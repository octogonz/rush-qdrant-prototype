//! Filesystem watcher for incremental re-indexing
//!
//! Watches catalog directories for file changes and triggers
//! incremental re-indexing when modifications are detected.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use notify::{Watcher, RecursiveMode, Event, EventKind};

use crate::engine::config::should_skip_path;
use crate::engine::chunker::chunk_file;
use crate::engine::ParallelEmbedder;
use crate::engine::QdrantUploader;
use crate::engine::util;
use crate::{CatalogConfig, is_text_file, chrono_timestamp};

/// Statistics from an incremental crawl
pub struct CrawlStats {
    pub new_files: usize,
    pub changed_files: usize,
    pub unchanged_files: usize,
    pub deleted_files: usize,
    pub chunks_embedded: usize,
}

/// Run an incremental crawl for a single catalog
///
/// Scans the catalog directory, compares with the existing Qdrant index,
/// and re-indexes changed/new files. Removes orphaned files.
pub fn run_incremental_crawl(
    catalog_name: &str,
    catalog_config: &CatalogConfig,
    embedder: &ParallelEmbedder,
    collection: &str,
    qdrant_url: Option<&str>,
) -> anyhow::Result<CrawlStats> {
    let directory = &catalog_config.path;
    let uploader = QdrantUploader::new(collection, qdrant_url)?;

    // Get existing files from Qdrant
    let existing_files = uploader.get_catalog_files(catalog_name)?;

    // Scan directory
    let mut files_to_process: Vec<(String, String)> = Vec::new(); // (absolute_path, relative_path)
    for entry in walkdir::WalkDir::new(directory)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path().to_string_lossy().to_string();
        if !should_skip_path(&path) && is_text_file(&path) {
            let rel_path = path
                .strip_prefix(directory)
                .unwrap_or(&path)
                .trim_start_matches('/')
                .trim_start_matches('\\')
                .to_string();
            files_to_process.push((path, rel_path));
        }
    }

    let rel_files_set: HashSet<String> = files_to_process.iter().map(|(_, rel)| rel.clone()).collect();

    let mut new_count = 0;
    let mut changed_count = 0;
    let mut unchanged_count = 0;
    let mut all_chunks: Vec<crate::engine::Chunk> = Vec::new();

    for (file_path, rel_path) in &files_to_process {
        // Read file and compute hash
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let current_hash = format!("sha256:{:x}", hasher.finalize());

        // Check if unchanged (using relative path for portability across machines)
        if let Some(existing_info) = existing_files.get(rel_path) {
            if existing_info.content_hash == current_hash && existing_info.file_complete {
                unchanged_count += 1;
                continue;
            }
            // Changed or incomplete — delete old chunks
            uploader.delete_file(rel_path, catalog_name)?;
            changed_count += 1;
        } else {
            new_count += 1;
        }

        // Chunk the file
        let package_name = if catalog_config.r#type == "monorepo" {
            crate::engine::package_lookup::find_package_name(file_path, directory)
        } else {
            std::path::Path::new(file_path)
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or(catalog_name)
                .to_string()
        };

        match chunk_file(file_path, catalog_name, directory, &package_name, 6000) {
            Ok(chunks) => {
                for mut chunk in chunks {
                    // Strip fallback marker from breadcrumb
                    chunk.breadcrumb = chunk.breadcrumb.replace(":[fallback-split]", "");
                    all_chunks.push(chunk);
                }
            }
            Err(e) => {
                eprintln!(
                    "[{}] Warning: failed to chunk {}: {}",
                    chrono_timestamp(),
                    file_path,
                    e
                );
            }
        }
    }

    // Delete orphaned files
    let mut deleted_count = 0;
    for (rel_path, _) in existing_files.iter() {
        if !rel_files_set.contains(rel_path) {
            uploader.delete_file(rel_path, catalog_name)?;
            deleted_count += 1;
        }
    }

    // Embed and upload chunks in streaming batches to limit memory
    let total_chunks = all_chunks.len();
    if total_chunks > 0 {
        use rayon::prelude::*;

        let use_cpu_batch = std::env::var("RUSH_QDRANT_CPU_BATCH").unwrap_or_default() == "1";

        // Track file completion across upload batches
        let mut file_chunks: HashMap<String, usize> = HashMap::new();
        let mut file_expected: HashMap<String, usize> = HashMap::new();

        if use_cpu_batch {
            // Parallel batching: sort by length, group into mini-batches,
            // distribute across workers for maximum throughput on CPU.
            all_chunks.sort_by_key(|c| c.text.len());

            // Adaptive mini-batch sizing: short chunks get larger batches
            const CHARS_PER_TOKEN: f64 = 4.0;
            let mut mini_batches: Vec<Vec<crate::engine::Chunk>> = Vec::new();
            let mut cursor = 0;
            while cursor < all_chunks.len() {
                let longest_chars = all_chunks[cursor..]
                    .iter()
                    .take(32)
                    .last()
                    .map(|c| c.text.len())
                    .unwrap_or(100);
                let est_tokens = (longest_chars as f64 / CHARS_PER_TOKEN).max(1.0);
                // CPU budget: scale batch size inversely with token count
                // Short chunks (<250 tokens): batch=16, medium: batch=8, long: batch=4
                let batch_size = ((4000.0 / est_tokens) as usize).clamp(4, 16);
                let end = (cursor + batch_size).min(all_chunks.len());
                mini_batches.push(all_chunks[cursor..end].to_vec());
                cursor = end;
            }
            // Free the original chunks now that mini_batches owns copies
            drop(all_chunks);

            let embedded: Vec<(crate::engine::Chunk, Vec<f32>)> = mini_batches
                .into_par_iter()
                .enumerate()
                .flat_map(|(i, batch)| {
                    let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
                    match embedder.encode_batch_on_worker(&texts, i) {
                        Ok(embeddings) => batch
                            .into_iter()
                            .zip(embeddings)
                            .collect::<Vec<_>>(),
                        Err(e) => {
                            eprintln!(
                                "[{}] Warning: batch embedding failed: {}",
                                chrono_timestamp(),
                                e
                            );
                            Vec::new()
                        }
                    }
                })
                .collect();

            // Upload in batches of 100 and free each batch
            for upload_batch in embedded.chunks(100) {
                uploader.upload_batch(upload_batch)?;
                for (chunk, _) in upload_batch {
                    let fid = util::display_file_id(chunk.file_id);
                    *file_chunks.entry(fid.clone()).or_insert(0) += 1;
                    file_expected.entry(fid).or_insert(chunk.chunk_count);
                }
            }
            drop(embedded);
        } else {
            // Default: individual encode per chunk, parallel across workers
            let embedded: Vec<(crate::engine::Chunk, Vec<f32>)> = all_chunks
                .into_par_iter()
                .enumerate()
                .filter_map(|(i, chunk)| match embedder.encode(&chunk.text, i) {
                    Ok(embedding) => Some((chunk, embedding)),
                    Err(e) => {
                        eprintln!(
                            "[{}] Warning: embedding failed: {}",
                            chrono_timestamp(),
                            e
                        );
                        None
                    }
                })
                .collect();

            // Upload in batches of 100 and free each batch
            for upload_batch in embedded.chunks(100) {
                uploader.upload_batch(upload_batch)?;
                for (chunk, _) in upload_batch {
                    let fid = util::display_file_id(chunk.file_id);
                    *file_chunks.entry(fid.clone()).or_insert(0) += 1;
                    file_expected.entry(fid).or_insert(chunk.chunk_count);
                }
            }
            drop(embedded);
        };

        // Mark files complete
        for (fid, count) in &file_chunks {
            if Some(count) == file_expected.get(fid) {
                let _ = uploader.mark_file_complete(fid, catalog_name);
            }
        }
    }

    Ok(CrawlStats {
        new_files: new_count,
        changed_files: changed_count,
        unchanged_files: unchanged_count,
        deleted_files: deleted_count,
        chunks_embedded: total_chunks,
    })
}

/// Start a file watcher for a single catalog
///
/// Spawns a background thread that watches for file changes,
/// debounces them (2 second quiet window), and triggers incremental re-indexing.
pub fn start_watcher(
    catalog_name: String,
    catalog_config: CatalogConfig,
    embedder: Arc<ParallelEmbedder>,
    collection: String,
    qdrant_url: Option<String>,
) {
    let watch_path = catalog_config.path.clone();

    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();

        let mut file_watcher = match notify::recommended_watcher(
            move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    match event.kind {
                        EventKind::Create(_)
                        | EventKind::Modify(_)
                        | EventKind::Remove(_) => {
                            let _ = tx.send(event);
                        }
                        _ => {}
                    }
                }
            },
        ) {
            Ok(w) => w,
            Err(e) => {
                eprintln!(
                    "[{}] Failed to create watcher for '{}': {}",
                    chrono_timestamp(),
                    catalog_name,
                    e
                );
                return;
            }
        };

        if let Err(e) =
            file_watcher.watch(std::path::Path::new(&watch_path), RecursiveMode::Recursive)
        {
            eprintln!(
                "[{}] Failed to watch '{}': {}",
                chrono_timestamp(),
                watch_path,
                e
            );
            return;
        }

        eprintln!(
            "[{}] Watching '{}' for changes (catalog: {})",
            chrono_timestamp(),
            watch_path,
            catalog_name
        );

        // Keep watcher alive by holding it in scope
        let _watcher = file_watcher;

        let mut has_pending = false;
        let mut quiet_since = Instant::now();

        loop {
            match rx.recv_timeout(Duration::from_millis(500)) {
                Ok(event) => {
                    // Only react to relevant file changes
                    let has_relevant = event.paths.iter().any(|p| {
                        let path_str = p.to_string_lossy();
                        !should_skip_path(&path_str) && is_text_file(&path_str)
                    });
                    if has_relevant {
                        has_pending = true;
                        quiet_since = Instant::now();
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }

            if has_pending && quiet_since.elapsed() >= Duration::from_secs(2) {
                eprintln!(
                    "[{}] Changes detected in '{}', re-indexing...",
                    chrono_timestamp(),
                    catalog_name
                );

                match run_incremental_crawl(
                    &catalog_name,
                    &catalog_config,
                    &embedder,
                    &collection,
                    qdrant_url.as_deref(),
                ) {
                    Ok(stats) => {
                        let total_changes =
                            stats.new_files + stats.changed_files + stats.deleted_files;
                        if total_changes > 0 {
                            eprintln!(
                                "[{}] Re-indexed '{}': {} new, {} changed, {} deleted ({} chunks)",
                                chrono_timestamp(),
                                catalog_name,
                                stats.new_files,
                                stats.changed_files,
                                stats.deleted_files,
                                stats.chunks_embedded
                            );
                        } else {
                            eprintln!(
                                "[{}] No indexable changes in '{}'",
                                chrono_timestamp(),
                                catalog_name
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "[{}] Re-index failed for '{}': {}",
                            chrono_timestamp(),
                            catalog_name,
                            e
                        );
                    }
                }

                has_pending = false;
            }
        }
    });
}
