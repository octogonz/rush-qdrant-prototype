//! rush-qdrant: Semantic search indexer for Rush monorepos
//! 
//! Uses Qdrant vector database with jina-embeddings-v2-base-code embeddings
//! Intelligently chunks code and documentation for high-quality semantic search

mod engine;

use clap::{Parser, Subcommand};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use engine::{
    config::should_skip_path,
    chunker::chunk_file,
    ParallelEmbedder,
    partitioner::{partition_typescript, PartitionConfig, ChunkQualityReport, PartitionDebug},
    uploader::{QdrantUploader, PointResult},
    SMALL_CHUNK_CHARS,
};

/// Qdrant configuration
#[derive(Debug, serde::Deserialize)]
struct QdrantConfig {
    url: Option<String>,
    collection: String,
}

/// Catalog configuration
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct CatalogConfig {
    /// Catalog type: "monorepo" or "folder"
    r#type: String,
    /// Path to scan
    path: String,
}

/// Main configuration file
#[derive(Debug, serde::Deserialize)]
struct Config {
    qdrant: QdrantConfig,
    catalogs: HashMap<String, CatalogConfig>,
}

/// Rush semantic search crawler for Qdrant
/// https://www.rushstack.io
#[derive(Parser)]
#[command(name = "rush-qdrant", version, about)]
struct Cli {
    /// Config file path (default: ~/.config/rush-qdrant/config.jsonc)
    #[arg(long)]
    config: Option<PathBuf>,
    
    #[command(subcommand)]
    command: Commands,
}

/// Available commands
#[derive(Subcommand)]
enum Commands {
    /// Crawl source and index into Qdrant (incremental sync).
    /// Reports warnings when AST chunking fails and fallback is used.
    /// These warnings indicate partitioner defects to investigate.
    Crawl {
        /// Catalog name (from config file)
        #[arg(long)]
        catalog: String,

        /// Allow files with chunking warnings to participate in incremental skipping
        #[arg(long, default_value_t = false)]
        incremental_warnings: bool,
    },
    
    /// Purge all chunks from a catalog or entire collection
    Purge {
        /// Catalog name to purge (if not specified, purges entire collection)
        #[arg(long)]
        catalog: Option<String>,
        
        /// Purge all catalogs (entire collection)
        #[arg(long)]
        all: bool,
    },
    
    /// Dump chunks for a TypeScript file (for debugging chunking algorithm).
    /// Uses AST-only mode by default to reveal partitioner issues.
    /// Add --with-fallback to see production behavior with fallback mitigation.
    DumpChunks {
        /// TypeScript file path
        #[arg(long)]
        file: PathBuf,
        
        /// Target chunk size in chars
        #[arg(long, default_value = "6000")]
        target_size: usize,
        
        /// Show visualization mode (full chunk contents)
        #[arg(long)]
        visualize: bool,
        
        /// Enable fallback line-based splitting for oversized chunks.
        /// By default, dump-chunks uses strict AST-only mode to reveal
        /// where the partitioner failed to find good split points.
        #[arg(long)]
        with_fallback: bool,
        
        /// Enable debug logging for partitioning decisions
        #[arg(long)]
        debug: bool,
    },
    
    /// Search with compact blurb output (for AI assistants)
    Search {
        /// Search query text
        #[arg(long)]
        text: String,
        
        /// Number of results
        #[arg(long, default_value = "10")]
        limit: usize,
        
        /// Filter by catalog (optional - searches all if omitted)
        #[arg(long)]
        catalog: Option<String>,
    },
    
    /// View chunks by their file IDs with optional selectors
    View {
        /// File IDs with optional selectors (can be specified multiple times)
        /// Formats: 
        ///   700a4ba232fe9ddc        - all chunks in file
        ///   700a4ba232fe9ddc:3      - chunk 3
        ///   700a4ba232fe9ddc:2-3    - chunks 2 through 3
        ///   700a4ba232fe9ddc:3-end  - chunk 3 through the last chunk
        #[arg(long)]
        id: Vec<String>,
        
        /// Show full filesystem paths
        #[arg(long)]
        full_paths: bool,
        
        /// Omit catalog preamble (show only chunks)
        #[arg(long)]
        chunks_only: bool,
    },
    
    /// Audit chunking quality across multiple files (AST-only mode).
    /// Scores reflect AST partitioning quality without fallback mitigation.
    /// Use after eliminating crawl warnings to find suboptimal chunk boundaries.
    AuditChunks {
        /// Number of files to sample
        #[arg(long, default_value = "20")]
        count: usize,
        
        /// Directory to sample from
        #[arg(long)]
        dir: String,
    },
}

const DEFAULT_CONFIG_PATH: &str = "~/.config/rush-qdrant/config.jsonc";

/// Get current timestamp for logging
fn chrono_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let h = (now / 3600) % 24;
    let m = (now / 60) % 60;
    let s = now % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}


/// Format duration in seconds to human-readable string (e.g., "1h 23m" or "5m 30s")
fn format_duration(secs: f64) -> String {
    let total_secs = secs as u64;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    
    if hours > 0 {
        format!("{}h {}m", hours, mins)
    } else if mins > 0 {
        format!("{}m {}s", mins, s)
    } else {
        format!("{}s", s)
    }
}

/// Format ETA in seconds to human-readable string
fn format_eta(secs: f64) -> String {
    if secs <= 0.0 || !secs.is_finite() {
        return "--".to_string();
    }
    format_duration(secs)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    // Load config
    let config_path = cli.config.unwrap_or_else(|| {
        PathBuf::from(shellexpand::tilde(DEFAULT_CONFIG_PATH).as_ref())
    });
    let config = load_config(&config_path)?;

    match cli.command {
        Commands::Crawl { catalog, incremental_warnings } => {
            run_crawl(&config, &catalog, incremental_warnings)?;
        }
        Commands::Purge { catalog, all } => {
            run_purge(&config, catalog.as_deref(), all)?;
        }
        Commands::DumpChunks { file, target_size, visualize, with_fallback, debug } => {
            run_dump_chunks(&file, target_size, visualize, with_fallback, debug)?;
        }
        Commands::Search { text, limit, catalog } => {
            run_search(&config, &text, limit, catalog.as_deref())?;
        }
        Commands::View { id, full_paths, chunks_only } => {
            run_view(&config, &id, full_paths, chunks_only)?;
        }
        Commands::AuditChunks { count, dir } => {
            run_audit_chunks(count, dir)?;
        }
    }

    Ok(())
}

fn load_config(path: &PathBuf) -> anyhow::Result<Config> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Failed to read config file {}: {}", path.display(), e))?;
    
    // Parse JSON (for now - will add JSONC support later)
    let config: Config = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Failed to parse config file: {}", e))?;
    
    Ok(config)
}

/// Run crawl (incremental sync)
fn run_crawl(config: &Config, catalog_name: &str, incremental_warnings: bool) -> anyhow::Result<()> {
    let total_start = std::time::Instant::now();
    println!("🔍 Starting crawl...");
    println!("Catalog: {}", catalog_name);
    
    // Get catalog config
    let catalog_config = config.catalogs.get(catalog_name)
        .ok_or_else(|| anyhow::anyhow!("Catalog '{}' not found in config", catalog_name))?;
    
    let directory = &catalog_config.path;
    
    println!("Directory: {}", directory);
    println!("Type: {}", catalog_config.r#type);
    println!("Collection: {}", config.qdrant.collection);
    println!();

    // Initialize parallel embedder
    println!("⚙️  Loading embedding model...");
    let embedder = ParallelEmbedder::new()?;
    println!();

    let uploader = QdrantUploader::new(&config.qdrant.collection, config.qdrant.url.as_deref())?;

    // Get existing files from DB for this catalog
    println!("📂 Checking existing index...");
    let existing_files = uploader.get_catalog_files(catalog_name)?;
    println!("Found {} files already indexed", existing_files.len());

    // Load persisted chunking warning files (sticky by default)
    let warning_state_path = PathBuf::from(shellexpand::tilde(&format!(
        "~/.config/rush-qdrant/warnings-{}.json",
        catalog_name
    )).as_ref());
    let warning_files: HashSet<String> = match std::fs::read_to_string(&warning_state_path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => HashSet::new(),
    };
    if !warning_files.is_empty() {
        println!("Found {} files with prior chunking warnings", warning_files.len());
    }
    println!();

    // Scan directory
    println!("📂 Scanning directory...");
    let mut files_to_process: Vec<String> = Vec::new();

    for entry in walkdir::WalkDir::new(directory)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path().to_string_lossy().to_string();
        
        if !should_skip_path(&path) && is_text_file(&path) {
            files_to_process.push(path);
        }
    }

    let total_files = files_to_process.len();
    println!("Found {} files in directory", total_files);
    println!();

    // Categorize files
    let mut new_count = 0;
    let mut changed_count = 0;
    let mut unchanged_count = 0;
    let mut orphaned_count = 0;
    
    // Find orphaned files (in DB but not on disk)
    let files_set: std::collections::HashSet<String> = files_to_process.iter().cloned().collect();
    for (file_path, _) in existing_files.iter() {
        if !files_set.contains(file_path) {
            orphaned_count += 1;
        }
    }

    // Phase 1: Collect all chunks (sequential - file I/O bound)
    println!("📦 Phase 1: Chunking files...");
    let mut all_chunks: Vec<engine::Chunk> = Vec::new();
    let mut chunks_by_type: HashMap<String, usize> = HashMap::new();
    let mut files_deleted = 0;
    let mut crawl_warning_files: HashSet<String> = HashSet::new();
    let mut warning_count: usize = 0;
    
    for (idx, file_path) in files_to_process.iter().enumerate() {
        // Progress indicator
        print!("\r  Chunking file {}/{} ({:.0}%) | warnings: {}   ", 
            idx + 1, total_files, 
            ((idx + 1) as f64 / total_files as f64) * 100.0,
            warning_count);
        std::io::Write::flush(&mut std::io::stdout())?;

        // Read file and compute hash
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("\n  ⚠️  Failed to read {}: {}", file_path, e);
                continue;
            }
        };
        
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let current_hash = format!("sha256:{:x}", hasher.finalize());

        // Check if file changed
        if let Some(existing_info) = existing_files.get(file_path) {
            if existing_info.content_hash == current_hash && existing_info.file_complete {
                let has_warning = warning_files.contains(file_path);
                if has_warning && !incremental_warnings {
                    // Sticky retry for warning files: always reprocess until clean
                } else {
                    unchanged_count += 1;
                    continue; // Skip unchanged and complete file
                }
            }
            
            // File changed or incomplete - delete old chunks
            uploader.delete_file(file_path, catalog_name)?;
            files_deleted += 1;
            if existing_info.content_hash != current_hash {
                changed_count += 1;
            } else {
                // Content hash matches but file was incomplete
                changed_count += 1;
            }
        } else {
            new_count += 1;
        }

        // Chunk the file
        let repo_root = &catalog_config.path;
        let package_name_or_folder = if catalog_config.r#type == "monorepo" {
            engine::package_lookup::find_package_name(file_path, repo_root)
        } else {
            // For folder type, use the folder name
            std::path::Path::new(file_path)
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or(catalog_name)
                .to_string()
        };
        
        match chunk_file(file_path, catalog_name, repo_root, &package_name_or_folder, 6000) {
            Ok(chunks) => {
                // Detect fallback warning marker in chunks (injected by partitioner via breadcrumb suffix)
                let had_warning = chunks.iter().any(|c| c.breadcrumb.contains("[fallback-split]"));
                if had_warning {
                    warning_count += 1;
                    crawl_warning_files.insert(file_path.clone());
                    println!();
                    println!("Warning: Couldn't find a splitpoint for {}", file_path);
                }

                for mut chunk in chunks {
                    if had_warning {
                        // Strip marker from stored breadcrumb; marker is only for signaling in-process
                        chunk.breadcrumb = chunk.breadcrumb.replace(":[fallback-split]", "");
                    }
                    *chunks_by_type.entry(chunk.chunk_type.clone()).or_insert(0) += 1;
                    all_chunks.push(chunk);
                }
            }
            Err(e) => {
                eprintln!("\n  ⚠️  Failed to chunk file {}: {}", file_path, e);
            }
        }
    }
    
    let total_chunks = all_chunks.len();
    println!("\n  Found {} chunks to embed", total_chunks);
    println!();

    // Phase 2: Embedding
    let use_batch = std::env::var("RUSH_QDRANT_BATCH").unwrap_or_default() == "1";
    let batch_size: usize = std::env::var("RUSH_QDRANT_BATCH_SIZE")
        .unwrap_or_else(|_| "0".to_string())
        .parse()
        .unwrap_or(0); // 0 = all at once

    if use_batch {
        // ── Batch-mode path: encode all chunks in large batches, then upload ──
        println!("⚡ Phase 2 (BATCH): Encoding {} chunks{}...",
            total_chunks,
            if batch_size > 0 { format!(" in batches of {}", batch_size) } else { " all at once".to_string() });
        let embed_start = std::time::Instant::now();

        let uploader = QdrantUploader::new(&config.qdrant.collection, config.qdrant.url.as_deref())?;

        let effective_batch_size = if batch_size > 0 { batch_size } else { total_chunks };
        let mut all_embedded: Vec<(engine::Chunk, Vec<f32>)> = Vec::with_capacity(total_chunks);

        for (batch_idx, batch_chunks) in all_chunks.chunks(effective_batch_size).enumerate() {
            let batch_start = std::time::Instant::now();
            let texts: Vec<&str> = batch_chunks.iter().map(|c| c.text.as_str()).collect();

            let embeddings = embedder.encode_batch(&texts)?;

            let batch_elapsed = batch_start.elapsed();
            let offset = batch_idx * effective_batch_size;
            eprintln!("[{}] Batch {} ({}-{}/{}): {} chunks in {:.1}s ({:.1} chunks/sec)",
                chrono_timestamp(),
                batch_idx + 1,
                offset + 1,
                (offset + batch_chunks.len()).min(total_chunks),
                total_chunks,
                batch_chunks.len(),
                batch_elapsed.as_secs_f64(),
                batch_chunks.len() as f64 / batch_elapsed.as_secs_f64().max(0.001));

            for (chunk, embedding) in batch_chunks.iter().zip(embeddings.into_iter()) {
                all_embedded.push((chunk.clone(), embedding));
            }
        }

        let embed_elapsed = embed_start.elapsed();
        let embed_rate = total_chunks as f64 / embed_elapsed.as_secs_f64().max(0.001);
        println!("  Embedding complete: {:.1} chunks/sec ({:.1}s)", embed_rate, embed_elapsed.as_secs_f64());

        // Upload all at once
        println!("  Uploading {} chunks to Qdrant...", all_embedded.len());
        let upload_start = std::time::Instant::now();

        // Upload in batches of 100 to avoid exceeding Qdrant limits
        for upload_batch in all_embedded.chunks(100) {
            uploader.upload_batch(upload_batch)?;
        }

        // Mark files complete
        let mut file_chunks: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut file_expected: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (chunk, _) in &all_embedded {
            let fid = engine::util::display_file_id(chunk.file_id);
            *file_chunks.entry(fid.clone()).or_insert(0) += 1;
            file_expected.entry(fid).or_insert(chunk.chunk_count);
        }
        for (fid, count) in &file_chunks {
            if Some(count) == file_expected.get(fid) {
                let _ = uploader.mark_file_complete(fid);
            }
        }

        let upload_elapsed = upload_start.elapsed();
        println!("  Upload complete: {:.1}s", upload_elapsed.as_secs_f64());

        let total_elapsed_phase2 = embed_elapsed.as_secs_f64() + upload_elapsed.as_secs_f64();
        println!();
        println!("  ✅ Embedded & uploaded {} chunks in {}", total_chunks, format_duration(total_elapsed_phase2));
        println!("  📊 Embedding rate: {:.1} chunks/sec (embed only)", embed_rate);
        println!("  📊 Overall Phase 2 rate: {:.1} chunks/sec (embed + upload)", total_chunks as f64 / total_elapsed_phase2.max(0.001));
        println!();

        // Phase 3: Cleanup orphaned files
        println!("🗑️  Cleaning up orphaned files...");
        for (file_path, _) in existing_files.iter() {
            if !files_set.contains(file_path) {
                uploader.delete_file(file_path, catalog_name)?;
                files_deleted += 1;
            }
        }

        println!();
        println!();
        let total_elapsed = total_start.elapsed();

        println!("✅ Crawl complete!");
        println!();
        println!("📊 Summary:");
        println!("  Total time: {:?}", total_elapsed);
        println!("  New files indexed: {}", new_count);
        println!("  Changed files re-indexed: {}", changed_count);
        println!("  Unchanged files skipped: {}", unchanged_count);
        println!("  Orphaned files deleted: {}", orphaned_count);
        println!();
        println!("Total chunks indexed: {}", total_chunks);
        println!("Overall rate: {:.1} chunks/sec", total_chunks as f64 / total_elapsed.as_secs_f64().max(0.001));
        println!("Files deleted from DB: {}", files_deleted);
        println!();

        // Update warning state
        let mut next_warning_files: HashSet<String> = HashSet::new();
        next_warning_files.extend(crawl_warning_files.iter().cloned());
        if incremental_warnings {
            next_warning_files.extend(warning_files.iter().cloned());
        }
        let json = serde_json::to_string_pretty(&next_warning_files)?;
        std::fs::write(&warning_state_path, json)?;

        if !crawl_warning_files.is_empty() {
            let plural = if crawl_warning_files.len() == 1 { "file" } else { "files" };
            println!("Chunking warnings in {} {}:", crawl_warning_files.len(), plural);
            for file in crawl_warning_files.iter().take(20) {
                println!("  - {}", file);
            }
            if crawl_warning_files.len() > 20 {
                println!("  ... and {} more", crawl_warning_files.len() - 20);
            }
            println!();
        }

        return Ok(());
    }

    // ── Original streaming path (non-batch) ──
    println!("⚡ Phase 2: Embedding {} chunks with {} parallel sessions...",
        total_chunks, embedder.num_workers());
    println!("  (Checkpoints every 60s - safe to CTRL+C)");
    let embed_start = std::time::Instant::now();

    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use crossbeam_channel::{unbounded, Sender, Receiver};
    
    // Channels for streaming embeddings to uploader
    let (embed_tx, embed_rx): (Sender<(engine::Chunk, Vec<f32>)>, Receiver<(engine::Chunk, Vec<f32>)>) = unbounded();
    
    let processed = Arc::new(AtomicUsize::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));
    
    // Track last upload time
    let last_upload_time = Arc::new(Mutex::new(std::time::Instant::now()));
    
    // Progress reporter thread - prints every 30 seconds
    let processed_clone = Arc::clone(&processed);
    let stop_clone = Arc::clone(&stop_flag);
    let total_chunks_for_thread = total_chunks;
    let embed_start_for_thread = std::time::Instant::now();
    let last_print_time = Arc::new(Mutex::new(std::time::Instant::now()));
    let last_print_clone = Arc::clone(&last_print_time);
    
    let progress_thread = std::thread::spawn(move || {
        while !stop_clone.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_secs(5));
            
            let mut last = last_print_clone.lock().unwrap();
            if last.elapsed() >= std::time::Duration::from_secs(30) {
                let current = processed_clone.load(Ordering::Relaxed);
                let elapsed = embed_start_for_thread.elapsed();
                let rate = current as f64 / elapsed.as_secs_f64().max(0.001);
                let remaining = (total_chunks_for_thread - current) as f64 / rate;
                let eta = format_eta(remaining);
                
                eprintln!("[{}] Embedded {}/{} ({:.0}%) - {:.1} chunks/sec - ETA: {}", 
                    chrono_timestamp(),
                    current, total_chunks_for_thread, 
                    (current as f64 / total_chunks_for_thread as f64) * 100.0,
                    rate, eta);
                
                *last = std::time::Instant::now();
            }
        }
    });
    
    // Wrap uploader in Arc<Mutex> for sharing across threads
    let uploader = Arc::new(Mutex::new(uploader));
    
    // Uploader thread - uploads accumulated embeddings every 60 seconds
    let stop_uploader = Arc::clone(&stop_flag);
    let last_upload_time_clone = Arc::clone(&last_upload_time);
    let uploader_clone = Arc::clone(&uploader);
    
    let uploader_thread = std::thread::spawn(move || {
        let mut accumulated: Vec<(engine::Chunk, Vec<f32>)> = Vec::new();
        
        // Track file completion:
        // - expected_count[file_id]: total chunks expected for this file (set once, from chunk.chunk_count)
        // - uploaded_count[file_id]: chunks successfully uploaded to Qdrant
        let mut expected_count: HashMap<String, usize> = HashMap::new();
        let mut uploaded_count: HashMap<String, usize> = HashMap::new();
        
        loop {
            // Check if we should upload (60s elapsed or stopped)
            let should_upload = {
                let mut last = last_upload_time_clone.lock().unwrap();
                if last.elapsed() >= std::time::Duration::from_secs(60) {
                    *last = std::time::Instant::now();
                    true
                } else {
                    false
                }
            };
            
            // Collect all available embeddings
            // Track expected count per file (set once on first observation)
            while let Ok(embedded) = embed_rx.try_recv() {
                let file_id = engine::util::display_file_id(embedded.0.file_id);
                
                // Set expected count on first observation of this file
                if let std::collections::hash_map::Entry::Vacant(e) = expected_count.entry(file_id.clone()) {
                    e.insert(embedded.0.chunk_count);
                } else {
                    // Validate consistency: all chunks for same file should report same chunk_count
                    let existing = expected_count.get(&file_id).unwrap();
                    if *existing != embedded.0.chunk_count {
                        eprintln!(
                            "[{}] ⚠️ Inconsistent chunk_count for file {}: expected {}, got {}",
                            chrono_timestamp(), file_id, existing, embedded.0.chunk_count
                        );
                    }
                }
                
                accumulated.push(embedded);
            }
            
            if should_upload && !accumulated.is_empty() {
                let count = accumulated.len();
                eprintln!("[{}] Uploading checkpoint ({} chunks)...", chrono_timestamp(), count);
                
                let uploader_guard = uploader_clone.lock().unwrap();
                match uploader_guard.upload_batch(&accumulated) {
                    Err(e) => {
                        eprintln!("[{}] ⚠️ Upload failed: {}", chrono_timestamp(), e);
                        // Do NOT update uploaded_count - these chunks were not persisted
                    }
                    Ok(_) => {
                        // Upload succeeded - now update uploaded_count per file
                        let mut files_in_batch: HashMap<String, usize> = HashMap::new();
                        for (chunk, _) in &accumulated {
                            let file_id = engine::util::display_file_id(chunk.file_id);
                            *files_in_batch.entry(file_id).or_insert(0) += 1;
                        }
                        
                        // Merge into uploaded_count
                        for (file_id, batch_count) in &files_in_batch {
                            *uploaded_count.entry(file_id.clone()).or_insert(0) += batch_count;
                        }
                        
                        // Check completion once per file (files where uploaded == expected)
                        let mut completed_files: Vec<String> = Vec::new();
                        for file_id in files_in_batch.keys() {
                            let uploaded = uploaded_count.get(file_id).copied().unwrap_or(0);
                            let expected = expected_count.get(file_id).copied().unwrap_or(0);
                            if uploaded == expected && expected > 0 {
                                completed_files.push(file_id.clone());
                            }
                        }
                        
                        // Mark completed files and clean up tracking state
                        for file_id in &completed_files {
                            if let Err(e) = uploader_guard.mark_file_complete(file_id) {
                                eprintln!("[{}] ⚠️ Failed to mark file complete: {}", chrono_timestamp(), e);
                            }
                            // Remove tracking state for completed files
                            uploaded_count.remove(file_id);
                            expected_count.remove(file_id);
                        }
                        
                        eprintln!("[{}] Checkpoint saved ({} files completed)", chrono_timestamp(), completed_files.len());
                    }
                }
                drop(uploader_guard);
                accumulated.clear();
            }
            
            // Check if done
            if stop_uploader.load(Ordering::Relaxed) {
                // Drain remaining
                while let Ok(embedded) = embed_rx.try_recv() {
                    let file_id = engine::util::display_file_id(embedded.0.file_id);
                    
                    if let std::collections::hash_map::Entry::Vacant(e) = expected_count.entry(file_id.clone()) {
                        e.insert(embedded.0.chunk_count);
                    }
                    
                    accumulated.push(embedded);
                }
                
                // Final upload
                if !accumulated.is_empty() {
                    eprintln!("[{}] Uploading final batch ({} chunks)...", chrono_timestamp(), accumulated.len());
                    
                    let uploader_guard = uploader_clone.lock().unwrap();
                    match uploader_guard.upload_batch(&accumulated) {
                        Err(e) => {
                            eprintln!("[{}] ⚠️ Final upload failed: {}", chrono_timestamp(), e);
                        }
                        Ok(_) => {
                            // Update uploaded_count
                            let mut files_in_batch: HashMap<String, usize> = HashMap::new();
                            for (chunk, _) in &accumulated {
                                let file_id = engine::util::display_file_id(chunk.file_id);
                                *files_in_batch.entry(file_id).or_insert(0) += 1;
                            }
                            
                            for (file_id, batch_count) in &files_in_batch {
                                *uploaded_count.entry(file_id.clone()).or_insert(0) += batch_count;
                            }
                            
                            // Check and mark completed files
                            let mut completed_files: Vec<String> = Vec::new();
                            for file_id in files_in_batch.keys() {
                                let uploaded = uploaded_count.get(file_id).copied().unwrap_or(0);
                                let expected = expected_count.get(file_id).copied().unwrap_or(0);
                                if uploaded == expected && expected > 0 {
                                    completed_files.push(file_id.clone());
                                }
                            }
                            
                            for file_id in &completed_files {
                                if let Err(e) = uploader_guard.mark_file_complete(file_id) {
                                    eprintln!("[{}] ⚠️ Failed to mark file complete: {}", chrono_timestamp(), e);
                                }
                            }
                        }
                    }
                }
                break;
            }
            
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });
    
    // Process all chunks in parallel, streaming results to uploader
    let embed_tx_clone = embed_tx.clone();
    let processed_embed = Arc::clone(&processed);
    
    all_chunks
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, chunk)| -> anyhow::Result<()> {
            let embedding = embedder.encode(&chunk.text, i)?;
            
            // Update counter
            processed_embed.fetch_add(1, Ordering::Relaxed);
            
            // Send to uploader
            embed_tx_clone.send((chunk, embedding))?;
            
            Ok(())
        })?;
    
    // Signal threads to stop
    stop_flag.store(true, Ordering::Relaxed);
    
    // Wait for threads
    drop(embed_tx); // Close channel
    let _ = progress_thread.join();
    let _ = uploader_thread.join();
    
    let embed_elapsed = embed_start.elapsed();
    let total_uploaded = processed.load(Ordering::Relaxed);
    let embed_rate = if embed_elapsed.as_secs() > 0 {
        total_uploaded as f64 / embed_elapsed.as_secs_f64()
    } else {
        total_uploaded as f64
    };
    println!();
    println!("  ✅ Embedded & uploaded {} chunks in {}", total_uploaded, format_duration(embed_elapsed.as_secs_f64()));
    println!("  📊 Embedding rate: {:.1} chunks/sec", embed_rate);
    println!();
    
    // Phase 3: Cleanup orphaned files
    println!("🗑️  Cleaning up orphaned files...");
    {
        let uploader_guard = uploader.lock().unwrap();
        for (file_path, _) in existing_files.iter() {
            if !files_set.contains(file_path) {
                uploader_guard.delete_file(file_path, catalog_name)?;
                files_deleted += 1;
            }
        }
    }

    println!();
    println!();
    let total_elapsed = total_start.elapsed();
    
    println!("✅ Crawl complete!");
    println!();
    println!("📊 Summary:");
    println!("  Total time: {:?}", total_elapsed);
    println!("  New files indexed: {}", new_count);
    println!("  Changed files re-indexed: {}", changed_count);
    println!("  Unchanged files skipped: {}", unchanged_count);
    println!("  Orphaned files deleted: {}", orphaned_count);
    println!();
    println!("Total chunks indexed: {}", total_chunks);
    println!("Overall rate: {:.1} chunks/sec", total_chunks as f64 / total_elapsed.as_secs_f64().max(0.001));
    println!("Files deleted from DB: {}", files_deleted);
    println!();

    // Update warning state: keep files that had warnings this crawl
    // plus any previous warning files that were skipped due to incremental mode.
    let mut next_warning_files: HashSet<String> = HashSet::new();
    next_warning_files.extend(crawl_warning_files.iter().cloned());
    if incremental_warnings {
        // In this mode, unchanged warning files may remain skipped; preserve prior state.
        next_warning_files.extend(warning_files.iter().cloned());
    }
    let json = serde_json::to_string_pretty(&next_warning_files)?;
    std::fs::write(&warning_state_path, json)?;

    // Warning summary
    if !crawl_warning_files.is_empty() {
        let plural = if crawl_warning_files.len() == 1 { "file" } else { "files" };
        println!("Chunking warnings in {} {}:", crawl_warning_files.len(), plural);
        for file in crawl_warning_files.iter().take(20) {
            println!("  - {}", file);
        }
        if crawl_warning_files.len() > 20 {
            println!("  ... and {} more", crawl_warning_files.len() - 20);
        }
        println!();
    }

    Ok(())
}

/// Run search with compact blurb output
fn run_search(config: &Config, text: &str, limit: usize, catalog: Option<&str>) -> anyhow::Result<()> {
    // Generate embedding for query
    let embedder = ParallelEmbedder::new()?;
    let embedding = embedder.encode(text, 0)?;
    
    // Query Qdrant
    let uploader = QdrantUploader::new(&config.qdrant.collection, config.qdrant.url.as_deref())?;
    let results = uploader.query(&embedding, limit, catalog)?;
    
    // Display results as blurbs
    for result in &results {
        // Line 1: file_id:chunk_number  score  breadcrumb
        let breadcrumb = result.payload.breadcrumb.as_deref().unwrap_or("unknown");
        println!("{}:{}  {:.3}  {}", 
            result.payload.file_id, 
            result.payload.chunk_number, 
            result.score, 
            breadcrumb);
        
        // Lines 2-4: first 3 lines of code (quoted with >)
        for line in result.payload.text.lines().take(3) {
            println!("> {}", line);
        }
        
        // Blank line between results
        println!();
    }
    
    Ok(())
}

/// Run view command to display full chunks by IDs
/// Parsed selector for file-based chunk queries
#[derive(Debug, Clone)]
enum ChunkSelector {
    /// All chunks in the file
    All,
    /// Single chunk at position N (1-indexed)
    Single(usize),
    /// Range from start to end (inclusive, 1-indexed)
    Range(usize, usize),
    /// Range from start to the end of file
    ToEnd(usize),
}

/// Parse file ID with optional selector
/// 
/// Formats:
/// - `700a4ba232fe9ddc` - all chunks in file
/// - `700a4ba232fe9ddc:3` - chunk 3
/// - `700a4ba232fe9ddc:2-3` - chunks 2 through 3
/// - `700a4ba232fe9ddc:3-end` - chunk 3 through the last chunk
fn parse_file_id_with_selector(s: &str) -> anyhow::Result<(String, ChunkSelector)> {
    let s = s.trim();
    
    // Check for selector suffix
    if let Some(colon_pos) = s.find(':') {
        let file_id = s[..colon_pos].to_string();
        let selector = &s[colon_pos + 1..];
        
        // Validate file_id is 16 hex chars
        if file_id.len() != 16 || !file_id.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(anyhow::anyhow!(
                "Invalid file ID '{}'. Expected 16 hex characters.",
                file_id
            ));
        }
        
        // Parse selector
        if selector == "end" {
            // Invalid: ":end" without start
            return Err(anyhow::anyhow!(
                "Invalid selector ':end'. Use ':N-end' format."
            ));
        } else if selector.ends_with("-end") {
            // :N-end format
            let start_str = &selector[..selector.len() - 4];
            let start: usize = start_str.parse()
                .map_err(|_| anyhow::anyhow!("Invalid chunk number in selector '{}'", selector))?;
            if start < 1 {
                return Err(anyhow::anyhow!("Chunk numbers are 1-indexed, got {}", start));
            }
            Ok((file_id, ChunkSelector::ToEnd(start)))
        } else if selector.contains('-') {
            // :N-M format
            let parts: Vec<&str> = selector.split('-').collect();
            if parts.len() != 2 {
                return Err(anyhow::anyhow!("Invalid selector '{}'. Expected ':N-M' format.", selector));
            }
            let start: usize = parts[0].parse()
                .map_err(|_| anyhow::anyhow!("Invalid start chunk in selector '{}'", selector))?;
            let end: usize = parts[1].parse()
                .map_err(|_| anyhow::anyhow!("Invalid end chunk in selector '{}'", selector))?;
            if start < 1 || end < 1 {
                return Err(anyhow::anyhow!("Chunk numbers are 1-indexed, got {}:{}", start, end));
            }
            if start > end {
                return Err(anyhow::anyhow!("Start chunk {} > end chunk {}", start, end));
            }
            Ok((file_id, ChunkSelector::Range(start, end)))
        } else {
            // :N format (single chunk)
            let chunk_num: usize = selector.parse()
                .map_err(|_| anyhow::anyhow!("Invalid chunk number in selector '{}'", selector))?;
            if chunk_num < 1 {
                return Err(anyhow::anyhow!("Chunk numbers are 1-indexed, got {}", chunk_num));
            }
            Ok((file_id, ChunkSelector::Single(chunk_num)))
        }
    } else {
        // No selector - validate file_id and return All selector
        if s.len() != 16 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(anyhow::anyhow!(
                "Invalid file ID '{}'. Expected 16 hex characters.",
                s
            ));
        }
        Ok((s.to_string(), ChunkSelector::All))
    }
}

fn run_view(config: &Config, id_specs: &[String], show_full_paths: bool, chunks_only: bool) -> anyhow::Result<()> {
    if id_specs.is_empty() {
        return Err(anyhow::anyhow!("No IDs provided. Use --id <file_id>[:<selector>]"));
    }
    
    // Parse all file IDs with selectors
    let mut requests: Vec<(String, ChunkSelector)> = Vec::new();
    for spec in id_specs {
        let (file_id, selector) = parse_file_id_with_selector(spec)?;
        requests.push((file_id, selector));
    }
    
    // Query Qdrant
    let uploader = QdrantUploader::new(&config.qdrant.collection, config.qdrant.url.as_deref())?;
    
    // Collect all results with their original selectors for display
    let mut all_results: Vec<(String, ChunkSelector, Vec<PointResult>)> = Vec::new();
    
    for (file_id, selector) in requests {
        let chunks = uploader.get_chunks_by_file_id(&file_id)?;
        
        // Filter by selector
        let filtered: Vec<PointResult> = match &selector {
            ChunkSelector::All => chunks,
            ChunkSelector::Single(n) => {
                chunks.into_iter().filter(|c| c.payload.chunk_number == *n).collect()
            }
            ChunkSelector::Range(start, end) => {
                chunks.into_iter().filter(|c| {
                    c.payload.chunk_number >= *start && c.payload.chunk_number <= *end
                }).collect()
            }
            ChunkSelector::ToEnd(start) => {
                chunks.into_iter().filter(|c| c.payload.chunk_number >= *start).collect()
            }
        };
        
        all_results.push((file_id, selector, filtered));
    }
    
    // Collect unique catalogs for preamble
    if !chunks_only {
        let catalogs: std::collections::HashSet<&str> = all_results
            .iter()
            .flat_map(|(_, _, results)| results.iter().map(|r| r.payload.catalog.as_str()))
            .collect();
        
        if !catalogs.is_empty() {
            println!("Catalogs:");
            for cat in catalogs {
                if let Some(cat_config) = config.catalogs.get(cat) {
                    println!("- {}", cat);
                    println!("  Catalog path: {}", cat_config.path);
                }
            }
            println!();
        }
    }
    
    // Display results
    for (file_id, selector, results) in &all_results {
        if results.is_empty() {
            // No chunks found
            let selector_str = match selector {
                ChunkSelector::All => String::new(),
                ChunkSelector::Single(n) => format!(":{}", n),
                ChunkSelector::Range(start, end) => format!(":{}-{}", start, end),
                ChunkSelector::ToEnd(start) => format!(":{}-end", start),
            };
            println!("{}{} ERROR: CHUNK NOT FOUND", file_id, selector_str);
            continue;
        }
        
        for result in results {
            let breadcrumb = result.payload.breadcrumb.as_deref().unwrap_or("unknown");
            let chunk_count = result.payload.chunk_count;
            let chunk_number = result.payload.chunk_number;
            
            // Header line: <file_id>:<chunk_number> (<n>/<total>) <breadcrumb>
            println!("{}:{} ({}/{}) {}", file_id, chunk_number, chunk_number, chunk_count, breadcrumb);
            
            // Source line
            println!("Source: {}:{}", result.payload.catalog, result.payload.relative_path);
            
            // Full path (optional)
            if show_full_paths {
                println!("Full path: {}", result.payload.source_uri);
            }
            
            // Lines and type
            println!("Lines: {}-{}", result.payload.start_line, result.payload.end_line);
            println!("Type: {}", result.payload.chunk_type);
            
            // Content
            println!();
            for line in result.payload.text.lines() {
                println!("> {}", line);
            }
            
            println!();
        }
    }
    
    Ok(())
}

/// Run purge command (delete all chunks from a catalog or entire collection)
fn run_purge(config: &Config, catalog: Option<&str>, all: bool) -> anyhow::Result<()> {
    let uploader = QdrantUploader::new(&config.qdrant.collection, config.qdrant.url.as_deref())?;

    if all {
        println!("🗑️  Purging entire collection: {}", config.qdrant.collection);
        println!("This will delete ALL data from the collection!");
        
        // Delete all points with empty filter
        let endpoint = format!(
            "{}/collections/{}/points/delete",
            config.qdrant.url.as_deref().unwrap_or("http://localhost:6333"),
            config.qdrant.collection
        );
        
        let empty_filter = serde_json::json!({"filter": {}});
        
        let response = reqwest::blocking::Client::new()
            .post(&endpoint)
            .json(&empty_filter)
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to purge collection: HTTP {}", response.status()));
        }
        
        println!("✅ Collection purged successfully");
    } else if let Some(catalog_name) = catalog {
        println!("🗑️  Purging catalog: {}", catalog_name);
        
        let operation_id = uploader.delete_catalog(catalog_name)?;
        println!("✅ Catalog purged successfully (operation ID: {})", operation_id);
    } else {
        return Err(anyhow::anyhow!("Must specify either --catalog <name> or --all"));
    }

    Ok(())
}

/// Check if a file is a text file we want to index
fn is_text_file(path: &str) -> bool {
    let extensions = [
        "ts", "tsx", "js", "jsx",           // TypeScript/JavaScript
        "md", "mdx",                        // Markdown
        "json",                             // JSON
        "yaml", "yml",                      // YAML
        "txt", "rst", "mdn",                // Text docs
        "toml", "ini", "conf",              // Config files
    ];

    let path_lower = path.to_lowercase();
    extensions.iter().any(|ext| path_lower.ends_with(&format!(".{}", ext)))
}

/// Run chunking diagnostics on a TypeScript file
fn run_dump_chunks(file: &PathBuf, target_size: usize, visualize: bool, with_fallback: bool, enable_debug: bool) -> anyhow::Result<()> {
    println!("📦 Chunks for: {}", file.display());
    if !with_fallback {
        println!("🔍 Strict mode: AST-only (fallback disabled)");
    }
    println!();
    
    // Read file
    let source = std::fs::read_to_string(file)
        .map_err(|e| anyhow::anyhow!("Failed to read file: {}", e))?;
    
    // Determine file name and package name
    let file_name = file.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown.ts");
    
    // Find package name by walking upward to find nearest package.json
    let file_path = file.to_string_lossy().to_string();
    let package_name = engine::package_lookup::find_package_name(&file_path, "");
    
    // Create config
    let config = PartitionConfig {
        target_size,
        file_name: file_name.to_string(),
        package_name: package_name.clone(),
        debug: PartitionDebug { enabled: enable_debug },
        allow_fallback: with_fallback,  // AST-only by default, enable fallback with flag
    };
    
    // Partition
    let chunks = partition_typescript(&source, &config, &file_path, &package_name);
    
    // Quality score
    let file_chars = source.len();
    let report = ChunkQualityReport::from_chunks(&chunks, file_chars);
    
    if visualize {
        // Visualization mode: show full chunk contents
        let lines: Vec<&str> = source.lines().collect();
        
        for (i, chunk) in chunks.iter().enumerate() {
            let line_count = chunk.end_line - chunk.start_line + 1;
            let size = chunk.text.len();
            
            println!("-- [CHUNK {}] [{} lines] [{} chars] --", i + 1, line_count, size);
            
            for line_num in chunk.start_line..=chunk.end_line {
                if line_num > 0 && line_num <= lines.len() {
                    println!("{}", lines[line_num - 1]);
                }
            }
            println!();
        }
        
        println!("=== QUALITY SCORE ===");
        println!("Score: {:.1}%", report.score);
        println!("Total chunks: {}", chunks.len());
        println!("Small chunks (<{} chars): {}", SMALL_CHUNK_CHARS, report.small_chunks);
        println!("Chars: {}-{} (mean {:.0})", report.min_chars, report.max_chars, report.mean_chars);
    } else {
        // Default mode: show summary with previews
        println!("Total chunks: {}", chunks.len());
        println!("Target size: {} chars", target_size);
        println!();
        
        let mut total_chars = 0;
        let mut oversized = 0;
        let mut undersized = 0;
        
        for (i, chunk) in chunks.iter().enumerate() {
            let text_size = chunk.text.len();
            let total_size = chunk.breadcrumb.len() + chunk.text.len();
            total_chars += total_size;
            
            if text_size > target_size {
                oversized += 1;
            } else if text_size < 200 {
                undersized += 1;
            }
            
            println!("━━━━━ Chunk {} ━━━━━", i + 1);
            println!("Breadcrumb: {}", chunk.breadcrumb);
            println!("Type: {}", chunk.chunk_type);
            if let Some(symbol) = &chunk.symbol_name {
                println!("Symbol: {}", symbol);
            }
            println!("Lines: {}-{}", chunk.start_line, chunk.end_line);
            println!("Size: {} chars (text: {}, breadcrumb: {})", 
                total_size, text_size, chunk.breadcrumb.len());
            if text_size > target_size {
                println!("⚠️  OVERSIZED (target: {}, actual: {})", target_size, text_size);
            } else if text_size < 200 {
                println!("⚡ Small chunk");
            }
            println!();
            println!("Preview (first 8 lines):");
            for line in chunk.text.lines().take(8) {
                println!("  {}", line);
            }
            if chunk.text.lines().count() > 8 {
                println!("  ... ({} more lines)", chunk.text.lines().count() - 8);
            }
            println!();
        }
        
        println!("━━━━━ Summary ━━━━━");
        println!("Total chunks: {}", chunks.len());
        println!("Total chars: {}", total_chars);
        println!("Average size: {:.0} chars", total_chars as f64 / chunks.len() as f64);
        println!("Oversized chunks (>{}): {}", target_size, oversized);
        println!("Small chunks (<200): {}", undersized);
        println!("Quality score: {:.1}%", report.score);
        println!("  Small chunks (<{} chars): {}", SMALL_CHUNK_CHARS, report.small_chunks);
    }
    
    Ok(())
}

/// Audit chunking quality across multiple files
fn run_audit_chunks(count: usize, dir: String) -> anyhow::Result<()> {
    use rand::seq::IndexedRandom;
    
    println!("📊 Sampling {} TypeScript files from: {}", count, dir);
    println!();
    
    // Collect all TypeScript files
    let ts_files: Vec<PathBuf> = walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            let ext = path.extension().map(|s| s.to_string_lossy()).unwrap_or_default();
            ext == "ts" && !path.to_string_lossy().contains("node_modules")
        })
        .map(|e| e.path().to_owned())
        .collect();
    
    println!("Found {} TypeScript files", ts_files.len());
    
    if ts_files.is_empty() {
        return Err(anyhow::anyhow!("No TypeScript files found"));
    }
    
    // Random sample
    let mut rng = rand::rng();
    let sample: Vec<_> = ts_files
        .choose_multiple(&mut rng, count)
        .into_iter()
        .collect();
    
    // Compute quality scores using AST-only mode (allow_fallback=false)
    // This measures how well the AST-based chunker performs, without fallback
    // masking the quality of split decisions.
    let mut results: Vec<_> = sample
        .into_iter()
        .filter_map(|path| {
            let source = std::fs::read_to_string(&path).ok()?;
            let file_name = path.file_name()?.to_string_lossy().to_string();
            let config = PartitionConfig {
                file_name,
                package_name: "n/a".to_string(),
                allow_fallback: false,  // AST-only mode for accurate quality measurement
                ..Default::default()
            };
            let chunks = partition_typescript(&source, &config, path.to_str().unwrap(), "n/a");
            let file_chars = source.len();
            let report = ChunkQualityReport::from_chunks(&chunks, file_chars);
            Some((path, report, chunks))
        })
        .collect();
    
    // Sort by score (worst first - ascending since higher is better)
    results.sort_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap());
    
    println!("\n=== Quality Scores (worst first) ===\n");
    for (i, (path, report, _)) in results.iter().enumerate() {
        let rel_path = path.strip_prefix(&dir).unwrap_or(path);
        println!("{}. {} {}", i + 1, report.format(), rel_path.display());
    }
    
    // Show top 3 worst for investigation
    println!("\n=== Top 3 Worst Files ===\n");
    for (path, report, chunks) in results.iter().take(3) {
        let rel_path = path.strip_prefix(&dir).unwrap_or(path);
        println!("--- {} ---", rel_path.display());
        println!("{}", report.format());
        
        // Show chunk breakdown
        for (i, chunk) in chunks.iter().enumerate() {
            let lines = chunk.end_line - chunk.start_line + 1;
            let tiny_marker = if lines < 20 { " [TINY]" } else { "" };
            println!(
                "  Chunk {}: {} lines ({}-{}) {} - {}",
                i + 1,
                lines,
                chunk.start_line,
                chunk.end_line,
                tiny_marker,
                chunk.breadcrumb
            );
        }
        println!();
    }
    
    Ok(())
}
