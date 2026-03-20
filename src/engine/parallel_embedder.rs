//! Parallel embedding generation using multiple ONNX sessions
//!
//! This module implements a pool of ONNX sessions for parallel embedding generation.
//! Each session runs with limited intra-op threads, allowing multiple sessions to
//! run concurrently without oversubscribing CPU cores.
//!
//! Based on benchmark findings:
//! - 4 parallel sessions × 3 intra-op threads = 12 cores utilized
//! - ~12ms per embedding (3.5x faster than single session)
//! - Individual processing (no batching) is faster on CPU
//!
//! GPU mode (set RUSH_QDRANT_GPU=1):
//! - Uses CUDA execution provider with a single session
//! - Set RUSH_QDRANT_MODEL to "fp16" or "int8" for alternate models

use anyhow::Result;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use tokenizers::Tokenizer;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::sync::{Arc, Mutex};

const MODEL_ID: &str = "jinaai/jina-embeddings-v2-base-code";
const MAX_LENGTH: usize = 8192;
const HIDDEN_SIZE: usize = 768;

/// Configuration for parallel embedding
pub struct ParallelConfig {
    /// Number of worker sessions (default: 4)
    pub num_workers: usize,
    /// Threads per session for intra-op parallelism (default: 3)
    pub intra_threads: usize,
    /// Use CUDA GPU execution provider
    pub use_cuda: bool,
    /// Model variant: "fp32", "fp16", or "int8"
    pub model_variant: String,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        let total_cores = num_cpus::get();
        let use_cuda = std::env::var("RUSH_QDRANT_GPU").unwrap_or_default() == "1";
        let model_variant = std::env::var("RUSH_QDRANT_MODEL").unwrap_or_else(|_| "fp32".to_string());
        let num_workers = if use_cuda { 1 } else { 4 };
        let intra_threads = if use_cuda { 1 } else { (total_cores / num_workers).max(1) };

        Self {
            num_workers,
            intra_threads,
            use_cuda,
            model_variant,
        }
    }
}

/// A pool of ONNX sessions for parallel embedding generation
pub struct ParallelEmbedder {
    // Each worker has its own session and tokenizer (both need &mut for encoding)
    workers: Vec<Arc<Mutex<(Session, Tokenizer)>>>,
}

impl ParallelEmbedder {
    /// Create a new parallel embedder with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ParallelConfig::default())
    }

    /// Create a new parallel embedder with custom configuration
    pub fn with_config(config: ParallelConfig) -> Result<Self> {
        // Download model files from HuggingFace (cached locally after first download)
        let api = Api::new()?;
        let repo = Repo::new(MODEL_ID.to_string(), RepoType::Model);
        let api = api.repo(repo);

        let tokenizer_path = api.get("tokenizer.json")?;

        // Select model file based on variant
        let model_file = match config.model_variant.as_str() {
            "fp16" => "onnx/model_fp16.onnx",
            "int8" => "onnx/model_quantized.onnx",
            _ => "onnx/model.onnx",
        };
        println!("Model variant: {} ({})", config.model_variant, model_file);
        let onnx_path = api.get(model_file)?;

        // Load base tokenizer (will be cloned for each worker)
        let base_tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        if config.use_cuda {
            println!("GPU mode: CUDA execution provider, {} worker(s)", config.num_workers);
        } else {
            println!("CPU mode: {} workers × {} threads each",
                config.num_workers, config.intra_threads);
        }

        // Create worker pool - each worker gets its own session AND tokenizer
        // This avoids lock contention on the tokenizer during parallel encoding
        let workers: Vec<Arc<Mutex<(Session, Tokenizer)>>> = (0..config.num_workers)
            .map(|i| {
                let mut builder = Session::builder()
                    .expect("Failed to create session builder");

                builder = builder
                    .with_optimization_level(GraphOptimizationLevel::All)
                    .expect("Failed to set optimization level");

                if config.use_cuda {
                    use ort::ep::CUDA;
                    let use_cuda_graph = std::env::var("RUSH_QDRANT_CUDA_GRAPH").unwrap_or_default() == "1";
                    let cuda_ep = CUDA::default()
                        .with_memory_limit(20 * 1024 * 1024 * 1024)
                        .with_cuda_graph(use_cuda_graph)
                        .build();
                    builder = builder
                        .with_execution_providers([cuda_ep])
                        .expect("Failed to register CUDA execution provider");
                    println!("  Worker {}: CUDA EP registered (arena: 20GB{})", i,
                        if use_cuda_graph { ", CUDA Graph ON" } else { "" });
                } else {
                    builder = builder
                        .with_intra_threads(config.intra_threads)
                        .expect("Failed to set intra threads");
                }

                let session = builder
                    .commit_from_file(&onnx_path)
                    .expect("Failed to commit session");

                // Clone tokenizer for this worker
                let tokenizer = base_tokenizer.clone();

                if i == 0 && !config.use_cuda {
                    println!("Worker pool created: {} workers × {} threads = {} total threads",
                        config.num_workers, config.intra_threads,
                        config.num_workers * config.intra_threads);
                }

                Arc::new(Mutex::new((session, tokenizer)))
            })
            .collect();

        Ok(Self { workers })
    }

    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Encode a batch of texts in a single ONNX inference call.
    /// Pads all sequences to the max length within the batch.
    /// Returns one embedding per input text.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.encode_batch_padded(texts, 0)
    }

    /// Encode a batch of texts, padding to a fixed length (bucket ceiling).
    /// If pad_to == 0, pads to the max length within the batch.
    /// A fixed pad_to enables CUDA Graphs (same tensor shapes across calls).
    pub fn encode_batch_padded(&self, texts: &[&str], pad_to: usize) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let worker = &self.workers[0];
        let mut guard = worker.lock().unwrap();
        let (session, tokenizer) = &mut *guard;

        // Tokenize all texts
        let encodings: Vec<_> = texts.iter()
            .map(|text| tokenizer.encode(*text, true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e)))
            .collect::<Result<Vec<_>>>()?;

        // Pad to bucket ceiling or batch max
        let max_len = if pad_to > 0 {
            pad_to
        } else {
            encodings.iter()
                .map(|e| e.get_ids().len().min(MAX_LENGTH))
                .max()
                .unwrap_or(0)
        };

        let batch_size = texts.len();

        // Build padded input tensors [batch_size, max_len]
        let mut input_ids_flat: Vec<i64> = vec![0; batch_size * max_len];
        let mut attention_mask_flat: Vec<i64> = vec![0; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let seq_len = ids.len().min(MAX_LENGTH).min(max_len);

            for j in 0..seq_len {
                input_ids_flat[i * max_len + j] = ids[j] as i64;
                attention_mask_flat[i * max_len + j] = mask[j] as i64;
            }
        }

        // Run batched inference
        let outputs = session.run(ort::inputs![
            "input_ids" => Tensor::from_array(([batch_size, max_len], input_ids_flat))?,
            "attention_mask" => Tensor::from_array(([batch_size, max_len], attention_mask_flat))?,
        ])?;

        // Extract output: shape [batch_size, max_len, HIDDEN_SIZE]
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        // Mean pooling per sequence (only over non-padded positions)
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let seq_len = encodings[i].get_ids().len().min(MAX_LENGTH).min(max_len);
            let embedding: Vec<f32> = (0..HIDDEN_SIZE)
                .map(|h| {
                    (0..seq_len)
                        .map(|j| data[i * max_len * HIDDEN_SIZE + j * HIDDEN_SIZE + h])
                        .sum::<f32>() / seq_len as f32
                })
                .collect();
            results.push(embedding);
        }

        Ok(results)
    }

    /// Encode a single text using a specific worker (for parallel processing)
    ///
    /// Call this from parallel iterator, passing worker_index = chunk_index % num_workers
    pub fn encode(&self, text: &str, worker_index: usize) -> Result<Vec<f32>> {
        let worker = &self.workers[worker_index % self.workers.len()];
        let mut guard = worker.lock().unwrap();
        let (session, tokenizer) = &mut *guard;

        // Tokenize with this worker's tokenizer
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Truncate if needed
        let seq_len = ids.len().min(MAX_LENGTH);

        // Create input tensors
        let input_ids: Vec<i64> = ids[..seq_len].iter().map(|&id| id as i64).collect();
        let attention_mask_data: Vec<i64> = attention_mask[..seq_len].iter().map(|&m| m as i64).collect();

        // Run inference
        let outputs = session.run(ort::inputs![
            "input_ids" => Tensor::from_array(([1, seq_len], input_ids))?,
            "attention_mask" => Tensor::from_array(([1, seq_len], attention_mask_data))?,
        ])?;

        // Extract output tensor
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        // Mean pooling over sequence dimension
        let embedding: Vec<f32> = (0..HIDDEN_SIZE)
            .map(|i| {
                (0..seq_len)
                    .map(|j| data[j * HIDDEN_SIZE + i])
                    .sum::<f32>() / seq_len as f32
            })
            .collect();

        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_parallel_encode() {
        let embedder = ParallelEmbedder::new().unwrap();
        let embedding = embedder.encode("function test() { return 42; }", 0);
        assert!(embedding.is_ok());
        let emb = embedding.unwrap();
        assert_eq!(emb.len(), 768);
    }

    #[test]
    fn test_parallel_performance() {
        let embedder = ParallelEmbedder::new().unwrap();
        let texts: Vec<&str> = (0..100).map(|_| "function test() { return 42; }").collect();

        use rayon::prelude::*;

        let start = Instant::now();
        let embeddings: Vec<_> = texts
            .par_iter()
            .enumerate()
            .map(|(i, text)| embedder.encode(text, i))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let elapsed = start.elapsed();

        println!("Embedded {} chunks in {:?}", embeddings.len(), elapsed);
        println!("Per embedding: {:?}", elapsed / embeddings.len() as u32);

        assert_eq!(embeddings.len(), 100);
        for emb in &embeddings {
            assert_eq!(emb.len(), 768);
        }
    }
}
