//! Reusable indexing engine for Qdrant semantic search
//! 
//! This module contains general-purpose indexing logic that works
//! for any Rush monorepo. It is designed to be reusable across projects.
//! 
//! Repository-specific configuration lives in `../config.rs`

pub mod config;
pub mod parallel_embedder;
pub mod chunker;
pub mod partitioner;
pub mod markdown_partitioner;
pub mod uploader;
pub mod util;
pub mod package_lookup;

// Re-export commonly used types for convenience
pub use chunker::Chunk;
pub use parallel_embedder::ParallelEmbedder;
pub use partitioner::SMALL_CHUNK_CHARS;
pub use uploader::QdrantUploader;
