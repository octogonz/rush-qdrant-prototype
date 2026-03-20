//! Qdrant client for batch uploading embeddings
//! 
//! This module handles HTTP communication with Qdrant to upload
//! chunks with their embeddings for semantic search.

use anyhow::{anyhow, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const DEFAULT_QDRANT_URL: &str = "http://localhost:6333";

/// Qdrant filter for queries
#[derive(Debug, Serialize)]
struct Filter {
    must: Vec<Condition>,
}

#[derive(Debug, Serialize)]
struct Condition {
    key: String,
    r#match: MatchValue,
}

#[derive(Debug, Serialize)]
struct MatchValue {
    value: String,
}

/// Request body for filter-based operations (delete, etc.)
#[derive(Debug, Serialize)]
struct FilterRequest {
    filter: Filter,
}

/// Qdrant client for uploading embeddings
pub struct QdrantUploader {
    client: Client,
    url: String,
    collection: String,
}

/// Request body for Qdrant upsert operation
#[derive(Debug, Serialize)]
struct UpsertRequest {
    points: Vec<Point>,
}

/// A single point in Qdrant
#[derive(Debug, Serialize)]
struct Point {
    id: String,  // Random UUID
    vector: Vec<f32>,
    payload: PointPayload,
}

/// Payload associated with a point
#[derive(Debug, Serialize, Deserialize)]
pub struct PointPayload {
    pub text: String,
    pub source_uri: String,
    pub source_type: String,
    pub catalog: String,
    pub content_hash: String,
    pub start_line: usize,
    pub end_line: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_name: Option<String>,
    pub chunk_type: String,
    pub chunk_kind: String,  // "content" | "imports" | "changelog" | "config"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub breadcrumb: Option<String>,
    // Phase 6+ fields
    pub file_id: String,        // 16-char hex hash of relative path
    pub relative_path: String,  // Path relative to catalog base
    pub chunk_number: usize,    // 1-indexed position in file
    pub chunk_count: usize,     // Total chunks in file
    // File completion tracking (Phase 10)
    #[serde(default)]
    pub file_complete: bool,    // true on chunk #1 when all chunks uploaded
}

/// Information about a file for incremental sync
#[derive(Debug, Clone)]
pub struct FileSyncInfo {
    pub content_hash: String,
    pub file_complete: bool,
}

/// Response from Qdrant upsert
#[derive(Debug, Deserialize)]
struct UpsertResponse {
    result: UpsertResult,
    status: String,
}

#[derive(Debug, Deserialize)]
struct UpsertResult {
    operation_id: u64,
}

/// Response from scroll (list points)
#[derive(Debug, Deserialize)]
struct ScrollResponse {
    result: ScrollResult,
    #[allow(dead_code)]
    status: String,
}

#[derive(Debug, Deserialize)]
struct ScrollResult {
    points: Vec<ScrollPoint>,
    #[serde(default)]
    next_page_offset: Option<QdrantId>,
}

/// Scroll point from Qdrant
#[derive(Debug, Deserialize)]
struct ScrollPoint {
    #[allow(dead_code)]
    id: QdrantId,
    payload: PointPayload,
}

/// Qdrant ID can be either a string (UUID) or integer (custom ID)
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum QdrantId {
    String(String),
    Integer(u64),
}

impl std::ops::Shr<i32> for QdrantId {
    type Output = u64;
    
    fn shr(self, rhs: i32) -> Self::Output {
        match self {
            QdrantId::Integer(n) => n >> rhs,
            QdrantId::String(_) => 0,
        }
    }
}

impl std::ops::Shr<i32> for &QdrantId {
    type Output = u64;
    
    fn shr(self, rhs: i32) -> Self::Output {
        match self {
            QdrantId::Integer(n) => *n >> rhs,
            QdrantId::String(_) => 0,
        }
    }
}

/// Response from delete
#[derive(Debug, Deserialize)]
struct DeleteResponse {
    result: DeleteResult,
    #[allow(dead_code)]
    status: String,
}

#[derive(Debug, Deserialize)]
struct DeleteResult {
    operation_id: u64,
}

/// Response from Qdrant search
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct SearchResponse {
    result: Vec<SearchResult>,
    status: String,
}

/// Search result from Qdrant
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct SearchResult {
    pub id: QdrantId,
    pub score: f32,
    pub payload: PointPayload,
}

impl QdrantUploader {
    /// Creates a new Qdrant uploader
    pub fn new(collection: &str, qdrant_url: Option<&str>) -> Result<Self> {
        let url = qdrant_url.unwrap_or(DEFAULT_QDRANT_URL).to_string();
        
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()?;

        Ok(Self {
            client,
            url,
            collection: collection.to_string(),
        })
    }

    /// Delete all points for a specific catalog
    #[allow(dead_code)]
    pub fn delete_catalog(&self, catalog: &str) -> Result<u64> {
        let endpoint = format!("{}/collections/{}/points/delete", self.url, self.collection);

        let request_body = FilterRequest {
            filter: Filter {
                must: vec![
                    Condition {
                        key: "catalog".to_string(),
                        r#match: MatchValue { value: catalog.to_string() },
                    }
                ],
            },
        };

        let response = self.client.post(&endpoint).json(&request_body).send()?;

        if !response.status().is_success() {
            return Err(anyhow!("Failed to delete catalog: HTTP {}", response.status()));
        }

        let delete_response: DeleteResponse = response.json()?;
        Ok(delete_response.result.operation_id)
    }

    /// Delete all points for a specific file
    pub fn delete_file(&self, file_path: &str, catalog: &str) -> Result<u64> {
        let endpoint = format!("{}/collections/{}/points/delete", self.url, self.collection);

        let request_body = FilterRequest {
            filter: Filter {
                must: vec![
                    Condition {
                        key: "catalog".to_string(),
                        r#match: MatchValue { value: catalog.to_string() },
                    },
                    Condition {
                        key: "source_uri".to_string(),
                        r#match: MatchValue { value: file_path.to_string() },
                    },
                ],
            },
        };

        let response = self.client.post(&endpoint).json(&request_body).send()?;

        if !response.status().is_success() {
            return Err(anyhow!("Failed to delete file: HTTP {}", response.status()));
        }

        let delete_response: DeleteResponse = response.json()?;
        Ok(delete_response.result.operation_id)
    }

    /// Get all points for a specific catalog
    /// Returns a map of file path → FileSyncInfo (content_hash and file_complete)
    /// Only queries chunk #1 per file for efficiency
    pub fn get_catalog_files(&self, catalog: &str) -> Result<std::collections::HashMap<String, FileSyncInfo>> {
        let mut files = std::collections::HashMap::new();
        let mut offset: Option<QdrantId> = None;
        const LIMIT: u32 = 1000;

        loop {
            let endpoint = format!(
                "{}/collections/{}/points/scroll?limit={}",
                self.url, self.collection, LIMIT
            );

            // Build filter with catalog AND chunk_number=1
            #[derive(Debug, Serialize)]
            struct ScrollRequestWithIntFilter {
                filter: FilterWithIntCondition,
                with_payload: bool,
                #[serde(skip_serializing_if = "Option::is_none")]
                offset: Option<QdrantId>,
            }

            #[derive(Debug, Serialize)]
            struct FilterWithIntCondition {
                must: Vec<serde_json::Value>,
            }

            let must_values = vec![
                serde_json::json!({
                    "key": "catalog",
                    "match": { "value": catalog }
                }),
                serde_json::json!({
                    "key": "chunk_number",
                    "match": { "value": 1 }
                }),
            ];

            let request_body = ScrollRequestWithIntFilter {
                filter: FilterWithIntCondition { must: must_values },
                with_payload: true,
                offset: offset.clone(),
            };

            let response = self.client.post(&endpoint).json(&request_body).send()?;

            if !response.status().is_success() {
                return Err(anyhow!("Failed to scroll catalog: HTTP {}", response.status()));
            }

            let response_text = response.text()?;
            let scroll_response: ScrollResponse = match serde_json::from_str(&response_text) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to deserialize Qdrant response: {}", e);
                    eprintln!("Raw response (first 2000 chars): {}", &response_text.chars().take(2000).collect::<String>());
                    return Err(anyhow!("Deserialization error: {}", e));
                }
            };

            if scroll_response.result.points.is_empty() {
                break;
            }

            for point in scroll_response.result.points {
                files.insert(
                    point.payload.source_uri.clone(),
                    FileSyncInfo {
                        content_hash: point.payload.content_hash.clone(),
                        file_complete: point.payload.file_complete,
                    },
                );
            }

            offset = scroll_response.result.next_page_offset;
            if offset.is_none() {
                break;
            }
        }

        Ok(files)
    }

    /// Uploads a batch of chunks with their embeddings
    pub fn upload_batch(&self, chunks: &[(crate::engine::Chunk, Vec<f32>)]) -> Result<u64> {
        if chunks.is_empty() {
            return Ok(0);
        }

        let points: Vec<Point> = chunks
            .iter()
            .map(|(chunk, embedding)| {
                Point {
                    id: Uuid::new_v4().to_string(),  // Random UUID
                    vector: embedding.clone(),
                    payload: PointPayload {
                        text: chunk.text.clone(),
                        source_uri: chunk.source_uri.clone(),
                        source_type: chunk.source_type.clone(),
                        catalog: chunk.catalog.clone(),
                        content_hash: chunk.content_hash.clone(),
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        symbol_name: chunk.symbol_name.clone(),
                        chunk_type: chunk.chunk_type.clone(),
                        chunk_kind: chunk.chunk_kind.clone(),
                        breadcrumb: Some(chunk.breadcrumb.clone()),
                        // Phase 6+ fields
                        file_id: super::util::display_file_id(chunk.file_id),
                        relative_path: chunk.relative_path.clone(),
                        chunk_number: chunk.chunk_number,
                        chunk_count: chunk.chunk_count,
                        // Phase 10: file completion tracking
                        file_complete: false,
                    },
                }
            })
            .collect();

        let request_body = UpsertRequest { points };

        let endpoint = format!("{}/collections/{}/points", self.url, self.collection);
        
        let response = self
            .client
            .put(&endpoint)
            .json(&request_body)
            .send()?
            .json::<UpsertResponse>()?;

        if response.status != "ok" {
            return Err(anyhow!("Qdrant upsert failed with status: {}", response.status));
        }

        Ok(response.result.operation_id)
    }

    /// Mark a file as complete by setting file_complete=true on chunk #1
    ///
    /// This is called after all chunks for a file have been uploaded.
    /// Uses Qdrant's payload update API to set the field without rewriting the point.
    ///
    /// The `catalog` parameter is required to disambiguate when multiple catalogs
    /// share the same file structure (e.g., multiple clones of the same repo).
    pub fn mark_file_complete(&self, file_id: &str, catalog: &str) -> Result<()> {
        // First, find the point ID for chunk #1 of this file in the correct catalog
        #[derive(Debug, Serialize)]
        struct ScrollRequestForId {
            filter: FilterForId,
            with_payload: bool,
            limit: u32,
        }

        #[derive(Debug, Serialize)]
        struct FilterForId {
            must: Vec<serde_json::Value>,
        }

        let must_values = vec![
            serde_json::json!({
                "key": "file_id",
                "match": { "value": file_id }
            }),
            serde_json::json!({
                "key": "chunk_number",
                "match": { "value": 1 }
            }),
            serde_json::json!({
                "key": "catalog",
                "match": { "value": catalog }
            }),
        ];

        let endpoint = format!(
            "{}/collections/{}/points/scroll?limit=1",
            self.url, self.collection
        );

        let request_body = ScrollRequestForId {
            filter: FilterForId { must: must_values },
            with_payload: false,
            limit: 1,
        };

        let response = self.client.post(&endpoint).json(&request_body).send()?;

        if !response.status().is_success() {
            return Err(anyhow!("Failed to find chunk #1 for file {}: HTTP {}", file_id, response.status()));
        }

        #[derive(Debug, Deserialize)]
        struct ScrollIdResponse {
            result: ScrollIdResult,
        }

        #[derive(Debug, Deserialize)]
        struct ScrollIdResult {
            points: Vec<ScrollIdPoint>,
        }

        #[derive(Debug, Deserialize)]
        struct ScrollIdPoint {
            id: QdrantId,
        }

        let scroll_response: ScrollIdResponse = response.json()?;

        let point_id = scroll_response
            .result
            .points
            .first()
            .ok_or_else(|| anyhow!("No chunk #1 found for file {}", file_id))?;

        // Now update the payload using Qdrant's set_payload API
        // The point ID should be passed as a plain string (UUID) or integer
        let point_id_str = match &point_id.id {
            QdrantId::String(s) => s.clone(),
            QdrantId::Integer(n) => n.to_string(),
        };

        #[derive(Debug, Serialize)]
        struct SetPayloadRequest {
            payload: std::collections::HashMap<String, serde_json::Value>,
            points: Vec<String>,
        }

        let mut payload = std::collections::HashMap::new();
        payload.insert("file_complete".to_string(), serde_json::json!(true));

        let request_body = SetPayloadRequest {
            payload,
            points: vec![point_id_str],
        };

        let endpoint = format!(
            "{}/collections/{}/points/payload",
            self.url, self.collection
        );

        let response = self.client.post(&endpoint).json(&request_body).send()?;

        if !response.status().is_success() {
            return Err(anyhow!("Failed to set file_complete for {}: HTTP {}", file_id, response.status()));
        }

        Ok(())
    }

    /// Queries the collection with an embedding
    pub fn query(&self, embedding: &[f32], limit: usize, catalog: Option<&str>) -> Result<Vec<SearchResult>> {
        #[derive(Debug, Serialize)]
        struct SearchRequest {
            vector: Vec<f32>,
            limit: usize,
            with_payload: bool,
            filter: Option<Filter>,
        }

        let filter = catalog.map(|cat| Filter {
            must: vec![Condition {
                key: "catalog".to_string(),
                r#match: MatchValue { value: cat.to_string() },
            }],
        });

        let request_body = SearchRequest {
            vector: embedding.to_vec(),
            limit,
            with_payload: true,
            filter,
        };

        let endpoint = format!("{}/collections/{}/points/search", self.url, self.collection);
        
        let response = self
            .client
            .post(&endpoint)
            .json(&request_body)
            .send()?
            .json::<SearchResponse>()?;

        Ok(response.result)
    }

    /// Get a single point by ID (legacy - uses old hash-based IDs)
    #[allow(dead_code)]
    pub fn get_point(&self, id: u64) -> Result<Option<PointResult>> {
        #[derive(Debug, Serialize)]
        struct PointRequest {
            ids: Vec<u64>,
            with_payload: bool,
        }

        let request_body = PointRequest {
            ids: vec![id],
            with_payload: true,
        };

        let endpoint = format!("{}/collections/{}/points", self.url, self.collection);
        
        let response = self
            .client
            .post(&endpoint)
            .json(&request_body)
            .send()?;
        
        if !response.status().is_success() {
            return Ok(None);
        }

        #[derive(Debug, Deserialize)]
        struct PointResponse {
            result: Vec<PointResult>,
        }

        let point_response: PointResponse = response.json()?;
        Ok(point_response.result.into_iter().next())
    }

    /// Get chunks by file_id with optional selector (Phase 7+)
    /// 
    /// # Arguments
    /// * `file_id` - 16-char hex file ID
    /// * `selector` - Which chunks to retrieve
    /// 
    /// # Returns
    /// Vector of points sorted by chunk_number, or error
    pub fn get_chunks_by_file_id(&self, file_id: &str) -> Result<Vec<PointResult>> {
        // Build scroll request with filter on file_id
        #[derive(Debug, Serialize)]
        struct ScrollRequestWithRange {
            filter: FilterWithRange,
            with_payload: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            offset: Option<QdrantId>,
        }

        #[derive(Debug, Serialize)]
        struct FilterWithRange {
            must: Vec<serde_json::Value>,
        }

        // Build file_id condition
        let must_values = vec![
            serde_json::json!({
                "key": "file_id",
                "match": { "value": file_id }
            })
        ];

        let mut results: Vec<PointResult> = Vec::new();
        let mut offset: Option<QdrantId> = None;
        const LIMIT: u32 = 100;

        loop {
            let endpoint = format!(
                "{}/collections/{}/points/scroll?limit={}",
                self.url, self.collection, LIMIT
            );

            let request_body = ScrollRequestWithRange {
                filter: FilterWithRange { must: must_values.clone() },
                with_payload: true,
                offset: offset.clone(),
            };

            let response = self.client.post(&endpoint).json(&request_body).send()?;

            if !response.status().is_success() {
                return Err(anyhow!("Failed to scroll file chunks: HTTP {}", response.status()));
            }

            let response_text = response.text()?;
            let scroll_response: ScrollResponse = match serde_json::from_str(&response_text) {
                Ok(r) => r,
                Err(e) => {
                    return Err(anyhow!("Deserialization error: {}", e));
                }
            };

            if scroll_response.result.points.is_empty() {
                break;
            }

            for point in scroll_response.result.points {
                results.push(PointResult {
                    id: point.id,
                    payload: point.payload,
                });
            }

            offset = scroll_response.result.next_page_offset;
            if offset.is_none() {
                break;
            }
        }

        // Sort by chunk_number
        results.sort_by_key(|p| p.payload.chunk_number);

        Ok(results)
    }
}

/// A point retrieved by ID (no score, unlike SearchResult)
#[derive(Debug, Deserialize)]
pub struct PointResult {
    pub id: QdrantId,
    pub payload: PointPayload,
}
