//! MCP server for semantic search over indexed catalogs
//!
//! Implements the StreamableHTTP MCP transport with JSON-RPC messaging.
//! Exposes `semantic_search` and `view_chunks` tools.

use actix_web::{web, App, HttpServer, HttpResponse};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::ParallelEmbedder;
use crate::engine::QdrantUploader;
use crate::{Config, CatalogConfig, parse_file_id_with_selector, ChunkSelector};
use crate::watcher;

/// Shared state for the MCP server
pub struct McpState {
    pub embedder: Arc<ParallelEmbedder>,
    pub collection: String,
    pub qdrant_url: Option<String>,
    pub catalogs: HashMap<String, CatalogConfig>,
}

/// JSON-RPC request envelope
#[derive(Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    #[serde(default)]
    id: Option<serde_json::Value>,
    method: String,
    #[serde(default)]
    params: Option<serde_json::Value>,
}

/// JSON-RPC response envelope
#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

/// JSON-RPC error object
#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

impl JsonRpcResponse {
    fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Option<serde_json::Value>, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError { code, message }),
        }
    }
}

/// Handle POST /mcp — main JSON-RPC endpoint
async fn handle_post(
    state: web::Data<McpState>,
    body: web::Json<JsonRpcRequest>,
) -> HttpResponse {
    let req = body.into_inner();

    match req.method.as_str() {
        "initialize" => {
            let result = serde_json::json!({
                "protocolVersion": "2025-03-26",
                "serverInfo": {
                    "name": "rush-qdrant",
                    "version": "0.1.0"
                },
                "capabilities": {
                    "tools": {}
                }
            });
            HttpResponse::Ok()
                .insert_header(("Content-Type", "application/json"))
                .json(JsonRpcResponse::success(req.id, result))
        }

        "notifications/initialized" => HttpResponse::Accepted().finish(),

        "tools/list" => {
            let catalogs: Vec<String> = state.catalogs.keys().cloned().collect();
            let catalog_desc = if catalogs.len() == 1 {
                format!("Searches the '{}' catalog.", catalogs[0])
            } else {
                format!(
                    "Available catalogs: {}. Omit to search all.",
                    catalogs.join(", ")
                )
            };

            let tools = serde_json::json!({
                "tools": [
                    {
                        "name": "semantic_search",
                        "description": format!(
                            "Search indexed code and documentation using semantic similarity. \
                             Returns file IDs, similarity scores, breadcrumbs, and code previews. {}",
                            catalog_desc
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Natural language search query"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of results (default: 10)",
                                    "default": 10
                                },
                                "catalog": {
                                    "type": "string",
                                    "description": "Filter to a specific catalog (optional)"
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "view_chunks",
                        "description": "Retrieve full chunk content by file ID. Use IDs from \
                                        semantic_search results. Selectors: '700a4ba232fe9ddc' \
                                        (all chunks), '700a4ba232fe9ddc:3' (chunk 3), \
                                        '700a4ba232fe9ddc:2-3' (range), '700a4ba232fe9ddc:3-end' \
                                        (to end).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "ids": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "File IDs with optional chunk selectors"
                                },
                                "full_paths": {
                                    "type": "boolean",
                                    "description": "Show full filesystem paths (default: false)",
                                    "default": false
                                }
                            },
                            "required": ["ids"]
                        }
                    }
                ]
            });

            HttpResponse::Ok()
                .insert_header(("Content-Type", "application/json"))
                .json(JsonRpcResponse::success(req.id, tools))
        }

        "tools/call" => {
            let params = req.params.unwrap_or(serde_json::json!({}));
            let tool_name = params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));

            // Log tool calls for usage tracking
            match tool_name {
                "semantic_search" => {
                    let query = arguments.get("query").and_then(|v| v.as_str()).unwrap_or("?");
                    let catalog = arguments.get("catalog").and_then(|v| v.as_str()).unwrap_or("*");
                    eprintln!("[query] semantic_search catalog={catalog} query={query:?}");
                    handle_search(state, req.id, arguments).await
                }
                "view_chunks" => {
                    let ids = arguments.get("ids").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                    eprintln!("[query] view_chunks ids={ids}");
                    handle_view(state, req.id, arguments).await
                }
                _ => HttpResponse::Ok()
                    .insert_header(("Content-Type", "application/json"))
                    .json(JsonRpcResponse::error(
                        req.id,
                        -32601,
                        format!("Unknown tool: {}", tool_name),
                    )),
            }
        }

        _ => {
            if req.id.is_some() {
                HttpResponse::Ok()
                    .insert_header(("Content-Type", "application/json"))
                    .json(JsonRpcResponse::error(
                        req.id,
                        -32601,
                        format!("Method not found: {}", req.method),
                    ))
            } else {
                // Notification — no response needed
                HttpResponse::Accepted().finish()
            }
        }
    }
}

/// Handle semantic_search tool call
async fn handle_search(
    state: web::Data<McpState>,
    id: Option<serde_json::Value>,
    args: serde_json::Value,
) -> HttpResponse {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;
    let catalog = args
        .get("catalog")
        .and_then(|v| v.as_str())
        .map(String::from);

    if query.is_empty() {
        return HttpResponse::Ok()
            .insert_header(("Content-Type", "application/json"))
            .json(JsonRpcResponse::error(
                id,
                -32602,
                "Missing required parameter: query".into(),
            ));
    }

    let embedder = state.embedder.clone();
    let collection = state.collection.clone();
    let qdrant_url = state.qdrant_url.clone();

    let result = web::block(move || -> anyhow::Result<String> {
        let embedding = embedder.encode(&query, 0)?;
        let uploader = QdrantUploader::new(&collection, qdrant_url.as_deref())?;
        let results = uploader.query(&embedding, limit, catalog.as_deref())?;

        let mut output = String::new();
        for r in &results {
            let breadcrumb = r.payload.breadcrumb.as_deref().unwrap_or("unknown");
            output.push_str(&format!(
                "{}:{}  {:.3}  {}\n",
                r.payload.file_id, r.payload.chunk_number, r.score, breadcrumb
            ));
            for line in r.payload.text.lines().take(3) {
                output.push_str(&format!("> {}\n", line));
            }
            output.push('\n');
        }

        if results.is_empty() {
            output.push_str("No results found.\n");
        }

        Ok(output)
    })
    .await;

    match result {
        Ok(Ok(text)) => {
            let content = serde_json::json!({
                "content": [{"type": "text", "text": text}]
            });
            HttpResponse::Ok()
                .insert_header(("Content-Type", "application/json"))
                .json(JsonRpcResponse::success(id, content))
        }
        Ok(Err(e)) => HttpResponse::Ok()
            .insert_header(("Content-Type", "application/json"))
            .json(JsonRpcResponse::error(
                id,
                -32603,
                format!("Search failed: {}", e),
            )),
        Err(e) => HttpResponse::Ok()
            .insert_header(("Content-Type", "application/json"))
            .json(JsonRpcResponse::error(
                id,
                -32603,
                format!("Internal error: {}", e),
            )),
    }
}

/// Handle view_chunks tool call
async fn handle_view(
    state: web::Data<McpState>,
    id: Option<serde_json::Value>,
    args: serde_json::Value,
) -> HttpResponse {
    let ids: Vec<String> = args
        .get("ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let full_paths = args
        .get("full_paths")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if ids.is_empty() {
        return HttpResponse::Ok()
            .insert_header(("Content-Type", "application/json"))
            .json(JsonRpcResponse::error(
                id,
                -32602,
                "Missing required parameter: ids".into(),
            ));
    }

    let collection = state.collection.clone();
    let qdrant_url = state.qdrant_url.clone();
    let catalogs = state.catalogs.clone();

    let result = web::block(move || -> anyhow::Result<String> {
        let uploader = QdrantUploader::new(&collection, qdrant_url.as_deref())?;
        let mut output = String::new();

        for spec in &ids {
            let (file_id, selector) = parse_file_id_with_selector(spec)?;
            let chunks = uploader.get_chunks_by_file_id(&file_id)?;

            let filtered: Vec<_> = match &selector {
                ChunkSelector::All => chunks,
                ChunkSelector::Single(n) => chunks
                    .into_iter()
                    .filter(|c| c.payload.chunk_number == *n)
                    .collect(),
                ChunkSelector::Range(start, end) => chunks
                    .into_iter()
                    .filter(|c| {
                        c.payload.chunk_number >= *start && c.payload.chunk_number <= *end
                    })
                    .collect(),
                ChunkSelector::ToEnd(start) => chunks
                    .into_iter()
                    .filter(|c| c.payload.chunk_number >= *start)
                    .collect(),
            };

            if filtered.is_empty() {
                let selector_str = match &selector {
                    ChunkSelector::All => String::new(),
                    ChunkSelector::Single(n) => format!(":{}", n),
                    ChunkSelector::Range(start, end) => format!(":{}-{}", start, end),
                    ChunkSelector::ToEnd(start) => format!(":{}-end", start),
                };
                output.push_str(&format!(
                    "{}{} ERROR: CHUNK NOT FOUND\n\n",
                    file_id, selector_str
                ));
                continue;
            }

            for r in &filtered {
                let breadcrumb = r.payload.breadcrumb.as_deref().unwrap_or("unknown");
                output.push_str(&format!(
                    "{}:{} ({}/{}) {}\n",
                    file_id,
                    r.payload.chunk_number,
                    r.payload.chunk_number,
                    r.payload.chunk_count,
                    breadcrumb
                ));
                output.push_str(&format!(
                    "Source: {}:{}\n",
                    r.payload.catalog, r.payload.relative_path
                ));
                if full_paths {
                    if let Some(cat_config) = catalogs.get(&r.payload.catalog) {
                        let full = format!("{}/{}", cat_config.path, r.payload.relative_path);
                        output.push_str(&format!("Full path: {}\n", full));
                    }
                }
                output.push_str(&format!(
                    "Lines: {}-{}\nType: {}\n\n",
                    r.payload.start_line, r.payload.end_line, r.payload.chunk_type
                ));
                for line in r.payload.text.lines() {
                    output.push_str(&format!("> {}\n", line));
                }
                output.push('\n');
            }
        }

        Ok(output)
    })
    .await;

    match result {
        Ok(Ok(text)) => {
            let content = serde_json::json!({
                "content": [{"type": "text", "text": text}]
            });
            HttpResponse::Ok()
                .insert_header(("Content-Type", "application/json"))
                .json(JsonRpcResponse::success(id, content))
        }
        Ok(Err(e)) => HttpResponse::Ok()
            .insert_header(("Content-Type", "application/json"))
            .json(JsonRpcResponse::error(
                id,
                -32603,
                format!("View failed: {}", e),
            )),
        Err(e) => HttpResponse::Ok()
            .insert_header(("Content-Type", "application/json"))
            .json(JsonRpcResponse::error(
                id,
                -32603,
                format!("Internal error: {}", e),
            )),
    }
}

/// Handle GET /mcp — SSE endpoint (not needed for request/response tools)
async fn handle_get() -> HttpResponse {
    HttpResponse::MethodNotAllowed().finish()
}

/// Handle DELETE /mcp — session termination
async fn handle_delete() -> HttpResponse {
    HttpResponse::Ok().finish()
}

/// Ensure the Qdrant collection exists, creating it if needed
fn ensure_collection(qdrant_url: &str, collection: &str) -> anyhow::Result<()> {
    let client = reqwest::blocking::Client::new();

    // Check if collection exists
    let url = format!("{}/collections/{}", qdrant_url, collection);
    let resp = client.get(&url).send()?;
    if resp.status().is_success() {
        return Ok(());
    }

    // Create collection with 768-dim cosine vectors
    eprintln!("Creating Qdrant collection '{}'...", collection);
    let body = serde_json::json!({
        "vectors": {
            "size": 768,
            "distance": "Cosine"
        }
    });
    let resp = client.put(&url).json(&body).send()?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        return Err(anyhow::anyhow!(
            "Failed to create collection '{}': HTTP {} - {}",
            collection,
            status,
            text
        ));
    }

    eprintln!("Collection '{}' created.", collection);
    Ok(())
}

/// Start the MCP server daemon
///
/// Loads the embedding model, runs initial indexing for all catalogs,
/// starts file watchers, and serves MCP tools over HTTP.
pub fn run_mcp(config: &Config, port: u16) -> anyhow::Result<()> {
    let qdrant_url = config
        .qdrant
        .url
        .as_deref()
        .unwrap_or("http://localhost:6333")
        .to_string();

    // Step 1: Ensure Qdrant collection exists
    ensure_collection(&qdrant_url, &config.qdrant.collection)?;

    // Step 2: Load embedding model (shared across all operations)
    eprintln!("Loading embedding model...");
    let embedder = Arc::new(ParallelEmbedder::new()?);
    eprintln!();

    // Step 3: Initial incremental crawl for each catalog
    for (name, catalog) in &config.catalogs {
        eprintln!("Initial index for catalog '{}'...", name);
        match watcher::run_incremental_crawl(
            name,
            catalog,
            &embedder,
            &config.qdrant.collection,
            Some(&qdrant_url),
        ) {
            Ok(stats) => {
                eprintln!(
                    "  {} new, {} changed, {} unchanged, {} deleted ({} chunks)",
                    stats.new_files,
                    stats.changed_files,
                    stats.unchanged_files,
                    stats.deleted_files,
                    stats.chunks_embedded
                );
            }
            Err(e) => {
                eprintln!("  Warning: initial crawl failed: {}", e);
            }
        }
    }

    // Step 3b: Release bulk-crawl embedder and recreate with small arena for queries.
    // This frees the large CUDA arena (~20GB) and replaces it with a 512MB one.
    drop(embedder);
    eprintln!("Reloading embedding model with query-optimized settings...");
    let embedder = Arc::new(ParallelEmbedder::with_config(
        crate::engine::ParallelConfig::for_query(),
    )?);
    eprintln!();

    // Step 4: Start file watchers for each catalog
    for (name, catalog) in &config.catalogs {
        watcher::start_watcher(
            name.clone(),
            catalog.clone(),
            embedder.clone(),
            config.qdrant.collection.clone(),
            Some(qdrant_url.clone()),
        );
    }

    // Step 5: Start HTTP server
    let catalog_count = config.catalogs.len();
    eprintln!();
    eprintln!(
        "MCP server ready at http://127.0.0.1:{}/mcp (serving {} catalog{})",
        port,
        catalog_count,
        if catalog_count == 1 { "" } else { "s" }
    );

    let state = McpState {
        embedder,
        collection: config.qdrant.collection.clone(),
        qdrant_url: Some(qdrant_url),
        catalogs: config.catalogs.clone(),
    };

    let data = web::Data::new(state);

    actix_web::rt::System::new().block_on(async move {
        HttpServer::new(move || {
            App::new()
                .app_data(data.clone())
                .route("/mcp", web::post().to(handle_post))
                .route("/mcp", web::get().to(handle_get))
                .route("/mcp", web::delete().to(handle_delete))
        })
        .bind(("127.0.0.1", port))?
        .run()
        .await
    })?;

    Ok(())
}
