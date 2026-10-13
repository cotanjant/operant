use anyhow::Result;
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::engine::{Engine, GenerationParams};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<Engine>>,
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn generate_id() -> String {
    format!("chatcmpl-{}", rand::random::<u32>())
}

async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let engine = state.engine.lock().await;
    let model_name = engine.model_name.clone();
    drop(engine);

    Json(ModelListResponse {
        object: "list",
        data: vec![ModelCard {
            id: model_name,
            object: "model",
            created: current_timestamp(),
            owned_by: "operant",
        }],
    })
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "operant-instruct",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, (StatusCode, String)> {
    let prompt = Engine::format_chat_prompt(&req.messages);

    let params = GenerationParams {
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.or(Some(0.9)),
        max_tokens: req.max_tokens.unwrap_or(512),
        seed: None,
        stop: req.stop.unwrap_or_default(),
    };

    let id = generate_id();
    let created = current_timestamp();

    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(100);
        let engine_arc = state.engine.clone();
        let prompt_clone = prompt.clone();
        let params_clone = params.clone();
        let id_clone = id.clone();

        tokio::task::spawn_blocking(move || {
            let mut engine = futures::executor::block_on(engine_arc.lock());
            let model_name = engine.model_name.clone();

            // First chunk with role
            let initial_chunk = StreamChunk {
                id: id_clone.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            if let Ok(json_str) = serde_json::to_string(&initial_chunk) {
                let _ = tx.blocking_send(format!("data: {}\n\n", json_str));
            }

            let send_tx = tx.clone();
            let model_name_clone = model_name.clone();
            let id_clone2 = id_clone.clone();

            let gen_res = engine.generate_stream_tokens(&prompt_clone, &params_clone, |token| {
                let chunk = StreamChunk {
                    id: id_clone2.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_name_clone.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: StreamDelta {
                            role: None,
                            content: Some(token.to_string()),
                        },
                        finish_reason: None,
                    }],
                };
                if let Ok(json_str) = serde_json::to_string(&chunk) {
                    let _ = send_tx.blocking_send(format!("data: {}\n\n", json_str));
                }
                Ok(())
            });

            let finish_reason = match gen_res {
                Ok(r) => r.finish_reason,
                Err(_) => "stop".to_string(),
            };

            let final_chunk = StreamChunk {
                id: id_clone.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name,
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some(finish_reason),
                }],
            };
            if let Ok(json_str) = serde_json::to_string(&final_chunk) {
                let _ = tx.blocking_send(format!("data: {}\n\n", json_str));
            }
            let _ = tx.blocking_send("data: [DONE]\n\n".to_string());
        });

        let body_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let body = Body::from_stream(body_stream.map(|chunk| Ok::<_, std::convert::Infallible>(chunk)));

        let mut res = Response::new(body);
        res.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        res.headers_mut().insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        );
        res.headers_mut().insert(
            header::CONNECTION,
            HeaderValue::from_static("keep-alive"),
        );
        Ok(res)
    } else {
        let mut engine = state.engine.lock().await;
        let model_name = req.model.unwrap_or_else(|| engine.model_name.clone());

        let res = engine
            .generate(&prompt, &params)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let response = ChatResponse {
            id,
            object: "chat.completion",
            created,
            model: model_name,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: res.text,
                },
                finish_reason: res.finish_reason,
            }],
            usage: Usage {
                prompt_tokens: res.prompt_tokens,
                completion_tokens: res.completion_tokens,
                total_tokens: res.prompt_tokens + res.completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

use tokio_stream::StreamExt;

pub fn create_router(engine: Arc<Mutex<Engine>>) -> Router {
    let state = AppState { engine };

    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chat/completions/", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/models/", get(list_models))
        .route("/health", get(health_check))
        .route("/", get(health_check))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

pub async fn run_server(engine: Arc<Mutex<Engine>>, host: &str, port: u16) -> Result<()> {
    let app = create_router(engine);
    let addr = format!("{}:{}", host, port);
    println!("Operant inference API listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
