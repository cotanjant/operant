use axum::{Json, Router, routing::post};
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, PartialEq, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, PartialEq, Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(default)]
    stream: bool,
}

#[derive(PartialEq, Debug, Serialize)]
struct Choice {
    index: usize,
    messages: Message,
    finish_reason: &'static str,
}

#[derive(PartialEq, Debug, Serialize)]
struct ChatResponse {
    object: &'static str,
    model: String,
    choices: Vec<Choice>,
}

async fn chat(Json(req): Json<ChatRequest>) -> Json<ChatResponse> {
    todo!("this should go to the model!!!");
    Json(ChatResponse {
        object: "chat.completion",
        model: req.model,
        choices: vec![Choice {
            index: 0,
            messages: Message {
                role: "assitant".into(),
                content: "hello operant!".into(),
            },
            finish_reason: "stop",
        }],
    })
}

async fn api() -> anyhow::Result<()> {
    let app = Router::new().route("/v1/chat/completions/", post(chat));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
