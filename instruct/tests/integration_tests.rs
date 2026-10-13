use axum::body::Body;
use axum::http::{Request, StatusCode};
use instruct::api::{ChatRequest, Message, create_router};
use instruct::engine::Engine;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower::ServiceExt;

#[tokio::test]
async fn test_api_routes() -> anyhow::Result<()> {
    let model_path = PathBuf::from("../model/out");
    if !model_path.exists() {
        println!("Model path does not exist, skipping integration test");
        return Ok(());
    }

    let engine = Engine::load(&model_path)?;
    let app = create_router(Arc::new(Mutex::new(engine)));

    // 1. Health check
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);

    // 2. Models list
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
    let body_json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body_json["object"], "list");

    // 3. Chat completion non-streaming
    let req = ChatRequest {
        model: Some("operant".to_string()),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Who am i?".to_string(),
        }],
        temperature: Some(0.0),
        top_p: None,
        max_tokens: Some(10),
        stream: false,
        stop: None,
    };

    let req_body = serde_json::to_vec(&req).unwrap();
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(req_body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
    let body_json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body_json["object"], "chat.completion");
    assert!(!body_json["choices"].as_array().unwrap().is_empty());
    println!("API response choices: {:?}", body_json["choices"]);

    Ok(())
}
