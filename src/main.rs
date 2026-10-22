mod batcher;
mod model;
mod yarn;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    routing::post,
    Json, Router,
};
use batcher::Job;
use model::EngineConfig;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::{mpsc::SyncSender, Arc},
    time::Duration,
};
use tokenizers::Tokenizer;
use tokio::sync::{oneshot, Semaphore};

const MODEL_ID: &str = "nvidia/Nemotron-3-Embed-1B";
const PAD_TOKEN_ID: u32 = 11;

#[derive(Clone)]
struct App {
    tokenizer: Arc<Tokenizer>,
    tokenize_slots: Arc<Semaphore>,
    queue: SyncSender<Job>,
    max_len: usize,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct Request {
    model: String,
    input: Input,
    dimensions: Option<usize>,
    encoding_format: Option<String>,
}

type ApiError = (StatusCode, Json<Value>);

fn error(status: StatusCode, message: impl Into<String>) -> ApiError {
    (status, Json(json!({
        "error": { "message": message.into() }
    })))
}

async fn embeddings(
    State(app): State<App>,
    Json(request): Json<Request>,
) -> Result<Json<Value>, ApiError> {
    if request.model != MODEL_ID {
        return Err(error(StatusCode::BAD_REQUEST, "unknown model"));
    }
    if request.dimensions.is_some_and(|d| d != 2048) {
        return Err(error(StatusCode::BAD_REQUEST, "only dimensions=2048 supported"));
    }
    if request.encoding_format.as_deref().unwrap_or("float") != "float" {
        return Err(error(StatusCode::BAD_REQUEST, "only float encoding supported"));
    }

    let texts = match request.input {
        Input::One(x) => vec![x],
        Input::Many(xs) => xs,
    };
    if texts.is_empty() || texts.len() > 128 {
        return Err(error(StatusCode::BAD_REQUEST, "input count must be 1..128"));
    }

    // Reject rather than building an unbounded tokenization wait queue.
    let permit = app.tokenize_slots.clone().try_acquire_owned()
        .map_err(|_| error(StatusCode::TOO_MANY_REQUESTS, "tokenizer overloaded"))?;

    let tokenizer = app.tokenizer.clone();
    let max_len = app.max_len;
    let sequences = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<u32>>, String> {
        let _permit = permit;
        let mut result = Vec::with_capacity(texts.len());

        for text in texts {
            // Clients supply query:/passage: prefixes.
            // add_special_tokens=true must match the reference configuration.
            let encoding = tokenizer.encode(text, true)
                .map_err(|e| e.to_string())?;
            let ids = encoding.get_ids().to_vec();

            if ids.is_empty() || ids.len() > max_len {
                return Err(format!("token count must be 1..{max_len}"));
            }
            if ids.contains(&PAD_TOKEN_ID) {
                return Err(
                    "literal <pad> tokens are not accepted; padding is server-owned"
                        .to_owned()
                );
            }
            result.push(ids);
        }
        Ok(result)
    }).await
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;

    let prompt_tokens: usize = sequences.iter().map(Vec::len).sum();
    let mut pending = Vec::with_capacity(sequences.len());

    for ids in sequences {
        let (reply, recv) = oneshot::channel();
        app.queue.try_send(Job { ids, reply }).map_err(|e| match e {
            std::sync::mpsc::TrySendError::Full(_) =>
                error(StatusCode::TOO_MANY_REQUESTS, "GPU queue full"),
            std::sync::mpsc::TrySendError::Disconnected(_) =>
                error(StatusCode::SERVICE_UNAVAILABLE, "GPU worker unavailable"),
        })?;
        pending.push(recv);
    }

    let collect = async move {
        let mut data = Vec::with_capacity(pending.len());
        for (index, recv) in pending.into_iter().enumerate() {
            let embedding = recv.await
                .map_err(|_| error(StatusCode::SERVICE_UNAVAILABLE, "worker stopped"))?
                .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;

            data.push(json!({
                "object": "embedding",
                "index": index,
                "embedding": embedding
            }));
        }
        Ok::<_, ApiError>(data)
    };

    let data = tokio::time::timeout(Duration::from_secs(30), collect).await
        .map_err(|_| error(StatusCode::GATEWAY_TIMEOUT, "embedding timeout"))??;

    Ok(Json(json!({
        "object": "list",
        "model": MODEL_ID,
        "data": data,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens
        }
    })))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(std::env::var("MODEL_DIR")?);
    let cfg: EngineConfig =
        serde_json::from_slice(&std::fs::read(dir.join("engine.json"))?)?;

    let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // No hidden tokenizer-side truncation or padding.
    tokenizer.with_truncation(None)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    tokenizer.with_padding(None);

    let cpu = std::env::var("DEVICE").as_deref() == Ok("cpu");
    let queue = batcher::start(dir, cfg.clone(), cpu)?;

    let state = App {
        tokenizer: Arc::new(tokenizer),
        tokenize_slots: Arc::new(Semaphore::new(8)),
        queue,
        max_len: cfg.max_len,
    };

    let router = Router::new()
        .route("/v1/embeddings", post(embeddings))
        .layer(DefaultBodyLimit::max(4 * 1024 * 1024))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    axum::serve(listener, router).await?;
    Ok(())
}
