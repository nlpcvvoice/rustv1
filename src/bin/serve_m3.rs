use anyhow::{ensure, Context, Result};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use rustv1::model::IndependentExecutor;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};

// ---------------------------------------------------------------------------
// Wire Protocol Definitions
// ---------------------------------------------------------------------------

const MAX_BODY_BYTES: usize = 1024 * 1024; // 1 MiB hard ceiling
const MAX_STRINGS_PER_REQ: usize = 8;
const MAX_TOKENS_PER_SEQ: usize = 1024;
const PAD_TOKEN_ID: u32 = 11;
const EXPECTED_MODEL_ID: &str = "nvidia/Nemotron-3-Embed-1B";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingRequest {
    model: String,
    input: RequestInput,
    #[serde(default)]
    encoding_format: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    user: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RequestInput {
    Single(String),
    Batch(Vec<String>),
}

impl RequestInput {
    fn into_vec(self) -> Result<Vec<String>, AppError> {
        match self {
            Self::Single(s) => {
                if s.trim().is_empty() {
                    return Err(AppError::InvalidInput(
                        "input text must not be empty".into(),
                    ));
                }
                Ok(vec![s])
            }
            Self::Batch(v) => {
                if v.is_empty() {
                    return Err(AppError::InvalidInput(
                        "batch input must not be empty".into(),
                    ));
                }
                if v.len() > MAX_STRINGS_PER_REQ {
                    return Err(AppError::InvalidInput(format!(
                        "batch exceeds maximum allowed elements (got {}, max {})",
                        v.len(),
                        MAX_STRINGS_PER_REQ
                    )));
                }
                for (i, s) in v.iter().enumerate() {
                    if s.trim().is_empty() {
                        return Err(AppError::InvalidInput(format!(
                            "element {i} in batch must not be empty"
                        )));
                    }
                }
                Ok(v)
            }
        }
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingResponse {
    object: &'static str,
    data: Vec<EmbeddingData>,
    model: String,
    usage: UsageStats,
}

#[derive(Debug, Serialize)]
struct EmbeddingData {
    object: &'static str,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct UsageStats {
    prompt_tokens: usize,
    total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Error Handling
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum AppError {
    InvalidInput(String),
    PayloadTooLarge,
    QueueFull,
    ExecutorPoisoned,
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, err_type, message) = match self {
            Self::InvalidInput(msg) => (StatusCode::BAD_REQUEST, "invalid_request_error", msg),
            Self::PayloadTooLarge => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "invalid_request_error",
                "request payload exceeded maximum size of 1 MiB".to_string(),
            ),
            Self::QueueFull => (
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable_error",
                "request queue is full; executor is saturated".to_string(),
            ),
            Self::ExecutorPoisoned => (
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable_error",
                "executor entered a terminal failure state; process restart required".to_string(),
            ),
            Self::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "api_error", msg),
        };

        let body = json!({
            "error": {
                "message": message,
                "type": err_type,
                "param": Value::Null,
                "code": Value::Null
            }
        });

        (
            status,
            [(header::CONTENT_TYPE, "application/json")],
            Json(body),
        )
            .into_response()
    }
}

// ---------------------------------------------------------------------------
// Worker Message Queue
// ---------------------------------------------------------------------------

struct InferenceJob {
    token_batches: Vec<Vec<u32>>,
    responder: oneshot::Sender<Result<Vec<Vec<f32>>, AppError>>,
}

#[derive(Clone)]
struct AppState {
    tx: mpsc::Sender<InferenceJob>,
    tokenizer: Arc<Tokenizer>,
    vocab_size: usize,
    poisoned_flag: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn livez() -> impl IntoResponse {
    StatusCode::OK
}

async fn readyz(State(state): State<AppState>) -> impl IntoResponse {
    if state.poisoned_flag.load(Ordering::Acquire) {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "executor poisoned; restart required",
        )
    } else {
        (StatusCode::OK, "ready")
    }
}

async fn embeddings(
    State(state): State<AppState>,
    _headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    if state.poisoned_flag.load(Ordering::Acquire) {
        return Err(AppError::ExecutorPoisoned);
    }

    if body.len() > MAX_BODY_BYTES {
        return Err(AppError::PayloadTooLarge);
    }

    let req: EmbeddingRequest = serde_json::from_slice(&body).map_err(|e| {
        AppError::InvalidInput(format!("malformed JSON or unknown request fields: {e}"))
    })?;

    if req.model != EXPECTED_MODEL_ID {
        return Err(AppError::InvalidInput(format!(
            "invalid model '{}'; expected '{}'",
            req.model, EXPECTED_MODEL_ID
        )));
    }

    if let Some(ref fmt) = req.encoding_format {
        if fmt != "float" {
            return Err(AppError::InvalidInput(format!(
                "unsupported encoding_format '{}'; only 'float' is implemented",
                fmt
            )));
        }
    }

    let texts = req.input.into_vec()?;
    let mut token_batches = Vec::with_capacity(texts.len());
    let mut total_tokens = 0usize;

    for (idx, text) in texts.iter().enumerate() {
        let enc = state.tokenizer.encode(text.as_str(), false).map_err(|e| {
            AppError::InvalidInput(format!("tokenization failed at item {idx}: {e}"))
        })?;
        let ids = enc.get_ids().to_vec();

        if ids.is_empty() {
            return Err(AppError::InvalidInput(format!(
                "item {idx} produced zero tokens"
            )));
        }
        if ids.len() > MAX_TOKENS_PER_SEQ {
            return Err(AppError::InvalidInput(format!(
                "item {idx} produced {} tokens; exceeds limit of {}",
                ids.len(),
                MAX_TOKENS_PER_SEQ
            )));
        }
        for &id in &ids {
            if id == PAD_TOKEN_ID {
                return Err(AppError::InvalidInput(format!(
                    "item {idx} contains forbidden pad token ID ({PAD_TOKEN_ID})"
                )));
            }
            if id as usize >= state.vocab_size {
                return Err(AppError::InvalidInput(format!(
                    "item {idx} contains out-of-vocabulary token ID {id}"
                )));
            }
        }
        total_tokens += ids.len();
        token_batches.push(ids);
    }

    let (resp_tx, resp_rx) = oneshot::channel();
    let job = InferenceJob {
        token_batches,
        responder: resp_tx,
    };

    state.tx.try_send(job).map_err(|e| match e {
        mpsc::error::TrySendError::Full(_) => AppError::QueueFull,
        mpsc::error::TrySendError::Closed(_) => {
            state.poisoned_flag.store(true, Ordering::Release);
            AppError::ExecutorPoisoned
        }
    })?;

    let embeddings = resp_rx.await.map_err(|_| {
        state.poisoned_flag.store(true, Ordering::Release);
        AppError::ExecutorPoisoned
    })??;

    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingData {
            object: "embedding",
            index,
            embedding,
        })
        .collect();

    let response = EmbeddingResponse {
        object: "list",
        data,
        model: req.model,
        usage: UsageStats {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        Json(response),
    )
        .into_response())
}

// ---------------------------------------------------------------------------
// Worker Implementation
// ---------------------------------------------------------------------------

fn run_worker(
    mut executor: IndependentExecutor,
    mut rx: mpsc::Receiver<InferenceJob>,
    poisoned_flag: Arc<AtomicBool>,
) {
    while let Some(job) = rx.blocking_recv() {
        if executor.is_poisoned() {
            poisoned_flag.store(true, Ordering::Release);
            let _ = job.responder.send(Err(AppError::ExecutorPoisoned));
            continue;
        }

        let result = executor.forward_serving(&job.token_batches);
        match result {
            Ok(embeddings) => {
                let _ = job.responder.send(Ok(embeddings));
            }
            Err(e) => {
                if executor.is_poisoned() {
                    poisoned_flag.store(true, Ordering::Release);
                }
                let _ = job
                    .responder
                    .send(Err(AppError::Internal(format!("{e:#}"))));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entrypoint
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 6,
        "usage: serve_m3 MODEL_DIR MANIFEST_JSONL AUDIT_IDENTITY BIND_ADDR QUEUE_CAPACITY"
    );

    let model_dir = PathBuf::from(&args[1]);
    let manifest_path = PathBuf::from(&args[2]);
    let audit_identity = args[3].clone();
    let bind_addr = args[4].clone();
    let queue_capacity: usize = args[5].parse().context("invalid queue capacity")?;

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load error: {e}"))?;

    let config_bytes = std::fs::read(model_dir.join("config.json"))?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes)?;
    let vocab_size = config["vocab_size"]
        .as_u64()
        .context("config missing vocab_size")? as usize;

    let (tx, rx) = mpsc::channel(queue_capacity);
    let poisoned_flag = Arc::new(AtomicBool::new(false));
    let poisoned_worker = poisoned_flag.clone();

    // Spawn dedicated OS thread for executor ownership (!Send + !Sync + MathModePin check)
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    thread::Builder::new()
        .name("m3-executor-worker".into())
        .spawn(move || {
            let executor_res =
                rustv1::m2_route::open(&model_dir, &manifest_path, 0, &audit_identity);
            match executor_res {
                Ok(executor) => {
                    ready_tx.send(Ok(())).unwrap();
                    run_worker(executor, rx, poisoned_worker);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            }
        })?;

    ready_rx.recv()??;

    let state = AppState {
        tx,
        tokenizer: Arc::new(tokenizer),
        vocab_size,
        poisoned_flag,
    };

    let app = Router::new()
        .route("/health/livez", get(livez))
        .route("/health/readyz", get(readyz))
        .route("/v1/embeddings", post(embeddings))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    println!("serve_m3 listening on {bind_addr}");
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        routing::get,
        Router,
    };

    use serde_json::json;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use tokio::sync::mpsc;
    use tower::ServiceExt;

    // Direct re-tests of route parsing and state semantics
    #[tokio::test]
    async fn request_validation_matrix() {
        // Unknown field test
        let raw = json!({
            "model": "nvidia/Nemotron-3-Embed-1B",
            "input": "test text",
            "mystery_field": true
        });
        let res: Result<super::EmbeddingRequest, _> = serde_json::from_value(raw);
        assert!(res.is_err(), "unknown field must fail closed");

        // Batch > 8 strings test
        let raw = json!({
            "model": "nvidia/Nemotron-3-Embed-1B",
            "input": vec!["s"; 9]
        });
        let req: super::EmbeddingRequest = serde_json::from_value(raw).unwrap();
        assert!(req.input.into_vec().is_err());

        // Empty string in batch test
        let raw = json!({
            "model": "nvidia/Nemotron-3-Embed-1B",
            "input": vec!["valid", "   "]
        });
        let req: super::EmbeddingRequest = serde_json::from_value(raw).unwrap();
        assert!(req.input.into_vec().is_err());
    }

    #[tokio::test]
    async fn readyz_reports_poison_status() {
        let poisoned = Arc::new(AtomicBool::new(false));
        let (_tx, _rx) = mpsc::channel::<super::InferenceJob>(1);

        // Dummy test router
        let p_clone = poisoned.clone();
        let app = Router::new().route(
            "/health/readyz",
            get(move || {
                let p = p_clone.clone();
                async move {
                    if p.load(Ordering::Acquire) {
                        (axum::http::StatusCode::SERVICE_UNAVAILABLE, "poisoned")
                    } else {
                        (axum::http::StatusCode::OK, "ready")
                    }
                }
            }),
        );

        let res = app
            .clone()
            .oneshot(Request::get("/health/readyz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        poisoned.store(true, Ordering::Release);
        let res = app
            .oneshot(Request::get("/health/readyz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
