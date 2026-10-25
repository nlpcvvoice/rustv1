//! CUDA-independent M3 HTTP serving.
//!
//! Executors are constructed, checked, used, and dropped on their owner OS
//! thread. Neither Backend nor its factory result needs to implement Send.
//!
//! The transport uses a deliberately conservative policy: each connection has
//! an absolute lifetime bounded by the smallest configured transport timeout.
//! This bounds slow headers, slow writes, and idle keepalive without relying on
//! a reverse proxy. It also limits active connections, including connections
//! which have not yet produced an HTTP request. Connections may therefore be
//! closed earlier than a phase-specific timeout implementation would close them.
//!
//! Install a sanitized process-wide panic hook before starting this service.
//! catch_unwind cannot suppress the process's existing panic hook.
//!
//! Executor audit manifests are separate, access-controlled provenance
//! artifacts. They are not the sanitized operational log implemented here.

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, FromRequest, Request, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    future::Future,
    io::{self, Write},
    net::SocketAddr,
    path::Path,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, MutexGuard,
    },
    task::{Context, Poll},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::{mpsc, oneshot, OwnedSemaphorePermit, Semaphore},
    time::{sleep, timeout, timeout_at, Instant as TokioInstant, Sleep},
};

pub const MODEL_ID: &str = "nvidia/Nemotron-3-Embed-1B";
pub const EMBEDDING_WIDTH: usize = 2048;
pub const BODY_LIMIT: usize = 1_048_576;
pub const PAD_ID: u32 = 11;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[derive(Clone, Debug)]
pub struct Config {
    pub admission_capacity: usize,
    pub tokenizer_capacity: usize,
    pub queue_capacity: usize,
    pub max_body_bytes: usize,
    pub max_sequence_tokens: usize,
    pub vocabulary_size: usize,
    pub body_timeout: Duration,
    pub request_timeout: Duration,
    pub startup_timeout: Duration,
    pub shutdown_grace: Duration,
    pub control_timeout: Duration,
    pub transport: TransportConfig,
}

impl Config {
    pub fn production(queue_capacity: usize, vocabulary_size: usize, limit: usize) -> Self {
        Self {
            admission_capacity: 64,
            tokenizer_capacity: 4,
            queue_capacity,
            max_body_bytes: BODY_LIMIT,
            max_sequence_tokens: limit.min(1024),
            vocabulary_size,
            body_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(60),
            startup_timeout: Duration::from_secs(300),
            shutdown_grace: Duration::from_secs(90),
            control_timeout: Duration::from_secs(10),
            transport: TransportConfig {
                max_connections: 128,
                header_timeout: Duration::from_secs(90),
                write_timeout: Duration::from_secs(90),
                idle_timeout: Duration::from_secs(90),
                connection_timeout: Duration::from_secs(90),
            },
        }
    }

    pub fn validate(&self) -> Result<(), Code> {
        let capacities = [
            self.admission_capacity,
            self.tokenizer_capacity,
            self.queue_capacity,
            self.transport.max_connections,
        ];
        if capacities.iter().any(|&n| n == 0 || n > 4096)
            || self.tokenizer_capacity > self.admission_capacity
            || self.max_body_bytes == 0
            || self.max_body_bytes > BODY_LIMIT
            || self.max_sequence_tokens == 0
            || self.max_sequence_tokens > 1024
            || self.vocabulary_size <= usize::try_from(PAD_ID).unwrap_or(usize::MAX)
            || [
                self.body_timeout,
                self.request_timeout,
                self.startup_timeout,
                self.shutdown_grace,
                self.control_timeout,
                self.transport.header_timeout,
                self.transport.write_timeout,
                self.transport.idle_timeout,
                self.transport.connection_timeout,
            ]
            .iter()
            .any(Duration::is_zero)
        {
            return Err(Code::InvalidRequest);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TransportConfig {
    pub max_connections: usize,
    pub header_timeout: Duration,
    pub write_timeout: Duration,
    pub idle_timeout: Duration,
    pub connection_timeout: Duration,
}

impl TransportConfig {
    fn lifetime(&self) -> Duration {
        self.header_timeout
            .min(self.write_timeout)
            .min(self.idle_timeout)
            .min(self.connection_timeout)
    }
}

/// The only operational error classifications exported by the service.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Code {
    EmptyBatch,
    EmptySequence,
    PadToken,
    OverLength,
    OutOfVocabulary,
    InvalidRequest,
    InvalidModel,
    UnsupportedEncodingFormat,
    UnsupportedMediaType,
    PayloadTooLarge,
    QueueFull,
    NotReady,
    ExecutorPoisoned,
    InferenceFailed,
    DeadlineExceeded,
}

impl Code {
    pub fn status(self) -> StatusCode {
        match self {
            Self::UnsupportedMediaType => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Self::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::QueueFull | Self::NotReady | Self::ExecutorPoisoned => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            Self::InferenceFailed => StatusCode::INTERNAL_SERVER_ERROR,
            Self::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
            _ => StatusCode::BAD_REQUEST,
        }
    }

    fn message(self) -> &'static str {
        match self {
            Self::EmptyBatch => "Input batch is empty",
            Self::EmptySequence => "Input sequence is empty",
            Self::PadToken => "Input contains a padding token",
            Self::OverLength => "Input exceeds the sequence limit",
            Self::OutOfVocabulary => "Input contains an out-of-vocabulary token",
            Self::InvalidRequest => "Invalid request",
            Self::InvalidModel => "Unsupported model",
            Self::UnsupportedEncodingFormat => "Unsupported encoding format",
            Self::UnsupportedMediaType => "Unsupported media type or content encoding",
            Self::PayloadTooLarge => "Request body exceeds the size limit",
            Self::QueueFull => "Admission capacity is exhausted",
            Self::NotReady => "Service is not ready",
            Self::ExecutorPoisoned => "Executor is retired; restart required",
            Self::InferenceFailed => "Inference failed",
            Self::DeadlineExceeded => "Request deadline exceeded",
        }
    }

    fn param(self) -> Option<&'static str> {
        match self {
            Self::EmptyBatch
            | Self::EmptySequence
            | Self::PadToken
            | Self::OverLength
            | Self::OutOfVocabulary => Some("input"),
            Self::InvalidModel => Some("model"),
            Self::UnsupportedEncodingFormat => Some("encoding_format"),
            _ => None,
        }
    }

    fn response_with_status(self, status: StatusCode) -> Response {
        let kind = if status.is_client_error() {
            "invalid_request_error"
        } else if status == StatusCode::SERVICE_UNAVAILABLE {
            "service_unavailable_error"
        } else {
            "api_error"
        };
        (
            status,
            Json(json!({
                "error": {
                    "message": self.message(),
                    "type": kind,
                    "param": self.param(),
                    "code": self
                }
            })),
        )
            .into_response()
    }
}

impl IntoResponse for Code {
    fn into_response(self) -> Response {
        self.response_with_status(self.status())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingRequest {
    model: String,
    input: RequestInput,
    #[serde(default)]
    encoding_format: Option<Value>,
    #[serde(default, rename = "user")]
    _user: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RequestInput {
    Single(String),
    Batch(Vec<String>),
}

impl RequestInput {
    fn texts(self) -> Result<Vec<String>, Code> {
        let texts = match self {
            Self::Single(text) => vec![text],
            Self::Batch(texts) => texts,
        };
        if texts.is_empty() {
            return Err(Code::EmptyBatch);
        }
        if texts.len() > 8 {
            return Err(Code::InvalidRequest);
        }
        if texts.iter().any(|text| text.trim().is_empty()) {
            return Err(Code::EmptySequence);
        }
        Ok(texts)
    }
}

#[derive(Clone, Copy)]
enum Encoding {
    Float,
    Base64,
}

impl Encoding {
    fn parse(value: Option<Value>) -> Result<Self, Code> {
        match value {
            None | Some(Value::Null) => Ok(Self::Float),
            Some(Value::String(value)) if value == "float" => Ok(Self::Float),
            Some(Value::String(value)) if value == "base64" => Ok(Self::Base64),
            _ => Err(Code::UnsupportedEncodingFormat),
        }
    }
}

#[derive(Serialize)]
struct EmbeddingResponse {
    object: &'static str,
    data: Vec<EmbeddingData>,
    model: &'static str,
    usage: Usage,
}

#[derive(Serialize)]
struct EmbeddingData {
    object: &'static str,
    index: usize,
    embedding: WireEmbedding,
}

#[derive(Serialize)]
#[serde(untagged)]
enum WireEmbedding {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

fn success(rows: Vec<Vec<f32>>, encoding: Encoding, tokens: usize) -> Response {
    let data = rows
        .into_iter()
        .enumerate()
        .map(|(index, row)| EmbeddingData {
            object: "embedding",
            index,
            embedding: match encoding {
                Encoding::Float => WireEmbedding::Float(row),
                Encoding::Base64 => {
                    let bytes: Vec<u8> = row.iter().flat_map(|value| value.to_le_bytes()).collect();
                    WireEmbedding::Base64(STANDARD.encode(bytes))
                }
            },
        })
        .collect();
    Json(EmbeddingResponse {
        object: "list",
        data,
        model: MODEL_ID,
        usage: Usage {
            prompt_tokens: tokens,
            total_tokens: tokens,
        },
    })
    .into_response()
}

fn check_media(headers: &HeaderMap) -> Result<(), Code> {
    if headers.contains_key(header::CONTENT_ENCODING)
        || headers.get_all(header::CONTENT_TYPE).iter().count() != 1
    {
        return Err(Code::UnsupportedMediaType);
    }
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .ok_or(Code::UnsupportedMediaType)?;
    let mut parts = content_type.split(';');
    if !parts
        .next()
        .is_some_and(|part| part.trim().eq_ignore_ascii_case("application/json"))
    {
        return Err(Code::UnsupportedMediaType);
    }
    let mut charset_seen = false;
    for part in parts {
        let (key, value) = part
            .trim()
            .split_once('=')
            .ok_or(Code::UnsupportedMediaType)?;
        if charset_seen
            || !key.trim().eq_ignore_ascii_case("charset")
            || !value.trim().trim_matches('"').eq_ignore_ascii_case("utf-8")
        {
            return Err(Code::UnsupportedMediaType);
        }
        charset_seen = true;
    }
    Ok(())
}

pub trait TextTokenizer: Send + Sync + 'static {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Code>;
}

pub struct PinnedTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl PinnedTokenizer {
    pub fn load(path: &Path, expected_sha256: &str, vocab_size: usize) -> Result<Self, Code> {
        let bytes = std::fs::read(path).map_err(|_| Code::InvalidRequest)?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if expected_sha256.len() != 64 || actual != expected_sha256 {
            return Err(Code::InvalidRequest);
        }
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(bytes).map_err(|_| Code::InvalidRequest)?;
        if tokenizer.get_padding().is_some() || tokenizer.get_truncation().is_some() {
            return Err(Code::InvalidRequest);
        }
        if tokenizer
            .get_vocab(true)
            .values()
            .any(|&id| usize::try_from(id).map_or(true, |converted| converted >= vocab_size))
        {
            return Err(Code::OutOfVocabulary);
        }
        Ok(Self { tokenizer })
    }
}

impl TextTokenizer for PinnedTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Code> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|_| Code::InvalidRequest)
    }
}

pub fn validate_tokens(batches: &[Vec<u32>], config: &Config) -> Result<usize, Code> {
    if batches.is_empty() {
        return Err(Code::EmptyBatch);
    }
    if batches.len() > 8 {
        return Err(Code::InvalidRequest);
    }
    let mut total = 0usize;
    for sequence in batches {
        if sequence.is_empty() {
            return Err(Code::EmptySequence);
        }
        if sequence.len() > config.max_sequence_tokens {
            return Err(Code::OverLength);
        }
        for &id in sequence {
            if id == PAD_ID {
                return Err(Code::PadToken);
            }
            if usize::try_from(id).map_or(true, |converted| converted >= config.vocabulary_size) {
                return Err(Code::OutOfVocabulary);
            }
        }
        total = total
            .checked_add(sequence.len())
            .ok_or(Code::InvalidRequest)?;
    }
    Ok(total)
}

pub fn validate_output(rows: &[Vec<f32>], count: usize) -> Result<(), Code> {
    if rows.len() != count {
        return Err(Code::InferenceFailed);
    }
    for row in rows {
        if row.len() != EMBEDDING_WIDTH || row.iter().any(|value| !value.is_finite()) {
            return Err(Code::InferenceFailed);
        }
        let norm = row
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > 1e-5 {
            return Err(Code::InferenceFailed);
        }
    }
    Ok(())
}

/// Exact HTTP/restart predicate. Decode JSON directly into Vec<f32>.
pub fn exact_f32(rows: &[Vec<f32>], golden: &[Vec<f32>]) -> bool {
    validate_output(rows, golden.len()).is_ok()
        && validate_output(golden, rows.len()).is_ok()
        && rows.iter().zip(golden).all(|(left, right)| {
            left.iter()
                .zip(right)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        })
}

fn output_digest(rows: &[Vec<f32>]) -> String {
    let mut digest = Sha256::new();
    for row in rows {
        for value in row {
            digest.update(value.to_le_bytes());
        }
    }
    digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Deliberately has no Send or Sync bound.
pub trait Backend: 'static {
    fn forward(&mut self, batches: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, Code>;
    fn inject_policy_drift(&mut self) -> Result<(), Code>;
    fn restore_policy(&mut self) -> Result<(), Code>;
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartupCheck {
    pub tokens: Vec<Vec<u32>>,
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Lifecycle {
    Starting,
    Ready,
    Draining,
    Stopped,
    Retired,
    Poisoned,
}

impl Lifecycle {
    fn terminal(self) -> bool {
        matches!(self, Self::Stopped | Self::Retired | Self::Poisoned)
    }
}

#[derive(Clone, Copy)]
enum AdmissionMode {
    Open,
    Quiesced,
    One,
}

struct Core {
    state: Lifecycle,
    worker_alive: bool,
    expected_exit: bool,
    admission: AdmissionMode,
    armed: bool,
    forward_calls: u64,
    startup_output_sha256: Option<String>,
}

struct Shared {
    core: Mutex<Core>,
    instance: String,
}

impl Shared {
    fn reject(core: &Core) -> Option<Code> {
        match core.state {
            Lifecycle::Poisoned | Lifecycle::Retired => Some(Code::ExecutorPoisoned),
            Lifecycle::Ready if core.worker_alive => None,
            _ => Some(Code::NotReady),
        }
    }

    fn retire(&self) {
        let mut core = lock(&self.core);
        if !core.state.terminal() {
            core.state = Lifecycle::Retired;
        }
    }

    fn poison(&self) {
        let mut core = lock(&self.core);
        if !core.state.terminal() {
            core.state = Lifecycle::Poisoned;
        }
    }
}

/// A fixed-schema logger: callers cannot supply arbitrary fields or messages.
#[derive(Clone)]
pub struct Logger {
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
}

#[derive(Serialize)]
struct LogRecord {
    event: &'static str,
    request_id: u64,
    route: &'static str,
    status: u16,
    code: Option<Code>,
    elapsed_ms: f64,
}

impl Logger {
    pub fn new(writer: impl Write + Send + 'static) -> Self {
        Self {
            writer: Arc::new(Mutex::new(Box::new(writer))),
        }
    }

    pub fn stderr() -> Self {
        Self::new(io::stderr())
    }

    fn complete(&self, id: u64, started: Instant, code: Option<Code>) {
        let record = LogRecord {
            event: "request_complete",
            request_id: id,
            route: "embeddings",
            status: code.map_or(200, |code| code.status().as_u16()),
            code,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
        // Serialize before acquiring the writer lock. No raw error is logged.
        if let Ok(mut bytes) = serde_json::to_vec(&record) {
            bytes.push(b'\n');
            let mut writer = lock(&self.writer);
            let _ = writer.write_all(&bytes);
            let _ = writer.flush();
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Queued,
    Running,
    Done,
    Expired,
}

struct Job {
    batches: Vec<Vec<u32>>,
    deadline: Instant,
    phase: Arc<Mutex<Phase>>,
    reply: oneshot::Sender<Result<Vec<Vec<f32>>, Code>>,
    _admission: Arc<OwnedSemaphorePermit>,
}

enum Message {
    Infer(Job),
    Control {
        command: ControlCommand,
        deadline: Instant,
        reply: oneshot::Sender<Result<Observation, Code>>,
    },
    Stop,
}

#[derive(Clone)]
pub struct BackendHandle {
    tx: mpsc::Sender<Message>,
    shared: Arc<Shared>,
}

impl BackendHandle {
    fn ready(&self) -> bool {
        let core = lock(&self.shared.core);
        core.state == Lifecycle::Ready
            && core.worker_alive
            && !matches!(core.admission, AdmissionMode::Quiesced)
            && !self.tx.is_closed()
    }

    async fn infer(
        &self,
        batches: Vec<Vec<u32>>,
        deadline: Instant,
        admission: Arc<OwnedSemaphorePermit>,
    ) -> Result<Vec<Vec<f32>>, Code> {
        let phase = Arc::new(Mutex::new(Phase::Queued));
        let (reply, receiver) = oneshot::channel();
        let job = Job {
            batches,
            deadline,
            phase: phase.clone(),
            reply,
            _admission: admission,
        };
        self.tx
            .try_send(Message::Infer(job))
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => Code::QueueFull,
                mpsc::error::TrySendError::Closed(_) => {
                    self.shared.poison();
                    Code::ExecutorPoisoned
                }
            })?;

        match timeout_at(TokioInstant::from_std(deadline), receiver).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => {
                self.shared.poison();
                Err(Code::ExecutorPoisoned)
            }
            Err(_) => {
                // Lock order is always core -> job phase, including the worker.
                let mut core = lock(&self.shared.core);
                let mut phase = lock(&phase);
                if *phase == Phase::Running && !core.state.terminal() {
                    core.state = Lifecycle::Retired;
                }
                if *phase != Phase::Done {
                    *phase = Phase::Expired;
                }
                Err(Code::DeadlineExceeded)
            }
        }
    }

    async fn command(
        &self,
        command: ControlCommand,
        duration: Duration,
    ) -> Result<Observation, Code> {
        let (reply, receiver) = oneshot::channel();
        self.tx
            .try_send(Message::Control {
                command,
                deadline: Instant::now() + duration,
                reply,
            })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => Code::QueueFull,
                mpsc::error::TrySendError::Closed(_) => Code::ExecutorPoisoned,
            })?;
        timeout(duration, receiver)
            .await
            .map_err(|_| Code::DeadlineExceeded)?
            .map_err(|_| Code::ExecutorPoisoned)?
    }
}

#[derive(Clone)]
pub struct App {
    config: Arc<Config>,
    backend: BackendHandle,
    tokenizer: Arc<dyn TextTokenizer>,
    admission: Arc<Semaphore>,
    tokenizers: Arc<Semaphore>,
    logger: Logger,
    next_request: Arc<AtomicU64>,
}

impl App {
    pub fn router(&self) -> Router {
        Router::new()
            .route("/health/livez", get(|| async { StatusCode::OK }))
            .route("/health/readyz", get(readyz))
            .route("/v1/embeddings", post(embeddings))
            .fallback(|| async { Code::InvalidRequest.response_with_status(StatusCode::NOT_FOUND) })
            .method_not_allowed_fallback(|| async {
                Code::InvalidRequest.response_with_status(StatusCode::METHOD_NOT_ALLOWED)
            })
            .layer(DefaultBodyLimit::max(self.config.max_body_bytes))
            .with_state(self.clone())
    }

    pub fn is_ready(&self) -> bool {
        self.backend.ready()
    }

    pub fn lifecycle(&self) -> Lifecycle {
        lock(&self.backend.shared.core).state
    }

    pub fn begin_drain(&self) {
        let mut core = lock(&self.backend.shared.core);
        if matches!(core.state, Lifecycle::Starting | Lifecycle::Ready) {
            core.state = Lifecycle::Draining;
        }
        core.admission = AdmissionMode::Quiesced;
    }

    fn admit(&self) -> Result<Arc<OwnedSemaphorePermit>, Code> {
        let mut core = lock(&self.backend.shared.core);
        if let Some(code) = Shared::reject(&core) {
            return Err(code);
        }
        if self.backend.tx.is_closed() {
            core.state = Lifecycle::Poisoned;
            return Err(Code::ExecutorPoisoned);
        }
        if matches!(core.admission, AdmissionMode::Quiesced) {
            return Err(Code::NotReady);
        }
        let permit = self
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| Code::QueueFull)?;
        if matches!(core.admission, AdmissionMode::One) {
            core.admission = AdmissionMode::Quiesced;
        }
        Ok(Arc::new(permit))
    }

    async fn request(
        &self,
        request: Request,
        started: Instant,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<Response, Code> {
        let deadline = started + self.config.request_timeout;
        check_media(request.headers())?;

        // Bytes::from_request observes DefaultBodyLimit. In particular, do not
        // replace this with unbounded BodyExt::collect().
        let read_budget = self
            .config
            .body_timeout
            .min(deadline.saturating_duration_since(Instant::now()));
        let bytes = timeout(read_budget, Bytes::from_request(request, &()))
            .await
            .map_err(|_| Code::DeadlineExceeded)?
            .map_err(|rejection| {
                if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
                    Code::PayloadTooLarge
                } else {
                    Code::InvalidRequest
                }
            })?;
        let parsed: EmbeddingRequest =
            serde_json::from_slice(&bytes).map_err(|_| Code::InvalidRequest)?;
        if parsed.model != MODEL_ID {
            return Err(Code::InvalidModel);
        }
        let encoding = Encoding::parse(parsed.encoding_format)?;
        let texts = parsed.input.texts()?;

        let tokenizer_permit = self
            .tokenizers
            .clone()
            .try_acquire_owned()
            .map_err(|_| Code::QueueFull)?;
        let tokenizer = self.tokenizer.clone();
        let config = self.config.clone();
        let blocking_admission = permit.clone();
        let work = tokio::task::spawn_blocking(move || {
            // These permits remain owned even if the async wait times out.
            let _tokenizer_permit = tokenizer_permit;
            let _admission_permit = blocking_admission;
            let batches = texts
                .iter()
                .map(|text| tokenizer.encode(text))
                .collect::<Result<Vec<_>, _>>()?;
            let tokens = validate_tokens(&batches, &config)?;
            Ok::<_, Code>((batches, tokens))
        });
        let (batches, tokens) = timeout_at(TokioInstant::from_std(deadline), work)
            .await
            .map_err(|_| Code::DeadlineExceeded)?
            .map_err(|_| Code::InvalidRequest)??;

        if Instant::now() >= deadline {
            return Err(Code::DeadlineExceeded);
        }
        let rows = self.backend.infer(batches, deadline, permit).await?;
        Ok(success(rows, encoding, tokens))
    }

    async fn control(&self, command: ControlCommand) -> Result<Observation, Code> {
        if command == ControlCommand::Quiesce {
            {
                let mut core = lock(&self.backend.shared.core);
                core.admission = AdmissionMode::Quiesced;
            }
            // The barrier includes body readers, tokenizer closures, queued
            // jobs, and executing jobs. It does not rely on queue emptiness.
            let count =
                u32::try_from(self.config.admission_capacity).map_err(|_| Code::InvalidRequest)?;
            let all = timeout(
                self.config.control_timeout,
                self.admission.clone().acquire_many_owned(count),
            )
            .await
            .map_err(|_| Code::DeadlineExceeded)?
            .map_err(|_| Code::NotReady)?;
            drop(all);
        }
        self.backend
            .command(command, self.config.control_timeout)
            .await
    }
}

async fn readyz(State(app): State<App>) -> Response {
    if app.is_ready() {
        StatusCode::OK.into_response()
    } else {
        let code = if matches!(app.lifecycle(), Lifecycle::Retired | Lifecycle::Poisoned) {
            Code::ExecutorPoisoned
        } else {
            Code::NotReady
        };
        code.into_response()
    }
}

async fn embeddings(State(app): State<App>, request: Request) -> Response {
    let started = Instant::now();
    let id = app.next_request.fetch_add(1, Ordering::Relaxed);
    let permit = match app.admit() {
        Ok(permit) => permit,
        Err(code) => {
            app.logger.complete(id, started, Some(code));
            return code.into_response();
        }
    };

    // Detaching the HTTP waiter must not cancel the execution deadline or
    // release permits while a blocking tokenizer / executor still owns work.
    let task = tokio::spawn(async move {
        let result = app.request(request, started, permit).await;
        app.logger
            .complete(id, started, result.as_ref().err().copied());
        result.unwrap_or_else(IntoResponse::into_response)
    });
    task.await
        .unwrap_or_else(|_| Code::InferenceFailed.into_response())
}

pub struct Service {
    pub app: App,
    worker: Option<thread::JoinHandle<()>>,
    startup: Option<oneshot::Receiver<Result<(), Code>>>,
}

impl Service {
    pub fn spawn<B, F>(
        config: Config,
        tokenizer: Arc<dyn TextTokenizer>,
        logger: Logger,
        check: StartupCheck,
        factory: F,
    ) -> Result<Self, Code>
    where
        B: Backend,
        F: FnOnce() -> Result<B, Code> + Send + 'static,
    {
        config.validate()?;
        validate_tokens(&check.tokens, &config)?;
        validate_output(&check.embeddings, check.tokens.len())?;

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Code::InvalidRequest)?
            .as_nanos();
        let shared = Arc::new(Shared {
            core: Mutex::new(Core {
                state: Lifecycle::Starting,
                worker_alive: false,
                expected_exit: false,
                admission: AdmissionMode::Open,
                armed: false,
                forward_calls: 0,
                startup_output_sha256: None,
            }),
            instance: format!("{}-{stamp}", std::process::id()),
        });
        let (tx, rx) = mpsc::channel(config.queue_capacity);
        let (startup_tx, startup_rx) = oneshot::channel();
        let owner_shared = shared.clone();
        let worker = thread::Builder::new()
            .name("m3-executor-owner".into())
            .spawn(move || {
                let _exit = WorkerExit(owner_shared.clone());
                let mut startup_tx = Some(startup_tx);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut backend = factory()?;
                    let output = backend.forward(&check.tokens)?;
                    if !exact_f32(&output, &check.embeddings) {
                        return Err(Code::InferenceFailed);
                    }
                    {
                        let mut core = lock(&owner_shared.core);
                        if core.state != Lifecycle::Starting {
                            return Err(Code::NotReady);
                        }
                        core.startup_output_sha256 = Some(output_digest(&output));
                        core.worker_alive = true;
                        core.state = Lifecycle::Ready;
                    }
                    if let Some(sender) = startup_tx.take() {
                        let _ = sender.send(Ok(()));
                    }
                    worker_loop(&mut backend, rx, &owner_shared);
                    Ok::<(), Code>(())
                }));
                if !matches!(result, Ok(Ok(()))) {
                    owner_shared.poison();
                }
                if let Some(sender) = startup_tx.take() {
                    let _ = sender.send(Err(Code::InferenceFailed));
                }
            })
            .map_err(|_| Code::InferenceFailed)?;
        let config = Arc::new(config);
        let app = App {
            admission: Arc::new(Semaphore::new(config.admission_capacity)),
            tokenizers: Arc::new(Semaphore::new(config.tokenizer_capacity)),
            config,
            tokenizer,
            logger,
            next_request: Arc::new(AtomicU64::new(1)),
            backend: BackendHandle { tx, shared },
        };
        Ok(Self {
            app,
            worker: Some(worker),
            startup: Some(startup_rx),
        })
    }

    pub async fn wait_ready(&mut self) -> Result<(), Code> {
        let receiver = self.startup.take().ok_or(Code::NotReady)?;
        match timeout(self.app.config.startup_timeout, receiver).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => {
                self.app.backend.shared.poison();
                Err(Code::ExecutorPoisoned)
            }
            Err(_) => {
                self.app.backend.shared.retire();
                Err(Code::DeadlineExceeded)
            }
        }
    }

    /// On Err the process must terminate, not construct another executor.
    /// A timed-out join remains owned by Tokio's blocking pool until it returns.
    pub async fn shutdown(&mut self) -> Result<(), Code> {
        self.app.begin_drain();
        let deadline = TokioInstant::now() + self.app.config.shutdown_grace;
        let count =
            u32::try_from(self.app.config.admission_capacity).map_err(|_| Code::InvalidRequest)?;

        let drain = timeout_at(
            deadline,
            self.app.admission.clone().acquire_many_owned(count),
        )
        .await;
        let all = match drain {
            Ok(Ok(permit)) => permit,
            _ => {
                self.app.backend.shared.retire();
                return Err(Code::DeadlineExceeded);
            }
        };

        {
            let mut core = lock(&self.app.backend.shared.core);
            core.expected_exit = true;
        }
        let sent = timeout_at(deadline, self.app.backend.tx.send(Message::Stop)).await;
        if !matches!(sent, Ok(Ok(()))) && !self.app.backend.tx.is_closed() {
            self.app.backend.shared.retire();
            return Err(Code::DeadlineExceeded);
        }
        drop(all);

        if let Some(worker) = self.worker.take() {
            let joined = tokio::task::spawn_blocking(move || worker.join());
            match timeout_at(deadline, joined).await {
                Ok(Ok(Ok(()))) => {}
                _ => {
                    self.app.backend.shared.retire();
                    return Err(Code::DeadlineExceeded);
                }
            }
        }
        Ok(())
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        // Dropping the HTTP/service handle never authorizes executor reuse.
        if self.worker.is_some() {
            self.app.backend.shared.retire();
            let _ = self.app.backend.tx.try_send(Message::Stop);
        }
    }
}

struct WorkerExit(Arc<Shared>);

impl Drop for WorkerExit {
    fn drop(&mut self) {
        let mut core = lock(&self.0.core);
        core.worker_alive = false;
        if core.expected_exit && core.state == Lifecycle::Draining {
            core.state = Lifecycle::Stopped;
        } else if !core.state.terminal() {
            core.state = Lifecycle::Poisoned;
        }
    }
}

fn worker_loop<B: Backend>(
    backend: &mut B,
    mut receiver: mpsc::Receiver<Message>,
    shared: &Shared,
) {
    while let Some(message) = receiver.blocking_recv() {
        match message {
            Message::Stop => {
                receiver.close();
                while let Ok(message) = receiver.try_recv() {
                    match message {
                        Message::Infer(job) => {
                            let _ = job.reply.send(Err(Code::NotReady));
                        }
                        Message::Control { reply, .. } => {
                            let _ = reply.send(Err(Code::NotReady));
                        }
                        Message::Stop => {}
                    }
                }
                return;
            }
            Message::Control {
                command,
                deadline,
                reply,
            } => {
                let result = if Instant::now() >= deadline {
                    Err(Code::DeadlineExceeded)
                } else {
                    owner_control(backend, shared, command)
                };
                let _ = reply.send(result);
            }
            Message::Infer(job) => run_job(backend, shared, job),
        }
    }
}

fn run_job<B: Backend>(backend: &mut B, shared: &Shared, job: Job) {
    let armed = {
        let mut core = lock(&shared.core);
        let mut phase = lock(&job.phase);
        if matches!(
            core.state,
            Lifecycle::Retired | Lifecycle::Poisoned | Lifecycle::Stopped
        ) {
            *phase = Phase::Done;
            let _ = job.reply.send(Err(Code::ExecutorPoisoned));
            return;
        }
        if *phase == Phase::Expired || Instant::now() >= job.deadline || job.reply.is_closed() {
            *phase = Phase::Expired;
            let _ = job.reply.send(Err(Code::DeadlineExceeded));
            return;
        }
        *phase = Phase::Running;
        core.forward_calls = match core.forward_calls.checked_add(1) {
            Some(value) => value,
            None => {
                core.state = Lifecycle::Poisoned;
                *phase = Phase::Done;
                let _ = job.reply.send(Err(Code::InferenceFailed));
                return;
            }
        };
        let armed = core.armed;
        core.armed = false;
        armed
    };

    // This call, including host output retrieval, owns the executor until it
    // returns. A timeout never moves or drops the backend on another thread.
    let mut result = if armed {
        backend
            .inject_policy_drift()
            .and_then(|()| backend.forward(&job.batches))
    } else {
        backend.forward(&job.batches)
    };
    if armed && backend.restore_policy().is_err() {
        result = Err(Code::InferenceFailed);
    }
    if let Ok(rows) = &result {
        if validate_output(rows, job.batches.len()).is_err() {
            result = Err(Code::InferenceFailed);
        }
    }
    // Backend classifications cannot escape as arbitrary public error codes.
    if result.is_err() {
        result = Err(Code::InferenceFailed);
    }

    {
        let mut core = lock(&shared.core);
        let mut phase = lock(&job.phase);
        if *phase == Phase::Expired || Instant::now() >= job.deadline {
            if !core.state.terminal() {
                core.state = Lifecycle::Retired;
            }
            *phase = Phase::Expired;
            result = Err(Code::DeadlineExceeded);
        } else if matches!(core.state, Lifecycle::Retired | Lifecycle::Poisoned) {
            *phase = Phase::Done;
            result = Err(Code::ExecutorPoisoned);
        } else {
            *phase = Phase::Done;
            if result.is_err() {
                core.state = Lifecycle::Poisoned;
            }
        }
    }
    let _ = job.reply.send(result);
}

/// Acceptance-only protocol, one authenticated JSON object per TCP connection.
///
/// Commands:
/// - quiesce: stop admission and wait for all admitted work, then owner barrier;
/// - inject_policy_drift: arm exactly the next admitted HTTP request;
/// - restore: restore flags on the owner, without clearing poison/retirement;
/// - observe: owner-thread observation and restart evidence.
///
/// If the one permitted request fails before submission, quiesce and arm again.
/// The drift is injected immediately before the designated backend call, and
/// automatically restored afterward. This is not an in-process recovery API.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControlCommand {
    Quiesce,
    InjectPolicyDrift,
    Restore,
    Observe,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ControlRequest {
    secret: String,
    command: ControlCommand,
}

#[derive(Serialize)]
pub struct Observation {
    instance: String,
    pid: u32,
    lifecycle: Lifecycle,
    worker_alive: bool,
    ready: bool,
    forward_calls: u64,
    startup_output_sha256: Option<String>,
}

fn owner_control<B: Backend>(
    backend: &mut B,
    shared: &Shared,
    command: ControlCommand,
) -> Result<Observation, Code> {
    if command == ControlCommand::Restore && backend.restore_policy().is_err() {
        shared.poison();
        return Err(Code::InferenceFailed);
    }
    let mut core = lock(&shared.core);
    match command {
        ControlCommand::Quiesce => {
            core.admission = AdmissionMode::Quiesced;
            core.armed = false;
        }
        ControlCommand::InjectPolicyDrift => {
            if core.state != Lifecycle::Ready
                || !matches!(core.admission, AdmissionMode::Quiesced)
                || core.armed
            {
                return Err(Code::NotReady);
            }
            core.armed = true;
            core.admission = AdmissionMode::One;
        }
        ControlCommand::Restore | ControlCommand::Observe => {}
    }
    Ok(Observation {
        instance: shared.instance.clone(),
        pid: std::process::id(),
        lifecycle: core.state,
        worker_alive: core.worker_alive,
        ready: core.state == Lifecycle::Ready
            && core.worker_alive
            && !matches!(core.admission, AdmissionMode::Quiesced),
        forward_calls: core.forward_calls,
        startup_output_sha256: core.startup_output_sha256.clone(),
    })
}

pub async fn bind_control(address: SocketAddr) -> Result<TcpListener, Code> {
    if !address.ip().is_loopback() {
        return Err(Code::InvalidRequest);
    }
    TcpListener::bind(address)
        .await
        .map_err(|_| Code::InvalidRequest)
}

fn secret_equal(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (left, right)| difference | (left ^ right))
        == 0
}

pub async fn serve_control(listener: TcpListener, app: App, secret: String) -> Result<(), Code> {
    if !listener
        .local_addr()
        .map_err(|_| Code::InvalidRequest)?
        .ip()
        .is_loopback()
        || secret.len() < 32
        || secret.len() > 256
    {
        return Err(Code::InvalidRequest);
    }
    // Sequential connections bound control memory and task count. The entire
    // exchange, including socket writes, has a frozen deadline.
    loop {
        let (mut socket, peer) = listener.accept().await.map_err(|_| Code::NotReady)?;
        if !peer.ip().is_loopback() {
            continue;
        }
        let exchange = async {
            let mut bytes = Vec::with_capacity(256);
            loop {
                let byte = socket.read_u8().await.map_err(|_| Code::InvalidRequest)?;
                if byte == b'\n' {
                    break;
                }
                if bytes.len() >= 4096 {
                    return Err(Code::PayloadTooLarge);
                }
                bytes.push(byte);
            }
            let request: ControlRequest =
                serde_json::from_slice(&bytes).map_err(|_| Code::InvalidRequest)?;
            if !secret_equal(request.secret.as_bytes(), secret.as_bytes()) {
                return Err(Code::InvalidRequest);
            }
            let result = app.control(request.command).await;
            let value = match result {
                Ok(observation) => json!({"ok": true, "observation": observation}),
                Err(code) => json!({"ok": false, "code": code}),
            };
            let mut output = serde_json::to_vec(&value).map_err(|_| Code::InferenceFailed)?;
            output.push(b'\n');
            socket
                .write_all(&output)
                .await
                .map_err(|_| Code::InvalidRequest)?;
            socket.shutdown().await.map_err(|_| Code::InvalidRequest)
        };
        let _ = timeout(app.config.control_timeout, exchange).await;
    }
}

/// Connection admission occurs before HTTP parsing or body buffering.
pub struct BoundedListener {
    listener: TcpListener,
    permits: Arc<Semaphore>,
    lifetime: Duration,
}

impl BoundedListener {
    pub fn new(listener: TcpListener, config: &TransportConfig) -> Result<Self, Code> {
        if config.max_connections == 0
            || config.max_connections > 4096
            || config.lifetime().is_zero()
        {
            return Err(Code::InvalidRequest);
        }
        Ok(Self {
            listener,
            permits: Arc::new(Semaphore::new(config.max_connections)),
            lifetime: config.lifetime(),
        })
    }
}

impl axum::serve::Listener for BoundedListener {
    type Io = TimedSocket;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.listener.accept().await {
                Ok((socket, address)) => {
                    let permit = match self.permits.clone().try_acquire_owned() {
                        Ok(permit) => permit,
                        Err(_) => {
                            drop(socket);
                            continue;
                        }
                    };
                    return (
                        TimedSocket {
                            socket,
                            expiry: Box::pin(sleep(self.lifetime)),
                            _permit: permit,
                        },
                        address,
                    );
                }
                Err(_) => sleep(Duration::from_millis(50)).await,
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}

pub struct TimedSocket {
    socket: TcpStream,
    expiry: Pin<Box<Sleep>>,
    _permit: OwnedSemaphorePermit,
}

impl TimedSocket {
    fn expired(&mut self, context: &mut Context<'_>) -> bool {
        self.expiry.as_mut().poll(context).is_ready()
    }

    fn timeout_error() -> io::Error {
        io::Error::new(io::ErrorKind::TimedOut, "transport deadline")
    }
}

impl AsyncRead for TimedSocket {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if this.expired(context) {
            return Poll::Ready(Err(Self::timeout_error()));
        }
        Pin::new(&mut this.socket).poll_read(context, buffer)
    }
}

impl AsyncWrite for TimedSocket {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if this.expired(context) {
            return Poll::Ready(Err(Self::timeout_error()));
        }
        Pin::new(&mut this.socket).poll_write(context, buffer)
    }

    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if this.expired(context) {
            return Poll::Ready(Err(Self::timeout_error()));
        }
        Pin::new(&mut this.socket).poll_flush(context)
    }

    fn poll_shutdown(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if this.expired(context) {
            return Poll::Ready(Err(Self::timeout_error()));
        }
        Pin::new(&mut this.socket).poll_shutdown(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request as HttpRequest};
    use std::{
        marker::PhantomData,
        rc::Rc,
        sync::atomic::{AtomicBool, AtomicUsize},
    };
    use tower::ServiceExt;

    fn vector(ids: &[u32]) -> Vec<f32> {
        let mut output = vec![0.0; EMBEDDING_WIDTH];
        output[0] = -0.0;
        let sum = ids.iter().fold(0u32, |sum, value| sum.wrapping_add(*value));
        let index = 1 + usize::try_from(sum % 2047).unwrap();
        output[index] = 1.0;
        output
    }

    struct Tokenizer;

    impl TextTokenizer for Tokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Code> {
            match text {
                "pad" => Ok(vec![PAD_ID]),
                "oov" => Ok(vec![u32::MAX]),
                "empty" => Ok(vec![]),
                "long" => Ok(vec![1; 1025]),
                "fail" => Err(Code::InvalidRequest),
                "two" => Ok(vec![2, 3]),
                _ => Ok(vec![1]),
            }
        }
    }

    struct Fake {
        calls: Arc<AtomicUsize>,
        delay: Arc<Mutex<Duration>>,
        fail: Arc<AtomicBool>,
        panic: Arc<AtomicBool>,
        drift: bool,
        _not_send: PhantomData<Rc<()>>,
    }

    impl Backend for Fake {
        fn forward(&mut self, batches: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, Code> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let delay = *lock(&self.delay);
            thread::sleep(delay);
            assert!(!self.panic.load(Ordering::SeqCst), "fake worker panic");
            if self.drift || self.fail.load(Ordering::SeqCst) {
                return Err(Code::InferenceFailed);
            }
            Ok(batches.iter().map(|ids| vector(ids)).collect())
        }

        fn inject_policy_drift(&mut self) -> Result<(), Code> {
            self.drift = true;
            Ok(())
        }

        fn restore_policy(&mut self) -> Result<(), Code> {
            self.drift = false;
            Ok(())
        }
    }

    #[derive(Clone)]
    struct MemoryLog(Arc<Mutex<Vec<u8>>>);

    impl Write for MemoryLog {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            lock(&self.0).extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct Fixture {
        service: Service,
        calls: Arc<AtomicUsize>,
        delay: Arc<Mutex<Duration>>,
        fail: Arc<AtomicBool>,
        panic: Arc<AtomicBool>,
        logs: Arc<Mutex<Vec<u8>>>,
    }

    async fn fixture(mut config: Config) -> Fixture {
        config.startup_timeout = Duration::from_secs(2);
        config.shutdown_grace = Duration::from_secs(3);
        let calls = Arc::new(AtomicUsize::new(0));
        let delay = Arc::new(Mutex::new(Duration::ZERO));
        let fail = Arc::new(AtomicBool::new(false));
        let panic = Arc::new(AtomicBool::new(false));
        let logs = Arc::new(Mutex::new(Vec::new()));
        let factory_calls = calls.clone();
        let factory_delay = delay.clone();
        let factory_fail = fail.clone();
        let factory_panic = panic.clone();
        let mut service = Service::spawn(
            config,
            Arc::new(Tokenizer),
            Logger::new(MemoryLog(logs.clone())),
            StartupCheck {
                tokens: vec![vec![1]],
                embeddings: vec![vector(&[1])],
            },
            move || {
                Ok(Fake {
                    calls: factory_calls,
                    delay: factory_delay,
                    fail: factory_fail,
                    panic: factory_panic,
                    drift: false,
                    _not_send: PhantomData,
                })
            },
        )
        .unwrap();
        service.wait_ready().await.unwrap();
        calls.store(0, Ordering::SeqCst);
        Fixture {
            service,
            calls,
            delay,
            fail,
            panic,
            logs,
        }
    }

    fn config() -> Config {
        Config::production(8, 131_072, 1024)
    }

    fn body(input: Value) -> Value {
        json!({"model": MODEL_ID, "input": input})
    }

    async fn send(app: &App, value: Value) -> (StatusCode, Vec<u8>) {
        raw(
            app,
            serde_json::to_vec(&value).unwrap(),
            Some("application/json"),
            None,
        )
        .await
    }

    async fn raw(
        app: &App,
        bytes: Vec<u8>,
        content_type: Option<&str>,
        content_encoding: Option<&str>,
    ) -> (StatusCode, Vec<u8>) {
        let mut request = HttpRequest::post("/v1/embeddings");
        if let Some(value) = content_type {
            request = request.header(header::CONTENT_TYPE, value);
        }
        if let Some(value) = content_encoding {
            request = request.header(header::CONTENT_ENCODING, value);
        }
        let response = app
            .router()
            .oneshot(request.body(Body::from(bytes)).unwrap())
            .await
            .unwrap();
        assert!(response.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("application/json"));
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1_000_000)
            .await
            .unwrap();
        (status, bytes.to_vec())
    }

    fn error(result: (StatusCode, Vec<u8>), expected: Code) {
        assert_eq!(result.0, expected.status());
        let parsed: Value = serde_json::from_slice(&result.1).unwrap();
        assert_eq!(
            parsed["error"]["code"],
            serde_json::to_value(expected).unwrap()
        );
        assert_eq!(parsed["error"]["message"], expected.message());
        assert_eq!(parsed.as_object().unwrap().len(), 1);
        assert_eq!(parsed["error"].as_object().unwrap().len(), 4);
    }

    async fn wait_calls(calls: &AtomicUsize, count: usize) {
        timeout(Duration::from_secs(2), async {
            while calls.load(Ordering::SeqCst) < count {
                sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
    }

    #[derive(Deserialize)]
    struct FloatResponse {
        data: Vec<FloatRow>,
        usage: FloatUsage,
    }

    #[derive(Deserialize)]
    struct FloatRow {
        index: usize,
        embedding: Vec<f32>,
    }

    #[derive(Deserialize)]
    struct FloatUsage {
        prompt_tokens: usize,
        total_tokens: usize,
    }

    #[tokio::test]
    async fn float_base64_order_usage_and_signed_zero() {
        let mut f = fixture(config()).await;
        for encoding in [Value::Null, json!("float"), json!("base64")] {
            let mut request = body(json!(["two", "one", "two"]));
            request["encoding_format"] = encoding.clone();
            let (status, bytes) = send(&f.service.app, request).await;
            assert_eq!(status, StatusCode::OK);
            let golden = vec![vector(&[2, 3]), vector(&[1]), vector(&[2, 3])];
            if encoding == "base64" {
                let parsed: Value = serde_json::from_slice(&bytes).unwrap();
                let rows: Vec<Vec<f32>> = parsed["data"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(index, row)| {
                        assert_eq!(row["index"], index);
                        let encoded = row["embedding"].as_str().unwrap();
                        assert!(encoded.ends_with('='));
                        let decoded = STANDARD.decode(encoded).unwrap();
                        assert_eq!(decoded.len(), 2048 * 4);
                        decoded
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .copied()
                            .map(f32::from_le_bytes)
                            .collect()
                    })
                    .collect();
                assert!(exact_f32(&rows, &golden));
                assert_eq!(parsed["usage"]["prompt_tokens"], 5);
                assert_eq!(parsed["usage"]["total_tokens"], 5);
            } else {
                let parsed: FloatResponse = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(parsed.usage.prompt_tokens, 5);
                assert_eq!(parsed.usage.total_tokens, 5);
                let rows = parsed
                    .data
                    .into_iter()
                    .enumerate()
                    .map(|(index, row)| {
                        assert_eq!(row.index, index);
                        row.embedding
                    })
                    .collect::<Vec<_>>();
                assert!(exact_f32(&rows, &golden));
            }
        }
        assert_eq!(
            send(&f.service.app, body(json!("one"))).await.0,
            StatusCode::OK
        );
        f.service.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn validation_matrix_and_whole_request_validation() {
        let mut f = fixture(config()).await;
        let cases = [
            (body(json!([])), Code::EmptyBatch),
            (body(json!(" \t\n")), Code::EmptySequence),
            (body(json!(["one", ""])), Code::EmptySequence),
            (body(json!("empty")), Code::EmptySequence),
            (body(json!("pad")), Code::PadToken),
            (body(json!("oov")), Code::OutOfVocabulary),
            (body(json!("long")), Code::OverLength),
            (body(json!("fail")), Code::InvalidRequest),
            (body(json!(["one", "pad"])), Code::PadToken),
            (body(json!(vec!["one"; 9])), Code::InvalidRequest),
            (body(json!([1, 2])), Code::InvalidRequest),
            (body(json!([["one"]])), Code::InvalidRequest),
            (body(Value::Null), Code::InvalidRequest),
            (json!({"input": "one"}), Code::InvalidRequest),
            (json!({"model": MODEL_ID}), Code::InvalidRequest),
            (
                json!({"model": "secret-model", "input": "one"}),
                Code::InvalidModel,
            ),
            (
                json!({"model": MODEL_ID, "input": "one", "dimensions": 3}),
                Code::InvalidRequest,
            ),
            (
                json!({"model": MODEL_ID, "input": "one", "user": 5}),
                Code::InvalidRequest,
            ),
            (
                json!({"model": MODEL_ID, "input": "one", "encoding_format": "other"}),
                Code::UnsupportedEncodingFormat,
            ),
            (
                json!({"model": MODEL_ID, "input": "one", "encoding_format": 42}),
                Code::UnsupportedEncodingFormat,
            ),
        ];
        for (request, code) in cases {
            error(send(&f.service.app, request).await, code);
        }
        error(
            raw(
                &f.service.app,
                b"{secret".to_vec(),
                Some("application/json"),
                None,
            )
            .await,
            Code::InvalidRequest,
        );
        error(
            raw(
                &f.service.app,
                b"{\"model\":\"x\",\"model\":\"y\",\"input\":\"z\"}".to_vec(),
                Some("application/json"),
                None,
            )
            .await,
            Code::InvalidRequest,
        );
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);

        for media in [None, Some("text/plain"), Some("application/json; x=y")] {
            error(
                raw(&f.service.app, b"{}".to_vec(), media, None).await,
                Code::UnsupportedMediaType,
            );
        }
        error(
            raw(
                &f.service.app,
                b"{}".to_vec(),
                Some("application/json"),
                Some("gzip"),
            )
            .await,
            Code::UnsupportedMediaType,
        );
        error(
            raw(
                &f.service.app,
                vec![b' '; BODY_LIMIT + 1],
                Some("application/json"),
                None,
            )
            .await,
            Code::PayloadTooLarge,
        );
        assert_eq!(
            raw(
                &f.service.app,
                serde_json::to_vec(&body(json!("one"))).unwrap(),
                Some("application/json; charset=utf-8"),
                None,
            )
            .await
            .0,
            StatusCode::OK
        );
        f.service.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn fallback_and_readiness_use_production_router() {
        let mut f = fixture(config()).await;
        for (method, path, status) in [
            ("GET", "/missing?private=query", StatusCode::NOT_FOUND),
            ("GET", "/v1/embeddings", StatusCode::METHOD_NOT_ALLOWED),
            ("GET", "/health/readyz", StatusCode::OK),
            ("GET", "/health/livez", StatusCode::OK),
        ] {
            let response = f
                .service
                .app
                .router()
                .oneshot(
                    HttpRequest::builder()
                        .method(method)
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), status);
        }
        f.service.app.begin_drain();
        assert!(!f.service.app.is_ready());
        error(
            send(&f.service.app, body(json!("one"))).await,
            Code::NotReady,
        );
        f.service.shutdown().await.unwrap();
        assert_eq!(f.service.app.lifecycle(), Lifecycle::Stopped);
    }

    #[tokio::test]
    async fn queue_full_and_expired_queued_job_do_not_retire() {
        let mut cfg = config();
        cfg.queue_capacity = 1;
        let mut f = fixture(cfg).await;
        *lock(&f.delay) = Duration::from_millis(150);
        let app = f.service.app.clone();
        let permit = app.admit().unwrap();
        let handle = app.backend.clone();
        let first = tokio::spawn(async move {
            handle
                .infer(
                    vec![vec![1]],
                    Instant::now() + Duration::from_secs(2),
                    permit,
                )
                .await
        });
        wait_calls(&f.calls, 1).await;

        let second_app = f.service.app.clone();
        let second_permit = second_app.admit().unwrap();
        let second = tokio::spawn(async move {
            second_app
                .backend
                .infer(
                    vec![vec![2]],
                    Instant::now() + Duration::from_millis(30),
                    second_permit,
                )
                .await
        });
        sleep(Duration::from_millis(5)).await;
        let permit = f.service.app.admit().unwrap();
        assert_eq!(
            f.service
                .app
                .backend
                .infer(
                    vec![vec![3]],
                    Instant::now() + Duration::from_secs(2),
                    permit
                )
                .await
                .unwrap_err(),
            Code::QueueFull
        );
        assert_eq!(second.await.unwrap().unwrap_err(), Code::DeadlineExceeded);
        first.await.unwrap().unwrap();
        sleep(Duration::from_millis(10)).await;
        assert!(f.service.app.is_ready());
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        f.service.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn execution_timeout_retires_and_discards_late_result() {
        let mut cfg = config();
        cfg.request_timeout = Duration::from_millis(40);
        let mut f = fixture(cfg).await;
        *lock(&f.delay) = Duration::from_millis(140);
        error(
            send(&f.service.app, body(json!("one"))).await,
            Code::DeadlineExceeded,
        );
        assert_eq!(f.service.app.lifecycle(), Lifecycle::Retired);
        error(
            send(&f.service.app, body(json!("one"))).await,
            Code::ExecutorPoisoned,
        );
        sleep(Duration::from_millis(160)).await;
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.service.app.lifecycle(), Lifecycle::Retired);
        f.service.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn disconnect_does_not_release_executor_or_admission() {
        let mut cfg = config();
        cfg.admission_capacity = 1;
        cfg.tokenizer_capacity = 1;
        let mut f = fixture(cfg).await;
        *lock(&f.delay) = Duration::from_millis(120);
        let app = f.service.app.clone();
        let client = tokio::spawn(async move { send(&app, body(json!("one"))).await });
        wait_calls(&f.calls, 1).await;
        client.abort();
        error(
            send(&f.service.app, body(json!("two"))).await,
            Code::QueueFull,
        );
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        sleep(Duration::from_millis(150)).await;
        assert!(f.service.app.is_ready());
        assert_eq!(
            send(&f.service.app, body(json!("two"))).await.0,
            StatusCode::OK
        );
        f.service.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn worker_failure_and_closed_queue_are_terminal() {
        for panic in [false, true] {
            let mut f = fixture(config()).await;
            if panic {
                f.panic.store(true, Ordering::SeqCst);
            } else {
                f.fail.store(true, Ordering::SeqCst);
            }
            let result = send(&f.service.app, body(json!("one"))).await;
            error(
                result,
                if panic {
                    Code::ExecutorPoisoned
                } else {
                    Code::InferenceFailed
                },
            );
            assert!(!f.service.app.is_ready());
            error(
                send(&f.service.app, body(json!("two"))).await,
                Code::ExecutorPoisoned,
            );
            assert_eq!(f.calls.load(Ordering::SeqCst), 1);
            f.service.shutdown().await.unwrap();
        }
    }

    #[tokio::test]
    async fn unexpected_idle_exit_is_supervised_without_a_request() {
        let mut f = fixture(config()).await;
        f.service.app.backend.tx.send(Message::Stop).await.unwrap();
        timeout(Duration::from_secs(2), async {
            while f.service.app.is_ready() {
                sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(f.service.app.lifecycle(), Lifecycle::Poisoned);
        assert!(f.service.app.backend.tx.is_closed());
        f.service.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn control_poison_restore_and_restart_evidence() {
        let mut f = fixture(config()).await;
        let before = f
            .service
            .app
            .control(ControlCommand::Observe)
            .await
            .unwrap();
        assert!(before.ready);
        f.service
            .app
            .control(ControlCommand::Quiesce)
            .await
            .unwrap();
        error(
            send(&f.service.app, body(json!("one"))).await,
            Code::NotReady,
        );
        f.service
            .app
            .control(ControlCommand::InjectPolicyDrift)
            .await
            .unwrap();
        error(
            send(&f.service.app, body(json!("one"))).await,
            Code::InferenceFailed,
        );
        let restored = f
            .service
            .app
            .control(ControlCommand::Restore)
            .await
            .unwrap();
        assert_eq!(restored.lifecycle, Lifecycle::Poisoned);
        assert!(!restored.ready);
        assert_eq!(restored.forward_calls, 1);
        error(
            send(&f.service.app, body(json!("one"))).await,
            Code::ExecutorPoisoned,
        );
        f.service.shutdown().await.unwrap();

        let mut fresh = fixture(config()).await;
        let after = fresh
            .service
            .app
            .control(ControlCommand::Observe)
            .await
            .unwrap();
        assert_ne!(before.instance, after.instance);
        assert_eq!(before.startup_output_sha256, after.startup_output_sha256);
        assert!(after.ready);
        fresh.service.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn graceful_drain_finishes_admitted_work() {
        let mut f = fixture(config()).await;
        *lock(&f.delay) = Duration::from_millis(80);
        let app = f.service.app.clone();
        let admitted = tokio::spawn(async move { send(&app, body(json!("one"))).await });
        wait_calls(&f.calls, 1).await;
        f.service.app.begin_drain();
        error(
            send(&f.service.app, body(json!("two"))).await,
            Code::NotReady,
        );
        f.service.shutdown().await.unwrap();
        assert_eq!(admitted.await.unwrap().0, StatusCode::OK);
        assert_eq!(f.service.app.lifecycle(), Lifecycle::Stopped);
    }

    #[tokio::test]
    async fn concurrent_association_and_sanitized_logs() {
        let mut f = fixture(config()).await;
        let mut clients = Vec::new();
        for index in 0..4 {
            let app = f.service.app.clone();
            clients.push(tokio::spawn(async move {
                let input = if index % 2 == 0 { "one" } else { "two" };
                let mut request = body(json!(input));
                request["user"] = json!("PRIVATE-USER");
                (input, send(&app, request).await)
            }));
        }
        for client in clients {
            let (input, (status, bytes)) = client.await.unwrap();
            assert_eq!(status, StatusCode::OK);
            let parsed: FloatResponse = serde_json::from_slice(&bytes).unwrap();
            let golden = if input == "one" {
                vector(&[1])
            } else {
                vector(&[2, 3])
            };
            assert!(exact_f32(&[parsed.data[0].embedding.clone()], &[golden]));
        }
        let _ = send(
            &f.service.app,
            json!({"model":"PRIVATE-MODEL","input":"PRIVATE-TEXT","user":"PRIVATE-USER"}),
        )
        .await;
        let logs = String::from_utf8(lock(&f.logs).clone()).unwrap();
        assert!(!logs.contains("PRIVATE"));
        for line in logs.lines() {
            let value: Value = serde_json::from_str(line).unwrap();
            let allowed = [
                "event",
                "request_id",
                "route",
                "status",
                "code",
                "elapsed_ms",
            ];
            assert!(value
                .as_object()
                .unwrap()
                .keys()
                .all(|key| allowed.contains(&key.as_str())));
        }
        f.service.shutdown().await.unwrap();
    }

    async fn socket_server(
        app: &App,
        transport: &TransportConfig,
    ) -> (SocketAddr, oneshot::Sender<()>, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let listener = BoundedListener::new(listener, transport).unwrap();
        let router = app.router();
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        (address, stop, task)
    }

    #[tokio::test]
    async fn real_socket_chunked_limit_and_shutdown() {
        let mut cfg = config();
        cfg.max_body_bytes = 128;
        let mut f = fixture(cfg).await;
        let (address, stop, server) =
            socket_server(&f.service.app, &f.service.app.config.transport).await;
        let mut socket = TcpStream::connect(address).await.unwrap();
        socket
            .write_all(
                b"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\n\
                  Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n\
                  Connection: close\r\n\r\n",
            )
            .await
            .unwrap();
        for _ in 0..3 {
            socket.write_all(b"40\r\n").await.unwrap();
            socket.write_all(&[b' '; 64]).await.unwrap();
            socket.write_all(b"\r\n").await.unwrap();
        }
        let _ = socket.write_all(b"0\r\n\r\n").await;
        let mut response = Vec::new();
        timeout(Duration::from_secs(2), socket.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        let text = String::from_utf8(response).unwrap();
        assert!(text.starts_with("HTTP/1.1 413"));
        assert!(text.contains("\"code\":\"payload_too_large\""));
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        f.service.app.begin_drain();
        stop.send(()).unwrap();
        f.service.shutdown().await.unwrap();
        timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap();
        assert!(TcpStream::connect(address).await.is_err());
    }

    #[tokio::test]
    async fn real_socket_slow_headers_are_bounded() {
        let mut f = fixture(config()).await;
        let mut transport = f.service.app.config.transport.clone();
        transport.header_timeout = Duration::from_millis(80);
        let (address, stop, server) = socket_server(&f.service.app, &transport).await;
        let mut socket = TcpStream::connect(address).await.unwrap();
        socket.write_all(b"POST /v1/").await.unwrap();
        sleep(Duration::from_millis(140)).await;
        let mut byte = [0u8; 1];
        let read = timeout(Duration::from_secs(1), socket.read(&mut byte))
            .await
            .unwrap();
        assert!(matches!(read, Ok(0) | Err(_)));
        stop.send(()).unwrap();
        f.service.shutdown().await.unwrap();
        timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn real_socket_slow_body_has_json_deadline() {
        let mut cfg = config();
        cfg.body_timeout = Duration::from_millis(60);
        let mut f = fixture(cfg).await;
        let (address, stop, server) =
            socket_server(&f.service.app, &f.service.app.config.transport).await;
        let mut socket = TcpStream::connect(address).await.unwrap();
        socket
            .write_all(
                b"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\n\
                  Content-Type: application/json\r\nContent-Length: 100\r\n\
                  Connection: close\r\n\r\n{",
            )
            .await
            .unwrap();
        let mut response = Vec::new();
        timeout(Duration::from_secs(2), socket.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        let text = String::from_utf8(response).unwrap();
        assert!(text.starts_with("HTTP/1.1 504"));
        assert!(text.contains("\"code\":\"deadline_exceeded\""));
        assert!(f.service.app.is_ready());
        stop.send(()).unwrap();
        f.service.shutdown().await.unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn transport_write_wait_is_bounded_without_a_reading_peer() {
        // Exercise the production transport wrapper directly with enough data
        // to exhaust TCP buffering. Small embedding responses alone cannot
        // reliably force write backpressure on loopback.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let _peer = TcpStream::connect(address).await.unwrap();
        let (socket, _) = listener.accept().await.unwrap();
        let permit = Arc::new(Semaphore::new(1)).acquire_owned().await.unwrap();
        let mut socket = TimedSocket {
            socket,
            expiry: Box::pin(sleep(Duration::from_millis(80))),
            _permit: permit,
        };
        let data = vec![0u8; 64 * 1024];
        let result = timeout(Duration::from_secs(2), async {
            loop {
                socket.write_all(&data).await?;
            }
            #[allow(unreachable_code)]
            Ok::<(), io::Error>(())
        })
        .await
        .unwrap();
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::TimedOut);
    }

    #[test]
    fn configuration_and_output_guards() {
        let mut cfg = config();
        cfg.queue_capacity = 0;
        assert_eq!(cfg.validate(), Err(Code::InvalidRequest));
        cfg.queue_capacity = 4097;
        assert_eq!(cfg.validate(), Err(Code::InvalidRequest));

        let row = vector(&[1]);
        assert!(validate_output(std::slice::from_ref(&row), 1).is_ok());
        assert!(validate_output(&[], 1).is_err());
        assert!(validate_output(&[vec![0.0; 2048]], 1).is_err());
        let mut invalid = row.clone();
        invalid[1] = f32::NAN;
        assert!(validate_output(&[invalid], 1).is_err());
        let mut different_zero = row.clone();
        different_zero[0] = 0.0;
        assert!(!exact_f32(&[row], &[different_zero]));
    }
}
