//! CUDA-independent M3 HTTP serving with A1–A6 and B1–B3 conformance.

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, FromRequest, Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::{self, Next},
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
    pub drain_seconds: u64,
    pub instance: Option<String>,
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
            drain_seconds: 30,
            transport: TransportConfig {
                max_connections: 128,
                header_timeout: Duration::from_secs(90),
                write_timeout: Duration::from_secs(90),
                idle_timeout: Duration::from_secs(90),
                connection_timeout: Duration::from_secs(90),
            },
            instance: None,
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
    launch_attestation: Mutex<Option<Value>>,
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
    pub config: Arc<Config>,
    pub backend: BackendHandle,
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
            .layer(middleware::from_fn_with_state(self.clone(), correlate))
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

    pub fn install_launch_attestation(&self, mut value: Value) -> Result<(), Code> {
        let core = lock(&self.backend.shared.core);
        if core.state != Lifecycle::Ready
            || !core.worker_alive
            || core.forward_calls != 0
            || core.startup_output_sha256.is_none()
            || !value.is_object()
        {
            return Err(Code::NotReady);
        }
        let mut slot = lock(&self.backend.shared.launch_attestation);
        if slot.is_some() {
            return Err(Code::InvalidRequest);
        }
        value["instance_id"] = json!(self.backend.shared.instance);
        value["startup"] = json!({
            "completed": true,
            "comparison": "exact_f32_bits",
            "output_f32le_sha256": core.startup_output_sha256,
        });
        value["capabilities"] = json!([
            "launch_attestation.v1",
            "acceptance-control.v1",
            "quiesce",
            "inject_policy_drift",
            "restore",
            "observe"
        ]);
        *slot = Some(value);
        Ok(())
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

        let tokenizer_permit = timeout_at(
            TokioInstant::from_std(deadline),
            self.tokenizers.clone().acquire_owned(),
        )
        .await
        .map_err(|_| Code::DeadlineExceeded)?
        .map_err(|_| Code::NotReady)?;

        let tokenizer = self.tokenizer.clone();
        let config = self.config.clone();
        let blocking_admission = permit.clone();
        let work = tokio::task::spawn_blocking(move || {
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

    pub async fn control(&self, command: ControlCommand) -> Result<Observation, Code> {
        if command == ControlCommand::Quiesce {
            {
                let mut core = lock(&self.backend.shared.core);
                core.admission = AdmissionMode::Quiesced;
            }
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

async fn correlate(State(app): State<App>, request: Request, next: Next) -> Response {
    let values = request.headers().get_all("x-acceptance-case-id");
    let count = values.iter().count();
    let case = values.iter().next().cloned();
    let valid = count <= 1
        && case.as_ref().is_none_or(|value| {
            value.to_str().is_ok_and(|text| {
                !text.is_empty()
                    && text.len() <= 128
                    && text.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
            })
        });
    let mut response = if valid {
        next.run(request).await
    } else {
        Code::InvalidRequest.into_response()
    };
    if let Ok(instance) = app.backend.shared.instance.parse() {
        response.headers_mut().insert("x-rv1-instance-id", instance);
    }
    if valid {
        if let Some(case) = case {
            response.headers_mut().insert("x-acceptance-case-id", case);
        }
    }
    response
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
            instance: config
                .instance
                .clone()
                .unwrap_or_else(|| format!("{}-{stamp}", std::process::id())),
            launch_attestation: Mutex::new(None),
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

    pub async fn shutdown(&mut self) -> Result<(), Code> {
        self.app.begin_drain();
        let deadline = TokioInstant::now() + Duration::from_secs(self.app.config.drain_seconds);
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

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControlCommand {
    Quiesce,
    InjectPolicyDrift,
    Restore,
    Observe,
    Attest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ControlRequest {
    secret: String,
    command: ControlCommand,
    #[serde(default)]
    challenge: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    attestation: Option<Value>,
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
    if command == ControlCommand::Attest
        && (core.state != Lifecycle::Ready
            || !core.worker_alive
            || core.forward_calls != 0
            || lock(&shared.launch_attestation).is_none())
    {
        return Err(Code::NotReady);
    }
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
        ControlCommand::Restore | ControlCommand::Observe | ControlCommand::Attest => {}
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
        attestation: if command == ControlCommand::Attest {
            lock(&shared.launch_attestation).clone()
        } else {
            None
        },
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
    loop {
        let (mut socket, peer) = listener.accept().await.map_err(|_| Code::NotReady)?;
        if !peer.ip().is_loopback() {
            continue;
        }
        let app = app.clone();
        let secret = secret.clone();
        let control_timeout = app.config.control_timeout;
        let exchange = async move {
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
            if request.command == ControlCommand::Attest {
                let challenge = request.challenge.as_deref().ok_or(Code::InvalidRequest)?;
                if challenge.len() != 64
                    || !challenge
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return Err(Code::InvalidRequest);
                }
            } else if request.challenge.is_some() {
                return Err(Code::InvalidRequest);
            }
            let result = app.control(request.command).await;
            let value = match result {
                Ok(observation) => json!({
                    "ok": true,
                    "challenge": request.challenge,
                    "observation": observation
                }),
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
        let _ = timeout(control_timeout, exchange).await;
    }
}

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
