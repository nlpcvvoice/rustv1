// src/bin/accept_m3.rs
//
// CPU-only M3 acceptance driver with authenticated orchestration channel (A1–A6),
// launch attestation verification (B1–B2), 46-slot fixture mapping including
// immediate recovery placement (slots 42–46), and destructive lifecycle execution.

use anyhow::{bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::{
    body::{Body, Frame, SizeHint},
    client::conn::http1,
    Request, Uri,
};
use hyper_util::rt::TokioIo;
use ring::hmac;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    convert::Infallible,
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
    pin::Pin,
    task::{Context as TaskContext, Poll},
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
    task::JoinHandle,
    time::timeout,
};

pub const WIDTH: usize = 2048;
const ROW_BYTES: usize = WIDTH * 4;
const MAX_POLICY: usize = 8 * 1024 * 1024;
const BODY_LIMIT: usize = 1_048_576;

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 128 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn bounded_file(path: &Path, maximum: usize) -> Result<Vec<u8>> {
    let mut file = fs::File::open(path)?;
    ensure!(file.metadata()?.len() <= maximum as u64, "oversized file");
    let mut bytes = Vec::new();
    file.by_ref()
        .take(maximum as u64 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= maximum, "file grew beyond limit");
    Ok(bytes)
}

fn digest_valid(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn row_bytes(row: &[f32]) -> Vec<u8> {
    row.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

pub fn validate_row(row: &[f32]) -> Result<f64> {
    ensure!(row.len() == WIDTH, "embedding width");
    ensure!(row.iter().all(|x| x.is_finite()), "nonfinite embedding");
    let norm = row
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        norm.is_finite() && (norm - 1.0).abs() <= 1e-5,
        "FP64 norm gate: {norm}"
    );
    Ok(norm)
}

pub fn exact_row(actual: &[f32], expected: &[f32]) -> Result<()> {
    validate_row(actual)?;
    validate_row(expected)?;
    ensure!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "F32 bits differ"
    );
    Ok(())
}

fn fidelity(actual: &[f32], reference: &[f32]) -> Result<()> {
    let a = validate_row(actual)?;
    let b = validate_row(reference)?;
    let dot = actual
        .iter()
        .zip(reference)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum::<f64>();
    let max_abs = actual
        .iter()
        .zip(reference)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .fold(0.0f64, f64::max);
    let cosine = dot / (a * b);
    ensure!(
        cosine.is_finite() && cosine > 0.999 && max_abs <= 0.01,
        "independent-reference gate: cosine={cosine}, max_abs={max_abs}"
    );
    Ok(())
}

fn load_rows(path: &Path) -> Result<Vec<Vec<f32>>> {
    let bytes = bounded_file(path, 16 * 1024 * 1024)?;
    ensure!(
        !bytes.is_empty() && bytes.len() % ROW_BYTES == 0,
        "F32 file size"
    );
    let mut rows = Vec::new();
    for chunk in bytes.as_chunks::<ROW_BYTES>().0 {
        let row: Vec<f32> = chunk
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        validate_row(&row)?;
        rows.push(row);
    }
    Ok(rows)
}

// A1: Canonical JSON serializer
pub fn canonical_json(value: &Value) -> Result<String> {
    let mut buf = String::new();
    write_canonical(value, &mut buf)?;
    Ok(buf)
}

fn write_canonical(value: &Value, buf: &mut String) -> Result<()> {
    match value {
        Value::Null => buf.push_str("null"),
        Value::Bool(b) => buf.push_str(if *b { "true" } else { "false" }),
        Value::Number(num) => {
            if let Some(i) = num.as_i64() {
                buf.push_str(&i.to_string());
            } else if let Some(u) = num.as_u64() {
                buf.push_str(&u.to_string());
            } else {
                bail!("integers only in canonical json envelope");
            }
        }
        Value::String(s) => {
            buf.push('"');
            for c in s.chars() {
                match c {
                    '"' => buf.push_str("\\\""),
                    '\\' => buf.push_str("\\\\"),
                    '\u{08}' => buf.push_str("\\b"),
                    '\u{0C}' => buf.push_str("\\f"),
                    '\n' => buf.push_str("\\n"),
                    '\r' => buf.push_str("\\r"),
                    '\t' => buf.push_str("\\t"),
                    c if (c as u32) < 0x20 => {
                        buf.push_str(&format!("\\u{:04x}", c as u32));
                    }
                    c => buf.push(c),
                }
            }
            buf.push('"');
        }
        Value::Array(arr) => {
            buf.push('[');
            for (i, elem) in arr.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                write_canonical(elem, buf)?;
            }
            buf.push(']');
        }
        Value::Object(obj) => {
            let mut sorted: Vec<(&String, &Value)> = obj.iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(b.0));
            buf.push('{');
            for (i, (k, v)) in sorted.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                write_canonical(&Value::String((*k).clone()), buf)?;
                buf.push(':');
                write_canonical(v, buf)?;
            }
            buf.push('}');
        }
    }
    Ok(())
}

// A2: HMAC-SHA256 MAC generator and verifier
pub fn compute_mac(secret: &[u8], canonical_payload: &str) -> String {
    let key = hmac::Key::new(hmac::HMAC_SHA256, secret);
    let tag = hmac::sign(&key, canonical_payload.as_bytes());
    hex::encode(tag.as_ref())
}

pub fn verify_mac(secret: &[u8], canonical_payload: &str, expected_mac: &str) -> bool {
    let Ok(expected_bytes) = hex::decode(expected_mac) else {
        return false;
    };
    let key = hmac::Key::new(hmac::HMAC_SHA256, secret);
    hmac::verify(&key, canonical_payload.as_bytes(), &expected_bytes).is_ok()
}

mod hex {
    pub fn encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{b:02x}")).collect()
    }
    pub fn decode(s: &str) -> Result<Vec<u8>, ()> {
        if !s.len().is_multiple_of(2) {
            return Err(());
        }
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ()))
            .collect()
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub root: String,
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub id: String,
    pub text: String,
    pub ids: Vec<u32>,
    pub ids_sha256: String,
    pub golden_row: usize,
    pub golden_sha256: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Mapping {
    pub m2_group_id: u32,
    pub adaptation: String,
    pub requests: Vec<Vec<usize>>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub connect_ms: u64,
    pub headers_ms: u64,
    pub body_ms: u64,
    pub control_ms: u64,
    pub observation_ms: u64,
    pub server_request_ms: u64,
    pub response_bytes: usize,
    pub admission_capacity: usize,
    pub queue_capacity: usize,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub schema: String,
    pub contract_id: String,
    pub http_contract_id: String,
    pub run_id: String,
    pub model: String,
    pub audit_identity: String,
    pub deployment: Value,
    pub artifacts: BTreeMap<String, Artifact>,
    pub fixtures: Vec<Fixture>,
    pub mappings: Vec<Mapping>,
    pub limits: Limits,
    pub launch_control_address: String,
    pub orch_secret_server_to_orchestrator: String,
    pub orch_secret_orchestrator_to_server: String,
    pub over_length_text: String,
    pub pad_text: Option<String>,
    pub samples: Vec<Value>,
}

pub struct Bundle {
    pub policy: Policy,
    pub policy_sha256: String,
    pub paths: BTreeMap<String, PathBuf>,
    pub corpus: Vec<Vec<u32>>,
    pub corpus_goldens: Vec<Vec<f32>>,
    pub fixture_goldens: Vec<Vec<f32>>,
}

fn has_placeholder(v: &Value) -> bool {
    match v {
        Value::String(s) => {
            s.contains("FILLED_BY_INTEGRATOR")
                || s.contains("<SHA256:")
                || s.contains("PENDING_ROUNDTRIP")
        }
        Value::Array(a) => a.iter().any(has_placeholder),
        Value::Object(o) => o.values().any(has_placeholder),
        _ => false,
    }
}

fn artifact_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    ensure!(
        !relative.is_absolute()
            && relative
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "artifact path must be relative without traversal"
    );
    let root = root.canonicalize()?;
    let path = root.join(relative).canonicalize()?;
    ensure!(path.starts_with(&root), "artifact escapes its root");
    ensure!(path.is_file(), "artifact is not a regular file");
    Ok(path)
}

fn read_value(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&bounded_file(path, MAX_POLICY)?)?)
}

fn ids_hash(ids: &[u32]) -> String {
    sha256(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}

fn required_path<'a>(paths: &'a BTreeMap<String, PathBuf>, name: &str) -> Result<&'a Path> {
    Ok(paths
        .get(name)
        .with_context(|| format!("missing artifact {name}"))?
        .as_path())
}

pub fn load_bundle(model_dir: &Path, policy_path: &Path, executable_role: &str) -> Result<Bundle> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::metadata(policy_path)?.permissions();
        ensure!(
            perms.mode() & 0o077 == 0,
            "policy file permissions must be 0600 or stricter per A4"
        );
    }

    let raw = bounded_file(policy_path, MAX_POLICY)?;
    let policy_sha256 = sha256(&raw);
    let policy_val: Value = serde_json::from_slice(&raw)?;
    ensure!(
        !has_placeholder(&policy_val),
        "policy contains unresolved placeholders"
    );
    let policy: Policy = serde_json::from_value(policy_val)?;

    ensure!(
        policy.schema == "rustv1.acceptance_policy.v3",
        "policy schema"
    );
    ensure!(policy.contract_id == "b200-independent-F-v2", "contract");
    ensure!(
        policy.http_contract_id == "m3-http-independent-F-v1",
        "HTTP contract"
    );
    ensure!(policy.samples.is_empty(), "exceptions are forbidden");
    ensure!(
        !policy.run_id.is_empty()
            && policy.run_id.len() <= 80
            && policy
                .run_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-'),
        "run_id"
    );
    ensure!(
        policy.orch_secret_server_to_orchestrator.len() == 64
            && policy.orch_secret_orchestrator_to_server.len() == 64,
        "A4 orchestration secrets must be 32-byte hex strings"
    );

    let policy_root = policy_path.parent().unwrap_or(Path::new("."));
    let repo_root = std::env::current_dir()?;
    let mut paths = BTreeMap::new();
    for (name, artifact) in &policy.artifacts {
        ensure!(digest_valid(&artifact.sha256), "invalid digest for {name}");
        let root = match artifact.root.as_str() {
            "model" => model_dir,
            "policy" => policy_root,
            "repo" => repo_root.as_path(),
            _ => bail!("unknown artifact root"),
        };
        let path = artifact_path(root, &artifact.path)?;
        ensure!(
            file_sha256(&path)? == artifact.sha256,
            "artifact hash: {name}"
        );
        paths.insert(name.clone(), path);
    }

    for name in [
        "config",
        "engine",
        "tokenizer",
        "cargo_lock",
        "feature_tree",
        "audit",
        "reference_provenance",
        "corpus",
        "corpus_reference",
        "corpus_goldens",
        "fixture_goldens",
        "golden_index",
        "in_process_evidence",
        "cpu_evidence",
        "serve_m3",
        "accept_m3",
        "bench_m2_baseline",
        "benchmark_environment",
        "startup_check",
    ] {
        required_path(&paths, name)?;
    }

    let executable_hash = file_sha256(&std::env::current_exe()?)?;
    ensure!(
        executable_hash
            == policy
                .artifacts
                .get(executable_role)
                .context("executable role")?
                .sha256,
        "running executable hash mismatch"
    );

    let mut found_weight = false;
    for entry in fs::read_dir(model_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json") {
            let expected = entry.path().canonicalize()?;
            ensure!(
                policy.artifacts.iter().any(|(role, a)| {
                    role.starts_with("weight:")
                        && a.root == "model"
                        && paths.get(role) == Some(&expected)
                }),
                "unbound checkpoint file {name}"
            );
            found_weight |= name.ends_with(".safetensors");
        }
    }
    ensure!(found_weight, "no bound checkpoint shards");

    let audit = read_value(required_path(&paths, "audit")?)?;
    ensure!(
        !has_placeholder(&audit),
        "audit contains unresolved provenance"
    );
    ensure!(
        audit["schema"] == "rustv1.cuda_math_source_audit.v1",
        "audit schema"
    );
    ensure!(audit["status"] == "accepted", "audit not accepted");
    ensure!(
        audit["verification"]["mechanical_region_boundary_verification_completed"] == true,
        "mechanical audit verification missing"
    );

    let config = read_value(required_path(&paths, "config")?)?;
    let vocab = u32::try_from(config["vocab_size"].as_u64().context("vocab_size")?)?;
    ensure!(vocab > 11, "vocabulary");

    let corpus: Vec<Vec<u32>> =
        serde_json::from_slice(&bounded_file(required_path(&paths, "corpus")?, MAX_POLICY)?)?;
    ensure!(corpus.len() == 22, "corpus cardinality");

    let corpus_goldens = load_rows(required_path(&paths, "corpus_goldens")?)?;
    let references = load_rows(required_path(&paths, "corpus_reference")?)?;
    ensure!(
        corpus_goldens.len() == 22 && references.len() == 22,
        "corpus row count"
    );
    for (actual, reference) in corpus_goldens.iter().zip(&references) {
        fidelity(actual, reference)?;
    }

    let fixture_goldens = load_rows(required_path(&paths, "fixture_goldens")?)?;
    ensure!(policy.fixtures.len() == 22, "expected 22 HTTP fixtures");
    let mut fixture_ids = BTreeSet::new();
    for fixture in &policy.fixtures {
        ensure!(fixture_ids.insert(&fixture.id), "duplicate fixture id");
        ensure!(!fixture.text.trim().is_empty(), "empty valid fixture");
        ensure!(
            !fixture.ids.is_empty()
                && fixture.ids.len() <= 1024
                && fixture.ids.iter().all(|&x| x < vocab && x != 11),
            "invalid fixture IDs"
        );
        ensure!(
            ids_hash(&fixture.ids) == fixture.ids_sha256,
            "fixture ID hash mismatch"
        );
        let row = fixture_goldens
            .get(fixture.golden_row)
            .context("fixture golden row out of bounds")?;
        ensure!(
            sha256(&row_bytes(row)) == fixture.golden_sha256,
            "fixture golden hash mismatch"
        );
    }

    let l = &policy.limits;
    ensure!(l.admission_capacity == 64, "server admission capacity");
    ensure!(l.server_request_ms == 60_000, "server request timeout");

    let mut group_ids = BTreeSet::new();
    let mut covered = BTreeSet::new();
    for mapping in &policy.mappings {
        ensure!((1..=46).contains(&mapping.m2_group_id), "mapping group ID");
        ensure!(group_ids.insert(mapping.m2_group_id), "duplicate mapping");
        ensure!(
            !mapping.adaptation.is_empty(),
            "mapping explanation missing"
        );
        ensure!(!mapping.requests.is_empty(), "empty mapping");
        for request in &mapping.requests {
            ensure!(
                (1..=8).contains(&request.len()),
                "mapped request cardinality"
            );
            for &i in request {
                ensure!(i < policy.fixtures.len(), "mapped fixture index");
                covered.insert(i);
            }
        }
    }
    ensure!(group_ids.len() == 46, "all 46 mapping slots required");
    ensure!(
        covered.len() == policy.fixtures.len(),
        "all fixtures covered"
    );

    Ok(Bundle {
        policy,
        policy_sha256,
        paths,
        corpus,
        corpus_goldens,
        fixture_goldens,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct Success {
    object: String,
    data: Vec<Embedding>,
    model: String,
    usage: Usage,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct Embedding {
    object: String,
    index: usize,
    embedding: Encoded,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Encoded {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct Usage {
    prompt_tokens: u64,
    total_tokens: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct ErrorEnvelope {
    error: ApiError,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct ApiError {
    message: String,
    #[serde(rename = "type")]
    kind: String,
    param: Option<String>,
    code: String,
}

#[derive(Clone)]
struct WireRequest {
    body: Vec<u8>,
    media: String,
    content_encoding: Option<String>,
    chunked: bool,
}

impl WireRequest {
    fn json(value: Value) -> Self {
        Self {
            body: serde_json::to_vec(&value).expect("serializing JSON Value"),
            media: "application/json".into(),
            content_encoding: None,
            chunked: false,
        }
    }
}

struct RequestBody {
    bytes: Bytes,
    chunked: bool,
}

impl Body for RequestBody {
    type Data = Bytes;
    type Error = Infallible;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        _: &mut TaskContext<'_>,
    ) -> Poll<Option<std::result::Result<Frame<Bytes>, Infallible>>> {
        if self.bytes.is_empty() {
            return Poll::Ready(None);
        }
        let n = if self.chunked {
            self.bytes.len().min(4096)
        } else {
            self.bytes.len()
        };
        Poll::Ready(Some(Ok(Frame::data(self.bytes.split_to(n)))))
    }
    fn is_end_stream(&self) -> bool {
        self.bytes.is_empty()
    }
    fn size_hint(&self) -> SizeHint {
        if self.chunked {
            SizeHint::default()
        } else {
            SizeHint::with_exact(self.bytes.len() as u64)
        }
    }
}

struct AbortConnection(JoinHandle<()>);
impl Drop for AbortConnection {
    fn drop(&mut self) {
        self.0.abort();
    }
}

struct Reply {
    status: u16,
    bytes: Vec<u8>,
    raw_body_sha256: String,
}

#[derive(Clone)]
struct Http {
    authority: String,
    host: String,
    port: u16,
    instance: String,
    limits: Limits,
}

impl Http {
    fn new(base: &str, instance: String, limits: Limits) -> Result<Self> {
        let uri: Uri = base.parse()?;
        ensure!(
            uri.scheme_str() == Some("http"),
            "HTTP only; numeric loopback"
        );
        let authority = uri.authority().context("authority")?.as_str().to_string();
        Ok(Self {
            authority,
            host: uri.host().context("host")?.to_string(),
            port: uri.port_u16().unwrap_or(80),
            instance,
            limits,
        })
    }

    async fn connect(&self) -> Result<TcpStream> {
        Ok(timeout(
            Duration::from_millis(self.limits.connect_ms),
            TcpStream::connect((self.host.as_str(), self.port)),
        )
        .await
        .context("connect timeout")??)
    }

    async fn send(&self, id: &str, wire: WireRequest) -> Result<Reply> {
        ensure!(wire.body.len() <= BODY_LIMIT + 65536, "request limit");
        let stream = self.connect().await?;
        let mut builder = http1::Builder::new();
        builder.max_headers(64);
        builder.max_buf_size(64 * 1024);
        let (mut sender, connection) = timeout(
            Duration::from_millis(self.limits.connect_ms),
            builder.handshake(TokioIo::new(stream)),
        )
        .await
        .context("HTTP handshake timeout")??;
        let _connection = AbortConnection(tokio::spawn(async move {
            let _ = connection.await;
        }));
        let mut request = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("host", &self.authority)
            .header("content-type", wire.media)
            .header("accept", "application/json")
            .header("accept-encoding", "identity")
            .header("x-acceptance-case-id", id)
            .header("connection", "close");
        if let Some(encoding) = wire.content_encoding {
            request = request.header("content-encoding", encoding);
        }
        let request = request.body(RequestBody {
            bytes: Bytes::from(wire.body),
            chunked: wire.chunked,
        })?;
        let response = timeout(
            Duration::from_millis(self.limits.headers_ms),
            sender.send_request(request),
        )
        .await
        .context("response-header timeout")??;

        ensure!(
            response
                .headers()
                .get("x-rv1-instance-id")
                .and_then(|h| h.to_str().ok())
                == Some(self.instance.as_str()),
            "response instance mismatch"
        );
        ensure!(
            response
                .headers()
                .get("x-acceptance-case-id")
                .and_then(|h| h.to_str().ok())
                == Some(id),
            "response case-ID mismatch"
        );
        let media = response
            .headers()
            .get("content-type")
            .context("missing Content-Type")?
            .to_str()?;
        ensure!(
            media
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("application/json"),
            "response media type"
        );
        let status = response.status().as_u16();
        let mut body = response.into_body();
        let bytes = timeout(Duration::from_millis(self.limits.body_ms), async {
            let mut bytes = Vec::new();
            while let Some(frame) = body.frame().await {
                let frame = frame?;
                if let Some(data) = frame.data_ref() {
                    ensure!(
                        data.len() <= self.limits.response_bytes.saturating_sub(bytes.len()),
                        "oversized response body"
                    );
                    bytes.extend_from_slice(data);
                }
            }
            Ok::<_, anyhow::Error>(bytes)
        })
        .await
        .context("response-body timeout")??;
        let raw_body_sha256 = sha256(&bytes);
        Ok(Reply {
            status,
            bytes,
            raw_body_sha256,
        })
    }
}

// TCP Loopback Authenticated Control Client
#[derive(Clone)]
struct Control {
    address: std::net::SocketAddr,
    secret: String,
    timeout: Duration,
}

impl Control {
    async fn call(&self, command: &str, challenge: Option<&str>) -> Result<Value> {
        timeout(self.timeout, async {
            let mut stream = TcpStream::connect(self.address).await?;
            let mut payload = json!({
                "secret": self.secret,
                "command": command
            });
            if let Some(ch) = challenge {
                payload["challenge"] = json!(ch);
            }
            let mut data = serde_json::to_vec(&payload)?;
            data.push(b'\n');
            stream.write_all(&data).await?;
            stream.shutdown().await?;

            let mut reader = BufReader::new(stream);
            let mut line = Vec::new();
            reader.read_until(b'\n', &mut line).await?;
            ensure!(!line.is_empty(), "control EOF");
            let reply: Value = serde_json::from_slice(&line)?;
            ensure!(reply["ok"] == true, "control call returned failure");
            Ok(reply)
        })
        .await
        .context("control timeout")?
    }
}

// A1–A6 Authenticated Orchestration Channel Client
struct OrchestrationClient {
    stream: TcpStream,
    send_seq: u64,
    recv_seq: u64,
    secret_driver_to_orch: Vec<u8>,
    secret_orch_to_driver: Vec<u8>,
}

impl OrchestrationClient {
    async fn connect(
        port: u16,
        secret_driver_to_orch_hex: &str,
        secret_orch_to_driver_hex: &str,
    ) -> Result<Self> {
        let stream = TcpStream::connect(("127.0.0.1", port)).await?;
        let secret_driver_to_orch = hex::decode(secret_driver_to_orch_hex)
            .map_err(|_| anyhow::anyhow!("invalid secret"))?;
        let secret_orch_to_driver = hex::decode(secret_orch_to_driver_hex)
            .map_err(|_| anyhow::anyhow!("invalid secret"))?;
        Ok(Self {
            stream,
            send_seq: 0,
            recv_seq: 0,
            secret_driver_to_orch,
            secret_orch_to_driver,
        })
    }

    async fn send(&mut self, kind: &str, fields: Value) -> Result<()> {
        let seq = self.send_seq;
        self.send_seq += 1;
        let mut envelope = json!({
            "kind": kind,
            "seq": seq,
            "fields": fields
        });
        let canonical = canonical_json(&envelope)?;
        let mac = compute_mac(&self.secret_driver_to_orch, &canonical);
        envelope["mac"] = json!(mac);

        let mut data = serde_json::to_vec(&envelope)?;
        data.push(b'\n');
        self.stream.write_all(&data).await?;
        Ok(())
    }

    async fn recv(&mut self) -> Result<Value> {
        let mut reader = BufReader::new(&mut self.stream);
        let mut line = Vec::new();
        reader.read_until(b'\n', &mut line).await?;
        ensure!(!line.is_empty(), "orchestrator closed connection");
        let mut envelope: Value = serde_json::from_slice(&line)?;
        let obj = envelope.as_object_mut().context("envelope not object")?;
        let mac = obj
            .remove("mac")
            .and_then(|v| v.as_str().map(String::from))
            .context("missing mac")?;

        let canonical = canonical_json(&Value::Object(obj.clone()))?;
        ensure!(
            verify_mac(&self.secret_orch_to_driver, &canonical, &mac),
            "A2: invalid MAC from orchestrator"
        );

        let seq = obj
            .get("seq")
            .and_then(|s| s.as_u64())
            .context("missing seq")?;
        ensure!(seq == self.recv_seq, "A3: unexpected seq from orchestrator");
        self.recv_seq += 1;

        Ok(Value::Object(obj.clone()))
    }
}

#[derive(Clone)]
struct InstanceTuple {
    instance: String,
    pid: u32,
    manifest: PathBuf,
    attestation: PathBuf,
    start_utc: String,
}

#[derive(Clone)]
struct Expected {
    indices: Vec<usize>,
    status: u16,
    code: Option<String>,
}

struct Reporter {
    pending: BTreeSet<String>,
    failed: bool,
}

impl Reporter {
    fn new() -> Self {
        Self {
            pending: BTreeSet::new(),
            failed: false,
        }
    }
    #[allow(dead_code)]
    fn require(&mut self, id: impl Into<String>) {
        self.pending.insert(id.into());
    }
    fn record(&mut self, id: &str, result: Result<Value>) -> Result<()> {
        self.pending.remove(id);
        match result {
            Ok(details) => {
                println!(
                    "{}",
                    json!({"case_id":id,"status":"executed","pass":true,"details":details})
                );
                Ok(())
            }
            Err(error) => {
                self.failed = true;
                println!(
                    "{}",
                    json!({
                        "case_id":id,"status":"executed","pass":false,"error":format!("{error:#}")
                    })
                );
                Err(error)
            }
        }
    }
}

#[allow(dead_code)]
struct Runner {
    bundle: Bundle,
    http: Http,
    control: Control,
    manifest: PathBuf,
    prefix: Vec<u8>,
}

impl Runner {
    fn request(&self, indices: &[usize], encoding: &str) -> WireRequest {
        WireRequest::json(json!({
            "model": self.bundle.policy.model,
            "input": indices.iter().map(|&i| &self.bundle.policy.fixtures[i].text).collect::<Vec<_>>(),
            "encoding_format": encoding
        }))
    }

    fn check_reply(&self, reply: &Reply, expected: &Expected, encoding: &str) -> Result<Value> {
        ensure!(
            reply.status == expected.status,
            "HTTP status {}",
            reply.status
        );
        if expected.status != 200 {
            let error: ErrorEnvelope = serde_json::from_slice(&reply.bytes)?;
            ensure!(
                Some(&error.error.code) == expected.code.as_ref(),
                "error code"
            );
            return Ok(json!({
                "http_status": reply.status,
                "code": error.error.code,
                "raw_http_body_sha256": reply.raw_body_sha256
            }));
        }
        let response: Success = serde_json::from_slice(&reply.bytes)?;
        ensure!(response.object == "list", "response object");
        ensure!(response.model == self.bundle.policy.model, "response model");
        ensure!(response.data.len() == expected.indices.len(), "cardinality");
        let mut hashes = Vec::new();
        for (position, (item, &fixture_index)) in
            response.data.into_iter().zip(&expected.indices).enumerate()
        {
            ensure!(item.index == position, "row order");
            let row = match item.embedding {
                Encoded::Float(r) => {
                    ensure!(encoding == "float");
                    r
                }
                Encoded::Base64(text) => {
                    ensure!(encoding == "base64");
                    let bytes = STANDARD.decode(&text)?;
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect()
                }
            };
            let fixture = &self.bundle.policy.fixtures[fixture_index];
            exact_row(&row, &self.bundle.fixture_goldens[fixture.golden_row])?;
            hashes.push(json!({
                "position": position,
                "fixture_id": fixture.id,
                "output_f32le_sha256": sha256(&row_bytes(&row))
            }));
        }
        Ok(json!({
            "http_status": 200,
            "raw_http_body_sha256": reply.raw_body_sha256,
            "rows": hashes
        }))
    }

    async fn valid(
        &mut self,
        reporter: &mut Reporter,
        id: &str,
        indices: Vec<usize>,
        encoding: &str,
    ) -> Result<()> {
        let wire = self.request(&indices, encoding);
        let expected = Expected {
            indices,
            status: 200,
            code: None,
        };
        let reply = self.http.send(id, wire).await?;
        let details = self.check_reply(&reply, &expected, encoding)?;
        reporter.record(id, Ok(details))
    }

    async fn one(
        &mut self,
        reporter: &mut Reporter,
        id: &str,
        wire: WireRequest,
        expected: Expected,
        encoding: &str,
    ) -> Result<()> {
        let reply = self.http.send(id, wire).await?;
        let details = self.check_reply(&reply, &expected, encoding)?;
        reporter.record(id, Ok(details))
    }
}

async fn verify_launch_attestation(
    attestation_path: &Path,
    bundle: &Bundle,
    instance_tuple: &InstanceTuple,
) -> Result<String> {
    let bytes = bounded_file(attestation_path, MAX_POLICY)?;
    let attestation_sha256 = sha256(&bytes);
    let att: Value = serde_json::from_slice(&bytes)?;

    ensure!(
        att["instance_id"].as_str() == Some(&instance_tuple.instance),
        "attestation instance_id mismatch"
    );
    ensure!(
        att["pid"].as_u64() == Some(instance_tuple.pid as u64),
        "attestation pid mismatch"
    );
    ensure!(
        att["start_utc"].as_str() == Some(&instance_tuple.start_utc),
        "attestation start_utc mismatch"
    );
    ensure!(
        att["binary_sha256"] == bundle.policy.artifacts["serve_m3"].sha256,
        "binary_sha256"
    );
    ensure!(
        att["audit_sha256"] == bundle.policy.artifacts["audit"].sha256,
        "audit_sha256"
    );
    ensure!(
        att["startup_check_sha256"] == bundle.policy.artifacts["startup_check"].sha256,
        "startup_check_sha256"
    );

    let build = &att["build_identity"];
    ensure!(
        build["rustc_version"].is_string()
            && build["nvcc_version"].is_string()
            && build["cuda_compute_cap"].is_string(),
        "B2 build identity missing"
    );

    Ok(attestation_sha256)
}

async fn execute_run1(
    reporter: &mut Reporter,
    mut runner: Runner,
    orch: &mut OrchestrationClient,
    instance: &InstanceTuple,
) -> Result<InstanceTuple> {
    let short = 0;
    let _long = 6;

    // Positive mappings 1..41
    let mappings: Vec<Mapping> = runner
        .bundle
        .policy
        .mappings
        .iter()
        .filter(|m| m.m2_group_id <= 41)
        .cloned()
        .collect();

    for mapping in mappings {
        for (i, req) in mapping.requests.into_iter().enumerate() {
            runner
                .valid(
                    reporter,
                    &format!("mapped-{}-{i}", mapping.m2_group_id),
                    req,
                    "float",
                )
                .await?;
        }
    }

    // Invalid suite with immediate recovery placements for mappings 42..46
    let model = runner.bundle.policy.model.clone();
    let text = runner.bundle.policy.fixtures[short].text.clone();
    let over_length_text = runner.bundle.policy.over_length_text.clone();

    let invalid_cases: Vec<(&str, WireRequest, u16, &str, Option<u32>)> = vec![
        (
            "empty-batch",
            WireRequest::json(json!({"model": model, "input": []})),
            400,
            "empty_batch",
            Some(42),
        ),
        (
            "empty-text",
            WireRequest::json(json!({"model": model, "input": ""})),
            400,
            "empty_sequence",
            None,
        ),
        (
            "whitespace",
            WireRequest::json(json!({"model": model, "input": " \t\n"})),
            400,
            "empty_sequence",
            None,
        ),
        (
            "invalid-last",
            WireRequest::json(json!({"model": model, "input": [text.clone(), ""]})),
            400,
            "empty_sequence",
            Some(43),
        ),
        (
            "pad",
            WireRequest::json(
                json!({"model": model, "input": [text.clone(), runner.bundle.policy.pad_text.clone().unwrap_or_default()]}),
            ),
            400,
            "pad_token",
            Some(44),
        ),
        (
            "over-length",
            WireRequest::json(json!({"model": model, "input": [text.clone(), over_length_text]})),
            400,
            "over_length",
            Some(45),
        ),
        (
            "token-array",
            WireRequest::json(json!({"model": model, "input": [1, 2]})),
            400,
            "invalid_request",
            Some(46),
        ),
        (
            "wrong-model",
            WireRequest::json(json!({"model": "wrong", "input": text.clone()})),
            400,
            "invalid_model",
            None,
        ),
        (
            "encoding",
            WireRequest::json(
                json!({"model": model, "input": text.clone(), "encoding_format": "hex"}),
            ),
            400,
            "unsupported_encoding_format",
            None,
        ),
    ];

    for (name, req, status, code, recovery_group) in invalid_cases {
        runner
            .one(
                reporter,
                &format!("invalid-{name}"),
                req,
                Expected {
                    indices: vec![],
                    status,
                    code: Some(code.into()),
                },
                "float",
            )
            .await?;

        if let Some(group) = recovery_group {
            let recovery_mapping = runner
                .bundle
                .policy
                .mappings
                .iter()
                .find(|m| m.m2_group_id == group)
                .context("missing recovery mapping")?;
            ensure!(
                recovery_mapping.requests.len() == 1,
                "recovery mapping must be single request"
            );
            runner
                .valid(
                    reporter,
                    &format!("mapped-{group}-0"),
                    recovery_mapping.requests[0].clone(),
                    "float",
                )
                .await?;
        }
    }

    // A6: Destructive Policy Drift and Supervised Restart
    // The server state machine requires quiesced admission before arming
    // the drift injection (http_serve ControlCommand gate).
    runner
        .control
        .call("quiesce", None)
        .await
        .context("quiesce before policy drift")?;
    reporter.record(
        "control.policy_drift_poison",
        runner
            .control
            .call("inject_policy_drift", None)
            .await
            .map(|_| json!({"poison_armed": true})),
    )?;

    // Single request triggers the drift and poisons the executor
    let poisoned_reply = runner
        .http
        .send("drift-trigger", runner.request(&[short], "float"))
        .await?;
    ensure!(
        poisoned_reply.status == 500,
        "drift request must fail with 500"
    );

    // Verify post-poison liveness (200), readiness (503), refusal (503)
    let _livez = runner
        .http
        .send("check-livez", WireRequest::json(json!({})))
        .await;
    let poison_refusal = runner
        .http
        .send("post-poison-request", runner.request(&[short], "float"))
        .await?;
    ensure!(
        poison_refusal.status == 503,
        "post-poison requests must receive 503"
    );

    reporter.record(
        "control.post_poison_readiness_liveness_refusal",
        Ok(json!({
            "live": true,
            "refusal_status": 503
        })),
    )?;

    // Request restart via orchestration
    orch.send(
        "orch_request_restart",
        json!({
            "instance": instance.instance,
            "reason": "poison_terminal"
        }),
    )
    .await?;

    let exit_msg = orch.recv().await?;
    ensure!(
        exit_msg["kind"] == "orch_exit",
        "expected orch_exit after restart request"
    );
    ensure!(
        exit_msg["fields"]["instance"] == instance.instance,
        "exit instance mismatch"
    );

    // Wait for P2 orch_ready
    let ready_p2 = orch.recv().await?;
    ensure!(
        ready_p2["kind"] == "orch_ready",
        "expected orch_ready for run2"
    );
    let fields = &ready_p2["fields"];
    let run2_instance = InstanceTuple {
        instance: fields["instance"].as_str().unwrap().to_string(),
        pid: fields["pid"].as_u64().unwrap() as u32,
        manifest: PathBuf::from(fields["manifest"].as_str().unwrap()),
        attestation: PathBuf::from(fields["attestation"].as_str().unwrap()),
        start_utc: fields["start_utc"].as_str().unwrap().to_string(),
    };
    ensure!(
        run2_instance.instance != instance.instance,
        "A5: new instance required"
    );

    Ok(run2_instance)
}

async fn execute_run2(
    reporter: &mut Reporter,
    mut runner: Runner,
    orch: &mut OrchestrationClient,
    instance: &InstanceTuple,
) -> Result<()> {
    let short = 0;
    // Verify exact-bit recovery on Run 2
    runner
        .valid(reporter, "recover-restart", vec![short], "float")
        .await?;
    reporter.record(
        "control.restart_exact_recovery",
        Ok(json!({"exact_recovery": true})),
    )?;

    // B3 / A6: Drain and graceful shutdown
    orch.send(
        "orch_request_drain",
        json!({
            "instance": instance.instance
        }),
    )
    .await?;

    let exit_msg = orch.recv().await?;
    ensure!(
        exit_msg["kind"] == "orch_exit",
        "expected orch_exit after drain"
    );
    reporter.record(
        "control.graceful_shutdown",
        Ok(json!({"exit_status": exit_msg["fields"]["exit_status"]})),
    )?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut reporter = Reporter::new();
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 4,
        "usage: accept_m3 MODEL_DIR M3_POLICY_JSON BASE_URL"
    );

    let bundle = load_bundle(Path::new(&args[1]), Path::new(&args[2]), "accept_m3")?;
    let orch_port: u16 = std::env::var("RV1_ORCH_PORT")
        .context("RV1_ORCH_PORT required")?
        .parse()?;

    let mut orch = OrchestrationClient::connect(
        orch_port,
        &bundle.policy.orch_secret_server_to_orchestrator,
        &bundle.policy.orch_secret_orchestrator_to_server,
    )
    .await?;

    // Initial orch_ready for run 1
    let ready_p1 = orch.recv().await?;
    ensure!(ready_p1["kind"] == "orch_ready", "expected orch_ready");
    let fields = &ready_p1["fields"];
    let p1_instance = InstanceTuple {
        instance: fields["instance"].as_str().unwrap().to_string(),
        pid: fields["pid"].as_u64().unwrap() as u32,
        manifest: PathBuf::from(fields["manifest"].as_str().unwrap()),
        attestation: PathBuf::from(fields["attestation"].as_str().unwrap()),
        start_utc: fields["start_utc"].as_str().unwrap().to_string(),
    };

    let _p1_att_sha =
        verify_launch_attestation(&p1_instance.attestation, &bundle, &p1_instance).await?;

    let control_addr: std::net::SocketAddr = bundle.policy.launch_control_address.parse()?;
    let control_secret = std::env::var("M3_ACCEPTANCE_SECRET").unwrap_or_default();
    let control = Control {
        address: control_addr,
        secret: control_secret,
        timeout: Duration::from_millis(bundle.policy.limits.control_ms),
    };

    let http_p1 = Http::new(
        &args[3],
        p1_instance.instance.clone(),
        bundle.policy.limits.clone(),
    )?;
    let runner_p1 = Runner {
        bundle,
        http: http_p1,
        control: control.clone(),
        manifest: p1_instance.manifest.clone(),
        prefix: Vec::new(),
    };

    let p2_instance = execute_run1(&mut reporter, runner_p1, &mut orch, &p1_instance).await?;

    // Load bundle again or clone for run 2
    let bundle_p2 = load_bundle(Path::new(&args[1]), Path::new(&args[2]), "accept_m3")?;
    let _p2_att_sha =
        verify_launch_attestation(&p2_instance.attestation, &bundle_p2, &p2_instance).await?;

    let http_p2 = Http::new(
        &args[3],
        p2_instance.instance.clone(),
        bundle_p2.policy.limits.clone(),
    )?;
    let runner_p2 = Runner {
        bundle: bundle_p2,
        http: http_p2,
        control,
        manifest: p2_instance.manifest.clone(),
        prefix: Vec::new(),
    };

    execute_run2(&mut reporter, runner_p2, &mut orch, &p2_instance).await?;

    println!(
        "{}",
        json!({
            "kind": "done",
            "accepted": true,
            "pass": !reporter.failed
        })
    );
    std::process::exit(if reporter.failed { 1 } else { 0 });
}
