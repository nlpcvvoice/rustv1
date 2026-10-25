// src/bin/accept_m3.rs
//
// CPU-only. No required-features entry.
//
// The local acceptance-control socket and manifest format are specified in
// consultation #20B, section 3. This is not a public fault-injection endpoint.
//
// Full acceptance intentionally remains false until the required destructive
// cases have an implemented orchestration driver.

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
    sync::Arc,
    task::{Context as TaskContext, Poll},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{TcpStream, UnixStream},
    sync::Barrier,
    task::JoinHandle,
    time::{sleep, timeout, Instant},
};

pub const WIDTH: usize = 2048;
const ROW_BYTES: usize = WIDTH * 4;
const MAX_POLICY: usize = 8 * 1024 * 1024;
const MAX_MANIFEST: usize = 128 * 1024 * 1024;
const MAX_CONTROL: usize = 64 * 1024;
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
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
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
    pub control_socket: String,
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
        Value::String(s) => s.contains("FILLED_BY_INTEGRATOR"),
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

/// Shared CPU-only preflight. `executable_role` is "accept_m3" or
/// "bench_m2_baseline"; executable hashes are never inferred from filenames.
pub fn load_bundle(model_dir: &Path, policy_path: &Path, executable_role: &str) -> Result<Bundle> {
    let raw = bounded_file(policy_path, MAX_POLICY)?;
    let policy_sha256 = sha256(&raw);
    let policy: Policy = serde_json::from_slice(&raw)?;
    ensure!(
        policy.schema == "rustv1.acceptance_policy.v2",
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

    // Require every top-level checkpoint shard, including an index if present.
    // The deployment contract does not permit unlisted nested shard locations.
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
    for (role, file) in [
        ("config", "config.json"),
        ("engine", "engine.json"),
        ("tokenizer", "tokenizer.json"),
    ] {
        ensure!(
            required_path(&paths, role)? == model_dir.join(file).canonicalize()?,
            "{role} does not bind MODEL_DIR/{file}"
        );
    }

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
        audit["crates"]["candle-core"]["version"] == "0.9.2",
        "Candle version"
    );
    ensure!(
        audit["crates"]["cudarc"]["version"] == "0.19.10",
        "cudarc version"
    );
    ensure!(
        audit["cargo_lock_sha256"] == policy.artifacts["cargo_lock"].sha256,
        "audit lock binding"
    );
    ensure!(
        audit["cargo_tree_features"]["sha256"] == policy.artifacts["feature_tree"].sha256,
        "audit feature-tree binding"
    );
    ensure!(
        audit["verification"]["mechanical_region_boundary_verification_completed"] == true,
        "mechanical audit verification missing"
    );
    let audit_artifact = &policy.artifacts["audit"];
    ensure!(
        audit_artifact.root == "repo",
        "audit must be repository-relative"
    );
    ensure!(
        policy.audit_identity
            == format!("sha256:{}:{}", audit_artifact.sha256, audit_artifact.path),
        "audit identity"
    );

    let config = read_value(required_path(&paths, "config")?)?;
    let vocab = u32::try_from(config["vocab_size"].as_u64().context("vocab_size")?)?;
    ensure!(vocab > 11, "vocabulary");
    let engine = read_value(required_path(&paths, "engine")?)?;
    ensure!(
        engine["max_len"].as_u64() == Some(1024),
        "effective length must be 1024"
    );
    let tokenizer = read_value(required_path(&paths, "tokenizer")?)?;
    ensure!(tokenizer["padding"].is_null(), "tokenizer padding enabled");
    ensure!(
        tokenizer["truncation"].is_null(),
        "tokenizer truncation enabled"
    );

    let corpus: Vec<Vec<u32>> =
        serde_json::from_slice(&bounded_file(required_path(&paths, "corpus")?, MAX_POLICY)?)?;
    ensure!(corpus.len() == 22, "corpus cardinality");
    for ids in &corpus {
        ensure!(!ids.is_empty() && ids.len() <= 1024, "corpus length");
        ensure!(ids.iter().all(|&x| x < vocab && x != 11), "corpus token");
    }
    ensure!(
        corpus.iter().any(|p| p.len() <= 16),
        "missing short corpus sample"
    );
    ensure!(
        corpus.iter().any(|p| p.len() == 1024),
        "missing length-1024 sample"
    );

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
    ensure!(policy.fixtures.len() >= 2, "insufficient HTTP fixtures");
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
            "fixture ID hash"
        );
        let row = fixture_goldens
            .get(fixture.golden_row)
            .context("fixture golden row")?;
        ensure!(
            sha256(&row_bytes(row)) == fixture.golden_sha256,
            "fixture golden hash"
        );
    }
    ensure!(
        policy.fixtures.iter().any(|f| f.ids.len() <= 16),
        "short HTTP fixture"
    );
    ensure!(
        policy.fixtures.iter().any(|f| f.ids.len() == 1024),
        "1024-token HTTP fixture"
    );

    let index = read_value(required_path(&paths, "golden_index")?)?;
    ensure!(
        index["schema"] == "rustv1.singleton_goldens.v1",
        "golden index schema"
    );
    ensure!(
        index["contract_id"] == policy.contract_id,
        "golden contract"
    );
    ensure!(
        index["corpus_sha256"] == policy.artifacts["corpus"].sha256,
        "index corpus"
    );
    ensure!(
        index["corpus_goldens_sha256"] == policy.artifacts["corpus_goldens"].sha256,
        "index corpus goldens"
    );
    ensure!(
        index["fixture_goldens_sha256"] == policy.artifacts["fixture_goldens"].sha256,
        "index fixture goldens"
    );
    ensure!(
        index["fixtures"] == serde_json::to_value(&policy.fixtures)?,
        "index fixtures"
    );
    let entries = index["corpus"].as_array().context("index corpus entries")?;
    ensure!(entries.len() == 22, "index corpus cardinality");
    for (i, entry) in entries.iter().enumerate() {
        ensure!(entry["index"].as_u64() == Some(i as u64), "index order");
        ensure!(entry["ids_sha256"] == ids_hash(&corpus[i]), "index ID hash");
        ensure!(
            entry["offset_bytes"].as_u64() == Some((i * ROW_BYTES) as u64),
            "offset"
        );
        ensure!(
            entry["bytes"].as_u64() == Some(ROW_BYTES as u64),
            "row byte count"
        );
        ensure!(
            entry["sha256"] == sha256(&row_bytes(&corpus_goldens[i])),
            "row hash"
        );
    }

    // These are hash-bound prerequisite attestations, not a re-execution of
    // the M2 suite, tokenizer, signature verifier, or source auditor.
    let process = read_value(required_path(&paths, "in_process_evidence")?)?;
    ensure!(
        process["schema"] == "rustv1.in_process_acceptance.v2",
        "M2 evidence schema"
    );
    ensure!(process["accepted"] == true, "M2 evidence not accepted");
    ensure!(process["contract_id"] == policy.contract_id, "M2 contract");
    ensure!(
        process["audit_identity"] == policy.audit_identity,
        "M2 audit"
    );
    ensure!(process["deployment"] == policy.deployment, "M2 deployment");
    for key in [
        "corpus",
        "corpus_reference",
        "corpus_goldens",
        "fixture_goldens",
        "golden_index",
        "reference_provenance",
    ] {
        ensure!(
            process["artifacts"][key] == policy.artifacts[key].sha256,
            "M2 evidence artifact {key}"
        );
    }
    for key in [
        "all_22_fidelity",
        "all_invariance_groups",
        "five_invalid_classes",
        "restart_exact_bits",
        "fixture_token_ids_verified",
        "audit_signature_verified",
    ] {
        ensure!(process["checks"][key] == true, "M2 prerequisite {key}");
    }

    let reference = read_value(required_path(&paths, "reference_provenance")?)?;
    ensure!(
        reference["schema"] == "rustv1.independent_reference.v1",
        "reference schema"
    );
    ensure!(
        reference["frozen_before_candidate"] == true,
        "reference not pre-frozen"
    );
    ensure!(
        reference["independent_from_candle"] == true,
        "reference independence"
    );
    ensure!(
        reference["corpus_sha256"] == policy.artifacts["corpus"].sha256,
        "reference corpus"
    );
    ensure!(
        reference["output_sha256"] == policy.artifacts["corpus_reference"].sha256,
        "reference output binding"
    );

    let cpu = read_value(required_path(&paths, "cpu_evidence")?)?;
    ensure!(
        cpu["schema"] == "rustv1.m3_cpu_evidence.v1",
        "CPU evidence schema"
    );
    ensure!(cpu["accepted"] == true, "CPU evidence not accepted");
    ensure!(
        cpu["server_sha256"] == policy.artifacts["serve_m3"].sha256,
        "CPU build binding"
    );
    for key in [
        "production_router",
        "float_signed_zero",
        "base64_roundtrip",
        "post_tokenization_pad",
        "post_tokenization_oov",
        "whole_request_validation",
        "chunked_body_limit",
        "header_body_write_idle_socket_deadlines",
        "worker_exit",
        "deadline_races",
        "disconnect_ownership",
        "sanitized_logs",
    ] {
        ensure!(cpu["checks"][key] == true, "CPU prerequisite {key}");
    }

    let l = &policy.limits;
    for ms in [
        l.connect_ms,
        l.headers_ms,
        l.body_ms,
        l.control_ms,
        l.observation_ms,
        l.server_request_ms,
    ] {
        ensure!((1..=300_000).contains(&ms), "invalid time bound");
    }
    ensure!(
        l.headers_ms > l.server_request_ms,
        "client must observe server deadline"
    );
    ensure!(
        l.observation_ms * 3 < l.server_request_ms,
        "observation budget"
    );
    ensure!(
        (64 * 1024..=16 * 1024 * 1024).contains(&l.response_bytes),
        "response cap"
    );
    ensure!(
        (8..=64).contains(&l.admission_capacity),
        "admission capacity"
    );
    ensure!(
        l.queue_capacity >= l.admission_capacity,
        "queue cannot hold admission set"
    );
    ensure!(l.queue_capacity <= 64, "queue capacity maximum");

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
        "uncovered HTTP fixture"
    );

    ensure!(
        !policy.over_length_text.is_empty(),
        "overlength fixture missing"
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
struct Success {
    object: String,
    data: Vec<Embedding>,
    model: String,
    usage: Usage,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
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
struct Usage {
    prompt_tokens: u64,
    total_tokens: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ErrorEnvelope {
    error: ApiError,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
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
            "HTTP only; test the bound deployed path"
        );
        ensure!(
            uri.path() == "/" || uri.path().is_empty(),
            "base URL must be an origin"
        );
        ensure!(uri.query().is_none(), "base URL query");
        let authority = uri.authority().context("authority")?.as_str().to_string();
        ensure!(!authority.contains('@'), "URL userinfo");
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
        ensure!(wire.body.len() <= BODY_LIMIT + 65536, "driver request cap");
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
            "response is not bound to the attested instance"
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
            .context("missing response Content-Type")?
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
        ensure!(
            response
                .headers()
                .get("content-encoding")
                .map(|h| h.as_bytes() == b"identity")
                .unwrap_or(true),
            "compressed response not supported"
        );
        if let Some(length) = response.headers().get("content-length") {
            ensure!(
                length.to_str()?.parse::<usize>()? <= self.limits.response_bytes,
                "oversized Content-Length"
            );
        }
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
                } else {
                    bail!("unexpected response trailers");
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

    async fn raw_request(&self, id: &str, wire: &WireRequest) -> Result<TcpStream> {
        let mut stream = self.connect().await?;
        let header = format!(
            "POST /v1/embeddings HTTP/1.1\r\nHost: {}\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\
             X-Acceptance-Case-Id: {}\r\nConnection: close\r\n\r\n",
            self.authority,
            wire.body.len(),
            id
        );
        timeout(Duration::from_millis(self.limits.connect_ms), async {
            stream.write_all(header.as_bytes()).await?;
            stream.write_all(&wire.body).await?;
            stream.flush().await?;
            Ok::<_, std::io::Error>(())
        })
        .await
        .context("request-write timeout")??;
        Ok(stream)
    }
}

#[derive(Clone)]
struct Control {
    path: String,
    run_id: String,
    instance: String,
    ms: u64,
}
impl Control {
    async fn call(&self, op: &str, args: Value) -> Result<Value> {
        timeout(Duration::from_millis(self.ms), async {
            let stream = UnixStream::connect(&self.path).await?;
            let (read, mut write) = stream.into_split();
            let command = json!({
                "protocol":"rustv1.acceptance-control.v1",
                "run_id":self.run_id,
                "instance_id":self.instance,
                "op":op,
                "args":args
            });
            let mut bytes = serde_json::to_vec(&command)?;
            bytes.push(b'\n');
            write.write_all(&bytes).await?;
            write.shutdown().await?;
            let mut reader = BufReader::new(read);
            let mut line = Vec::new();
            loop {
                let available = reader.fill_buf().await?;
                ensure!(!available.is_empty(), "control EOF before response");
                let n = available
                    .iter()
                    .position(|&b| b == b'\n')
                    .map(|n| n + 1)
                    .unwrap_or(available.len());
                ensure!(line.len() + n <= MAX_CONTROL, "oversized control response");
                let finished = available[n - 1] == b'\n';
                line.extend_from_slice(&available[..n]);
                reader.consume(n);
                if finished {
                    break;
                }
            }
            let reply: Value = serde_json::from_slice(&line)?;
            ensure!(
                reply["protocol"] == "rustv1.acceptance-control.v1",
                "control protocol"
            );
            ensure!(reply["run_id"] == self.run_id, "control run");
            ensure!(reply["ok"] == true, "control command rejected");
            if !self.instance.is_empty() {
                ensure!(reply["instance_id"] == self.instance, "control instance");
            }
            Ok(reply)
        })
        .await
        .context("control timeout")?
    }

    async fn observe(
        &self,
        milliseconds: u64,
        predicate: impl Fn(&Value) -> bool,
    ) -> Result<Value> {
        let end = Instant::now() + Duration::from_millis(milliseconds);
        loop {
            let value = self.call("snapshot", json!({})).await?;
            if predicate(&value) {
                return Ok(value);
            }
            ensure!(
                Instant::now() < end,
                "required lifecycle state was not observed"
            );
            sleep(Duration::from_millis(2)).await;
        }
    }
}

#[derive(Clone)]
struct Expected {
    indices: Vec<usize>,
    status: u16,
    code: Option<String>,
    forwards: bool,
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
    fn unexecuted(&mut self, id: &str, reason: &str) {
        self.pending.remove(id);
        println!(
            "{}",
            json!({
                "case_id":id,"status":"unexecuted","required":true,"pass":false,"reason":reason
            })
        );
    }
}

struct Runner {
    bundle: Bundle,
    http: Http,
    control: Control,
    manifest: PathBuf,
    prefix: Vec<u8>,
    seq: u64,
    expected: BTreeMap<String, Expected>,
}

impl Runner {
    fn request(&self, indices: &[usize], encoding: &str) -> WireRequest {
        WireRequest::json(json!({
            "model":self.bundle.policy.model,
            "input":indices.iter().map(|&i| &self.bundle.policy.fixtures[i].text).collect::<Vec<_>>(),
            "encoding_format":encoding
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
            let kind = if reply.status == 503 {
                "service_unavailable_error"
            } else if reply.status >= 500 {
                "api_error"
            } else {
                "invalid_request_error"
            };
            ensure!(error.error.kind == kind, "error type");
            ensure!(
                !error.error.message.is_empty() && error.error.message.len() <= 256,
                "public message shape"
            );
            if let Some(param) = error.error.param {
                ensure!(
                    ["input", "model", "encoding_format", "user"].contains(&param.as_str()),
                    "error param"
                );
            }
            return Ok(json!({
                "http_status":reply.status,
                "code":error.error.code,
                "raw_http_body_sha256":reply.raw_body_sha256,
                "output_f32le_sha256":Value::Null
            }));
        }
        // Deserialize number arrays directly into Vec<f32>, not via Value/f64.
        let response: Success = serde_json::from_slice(&reply.bytes)?;
        ensure!(response.object == "list", "response object");
        ensure!(response.model == self.bundle.policy.model, "response model");
        ensure!(
            response.data.len() == expected.indices.len(),
            "response cardinality"
        );
        let tokens: usize = expected
            .indices
            .iter()
            .map(|&i| self.bundle.policy.fixtures[i].ids.len())
            .sum();
        ensure!(
            response.usage.prompt_tokens == tokens as u64
                && response.usage.total_tokens == tokens as u64,
            "usage"
        );
        let mut hashes = Vec::new();
        for (position, (item, &fixture_index)) in
            response.data.into_iter().zip(&expected.indices).enumerate()
        {
            ensure!(
                item.object == "embedding" && item.index == position,
                "row/index order"
            );
            let row = match item.embedding {
                Encoded::Float(row) => {
                    ensure!(encoding == "float", "expected base64");
                    row
                }
                Encoded::Base64(text) => {
                    ensure!(encoding == "base64", "expected float");
                    ensure!(text.len() == ROW_BYTES.div_ceil(3) * 4, "base64 length");
                    let bytes = STANDARD.decode(&text)?;
                    ensure!(bytes.len() == ROW_BYTES, "base64 decoded length");
                    ensure!(STANDARD.encode(&bytes) == text, "noncanonical base64");
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
                "position":position,
                "fixture_id":fixture.id,
                "output_f32le_sha256":sha256(&row_bytes(&row))
            }));
        }
        Ok(json!({
            "http_status":200,"raw_http_body_sha256":reply.raw_body_sha256,
            "rows":hashes,"prompt_tokens":tokens
        }))
    }

    async fn reconcile(&mut self) -> Result<Value> {
        let barrier = self.control.call("barrier", json!({})).await?;
        ensure!(barrier["quiescent"] == true, "not quiescent");
        let bytes = bounded_file(&self.manifest, MAX_MANIFEST)?;
        ensure!(
            bytes.starts_with(&self.prefix),
            "manifest changed/truncated"
        );
        ensure!(bytes.last() == Some(&b'\n'), "partial manifest line");
        ensure!(
            barrier["manifest_bytes"].as_u64() == Some(bytes.len() as u64),
            "barrier manifest size"
        );
        ensure!(
            barrier["manifest_sha256"] == sha256(&bytes),
            "barrier manifest hash"
        );

        let delta = &bytes[self.prefix.len()..];
        let mut active: Option<(String, usize)> = None;
        let mut seen: BTreeMap<String, usize> = BTreeMap::new();
        let mut terminals = BTreeSet::new();
        for line in delta.split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
            let event: Value = serde_json::from_slice(line)?;
            self.seq += 1;
            ensure!(event["seq"].as_u64() == Some(self.seq), "manifest sequence");
            ensure!(
                event["instance_id"] == self.control.instance,
                "manifest instance"
            );
            let id = event["case_id"].as_str().context("manifest case ID")?;
            let expected = self
                .expected
                .get(id)
                .context("unattributed physical/request work")?;
            match event["event"].as_str().context("manifest event")? {
                "forward_begin" => {
                    ensure!(expected.forwards, "work for rejected/cancelled case");
                    ensure!(active.is_none(), "overlapping physical forwards");
                    ensure!(!terminals.contains(id), "forward after terminal");
                    let position = *seen.get(id).unwrap_or(&0);
                    ensure!(
                        event["input_index"].as_u64() == Some(position as u64),
                        "forward index"
                    );
                    let fixture_index = *expected.indices.get(position).context("extra forward")?;
                    ensure!(
                        event["token_count"].as_u64()
                            == Some(self.bundle.policy.fixtures[fixture_index].ids.len() as u64),
                        "physical token count"
                    );
                    ensure!(
                        event["physical_batch"].as_u64() == Some(1),
                        "physical batch"
                    );
                    ensure!(
                        event["audit_identity"] == self.bundle.policy.audit_identity,
                        "forward audit"
                    );
                    ensure!(
                        event["math"]["reduced_precision"]
                            == json!({"f32":false,"f16":false,"bf16":false}),
                        "math flags"
                    );
                    active = Some((id.to_string(), position));
                }
                "forward_end" => {
                    let (active_id, position) = active.take().context("end without begin")?;
                    ensure!(active_id == id, "mismatched end");
                    ensure!(
                        event["input_index"].as_u64() == Some(position as u64),
                        "end index"
                    );
                    ensure!(
                        event["audit_identity"] == self.bundle.policy.audit_identity,
                        "end audit"
                    );
                    ensure!(
                        event["math"]["reduced_precision"]
                            == json!({"f32":false,"f16":false,"bf16":false}),
                        "end math flags"
                    );
                    let fixture = &self.bundle.policy.fixtures[expected.indices[position]];
                    ensure!(
                        event["output_f32le_sha256"] == fixture.golden_sha256,
                        "manifest output"
                    );
                    seen.insert(id.to_string(), position + 1);
                }
                "request_terminal" => {
                    ensure!(
                        active.as_ref().map(|(a, _)| a.as_str()) != Some(id),
                        "terminal during work"
                    );
                    ensure!(terminals.insert(id.to_string()), "duplicate terminal");
                    ensure!(
                        event["status"].as_u64() == Some(expected.status as u64),
                        "terminal status"
                    );
                    ensure!(event["code"] == json!(expected.code), "terminal code");
                }
                _ => bail!("unrecognized event in physical manifest"),
            }
        }
        ensure!(active.is_none(), "unpaired physical forward");
        for (id, expected) in &self.expected {
            ensure!(terminals.contains(id), "missing terminal for {id}");
            let count = *seen.get(id).unwrap_or(&0);
            ensure!(
                count
                    == if expected.forwards {
                        expected.indices.len()
                    } else {
                        0
                    },
                "physical-forward count for {id}"
            );
        }
        let details = json!({
            "quiescent":true,"manifest_sha256":sha256(&bytes),
            "reconciled_requests":self.expected.len(),"last_seq":self.seq
        });
        self.prefix = bytes;
        self.expected.clear();
        Ok(details)
    }

    async fn one(
        &mut self,
        reporter: &mut Reporter,
        id: &str,
        wire: WireRequest,
        expected: Expected,
        encoding: &str,
    ) -> Result<()> {
        self.expected.insert(id.to_string(), expected.clone());
        let result = async {
            let reply = self.http.send(id, wire).await?;
            let wire_details = self.check_reply(&reply, &expected, encoding)?;
            let manifest = self.reconcile().await?;
            Ok(json!({"wire":wire_details,"manifest":manifest}))
        }
        .await;
        reporter.record(id, result)
    }

    async fn valid(
        &mut self,
        reporter: &mut Reporter,
        id: &str,
        indices: Vec<usize>,
        encoding: &str,
    ) -> Result<()> {
        let wire = self.request(&indices, encoding);
        self.one(
            reporter,
            id,
            wire,
            Expected {
                indices,
                status: 200,
                code: None,
                forwards: true,
            },
            encoding,
        )
        .await
    }

    async fn batch(
        &mut self,
        reporter: &mut Reporter,
        prefix: &str,
        requests: Vec<Vec<usize>>,
    ) -> Result<()> {
        let barrier = Arc::new(Barrier::new(requests.len() + 1));
        let mut jobs = Vec::new();
        for (slot, indices) in requests.into_iter().enumerate() {
            let id = format!("{prefix}-{slot}");
            let encoding = if slot % 2 == 0 { "float" } else { "base64" };
            let expected = Expected {
                indices: indices.clone(),
                status: 200,
                code: None,
                forwards: true,
            };
            self.expected.insert(id.clone(), expected.clone());
            let wire = self.request(&indices, encoding);
            let http = self.http.clone();
            let start = barrier.clone();
            let task_id = id.clone();
            let task = tokio::spawn(async move {
                start.wait().await;
                http.send(&task_id, wire).await
            });
            jobs.push((id, expected, encoding, task));
        }
        barrier.wait().await;
        let mut failed = false;
        // Join every task, including after a failed request. Association is by
        // the frozen client ID, never by completion order.
        for (id, expected, encoding, task) in jobs {
            let result = match task.await {
                Ok(Ok(reply)) => self.check_reply(&reply, &expected, encoding),
                Ok(Err(error)) => Err(error),
                Err(error) => Err(error.into()),
            };
            if reporter.record(&id, result).is_err() {
                failed = true;
            }
        }
        let id = format!("{prefix}-manifest");
        let reconciled = self.reconcile().await;
        if reporter.record(&id, reconciled).is_err() {
            failed = true;
        }
        ensure!(!failed, "burst failed");
        Ok(())
    }

    async fn saturation(&mut self, reporter: &mut Reporter, short: usize) -> Result<()> {
        self.control.call("hold_worker", json!({})).await?;
        let result = async {
            let n = self.bundle.policy.limits.admission_capacity;
            let mut tasks = Vec::new();
            for i in 0..n {
                let id = format!("saturation-admitted-{i}");
                let wire = self.request(&[short], "float");
                let http = self.http.clone();
                let task_id = id.clone();
                self.expected.insert(
                    id.clone(),
                    Expected {
                        indices: vec![short],
                        status: 200,
                        code: None,
                        forwards: true,
                    },
                );
                tasks.push((
                    id,
                    tokio::spawn(async move { http.send(&task_id, wire).await }),
                ));
            }
            self.control
                .observe(self.bundle.policy.limits.observation_ms, |v| {
                    v["admitted"].as_u64() == Some(n as u64)
                        && v["queued"].as_u64() == Some(n as u64)
                        && v["active_case_id"].is_null()
                })
                .await?;
            let id = "saturation-rejected";
            let expected = Expected {
                indices: vec![],
                status: 503,
                code: Some("queue_full".into()),
                forwards: false,
            };
            self.expected.insert(id.into(), expected.clone());
            let reply = self.http.send(id, self.request(&[short], "float")).await?;
            reporter.record(id, self.check_reply(&reply, &expected, "float"))?;
            self.control.call("release_worker", json!({})).await?;

            let mut failed = false;
            for (id, task) in tasks {
                let expected = self.expected[&id].clone();
                let result = match task.await {
                    Ok(Ok(reply)) => self.check_reply(&reply, &expected, "float"),
                    Ok(Err(e)) => Err(e),
                    Err(e) => Err(e.into()),
                };
                if reporter.record(&id, result).is_err() {
                    failed = true;
                }
            }
            let manifest = self.reconcile().await;
            if reporter.record("saturation-manifest", manifest).is_err() {
                failed = true;
            }
            ensure!(!failed, "saturation admitted requests failed");
            Ok(())
        }
        .await;
        // Idempotent cleanup. Not a correctness-request retry.
        let release = self.control.call("release_worker", json!({})).await;
        reporter.record(
            "saturation-control",
            result.and(release.map(|_| json!({"released":true}))),
        )
    }

    async fn queued_deadline(&mut self, reporter: &mut Reporter, short: usize) -> Result<()> {
        self.control.call("hold_worker", json!({})).await?;
        let id = "queued-deadline";
        let expected = Expected {
            indices: vec![],
            status: 504,
            code: Some("deadline_exceeded".into()),
            forwards: false,
        };
        self.expected.insert(id.into(), expected.clone());
        let result = async {
            let reply = self.http.send(id, self.request(&[short], "float")).await?;
            self.check_reply(&reply, &expected, "float")
        }
        .await;
        let release = self.control.call("release_worker", json!({})).await;
        let result = match (result, release) {
            (Ok(details), Ok(_)) => self
                .reconcile()
                .await
                .map(|m| json!({"wire":details,"manifest":m})),
            (Err(e), _) | (_, Err(e)) => Err(e),
        };
        reporter.record(id, result)
    }

    async fn disconnect(
        &mut self,
        reporter: &mut Reporter,
        indices: Vec<usize>,
        queued: bool,
    ) -> Result<()> {
        let id = if queued {
            "disconnect-queued"
        } else {
            "disconnect-active"
        };
        if queued {
            self.control.call("hold_worker", json!({})).await?;
        }
        let result = async {
            let stream = self
                .http
                .raw_request(id, &self.request(&indices, "float"))
                .await?;
            self.control
                .observe(self.bundle.policy.limits.observation_ms, |v| {
                    if queued {
                        v["queued_case_ids"]
                            .as_array()
                            .map(|a| a.iter().any(|x| x == id))
                            .unwrap_or(false)
                    } else {
                        v["active_case_id"].as_str() == Some(id)
                    }
                })
                .await?;
            drop(stream); // Close both directions; do not merely cancel a body future.
            self.control
                .observe(self.bundle.policy.limits.observation_ms, |v| {
                    v["disconnected_case_ids"]
                        .as_array()
                        .map(|a| a.iter().any(|x| x == id))
                        .unwrap_or(false)
                })
                .await?;
            self.expected.insert(
                id.into(),
                Expected {
                    indices,
                    status: 499,
                    code: Some("client_disconnected".into()),
                    forwards: !queued,
                },
            );
            Ok(())
        }
        .await;
        let release = self.control.call("release_worker", json!({})).await;
        let result = match (result, release) {
            (Ok(()), Ok(_)) => self.reconcile().await,
            (Err(e), _) | (_, Err(e)) => Err(e),
        };
        reporter.record(id, result)
    }
}

fn register_suite(reporter: &mut Reporter, policy: &Policy) {
    for id in [
        "launch",
        "wire-string",
        "wire-null",
        "wire-user",
        "wire-charset",
        "wire-eight",
    ] {
        reporter.require(id);
    }
    for i in 0..policy.fixtures.len() {
        for encoding in ["float", "base64"] {
            reporter.require(format!("serial-{i}-{encoding}"));
        }
    }
    for mapping in &policy.mappings {
        for i in 0..mapping.requests.len() {
            reporter.require(format!("mapped-{}-{i}", mapping.m2_group_id));
        }
    }
    for n in [2, 4, 8] {
        for burst in 0..10 {
            for slot in 0..n {
                reporter.require(format!("burst-{n}-{burst}-{slot}"));
            }
            reporter.require(format!("burst-{n}-{burst}-manifest"));
        }
    }
    for name in [
        "malformed",
        "empty-batch",
        "empty-text",
        "whitespace",
        "invalid-last",
        "wrong-model",
        "encoding",
        "encoding-type",
        "dimensions",
        "user-type",
        "token-array",
        "too-many",
        "over-length",
        "media",
        "compression",
        "oversized",
        "oversized-chunked",
    ] {
        reporter.require(format!("invalid-{name}"));
        reporter.require(format!("recover-{name}"));
    }
    if policy.pad_text.is_some() {
        reporter.require("invalid-pad");
        reporter.require("recover-pad");
    }
    for i in 0..policy.limits.admission_capacity {
        reporter.require(format!("saturation-admitted-{i}"));
    }
    for id in [
        "saturation-rejected",
        "saturation-manifest",
        "saturation-control",
        "recover-saturation",
        "queued-deadline",
        "recover-deadline",
        "disconnect-queued",
        "recover-disconnect-queued",
        "disconnect-active",
        "recover-disconnect-active",
        "final-manifest",
    ] {
        reporter.require(id);
    }
}

async fn execute(
    reporter: &mut Reporter,
    bundle: Bundle,
    manifest: PathBuf,
    base: &str,
) -> Result<()> {
    register_suite(reporter, &bundle.policy);
    let mut control = Control {
        path: bundle.policy.control_socket.clone(),
        run_id: bundle.policy.run_id.clone(),
        instance: String::new(),
        ms: bundle.policy.limits.control_ms,
    };
    let launch_result = async {
        let nonce = format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        );
        let hello = control.call("hello", json!({"challenge":nonce})).await?;
        ensure!(hello["challenge"] == nonce, "launch challenge");
        let instance = hello["instance_id"]
            .as_str()
            .context("instance")?
            .to_string();
        ensure!(!instance.is_empty(), "empty instance");
        ensure!(
            hello["policy_sha256"] == bundle.policy_sha256,
            "server policy"
        );
        ensure!(
            hello["audit_identity"] == bundle.policy.audit_identity,
            "server audit"
        );
        ensure!(
            hello["deployment"] == bundle.policy.deployment,
            "server deployment"
        );
        ensure!(hello["state"] == "Ready", "server not ready");
        let pid = hello["pid"].as_u64().context("server PID")?;
        ensure!(pid > 1 && pid <= u32::MAX as u64, "PID range");
        ensure!(
            file_sha256(Path::new(&format!("/proc/{pid}/exe")))?
                == bundle.policy.artifacts["serve_m3"].sha256,
            "actual local server executable"
        );
        let prefix = bounded_file(&manifest, MAX_MANIFEST)?;
        ensure!(prefix.last() == Some(&b'\n'), "startup manifest incomplete");
        let lines: Vec<&[u8]> = prefix
            .split(|&b| b == b'\n')
            .filter(|l| !l.is_empty())
            .collect();
        ensure!(lines.len() == 1, "fresh isolated manifest required");
        let startup: Value = serde_json::from_slice(lines[0])?;
        ensure!(
            startup["event"] == "startup" && startup["seq"] == 0,
            "startup record"
        );
        ensure!(startup["instance_id"] == instance, "startup instance");
        ensure!(
            startup["policy_sha256"] == bundle.policy_sha256,
            "startup policy"
        );
        ensure!(
            startup["audit_identity"] == bundle.policy.audit_identity,
            "startup audit"
        );
        ensure!(
            startup["deployment"] == bundle.policy.deployment,
            "startup deployment"
        );
        ensure!(
            startup["server_sha256"] == bundle.policy.artifacts["serve_m3"].sha256,
            "startup executable"
        );
        ensure!(
            hello["manifest_sha256"] == sha256(&prefix),
            "launch manifest"
        );
        Ok((instance, prefix, hello))
    }
    .await;
    let (instance, prefix, hello) = match launch_result {
        Ok(x) => x,
        Err(e) => {
            reporter.record("launch", Err(e))?;
            unreachable!()
        }
    };
    reporter.record("launch", Ok(hello))?;
    control.instance = instance.clone();
    let http = Http::new(base, instance, bundle.policy.limits.clone())?;
    let mut runner = Runner {
        bundle,
        http,
        control,
        manifest,
        prefix,
        seq: 0,
        expected: BTreeMap::new(),
    };
    let short = runner
        .bundle
        .policy
        .fixtures
        .iter()
        .enumerate()
        .min_by_key(|(_, f)| f.ids.len())
        .unwrap()
        .0;
    let long = runner
        .bundle
        .policy
        .fixtures
        .iter()
        .position(|f| f.ids.len() == 1024)
        .unwrap();

    let model = runner.bundle.policy.model.clone();
    let text = runner.bundle.policy.fixtures[short].text.clone();
    for (id, value) in [
        ("wire-string", json!({"model":model,"input":text})),
        (
            "wire-null",
            json!({"model":model,"input":[text],"encoding_format":null}),
        ),
        (
            "wire-user",
            json!({"model":model,"input":[text],"user":"not-to-be-logged"}),
        ),
    ] {
        runner
            .one(
                reporter,
                id,
                WireRequest::json(value),
                Expected {
                    indices: vec![short],
                    status: 200,
                    code: None,
                    forwards: true,
                },
                "float",
            )
            .await?;
    }
    let mut charset = runner.request(&[short], "float");
    charset.media = "application/json; charset=utf-8".into();
    runner
        .one(
            reporter,
            "wire-charset",
            charset,
            Expected {
                indices: vec![short],
                status: 200,
                code: None,
                forwards: true,
            },
            "float",
        )
        .await?;
    runner
        .valid(reporter, "wire-eight", vec![short; 8], "float")
        .await?;

    for i in 0..runner.bundle.policy.fixtures.len() {
        for encoding in ["float", "base64"] {
            runner
                .valid(
                    reporter,
                    &format!("serial-{i}-{encoding}"),
                    vec![i],
                    encoding,
                )
                .await?;
        }
    }
    let mappings = runner.bundle.policy.mappings.clone();
    for mapping in mappings {
        for (i, request) in mapping.requests.into_iter().enumerate() {
            runner
                .valid(
                    reporter,
                    &format!("mapped-{}-{i}", mapping.m2_group_id),
                    request,
                    "float",
                )
                .await?;
        }
    }
    // Fixed schedule v1. Every burst contains short and long inputs.
    // N>=2 always includes a multi-string request and a duplicate-input request.
    for n in [2, 4, 8] {
        for burst in 0..10 {
            let requests = (0..n)
                .map(|slot| match slot % 4 {
                    0 => vec![short, long],
                    1 => vec![short, short],
                    2 => vec![long],
                    _ => vec![(burst + slot) % runner.bundle.policy.fixtures.len(), short],
                })
                .collect();
            runner
                .batch(reporter, &format!("burst-{n}-{burst}"), requests)
                .await?;
        }
    }

    let mut invalid: Vec<(&str, WireRequest, u16, &str)> = Vec::new();
    let valid = runner.request(&[short], "float");
    let mut malformed = valid.clone();
    malformed.body = b"{".to_vec();
    invalid.push(("malformed", malformed, 400, "invalid_request"));
    for (name, value, code) in [
        (
            "empty-batch",
            json!({"model":model,"input":[]}),
            "empty_batch",
        ),
        (
            "empty-text",
            json!({"model":model,"input":""}),
            "empty_sequence",
        ),
        (
            "whitespace",
            json!({"model":model,"input":" \n\t"}),
            "empty_sequence",
        ),
        (
            "invalid-last",
            json!({"model":model,"input":[text,""]}),
            "empty_sequence",
        ),
        (
            "wrong-model",
            json!({"model":"wrong","input":text}),
            "invalid_model",
        ),
        (
            "encoding",
            json!({"model":model,"input":text,"encoding_format":"hex"}),
            "unsupported_encoding_format",
        ),
        (
            "encoding-type",
            json!({"model":model,"input":text,"encoding_format":3}),
            "unsupported_encoding_format",
        ),
        (
            "dimensions",
            json!({"model":model,"input":text,"dimensions":2048}),
            "invalid_request",
        ),
        (
            "user-type",
            json!({"model":model,"input":text,"user":3}),
            "invalid_request",
        ),
        (
            "token-array",
            json!({"model":model,"input":[1,2]}),
            "invalid_request",
        ),
        (
            "too-many",
            json!({"model":model,"input":vec![text.clone();9]}),
            "invalid_request",
        ),
        (
            "over-length",
            json!({"model":model,"input":[text,runner.bundle.policy.over_length_text]}),
            "over_length",
        ),
    ] {
        invalid.push((name, WireRequest::json(value), 400, code));
    }
    let mut media = valid.clone();
    media.media = "text/plain".into();
    invalid.push(("media", media, 415, "unsupported_media_type"));
    let mut compressed = valid.clone();
    compressed.content_encoding = Some("gzip".into());
    invalid.push(("compression", compressed, 415, "unsupported_media_type"));
    let mut oversized = valid.clone();
    oversized.body = vec![b' '; BODY_LIMIT + 1];
    invalid.push(("oversized", oversized.clone(), 413, "payload_too_large"));
    oversized.chunked = true;
    invalid.push(("oversized-chunked", oversized, 413, "payload_too_large"));
    if let Some(pad_text) = &runner.bundle.policy.pad_text {
        invalid.push((
            "pad",
            WireRequest::json(json!({
                "model":model,"input":[text,pad_text]
            })),
            400,
            "pad_token",
        ));
    }
    for (name, request, status, code) in invalid {
        runner
            .one(
                reporter,
                &format!("invalid-{name}"),
                request,
                Expected {
                    indices: vec![],
                    status,
                    code: Some(code.into()),
                    forwards: false,
                },
                "float",
            )
            .await?;
        runner
            .valid(reporter, &format!("recover-{name}"), vec![short], "float")
            .await?;
    }

    runner.saturation(reporter, short).await?;
    runner
        .valid(reporter, "recover-saturation", vec![short], "float")
        .await?;
    runner.queued_deadline(reporter, short).await?;
    runner
        .valid(reporter, "recover-deadline", vec![short], "float")
        .await?;
    runner.disconnect(reporter, vec![short], true).await?;
    runner
        .valid(reporter, "recover-disconnect-queued", vec![short], "float")
        .await?;
    runner.disconnect(reporter, vec![long; 8], false).await?;
    runner
        .valid(reporter, "recover-disconnect-active", vec![short], "float")
        .await?;
    let final_manifest = runner.reconcile().await;
    reporter.record("final-manifest", final_manifest)?;
    Ok(())
}

async fn entry(reporter: &mut Reporter) -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 5,
        "usage: accept_m3 MODEL_DIR SERVER_MANIFEST_JSONL M3_POLICY_JSON BASE_URL"
    );
    let result = load_bundle(Path::new(&args[1]), Path::new(&args[3]), "accept_m3");
    let bundle = match result {
        Ok(bundle) => bundle,
        Err(error) => {
            reporter.record("preflight", Err(error))?;
            unreachable!()
        }
    };
    reporter.record(
        "preflight",
        Ok(json!({
            "policy_sha256":bundle.policy_sha256,
            "artifact_hashes_verified":true,
            "prerequisite_attestations_hash_bound":true,
            "source_audit_and_signature_not_rederived_by_driver":true
        })),
    )?;
    execute(reporter, bundle, PathBuf::from(&args[2]), &args[4]).await
}

#[tokio::main]
async fn main() {
    let mut reporter = Reporter::new();
    reporter.require("preflight");
    let result = entry(&mut reporter).await;
    if let Err(error) = &result {
        eprintln!("accept_m3 stopped: {error:#}");
    }
    let remaining: Vec<String> = reporter.pending.iter().cloned().collect();
    for id in remaining {
        reporter.unexecuted(&id, "earlier required failure or unavailable prerequisite");
    }
    for id in [
        "control.policy_drift_poison",
        "control.post_poison_readiness_liveness_refusal",
        "control.restart_exact_recovery",
        "control.deadline_during_execution_retirement",
        "control.graceful_shutdown",
    ] {
        reporter.unexecuted(id, "requires destructive acceptance-control orchestration");
    }
    println!(
        "{}",
        json!({
            "kind":"done",
            "non_destructive_pass":result.is_ok() && !reporter.failed,
            "accepted":false,
            "required_destructive_evidence_missing":true
        })
    );
    std::process::exit(1);
}
