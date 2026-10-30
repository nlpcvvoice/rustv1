//! Live, serialized HTTP-to-executor reconciliation.
//!
//! Trust input:
//! - orchestrator-produced handoff;
//! - inherited M3_ACCEPTANCE_SECRET;
//! - unchanged contract::load for all consumed artifact bindings.
//!
//! This phase creates no CUDA owner. It tests the already-running batched
//! server, then leaves admission quiesced. No restart or independent-mode
//! regression is implied by completion.

use crate::contract::{
    self, bounded, compare, digest_valid, file_sha, label_valid, norm, row_bytes, sha, unresolved,
    value, write_new, Bundle, ROW_BYTES,
};
use anyhow::{bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::{
    body::{Body, Frame, SizeHint},
    client::conn::http1,
    header::HeaderMap,
    Request, Uri,
};
use hyper_util::rt::TokioIo;
use ring::rand::{SecureRandom, SystemRandom};
use rustv1::model::{BATCHED_EXECUTION_POLICY, BATCHED_HTTP_CONTRACT};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    convert::Infallible,
    fs::{self, File, OpenOptions},
    io::Write,
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    pin::Pin,
    task::{Context as TaskContext, Poll},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    task::JoinHandle,
    time::timeout,
};

const MODEL: &str = "nvidia/Nemotron-3-Embed-1B";
const PHASE: &str = "http_reconciliation";
const JSON_LIMIT: usize = 32 * 1024 * 1024;
const MANIFEST_LIMIT: usize = 256 * 1024 * 1024;
const BODY_LIMIT: usize = 1_048_576;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Handoff {
    schema: String,
    model_root: PathBuf,
    prepass_input_path: PathBuf,
    prepass_input_sha256: String,
    live_manifest_path: PathBuf,
    instance: String,
    pid: u32,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Limits {
    connect_ms: u64,
    headers_ms: u64,
    body_ms: u64,
    control_ms: u64,
    observation_ms: u64,
    server_request_ms: u64,
    response_bytes: usize,
    admission_capacity: usize,
    queue_capacity: usize,
}

impl Limits {
    fn check(&self) -> Result<()> {
        for n in [
            self.connect_ms,
            self.headers_ms,
            self.body_ms,
            self.control_ms,
            self.observation_ms,
        ] {
            ensure!((1..=300_000).contains(&n), "invalid client deadline");
        }
        ensure!(
            self.server_request_ms == 60_000
                && self.response_bytes == BODY_LIMIT
                && self.admission_capacity == 64
                && self.queue_capacity == 64,
            "unexpected server limits"
        );
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Mapping {
    m2_group_id: usize,
    adaptation: String,
    requests: Vec<Vec<usize>>,
}

fn closed(value: &Value, fields: &[&str]) -> Result<()> {
    let object = value.as_object().context("expected JSON object")?;
    ensure!(
        object.len() == fields.len() && fields.iter().all(|field| object.contains_key(*field)),
        "unexpected or missing JSON fields"
    );
    Ok(())
}

fn integer(value: &Value, field: &str) -> Result<u64> {
    value[field]
        .as_u64()
        .with_context(|| format!("missing/non-integer {field}"))
}

fn math(record: &Value, bundle: &Bundle) -> Result<()> {
    let m = &record["math"];
    ensure!(
        m["audit_identity"] == bundle.audit_identity
            && m["candle_core_version"] == "0.9.2"
            && m["reduced_precision"]["f32"] == false
            && m["reduced_precision"]["f16"] == false
            && m["reduced_precision"]["bf16"] == false,
        "executor math binding"
    );
    Ok(())
}

fn lines(bytes: &[u8]) -> Result<Vec<Value>> {
    ensure!(
        !bytes.is_empty() && bytes.ends_with(b"\n"),
        "empty/truncated JSONL"
    );
    bytes[..bytes.len() - 1]
        .split(|&b| b == b'\n')
        .map(|line| {
            ensure!(!line.is_empty(), "empty JSONL record");
            Ok(serde_json::from_slice(line)?)
        })
        .collect()
}

#[derive(Clone)]
struct Control {
    address: SocketAddr,
    secret: String,
    call_budget: Duration,
    observation_budget: Duration,
}

impl Control {
    async fn call(&self, command: &str, challenge: Option<&str>) -> Result<Value> {
        let budget = if command == "observe" {
            self.observation_budget
        } else {
            self.call_budget
        };
        timeout(budget, async {
            let mut socket = TcpStream::connect(self.address).await?;
            let mut payload = json!({
                "secret": self.secret,
                "command": command
            });
            if let Some(challenge) = challenge {
                payload["challenge"] = json!(challenge);
            }
            let mut bytes = serde_json::to_vec(&payload)?;
            bytes.push(b'\n');
            socket.write_all(&bytes).await?;
            socket.shutdown().await?;

            // Bounded before allocation, including a peer that never sends LF.
            let mut reply = Vec::new();
            loop {
                let byte = socket.read_u8().await.context("control EOF")?;
                ensure!(reply.len() < 1024 * 1024, "control response too large");
                if byte == b'\n' {
                    break;
                }
                reply.push(byte);
            }
            let reply: Value = serde_json::from_slice(&reply)?;
            closed(&reply, &["ok", "challenge", "observation"])?;
            ensure!(reply["ok"] == true, "control command failed");
            ensure!(
                reply["challenge"] == json!(challenge),
                "control challenge mismatch"
            );
            Ok(reply["observation"].clone())
        })
        .await
        .context("control deadline")?
    }
}

fn observation(observation: &Value, handoff: &Handoff, ready: bool, attested: bool) -> Result<u64> {
    let mut fields = vec![
        "instance",
        "pid",
        "lifecycle",
        "worker_alive",
        "ready",
        "forward_calls",
        "startup_output_sha256",
    ];
    if attested {
        fields.push("attestation");
    }
    closed(observation, &fields)?;
    ensure!(
        observation["instance"] == handoff.instance
            && observation["pid"] == handoff.pid
            && observation["lifecycle"] == "ready"
            && observation["worker_alive"] == true
            && observation["ready"] == ready,
        "observation identity/lifecycle mismatch"
    );
    ensure!(
        observation["startup_output_sha256"]
            .as_str()
            .is_some_and(digest_valid),
        "missing startup digest"
    );
    integer(observation, "forward_calls")
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

#[derive(Clone)]
struct Wire {
    body: Vec<u8>,
    media: String,
    content_encoding: Option<String>,
    chunked: bool,
}

impl Wire {
    fn json(value: Value) -> Result<Self> {
        Ok(Self {
            body: serde_json::to_vec(&value)?,
            media: "application/json".into(),
            content_encoding: None,
            chunked: false,
        })
    }
}

struct Reply {
    status: u16,
    request_id: Option<u64>,
    headers: Value,
    bytes: Vec<u8>,
}

fn one_header<'a>(headers: &'a HeaderMap, name: &str) -> Result<Option<&'a str>> {
    let mut values = headers.get_all(name).iter();
    let first = values.next();
    ensure!(values.next().is_none(), "duplicate response header {name}");
    first
        .map(|value| value.to_str().map_err(Into::into))
        .transpose()
}

struct Http {
    authority: String,
    address: SocketAddr,
    instance: String,
    limits: Limits,
}

impl Http {
    fn new(base: &str, instance: &str, limits: Limits) -> Result<Self> {
        let uri: Uri = base.parse()?;
        ensure!(uri.scheme_str() == Some("http"), "HTTP scheme required");
        ensure!(
            uri.path().is_empty() || uri.path() == "/",
            "BASE_URL must not contain a path"
        );
        ensure!(uri.query().is_none(), "BASE_URL query forbidden");
        let authority = uri.authority().context("HTTP authority")?;
        ensure!(!authority.as_str().contains('@'), "userinfo forbidden");
        let host = uri.host().context("HTTP host")?;
        let host = host.trim_start_matches('[').trim_end_matches(']');
        let ip: IpAddr = host.parse().context("numeric loopback required")?;
        ensure!(ip.is_loopback(), "HTTP must use loopback");
        Ok(Self {
            authority: authority.as_str().to_owned(),
            address: SocketAddr::new(ip, uri.port_u16().unwrap_or(80)),
            instance: instance.to_owned(),
            limits,
        })
    }

    async fn send(&self, case: &str, wire: &Wire) -> Result<Reply> {
        ensure!(label_valid(case), "invalid case label");
        ensure!(wire.body.len() <= BODY_LIMIT + 65536, "request size bound");

        let stream = timeout(
            Duration::from_millis(self.limits.connect_ms),
            TcpStream::connect(self.address),
        )
        .await
        .context("HTTP connect deadline")??;

        let mut builder = http1::Builder::new();
        builder.max_headers(64);
        builder.max_buf_size(64 * 1024);
        let (mut sender, connection) = timeout(
            Duration::from_millis(self.limits.connect_ms),
            builder.handshake(TokioIo::new(stream)),
        )
        .await
        .context("HTTP handshake deadline")??;
        let _connection = AbortConnection(tokio::spawn(async move {
            let _ = connection.await;
        }));

        let mut request = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("host", &self.authority)
            .header("content-type", &wire.media)
            .header("accept", "application/json")
            .header("accept-encoding", "identity")
            .header("connection", "close")
            .header("x-acceptance-case-id", case);
        if let Some(encoding) = &wire.content_encoding {
            request = request.header("content-encoding", encoding);
        }
        let request = request.body(RequestBody {
            bytes: Bytes::from(wire.body.clone()),
            chunked: wire.chunked,
        })?;
        let response = timeout(
            Duration::from_millis(self.limits.headers_ms),
            sender.send_request(request),
        )
        .await
        .context("HTTP response-header deadline")??;

        let headers = response.headers();
        ensure!(
            one_header(headers, "x-rv1-instance-id")? == Some(self.instance.as_str())
                && one_header(headers, "x-acceptance-case-id")? == Some(case),
            "HTTP correlation headers"
        );
        let media = one_header(headers, "content-type")?.context("Content-Type")?;
        ensure!(
            media
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("application/json"),
            "HTTP response media"
        );
        ensure!(
            one_header(headers, "content-encoding")?.is_none(),
            "encoded response forbidden"
        );

        let request_id = one_header(headers, "x-rv1-request-id")?
            .map(|text| -> Result<u64> {
                let id: u64 = text.parse()?;
                ensure!(id > 0 && text == id.to_string(), "noncanonical request ID");
                Ok(id)
            })
            .transpose()?;

        let recorded_headers = json!({
            "x-rv1-instance-id": self.instance,
            "x-acceptance-case-id": case,
            "x-rv1-request-id": request_id,
            "content-type": media
        });
        let status = response.status().as_u16();
        let mut body = response.into_body();
        let bytes = timeout(Duration::from_millis(self.limits.body_ms), async {
            let mut bytes = Vec::new();
            while let Some(frame) = body.frame().await {
                let frame = frame?;
                if let Some(data) = frame.data_ref() {
                    ensure!(
                        data.len() <= self.limits.response_bytes.saturating_sub(bytes.len()),
                        "response body exceeds bound"
                    );
                    bytes.extend_from_slice(data);
                } else {
                    bail!("unexpected response trailers");
                }
            }
            Ok::<_, anyhow::Error>(bytes)
        })
        .await
        .context("HTTP response-body deadline")??;

        Ok(Reply {
            status,
            request_id,
            headers: recorded_headers,
            bytes,
        })
    }
}

enum Expected {
    Success {
        indices: Vec<usize>,
        encoding: &'static str,
    },
    Rejection {
        status: u16,
        code: &'static str,
        closed_admission: bool,
        gate: &'static str,
    },
}

struct Case {
    id: String,
    wire: Wire,
    expected: Expected,
}

fn valid_case(
    bundle: &Bundle,
    id: String,
    indices: Vec<usize>,
    encoding: &'static str,
) -> Result<Case> {
    ensure!(
        !indices.is_empty()
            && indices.len() <= 8
            && indices.iter().all(|&i| i < bundle.fixtures.len()),
        "invalid positive layout"
    );
    let input: Vec<&str> = indices
        .iter()
        .map(|&i| bundle.fixtures[i].text.as_str())
        .collect();
    Ok(Case {
        id,
        wire: Wire::json(json!({
            "model": MODEL,
            "input": input,
            "encoding_format": encoding
        }))?,
        expected: Expected::Success { indices, encoding },
    })
}

fn rejection(
    id: &str,
    body: Value,
    status: u16,
    code: &'static str,
    gate: &'static str,
) -> Result<Case> {
    Ok(Case {
        id: id.into(),
        wire: Wire::json(body)?,
        expected: Expected::Rejection {
            status,
            code,
            closed_admission: false,
            gate,
        },
    })
}

fn error_contract(code: &str) -> Result<(&'static str, Option<&'static str>)> {
    Ok(match code {
        "empty_batch" => ("Input batch is empty", Some("input")),
        "empty_sequence" => ("Input sequence is empty", Some("input")),
        "pad_token" => ("Input contains a padding token", Some("input")),
        "over_length" => ("Input exceeds the sequence limit", Some("input")),
        "invalid_request" => ("Invalid request", None),
        "invalid_model" => ("Unsupported model", Some("model")),
        "unsupported_encoding_format" => ("Unsupported encoding format", Some("encoding_format")),
        "unsupported_media_type" => ("Unsupported media type or content encoding", None),
        "payload_too_large" => ("Request body exceeds the size limit", None),
        "not_ready" => ("Service is not ready", None),
        _ => bail!("unrecognized planned rejection"),
    })
}

fn check_error(reply: &Reply, status: u16, code: &str) -> Result<Value> {
    ensure!(reply.status == status, "wrong rejection status");
    let value: Value = serde_json::from_slice(&reply.bytes)?;
    let (message, param) = error_contract(code)?;
    let kind = if (400..500).contains(&status) {
        "invalid_request_error"
    } else if status == 503 {
        "service_unavailable_error"
    } else {
        "api_error"
    };
    ensure!(
        value
            == json!({
                "error": {
                    "message": message,
                    "type": kind,
                    "param": param,
                    "code": code
                }
            }),
        "error envelope mismatch or partial success payload"
    );
    Ok(value)
}

fn decode_success(
    reply: &Reply,
    bundle: &Bundle,
    indices: &[usize],
    encoding: &str,
) -> Result<(Vec<Vec<f32>>, Value)> {
    ensure!(reply.status == 200, "positive HTTP status");
    let value: Value = serde_json::from_slice(&reply.bytes)?;
    closed(&value, &["object", "data", "model", "usage"])?;
    ensure!(
        value["object"] == "list" && value["model"] == MODEL,
        "success object/model"
    );
    let tokens: usize = indices.iter().map(|&i| bundle.fixtures[i].ids.len()).sum();
    ensure!(
        value["usage"] == json!({"prompt_tokens": tokens, "total_tokens": tokens}),
        "usage must count valid input tokens, including duplicates, not padding"
    );
    let items = value["data"].as_array().context("embedding data")?;
    ensure!(items.len() == indices.len(), "response row cardinality");

    let mut output = Vec::new();
    for (position, item) in items.iter().enumerate() {
        closed(item, &["object", "index", "embedding"])?;
        ensure!(
            item["object"] == "embedding" && item["index"] == position,
            "embedding object/index/order"
        );
        let row: Vec<f32> = match encoding {
            "float" => {
                ensure!(item["embedding"].is_array(), "float encoding required");
                serde_json::from_value(item["embedding"].clone())?
            }
            "base64" => {
                let text = item["embedding"]
                    .as_str()
                    .context("base64 string required")?;
                let bytes = STANDARD.decode(text)?;
                ensure!(bytes.len() == ROW_BYTES, "base64 decoded byte length");
                ensure!(STANDARD.encode(&bytes) == text, "noncanonical base64");
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
            _ => bail!("invalid expected encoding"),
        };
        norm(&row)?;
        output.push(row);
    }
    Ok((output, value["usage"].clone()))
}

fn build_plan(bundle: &Bundle, policy: &Value) -> Result<Vec<Case>> {
    // This artifact is resolved exclusively through the verified input bundle.
    let mapping_value = value(bundle.path("mappings_v1")?)?;
    ensure!(
        policy["mappings"] == mapping_value,
        "materialized mappings mismatch"
    );
    let mappings: Vec<Mapping> = serde_json::from_value(mapping_value)?;
    ensure!(mappings.len() == 46, "46 mappings required");
    for (position, mapping) in mappings.iter().enumerate() {
        ensure!(
            mapping.m2_group_id == position + 1 && !mapping.adaptation.is_empty(),
            "mapping order/identity"
        );
        ensure!(!mapping.requests.is_empty(), "empty mapping");
        for request in &mapping.requests {
            ensure!(
                (1..=8).contains(&request.len()) && request.iter().all(|&i| i < 22),
                "mapping request layout"
            );
        }
    }
    for mapping in &mappings[41..] {
        ensure!(mapping.requests.len() == 1, "recovery must be one request");
    }

    let tokenizer =
        tokenizers::Tokenizer::from_bytes(&bounded(bundle.path("tokenizer")?, JSON_LIMIT)?)
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
    let encode = |text: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?
            .get_ids()
            .to_vec())
    };

    let over = policy["over_length_text"]
        .as_str()
        .context("overlength text")?;
    ensure!(!over.trim().is_empty(), "empty overlength text");
    let over_ids = encode(over)?;
    ensure!(
        over_ids.len() > bundle.engine.max_len
            && over_ids.iter().all(|&id| id < 131072 && id != 11),
        "overlength probe must really exceed max_len without padding/OOV"
    );

    let short = &bundle.fixtures[0].text;
    let mut plan = Vec::new();
    for mapping in &mappings[..41] {
        for (request, indices) in mapping.requests.iter().enumerate() {
            plan.push(valid_case(
                bundle,
                format!("mapped-{}-{request}", mapping.m2_group_id),
                indices.clone(),
                "float",
            )?);
        }
    }

    let pad = match policy.get("pad_text").context("pad_text field")? {
        Value::Null => None,
        Value::String(text) => {
            ensure!(
                !text.trim().is_empty() && encode(text)? == vec![11],
                "declared pad_text must encode to exactly [11]"
            );
            Some(text.as_str())
        }
        _ => bail!("pad_text must be string or null"),
    };
    let recovery_rejections = vec![
        rejection(
            "invalid-empty-batch",
            json!({"model": MODEL, "input": []}),
            400,
            "empty_batch",
            "http_input_validation",
        )?,
        rejection(
            "invalid-last-empty",
            json!({"model": MODEL, "input": [short, ""]}),
            400,
            "empty_sequence",
            "http_input_validation",
        )?,
        if let Some(pad) = pad {
            rejection(
                "invalid-last-pad",
                json!({"model": MODEL, "input": [short, pad]}),
                400,
                "pad_token",
                "token_validation",
            )?
        } else {
            rejection(
                "invalid-last-empty-pad-substitute",
                json!({"model": MODEL, "input": [short, ""]}),
                400,
                "empty_sequence",
                "http_input_validation-not-pad-token-coverage",
            )?
        },
        rejection(
            "invalid-last-overlength",
            json!({"model": MODEL, "input": [short, over]}),
            400,
            "over_length",
            "token_length_validation",
        )?,
        rejection(
            "invalid-token-array",
            json!({"model": MODEL, "input": [1, 2]}),
            400,
            "invalid_request",
            "http_schema-not-oov-token-coverage",
        )?,
    ];
    for (offset, rejection) in recovery_rejections.into_iter().enumerate() {
        plan.push(rejection);
        let mapping = &mappings[41 + offset];
        plan.push(valid_case(
            bundle,
            format!("mapped-{}-0", mapping.m2_group_id),
            mapping.requests[0].clone(),
            "float",
        )?);
    }

    // Explicit singleton wire-encoding coverage for all fixture indices.
    for fixture in 0..22 {
        plan.push(valid_case(
            bundle,
            format!("base64-singleton-{fixture}"),
            vec![fixture],
            "base64",
        )?);
    }

    for (name, indices) in [("mixed", vec![0, 6, 0, 1]), ("eight-long", vec![6; 8])] {
        for encoding in ["float", "base64"] {
            plan.push(valid_case(
                bundle,
                format!("{name}-{encoding}"),
                indices.clone(),
                encoding,
            )?);
        }
    }

    // Scalar input, omitted encoding, explicit null encoding, and user field.
    let mut scalar = valid_case(bundle, "scalar-default".into(), vec![0], "float")?;
    scalar.wire = Wire::json(json!({"model": MODEL, "input": short}))?;
    plan.push(scalar);
    let mut null = valid_case(bundle, "scalar-null-encoding".into(), vec![0], "float")?;
    null.wire = Wire::json(json!({
        "model": MODEL, "input": short, "encoding_format": null, "user": "accept-m4"
    }))?;
    plan.push(null);

    let mut chunked = valid_case(bundle, "chunked-valid".into(), vec![0, 1], "base64")?;
    chunked.wire.chunked = true;
    chunked.wire.media = "application/json; charset=utf-8".into();
    plan.push(chunked);

    for (name, input, code) in [
        ("empty-text", json!(""), "empty_sequence"),
        ("whitespace", json!(" \t\n"), "empty_sequence"),
        ("null-input", Value::Null, "invalid_request"),
        ("mixed-input-types", json!([short, 1]), "invalid_request"),
    ] {
        plan.push(rejection(
            &format!("invalid-{name}"),
            json!({"model": MODEL, "input": input}),
            400,
            code,
            "http_input_validation",
        )?);
    }
    plan.push(rejection(
        "invalid-model",
        json!({"model": "wrong", "input": short}),
        400,
        "invalid_model",
        "http_model_validation",
    )?);
    for (name, encoding) in [
        ("string", json!("hex")),
        ("number", json!(1)),
        ("object", json!({})),
        ("array", json!([])),
        ("boolean", json!(true)),
    ] {
        plan.push(rejection(
            &format!("invalid-encoding-{name}"),
            json!({"model": MODEL, "input": short, "encoding_format": encoding}),
            400,
            "unsupported_encoding_format",
            "http_encoding_validation",
        )?);
    }
    plan.push(rejection(
        "invalid-unknown-field",
        json!({"model": MODEL, "input": short, "unknown": true}),
        400,
        "invalid_request",
        "http_schema",
    )?);

    // Both hit RequestInput::texts, not the executor capacity gate.
    for count in [9, bundle.shapes.batching.max_batch_size + 1] {
        plan.push(rejection(
            &format!("http-input-count-{count}"),
            json!({"model": MODEL, "input": vec![short; count]}),
            400,
            "invalid_request",
            "http_input_count-not-physical_capacity",
        )?);
    }

    let mut malformed = rejection(
        "invalid-malformed-json",
        json!({}),
        400,
        "invalid_request",
        "http_json",
    )?;
    malformed.wire.body = b"{".to_vec();
    plan.push(malformed);

    let mut media = rejection(
        "invalid-media",
        json!({"model": MODEL, "input": short}),
        415,
        "unsupported_media_type",
        "http_media",
    )?;
    media.wire.media = "text/plain".into();
    plan.push(media);

    let mut compressed = rejection(
        "invalid-content-encoding",
        json!({"model": MODEL, "input": short}),
        415,
        "unsupported_media_type",
        "http_content_encoding",
    )?;
    compressed.wire.content_encoding = Some("gzip".into());
    plan.push(compressed);

    for chunked in [false, true] {
        let mut oversized = rejection(
            if chunked {
                "payload-too-large-chunked"
            } else {
                "payload-too-large-length"
            },
            json!({}),
            413,
            "payload_too_large",
            "http_body_limit",
        )?;
        oversized.wire.body = vec![b' '; BODY_LIMIT + 1];
        oversized.wire.chunked = chunked;
        plan.push(oversized);
    }

    // Demonstrate recovery after the remaining HTTP-only rejection suite.
    plan.push(valid_case(
        bundle,
        "final-open-recovery".into(),
        vec![0, 6],
        "float",
    )?);

    let mut closed = rejection(
        "closed-admission",
        json!({"model": MODEL, "input": short}),
        503,
        "not_ready",
        "closed_admission",
    )?;
    closed.expected = Expected::Rejection {
        status: 503,
        code: "not_ready",
        closed_admission: true,
        gate: "closed_admission",
    };
    plan.push(closed);

    let mut ids = BTreeSet::new();
    for case in &plan {
        ensure!(
            label_valid(&case.id) && ids.insert(case.id.clone()),
            "case ID collision"
        );
        ensure!(
            case.wire.body.len() <= BODY_LIMIT + 65536,
            "planned request bound"
        );
    }
    Ok(plan)
}

struct Ledger {
    file: File,
    path: PathBuf,
    required: BTreeSet<String>,
    completed: BTreeSet<String>,
    order: Vec<String>,
}

impl Ledger {
    fn new(out: &Path, plan: &[Case]) -> Result<Self> {
        let path = out.join("accept_records.jsonl");
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let required = plan.iter().map(|case| case.id.clone()).collect();
        Ok(Self {
            file,
            path,
            required,
            completed: BTreeSet::new(),
            order: Vec::new(),
        })
    }

    fn pass(&mut self, id: &str, details: Value) -> Result<()> {
        ensure!(
            self.required.contains(id) && !self.completed.contains(id),
            "unexpected/duplicate ledger completion"
        );
        let record = json!({
            "acceptance_phase": PHASE,
            "case_id": id,
            "status": "executed",
            "pass": true,
            "details": details
        });
        serde_json::to_writer(&mut self.file, &record)?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        self.completed.insert(id.into());
        self.order.push(id.into());
        println!("{record}");
        Ok(())
    }

    fn failure(&mut self, id: &str, error: &anyhow::Error) -> Result<()> {
        let record = json!({
            "acceptance_phase": PHASE,
            "case_id": id,
            "status": "executed",
            "pass": false,
            "error": format!("{error:#}")
        });
        serde_json::to_writer(&mut self.file, &record)?;
        self.file.write_all(b"\n")?;
        self.file.sync_all()?;
        println!("{record}");
        Ok(())
    }

    fn finish(&mut self, plan: &[Case]) -> Result<()> {
        ensure!(self.completed == self.required, "incomplete item-3 ledger");
        ensure!(
            self.order == plan.iter().map(|case| case.id.clone()).collect::<Vec<_>>(),
            "ledger order differs from required plan"
        );
        self.file.sync_all()?;
        let records = lines(&bounded(&self.path, MANIFEST_LIMIT)?)?;
        ensure!(records.len() == plan.len(), "ledger record count");
        for (record, case) in records.iter().zip(plan) {
            ensure!(
                record["acceptance_phase"] == PHASE
                    && record["case_id"] == case.id
                    && record["status"] == "executed"
                    && record["pass"] == true,
                "ledger reread validation"
            );
        }
        Ok(())
    }
}

struct Runner<'a> {
    bundle: &'a Bundle,
    handoff: &'a Handoff,
    http: Http,
    control: Control,
    out: &'a Path,
    prefix: Vec<u8>,
    next_forward: u64,
    last_observation: Value,
    last_request_id: Option<u64>,
    seen_request_ids: BTreeSet<u64>,
    layout_hashes: BTreeMap<Vec<usize>, Vec<String>>,
    positives: u64,
    rejections: u64,
}

impl Runner<'_> {
    fn read_manifest(&self) -> Result<Vec<u8>> {
        let bytes = bounded(&self.handoff.live_manifest_path, MANIFEST_LIMIT)?;
        ensure!(
            bytes.starts_with(&self.prefix),
            "live manifest rewritten/truncated"
        );
        Ok(bytes)
    }

    async fn quiesce(&mut self) -> Result<()> {
        let before = self.control.call("observe", None).await?;
        observation(&before, self.handoff, true, false)?;
        ensure!(
            before == self.last_observation,
            "unaccounted work before quiesce"
        );
        ensure!(
            self.read_manifest()? == self.prefix,
            "unaccounted manifest growth"
        );

        let after = self.control.call("quiesce", None).await?;
        observation(&after, self.handoff, false, false)?;
        let mut expected = before;
        expected["ready"] = json!(false);
        ensure!(after == expected, "quiesce changed execution observations");
        ensure!(
            self.read_manifest()? == self.prefix,
            "quiesce produced GPU work"
        );
        self.last_observation = after;
        Ok(())
    }

    fn reconcile_positive(
        &mut self,
        case: &Case,
        request_id: u64,
        indices: &[usize],
        output: &[Vec<f32>],
        suffix: &[u8],
    ) -> Result<Value> {
        let records = lines(suffix)?;
        ensure!(
            records.len() == 2,
            "serialized batched HTTP request must produce exactly one physical forward"
        );
        let begin = &records[0];
        let end = &records[1];
        let b = indices.len();
        let s = indices
            .iter()
            .map(|&i| self.bundle.fixtures[i].ids.len())
            .max()
            .context("empty successful request")?;
        let valid: usize = indices
            .iter()
            .map(|&i| self.bundle.fixtures[i].ids.len())
            .sum();

        ensure!(
            begin["kind"] == "physical_forward_begin"
                && end["kind"] == "physical_forward_end"
                && end["status"] == "ok",
            "physical forward begin/end/status"
        );
        for record in [begin, end] {
            ensure!(
                record["forward_id"] == self.next_forward
                    && record["execution_policy"] == BATCHED_EXECUTION_POLICY
                    && record["profile"] == "F"
                    && record["B"] == b
                    && record["S"] == s
                    && record["projection_rows"] == b * s
                    && record["attention_cells_before_heads"] == b * s * s
                    && record["valid_tokens"] == valid,
                "physical forward identity/geometry/accounting"
            );
            math(record, self.bundle)?;
        }
        ensure!(
            begin["padding_tokens"] == b * s - valid,
            "padding accounting"
        );
        let starts = begin["rows"].as_array().context("physical begin rows")?;
        let ends = end["rows"].as_array().context("physical end rows")?;
        ensure!(
            starts.len() == b && ends.len() == b,
            "physical row cardinality"
        );

        let mut evidence = Vec::new();
        let mut hashes = Vec::new();
        for (position, (&fixture_index, row)) in indices.iter().zip(output).enumerate() {
            let fixture = &self.bundle.fixtures[fixture_index];
            let hash = sha(&row_bytes(row));
            let member = json!({
                "physical_row": position,
                "request_id": request_id,
                "request_row": position,
                "case_id": case.id,
                "valid_length": fixture.ids.len(),
                "token_sha256": fixture.ids_sha256
            });
            ensure!(
                starts[position] == member,
                "request/input/physical begin association"
            );
            let mut completed = member;
            completed["output_sha256"] = json!(hash);
            ensure!(
                ends[position] == completed,
                "request/input/physical/output end association"
            );

            let singleton = compare(row, &self.bundle.singleton[fixture_index])?;
            let reference = compare(row, &self.bundle.reference[fixture_index])?;
            if b == 1 {
                ensure!(hash == fixture.golden_sha256, "physical B1 exact mismatch");
            }
            let (candidate, selection) =
                match self.bundle.candidates.get(&(indices.to_vec(), position)) {
                    Some(candidate) => (candidate.as_slice(), "shape_specific"),
                    None => (
                        self.bundle.canonical(fixture_index, position % 2)?,
                        "canonical_same_fixture_position_modulo_two",
                    ),
                };
            let candidate_comparison = compare(row, candidate)?;
            hashes.push(hash.clone());
            evidence.push(json!({
                "fixture_index": fixture_index,
                "fixture_id": fixture.id,
                "request_row": position,
                "physical_row": position,
                "forward_id": self.next_forward,
                "valid_length": fixture.ids.len(),
                "token_sha256": fixture.ids_sha256,
                "output_f32le_sha256": hash,
                "singleton_comparison": singleton,
                "independent_reference_comparison": reference,
                "candidate_comparison": candidate_comparison,
                "candidate_selection": selection
            }));
        }

        // The phase is serialized and requires one intact physical batch.
        // Thus repeated layouts, including float/base64, must match exactly.
        let repeated = if let Some(previous) = self.layout_hashes.get(indices) {
            ensure!(
                previous == &hashes,
                "identical-layout exact repeat mismatch"
            );
            true
        } else {
            false
        };
        self.layout_hashes.insert(indices.to_vec(), hashes);
        let forward = self.next_forward;
        self.next_forward = forward.checked_add(1).context("forward ID overflow")?;

        Ok(json!({
            "route": "IndependentExecutor::forward_batch",
            "physical_forwards": 1,
            "forward_id": forward,
            "B": b,
            "S": s,
            "valid_tokens": valid,
            "padding_tokens": b * s - valid,
            "rows": evidence,
            "previous_identical_layout_checked_exactly": repeated,
            "association_source": "executor_rows"
        }))
    }

    async fn one(&mut self, case: &Case) -> Result<Value> {
        let closed_admission = matches!(
            &case.expected,
            Expected::Rejection {
                closed_admission: true,
                ..
            }
        );
        let before = self.control.call("observe", None).await?;
        let before_calls = observation(&before, self.handoff, !closed_admission, false)?;
        ensure!(
            before == self.last_observation,
            "unaccounted observation change"
        );
        let before_bytes = self.read_manifest()?;
        ensure!(before_bytes == self.prefix, "unaccounted manifest growth");
        let start = before_bytes.len();

        write_new(
            &self.out.join(format!("{}.request.bin", case.id)),
            &case.wire.body,
        )?;
        let reply = self.http.send(&case.id, &case.wire).await?;
        write_new(
            &self.out.join(format!("{}.response.bin", case.id)),
            &reply.bytes,
        )?;

        if let Some(id) = reply.request_id {
            ensure!(
                self.last_request_id.is_none_or(|last| id > last)
                    && self.seen_request_ids.insert(id),
                "reused or nonmonotonic server request ID"
            );
            self.last_request_id = Some(id);
        }

        // Observe is an owner-queue barrier after the complete HTTP response.
        let after = self.control.call("observe", None).await?;
        let after_calls = observation(&after, self.handoff, !closed_admission, false)?;
        let after_bytes = self.read_manifest()?;
        ensure!(
            after_bytes.starts_with(&before_bytes),
            "manifest prefix changed across request"
        );

        let (execution, response) = match &case.expected {
            Expected::Success { indices, encoding } => {
                let request_id = reply
                    .request_id
                    .context("successful response missing request ID")?;
                ensure!(
                    after_calls
                        == before_calls
                            .checked_add(1)
                            .context("logical counter overflow")?,
                    "success must add exactly one logical backend request"
                );
                let mut expected_observation = before.clone();
                expected_observation["forward_calls"] = json!(after_calls);
                ensure!(
                    after == expected_observation,
                    "unexpected success observation changes"
                );
                let (output, usage) = decode_success(&reply, self.bundle, indices, encoding)?;
                let execution = self.reconcile_positive(
                    case,
                    request_id,
                    indices,
                    &output,
                    &after_bytes[start..],
                )?;
                self.positives += 1;
                (execution, json!({"encoding": encoding, "usage": usage}))
            }
            Expected::Rejection {
                status,
                code,
                closed_admission,
                gate,
            } => {
                if *closed_admission {
                    ensure!(
                        reply.request_id.is_none(),
                        "closed-admission branch unexpectedly supplied a request ID"
                    );
                } else {
                    ensure!(
                        reply.request_id.is_some(),
                        "admitted rejection missing server request ID"
                    );
                }
                let error = check_error(&reply, *status, code)?;
                ensure!(
                    after == before && after_bytes == before_bytes,
                    "rejected request executed or changed observations/manifest"
                );
                self.rejections += 1;
                (
                    json!({
                        "physical_forwards": 0,
                        "logical_backend_request_delta": 0,
                        "zero_execution_proof": {
                            "observations_identical": true,
                            "manifest_bytes_identical": true
                        },
                        "rejection_gate": gate,
                        "physical_capacity_gate_proven": false,
                        "association_source": if *closed_admission {
                            "null-request-id-observation-and-byte-invariance"
                        } else {
                            "http-request-id-observation-and-byte-invariance"
                        }
                    }),
                    error,
                )
            }
        };

        let details = json!({
            "instance": self.handoff.instance,
            "pid": self.handoff.pid,
            "server_request_id": reply.request_id,
            "request": {
                "path": format!("{}.request.bin", case.id),
                "sha256": sha(&case.wire.body),
                "media": case.wire.media,
                "content_encoding": case.wire.content_encoding,
                "chunked": case.wire.chunked
            },
            "response": {
                "path": format!("{}.response.bin", case.id),
                "sha256": sha(&reply.bytes),
                "http_status": reply.status,
                "headers": reply.headers,
                "validation": response
            },
            "observation_before": before,
            "observation_after": after,
            "logical_backend_request_delta": after_calls - before_calls,
            "manifest_interval": {
                "start_byte": start,
                "end_byte": after_bytes.len(),
                "prefix_before_sha256": sha(&before_bytes),
                "prefix_after_sha256": sha(&after_bytes),
                "interval_sha256": sha(&after_bytes[start..])
            },
            "execution": execution
        });
        self.prefix = after_bytes;
        self.last_observation = after;
        Ok(details)
    }
}

fn check_initial_manifest(bytes: &[u8], bundle: &Bundle, attestation: &Value) -> Result<u64> {
    let records = lines(bytes)?;
    let start = &records[0];
    ensure!(
        start["kind"] == "executor_start"
            && start["execution_policy"] == BATCHED_EXECUTION_POLICY
            && start["profile"] == "F"
            && start["batching"] == serde_json::to_value(&bundle.shapes.batching)?,
        "live executor start policy"
    );
    math(start, bundle)?;

    // serve_m3 hashes the pretty-serialized base launch object before adding
    // these runtime control-response fields. Reconstruct those exact bytes.
    let mut base = attestation.clone();
    let base_object = base.as_object_mut().context("attestation object")?;
    for key in ["http_address", "control_address", "startup", "capabilities"] {
        ensure!(
            base_object.remove(key).is_some(),
            "missing runtime attestation field"
        );
    }
    let base_bytes = serde_json::to_vec_pretty(&base)?;
    ensure!(
        start["attestation_sha256"] == sha(&base_bytes),
        "manifest/launch-attestation digest mismatch"
    );

    ensure!(
        (records.len() - 1) % 2 == 0,
        "incomplete initial physical forward"
    );
    let mut next = 0u64;
    // No logical HTTP calls have occurred. Initial forwards must be complete
    // forward_serving startup rows, not batched/request-associated traffic.
    for pair in records[1..].as_chunks::<2>().0 {
        let begin = &pair[0];
        let end = &pair[1];
        ensure!(
            begin["kind"] == "physical_forward_begin"
                && end["kind"] == "physical_forward_end"
                && end["status"] == "ok"
                && begin["forward_id"] == next
                && end["forward_id"] == next
                && begin["logical_request_row"] == next
                && begin["B"] == 1
                && end["B"] == 1
                && begin["S"] == end["S"]
                && begin["projection_rows"] == begin["S"]
                && begin["profile"] == "F"
                && end["profile"] == "F"
                && begin.get("rows").is_none()
                && end.get("rows").is_none()
                && begin["token_sha256"] == end["token_sha256"],
            "initial startup manifest structure"
        );
        ensure!(
            integer(begin, "S")? > 0
                && integer(begin, "S")? <= bundle.engine.max_len as u64
                && begin["token_sha256"].as_str().is_some_and(digest_valid)
                && end["output_sha256"].as_str().is_some_and(digest_valid),
            "initial startup forward fields"
        );
        math(begin, bundle)?;
        math(end, bundle)?;
        next = next.checked_add(1).context("forward ID overflow")?;
    }
    ensure!(next > 0, "missing startup physical forwards");
    Ok(next)
}

fn check_policy(policy: &Value, bundle: &Bundle, control: SocketAddr) -> Result<Limits> {
    ensure!(!unresolved(policy), "unresolved policy");
    ensure!(
        policy["schema"] == "rustv1.acceptance_policy.v3"
            && policy["contract_id"] == "b200-independent-F-v2"
            && policy["http_contract_id"] == BATCHED_HTTP_CONTRACT
            && policy["model"] == MODEL
            && policy["audit_identity"] == bundle.audit_identity
            && policy["samples"] == json!([])
            && policy["deployment"]["execution_policy"] == BATCHED_EXECUTION_POLICY
            && policy["deployment"]["profile"] == "F"
            && policy["deployment"]["batching"] == serde_json::to_value(&bundle.shapes.batching)?
            && policy["deployment"]["scheduler"]["max_http_inputs"] == 8
            && policy["deployment"]["scheduler"]["split_http_request"] == false,
        "HTTP phase policy binding"
    );
    ensure!(
        policy["deployment"]["materialization"]
            == json!({
                "fixtures": {"artifact_role": "fixtures_exec", "count": 22},
                "mappings": {"artifact_role": "mappings_v1", "count": 46}
            }),
        "materialization declarations"
    );
    let policy_control: SocketAddr = policy["launch_control_address"]
        .as_str()
        .context("policy control address")?
        .parse()?;
    ensure!(policy_control == control, "CLI/policy control mismatch");
    ensure!(
        policy["fixtures"] == value(bundle.path("fixtures_exec")?)?,
        "materialized fixture mismatch"
    );

    // Compare consumed policy artifact identities with the verified bundle.
    // Paths are never resolved from the finalized policy in this module.
    for role in [
        "config",
        "engine",
        "tokenizer",
        "cargo_lock",
        "feature_tree",
        "audit",
        "reference_provenance",
        "fixtures_exec",
        "fixture_goldens",
        "fixture_reference",
        "batch_goldens",
        "batch_golden_index",
        "mappings_v1",
    ] {
        let binding = bundle
            .artifacts
            .get(role)
            .context("missing pre-pass binding")?;
        ensure!(
            policy["artifacts"][role]["sha256"] == binding.sha256,
            "policy/pre-pass artifact mismatch: {role}"
        );
    }
    let limits: Limits = serde_json::from_value(policy["limits"].clone())?;
    limits.check()?;
    Ok(limits)
}

async fn execute(
    policy_path: &Path,
    base: &str,
    control_address: &str,
    out: &Path,
    handoff_path: &Path,
) -> Result<String> {
    let handoff_bytes = bounded(handoff_path, JSON_LIMIT)?;
    let handoff: Handoff = serde_json::from_slice(&handoff_bytes)?;
    ensure!(
        handoff.schema == "rustv1.accept_m4_handoff.v1"
            && handoff.pid > 0
            && label_valid(&handoff.instance)
            && handoff.model_root.is_absolute()
            && handoff.prepass_input_path.is_absolute()
            && handoff.live_manifest_path.is_absolute()
            && handoff.live_manifest_path.is_file(),
        "handoff schema/identity/paths"
    );
    let bundle = contract::load(
        &handoff.model_root,
        &handoff.prepass_input_path,
        &handoff.prepass_input_sha256,
    )?;
    ensure!(
        !bundle
            .paths
            .values()
            .any(|path| { handoff.live_manifest_path.canonicalize().ok().as_ref() == Some(path) }),
        "live manifest must not alias a bound immutable artifact"
    );

    let policy_bytes = bounded(policy_path, JSON_LIMIT)?;
    let policy: Value = serde_json::from_slice(&policy_bytes)?;
    let control_address: SocketAddr = control_address.parse()?;
    ensure!(
        control_address.ip().is_loopback(),
        "control must use loopback"
    );
    let limits = check_policy(&policy, &bundle, control_address)?;
    let plan = build_plan(&bundle, &policy)?;
    let secret =
        std::env::var("M3_ACCEPTANCE_SECRET").context("M3_ACCEPTANCE_SECRET must be inherited")?;
    ensure!(
        (32..=256).contains(&secret.len()),
        "control credential length"
    );
    let control = Control {
        address: control_address,
        secret,
        call_budget: Duration::from_millis(limits.control_ms),
        observation_budget: Duration::from_millis(limits.observation_ms),
    };
    let http = Http::new(base, &handoff.instance, limits)?;
    let binary_sha256 = file_sha(&std::env::current_exe()?)?;

    fs::create_dir(out).context("OUT_DIR must not already exist")?;
    let mut ledger = Ledger::new(out, &plan)?;

    let mut random = [0u8; 32];
    SystemRandom::new()
        .fill(&mut random)
        .map_err(|_| anyhow::anyhow!("challenge randomness"))?;
    let challenge: String = random.iter().map(|b| format!("{b:02x}")).collect();

    // The orchestrator already attested the launch. Repeat the authenticated
    // challenge before traffic to bind this phase to the still-fresh instance.
    let attested = control.call("attest", Some(&challenge)).await?;
    ensure!(
        observation(&attested, &handoff, true, true)? == 0,
        "HTTP phase requires a fresh instance with no prior logical requests"
    );
    let att = &attested["attestation"];
    ensure!(
        att["schema"] == "rustv1.server_launch_attestation.v1"
            && att["instance_id"] == handoff.instance
            && att["pid"] == handoff.pid
            && att["settings_sha256"] == sha(&policy_bytes)
            && att["audit_sha256"] == bundle.artifacts["audit"].sha256
            && att["http_contract_id"] == BATCHED_HTTP_CONTRACT
            && att["execution_policy"] == BATCHED_EXECUTION_POLICY
            && att["batching"] == serde_json::to_value(&bundle.shapes.batching)?
            && att["startup"]["completed"] == true
            && att["startup"]["comparison"] == "exact_f32_bits"
            && att["startup"]["output_f32le_sha256"] == attested["startup_output_sha256"],
        "fresh launch attestation binding"
    );
    let att_http: SocketAddr = att["http_address"]
        .as_str()
        .context("attested HTTP address")?
        .parse()?;
    let att_control: SocketAddr = att["control_address"]
        .as_str()
        .context("attested control address")?
        .parse()?;
    ensure!(
        att_control == control_address
            && att_http.port() == http.address.port()
            && (att_http.ip() == http.address.ip() || att_http.ip().is_unspecified()),
        "attested endpoint mismatch"
    );

    let initial = bounded(&handoff.live_manifest_path, MANIFEST_LIMIT)?;
    let next_forward = check_initial_manifest(&initial, &bundle, att)?;
    let initial_observation = control.call("observe", None).await?;
    ensure!(
        observation(&initial_observation, &handoff, true, false)? == 0,
        "traffic before phase start"
    );
    let mut without_attestation = attested.clone();
    without_attestation
        .as_object_mut()
        .context("observation object")?
        .remove("attestation");
    ensure!(
        initial_observation == without_attestation,
        "launch observation changed"
    );
    ensure!(
        bounded(&handoff.live_manifest_path, MANIFEST_LIMIT)? == initial,
        "manifest changed during launch binding"
    );

    write_new(
        &out.join("launch_binding.json"),
        &serde_json::to_vec_pretty(&json!({
            "acceptance_phase": PHASE,
            "challenge": challenge,
            "observation": attested,
            "handoff_sha256": sha(&handoff_bytes),
            "policy_sha256": sha(&policy_bytes),
            "prepass_input_sha256": bundle.input_sha256,
            "initial_manifest_bytes": initial.len(),
            "initial_manifest_sha256": sha(&initial),
            "initial_next_forward_id": next_forward
        }))?,
    )?;
    write_new(&out.join("initial_executor_manifest.jsonl"), &initial)?;

    let mut runner = Runner {
        bundle: &bundle,
        handoff: &handoff,
        http,
        control,
        out,
        prefix: initial.clone(),
        next_forward,
        last_observation: initial_observation.clone(),
        last_request_id: None,
        seen_request_ids: BTreeSet::new(),
        layout_hashes: BTreeMap::new(),
        positives: 0,
        rejections: 0,
    };

    for case in &plan {
        let result = async {
            if matches!(
                &case.expected,
                Expected::Rejection {
                    closed_admission: true,
                    ..
                }
            ) {
                runner.quiesce().await?;
            }
            runner.one(case).await
        }
        .await;
        match result {
            Ok(details) => ledger.pass(&case.id, details)?,
            Err(error) => {
                ledger.failure(&case.id, &error)?;
                return Err(error).with_context(|| format!("case {}", case.id));
            }
        }
    }

    ensure!(
        runner.positives > 0
            && runner.rejections > 0
            && runner.prefix.len() > initial.len()
            && runner.next_forward - next_forward == runner.positives,
        "manifest did not grow exactly with successful physical forwards"
    );
    ledger.finish(&plan)?;
    bundle.recheck_bindings()?;
    ensure!(
        file_sha(&handoff.prepass_input_path)? == handoff.prepass_input_sha256
            && bounded(handoff_path, JSON_LIMIT)? == handoff_bytes
            && bounded(policy_path, JSON_LIMIT)? == policy_bytes
            && file_sha(&std::env::current_exe()?)? == binary_sha256,
        "phase input or executable changed"
    );

    let final_observation = runner.control.call("observe", None).await?;
    observation(&final_observation, &handoff, false, false)?;
    ensure!(
        final_observation == runner.last_observation && runner.read_manifest()? == runner.prefix,
        "unreconciled final activity"
    );

    write_new(
        &out.join("executor_manifest_snapshot.jsonl"),
        &runner.prefix,
    )?;
    write_new(
        &out.join("final_observation.json"),
        &serde_json::to_vec_pretty(&final_observation)?,
    )?;

    let artifact_hashes: BTreeMap<&str, &str> = bundle
        .artifacts
        .iter()
        .map(|(role, artifact)| (role.as_str(), artifact.sha256.as_str()))
        .collect();

    // All evidence files are closed/synced or written through write_new before
    // constructing the completion marker. Policy secrets are never copied.
    let mut outputs = BTreeMap::new();
    for entry in fs::read_dir(out)? {
        let entry = entry?;
        ensure!(entry.file_type()?.is_file(), "unexpected output entry");
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 output filename"))?;
        outputs.insert(name, file_sha(&entry.path())?);
    }

    // A final owner observation plus byte equality closes the captured interval.
    let final_check = runner.control.call("observe", None).await?;
    ensure!(
        final_check == final_observation && runner.read_manifest()? == runner.prefix,
        "activity while finalizing evidence"
    );

    let manifest = json!({
        "schema": "rustv1.accept_m4_http_manifest.v1",
        "acceptance_phase": PHASE,
        "scope": "boundary_item_3_live_batched_http_reconciliation",
        "producer": {
            "component": "accept_m4",
            "command": "acceptance-http",
            "binary_sha256": binary_sha256
        },
        "handoff_sha256": sha(&handoff_bytes),
        "policy_sha256": sha(&policy_bytes),
        "prepass_input_sha256": bundle.input_sha256,
        "artifact_sha256": artifact_hashes,
        "instance": handoff.instance,
        "pid": handoff.pid,
        "execution_policy": BATCHED_EXECUTION_POLICY,
        "http_contract_id": BATCHED_HTTP_CONTRACT,
        "live_manifest_path": handoff.live_manifest_path,
        "initial_manifest_bytes": initial.len(),
        "initial_manifest_sha256": sha(&initial),
        "final_manifest_bytes": runner.prefix.len(),
        "final_manifest_sha256": sha(&runner.prefix),
        "initial_observation": initial_observation,
        "final_observation": final_observation,
        "required_case_ids": plan.iter().map(|case| &case.id).collect::<Vec<_>>(),
        "completed_case_ids": ledger.order,
        "pending_required_cases_in_scope": [],
        "exceptions": [],
        "successful_http_requests": runner.positives,
        "rejected_http_requests": runner.rejections,
        "logical_backend_requests": runner.positives,
        "reconciled_physical_forwards": runner.positives,
        "unmatched_physical_records": 0,
        "gates": {
            "handoff_instance_and_pid_verified": true,
            "prepass_loader_reused_unchanged": true,
            "fresh_control_challenge_verified": true,
            "launch_attestation_manifest_binding": true,
            "live_manifest_grew": true,
            "append_only_manifest_prefixes": true,
            "response_schema_usage_encoding": true,
            "server_request_ids_unique_and_monotonic": true,
            "request_observation_interval_physical_row_association": true,
            "rejections_zero_execution": true,
            "closed_admission_request_id_is_null": true,
            "complete_item_3_ledger": true
        },
        "output_sha256": outputs,
        "server_left_quiesced": true,
        "independent_mode_regression_completed": false,
        "scheduler_families_53_through_58_completed": false,
        "physical_capacity_gate_proven": false,
        "pass": true,
        "m4_accepted": false
    });
    let bytes = serde_json::to_vec_pretty(&manifest)?;

    // LAST output file: absent on any preceding failure.
    write_new(&out.join("acceptance_http_manifest.json"), &bytes)?;
    Ok(sha(&bytes))
}

pub fn run(
    policy_path: &Path,
    base: &str,
    control_address: &str,
    out: &Path,
    handoff_path: &Path,
) -> Result<String> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(execute(
        policy_path,
        base,
        control_address,
        out,
        handoff_path,
    ));
    runtime.shutdown_timeout(Duration::from_secs(1));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_objects_reject_missing_and_extra_fields() {
        assert!(closed(&json!({"a": 1}), &["a"]).is_ok());
        assert!(closed(&json!({}), &["a"]).is_err());
        assert!(closed(&json!({"a": 1, "b": 2}), &["a"]).is_err());
    }

    #[test]
    fn jsonl_rejects_truncation_and_blank_records() {
        assert!(lines(b"{\"a\":1}\n").is_ok());
        assert!(lines(b"{\"a\":1}").is_err());
        assert!(lines(b"{\"a\":1}\n\n").is_err());
        assert!(lines(b"").is_err());
    }

    #[test]
    fn error_validation_rejects_partial_success_and_wrong_code() {
        let mut reply = Reply {
            status: 400,
            request_id: Some(1),
            headers: json!({}),
            bytes: serde_json::to_vec(&json!({
                "error": {
                    "message": "Input batch is empty",
                    "type": "invalid_request_error",
                    "param": "input",
                    "code": "empty_batch"
                }
            }))
            .unwrap(),
        };
        assert!(check_error(&reply, 400, "empty_batch").is_ok());
        assert!(check_error(&reply, 400, "empty_sequence").is_err());
        let mut body: Value = serde_json::from_slice(&reply.bytes).unwrap();
        body["data"] = json!([]);
        reply.bytes = serde_json::to_vec(&body).unwrap();
        assert!(check_error(&reply, 400, "empty_batch").is_err());
    }

    #[test]
    fn http_endpoint_requires_numeric_loopback_and_no_path() {
        let limits = Limits {
            connect_ms: 1000,
            headers_ms: 65000,
            body_ms: 10000,
            control_ms: 30000,
            observation_ms: 2000,
            server_request_ms: 60000,
            response_bytes: BODY_LIMIT,
            admission_capacity: 64,
            queue_capacity: 64,
        };
        assert!(Http::new("http://127.0.0.1:9000", "instance", limits.clone()).is_ok());
        assert!(Http::new("http://localhost:9000", "instance", limits.clone()).is_err());
        assert!(Http::new("http://192.0.2.1:9000", "instance", limits.clone()).is_err());
        assert!(Http::new("http://127.0.0.1:9000/path", "instance", limits).is_err());
    }

    #[test]
    fn token_digest_uses_little_endian_u32() {
        assert_eq!(contract::ids_sha(&[1]), sha(&[1, 0, 0, 0]));
    }
}
