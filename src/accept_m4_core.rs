//! M4 coordinator prefix: finalized policy through independent owner join.
//!
//! This module deliberately has no overall-acceptance entry point.
//! run_prefix returns the still-incomplete global ledger, its runtime, and
//! the authenticated orchestration connection for the remaining phases.
//!
//! Initial launch is performed by the existing external supervisor.
//! The server control socket authenticates/binds that launch; it cannot
//! launch or shut down a process.
//!
//! Linux is required for independent old-process-exit proof. Merely receiving
//! orch_exit, losing HTTP connectivity, or observing a terminal worker is
//! never sufficient.
//!
//! No completion manifest is written here. The final coordinator must finish
//! the full required ledger and write its completion manifest LAST.
#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_imports))]

use crate::{
    acceptance_http,
    acceptance_policy::{self as policy, Ledger, Loaded},
    contract::{
        bounded, digest_valid, file_sha, ids_sha, label_valid, row_bytes, sha, value, write_new,
    },
};
use anyhow::{bail, ensure, Context, Result};
use ring::{
    hmac,
    rand::{SecureRandom, SystemRandom},
};
use rustv1::{
    http_serve::StartupCheck,
    model::{BATCHED_EXECUTION_POLICY, BATCHED_HTTP_CONTRACT},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::Write,
    net::SocketAddr,
    path::{Component, Path, PathBuf},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    time::{sleep, timeout, Instant},
};

const JSON_LIMIT: usize = 32 * 1024 * 1024;
const RECORD_LIMIT: usize = 256 * 1024 * 1024;
const FRAME_LIMIT: usize = 1024 * 1024;
const EXIT_BUDGET: Duration = Duration::from_secs(120);

fn text<'a>(v: &'a Value, field: &str) -> Result<&'a str> {
    v[field]
        .as_str()
        .with_context(|| format!("missing string: {field}"))
}

fn uint(v: &Value, field: &str) -> Result<u64> {
    v[field]
        .as_u64()
        .with_context(|| format!("missing unsigned integer: {field}"))
}

fn closed(v: &Value, fields: &[&str]) -> Result<()> {
    let object = v.as_object().context("expected object")?;
    ensure!(
        object.len() == fields.len() && fields.iter().all(|field| object.contains_key(*field)),
        "unexpected or missing fields"
    );
    Ok(())
}

pub fn jsonl(bytes: &[u8]) -> Result<Vec<Value>> {
    ensure!(
        !bytes.is_empty() && bytes.ends_with(b"\n"),
        "empty or truncated JSONL"
    );
    bytes[..bytes.len() - 1]
        .split(|b| *b == b'\n')
        .map(|line| {
            ensure!(!line.is_empty(), "blank JSONL record");
            Ok(serde_json::from_slice(line)?)
        })
        .collect()
}

pub(crate) fn random_label() -> Result<String> {
    let mut bytes = [0u8; 32];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| anyhow::anyhow!("randomness unavailable"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn private_directory(path: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(path)
        .context("output directory must be new")?;
    Ok(())
}

fn private_file(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

fn write_json(path: &Path, v: &Value) -> Result<String> {
    let bytes = serde_json::to_vec_pretty(v)?;
    write_new(path, &bytes)?;
    Ok(sha(&bytes))
}

fn math(record: &Value, loaded: &Loaded) -> Result<()> {
    let m = &record["math"];
    ensure!(
        m["audit_identity"] == loaded.bundle.audit_identity
            && m["candle_core_version"] == "0.9.2"
            && m["reduced_precision"] == json!({"f32": false, "f16": false, "bf16": false}),
        "executor math identity"
    );
    Ok(())
}

fn startup_check(loaded: &Loaded) -> Result<StartupCheck> {
    Ok(serde_json::from_slice(&bounded(
        &loaded.paths["startup_check"],
        JSON_LIMIT,
    )?)?)
}

fn startup_digest(check: &StartupCheck) -> String {
    let bytes: Vec<u8> = check
        .embeddings
        .iter()
        .flat_map(|row| row_bytes(row))
        .collect();
    sha(&bytes)
}

/// Validate only forward_serving pairs, not a batched rows manifest.
///
/// logical_request_row restarts at zero for every forward_serving call.
/// Server request IDs are intentionally absent from this manifest format.
fn serving_pairs(
    records: &[Value],
    next: u64,
    tokens: &[Vec<u32>],
    hashes: &[String],
    loaded: &Loaded,
) -> Result<Vec<Value>> {
    ensure!(
        !tokens.is_empty() && tokens.len() == hashes.len() && records.len() == tokens.len() * 2,
        "forward_serving interval cardinality"
    );
    let mut evidence = Vec::with_capacity(tokens.len());
    for (position, (ids, output_hash)) in tokens.iter().zip(hashes).enumerate() {
        ensure!(digest_valid(output_hash), "output digest");
        let forward = next
            .checked_add(u64::try_from(position)?)
            .context("forward counter overflow")?;
        let begin = &records[2 * position];
        let end = &records[2 * position + 1];
        let token_hash = ids_sha(ids);
        ensure!(
            begin["kind"] == "physical_forward_begin"
                && end["kind"] == "physical_forward_end"
                && end["status"] == "ok"
                && begin["logical_request_row"] == position
                && begin["projection_rows"] == ids.len()
                && end["output_sha256"] == *output_hash,
            "forward_serving pair structure/output"
        );
        for record in [begin, end] {
            ensure!(
                record["forward_id"] == forward
                    && record["B"] == 1
                    && record["S"] == ids.len()
                    && record["profile"] == "F"
                    && record["token_sha256"] == token_hash
                    && record.get("rows").is_none()
                    && record.get("request_id").is_none()
                    && record.get("request_row").is_none()
                    && record.get("case_id").is_none(),
                "forward_serving geometry/identity"
            );
            math(record, loaded)?;
        }
        evidence.push(json!({
            "logical_request_row": position,
            "physical_row": 0,
            "forward_id": forward,
            "B": 1,
            "S": ids.len(),
            "token_sha256": token_hash,
            "output_f32le_sha256": output_hash
        }));
    }
    Ok(evidence)
}

pub(crate) fn check_startup(
    bytes: &[u8],
    loaded: &Loaded,
    attestation_sha256: &str,
    batched: bool,
) -> Result<Value> {
    let records = jsonl(bytes)?;
    let start = records.first().context("executor start")?;
    ensure!(
        start["kind"] == "executor_start"
            && start["profile"] == "F"
            && start["attestation_sha256"] == attestation_sha256
            && start["stream_policy"] == "single_owner_no_overlap",
        "executor start binding"
    );
    if batched {
        ensure!(
            start["execution_policy"] == BATCHED_EXECUTION_POLICY
                && start["batching"] == serde_json::to_value(&loaded.bundle.shapes.batching)?,
            "batched executor start"
        );
    } else {
        ensure!(
            start["execution_policy"] == "independent_sequence_v1"
                && start["physical_B"] == 1
                && start["physical_S"] == "valid_token_count"
                && start.get("batching").is_none(),
            "independent executor start"
        );
    }
    math(start, loaded)?;
    let check = startup_check(loaded)?;
    let hashes: Vec<String> = check
        .embeddings
        .iter()
        .map(|row| sha(&row_bytes(row)))
        .collect();
    let rows = serving_pairs(&records[1..], 0, &check.tokens, &hashes, loaded)?;
    Ok(json!({
        "route": "IndependentExecutor::forward_serving",
        "physical_forwards": check.tokens.len(),
        "next_forward_id": check.tokens.len(),
        "startup_output_sha256": startup_digest(&check),
        "manifest_bytes": bytes.len(),
        "manifest_sha256": sha(bytes),
        "rows": rows,
        "unmatched_records": 0
    }))
}

/* Existing A1-A6 orchestration wire format. */

fn decode_key(s: &str) -> Result<Vec<u8>> {
    ensure!(digest_valid(s), "invalid orchestration key");
    (0..s.len())
        .step_by(2)
        .map(|i| Ok(u8::from_str_radix(&s[i..i + 2], 16)?))
        .collect()
}

/// Matches the attached accept_m3 canonicalization:
/// sorted object keys, UTF-8 strings, integer numbers only.
pub fn canonical_json(v: &Value) -> Result<String> {
    fn emit(v: &Value, out: &mut String) -> Result<()> {
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    out.push_str(&i.to_string());
                } else if let Some(u) = n.as_u64() {
                    out.push_str(&u.to_string());
                } else {
                    bail!("orchestration envelope numbers must be integers");
                }
            }
            Value::String(s) => out.push_str(&serde_json::to_string(s)?),
            Value::Array(values) => {
                out.push('[');
                for (i, v) in values.iter().enumerate() {
                    if i != 0 {
                        out.push(',');
                    }
                    emit(v, out)?;
                }
                out.push(']');
            }
            Value::Object(object) => {
                let sorted: BTreeMap<&str, &Value> =
                    object.iter().map(|(k, v)| (k.as_str(), v)).collect();
                out.push('{');
                for (i, (k, v)) in sorted.into_iter().enumerate() {
                    if i != 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(k)?);
                    out.push(':');
                    emit(v, out)?;
                }
                out.push('}');
            }
        }
        Ok(())
    }
    let mut out = String::new();
    emit(v, &mut out)?;
    Ok(out)
}

async fn read_frame(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    loop {
        let byte = stream
            .read_u8()
            .await
            .context("control/orchestration EOF")?;
        if byte == b'\n' {
            ensure!(!bytes.is_empty(), "empty frame");
            return Ok(bytes);
        }
        ensure!(bytes.len() < FRAME_LIMIT, "frame exceeds bound");
        bytes.push(byte);
    }
}

/// Never derive Debug or Serialize: this object contains credentials.
///
/// Any failed or timed-out operation makes the connection unusable. This
/// prevents a cancelled partial frame from being retried as a new message.
pub struct Orchestration {
    stream: TcpStream,
    send_key: hmac::Key,
    receive_key: hmac::Key,
    send_sequence: u64,
    receive_sequence: u64,
    budget: Duration,
    usable: bool,
    transcript: File,
}

impl Orchestration {
    pub async fn connect(loaded: &Loaded, transcript: &Path) -> Result<Self> {
        let port: u16 = std::env::var("RV1_ORCH_PORT")
            .context("RV1_ORCH_PORT required")?
            .parse()?;
        ensure!(port != 0, "zero orchestration port");
        let budget = Duration::from_millis(uint(&loaded.policy["limits"], "control_ms")?);
        let stream = timeout(
            budget,
            TcpStream::connect(SocketAddr::from(([127, 0, 0, 1], port))),
        )
        .await
        .context("orchestration connect deadline")??;
        // Preserve the attached accept_m3 directional interpretation.
        let send_key = decode_key(text(&loaded.policy, "orch_secret_server_to_orchestrator")?)?;
        let receive_key = decode_key(text(&loaded.policy, "orch_secret_orchestrator_to_server")?)?;
        Ok(Self {
            stream,
            send_key: hmac::Key::new(hmac::HMAC_SHA256, &send_key),
            receive_key: hmac::Key::new(hmac::HMAC_SHA256, &receive_key),
            send_sequence: 0,
            receive_sequence: 0,
            budget,
            usable: true,
            transcript: private_file(transcript)?,
        })
    }

    fn record(&mut self, direction: &str, envelope: &Value) -> Result<()> {
        serde_json::to_writer(
            &mut self.transcript,
            &json!({"direction": direction, "envelope": envelope}),
        )?;
        self.transcript.write_all(b"\n")?;
        self.transcript.sync_data()?;
        Ok(())
    }

    pub async fn send(&mut self, kind: &str, fields: Value) -> Result<Value> {
        ensure!(self.usable, "orchestration connection is terminal");
        ensure!(fields.is_object(), "orchestration fields must be an object");
        self.usable = false;
        let next = self
            .send_sequence
            .checked_add(1)
            .context("send sequence overflow")?;
        let mut envelope = json!({"kind": kind, "seq": self.send_sequence, "fields": fields});
        let canonical = canonical_json(&envelope)?;
        let tag = hmac::sign(&self.send_key, canonical.as_bytes());
        let mac: String = tag.as_ref().iter().map(|b| format!("{b:02x}")).collect();
        envelope["mac"] = json!(mac);
        let mut bytes = serde_json::to_vec(&envelope)?;
        ensure!(bytes.len() <= FRAME_LIMIT, "outgoing frame bound");
        bytes.push(b'\n');
        timeout(self.budget, self.stream.write_all(&bytes))
            .await
            .context("orchestration send deadline")??;
        self.record("driver_to_orchestrator", &envelope)?;
        self.send_sequence = next;
        self.usable = true;
        Ok(envelope)
    }

    pub async fn receive(&mut self) -> Result<Value> {
        self.receive_with_budget(self.budget).await
    }

    /// Replacement startup includes server load and startup-check budgets.
    /// This changes only the receive watchdog, not framing/authentication.
    pub async fn receive_with_budget(&mut self, budget: Duration) -> Result<Value> {
        ensure!(
            !budget.is_zero() && budget <= Duration::from_secs(660),
            "invalid orchestration receive budget"
        );
        ensure!(self.usable, "orchestration connection is terminal");
        self.usable = false;
        let next = self
            .receive_sequence
            .checked_add(1)
            .context("receive sequence overflow")?;
        let bytes = timeout(budget, read_frame(&mut self.stream))
            .await
            .context("orchestration receive deadline")??;
        let envelope: Value = serde_json::from_slice(&bytes)?;
        closed(&envelope, &["kind", "seq", "fields", "mac"])?;
        ensure!(
            envelope["kind"].is_string() && envelope["fields"].is_object(),
            "orchestration envelope types"
        );
        let mac = decode_key(text(&envelope, "mac")?)?;
        let payload = json!({
            "kind": envelope["kind"],
            "seq": envelope["seq"],
            "fields": envelope["fields"]
        });
        let canonical = canonical_json(&payload)?;
        hmac::verify(&self.receive_key, canonical.as_bytes(), &mac)
            .map_err(|_| anyhow::anyhow!("orchestration MAC verification failed"))?;
        ensure!(
            uint(&envelope, "seq")? == self.receive_sequence,
            "orchestration sequence mismatch"
        );
        self.record("orchestrator_to_driver", &envelope)?;
        self.receive_sequence = next;
        self.usable = true;
        Ok(envelope)
    }

    pub fn sync(&self) -> Result<()> {
        self.transcript.sync_all()?;
        Ok(())
    }
}

pub struct Control {
    address: SocketAddr,
    secret: String,
    budget: Duration,
    observation_budget: Duration,
}

impl Control {
    pub fn from_policy(loaded: &Loaded) -> Result<Self> {
        let address: SocketAddr = text(&loaded.policy, "launch_control_address")?.parse()?;
        ensure!(
            address.ip().is_loopback() && address.port() != 0,
            "invalid control address"
        );
        let secret = std::env::var("M3_ACCEPTANCE_SECRET")
            .context("M3_ACCEPTANCE_SECRET must be inherited")?;
        ensure!(
            (32..=256).contains(&secret.len()),
            "invalid control credential length"
        );
        Ok(Self {
            address,
            secret,
            budget: Duration::from_millis(uint(&loaded.policy["limits"], "control_ms")?),
            observation_budget: Duration::from_millis(uint(
                &loaded.policy["limits"],
                "observation_ms",
            )?),
        })
    }

    pub async fn call(&self, command: &str, challenge: Option<&str>) -> Result<Value> {
        let budget = if command == "observe" {
            self.observation_budget
        } else {
            self.budget
        };
        timeout(budget, async {
            let mut stream = TcpStream::connect(self.address).await?;
            let mut payload = json!({"secret": self.secret, "command": command});
            if let Some(challenge) = challenge {
                payload["challenge"] = json!(challenge);
            }
            let mut bytes = serde_json::to_vec(&payload)?;
            bytes.push(b'\n');
            stream.write_all(&bytes).await?;
            stream.shutdown().await?;
            let reply: Value = serde_json::from_slice(&read_frame(&mut stream).await?)?;
            closed(&reply, &["ok", "challenge", "observation"])?;
            ensure!(
                reply["ok"] == true && reply["challenge"] == json!(challenge),
                "control response/challenge mismatch"
            );
            ensure!(reply["observation"].is_object(), "control observation");
            Ok(reply["observation"].clone())
        })
        .await
        .context("control deadline")?
    }
}

/* Linux process identity and exit proof. */

#[derive(Clone)]
pub struct ProcessIdentity {
    pub pid: u32,
    pub start_ticks: u64,
    pub boot_id: String,
    pub executable_sha256: String,
}

fn proc_start_ticks(bytes: &[u8], expected_pid: u32) -> Result<u64> {
    let s = std::str::from_utf8(bytes)?;
    let opening = s.find(" (").context("proc stat opening")?;
    ensure!(
        s[..opening].parse::<u32>()? == expected_pid,
        "proc stat PID"
    );
    // comm may itself contain spaces and parentheses.
    let closing = s.rfind(") ").context("proc stat closing")?;
    let fields: Vec<&str> = s[closing + 2..].split_whitespace().collect();
    // fields[0] is field 3 (state); starttime is field 22.
    let start = fields.get(19).context("proc stat starttime")?.parse()?;
    Ok(start)
}

impl ProcessIdentity {
    pub fn capture(pid: u32, expected_executable: &str) -> Result<Self> {
        ensure!(
            cfg!(target_os = "linux"),
            "process-exit proof requires Linux"
        );
        ensure!(
            pid > 0 && pid != std::process::id(),
            "external server PID required"
        );
        let stat = PathBuf::from(format!("/proc/{pid}/stat"));
        let start_ticks = proc_start_ticks(&bounded(&stat, 64 * 1024)?, pid)?;
        let executable_sha256 = file_sha(&PathBuf::from(format!("/proc/{pid}/exe")))?;
        ensure!(
            executable_sha256 == expected_executable,
            "running server executable mismatch"
        );
        let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")?
            .trim()
            .to_owned();
        ensure!(!boot_id.is_empty(), "empty boot identity");
        let identity = Self {
            pid,
            start_ticks,
            boot_id,
            executable_sha256,
        };
        identity.check_live()?;
        Ok(identity)
    }

    pub fn check_live(&self) -> Result<()> {
        ensure!(
            fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim() == self.boot_id,
            "boot identity changed"
        );
        let bytes = bounded(
            &PathBuf::from(format!("/proc/{}/stat", self.pid)),
            64 * 1024,
        )?;
        ensure!(
            proc_start_ticks(&bytes, self.pid)? == self.start_ticks,
            "server PID identity changed"
        );
        Ok(())
    }

    pub async fn prove_absent(&self) -> Result<Value> {
        let deadline = Instant::now() + EXIT_BUDGET;
        let path = PathBuf::from(format!("/proc/{}", self.pid));
        loop {
            ensure!(
                fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim() == self.boot_id,
                "boot identity changed during exit proof"
            );
            match fs::symlink_metadata(&path) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(json!({
                        "method": "linux-proc-pid-directory-absent",
                        "pid": self.pid,
                        "previous_start_ticks": self.start_ticks,
                        "boot_id": self.boot_id,
                        "previous_executable_sha256": self.executable_sha256,
                        "old_process_absent": true,
                        "zombie_counted_as_exit": false
                    }));
                }
                Err(error) => return Err(error.into()),
                Ok(_) => {
                    // PID reuse is not accepted as a shortcut. Fail closed.
                    self.check_live()?;
                }
            }
            ensure!(Instant::now() < deadline, "old process did not exit");
            sleep(Duration::from_millis(20)).await;
        }
    }
}

pub struct Launch {
    pub instance: String,
    pub pid: u32,
    pub manifest: PathBuf,
    pub attestation: PathBuf,
    pub start_utc: String,
    pub process: ProcessIdentity,
    pub initial_manifest: Vec<u8>,
    pub initial_observation: Value,
    pub startup: Value,
    pub binding_sha256: String,
}

fn evidence_path(raw: &str, loaded: &Loaded) -> Result<PathBuf> {
    let path = Path::new(raw).canonicalize()?;
    ensure!(path.is_file(), "launch evidence is not a file");
    ensure!(
        path != loaded.policy_path
            && !loaded.paths.values().any(|p| p == &path)
            && !loaded.bundle.paths.values().any(|p| p == &path),
        "mutable launch evidence aliases a bound artifact"
    );
    Ok(path)
}

fn ready_observation(v: &Value, instance: &str, pid: u32, ready: bool) -> Result<u64> {
    ensure!(
        v["instance"] == instance
            && v["pid"] == pid
            && v["lifecycle"] == "ready"
            && v["worker_alive"] == true
            && v["ready"] == ready
            && v["startup_output_sha256"]
                .as_str()
                .is_some_and(digest_valid),
        "observation identity/lifecycle"
    );
    uint(v, "forward_calls")
}

pub(crate) fn base_address(base: &str) -> Result<SocketAddr> {
    let uri: hyper::Uri = base.parse()?;
    ensure!(
        uri.scheme_str() == Some("http")
            && (uri.path().is_empty() || uri.path() == "/")
            && uri.query().is_none(),
        "base URL must be plain HTTP without path/query"
    );
    let authority = uri.authority().context("HTTP authority")?;
    ensure!(!authority.as_str().contains('@'), "HTTP userinfo forbidden");
    let host = uri
        .host()
        .context("HTTP host")?
        .trim_start_matches('[')
        .trim_end_matches(']');
    let address = SocketAddr::new(host.parse()?, uri.port_u16().unwrap_or(80));
    ensure!(
        address.ip().is_loopback() && address.port() != 0,
        "numeric loopback HTTP endpoint required"
    );
    Ok(address)
}

async fn bind_initial(
    loaded: &Loaded,
    ledger: &mut Ledger,
    orch: &mut Orchestration,
    control: &Control,
    base: &str,
    input: &Path,
    out: &Path,
) -> Result<Launch> {
    let ready = orch.receive().await?;
    ensure!(ready["kind"] == "orch_ready", "expected initial orch_ready");
    let fields = &ready["fields"];
    let instance = text(fields, "instance")?.to_owned();
    ensure!(label_valid(&instance), "invalid instance");
    let pid = u32::try_from(uint(fields, "pid")?)?;
    let start_utc = text(fields, "start_utc")?.to_owned();
    ensure!(
        !start_utc.is_empty() && start_utc.len() <= 80,
        "invalid start identity"
    );
    let manifest = evidence_path(text(fields, "manifest")?, loaded)?;
    let attestation = evidence_path(text(fields, "attestation")?, loaded)?;
    let input = input.canonicalize()?;
    ensure!(
        manifest != attestation && manifest != input && attestation != input,
        "launch evidence aliases another input"
    );
    let process = ProcessIdentity::capture(pid, &loaded.artifacts["serve_m3"].sha256)?;
    ledger.append(
        policy::LAUNCH,
        "authenticated-orchestration-ready",
        true,
        json!({
            "authenticated_envelope": ready,
            "instance": instance,
            "pid": pid,
            "process_start_ticks": process.start_ticks,
            "boot_id": process.boot_id,
            "running_binary_sha256": process.executable_sha256
        }),
        true,
    )?;

    let challenge = random_label()?;
    let attested = control.call("attest", Some(&challenge)).await?;
    ensure!(
        ready_observation(&attested, &instance, pid, true)? == 0,
        "launch is not fresh"
    );
    let att = &attested["attestation"];
    let saved_bytes = bounded(&attestation, JSON_LIMIT)?;
    let saved: Value = serde_json::from_slice(&saved_bytes)?;
    ensure!(
        saved["schema"] == "rustv1.server_launch_attestation.v1"
            && saved["instance_id"] == instance
            && saved["pid"] == pid
            && saved["start_utc"] == start_utc
            && saved["settings_sha256"] == loaded.policy_sha256
            && saved["binary_sha256"] == loaded.artifacts["serve_m3"].sha256
            && saved["audit_sha256"] == loaded.artifacts["audit"].sha256
            && saved["startup_check_sha256"] == loaded.artifacts["startup_check"].sha256
            && saved["execution_policy"] == BATCHED_EXECUTION_POLICY
            && saved["http_contract_id"] == BATCHED_HTTP_CONTRACT
            && saved["batching"] == serde_json::to_value(&loaded.bundle.shapes.batching)?,
        "saved launch attestation binding"
    );
    for key in [
        "rustc_version",
        "nvcc_version",
        "cuda_compute_cap",
        "target_features",
    ] {
        ensure!(
            !text(&saved["build_identity"], key)?.is_empty(),
            "empty server build identity"
        );
    }
    let mut reconstructed = att.clone();
    let object = reconstructed
        .as_object_mut()
        .context("runtime attestation object")?;
    for key in ["http_address", "control_address", "startup", "capabilities"] {
        ensure!(
            object.remove(key).is_some(),
            "missing runtime attestation field"
        );
    }
    ensure!(
        reconstructed == saved && serde_json::to_vec_pretty(&saved)? == saved_bytes,
        "runtime/saved attestation mismatch"
    );
    let expected_http = base_address(base)?;
    let actual_http: SocketAddr = text(att, "http_address")?.parse()?;
    let actual_control: SocketAddr = text(att, "control_address")?.parse()?;
    ensure!(
        actual_control == control.address
            && actual_http.port() == expected_http.port()
            && (actual_http.ip() == expected_http.ip() || actual_http.ip().is_unspecified()),
        "attested endpoint mismatch"
    );
    let check = startup_check(loaded)?;
    ensure!(
        att["startup"]
            == json!({
                "completed": true,
                "comparison": "exact_f32_bits",
                "output_f32le_sha256": startup_digest(&check)
            })
            && attested["startup_output_sha256"] == startup_digest(&check),
        "startup output binding"
    );

    let initial_manifest = bounded(&manifest, RECORD_LIMIT)?;
    let startup = check_startup(&initial_manifest, loaded, &sha(&saved_bytes), true)?;
    let initial_observation = control.call("observe", None).await?;
    let mut plain = attested.clone();
    plain
        .as_object_mut()
        .context("observation object")?
        .remove("attestation");
    ensure!(
        initial_observation == plain && bounded(&manifest, RECORD_LIMIT)? == initial_manifest,
        "activity during launch binding"
    );
    process.check_live()?;
    let binding_sha256 = write_json(
        &out.join("initial_launch_binding.json"),
        &json!({
            "challenge": challenge,
            "authenticated_ready": ready,
            "authenticated_observation": attested,
            "policy_sha256": loaded.policy_sha256,
            "attestation_sha256": sha(&saved_bytes),
            "startup": startup,
            "process_start_ticks": process.start_ticks,
            "boot_id": process.boot_id
        }),
    )?;
    write_new(
        &out.join("initial_executor_manifest.jsonl"),
        &initial_manifest,
    )?;
    ledger.append(
        policy::LAUNCH,
        "fresh-launch-attestation",
        true,
        json!({"launch_binding_sha256": binding_sha256}),
        true,
    )?;
    ledger.append(
        policy::LAUNCH,
        "startup-manifest-reconciliation",
        true,
        startup.clone(),
        true,
    )?;

    write_json(
        &out.join("http_handoff.json"),
        &json!({
            "schema": "rustv1.accept_m4_handoff.v1",
            "model_root": loaded.bundle.model_dir,
            "prepass_input_path": input,
            "prepass_input_sha256": loaded.bundle.input_sha256,
            "live_manifest_path": manifest,
            "instance": instance,
            "pid": pid
        }),
    )?;
    Ok(Launch {
        instance,
        pid,
        manifest,
        attestation,
        start_utc,
        process,
        initial_manifest,
        initial_observation,
        startup,
        binding_sha256,
    })
}

fn verify_http_outputs(out: &Path, manifest: &Value) -> Result<()> {
    let hashes = manifest["output_sha256"]
        .as_object()
        .context("HTTP output hashes")?;
    let mut actual = BTreeMap::new();
    for entry in fs::read_dir(out)? {
        let entry = entry?;
        ensure!(entry.file_type()?.is_file(), "unexpected HTTP output entry");
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 output name"))?;
        if name != "acceptance_http_manifest.json" {
            actual.insert(name, file_sha(&entry.path())?);
        }
    }
    ensure!(actual.len() == hashes.len(), "HTTP output coverage");
    for (name, digest) in hashes {
        let p = Path::new(name);
        ensure!(
            p.components().count() == 1
                && matches!(p.components().next(), Some(Component::Normal(_)))
                && digest.as_str().is_some_and(digest_valid)
                && actual.get(name).map(String::as_str) == digest.as_str(),
            "HTTP output binding"
        );
    }
    Ok(())
}

async fn http_phase(
    loaded: &Loaded,
    ledger: &mut Ledger,
    launch: &Launch,
    base: &str,
    out: &Path,
) -> Result<Value> {
    let http_out = out.join("http");
    let policy_path = loaded.policy_path.clone();
    let base = base.to_owned();
    let control = text(&loaded.policy, "launch_control_address")?.to_owned();
    let handoff = out.join("http_handoff.json");
    let destination = http_out.clone();

    // run() owns its Tokio runtime. It must not run directly inside this
    // coordinator's runtime. The unchanged producer prints each HTTP case.
    let digest = tokio::task::spawn_blocking(move || {
        acceptance_http::run(&policy_path, &base, &control, &destination, &handoff)
    })
    .await
    .context("HTTP phase task failed")??;

    let path = http_out.join("acceptance_http_manifest.json");
    ensure!(file_sha(&path)? == digest, "HTTP returned manifest digest");
    let manifest = value(&path)?;
    ensure!(
        manifest["schema"] == "rustv1.accept_m4_http_manifest.v1"
            && manifest["acceptance_phase"] == policy::HTTP
            && manifest["policy_sha256"] == loaded.policy_sha256
            && manifest["prepass_input_sha256"] == loaded.bundle.input_sha256
            && manifest["instance"] == launch.instance
            && manifest["pid"] == launch.pid
            && manifest["pass"] == true
            && manifest["m4_accepted"] == false
            && manifest["server_left_quiesced"] == true
            && manifest["pending_required_cases_in_scope"] == json!([])
            && manifest["exceptions"] == json!([])
            && manifest["unmatched_physical_records"] == 0
            && manifest["initial_manifest_sha256"] == sha(&launch.initial_manifest)
            && manifest["initial_observation"] == launch.initial_observation,
        "HTTP phase completion binding"
    );
    verify_http_outputs(&http_out, &manifest)?;
    ensure!(
        bounded(
            &http_out.join("initial_executor_manifest.jsonl"),
            RECORD_LIMIT
        )? == launch.initial_manifest,
        "HTTP initial snapshot differs"
    );
    let snapshot = bounded(
        &http_out.join("executor_manifest_snapshot.jsonl"),
        RECORD_LIMIT,
    )?;
    ensure!(
        snapshot.starts_with(&launch.initial_manifest)
            && bounded(&launch.manifest, RECORD_LIMIT)? == snapshot
            && manifest["final_manifest_sha256"] == sha(&snapshot)
            && manifest["final_manifest_bytes"] == snapshot.len(),
        "HTTP final live-manifest reconciliation"
    );
    let records_bytes = bounded(&http_out.join("accept_records.jsonl"), RECORD_LIMIT)?;
    let records = jsonl(&records_bytes)?;
    ledger.import(policy::HTTP, &records, &sha(&records_bytes), false)?;
    loaded.recheck()?;
    Ok(manifest)
}

/// Existing supervisor drain exchange, followed by an independent process
/// absence check. No replacement launch is requested by this operation.
pub async fn supervised_drain(
    loaded: &Loaded,
    launch: &Launch,
    control: &Control,
    orch: &mut Orchestration,
    expected_observation: &Value,
    expected_manifest: &[u8],
) -> Result<(Value, Value)> {
    launch.process.check_live()?;
    ensure!(
        control.call("observe", None).await? == *expected_observation
            && bounded(&launch.manifest, RECORD_LIMIT)? == expected_manifest,
        "unaccounted activity before drain"
    );
    loaded.recheck()?;
    let request = orch
        .send("orch_request_drain", json!({"instance": launch.instance}))
        .await?;
    let exit = orch.receive().await?;
    ensure!(
        exit["kind"] == "orch_exit"
            && exit["fields"]["instance"] == launch.instance
            && exit["fields"]["exit_status"] == 0,
        "supervised graceful exit acknowledgement"
    );
    // Some supervisors additionally send PID. If present it must match.
    if let Some(pid) = exit["fields"].get("pid") {
        ensure!(*pid == launch.pid, "exit acknowledgement PID mismatch");
    }
    let absence = launch.process.prove_absent().await?;
    ensure!(
        bounded(&launch.manifest, RECORD_LIMIT)? == expected_manifest,
        "unreconciled execution during shutdown"
    );
    orch.sync()?;
    Ok((
        json!({
            "request": request,
            "acknowledgement": exit,
            "instance": launch.instance,
            "pid": launch.pid
        }),
        absence,
    ))
}

/// State passed to the remaining scheduler and lifecycle implementations.
/// It contains policy secrets indirectly; do not serialize this struct.
pub struct CoreState {
    pub loaded: Loaded,
    pub ledger: Ledger,
    pub orchestration: Orchestration,
    pub control: Control,
    pub initial_launch: Launch,
    pub initial_exit: Value,
    pub independent_join: Value,
    pub out: PathBuf,
    pub base_url: String,
}

/// Preserve the runtime that owns the orchestration socket registration.
pub struct Continuation {
    pub runtime: tokio::runtime::Runtime,
    pub state: CoreState,
}

#[cfg(feature = "cuda")]
pub fn run_prefix(
    model: &Path,
    input: &Path,
    expected_input_sha256: &str,
    policy_path: &Path,
    base: &str,
    out: &Path,
) -> Result<Continuation> {
    ensure!(cfg!(target_os = "linux"), "acceptance core requires Linux");
    base_address(base)?;
    let loaded = policy::load(model, input, expected_input_sha256, policy_path)?;
    private_directory(out)?;
    let out = out.canonicalize()?;
    let input = input.canonicalize()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(prefix(loaded, &input, base, &out));
    match result {
        Ok(state) => Ok(Continuation { runtime, state }),
        Err(error) => {
            runtime.shutdown_timeout(Duration::from_secs(1));
            Err(error)
        }
    }
}

#[cfg(feature = "cuda")]
async fn prefix(loaded: Loaded, input: &Path, base: &str, out: &Path) -> Result<CoreState> {
    let mut ledger = Ledger::create(out, &loaded)?;
    let evidence = value(&loaded.paths["batch_in_process_evidence"])?;
    let source = text(&evidence["records"], "sha256")?;
    ledger.import(policy::PREPASS, &loaded.prepass_records, source, true)?;
    ledger.append(
        policy::POLICY,
        "finalized-policy-bindings",
        true,
        loaded.binding_summary(),
        true,
    )?;
    ledger.append(
        policy::POLICY,
        "prepass-evidence-consumed",
        true,
        json!({
            "evidence_sha256": loaded.artifacts["batch_in_process_evidence"].sha256,
            "source_records_sha256": source,
            "records": loaded.prepass_records.len(),
            "consumer": "accept_m4_policy::load"
        }),
        true,
    )?;

    let control = Control::from_policy(&loaded)?;
    let mut orchestration =
        Orchestration::connect(&loaded, &out.join("orchestration.jsonl")).await?;
    let launch = bind_initial(
        &loaded,
        &mut ledger,
        &mut orchestration,
        &control,
        base,
        input,
        out,
    )
    .await?;
    let http = http_phase(&loaded, &mut ledger, &launch, base, out).await?;
    let final_bytes = bounded(&launch.manifest, RECORD_LIMIT)?;
    ensure!(
        http["final_manifest_sha256"] == sha(&final_bytes),
        "HTTP manifest changed before shutdown"
    );
    let (shutdown, absence) = supervised_drain(
        &loaded,
        &launch,
        &control,
        &mut orchestration,
        &http["final_observation"],
        &final_bytes,
    )
    .await?;
    ledger.append(
        policy::EXIT,
        "authenticated-supervised-shutdown",
        true,
        shutdown.clone(),
        true,
    )?;
    ledger.append(
        policy::EXIT,
        "old-process-exit-proven",
        true,
        absence.clone(),
        true,
    )?;
    let initial_exit = json!({
        "shutdown": shutdown,
        "process_exit": absence,
        "final_manifest_sha256": sha(&final_bytes)
    });
    let exit_digest = write_json(&out.join("initial_exit.json"), &initial_exit)?;
    write_new(
        &out.join("batched_final_executor_manifest.jsonl"),
        &final_bytes,
    )?;

    // The only local CUDA launch in this prefix occurs after the authenticated
    // exit exchange AND the independent /proc absence proof above.
    let independent_join = independent::run(&loaded, &mut ledger, out, &exit_digest).await?;

    loaded.recheck()?;
    orchestration.sync()?;
    ensure!(
        !ledger.pending().is_empty(),
        "prefix unexpectedly completed the global ledger"
    );
    Ok(CoreState {
        loaded,
        ledger,
        orchestration,
        control,
        initial_launch: launch,
        initial_exit,
        independent_join,
        out: out.to_owned(),
        base_url: base.to_owned(),
    })
}

#[cfg(feature = "cuda")]
mod independent {
    use super::*;
    use crate::contract::compare;
    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::{client::conn::http1, header::HeaderMap, Request};
    use hyper_util::rt::TokioIo;
    use rustv1::{
        http_serve::{
            App, Backend, BoundedListener, Code, Config, ControlCommand, Lifecycle, Logger,
            PinnedTokenizer, Service, MODEL_ID,
        },
        model::IndependentExecutor,
    };
    use std::sync::Arc;
    use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle};

    fn service_result<T>(result: std::result::Result<T, Code>) -> Result<T> {
        result.map_err(|code| anyhow::anyhow!("Service error: {code:?}"))
    }

    struct CudaBackend {
        executor: IndependentExecutor,
    }

    impl Backend for CudaBackend {
        fn forward(&mut self, batches: &[Vec<u32>]) -> std::result::Result<Vec<Vec<f32>>, Code> {
            self.executor
                .forward_serving(batches)
                .map_err(|_| Code::InferenceFailed)
        }

        // This phase never arms policy drift. Refuse it rather than mutating
        // the coordinator's process-global CUDA policy.
        fn inject_policy_drift(&mut self) -> std::result::Result<(), Code> {
            Err(Code::InvalidRequest)
        }

        fn restore_policy(&mut self) -> std::result::Result<(), Code> {
            Err(Code::InvalidRequest)
        }
    }

    struct AbortConnection(JoinHandle<()>);

    impl Drop for AbortConnection {
        fn drop(&mut self) {
            self.0.abort();
        }
    }

    fn header<'a>(headers: &'a HeaderMap, name: &str) -> Result<&'a str> {
        let mut values = headers.get_all(name).iter();
        let value = values.next().context("missing response header")?;
        ensure!(values.next().is_none(), "duplicate response header");
        Ok(value.to_str()?)
    }

    struct Reply {
        status: u16,
        request_id: u64,
        bytes: Vec<u8>,
    }

    struct Client {
        address: SocketAddr,
        instance: String,
        connect: Duration,
        headers: Duration,
        body: Duration,
    }

    impl Client {
        async fn send(&self, case: &str, bytes: &[u8]) -> Result<Reply> {
            ensure!(label_valid(case), "invalid case label");
            let stream = timeout(self.connect, TcpStream::connect(self.address))
                .await
                .context("independent HTTP connect deadline")??;
            let mut builder = http1::Builder::new();
            builder.max_headers(64);
            builder.max_buf_size(64 * 1024);
            let (mut sender, connection) =
                timeout(self.connect, builder.handshake(TokioIo::new(stream)))
                    .await
                    .context("independent HTTP handshake deadline")??;
            let _connection = AbortConnection(tokio::spawn(async move {
                let _ = connection.await;
            }));
            let request = Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("host", self.address.to_string())
                .header("content-type", "application/json")
                .header("accept", "application/json")
                .header("accept-encoding", "identity")
                .header("connection", "close")
                .header("x-acceptance-case-id", case)
                .body(Full::new(Bytes::copy_from_slice(bytes)))?;
            let response = timeout(self.headers, sender.send_request(request))
                .await
                .context("independent HTTP headers deadline")??;
            ensure!(
                header(response.headers(), "x-rv1-instance-id")? == self.instance
                    && header(response.headers(), "x-acceptance-case-id")? == case,
                "independent HTTP correlation"
            );
            let media = header(response.headers(), "content-type")?;
            ensure!(
                media
                    .split(';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .eq_ignore_ascii_case("application/json")
                    && !response.headers().contains_key("content-encoding"),
                "independent HTTP response encoding"
            );
            let request_text = header(response.headers(), "x-rv1-request-id")?;
            let request_id: u64 = request_text.parse()?;
            ensure!(
                request_id > 0 && request_text == request_id.to_string(),
                "noncanonical request ID"
            );
            let status = response.status().as_u16();
            let mut body = response.into_body();
            let bytes = timeout(self.body, async {
                let mut bytes = Vec::new();
                while let Some(frame) = body.frame().await {
                    let frame = frame?;
                    let data = frame.data_ref().context("unexpected HTTP trailers")?;
                    ensure!(
                        data.len() <= 1_048_576usize.saturating_sub(bytes.len()),
                        "independent HTTP response exceeds bound"
                    );
                    bytes.extend_from_slice(data);
                }
                Ok::<_, anyhow::Error>(bytes)
            })
            .await
            .context("independent HTTP body deadline")??;
            Ok(Reply {
                status,
                request_id,
                bytes,
            })
        }
    }

    enum Expected {
        Success(Vec<usize>),
        Rejection(&'static str),
    }

    struct Case {
        id: String,
        body: Value,
        expected: Expected,
    }

    fn positive(loaded: &Loaded, id: String, indices: &[usize]) -> Case {
        Case {
            id,
            body: json!({
                "model": MODEL_ID,
                "input": indices.iter()
                    .map(|i| loaded.bundle.fixtures[*i].text.as_str())
                    .collect::<Vec<_>>(),
                "encoding_format": "float"
            }),
            expected: Expected::Success(indices.to_vec()),
        }
    }

    fn plan(loaded: &Loaded) -> Result<Vec<Case>> {
        let mut cases = Vec::new();
        for mapping in &loaded.mappings[..41] {
            for (position, indices) in mapping.requests.iter().enumerate() {
                cases.push(positive(
                    loaded,
                    format!("mapped-{}-{position}", mapping.m2_group_id),
                    indices,
                ));
            }
        }
        let short = loaded.bundle.fixtures[0].text.as_str();
        let pad = loaded.policy["pad_text"].as_str();
        let rejects = [
            ("invalid-empty-batch", json!([]), "empty_batch"),
            ("invalid-last-empty", json!([short, ""]), "empty_sequence"),
            (
                if pad.is_some() {
                    "invalid-last-pad"
                } else {
                    "invalid-last-empty-pad-substitute"
                },
                json!([short, pad.unwrap_or("")]),
                if pad.is_some() {
                    "pad_token"
                } else {
                    "empty_sequence"
                },
            ),
            (
                "invalid-last-overlength",
                json!([short, text(&loaded.policy, "over_length_text")?]),
                "over_length",
            ),
            ("invalid-token-array", json!([1, 2]), "invalid_request"),
        ];
        for (offset, (id, input, code)) in rejects.into_iter().enumerate() {
            cases.push(Case {
                id: id.into(),
                body: json!({"model": MODEL_ID, "input": input}),
                expected: Expected::Rejection(code),
            });
            let mapping = &loaded.mappings[41 + offset];
            cases.push(positive(
                loaded,
                format!("mapped-{}-0", mapping.m2_group_id),
                &mapping.requests[0],
            ));
        }
        let ids: Vec<String> = cases.iter().map(|c| c.id.clone()).collect();
        ensure!(
            ids == policy::mapping_case_ids(&loaded.mappings, pad.is_some()),
            "independent plan differs from required mapping order"
        );
        Ok(cases)
    }

    fn error_body(code: &str) -> Result<Value> {
        let (message, param) = match code {
            "empty_batch" => ("Input batch is empty", Some("input")),
            "empty_sequence" => ("Input sequence is empty", Some("input")),
            "pad_token" => ("Input contains a padding token", Some("input")),
            "over_length" => ("Input exceeds the sequence limit", Some("input")),
            "invalid_request" => ("Invalid request", None),
            _ => bail!("unplanned independent rejection"),
        };
        Ok(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": code
            }
        }))
    }

    fn outputs(reply: &Reply, indices: &[usize], loaded: &Loaded) -> Result<Vec<Vec<f32>>> {
        ensure!(reply.status == 200, "independent success HTTP status");
        let v: Value = serde_json::from_slice(&reply.bytes)?;
        closed(&v, &["object", "data", "model", "usage"])?;
        let total: usize = indices
            .iter()
            .map(|i| loaded.bundle.fixtures[*i].ids.len())
            .sum();
        ensure!(
            v["object"] == "list"
                && v["model"] == MODEL_ID
                && v["usage"] == json!({"prompt_tokens": total, "total_tokens": total}),
            "independent response object/model/usage"
        );
        let data = v["data"].as_array().context("embedding rows")?;
        ensure!(
            data.len() == indices.len(),
            "independent response cardinality"
        );
        let mut rows = Vec::new();
        for (position, (item, fixture)) in data.iter().zip(indices).enumerate() {
            closed(item, &["object", "index", "embedding"])?;
            ensure!(
                item["object"] == "embedding"
                    && item["index"] == position
                    && item["embedding"].is_array(),
                "independent response row identity"
            );
            let row: Vec<f32> = serde_json::from_value(item["embedding"].clone())?;
            crate::contract::exact(
                std::slice::from_ref(&row),
                std::slice::from_ref(&loaded.bundle.singleton[*fixture]),
            )?;
            compare(&row, &loaded.bundle.reference[*fixture])?;
            rows.push(row);
        }
        Ok(rows)
    }

    async fn observe(app: &App) -> Result<Value> {
        Ok(serde_json::to_value(service_result(
            app.control(ControlCommand::Observe).await,
        )?)?)
    }

    struct Runner<'a> {
        loaded: &'a Loaded,
        app: App,
        client: Client,
        out: PathBuf,
        http_out: PathBuf,
        manifest: PathBuf,
        prefix: Vec<u8>,
        observation: Value,
        next_forward: u64,
        last_request: u64,
    }

    impl Runner<'_> {
        async fn one(&mut self, case: &Case) -> Result<Value> {
            let before = observe(&self.app).await?;
            let before_calls =
                ready_observation(&before, &self.client.instance, std::process::id(), true)?;
            ensure!(
                before == self.observation && bounded(&self.manifest, RECORD_LIMIT)? == self.prefix,
                "unaccounted independent activity"
            );
            let before_len = self.prefix.len();

            // Replay the exact request bytes previously used by the unchanged
            // batched HTTP producer, after validating their planned semantics.
            let request = bounded(
                &self.http_out.join(format!("{}.request.bin", case.id)),
                1_048_576,
            )?;
            ensure!(
                serde_json::from_slice::<Value>(&request)? == case.body
                    && request == serde_json::to_vec(&case.body)?,
                "mapping request changed between execution modes"
            );
            write_new(&self.out.join(format!("{}.request.bin", case.id)), &request)?;
            let reply = self.client.send(&case.id, &request).await?;
            write_new(
                &self.out.join(format!("{}.response.bin", case.id)),
                &reply.bytes,
            )?;
            ensure!(
                reply.request_id > self.last_request,
                "independent request ID reuse/reordering"
            );

            let after = observe(&self.app).await?;
            let after_calls =
                ready_observation(&after, &self.client.instance, std::process::id(), true)?;
            let bytes = bounded(&self.manifest, RECORD_LIMIT)?;
            ensure!(
                bytes.starts_with(&self.prefix),
                "independent manifest rewritten"
            );

            let execution = match &case.expected {
                Expected::Rejection(code) => {
                    ensure!(
                        reply.status == 400
                            && serde_json::from_slice::<Value>(&reply.bytes)? == error_body(code)?
                            && after == before
                            && bytes == self.prefix,
                        "independent rejection contract or zero-execution failure"
                    );
                    json!({
                        "physical_forwards": 0,
                        "logical_backend_request_delta": 0,
                        "observations_identical": true,
                        "manifest_bytes_identical": true,
                        "code": code,
                        "pad_token_coverage": *code == "pad_token",
                        "oov_token_coverage": false
                    })
                }
                Expected::Success(indices) => {
                    ensure!(
                        after_calls == before_calls.checked_add(1).context("logical overflow")?,
                        "independent logical request delta"
                    );
                    let mut expected_observation = before.clone();
                    expected_observation["forward_calls"] = json!(after_calls);
                    ensure!(
                        after == expected_observation,
                        "independent observation changes"
                    );
                    let output = outputs(&reply, indices, self.loaded)?;
                    let tokens: Vec<Vec<u32>> = indices
                        .iter()
                        .map(|i| self.loaded.bundle.fixtures[*i].ids.clone())
                        .collect();
                    let hashes: Vec<String> = output.iter().map(|r| sha(&row_bytes(r))).collect();
                    let interval = jsonl(&bytes[before_len..])?;
                    let mut rows =
                        serving_pairs(&interval, self.next_forward, &tokens, &hashes, self.loaded)?;
                    for (position, (row, fixture)) in rows.iter_mut().zip(indices).enumerate() {
                        row["fixture_index"] = json!(fixture);
                        row["fixture_id"] = json!(self.loaded.bundle.fixtures[*fixture].id);
                        row["response_row"] = json!(position);
                        row["singleton_exact_f32_bits"] = json!(true);
                        row["independent_reference_comparison"] =
                            compare(&output[position], &self.loaded.bundle.reference[*fixture])?;
                    }
                    self.next_forward = self
                        .next_forward
                        .checked_add(u64::try_from(indices.len())?)
                        .context("forward counter overflow")?;
                    json!({
                        "route": "IndependentExecutor::forward_serving",
                        "physical_forwards": indices.len(),
                        "logical_backend_request_delta": 1,
                        "rows": rows,
                        "association_source":
                            "serialized-http-request-observation-and-exclusive-manifest-interval",
                        "executor_emitted_server_request_ids": false
                    })
                }
            };
            let details = json!({
                "instance": self.client.instance,
                "pid": std::process::id(),
                "server_request_id": reply.request_id,
                "request_sha256": sha(&request),
                "request_bytes_identical_to_batched_phase": true,
                "response_sha256": sha(&reply.bytes),
                "http_status": reply.status,
                "observation_before": before,
                "observation_after": after,
                "manifest_interval": {
                    "start_byte": before_len,
                    "end_byte": bytes.len(),
                    "prefix_before_sha256": sha(&self.prefix),
                    "prefix_after_sha256": sha(&bytes),
                    "interval_sha256": sha(&bytes[before_len..])
                },
                "execution": execution
            });
            self.prefix = bytes;
            self.observation = after;
            self.last_request = reply.request_id;
            Ok(details)
        }
    }

    pub(super) async fn run(
        loaded: &Loaded,
        ledger: &mut Ledger,
        root: &Path,
        old_exit_sha256: &str,
    ) -> Result<Value> {
        ensure!(digest_valid(old_exit_sha256), "old exit evidence digest");
        loaded.recheck()?;
        let out = root.join("independent");
        private_directory(&out)?;
        let manifest = out.join("executor_manifest.jsonl");
        let instance = format!("independent-{}", random_label()?);
        let mut config = Config::production(64, 131072, 1024);
        config.batching = None;
        config.instance = Some(instance.clone());
        service_result(config.validate())?;
        let check = startup_check(loaded)?;
        let expected_startup_digest = startup_digest(&check);
        let startup_count = check.tokens.len();

        let tokenizer = Arc::new(service_result(PinnedTokenizer::load(
            &loaded.paths["tokenizer"],
            &loaded.artifacts["tokenizer"].sha256,
            131072,
        ))?);
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let listener = service_result(BoundedListener::new(listener, &config.transport))?;
        let client = Client {
            address,
            instance: instance.clone(),
            connect: Duration::from_millis(uint(&loaded.policy["limits"], "connect_ms")?),
            headers: Duration::from_millis(uint(&loaded.policy["limits"], "headers_ms")?),
            body: Duration::from_millis(uint(&loaded.policy["limits"], "body_ms")?),
        };
        let launch_digest = write_json(
            &out.join("owner_binding.json"),
            &json!({
                "schema": "rustv1.accept_m4_independent_owner.v1",
                "instance": instance,
                "pid": std::process::id(),
                "policy_sha256": loaded.policy_sha256,
                "driver_binary_sha256": loaded.artifacts["accept_m4"].sha256,
                "startup_check_sha256": loaded.artifacts["startup_check"].sha256,
                "tokenizer_sha256": loaded.artifacts["tokenizer"].sha256,
                "audit_identity": loaded.bundle.audit_identity,
                "previous_process_exit_evidence_sha256": old_exit_sha256,
                "execution_policy": "independent_sequence_v1",
                "batching": null,
                "backend_forward": "IndependentExecutor::forward_serving",
                "http_address": address.to_string()
            }),
        )?;
        let model = loaded.bundle.model_dir.clone();
        let audit = loaded.bundle.audit_identity.clone();
        let executor_manifest = manifest.clone();
        let executor_binding = launch_digest.clone();
        let logger = Logger::new(private_file(&out.join("service_log.jsonl"))?);
        let cases = plan(loaded)?;

        // Only Send inputs cross the thread boundary. IndependentExecutor is
        // constructed and retained on Service's owner thread.
        let mut service = service_result(Service::spawn(
            config,
            tokenizer,
            logger,
            check,
            move || {
                let executor = rustv1::m2_route::open_with_attestation(
                    &model,
                    &executor_manifest,
                    0,
                    &audit,
                    Some(&executor_binding),
                )
                .map_err(|_| Code::InferenceFailed)?;
                if executor.batching().is_some() {
                    return Err(Code::InferenceFailed);
                }
                Ok(CudaBackend { executor })
            },
        ))?;
        let app = service.app.clone();
        let router = app.router();
        let (stop, stopped) = oneshot::channel();
        let mut server = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
        });

        // Every normal error after Service creation flows through cleanup.
        let work: Result<(Vec<u8>, u64, Value)> = async {
            let startup_result: Result<(Vec<u8>, Value)> = async {
                service_result(service.wait_ready().await)?;
                ensure!(
                    service.app.config.batching.is_none(),
                    "independent batching enabled"
                );
                let bytes = bounded(&manifest, RECORD_LIMIT)?;
                let startup = check_startup(&bytes, loaded, &launch_digest, false)?;
                let observation = observe(&app).await?;
                ensure!(
                    ready_observation(&observation, &instance, std::process::id(), true)? == 0
                        && observation["startup_output_sha256"] == expected_startup_digest,
                    "independent startup observation"
                );
                ensure!(
                    bounded(&manifest, RECORD_LIMIT)? == bytes,
                    "activity during independent startup binding"
                );
                ledger.append(
                    policy::INDEPENDENT,
                    "independent-startup-reconciled",
                    true,
                    json!({
                        "owner_binding_sha256": launch_digest,
                        "startup": startup,
                        "observation": observation
                    }),
                    true,
                )?;
                Ok((bytes, observation))
            }
            .await;
            let (bytes, observation) = match startup_result {
                Ok(value) => value,
                Err(error) => {
                    // If append itself failed, the ledger may already be
                    // terminal; preserve the original error either way.
                    let _ = ledger.append(
                        policy::INDEPENDENT,
                        "independent-startup-reconciled",
                        false,
                        json!({"error": format!("{error:#}")}),
                        true,
                    );
                    return Err(error);
                }
            };
            let mut runner = Runner {
                loaded,
                app: app.clone(),
                client,
                out: out.clone(),
                http_out: root.join("http"),
                manifest: manifest.clone(),
                prefix: bytes,
                observation,
                next_forward: u64::try_from(startup_count)?,
                last_request: 0,
            };
            for case in &cases {
                match runner.one(case).await {
                    Ok(details) => {
                        ledger.append(policy::INDEPENDENT, &case.id, true, details, true)?
                    }
                    Err(error) => {
                        let _ = ledger.append(
                            policy::INDEPENDENT,
                            &case.id,
                            false,
                            json!({"error": format!("{error:#}")}),
                            true,
                        );
                        return Err(error);
                    }
                }
            }
            ensure!(
                observe(&app).await? == runner.observation
                    && bounded(&manifest, RECORD_LIMIT)? == runner.prefix,
                "unreconciled final independent work"
            );
            Ok((runner.prefix, runner.next_forward, runner.observation))
        }
        .await;

        // Service::shutdown performs the owner-thread join on success.
        // Drop alone is expressly not treated as a join.
        let shutdown = service_result(service.shutdown().await);
        let _ = stop.send(());
        let server_result = timeout(Duration::from_secs(30), &mut server).await;
        let transport = match server_result {
            Ok(Ok(Ok(()))) => Ok(()),
            other => {
                server.abort();
                let _ = server.await;
                Err(anyhow::anyhow!(
                    "independent HTTP shutdown failed: {other:?}"
                ))
            }
        };

        if let Err(error) = work {
            return Err(error).with_context(|| {
                format!(
                    "independent phase failed; owner_shutdown_ok={}, transport_shutdown_ok={}",
                    shutdown.is_ok(),
                    transport.is_ok()
                )
            });
        }
        let (final_bytes, forwards, final_observation) = work?;
        shutdown?;
        transport?;
        ensure!(
            app.lifecycle() == Lifecycle::Stopped && !app.is_ready(),
            "independent owner not stopped after join"
        );
        ensure!(
            bounded(&manifest, RECORD_LIMIT)? == final_bytes,
            "independent manifest changed during shutdown"
        );
        let records = jsonl(&final_bytes)?;
        ensure!(
            records.len() == 1 + usize::try_from(forwards)? * 2,
            "independent whole-manifest physical count"
        );
        loaded.recheck()?;
        let proof = json!({
            "instance": instance,
            "pid": std::process::id(),
            "owner_binding_sha256": launch_digest,
            "service_shutdown_returned_ok": true,
            "owner_thread_joined": true,
            "http_server_joined": true,
            "lifecycle": "stopped",
            "ready": false,
            "final_pre_shutdown_observation": final_observation,
            "executor_manifest_sha256": sha(&final_bytes),
            "executor_manifest_bytes": final_bytes.len(),
            "physical_forwards_including_startup": forwards,
            "startup_physical_forwards": startup_count,
            "unmatched_physical_records": 0,
            "later_cuda_launch_permitted_only_after_this_join": true
        });
        write_json(&out.join("owner_join.json"), &proof)?;
        ledger.append(
            policy::JOIN,
            "independent-owner-shutdown-and-join",
            true,
            proof.clone(),
            true,
        )?;
        Ok(proof)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn rejected_schema_has_no_success_payload() {
            let body = error_body("invalid_request").unwrap();
            assert!(body.get("data").is_none());
            assert_eq!(body["error"]["param"], Value::Null);
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalization_matches_integer_envelope_contract() {
        assert_eq!(
            canonical_json(&json!({"z": [true, null, 2], "a": "\n\"\\é"})).unwrap(),
            "{\"a\":\"\\n\\\"\\\\é\",\"z\":[true,null,2]}"
        );
        assert!(canonical_json(&json!({"x": 0.5})).is_err());
    }

    #[test]
    fn jsonl_rejects_truncation_and_blank_lines() {
        assert!(jsonl(b"{}\n").is_ok());
        assert!(jsonl(b"{}").is_err());
        assert!(jsonl(b"{}\n\n").is_err());
        assert!(jsonl(b"").is_err());
    }

    #[test]
    fn proc_stat_parser_handles_parentheses_in_comm() {
        let mut fields = vec!["S".to_owned()];
        fields.extend((4..=21).map(|_| "0".to_owned()));
        fields.push("12345".to_owned());
        let bytes = format!("42 (name with ) parentheses) {}\n", fields.join(" "));
        assert_eq!(proc_start_ticks(bytes.as_bytes(), 42).unwrap(), 12345);
        assert!(proc_start_ticks(bytes.as_bytes(), 43).is_err());
    }

    #[test]
    fn endpoint_is_numeric_loopback_only() {
        assert!(base_address("http://127.0.0.1:9700").is_ok());
        assert!(base_address("http://[::1]:9700/").is_ok());
        assert!(base_address("http://localhost:9700").is_err());
        assert!(base_address("http://192.0.2.1:9700").is_err());
        assert!(base_address("http://127.0.0.1:9700/path").is_err());
        assert!(base_address("http://127.0.0.1:9700/?x=1").is_err());
    }
}
