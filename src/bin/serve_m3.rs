//! Policy-bound batched server. The positional CLI and environment names are
//! retained; RV1_M3_POLICY must now identify the materialized M4 policy.
//!
//! Startup remains forward_serving plus exact F32 bits. No changes are made to
//! startup_check_gen, golden_export, model.rs or m2_route.rs signatures.
//!
//! Supervised restart remains external: poison/retirement is terminal, health
//! and observation remain available, and SIGTERM drains or exits nonzero.
//! A replacement process must not overlap the old process's CUDA lifetime.
//!
//! Renewed audit materialization contract used here:
//!   scope.execution_policy = "batched_execution_v2"
//!   scope.maximum_physical_b >= configured max_batch_size
//!   scope.sequence_length_max >= engine.max_len
//! together with accepted status, completed mechanical verification, matching
//! lockfile/feature-tree identities and the existing audit identity convention.
//! The B=1-only v2 artifact cannot satisfy these requirements.

use ring::rand::{SecureRandom, SystemRandom};
use rustv1::{
    http_serve::{
        bind_control, serve_control, Backend, BoundedListener, Code, Config, Logger,
        PinnedTokenizer, Service, StartupCheck, MODEL_ID,
    },
    model::{
        BatchedExecution, EngineConfig, ExecutionRow, BATCHED_EXECUTION_POLICY,
        BATCHED_HTTP_CONTRACT,
    },
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    net::SocketAddr,
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{net::TcpListener, sync::oneshot, time::timeout};

const MAX_JSON: usize = 8 * 1024 * 1024;

struct CudaBackend {
    executor: rustv1::model::IndependentExecutor,
}

impl Backend for CudaBackend {
    fn forward(&mut self, batches: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, Code> {
        self.executor
            .forward_serving(batches)
            .map_err(|_| Code::InferenceFailed)
    }

    fn forward_batch(
        &mut self,
        batches: &[Vec<u32>],
        associations: &[ExecutionRow],
    ) -> Result<Vec<Vec<f32>>, Code> {
        self.executor
            .forward_batch(batches, associations)
            .map_err(|_| Code::InferenceFailed)
    }

    fn inject_policy_drift(&mut self) -> Result<(), Code> {
        candle_core::cuda_backend::set_gemm_reduced_precision_f32(true);
        Ok(())
    }

    fn restore_policy(&mut self) -> Result<(), Code> {
        candle_core::cuda_backend::set_gemm_reduced_precision_f32(false);
        candle_core::cuda_backend::set_gemm_reduced_precision_f16(false);
        candle_core::cuda_backend::set_gemm_reduced_precision_bf16(false);
        Ok(())
    }
}

struct Args {
    model_dir: PathBuf,
    manifest: PathBuf,
    audit_identity: String,
    bind: SocketAddr,
    queue_capacity: usize,
    tokenizer_sha256: String,
    startup_check: PathBuf,
    startup_check_sha256: String,
    control: Option<SocketAddr>,
}

impl Args {
    fn parse() -> Result<Self, Code> {
        let args: Vec<String> = std::env::args().collect();
        if args.len() != 10 {
            return Err(Code::InvalidRequest);
        }
        let control = if args[9] == "-" {
            None
        } else {
            let address: SocketAddr = args[9].parse().map_err(|_| Code::InvalidRequest)?;
            if !address.ip().is_loopback() {
                return Err(Code::InvalidRequest);
            }
            Some(address)
        };
        let queue_capacity = args[5].parse().map_err(|_| Code::InvalidRequest)?;
        if queue_capacity != 64 {
            return Err(Code::InvalidRequest);
        }
        Ok(Self {
            model_dir: PathBuf::from(&args[1]),
            manifest: PathBuf::from(&args[2]),
            audit_identity: args[3].clone(),
            bind: args[4].parse().map_err(|_| Code::InvalidRequest)?,
            queue_capacity,
            tokenizer_sha256: args[6].clone(),
            startup_check: PathBuf::from(&args[7]),
            startup_check_sha256: args[8].clone(),
            control,
        })
    }
}

fn require(condition: bool) -> Result<(), Code> {
    if condition {
        Ok(())
    } else {
        Err(Code::InvalidRequest)
    }
}

fn integer(value: &Value, key: &str) -> Result<usize, Code> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(Code::InvalidRequest)
}

fn digest_valid(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn bounded_read(path: &Path) -> Result<Vec<u8>, Code> {
    let file = fs::File::open(path).map_err(|_| Code::InvalidRequest)?;
    require(file.metadata().map_err(|_| Code::InvalidRequest)?.len() <= MAX_JSON as u64)?;
    let mut bytes = Vec::new();
    file.take(MAX_JSON as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| Code::InvalidRequest)?;
    require(bytes.len() <= MAX_JSON)?;
    Ok(bytes)
}

fn read_value(path: &Path) -> Result<Value, Code> {
    serde_json::from_slice(&bounded_read(path)?).map_err(|_| Code::InvalidRequest)
}

fn file_sha256(path: &Path) -> Result<String, Code> {
    let mut file = fs::File::open(path).map_err(|_| Code::InvalidRequest)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|_| Code::InvalidRequest)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn placeholder(value: &Value) -> bool {
    match value {
        Value::String(s) => [
            "RUNTIME",
            "FILLED_BY_INTEGRATOR",
            "<SHA256:",
            "PENDING_ROUNDTRIP",
        ]
        .iter()
        .any(|needle| s.contains(needle)),
        Value::Array(values) => values.iter().any(placeholder),
        Value::Object(values) => values
            .iter()
            .any(|(key, value)| key.contains("RUNTIME") || placeholder(value)),
        _ => false,
    }
}

fn artifact_path(root: &Path, relative: &str) -> Result<PathBuf, Code> {
    let relative = Path::new(relative);
    require(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
    )?;
    let root = root.canonicalize().map_err(|_| Code::InvalidRequest)?;
    let path = root
        .join(relative)
        .canonicalize()
        .map_err(|_| Code::InvalidRequest)?;
    require(path.starts_with(root) && path.is_file())?;
    Ok(path)
}

fn scheduler_contract() -> Value {
    json!({
        "kind": "fifo_opportunistic_static",
        "single_owner": true,
        "overlapping_forwards": false,
        "max_http_inputs": 8,
        "split_http_request": false,
        "head_bypass": false,
        "dummy_batch_rows": false,
        "sequence_width_rounding": false,
        "wait_origin": "queue_insertion",
        "assembly_deadline": "minimum_of_oldest_batch_wait_and_member_request_deadlines",
        "control_messages_are_batch_boundaries": true,
        "queue_capacity_unit": "logical_requests",
        "scheduler_local_jobs_retain_admission": true,
        "expired_queued_requests_execute": false,
        "running_deadline_expiry": "retire_executor",
        "physical_failure": "fail_all_members_and_poison",
        "oom_retry": false,
        "chunked_prefill": false
    })
}

fn padding_contract() -> Value {
    json!({
        "side": "right",
        "pad_id": 11,
        "user_pad_ids": "reject",
        "physical_b": "real_sequence_count",
        "physical_s": "maximum_valid_length",
        "attention": "bidirectional",
        "mask_dtype": "F32",
        "valid_key_mask": 0,
        "invalid_key_mask": "negative_infinity",
        "mask_applies_to": "all_queries_in_every_layer",
        "padded_queries": "attend_valid_keys_then_discard",
        "position_ids": "zero_based_per_sequence",
        "pooling": "sum_valid_states_divide_by_valid_length",
        "normalization": "per_sequence_l2"
    })
}

fn precision_contract() -> Value {
    json!({
        "weights_and_projections": "BF16",
        "profile": "F",
        "gemm_reduced_precision_f32": false,
        "gemm_reduced_precision_f16": false,
        "gemm_reduced_precision_bf16": false,
        "physical_b1_comparison": "exact_f32_bits",
        "physical_b_gt_1_comparison": {
            "cosine_strictly_greater_than": 0.999,
            "max_abs_at_most": 0.01,
            "compare_directly_against": [
                "batched_candidate", "independent_reference", "frozen_singleton_candidate"
            ]
        },
        "all_outputs": {
            "width": 2048,
            "finite": true,
            "norm_computation": "FP64",
            "l2_norm_abs_error_at_most": 0.00001
        },
        "controlled_identical_batch_repeats": "exact_f32_bits",
        "arbitrary_batch_shape_bit_invariance": false,
        "batch_invariance": "direct_pairwise_numerical_gates"
    })
}

struct Loaded {
    args: Args,
    policy_bytes: Vec<u8>,
    policy: Value,
    config: Config,
    tokenizer: Arc<PinnedTokenizer>,
    check: StartupCheck,
    binary_sha256: String,
}

fn load(args: Args, policy_path: PathBuf) -> Result<Loaded, Code> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        require(
            fs::metadata(&policy_path)
                .map_err(|_| Code::InvalidRequest)?
                .permissions()
                .mode()
                & 0o077
                == 0,
        )?;
    }

    let policy_bytes = bounded_read(&policy_path)?;
    let policy: Value = serde_json::from_slice(&policy_bytes).map_err(|_| Code::InvalidRequest)?;
    require(!placeholder(&policy))?;
    require(
        policy["schema"] == "rustv1.acceptance_policy.v3"
            && policy["contract_id"] == "b200-independent-F-v2"
            && policy["http_contract_id"] == BATCHED_HTTP_CONTRACT
            && policy["model"] == MODEL_ID
            && policy["audit_identity"] == args.audit_identity
            && policy["samples"].as_array().is_some_and(Vec::is_empty),
    )?;
    let expected_control: SocketAddr = policy["launch_control_address"]
        .as_str()
        .ok_or(Code::InvalidRequest)?
        .parse()
        .map_err(|_| Code::InvalidRequest)?;
    require(expected_control.ip().is_loopback() && args.control == Some(expected_control))?;
    require(
        policy["limits"]["admission_capacity"] == 64
            && policy["limits"]["queue_capacity"] == 64
            && args.queue_capacity == 64
            && policy["limits"]["server_request_ms"] == 60_000
            && policy["limits"]["response_bytes"] == 1_048_576,
    )?;
    require(
        policy["artifacts"]["tokenizer"]["sha256"] == args.tokenizer_sha256
            && policy["artifacts"]["startup_check"]["sha256"] == args.startup_check_sha256,
    )?;

    let deployment = &policy["deployment"];
    require(
        deployment["gpu"] == "NVIDIA B200"
            && deployment["route"] == "sm_100 native"
            && deployment["engine"] == "candle 0.9.2"
            && deployment["cudarc"] == "0.19.10"
            && deployment["rust_toolchain"] == "1.99.1"
            && deployment["cuda_toolkit"] == "12.8.1"
            && deployment["profile"] == "F"
            && deployment["execution_policy"] == BATCHED_EXECUTION_POLICY
            && deployment["scheduler"] == scheduler_contract()
            && deployment["padding"] == padding_contract()
            && deployment["precision"] == precision_contract()
            && deployment["startup"]
                == json!({
                    "route": "forward_serving",
                    "physical_b": 1,
                    "physical_s": "valid_length",
                    "comparison": "exact_f32_bits",
                    "existing_startup_artifact_format": true,
                    "launch_attestation_binding_required": true
                }),
    )?;

    let batching: BatchedExecution =
        serde_json::from_value(deployment["batching"].clone()).map_err(|_| Code::InvalidRequest)?;
    batching.validate().map_err(|_| Code::InvalidRequest)?;

    let repo_root = std::env::current_dir().map_err(|_| Code::InvalidRequest)?;
    let policy_root = policy_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let artifacts = policy["artifacts"]
        .as_object()
        .ok_or(Code::InvalidRequest)?;
    let mut paths = BTreeMap::new();
    for (role, artifact) in artifacts {
        let expected = artifact["sha256"].as_str().ok_or(Code::InvalidRequest)?;
        require(digest_valid(expected))?;
        let root = match artifact["root"].as_str() {
            Some("model") => args.model_dir.as_path(),
            Some("repo") => repo_root.as_path(),
            Some("policy") => policy_root,
            _ => return Err(Code::InvalidRequest),
        };
        let path = artifact_path(root, artifact["path"].as_str().ok_or(Code::InvalidRequest)?)?;
        require(file_sha256(&path)? == expected)?;
        paths.insert(role.clone(), path);
    }
    let required = |role: &str| -> Result<&Path, Code> {
        paths
            .get(role)
            .map(PathBuf::as_path)
            .ok_or(Code::InvalidRequest)
    };
    for role in [
        "config",
        "engine",
        "tokenizer",
        "startup_check",
        "serve_m3",
        "cargo_lock",
        "feature_tree",
        "audit",
        "audit_b1_base",
        "fixtures_exec",
        "mappings_v1",
        "batch_cases",
        "batch_goldens",
        "batch_golden_index",
    ] {
        required(role)?;
    }
    for (role, actual) in [
        ("config", args.model_dir.join("config.json")),
        ("engine", args.model_dir.join("engine.json")),
        ("tokenizer", args.model_dir.join("tokenizer.json")),
        ("startup_check", args.startup_check.clone()),
    ] {
        require(required(role)? == actual.canonicalize().map_err(|_| Code::InvalidRequest)?)?;
    }

    let binary_sha256 = file_sha256(&std::env::current_exe().map_err(|_| Code::InvalidRequest)?)?;
    require(policy["artifacts"]["serve_m3"]["sha256"] == binary_sha256)?;
    require(
        file_sha256(&repo_root.join("Cargo.lock"))?
            == policy["artifacts"]["cargo_lock"]["sha256"]
                .as_str()
                .ok_or(Code::InvalidRequest)?,
    )?;

    let mut found_weight = false;
    for entry in fs::read_dir(&args.model_dir).map_err(|_| Code::InvalidRequest)? {
        let entry = entry.map_err(|_| Code::InvalidRequest)?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json") {
            let path = entry
                .path()
                .canonicalize()
                .map_err(|_| Code::InvalidRequest)?;
            require(artifacts.iter().any(|(role, artifact)| {
                role.starts_with("weight:")
                    && artifact["root"] == "model"
                    && paths.get(role) == Some(&path)
            }))?;
            found_weight |= name.ends_with(".safetensors");
        }
    }
    require(found_weight)?;

    let engine: EngineConfig = serde_json::from_slice(&bounded_read(required("engine")?)?)
        .map_err(|_| Code::InvalidRequest)?;
    batching
        .validate_engine(&engine)
        .map_err(|_| Code::InvalidRequest)?;
    let model = read_value(required("config")?)?;
    let vocabulary_size = integer(&model, "vocab_size")?;
    require(
        !engine.causal
            && engine.rope_layout == "split_half"
            && engine.max_len <= integer(&model, "max_position_embeddings")?
            && vocabulary_size == 131072
            && model["pad_token_id"] == 11,
    )?;

    let audit = read_value(required("audit")?)?;
    require(
        !placeholder(&audit)
            && audit["schema"] == "rustv1.cuda_math_source_audit.v1"
            && audit["status"] == "accepted"
            && audit["signed"] == true
            && audit["verification"]["mechanical_region_boundary_verification_completed"] == true
            && audit["contract_id"] == "b200-independent-F-v2"
            && audit["scope"]["execution_policy"] == BATCHED_EXECUTION_POLICY
            && audit["scope"]["precision_profile"] == "F"
            && audit["scope"]["maximum_physical_b"]
                .as_u64()
                .is_some_and(|n| n >= batching.max_batch_size as u64)
            && audit["scope"]["sequence_length_max"]
                .as_u64()
                .is_some_and(|n| n >= engine.max_len as u64)
            && audit["scope"]["physical_sequence_batch_size"] != 1
            && audit["cargo_lock_sha256"] == policy["artifacts"]["cargo_lock"]["sha256"]
            && audit["cargo_tree_features"]["sha256"]
                == policy["artifacts"]["feature_tree"]["sha256"],
    )?;
    let audit_path = policy["artifacts"]["audit"]["path"]
        .as_str()
        .ok_or(Code::InvalidRequest)?;
    let audit_sha = policy["artifacts"]["audit"]["sha256"]
        .as_str()
        .ok_or(Code::InvalidRequest)?;
    require(
        audit["intended_path"] == audit_path
            && args.audit_identity == format!("sha256:{audit_sha}:{audit_path}"),
    )?;
    require(
        policy["artifacts"]["audit"]["sha256"] != policy["artifacts"]["audit_b1_base"]["sha256"],
    )?;

    // These checks bind the retained fixture/mapping objects; they are not
    // substitutes for the M4 numerical or batch-case acceptance driver.
    require(
        policy["fixtures"] == read_value(required("fixtures_exec")?)?
            && policy["mappings"] == read_value(required("mappings_v1")?)?,
    )?;
    require(policy["fixtures"].as_array().is_some_and(|v| v.len() == 22))?;
    let mappings = policy["mappings"].as_array().ok_or(Code::InvalidRequest)?;
    let mut groups = BTreeSet::new();
    for mapping in mappings {
        let group = mapping["m2_group_id"]
            .as_u64()
            .ok_or(Code::InvalidRequest)?;
        require((1..=46).contains(&group) && groups.insert(group))?;
        let requests = mapping["requests"].as_array().ok_or(Code::InvalidRequest)?;
        require(!requests.is_empty())?;
        for request in requests {
            let indices = request.as_array().ok_or(Code::InvalidRequest)?;
            require(
                (1..=8).contains(&indices.len())
                    && indices.iter().all(|v| v.as_u64().is_some_and(|n| n < 22)),
            )?;
        }
    }
    require(groups.len() == 46)?;

    let mut config = Config::production(args.queue_capacity, vocabulary_size, engine.max_len);
    config.batching = Some(batching);
    // These policy fields are client-side deadlines too; retain the established
    // server body/control defaults rather than silently interpreting them as
    // new server settings.
    config.validate()?;
    let tokenizer = Arc::new(PinnedTokenizer::load(
        required("tokenizer")?,
        &args.tokenizer_sha256,
        vocabulary_size,
    )?);
    let check_bytes = bounded_read(required("startup_check")?)?;
    require(format!("{:x}", Sha256::digest(&check_bytes)) == args.startup_check_sha256)?;
    let check = serde_json::from_slice(&check_bytes).map_err(|_| Code::InvalidRequest)?;

    Ok(Loaded {
        args,
        policy_bytes,
        policy,
        config,
        tokenizer,
        check,
        binary_sha256,
    })
}

async fn termination() -> Result<(), Code> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut terminate = signal(SignalKind::terminate()).map_err(|_| Code::NotReady)?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.map_err(|_| Code::NotReady),
            _ = terminate.recv() => Ok(()),
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await.map_err(|_| Code::NotReady)
    }
}

fn start_time() -> Result<String, Code> {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Code::InvalidRequest)?;
    Ok(format!("{}.{:06}Z", d.as_secs(), d.subsec_micros()))
}

fn instance_id() -> Result<String, Code> {
    let mut bytes = [0u8; 16];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| Code::InferenceFailed)?;
    let suffix: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    Ok(format!("{}-{suffix}", std::process::id()))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Code> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(|_| Code::InvalidRequest)?;
    file.write_all(bytes).map_err(|_| Code::InvalidRequest)?;
    file.flush().map_err(|_| Code::InvalidRequest)?;
    file.sync_all().map_err(|_| Code::InvalidRequest)
}

async fn run() -> Result<(), Code> {
    let args = Args::parse()?;
    let secret = std::env::var("M3_ACCEPTANCE_SECRET").map_err(|_| Code::InvalidRequest)?;
    require((32..=256).contains(&secret.len()))?;
    let policy_path =
        PathBuf::from(std::env::var("RV1_M3_POLICY").map_err(|_| Code::InvalidRequest)?);
    let load_task = tokio::task::spawn_blocking(move || load(args, policy_path));
    let Loaded {
        args,
        policy_bytes,
        policy,
        mut config,
        tokenizer,
        check,
        binary_sha256,
    } = timeout(Duration::from_secs(300), load_task)
        .await
        .map_err(|_| Code::DeadlineExceeded)?
        .map_err(|_| Code::InvalidRequest)??;

    let listener = TcpListener::bind(args.bind)
        .await
        .map_err(|_| Code::InvalidRequest)?;
    let actual_http_address = listener.local_addr().map_err(|_| Code::InvalidRequest)?;
    let listener = BoundedListener::new(listener, &config.transport)?;
    let control_listener = bind_control(args.control.ok_or(Code::InvalidRequest)?).await?;
    let actual_control_address = control_listener
        .local_addr()
        .map_err(|_| Code::InvalidRequest)?;

    // Assign once. Every later identity is read from Config.instance.
    config.instance = Some(instance_id()?);
    config.validate()?;
    let start_utc = start_time()?;
    let batching = config.batching.clone().ok_or(Code::InvalidRequest)?;
    let build_identity = json!({
        "rustc_version": env!("RV1_BUILD_RUSTC_VERSION"),
        "nvcc_version": env!("RV1_BUILD_NVCC_VERSION"),
        "cuda_compute_cap": env!("RV1_BUILD_CUDA_COMPUTE_CAP"),
        "target_features": env!("RV1_BUILD_TARGET_FEATURES"),
    });
    // Build metadata is retained as attestation, not misrepresented as proof
    // of native cubin contents or of the running GPU's identity.
    let attestation_value = json!({
        "schema": "rustv1.server_launch_attestation.v1",
        "instance_id": config.instance.as_deref().ok_or(Code::InvalidRequest)?,
        "pid": std::process::id(),
        "start_utc": start_utc,
        "binary_sha256": binary_sha256,
        "settings_sha256": format!("{:x}", Sha256::digest(&policy_bytes)),
        "audit_sha256": policy["artifacts"]["audit"]["sha256"],
        "startup_check_sha256": policy["artifacts"]["startup_check"]["sha256"],
        "build_identity": build_identity,
        "http_contract_id": BATCHED_HTTP_CONTRACT,
        "execution_policy": BATCHED_EXECUTION_POLICY,
        "batching": batching,
    });
    let attestation_bytes =
        serde_json::to_vec_pretty(&attestation_value).map_err(|_| Code::InvalidRequest)?;
    let attestation_sha256 = format!("{:x}", Sha256::digest(&attestation_bytes));
    let attestation_path = PathBuf::from(
        std::env::var("RV1_ATTESTATION_PATH")
            .unwrap_or_else(|_| format!("{}.att.json", args.manifest.display())),
    );
    write_new(&attestation_path, &attestation_bytes)?;

    let manifest = args.manifest;
    let model_dir = args.model_dir;
    let audit_identity = args.audit_identity;
    let mut service = Service::spawn(config, tokenizer, Logger::stderr(), check, move || {
        let executor = rustv1::m2_route::open_batched_with_attestation(
            &model_dir,
            &manifest,
            0,
            &audit_identity,
            Some(&attestation_sha256),
            batching,
        )
        .map_err(|_| Code::InferenceFailed)?;
        Ok(CudaBackend { executor })
    })?;
    service.wait_ready().await?;

    let mut launch_att = attestation_value;
    launch_att["http_address"] = json!(actual_http_address.to_string());
    launch_att["control_address"] = json!(actual_control_address.to_string());
    service.app.install_launch_attestation(launch_att)?;

    let app = service.app.clone();
    let mut control_task =
        tokio::spawn(async move { serve_control(control_listener, app, secret).await });
    let (stop, stopped) = oneshot::channel();
    let router = service.app.router();
    let mut server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });

    // Preserve the terminal observation window for an external supervisor.
    // No internal CUDA reinitialization or overlapping replacement is allowed.
    let mut server_finished = false;
    let mut control_finished = false;
    let normal_signal = tokio::select! {
        result = termination() => result.is_ok(),
        _ = &mut server => {
            server_finished = true;
            false
        },
        _ = &mut control_task => {
            control_finished = true;
            false
        },
    };
    let terminal = matches!(
        service.app.lifecycle(),
        rustv1::http_serve::Lifecycle::Poisoned | rustv1::http_serve::Lifecycle::Retired
    );
    service.app.begin_drain();
    let _ = stop.send(());
    if !control_finished {
        control_task.abort();
        let _ = control_task.await;
    }

    let shutdown = service.shutdown().await;
    if shutdown.is_err() {
        server.abort();
        return shutdown;
    }
    if server_finished {
        return Err(Code::NotReady);
    }
    let server_result = timeout(
        Duration::from_secs(service.app.config.drain_seconds),
        &mut server,
    )
    .await;
    if !matches!(server_result, Ok(Ok(Ok(())))) {
        server.abort();
        return Err(Code::DeadlineExceeded);
    }
    if normal_signal && !terminal {
        Ok(())
    } else {
        Err(Code::ExecutorPoisoned)
    }
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(_) => std::process::exit(1),
    };
    if runtime.block_on(run()).is_err() {
        // A timed-out or failed CUDA owner must not survive a supervisor's
        // restart decision merely because a detached worker still exists.
        std::process::exit(1);
    }
    runtime.shutdown_timeout(Duration::from_secs(1));
}
