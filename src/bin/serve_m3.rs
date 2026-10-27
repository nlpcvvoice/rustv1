// src/bin/serve_m3.rs

use rustv1::{
    http_serve::{
        bind_control, serve_control, Backend, BoundedListener, Code, Config, Logger,
        PinnedTokenizer, Service, StartupCheck,
    },
    model::EngineConfig,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{net::TcpListener, sync::oneshot, time::timeout};

struct CudaBackend {
    executor: rustv1::model::IndependentExecutor,
}

impl Backend for CudaBackend {
    fn forward(&mut self, batches: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, Code> {
        self.executor
            .forward_serving(batches)
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
        if !(1..=4096).contains(&queue_capacity) {
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

fn integer(value: &Value, key: &str) -> Result<usize, Code> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .ok_or(Code::InvalidRequest)
}

fn file_sha256(path: &Path) -> Result<String, Code> {
    let bytes = fs::read(path).map_err(|_| Code::InvalidRequest)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
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

async fn run() -> Result<(), Code> {
    let args = Args::parse()?;
    let control_secret = if args.control.is_some() {
        let secret = std::env::var("M3_ACCEPTANCE_SECRET").map_err(|_| Code::InvalidRequest)?;
        if !(32..=256).contains(&secret.len()) {
            return Err(Code::InvalidRequest);
        }
        Some(secret)
    } else {
        None
    };

    let launch_policy_path =
        PathBuf::from(std::env::var("RV1_M3_POLICY").map_err(|_| Code::InvalidRequest)?);
    let policy_bytes = fs::read(&launch_policy_path).map_err(|_| Code::InvalidRequest)?;
    let policy: Value = serde_json::from_slice(&policy_bytes).map_err(|_| Code::InvalidRequest)?;

    let expected_control: SocketAddr = policy["launch_control_address"]
        .as_str()
        .ok_or(Code::InvalidRequest)?
        .parse()
        .map_err(|_| Code::InvalidRequest)?;
    if !expected_control.ip().is_loopback()
        || args.control != Some(expected_control)
        || policy["audit_identity"] != args.audit_identity
        || policy["artifacts"]["tokenizer"]["sha256"] != args.tokenizer_sha256
        || policy["artifacts"]["startup_check"]["sha256"] != args.startup_check_sha256
        || policy["limits"]["admission_capacity"].as_u64() != Some(64)
        || policy["limits"]["queue_capacity"].as_u64() != Some(args.queue_capacity as u64)
        || policy["limits"]["server_request_ms"].as_u64() != Some(60_000)
        || policy["model"] != rustv1::http_serve::MODEL_ID
    {
        return Err(Code::InvalidRequest);
    }

    let bind = args.bind;
    let control_address = args.control;
    let model_dir = args.model_dir.clone();
    let tokenizer_sha256 = args.tokenizer_sha256;
    let startup_path = args.startup_check;
    let startup_sha256 = args.startup_check_sha256;
    let queue_capacity = args.queue_capacity;

    let load = tokio::task::spawn_blocking(move || {
        let config_bytes =
            std::fs::read(model_dir.join("config.json")).map_err(|_| Code::InvalidRequest)?;
        let model: Value =
            serde_json::from_slice(&config_bytes).map_err(|_| Code::InvalidRequest)?;
        let engine_bytes =
            std::fs::read(model_dir.join("engine.json")).map_err(|_| Code::InvalidRequest)?;
        let engine: EngineConfig =
            serde_json::from_slice(&engine_bytes).map_err(|_| Code::InvalidRequest)?;

        let vocabulary_size = integer(&model, "vocab_size")?;
        let model_limit = integer(&model, "max_position_embeddings")?;
        if engine.max_len == 0
            || engine.max_len > 1024
            || engine.max_len > model_limit
            || engine.max_seqs == 0
            || engine.max_tokens < engine.max_len
            || model.get("pad_token_id").and_then(Value::as_u64) != Some(11)
        {
            return Err(Code::InvalidRequest);
        }
        let config = Config::production(queue_capacity, vocabulary_size, engine.max_len);
        config.validate()?;
        let tokenizer = PinnedTokenizer::load(
            &model_dir.join("tokenizer.json"),
            &tokenizer_sha256,
            vocabulary_size,
        )?;
        let check_bytes = std::fs::read(startup_path).map_err(|_| Code::InvalidRequest)?;
        if startup_sha256.len() != 64
            || format!("{:x}", Sha256::digest(&check_bytes)) != startup_sha256
        {
            return Err(Code::InvalidRequest);
        }
        let check: StartupCheck =
            serde_json::from_slice(&check_bytes).map_err(|_| Code::InvalidRequest)?;
        Ok::<_, Code>((config, Arc::new(tokenizer), check))
    });
    let (mut config, tokenizer, check) = timeout(Duration::from_secs(60), load)
        .await
        .map_err(|_| Code::DeadlineExceeded)?
        .map_err(|_| Code::InvalidRequest)??;

    let listener = TcpListener::bind(bind)
        .await
        .map_err(|_| Code::InvalidRequest)?;
    let actual_http_address = listener.local_addr().map_err(|_| Code::InvalidRequest)?;
    let listener = BoundedListener::new(listener, &config.transport)?;
    let control_listener = match control_address {
        Some(address) => Some(bind_control(address).await?),
        None => None,
    };

    // B1: Prepare Launch Attestation file and compute digest
    let binary_sha256 = file_sha256(&std::env::current_exe().map_err(|_| Code::InvalidRequest)?)?;
    let start_utc = chrono_or_epoch();
    let instance_id = format!(
        "{}-{}",
        std::process::id(),
        start_utc.replace([':', '-'], "")
    );
    config.instance = Some(instance_id.clone());
    let build_identity = json!({
        "rustc_version": env!("RV1_BUILD_RUSTC_VERSION"),
        "nvcc_version": env!("RV1_BUILD_NVCC_VERSION"),
        "cuda_compute_cap": env!("RV1_BUILD_CUDA_COMPUTE_CAP"),
        "target_features": env!("RV1_BUILD_TARGET_FEATURES"),
    });

    let attestation_value = json!({
        "schema": "rustv1.server_launch_attestation.v1",
        "instance_id": instance_id,
        "pid": std::process::id(),
        "start_utc": start_utc,
        "binary_sha256": binary_sha256,
        "settings_sha256": format!("{:x}", Sha256::digest(&policy_bytes)),
        "audit_sha256": policy["artifacts"]["audit"]["sha256"],
        "startup_check_sha256": policy["artifacts"]["startup_check"]["sha256"],
        "build_identity": build_identity,
    });
    let attestation_bytes =
        serde_json::to_vec_pretty(&attestation_value).map_err(|_| Code::InvalidRequest)?;
    let attestation_sha256 = format!("{:x}", Sha256::digest(&attestation_bytes));

    let attestation_path = std::env::var("RV1_ATTESTATION_PATH")
        .unwrap_or_else(|_| format!("{}.att.json", args.manifest.display()));
    fs::write(&attestation_path, &attestation_bytes).map_err(|_| Code::InvalidRequest)?;

    let manifest_path = args.manifest.clone();
    let model_dir_clone = args.model_dir.clone();
    let audit_identity_clone = args.audit_identity.clone();
    let attestation_sha256_clone = attestation_sha256.clone();

    let mut service = Service::spawn(config, tokenizer, Logger::stderr(), check, move || {
        let executor = rustv1::m2_route::open_with_attestation(
            &model_dir_clone,
            &manifest_path,
            0,
            &audit_identity_clone,
            Some(&attestation_sha256_clone),
        )
        .map_err(|_| Code::InferenceFailed)?;
        Ok(CudaBackend { executor })
    })?;
    service.wait_ready().await?;

    let mut launch_att = attestation_value;
    launch_att["http_address"] = json!(actual_http_address.to_string());
    launch_att["control_address"] = json!(control_listener
        .as_ref()
        .ok_or(Code::InvalidRequest)?
        .local_addr()
        .map_err(|_| Code::InvalidRequest)?
        .to_string());
    service.app.install_launch_attestation(launch_att)?;

    let control_task = match (control_listener, control_secret) {
        (Some(listener), Some(secret)) => {
            let app = service.app.clone();
            Some(tokio::spawn(async move {
                serve_control(listener, app, secret).await
            }))
        }
        _ => None,
    };

    let (stop, stopped) = oneshot::channel();
    let router = service.app.router();
    let mut server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });

    let normal_signal = tokio::select! {
        result = termination() => result.is_ok(),
        _ = &mut server => false,
    };

    service.app.begin_drain();
    let _ = stop.send(());
    if let Some(task) = control_task {
        task.abort();
        let _ = task.await;
    }

    let shutdown = service.shutdown().await;
    if shutdown.is_err() {
        server.abort();
        return shutdown;
    }

    if normal_signal {
        match timeout(
            Duration::from_secs(service.app.config.drain_seconds),
            &mut server,
        )
        .await
        {
            Ok(Ok(Ok(()))) => Ok(()),
            _ => {
                server.abort();
                Err(Code::DeadlineExceeded)
            }
        }
    } else {
        Err(Code::NotReady)
    }
}

fn chrono_or_epoch() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:06}Z", d.as_secs(), d.subsec_micros())
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
    let result = runtime.block_on(run());

    if result.is_err() {
        std::process::exit(1);
    }
    runtime.shutdown_timeout(Duration::from_secs(1));
}
