use rustv1::{
    http_serve::{
        bind_control, serve_control, Backend, BoundedListener, Code, Config, Logger,
        PinnedTokenizer, Service, StartupCheck,
    },
    model::{EngineConfig, IndependentExecutor},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};
use tokio::{net::TcpListener, sync::oneshot, time::timeout};

struct CudaBackend {
    executor: IndependentExecutor,
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
        // This intentionally does not replace or unpoison the executor.
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
        // serve_m3 MODEL_DIR MANIFEST_JSONL AUDIT_IDENTITY BIND_ADDR
        //          QUEUE_CAPACITY TOKENIZER_SHA256 STARTUP_CHECK_JSON
        //          STARTUP_CHECK_SHA256 CONTROL_ADDR_OR_DASH
        //
        // STARTUP_CHECK_JSON:
        // {"tokens":[[...]],"embeddings":[[...2048 F32 values...]]}
        //
        // The startup artifact must come from accepted in-process evidence,
        // not from this server's own startup execution.
        //
        // Control is disabled with "-". Otherwise its address must be loopback,
        // and M3_ACCEPTANCE_SECRET must contain 32..256 bytes.
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

    let bind = args.bind;
    let control_address = args.control;
    let model_dir = args.model_dir.clone();
    let tokenizer_sha256 = args.tokenizer_sha256;
    let startup_path = args.startup_check;
    let startup_sha256 = args.startup_check_sha256;
    let queue_capacity = args.queue_capacity;

    // Filesystem IO, tokenizer construction, and hash verification do not run
    // on a Tokio scheduler thread.
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
    let (config, tokenizer, check) = timeout(Duration::from_secs(60), load)
        .await
        .map_err(|_| Code::DeadlineExceeded)?
        .map_err(|_| Code::InvalidRequest)??;

    // Bind before allocating CUDA resources, but do not accept requests until
    // the owner-thread startup check has completed.
    let listener = TcpListener::bind(bind)
        .await
        .map_err(|_| Code::InvalidRequest)?;
    let listener = BoundedListener::new(listener, &config.transport)?;
    let control_listener = match control_address {
        Some(address) => Some(bind_control(address).await?),
        None => None,
    };
    let grace = config.shutdown_grace;
    let mut service = Service::spawn(config, tokenizer, Logger::stderr(), check, move || {
        // Use the unchanged production M2 path. No second math pin,
        // alternative loader, or CUDA initialization is introduced here.
        //
        // Important: the attached implementation of m2_route::open calls
        // MathModePin::pin, whose visible audit-identity validation is a
        // nonempty-string check. Stronger audit-artifact/lock validation
        // is not implemented by those attached bytes. This adapter does
        // not claim otherwise or silently substitute a different path.
        let executor =
            rustv1::m2_route::open(&args.model_dir, &args.manifest, 0, &args.audit_identity)
                .map_err(|_| Code::InferenceFailed)?;
        Ok(CudaBackend { executor })
    })?;
    service.wait_ready().await?;

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

    // Admission closes before the listener's graceful-shutdown notification.
    service.app.begin_drain();
    let _ = stop.send(());
    if let Some(task) = control_task {
        task.abort();
        let _ = task.await;
    }

    let shutdown = service.shutdown().await;
    if shutdown.is_err() {
        // The caller exits the process immediately. It must not await runtime
        // destruction with a potentially stuck CUDA owner/blocking join.
        server.abort();
        return shutdown;
    }

    if normal_signal {
        match timeout(grace, &mut server).await {
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

fn main() {
    // No panic payload, raw error chain, CLI value, or path reaches stderr.
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
        // In particular, do not block in Runtime::drop after a hard owner
        // timeout. Process termination is the only recovery boundary.
        std::process::exit(1);
    }
    runtime.shutdown_timeout(Duration::from_secs(1));
}
