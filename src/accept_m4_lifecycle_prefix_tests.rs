use super::*;
use rustv1::{
    http_serve::{
        Backend, BoundedListener, Code, Config, ControlCommand, Logger, Service, StartupCheck,
        TextTokenizer,
    },
    model::{BatchedExecution, ExecutionRow},
};
use std::{
    process::{Command, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};

const CHILD_TEST: &str =
    "acceptance_scheduler::lifecycle_prefix::tests::terminal_routes_child_process";
const CHILD_ENV: &str = "RV1_M4_TERMINAL_ROUTE_TEST";
const RESTORE_ENV: &str = "RV1_M4_RESTORE_ROUTE_TEST";

fn limits() -> Limits {
    serde_json::from_value(json!({
        "connect_ms": 5000,
        "headers_ms": 65000,
        "body_ms": 10000,
        "control_ms": 30000,
        "observation_ms": 2000,
        "server_request_ms": 60000,
        "response_bytes": 1048576,
        "admission_capacity": 64,
        "queue_capacity": 64
    }))
    .unwrap()
}

fn row() -> Vec<f32> {
    let mut row = vec![0.0; 2048];
    row[100] = 1.0;
    row
}

struct Tokens;
impl TextTokenizer for Tokens {
    fn encode(&self, text: &str) -> std::result::Result<Vec<u32>, Code> {
        if text == "valid" {
            Ok(vec![100])
        } else {
            Err(Code::InvalidRequest)
        }
    }
}

struct DriftBackend {
    drift: bool,
    restores: Arc<std::sync::atomic::AtomicUsize>,
}
impl Backend for DriftBackend {
    fn forward(&mut self, batches: &[Vec<u32>]) -> std::result::Result<Vec<Vec<f32>>, Code> {
        Ok(batches.iter().map(|_| row()).collect())
    }

    fn forward_batch(
        &mut self,
        batches: &[Vec<u32>],
        _: &[ExecutionRow],
    ) -> std::result::Result<Vec<Vec<f32>>, Code> {
        if self.drift {
            Err(Code::InferenceFailed)
        } else {
            Ok(batches.iter().map(|_| row()).collect())
        }
    }

    fn inject_policy_drift(&mut self) -> std::result::Result<(), Code> {
        self.drift = true;
        Ok(())
    }

    fn restore_policy(&mut self) -> std::result::Result<(), Code> {
        self.drift = false;
        self.restores
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

fn service<T>(result: std::result::Result<T, Code>) -> Result<T> {
    result.map_err(|code| anyhow::anyhow!("production Service: {code:?}"))
}

async fn child_work(out: &Path, check_restore: bool) -> Result<()> {
    let instance = format!("terminal-{}", core::random_label()?);
    let mut config = Config::production(64, 131072, 1024);
    config.batching = Some(BatchedExecution::default());
    config.instance = Some(instance.clone());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let base = format!("http://{}", listener.local_addr()?);
    let listener = service(BoundedListener::new(listener, &config.transport))?;
    let restores = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let backend_restores = restores.clone();
    let mut owner = service(Service::spawn(
        config,
        Arc::new(Tokens),
        Logger::new(std::io::sink()),
        StartupCheck {
            tokens: vec![vec![100]],
            embeddings: vec![row()],
        },
        move || {
            Ok(DriftBackend {
                drift: false,
                restores: backend_restores,
            })
        },
    ))?;
    let app = owner.app.clone();
    let router = app.router();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let mut server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });

    let work: Result<()> = async {
        service(owner.wait_ready().await)?;
        let before = serde_json::to_value(service(app.control(ControlCommand::Observe).await)?)?;
        let quiesced = serde_json::to_value(service(app.control(ControlCommand::Quiesce).await)?)?;
        let armed = serde_json::to_value(service(
            app.control(ControlCommand::InjectPolicyDrift).await,
        )?)?;
        save_json(
            &out.join("controls.json"),
            &json!({
                "before": before, "quiesced": quiesced, "armed": armed
            }),
        )?;
        ensure!(
            quiesced["ready"] == false && armed == before,
            "production arm/quiesce observations"
        );

        let http = Http::new(&base, &instance, limits())?;
        let body = json!({"model": MODEL_ID, "input": ["valid"], "encoding_format": "float"});
        let trigger = http
            .send("cpu-drift-trigger", &Wire::json(body.clone())?)
            .await?;
        save_reply(out, "cpu-drift-trigger", &trigger)?;
        production_error(&trigger, "inference_failed")?;

        let observation =
            serde_json::to_value(service(app.control(ControlCommand::Observe).await)?)?;
        save_json(&out.join("poisoned.json"), &observation)?;
        ensure!(
            observation == poisoned(&before)?,
            "production poisoned observation"
        );

        let live = http.get_health("cpu-livez", "/health/livez").await?;
        save_reply(out, "cpu-livez", &live)?;
        ensure!(live.status == 200 && live.bytes.is_empty(), "real livez");

        let ready = http.get_health("cpu-readyz", "/health/readyz").await?;
        save_reply(out, "cpu-readyz", &ready)?;
        production_error(&ready, "executor_poisoned")?;
        ensure!(
            !app.is_ready() && ready.request_id.is_none(),
            "real readyz/App::is_ready path"
        );

        let refusal = http.send("cpu-refusal", &Wire::json(body)?).await?;
        save_reply(out, "cpu-refusal", &refusal)?;
        production_error(&refusal, "executor_poisoned")?;
        ensure!(
            refusal.request_id.is_none(),
            "terminal admission refusal ID"
        );
        ensure!(
            serde_json::to_value(service(app.control(ControlCommand::Observe).await)?)?
                == observation,
            "health/refusal altered logical counter"
        );

        if check_restore {
            ensure!(
                restores.load(std::sync::atomic::Ordering::SeqCst) == 1,
                "drift trigger did not perform its automatic cleanup restore"
            );
            let restore_out = out.join("explicit-restore");
            private_directory(&restore_out)?;
            let probe = super::restart_prefix::restore_probe(
                |name| {
                    let app = &app;
                    async move {
                        let command = match name {
                            "observe" => ControlCommand::Observe,
                            "restore" => ControlCommand::Restore,
                            _ => anyhow::bail!("unplanned restore harness control"),
                        };
                        Ok(serde_json::to_value(service(app.control(command).await)?)?)
                    }
                },
                &http,
                json!({
                    "model": MODEL_ID,
                    "input": ["valid"],
                    "encoding_format": "float"
                }),
                &observation,
                &restore_out,
            )
            .await?;
            ensure!(
                restores.load(std::sync::atomic::Ordering::SeqCst) == 2
                    && probe["restore_observation"] == observation
                    && probe["observation_after"] == observation
                    && !app.is_ready(),
                "explicit production restore was skipped or reopened the executor"
            );
            save_json(&restore_out.join("evidence.json"), &probe)?;
        }
        Ok(())
    }
    .await;

    let connections = http::join_connections(&instance).await?;
    save_json(&out.join("connections.json"), &json!(connections))?;

    // Do not clear operational quarantine to make shutdown pass.
    let shutdown = owner.shutdown().await;
    let _ = stop.send(());
    let transport = match tokio::time::timeout(Duration::from_secs(10), &mut server).await {
        Ok(Ok(Ok(()))) => true,
        _ => {
            server.abort();
            let _ = server.await;
            false
        }
    };
    let closure = json!({
        "owner_thread_joined": shutdown.is_ok(),
        "shutdown_error": shutdown.err(),
        "http_server_joined": transport,
        "quarantine_cleared": false,
        "child_runtime_threads_joined": false,
        "child_watchdog_thread_joined": false,
        "containment_required": "parent-reap"
    });
    save_json(&out.join("owner_closure.json"), &closure)?;
    work?;
    ensure!(
        closure["owner_thread_joined"] == false
            && closure["shutdown_error"] == "deadline_exceeded"
            && transport,
        "quarantine/transport closure"
    );
    Ok(())
}

#[test]
#[ignore = "private child-process protocol; invoked by parent harness"]
fn terminal_routes_child_process() {
    let out = PathBuf::from(std::env::var(CHILD_ENV).expect("child evidence directory"));
    let watchdog_out = out.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(90));
        let _ = save_json(&watchdog_out.join("watchdog.json"), &json!({"fired": true}));
        std::process::exit(124);
    });

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        let check_restore = std::env::var(RESTORE_ENV).as_deref() == Ok("1");
        let result = runtime.block_on(child_work(&out, check_restore));
        std::mem::forget(runtime);
        result
    }));
    let success = matches!(&result, Ok(Ok(())));
    let error = match result {
        Ok(Err(error)) => Some(format!("{error:#}")),
        Err(_) => Some("child harness panicked".to_owned()),
        Ok(Ok(())) => None,
    };
    let _ = save_json(
        &out.join("child_result.json"),
        &json!({
            "pid": std::process::id(),
            "success": success,
            "error": error,
            "backend": "synthetic-drift-stimulus",
            "live_cuda_policy_drift_proof": false
        }),
    );
    std::process::exit(if success { 0 } else { 1 });
}

fn parent_routes_in_reaped_child(check_restore: bool) {
    let root = std::env::temp_dir().join(format!(
        "rv1-terminal-routes-{}-{}",
        std::process::id(),
        core::random_label().unwrap()
    ));
    private_directory(&root).unwrap();

    let result: Result<()> = (|| {
        let mut child = Command::new(std::env::current_exe()?)
            .arg("--exact")
            .arg(CHILD_TEST)
            .arg("--ignored")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env_clear()
            .env(CHILD_ENV, &root)
            .env(RESTORE_ENV, if check_restore { "1" } else { "0" })
            .stdin(Stdio::null())
            .stdout(Stdio::from(fs::File::create(root.join("stdout.log"))?))
            .stderr(Stdio::from(fs::File::create(root.join("stderr.log"))?))
            .spawn()?;
        let pid = child.id();
        let deadline = Instant::now() + Duration::from_secs(100);
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                other => {
                    let _ = child.kill();
                    let status = child.wait()?;
                    save_json(
                        &root.join("parent_failure.json"),
                        &json!({
                            "poll": format!("{other:?}"),
                            "reaped": true,
                            "exit_code": status.code()
                        }),
                    )?;
                    anyhow::bail!("terminal child failed or exceeded watchdog");
                }
            }
        };
        save_json(
            &root.join("parent_reap.json"),
            &json!({
                "pid": pid,
                "child_process_reaped": true,
                "exit_code": status.code(),
                "child_runtime_threads_joined": false,
                "child_watchdog_thread_joined": false
            }),
        )?;
        ensure!(status.success(), "terminal child failed");
        let closure = value(&root.join("owner_closure.json"))?;
        ensure!(
            closure["owner_thread_joined"] == false
                && closure["shutdown_error"] == "deadline_exceeded"
                && closure["quarantine_cleared"] == false,
            "parent must preserve child-local unjoined closure"
        );
        Ok(())
    })();

    if result.is_ok() {
        fs::remove_dir_all(&root).unwrap();
    }
    result.unwrap_or_else(|error| {
        panic!(
            "terminal route harness failed; evidence retained at {}: {error:#}",
            root.display()
        )
    });
}

#[test]
fn connection_reconciliation_rejects_missing_extra_and_relabelled_owners() {
    let completed = json!({
        "instance": "instance",
        "connection_id": 1,
        "join_awaited": true,
        "join_outcome": "completed"
    });
    assert!(check_connections("instance", &[completed.clone()], &[completed.clone()]).is_ok());
    assert!(check_connections("instance", &[completed.clone()], &[]).is_err());
    assert!(check_connections(
        "instance",
        &[completed.clone()],
        &[completed.clone(), completed.clone()]
    )
    .is_err());

    let mut changed = completed.clone();
    changed["join_outcome"] = json!("panicked");
    assert!(check_connections("instance", &[completed], &[changed]).is_err());
}

#[test]
fn production_terminal_health_routes_in_reaped_child() {
    parent_routes_in_reaped_child(false);
}

#[test]
fn production_restore_stays_poisoned_in_reaped_child() {
    parent_routes_in_reaped_child(true);
}
