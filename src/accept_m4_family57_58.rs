// Included inside acceptance_scheduler::physical_b::family57_58.
//
// The enclosing module is gated by scheduler-test-seam.
// No production scheduling, retirement, quarantine, or drain behavior changes.
//
// Destructive probes run only in a separate process. In particular, the
// operational-failure probe never clears quarantine to manufacture a join.

use super::*;
use serde::Deserialize;
use std::{
    fs::File,
    future::{poll_fn, Future},
    io::Write,
    process::{Child, Command, Stdio},
    sync::Condvar,
    task::Poll,
    time::Instant,
};
use tokio::sync::watch;

const DRAIN: &str = "family-57-drain-running-work";
const FAILURE: &str = "family-58-physical-failure-fanout";
const EXPIRY: &str = "family-58-running-expiry-retirement";
const NEXT: &str = "scheduler-owners-joined";

const CHILD_COMMAND: &str = "scheduler-destructive-child";
#[cfg(test)]
const CHILD_TEST: &str =
    "acceptance_scheduler::physical_b::family57_58::tests::\
     scheduler_destructive_child_process";
#[cfg(test)]
const CHILD_ENV: &str = "RV1_M4_DESTRUCTIVE_SPEC";

const REQUEST_US: u64 = 60_000_000;
const CHILD_BUDGET: Duration = Duration::from_secs(100);
const PARENT_BUDGET: Duration = Duration::from_secs(120);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Probe {
    Drain,
    Failure,
    Expiry,
}

impl Probe {
    fn parse(case: &str) -> Result<Self> {
        match case {
            DRAIN => Ok(Self::Drain),
            FAILURE => Ok(Self::Failure),
            EXPIRY => Ok(Self::Expiry),
            _ => anyhow::bail!("unknown destructive case"),
        }
    }

    fn case(self) -> &'static str {
        match self {
            Self::Drain => DRAIN,
            Self::Failure => FAILURE,
            Self::Expiry => EXPIRY,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChildSpec {
    case: String,
    out: PathBuf,
    nonce: String,
    limits: Value,
    parent_pid: u32,
}

fn private_file(path: &Path) -> Result<File> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

fn append_record(file: &mut File, record: &Value) -> std::result::Result<(), Code> {
    serde_json::to_writer(&mut *file, record).map_err(|_| Code::InferenceFailed)?;
    file.write_all(b"\n").map_err(|_| Code::InferenceFailed)?;
    file.sync_data().map_err(|_| Code::InferenceFailed)
}

fn startup_record() -> Value {
    json!({
        "kind": "startup",
        "tokens": [[100]],
        "output_sha256": [sha(&row_bytes(&vector(100)))]
    })
}

/// A backend-entry gate, not a replacement scheduler.
///
/// Entry is published after the actual backend begin record has been synced.
/// Waiting here therefore witnesses execution inside Backend::forward_batch,
/// not merely assembly or a scheduler launch decision.
struct Gate {
    open: Mutex<bool>,
    progress: Condvar,
    entered: watch::Sender<bool>,
}

impl Gate {
    fn new() -> Self {
        let (entered, _) = watch::channel(false);
        Self {
            open: Mutex::new(false),
            progress: Condvar::new(),
            entered,
        }
    }

    fn enter(&self) {
        let mut open = guard(&self.open);
        self.entered.send_replace(true);
        while !*open {
            open = self
                .progress
                .wait(open)
                .unwrap_or_else(|error| error.into_inner());
        }
    }

    fn release(&self) {
        *guard(&self.open) = true;
        self.progress.notify_all();
    }

    async fn wait_entered(&self) -> Result<()> {
        timeout(WATCHDOG, async {
            let mut receiver = self.entered.subscribe();
            loop {
                if *receiver.borrow_and_update() {
                    return Ok::<(), anyhow::Error>(());
                }
                receiver.changed().await?;
            }
        })
        .await
        .context("backend-entry watchdog; not semantic time")?
    }
}

/// Produces deterministic rows and records actual physical input associations.
///
/// Failure stimulus: only the output corresponding to token 101 has zero norm.
/// The production validate_output call rejects it. This backend neither
/// poisons the executor nor fans errors out to logical requests.
struct ProbeBackend {
    file: File,
    gate: Arc<Gate>,
    probe: Probe,
    next: u64,
}

impl Backend for ProbeBackend {
    fn forward(
        &mut self,
        batches: &[Vec<u32>],
    ) -> std::result::Result<Vec<Vec<f32>>, Code> {
        let output: Vec<Vec<f32>> =
            batches.iter().map(|tokens| vector(tokens[0])).collect();
        append_record(
            &mut self.file,
            &json!({
                "kind": "startup",
                "tokens": batches,
                "output_sha256": output.iter()
                    .map(|row| sha(&row_bytes(row))).collect::<Vec<_>>()
            }),
        )?;
        Ok(output)
    }

    fn forward_batch(
        &mut self,
        batches: &[Vec<u32>],
        associations: &[ExecutionRow],
    ) -> std::result::Result<Vec<Vec<f32>>, Code> {
        let id = self.next;
        self.next += 1;
        let b = batches.len();
        let s = batches.iter().map(Vec::len).max().unwrap_or(0);
        let valid: usize = batches.iter().map(Vec::len).sum();

        let rows: Vec<Value> = batches
            .iter()
            .zip(associations)
            .enumerate()
            .map(|(physical_row, (tokens, association))| {
                json!({
                    "physical_row": physical_row,
                    "request_id": association.request_id,
                    "request_row": association.request_row,
                    "case_id": association.case_id,
                    "valid_length": tokens.len(),
                    "token_sha256": ids_sha(tokens)
                })
            })
            .collect();

        append_record(
            &mut self.file,
            &json!({
                "kind": "begin",
                "forward_id": id,
                "B": b,
                "S": s,
                "valid_tokens": valid,
                "padding_tokens": b * s - valid,
                "input_rows": batches.len(),
                "association_rows": associations.len(),
                "rows": rows
            }),
        )?;

        if batches.iter().any(|tokens| tokens == &[100]) {
            self.gate.enter();
        }

        let mut invalid = Vec::new();
        let output: Vec<Vec<f32>> = batches
            .iter()
            .enumerate()
            .map(|(position, tokens)| {
                if self.probe == Probe::Failure && tokens == &[101] {
                    invalid.push(position);
                    vec![0.0; 2048]
                } else {
                    vector(tokens[0])
                }
            })
            .collect();

        let returned: Vec<Value> = rows
            .iter()
            .zip(&output)
            .map(|(row, output)| {
                let mut row = row.clone();
                row["output_sha256"] = json!(sha(&row_bytes(output)));
                row
            })
            .collect();

        // "returned_rows" is deliberately not a scheduler success claim.
        // In the failure probe the production validator rejects this return.
        append_record(
            &mut self.file,
            &json!({
                "kind": "returned_rows",
                "forward_id": id,
                "rows": returned,
                "injected_invalid_output_rows": invalid
            }),
        )?;
        Ok(output)
    }

    fn inject_policy_drift(&mut self) -> std::result::Result<(), Code> {
        Err(Code::InvalidRequest)
    }

    fn restore_policy(&mut self) -> std::result::Result<(), Code> {
        Err(Code::InvalidRequest)
    }
}

fn check_error(reply: &http::Reply, code: Code) -> Result<()> {
    let (message, kind) = match code {
        Code::InferenceFailed => ("Inference failed", "api_error"),
        Code::DeadlineExceeded => ("Request deadline exceeded", "api_error"),
        Code::ExecutorPoisoned => (
            "Executor is retired; restart required",
            "service_unavailable_error",
        ),
        _ => anyhow::bail!("unplanned destructive error"),
    };
    ensure!(
        reply.status == code.status().as_u16()
            && serde_json::from_slice::<Value>(&reply.bytes)?
                == json!({
                    "error": {
                        "message": message,
                        "type": kind,
                        "param": null,
                        "code": code
                    }
                }),
        "wrong production error envelope or partial success"
    );
    Ok(())
}

struct Local {
    service: Service,
    app: App,
    control: SchedulerControl,
    gate: Arc<Gate>,
    server: tokio::task::JoinHandle<std::io::Result<()>>,
    stop: Option<oneshot::Sender<()>>,
    base: String,
    instance: String,
    limits: Limits,
    out: PathBuf,
    manifest: PathBuf,
    tasks: JoinSet<Result<(Plan, http::Reply)>>,
    plans: BTreeMap<u64, Plan>,
    insertion: Vec<Value>,
    responses: BTreeMap<u64, Value>,
}

impl Local {
    async fn create(limits: Limits, out: &Path, probe: Probe) -> Result<Self> {
        let control = SchedulerControl::new();
        let gate = Arc::new(Gate::new());
        let instance = format!("f5758-{}", core::random_label()?);
        let manifest = out.join("backend_manifest.jsonl");
        let file = private_file(&manifest)?;
        let logger = Logger::new(private_file(&out.join("service_log.jsonl"))?);

        let mut config = Config::production(64, 131072, 1024);
        config.batching = Some(BatchedExecution::default());
        config.scheduler_control = Some(control.clone());
        config.instance = Some(instance.clone());
        service_result(config.validate())?;
        ensure!(
            config.request_timeout == Duration::from_micros(REQUEST_US)
                && config.drain_seconds == 30
                && config.shutdown_grace == Duration::from_secs(90),
            "production timeout/drain envelope changed"
        );

        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let base = format!("http://{}", listener.local_addr()?);
        let listener = service_result(BoundedListener::new(listener, &config.transport))?;
        let _ = Http::new(&base, &instance, limits.clone())?;
        let backend_gate = gate.clone();

        let service = service_result(Service::spawn(
            config,
            Arc::new(Tokenizer),
            logger,
            StartupCheck {
                tokens: vec![vec![100]],
                embeddings: vec![vector(100)],
            },
            move || {
                Ok(ProbeBackend {
                    file,
                    gate: backend_gate,
                    probe,
                    next: 0,
                })
            },
        ))?;
        let app = service.app.clone();
        let router = app.router();
        let (stop, stopped) = oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
        });

        Ok(Self {
            service,
            app,
            control,
            gate,
            server,
            stop: Some(stop),
            base,
            instance,
            limits,
            out: out.to_owned(),
            manifest,
            tasks: JoinSet::new(),
            plans: BTreeMap::new(),
            insertion: Vec::new(),
            responses: BTreeMap::new(),
        })
    }

    async fn submit(&mut self, tokens: &[u32]) -> Result<u64> {
        let now = self.control.now_us();
        let plan = Plan {
            case: format!("sched5758-request-{}", tokens[0]),
            tokens: tokens.to_vec(),
            body: json!({
                "model": MODEL_ID,
                "input": tokens.iter()
                    .map(|token| format!("token-{token}")).collect::<Vec<_>>(),
                "encoding_format": "float"
            }),
        };
        write_new(
            &self.out.join(format!("{}.request.bin", plan.case)),
            &serde_json::to_vec(&plan.body)?,
        )?;
        let client = Http::new(&self.base, &self.instance, self.limits.clone())?;
        let submitted = plan.clone();
        self.tasks.spawn(async move {
            let reply = client
                .send(&submitted.case, &Wire::json(submitted.body.clone())?)
                .await?;
            Ok((submitted, reply))
        });

        let event = find_event(&self.control, |event| {
            event["kind"] == "inserted" && event["case_id"] == plan.case
        })
        .await?;
        let id = uint(&event, "request_id")?;
        ensure!(
            id > 0
                && event["B"] == tokens.len()
                && event["S"] == 1
                && event["inserted_us"] == now
                && event["now_us"] == now
                && event["deadline_us"] == now + REQUEST_US
                && self.plans.insert(id, plan).is_none(),
            "insertion identity/geometry/time"
        );
        self.insertion.push(event);
        Ok(id)
    }

    async fn assemble(&self, ids: &[u64], boundary: u64) -> Result<Value> {
        let first = checkpoint(&self.control, "assembly_start", ids[0]).await?;
        release(&self.control, &first)?;

        let growth = find_event(&self.control, |event| {
            event["kind"] == "assembly" && event["request_ids"] == json!(ids)
        })
        .await?;
        let seq = uint(&growth, "seq")?;
        ensure!(growth["boundary_us"] == boundary, "assembly boundary");
        find_event(&self.control, |event| {
            event["kind"] == "assembly_receive"
                && event["seq"].as_u64().is_some_and(|value| value > seq)
                && event["boundary_us"] == boundary
        })
        .await?;

        service_result(self.control.advance_to_us(boundary))?;
        let assembled = checkpoint(&self.control, "assembled", ids[0]).await?;
        let b: usize = ids.iter().map(|id| self.plans[id].tokens.len()).sum();
        ensure!(
            assembled["request_ids"] == json!(ids)
                && assembled["B"] == b
                && assembled["S"] == 1
                && assembled["now_us"] == boundary
                && assembled["boundary_us"] == boundary,
            "assembled membership/geometry/time"
        );
        release(&self.control, &assembled)?;
        Ok(assembled)
    }

    async fn collect(&mut self, expected: &BTreeMap<u64, Option<Code>>) -> Result<()> {
        let mut seen = BTreeSet::new();
        for _ in 0..expected.len() {
            let joined = timeout(WATCHDOG, self.tasks.join_next())
                .await
                .context("HTTP collection watchdog")?
                .context("missing HTTP task")?;
            let (plan, reply) = joined.context("HTTP task join")??;
            write_new(
                &self.out.join(format!("{}.response.bin", plan.case)),
                &reply.bytes,
            )?;
            let id = reply.request_id.context("admitted HTTP response missing ID")?;
            let code = *expected.get(&id).context("unexpected HTTP response ID")?;
            ensure!(
                seen.insert(id)
                    && self.plans.get(&id).is_some_and(|p| {
                        p.case == plan.case && p.tokens == plan.tokens && p.body == plan.body
                    }),
                "duplicate response or insertion/HTTP mismatch"
            );

            let hashes = if let Some(code) = code {
                check_error(&reply, code)?;
                None
            } else {
                Some(
                    decode(&reply, &plan)?
                        .iter()
                        .map(|row| sha(&row_bytes(row)))
                        .collect::<Vec<_>>(),
                )
            };
            let evidence = json!({
                "request_id": id,
                "case_id": plan.case,
                "http_status": reply.status,
                "code": code,
                "headers": reply.headers,
                "request_sha256": sha(&serde_json::to_vec(&plan.body)?),
                "response_sha256": sha(&reply.bytes),
                "output_sha256": hashes
            });
            ensure!(
                self.responses.insert(id, evidence.clone()).is_none(),
                "response repeated across cohorts"
            );
            save_json(
                &self.out.join(format!("response-{id}.json")),
                &evidence,
            )?;
        }
        ensure!(
            seen == expected.keys().copied().collect(),
            "missing expected HTTP response"
        );
        Ok(())
    }

    async fn control_request(&mut self) -> Result<u64> {
        let id = self.submit(&[90]).await?;
        self.assemble(&[id], 1000).await?;
        self.collect(&BTreeMap::from([(id, None)])).await?;
        let observation = observe(&self.app).await?;
        ensure!(
            observation["forward_calls"] == 1
                && observation["lifecycle"] == "ready"
                && observation["ready"] == true,
            "outside-batch control request did not complete"
        );
        Ok(id)
    }

    /// Cleanup preserves the production quarantine contract.
    ///
    /// On the failure probe, shutdown's DeadlineExceeded is expected and is
    /// recorded as NOT an owner join. Process exit is the cleanup boundary.
    async fn cleanup(&mut self) -> Result<(Value, Option<Code>)> {
        self.app.begin_drain();
        self.gate.release();
        self.control.open_for_cleanup();

        // Cleanup-only advance, after all semantic witnesses have been saved.
        service_result(self.control.advance_to_us(120_000_000))?;
        self.tasks.abort_all();
        while self.tasks.join_next().await.is_some() {}

        // Separate from request-task draining: cancelled request futures can
        // leave connection handles in the strict client's retained inventory.
        let http_connections = http::join_connections(&self.instance).await?;

        let shutdown = self.service.shutdown().await;
        let shutdown_code = shutdown.as_ref().err().copied();

        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let transport = match timeout(WATCHDOG, &mut self.server).await {
            Ok(Ok(Ok(()))) => true,
            _ => {
                self.server.abort();
                let _ = (&mut self.server).await;
                false
            }
        };

        let closure = json!({
            "service_shutdown_returned_ok": shutdown.is_ok(),
            "owner_thread_joined": shutdown.is_ok(),
            "shutdown_error": shutdown_code,
            "http_server_joined": transport,
            "client_tasks_drained": self.tasks.is_empty(),
            "http_connections": http_connections,
            "lifecycle": self.app.lifecycle(),
            "ready": self.app.is_ready(),
            "quarantine_cleared": false,
            "drop_counted_as_join": false,
            "shutdown_timeout_counted_as_join": false,
            "process_exit_required": shutdown.is_err()
        });
        save_json(&self.out.join("owner_closure.json"), &closure)?;
        ensure!(transport, "local HTTP server cleanup failed");
        Ok((closure, shutdown_code))
    }
}

fn expected_rows(local: &Local, ids: &[u64]) -> Result<Vec<Value>> {
    let mut rows = Vec::new();
    for id in ids {
        let plan = local.plans.get(id).context("unknown physical member")?;
        for (request_row, token) in plan.tokens.iter().enumerate() {
            rows.push(json!({
                "physical_row": rows.len(),
                "request_id": id,
                "request_row": request_row,
                "case_id": plan.case,
                "valid_length": 1,
                "token_sha256": ids_sha(&[*token])
            }));
        }
    }
    Ok(rows)
}

/// Reconcile every inserted request, physical row, response, and production
/// request-completion log. No summary boolean substitutes for this coverage.
fn reconcile(
    local: &Local,
    probe: Probe,
    control_id: u64,
    members: &[u64],
) -> Result<Value> {
    let events = service_result(local.control.snapshot())?;
    for (position, event) in events.iter().enumerate() {
        ensure!(event["seq"] == position, "scheduler event sequence gap");
        ensure!(
            matches!(
                event["kind"].as_str(),
                Some("inserted" | "checkpoint" | "assembly" | "assembly_receive" | "launch")
            ),
            "unexpected scheduler event"
        );
    }
    let inserted: Vec<Value> = events
        .iter()
        .filter(|event| event["kind"] == "inserted")
        .cloned()
        .collect();
    ensure!(inserted == local.insertion, "unmatched insertion");

    let launches: Vec<&Value> = events
        .iter()
        .filter(|event| event["kind"] == "launch")
        .collect();
    let bytes = bounded(&local.manifest, MANIFEST_LIMIT)?;
    let records = core::jsonl(&bytes)?;
    ensure!(
        records.len() == 5
            && records[0] == startup_record()
            && launches.len() == 2
            && local.plans.len() == 3
            && local.responses.len() == 3
            && local.tasks.is_empty(),
        "complete request/forward/response coverage"
    );

    let groups = [vec![control_id], members.to_vec()];
    let mut executed = BTreeSet::new();
    for (forward, ids) in groups.iter().enumerate() {
        for id in ids {
            ensure!(executed.insert(*id), "request split or executed twice");
        }
        let rows = expected_rows(local, ids)?;
        let begin = &records[1 + 2 * forward];
        let returned = &records[2 + 2 * forward];
        let b = rows.len();
        ensure!(
            *begin == json!({
                "kind": "begin",
                "forward_id": forward,
                "B": b,
                "S": 1,
                "valid_tokens": b,
                "padding_tokens": 0,
                "input_rows": b,
                "association_rows": b,
                "rows": rows
            })
                && launches[forward]["request_ids"] == json!(ids)
                && launches[forward]["B"] == b
                && launches[forward]["S"] == 1
                && launches[forward]["now_us"] == if forward == 0 { 1000 } else { 2000 },
            "physical input/launch association or timing"
        );

        let mut expected_returned = Vec::new();
        let mut physical = 0usize;
        let mut invalid = Vec::new();
        for id in ids {
            let plan = &local.plans[id];
            for token in &plan.tokens {
                let bad = probe == Probe::Failure && *token == 101;
                let output = if bad {
                    invalid.push(physical);
                    vec![0.0; 2048]
                } else {
                    vector(*token)
                };
                let mut row = rows[physical].clone();
                row["output_sha256"] = json!(sha(&row_bytes(&output)));
                expected_returned.push(row);
                physical += 1;
            }
        }
        ensure!(
            *returned == json!({
                "kind": "returned_rows",
                "forward_id": forward,
                "rows": expected_returned,
                "injected_invalid_output_rows": invalid
            }),
            "backend returned-row coverage"
        );
    }

    ensure!(
        executed == local.responses.keys().copied().collect(),
        "unmatched executed request or HTTP response"
    );
    ensure!(
        local.responses[&control_id]["http_status"] == 200
            && local.responses[&control_id]["code"].is_null(),
        "outside-batch control request affected"
    );

    let expected_codes = match probe {
        Probe::Drain => [None, None],
        Probe::Failure => [Some(Code::InferenceFailed), Some(Code::InferenceFailed)],
        Probe::Expiry => [Some(Code::DeadlineExceeded), Some(Code::ExecutorPoisoned)],
    };
    for (id, code) in members.iter().zip(expected_codes) {
        ensure!(
            local.responses[id]["code"] == json!(code),
            "wrong member production outcome"
        );
    }

    // Logger::complete runs before HTTP task completion. Complete collection
    // therefore permits an exact, stable readback of production completions.
    let log_bytes = bounded(&local.out.join("service_log.jsonl"), JSON_LIMIT)?;
    let logs = core::jsonl(&log_bytes)?;
    ensure!(logs.len() == 3, "missing or duplicated production completion log");
    let mut logged = BTreeSet::new();
    for log in &logs {
        let id = uint(log, "request_id")?;
        let response = local.responses.get(&id).context("unsubmitted logged request")?;
        ensure!(
            logged.insert(id)
                && log["event"] == "request_complete"
                && log["route"] == "embeddings"
                && log["status"] == response["http_status"]
                && log["code"] == response["code"],
            "production completion/HTTP mismatch or duplicate"
        );
    }
    ensure!(logged == executed, "production completion coverage");

    Ok(json!({
        "control_request_id": control_id,
        "physical_failure_or_drain_members": members,
        "physical_partition": groups,
        "physical_rows": [1, 3],
        "logical_requests": 3,
        "physical_forwards": 2,
        "outside_batch_control_unchanged": local.responses[&control_id],
        "production_completion_log_sha256": sha(&log_bytes),
        "backend_manifest_sha256": sha(&bytes),
        "scheduler_events": events,
        "backend_records": records,
        "responses": local.responses,
        "unmatched_insertions": 0,
        "unmatched_physical_rows": 0,
        "missing_responses": 0,
        "duplicate_responses": 0,
        "duplicate_production_completions": 0
    }))
}

async fn exercise(local: &mut Local, probe: Probe) -> Result<Value> {
    service_result(local.service.wait_ready().await)?;
    let before = observe(&local.app).await?;
    ensure!(
        before["instance"] == local.instance
            && before["pid"] == std::process::id()
            && before["lifecycle"] == "ready"
            && before["worker_alive"] == true
            && before["ready"] == true
            && before["forward_calls"] == 0,
        "child initial observation"
    );
    save_json(&local.out.join("observation_before.json"), &before)?;

    // This control belongs to another physical input and completes before
    // destructive work. No queued-work-survives-poison claim is made.
    let control_id = local.control_request().await?;
    let control_response = local.responses[&control_id].clone();

    let first = local.submit(&[100, 101]).await?;
    if probe == Probe::Expiry {
        // Distinct deadlines: only the first logical request expires.
        service_result(local.control.advance_to_us(1100))?;
    }
    let second = local.submit(&[102]).await?;
    let members = [first, second];
    let assembled = local.assemble(&members, 2000).await?;
    local.gate.wait_entered().await?;

    let running_events = service_result(local.control.snapshot())?;
    let running_bytes = bounded(&local.manifest, MANIFEST_LIMIT)?;
    let running_records = core::jsonl(&running_bytes)?;
    ensure!(
        running_records.len() == 4
            && running_records[3]["kind"] == "begin"
            && running_records[3]["forward_id"] == 1
            && running_records[3]["rows"] == json!(expected_rows(local, &members)?)
            && local.responses.len() == 1
            && local.tasks.len() == 2,
        "work not held inside physical backend execution"
    );
    save_json(&local.out.join("running_events.json"), &json!(running_events))?;
    write_new(&local.out.join("running_manifest.jsonl"), &running_bytes)?;

    let witness = match probe {
        Probe::Drain => {
            // Poll actual shutdown once. It must begin draining and wait on
            // admission retained by the executing physical batch.
            let shutdown = local.service.shutdown();
            tokio::pin!(shutdown);
            let first_poll = poll_fn(|cx| Poll::Ready(shutdown.as_mut().poll(cx))).await;
            ensure!(
                first_poll.is_pending()
                    && local.app.lifecycle() == Lifecycle::Draining
                    && !local.app.is_ready()
                    && bounded(&local.manifest, MANIFEST_LIMIT)? == running_bytes,
                "production drain did not wait for running work"
            );
            local.gate.release();
            service_result(
                timeout(WATCHDOG, shutdown.as_mut())
                    .await
                    .context("running drain watchdog")?,
            )?;
            ensure!(
                local.app.lifecycle() == Lifecycle::Stopped && !local.app.is_ready(),
                "drain did not stop after owner join"
            );
            json!({
                "shutdown_initial_poll_pending": true,
                "running_batch_held_when_drain_started": true,
                "production_shutdown_completed": true,
                "owner_thread_joined_by_measured_shutdown": true,
                "outcome": "running_work_completed_before_owner_join"
            })
        }
        Probe::Failure => {
            // Supplemental verification of the exact production validator.
            // The actual probe separately traverses run_batch -> validate_output.
            let valid = vec![vector(100), vector(101), vector(102)];
            service_result(validate_output(&valid, 3))?;
            let mut invalid = valid.clone();
            invalid[1] = vec![0.0; 2048];
            ensure!(
                validate_output(&invalid, 3) == Err(Code::InferenceFailed)
                    && validate_output(&[invalid[0].clone(), invalid[2].clone()], 2).is_ok(),
                "single-row failure stimulus not isolated"
            );
            local.gate.release();
            json!({
                "fault": "one_zero_norm_physical_output_row",
                "fault_physical_row": 1,
                "fault_request_id": first,
                "fault_request_row": 1,
                "production_failure_gate": "http_serve::run_batch -> validate_output",
                "backend_implements_fanout": false,
                "backend_implements_poison": false,
                "queued_requests_outside_failed_batch": 0,
                "queued_requests_survive_poison_claimed": false
            })
        }
        Probe::Expiry => {
            let deadline = 1000 + REQUEST_US;
            ensure!(
                local.insertion[1]["deadline_us"] == deadline
                    && local.insertion[2]["deadline_us"] == deadline + 100,
                "staggered running deadlines"
            );
            service_result(local.control.advance_to_us(deadline))?;

            // Collect while backend remains blocked. This proves the HTTP
            // running-deadline branch retires the executor independently of
            // backend return, rather than merely filtering queued requests.
            local
                .collect(&BTreeMap::from([
                    (first, Some(Code::DeadlineExceeded)),
                    (second, Some(Code::ExecutorPoisoned)),
                ]))
                .await?;
            ensure!(
                local.app.lifecycle() == Lifecycle::Retired
                    && !local.app.is_ready()
                    && bounded(&local.manifest, MANIFEST_LIMIT)? == running_bytes
                    && service_result(local.control.snapshot())? == running_events,
                "running expiry did not retire while backend remained executing"
            );
            save_json(
                &local.out.join("retired_while_backend_held.json"),
                &json!({
                    "now_us": local.control.now_us(),
                    "lifecycle": local.app.lifecycle(),
                    "ready": local.app.is_ready(),
                    "backend_manifest_sha256": sha(&running_bytes),
                    "responses": local.responses
                }),
            )?;
            local.gate.release();
            json!({
                "running_deadline_expiry": "retire_executor",
                "expired_request_id": first,
                "expired_deadline_us": deadline,
                "other_member_request_id": second,
                "other_member_deadline_us": deadline + 100,
                "retired_before_backend_return": true,
                "http_success_after_expiry": false,
                "production_error_codes": ["deadline_exceeded", "executor_poisoned"]
            })
        }
    };

    if probe != Probe::Expiry {
        let code = if probe == Probe::Failure {
            Some(Code::InferenceFailed)
        } else {
            None
        };
        local
            .collect(&BTreeMap::from([(first, code), (second, code)]))
            .await?;
    }

    let after = if probe == Probe::Drain {
        // Stop consumed the owner queue; observe cannot be sent after join.
        json!({
            "lifecycle": local.app.lifecycle(),
            "ready": local.app.is_ready(),
            "owner_already_joined": true
        })
    } else {
        // Owner observation is a barrier after return from the held backend.
        let after = observe(&local.app).await?;
        let mut expected = before.clone();
        expected["forward_calls"] = json!(3);
        expected["ready"] = json!(false);
        expected["lifecycle"] = json!(if probe == Probe::Failure {
            "poisoned"
        } else {
            "retired"
        });
        ensure!(after == expected, "terminal child observation");
        after
    };
    ensure!(
        local.responses[&control_id] == control_response,
        "outside-batch control response changed"
    );
    let reconciliation = reconcile(local, probe, control_id, &members)?;
    save_json(&local.out.join("observation_after.json"), &after)?;

    Ok(json!({
        "case_id": probe.case(),
        "instance": local.instance,
        "pid": std::process::id(),
        "http_base": local.base,
        "batching": BatchedExecution::default(),
        "semantic_clock": "manual_microseconds",
        "request_timeout_us": REQUEST_US,
        "request_timeout_shortened": false,
        "wall_watchdogs_are_semantic_time": false,
        "observation_before": before,
        "observation_after": after,
        "assembled": assembled,
        "witness": witness,
        "reconciliation": reconciliation,
        "production_source_sha256": sha(include_bytes!("http_serve.rs")),
        "production_scheduler_used": true,
        "backend": "synthetic_scheduler_probe",
        "numerical_model_evidence": false
    }))
}

async fn child_probe(spec: &ChildSpec, probe: Probe) -> Result<Value> {
    let limits: Limits = serde_json::from_value(spec.limits.clone())?;
    let mut local = Local::create(limits, &spec.out, probe).await?;

    let work = exercise(&mut local, probe).await;
    // Save semantic results before cleanup's clock advance or shutdown wait.
    if let Ok(evidence) = &work {
        save_json(&spec.out.join("semantic_evidence.json"), evidence)?;
    }

    let cleanup = local.cleanup().await;
    let final_events = service_result(local.control.snapshot());
    if let Ok(events) = &final_events {
        save_json(&spec.out.join("final_scheduler_events.json"), &json!(events))?;
    }
    if let Err(error) = &work {
        let _ = save_json(
            &spec.out.join("work_failure.json"),
            &json!({"error": format!("{error:#}")}),
        );
    }

    let mut evidence = work?;
    let (closure, shutdown_code) = cleanup?;
    let events = final_events?;
    ensure!(
        evidence["reconciliation"]["scheduler_events"] == json!(events)
            && evidence["reconciliation"]["backend_manifest_sha256"]
                == sha(&bounded(&local.manifest, MANIFEST_LIMIT)?),
        "additional execution or scheduler activity during cleanup"
    );

    match probe {
        Probe::Failure => {
            ensure!(
                shutdown_code == Some(Code::DeadlineExceeded)
                    && closure["owner_thread_joined"] == false
                    && closure["quarantine_cleared"] == false
                    && local.app.lifecycle() == Lifecycle::Poisoned,
                "operational quarantine was cleared or timeout counted as join"
            );
        }
        Probe::Drain | Probe::Expiry => {
            ensure!(
                shutdown_code.is_none() && closure["owner_thread_joined"] == true,
                "non-quarantined child owner did not join"
            );
        }
    }
    evidence["cleanup"] = closure;
    evidence["process_termination_is_quarantine_cleanup"] =
        json!(probe == Probe::Failure);
    Ok(evidence)
}

/// Exit codes:
///   0: semantic checks and required child-local cleanup checks passed
///   1: probe/setup/evidence failure
///   2: invalid child protocol
/// 124: independent child watchdog
///
/// For operational failure, exit 0 is NOT Service::shutdown success. The
/// evidence explicitly records its expected DeadlineExceeded and lack of join.
pub fn child_entry(raw: &str) -> ! {
    let parsed: Result<(ChildSpec, Probe)> = (|| {
        ensure!(raw.len() <= 16 * 1024, "child spec too large");
        let spec: ChildSpec = serde_json::from_str(raw)?;
        let probe = Probe::parse(&spec.case)?;
        ensure!(
            label_valid(&spec.nonce)
                && spec.parent_pid > 0
                && spec.parent_pid != std::process::id()
                && spec.out.is_absolute()
                && spec.out.canonicalize()? == spec.out
                && fs::symlink_metadata(&spec.out)?.is_dir(),
            "invalid child identity/directory"
        );
        Ok((spec, probe))
    })();
    let (spec, probe) = match parsed {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("invalid destructive child protocol: {error:#}");
            std::process::exit(2);
        }
    };

    // Independent of Tokio progress, backend progress, and runtime shutdown.
    let watchdog_out = spec.out.clone();
    std::thread::spawn(move || {
        std::thread::sleep(CHILD_BUDGET);
        let _ = save_json(
            &watchdog_out.join("watchdog.json"),
            &json!({"exit_code": 124, "semantic_time_evidence": false}),
        );
        std::process::exit(124);
    });

    let result: Result<()> = (|| {
        save_json(
            &spec.out.join("child_identity.json"),
            &json!({
                "pid": std::process::id(),
                "parent_pid": spec.parent_pid,
                "nonce": spec.nonce,
                "case_id": spec.case,
                "evidence_directory": spec.out,
                "binary_sha256": file_sha(&std::env::current_exe()?)?
            }),
        )?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        let result = runtime.block_on(child_probe(&spec, probe));
        match result {
            Ok(probe) => {
                save_json(
                    &spec.out.join("evidence.json"),
                    &json!({
                        "schema": "rustv1.accept_m4_destructive_child.v1",
                        "case_id": spec.case,
                        "nonce": spec.nonce,
                        "pid": std::process::id(),
                        "parent_pid": spec.parent_pid,
                        "evidence_directory": spec.out,
                        "probe": probe,
                        "child_checks_passed": true
                    }),
                )?;
                // Never wait for runtime destruction as a substitute for
                // terminating a quarantined child. The parent must reap us.
                std::mem::forget(runtime);
                Ok(())
            }
            Err(error) => {
                std::mem::forget(runtime);
                Err(error)
            }
        }
    })();

    match result {
        Ok(()) => std::process::exit(0),
        Err(error) => {
            let _ = save_json(
                &spec.out.join("failure.json"),
                &json!({"error": format!("{error:#}"), "exit_code": 1}),
            );
            eprintln!("destructive child failed: {error:#}");
            std::process::exit(1);
        }
    }
}

/// Also reaps on cancellation or an unexpected parent-side error.
/// Normal success uses try_wait, which reaps and returns the exit status.
struct ReapChild(Option<Child>);

impl Drop for ReapChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn child_files(out: &Path) -> Result<Value> {
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(out)? {
        let entry = entry?;
        ensure!(entry.file_type()?.is_file(), "unexpected child evidence entry");
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 child evidence name"))?;
        files.insert(name, file_sha(&entry.path())?);
    }
    Ok(json!(files))
}

async fn spawn_probe(case: &str, out: &Path, limits: Value) -> Result<Value> {
    Probe::parse(case)?;
    let child_out = out.join("child");
    private_directory(&child_out)?;
    let child_out = child_out.canonicalize()?;
    let nonce = core::random_label()?;
    let spec = json!({
        "case": case,
        "out": child_out,
        "nonce": nonce,
        "limits": limits,
        "parent_pid": std::process::id()
    });
    save_json(&out.join("child_spec.json"), &spec)?;
    let executable = std::env::current_exe()?;
    let binary_sha = file_sha(&executable)?;
    let raw = serde_json::to_string(&spec)?;

    let mut command = Command::new(&executable);
    // Do not inherit acceptance credentials, live endpoint configuration,
    // orchestration ports, or policy paths into the child.
    command.env_clear();
    command.stdin(Stdio::null());
    command.stdout(Stdio::from(private_file(&out.join("child_stdout.log"))?));
    command.stderr(Stdio::from(private_file(&out.join("child_stderr.log"))?));

    #[cfg(test)]
    {
        command
            .arg("--exact")
            .arg(CHILD_TEST)
            .arg("--ignored")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(CHILD_ENV, &raw);
    }
    #[cfg(not(test))]
    {
        command.arg(CHILD_COMMAND).arg(&raw);
    }

    let child = command.spawn().context("spawn destructive child")?;
    let pid = child.id();
    ensure!(pid != std::process::id(), "child PID isolation");
    let mut child = ReapChild(Some(child));
    let deadline = Instant::now() + PARENT_BUDGET;
    let mut watchdog = false;

    let status = loop {
        let process = child.0.as_mut().context("missing child process")?;
        if let Some(status) = process.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            watchdog = true;
            // Kill only the owned child handle, never the live server PID.
            process.kill().context("kill timed-out destructive child")?;
            break process.wait().context("reap timed-out destructive child")?;
        }
        sleep(Duration::from_millis(10)).await;
    };
    child.0.take();

    let exit = json!({
        "pid": pid,
        "parent_pid": std::process::id(),
        "exit_code": status.code(),
        "success": status.success(),
        "parent_watchdog_fired": watchdog,
        "child_process_reaped": true,
        "process_drop_counted_as_reap": false,
        "evidence_directory": child_out,
        "binary_sha256": binary_sha
    });
    save_json(&out.join("child_exit.json"), &exit)?;
    ensure!(
        !watchdog && status.code() == Some(0),
        "destructive child failed; retained evidence at {}",
        out.display()
    );

    // Read only after process exit/reap: the child cannot mutate its evidence.
    let evidence = value(&child_out.join("evidence.json"))?;
    let identity = value(&child_out.join("child_identity.json"))?;
    ensure!(
        evidence["schema"] == "rustv1.accept_m4_destructive_child.v1"
            && evidence["case_id"] == case
            && evidence["nonce"] == nonce
            && evidence["pid"] == pid
            && evidence["parent_pid"] == std::process::id()
            && evidence["evidence_directory"] == json!(child_out)
            && evidence["child_checks_passed"] == true
            && identity["binary_sha256"] == binary_sha
            && identity["pid"] == pid
            && identity["nonce"] == nonce,
        "child evidence identity/protocol mismatch"
    );
    ensure!(
        file_sha(&executable)? == binary_sha,
        "child executable changed"
    );

    Ok(json!({
        "exit": exit,
        "identity": identity,
        "evidence": evidence,
        "child_output_sha256": child_files(&child_out)?
    }))
}

#[derive(Clone)]
struct Preserved {
    observation: Value,
    manifest: Vec<u8>,
    next_forward: u64,
    request_ids: BTreeSet<u64>,
}

impl Preserved {
    fn capture(state: &SchedulerState) -> Self {
        Self {
            observation: state.observation.clone(),
            manifest: state.manifest_prefix.clone(),
            next_forward: state.next_forward,
            request_ids: state.seen_request_ids.clone(),
        }
    }

    async fn verify(&self, state: &SchedulerState) -> Result<()> {
        state.launch.process.check_live()?;
        ensure!(
            state.observation == self.observation
                && state.manifest_prefix == self.manifest
                && state.next_forward == self.next_forward
                && state.seen_request_ids == self.request_ids,
            "retained live reconciliation state changed"
        );
        ensure!(
            state.core.control.call("observe", None).await? == self.observation
                && bounded(&state.launch.manifest, MANIFEST_LIMIT)? == self.manifest,
            "live server observation or manifest changed"
        );
        state.launch.process.check_live()?;
        Ok(())
    }

    fn evidence(&self) -> Value {
        json!({
            "observation": self.observation,
            "manifest_sha256": sha(&self.manifest),
            "manifest_bytes": self.manifest.len(),
            "next_forward": self.next_forward,
            "request_ids": self.request_ids
        })
    }
}

async fn execute_case(
    state: &mut SchedulerState,
    preserved: &Preserved,
    case: &str,
) -> Result<()> {
    ensure!(
        state.core.ledger.pending().first()
            == Some(&json!({
                "acceptance_phase": policy::SCHEDULER,
                "case_id": case
            })),
        "destructive runner called at wrong global ledger position"
    );

    let result: Result<Value> = async {
        state.core.loaded.recheck()?;
        envelope(&state.core.loaded)?;
        ensure!(
            state.core.loaded.policy["deployment"]["scheduler"]["running_deadline_expiry"]
                == "retire_executor"
                && state.core.loaded.policy["deployment"]["scheduler"]["physical_failure"]
                    == "fail_all_members_and_poison",
            "destructive policy binding"
        );
        preserved.verify(state).await?;

        let out = state.directory.join(case);
        private_directory(&out)?;
        let out = out.canonicalize()?;
        let child_dir = out.join("child");
        ensure!(
            !state.launch.manifest.starts_with(&child_dir)
                && !state.launch.attestation.starts_with(&child_dir)
                && !state.core.loaded.policy_path.starts_with(&child_dir),
            "child evidence aliases live or policy evidence"
        );
        save_json(&out.join("live_before.json"), &preserved.evidence())?;

        // Do not propagate child failure before the post-child isolation check.
        let child = spawn_probe(
            case,
            &out,
            state.core.loaded.policy["limits"].clone(),
        )
        .await;
        let isolation = preserved.verify(state).await;
        save_json(
            &out.join("live_after_check.json"),
            &json!({
                "unchanged": isolation.is_ok(),
                "preserved": preserved.evidence(),
                "error": isolation.as_ref().err().map(|e| format!("{e:#}"))
            }),
        )?;
        isolation?;
        let child = child?;

        ensure!(
            child["exit"]["pid"] != state.launch.pid
                && child["exit"]["child_process_reaped"] == true,
            "child/live process isolation or missing reap"
        );
        state.core.loaded.recheck()?;
        preserved.verify(state).await?;
        state.core.orchestration.sync()?;

        let evidence = json!({
            "schema": "rustv1.accept_m4_destructive_case.v1",
            "case_id": case,
            "policy_sha256": state.core.loaded.policy_sha256,
            "driver_binary_sha256": state.core.loaded.artifacts["accept_m4"].sha256,
            "live_instance": state.launch.instance,
            "live_pid": state.launch.pid,
            "live_reconciliation_unchanged": preserved.evidence(),
            "child_evidence_directory_separate": true,
            "child": child,
            "child_process_reaped_before_ledger_pass": true,
            "production_scheduler_used": true,
            "backend": "synthetic_scheduler_probe",
            "numerical_model_evidence": false
        });
        let digest = save_json(&out.join("evidence.json"), &evidence)?;
        Ok(json!({
            "evidence_path": format!("scheduler/{case}/evidence.json"),
            "evidence_sha256": digest,
            "child_process_reaped": true,
            "live_reconciliation_unchanged": true,
            "production_scheduler_used": true,
            "numerical_model_evidence": false
        }))
    }
    .await;

    match result {
        Ok(details) => {
            state.core.ledger.append(policy::SCHEDULER, case, true, details, true)?;
            Ok(())
        }
        Err(error) => {
            let _ = state.core.ledger.append(
                policy::SCHEDULER,
                case,
                false,
                json!({
                    "error": format!("{error:#}"),
                    "retained_evidence_path": format!("scheduler/{case}")
                }),
                true,
            );
            Err(error)
        }
    }
}

async fn execute(mut state: SchedulerState) -> Result<SchedulerState> {
    let preserved = Preserved::capture(&state);
    for case in [DRAIN, FAILURE, EXPIRY] {
        execute_case(&mut state, &preserved, case).await?;
    }
    preserved.verify(&state).await?;
    ensure!(
        state.core.ledger.pending().first()
            == Some(&json!({
                "acceptance_phase": policy::SCHEDULER,
                "case_id": NEXT
            })),
        "unexpected post-destructive ledger boundary"
    );
    Ok(state)
}

pub fn run(continuation: SchedulerContinuation) -> Result<SchedulerContinuation> {
    let SchedulerContinuation { runtime, state } = continuation;
    let state = runtime.block_on(execute(state))?;
    Ok(SchedulerContinuation { runtime, state })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits_value() -> Value {
        json!({
            "connect_ms": 5000,
            "headers_ms": 65000,
            "body_ms": 10000,
            "control_ms": 30000,
            "observation_ms": 2000,
            "server_request_ms": 60000,
            "response_bytes": 1048576,
            "admission_capacity": 64,
            "queue_capacity": 64
        })
    }

    /// Invoked only by the parent harness in a separate libtest process.
    #[test]
    #[ignore = "child-process protocol; invoked by the destructive parent harness"]
    fn scheduler_destructive_child_process() {
        child_entry(&std::env::var(CHILD_ENV).unwrap_or_default());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn production_scheduler_family57_58_destructive_probes() {
        let root = std::env::temp_dir().join(format!(
            "rustv1-family5758-{}-{}",
            std::process::id(),
            core::random_label().unwrap()
        ));
        private_directory(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let live_out = root.join("parent-live");
        private_directory(&live_out).unwrap();

        // A real, independently owned parent-local Service/HTTP server remains
        // alive during every child. Seed nonempty reconciliation state.
        let mut live = Local::create(
            serde_json::from_value(limits_value()).unwrap(),
            &live_out,
            Probe::Drain,
        )
        .await
        .unwrap();

        let work: Result<()> = async {
            service_result(live.service.wait_ready().await)?;
            let id = live.control_request().await?;
            let observation = observe(&live.app).await?;
            let manifest = bounded(&live.manifest, MANIFEST_LIMIT)?;
            let next_forward = 1u64;
            let request_ids = BTreeSet::from([id]);
            let retained_next = next_forward;
            let retained_ids = request_ids.clone();

            for case in [DRAIN, FAILURE, EXPIRY] {
                let out = root.join(case);
                private_directory(&out)?;
                let child = spawn_probe(case, &out, limits_value()).await;

                // As in the continuation runner, recheck even on child error.
                ensure!(
                    observe(&live.app).await? == observation
                        && bounded(&live.manifest, MANIFEST_LIMIT)? == manifest
                        && next_forward == retained_next
                        && request_ids == retained_ids
                        && live.responses.keys().copied().collect::<BTreeSet<_>>()
                            == request_ids,
                    "child changed parent-live reconciliation state"
                );
                let child = child?;
                ensure!(
                    child["exit"]["child_process_reaped"] == true
                        && child["exit"]["pid"] != std::process::id()
                        && child["evidence"]["evidence_directory"]
                            != json!(live_out),
                    "harness process/directory isolation"
                );

                let probe = &child["evidence"]["probe"];
                ensure!(
                    probe["reconciliation"]["missing_responses"] == 0
                        && probe["reconciliation"]["duplicate_responses"] == 0
                        && probe["reconciliation"]["duplicate_production_completions"] == 0
                        && probe["cleanup"]["http_server_joined"] == true,
                    "destructive reconciliation/transport summary"
                );
                match Probe::parse(case)? {
                    Probe::Drain => ensure!(
                        probe["witness"]["shutdown_initial_poll_pending"] == true
                            && probe["witness"]["owner_thread_joined_by_measured_shutdown"] == true,
                        "running drain witness"
                    ),
                    Probe::Failure => ensure!(
                        probe["cleanup"]["owner_thread_joined"] == false
                            && probe["cleanup"]["shutdown_error"] == "deadline_exceeded"
                            && probe["cleanup"]["quarantine_cleared"] == false
                            && probe["observation_after"]["lifecycle"] == "poisoned",
                        "failure quarantine/poison witness"
                    ),
                    Probe::Expiry => ensure!(
                        probe["witness"]["retired_before_backend_return"] == true
                            && probe["observation_after"]["lifecycle"] == "retired"
                            && probe["cleanup"]["owner_thread_joined"] == true,
                        "running expiry witness"
                    ),
                }
            }
            Ok(())
        }
        .await;

        let cleanup = live.cleanup().await;
        let result = work.and_then(|()| {
            let (closure, code) = cleanup?;
            ensure!(
                code.is_none()
                    && closure["owner_thread_joined"] == true
                    && closure["http_server_joined"] == true,
                "parent-live owner cleanup"
            );
            Ok(())
        });

        if result.is_ok() {
            fs::remove_dir_all(&root).unwrap();
        }
        result.unwrap_or_else(|error| {
            panic!(
                "destructive family-57/58 harness failed; evidence retained at {}: {error:#}",
                root.display()
            )
        });
    }
}
