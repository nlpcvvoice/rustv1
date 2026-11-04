// Included inside acceptance_scheduler::physical_b::family56_57.
//
// The enclosing module is gated by scheduler-test-seam.
// Production scheduling and token validation are unchanged.
// No CUDA backend is constructed.
//
// Local HTTP IDs remain local: they never enter the retained live server's
// request-ID set.

use super::*;
use rustv1::http_serve::{validate_tokens, PAD_ID};
use std::{
    fs::File,
    future::{poll_fn, Future},
    io::Write,
    task::Poll,
};

const INVALID_LAST: &str = "family-56-invalid-last-isolation";
const TOKEN_GATES: &str = "family-56-pad-and-oov-token-gates";
const QUIESCE: &str = "family-57-quiesce-assembled-work";
const NEXT: &str = "family-57-drain-running-work";

const PAD_GATE: &str =
    "http_serve::validate_tokens: id == PAD_ID -> Code::PadToken";
const OOV_GATE: &str =
    "http_serve::validate_tokens: \
     usize::try_from(id).map_or(true, |n| n >= config.vocabulary_size) \
     -> Code::OutOfVocabulary";
const CLOSED_GATE: &str =
    "http_serve::App::admit: AdmissionMode::Quiesced -> Code::NotReady";

const REQUEST_US: u64 = 60_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Probe {
    InvalidLast,
    Pad,
    Oov,
    Quiesce,
}

impl Probe {
    fn name(self) -> &'static str {
        match self {
            Self::InvalidLast => "invalid-last",
            Self::Pad => "pad",
            Self::Oov => "oov",
            Self::Quiesce => "quiesce",
        }
    }

    fn has_work(self) -> bool {
        matches!(self, Self::InvalidLast | Self::Quiesce)
    }
}

/// Records actual successful tokenizer returns, including invalid token IDs.
///
/// This tokenizer introduces no token validity predicate. The authoritative
/// checks remain in production App::request -> validate_tokens.
struct TraceTokenizer {
    trace: Arc<Mutex<Vec<Value>>>,
}

impl TextTokenizer for TraceTokenizer {
    fn encode(&self, text: &str) -> std::result::Result<Vec<u32>, Code> {
        let tokens = if let Some(ids) = text.strip_prefix("ids-") {
            ids.split('-')
                .map(|part| part.parse::<u32>().map_err(|_| Code::InvalidRequest))
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            Tokenizer.encode(text)?
        };
        guard(&self.trace).push(json!({
            "text": text,
            "tokens": tokens
        }));
        Ok(tokens)
    }
}

/// Persist the enclosing recording backend's records on every backend call.
///
/// The file exists before startup and is updated by the backend, not
/// synthesized after asserting zero execution.
struct FileBackend {
    inner: ControlledBackend,
    file: File,
    written: usize,
}

impl FileBackend {
    fn persist(&mut self) -> std::result::Result<(), Code> {
        let records = guard(&self.inner.recording).records.clone();
        for record in &records[self.written..] {
            serde_json::to_writer(&mut self.file, record)
                .map_err(|_| Code::InferenceFailed)?;
            self.file
                .write_all(b"\n")
                .map_err(|_| Code::InferenceFailed)?;
        }
        self.file.sync_data().map_err(|_| Code::InferenceFailed)?;
        self.written = records.len();
        Ok(())
    }
}

impl Backend for FileBackend {
    fn forward(
        &mut self,
        batches: &[Vec<u32>],
    ) -> std::result::Result<Vec<Vec<f32>>, Code> {
        let output = self.inner.forward(batches)?;
        self.persist()?;
        Ok(output)
    }

    fn forward_batch(
        &mut self,
        batches: &[Vec<u32>],
        associations: &[ExecutionRow],
    ) -> std::result::Result<Vec<Vec<f32>>, Code> {
        let output = self.inner.forward_batch(batches, associations)?;
        self.persist()?;
        Ok(output)
    }

    fn inject_policy_drift(&mut self) -> std::result::Result<(), Code> {
        Err(Code::InvalidRequest)
    }

    fn restore_policy(&mut self) -> std::result::Result<(), Code> {
        Err(Code::InvalidRequest)
    }
}

fn manifest_file(path: &Path) -> Result<File> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

fn record_bytes(records: &[Value]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for record in records {
        serde_json::to_writer(&mut bytes, record)?;
        bytes.push(b'\n');
    }
    Ok(bytes)
}

fn startup_record() -> Value {
    json!({
        "kind": "controlled_startup",
        "backend": "synthetic_scheduler_probe",
        "tokens": [[100]],
        "output_sha256": [sha(&row_bytes(&vector(100)))]
    })
}

fn check_error(reply: &http::Reply, code: Code) -> Result<()> {
    let (message, param, kind) = match code {
        Code::PadToken => (
            "Input contains a padding token",
            Some("input"),
            "invalid_request_error",
        ),
        Code::OutOfVocabulary => (
            "Input contains an out-of-vocabulary token",
            Some("input"),
            "invalid_request_error",
        ),
        Code::NotReady => (
            "Service is not ready",
            None,
            "service_unavailable_error",
        ),
        _ => anyhow::bail!("unplanned family-56/57 rejection"),
    };
    ensure!(
        reply.status == code.status().as_u16()
            && serde_json::from_slice::<Value>(&reply.bytes)?
                == json!({
                    "error": {
                        "message": message,
                        "type": kind,
                        "param": param,
                        "code": code
                    }
                }),
        "rejection status/envelope or partial success payload"
    );
    Ok(())
}

type Successful = BTreeMap<u64, (Plan, Vec<Vec<f32>>)>;

struct Driver {
    base: String,
    instance: String,
    limits: Limits,
    out: PathBuf,
    control: SchedulerControl,
    tasks: JoinSet<Result<(Plan, http::Reply)>>,
    insertion: Vec<Value>,
    responses: Vec<Value>,
    rejections: Vec<Value>,
    exposed_ids: BTreeSet<u64>,
    expected_trace: Vec<Value>,
}

impl Driver {
    async fn submit(&mut self, token: u32) -> Result<()> {
        let plan = Plan {
            case: format!("sched5657-valid-{token}"),
            tokens: vec![token],
            body: json!({
                "model": MODEL_ID,
                "input": [format!("token-{token}")],
                "encoding_format": "float"
            }),
        };
        write_new(
            &self.out.join(format!("{}.request.bin", plan.case)),
            &serde_json::to_vec(&plan.body)?,
        )?;
        self.expected_trace.push(json!({
            "text": format!("token-{token}"),
            "tokens": [token]
        }));
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
                && event["B"] == 1
                && event["S"] == 1
                && event["inserted_us"] == 0
                && event["now_us"] == 0
                && event["deadline_us"] == REQUEST_US
                && self.insertion.iter().all(|old| old["request_id"] != id),
            "valid insertion identity/geometry/time"
        );
        self.insertion.push(event);
        Ok(())
    }

    async fn rejection(
        &mut self,
        case: &str,
        body: Value,
        code: Code,
        gate: &str,
    ) -> Result<Value> {
        let request = serde_json::to_vec(&body)?;
        write_new(
            &self.out.join(format!("{case}.request.bin")),
            &request,
        )?;
        let client = Http::new(&self.base, &self.instance, self.limits.clone())?;
        let reply = client.send(case, &Wire::json(body)?).await?;
        write_new(
            &self.out.join(format!("{case}.response.bin")),
            &reply.bytes,
        )?;
        check_error(&reply, code)?;

        if code == Code::NotReady {
            ensure!(
                reply.request_id.is_none(),
                "closed HTTP admission unexpectedly exposed an ID"
            );
        } else {
            // Token rejection occurs after HTTP admission, not before ID
            // allocation. Use only the actual response ID.
            let id = reply.request_id.context("token rejection missing HTTP ID")?;
            ensure!(
                self.exposed_ids.insert(id)
                    && self.insertion.iter().all(|event| event["request_id"] != id),
                "rejected ID reused or present in scheduler insertion"
            );
        }
        ensure!(
            service_result(self.control.snapshot())?
                .iter()
                .all(|event| event["case_id"] != case),
            "rejected request entered scheduler"
        );
        let evidence = json!({
            "case_id": case,
            "request_id": reply.request_id,
            "http_status": reply.status,
            "headers": reply.headers,
            "request_sha256": sha(&request),
            "response_sha256": sha(&reply.bytes),
            "rejection_gate": gate,
            "code": code,
            "scheduler_inserted": false,
            "fabricated_request_id": false
        });
        self.rejections.push(evidence.clone());
        Ok(evidence)
    }

    async fn token_rejection(&mut self, app: &App, probe: Probe) -> Result<Value> {
        let (sequences, code, gate) = match probe {
            Probe::InvalidLast => (
                vec![vec![102], vec![103, 104, 131072]],
                Code::OutOfVocabulary,
                OOV_GATE,
            ),
            Probe::Pad => (vec![vec![105, PAD_ID]], Code::PadToken, PAD_GATE),
            Probe::Oov => (
                vec![vec![106, 131072]],
                Code::OutOfVocabulary,
                OOV_GATE,
            ),
            Probe::Quiesce => anyhow::bail!("quiesce is not a token rejection"),
        };
        ensure!(
            app.config.vocabulary_size == 131072
                && PAD_ID == 11
                && app.config.max_sequence_tokens == 1024,
            "token qualification envelope"
        );

        // Call the actual public production validator, not a replacement
        // predicate. Removing only the final token must make the request
        // valid; the complete request must return the exact planned Code.
        let mut valid_prefix = sequences.clone();
        valid_prefix.last_mut().context("last sequence")?.pop();
        service_result(validate_tokens(&valid_prefix, &app.config))?;
        ensure!(
            validate_tokens(&sequences, &app.config) == Err(code),
            "production token gate differs from planned rejection"
        );

        let texts: Vec<String> = sequences
            .iter()
            .map(|ids| {
                format!(
                    "ids-{}",
                    ids.iter().map(u32::to_string).collect::<Vec<_>>().join("-")
                )
            })
            .collect();
        for (text, tokens) in texts.iter().zip(&sequences) {
            self.expected_trace.push(json!({
                "text": text,
                "tokens": tokens
            }));
        }
        let case = format!("sched5657-{}", probe.name());
        let mut evidence = self
            .rejection(
                &case,
                json!({
                    "model": MODEL_ID,
                    "input": texts,
                    "encoding_format": "float"
                }),
                code,
                gate,
            )
            .await?;
        evidence["tokenized_sequences"] = json!(sequences);
        evidence["valid_without_final_token"] = json!(valid_prefix);
        evidence["production_validator_called_directly_as_supplement"] = json!(true);
        evidence["http_path_exercised_separately"] = json!(true);
        evidence["rejection_before_backend_infer"] = json!(true);
        evidence["http_admission_acquired"] = json!(true);
        evidence["executor_validation_gate_claimed"] = json!(false);
        Ok(evidence)
    }

    async fn closed_attempt(&mut self, name: &str) -> Result<Value> {
        self.rejection(
            name,
            json!({
                "model": MODEL_ID,
                "input": ["token-107"],
                "encoding_format": "float"
            }),
            Code::NotReady,
            CLOSED_GATE,
        )
        .await
    }

    async fn collect(&mut self) -> Result<Successful> {
        let mut successful = BTreeMap::new();
        while let Some(joined) = timeout(WATCHDOG, self.tasks.join_next())
            .await
            .context("HTTP task watchdog; not semantic time evidence")?
        {
            let (plan, reply) = joined.context("family-56/57 HTTP task")??;
            write_new(
                &self.out.join(format!("{}.response.bin", plan.case)),
                &reply.bytes,
            )?;
            let id = reply.request_id.context("successful HTTP response ID")?;
            let ack = self
                .insertion
                .iter()
                .find(|event| event["case_id"] == plan.case)
                .context("successful response without insertion")?;
            ensure!(
                ack["request_id"] == id && self.exposed_ids.insert(id),
                "HTTP/insertion identity mismatch or duplicate exposed ID"
            );
            let output = decode(&reply, &plan)?;
            self.responses.push(json!({
                "case_id": plan.case,
                "request_id": id,
                "http_status": reply.status,
                "headers": reply.headers,
                "request_sha256": sha(&serde_json::to_vec(&plan.body)?),
                "response_sha256": sha(&reply.bytes),
                "output_sha256": output.iter()
                    .map(|row| sha(&row_bytes(row))).collect::<Vec<_>>()
            }));
            ensure!(
                successful.insert(id, (plan, output)).is_none(),
                "duplicate successful response"
            );
        }
        ensure!(
            successful.len() == self.insertion.len(),
            "missing successful response"
        );
        Ok(successful)
    }
}

fn verify_trace(actual: &[Value], expected: &[Value]) -> Result<()> {
    let mut actual = actual.to_vec();
    let mut expected = expected.to_vec();
    actual.sort_by(|a, b| a["text"].as_str().cmp(&b["text"].as_str()));
    expected.sort_by(|a, b| a["text"].as_str().cmp(&b["text"].as_str()));
    ensure!(actual == expected, "unmatched tokenizer return or refused input tokenized");
    Ok(())
}

async fn assemble(driver: &Driver) -> Result<Value> {
    ensure!(driver.insertion.len() == 2, "assembly requires two insertions");
    let ids: Vec<u64> = driver.insertion
        .iter()
        .map(|event| uint(event, "request_id"))
        .collect::<Result<_>>()?;
    let start = checkpoint(&driver.control, "assembly_start", ids[0]).await?;
    ensure!(start["now_us"] == 0, "assembly start time");
    release(&driver.control, &start)?;

    let growth = find_event(&driver.control, |event| {
        event["kind"] == "assembly" && event["request_ids"] == json!(ids)
    })
    .await?;
    ensure!(
        growth["B"] == 2
            && growth["S"] == 1
            && growth["now_us"] == 0
            && growth["boundary_us"] == 1000,
        "assembled membership growth"
    );
    let sequence = uint(&growth, "seq")?;
    find_event(&driver.control, |event| {
        event["kind"] == "assembly_receive"
            && event["seq"].as_u64().is_some_and(|seq| seq > sequence)
            && event["boundary_us"] == 1000
            && event["now_us"] == 0
    })
    .await?;
    service_result(driver.control.advance_to_us(1000))?;
    let assembled = checkpoint(&driver.control, "assembled", ids[0]).await?;
    ensure!(
        assembled["request_ids"] == json!(ids)
            && assembled["B"] == 2
            && assembled["S"] == 1
            && assembled["now_us"] == 1000
            && assembled["boundary_us"] == 1000,
        "assembled checkpoint identity/geometry/time"
    );
    Ok(assembled)
}

/// Reconcile the complete scheduler interval and every backend record.
///
/// Positive probes have exactly one two-row physical forward. Gate-only
/// probes have no serving execution and no scheduler events.
fn reconcile(
    driver: &Driver,
    probe: Probe,
    records: &[Value],
    events: &[Value],
    successful: &Successful,
) -> Result<Value> {
    ensure!(
        records.first() == Some(&startup_record()),
        "controlled startup record"
    );
    let inserted: Vec<&Value> = events
        .iter()
        .filter(|event| event["kind"] == "inserted")
        .collect();
    ensure!(
        inserted.len() == driver.insertion.len()
            && inserted.iter().zip(&driver.insertion).all(|(a, b)| *a == b),
        "unmatched or reordered insertions"
    );
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

    if !probe.has_work() {
        ensure!(
            events.is_empty()
                && records == [startup_record()]
                && successful.is_empty(),
            "token-gate-only probe executed or entered scheduler"
        );
        return Ok(json!({
            "physical_forwards": 0,
            "logical_backend_requests": 0,
            "rejected_requests_executed": 0,
            "unmatched_backend_records": 0,
            "unmatched_insertions": 0
        }));
    }

    ensure!(
        successful.len() == 2 && driver.insertion.len() == 2 && records.len() == 3,
        "positive probe request/backend coverage"
    );
    let ids: Vec<u64> = driver.insertion
        .iter()
        .map(|event| uint(event, "request_id"))
        .collect::<Result<_>>()?;
    let assemblies: Vec<&Value> = events
        .iter()
        .filter(|event| event["kind"] == "assembly")
        .collect();
    let checkpoints: Vec<&Value> = events
        .iter()
        .filter(|event| event["kind"] == "checkpoint")
        .collect();
    let launches: Vec<&Value> = events
        .iter()
        .filter(|event| event["kind"] == "launch")
        .collect();
    ensure!(
        assemblies.len() == 2 && checkpoints.len() == 2 && launches.len() == 1,
        "additional or missing assembly/checkpoint/launch"
    );
    ensure!(
        assemblies[0]["request_ids"] == json!(&ids[..1])
            && assemblies[1]["request_ids"] == json!(ids)
            && checkpoints[0]["stage"] == "assembly_start"
            && checkpoints[0]["request_ids"] == json!(&ids[..1])
            && checkpoints[0]["now_us"] == 0
            && checkpoints[0]["boundary_us"].is_null()
            && checkpoints[1]["stage"] == "assembled"
            && checkpoints[1]["request_ids"] == json!(ids)
            && checkpoints[1]["now_us"] == 1000
            && checkpoints[1]["boundary_us"] == 1000,
        "assembly/checkpoint membership or timing"
    );
    for (position, assembly) in assemblies.iter().enumerate() {
        ensure!(
            assembly["B"] == position + 1
                && assembly["S"] == 1
                && assembly["now_us"] == 0
                && assembly["boundary_us"] == 1000,
            "assembly geometry/time"
        );
    }
    let first_assembly = uint(assemblies[0], "seq")?;
    ensure!(
        driver.insertion.iter().all(|event| {
            event["seq"].as_u64().is_some_and(|seq| seq < first_assembly)
        }),
        "assembly began before both valid insertion acknowledgements"
    );
    for event in events.iter().filter(|event| event["kind"] == "assembly_receive") {
        ensure!(
            event["boundary_us"] == 1000 && event["now_us"] == 0,
            "unexpected assembly receive"
        );
    }

    let launch_us = if probe == Probe::Quiesce { 5000 } else { 1000 };
    ensure!(
        launches[0]["request_ids"] == json!(ids)
            && launches[0]["B"] == 2
            && launches[0]["S"] == 1
            && launches[0]["now_us"] == launch_us
            && launches[0]["boundary_us"].is_null(),
        "launch identity/geometry/time"
    );

    let mut rows = Vec::new();
    let mut completed = Vec::new();
    for id in &ids {
        let (plan, output) = successful.get(id).context("unknown launched request")?;
        ensure!(plan.tokens.len() == 1 && output.len() == 1, "singleton geometry");
        let row = json!({
            "physical_row": rows.len(),
            "request_id": id,
            "request_row": 0,
            "case_id": plan.case,
            "valid_length": 1,
            "token_sha256": ids_sha(&plan.tokens)
        });
        let mut end = row.clone();
        end["output_sha256"] = json!(sha(&row_bytes(&output[0])));
        rows.push(row);
        completed.push(end);
    }
    let begin = json!({
        "kind": "controlled_backend_forward_begin",
        "backend": "synthetic_scheduler_probe",
        "forward_id": 0,
        "B": 2,
        "S": 1,
        "projection_rows": 2,
        "attention_cells_before_heads": 2,
        "valid_tokens": 2,
        "padding_tokens": 0,
        "input_rows": 2,
        "association_rows": 2,
        "rows": rows
    });
    let mut end = begin.clone();
    end["kind"] = json!("controlled_backend_forward_end");
    end["status"] = json!("ok");
    end["rows"] = json!(completed);
    ensure!(
        records[1] == begin && records[2] == end,
        "complete backend pair/HTTP physical-row reconciliation"
    );
    Ok(json!({
        "physical_forwards": 1,
        "logical_backend_requests": 2,
        "launch_us": launch_us,
        "request_ids": ids,
        "rows": completed,
        "rejected_requests_executed": 0,
        "unmatched_backend_records": 0,
        "unmatched_insertions": 0,
        "unmatched_successful_responses": 0
    }))
}

async fn exercise(
    driver: &mut Driver,
    app: &App,
    recording: &Mutex<Recording>,
    trace: &Mutex<Vec<Value>>,
    manifest: &Path,
    probe: Probe,
) -> Result<Value> {
    let before = observe(app).await?;
    ensure!(
        before["instance"] == driver.instance
            && before["pid"] == std::process::id()
            && before["lifecycle"] == "ready"
            && before["worker_alive"] == true
            && before["ready"] == true
            && before["forward_calls"] == 0,
        "initial local observation"
    );
    let initial_records = guard(recording).records.clone();
    let initial_bytes = bounded(manifest, MANIFEST_LIMIT)?;
    ensure!(
        initial_records == [startup_record()]
            && initial_bytes == record_bytes(&initial_records)?
            && service_result(driver.control.snapshot())?.is_empty()
            && guard(trace).is_empty(),
        "initial backend/scheduler/tokenizer state"
    );
    save_json(&driver.out.join("observation_before.json"), &before)?;
    write_new(&driver.out.join("manifest_before.jsonl"), &initial_bytes)?;

    let successful;
    let witness = if probe == Probe::Quiesce {
        driver.submit(100).await?;
        driver.submit(101).await?;
        let assembled = assemble(driver).await?;
        let held_events = service_result(driver.control.snapshot())?;
        ensure!(
            guard(recording).records == initial_records
                && bounded(manifest, MANIFEST_LIMIT)? == initial_bytes,
            "assembled work executed before quiesce"
        );

        // Poll the actual production control future once. With admitted work
        // held at assembled, it must close admission and suspend on the
        // all-permits barrier. No sleep or task-start guess establishes this.
        let quiesce = app.control(ControlCommand::Quiesce);
        tokio::pin!(quiesce);
        let first_poll = poll_fn(|cx| Poll::Ready(quiesce.as_mut().poll(cx))).await;
        ensure!(first_poll.is_pending(), "quiesce completed with assembled permits held");
        ensure!(
            !app.is_ready() && app.lifecycle() == Lifecycle::Ready,
            "quiesce did not close admission without changing lifecycle"
        );

        let during = driver.closed_attempt("sched5657-closed-during").await?;
        service_result(driver.control.advance_to_us(5000))?;
        ensure!(
            service_result(driver.control.snapshot())? == held_events
                && guard(recording).records == initial_records
                && bounded(manifest, MANIFEST_LIMIT)? == initial_bytes
                && !app.is_ready()
                && app.lifecycle() == Lifecycle::Ready,
            "held quiesce window changed assembled state or executed"
        );

        // All pre-existing work is already assembled. No backlog is hidden
        // behind the checkpoint. Allow that batch to complete normally.
        release(&driver.control, &assembled)?;
        successful = driver.collect().await?;
        let quiesced = serde_json::to_value(service_result(
            timeout(WATCHDOG, quiesce.as_mut())
                .await
                .context("quiesce watchdog; not semantic time evidence")?,
        )?)?;
        let mut expected = before.clone();
        expected["ready"] = json!(false);
        expected["forward_calls"] = json!(2);
        ensure!(
            quiesced == expected && observe(app).await? == expected,
            "quiesce completion changed more than readiness and assembled request count"
        );

        let stable_events = service_result(driver.control.snapshot())?;
        let stable_records = guard(recording).records.clone();
        let stable_bytes = bounded(manifest, MANIFEST_LIMIT)?;
        ensure!(
            stable_events.starts_with(&held_events)
                && stable_events.len() == held_events.len() + 1
                && stable_events.last().context("quiesce launch event")?["kind"] == "launch",
            "new assembly or insertion began after admission closed"
        );

        // Manual semantic time advances, each followed by a real refusal and
        // an owner-queue observation barrier. Wall time is only a watchdog.
        let mut windows = Vec::new();
        for (now, case) in [
            (15_000, "sched5657-closed-at-15000"),
            (1_000_000, "sched5657-closed-at-1000000"),
        ] {
            service_result(driver.control.advance_to_us(now))?;
            let refusal = driver.closed_attempt(case).await?;
            let observation = observe(app).await?;
            ensure!(
                observation == quiesced
                    && service_result(driver.control.snapshot())? == stable_events
                    && guard(recording).records == stable_records
                    && bounded(manifest, MANIFEST_LIMIT)? == stable_bytes,
                "quiesced scheduler/owner/backend state not stable"
            );
            windows.push(json!({
                "now_us": now,
                "refusal": refusal,
                "observation": observation,
                "scheduler_events_sha256": sha(&serde_json::to_vec(&stable_events)?),
                "backend_manifest_sha256": sha(&stable_bytes)
            }));
        }

        json!({
            "assembled_before_quiesce": assembled,
            "admission_closed_us": 1000,
            "held_through_us": 5000,
            "quiesce_initial_poll_pending": true,
            "during_quiesce_refusal": during,
            "quiesce_observation": quiesced,
            "assembled_logical_requests_completed": 2,
            "new_assembly_starts_after_admission_close": 0,
            "new_insertions_after_admission_close": 0,
            "post_close_scheduler_suffix": &stable_events[held_events.len()..],
            "stable_quiesced_windows": windows,
            "pre_admitted_unassembled_backlog": 0,
            "general_backlog_assembly_suppression_claimed": false,
            "executor_retired_or_poisoned": false
        })
    } else {
        if probe == Probe::InvalidLast {
            // Both valid requests remain outstanding while the invalid-last
            // request traverses HTTP token validation.
            driver.submit(100).await?;
            driver.submit(101).await?;
            let first = uint(&driver.insertion[0], "request_id")?;
            checkpoint(&driver.control, "assembly_start", first).await?;
        }
        let held_events = service_result(driver.control.snapshot())?;
        let rejection = driver.token_rejection(app, probe).await?;
        ensure!(
            service_result(driver.control.snapshot())? == held_events
                && guard(recording).records == initial_records
                && bounded(manifest, MANIFEST_LIMIT)? == initial_bytes
                && app.is_ready()
                && app.lifecycle() == Lifecycle::Ready,
            "token rejection changed scheduler/backend/lifecycle"
        );
        save_json(&driver.out.join("rejection.json"), &rejection)?;
        save_json(&driver.out.join("rejection_events.json"), &json!(held_events))?;
        write_new(
            &driver.out.join("manifest_after_rejection.jsonl"),
            &bounded(manifest, MANIFEST_LIMIT)?,
        )?;

        if probe == Probe::InvalidLast {
            let assembled = assemble(driver).await?;
            release(&driver.control, &assembled)?;
            successful = driver.collect().await?;
            logical_delta(&before, &observe(app).await?, 2)?;
        } else {
            successful = driver.collect().await?;
            ensure!(
                observe(app).await? == before,
                "token rejection changed owner observation"
            );
        }

        json!({
            "rejection": rejection,
            "co_submitted_valid_logical_requests": driver.insertion.len(),
            "co_submitted_valid_requests_completed": successful.len(),
            "manifest_bytes_identical_through_rejection": true,
            "scheduler_events_identical_through_rejection": true,
            "backend_records_identical_through_rejection": true,
            "entire_invalid_request_executed": false,
            "valid_prefix_of_invalid_request_executed": false,
            "executor_validation_gate_claimed": false
        })
    };

    let after = observe(app).await?;
    let mut expected = before.clone();
    expected["forward_calls"] = json!(successful.len());
    if probe == Probe::Quiesce {
        expected["ready"] = json!(false);
    }
    ensure!(after == expected, "final local observation");
    let events = service_result(driver.control.snapshot())?;
    let records = guard(recording).records.clone();
    let bytes = bounded(manifest, MANIFEST_LIMIT)?;
    ensure!(
        bytes.starts_with(&initial_bytes) && bytes == record_bytes(&records)?,
        "backend manifest binding"
    );
    let actual_trace = guard(trace).clone();
    verify_trace(&actual_trace, &driver.expected_trace)?;
    let reconciliation = reconcile(driver, probe, &records, &events, &successful)?;
    save_json(&driver.out.join("observation_after.json"), &after)?;

    Ok(json!({
        "schema": "rustv1.accept_m4_family56_57_probe.v1",
        "probe": probe.name(),
        "instance": driver.instance,
        "pid": std::process::id(),
        "batching": BatchedExecution::default(),
        "semantic_clock": "manual_microseconds",
        "watchdogs_are_semantic_time_evidence": false,
        "production_source_sha256": sha(include_bytes!("http_serve.rs")),
        "observation_before": before,
        "observation_after": after,
        "manifest_before_sha256": sha(&initial_bytes),
        "backend_manifest_sha256": sha(&bytes),
        "backend_manifest_bytes": bytes.len(),
        "insertion_acknowledgements": driver.insertion,
        "responses": driver.responses,
        "rejections": driver.rejections,
        "tokenizer_returns": actual_trace,
        "witness": witness,
        "scheduler_events": events,
        "backend_records": records,
        "reconciliation": reconciliation,
        "production_scheduler_used": true,
        "backend": "synthetic_scheduler_probe",
        "numerical_model_evidence": false
    }))
}

async fn local(limits: Limits, out: &Path, probe: Probe) -> Result<Value> {
    private_directory(out)?;
    let control = SchedulerControl::new();
    let recording = Arc::new(Mutex::new(Recording::default()));
    let trace = Arc::new(Mutex::new(Vec::new()));
    let instance = format!("f5657-{}", core::random_label()?);
    let manifest = out.join("controlled_backend_manifest.jsonl");
    let file = manifest_file(&manifest)?;

    let mut config = Config::production(64, 131072, 1024);
    config.batching = Some(BatchedExecution::default());
    config.scheduler_control = Some(control.clone());
    config.instance = Some(instance.clone());
    service_result(config.validate())?;
    ensure!(
        config.admission_capacity == 64
            && config.queue_capacity == 64
            && config.request_timeout == Duration::from_micros(REQUEST_US),
        "local qualification envelope"
    );

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let base = format!("http://{}", listener.local_addr()?);
    let listener = service_result(BoundedListener::new(listener, &config.transport))?;
    let _ = Http::new(&base, &instance, limits.clone())?;
    let backend_recording = recording.clone();
    let mut service = service_result(Service::spawn(
        config,
        Arc::new(TraceTokenizer { trace: trace.clone() }),
        Logger::new(std::io::sink()),
        StartupCheck {
            tokens: vec![vec![100]],
            embeddings: vec![vector(100)],
        },
        move || {
            Ok(FileBackend {
                inner: ControlledBackend { recording: backend_recording },
                file,
                written: 0,
            })
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
    let mut driver = Driver {
        base,
        instance,
        limits,
        out: out.to_owned(),
        control: control.clone(),
        tasks: JoinSet::new(),
        insertion: Vec::new(),
        responses: Vec::new(),
        rejections: Vec::new(),
        exposed_ids: BTreeSet::new(),
        expected_trace: Vec::new(),
    };

    // Every ordinary error after owner creation flows through cleanup.
    let work: Result<Value> = async {
        service_result(service.wait_ready().await)?;
        exercise(&mut driver, &app, &recording, &trace, &manifest, probe).await
    }
    .await;

    app.begin_drain();
    control.open_for_cleanup();
    let cleanup_control = control.clone();
    let (pump_stop, mut pump_stopped) = oneshot::channel();
    let pump = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut pump_stopped => return Ok::<(), Code>(()),
                _ = sleep(Duration::from_millis(1)) => {
                    cleanup_control.advance_by_us(1000)?;
                }
            }
        }
    });

    driver.tasks.abort_all();
    while driver.tasks.join_next().await.is_some() {}

    let shutdown = service_result(service.shutdown().await);
    let _ = stop.send(());
    let transport = match timeout(WATCHDOG, &mut server).await {
        Ok(Ok(Ok(()))) => Ok(()),
        other => {
            server.abort();
            let _ = server.await;
            Err(anyhow::anyhow!("family-56/57 HTTP cleanup failed: {other:?}"))
        }
    };
    let _ = pump_stop.send(());
    let pump_result = pump
        .await
        .context("family-56/57 cleanup clock task")
        .and_then(service_result);

    let final_records = guard(&recording).records.clone();
    let final_trace = guard(&trace).clone();
    let final_events = service_result(control.snapshot());
    let closure = json!({
        "service_shutdown_returned_ok": shutdown.is_ok(),
        "owner_thread_joined": shutdown.is_ok(),
        "http_server_joined": transport.is_ok(),
        "cleanup_clock_task_joined": pump_result.is_ok(),
        "client_tasks_drained": driver.tasks.is_empty(),
        "lifecycle": app.lifecycle(),
        "ready": app.is_ready(),
        "drop_counted_as_join": false,
        "shutdown_timeout_counted_as_cleanup": false,
        "quarantine_cleared": false,
        "work_error": work.as_ref().err().map(|error| format!("{error:#}")),
        "shutdown_error": shutdown.as_ref().err().map(|error| format!("{error:#}")),
        "transport_error": transport.as_ref().err().map(|error| format!("{error:#}")),
        "clock_error": pump_result.as_ref().err().map(|error| format!("{error:#}"))
    });
    let diagnostic: Result<()> = (|| {
        save_json(&out.join("owner_closure.json"), &closure)?;
        save_json(&out.join("backend_records.json"), &json!(final_records))?;
        save_json(&out.join("tokenizer_returns.json"), &json!(final_trace))?;
        save_json(&out.join("responses.json"), &json!(driver.responses))?;
        save_json(&out.join("rejections.json"), &json!(driver.rejections))?;
        match &final_events {
            Ok(events) => {
                save_json(&out.join("scheduler_events.json"), &json!(events))?;
            }
            Err(error) => {
                save_json(
                    &out.join("scheduler_event_failure.json"),
                    &json!({"error": format!("{error:#}")}),
                )?;
            }
        }
        Ok(())
    })();

    if let Err(error) = work {
        return Err(error).with_context(|| {
            format!(
                "family-56/57 probe failed; retained evidence at {}; \
                 owner_join_ok={}, http_join_ok={}, clock_join_ok={}, diagnostics_ok={}",
                out.display(),
                shutdown.is_ok(),
                transport.is_ok(),
                pump_result.is_ok(),
                diagnostic.is_ok()
            )
        });
    }
    let mut evidence = work?;
    shutdown?;
    transport?;
    pump_result?;
    diagnostic?;
    let final_events = final_events?;
    ensure!(
        app.lifecycle() == Lifecycle::Stopped && !app.is_ready(),
        "local owner not stopped after successful shutdown/join"
    );
    let final_bytes = bounded(&manifest, MANIFEST_LIMIT)?;
    ensure!(
        evidence["backend_records"] == json!(final_records)
            && evidence["scheduler_events"] == json!(final_events)
            && evidence["tokenizer_returns"] == json!(final_trace)
            && final_bytes == record_bytes(&final_records)?
            && evidence["backend_manifest_sha256"] == sha(&final_bytes),
        "additional execution, scheduler activity, or tokenization during cleanup"
    );
    evidence["cleanup"] = closure;
    evidence["all_local_owners_joined"] = json!(true);
    save_json(&out.join("evidence.json"), &evidence)?;
    Ok(evidence)
}

fn probes(case: &str) -> Result<Vec<Probe>> {
    Ok(match case {
        INVALID_LAST => vec![Probe::InvalidLast],
        TOKEN_GATES => vec![Probe::Pad, Probe::Oov],
        QUIESCE => vec![Probe::Quiesce],
        _ => anyhow::bail!("unknown family-56/57 case"),
    })
}

async fn live_unchanged(state: &SchedulerState) -> Result<()> {
    state.launch.process.check_live()?;
    ensure!(
        state.core.control.call("observe", None).await? == state.observation
            && bounded(&state.launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
        "unreconciled live-server activity during local family-56/57 probes"
    );
    Ok(())
}

async fn execute_case(state: &mut SchedulerState, case: &str) -> Result<()> {
    ensure!(
        state.core.ledger.pending().first()
            == Some(&json!({
                "acceptance_phase": policy::SCHEDULER,
                "case_id": case
            })),
        "family-56/57 runner called at wrong global ledger position"
    );
    let result: Result<Value> = async {
        state.core.loaded.recheck()?;
        envelope(&state.core.loaded)?;
        live_unchanged(state).await?;
        let out = state.directory.join(case);
        private_directory(&out)?;
        let mut evidence_probes = Vec::new();
        let mut bindings = Vec::new();
        for probe in probes(case)? {
            let limits: Limits =
                serde_json::from_value(state.core.loaded.policy["limits"].clone())?;
            let probe_out = out.join(probe.name());
            let evidence = local(limits, &probe_out, probe).await?;
            ensure!(
                evidence["all_local_owners_joined"] == true,
                "probe returned without local owner joins"
            );
            live_unchanged(state).await?;
            bindings.push(json!({
                "path": format!("{}/evidence.json", probe.name()),
                "sha256": file_sha(&probe_out.join("evidence.json"))?
            }));
            evidence_probes.push(evidence);
        }
        state.core.loaded.recheck()?;
        live_unchanged(state).await?;
        state.core.orchestration.sync()?;
        let evidence = json!({
            "schema": "rustv1.accept_m4_family56_57_case.v1",
            "case_id": case,
            "policy_sha256": state.core.loaded.policy_sha256,
            "driver_binary_sha256": state.core.loaded.artifacts["accept_m4"].sha256,
            "batching": state.core.loaded.bundle.shapes.batching,
            "probe_bindings": bindings,
            "probes": evidence_probes,
            "live_instance": state.launch.instance,
            "live_pid": state.launch.pid,
            "live_observation_unchanged": state.observation,
            "live_manifest_unchanged_sha256": sha(&state.manifest_prefix),
            "live_next_forward_id_unchanged": state.next_forward,
            "live_request_ids_unchanged": state.seen_request_ids,
            "production_scheduler_used": true,
            "backend": "synthetic_scheduler_probe",
            "numerical_model_evidence": false,
            "all_local_owners_joined": true
        });
        let digest = save_json(&out.join("evidence.json"), &evidence)?;
        Ok(json!({
            "evidence_path": format!("scheduler/{case}/evidence.json"),
            "evidence_sha256": digest,
            "production_scheduler_used": true,
            "backend": "synthetic_scheduler_probe",
            "numerical_model_evidence": false,
            "all_local_owners_joined": true
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
    let observation = state.observation.clone();
    let manifest = state.manifest_prefix.clone();
    let next_forward = state.next_forward;
    let request_ids = state.seen_request_ids.clone();

    for case in [INVALID_LAST, TOKEN_GATES, QUIESCE] {
        execute_case(&mut state, case).await?;
    }

    ensure!(
        state.observation == observation
            && state.manifest_prefix == manifest
            && state.next_forward == next_forward
            && state.seen_request_ids == request_ids,
        "family-56/57 changed retained live reconciliation state"
    );
    ensure!(
        state.core.ledger.pending().first()
            == Some(&json!({
                "acceptance_phase": policy::SCHEDULER,
                "case_id": NEXT
            })),
        "unexpected post-quiesce ledger boundary"
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn production_scheduler_family56_57_probes() {
        let root = std::env::temp_dir().join(format!(
            "rustv1-family5657-test-{}-{}",
            std::process::id(),
            core::random_label().unwrap()
        ));
        private_directory(&root).unwrap();

        let result: Result<()> = async {
            for case in [INVALID_LAST, TOKEN_GATES, QUIESCE] {
                for probe in probes(case)? {
                    let evidence = local(limits(), &root.join(probe.name()), probe).await?;
                    ensure!(
                        evidence["cleanup"]["owner_thread_joined"] == true
                            && evidence["cleanup"]["http_server_joined"] == true
                            && evidence["cleanup"]["cleanup_clock_task_joined"] == true
                            && evidence["all_local_owners_joined"] == true
                            && evidence["reconciliation"]["rejected_requests_executed"] == 0,
                        "family-56/57 cleanup or rejection summary"
                    );
                    match probe {
                        Probe::InvalidLast => {
                            ensure!(
                                evidence["witness"]["co_submitted_valid_requests_completed"] == 2
                                    && evidence["witness"]["rejection"]["rejection_gate"] == OOV_GATE
                                    && evidence["witness"]["rejection"]["request_id"].is_u64(),
                                "invalid-last isolation summary"
                            );
                        }
                        Probe::Pad | Probe::Oov => {
                            ensure!(
                                evidence["observation_before"] == evidence["observation_after"]
                                    && evidence["manifest_before_sha256"]
                                        == evidence["backend_manifest_sha256"]
                                    && evidence["reconciliation"]["physical_forwards"] == 0
                                    && evidence["witness"]["rejection"]["request_id"].is_u64()
                                    && evidence["witness"]["rejection"]["rejection_gate"]
                                        == if probe == Probe::Pad { PAD_GATE } else { OOV_GATE },
                                "token-gate zero-execution summary"
                            );
                        }
                        Probe::Quiesce => {
                            ensure!(
                                evidence["witness"]["assembled_logical_requests_completed"] == 2
                                    && evidence["witness"]["new_assembly_starts_after_admission_close"] == 0
                                    && evidence["witness"]["new_insertions_after_admission_close"] == 0
                                    && evidence["witness"]["stable_quiesced_windows"]
                                        .as_array().is_some_and(|windows| windows.len() == 2)
                                    && evidence["observation_after"]["lifecycle"] == "ready"
                                    && evidence["observation_after"]["worker_alive"] == true
                                    && evidence["observation_after"]["ready"] == false,
                                "quiesce summary"
                            );
                        }
                    }
                }
            }
            Ok(())
        }
        .await;

        if result.is_ok() {
            fs::remove_dir_all(&root).unwrap();
        }
        result.unwrap_or_else(|error| {
            panic!(
                "family-56/57 harness failed; retained evidence at {}: {error:#}",
                root.display()
            )
        });
    }
}
