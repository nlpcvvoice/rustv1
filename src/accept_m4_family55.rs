// Included inside acceptance_scheduler::physical_b::family55.
//
// The enclosing module is gated by scheduler-test-seam.
// No CUDA backend is constructed. Production scheduling is unchanged.
//
// Local IDs belong only to their local Service instance and are never
// inserted into the retained live server's request-ID set.

use super::*;
use std::{
    fs::File,
    io::Write,
    sync::Condvar,
};
use tokio::sync::watch;

const SATURATION: &str = "family-55-queue-saturation";
const EXPIRY: &str = "family-55-queued-expiry-zero-execution";
const WAIT: &str = "family-55-batching-wait-bound";
const NEXT: &str = "family-56-invalid-last-isolation";
const REQUEST_US: u64 = 60_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Probe {
    Saturation,
    Expiry,
    Wait,
    Deadline,
}

impl Probe {
    fn name(self) -> &'static str {
        match self {
            Self::Saturation => "saturation",
            Self::Expiry => "queued-expiry",
            Self::Wait => "wait-dominated",
            Self::Deadline => "deadline-dominated",
        }
    }

    fn expires_all(self) -> bool {
        matches!(self, Self::Saturation | Self::Expiry)
    }
}

/// Delay tokenization of token-100 only in the deadline-dominated probe.
///
/// The request has already acquired admission and established its production
/// deadline before TextTokenizer::encode is called. This gate changes neither
/// that deadline nor the scheduler's insertion timestamp.
struct TokenGate {
    open: Mutex<bool>,
    progress: Condvar,
    entered: watch::Sender<bool>,
}

impl TokenGate {
    fn new(open: bool) -> Self {
        let (entered, _) = watch::channel(false);
        Self {
            open: Mutex::new(open),
            progress: Condvar::new(),
            entered,
        }
    }

    fn pass(&self) {
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
            let mut entered = self.entered.subscribe();
            loop {
                if *entered.borrow_and_update() {
                    return Ok::<(), anyhow::Error>(());
                }
                entered.changed().await?;
            }
        })
        .await
        .context("tokenizer-entry watchdog; not a semantic time witness")?
    }
}

struct GatedTokenizer {
    gate: Arc<TokenGate>,
}

impl TextTokenizer for GatedTokenizer {
    fn encode(&self, text: &str) -> std::result::Result<Vec<u32>, Code> {
        if text == "token-100" {
            self.gate.pass();
        }
        Tokenizer.encode(text)
    }
}

/// Persist the reused recording backend's actual records as they are emitted.
///
/// This is an actual backend-produced file, not an evidence file synthesized
/// after the zero-execution assertion. Every backend call synchronizes its
/// appended records before returning.
struct FileBackend {
    inner: ControlledBackend,
    manifest: File,
    written: usize,
}

impl FileBackend {
    fn persist(&mut self) -> std::result::Result<(), Code> {
        let records = guard(&self.inner.recording).records.clone();
        for record in &records[self.written..] {
            serde_json::to_writer(&mut self.manifest, record)
                .map_err(|_| Code::InferenceFailed)?;
            self.manifest
                .write_all(b"\n")
                .map_err(|_| Code::InferenceFailed)?;
        }
        self.manifest
            .sync_data()
            .map_err(|_| Code::InferenceFailed)?;
        self.written = records.len();
        Ok(())
    }
}

impl Backend for FileBackend {
    fn forward(&mut self, batches: &[Vec<u32>]) -> std::result::Result<Vec<Vec<f32>>, Code> {
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

fn request_plan(token: u32) -> Plan {
    Plan {
        case: format!("sched55-request-{token}"),
        tokens: vec![token],
        body: json!({
            "model": MODEL_ID,
            "input": [format!("token-{token}")],
            "encoding_format": "float"
        }),
    }
}

fn error_reply(reply: &http::Reply, code: &str) -> Result<()> {
    let (status, message, kind) = match code {
        "queue_full" => (
            503,
            "Admission capacity is exhausted",
            "service_unavailable_error",
        ),
        "deadline_exceeded" => (504, "Request deadline exceeded", "api_error"),
        _ => anyhow::bail!("unplanned family-55 error"),
    };
    ensure!(
        reply.status == status
            && serde_json::from_slice::<Value>(&reply.bytes)?
                == json!({
                    "error": {
                        "message": message,
                        "type": kind,
                        "param": null,
                        "code": code
                    }
                }),
        "family-55 error status/envelope or partial success payload"
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
}

impl Driver {
    fn submit(&mut self, token: u32) -> Result<Plan> {
        let plan = request_plan(token);
        write_new(
            &self.out.join(format!("{}.request.bin", plan.case)),
            &serde_json::to_vec(&plan.body)?,
        )?;
        let client = Http::new(&self.base, &self.instance, self.limits.clone())?;
        let submitted = plan.clone();
        self.tasks.spawn(async move {
            let wire = Wire::json(submitted.body.clone())?;
            let reply = client.send(&submitted.case, &wire).await?;
            Ok((submitted, reply))
        });
        Ok(plan)
    }

    async fn acknowledge(&mut self, plan: &Plan) -> Result<Value> {
        let event = find_event(&self.control, |event| {
            event["kind"] == "inserted" && event["case_id"] == plan.case
        })
        .await?;
        ensure!(
            event["B"] == 1
                && event["S"] == 1
                && self
                    .insertion
                    .iter()
                    .all(|previous| previous["request_id"] != event["request_id"]),
            "insertion geometry or duplicate acknowledged ID"
        );
        self.insertion.push(event.clone());
        Ok(event)
    }

    async fn overflow(&self) -> Result<Value> {
        let plan = request_plan(1000);
        let bytes = serde_json::to_vec(&plan.body)?;
        write_new(
            &self.out.join(format!("{}.request.bin", plan.case)),
            &bytes,
        )?;
        let client = Http::new(&self.base, &self.instance, self.limits.clone())?;
        let reply = client.send(&plan.case, &Wire::json(plan.body)?).await?;
        write_new(
            &self.out.join(format!("{}.response.bin", plan.case)),
            &reply.bytes,
        )?;
        error_reply(&reply, "queue_full")?;
        ensure!(
            reply.request_id.is_none(),
            "HTTP admission refusal unexpectedly exposed a request ID"
        );
        ensure!(
            service_result(self.control.snapshot())?
                .iter()
                .all(|event| event["case_id"] != plan.case),
            "admission-refused request reached scheduler insertion"
        );
        Ok(json!({
            "case_id": plan.case,
            "request_id": reply.request_id,
            "request_sha256": sha(&bytes),
            "response_sha256": sha(&reply.bytes),
            "http_status": reply.status,
            "headers": reply.headers,
            "rejection_gate": "http_admission_gate",
            "independent_queue_slot_predicate_proven": false,
            "executor_capacity_predicate_proven": false
        }))
    }

    async fn collect(&mut self, probe: Probe) -> Result<(Vec<Value>, Successful)> {
        let mut responses = Vec::new();
        let mut successful = BTreeMap::new();
        let mut completed = BTreeSet::new();

        while let Some(joined) = timeout(WATCHDOG, self.tasks.join_next())
            .await
            .context("HTTP task watchdog; not a semantic time witness")?
        {
            let (plan, reply) = joined.context("family-55 HTTP task")??;
            write_new(
                &self.out.join(format!("{}.response.bin", plan.case)),
                &reply.bytes,
            )?;
            let acknowledged = self
                .insertion
                .iter()
                .find(|event| event["case_id"] == plan.case)
                .context("response without acknowledged insertion")?;

            // Obtain the ID from the actual HTTP response and reconcile it
            // against the actual insertion event. Never synthesize an ID.
            let id = reply.request_id.context("admitted response missing ID")?;
            ensure!(
                acknowledged["request_id"] == id && completed.insert(id),
                "HTTP/insertion identity mismatch or duplicate response"
            );

            let expired = probe.expires_all()
                || (probe == Probe::Deadline && plan.tokens == [100]);
            let output = if expired {
                error_reply(&reply, "deadline_exceeded")?;
                None
            } else {
                Some(decode(&reply, &plan)?)
            };

            responses.push(json!({
                "case_id": plan.case,
                "request_id": reply.request_id,
                "request_sha256": sha(&serde_json::to_vec(&plan.body)?),
                "response_sha256": sha(&reply.bytes),
                "http_status": reply.status,
                "headers": reply.headers,
                "expired_while_queued": expired,
                "output_sha256": output.as_ref().map(|rows| {
                    rows.iter().map(|row| sha(&row_bytes(row))).collect::<Vec<_>>()
                })
            }));
            if let Some(output) = output {
                ensure!(
                    successful.insert(id, (plan, output)).is_none(),
                    "duplicate successful response"
                );
            }
        }

        ensure!(
            completed.len() == self.insertion.len(),
            "missing admitted HTTP response"
        );
        Ok((responses, successful))
    }
}

fn no_completion(events: &[Value]) -> Result<()> {
    ensure!(
        events.iter().all(|event| {
            event["kind"] != "launch"
                && !(event["kind"] == "checkpoint" && event["stage"] == "assembled")
        }),
        "assembly completed or execution launched before intended boundary"
    );
    Ok(())
}

async fn receive_after(
    control: &SchedulerControl,
    sequence: u64,
    boundary: u64,
    now: u64,
) -> Result<Value> {
    let event = find_event(control, |event| {
        event["kind"] == "assembly_receive"
            && event["seq"].as_u64().is_some_and(|seq| seq > sequence)
    })
    .await?;
    ensure!(
        event["boundary_us"] == boundary && event["now_us"] == now,
        "production receive boundary/time"
    );
    Ok(event)
}

/// Complete coverage of startup, launches, physical rows, and successful HTTP
/// responses. Expired requests cannot occur in any backend serving record.
fn reconcile(
    records: &[Value],
    events: &[Value],
    insertion: &[Value],
    successful: &Successful,
) -> Result<Value> {
    ensure!(
        records.first()
            == Some(&json!({
                "kind": "controlled_startup",
                "backend": "synthetic_scheduler_probe",
                "tokens": [[100]],
                "output_sha256": [sha(&row_bytes(&vector(100)))]
            })),
        "startup record mismatch"
    );

    let actual_insertions: Vec<&Value> = events
        .iter()
        .filter(|event| event["kind"] == "inserted")
        .collect();
    ensure!(
        actual_insertions.len() == insertion.len()
            && actual_insertions
                .iter()
                .zip(insertion)
                .all(|(actual, expected)| *actual == expected),
        "unmatched or reordered insertions"
    );
    for (position, event) in events.iter().enumerate() {
        ensure!(event["seq"] == position, "scheduler sequence gap");
        ensure!(
            matches!(
                event["kind"].as_str(),
                Some("inserted" | "assembly" | "assembly_receive" | "checkpoint" | "launch")
            ),
            "unexpected scheduler event"
        );
    }

    let launches: Vec<&Value> = events
        .iter()
        .filter(|event| event["kind"] == "launch")
        .collect();
    ensure!(
        records.len() == 1 + 2 * launches.len(),
        "backend pair/launch coverage"
    );

    let mut seen = BTreeSet::new();
    let mut forwards = Vec::new();
    for (forward, launch) in launches.iter().enumerate() {
        let ids = launch["request_ids"]
            .as_array()
            .context("launch members")?;
        let mut rows = Vec::new();
        let mut completed_rows = Vec::new();

        for member in ids {
            let id = member.as_u64().context("launch request ID")?;
            ensure!(seen.insert(id), "request executed more than once");
            let (plan, output) = successful
                .get(&id)
                .context("expired, refused, or unknown request executed")?;
            ensure!(
                plan.tokens.len() == 1 && output.len() == 1,
                "singleton probe geometry"
            );
            let row = json!({
                "physical_row": rows.len(),
                "request_id": id,
                "request_row": 0,
                "case_id": plan.case,
                "valid_length": 1,
                "token_sha256": ids_sha(&plan.tokens)
            });
            let mut completed = row.clone();
            completed["output_sha256"] = json!(sha(&row_bytes(&output[0])));
            rows.push(row);
            completed_rows.push(completed);
        }

        let b = rows.len();
        ensure!(
            b > 0
                && BatchedExecution::default().fits(b, 1)
                && launch["B"] == b
                && launch["S"] == 1
                && launch["boundary_us"].is_null(),
            "launch geometry"
        );
        let begin = &records[1 + 2 * forward];
        let end = &records[2 + 2 * forward];
        ensure!(
            begin["kind"] == "controlled_backend_forward_begin"
                && end["kind"] == "controlled_backend_forward_end"
                && end["status"] == "ok"
                && begin["rows"] == json!(rows)
                && end["rows"] == json!(completed_rows),
            "backend pair and physical/HTTP row association"
        );
        for record in [begin, end] {
            ensure!(
                record["backend"] == "synthetic_scheduler_probe"
                    && record["forward_id"] == forward
                    && record["B"] == b
                    && record["S"] == 1
                    && record["projection_rows"] == b
                    && record["attention_cells_before_heads"] == b
                    && record["valid_tokens"] == b
                    && record["padding_tokens"] == 0
                    && record["input_rows"] == b
                    && record["association_rows"] == b,
                "backend geometry/accounting"
            );
        }
        forwards.push(json!({
            "forward_id": forward,
            "launch_us": launch["now_us"],
            "request_ids": ids,
            "rows": completed_rows
        }));
    }
    ensure!(
        seen.len() == successful.len(),
        "successful HTTP response absent from backend records"
    );
    Ok(json!({
        "physical_forwards": forwards,
        "logical_backend_requests": successful.len(),
        "expired_or_refused_requests_executed": 0,
        "unmatched_backend_records": 0,
        "unmatched_insertions": 0,
        "unmatched_successful_responses": 0
    }))
}

async fn exercise(
    driver: &mut Driver,
    app: &App,
    gate: &TokenGate,
    recording: &Mutex<Recording>,
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
        "initial family-55 observation"
    );
    let initial_records = guard(recording).records.clone();
    let initial_bytes = bounded(manifest, MANIFEST_LIMIT)?;
    ensure!(
        initial_bytes == record_bytes(&initial_records)? && initial_records.len() == 1,
        "initial backend file/record binding"
    );
    write_new(&driver.out.join("manifest_before.jsonl"), &initial_bytes)?;
    save_json(&driver.out.join("observation_before.json"), &before)?;

    let responses;
    let successful;

    let witness = if probe.expires_all() {
        let count = if probe == Probe::Saturation { 64 } else { 2 };
        for position in 0..count {
            let plan = driver.submit(100 + u32::try_from(position)?)?;
            let ack = driver.acknowledge(&plan).await?;
            ensure!(
                ack["inserted_us"] == 0
                    && ack["now_us"] == 0
                    && ack["deadline_us"] == REQUEST_US,
                "queued probe insertion/deadline"
            );
        }
        let first_id = uint(&driver.insertion[0], "request_id")?;
        let start = checkpoint(&driver.control, "assembly_start", first_id).await?;
        ensure!(
            start["now_us"] == 0 && start["request_ids"] == json!([first_id]),
            "first queued checkpoint"
        );

        let refusal = if probe == Probe::Saturation {
            Some(driver.overflow().await?)
        } else {
            None
        };
        let held_events = service_result(driver.control.snapshot())?;
        no_completion(&held_events)?;
        ensure!(
            held_events.iter().filter(|event| event["kind"] == "checkpoint").count() == 1
                && held_events.iter().all(|event| event["kind"] != "assembly")
                && guard(recording).records == initial_records
                && bounded(manifest, MANIFEST_LIMIT)? == initial_bytes,
            "queued work progressed while the first checkpoint was held"
        );
        save_json(&driver.out.join("held_events.json"), &json!(held_events))?;

        // All requests have insertion acknowledgements. In particular, the
        // second request is still behind the held first request, not running.
        service_result(driver.control.advance_to_us(REQUEST_US))?;
        (responses, successful) = driver.collect(probe).await?;
        ensure!(
            successful.is_empty()
                && bounded(manifest, MANIFEST_LIMIT)? == initial_bytes
                && guard(recording).records == initial_records,
            "queued deadline expiry executed backend work"
        );

        // The first request passed queued_error before entering the held
        // checkpoint. It will finish assembly, then be filtered by run_batch.
        // Other expired requests are rejected directly by queued_error.
        release(&driver.control, &start)?;
        let assembled = checkpoint(&driver.control, "assembled", first_id).await?;
        ensure!(
            assembled["request_ids"] == json!([first_id])
                && assembled["now_us"] == REQUEST_US
                && assembled["boundary_us"] == 1000,
            "expired first request assembly cleanup"
        );
        release(&driver.control, &assembled)?;

        // This owner command follows every queued Infer message. Its reply
        // proves the owner consumed the expired channel backlog.
        let after = observe(app).await?;
        ensure!(
            after == before
                && bounded(manifest, MANIFEST_LIMIT)? == initial_bytes
                && guard(recording).records == initial_records,
            "zero-execution owner barrier/manifest/record invariance"
        );

        json!({
            "configured_admission_capacity": app.config.admission_capacity,
            "configured_queue_capacity": app.config.queue_capacity,
            "acknowledged_waiting_logical_requests": count,
            "scheduler_local_waiting_requests": 1,
            "channel_waiting_requests": count - 1,
            "expiry_us": REQUEST_US,
            "held_checkpoint": start,
            "expired_first_assembled_checkpoint": assembled,
            "admission_refusal": refusal,
            "rejection_gate": if probe == Probe::Saturation {
                Some("http_admission_gate")
            } else {
                None
            },
            "independent_queue_slot_predicate_proven": false,
            "executor_capacity_predicate_proven": false,
            "observations_identical": true,
            "manifest_bytes_identical": true,
            "backend_records_identical": true,
            "physical_forwards": 0,
            "logical_backend_request_delta": 0,
            "observation_after": after
        })
    } else {
        if probe == Probe::Wait {
            // Nonzero insertion proves the origin is insertion, not zero.
            service_result(driver.control.advance_to_us(10_000))?;
        }

        let first_plan = driver.submit(100)?;
        if probe == Probe::Deadline {
            gate.wait_entered().await?;
            ensure!(
                driver.control.now_us() == 0
                    && service_result(driver.control.snapshot())?.is_empty(),
                "delayed request inserted before tokenizer release"
            );
            service_result(driver.control.advance_to_us(REQUEST_US - 500))?;
            gate.release();
        }
        let first_ack = driver.acknowledge(&first_plan).await?;
        let first_id = uint(&first_ack, "request_id")?;
        let inserted = uint(&first_ack, "inserted_us")?;
        let deadline = uint(&first_ack, "deadline_us")?;
        let wait_end = inserted
            .checked_add(BatchedExecution::default().max_wait_us)
            .context("wait boundary overflow")?;
        let bound = wait_end.min(deadline);

        ensure!(
            first_ack["now_us"] == inserted
                && if probe == Probe::Wait {
                    inserted == 10_000
                        && deadline == REQUEST_US + 10_000
                        && bound == 11_000
                        && wait_end < deadline
                } else {
                    inserted == REQUEST_US - 500
                        && deadline == REQUEST_US
                        && bound == REQUEST_US
                        && deadline < wait_end
                },
            "wait/deadline-dominated request geometry"
        );

        let start = checkpoint(&driver.control, "assembly_start", first_id).await?;
        release(&driver.control, &start)?;
        let first_assembly = find_event(&driver.control, |event| {
            event["kind"] == "assembly"
                && event["request_ids"] == json!([first_id])
        })
        .await?;
        ensure!(
            first_assembly["boundary_us"] == bound
                && first_assembly["now_us"] == inserted,
            "initial assembly boundary"
        );
        receive_after(
            &driver.control,
            uint(&first_assembly, "seq")?,
            bound,
            inserted,
        )
        .await?;

        // Stimulus at bound-1 makes the owner actually process another
        // complete request at that time. A mere wall-clock sleep followed by
        // an empty snapshot would not establish this progress witness.
        service_result(driver.control.advance_to_us(bound - 1))?;
        let second_plan = driver.submit(101)?;
        let second_ack = driver.acknowledge(&second_plan).await?;
        let second_id = uint(&second_ack, "request_id")?;
        ensure!(
            second_ack["inserted_us"] == bound - 1
                && second_ack["now_us"] == bound - 1
                && second_ack["deadline_us"] == bound - 1 + REQUEST_US,
            "late follower timing"
        );
        let members = json!([first_id, second_id]);
        let growth = find_event(&driver.control, |event| {
            event["kind"] == "assembly" && event["request_ids"] == members
        })
        .await?;
        ensure!(
            growth["now_us"] == bound - 1
                && growth["boundary_us"] == bound
                && growth["B"] == 2
                && growth["S"] == 1,
            "late follower changed the oldest-insertion/deadline boundary"
        );
        let below = receive_after(
            &driver.control,
            uint(&growth, "seq")?,
            bound,
            bound - 1,
        )
        .await?;
        let below_events = service_result(driver.control.snapshot())?;
        no_completion(&below_events)?;
        ensure!(
            bounded(manifest, MANIFEST_LIMIT)? == initial_bytes
                && guard(recording).records == initial_records,
            "backend ran before assembly boundary"
        );
        save_json(
            &driver.out.join("below_bound_events.json"),
            &json!(below_events),
        )?;

        service_result(driver.control.advance_to_us(bound))?;
        let assembled = checkpoint(&driver.control, "assembled", first_id).await?;
        ensure!(
            assembled["request_ids"] == members
                && assembled["B"] == 2
                && assembled["S"] == 1
                && assembled["now_us"] == bound
                && assembled["boundary_us"] == bound,
            "assembly did not complete exactly at the minimum bound"
        );
        ensure!(
            bounded(manifest, MANIFEST_LIMIT)? == initial_bytes,
            "backend ran before assembled checkpoint release"
        );
        release(&driver.control, &assembled)?;
        (responses, successful) = driver.collect(probe).await?;

        let expected_successes = if probe == Probe::Wait { 2 } else { 1 };
        let after = observe(app).await?;
        logical_delta(&before, &after, expected_successes)?;
        ensure!(
            successful.len() == expected_successes
                && (probe != Probe::Deadline || !successful.contains_key(&first_id)),
            "deadline-boundary expiry did not exclude the expired member"
        );

        let final_events = service_result(driver.control.snapshot())?;
        let completions: Vec<&Value> = final_events
            .iter()
            .filter(|event| {
                event["kind"] == "checkpoint" && event["stage"] == "assembled"
            })
            .collect();
        let launches: Vec<&Value> = final_events
            .iter()
            .filter(|event| event["kind"] == "launch")
            .collect();
        ensure!(
            completions == vec![&assembled]
                && launches.len() == 1
                && launches[0]["now_us"] == bound
                && launches[0]["request_ids"]
                    == if probe == Probe::Wait {
                        members
                    } else {
                        json!([second_id])
                    },
            "early/additional completion or unexpected launch membership/time"
        );

        json!({
            "oldest_inserted_us": inserted,
            "batch_wait_us": BatchedExecution::default().max_wait_us,
            "oldest_insertion_plus_wait_us": wait_end,
            "earliest_member_deadline_us": deadline,
            "computed_minimum_bound_us": bound,
            "dominant_term": if probe == Probe::Wait {
                "oldest_insertion_plus_wait"
            } else {
                "earliest_request_deadline"
            },
            "below_bound_receive": below,
            "late_follower_assembly": growth,
            "assembled_checkpoint": assembled,
            "no_assembly_completion_before_bound": true,
            "assembly_completed_at_bound": true,
            "assembly_member_growth_before_bound_is_expected": true,
            "late_arrival_reset_wait_origin": false,
            "capacity_boundary_present": false,
            "control_message_boundary_present": false,
            "request_timeout_us": REQUEST_US,
            "request_timeout_shortened": false,
            "deadline_member_expired_without_execution": probe == Probe::Deadline,
            "observation_after": after
        })
    };

    let after = observe(app).await?;
    logical_delta(&before, &after, successful.len())?;
    let events = service_result(driver.control.snapshot())?;
    let records = guard(recording).records.clone();
    let bytes = bounded(manifest, MANIFEST_LIMIT)?;
    ensure!(
        bytes == record_bytes(&records)? && bytes.starts_with(&initial_bytes),
        "backend file does not exactly match complete backend records"
    );
    if probe.expires_all() {
        ensure!(
            bytes == initial_bytes
                && records == initial_records
                && events.iter().all(|event| event["kind"] != "launch"),
            "zero-execution interval contains serving work"
        );
    }
    let reconciliation = reconcile(
        &records,
        &events,
        &driver.insertion,
        &successful,
    )?;
    save_json(&driver.out.join("observation_after.json"), &after)?;

    Ok(json!({
        "schema": "rustv1.accept_m4_family55_probe.v1",
        "probe": probe.name(),
        "instance": driver.instance,
        "pid": std::process::id(),
        "batching": BatchedExecution::default(),
        "admission_capacity": app.config.admission_capacity,
        "queue_capacity": app.config.queue_capacity,
        "semantic_clock": "manual_microseconds",
        "wall_watchdogs_are_semantic_time_evidence": false,
        "observation_before": before,
        "observation_after": after,
        "manifest_before_sha256": sha(&initial_bytes),
        "backend_manifest_sha256": sha(&bytes),
        "backend_manifest_bytes": bytes.len(),
        "insertion_acknowledgements": driver.insertion,
        "responses": responses,
        "witness": witness,
        "scheduler_events": events,
        "backend_records": records,
        "reconciliation": reconciliation,
        "production_scheduler_used": true,
        "backend": "synthetic_scheduler_probe",
        "numerical_model_evidence": false,
        "scheduler_phase_complete": false,
        "m4_accepted": false
    }))
}

async fn local(limits: Limits, out: &Path, probe: Probe) -> Result<Value> {
    private_directory(out)?;
    let control = SchedulerControl::new();
    let recording = Arc::new(Mutex::new(Recording::default()));
    let gate = Arc::new(TokenGate::new(probe != Probe::Deadline));
    let instance = format!("f55-{}", core::random_label()?);
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
        "family-55 qualification envelope"
    );

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let base = format!("http://{}", listener.local_addr()?);
    let listener = service_result(BoundedListener::new(listener, &config.transport))?;
    let _ = Http::new(&base, &instance, limits.clone())?;

    let backend_recording = recording.clone();
    let mut service = service_result(Service::spawn(
        config,
        Arc::new(GatedTokenizer { gate: gate.clone() }),
        Logger::new(std::io::sink()),
        StartupCheck {
            tokens: vec![vec![100]],
            embeddings: vec![vector(100)],
        },
        move || {
            Ok(FileBackend {
                inner: ControlledBackend {
                    recording: backend_recording,
                },
                manifest: file,
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
    };

    // Every ordinary error after owner creation flows through cleanup.
    let work: Result<Value> = async {
        service_result(service.wait_ready().await)?;
        exercise(
            &mut driver,
            &app,
            &gate,
            &recording,
            &manifest,
            probe,
        )
        .await
    }
    .await;

    app.begin_drain();
    gate.release();
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
            Err(anyhow::anyhow!("family-55 HTTP cleanup failed: {other:?}"))
        }
    };
    let _ = pump_stop.send(());
    let pump_result = pump
        .await
        .context("family-55 cleanup clock task")
        .and_then(service_result);

    // Retain diagnostics on failure too, after attempting every cleanup.
    // The backend file has already been written by the backend itself.
    let final_records = guard(&recording).records.clone();
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
                "family-55 probe failed; evidence retained at {}; \
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
        "family-55 owner did not stop after successful shutdown/join"
    );
    let final_bytes = bounded(&manifest, MANIFEST_LIMIT)?;
    ensure!(
        evidence["backend_records"] == json!(final_records)
            && evidence["scheduler_events"] == json!(final_events)
            && final_bytes == record_bytes(&final_records)?
            && evidence["backend_manifest_sha256"] == sha(&final_bytes),
        "additional execution, scheduler activity, or manifest change during cleanup"
    );

    evidence["cleanup"] = closure;
    evidence["all_local_owners_joined"] = json!(true);
    save_json(&out.join("evidence.json"), &evidence)?;
    Ok(evidence)
}

fn probes(case: &str) -> Result<Vec<Probe>> {
    Ok(match case {
        SATURATION => vec![Probe::Saturation],
        EXPIRY => vec![Probe::Expiry],
        WAIT => vec![Probe::Wait, Probe::Deadline],
        _ => anyhow::bail!("unknown family-55 case"),
    })
}

async fn live_unchanged(state: &SchedulerState) -> Result<()> {
    state.launch.process.check_live()?;
    ensure!(
        state.core.control.call("observe", None).await? == state.observation
            && bounded(&state.launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
        "unreconciled live-server activity during family-55 local probes"
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
        "family-55 runner called at wrong global ledger position"
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
                "probe returned without all owner joins"
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
            "schema": "rustv1.accept_m4_family55_case.v1",
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
            "all_local_owners_joined": true,
            "scheduler_phase_complete": false,
            "m4_accepted": false
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
            state
                .core
                .ledger
                .append(policy::SCHEDULER, case, true, details, true)?;
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

    for case in [SATURATION, EXPIRY, WAIT] {
        execute_case(&mut state, case).await?;
    }

    ensure!(
        state.observation == observation
            && state.manifest_prefix == manifest
            && state.next_forward == next_forward
            && state.seen_request_ids == request_ids,
        "family-55 changed the retained live reconciliation cursor"
    );
    ensure!(
        state.core.ledger.pending().first()
            == Some(&json!({
                "acceptance_phase": policy::SCHEDULER,
                "case_id": NEXT
            })),
        "unexpected post-family-55 ledger position"
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
    async fn production_scheduler_family55_probes() {
        let root = std::env::temp_dir().join(format!(
            "rustv1-family55-test-{}-{}",
            std::process::id(),
            core::random_label().unwrap()
        ));
        private_directory(&root).unwrap();

        let result: Result<()> = async {
            for case in [SATURATION, EXPIRY, WAIT] {
                for probe in probes(case)? {
                    let evidence = local(limits(), &root.join(probe.name()), probe).await?;
                    ensure!(
                        evidence["cleanup"]["owner_thread_joined"] == true
                            && evidence["cleanup"]["http_server_joined"] == true
                            && evidence["cleanup"]["cleanup_clock_task_joined"] == true
                            && evidence["all_local_owners_joined"] == true,
                        "family-55 test owner cleanup"
                    );
                    if probe.expires_all() {
                        ensure!(
                            evidence["observation_before"] == evidence["observation_after"]
                                && evidence["manifest_before_sha256"]
                                    == evidence["backend_manifest_sha256"]
                                && evidence["reconciliation"]["logical_backend_requests"] == 0,
                            "family-55 zero-execution test summary"
                        );
                    } else {
                        ensure!(
                            evidence["witness"]["no_assembly_completion_before_bound"] == true
                                && evidence["witness"]["assembly_completed_at_bound"] == true,
                            "family-55 wait-bound test summary"
                        );
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
                "family-55 harness failed; retained evidence at {}: {error:#}",
                root.display()
            )
        });
    }
}
