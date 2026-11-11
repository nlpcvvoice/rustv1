// Included inside acceptance_scheduler::physical_b::capacity.
//
// These are full-envelope boundary/accounting probes, not claims that the
// padded-token and attention rejection predicates can be independently
// triggered at B<=32 and S<=1024.
//
// Every local owner is explicitly shut down and joined before its global
// ledger case can pass. No CUDA backend is constructed by this file.

use super::*;

const PADDED_CASE: &str = "family-54-padded-token-capacity";
const ATTENTION_CASE: &str = "family-54-attention-cell-capacity";
const FIFO_CASE: &str = "family-54-fifo-no-split-no-bypass";
const FOLLOWING_CASE: &str = "family-55-queue-saturation";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Padded,
    Attention,
    Fifo,
}

#[derive(Clone, Copy)]
struct Spec {
    mode: Mode,
    width: usize,
    name: &'static str,
}

impl Spec {
    fn counts(self) -> &'static [usize] {
        match self.mode {
            Mode::Padded | Mode::Attention => &[8, 8, 8, 8, 1],
            Mode::Fifo => &[8, 8, 8, 7, 2, 1],
        }
    }

    fn first_requests(self) -> usize {
        4
    }

    fn first_b(self) -> usize {
        self.counts()[..self.first_requests()].iter().sum()
    }

    fn tail_b(self) -> usize {
        self.counts()[self.first_requests()..].iter().sum()
    }

    fn length(self, physical_input: usize) -> usize {
        match self.mode {
            Mode::Padded if physical_input == 0 => self.width,
            Mode::Padded | Mode::Fifo => 1,
            Mode::Attention => self.width,
        }
    }

    fn validate(self) -> Result<()> {
        ensure!(
            matches!(self.width, 1 | 1023 | 1024),
            "unexpected capacity-probe width"
        );
        ensure!(
            (self.mode == Mode::Fifo) == (self.width == 1),
            "capacity-probe mode/width mismatch"
        );
        ensure!(
            self.counts().iter().all(|n| (1..=8).contains(n))
                && self.counts().iter().sum::<usize>()
                    == if self.mode == Mode::Fifo { 34 } else { 33 }
                && self.first_requests() < self.counts().len(),
            "capacity-probe request geometry"
        );
        ensure!(
            label_valid(self.name),
            "invalid capacity-probe output name"
        );
        Ok(())
    }
}

#[derive(Clone)]
struct CapacityPlan {
    case: String,
    sequences: Vec<Vec<u32>>,
    body: Value,
}

/// Short HTTP strings specify deterministic tokenization, not model text.
///
/// The normal Service request path still calls production validate_tokens
/// and the per-request batching-policy check. No token-ID HTTP route or
/// private BackendHandle::infer access is introduced.
struct LengthTokenizer;

impl TextTokenizer for LengthTokenizer {
    fn encode(&self, text: &str) -> std::result::Result<Vec<u32>, Code> {
        let (token, length) = text
            .strip_prefix("probe-")
            .and_then(|rest| rest.split_once('-'))
            .ok_or(Code::InvalidRequest)?;
        let token = token.parse::<u32>().map_err(|_| Code::InvalidRequest)?;
        let length = length
            .parse::<usize>()
            .map_err(|_| Code::InvalidRequest)?;
        if !(1..=1024).contains(&length) {
            return Err(Code::InvalidRequest);
        }
        Ok(vec![token; length])
    }
}

fn capacity_plans(spec: Spec) -> Result<Vec<CapacityPlan>> {
    spec.validate()?;
    let mut input_position = 0usize;
    let mut plans = Vec::new();

    for (request_position, count) in spec.counts().iter().copied().enumerate() {
        let mut sequences = Vec::new();
        let mut texts = Vec::new();

        for _ in 0..count {
            let token = 100u32 + u32::try_from(input_position)?;
            let length = spec.length(input_position);
            sequences.push(vec![token; length]);
            texts.push(format!("probe-{token}-{length}"));
            input_position += 1;
        }

        let case = format!("sched54-{}-request-{request_position}", spec.name);
        ensure!(label_valid(&case), "invalid controlled request label");
        plans.push(CapacityPlan {
            case,
            sequences,
            body: json!({
                "model": MODEL_ID,
                "input": texts,
                "encoding_format": "float"
            }),
        });
    }

    ensure!(
        input_position == if spec.mode == Mode::Fifo { 34 } else { 33 },
        "controlled row coverage"
    );
    Ok(plans)
}

fn geometry(plans: &[CapacityPlan]) -> Result<(usize, usize, usize)> {
    let b = plans.iter().map(|p| p.sequences.len()).sum::<usize>();
    let s = plans
        .iter()
        .flat_map(|p| p.sequences.iter().map(Vec::len))
        .max()
        .context("empty controlled geometry")?;
    let valid = plans
        .iter()
        .flat_map(|p| p.sequences.iter().map(Vec::len))
        .sum::<usize>();
    Ok((b, s, valid))
}

fn decode_capacity(reply: &http::Reply, plan: &CapacityPlan) -> Result<Vec<Vec<f32>>> {
    ensure!(reply.status == 200, "controlled capacity HTTP status");
    let body: Value = serde_json::from_slice(&reply.bytes)?;
    closed(&body, &["object", "data", "model", "usage"])?;
    let valid = plan.sequences.iter().map(Vec::len).sum::<usize>();
    ensure!(
        body["object"] == "list"
            && body["model"] == MODEL_ID
            && body["usage"] == json!({
                "prompt_tokens": valid,
                "total_tokens": valid
            }),
        "controlled capacity object/model/valid-token usage"
    );

    let data = body["data"].as_array().context("capacity response rows")?;
    ensure!(
        data.len() == plan.sequences.len(),
        "controlled capacity response cardinality"
    );

    let mut output = Vec::new();
    for (position, item) in data.iter().enumerate() {
        closed(item, &["object", "index", "embedding"])?;
        ensure!(
            item["object"] == "embedding"
                && item["index"] == position
                && item["embedding"].is_array(),
            "controlled capacity response row identity"
        );
        output.push(serde_json::from_value::<Vec<f32>>(
            item["embedding"].clone(),
        )?);
    }

    service_result(validate_output(&output, plan.sequences.len()))?;
    let expected: Vec<Vec<f32>> = plan
        .sequences
        .iter()
        .map(|ids| vector(ids[0]))
        .collect();
    exact(&output, &expected)?;
    Ok(output)
}

/// Checks the actual production fits implementation over the entire legal
/// B/S domain, plus the immediately adjacent B/S boundaries.
///
/// This is supplemental arithmetic evidence. It does not replace the
/// production assembly/launch observations or physical-row reconciliation.
fn envelope_relationship() -> Result<Value> {
    let policy = BatchedExecution::default();
    ensure!(
        policy.max_batch_size == 32
            && policy.max_padded_tokens == 32768
            && policy.max_attention_cells == 33554432
            && policy.max_wait_us == 1000,
        "default qualification envelope changed"
    );

    for b in 0usize..=33 {
        for s in 0usize..=1025 {
            let in_domain = (1..=32).contains(&b) && (1..=1024).contains(&s);
            ensure!(
                policy.fits(b, s) == in_domain,
                "full-envelope fits differs from B/S domain at B={b}, S={s}"
            );
            if in_domain {
                ensure!(
                    b * s <= policy.max_padded_tokens
                        && b * s * s <= policy.max_attention_cells,
                    "legal B/S shape exceeds a derived ceiling"
                );
            }
        }
    }

    Ok(json!({
        "batching": policy,
        "max_sequence_tokens": 1024,
        "maximum_legal_padded_tokens": 32 * 1024,
        "maximum_legal_attention_cells_before_heads": 32 * 1024 * 1024,
        "checked_b_range_inclusive": [0, 33],
        "checked_s_range_inclusive": [0, 1025],
        "legal_domain_exhaustively_checked": true,
        "isolated_padded_limit_violation_in_legal_b_s_domain": false,
        "isolated_attention_limit_violation_in_legal_b_s_domain": false,
        "lowered_test_policy_used": false,
        "overlength_tokens_used": false
    }))
}

fn check_members(
    event: &Value,
    ids: &[u64],
    plans: &[CapacityPlan],
    now: u64,
) -> Result<()> {
    let (b, s, _) = geometry(plans)?;
    ensure!(
        event["request_ids"] == json!(ids)
            && event["B"] == b
            && event["S"] == s
            && event["now_us"] == now,
        "production event membership/geometry/time"
    );
    Ok(())
}

fn reconcile_capacity(
    spec: Spec,
    plans: &[CapacityPlan],
    insertion: &[Value],
    events: &[Value],
    records: &[Value],
    requests: &BTreeMap<u64, (CapacityPlan, Vec<Vec<f32>>)>,
) -> Result<Value> {
    ensure!(
        plans.len() == spec.counts().len()
            && insertion.len() == plans.len()
            && requests.len() == plans.len(),
        "controlled capacity logical request coverage"
    );
    ensure!(
        records.len() == 5
            && records[0] == json!({
                "kind": "controlled_startup",
                "backend": "synthetic_scheduler_probe",
                "tokens": [[100]],
                "output_sha256": [sha(&row_bytes(&vector(100)))]
            }),
        "controlled startup or physical-call count"
    );

    let ids: Vec<u64> = insertion
        .iter()
        .map(|event| uint(event, "request_id"))
        .collect::<Result<_>>()?;
    ensure!(
        ids.iter().copied().collect::<BTreeSet<_>>().len() == ids.len(),
        "duplicate acknowledged request ID"
    );

    let inserted: Vec<&Value> = events
        .iter()
        .filter(|event| event["kind"] == "inserted")
        .collect();
    ensure!(
        inserted.len() == insertion.len()
            && inserted.iter().zip(insertion).all(|(a, b)| *a == b),
        "unmatched or reordered insertions"
    );

    for ((event, plan), count) in insertion.iter().zip(plans).zip(spec.counts()) {
        let s = plan.sequences.iter().map(Vec::len).max().context("empty plan")?;
        ensure!(
            event["case_id"] == plan.case
                && event["B"] == *count
                && event["S"] == s
                && event["inserted_us"] == 0
                && event["now_us"] == 0
                && event["deadline_us"] == 60_000_000,
            "insertion acknowledgement identity/geometry/time"
        );
    }

    let launches: Vec<&Value> = events
        .iter()
        .filter(|event| event["kind"] == "launch")
        .collect();
    let assemblies: Vec<&Value> = events
        .iter()
        .filter(|event| event["kind"] == "assembly")
        .collect();
    let checkpoints: Vec<&Value> = events
        .iter()
        .filter(|event| event["kind"] == "checkpoint")
        .collect();

    ensure!(
        launches.len() == 2
            && assemblies.len() == plans.len()
            && checkpoints.len() == 4,
        "assembly/checkpoint/launch coverage"
    );
    ensure!(
        events.iter().all(|event| matches!(
            event["kind"].as_str(),
            Some("inserted" | "assembly" | "assembly_receive" | "checkpoint" | "launch")
        )),
        "unknown scheduler event"
    );
    for (position, event) in events.iter().enumerate() {
        ensure!(event["seq"] == position, "scheduler event sequence");
        if event["kind"] == "assembly_receive" {
            ensure!(
                event["boundary_us"] == 1000 && event["now_us"] == 0,
                "unexpected assembly receive boundary/time"
            );
        }
    }

    let cut = spec.first_requests();
    let groups = [
        (&ids[..cut], &plans[..cut], 0u64),
        (&ids[cut..], &plans[cut..], 1000u64),
    ];

    // There is one assembly record for each added complete request. The tail
    // is assembled at zero and then waits until its explicit 1000-us boundary.
    let mut assembly_position = 0usize;
    for (members, group_plans, _) in groups {
        for added in 1..=members.len() {
            let event = assemblies[assembly_position];
            check_members(event, &members[..added], &group_plans[..added], 0)?;
            ensure!(
                event["boundary_us"] == 1000,
                "assembly deadline differs from oldest insertion + wait"
            );
            assembly_position += 1;
        }
    }

    let mut seen = BTreeSet::new();
    let mut forward_evidence = Vec::new();

    for (forward, (members, group_plans, now)) in groups.into_iter().enumerate() {
        let (b, s, valid) = geometry(group_plans)?;
        ensure!(
            BatchedExecution::default().fits(b, s),
            "launched shape exceeds full envelope"
        );

        check_members(launches[forward], members, group_plans, now)?;
        ensure!(
            launches[forward]["boundary_us"].is_null(),
            "launch event boundary format"
        );

        let start_checkpoint = checkpoints[2 * forward];
        let assembled_checkpoint = checkpoints[2 * forward + 1];
        ensure!(
            start_checkpoint["stage"] == "assembly_start"
                && start_checkpoint["boundary_us"].is_null()
                && assembled_checkpoint["stage"] == "assembled"
                && assembled_checkpoint["boundary_us"] == 1000,
            "owner checkpoint stages/boundaries"
        );
        check_members(
            start_checkpoint,
            &members[..1],
            &group_plans[..1],
            0,
        )?;
        check_members(assembled_checkpoint, members, group_plans, now)?;

        let begin = &records[1 + 2 * forward];
        let end = &records[2 + 2 * forward];
        ensure!(
            begin["kind"] == "controlled_backend_forward_begin"
                && end["kind"] == "controlled_backend_forward_end"
                && end["status"] == "ok",
            "controlled physical begin/end/status"
        );

        let mut expected_rows = Vec::new();
        let mut completed_rows = Vec::new();

        for (id, expected_plan) in members.iter().zip(group_plans) {
            ensure!(seen.insert(*id), "request split or repeated");
            let (actual_plan, output) = requests.get(id).context("missing HTTP response")?;
            ensure!(
                actual_plan.case == expected_plan.case
                    && actual_plan.body == expected_plan.body
                    && actual_plan.sequences == expected_plan.sequences
                    && output.len() == expected_plan.sequences.len(),
                "HTTP plan differs from acknowledged request"
            );

            for (request_row, tokens) in expected_plan.sequences.iter().enumerate() {
                let row = json!({
                    "physical_row": expected_rows.len(),
                    "request_id": id,
                    "request_row": request_row,
                    "case_id": expected_plan.case,
                    "valid_length": tokens.len(),
                    "token_sha256": ids_sha(tokens)
                });
                let mut completed = row.clone();
                completed["output_sha256"] =
                    json!(sha(&row_bytes(&output[request_row])));
                expected_rows.push(row);
                completed_rows.push(completed);
            }
        }

        ensure!(expected_rows.len() == b, "physical row coverage");
        for record in [begin, end] {
            ensure!(
                record["backend"] == "synthetic_scheduler_probe"
                    && record["forward_id"] == forward
                    && record["B"] == b
                    && record["S"] == s
                    && record["projection_rows"] == b * s
                    && record["attention_cells_before_heads"] == b * s * s
                    && record["valid_tokens"] == valid
                    && record["padding_tokens"] == b * s - valid
                    && record["input_rows"] == b
                    && record["association_rows"] == b,
                "controlled manifest geometry/accounting"
            );
        }
        ensure!(
            begin["rows"] == json!(expected_rows)
                && end["rows"] == json!(completed_rows),
            "four-way physical/HTTP/request-row/case/output association"
        );

        forward_evidence.push(json!({
            "forward_id": forward,
            "request_ids": members,
            "B": b,
            "S": s,
            "projection_rows": b * s,
            "attention_cells_before_heads": b * s * s,
            "valid_tokens": valid,
            "padding_tokens": b * s - valid,
            "launch_us": now,
            "rows": completed_rows
        }));
    }

    ensure!(
        seen.len() == requests.len()
            && forward_evidence[0]["B"] == spec.first_b()
            && forward_evidence[1]["B"] == spec.tail_b(),
        "unmatched request or unexpected physical partition"
    );

    // All requests were acknowledged while the first owner checkpoint was
    // held. Insertion may race the first checkpoint's diagnostic event, but
    // every insertion must precede the first actual assembly observation.
    let first_assembly_seq = uint(assemblies[0], "seq")?;
    ensure!(
        insertion.iter().all(|event| {
            event["seq"].as_u64().is_some_and(|seq| seq < first_assembly_seq)
        }),
        "assembly began before all insertion acknowledgements"
    );

    let policy = BatchedExecution::default();
    let fifo_witness = if spec.mode == Mode::Fifo {
        ensure!(
            spec.first_b() == 31
                && plans[cut].sequences.len() == 2
                && plans[cut + 1].sequences.len() == 1
                && policy.fits(31, 1)
                && !policy.fits(33, 1)
                && policy.fits(32, 1)
                && policy.fits(3, 1),
            "FIFO counterfactual geometry"
        );
        json!({
            "assembled_rows": 31,
            "head_request_id": ids[cut],
            "head_request_rows": 2,
            "later_request_id": ids[cut + 1],
            "later_request_rows": 1,
            "whole_head_would_fit_first_batch": false,
            "one_head_row_would_fit_first_batch": true,
            "later_request_would_fit_first_batch": true,
            "observed_physical_batches": [31, 3],
            "head_request_split": false,
            "head_request_bypassed": false,
            "fifo_source": "acknowledged_insertion_not_request_id_sort"
        })
    } else {
        let (_, s, _) = geometry(&plans[..cut])?;
        json!({
            "observed_physical_batches": [32, 1],
            "first_batch_width": s,
            "overflow_request_id": ids[cut],
            "all_requests_intact": true,
            "overflow_attributable_to_b_ceiling": true,
            "isolated_padded_or_attention_rejection_claimed": false
        })
    };

    Ok(json!({
        "submitted_logical_requests": plans.len(),
        "submitted_real_rows": plans.iter().map(|p| p.sequences.len()).sum::<usize>(),
        "physical_forwards": forward_evidence,
        "partition_witness": fifo_witness,
        "unmatched_backend_records": 0,
        "unmatched_insertions": 0,
        "unmatched_http_responses": 0,
        "request_split": false,
        "dummy_rows": 0,
        "sequence_width_rounding": false,
        "numerical_model_evidence": false,
        "backend": "synthetic_scheduler_probe"
    }))
}

/// Run one complete local probe.
///
/// This function accepts client limits rather than Loaded so the exact local
/// harness can also be exercised by feature-gated CPU tests.
async fn local_capacity(limits: Limits, out: &Path, spec: Spec) -> Result<Value> {
    let plans = capacity_plans(spec)?;
    private_directory(out)?;

    let control = SchedulerControl::new();
    let recording = Arc::new(Mutex::new(Recording::default()));
    let instance = format!("cap54-{}", core::random_label()?);
    let mut config = Config::production(64, 131072, 1024);
    config.batching = Some(BatchedExecution::default());
    config.scheduler_control = Some(control.clone());
    config.instance = Some(instance.clone());
    service_result(config.validate())?;

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let listener = service_result(BoundedListener::new(listener, &config.transport))?;
    let client_base = format!("http://{address}");

    // Validate client construction before creating the Service owner.
    let _ = Http::new(&client_base, &instance, limits.clone())?;

    let backend_recording = recording.clone();
    let mut service = service_result(Service::spawn(
        config,
        Arc::new(LengthTokenizer),
        Logger::new(std::io::sink()),
        StartupCheck {
            tokens: vec![vec![100]],
            embeddings: vec![vector(100)],
        },
        move || {
            Ok(ControlledBackend {
                recording: backend_recording,
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
    let mut tasks = JoinSet::new();

    // Every ordinary error after owner creation flows through cleanup.
    let work: Result<Value> = async {
        service_result(service.wait_ready().await)?;
        let before = observe(&app).await?;
        ensure!(
            before["instance"] == instance
                && before["pid"] == std::process::id()
                && before["lifecycle"] == "ready"
                && before["worker_alive"] == true
                && before["ready"] == true
                && before["forward_calls"] == 0,
            "controlled capacity initial observation"
        );

        let mut insertion = Vec::new();
        for plan in &plans {
            let bytes = serde_json::to_vec(&plan.body)?;
            write_new(
                &out.join(format!("{}.request.bin", plan.case)),
                &bytes,
            )?;
            let client = Http::new(&client_base, &instance, limits.clone())?;
            let submitted = plan.clone();
            tasks.spawn(async move {
                let wire = Wire::json(submitted.body.clone())?;
                let reply = client.send(&submitted.case, &wire).await?;
                Ok::<_, anyhow::Error>((submitted, reply))
            });

            insertion.push(
                find_event(&control, |event| {
                    event["kind"] == "inserted" && event["case_id"] == plan.case
                })
                .await?,
            );
        }

        let ids: Vec<u64> = insertion
            .iter()
            .map(|event| uint(event, "request_id"))
            .collect::<Result<_>>()?;
        let cut = spec.first_requests();

        let first = checkpoint(&control, "assembly_start", ids[0]).await?;
        ensure!(
            control.now_us() == 0 && guard(&recording).records.len() == 1,
            "execution or clock progress before first assembly release"
        );
        release(&control, &first)?;

        // No clock advance is made here. B32 terminates assembly directly;
        // FIFO31 terminates upon encountering its non-fitting two-row head.
        let assembled = checkpoint(&control, "assembled", ids[0]).await?;
        check_members(&assembled, &ids[..cut], &plans[..cut], 0)?;
        ensure!(
            assembled["boundary_us"] == 1000,
            "first assembled boundary"
        );
        release(&control, &assembled)?;

        let tail_start = checkpoint(&control, "assembly_start", ids[cut]).await?;
        release(&control, &tail_start)?;

        // Wait for the entire tail to be assembled, then for the following
        // receive observation. Advancing after only assembly_start could
        // race tail coalescing and would not be a deterministic witness.
        let tail_assembly = find_event(&control, |event| {
            event["kind"] == "assembly"
                && event["request_ids"] == json!(&ids[cut..])
        })
        .await?;
        check_members(&tail_assembly, &ids[cut..], &plans[cut..], 0)?;
        let tail_assembly_seq = uint(&tail_assembly, "seq")?;
        find_event(&control, |event| {
            event["kind"] == "assembly_receive"
                && event["boundary_us"] == 1000
                && event["seq"]
                    .as_u64()
                    .is_some_and(|seq| seq > tail_assembly_seq)
        })
        .await?;

        service_result(control.advance_to_us(1000))?;
        let tail = checkpoint(&control, "assembled", ids[cut]).await?;
        check_members(&tail, &ids[cut..], &plans[cut..], 1000)?;
        ensure!(tail["boundary_us"] == 1000, "tail assembled boundary");
        release(&control, &tail)?;

        let mut requests = BTreeMap::new();
        let mut responses = Vec::new();

        while let Some(joined) = tasks.join_next().await {
            let (plan, reply) = joined.context("controlled capacity client task")??;
            write_new(
                &out.join(format!("{}.response.bin", plan.case)),
                &reply.bytes,
            )?;
            let id = reply.request_id.context("capacity HTTP request ID")?;
            let acknowledged = insertion
                .iter()
                .find(|event| event["case_id"] == plan.case)
                .context("HTTP response without acknowledged insertion")?;
            ensure!(
                acknowledged["request_id"] == id,
                "insertion/HTTP request-ID mismatch"
            );
            let output = decode_capacity(&reply, &plan)?;
            responses.push(json!({
                "case_id": plan.case,
                "request_id": id,
                "request_sha256": sha(&serde_json::to_vec(&plan.body)?),
                "response_sha256": sha(&reply.bytes),
                "http_status": reply.status,
                "headers": reply.headers,
                "rows": output.iter().enumerate().map(|(request_row, row)| {
                    json!({
                        "request_id": id,
                        "request_row": request_row,
                        "case_id": plan.case,
                        "output_sha256": sha(&row_bytes(row))
                    })
                }).collect::<Vec<_>>()
            }));
            ensure!(
                requests.insert(id, (plan, output)).is_none(),
                "duplicate capacity HTTP response ID"
            );
        }

        let after = observe(&app).await?;
        logical_delta(&before, &after, plans.len())?;
        let events = service_result(control.snapshot())?;
        let records = guard(&recording).records.clone();
        let reconciliation = reconcile_capacity(
            spec,
            &plans,
            &insertion,
            &events,
            &records,
            &requests,
        )?;

        Ok(json!({
            "schema": "rustv1.accept_m4_controlled_capacity_probe.v1",
            "probe": spec.name,
            "instance": instance,
            "pid": std::process::id(),
            "batching": BatchedExecution::default(),
            "max_sequence_tokens": 1024,
            "semantic_clock": "manual_microseconds",
            "watchdogs_are_capacity_witnesses": false,
            "observation_before": before,
            "observation_after": after,
            "logical_backend_request_delta": plans.len(),
            "responses": responses,
            "insertion_acknowledgements": insertion,
            "scheduler_events": events,
            "backend_records": records,
            "reconciliation": reconciliation
        }))
    }
    .await;

    // Close admission before opening progress gates. This does not release
    // quarantine, change job outcomes, or substitute for an owner join.
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

    tasks.abort_all();
    while tasks.join_next().await.is_some() {}

    let shutdown = service_result(service.shutdown().await);
    let _ = stop.send(());
    let transport = match timeout(WATCHDOG, &mut server).await {
        Ok(Ok(Ok(()))) => Ok(()),
        other => {
            server.abort();
            let _ = server.await;
            Err(anyhow::anyhow!(
                "controlled capacity HTTP cleanup failed: {other:?}"
            ))
        }
    };
    let _ = pump_stop.send(());
    let pump_result = pump
        .await
        .context("capacity cleanup clock task")
        .and_then(service_result);

    if let Err(error) = work {
        return Err(error).with_context(|| {
            format!(
                "capacity probe failed; owner_join_ok={}, http_join_ok={}, clock_join_ok={}",
                shutdown.is_ok(),
                transport.is_ok(),
                pump_result.is_ok()
            )
        });
    }

    let mut evidence = work?;
    shutdown?;
    transport?;
    pump_result?;
    ensure!(
        app.lifecycle() == Lifecycle::Stopped && !app.is_ready(),
        "capacity owner did not stop after successful shutdown/join"
    );

    let final_records = guard(&recording).records.clone();
    let final_events = service_result(control.snapshot())?;
    ensure!(
        evidence["backend_records"] == json!(final_records)
            && evidence["scheduler_events"] == json!(final_events),
        "additional execution or scheduler activity during successful cleanup"
    );

    let mut manifest_bytes = Vec::new();
    for record in &final_records {
        serde_json::to_writer(&mut manifest_bytes, record)?;
        manifest_bytes.push(b'\n');
    }
    write_new(
        &out.join("controlled_backend_manifest.jsonl"),
        &manifest_bytes,
    )?;
    let event_bytes = serde_json::to_vec_pretty(&final_events)?;
    write_new(&out.join("scheduler_events.json"), &event_bytes)?;

    evidence["backend_manifest_sha256"] = json!(sha(&manifest_bytes));
    evidence["scheduler_events_sha256"] = json!(sha(&event_bytes));
    evidence["cleanup"] = json!({
        "service_shutdown_returned_ok": true,
        "owner_thread_joined": true,
        "http_server_joined": true,
        "cleanup_clock_task_joined": true,
        "lifecycle": "stopped",
        "ready": false,
        "quarantine_cleared": false,
        "drop_counted_as_join": false,
        "shutdown_timeout_counted_as_cleanup": false
    });
    evidence["numerical_model_evidence"] = json!(false);
    evidence["scheduler_phase_complete"] = json!(false);
    evidence["m4_accepted"] = json!(false);
    Ok(evidence)
}

async fn check_live_unchanged(state: &SchedulerState) -> Result<()> {
    state.launch.process.check_live()?;
    ensure!(
        state.core.control.call("observe", None).await? == state.observation
            && bounded(&state.launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
        "unreconciled external server activity during local capacity probes"
    );
    Ok(())
}

fn case_specs(case: &str) -> Result<Vec<Spec>> {
    Ok(match case {
        PADDED_CASE => vec![
            Spec {
                mode: Mode::Padded,
                width: 1023,
                name: "padded-below",
            },
            Spec {
                mode: Mode::Padded,
                width: 1024,
                name: "padded-at",
            },
        ],
        ATTENTION_CASE => vec![
            Spec {
                mode: Mode::Attention,
                width: 1023,
                name: "attention-below",
            },
            Spec {
                mode: Mode::Attention,
                width: 1024,
                name: "attention-at",
            },
        ],
        FIFO_CASE => vec![Spec {
            mode: Mode::Fifo,
            width: 1,
            name: "fifo",
        }],
        _ => anyhow::bail!("unknown family-54 capacity case"),
    })
}

fn boundary_summary(case: &str, probes: &[Value]) -> Result<Value> {
    match case {
        PADDED_CASE => {
            ensure!(probes.len() == 2, "padded probe coverage");
            let below = &probes[0]["reconciliation"]["physical_forwards"][0];
            let at = &probes[1]["reconciliation"]["physical_forwards"][0];
            ensure!(
                below["B"] == 32
                    && below["S"] == 1023
                    && below["projection_rows"] == 32736
                    && below["valid_tokens"] == 1054
                    && below["padding_tokens"] == 31682
                    && at["B"] == 32
                    && at["S"] == 1024
                    && at["projection_rows"] == 32768
                    && at["valid_tokens"] == 1055
                    && at["padding_tokens"] == 31713,
                "padded-token boundary/accounting witness"
            );
            Ok(json!({
                "scope": "full_envelope_padded_token_boundary_and_accounting",
                "below_boundary_padded_tokens": 32736,
                "at_boundary_padded_tokens": 32768,
                "at_boundary_valid_tokens": 1055,
                "at_boundary_padding_tokens": 31713,
                "uses_padded_not_valid_token_count": true,
                "physical_batches_in_each_probe": [32, 1],
                "independent_padded_rejection_predicate_proven": false,
                "independent_rejection_unreachable_under_fixed_b_s_bounds": true
            }))
        }
        ATTENTION_CASE => {
            ensure!(probes.len() == 2, "attention probe coverage");
            let below = &probes[0]["reconciliation"]["physical_forwards"][0];
            let at = &probes[1]["reconciliation"]["physical_forwards"][0];
            ensure!(
                below["B"] == 32
                    && below["S"] == 1023
                    && below["attention_cells_before_heads"] == 32 * 1023 * 1023
                    && below["projection_rows"] == 32736
                    && below["padding_tokens"] == 0
                    && at["B"] == 32
                    && at["S"] == 1024
                    && at["attention_cells_before_heads"] == 33554432
                    && at["projection_rows"] == 32768
                    && at["padding_tokens"] == 0,
                "attention-cell boundary/accounting witness"
            );
            Ok(json!({
                "scope": "full_envelope_attention_cell_boundary_and_accounting",
                "below_boundary_attention_cells_before_heads": 32 * 1023 * 1023,
                "at_boundary_attention_cells_before_heads": 33554432,
                "attention_count_formula": "B*S*S_before_heads",
                "physical_batches_in_each_probe": [32, 1],
                "independent_attention_rejection_predicate_proven": false,
                "independent_rejection_unreachable_under_fixed_b_s_bounds": true
            }))
        }
        FIFO_CASE => {
            ensure!(probes.len() == 1, "FIFO probe coverage");
            let witness = &probes[0]["reconciliation"]["partition_witness"];
            ensure!(
                witness["observed_physical_batches"] == json!([31, 3])
                    && witness["whole_head_would_fit_first_batch"] == false
                    && witness["one_head_row_would_fit_first_batch"] == true
                    && witness["later_request_would_fit_first_batch"] == true
                    && witness["head_request_split"] == false
                    && witness["head_request_bypassed"] == false,
                "FIFO/no-split/no-bypass witness"
            );
            Ok(witness.clone())
        }
        _ => anyhow::bail!("unknown capacity summary case"),
    }
}

async fn execute_case(state: &mut SchedulerState, case: &str) -> Result<()> {
    ensure!(
        state.core.ledger.pending().first() == Some(&json!({
            "acceptance_phase": policy::SCHEDULER,
            "case_id": case
        })),
        "remaining capacity runner called at wrong ledger position"
    );

    let result: Result<Value> = async {
        state.core.loaded.recheck()?;
        envelope(&state.core.loaded)?;
        check_live_unchanged(state).await?;

        let relation = envelope_relationship()?;
        let out = state.directory.join(case);
        private_directory(&out)?;
        let mut probes = Vec::new();
        let mut probe_bindings = Vec::new();

        for spec in case_specs(case)? {
            let limits: Limits =
                serde_json::from_value(state.core.loaded.policy["limits"].clone())?;
            let probe_out = out.join(spec.name);
            let mut evidence = local_capacity(limits, &probe_out, spec).await?;

            // Do not establish a new live-server baseline after local work.
            check_live_unchanged(state).await?;
            evidence["policy_sha256"] = json!(state.core.loaded.policy_sha256);
            evidence["driver_binary_sha256"] =
                json!(state.core.loaded.artifacts["accept_m4"].sha256);
            evidence["live_manifest_unchanged_sha256"] =
                json!(sha(&state.manifest_prefix));

            let digest = save_json(&probe_out.join("evidence.json"), &evidence)?;
            probe_bindings.push(json!({
                "path": format!("{}/evidence.json", spec.name),
                "sha256": digest,
                "owner_thread_joined": true,
                "http_server_joined": true,
                "cleanup_clock_task_joined": true
            }));
            probes.push(evidence);
        }

        let summary = boundary_summary(case, &probes)?;
        state.core.loaded.recheck()?;
        check_live_unchanged(state).await?;
        state.core.orchestration.sync()?;

        // Includes every reconciled physical row, so the global ledger's
        // digest-bound evidence retains request ID, request row, case ID,
        // and output hash rather than just an aggregate batch-size claim.
        let evidence = json!({
            "schema": "rustv1.accept_m4_capacity_case.v1",
            "case_id": case,
            "policy_sha256": state.core.loaded.policy_sha256,
            "driver_binary_sha256": state.core.loaded.artifacts["accept_m4"].sha256,
            "batching": state.core.loaded.bundle.shapes.batching,
            "max_sequence_tokens": 1024,
            "envelope_relationship": relation,
            "summary": summary,
            "probe_bindings": probe_bindings,
            "probes": probes,
            "live_instance": state.launch.instance,
            "live_pid": state.launch.pid,
            "live_observation_unchanged": state.observation,
            "live_manifest_unchanged_sha256": sha(&state.manifest_prefix),
            "live_next_forward_id_unchanged": state.next_forward,
            "production_scheduler_used": true,
            "backend": "synthetic_scheduler_probe",
            "numerical_model_evidence": false,
            "all_local_owners_joined": true,
            "unmatched_backend_records": 0,
            "scheduler_phase_complete": false,
            "m4_accepted": false
        });
        let evidence_sha256 = save_json(&out.join("evidence.json"), &evidence)?;
        Ok(json!({
            "evidence_path": format!("scheduler/{case}/evidence.json"),
            "evidence_sha256": evidence_sha256,
            "summary": summary,
            "production_scheduler_used": true,
            "backend": "synthetic_scheduler_probe",
            "numerical_model_evidence": false,
            "all_local_owners_joined": true,
            "unmatched_backend_records": 0
        }))
    }
    .await;

    match result {
        Ok(details) => {
            state.core.ledger.append(
                policy::SCHEDULER,
                case,
                true,
                details,
                true,
            )?;
            Ok(())
        }
        Err(error) => {
            let _ = state.core.ledger.append(
                policy::SCHEDULER,
                case,
                false,
                json!({"error": format!("{error:#}")}),
                true,
            );
            Err(error)
        }
    }
}

async fn execute_remaining(mut state: SchedulerState) -> Result<SchedulerState> {
    for case in [PADDED_CASE, ATTENTION_CASE, FIFO_CASE] {
        execute_case(&mut state, case).await?;
    }

    ensure!(
        state.core.ledger.pending().first() == Some(&json!({
            "acceptance_phase": policy::SCHEDULER,
            "case_id": FOLLOWING_CASE
        })),
        "unexpected post-family-54 ledger position"
    );
    Ok(state)
}

pub fn run(continuation: SchedulerContinuation) -> Result<SchedulerContinuation> {
    let SchedulerContinuation { runtime, state } = continuation;
    let state = runtime.block_on(execute_remaining(state))?;
    Ok(SchedulerContinuation { runtime, state })
}

#[cfg(test)]
mod capacity_tests {
    use super::*;

    fn test_limits() -> Limits {
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

    #[test]
    fn full_envelope_has_no_isolated_token_or_attention_violation() {
        let relation = envelope_relationship().unwrap();
        assert_eq!(
            relation["isolated_padded_limit_violation_in_legal_b_s_domain"],
            false
        );
        assert_eq!(
            relation["isolated_attention_limit_violation_in_legal_b_s_domain"],
            false
        );
    }

    #[test]
    fn padded_probe_distinguishes_valid_from_padded_tokens() {
        let spec = Spec {
            mode: Mode::Padded,
            width: 1024,
            name: "padded-at",
        };
        let plans = capacity_plans(spec).unwrap();
        assert_eq!(geometry(&plans[..4]).unwrap(), (32, 1024, 1055));
        assert_eq!(geometry(&plans[4..]).unwrap(), (1, 1, 1));

        let tokenizer = LengthTokenizer;
        for plan in plans {
            let texts = plan.body["input"].as_array().unwrap();
            for (text, tokens) in texts.iter().zip(&plan.sequences) {
                assert_eq!(
                    tokenizer.encode(text.as_str().unwrap()).unwrap(),
                    *tokens
                );
            }
        }
    }

    #[test]
    fn fifo_plan_exposes_both_split_and_bypass_opportunities() {
        let spec = Spec {
            mode: Mode::Fifo,
            width: 1,
            name: "fifo",
        };
        let plans = capacity_plans(spec).unwrap();
        let policy = BatchedExecution::default();

        assert_eq!(geometry(&plans[..4]).unwrap(), (31, 1, 31));
        assert_eq!(plans[4].sequences.len(), 2);
        assert_eq!(plans[5].sequences.len(), 1);
        assert!(!policy.fits(31 + plans[4].sequences.len(), 1));
        assert!(policy.fits(31 + plans[5].sequences.len(), 1));
        assert_eq!(geometry(&plans[4..]).unwrap(), (3, 1, 3));
    }

    // Runs the actual local HTTP/Service/manual-clock harness. No finalized
    // model policy, CUDA device, external supervisor, or global ledger needed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn production_scheduler_capacity_probes() {
        let root = std::env::temp_dir().join(format!(
            "rustv1-capacity-test-{}-{}",
            std::process::id(),
            core::random_label().unwrap()
        ));
        private_directory(&root).unwrap();

        let result: Result<()> = async {
            for case in [PADDED_CASE, ATTENTION_CASE, FIFO_CASE] {
                let mut probes = Vec::new();
                for spec in case_specs(case)? {
                    let probe_out = root.join(spec.name);
                    let evidence =
                        local_capacity(test_limits(), &probe_out, spec).await?;
                    save_json(&probe_out.join("evidence.json"), &evidence)?;
                    super::super::saved_local::mutation_checks(
                        &probe_out,
                        &evidence,
                    )?;
                    ensure!(
                        evidence["cleanup"]["owner_thread_joined"] == true
                            && evidence["cleanup"]["http_server_joined"] == true
                            && evidence["cleanup"]["cleanup_clock_task_joined"] == true,
                        "test probe did not join all local owners"
                    );
                    probes.push(evidence);
                }
                boundary_summary(case, &probes)?;
            }
            Ok(())
        }
        .await;

        if result.is_ok() {
            fs::remove_dir_all(&root).unwrap();
        }
        result.unwrap_or_else(|error| {
            panic!(
                "capacity harness failed; retained evidence at {}: {error:#}",
                root.display()
            )
        });
    }
}
