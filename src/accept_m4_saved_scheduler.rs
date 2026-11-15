// Included at acceptance_scheduler::saved_initial::saved_scheduler.
//
// Offline scheduler launch/coalescing/manifest/transcript reconciliation.
//
// No control or HTTP calls, process observations, owner joins, child reaps,
// orchestration exchanges, ledger appends, or completion markers.
//
// Family-53 has no historical driver_binary_sha256 producer field.
// Preserve that schema rather than adding a retrospective commitment.

use super::*;

const SCHEDULER_BOUNDARY: &str =
    "SCHEDULER-ARTIFACTS-MANIFEST-AND-TRANSCRIPT-RECONCILED";
const BINDING_CASE: &str = "replacement-launch-after-independent-join";

pub(crate) fn retained(
    continuation: &IndependentArtifactsContinuation,
) -> &FinalDrainState {
    &continuation.poisoned_launch.destructive_owners.recovery_http
        .recovery_lineage.initial_artifacts.imported_artifacts
        .local_artifacts.final_drain.state
}

fn array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    value[key].as_array().with_context(|| format!("scheduler array: {key}"))
}

/// Preserve the exact bytes preceding a JSONL record, not a reserialization.
fn prefix(bytes: &[u8], count: usize) -> Result<&[u8]> {
    let records = core::jsonl(bytes)?;
    ensure!(count <= records.len(), "scheduler JSONL prefix count");
    let end: usize = bytes.split_inclusive(|b| *b == b'\n')
        .take(count).map(<[u8]>::len).sum();
    Ok(&bytes[..end])
}

fn frame(direction: &str, envelope: &Value) -> Value {
    json!({"direction": direction, "envelope": envelope})
}

/// The caller has authenticated all frames with verify_transcript.
/// The preceding reconciliation chain checks positions 0..3 and 6..14.
fn scheduler_frames(
    frames: &[Value],
    request: &Value,
    accepted: &Value,
    ready: &Value,
) -> Result<()> {
    ensure!(
        frames.len() == 14
            && frames[3..6] == [
                frame("driver_to_orchestrator", request),
                frame("orchestrator_to_driver", accepted),
                frame("orchestrator_to_driver", ready),
            ],
        "scheduler handshake is not the exact remaining transcript segment"
    );
    Ok(())
}

/// Check complete authorization/acknowledgement field objects.
///
/// This is historical authorization ordering, not an independent measurement
/// of process-creation time.
fn authorization(
    binding: &Value,
    initial_instance: &str,
    initial_pid: u32,
    policy_digest: &str,
    join_digest: &str,
    absence: &Value,
) -> Result<()> {
    let request = &binding["authenticated_launch_request"];
    let accepted = &binding["authenticated_launch_accepted"];
    let ready = &binding["authenticated_ready"];
    let nonce = text(&request["fields"], "launch_request_id")?;
    ensure!(
        crate::contract::digest_valid(nonce)
            && request["kind"] == "orch_request_launch"
            && request["fields"] == json!({
                "launch_request_id": nonce,
                "reason": "after_independent_join",
                "previous_instance": initial_instance,
                "previous_pid": initial_pid,
                "policy_sha256": policy_digest,
                "owner_closure_sha256": join_digest,
                "old_absence_sha256": sha(&serde_json::to_vec(absence)?)
            })
            && accepted["kind"] == "orch_launch_accepted"
            && accepted["fields"] == json!({
                "launch_request_id": nonce,
                "request_seq": uint(request, "seq")?,
                "previous_instance": initial_instance
            })
            && ready["kind"] == "orch_ready"
            && ready["fields"]["launch_request_id"] == nonce,
        "scheduler launch authorization/acknowledgement changed"
    );
    closed(&ready["fields"], &[
        "launch_request_id", "instance", "pid", "start_utc",
        "manifest", "attestation",
    ])?;
    Ok(())
}

fn launch_artifacts(
    old: &SchedulerState,
    frames: &[Value],
    inventory: &mut BTreeMap<String, String>,
) -> Result<Value> {
    let state = &old.core;
    let loaded = &state.loaded;
    let out = &state.out;
    let launch = &old.launch;
    let initial = &state.initial_launch;
    envelope(loaded)?;

    let bytes = read_committed(
        out, "scheduler/launch_binding.json", &launch.binding_sha256,
        JSON_LIMIT, inventory,
    )?;
    let binding: Value = serde_json::from_slice(&bytes)?;
    let join_bytes = read_output(
        out, "independent/owner_join.json", JSON_LIMIT, inventory,
    )?;
    ensure!(
        join_bytes == serde_json::to_vec_pretty(&state.independent_join)?
            && state.independent_join["owner_thread_joined"] == true
            && state.independent_join["service_shutdown_returned_ok"] == true
            && state.independent_join["http_server_joined"] == true
            && state.independent_join["lifecycle"] == "stopped"
            && state.independent_join["ready"] == false,
        "scheduler prerequisite independent join changed"
    );
    let join_digest = sha(&join_bytes);
    let absence = absence_record(&initial.process);
    ensure!(
        state.initial_exit["process_exit"] == absence,
        "scheduler prerequisite initial absence changed"
    );
    authorization(
        &binding, &initial.instance, initial.pid,
        &loaded.policy_sha256, &join_digest, &absence,
    )?;

    let request = &binding["authenticated_launch_request"];
    let accepted = &binding["authenticated_launch_accepted"];
    let ready = &binding["authenticated_ready"];
    scheduler_frames(frames, request, accepted, ready)?;

    let fields = &ready["fields"];
    ensure!(
        label_valid(&launch.instance)
            && launch.instance != initial.instance
            && launch.pid > 0
            && launch.pid != initial.pid
            && launch.pid != std::process::id()
            && launch.process.pid == launch.pid
            && launch.process.executable_sha256
                == loaded.artifacts["serve_m3"].sha256
            && fields["instance"] == launch.instance
            && fields["pid"] == launch.pid
            && fields["start_utc"] == launch.start_utc
            && !launch.start_utc.is_empty()
            && launch.start_utc.len() <= 80,
        "scheduler saved launch/process identity"
    );

    // Reuse only the filesystem isolation helper, never bind_replacement.
    let handoff = policy::Handoff::read(&out.join("http_handoff.json"))?;
    ensure!(
        new_evidence_path(text(fields, "manifest")?, state, &handoff)?
            == launch.manifest
            && new_evidence_path(text(fields, "attestation")?, state, &handoff)?
                == launch.attestation
            && launch.manifest != launch.attestation,
        "scheduler saved external path isolation"
    );

    let attestation_bytes =
        regular_external_bytes(&launch.attestation, JSON_LIMIT)?;
    let attestation: Value = serde_json::from_slice(&attestation_bytes)?;
    ensure!(
        attestation["schema"] == "rustv1.server_launch_attestation.v1"
            && attestation["instance_id"] == launch.instance
            && attestation["pid"] == launch.pid
            && attestation["start_utc"] == launch.start_utc
            && attestation["settings_sha256"] == loaded.policy_sha256
            && attestation["binary_sha256"] == loaded.artifacts["serve_m3"].sha256
            && attestation["audit_sha256"] == loaded.artifacts["audit"].sha256
            && attestation["startup_check_sha256"]
                == loaded.artifacts["startup_check"].sha256
            && attestation["execution_policy"] == BATCHED_EXECUTION_POLICY
            && attestation["http_contract_id"] == BATCHED_HTTP_CONTRACT
            && attestation["batching"]
                == serde_json::to_value(&loaded.bundle.shapes.batching)?,
        "scheduler saved attestation policy binding"
    );
    for name in [
        "rustc_version", "nvcc_version", "cuda_compute_cap", "target_features",
    ] {
        ensure!(
            !text(&attestation["build_identity"], name)?.is_empty(),
            "empty scheduler build identity"
        );
    }

    let attested = &binding["authenticated_observation"];
    let mut plain = attested.clone();
    ensure!(
        plain.as_object_mut().context("scheduler attested observation")?
            .remove("attestation").is_some()
            && plain == launch.initial_observation
            && ready_observation(&plain, launch)? == 0,
        "scheduler fresh historical observation"
    );
    let runtime = &attested["attestation"];
    let mut base = runtime.clone();
    let object = base.as_object_mut().context("scheduler runtime attestation")?;
    for name in ["http_address", "control_address", "startup", "capabilities"] {
        ensure!(object.remove(name).is_some(), "missing runtime field");
    }
    ensure!(
        base == attestation
            && serde_json::to_vec_pretty(&base)? == attestation_bytes,
        "scheduler runtime/base attestation changed"
    );

    let expected_http = core::base_address(&state.base_url)?;
    let actual_http: SocketAddr = text(runtime, "http_address")?.parse()?;
    let actual_control: SocketAddr = text(runtime, "control_address")?.parse()?;
    let expected_control: SocketAddr =
        text(&loaded.policy, "launch_control_address")?.parse()?;
    ensure!(
        actual_control == expected_control
            && actual_http.port() == expected_http.port()
            && (actual_http.ip() == expected_http.ip()
                || actual_http.ip().is_unspecified()),
        "scheduler saved endpoint binding"
    );

    let startup = core::check_startup(
        &launch.initial_manifest, loaded, &sha(&attestation_bytes), true,
    )?;
    ensure!(
        startup == launch.startup
            && runtime["startup"] == json!({
                "completed": true,
                "comparison": "exact_f32_bits",
                "output_f32le_sha256": startup["startup_output_sha256"]
            })
            && plain["startup_output_sha256"] == startup["startup_output_sha256"]
            && crate::contract::digest_valid(text(&binding, "challenge")?),
        "scheduler startup/challenge binding"
    );

    let expected = json!({
        "schema": "rustv1.accept_m4_scheduler_launch_binding.v1",
        "authenticated_ready": ready,
        "challenge": binding["challenge"],
        "authenticated_observation": attested,
        "policy_sha256": loaded.policy_sha256,
        "attestation_sha256": sha(&attestation_bytes),
        "startup": startup,
        "independent_join_sha256": join_digest,
        "old_process_absence_rechecked": absence,
        "process_start_ticks": launch.process.start_ticks,
        "boot_id": launch.process.boot_id,
        "running_binary_sha256": launch.process.executable_sha256,
        "launch_performed_by": "external_supervisor",
        "authenticated_launch_request": request,
        "authenticated_launch_accepted": accepted,
        "launch_authorization_after_independent_join": true,
        "process_creation_after_join_independently_measured": false
    });
    ensure!(
        binding == expected && bytes == serde_json::to_vec_pretty(&expected)?,
        "complete scheduler launch binding changed"
    );
    expect_bytes(
        out, "scheduler/launch_attestation.json",
        &attestation_bytes, JSON_LIMIT, inventory,
    )?;
    expect_bytes(
        out, "scheduler/startup_executor_manifest.jsonl",
        &launch.initial_manifest, MANIFEST_LIMIT, inventory,
    )?;
    Ok(binding)
}

fn consumed_binding(
    old: &SchedulerState,
    records: &[Value],
    transcript: &[u8],
    binding: &Value,
    inventory: &mut BTreeMap<String, String>,
) -> Result<()> {
    let rows: Vec<_> = records.iter().filter(|row| {
        row["acceptance_phase"] == policy::LIFECYCLE
            && row["case_id"] == BINDING_CASE
    }).collect();
    ensure!(rows.len() == 1, "scheduler consumed-binding ledger coverage");
    let relative = format!("scheduler/{BINDING_CASE}/evidence.json");
    let digest = text(&rows[0]["details"], "evidence_sha256")?;
    ensure!(
        rows[0]["details"] == json!({
            "evidence_path": relative,
            "evidence_sha256": digest,
            "instance": old.launch.instance,
            "pid": old.launch.pid
        }),
        "scheduler consumed-binding ledger details"
    );
    let bytes = read_committed(
        &old.core.out, &relative, digest, JSON_LIMIT, inventory,
    )?;
    let expected = json!({
        "schema": "rustv1.accept_m4_consumed_replacement_binding.v1",
        "case_id": BINDING_CASE,
        "launch_binding_path": "scheduler/launch_binding.json",
        "launch_binding_sha256": old.launch.binding_sha256,
        "independent_join_sha256": binding["independent_join_sha256"],
        "orchestration_transcript_sha256": sha(prefix(transcript, 6)?),
        "transcript_positions": {
            "request": 3,
            "acknowledgement": 4,
            "ready": 5
        },
        "request": binding["authenticated_launch_request"],
        "acknowledgement": binding["authenticated_launch_accepted"],
        "ready": binding["authenticated_ready"],
        "instance": old.launch.instance,
        "pid": old.launch.pid,
        "launch_authorization_after_independent_join": true,
        "process_creation_after_join_independently_measured": false,
        "additional_launch_requested": false,
        "additional_supervisor_message_received": false,
        "policy_sha256": old.core.loaded.policy_sha256,
        "driver_binary_sha256": old.core.loaded.artifacts["accept_m4"].sha256
    });
    ensure!(
        serde_json::from_slice::<Value>(&bytes)? == expected
            && bytes == serde_json::to_vec_pretty(&expected)?,
        "complete consumed scheduler launch binding changed"
    );
    Ok(())
}

fn interval<'a>(
    manifest: &'a [u8],
    commitment: &Value,
    cursor: usize,
    name: &str,
) -> Result<&'a [u8]> {
    closed(commitment, &[
        "path", "start_byte", "end_byte", "prefix_before_sha256",
        "prefix_after_sha256", "interval_sha256",
    ])?;
    let start = usize::try_from(uint(commitment, "start_byte")?)?;
    let end = usize::try_from(uint(commitment, "end_byte")?)?;
    ensure!(
        commitment["path"] == name
            && start == cursor && start < end && end <= manifest.len(),
        "scheduler interval name/gap/overlap/bounds"
    );
    ensure!(
        commitment["prefix_before_sha256"] == sha(&manifest[..start])
            && commitment["prefix_after_sha256"] == sha(&manifest[..end])
            && commitment["interval_sha256"] == sha(&manifest[start..end]),
        "scheduler interval commitment changed"
    );
    Ok(&manifest[start..end])
}

/// Check only retained connection evidence. Never consult the live registry.
fn response_connection(
    response: &Value,
    instance: &str,
    case: &str,
    request_digest: &str,
    endpoint: SocketAddr,
) -> Result<u64> {
    closed(response, &[
        "case_id", "request_id", "request_sha256", "response_sha256",
        "http_status", "headers", "usage",
    ])?;
    let id = uint(response, "request_id")?;
    let headers = &response["headers"];
    closed(headers, &[
        "x-rv1-instance-id", "x-acceptance-case-id",
        "x-rv1-request-id", "content-type", "_connection_owner",
    ])?;
    ensure!(
        id > 0
            && response["case_id"] == case
            && response["http_status"] == 200
            && response["request_sha256"] == request_digest
            && headers["x-rv1-instance-id"] == instance
            && headers["x-acceptance-case-id"] == case
            && headers["x-rv1-request-id"] == id
            && text(headers, "content-type")?.split(';').next()
                .unwrap_or("").trim().eq_ignore_ascii_case("application/json"),
        "coalescing response identity/correlation/media"
    );
    let connection = &headers["_connection_owner"];
    closed(connection, &[
        "instance", "case_id", "method", "path",
        "local_address", "peer_address", "request_sha256",
        "connection_id", "owner_pid", "tokio_task_id",
        "join_outcome", "abort_requested", "connection_result",
        "join_error", "join_awaited",
    ])?;
    let local: SocketAddr = text(connection, "local_address")?.parse()?;
    let peer: SocketAddr = text(connection, "peer_address")?.parse()?;
    ensure!(
        connection["instance"] == instance
            && connection["case_id"] == case
            && connection["method"] == "POST"
            && connection["path"] == "/v1/embeddings"
            && connection["request_sha256"] == request_digest
            && connection["owner_pid"] == std::process::id()
            && connection["join_awaited"] == true
            && connection["abort_requested"].is_boolean()
            && !text(connection, "tokio_task_id")?.is_empty()
            && local.ip().is_loopback() && local.port() != 0
            && peer == endpoint,
        "coalescing saved connection identity"
    );
    match text(connection, "join_outcome")? {
        "completed" => {
            let result = &connection["connection_result"];
            closed(result, &["ok", "error"])?;
            ensure!(
                connection["join_error"].is_null()
                    && ((result["ok"] == true && result["error"].is_null())
                        || (result["ok"] == false && result["error"].is_string())),
                "completed coalescing connection result"
            );
        }
        "cancelled-after-abort-and-awaited" => {
            ensure!(
                connection["abort_requested"] == true
                    && connection["connection_result"].is_null()
                    && !text(connection, "join_error")?.is_empty(),
                "coalescing connection cancellation proof"
            );
        }
        _ => anyhow::bail!("unjoined or panicked coalescing connection"),
    }
    uint(connection, "connection_id")
}

/// Stop at the first successful coalescing wave, as the producer did.
/// Earlier non-coalescing waves remain part of complete manifest coverage.
fn wave_termination(waves: &[Value]) -> Result<()> {
    ensure!((1..=MAX_WAVES).contains(&waves.len()), "coalescing wave bound");
    for (index, wave) in waves.iter().enumerate() {
        ensure!(
            !array(wave, "coalescing_witnesses")?.is_empty()
                == (index + 1 == waves.len()),
            "coalescing waves did not stop at first witness"
        );
    }
    Ok(())
}

fn coalescing(
    old: &SchedulerState,
    records: &[Value],
    inventory: &mut BTreeMap<String, String>,
) -> Result<Value> {
    let out = &old.core.out;
    let loaded = &old.core.loaded;
    let launch = &old.launch;
    let rows: Vec<_> = records.iter().filter(|row| {
        row["acceptance_phase"] == policy::SCHEDULER
            && row["case_id"] == FIRST_CASE
    }).collect();
    ensure!(rows.len() == 1, "coalescing ledger coverage");
    ensure!(
        rows[0]["details"] == json!({
            "evidence_path": "scheduler/family-53.evidence.json",
            "evidence_sha256": old.coalescing_evidence_sha256,
            "launch_binding_sha256": launch.binding_sha256,
            "instance": launch.instance,
            "pid": launch.pid,
            "unmatched_physical_records": 0
        }),
        "complete coalescing ledger details changed"
    );
    let bytes = read_committed(
        out, "scheduler/family-53.evidence.json",
        &old.coalescing_evidence_sha256, JSON_LIMIT, inventory,
    )?;
    let saved: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        saved.get("driver_binary_sha256").is_none(),
        "family-53 historical producer schema changed"
    );
    let waves = array(&saved, "waves")?;
    wave_termination(waves)?;

    let manifest = &old.manifest_prefix;
    ensure!(
        manifest.starts_with(&launch.initial_manifest),
        "scheduler final manifest does not extend startup"
    );
    let records_count = core::jsonl(manifest)?.len();
    let mut cursor = launch.initial_manifest.len();
    let mut next = uint(&launch.startup, "next_forward_id")?;
    let mut observation = launch.initial_observation.clone();
    let mut ids = BTreeSet::new();
    let mut connection_ids = BTreeSet::new();
    let mut checked_waves = Vec::new();
    let endpoint = core::base_address(&old.core.base_url)?;

    for (index, wave) in waves.iter().enumerate() {
        let plans = wave_plan(&loaded.bundle, index)?;
        ensure!(plans.len() == WAVE_REQUESTS, "coalescing producer plan count");
        let responses = array(wave, "responses")?;
        ensure!(responses.len() == plans.len(), "coalescing response count");
        let mut cases = BTreeSet::new();
        let mut completed = BTreeMap::new();

        // Responses were recorded in JoinSet completion order. Preserve that
        // order; match each response to the independent plan by case label.
        for response in responses {
            let case = text(response, "case_id")?;
            ensure!(cases.insert(case.to_owned()), "duplicate wave response case");
            let plan = plans.iter().find(|plan| plan.case == case)
                .context("unplanned coalescing response case")?;
            let request_digest = sha(&plan.bytes);
            expect_bytes(
                out, &format!("scheduler/{case}.request.bin"),
                &plan.bytes, JSON_LIMIT, inventory,
            )?;
            let connection = response_connection(
                response, &launch.instance, case, &request_digest, endpoint,
            )?;
            ensure!(
                connection_ids.insert(connection),
                "coalescing connection identity reused"
            );
            let id = uint(response, "request_id")?;
            ensure!(ids.insert(id), "coalescing request identity reused");
            let response_bytes = read_committed(
                out, &format!("scheduler/{case}.response.bin"),
                text(response, "response_sha256")?, JSON_LIMIT, inventory,
            )?;
            let reply = http::Reply {
                status: 200,
                request_id: Some(id),
                headers: response["headers"].clone(),
                bytes: response_bytes,
            };
            let (output, usage) = http::decode_success(
                &reply, &loaded.bundle, &plan.fixtures, "float",
            )?;
            ensure!(response["usage"] == usage, "coalescing usage changed");
            ensure!(
                completed.insert(id, CompletedRequest {
                    case: case.to_owned(),
                    fixtures: plan.fixtures.clone(),
                    output,
                }).is_none(),
                "duplicate completed coalescing request"
            );
        }

        let interval_name = format!("wave-{index}.executor_interval.jsonl");
        let suffix = interval(
            manifest, &wave["manifest_interval"], cursor, &interval_name,
        )?;
        expect_bytes(
            out, &format!("scheduler/{interval_name}"),
            suffix, MANIFEST_LIMIT, inventory,
        )?;
        let physical = reconcile_wave(suffix, next, &completed, &loaded.bundle)?;
        let after = &wave["observation_after"];
        ready_observation(after, launch)?;
        logical_delta(&observation, after, WAVE_REQUESTS)?;
        let end = usize::try_from(uint(&wave["manifest_interval"], "end_byte")?)?;
        let expected = json!({
            "wave": index,
            "instance": launch.instance,
            "pid": launch.pid,
            "submitted_logical_requests": WAVE_REQUESTS,
            "logical_backend_request_delta": WAVE_REQUESTS,
            "observation_before": observation,
            "observation_after": after,
            "responses": responses,
            "manifest_interval": {
                "path": interval_name,
                "start_byte": cursor,
                "end_byte": end,
                "prefix_before_sha256": sha(&manifest[..cursor]),
                "prefix_after_sha256": sha(&manifest[..end]),
                "interval_sha256": sha(suffix)
            },
            "physical_forwards": physical.forwards,
            "coalescing_witnesses": physical.witnesses,
            "unmatched_physical_records": 0,
            "request_id_order_used_as_fifo_insertion_proof": false
        });
        ensure!(*wave == expected, "complete coalescing wave evidence changed");
        expect_pretty(
            out, &format!("scheduler/wave-{index}.json"),
            &expected, inventory,
        )?;
        checked_waves.push(expected);
        cursor = end;
        next = physical.next_forward;
        observation = after.clone();
    }

    let expected_count = usize::try_from(next)?
        .checked_mul(2).and_then(|n| n.checked_add(1))
        .context("scheduler record count overflow")?;
    ensure!(
        cursor == manifest.len()
            && next == old.next_forward
            && records_count == expected_count
            && ids.len() == waves.len() * WAVE_REQUESTS,
        "scheduler complete startup/serving manifest coverage"
    );
    logical_delta(
        &launch.initial_observation, &observation, waves.len() * WAVE_REQUESTS,
    )?;

    let expected = json!({
        "schema": "rustv1.accept_m4_live_coalescing.v1",
        "case_id": FIRST_CASE,
        "instance": launch.instance,
        "pid": launch.pid,
        "launch_binding_sha256": launch.binding_sha256,
        "policy_sha256": loaded.policy_sha256,
        "batching": loaded.bundle.shapes.batching,
        "maximum_sequence_length": 1024,
        "maximum_waves": MAX_WAVES,
        "maximum_concurrent_http_requests": WAVE_REQUESTS,
        "waves": checked_waves,
        "final_observation": observation,
        "final_manifest_sha256": sha(manifest),
        "next_forward_id": next,
        "unmatched_physical_records": 0,
        "coalescing_proven_by": [
            "logical_request_observation_delta",
            "complete_executor_manifest_delta",
            "one_physical_forward_with_multiple_distinct_request_ids"
        ],
        "fifo_insertion_order_proven": false,
        "batching_wait_bound_measured": false,
        "destructive_restart_proven": false,
        "scheduler_phase_complete": false,
        "m4_accepted": false
    });
    ensure!(
        saved == expected && bytes == serde_json::to_vec_pretty(&expected)?,
        "complete family-53 producer evidence changed"
    );
    expect_bytes(
        out, "scheduler/family-53.executor_snapshot.jsonl",
        manifest, MANIFEST_LIMIT, inventory,
    )?;
    ensure!(
        regular_external_bytes(&launch.manifest, MANIFEST_LIMIT)? == *manifest,
        "external scheduler manifest differs from complete reconciled bytes"
    );

    // The preceding poison reconciliation independently checks the trigger
    // request and its zero-byte physical interval. Bind that history here.
    let mut poisoned = observation.clone();
    poisoned["ready"] = json!(false);
    poisoned["lifecycle"] = json!("poisoned");
    poisoned["forward_calls"] = json!(
        uint(&observation, "forward_calls")?.checked_add(1)
            .context("scheduler poison logical overflow")?
    );
    ensure!(
        old.observation == poisoned
            && old.seen_request_ids.is_superset(&ids)
            && old.seen_request_ids.len() == ids.len() + 1,
        "scheduler coalescing/poison historical cursor continuity"
    );

    Ok(json!({
        "waves": waves.len(),
        "logical_requests": ids.len(),
        "saved_connection_joins_checked": connection_ids.len(),
        "startup_physical_forwards": launch.startup["next_forward_id"],
        "serving_physical_forwards":
            next - uint(&launch.startup, "next_forward_id")?,
        "physical_forwards_including_startup": next,
        "manifest_bytes": manifest.len(),
        "manifest_sha256": sha(manifest),
        "unmatched_manifest_bytes": 0,
        "unmatched_physical_records": 0,
        "historical_coalescing_observation": observation,
        "historical_poisoned_observation": old.observation,
        "drift_logical_delta": 1,
        "drift_physical_manifest_delta_bytes": 0,
        "request_id_order_used_as_fifo_insertion_proof": false
    }))
}

pub(crate) fn reconcile(
    continuation: &IndependentArtifactsContinuation,
) -> Result<Value> {
    ensure!(
        super::saved_independent::reconcile(&continuation.poisoned_launch)?
            == continuation.independent_reconciliation,
        "preceding independent reconciliation changed"
    );
    let state = retained(continuation);
    let old = &state.recovery.ready.poison_exit.old;
    let loaded = &old.core.loaded;
    let out = &old.core.out;
    loaded.recheck()?;
    ensure!(
        old.core.ledger.pending() == vec![json!({
            "acceptance_phase": policy::LIFECYCLE,
            "case_id": "final-artifact-and-manifest-reconciliation"
        })],
        "scheduler saved reconciliation at wrong ledger boundary"
    );
    let ledger = bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)?;
    let transcript = bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)?;
    ensure!(
        ledger == state.ledger_prefix && transcript == state.orchestration_prefix,
        "scheduler retained ledger/transcript changed"
    );
    let records = core::jsonl(&ledger)?;
    let frames = lifecycle_prefix::verify_transcript(&transcript, &loaded.policy)?;
    let mut inventory = BTreeMap::new();
    let binding = launch_artifacts(old, &frames, &mut inventory)?;
    consumed_binding(old, &records, &transcript, &binding, &mut inventory)?;
    let summary = coalescing(old, &records, &mut inventory)?;

    // All 14 positions now have semantic checks:
    // 0..3 initial; 3..6 scheduler; 6..9 restart;
    // 9..12 recovery launch; 12..14 final drain.
    ensure!(
        frames.len() == 14,
        "unaccounted orchestration frame"
    );

    ensure!(
        super::saved_independent::reconcile(&continuation.poisoned_launch)?
            == continuation.independent_reconciliation,
        "preceding artifacts changed during scheduler reconciliation"
    );
    loaded.recheck()?;
    for (relative, digest) in &inventory {
        ensure!(
            file_sha(&output_path(out, relative)?)? == *digest,
            "scheduler artifact changed during reconciliation"
        );
    }
    ensure!(
        regular_external_bytes(&old.launch.manifest, MANIFEST_LIMIT)?
            == old.manifest_prefix
            && sha(&regular_external_bytes(&old.launch.attestation, JSON_LIMIT)?)
                == text(&binding, "attestation_sha256")?
            && bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)? == ledger
            && bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)? == transcript,
        "scheduler closing byte stability"
    );

    Ok(json!({
        "schema": "rustv1.accept_m4_scheduler_artifact_reconciliation.v1",
        "boundary": SCHEDULER_BOUNDARY,
        "policy_sha256": loaded.policy_sha256,
        "preceding_ledger_sha256": sha(&ledger),
        "orchestration_sha256": sha(&transcript),
        "preceding_independent_reconciliation_sha256":
            sha(&serde_json::to_vec(&continuation.independent_reconciliation)?),
        "scheduler_launch_binding_sha256": old.launch.binding_sha256,
        "scheduler_attestation_sha256": binding["attestation_sha256"],
        "scheduler_startup_manifest_sha256": sha(&old.launch.initial_manifest),
        "coalescing": summary,
        "scheduler_launch_and_coalescing_reconciliation_complete": true,
        "complete_scheduler_manifest_reconciled_here": true,
        "authenticated_transcript_frames_checked": 14,
        "scheduler_launch_semantic_positions": [3, 4, 5],
        "whole_transcript_semantics_reconciled": true,
        "semantic_frame_coverage": {
            "initial": [0, 1, 2],
            "scheduler_launch": [3, 4, 5],
            "poison_restart": [6, 7, 8],
            "recovery_launch": [9, 10, 11],
            "final_drain": [12, 13]
        },
        "family53_driver_binary_producer_field_present": false,
        "process_creation_after_join_independently_measured": false,
        "historical_observations_relabelled": false,
        "verified_output_sha256": inventory,
        "complete_output_directory_coverage": false,
        "post_exit_control_calls": 0,
        "process_observations": 0,
        "owner_joins_performed": 0,
        "child_reaps_performed": 0,
        "ledger_rows_appended": 0,
        "final_artifact_reconciliation_complete": false,
        "scheduler_phase_complete": false,
        "lifecycle_phase_complete": false,
        "m4_accepted": false
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_handshake_requires_exact_remaining_positions() {
        // Semantic format fixtures only; production caller verifies MACs.
        let request = json!({"kind": "request"});
        let accepted = json!({"kind": "accepted"});
        let ready = json!({"kind": "ready"});
        let mut frames = vec![Value::Null; 14];
        frames[3] = frame("driver_to_orchestrator", &request);
        frames[4] = frame("orchestrator_to_driver", &accepted);
        frames[5] = frame("orchestrator_to_driver", &ready);
        assert!(scheduler_frames(&frames, &request, &accepted, &ready).is_ok());
        let mut reordered = frames.clone();
        reordered.swap(3, 4);
        assert!(scheduler_frames(&reordered, &request, &accepted, &ready).is_err());
        assert!(scheduler_frames(&frames[..13], &request, &accepted, &ready).is_err());
        frames.push(Value::Null);
        assert!(scheduler_frames(&frames, &request, &accepted, &ready).is_err());
    }

    #[test]
    fn scheduler_authorization_binds_join_absence_and_request_sequence() {
        let absence = json!({"historical_absence": true});
        let policy = sha(b"policy");
        let join = sha(b"join");
        let nonce = sha(b"nonce");
        let binding = json!({
            "authenticated_launch_request": {
                "kind": "orch_request_launch",
                "seq": 1,
                "fields": {
                    "launch_request_id": nonce,
                    "reason": "after_independent_join",
                    "previous_instance": "initial",
                    "previous_pid": 42,
                    "policy_sha256": policy,
                    "owner_closure_sha256": join,
                    "old_absence_sha256": sha(&serde_json::to_vec(&absence).unwrap())
                }
            },
            "authenticated_launch_accepted": {
                "kind": "orch_launch_accepted",
                "fields": {
                    "launch_request_id": nonce,
                    "request_seq": 1,
                    "previous_instance": "initial"
                }
            },
            "authenticated_ready": {
                "kind": "orch_ready",
                "fields": {
                    "launch_request_id": nonce,
                    "instance": "scheduler",
                    "pid": 43,
                    "start_utc": "historical",
                    "manifest": "/fixture/manifest",
                    "attestation": "/fixture/attestation"
                }
            }
        });
        let verify = |v: &Value| authorization(
            v, "initial", 42, &policy, &join, &absence,
        );
        assert!(verify(&binding).is_ok());
        for (section, field, replacement) in [
            ("authenticated_launch_request", "owner_closure_sha256", json!(sha(b"other"))),
            ("authenticated_launch_request", "old_absence_sha256", json!(sha(b"other"))),
            ("authenticated_launch_accepted", "request_seq", json!(2)),
            ("authenticated_ready", "launch_request_id", json!(sha(b"other"))),
        ] {
            let mut changed = binding.clone();
            changed[section]["fields"][field] = replacement;
            assert!(verify(&changed).is_err());
        }
        let mut extra = binding;
        extra["authenticated_launch_request"]["fields"]["owner_join_invented"] = json!(true);
        assert!(verify(&extra).is_err());
    }

    #[test]
    fn scheduler_intervals_reject_gap_overlap_empty_and_mutation() {
        let bytes = b"startup\nbegin\nend\n";
        let start = b"startup\n".len();
        let proof = json!({
            "path": "wave-0.executor_interval.jsonl",
            "start_byte": start,
            "end_byte": bytes.len(),
            "prefix_before_sha256": sha(&bytes[..start]),
            "prefix_after_sha256": sha(bytes),
            "interval_sha256": sha(&bytes[start..])
        });
        let name = "wave-0.executor_interval.jsonl";
        assert!(interval(bytes, &proof, start, name).is_ok());
        assert!(interval(bytes, &proof, start - 1, name).is_err());
        assert!(interval(bytes, &proof, start + 1, name).is_err());
        assert!(interval(bytes, &proof, start, "other").is_err());
        let mut empty = proof.clone();
        empty["end_byte"] = json!(start);
        assert!(interval(bytes, &empty, start, name).is_err());
        let mut oversized = proof.clone();
        oversized["end_byte"] = json!(bytes.len() + 1);
        assert!(interval(bytes, &oversized, start, name).is_err());
        let mut changed = bytes.to_vec();
        changed[start] = b'X';
        assert!(interval(&changed, &proof, start, name).is_err());
    }

    #[test]
    fn scheduler_waves_preserve_failed_attempts_and_stop_at_first_witness() {
        let no = json!({"coalescing_witnesses": []});
        let yes = json!({"coalescing_witnesses": [{"physical": "fixture"}]});
        assert!(wave_termination(&[yes.clone()]).is_ok());
        assert!(wave_termination(&[no.clone(), yes.clone()]).is_ok());
        assert!(wave_termination(&[]).is_err());
        assert!(wave_termination(&[no.clone()]).is_err());
        assert!(wave_termination(&[yes.clone(), no.clone()]).is_err());
        assert!(wave_termination(&[yes.clone(), yes.clone()]).is_err());
        let mut too_many = vec![no; MAX_WAVES];
        too_many.push(yes);
        assert!(wave_termination(&too_many).is_err());
    }

    #[test]
    fn scheduler_prefix_retains_original_transcript_bytes() {
        let bytes = b"{ \"a\": 1 }\n{\"b\":2}\n";
        assert_eq!(prefix(bytes, 1).unwrap(), b"{ \"a\": 1 }\n");
        assert_eq!(prefix(bytes, 2).unwrap(), bytes);
        assert!(prefix(bytes, 3).is_err());
        assert!(prefix(b"{}", 1).is_err());
    }
}
