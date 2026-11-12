use super::*;

const CASE: &str = "final-authenticated-drain-and-exit";
const FOLLOWING: &str = "final-artifact-and-manifest-reconciliation";

/// Contains credentials indirectly. Do not derive Debug or Serialize.
pub struct FinalDrainContinuation {
    pub runtime: tokio::runtime::Runtime,
    pub state: FinalDrainState,
}

/// Neither preceding observation is rewritten to describe process exit.
///
/// recovery.observation remains the historical ready recovery boundary.
/// recovery.ready.poison_exit.old remains the historical poisoned launch.
/// The quiescence and exit evidence below describe subsequent events.
pub struct FinalDrainState {
    pub recovery: ExactRecoveryState,
    pub quiesced_observation: Value,
    pub drain_request: Value,
    pub authenticated_exit: Value,
    pub process_absence: Value,
    pub drain_evidence_sha256: String,
    pub ledger_prefix: Vec<u8>,
    pub orchestration_prefix: Vec<u8>,
}

fn quiescence(before: &Value, after: &Value) -> Result<()> {
    closed(
        before,
        &[
            "instance",
            "pid",
            "lifecycle",
            "worker_alive",
            "ready",
            "forward_calls",
            "startup_output_sha256",
        ],
    )?;
    ensure!(
        before["lifecycle"] == "ready" && before["worker_alive"] == true && before["ready"] == true,
        "final quiescence requires the recovered ready worker"
    );
    let mut expected = before.clone();
    expected["ready"] = json!(false);
    ensure!(
        *after == expected,
        "quiescence changed identity, lifecycle, liveness or execution cursor"
    );
    Ok(())
}

fn validate_drain_fields(fields: &Value) -> Result<()> {
    closed(
        fields,
        &[
            "drain_request_id",
            "reason",
            "instance",
            "pid",
            "process_start_ticks",
            "boot_id",
            "running_binary_sha256",
            "policy_sha256",
            "launch_binding_sha256",
            "recovery_evidence_sha256",
            "manifest_sha256",
            "quiesced_observation_sha256",
            "preceding_ledger_sha256",
            "replacement_launch",
        ],
    )?;
    for name in [
        "drain_request_id",
        "running_binary_sha256",
        "policy_sha256",
        "launch_binding_sha256",
        "recovery_evidence_sha256",
        "manifest_sha256",
        "quiesced_observation_sha256",
        "preceding_ledger_sha256",
    ] {
        ensure!(
            crate::contract::digest_valid(text(fields, name)?),
            "invalid final-drain nonce/digest: {name}"
        );
    }
    ensure!(
        fields["reason"] == "after_exact_recovery"
            && label_valid(text(fields, "instance")?)
            && uint(fields, "pid")? > 0
            && uint(fields, "pid")? <= u64::from(u32::MAX)
            && uint(fields, "process_start_ticks")? > 0
            && !text(fields, "boot_id")?.is_empty()
            && text(fields, "boot_id")?.len() <= 80
            && fields["replacement_launch"] == "forbidden",
        "invalid final-drain identity or launch prohibition"
    );
    Ok(())
}

fn check_final_exit(request: &Value, exit: &Value) -> Result<()> {
    ensure!(
        request["kind"] == "orch_request_final_drain" && exit["kind"] == "orch_final_exit",
        "unexpected final-drain exchange kind"
    );
    validate_drain_fields(&request["fields"])?;
    uint(request, "seq")?;
    closed(
        &exit["fields"],
        &[
            "drain_request_id",
            "request_seq",
            "instance",
            "pid",
            "termination_signal",
            "exit_status",
            "exit_signal",
            "process_reaped",
            "replacement_launched",
        ],
    )?;
    let fields = &exit["fields"];
    ensure!(
        fields["drain_request_id"] == request["fields"]["drain_request_id"]
            && fields["request_seq"] == request["seq"]
            && fields["instance"] == request["fields"]["instance"]
            && fields["pid"] == request["fields"]["pid"]
            && fields["termination_signal"] == "SIGTERM"
            && fields["exit_status"] == 0
            && fields["exit_signal"].is_null()
            && fields["process_reaped"] == true
            && fields["replacement_launched"] == false,
        "final exit identity/status/reap/launch binding"
    );
    Ok(())
}

async fn drain_exchange(
    orchestration: &mut core::Orchestration,
    fields: Value,
) -> Result<(Value, Value)> {
    validate_drain_fields(&fields)?;
    let request = orchestration
        .send("orch_request_final_drain", fields)
        .await?;
    let exit = orchestration
        .receive_with_budget(Duration::from_secs(120))
        .await?;
    check_final_exit(&request, &exit)?;
    orchestration.sync()?;
    Ok((request, exit))
}

/// Verify a saved interval against its preceding commitment and exact bytes.
/// The caller also enforces contiguous coverage of the complete manifest.
fn saved_interval<'a>(
    root: &Path,
    proof: &Value,
    manifest: &'a [u8],
    expected_start: usize,
) -> Result<&'a [u8]> {
    let interval = &proof["manifest_interval"];
    let start = usize::try_from(uint(interval, "start_byte")?)?;
    let end = usize::try_from(uint(interval, "end_byte")?)?;
    ensure!(
        start == expected_start && start < end && end <= manifest.len(),
        "recovery interval gap, overlap or invalid bounds"
    );
    let bytes = bounded(&under(root, text(interval, "path")?)?, MANIFEST_LIMIT)?;
    ensure!(
        bytes == manifest[start..end]
            && sha(&bytes) == text(interval, "interval_sha256")?
            && sha(&manifest[..start]) == text(interval, "prefix_before_sha256")?
            && sha(&manifest[..end]) == text(interval, "prefix_after_sha256")?,
        "saved recovery interval differs from preceding commitment"
    );
    Ok(&manifest[start..end])
}

/// Consume the recovery boundary without calling consume_ready(), whose
/// pending-case guard correctly belongs to the earlier recovery position.
///
/// This checks the immediate predecessor and its saved HTTP/manifest interval.
/// It is not a substitute for the pending cross-phase final reconciliation.
fn consume_recovery(state: &ExactRecoveryState) -> Result<Value> {
    let retained = &state.ready;
    let old = &retained.poison_exit.old;
    let launch = &retained.replacement;

    pending(old, CASE)?;
    old.core.loaded.recheck()?;
    ensure!(
        ledger_bytes(old)? == state.ledger_prefix,
        "retained exact-recovery ledger changed"
    );
    let records = check_ledger_prefix(old, &state.ledger_prefix)?;
    let record = one_record_extension(
        &retained.ledger_prefix,
        &state.ledger_prefix,
        policy::LIFECYCLE,
        RECOVERY,
        &state.recovery_evidence_sha256,
    )?;
    ensure!(
        record["details"]["instance"] == launch.instance
            && record["details"]["pid"] == launch.pid
            && record["details"]["logical_requests"] == 44
            && record["details"]["physical_forwards"] == 44
            && record["details"]["unmatched_physical_records"] == 0,
        "recovery ledger identity/count binding"
    );

    let ready_record = one_record_extension(
        &retained.poison_exit.ledger_prefix,
        &retained.ledger_prefix,
        policy::LIFECYCLE,
        super::super::CASE,
        &retained.reattestation_evidence_sha256,
    )?;
    ensure!(
        ready_record["details"]["instance"] == launch.instance
            && ready_record["details"]["pid"] == launch.pid,
        "reattestation ledger identity changed"
    );

    let saved: Value = serde_json::from_slice(&read_bound(
        &old.core.out,
        &format!("scheduler/{RECOVERY}/evidence.json"),
        &state.recovery_evidence_sha256,
    )?)?;
    ensure!(
        saved["schema"] == "rustv1.accept_m4_exact_recovery.v1"
            && saved["case_id"] == RECOVERY
            && saved["instance"] == launch.instance
            && saved["pid"] == launch.pid
            && saved["policy_sha256"] == old.core.loaded.policy_sha256
            && saved["driver_binary_sha256"] == old.core.loaded.artifacts["accept_m4"].sha256
            && saved["preceding_ledger_sha256"] == sha(&retained.ledger_prefix)
            && saved["reattestation_evidence_sha256"] == retained.reattestation_evidence_sha256
            && saved["launch_binding_sha256"] == launch.binding_sha256
            && saved["orchestration_prefix_sha256"] == sha(&state.orchestration_prefix)
            && state.orchestration_prefix == retained.orchestration_prefix
            && saved["initial_observation"] == launch.initial_observation
            && saved["startup_manifest_sha256"] == sha(&launch.initial_manifest)
            && saved["final_observation"] == state.observation
            && saved["final_manifest_sha256"] == sha(&state.manifest_prefix)
            && saved["final_manifest_bytes"] == state.manifest_prefix.len()
            && saved["next_forward_id"] == state.next_forward
            && saved["replacement_request_ids"] == json!(state.seen_request_ids)
            && saved["replacement_logical_requests"] == 44
            && saved["reconciled_recovery_physical_forwards"] == 44
            && saved["fixture_indices"] == json!((0usize..22).collect::<Vec<_>>())
            && saved["encodings_per_fixture"] == json!(["float", "base64"])
            && saved["serialized_physical_b1_exact_f32_bits"] == true
            && saved["unmatched_physical_records"] == 0
            && saved["replacement_left_ready"] == true
            && saved["final_drain_executed"] == false
            && saved["lifecycle_phase_complete"] == false
            && saved["m4_accepted"] == false,
        "recovery evidence/cursor semantics changed"
    );
    ready_observation(&state.observation, launch)?;
    logical_delta(&launch.initial_observation, &state.observation, 44)?;

    let (old_exit, _) = preceding_case(old, &records, EXIT_CASE)?;
    historical_cursor(
        &old_exit,
        &old.observation,
        &old.manifest_prefix,
        old.next_forward,
        &old.seen_request_ids,
    )?;
    ensure!(
        saved["old_historical_observation"] == old.observation
            && saved["old_manifest_unchanged_sha256"] == sha(&old.manifest_prefix)
            && saved["old_next_forward_id_unchanged"] == old.next_forward
            && saved["old_request_ids_unchanged"] == json!(old.seen_request_ids)
            && saved["old_exit_evidence_sha256"] == retained.poison_exit.exit_evidence_sha256
            && saved["old_service_owner_join_claimed"] == false
            && saved["old_quarantine_cleared_in_process"] == false
            && old_exit["independent_process_absence"] == retained.poison_exit.old_absence
            && bounded(&old.launch.manifest, MANIFEST_LIMIT)? == old.manifest_prefix,
        "historical poisoned launch changed or was relabelled"
    );
    read_bound(
        &old.core.out,
        &format!("scheduler/{EXIT_CASE}/evidence.json"),
        &retained.poison_exit.exit_evidence_sha256,
    )?;

    let ready_path = format!("scheduler/{}/evidence.json", super::super::CASE);
    let ready: Value = serde_json::from_slice(&read_bound(
        &old.core.out,
        &ready_path,
        &retained.reattestation_evidence_sha256,
    )?)?;
    ensure!(
        saved["reattestation_evidence_path"] == ready_path
            && ready["instance"] == launch.instance
            && ready["pid"] == launch.pid
            && ready["initial_observation"] == launch.initial_observation
            && ready["startup_manifest_sha256"] == sha(&launch.initial_manifest)
            && ready["launch_binding_sha256"] == launch.binding_sha256
            && ready["preceding_ledger_sha256"] == sha(&retained.poison_exit.ledger_prefix),
        "recovery predecessor binding changed"
    );
    let binding: Value = serde_json::from_slice(&read_bound(
        &old.core.out,
        &format!("scheduler/{}/launch_binding.json", super::super::CASE),
        &launch.binding_sha256,
    )?)?;
    ensure!(
        ready["launch_binding"] == binding
            && binding["process_start_ticks"] == launch.process.start_ticks
            && binding["boot_id"] == launch.process.boot_id
            && binding["running_binary_sha256"] == launch.process.executable_sha256
            && launch.process.pid == launch.pid
            && launch.process.executable_sha256 == old.core.loaded.artifacts["serve_m3"].sha256
            && file_sha(&launch.attestation)? == text(&binding, "attestation_sha256")?
            && core::check_startup(
                &launch.initial_manifest,
                &old.core.loaded,
                text(&binding, "attestation_sha256")?,
                true,
            )? == launch.startup,
        "retained recovery launch/startup changed"
    );

    let frames = verify_transcript(&state.orchestration_prefix, &old.core.loaded.policy)?;
    ensure!(
        !frames.iter().any(|frame| {
            matches!(
                frame["envelope"]["kind"].as_str(),
                Some("orch_request_final_drain" | "orch_final_exit")
            )
        }),
        "final drain already present in preceding transcript"
    );

    let root = old.directory.join(RECOVERY);
    let requests = array(&saved, "requests")?;
    ensure!(requests.len() == 44, "saved recovery request coverage");
    manifest_suffix(&launch.initial_manifest, &state.manifest_prefix)?;
    let mut cursor = launch.initial_manifest.len();
    let mut next = uint(&launch.startup, "next_forward_id")?;
    let mut observation = launch.initial_observation.clone();
    let mut ids = BTreeSet::new();
    let mut last_id = None;
    let mut connections = Vec::new();
    let bundle = &old.core.loaded.bundle;
    ensure!(bundle.fixtures.len() == 22, "frozen fixture coverage");

    for (position, proof) in requests.iter().enumerate() {
        let fixture = position / 2;
        let encoding = if position % 2 == 0 { "float" } else { "base64" };
        let case = format!("sched58-recovery-{fixture}-{encoding}");
        let request = serde_json::to_vec(&json!({
            "model": MODEL_ID,
            "input": [bundle.fixtures[fixture].text],
            "encoding_format": encoding
        }))?;
        ensure!(
            proof["case_id"] == case
                && proof["fixture_index"] == fixture
                && proof["fixture_id"] == bundle.fixtures[fixture].id
                && proof["encoding"] == encoding
                && proof["request_path"] == format!("{case}.request.bin")
                && proof["request_sha256"] == sha(&request)
                && bounded(&root.join(format!("{case}.request.bin")), JSON_LIMIT)? == request
                && value(&root.join(format!("{case}.evidence.json")))? == *proof
                && proof["observation_before"] == observation
                && proof["logical_backend_request_delta"] == 1
                && proof["unmatched_physical_records"] == 0
                && proof["exact_f32_bits"] == true
                && proof["prior_float_output_checked_exactly"] == (position % 2 == 1),
            "saved recovery request/proof binding"
        );

        let response = &proof["response"];
        check_saved_reply(&root, response)?;
        let id = uint(proof, "request_id")?;
        ensure!(
            id > 0
                && ids.insert(id)
                && last_id.is_none_or(|previous| id > previous)
                && response["case_id"] == case
                && response["request_id"] == id,
            "saved recovery request ID association"
        );
        let reply = http::Reply {
            status: u16::try_from(uint(response, "http_status")?)?,
            request_id: Some(id),
            headers: response["headers"].clone(),
            bytes: bounded(&under(&root, text(response, "response_path")?)?, JSON_LIMIT)?,
        };
        let (output, usage) = http::decode_success(&reply, bundle, &[fixture], encoding)?;
        let digest = exact_singleton(
            &output,
            &bundle.singleton[fixture],
            &bundle.fixtures[fixture].golden_sha256,
        )?;
        ensure!(
            proof["usage"] == usage
                && proof["output_f32le_sha256"] == digest
                && proof["frozen_singleton_sha256"] == digest,
            "saved recovery numerical commitment changed"
        );

        let suffix = saved_interval(&root, proof, &state.manifest_prefix, cursor)?;
        ensure!(
            core::jsonl(suffix)?.len() == 2,
            "recovery physical pair count"
        );
        let completed = BTreeMap::from([(
            id,
            CompletedRequest {
                case: case.clone(),
                fixtures: vec![fixture],
                output,
            },
        )]);
        let reconciled = reconcile_wave(suffix, next, &completed, bundle)?;
        ensure!(
            reconciled.forwards.len() == 1
                && reconciled.forwards[0]["B"] == 1
                && reconciled.witnesses.is_empty()
                && proof["physical_forwards"] == json!(reconciled.forwards),
            "saved recovery physical reconciliation changed"
        );
        logical_delta(&observation, &proof["observation_after"], 1)?;

        let connection = &reply.headers["_connection_owner"];
        ensure!(
            connection["instance"] == launch.instance
                && connection["owner_pid"] == std::process::id()
                && connection["case_id"] == case
                && connection["method"] == "POST"
                && connection["path"] == "/v1/embeddings"
                && connection["request_sha256"] == sha(&request),
            "saved recovery connection association"
        );
        connections.push(connection.clone());
        cursor = usize::try_from(uint(&proof["manifest_interval"], "end_byte")?)?;
        next = reconciled.next_forward;
        observation = proof["observation_after"].clone();
        last_id = Some(id);
    }

    check_connections(
        &launch.instance,
        &connections,
        array(&saved, "http_connections")?,
    )?;
    ensure!(
        cursor == state.manifest_prefix.len()
            && next == state.next_forward
            && ids == state.seen_request_ids
            && observation == state.observation
            && bounded(
                &root.join("executor_manifest_snapshot.jsonl"),
                MANIFEST_LIMIT
            )? == state.manifest_prefix
            && value(&root.join("final_observation.json"))? == state.observation
            && value(&root.join("http_connections.json"))? == saved["http_connections"],
        "recovery closing artifacts or complete interval coverage changed"
    );
    Ok(saved)
}

async fn execute(mut state: ExactRecoveryState) -> Result<FinalDrainState> {
    #[allow(clippy::type_complexity)]
    let work: Result<(Value, Value, Value, Value, String, Vec<u8>, Vec<u8>)> = async {
        let preceding = consume_recovery(&state)?;
        let old = &state.ready.poison_exit.old;
        let launch = &state.ready.replacement;
        let out = stage_directory(old, CASE)?;
        transcript_extension(old, &state.orchestration_prefix, &[])?;

        ensure!(
            old.launch.process.prove_absent().await? == state.ready.poison_exit.old_absence,
            "old poisoned process absence changed"
        );
        live_cursor(&state.ready, &state.observation, &state.manifest_prefix).await?;

        let connections = http::join_connections(&launch.instance).await?;
        check_connections(
            &launch.instance,
            array(&preceding, "http_connections")?,
            &connections,
        )?;
        save_json(&out.join("http_connections.json"), &json!(connections))?;

        let quiesced = old.core.control.call("quiesce", None).await?;
        quiescence(&state.observation, &quiesced)?;
        let quiesced_bytes = serde_json::to_vec(&quiesced)?;
        write_new(&out.join("quiesced_observation.json"), &quiesced_bytes)?;
        ensure!(
            old.core.control.call("observe", None).await? == quiesced
                && bounded(&launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
            "unreconciled work after final quiescence"
        );
        launch.process.check_live()?;

        // Validate preceding bytes again before granting termination authority.
        ensure!(
            consume_recovery(&state)? == preceding,
            "preceding recovery evidence changed before drain authorization"
        );
        let nonce = core::random_label()?;
        let frames = verify_transcript(&state.orchestration_prefix, &old.core.loaded.policy)?;
        ensure!(
            !frames.iter().any(|frame| {
                frame["envelope"]["fields"]
                    .as_object()
                    .is_some_and(|fields| fields.values().any(|value| value == &json!(nonce)))
            }),
            "final drain nonce collision"
        );

        let fields = json!({
            "drain_request_id": nonce,
            "reason": "after_exact_recovery",
            "instance": launch.instance,
            "pid": launch.pid,
            "process_start_ticks": launch.process.start_ticks,
            "boot_id": launch.process.boot_id,
            "running_binary_sha256": launch.process.executable_sha256,
            "policy_sha256": old.core.loaded.policy_sha256,
            "launch_binding_sha256": launch.binding_sha256,
            "recovery_evidence_sha256": state.recovery_evidence_sha256,
            "manifest_sha256": sha(&state.manifest_prefix),
            "quiesced_observation_sha256": sha(&quiesced_bytes),
            "preceding_ledger_sha256": sha(&state.ledger_prefix),
            "replacement_launch": "forbidden"
        });

        let (request, exit) =
            drain_exchange(&mut state.ready.poison_exit.old.core.orchestration, fields).await?;
        let old = &state.ready.poison_exit.old;
        let launch = &state.ready.replacement;
        save_json(&out.join("request.json"), &request)?;
        save_json(&out.join("authenticated_exit.json"), &exit)?;

        // No observe/check_live is attempted after termination authorization.
        let absence = launch.process.prove_absent().await?;
        save_json(&out.join("process_absence.json"), &absence)?;
        ensure!(
            bounded(&launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix
                && old.launch.process.prove_absent().await? == state.ready.poison_exit.old_absence,
            "executor activity during final exit or changed old absence"
        );
        write_new(
            &out.join("executor_manifest_snapshot.jsonl"),
            &state.manifest_prefix,
        )?;

        let transcript = transcript_extension(
            old,
            &state.orchestration_prefix,
            &[
                json!({
                    "direction": "driver_to_orchestrator",
                    "envelope": request
                }),
                json!({
                    "direction": "orchestrator_to_driver",
                    "envelope": exit
                }),
            ],
        )?;
        ensure!(
            consume_recovery(&state)? == preceding
                && bounded(&out.join("quiesced_observation.json"), JSON_LIMIT)? == quiesced_bytes
                && value(&out.join("request.json"))? == request
                && value(&out.join("authenticated_exit.json"))? == exit
                && value(&out.join("process_absence.json"))? == absence
                && value(&out.join("http_connections.json"))? == json!(connections)
                && bounded(
                    &out.join("executor_manifest_snapshot.jsonl"),
                    MANIFEST_LIMIT
                )? == state.manifest_prefix,
            "preceding or final-drain artifacts changed"
        );

        let evidence = json!({
            "schema": "rustv1.accept_m4_final_drain_exit.v1",
            "case_id": CASE,
            "instance": launch.instance,
            "pid": launch.pid,
            "policy_sha256": old.core.loaded.policy_sha256,
            "driver_binary_sha256": old.core.loaded.artifacts["accept_m4"].sha256,
            "preceding_ledger_sha256": sha(&state.ledger_prefix),
            "recovery_evidence_path": format!("scheduler/{RECOVERY}/evidence.json"),
            "recovery_evidence_sha256": state.recovery_evidence_sha256,
            "launch_binding_sha256": launch.binding_sha256,
            "recovery_observation_preserved": state.observation,
            "recovery_observation_is_historical": true,
            "quiesced_observation": quiesced,
            "quiesced_observation_sha256": sha(&quiesced_bytes),
            "request": request,
            "authenticated_exit": exit,
            "independent_process_absence": absence,
            "preceding_orchestration_sha256": sha(&state.orchestration_prefix),
            "orchestration_prefix_sha256": sha(&transcript),
            "manifest_unchanged_sha256": sha(&state.manifest_prefix),
            "manifest_bytes": state.manifest_prefix.len(),
            "executor_manifest_interval_bytes": 0,
            "executor_next_forward_id_unchanged": state.next_forward,
            "exposed_request_ids_unchanged": state.seen_request_ids,
            "http_connections": connections,
            "supervisor_reported_process_reaped": true,
            "supervisor_reap_independently_awaited_by_driver": false,
            "process_absence_independently_measured_by_driver": true,
            "replacement_service_owner_join_independently_observed": false,
            "replacement_post_exit_observation_fabricated": false,
            "old_historical_observation": old.observation,
            "old_manifest_unchanged_sha256": sha(&old.manifest_prefix),
            "old_next_forward_id_unchanged": old.next_forward,
            "old_request_ids_unchanged": old.seen_request_ids,
            "old_exit_evidence_sha256": state.ready.poison_exit.exit_evidence_sha256,
            "old_service_owner_join_claimed": false,
            "old_quarantine_cleared_in_process": false,
            "replacement_launch_authorized": false,
            "final_artifact_reconciliation_executed": false,
            "lifecycle_phase_complete": false,
            "m4_accepted": false
        });
        let digest = save_json(&out.join("evidence.json"), &evidence)?;
        old.core.loaded.recheck()?;
        transcript_extension(old, &transcript, &[])?;
        ensure!(
            ledger_bytes(old)? == state.ledger_prefix
                && bounded(&launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
            "final-drain closing cursor changed"
        );

        state.ready.poison_exit.old.core.ledger.append(
            policy::LIFECYCLE,
            CASE,
            true,
            json!({
                "evidence_path": format!("scheduler/{CASE}/evidence.json"),
                "evidence_sha256": digest,
                "instance": state.ready.replacement.instance,
                "pid": state.ready.replacement.pid,
                "exit_status": 0,
                "process_absence_independently_measured": true
            }),
            true,
        )?;
        let ledger = ledger_bytes(&state.ready.poison_exit.old)?;
        one_record_extension(
            &state.ledger_prefix,
            &ledger,
            policy::LIFECYCLE,
            CASE,
            &digest,
        )?;
        check_ledger_prefix(&state.ready.poison_exit.old, &ledger)?;
        pending(&state.ready.poison_exit.old, FOLLOWING)?;
        Ok((quiesced, request, exit, absence, digest, ledger, transcript))
    }
    .await;

    match work {
        Ok((quiesced, request, exit, absence, digest, ledger, transcript)) => Ok(FinalDrainState {
            recovery: state,
            quiesced_observation: quiesced,
            drain_request: request,
            authenticated_exit: exit,
            process_absence: absence,
            drain_evidence_sha256: digest,
            ledger_prefix: ledger,
            orchestration_prefix: transcript,
        }),
        Err(error) => {
            fail(&mut state.ready.poison_exit.old, CASE, &error);
            Err(error)
        }
    }
}

pub(crate) fn run(continuation: ExactRecoveryContinuation) -> Result<FinalDrainContinuation> {
    let ExactRecoveryContinuation { runtime, state } = continuation;
    let directory = state.ready.poison_exit.old.directory.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.block_on(execute(state))
    }));
    match result {
        Ok(result) => Ok(FinalDrainContinuation {
            runtime,
            state: result?,
        }),
        Err(payload) => {
            let _ = save_json(
                &directory.join("final_drain_panic.json"),
                &json!({
                    "panic": true,
                    "completion_claimed": false,
                    "retained_evidence_directory": directory
                }),
            );
            std::panic::resume_unwind(payload)
        }
    }
}

#[cfg(test)]
#[path = "accept_m4_lifecycle_final_drain_tests.rs"]
mod tests;

#[cfg(feature = "scheduler-test-seam")]
pub(crate) mod saved_lineage {
    include!("accept_m4_saved_recovery_lineage.rs");
}
