// Included inside:
// acceptance_scheduler::lifecycle_prefix::saved_owners::saved_poison
//
// Offline reconciliation of the historical poison lifecycle artifacts.
//
// No control/HTTP requests, process observations, owner joins, reaps,
// orchestration exchanges, ledger appends, or completion markers.
//
// Scheduler launch/startup/coalescing reconciliation remains a separate
// prerequisite. This module closes the five poison lifecycle subtrees.

use super::*;

const POISON_BOUNDARY: &str = "POISONED-LAUNCH-ARTIFACTS-RECONCILED";

const POISON_CASES: [&str; 5] = [
    "family-58-live-policy-drift-poison",
    "family-58-terminal-liveness-readiness-refusal",
    "family-58-restore-does-not-recover",
    "family-58-authenticated-supervised-restart",
    "family-58-old-process-exit",
];

#[derive(Clone, Copy)]
enum ResponseKind {
    Live,
    Poison,
    Drift,
}

fn root_for(case: &str) -> String {
    format!("scheduler/{case}")
}

fn case_path(case: &str) -> String {
    format!("{}/evidence.json", root_for(case))
}

/// Load only the expected successful lifecycle row and original commitment.
fn load_case(
    files: &mut Files<'_>,
    records: &[Value],
    case: &str,
    old: &SchedulerState,
) -> Result<(usize, Value, String)> {
    let matches: Vec<_> = records.iter().enumerate()
        .filter(|(_, row)| {
            row["acceptance_phase"] == policy::LIFECYCLE
                && row["case_id"] == case
        })
        .collect();
    ensure!(matches.len() == 1, "missing/duplicate poison lifecycle row");

    let (index, row) = matches[0];
    let relative = case_path(case);
    let digest = text(&row["details"], "evidence_sha256")?.to_owned();
    ensure!(
        row["status"] == "executed"
            && row["pass"] == true
            && row["details"] == json!({
                "evidence_path": relative,
                "evidence_sha256": digest,
                "instance": old.launch.instance,
                "pid": old.launch.pid
            }),
        "poison lifecycle ledger identity/path/schema"
    );
    let bytes = files.committed(&relative, &digest)?;
    let evidence: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        evidence["case_id"] == case
            && evidence["instance"] == old.launch.instance
            && evidence["pid"] == old.launch.pid
            && evidence["policy_sha256"] == old.core.loaded.policy_sha256
            && evidence["driver_binary_sha256"]
                == old.core.loaded.artifacts["accept_m4"].sha256,
        "poison lifecycle evidence identity/policy/driver"
    );
    Ok((index, evidence, digest))
}

/// Exact producer-format comparison after independently reconstructing fields.
fn parent(
    files: &mut Files<'_>,
    case: &str,
    saved: &Value,
    mut expected: Value,
    old: &SchedulerState,
) -> Result<()> {
    expected["policy_sha256"] = json!(old.core.loaded.policy_sha256);
    expected["driver_binary_sha256"] =
        json!(old.core.loaded.artifacts["accept_m4"].sha256);
    ensure!(*saved == expected, "complete historical poison evidence changed");
    files.pretty(&case_path(case), &expected)
}

/// Validate the actual saved connection-join record.
///
/// A completed task may contain a transport error: the original client
/// accepted that after receiving the complete HTTP response. Preserve that
/// distinction rather than retroactively requiring connection_result.ok.
fn connection_record(
    value: &Value,
    instance: &str,
    case: &str,
    method: &str,
    path: &str,
    request: &[u8],
    endpoint: SocketAddr,
) -> Result<()> {
    closed(value, &[
        "instance",
        "case_id",
        "method",
        "path",
        "local_address",
        "peer_address",
        "request_sha256",
        "connection_id",
        "owner_pid",
        "tokio_task_id",
        "join_outcome",
        "abort_requested",
        "connection_result",
        "join_error",
        "join_awaited",
    ])?;
    let local: SocketAddr = text(value, "local_address")?.parse()?;
    let peer: SocketAddr = text(value, "peer_address")?.parse()?;
    ensure!(
        value["instance"] == instance
            && value["case_id"] == case
            && value["method"] == method
            && value["path"] == path
            && value["request_sha256"] == sha(request)
            && value["owner_pid"] == std::process::id()
            && value["join_awaited"] == true
            && value["abort_requested"].is_boolean()
            && !text(value, "tokio_task_id")?.is_empty()
            && local.ip().is_loopback()
            && local.port() != 0
            && peer == endpoint,
        "poison HTTP connection identity/request/endpoint"
    );
    uint(value, "connection_id")?;

    match text(value, "join_outcome")? {
        "completed" => {
            closed(&value["connection_result"], &["ok", "error"])?;
            let result = &value["connection_result"];
            ensure!(
                value["join_error"].is_null()
                    && ((result["ok"] == true && result["error"].is_null())
                        || (result["ok"] == false && result["error"].is_string())),
                "completed connection outcome schema"
            );
        }
        "cancelled-after-abort-and-awaited" => {
            ensure!(
                value["abort_requested"] == true
                    && value["connection_result"].is_null()
                    && !text(value, "join_error")?.is_empty(),
                "cancelled connection lacks abort-and-await evidence"
            );
        }
        _ => anyhow::bail!("unjoined or panicked poison HTTP connection"),
    }
    Ok(())
}

/// Reconcile a digest-bound saved response and its exact sidecar.
///
/// This function performs no HTTP request and does not inspect the live
/// connection registry.
#[allow(clippy::too_many_arguments)]
fn saved_response(
    files: &mut Files<'_>,
    root: &str,
    response: &Value,
    instance: &str,
    endpoint: SocketAddr,
    case: &str,
    method: &str,
    path: &str,
    request: &[u8],
    request_id: Option<u64>,
    kind: ResponseKind,
) -> Result<Value> {
    let status = match kind {
        ResponseKind::Live => 200u16,
        ResponseKind::Poison => 503,
        ResponseKind::Drift => 500,
    };
    if let Some(id) = request_id {
        ensure!(id > 0, "zero poison HTTP request ID");
    }

    closed(response, &[
        "case_id",
        "http_status",
        "request_id",
        "headers",
        "response_path",
        "response_sha256",
    ])?;
    ensure!(
        response["case_id"] == case
            && response["http_status"] == status
            && response["request_id"] == json!(request_id)
            && response["response_path"] == format!("{case}.response.bin"),
        "poison response identity/status/path"
    );

    let headers = &response["headers"];
    closed(headers, &[
        "x-rv1-instance-id",
        "x-acceptance-case-id",
        "x-rv1-request-id",
        "content-type",
        "_connection_owner",
    ])?;
    ensure!(
        headers["x-rv1-instance-id"] == instance
            && headers["x-acceptance-case-id"] == case
            && headers["x-rv1-request-id"] == json!(request_id),
        "poison response correlation"
    );
    if !matches!(kind, ResponseKind::Live) {
        ensure!(
            text(headers, "content-type")?.split(';').next()
                .unwrap_or("").trim().eq_ignore_ascii_case("application/json"),
            "poison response media"
        );
    } else {
        ensure!(
            headers["content-type"].is_null() || headers["content-type"].is_string(),
            "livez saved media type"
        );
    }

    let connection = &headers["_connection_owner"];
    connection_record(
        connection, instance, case, method, path, request, endpoint,
    )?;

    let bytes = files.committed(
        &format!("{root}/{case}.response.bin"),
        text(response, "response_sha256")?,
    )?;
    let reply = http::Reply {
        status,
        request_id,
        headers: headers.clone(),
        bytes,
    };
    match kind {
        ResponseKind::Live => {
            ensure!(
                reply.bytes.is_empty() && request_id.is_none(),
                "historical livez was not empty 200 without request ID"
            );
        }
        ResponseKind::Poison => {
            ensure!(request_id.is_none(), "terminal refusal exposed an ID");
            production_error(&reply, "executor_poisoned")?;
        }
        ResponseKind::Drift => {
            ensure!(request_id.is_some(), "admitted drift trigger lost its ID");
            production_error(&reply, "inference_failed")?;
        }
    }
    files.pretty(&format!("{root}/{case}.response.json"), response)?;
    Ok(connection.clone())
}

fn historical_transition(
    before: &Value,
    after: &Value,
    before_ids: &BTreeSet<u64>,
    after_ids: &BTreeSet<u64>,
    trigger_id: u64,
) -> Result<()> {
    ensure!(
        trigger_id > 0
            && before["lifecycle"] == "ready"
            && before["worker_alive"] == true
            && before["ready"] == true
            && !before_ids.contains(&trigger_id)
            && *after == poisoned(before)?,
        "historical drift transition changed"
    );
    let mut expected_ids = before_ids.clone();
    ensure!(expected_ids.insert(trigger_id), "drift request ID collision");
    ensure!(
        *after_ids == expected_ids,
        "historical drift request set changed"
    );
    Ok(())
}

pub(crate) fn reconcile(
    continuation: &crate::acceptance_scheduler::DestructiveOwnersArtifactsContinuation,
) -> Result<Value> {
    ensure!(
        super::reconcile(&continuation.recovery_http)?
            == continuation.destructive_owners_reconciliation,
        "preceding destructive/owner reconciliation changed"
    );

    let state = &continuation.recovery_http.recovery_lineage.initial_artifacts
        .imported_artifacts.local_artifacts.final_drain.state;
    let poison = &state.recovery.ready.poison_exit;
    let old = &poison.old;
    let loaded = &old.core.loaded;
    let out = &old.core.out;
    loaded.recheck()?;
    ensure!(
        old.core.ledger.pending() == vec![json!({
            "acceptance_phase": policy::LIFECYCLE,
            "case_id": "final-artifact-and-manifest-reconciliation"
        })],
        "poison reconciliation called at wrong ledger boundary"
    );

    let ledger = bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)?;
    let transcript = bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)?;
    ensure!(
        ledger == state.ledger_prefix && transcript == state.orchestration_prefix,
        "retained poison reconciliation inputs changed"
    );
    let records = core::jsonl(&ledger)?;
    let frames = verify_transcript(&transcript, &loaded.policy)?;
    ensure!(frames.len() == 14, "final authenticated transcript coverage");

    let mut files = Files::new(out)?;
    let mut cases = Vec::new();
    for case in POISON_CASES {
        cases.push(load_case(&mut files, &records, case, old)?);
    }
    ensure!(
        cases.windows(2).all(|pair| pair[1].0 == pair[0].0 + 1),
        "poison lifecycle cases are not contiguous"
    );
    let (_, drift, drift_digest) = &cases[0];
    let (_, terminal, terminal_digest) = &cases[1];
    let (restore_index, restore, restore_digest) = &cases[2];
    let (restart_index, restart, restart_digest) = &cases[3];
    let (exit_index, exited, exit_digest) = &cases[4];

    ensure!(
        *restore_digest == poison.restore_evidence_sha256
            && *restart_digest == poison.restart_evidence_sha256
            && *exit_digest == poison.exit_evidence_sha256,
        "retained poison lifecycle commitments changed"
    );

    let coalescing_bytes = files.committed(
        "scheduler/family-53.evidence.json",
        &old.coalescing_evidence_sha256,
    )?;
    let coalescing: Value = serde_json::from_slice(&coalescing_bytes)?;
    let before = &coalescing["final_observation"];
    ready_observation(before, &old.launch)?;
    ensure!(
        coalescing["instance"] == old.launch.instance
            && coalescing["pid"] == old.launch.pid
            && coalescing["final_manifest_sha256"] == sha(&old.manifest_prefix)
            && coalescing["next_forward_id"] == old.next_forward,
        "poison baseline differs from coalescing boundary"
    );

    let mut before_ids = BTreeSet::new();
    for wave in array(&coalescing, "waves")? {
        for response in array(wave, "responses")? {
            ensure!(
                before_ids.insert(uint(response, "request_id")?),
                "duplicate coalescing request ID"
            );
        }
    }
    let trigger_id = uint(&drift["response"], "request_id")?;
    historical_transition(
        before, &old.observation, &before_ids, &old.seen_request_ids, trigger_id,
    )?;
    let mut quiesced = before.clone();
    quiesced["ready"] = json!(false);

    let request = serde_json::to_vec(&trigger_body(old))?;
    let endpoint = core::base_address(&old.core.base_url)?;
    let instance = &old.launch.instance;
    let mut connections = Vec::new();

    // Drift: the logical call increments, but math.check fails before
    // physical forward-ID allocation. No physical pair may be invented.
    let drift_root = root_for(POISON_CASES[0]);
    for (name, expected) in [
        ("observation_before.json", before),
        ("quiesced.json", &quiesced),
        ("armed.json", before),
        ("poisoned.json", &old.observation),
    ] {
        files.pretty(&format!("{drift_root}/{name}"), expected)?;
    }
    for name in ["manifest_before.jsonl", "manifest_after.jsonl"] {
        files.exact(&format!("{drift_root}/{name}"), &old.manifest_prefix)?;
    }
    files.exact(
        &format!("{drift_root}/sched58-drift-trigger.request.bin"),
        &request,
    )?;
    connections.push(saved_response(
        &mut files, &drift_root, &drift["response"], instance, endpoint,
        "sched58-drift-trigger", "POST", "/v1/embeddings", &request,
        Some(trigger_id), ResponseKind::Drift,
    )?);
    parent(&mut files, POISON_CASES[0], drift, json!({
        "schema": "rustv1.accept_m4_live_policy_drift.v1",
        "case_id": POISON_CASES[0],
        "instance": instance,
        "pid": old.launch.pid,
        "launch_binding_sha256": old.launch.binding_sha256,
        "observation_before": before,
        "quiesced": quiesced,
        "armed": before,
        "response": drift["response"],
        "observation_after": old.observation,
        "submitted_trigger_requests": 1,
        "fixture_index": 0,
        "fixture_token_sha256": loaded.bundle.fixtures[0].ids_sha256,
        "request_sha256": sha(&request),
        "logical_backend_request_delta": 1,
        "executor_manifest_interval_bytes": 0,
        "executor_manifest_sha256": sha(&old.manifest_prefix),
        "executor_next_forward_id_unchanged": old.next_forward,
        "physical_begin_records_added": 0,
        "physical_end_records_added": 0,
        "physical_forward_pair_invented": false,
        "failure_boundary":
            "IndependentExecutor::forward_batch_validated: math.check before forward ID",
        "request_ids_before": before_ids,
        "request_ids_after": old.seen_request_ids
    }), old)?;

    // Terminal HTTP routes: all responses are historical, pre-exit.
    let terminal_root = root_for(POISON_CASES[1]);
    files.exact(
        &format!("{terminal_root}/sched58-terminal-refusal.request.bin"),
        &request,
    )?;
    for (response, case, method, path, bytes, kind) in [
        (
            &terminal["livez"]["response"], "sched58-livez",
            "GET", "/health/livez", &[][..], ResponseKind::Live,
        ),
        (
            &terminal["readyz"]["response"], "sched58-readyz",
            "GET", "/health/readyz", &[][..], ResponseKind::Poison,
        ),
        (
            &terminal["subsequent_embeddings_refusal"], "sched58-terminal-refusal",
            "POST", "/v1/embeddings", request.as_slice(), ResponseKind::Poison,
        ),
    ] {
        connections.push(saved_response(
            &mut files, &terminal_root, response, instance, endpoint,
            case, method, path, bytes, None, kind,
        )?);
    }
    parent(&mut files, POISON_CASES[1], terminal, json!({
        "schema": "rustv1.accept_m4_terminal_http_routes.v1",
        "case_id": POISON_CASES[1],
        "instance": instance,
        "pid": old.launch.pid,
        "livez": {
            "method": "GET",
            "path": "/health/livez",
            "response": terminal["livez"]["response"]
        },
        "readyz": {
            "method": "GET",
            "path": "/health/readyz",
            "response": terminal["readyz"]["response"],
            "production_path": "readyz -> App::is_ready -> BackendHandle::ready"
        },
        "subsequent_embeddings_refusal": terminal["subsequent_embeddings_refusal"],
        "observation_unchanged": old.observation,
        "manifest_unchanged_sha256": sha(&old.manifest_prefix),
        "executor_next_forward_id_unchanged": old.next_forward,
        "exposed_request_ids_unchanged": old.seen_request_ids,
        "harness_defined_readiness_predicate": false
    }), old)?;

    // Restore is cleanup, not recovery. Its original three observations
    // remain exactly the same poisoned observation.
    let restore_root = root_for(POISON_CASES[2]);
    let probe = &restore["probe"];
    for name in [
        "observation_before.json",
        "restore_observation.json",
        "observation_after.json",
    ] {
        files.pretty(&format!("{restore_root}/{name}"), &old.observation)?;
    }
    for name in ["manifest_before.jsonl", "manifest_after.jsonl"] {
        files.exact(&format!("{restore_root}/{name}"), &old.manifest_prefix)?;
    }
    files.exact(
        &format!("{restore_root}/sched58-restored-refusal.request.bin"),
        &request,
    )?;
    for (response, case, method, path, bytes, kind) in [
        (
            &probe["livez"], "sched58-restored-livez",
            "GET", "/health/livez", &[][..], ResponseKind::Live,
        ),
        (
            &probe["readyz"], "sched58-restored-readyz",
            "GET", "/health/readyz", &[][..], ResponseKind::Poison,
        ),
        (
            &probe["embeddings_refusal"], "sched58-restored-refusal",
            "POST", "/v1/embeddings", request.as_slice(), ResponseKind::Poison,
        ),
    ] {
        connections.push(saved_response(
            &mut files, &restore_root, response, instance, endpoint,
            case, method, path, bytes, None, kind,
        )?);
    }
    let expected_probe = json!({
        "observation_before": old.observation,
        "restore_observation": old.observation,
        "observation_after": old.observation,
        "livez": probe["livez"],
        "readyz": probe["readyz"],
        "embeddings_refusal": probe["embeddings_refusal"],
        "request_sha256": sha(&request),
        "explicit_restore_control_completed": true,
        "restoration_is_recovery": false,
        "quarantine_cleared": false,
        "logical_backend_request_delta": 0,
        "exposed_new_request_ids": []
    });
    ensure!(*probe == expected_probe, "complete historical restore probe changed");
    parent(&mut files, POISON_CASES[2], restore, json!({
        "schema": "rustv1.accept_m4_restore_stays_poisoned.v1",
        "case_id": POISON_CASES[2],
        "instance": instance,
        "pid": old.launch.pid,
        "launch_binding_sha256": old.launch.binding_sha256,
        "preceding_ledger_sha256": sha(ledger_prefix(&ledger, *restore_index)?),
        "sources": [
            source(&case_path(POISON_CASES[0]), drift_digest, ""),
            source(&case_path(POISON_CASES[1]), terminal_digest, "")
        ],
        "probe": expected_probe,
        "manifest_unchanged_sha256": sha(&old.manifest_prefix),
        "executor_manifest_interval_bytes": 0,
        "executor_next_forward_id_unchanged": old.next_forward,
        "exposed_request_ids_unchanged": old.seen_request_ids,
        "old_observation_preserved": old.observation,
        "owner_join_claimed": false,
        "process_exit_claimed": false
    }), old)?;

    // The preceding saved-lineage validator has already checked MACs,
    // exact restart schemas, target identities and saved absence semantics.
    // Bind the exact frames and complete producer evidence again here.
    ensure!(
        frames[6] == json!({
            "direction": "driver_to_orchestrator",
            "envelope": poison.restart_request
        })
            && frames[7] == json!({
                "direction": "orchestrator_to_driver",
                "envelope": poison.restart_accepted
            })
            && frames[8] == json!({
                "direction": "orchestrator_to_driver",
                "envelope": poison.restart_exit
            })
            && ledger_prefix(&transcript, 9)? == poison.orchestration_prefix
            && ledger_prefix(&ledger, *exit_index + 1)? == poison.ledger_prefix,
        "historical restart frame/byte boundary changed"
    );

    let restart_root = root_for(POISON_CASES[3]);
    files.pretty(
        &format!("{restart_root}/request.json"),
        &poison.restart_request,
    )?;
    files.pretty(
        &format!("{restart_root}/accepted.json"),
        &poison.restart_accepted,
    )?;
    parent(&mut files, POISON_CASES[3], restart, json!({
        "schema": "rustv1.accept_m4_authenticated_restart_authorization.v1",
        "case_id": POISON_CASES[3],
        "instance": instance,
        "pid": old.launch.pid,
        "preceding_ledger_sha256": sha(ledger_prefix(&ledger, *restart_index)?),
        "restore_evidence_path": case_path(POISON_CASES[2]),
        "restore_evidence_sha256": restore_digest,
        "request": poison.restart_request,
        "acceptance": poison.restart_accepted,
        "orchestration_prefix_sha256": sha(ledger_prefix(&transcript, 8)?),
        "old_pre_restart_observation": old.observation,
        "old_manifest_sha256": sha(&old.manifest_prefix),
        "replacement_launch": "await_explicit_authorization_after_driver_absence_proof",
        "restart_authorization_authenticated": true,
        "old_process_exit_independently_proven": false,
        "replacement_reattested": false,
        "owner_join_claimed": false
    }), old)?;

    let exit_root = root_for(POISON_CASES[4]);
    files.pretty(
        &format!("{exit_root}/authenticated_exit.json"),
        &poison.restart_exit,
    )?;
    files.pretty(
        &format!("{exit_root}/process_absence.json"),
        &poison.old_absence,
    )?;
    files.exact(
        &format!("{exit_root}/executor_manifest_snapshot.jsonl"),
        &old.manifest_prefix,
    )?;
    parent(&mut files, POISON_CASES[4], exited, json!({
        "schema": "rustv1.accept_m4_poisoned_process_exit.v1",
        "case_id": POISON_CASES[4],
        "instance": instance,
        "pid": old.launch.pid,
        "preceding_ledger_sha256": sha(ledger_prefix(&ledger, *exit_index)?),
        "restart_evidence_path": case_path(POISON_CASES[3]),
        "restart_evidence_sha256": restart_digest,
        "restore_evidence_sha256": restore_digest,
        "request": poison.restart_request,
        "acceptance": poison.restart_accepted,
        "authenticated_exit": poison.restart_exit,
        "independent_process_absence": poison.old_absence,
        "orchestration_prefix_sha256": sha(&poison.orchestration_prefix),
        "old_pre_exit_observation": old.observation,
        "old_observation_is_historical": true,
        "manifest_unchanged_sha256": sha(&old.manifest_prefix),
        "executor_next_forward_id_unchanged": old.next_forward,
        "exposed_request_ids_unchanged": old.seen_request_ids,
        "containment_boundary": "external_process_exit_and_supervisor_reap",
        "supervisor_reap_independently_awaited_by_driver": false,
        "process_absence_independently_measured_by_driver": true,
        "service_owner_join_claimed": false,
        "quarantine_cleared_in_process": false,
        "replacement_launch_authorized": false
    }), old)?;

    // Seven lifecycle HTTP connections, distinct from the earlier
    // coalescing connections. Do not use a current registry lookup.
    ensure!(connections.len() == 7, "poison HTTP response coverage");
    let mut connection_ids = BTreeSet::new();
    for wave in array(&coalescing, "waves")? {
        for response in array(wave, "responses")? {
            ensure!(
                connection_ids.insert(uint(
                    &response["headers"]["_connection_owner"], "connection_id",
                )?),
                "coalescing connection ID reused"
            );
        }
    }
    for connection in &connections {
        ensure!(
            connection_ids.insert(uint(connection, "connection_id")?),
            "poison connection ID reused across lifecycle/coalescing"
        );
    }

    for case in POISON_CASES {
        flat(&files, &root_for(case))?;
    }

    // Revalidate the preceding chain, including saved historical absence and
    // external final-manifest equality. This is not a live-boundary consumer.
    ensure!(
        super::reconcile(&continuation.recovery_http)?
            == continuation.destructive_owners_reconciliation,
        "preceding artifacts changed during poison reconciliation"
    );
    loaded.recheck()?;
    files.stable()?;
    for case in POISON_CASES {
        flat(&files, &root_for(case))?;
    }
    ensure!(
        bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)? == ledger
            && bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)? == transcript,
        "poison ledger/transcript changed during reconciliation"
    );

    let sources: Vec<Value> = POISON_CASES.iter().zip(&cases)
        .map(|(case, (_, _, digest))| source(&case_path(case), digest, ""))
        .collect();

    Ok(json!({
        "schema": "rustv1.accept_m4_poison_artifact_reconciliation.v1",
        "boundary": POISON_BOUNDARY,
        "policy_sha256": loaded.policy_sha256,
        "preceding_ledger_sha256": sha(&ledger),
        "orchestration_sha256": sha(&transcript),
        "preceding_destructive_owner_reconciliation_sha256":
            sha(&serde_json::to_vec(
                &continuation.destructive_owners_reconciliation
            )?),
        "sources": sources,
        "exact_output_subtrees":
            POISON_CASES.iter().map(|case| root_for(case)).collect::<Vec<_>>(),
        "poison_lifecycle_artifact_coverage_complete": true,
        "complete_parent_evidence_reconstructed": 5,
        "saved_http_responses_checked": 7,
        "saved_http_connection_joins_checked": 7,
        "historical_poisoned_observation": old.observation,
        "historical_observations_relabelled": false,
        "drift_logical_backend_request_delta": 1,
        "drift_physical_manifest_delta_bytes": 0,
        "physical_forward_pair_invented": false,
        "restore_is_recovery": false,
        "old_service_owner_join_claimed": false,
        "old_quarantine_cleared_in_process": false,
        "saved_absence_remeasured": false,
        "supervisor_reap_independently_awaited_by_driver": false,
        "verified_output_sha256": files.hashes,
        "scheduler_launch_and_coalescing_reconciliation_complete": false,
        "complete_scheduler_manifest_reconciled_here": false,
        "whole_transcript_semantics_reconciled": false,
        "independent_mode_artifacts_reconciled": false,
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
    fn poison_transition_preserves_history_and_requires_exact_id_delta() {
        let before = json!({
            "instance": "historical",
            "pid": 42,
            "lifecycle": "ready",
            "worker_alive": true,
            "ready": true,
            "forward_calls": 8,
            "startup_output_sha256": sha(b"startup")
        });
        let after = poisoned(&before).unwrap();
        let old_ids = BTreeSet::from([1u64, 2]);
        let new_ids = BTreeSet::from([1u64, 2, 3]);
        assert!(historical_transition(
            &before, &after, &old_ids, &new_ids, 3
        ).is_ok());

        let mut stopped = after.clone();
        stopped["lifecycle"] = json!("stopped");
        stopped["worker_alive"] = json!(false);
        assert!(historical_transition(
            &before, &stopped, &old_ids, &new_ids, 3
        ).is_err());

        let mut recovered = after.clone();
        recovered["ready"] = json!(true);
        assert!(historical_transition(
            &before, &recovered, &old_ids, &new_ids, 3
        ).is_err());

        assert!(historical_transition(
            &before, &after, &old_ids, &old_ids, 3
        ).is_err());
        assert!(historical_transition(
            &before, &after, &old_ids, &new_ids, 2
        ).is_err());

        let mut overflow = before;
        overflow["forward_calls"] = json!(u64::MAX);
        assert!(poisoned(&overflow).is_err());
    }

    fn connection_fixture(request: &[u8]) -> Value {
        json!({
            "instance": "historical",
            "case_id": "refusal",
            "method": "POST",
            "path": "/v1/embeddings",
            "local_address": "127.0.0.1:32000",
            "peer_address": "127.0.0.1:9700",
            "request_sha256": sha(request),
            "connection_id": 7,
            "owner_pid": std::process::id(),
            "tokio_task_id": "fixture-task",
            "join_outcome": "completed",
            "abort_requested": false,
            "connection_result": {"ok": true, "error": null},
            "join_error": null,
            "join_awaited": true
        })
    }

    #[test]
    fn poison_connections_preserve_transport_error_and_reject_fake_join() {
        // Format fixtures only; not acceptance execution evidence.
        let request = b"request";
        let endpoint: SocketAddr = "127.0.0.1:9700".parse().unwrap();
        let verify = |v: &Value| connection_record(
            v, "historical", "refusal", "POST", "/v1/embeddings",
            request, endpoint,
        );
        let original = connection_fixture(request);
        assert!(verify(&original).is_ok());

        let mut transport_error = original.clone();
        transport_error["connection_result"] =
            json!({"ok": false, "error": "historical transport error"});
        assert!(verify(&transport_error).is_ok());

        let mut unjoined = original.clone();
        unjoined["join_awaited"] = json!(false);
        assert!(verify(&unjoined).is_err());

        let mut cancelled = original.clone();
        cancelled["join_outcome"] = json!("cancelled-after-abort-and-awaited");
        cancelled["connection_result"] = Value::Null;
        cancelled["join_error"] = json!("historical cancellation");
        assert!(verify(&cancelled).is_err());
        cancelled["abort_requested"] = json!(true);
        assert!(verify(&cancelled).is_ok());

        let mut wrong_request = original.clone();
        wrong_request["request_sha256"] = json!(sha(b"other"));
        assert!(verify(&wrong_request).is_err());

        let mut panicked = original;
        panicked["join_outcome"] = json!("panicked");
        assert!(verify(&panicked).is_err());
    }

    #[test]
    fn poison_saved_response_rejects_changed_missing_extra_and_relabelled_files() {
        // Disposable format fixtures, not a production-Service harness.
        let root = std::env::temp_dir().join(format!(
            "rv1-poison-files-{}-{}",
            std::process::id(),
            core::random_label().unwrap()
        ));
        private_directory(&root).unwrap();
        let subtree = root.join("probe");
        private_directory(&subtree).unwrap();

        let result: Result<()> = (|| {
            let request = b"request";
            let body = serde_json::to_vec(&json!({
                "error": {
                    "message": "Executor is retired; restart required",
                    "type": "service_unavailable_error",
                    "param": null,
                    "code": "executor_poisoned"
                }
            }))?;
            let response = json!({
                "case_id": "refusal",
                "http_status": 503,
                "request_id": null,
                "headers": {
                    "x-rv1-instance-id": "historical",
                    "x-acceptance-case-id": "refusal",
                    "x-rv1-request-id": null,
                    "content-type": "application/json",
                    "_connection_owner": connection_fixture(request)
                },
                "response_path": "refusal.response.bin",
                "response_sha256": sha(&body)
            });
            let sidecar = serde_json::to_vec_pretty(&response)?;
            let body_path = subtree.join("refusal.response.bin");
            let sidecar_path = subtree.join("refusal.response.json");
            write_new(&body_path, &body)?;
            write_new(&sidecar_path, &sidecar)?;

            let verify = |response: &Value| -> Result<()> {
                let mut files = Files::new(&root)?;
                saved_response(
                    &mut files, "probe", response, "historical",
                    "127.0.0.1:9700".parse()?,
                    "refusal", "POST", "/v1/embeddings", request,
                    None, ResponseKind::Poison,
                )?;
                flat(&files, "probe")?;
                files.stable()
            };
            verify(&response)?;

            fs::write(&body_path, b"{}")?;
            let changed = verify(&response);
            fs::write(&body_path, &body)?;
            ensure!(changed.is_err(), "changed poison response accepted");

            let hidden = subtree.join("hidden");
            fs::rename(&sidecar_path, &hidden)?;
            let missing = verify(&response);
            fs::rename(&hidden, &sidecar_path)?;
            ensure!(missing.is_err(), "missing poison sidecar accepted");

            let extra = subtree.join("unexpected");
            write_new(&extra, b"x")?;
            let additional = verify(&response);
            fs::remove_file(&extra)?;
            ensure!(additional.is_err(), "extra poison artifact accepted");

            let mut relabelled = response.clone();
            relabelled["request_id"] = json!(99);
            ensure!(
                verify(&relabelled).is_err(),
                "terminal refusal relabelled as admitted"
            );

            #[cfg(unix)]
            {
                let link = subtree.join("linked-response");
                std::os::unix::fs::symlink(&body_path, &link)?;
                let linked = verify(&response);
                fs::remove_file(&link)?;
                ensure!(linked.is_err(), "poison subtree symlink accepted");
            }

            verify(&response)
        })();

        if result.is_ok() {
            fs::remove_dir_all(&root).unwrap();
        }
        result.unwrap_or_else(|error| {
            panic!("poison file checks failed; retained {}: {error:#}", root.display())
        });
    }
}
