use super::*;
use std::{future::Future, time::Duration};

const RESTORE: &str = "family-58-restore-does-not-recover";
const RESTART: &str = "family-58-authenticated-supervised-restart";
const EXIT_CASE: &str = "family-58-old-process-exit";
const FOLLOWING: &str = "family-58-replacement-reattestation";

const LAUNCH_GATE: &str = "await_explicit_authorization_after_driver_absence_proof";

/// Contains credentials indirectly through the retained SchedulerState.
/// Do not derive Debug or Serialize.
pub struct PoisonExitContinuation {
    pub runtime: tokio::runtime::Runtime,
    pub state: PoisonExitState,
}

/// The old observation is historical, not a claim that the exited process
/// still has a live worker. Its frozen fields are deliberately not relabelled.
pub struct PoisonExitState {
    pub old: SchedulerState,
    pub restart_request: Value,
    pub restart_accepted: Value,
    pub restart_exit: Value,
    pub old_absence: Value,
    pub restore_evidence_sha256: String,
    pub restart_evidence_sha256: String,
    pub exit_evidence_sha256: String,
    pub ledger_prefix: Vec<u8>,
    pub orchestration_prefix: Vec<u8>,
}

fn pending(state: &SchedulerState, case: &str) -> Result<()> {
    ensure!(
        state.core.ledger.pending().first()
            == Some(&json!({
                "acceptance_phase": policy::LIFECYCLE,
                "case_id": case
            })),
        "restore/restart runner called at wrong global ledger position"
    );
    Ok(())
}

fn ledger_bytes(state: &SchedulerState) -> Result<Vec<u8>> {
    bounded(&state.core.out.join("accept_records.jsonl"), MANIFEST_LIMIT)
}

fn check_ledger_prefix(state: &SchedulerState, bytes: &[u8]) -> Result<Vec<Value>> {
    let records = core::jsonl(bytes)?;
    let required: Vec<(&str, &str)> = state
        .core
        .loaded
        .required
        .phases
        .iter()
        .flat_map(|(phase, ids)| ids.iter().map(move |id| (phase.as_str(), id.as_str())))
        .collect();

    let expected_count = required
        .len()
        .checked_sub(state.core.ledger.pending().len())
        .context("ledger pending count exceeds required count")?;

    ensure!(records.len() == expected_count, "ledger prefix coverage");

    for (record, (phase, case)) in records.iter().zip(required) {
        closed(
            record,
            &["acceptance_phase", "case_id", "status", "pass", "details"],
        )?;
        ensure!(
            record["acceptance_phase"] == phase
                && record["case_id"] == case
                && record["status"] == "executed"
                && record["pass"] == true
                && record["details"].is_object(),
            "preceding ledger order/status/schema changed"
        );
    }
    Ok(records)
}

/// Read an earlier artifact through its preceding ledger digest.
/// Never replace that digest with a newly computed baseline.
fn preceding_case(state: &SchedulerState, records: &[Value], case: &str) -> Result<(Value, Value)> {
    let matches: Vec<&Value> = records
        .iter()
        .filter(|record| {
            record["acceptance_phase"] == policy::LIFECYCLE && record["case_id"] == case
        })
        .collect();
    ensure!(
        matches.len() == 1,
        "missing/duplicate preceding lifecycle case"
    );

    let details = &matches[0]["details"];
    let relative = format!("scheduler/{case}/evidence.json");
    ensure!(
        details["evidence_path"] == relative,
        "preceding lifecycle evidence path changed"
    );
    let digest = text(details, "evidence_sha256")?;
    let bytes = bounded(&under(&state.core.out, &relative)?, JSON_LIMIT)?;
    ensure!(
        sha(&bytes) == digest,
        "preceding lifecycle evidence changed"
    );

    let evidence: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        evidence["case_id"] == case
            && evidence["policy_sha256"] == state.core.loaded.policy_sha256
            && evidence["driver_binary_sha256"] == state.core.loaded.artifacts["accept_m4"].sha256
            && evidence["instance"] == state.launch.instance
            && evidence["pid"] == state.launch.pid,
        "preceding lifecycle identity/binding"
    );
    Ok((evidence, source(&relative, digest, "")))
}

fn check_saved_reply(root: &Path, evidence: &Value) -> Result<()> {
    let name = text(evidence, "case_id")?;
    ensure!(label_valid(name), "invalid saved reply label");
    ensure!(
        evidence["response_path"] == format!("{name}.response.bin"),
        "saved response path mismatch"
    );
    let path = under(root, text(evidence, "response_path")?)?;
    ensure!(
        file_sha(&path)? == text(evidence, "response_sha256")?
            && value(&under(root, &format!("{name}.response.json"))?)? == *evidence,
        "preceding response bytes/metadata changed"
    );
    Ok(())
}

fn poison_observation(before: &Value) -> Result<()> {
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
        before["lifecycle"] == "poisoned"
            && before["worker_alive"] == true
            && before["ready"] == false,
        "restore requires the retained operationally poisoned worker"
    );
    Ok(())
}

/// Shared by the live runner and the CPU production-Service harness.
///
/// The callback invokes actual production controls. This helper implements
/// assertions and HTTP stimulus, not lifecycle transitions or recovery.
pub(super) async fn restore_probe<F, Fut>(
    mut command: F,
    http: &Http,
    body: Value,
    before: &Value,
    out: &Path,
) -> Result<Value>
where
    F: FnMut(&'static str) -> Fut,
    Fut: Future<Output = Result<Value>>,
{
    poison_observation(before)?;
    ensure!(
        command("observe").await? == *before,
        "observation changed before explicit restore"
    );
    save_json(&out.join("observation_before.json"), before)?;

    let restored = command("restore").await?;
    save_json(&out.join("restore_observation.json"), &restored)?;
    ensure!(
        restored == *before && command("observe").await? == *before,
        "restore changed poisoned lifecycle, readiness, identity or counters"
    );

    let live = http
        .get_health("sched58-restored-livez", "/health/livez")
        .await?;
    let live_evidence = save_reply(out, "sched58-restored-livez", &live)?;
    ensure!(
        live.status == 200 && live.bytes.is_empty() && live.request_id.is_none(),
        "restored poisoned process lost its real livez route"
    );

    let ready = http
        .get_health("sched58-restored-readyz", "/health/readyz")
        .await?;
    let ready_evidence = save_reply(out, "sched58-restored-readyz", &ready)?;
    production_error(&ready, "executor_poisoned")?;
    ensure!(ready.request_id.is_none(), "readyz supplied request ID");

    let request = serde_json::to_vec(&body)?;
    write_new(&out.join("sched58-restored-refusal.request.bin"), &request)?;
    let refusal = http
        .send("sched58-restored-refusal", &Wire::json(body)?)
        .await?;
    let refusal_evidence = save_reply(out, "sched58-restored-refusal", &refusal)?;
    production_error(&refusal, "executor_poisoned")?;
    ensure!(
        refusal.request_id.is_none(),
        "restore reopened embeddings admission"
    );

    let after = command("observe").await?;
    save_json(&out.join("observation_after.json"), &after)?;
    ensure!(
        after == *before,
        "post-restore probes changed frozen observation"
    );

    Ok(json!({
        "observation_before": before,
        "restore_observation": restored,
        "observation_after": after,
        "livez": live_evidence,
        "readyz": ready_evidence,
        "embeddings_refusal": refusal_evidence,
        "request_sha256": sha(&request),
        "explicit_restore_control_completed": true,
        "restoration_is_recovery": false,
        "quarantine_cleared": false,
        "logical_backend_request_delta": 0,
        "exposed_new_request_ids": []
    }))
}

async fn restore(state: &SchedulerState, out: &Path, preceding_ledger: &[u8]) -> Result<Value> {
    unchanged(state).await?;
    poison_observation(&state.observation)?;
    let records = check_ledger_prefix(state, preceding_ledger)?;

    let (drift, drift_source) = preceding_case(state, &records, DRIFT)?;
    let (terminal, terminal_source) = preceding_case(state, &records, TERMINAL)?;

    ensure!(
        drift["launch_binding_sha256"] == state.launch.binding_sha256
            && drift["observation_after"] == state.observation
            && drift["executor_manifest_sha256"] == sha(&state.manifest_prefix)
            && drift["executor_next_forward_id_unchanged"] == state.next_forward
            && drift["request_ids_after"] == json!(state.seen_request_ids)
            && terminal["observation_unchanged"] == state.observation
            && terminal["manifest_unchanged_sha256"] == sha(&state.manifest_prefix)
            && terminal["executor_next_forward_id_unchanged"] == state.next_forward
            && terminal["exposed_request_ids_unchanged"] == json!(state.seen_request_ids),
        "restore cursor differs from preceding digest-bound lifecycle evidence"
    );

    let drift_out = state.directory.join(DRIFT);
    let terminal_out = state.directory.join(TERMINAL);
    ensure!(
        bounded(&drift_out.join("manifest_after.jsonl"), MANIFEST_LIMIT)? == state.manifest_prefix
            && value(&drift_out.join("poisoned.json"))? == state.observation,
        "saved poisoned snapshot changed"
    );
    check_saved_reply(&drift_out, &drift["response"])?;
    ensure!(
        file_sha(&drift_out.join("sched58-drift-trigger.request.bin"))?
            == text(&drift, "request_sha256")?,
        "preceding drift request changed"
    );

    for response in [
        &terminal["livez"]["response"],
        &terminal["readyz"]["response"],
        &terminal["subsequent_embeddings_refusal"],
    ] {
        check_saved_reply(&terminal_out, response)?;
    }

    let body = trigger_body(state);
    let expected_request = serde_json::to_vec(&body)?;
    ensure!(
        bounded(
            &terminal_out.join("sched58-terminal-refusal.request.bin"),
            JSON_LIMIT
        )? == expected_request,
        "preceding terminal refusal request changed"
    );

    write_new(&out.join("manifest_before.jsonl"), &state.manifest_prefix)?;
    let http = client(state)?;
    let probe = restore_probe(
        |command| state.core.control.call(command, None),
        &http,
        body,
        &state.observation,
        out,
    )
    .await?;

    unchanged(state).await?;
    let manifest = bounded(&state.launch.manifest, MANIFEST_LIMIT)?;
    write_new(&out.join("manifest_after.jsonl"), &manifest)?;
    ensure!(
        manifest == state.manifest_prefix,
        "restore produced physical work"
    );

    Ok(json!({
        "schema": "rustv1.accept_m4_restore_stays_poisoned.v1",
        "case_id": RESTORE,
        "instance": state.launch.instance,
        "pid": state.launch.pid,
        "launch_binding_sha256": state.launch.binding_sha256,
        "preceding_ledger_sha256": sha(preceding_ledger),
        "sources": [drift_source, terminal_source],
        "probe": probe,
        "manifest_unchanged_sha256": sha(&manifest),
        "executor_manifest_interval_bytes": 0,
        "executor_next_forward_id_unchanged": state.next_forward,
        "exposed_request_ids_unchanged": state.seen_request_ids,
        "old_observation_preserved": state.observation,
        "owner_join_claimed": false,
        "process_exit_claimed": false
    }))
}

fn validate_request_fields(fields: &Value) -> Result<()> {
    closed(
        fields,
        &[
            "restart_request_id",
            "reason",
            "instance",
            "pid",
            "process_start_ticks",
            "boot_id",
            "running_binary_sha256",
            "policy_sha256",
            "launch_binding_sha256",
            "restore_evidence_sha256",
            "manifest_sha256",
            "replacement_launch",
        ],
    )?;
    for key in [
        "restart_request_id",
        "running_binary_sha256",
        "policy_sha256",
        "launch_binding_sha256",
        "restore_evidence_sha256",
        "manifest_sha256",
    ] {
        ensure!(
            crate::contract::digest_valid(text(fields, key)?),
            "invalid restart nonce/digest: {key}"
        );
    }
    ensure!(
        fields["reason"] == "poison_terminal"
            && label_valid(text(fields, "instance")?)
            && uint(fields, "pid")? > 0
            && uint(fields, "pid")? <= u64::from(u32::MAX)
            && uint(fields, "process_start_ticks")? > 0
            && !text(fields, "boot_id")?.is_empty()
            && text(fields, "boot_id")?.len() <= 80
            && fields["replacement_launch"] == LAUNCH_GATE,
        "restart request identity/launch gate"
    );
    Ok(())
}

fn check_accepted(request: &Value, accepted: &Value) -> Result<()> {
    ensure!(
        request["kind"] == "orch_request_restart" && accepted["kind"] == "orch_restart_accepted",
        "unexpected restart request/acceptance kind"
    );
    validate_request_fields(&request["fields"])?;
    uint(request, "seq")?;
    closed(
        &accepted["fields"],
        &[
            "restart_request_id",
            "request_seq",
            "instance",
            "pid",
            "replacement_launch",
        ],
    )?;
    let fields = &accepted["fields"];
    ensure!(
        fields["restart_request_id"] == request["fields"]["restart_request_id"]
            && fields["request_seq"] == request["seq"]
            && fields["instance"] == request["fields"]["instance"]
            && fields["pid"] == request["fields"]["pid"]
            && fields["replacement_launch"] == LAUNCH_GATE,
        "restart acceptance does not bind the exact request"
    );
    Ok(())
}

fn check_exit(request: &Value, exit: &Value) -> Result<()> {
    ensure!(
        request["kind"] == "orch_request_restart" && exit["kind"] == "orch_restart_exit",
        "unexpected restart exit kind"
    );
    validate_request_fields(&request["fields"])?;
    closed(
        &exit["fields"],
        &[
            "restart_request_id",
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
        fields["restart_request_id"] == request["fields"]["restart_request_id"]
            && fields["request_seq"] == request["seq"]
            && fields["instance"] == request["fields"]["instance"]
            && fields["pid"] == request["fields"]["pid"]
            && fields["termination_signal"] == "SIGTERM"
            && fields["exit_status"] == 1
            && fields["exit_signal"].is_null()
            && fields["process_reaped"] == true
            && fields["replacement_launched"] == false,
        "poisoned exit status/reap/identity/launch gate mismatch"
    );
    Ok(())
}

async fn request_restart(
    orchestration: &mut core::Orchestration,
    fields: Value,
) -> Result<(Value, Value)> {
    validate_request_fields(&fields)?;
    let request = orchestration.send("orch_request_restart", fields).await?;
    let accepted = orchestration.receive().await?;
    check_accepted(&request, &accepted)?;
    orchestration.sync()?;
    Ok((request, accepted))
}

async fn receive_exit(orchestration: &mut core::Orchestration, request: &Value) -> Result<Value> {
    let exit = orchestration
        .receive_with_budget(Duration::from_secs(120))
        .await?;
    check_exit(request, &exit)?;
    orchestration.sync()?;
    Ok(exit)
}

/// Check the exact transcript suffix, not just whether matching envelopes
/// can be found somewhere in an otherwise unconstrained transcript.
fn transcript_extension(
    state: &SchedulerState,
    previous: &[u8],
    expected: &[Value],
) -> Result<Vec<u8>> {
    state.core.orchestration.sync()?;
    let bytes = bounded(&state.core.out.join("orchestration.jsonl"), JSON_LIMIT)?;
    ensure!(
        bytes.starts_with(previous),
        "orchestration transcript prefix changed"
    );

    let records = verify_transcript(&bytes, &state.core.loaded.policy)?;
    let old = verify_transcript(previous, &state.core.loaded.policy)?;
    ensure!(
        records.len() == old.len() + expected.len()
            && records[..old.len()] == old
            && records[old.len()..] == *expected,
        "unexpected/missing orchestration exchange"
    );
    Ok(bytes)
}

fn stage_directory(state: &SchedulerState, case: &str) -> Result<PathBuf> {
    pending(state, case)?;
    let out = state.directory.join(case);
    private_directory(&out)?;
    Ok(out)
}

fn append_pass(
    state: &mut SchedulerState,
    case: &str,
    mut evidence: Value,
    previous_ledger: &mut Vec<u8>,
) -> Result<String> {
    pending(state, case)?;
    ensure!(
        ledger_bytes(state)? == *previous_ledger,
        "global ledger changed during lifecycle case"
    );
    state.core.loaded.recheck()?;
    state.core.orchestration.sync()?;
    evidence["policy_sha256"] = json!(state.core.loaded.policy_sha256);
    evidence["driver_binary_sha256"] = json!(state.core.loaded.artifacts["accept_m4"].sha256);

    let digest = save_json(&state.directory.join(case).join("evidence.json"), &evidence)?;
    state.core.ledger.append(
        policy::LIFECYCLE,
        case,
        true,
        json!({
            "evidence_path": format!("scheduler/{case}/evidence.json"),
            "evidence_sha256": digest,
            "instance": state.launch.instance,
            "pid": state.launch.pid
        }),
        true,
    )?;
    let bytes = ledger_bytes(state)?;
    ensure!(
        bytes.starts_with(previous_ledger),
        "ledger append rewrote prefix"
    );
    check_ledger_prefix(state, &bytes)?;
    *previous_ledger = bytes;
    Ok(digest)
}

fn fail(state: &mut SchedulerState, case: &str, error: &anyhow::Error) {
    let _ = save_json(
        &state.directory.join(case).join("failure.json"),
        &json!({
            "error": format!("{error:#}"),
            "completion_claimed": false
        }),
    );
    let _ = state.core.ledger.append(
        policy::LIFECYCLE,
        case,
        false,
        json!({
            "error": format!("{error:#}"),
            "retained_evidence_path": format!("scheduler/{case}")
        }),
        true,
    );
}

async fn execute(mut state: SchedulerState) -> Result<PoisonExitState> {
    pending(&state, RESTORE)?;
    state.core.loaded.recheck()?;
    let mut ledger = ledger_bytes(&state)?;
    check_ledger_prefix(&state, &ledger)?;
    state.core.orchestration.sync()?;
    let mut transcript = bounded(&state.core.out.join("orchestration.jsonl"), JSON_LIMIT)?;
    verify_transcript(&transcript, &state.core.loaded.policy)?;

    let work: Result<String> = async {
        let out = stage_directory(&state, RESTORE)?;
        let evidence = restore(&state, &out, &ledger).await?;
        transcript = transcript_extension(&state, &transcript, &[])?;
        append_pass(&mut state, RESTORE, evidence, &mut ledger)
    }
    .await;
    let restore_digest = match work {
        Ok(digest) => digest,
        Err(error) => {
            fail(&mut state, RESTORE, &error);
            return Err(error);
        }
    };

    let work: Result<(Value, Value, String)> = async {
        let out = stage_directory(&state, RESTART)?;
        state.core.loaded.recheck()?;
        unchanged(&state).await?;

        let restore_path = state.directory.join(RESTORE).join("evidence.json");
        ensure!(
            file_sha(&restore_path)? == restore_digest,
            "saved restore evidence changed before restart authorization"
        );
        let fields = json!({
            "restart_request_id": core::random_label()?,
            "reason": "poison_terminal",
            "instance": state.launch.instance,
            "pid": state.launch.pid,
            "process_start_ticks": state.launch.process.start_ticks,
            "boot_id": state.launch.process.boot_id,
            "running_binary_sha256": state.launch.process.executable_sha256,
            "policy_sha256": state.core.loaded.policy_sha256,
            "launch_binding_sha256": state.launch.binding_sha256,
            "restore_evidence_sha256": restore_digest,
            "manifest_sha256": sha(&state.manifest_prefix),
            "replacement_launch": LAUNCH_GATE
        });
        let (request, accepted) = request_restart(&mut state.core.orchestration, fields).await?;

        // The supervisor may terminate the process immediately after its
        // acceptance. No subsequent check_live/observe is appropriate here.
        save_json(&out.join("request.json"), &request)?;
        save_json(&out.join("accepted.json"), &accepted)?;
        transcript = transcript_extension(
            &state,
            &transcript,
            &[
                json!({
                    "direction": "driver_to_orchestrator",
                    "envelope": request
                }),
                json!({
                    "direction": "orchestrator_to_driver",
                    "envelope": accepted
                }),
            ],
        )?;
        let evidence = json!({
            "schema": "rustv1.accept_m4_authenticated_restart_authorization.v1",
            "case_id": RESTART,
            "instance": state.launch.instance,
            "pid": state.launch.pid,
            "preceding_ledger_sha256": sha(&ledger),
            "restore_evidence_path": format!("scheduler/{RESTORE}/evidence.json"),
            "restore_evidence_sha256": restore_digest,
            "request": request,
            "acceptance": accepted,
            "orchestration_prefix_sha256": sha(&transcript),
            "old_pre_restart_observation": state.observation,
            "old_manifest_sha256": sha(&state.manifest_prefix),
            "replacement_launch": LAUNCH_GATE,
            "restart_authorization_authenticated": true,
            "old_process_exit_independently_proven": false,
            "replacement_reattested": false,
            "owner_join_claimed": false
        });
        let digest = append_pass(&mut state, RESTART, evidence, &mut ledger)?;
        Ok((request, accepted, digest))
    }
    .await;
    let (request, accepted, restart_digest) = match work {
        Ok(result) => result,
        Err(error) => {
            fail(&mut state, RESTART, &error);
            return Err(error);
        }
    };

    let work: Result<(Value, Value, String)> = async {
        let out = stage_directory(&state, EXIT_CASE)?;
        let exit = receive_exit(&mut state.core.orchestration, &request).await?;
        save_json(&out.join("authenticated_exit.json"), &exit)?;

        // Receiving the authenticated reap report is not the independent
        // absence measurement. A zombie or reused PID cannot satisfy this.
        let absence = state.launch.process.prove_absent().await?;
        save_json(&out.join("process_absence.json"), &absence)?;

        ensure!(
            bounded(&state.launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
            "unreconciled executor activity during poisoned process shutdown"
        );
        write_new(
            &out.join("executor_manifest_snapshot.jsonl"),
            &state.manifest_prefix,
        )?;
        ensure!(
            file_sha(&state.directory.join(RESTART).join("evidence.json"))? == restart_digest
                && file_sha(&state.directory.join(RESTORE).join("evidence.json"))?
                    == restore_digest,
            "preceding restart/restore evidence changed"
        );

        transcript = transcript_extension(
            &state,
            &transcript,
            &[json!({
                "direction": "orchestrator_to_driver",
                "envelope": exit
            })],
        )?;
        let evidence = json!({
            "schema": "rustv1.accept_m4_poisoned_process_exit.v1",
            "case_id": EXIT_CASE,
            "instance": state.launch.instance,
            "pid": state.launch.pid,
            "preceding_ledger_sha256": sha(&ledger),
            "restart_evidence_path": format!("scheduler/{RESTART}/evidence.json"),
            "restart_evidence_sha256": restart_digest,
            "restore_evidence_sha256": restore_digest,
            "request": request,
            "acceptance": accepted,
            "authenticated_exit": exit,
            "independent_process_absence": absence,
            "orchestration_prefix_sha256": sha(&transcript),
            "old_pre_exit_observation": state.observation,
            "old_observation_is_historical": true,
            "manifest_unchanged_sha256": sha(&state.manifest_prefix),
            "executor_next_forward_id_unchanged": state.next_forward,
            "exposed_request_ids_unchanged": state.seen_request_ids,
            "containment_boundary": "external_process_exit_and_supervisor_reap",
            "supervisor_reap_independently_awaited_by_driver": false,
            "process_absence_independently_measured_by_driver": true,
            "service_owner_join_claimed": false,
            "quarantine_cleared_in_process": false,
            "replacement_launch_authorized": false
        });
        let digest = append_pass(&mut state, EXIT_CASE, evidence, &mut ledger)?;
        Ok((exit, absence, digest))
    }
    .await;
    let (exit, absence, exit_digest) = match work {
        Ok(result) => result,
        Err(error) => {
            fail(&mut state, EXIT_CASE, &error);
            return Err(error);
        }
    };

    pending(&state, FOLLOWING)?;
    ensure!(ledger_bytes(&state)? == ledger, "post-exit ledger changed");
    transcript = transcript_extension(&state, &transcript, &[])?;

    Ok(PoisonExitState {
        old: state,
        restart_request: request,
        restart_accepted: accepted,
        restart_exit: exit,
        old_absence: absence,
        restore_evidence_sha256: restore_digest,
        restart_evidence_sha256: restart_digest,
        exit_evidence_sha256: exit_digest,
        ledger_prefix: ledger,
        orchestration_prefix: transcript,
    })
}

pub(crate) fn run(continuation: SchedulerContinuation) -> Result<PoisonExitContinuation> {
    let SchedulerContinuation { runtime, state } = continuation;
    let retained = state.directory.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.block_on(execute(state))
    }));
    match result {
        Ok(result) => Ok(PoisonExitContinuation {
            runtime,
            state: result?,
        }),
        Err(payload) => {
            let _ = save_json(
                &retained.join("restore_restart_exit_panic.json"),
                &json!({
                    "panic": true,
                    "retained_evidence_directory": retained,
                    "completion_claimed": false
                }),
            );
            eprintln!(
                "restore/restart/exit panic; evidence retained at {}",
                retained.display()
            );
            std::panic::resume_unwind(payload)
        }
    }
}

#[cfg(test)]
#[path = "accept_m4_lifecycle_restart_prefix_tests.rs"]
mod tests;
