use super::*;

const CASE: &str = "family-58-replacement-reattestation";
const NEXT_CASE: &str = "family-58-exact-recovery";

/// Contains credentials indirectly. Do not derive Debug or Serialize.
pub struct RecoveryReadyContinuation {
    pub runtime: tokio::runtime::Runtime,
    pub state: RecoveryReadyState,
}

/// The exited process and the fresh process remain separate identities.
///
/// poison_exit.old retains its historical poisoned observation, manifest
/// cursor and request IDs. Only its coordinator ledger/orchestration advance.
///
/// The replacement has completed startup and has zero logical requests.
/// No HTTP recovery stimulus has yet been submitted.
pub struct RecoveryReadyState {
    pub poison_exit: PoisonExitState,
    pub replacement: Launch,
    pub reattestation_evidence_sha256: String,
    pub ledger_prefix: Vec<u8>,
    pub orchestration_prefix: Vec<u8>,
}

fn read_bound(root: &Path, relative: &str, digest: &str) -> Result<Vec<u8>> {
    ensure!(
        crate::contract::digest_valid(digest),
        "invalid preceding artifact digest"
    );
    let bytes = bounded(&under(root, relative)?, JSON_LIMIT)?;
    ensure!(sha(&bytes) == digest, "preceding artifact changed");
    Ok(bytes)
}

/// Exact byte prefix preceding record index, including its final LF.
///
/// Validates the entire JSONL before using line boundaries. It does not
/// reserialize records and accidentally substitute a different byte chain.
fn prefix_before(bytes: &[u8], index: usize) -> Result<&[u8]> {
    let records = core::jsonl(bytes)?;
    ensure!(index <= records.len(), "JSONL prefix index");
    let offset = bytes
        .split_inclusive(|byte| *byte == b'\n')
        .take(index)
        .map(<[u8]>::len)
        .sum::<usize>();
    Ok(&bytes[..offset])
}

fn historical_cursor(
    exit: &Value,
    observation: &Value,
    manifest: &[u8],
    next_forward: u64,
    request_ids: &BTreeSet<u64>,
) -> Result<()> {
    poison_observation(observation)?;
    ensure!(
        exit["old_pre_exit_observation"] == *observation
            && exit["old_observation_is_historical"] == true
            && exit["manifest_unchanged_sha256"] == sha(manifest)
            && exit["executor_next_forward_id_unchanged"] == next_forward
            && exit["exposed_request_ids_unchanged"] == json!(request_ids)
            && exit["service_owner_join_claimed"] == false
            && exit["quarantine_cleared_in_process"] == false
            && exit["replacement_launch_authorized"] == false,
        "historical poisoned cursor was changed or relabelled"
    );
    Ok(())
}

/// Validate retained state against already-produced evidence before granting
/// any additional launch authority. No live observation of the exited
/// process is attempted.
fn consume_exit(retained: &PoisonExitState) -> Result<Value> {
    let state = &retained.old;
    pending(state, CASE)?;
    state.core.loaded.recheck()?;

    ensure!(
        ledger_bytes(state)? == retained.ledger_prefix,
        "retained exit ledger prefix changed"
    );
    let records = check_ledger_prefix(state, &retained.ledger_prefix)?;
    ensure!(records.len() >= 3, "missing restart ledger suffix");

    let mut sources = Vec::new();
    let mut evidence = Vec::new();
    for (offset, (case, expected_digest)) in [
        (RESTORE, retained.restore_evidence_sha256.as_str()),
        (RESTART, retained.restart_evidence_sha256.as_str()),
        (EXIT_CASE, retained.exit_evidence_sha256.as_str()),
    ]
    .into_iter()
    .enumerate()
    {
        let index = records.len() - 3 + offset;
        let record = &records[index];
        ensure!(
            record["acceptance_phase"] == policy::LIFECYCLE
                && record["case_id"] == case
                && record["details"]["evidence_sha256"] == expected_digest,
            "restore/restart/exit ledger suffix or retained digest changed"
        );
        let (saved, src) = preceding_case(state, &records, case)?;
        ensure!(
            saved["preceding_ledger_sha256"] == sha(prefix_before(&retained.ledger_prefix, index)?),
            "preceding lifecycle ledger byte-chain mismatch"
        );
        sources.push(src);
        evidence.push(saved);
    }

    let restore = &evidence[0];
    let restart = &evidence[1];
    let exit = &evidence[2];

    historical_cursor(
        exit,
        &state.observation,
        &state.manifest_prefix,
        state.next_forward,
        &state.seen_request_ids,
    )?;
    ensure!(
        restore["old_observation_preserved"] == state.observation
            && restore["manifest_unchanged_sha256"] == sha(&state.manifest_prefix)
            && restore["executor_next_forward_id_unchanged"] == state.next_forward
            && restore["exposed_request_ids_unchanged"] == json!(state.seen_request_ids)
            && restart["old_pre_restart_observation"] == state.observation
            && restart["old_manifest_sha256"] == sha(&state.manifest_prefix)
            && restart["restore_evidence_sha256"] == retained.restore_evidence_sha256
            && exit["restore_evidence_sha256"] == retained.restore_evidence_sha256
            && exit["restart_evidence_sha256"] == retained.restart_evidence_sha256
            && restart["request"] == retained.restart_request
            && restart["acceptance"] == retained.restart_accepted
            && exit["request"] == retained.restart_request
            && exit["acceptance"] == retained.restart_accepted
            && exit["authenticated_exit"] == retained.restart_exit
            && exit["independent_process_absence"] == retained.old_absence,
        "retained restart exchange differs from preceding evidence"
    );

    check_accepted(&retained.restart_request, &retained.restart_accepted)?;
    check_exit(&retained.restart_request, &retained.restart_exit)?;

    let request = &retained.restart_request["fields"];
    ensure!(
        request["instance"] == state.launch.instance
            && request["pid"] == state.launch.pid
            && request["process_start_ticks"] == state.launch.process.start_ticks
            && request["boot_id"] == state.launch.process.boot_id
            && request["running_binary_sha256"] == state.launch.process.executable_sha256
            && request["policy_sha256"] == state.core.loaded.policy_sha256
            && request["launch_binding_sha256"] == state.launch.binding_sha256
            && request["restore_evidence_sha256"] == retained.restore_evidence_sha256
            && request["manifest_sha256"] == sha(&state.manifest_prefix),
        "restart target differs from retained process identity"
    );

    ensure!(
        file_sha(&state.directory.join("launch_binding.json"))? == state.launch.binding_sha256
            && bounded(&state.launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix
            && bounded(
                &state
                    .directory
                    .join(EXIT_CASE)
                    .join("executor_manifest_snapshot.jsonl"),
                MANIFEST_LIMIT,
            )? == state.manifest_prefix
            && value(&state.directory.join(EXIT_CASE).join("process_absence.json"))?
                == retained.old_absence
            && value(
                &state
                    .directory
                    .join(EXIT_CASE)
                    .join("authenticated_exit.json")
            )? == retained.restart_exit
            && value(&state.directory.join(RESTART).join("request.json"))?
                == retained.restart_request
            && value(&state.directory.join(RESTART).join("accepted.json"))?
                == retained.restart_accepted,
        "saved process-exit artifacts changed"
    );

    let transcript = transcript_extension(state, &retained.orchestration_prefix, &[])?;
    let frames = verify_transcript(&transcript, &state.core.loaded.policy)?;
    ensure!(frames.len() >= 3, "missing authenticated restart suffix");
    let suffix = &frames[frames.len() - 3..];
    ensure!(
        suffix
            == [
                json!({
                    "direction": "driver_to_orchestrator",
                    "envelope": retained.restart_request
                }),
                json!({
                    "direction": "orchestrator_to_driver",
                    "envelope": retained.restart_accepted
                }),
                json!({
                    "direction": "orchestrator_to_driver",
                    "envelope": retained.restart_exit
                }),
            ]
            && restart["orchestration_prefix_sha256"]
                == sha(prefix_before(&transcript, frames.len() - 1)?)
            && exit["orchestration_prefix_sha256"] == sha(&transcript),
        "restart transcript suffix or evidence commitment changed"
    );
    ensure!(
        !frames.iter().any(|frame| {
            matches!(
                frame["envelope"]["kind"].as_str(),
                Some("orch_request_recovery_launch" | "orch_recovery_launch_accepted")
            )
        }),
        "recovery launch was already authorized"
    );

    Ok(json!({
        "sources": sources,
        "historical_observation": state.observation,
        "historical_manifest_sha256": sha(&state.manifest_prefix),
        "historical_next_forward_id": state.next_forward,
        "historical_request_ids": state.seen_request_ids,
        "consumed_exit_evidence_sha256": retained.exit_evidence_sha256,
        "consumed_ledger_sha256": sha(&retained.ledger_prefix),
        "consumed_orchestration_sha256": sha(&retained.orchestration_prefix)
    }))
}

fn validate_launch_fields(fields: &Value) -> Result<()> {
    closed(
        fields,
        &[
            "launch_request_id",
            "reason",
            "restart_request_id",
            "restart_request_seq",
            "previous_instance",
            "previous_pid",
            "previous_process_start_ticks",
            "previous_boot_id",
            "policy_sha256",
            "previous_launch_binding_sha256",
            "exit_evidence_sha256",
            "old_absence_sha256",
        ],
    )?;
    for name in [
        "launch_request_id",
        "restart_request_id",
        "policy_sha256",
        "previous_launch_binding_sha256",
        "exit_evidence_sha256",
        "old_absence_sha256",
    ] {
        ensure!(
            crate::contract::digest_valid(text(fields, name)?),
            "invalid recovery launch nonce/digest: {name}"
        );
    }
    uint(fields, "restart_request_seq")?;
    ensure!(
        fields["reason"] == "after_poison_exit"
            && fields["launch_request_id"] != fields["restart_request_id"]
            && label_valid(text(fields, "previous_instance")?)
            && uint(fields, "previous_pid")? > 0
            && uint(fields, "previous_pid")? <= u64::from(u32::MAX)
            && uint(fields, "previous_process_start_ticks")? > 0
            && !text(fields, "previous_boot_id")?.is_empty()
            && text(fields, "previous_boot_id")?.len() <= 80,
        "invalid recovery launch identity"
    );
    Ok(())
}

fn check_launch_accepted(request: &Value, accepted: &Value) -> Result<()> {
    ensure!(
        request["kind"] == "orch_request_recovery_launch"
            && accepted["kind"] == "orch_recovery_launch_accepted",
        "unexpected recovery authorization/acceptance kind"
    );
    validate_launch_fields(&request["fields"])?;
    uint(request, "seq")?;
    closed(
        &accepted["fields"],
        &[
            "launch_request_id",
            "request_seq",
            "restart_request_id",
            "previous_instance",
            "previous_pid",
        ],
    )?;
    let fields = &accepted["fields"];
    ensure!(
        fields["launch_request_id"] == request["fields"]["launch_request_id"]
            && fields["request_seq"] == request["seq"]
            && fields["restart_request_id"] == request["fields"]["restart_request_id"]
            && fields["previous_instance"] == request["fields"]["previous_instance"]
            && fields["previous_pid"] == request["fields"]["previous_pid"],
        "recovery acceptance does not bind the exact authorization"
    );
    Ok(())
}

fn check_ready(request: &Value, ready: &Value) -> Result<()> {
    ensure!(
        ready["kind"] == "orch_ready",
        "expected recovery orch_ready"
    );
    closed(
        &ready["fields"],
        &[
            "launch_request_id",
            "instance",
            "pid",
            "start_utc",
            "manifest",
            "attestation",
        ],
    )?;
    let fields = &ready["fields"];
    ensure!(
        fields["launch_request_id"] == request["fields"]["launch_request_id"]
            && label_valid(text(fields, "instance")?)
            && fields["instance"] != request["fields"]["previous_instance"]
            && uint(fields, "pid")? > 0
            && uint(fields, "pid")? <= u64::from(u32::MAX)
            && fields["pid"] != request["fields"]["previous_pid"]
            && !text(fields, "start_utc")?.is_empty()
            && text(fields, "start_utc")?.len() <= 80
            && Path::new(text(fields, "manifest")?).is_absolute()
            && Path::new(text(fields, "attestation")?).is_absolute(),
        "recovery ready identity/path/authorization mismatch"
    );
    Ok(())
}

async fn authorize(
    orchestration: &mut core::Orchestration,
    fields: Value,
) -> Result<(Value, Value, Value)> {
    validate_launch_fields(&fields)?;
    let request = orchestration
        .send("orch_request_recovery_launch", fields)
        .await?;
    let accepted = orchestration.receive().await?;
    check_launch_accepted(&request, &accepted)?;
    let ready = orchestration
        .receive_with_budget(Duration::from_secs(660))
        .await?;
    check_ready(&request, &ready)?;
    orchestration.sync()?;
    Ok((request, accepted, ready))
}

fn distinct_files(left: &Path, right: &Path) -> Result<()> {
    ensure!(
        left.canonicalize()? != right.canonicalize()?,
        "recovery evidence path aliases preceding evidence"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let a = fs::metadata(left)?;
        let b = fs::metadata(right)?;
        ensure!(
            (a.dev(), a.ino()) != (b.dev(), b.ino()),
            "recovery evidence hard-links preceding evidence"
        );
    }
    Ok(())
}

fn recovery_paths(state: &SchedulerState, ready: &Value) -> Result<(PathBuf, PathBuf)> {
    let handoff = policy::Handoff::read(&state.core.out.join("http_handoff.json"))?;
    let manifest = new_evidence_path(text(&ready["fields"], "manifest")?, &state.core, &handoff)?;
    let attestation = new_evidence_path(
        text(&ready["fields"], "attestation")?,
        &state.core,
        &handoff,
    )?;
    distinct_files(&manifest, &attestation)?;

    for candidate in [&manifest, &attestation] {
        for previous in [
            &state.launch.manifest,
            &state.launch.attestation,
            &state.core.initial_launch.manifest,
            &state.core.initial_launch.attestation,
            &state.core.loaded.policy_path,
            &handoff.prepass_input_path,
        ] {
            distinct_files(candidate, previous)?;
        }
        for previous in state
            .core
            .loaded
            .paths
            .values()
            .chain(state.core.loaded.bundle.paths.values())
        {
            distinct_files(candidate, previous)?;
        }
    }
    Ok((manifest, attestation))
}

async fn bind_ready(state: &SchedulerState, ready: &Value, out: &Path) -> Result<(Launch, Value)> {
    state.core.loaded.recheck()?;
    envelope(&state.core.loaded)?;
    let fields = &ready["fields"];
    let instance = text(fields, "instance")?.to_owned();
    let pid = u32::try_from(uint(fields, "pid")?)?;
    let start_utc = text(fields, "start_utc")?.to_owned();
    ensure!(
        instance != state.launch.instance
            && instance != state.core.initial_launch.instance
            && pid != state.launch.pid
            && pid != state.core.initial_launch.pid,
        "recovery reused a preceding external launch identity"
    );
    let (manifest, attestation) = recovery_paths(state, ready)?;
    let process = ProcessIdentity::capture(pid, &state.core.loaded.artifacts["serve_m3"].sha256)?;

    let challenge = core::random_label()?;
    let attested = state.core.control.call("attest", Some(&challenge)).await?;
    closed(
        &attested,
        &[
            "instance",
            "pid",
            "lifecycle",
            "worker_alive",
            "ready",
            "forward_calls",
            "startup_output_sha256",
            "attestation",
        ],
    )?;
    ensure!(
        attested["instance"] == instance
            && attested["pid"] == pid
            && attested["lifecycle"] == "ready"
            && attested["worker_alive"] == true
            && attested["ready"] == true
            && attested["forward_calls"] == 0,
        "recovery process is not fresh and ready"
    );

    let saved_bytes = bounded(&attestation, JSON_LIMIT)?;
    let saved: Value = serde_json::from_slice(&saved_bytes)?;
    ensure!(
        saved["schema"] == "rustv1.server_launch_attestation.v1"
            && saved["instance_id"] == instance
            && saved["pid"] == pid
            && saved["start_utc"] == start_utc
            && saved["settings_sha256"] == state.core.loaded.policy_sha256
            && saved["binary_sha256"] == state.core.loaded.artifacts["serve_m3"].sha256
            && saved["audit_sha256"] == state.core.loaded.artifacts["audit"].sha256
            && saved["startup_check_sha256"] == state.core.loaded.artifacts["startup_check"].sha256
            && saved["execution_policy"] == BATCHED_EXECUTION_POLICY
            && saved["http_contract_id"] == BATCHED_HTTP_CONTRACT
            && saved["batching"]
                == serde_json::to_value(&state.core.loaded.bundle.shapes.batching)?,
        "recovery saved attestation binding"
    );
    for name in [
        "rustc_version",
        "nvcc_version",
        "cuda_compute_cap",
        "target_features",
    ] {
        ensure!(
            !text(&saved["build_identity"], name)?.is_empty(),
            "empty recovery build identity"
        );
    }

    let runtime_attestation = &attested["attestation"];
    let mut reconstructed = runtime_attestation.clone();
    let object = reconstructed
        .as_object_mut()
        .context("runtime attestation object")?;
    for name in ["http_address", "control_address", "startup", "capabilities"] {
        ensure!(
            object.remove(name).is_some(),
            "missing runtime attestation field"
        );
    }
    ensure!(
        reconstructed == saved && serde_json::to_vec_pretty(&saved)? == saved_bytes,
        "recovery runtime/saved attestation mismatch"
    );

    let expected_http = core::base_address(&state.core.base_url)?;
    let actual_http: SocketAddr = text(runtime_attestation, "http_address")?.parse()?;
    let actual_control: SocketAddr = text(runtime_attestation, "control_address")?.parse()?;
    let expected_control: SocketAddr =
        text(&state.core.loaded.policy, "launch_control_address")?.parse()?;
    ensure!(
        actual_control == expected_control
            && actual_http.port() == expected_http.port()
            && (actual_http.ip() == expected_http.ip() || actual_http.ip().is_unspecified()),
        "recovery attested endpoint mismatch"
    );

    let initial_manifest = bounded(&manifest, MANIFEST_LIMIT)?;
    let startup = core::check_startup(
        &initial_manifest,
        &state.core.loaded,
        &sha(&saved_bytes),
        true,
    )?;
    ensure!(
        runtime_attestation["startup"]
            == json!({
                "completed": true,
                "comparison": "exact_f32_bits",
                "output_f32le_sha256": startup["startup_output_sha256"]
            })
            && attested["startup_output_sha256"] == startup["startup_output_sha256"],
        "recovery startup output binding"
    );

    let initial_observation = state.core.control.call("observe", None).await?;
    let mut plain = attested.clone();
    ensure!(
        plain
            .as_object_mut()
            .context("attested observation")?
            .remove("attestation")
            .is_some(),
        "missing authenticated attestation"
    );
    ensure!(
        initial_observation == plain
            && bounded(&manifest, MANIFEST_LIMIT)? == initial_manifest
            && bounded(&attestation, JSON_LIMIT)? == saved_bytes,
        "recovery activity/evidence mutation during reattestation"
    );
    process.check_live()?;

    write_new(
        &out.join("startup_executor_manifest.jsonl"),
        &initial_manifest,
    )?;
    write_new(&out.join("launch_attestation.json"), &saved_bytes)?;

    let binding = json!({
        "schema": "rustv1.accept_m4_recovery_launch_binding.v1",
        "authenticated_ready": ready,
        "challenge": challenge,
        "authenticated_observation": attested,
        "policy_sha256": state.core.loaded.policy_sha256,
        "attestation_sha256": sha(&saved_bytes),
        "startup": startup,
        "process_start_ticks": process.start_ticks,
        "boot_id": process.boot_id,
        "running_binary_sha256": process.executable_sha256,
        "launch_performed_by": "external_supervisor",
        "process_creation_after_old_exit_independently_measured": false
    });
    let binding_sha256 = save_json(&out.join("launch_binding.json"), &binding)?;

    Ok((
        Launch {
            instance,
            pid,
            manifest,
            attestation,
            start_utc,
            process,
            initial_manifest,
            initial_observation,
            startup,
            binding_sha256,
        },
        binding,
    ))
}

async fn execute(mut retained: PoisonExitState) -> Result<RecoveryReadyState> {
    let work: Result<(Launch, String, Vec<u8>, Vec<u8>)> = async {
        let consumed = consume_exit(&retained)?;
        let out = stage_directory(&retained.old, CASE)?;

        // This is an independent remeasurement before the authorization,
        // not an interpretation of the authenticated supervisor exit frame.
        let absence = retained.old.launch.process.prove_absent().await?;
        ensure!(
            absence == retained.old_absence,
            "old process absence identity changed"
        );
        let absence_bytes = serde_json::to_vec(&absence)?;
        write_new(&out.join("old_absence_rechecked.json"), &absence_bytes)?;

        // Bind inputs again after the potentially waiting absence proof.
        ensure!(
            consume_exit(&retained)? == consumed,
            "exit prerequisites changed while proving absence"
        );

        let nonce = core::random_label()?;
        let frames = verify_transcript(
            &retained.orchestration_prefix,
            &retained.old.core.loaded.policy,
        )?;
        ensure!(
            !frames.iter().any(|frame| {
                let fields = &frame["envelope"]["fields"];
                fields["launch_request_id"] == nonce || fields["restart_request_id"] == nonce
            }),
            "recovery authorization nonce collision"
        );

        let fields = json!({
            "launch_request_id": nonce,
            "reason": "after_poison_exit",
            "restart_request_id": retained.restart_request["fields"]["restart_request_id"],
            "restart_request_seq": retained.restart_request["seq"],
            "previous_instance": retained.old.launch.instance,
            "previous_pid": retained.old.launch.pid,
            "previous_process_start_ticks": retained.old.launch.process.start_ticks,
            "previous_boot_id": retained.old.launch.process.boot_id,
            "policy_sha256": retained.old.core.loaded.policy_sha256,
            "previous_launch_binding_sha256": retained.old.launch.binding_sha256,
            "exit_evidence_sha256": retained.exit_evidence_sha256,
            "old_absence_sha256": sha(&absence_bytes)
        });
        let (request, accepted, ready) =
            authorize(&mut retained.old.core.orchestration, fields).await?;

        save_json(&out.join("request.json"), &request)?;
        save_json(&out.join("accepted.json"), &accepted)?;
        save_json(&out.join("ready.json"), &ready)?;

        let transcript = transcript_extension(
            &retained.old,
            &retained.orchestration_prefix,
            &[
                json!({
                    "direction": "driver_to_orchestrator",
                    "envelope": request
                }),
                json!({
                    "direction": "orchestrator_to_driver",
                    "envelope": accepted
                }),
                json!({
                    "direction": "orchestrator_to_driver",
                    "envelope": ready
                }),
            ],
        )?;

        let (replacement, binding) = bind_ready(&retained.old, &ready, &out).await?;

        // Old process exit is not relabelled as an owner-thread join.
        // The old historical poisoned state is never overwritten.
        ensure!(
            retained.old.launch.process.prove_absent().await? == retained.old_absence
                && bounded(&retained.old.launch.manifest, MANIFEST_LIMIT)?
                    == retained.old.manifest_prefix,
            "old process/evidence changed during replacement launch"
        );
        for (case, digest) in [
            (RESTORE, retained.restore_evidence_sha256.as_str()),
            (RESTART, retained.restart_evidence_sha256.as_str()),
            (EXIT_CASE, retained.exit_evidence_sha256.as_str()),
        ] {
            read_bound(
                &retained.old.core.out,
                &format!("scheduler/{case}/evidence.json"),
                digest,
            )?;
        }
        retained.old.core.loaded.recheck()?;
        replacement.process.check_live()?;
        ensure!(
            retained.old.core.control.call("observe", None).await?
                == replacement.initial_observation
                && bounded(&replacement.manifest, MANIFEST_LIMIT)? == replacement.initial_manifest,
            "unreconciled replacement activity before case completion"
        );
        transcript_extension(&retained.old, &transcript, &[])?;
        ensure!(
            ledger_bytes(&retained.old)? == retained.ledger_prefix,
            "ledger changed during reattestation"
        );

        let evidence = json!({
            "schema": "rustv1.accept_m4_replacement_reattestation.v1",
            "case_id": CASE,
            "instance": replacement.instance,
            "pid": replacement.pid,
            "policy_sha256": retained.old.core.loaded.policy_sha256,
            "driver_binary_sha256":
                retained.old.core.loaded.artifacts["accept_m4"].sha256,
            "preceding_ledger_sha256": sha(&retained.ledger_prefix),
            "preceding_exit": consumed,
            "previous_instance": retained.old.launch.instance,
            "previous_pid": retained.old.launch.pid,
            "old_absence_rechecked": absence,
            "old_absence_artifact_sha256": sha(&absence_bytes),
            "request": request,
            "acceptance": accepted,
            "ready": ready,
            "launch_binding_path": format!("scheduler/{CASE}/launch_binding.json"),
            "launch_binding_sha256": replacement.binding_sha256,
            "launch_binding": binding,
            "orchestration_prefix_sha256": sha(&transcript),
            "initial_observation": replacement.initial_observation,
            "startup_manifest_sha256": sha(&replacement.initial_manifest),
            "next_forward_id": replacement.startup["next_forward_id"],
            "replacement_request_ids": [],
            "replacement_logical_requests": 0,
            "launch_authorization_after_independent_old_absence_proof": true,
            "process_creation_after_old_exit_independently_measured": false,
            "old_service_owner_join_claimed": false,
            "old_quarantine_cleared_in_process": false,
            "exact_http_recovery_executed": false
        });
        let digest = save_json(&out.join("evidence.json"), &evidence)?;

        pending(&retained.old, CASE)?;
        retained.old.core.ledger.append(
            policy::LIFECYCLE,
            CASE,
            true,
            json!({
                "evidence_path": format!("scheduler/{CASE}/evidence.json"),
                "evidence_sha256": digest,
                "instance": replacement.instance,
                "pid": replacement.pid,
                "previous_instance": retained.old.launch.instance,
                "previous_pid": retained.old.launch.pid
            }),
            true,
        )?;
        let ledger = ledger_bytes(&retained.old)?;
        ensure!(
            ledger.starts_with(&retained.ledger_prefix),
            "reattestation append rewrote the ledger prefix"
        );
        check_ledger_prefix(&retained.old, &ledger)?;
        pending(&retained.old, NEXT_CASE)?;
        Ok((replacement, digest, ledger, transcript))
    }
    .await;

    match work {
        Ok((replacement, digest, ledger, transcript)) => Ok(RecoveryReadyState {
            poison_exit: retained,
            replacement,
            reattestation_evidence_sha256: digest,
            ledger_prefix: ledger,
            orchestration_prefix: transcript,
        }),
        Err(error) => {
            fail(&mut retained.old, CASE, &error);
            Err(error)
        }
    }
}

pub(crate) fn run(continuation: PoisonExitContinuation) -> Result<RecoveryReadyContinuation> {
    let PoisonExitContinuation { runtime, state } = continuation;
    let directory = state.old.directory.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.block_on(execute(state))
    }));
    match result {
        Ok(result) => Ok(RecoveryReadyContinuation {
            runtime,
            state: result?,
        }),
        Err(payload) => {
            let _ = save_json(
                &directory.join("replacement_reattestation_panic.json"),
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
#[path = "accept_m4_lifecycle_reattestation_tests.rs"]
mod tests;
