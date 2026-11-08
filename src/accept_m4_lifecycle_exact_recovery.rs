use super::*;

const RECOVERY: &str = "family-58-exact-recovery";
const DRAIN_NEXT: &str = "final-authenticated-drain-and-exit";

/// Contains credentials indirectly. Do not derive Debug or Serialize.
pub struct ExactRecoveryContinuation {
    pub runtime: tokio::runtime::Runtime,
    pub state: ExactRecoveryState,
}

/// Separate recovery cursor; never overwrites poison_exit.old.
///
/// ready retains the original reattestation boundary and startup baseline.
/// The fields below describe only the replacement's completed HTTP interval.
pub struct ExactRecoveryState {
    pub ready: RecoveryReadyState,
    pub manifest_prefix: Vec<u8>,
    pub observation: Value,
    pub next_forward: u64,
    pub seen_request_ids: BTreeSet<u64>,
    pub recovery_evidence_sha256: String,
    pub ledger_prefix: Vec<u8>,
    pub orchestration_prefix: Vec<u8>,
}

/// Require exactly one appended ledger record, without reserializing the
/// preceding bytes. Used to consume and publish continuation boundaries.
fn one_record_extension(
    previous: &[u8],
    current: &[u8],
    phase: &str,
    case: &str,
    digest: &str,
) -> Result<Value> {
    ensure!(
        crate::contract::digest_valid(digest),
        "invalid boundary evidence digest"
    );
    let old = core::jsonl(previous)?;
    let records = core::jsonl(current)?;
    ensure!(
        current.starts_with(previous)
            && records.len() == old.len() + 1
            && prefix_before(current, old.len())? == previous,
        "boundary is not an exact one-record byte extension"
    );
    let record = records.last().context("missing boundary record")?;
    closed(
        record,
        &["acceptance_phase", "case_id", "status", "pass", "details"],
    )?;
    ensure!(
        record["acceptance_phase"] == phase
            && record["case_id"] == case
            && record["status"] == "executed"
            && record["pass"] == true
            && record["details"]["evidence_sha256"] == digest
            && record["details"]["evidence_path"] == format!("scheduler/{case}/evidence.json"),
        "boundary case/status/evidence binding"
    );
    Ok(record.clone())
}

/// Validate an append-only executor interval before slicing it.
///
/// Equal bytes are permitted here: the physical reconciliation below rejects
/// a successful request with no physical forward.
fn manifest_suffix<'a>(previous: &[u8], current: &'a [u8]) -> Result<&'a [u8]> {
    core::jsonl(previous)?;
    core::jsonl(current)?;
    ensure!(
        current.starts_with(previous),
        "replacement executor manifest rewritten or truncated"
    );
    Ok(&current[previous.len()..])
}

/// Exact F32 bits, not tolerance equality or JSON-number equality.
///
/// The digest is the pre-existing frozen fixture commitment, never a newly
/// established baseline from the recovered response.
fn exact_singleton(output: &[Vec<f32>], frozen: &[f32], frozen_digest: &str) -> Result<String> {
    ensure!(
        output.len() == 1 && crate::contract::digest_valid(frozen_digest),
        "exact recovery singleton cardinality/digest"
    );
    crate::contract::norm(frozen)?;
    crate::contract::norm(&output[0])?;
    ensure!(
        sha(&row_bytes(frozen)) == frozen_digest,
        "retained singleton differs from frozen fixture digest"
    );
    ensure!(
        output[0]
            .iter()
            .zip(frozen)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
        "recovery differs from frozen singleton F32 bits"
    );
    let digest = sha(&row_bytes(&output[0]));
    ensure!(
        digest == frozen_digest,
        "recovery singleton digest mismatch"
    );
    Ok(digest)
}

/// Consume immutable prerequisites at the reattestation boundary.
///
/// This function intentionally does not demand that the live replacement
/// still have its startup-only manifest. It is called again after recovery
/// to detect mutation of preceding commitments, while the new HTTP interval
/// is checked separately against its own cursor.
fn consume_ready(retained: &RecoveryReadyState) -> Result<Value> {
    let exit = &retained.poison_exit;
    let old = &exit.old;
    let launch = &retained.replacement;

    pending(old, RECOVERY)?;
    old.core.loaded.recheck()?;
    ensure!(
        ledger_bytes(old)? == retained.ledger_prefix,
        "retained reattestation ledger changed"
    );
    let records = check_ledger_prefix(old, &retained.ledger_prefix)?;
    let record = one_record_extension(
        &exit.ledger_prefix,
        &retained.ledger_prefix,
        policy::LIFECYCLE,
        super::CASE,
        &retained.reattestation_evidence_sha256,
    )?;
    ensure!(
        record["details"]["instance"] == launch.instance
            && record["details"]["pid"] == launch.pid
            && record["details"]["previous_instance"] == old.launch.instance
            && record["details"]["previous_pid"] == old.launch.pid,
        "reattestation ledger launch identity changed"
    );

    let relative = format!("scheduler/{}/evidence.json", super::CASE);
    let saved: Value = serde_json::from_slice(&read_bound(
        &old.core.out,
        &relative,
        &retained.reattestation_evidence_sha256,
    )?)?;
    ensure!(
        saved["schema"] == "rustv1.accept_m4_replacement_reattestation.v1"
            && saved["case_id"] == super::CASE
            && saved["instance"] == launch.instance
            && saved["pid"] == launch.pid
            && saved["previous_instance"] == old.launch.instance
            && saved["previous_pid"] == old.launch.pid
            && saved["policy_sha256"] == old.core.loaded.policy_sha256
            && saved["driver_binary_sha256"] == old.core.loaded.artifacts["accept_m4"].sha256
            && saved["preceding_ledger_sha256"] == sha(&exit.ledger_prefix)
            && saved["orchestration_prefix_sha256"] == sha(&retained.orchestration_prefix)
            && saved["initial_observation"] == launch.initial_observation
            && saved["startup_manifest_sha256"] == sha(&launch.initial_manifest)
            && saved["next_forward_id"] == launch.startup["next_forward_id"]
            && saved["replacement_request_ids"] == json!([])
            && saved["replacement_logical_requests"] == 0
            && saved["launch_authorization_after_independent_old_absence_proof"] == true
            && saved["process_creation_after_old_exit_independently_measured"] == false
            && saved["old_service_owner_join_claimed"] == false
            && saved["old_quarantine_cleared_in_process"] == false
            && saved["exact_http_recovery_executed"] == false,
        "reattestation evidence/cursor semantics changed"
    );
    ensure!(
        ready_observation(&launch.initial_observation, launch)? == 0,
        "recovery baseline is not a fresh ready observation"
    );

    // The old lifecycle evidence still belongs to the old launch.
    let (old_exit, _) = preceding_case(old, &records, EXIT_CASE)?;
    let (restart, _) = preceding_case(old, &records, RESTART)?;
    let (restore, _) = preceding_case(old, &records, RESTORE)?;
    for (case, expected) in [
        (EXIT_CASE, exit.exit_evidence_sha256.as_str()),
        (RESTART, exit.restart_evidence_sha256.as_str()),
        (RESTORE, exit.restore_evidence_sha256.as_str()),
    ] {
        let position = records
            .iter()
            .position(|r| r["acceptance_phase"] == policy::LIFECYCLE && r["case_id"] == case)
            .context("missing old lifecycle record")?;
        ensure!(
            records[position]["details"]["evidence_sha256"] == expected,
            "old retained evidence digest changed"
        );
        let evidence: Value = serde_json::from_slice(&read_bound(
            &old.core.out,
            &format!("scheduler/{case}/evidence.json"),
            expected,
        )?)?;
        ensure!(
            evidence["preceding_ledger_sha256"]
                == sha(prefix_before(&retained.ledger_prefix, position)?),
            "old lifecycle preceding-byte chain changed"
        );
    }
    historical_cursor(
        &old_exit,
        &old.observation,
        &old.manifest_prefix,
        old.next_forward,
        &old.seen_request_ids,
    )?;
    ensure!(
        old_exit["request"] == exit.restart_request
            && old_exit["acceptance"] == exit.restart_accepted
            && old_exit["authenticated_exit"] == exit.restart_exit
            && old_exit["independent_process_absence"] == exit.old_absence
            && old_exit["restart_evidence_sha256"] == exit.restart_evidence_sha256
            && old_exit["restore_evidence_sha256"] == exit.restore_evidence_sha256
            && restart["request"] == exit.restart_request
            && restart["acceptance"] == exit.restart_accepted
            && restart["old_pre_restart_observation"] == old.observation
            && restart["old_manifest_sha256"] == sha(&old.manifest_prefix)
            && restore["old_observation_preserved"] == old.observation
            && restore["manifest_unchanged_sha256"] == sha(&old.manifest_prefix)
            && restore["executor_next_forward_id_unchanged"] == old.next_forward
            && restore["exposed_request_ids_unchanged"] == json!(old.seen_request_ids),
        "old restart/exit/historical cursor changed"
    );
    check_accepted(&exit.restart_request, &exit.restart_accepted)?;
    check_exit(&exit.restart_request, &exit.restart_exit)?;

    let consumed = &saved["preceding_exit"];
    ensure!(
        consumed["historical_observation"] == old.observation
            && consumed["historical_manifest_sha256"] == sha(&old.manifest_prefix)
            && consumed["historical_next_forward_id"] == old.next_forward
            && consumed["historical_request_ids"] == json!(old.seen_request_ids)
            && consumed["consumed_exit_evidence_sha256"] == exit.exit_evidence_sha256
            && consumed["consumed_ledger_sha256"] == sha(&exit.ledger_prefix)
            && consumed["consumed_orchestration_sha256"] == sha(&exit.orchestration_prefix)
            && saved["old_absence_rechecked"] == exit.old_absence,
        "reattestation did not consume the retained old exit"
    );

    let out = old.directory.join(super::CASE);
    let request = &saved["request"];
    let accepted = &saved["acceptance"];
    let ready = &saved["ready"];
    check_launch_accepted(request, accepted)?;
    check_ready(request, ready)?;
    ensure!(
        request["fields"]["restart_request_id"]
            == exit.restart_request["fields"]["restart_request_id"]
            && request["fields"]["restart_request_seq"] == exit.restart_request["seq"]
            && request["fields"]["previous_instance"] == old.launch.instance
            && request["fields"]["previous_pid"] == old.launch.pid
            && request["fields"]["previous_process_start_ticks"] == old.launch.process.start_ticks
            && request["fields"]["previous_boot_id"] == old.launch.process.boot_id
            && request["fields"]["previous_launch_binding_sha256"] == old.launch.binding_sha256
            && request["fields"]["policy_sha256"] == old.core.loaded.policy_sha256
            && request["fields"]["exit_evidence_sha256"] == exit.exit_evidence_sha256
            && request["fields"]["old_absence_sha256"] == saved["old_absence_artifact_sha256"],
        "recovery authorization changed target or prerequisites"
    );
    ensure!(
        ready["fields"]["instance"] == launch.instance
            && ready["fields"]["pid"] == launch.pid
            && ready["fields"]["start_utc"] == launch.start_utc
            && Path::new(text(&ready["fields"], "manifest")?).canonicalize()? == launch.manifest
            && Path::new(text(&ready["fields"], "attestation")?).canonicalize()?
                == launch.attestation
            && value(&out.join("request.json"))? == *request
            && value(&out.join("accepted.json"))? == *accepted
            && value(&out.join("ready.json"))? == *ready,
        "saved recovery exchange or retained Launch changed"
    );

    let binding_path = format!("scheduler/{}/launch_binding.json", super::CASE);
    ensure!(
        saved["launch_binding_path"] == binding_path
            && saved["launch_binding_sha256"] == launch.binding_sha256,
        "recovery launch binding commitment changed"
    );
    let binding: Value = serde_json::from_slice(&read_bound(
        &old.core.out,
        &binding_path,
        &launch.binding_sha256,
    )?)?;
    ensure!(
        binding == saved["launch_binding"]
            && binding["schema"] == "rustv1.accept_m4_recovery_launch_binding.v1"
            && binding["authenticated_ready"] == *ready
            && binding["policy_sha256"] == old.core.loaded.policy_sha256
            && binding["process_start_ticks"] == launch.process.start_ticks
            && binding["boot_id"] == launch.process.boot_id
            && binding["running_binary_sha256"] == launch.process.executable_sha256
            && launch.process.pid == launch.pid
            && launch.process.executable_sha256 == old.core.loaded.artifacts["serve_m3"].sha256
            && binding["startup"] == launch.startup
            && binding["launch_performed_by"] == "external_supervisor"
            && binding["process_creation_after_old_exit_independently_measured"] == false,
        "recovery launch binding identity changed"
    );
    ensure!(
        crate::contract::digest_valid(text(&binding, "challenge")?),
        "invalid saved attestation challenge"
    );
    let attested = &binding["authenticated_observation"];
    let mut plain = attested.clone();
    ensure!(
        plain
            .as_object_mut()
            .context("saved authenticated observation")?
            .remove("attestation")
            .is_some()
            && plain == launch.initial_observation,
        "saved authenticated startup observation changed"
    );

    let attestation = bounded(&launch.attestation, JSON_LIMIT)?;
    let attestation_digest = text(&binding, "attestation_sha256")?;
    ensure!(
        sha(&attestation) == attestation_digest
            && bounded(&out.join("launch_attestation.json"), JSON_LIMIT)? == attestation
            && bounded(&out.join("startup_executor_manifest.jsonl"), MANIFEST_LIMIT)?
                == launch.initial_manifest
            && core::check_startup(
                &launch.initial_manifest,
                &old.core.loaded,
                attestation_digest,
                true,
            )? == launch.startup,
        "recovery startup/attestation artifacts changed"
    );

    let mut reconstructed = attested["attestation"].clone();
    let object = reconstructed
        .as_object_mut()
        .context("saved runtime attestation")?;
    for name in ["http_address", "control_address", "startup", "capabilities"] {
        ensure!(
            object.remove(name).is_some(),
            "missing runtime attestation field"
        );
    }
    ensure!(
        serde_json::to_vec_pretty(&reconstructed)? == attestation,
        "saved runtime/base attestation mismatch"
    );

    let absence_bytes = bounded(&out.join("old_absence_rechecked.json"), JSON_LIMIT)?;
    ensure!(
        sha(&absence_bytes) == text(&saved, "old_absence_artifact_sha256")?
            && absence_bytes == serde_json::to_vec(&exit.old_absence)?
            && bounded(&old.launch.manifest, MANIFEST_LIMIT)? == old.manifest_prefix
            && bounded(
                &old.directory
                    .join(EXIT_CASE)
                    .join("executor_manifest_snapshot.jsonl"),
                MANIFEST_LIMIT,
            )? == old.manifest_prefix
            && file_sha(&old.directory.join("launch_binding.json"))? == old.launch.binding_sha256,
        "preceding absence/old-manifest artifacts changed"
    );

    let transcript = transcript_extension(
        old,
        &exit.orchestration_prefix,
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
    ensure!(
        transcript == retained.orchestration_prefix,
        "retained recovery transcript changed"
    );

    Ok(saved)
}

async fn live_cursor(
    retained: &RecoveryReadyState,
    observation: &Value,
    manifest: &[u8],
) -> Result<()> {
    let old = &retained.poison_exit.old;
    retained.replacement.process.check_live()?;
    ready_observation(observation, &retained.replacement)?;
    ensure!(
        old.core.control.call("observe", None).await? == *observation
            && bounded(&retained.replacement.manifest, MANIFEST_LIMIT)? == manifest,
        "unreconciled replacement activity"
    );
    Ok(())
}

async fn execute(mut retained: RecoveryReadyState) -> Result<ExactRecoveryState> {
    #[allow(clippy::type_complexity)]
    let work: Result<(Vec<u8>, Value, u64, BTreeSet<u64>, String, Vec<u8>)> = async {
        let preceding = consume_ready(&retained)?;
        let old = &retained.poison_exit.old;
        let launch = &retained.replacement;
        let out = stage_directory(old, RECOVERY)?;

        ensure!(
            old.launch.process.prove_absent().await? == retained.poison_exit.old_absence,
            "old process absence changed before recovery"
        );
        live_cursor(
            &retained,
            &launch.initial_observation,
            &launch.initial_manifest,
        )
        .await?;

        let client = Http::new(
            &old.core.base_url,
            &launch.instance,
            serde_json::from_value(old.core.loaded.policy["limits"].clone())?,
        )?;
        let bundle = &old.core.loaded.bundle;
        ensure!(
            bundle.fixtures.len() == 22
                && bundle.singleton.len() == 22
                && bundle.reference.len() == 22,
            "frozen recovery fixture coverage"
        );

        let mut manifest = launch.initial_manifest.clone();
        let mut observation = launch.initial_observation.clone();
        let mut next_forward = uint(&launch.startup, "next_forward_id")?;
        let mut request_ids = BTreeSet::new();
        let mut last_request_id = None;
        let mut requests = Vec::new();
        let mut expected_connections = Vec::new();

        // Each request is awaited and reconciled before the next submission.
        // Thus each interval must contain exactly one physical B=1 forward.
        for fixture in 0..22 {
            let mut previous_output: Option<Vec<Vec<f32>>> = None;
            for encoding in ["float", "base64"] {
                live_cursor(&retained, &observation, &manifest).await?;
                let case = format!("sched58-recovery-{fixture}-{encoding}");
                let body = json!({
                    "model": MODEL_ID,
                    "input": [bundle.fixtures[fixture].text],
                    "encoding_format": encoding
                });
                let request_bytes = serde_json::to_vec(&body)?;
                write_new(&out.join(format!("{case}.request.bin")), &request_bytes)?;

                let reply = client.send(&case, &Wire::json(body)?).await?;
                let response = save_reply(&out, &case, &reply)?;
                let id = reply
                    .request_id
                    .context("recovery success missing request ID")?;
                ensure!(
                    id > 0
                        && last_request_id.is_none_or(|previous| id > previous)
                        && request_ids.insert(id),
                    "recovery reused or reordered an exposed request ID"
                );

                let (output, usage) = http::decode_success(&reply, bundle, &[fixture], encoding)?;
                let output_digest = exact_singleton(
                    &output,
                    &bundle.singleton[fixture],
                    &bundle.fixtures[fixture].golden_sha256,
                )?;
                if let Some(previous) = &previous_output {
                    crate::contract::exact(&output, previous)?;
                }

                let after = old.core.control.call("observe", None).await?;
                ready_observation(&after, launch)?;
                logical_delta(&observation, &after, 1)?;
                let bytes = bounded(&launch.manifest, MANIFEST_LIMIT)?;
                let suffix = manifest_suffix(&manifest, &bytes)?;
                ensure!(
                    core::jsonl(suffix)?.len() == 2,
                    "serialized singleton recovery must add exactly one physical pair"
                );

                let completed = BTreeMap::from([(
                    id,
                    CompletedRequest {
                        case: case.clone(),
                        fixtures: vec![fixture],
                        output: output.clone(),
                    },
                )]);
                let reconciled = reconcile_wave(suffix, next_forward, &completed, bundle)?;
                ensure!(
                    reconciled.forwards.len() == 1
                        && reconciled.forwards[0]["B"] == 1
                        && reconciled.forwards[0]["distinct_request_ids"] == json!([id])
                        && reconciled.witnesses.is_empty()
                        && reconciled.next_forward
                            == next_forward.checked_add(1).context("forward overflow")?,
                    "recovery physical interval cardinality"
                );
                let interval_name = format!("{case}.executor_interval.jsonl");
                write_new(&out.join(&interval_name), suffix)?;

                let connection = &reply.headers["_connection_owner"];
                ensure!(
                    connection["instance"] == launch.instance
                        && connection["owner_pid"] == std::process::id()
                        && connection["case_id"] == case
                        && connection["method"] == "POST"
                        && connection["path"] == "/v1/embeddings"
                        && connection["request_sha256"] == sha(&request_bytes)
                        && connection["join_awaited"] == true,
                    "recovery HTTP connection ownership/request binding"
                );
                expected_connections.push(connection.clone());
                let proof = json!({
                    "case_id": case,
                    "fixture_index": fixture,
                    "fixture_id": bundle.fixtures[fixture].id,
                    "encoding": encoding,
                    "request_id": id,
                    "request_path": format!("{case}.request.bin"),
                    "request_sha256": sha(&request_bytes),
                    "response": response,
                    "usage": usage,
                    "output_f32le_sha256": output_digest,
                    "frozen_singleton_sha256": bundle.fixtures[fixture].golden_sha256,
                    "exact_f32_bits": true,
                    "prior_float_output_checked_exactly": previous_output.is_some(),
                    "observation_before": observation,
                    "observation_after": after,
                    "logical_backend_request_delta": 1,
                    "manifest_interval": {
                        "path": interval_name,
                        "start_byte": manifest.len(),
                        "end_byte": bytes.len(),
                        "prefix_before_sha256": sha(&manifest),
                        "prefix_after_sha256": sha(&bytes),
                        "interval_sha256": sha(suffix)
                    },
                    "physical_forwards": reconciled.forwards,
                    "unmatched_physical_records": 0
                });
                save_json(&out.join(format!("{case}.evidence.json")), &proof)?;
                requests.push(proof);

                previous_output = Some(output);
                manifest = bytes;
                observation = after;
                next_forward = reconciled.next_forward;
                last_request_id = Some(id);
            }
        }

        ensure!(
            requests.len() == 44
                && request_ids.len() == 44
                && next_forward
                    == uint(&launch.startup, "next_forward_id")?
                        .checked_add(44)
                        .context("recovery total forward overflow")?,
            "recovery request/physical coverage"
        );
        logical_delta(&launch.initial_observation, &observation, 44)?;

        let connections = http::join_connections(&launch.instance).await?;
        check_connections(&launch.instance, &expected_connections, &connections)?;
        save_json(&out.join("http_connections.json"), &json!(connections))?;
        write_new(&out.join("executor_manifest_snapshot.jsonl"), &manifest)?;
        save_json(&out.join("final_observation.json"), &observation)?;

        // Recheck all preceding immutable commitments after HTTP execution.
        ensure!(
            consume_ready(&retained)? == preceding,
            "preceding reattestation evidence changed during recovery"
        );
        ensure!(
            old.launch.process.prove_absent().await? == retained.poison_exit.old_absence,
            "old process absence changed during recovery"
        );
        live_cursor(&retained, &observation, &manifest).await?;

        // Read saved request/response/interval files through the commitments
        // made at each actual execution, rather than hashing them as new truth.
        for proof in &requests {
            let case = text(proof, "case_id")?;
            ensure!(
                file_sha(&under(&out, text(proof, "request_path")?)?)?
                    == text(proof, "request_sha256")?
                    && value(&out.join(format!("{case}.evidence.json")))? == *proof,
                "saved recovery request/proof changed"
            );
            check_saved_reply(&out, &proof["response"])?;
            let interval = &proof["manifest_interval"];
            let start = usize::try_from(uint(interval, "start_byte")?)?;
            let end = usize::try_from(uint(interval, "end_byte")?)?;
            ensure!(
                start <= end && end <= manifest.len(),
                "saved recovery interval bounds"
            );
            let interval_bytes = bounded(&under(&out, text(interval, "path")?)?, MANIFEST_LIMIT)?;
            ensure!(
                interval_bytes == manifest[start..end]
                    && sha(&interval_bytes) == text(interval, "interval_sha256")?
                    && sha(&manifest[..start]) == text(interval, "prefix_before_sha256")?
                    && sha(&manifest[..end]) == text(interval, "prefix_after_sha256")?,
                "saved recovery executor interval changed"
            );
        }
        ensure!(
            bounded(
                &out.join("executor_manifest_snapshot.jsonl"),
                MANIFEST_LIMIT
            )? == manifest
                && value(&out.join("final_observation.json"))? == observation
                && value(&out.join("http_connections.json"))? == json!(connections),
            "saved recovery closing artifacts changed"
        );

        let evidence = json!({
            "schema": "rustv1.accept_m4_exact_recovery.v1",
            "case_id": RECOVERY,
            "instance": launch.instance,
            "pid": launch.pid,
            "policy_sha256": old.core.loaded.policy_sha256,
            "driver_binary_sha256": old.core.loaded.artifacts["accept_m4"].sha256,
            "preceding_ledger_sha256": sha(&retained.ledger_prefix),
            "reattestation_evidence_path":
                format!("scheduler/{}/evidence.json", super::CASE),
            "reattestation_evidence_sha256": retained.reattestation_evidence_sha256,
            "launch_binding_sha256": launch.binding_sha256,
            "orchestration_prefix_sha256": sha(&retained.orchestration_prefix),
            "initial_observation": launch.initial_observation,
            "startup_manifest_sha256": sha(&launch.initial_manifest),
            "requests": requests,
            "http_connections": connections,
            "final_observation": observation,
            "final_manifest_sha256": sha(&manifest),
            "final_manifest_bytes": manifest.len(),
            "next_forward_id": next_forward,
            "replacement_request_ids": request_ids,
            "replacement_logical_requests": 44,
            "reconciled_recovery_physical_forwards": 44,
            "fixture_indices": (0usize..22).collect::<Vec<_>>(),
            "encodings_per_fixture": ["float", "base64"],
            "serialized_physical_b1_exact_f32_bits": true,
            "unmatched_physical_records": 0,
            "old_historical_observation": old.observation,
            "old_manifest_unchanged_sha256": sha(&old.manifest_prefix),
            "old_next_forward_id_unchanged": old.next_forward,
            "old_request_ids_unchanged": old.seen_request_ids,
            "old_exit_evidence_sha256": retained.poison_exit.exit_evidence_sha256,
            "old_service_owner_join_claimed": false,
            "old_quarantine_cleared_in_process": false,
            "replacement_left_ready": true,
            "final_drain_executed": false,
            "lifecycle_phase_complete": false,
            "m4_accepted": false
        });
        let digest = save_json(&out.join("evidence.json"), &evidence)?;
        old.core.loaded.recheck()?;
        live_cursor(&retained, &observation, &manifest).await?;
        transcript_extension(old, &retained.orchestration_prefix, &[])?;
        ensure!(
            ledger_bytes(old)? == retained.ledger_prefix,
            "ledger changed during recovery"
        );

        // Append with the replacement identity, not the old poisoned launch.
        pending(&retained.poison_exit.old, RECOVERY)?;
        retained.poison_exit.old.core.ledger.append(
            policy::LIFECYCLE,
            RECOVERY,
            true,
            json!({
                "evidence_path": format!("scheduler/{RECOVERY}/evidence.json"),
                "evidence_sha256": digest,
                "instance": retained.replacement.instance,
                "pid": retained.replacement.pid,
                "logical_requests": 44,
                "physical_forwards": 44,
                "unmatched_physical_records": 0
            }),
            true,
        )?;
        let ledger = ledger_bytes(&retained.poison_exit.old)?;
        one_record_extension(
            &retained.ledger_prefix,
            &ledger,
            policy::LIFECYCLE,
            RECOVERY,
            &digest,
        )?;
        check_ledger_prefix(&retained.poison_exit.old, &ledger)?;
        pending(&retained.poison_exit.old, DRAIN_NEXT)?;

        Ok((
            manifest,
            observation,
            next_forward,
            request_ids,
            digest,
            ledger,
        ))
    }
    .await;

    match work {
        Ok((manifest, observation, next_forward, request_ids, digest, ledger)) => {
            let transcript = retained.orchestration_prefix.clone();
            Ok(ExactRecoveryState {
                ready: retained,
                manifest_prefix: manifest,
                observation,
                next_forward,
                seen_request_ids: request_ids,
                recovery_evidence_sha256: digest,
                ledger_prefix: ledger,
                orchestration_prefix: transcript,
            })
        }
        Err(error) => {
            fail(&mut retained.poison_exit.old, RECOVERY, &error);
            Err(error)
        }
    }
}

pub(crate) fn run(continuation: RecoveryReadyContinuation) -> Result<ExactRecoveryContinuation> {
    let RecoveryReadyContinuation { runtime, state } = continuation;
    let directory = state.poison_exit.old.directory.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.block_on(execute(state))
    }));
    match result {
        Ok(result) => Ok(ExactRecoveryContinuation {
            runtime,
            state: result?,
        }),
        Err(payload) => {
            let _ = save_json(
                &directory.join("exact_recovery_panic.json"),
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
#[path = "accept_m4_lifecycle_exact_recovery_tests.rs"]
mod tests;
