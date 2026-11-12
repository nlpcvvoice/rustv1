// Included at acceptance_scheduler::saved_initial.
//
// Offline reconciliation of the initial external launch and HTTP marker.
//
// No process observation, control request, HTTP request, owner join,
// orchestration exchange, or ledger append.
//
// Important historical limitation:
// core::http_phase did not retain the producer-returned HTTP marker digest.
// The marker is checked against reconstructed producer-format contents.
// Its newly calculated inventory digest is NOT an original commitment.
//
// The HTTP producer's independent challenge likewise has no separately
// retained historical commitment. Its syntax and saved binding semantics
// are checked without claiming to authenticate its original value.

use super::*;
use std::path::Component;

const BOUNDARY: &str = "INITIAL-LAUNCH-AND-HTTP-MARKER-RECONCILED";
const HTTP_MARKER: &str = "acceptance_http_manifest.json";

fn safe_relative(relative: &str) -> Result<()> {
    let path = Path::new(relative);
    ensure!(
        !relative.is_empty()
            && path
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "unsafe saved-initial artifact path"
    );
    Ok(())
}

fn regular_directory(path: &Path) -> Result<()> {
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_dir(),
        "saved-initial directory is not a regular non-symlink directory"
    );
    Ok(())
}

/// Check each named path component, not only the final file.
fn output_path(root: &Path, relative: &str) -> Result<PathBuf> {
    safe_relative(relative)?;
    regular_directory(root)?;
    let mut path = root.to_owned();
    let parts: Vec<_> = Path::new(relative).components().collect();

    for (position, part) in parts.iter().enumerate() {
        let Component::Normal(name) = part else {
            anyhow::bail!("invalid saved-initial path component");
        };
        path.push(name);
        let kind = fs::symlink_metadata(&path)?.file_type();
        ensure!(
            if position + 1 == parts.len() {
                kind.is_file()
            } else {
                kind.is_dir()
            },
            "symlink or unexpected saved-initial artifact type"
        );
    }
    Ok(path)
}

fn read_output(
    root: &Path,
    relative: &str,
    limit: usize,
    inventory: &mut BTreeMap<String, String>,
) -> Result<Vec<u8>> {
    let bytes = bounded(&output_path(root, relative)?, limit)?;
    let digest = sha(&bytes);
    if let Some(previous) = inventory.get(relative) {
        ensure!(
            *previous == digest,
            "saved-initial artifact changed between reads"
        );
    } else {
        inventory.insert(relative.to_owned(), digest);
    }
    Ok(bytes)
}

fn read_committed(
    root: &Path,
    relative: &str,
    digest: &str,
    limit: usize,
    inventory: &mut BTreeMap<String, String>,
) -> Result<Vec<u8>> {
    ensure!(
        crate::contract::digest_valid(digest),
        "invalid saved-initial commitment"
    );
    let bytes = read_output(root, relative, limit, inventory)?;
    ensure!(
        sha(&bytes) == digest,
        "saved-initial artifact commitment mismatch: {relative}"
    );
    Ok(bytes)
}

fn expect_bytes(
    root: &Path,
    relative: &str,
    expected: &[u8],
    limit: usize,
    inventory: &mut BTreeMap<String, String>,
) -> Result<()> {
    ensure!(
        read_output(root, relative, limit, inventory)? == expected,
        "saved-initial reconstructed bytes differ: {relative}"
    );
    Ok(())
}

fn expect_pretty(
    root: &Path,
    relative: &str,
    expected: &Value,
    inventory: &mut BTreeMap<String, String>,
) -> Result<()> {
    expect_bytes(
        root,
        relative,
        &serde_json::to_vec_pretty(expected)?,
        JSON_LIMIT,
        inventory,
    )
}

fn phase_rows<'a>(records: &'a [Value], phase: &str) -> Vec<&'a Value> {
    records
        .iter()
        .filter(|record| record["acceptance_phase"] == phase)
        .collect()
}

fn required_ids(loaded: &Loaded, phase: &str) -> Result<Vec<String>> {
    let matches: Vec<_> = loaded
        .required
        .phases
        .iter()
        .filter(|(name, _)| name == phase)
        .collect();
    ensure!(matches.len() == 1, "missing/duplicate required phase");
    Ok(matches[0].1.clone())
}

fn regular_external_bytes(path: &Path, limit: usize) -> Result<Vec<u8>> {
    // Launch paths were canonicalized at the live binding boundary.
    ensure!(
        path.is_absolute()
            && path.canonicalize()? == path
            && fs::symlink_metadata(path)?.file_type().is_file(),
        "external artifact path/type changed"
    );
    bounded(path, limit)
}

/// Exact historical absence record produced by ProcessIdentity::prove_absent.
///
/// This reconstructs the record's expected identity fields. It does not
/// perform a new /proc observation or assert current absence.
fn absence_record(process: &ProcessIdentity) -> Value {
    json!({
        "method": "linux-proc-pid-directory-absent",
        "pid": process.pid,
        "previous_start_ticks": process.start_ticks,
        "boot_id": process.boot_id,
        "previous_executable_sha256": process.executable_sha256,
        "old_process_absent": true,
        "zombie_counted_as_exit": false
    })
}

/// The initial protocol historically allows additional orch_ready/och_exit
/// fields. Do not impose a later, closed replacement-launch schema on it.
///
/// Exact envelopes are nevertheless bound by both the saved transcript and
/// the already-appended ledger rows.
fn initial_frames(
    frames: &[Value],
    ready: &Value,
    shutdown: &Value,
    instance: &str,
    pid: u32,
) -> Result<()> {
    ensure!(frames.len() >= 3, "missing initial orchestration exchange");

    closed(shutdown, &["request", "acknowledgement", "instance", "pid"])?;
    let request = &shutdown["request"];
    let exit = &shutdown["acknowledgement"];

    ensure!(
        ready["kind"] == "orch_ready"
            && ready["fields"]["instance"] == instance
            && ready["fields"]["pid"] == pid
            && request["kind"] == "orch_request_drain"
            && request["fields"] == json!({"instance": instance})
            && exit["kind"] == "orch_exit"
            && exit["fields"]["instance"] == instance
            && exit["fields"]["exit_status"] == 0
            && shutdown["instance"] == instance
            && shutdown["pid"] == pid,
        "initial ready/drain/exit semantics"
    );
    if let Some(reported_pid) = exit["fields"].get("pid") {
        ensure!(*reported_pid == pid, "initial exit PID changed");
    }

    let expected = [
        json!({
            "direction": "orchestrator_to_driver",
            "envelope": ready
        }),
        json!({
            "direction": "driver_to_orchestrator",
            "envelope": request
        }),
        json!({
            "direction": "orchestrator_to_driver",
            "envelope": exit
        }),
    ];
    ensure!(
        frames[..3] == expected,
        "initial transcript is not the exact expected three-frame prefix"
    );
    Ok(())
}

/// Compare against the exact serialization used by the producer.
///
/// The caller must construct expected from independently checked state,
/// rather than copying arbitrary marker fields into it.
fn exact_marker(bytes: &[u8], expected: &Value) -> Result<()> {
    ensure!(
        serde_json::from_slice::<Value>(bytes)? == *expected
            && bytes == serde_json::to_vec_pretty(expected)?,
        "HTTP completion marker differs from reconstructed producer contents"
    );
    Ok(())
}

fn http_directory_coverage(
    root: &Path,
    outputs: &BTreeMap<String, String>,
) -> Result<()> {
    regular_directory(root)?;
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        ensure!(
            entry.file_type()?.is_file(),
            "HTTP output contains a symlink or non-file"
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 HTTP output name"))?;
        ensure!(actual.insert(name), "duplicate HTTP directory entry");
    }
    let expected: BTreeSet<String> = outputs
        .keys()
        .cloned()
        .chain(std::iter::once(HTTP_MARKER.to_owned()))
        .collect();
    ensure!(actual == expected, "HTTP completion output coverage changed");
    Ok(())
}

fn http_marker(
    continuation: &ImportedArtifactsContinuation,
    initial_binding: &Value,
    inventory: &mut BTreeMap<String, String>,
) -> Result<Value> {
    let state = &continuation.local_artifacts.final_drain.state;
    let core = &state.recovery.ready.poison_exit.old.core;
    let loaded = &core.loaded;
    let launch = &core.initial_launch;
    let out = &core.out;
    let summary = &continuation.import_reconciliation["http"];

    // Handoff has no retained original byte digest in CoreState.
    // Reconstruct its exact producer format after checking its semantic
    // binding to the retained launch and policy-loaded input.
    let handoff_bytes = read_output(
        out,
        "http_handoff.json",
        JSON_LIMIT,
        inventory,
    )?;
    let handoff: policy::Handoff = serde_json::from_slice(&handoff_bytes)?;
    ensure!(
        handoff.schema == "rustv1.accept_m4_handoff.v1"
            && handoff.instance == launch.instance
            && handoff.pid == launch.pid
            && handoff.model_root == loaded.bundle.model_dir
            && handoff.prepass_input_sha256 == loaded.bundle.input_sha256
            && handoff.prepass_input_path.is_absolute()
            && handoff.prepass_input_path.canonicalize()?
                == handoff.prepass_input_path
            && fs::symlink_metadata(&handoff.prepass_input_path)?
                .file_type().is_file()
            && file_sha(&handoff.prepass_input_path)?
                == loaded.bundle.input_sha256
            && handoff.live_manifest_path == launch.manifest,
        "saved HTTP handoff semantic binding"
    );
    let expected_handoff = json!({
        "schema": "rustv1.accept_m4_handoff.v1",
        "model_root": loaded.bundle.model_dir,
        "prepass_input_path": handoff.prepass_input_path,
        "prepass_input_sha256": loaded.bundle.input_sha256,
        "live_manifest_path": launch.manifest,
        "instance": launch.instance,
        "pid": launch.pid
    });
    ensure!(
        handoff_bytes == serde_json::to_vec_pretty(&expected_handoff)?,
        "HTTP handoff differs from producer serialization"
    );

    let binding_bytes = read_output(
        out,
        "http/launch_binding.json",
        JSON_LIMIT,
        inventory,
    )?;
    let binding: Value = serde_json::from_slice(&binding_bytes)?;
    let challenge = text(&binding, "challenge")?;
    ensure!(
        crate::contract::digest_valid(challenge),
        "invalid saved HTTP challenge"
    );

    // Both attestations were taken before any logical traffic. The runtime
    // launch attestation and plain observation must therefore be identical.
    // The independent challenge itself is NOT historically authenticated
    // here: the original completion-marker digest was discarded.
    let expected_binding = json!({
        "acceptance_phase": policy::HTTP,
        "challenge": challenge,
        "observation": initial_binding["authenticated_observation"],
        "handoff_sha256": sha(&handoff_bytes),
        "policy_sha256": loaded.policy_sha256,
        "prepass_input_sha256": loaded.bundle.input_sha256,
        "initial_manifest_bytes": launch.initial_manifest.len(),
        "initial_manifest_sha256": sha(&launch.initial_manifest),
        "initial_next_forward_id": launch.startup["next_forward_id"]
    });
    ensure!(
        binding == expected_binding
            && binding_bytes == serde_json::to_vec_pretty(&expected_binding)?,
        "HTTP saved launch binding differs from retained initial launch"
    );

    let source_bytes = read_output(
        out,
        "http/accept_records.jsonl",
        MANIFEST_LIMIT,
        inventory,
    )?;
    ensure!(
        sha(&source_bytes)
            == text(&continuation.import_reconciliation, "http_source_records_sha256")?,
        "HTTP source changed after imported reconciliation"
    );
    let source = core::jsonl(&source_bytes)?;
    let final_observation = source
        .last()
        .context("missing HTTP producer records")?["details"]["observation_after"]
        .clone();

    let final_manifest = read_output(
        out,
        "http/executor_manifest_snapshot.jsonl",
        MANIFEST_LIMIT,
        inventory,
    )?;
    let successful = uint(summary, "successful_requests")?;
    let rejected = uint(summary, "rejected_requests")?;
    ensure!(
        successful > 0
            && rejected > 0
            && successful.checked_add(rejected)
                == Some(u64::try_from(source.len())?)
            && summary["final_manifest_sha256"] == sha(&final_manifest)
            && summary["final_manifest_bytes"] == final_manifest.len()
            && summary["serving_physical_forwards"] == successful,
        "HTTP reconstructed completion counters"
    );

    // Build the output table from artifacts already checked against imported
    // producer evidence, plus the semantically reconstructed launch binding.
    // Never use the marker's own output table as the trust source.
    let imported_inventory = continuation.import_reconciliation["verified_output_sha256"]
        .as_object()
        .context("imported output inventory")?;
    let mut outputs = BTreeMap::new();
    for (relative, digest) in imported_inventory {
        let Some(name) = relative.strip_prefix("http/") else {
            continue;
        };
        ensure!(
            Path::new(name).components().count() == 1
                && name != HTTP_MARKER
                && name != "launch_binding.json",
            "unexpected imported HTTP inventory path"
        );
        let expected_digest = digest.as_str().context("imported digest")?;
        read_committed(
            out,
            relative,
            expected_digest,
            MANIFEST_LIMIT,
            inventory,
        )?;
        ensure!(
            outputs.insert(name.to_owned(), expected_digest.to_owned()).is_none(),
            "duplicate HTTP output inventory"
        );
    }
    ensure!(
        outputs
            .insert("launch_binding.json".to_owned(), sha(&binding_bytes))
            .is_none(),
        "duplicate HTTP launch binding"
    );
    http_directory_coverage(&out.join("http"), &outputs)?;

    let required = required_ids(loaded, policy::HTTP)?;
    ensure!(
        source.iter().map(|record| text(record, "case_id"))
            .collect::<Result<Vec<_>>>()?
            == required.iter().map(String::as_str).collect::<Vec<_>>(),
        "HTTP marker required/completed order"
    );
    let artifact_hashes: BTreeMap<&str, &str> = loaded
        .bundle
        .artifacts
        .iter()
        .map(|(role, artifact)| (role.as_str(), artifact.sha256.as_str()))
        .collect();

    let expected = json!({
        "schema": "rustv1.accept_m4_http_manifest.v1",
        "acceptance_phase": policy::HTTP,
        "scope": "boundary_item_3_live_batched_http_reconciliation",
        "producer": {
            "component": "accept_m4",
            "command": "acceptance-http",
            "binary_sha256": loaded.artifacts["accept_m4"].sha256
        },
        "handoff_sha256": sha(&handoff_bytes),
        "policy_sha256": loaded.policy_sha256,
        "prepass_input_sha256": loaded.bundle.input_sha256,
        "artifact_sha256": artifact_hashes,
        "instance": launch.instance,
        "pid": launch.pid,
        "execution_policy": BATCHED_EXECUTION_POLICY,
        "http_contract_id": BATCHED_HTTP_CONTRACT,
        "live_manifest_path": launch.manifest,
        "initial_manifest_bytes": launch.initial_manifest.len(),
        "initial_manifest_sha256": sha(&launch.initial_manifest),
        "final_manifest_bytes": final_manifest.len(),
        "final_manifest_sha256": sha(&final_manifest),
        "initial_observation": launch.initial_observation,
        "final_observation": final_observation,
        "required_case_ids": required,
        "completed_case_ids": required,
        "pending_required_cases_in_scope": [],
        "exceptions": [],
        "successful_http_requests": successful,
        "rejected_http_requests": rejected,
        "logical_backend_requests": successful,
        "reconciled_physical_forwards": successful,
        "unmatched_physical_records": 0,
        "gates": {
            "handoff_instance_and_pid_verified": true,
            "prepass_loader_reused_unchanged": true,
            "fresh_control_challenge_verified": true,
            "launch_attestation_manifest_binding": true,
            "live_manifest_grew": true,
            "append_only_manifest_prefixes": true,
            "response_schema_usage_encoding": true,
            "server_request_ids_unique_and_monotonic": true,
            "request_observation_interval_physical_row_association": true,
            "rejections_zero_execution": true,
            "closed_admission_request_id_is_null": true,
            "complete_item_3_ledger": true
        },
        "output_sha256": outputs,
        "server_left_quiesced": true,
        "independent_mode_regression_completed": false,
        "scheduler_families_53_through_58_completed": false,
        "physical_capacity_gate_proven": false,
        "pass": true,
        "m4_accepted": false
    });

    let marker_bytes = read_output(
        out,
        &format!("http/{HTTP_MARKER}"),
        JSON_LIMIT,
        inventory,
    )?;
    exact_marker(&marker_bytes, &expected)?;

    Ok(json!({
        "path": format!("http/{HTTP_MARKER}"),
        "inventory_sha256": sha(&marker_bytes),
        "validation": "reconstructed_producer_format_and_bound_artifact_semantics",
        "original_producer_returned_digest_retained": false,
        "original_marker_byte_commitment_authenticated": false,
        "new_digest_described_as_historical_commitment": false,
        "producer_fresh_challenge_gate_preserved_as_historical_claim": true,
        "http_challenge_original_value_independently_authenticated": false,
        "handoff_original_path_spelling_independently_authenticated": false,
        "handoff_input_content_policy_bound": true,
        "required_case_count": source.len(),
        "output_files_checked": outputs.len(),
        "m4_accepted": false
    }))
}

pub(super) fn reconcile(
    continuation: &ImportedArtifactsContinuation,
) -> Result<Value> {
    ensure!(
        saved_imports::reconcile(&continuation.local_artifacts)?
            == continuation.import_reconciliation,
        "preceding imported-artifact reconciliation changed"
    );

    let state = &continuation.local_artifacts.final_drain.state;
    let core = &state.recovery.ready.poison_exit.old.core;
    let loaded = &core.loaded;
    let launch = &core.initial_launch;
    let out = &core.out;
    loaded.recheck()?;

    let ledger = bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)?;
    ensure!(ledger == state.ledger_prefix, "initial reconciliation ledger changed");
    let records = core::jsonl(&ledger)?;
    let mut inventory = BTreeMap::new();

    let binding_bytes = read_committed(
        out,
        "initial_launch_binding.json",
        &launch.binding_sha256,
        JSON_LIMIT,
        &mut inventory,
    )?;
    let binding: Value = serde_json::from_slice(&binding_bytes)?;
    closed(
        &binding,
        &[
            "challenge",
            "authenticated_ready",
            "authenticated_observation",
            "policy_sha256",
            "attestation_sha256",
            "startup",
            "process_start_ticks",
            "boot_id",
        ],
    )?;
    ensure!(
        crate::contract::digest_valid(text(&binding, "challenge")?)
            && binding["policy_sha256"] == loaded.policy_sha256
            && binding["startup"] == launch.startup
            && binding["process_start_ticks"] == launch.process.start_ticks
            && binding["boot_id"] == launch.process.boot_id
            && launch.process.pid == launch.pid
            && launch.process.executable_sha256 == loaded.artifacts["serve_m3"].sha256,
        "initial launch binding/retained process identity"
    );

    let ready = &binding["authenticated_ready"];
    ensure!(
        ready["kind"] == "orch_ready"
            && ready["fields"]["instance"] == launch.instance
            && ready["fields"]["pid"] == launch.pid
            && ready["fields"]["start_utc"] == launch.start_utc
            && !launch.start_utc.is_empty()
            && launch.start_utc.len() <= 80
            && Path::new(text(&ready["fields"], "manifest")?).canonicalize()?
                == launch.manifest
            && Path::new(text(&ready["fields"], "attestation")?).canonicalize()?
                == launch.attestation,
        "initial authenticated ready/Launch association"
    );

    let launch_rows = phase_rows(&records, policy::LAUNCH);
    ensure!(
        launch_rows.len() == 3
            && launch_rows[0]["case_id"] == "authenticated-orchestration-ready"
            && launch_rows[0]["details"] == json!({
                "authenticated_envelope": ready,
                "instance": launch.instance,
                "pid": launch.pid,
                "process_start_ticks": launch.process.start_ticks,
                "boot_id": launch.process.boot_id,
                "running_binary_sha256": launch.process.executable_sha256
            })
            && launch_rows[1]["case_id"] == "fresh-launch-attestation"
            && launch_rows[1]["details"] == json!({
                "launch_binding_sha256": launch.binding_sha256
            })
            && launch_rows[2]["case_id"] == "startup-manifest-reconciliation"
            && launch_rows[2]["details"] == launch.startup,
        "initial launch ledger binding"
    );

    let attestation_bytes = regular_external_bytes(&launch.attestation, JSON_LIMIT)?;
    ensure!(
        sha(&attestation_bytes) == text(&binding, "attestation_sha256")?,
        "initial external attestation changed"
    );
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
        "initial external attestation policy binding"
    );
    for key in [
        "rustc_version",
        "nvcc_version",
        "cuda_compute_cap",
        "target_features",
    ] {
        ensure!(
            !text(&attestation["build_identity"], key)?.is_empty(),
            "empty initial build identity"
        );
    }

    let attested = &binding["authenticated_observation"];
    let mut plain = attested.clone();
    ensure!(
        plain.as_object_mut()
            .context("initial authenticated observation")?
            .remove("attestation")
            .is_some()
            && plain == launch.initial_observation
            && ready_observation(&plain, launch)? == 0,
        "initial authenticated observation changed"
    );
    let runtime_attestation = &attested["attestation"];
    let mut reconstructed = runtime_attestation.clone();
    let object = reconstructed
        .as_object_mut()
        .context("initial runtime attestation")?;
    for key in ["http_address", "control_address", "startup", "capabilities"] {
        ensure!(
            object.remove(key).is_some(),
            "missing initial runtime attestation field"
        );
    }
    ensure!(
        reconstructed == attestation
            && serde_json::to_vec_pretty(&reconstructed)? == attestation_bytes,
        "initial runtime/base attestation mismatch"
    );

    let expected_http = core::base_address(&core.base_url)?;
    let actual_http: SocketAddr = text(runtime_attestation, "http_address")?.parse()?;
    let actual_control: SocketAddr = text(runtime_attestation, "control_address")?.parse()?;
    let expected_control: SocketAddr =
        text(&loaded.policy, "launch_control_address")?.parse()?;
    ensure!(
        actual_control == expected_control
            && actual_http.port() == expected_http.port()
            && (actual_http.ip() == expected_http.ip() || actual_http.ip().is_unspecified()),
        "initial saved endpoint association"
    );

    ensure!(
        core::check_startup(
            &launch.initial_manifest,
            loaded,
            text(&binding, "attestation_sha256")?,
            true,
        )? == launch.startup
            && runtime_attestation["startup"] == json!({
                "completed": true,
                "comparison": "exact_f32_bits",
                "output_f32le_sha256": launch.startup["startup_output_sha256"]
            })
            && launch.initial_observation["startup_output_sha256"]
                == launch.startup["startup_output_sha256"],
        "initial startup reconciliation changed"
    );
    expect_bytes(
        out,
        "initial_executor_manifest.jsonl",
        &launch.initial_manifest,
        MANIFEST_LIMIT,
        &mut inventory,
    )?;

    closed(
        &core.initial_exit,
        &["shutdown", "process_exit", "final_manifest_sha256"],
    )?;
    let absence = absence_record(&launch.process);
    ensure!(
        core.initial_exit["process_exit"] == absence,
        "initial historical absence record changed"
    );
    let exit_rows = phase_rows(&records, policy::EXIT);
    ensure!(
        exit_rows.len() == 2
            && exit_rows[0]["case_id"] == "authenticated-supervised-shutdown"
            && exit_rows[0]["details"] == core.initial_exit["shutdown"]
            && exit_rows[1]["case_id"] == "old-process-exit-proven"
            && exit_rows[1]["details"] == absence,
        "initial exit ledger binding"
    );
    expect_pretty(
        out,
        "initial_exit.json",
        &core.initial_exit,
        &mut inventory,
    )?;

    let transcript = bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)?;
    ensure!(
        transcript == state.orchestration_prefix,
        "saved final transcript changed"
    );
    let frames = lifecycle_prefix::verify_transcript(&transcript, &loaded.policy)?;
    initial_frames(
        &frames,
        ready,
        &core.initial_exit["shutdown"],
        &launch.instance,
        launch.pid,
    )?;

    let final_manifest = read_committed(
        out,
        "http/executor_manifest_snapshot.jsonl",
        text(&continuation.import_reconciliation["http"], "final_manifest_sha256")?,
        MANIFEST_LIMIT,
        &mut inventory,
    )?;
    ensure!(
        final_manifest.starts_with(&launch.initial_manifest)
            && core.initial_exit["final_manifest_sha256"] == sha(&final_manifest)
            && regular_external_bytes(&launch.manifest, MANIFEST_LIMIT)? == final_manifest,
        "initial external final manifest differs from reconciled HTTP interval"
    );
    expect_bytes(
        out,
        "batched_final_executor_manifest.jsonl",
        &final_manifest,
        MANIFEST_LIMIT,
        &mut inventory,
    )?;

    let marker = http_marker(continuation, &binding, &mut inventory)?;

    // Stability checks do not establish new historical baselines.
    for (relative, digest) in &inventory {
        ensure!(
            file_sha(&output_path(out, relative)?)? == *digest,
            "initial/HTTP artifact changed during reconciliation"
        );
    }
    ensure!(
        regular_external_bytes(&launch.attestation, JSON_LIMIT)? == attestation_bytes
            && regular_external_bytes(&launch.manifest, MANIFEST_LIMIT)? == final_manifest,
        "initial external artifact changed during reconciliation"
    );
    loaded.recheck()?;
    ensure!(
        bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)? == ledger
            && bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)? == transcript,
        "ledger/transcript changed during initial reconciliation"
    );

    Ok(json!({
        "schema": "rustv1.accept_m4_initial_launch_reconciliation.v1",
        "boundary": BOUNDARY,
        "policy_sha256": loaded.policy_sha256,
        "preceding_ledger_sha256": sha(&ledger),
        "preceding_import_reconciliation_sha256":
            sha(&serde_json::to_vec(&continuation.import_reconciliation)?),
        "orchestration_sha256": sha(&transcript),
        "authenticated_transcript_frames_checked": frames.len(),
        "initial_transcript_semantic_positions": [0, 1, 2],
        "whole_transcript_semantics_reconciled": false,
        "initial_launch": {
            "instance": launch.instance,
            "pid": launch.pid,
            "launch_binding_sha256": launch.binding_sha256,
            "attestation_sha256": sha(&attestation_bytes),
            "startup_manifest_sha256": sha(&launch.initial_manifest),
            "final_manifest_sha256": sha(&final_manifest),
            "final_manifest_bytes": final_manifest.len(),
            "startup_and_http_interval_coverage_complete": true,
            "historical_absence": absence,
            "historical_observations_relabelled": false,
            "new_process_absence_measurement": false,
            "supervisor_reap_independently_awaited_by_driver": false,
            "service_owner_join_claimed": false
        },
        "http_completion_marker": marker,
        "http_completion_marker_semantics_reconciled": true,
        "http_completion_marker_original_digest_authenticated": false,
        "verified_output_sha256": inventory,
        "all_external_launches_reconciled": false,
        "destructive_children_reconciled": false,
        "historical_poisoned_launch_reconciled": false,
        "independent_mode_artifacts_reconciled": false,
        "post_exit_control_calls": 0,
        "process_observations": 0,
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
    fn http_marker_rejects_changed_claims_extra_fields_and_reserialization() {
        // Disposable format fixtures, not execution evidence.
        let expected = json!({
            "schema": "test-marker",
            "output_sha256": {"records.jsonl": sha(b"original\n")},
            "pass": true,
            "m4_accepted": false
        });
        let original = serde_json::to_vec_pretty(&expected).unwrap();
        assert!(exact_marker(&original, &expected).is_ok());

        let mut changed = expected.clone();
        changed["m4_accepted"] = json!(true);
        assert!(exact_marker(
            &serde_json::to_vec_pretty(&changed).unwrap(),
            &expected,
        ).is_err());

        let mut rebased = expected.clone();
        rebased["output_sha256"]["records.jsonl"] = json!(sha(b"changed\n"));
        assert!(exact_marker(
            &serde_json::to_vec_pretty(&rebased).unwrap(),
            &expected,
        ).is_err());

        let mut extra = expected.clone();
        extra["additional_claim"] = json!(true);
        assert!(exact_marker(
            &serde_json::to_vec_pretty(&extra).unwrap(),
            &expected,
        ).is_err());

        assert!(exact_marker(
            &serde_json::to_vec(&expected).unwrap(),
            &expected,
        ).is_err());
    }

    #[test]
    fn initial_exchange_requires_exact_prefix_and_preserves_optional_exit_pid() {
        // MAC verification is performed by verify_transcript in the caller.
        // These fixtures exercise only the initial semantic-prefix validator.
        let ready = json!({
            "kind": "orch_ready",
            "seq": 0,
            "fields": {"instance": "initial", "pid": 42},
            "mac": "fixture"
        });
        let request = json!({
            "kind": "orch_request_drain",
            "seq": 0,
            "fields": {"instance": "initial"},
            "mac": "fixture"
        });
        let exit = json!({
            "kind": "orch_exit",
            "seq": 1,
            "fields": {"instance": "initial", "exit_status": 0},
            "mac": "fixture"
        });
        let shutdown = json!({
            "request": request,
            "acknowledgement": exit,
            "instance": "initial",
            "pid": 42
        });
        let frames = vec![
            json!({"direction": "orchestrator_to_driver", "envelope": ready}),
            json!({"direction": "driver_to_orchestrator", "envelope": request}),
            json!({"direction": "orchestrator_to_driver", "envelope": exit}),
        ];
        assert!(initial_frames(&frames, &ready, &shutdown, "initial", 42).is_ok());

        let mut reordered = frames.clone();
        reordered.swap(0, 1);
        assert!(initial_frames(&reordered, &ready, &shutdown, "initial", 42).is_err());
        assert!(initial_frames(&frames[..2], &ready, &shutdown, "initial", 42).is_err());

        let mut with_pid = shutdown.clone();
        with_pid["acknowledgement"]["fields"]["pid"] = json!(42);
        let mut pid_frames = frames.clone();
        pid_frames[2]["envelope"] = with_pid["acknowledgement"].clone();
        assert!(initial_frames(
            &pid_frames,
            &ready,
            &with_pid,
            "initial",
            42,
        ).is_ok());

        with_pid["acknowledgement"]["fields"]["pid"] = json!(43);
        pid_frames[2]["envelope"] = with_pid["acknowledgement"].clone();
        assert!(initial_frames(
            &pid_frames,
            &ready,
            &with_pid,
            "initial",
            42,
        ).is_err());

        let mut relabelled = shutdown;
        relabelled["acknowledgement"]["fields"]["exit_status"] = json!(1);
        let mut changed_frames = frames;
        changed_frames[2]["envelope"] = relabelled["acknowledgement"].clone();
        assert!(initial_frames(
            &changed_frames,
            &ready,
            &relabelled,
            "initial",
            42,
        ).is_err());
    }

    #[test]
    fn saved_initial_paths_reject_absolute_and_traversal_components() {
        assert!(safe_relative("http/launch_binding.json").is_ok());
        assert!(safe_relative("initial_exit.json").is_ok());
        assert!(safe_relative("").is_err());
        assert!(safe_relative("/tmp/evidence.json").is_err());
        assert!(safe_relative("../evidence.json").is_err());
        assert!(safe_relative("http/../evidence.json").is_err());
    }
}
