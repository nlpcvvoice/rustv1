// Included at:
// acceptance_scheduler::lifecycle_prefix::restart_prefix::reattestation::
// exact_recovery::final_drain::saved_lineage
//
// Offline recovery-launch lineage and startup reconciliation.
//
// Reuses pure validators only. In particular, do not call consume_exit,
// consume_ready, consume_recovery, transcript_extension, live_cursor,
// check_live, prove_absent, or any control/orchestration exchange here.
//
// This is NOT complete manifest/artifact reconciliation.

use super::*;
use std::path::Component;

const LINEAGE_BOUNDARY: &str = "RECOVERY-LAUNCH-LINEAGE-RECONCILED";
const RESTORE_ID: &str = "family-58-restore-does-not-recover";
const RESTART_ID: &str = "family-58-authenticated-supervised-restart";
const EXIT_ID: &str = "family-58-old-process-exit";
const READY_ID: &str = "family-58-replacement-reattestation";
const RECOVERY_ID: &str = "family-58-exact-recovery";
const DRAIN_ID: &str = "final-authenticated-drain-and-exit";
const FINAL_ID: &str = "final-artifact-and-manifest-reconciliation";

struct Files<'a> {
    root: &'a Path,
    hashes: BTreeMap<String, String>,
}

impl<'a> Files<'a> {
    fn new(root: &'a Path) -> Result<Self> {
        ensure!(
            fs::symlink_metadata(root)?.file_type().is_dir(),
            "lineage output root must be a non-symlink directory"
        );
        Ok(Self {
            root,
            hashes: BTreeMap::new(),
        })
    }

    fn path(&self, relative: &str) -> Result<PathBuf> {
        let parts: Vec<_> = Path::new(relative).components().collect();
        ensure!(
            !parts.is_empty()
                && parts.iter().all(|part| matches!(part, Component::Normal(_))),
            "unsafe lineage artifact path"
        );
        let mut path = self.root.to_owned();
        for (index, part) in parts.iter().enumerate() {
            let Component::Normal(name) = part else {
                anyhow::bail!("invalid lineage path component");
            };
            path.push(name);
            let kind = fs::symlink_metadata(&path)?.file_type();
            ensure!(
                if index + 1 == parts.len() {
                    kind.is_file()
                } else {
                    kind.is_dir()
                },
                "symlink or unexpected lineage artifact type"
            );
        }
        Ok(path)
    }

    fn read(&mut self, relative: &str, limit: usize) -> Result<Vec<u8>> {
        let bytes = bounded(&self.path(relative)?, limit)?;
        let digest = sha(&bytes);
        if let Some(previous) = self.hashes.get(relative) {
            ensure!(*previous == digest, "lineage artifact changed between reads");
        } else {
            self.hashes.insert(relative.to_owned(), digest);
        }
        Ok(bytes)
    }

    fn committed(&mut self, relative: &str, digest: &str) -> Result<Vec<u8>> {
        ensure!(
            crate::contract::digest_valid(digest),
            "invalid lineage artifact commitment"
        );
        let bytes = self.read(relative, JSON_LIMIT)?;
        ensure!(sha(&bytes) == digest, "lineage commitment mismatch: {relative}");
        Ok(bytes)
    }

    fn exact(&mut self, relative: &str, expected: &[u8]) -> Result<()> {
        ensure!(
            self.read(relative, MANIFEST_LIMIT)? == expected,
            "lineage artifact bytes differ: {relative}"
        );
        Ok(())
    }

    fn pretty(&mut self, relative: &str, expected: &Value) -> Result<()> {
        self.exact(relative, &serde_json::to_vec_pretty(expected)?)
    }

    fn stable(&self) -> Result<()> {
        for (relative, digest) in &self.hashes {
            ensure!(
                file_sha(&self.path(relative)?)? == *digest,
                "lineage artifact changed during reconciliation"
            );
        }
        Ok(())
    }
}

fn external(path: &Path, limit: usize) -> Result<Vec<u8>> {
    ensure!(
        path.is_absolute()
            && path.canonicalize()? == path
            && fs::symlink_metadata(path)?.file_type().is_file(),
        "lineage external artifact path/type changed"
    );
    bounded(path, limit)
}

/// Reconstruct the format of a historical measurement, without measuring
/// process state again.
fn historical_absence(process: &ProcessIdentity) -> Value {
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

/// Both inputs must have already passed verify_transcript.
///
/// Require an exact semantic extension, including no extra frames. Byte-prefix
/// checks are performed separately against the retained transcript bytes.
fn exact_extension(
    previous: &[Value],
    current: &[Value],
    suffix: &[Value],
) -> Result<()> {
    ensure!(
        current.len() == previous.len() + suffix.len()
            && current.starts_with(previous)
            && current[previous.len()..] == *suffix,
        "unexpected authenticated transcript extension"
    );
    Ok(())
}

fn frame(direction: &str, envelope: &Value) -> Value {
    json!({"direction": direction, "envelope": envelope})
}

fn read_case(
    files: &mut Files<'_>,
    ledger: &[u8],
    case: &str,
    digest: &str,
    loaded: &Loaded,
    launch: &Launch,
) -> Result<Value> {
    let records = core::jsonl(ledger)?;
    let matches: Vec<_> = records.iter().enumerate()
        .filter(|(_, record)| {
            record["acceptance_phase"] == policy::LIFECYCLE
                && record["case_id"] == case
        })
        .collect();
    ensure!(matches.len() == 1, "missing/duplicate lineage case");
    let (index, record) = matches[0];
    let relative = format!("scheduler/{case}/evidence.json");
    ensure!(
        record["status"] == "executed"
            && record["pass"] == true
            && record["details"]["evidence_path"] == relative
            && record["details"]["evidence_sha256"] == digest
            && record["details"]["instance"] == launch.instance
            && record["details"]["pid"] == launch.pid,
        "lineage ledger commitment/identity"
    );
    let bytes = files.committed(&relative, digest)?;
    let saved: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        saved["case_id"] == case
            && saved["instance"] == launch.instance
            && saved["pid"] == launch.pid
            && saved["policy_sha256"] == loaded.policy_sha256
            && saved["driver_binary_sha256"] == loaded.artifacts["accept_m4"].sha256
            && saved["preceding_ledger_sha256"]
                == sha(prefix_before(ledger, index)?),
        "lineage evidence identity or preceding ledger bytes"
    );
    Ok(saved)
}

fn unique_nonce(frames: &[Value], nonce: &str) -> Result<()> {
    ensure!(
        crate::contract::digest_valid(nonce)
            && !frames.iter().any(|frame| {
                frame["envelope"]["fields"].as_object().is_some_and(|fields| {
                    fields.values().any(|value| value.as_str() == Some(nonce))
                })
            }),
        "lineage authorization nonce reused"
    );
    Ok(())
}

pub(crate) fn reconcile(
    continuation: &InitialLaunchArtifactsContinuation,
) -> Result<Value> {
    ensure!(
        crate::acceptance_scheduler::saved_initial::reconcile(
            &continuation.imported_artifacts,
        )? == continuation.initial_launch_reconciliation,
        "preceding initial-launch reconciliation changed"
    );

    let state = &continuation.imported_artifacts.local_artifacts.final_drain.state;
    let recovery = &state.recovery;
    let ready = &recovery.ready;
    let poison = &ready.poison_exit;
    let old = &poison.old;
    let core = &old.core;
    let loaded = &core.loaded;
    let launch = &ready.replacement;
    let out = &core.out;

    loaded.recheck()?;
    ensure!(
        core.ledger.pending() == vec![json!({
            "acceptance_phase": policy::LIFECYCLE,
            "case_id": FINAL_ID
        })],
        "lineage reconciliation called at wrong ledger boundary"
    );

    let ledger = bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)?;
    let transcript = bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)?;
    ensure!(
        ledger == state.ledger_prefix
            && transcript == state.orchestration_prefix,
        "final saved ledger/transcript changed"
    );

    // Retained boundaries must be exact byte extensions, not reserialized
    // arrays or newly selected baselines.
    one_record_extension(
        &poison.ledger_prefix,
        &ready.ledger_prefix,
        policy::LIFECYCLE,
        READY_ID,
        &ready.reattestation_evidence_sha256,
    )?;
    one_record_extension(
        &ready.ledger_prefix,
        &recovery.ledger_prefix,
        policy::LIFECYCLE,
        RECOVERY_ID,
        &recovery.recovery_evidence_sha256,
    )?;
    one_record_extension(
        &recovery.ledger_prefix,
        &state.ledger_prefix,
        policy::LIFECYCLE,
        DRAIN_ID,
        &state.drain_evidence_sha256,
    )?;

    let mut files = Files::new(out)?;
    let restore = read_case(
        &mut files, &ledger, RESTORE_ID,
        &poison.restore_evidence_sha256, loaded, &old.launch,
    )?;
    let restart = read_case(
        &mut files, &ledger, RESTART_ID,
        &poison.restart_evidence_sha256, loaded, &old.launch,
    )?;
    let exited = read_case(
        &mut files, &ledger, EXIT_ID,
        &poison.exit_evidence_sha256, loaded, &old.launch,
    )?;
    let reattested = read_case(
        &mut files, &ledger, READY_ID,
        &ready.reattestation_evidence_sha256, loaded, launch,
    )?;
    let recovered = read_case(
        &mut files, &ledger, RECOVERY_ID,
        &recovery.recovery_evidence_sha256, loaded, launch,
    )?;
    let drained = read_case(
        &mut files, &ledger, DRAIN_ID,
        &state.drain_evidence_sha256, loaded, launch,
    )?;

    // Preserve the old poisoned observation exactly. worker_alive=true is a
    // historical observation, not a claim about the exited process now.
    historical_cursor(
        &exited,
        &old.observation,
        &old.manifest_prefix,
        old.next_forward,
        &old.seen_request_ids,
    )?;
    ensure!(
        poison.old_absence == historical_absence(&old.launch.process)
            && restore["old_observation_preserved"] == old.observation
            && restore["manifest_unchanged_sha256"] == sha(&old.manifest_prefix)
            && restore["executor_next_forward_id_unchanged"] == old.next_forward
            && restore["exposed_request_ids_unchanged"] == json!(old.seen_request_ids)
            && restore["owner_join_claimed"] == false
            && restore["process_exit_claimed"] == false
            && restart["old_pre_restart_observation"] == old.observation
            && restart["old_manifest_sha256"] == sha(&old.manifest_prefix)
            && restart["owner_join_claimed"] == false
            && exited["independent_process_absence"] == poison.old_absence
            && exited["supervisor_reap_independently_awaited_by_driver"] == false
            && exited["containment_boundary"]
                == "external_process_exit_and_supervisor_reap",
        "historical poisoned lineage changed or was relabelled"
    );

    check_accepted(&poison.restart_request, &poison.restart_accepted)?;
    check_exit(&poison.restart_request, &poison.restart_exit)?;
    let request = &poison.restart_request["fields"];
    ensure!(
        request["instance"] == old.launch.instance
            && request["pid"] == old.launch.pid
            && request["process_start_ticks"] == old.launch.process.start_ticks
            && request["boot_id"] == old.launch.process.boot_id
            && request["running_binary_sha256"] == old.launch.process.executable_sha256
            && request["policy_sha256"] == loaded.policy_sha256
            && request["launch_binding_sha256"] == old.launch.binding_sha256
            && request["restore_evidence_sha256"] == poison.restore_evidence_sha256
            && request["manifest_sha256"] == sha(&old.manifest_prefix)
            && restart["request"] == poison.restart_request
            && restart["acceptance"] == poison.restart_accepted
            && restart["restore_evidence_sha256"] == poison.restore_evidence_sha256
            && exited["request"] == poison.restart_request
            && exited["acceptance"] == poison.restart_accepted
            && exited["authenticated_exit"] == poison.restart_exit
            && exited["restart_evidence_sha256"] == poison.restart_evidence_sha256
            && exited["restore_evidence_sha256"] == poison.restore_evidence_sha256,
        "restart target or saved exchange changed"
    );

    for (relative, expected) in [
        (format!("scheduler/{RESTART_ID}/request.json"), &poison.restart_request),
        (format!("scheduler/{RESTART_ID}/accepted.json"), &poison.restart_accepted),
        (format!("scheduler/{EXIT_ID}/authenticated_exit.json"), &poison.restart_exit),
        (format!("scheduler/{EXIT_ID}/process_absence.json"), &poison.old_absence),
    ] {
        files.pretty(&relative, expected)?;
    }
    files.exact(
        &format!("scheduler/{EXIT_ID}/executor_manifest_snapshot.jsonl"),
        &old.manifest_prefix,
    )?;

    let old_frames = verify_transcript(&poison.orchestration_prefix, &loaded.policy)?;
    let ready_frames = verify_transcript(&ready.orchestration_prefix, &loaded.policy)?;
    let final_frames = verify_transcript(&transcript, &loaded.policy)?;
    ensure!(
        old_frames.len() == 9
            && ready.orchestration_prefix.starts_with(&poison.orchestration_prefix)
            && recovery.orchestration_prefix == ready.orchestration_prefix
            && transcript.starts_with(&recovery.orchestration_prefix),
        "retained orchestration byte lineage"
    );
    exact_extension(
        &old_frames[..6],
        &old_frames,
        &[
            frame("driver_to_orchestrator", &poison.restart_request),
            frame("orchestrator_to_driver", &poison.restart_accepted),
            frame("orchestrator_to_driver", &poison.restart_exit),
        ],
    )?;
    ensure!(
        restart["orchestration_prefix_sha256"]
            == sha(prefix_before(&poison.orchestration_prefix, 8)?)
            && exited["orchestration_prefix_sha256"] == sha(&poison.orchestration_prefix),
        "restart/exit transcript byte commitments changed"
    );

    let request = &reattested["request"];
    let accepted = &reattested["acceptance"];
    let ready_envelope = &reattested["ready"];
    check_launch_accepted(request, accepted)?;
    check_ready(request, ready_envelope)?;
    unique_nonce(&old_frames, text(&request["fields"], "launch_request_id")?)?;

    let absence_bytes = serde_json::to_vec(&poison.old_absence)?;
    let fields = &request["fields"];
    ensure!(
        fields["restart_request_id"]
            == poison.restart_request["fields"]["restart_request_id"]
            && fields["restart_request_seq"] == poison.restart_request["seq"]
            && fields["previous_instance"] == old.launch.instance
            && fields["previous_pid"] == old.launch.pid
            && fields["previous_process_start_ticks"] == old.launch.process.start_ticks
            && fields["previous_boot_id"] == old.launch.process.boot_id
            && fields["policy_sha256"] == loaded.policy_sha256
            && fields["previous_launch_binding_sha256"] == old.launch.binding_sha256
            && fields["exit_evidence_sha256"] == poison.exit_evidence_sha256
            && fields["old_absence_sha256"] == sha(&absence_bytes)
            && reattested["old_absence_rechecked"] == poison.old_absence
            && reattested["old_absence_artifact_sha256"] == sha(&absence_bytes)
            && reattested["previous_instance"] == old.launch.instance
            && reattested["previous_pid"] == old.launch.pid
            && reattested["launch_authorization_after_independent_old_absence_proof"] == true
            && reattested["process_creation_after_old_exit_independently_measured"] == false
            && reattested["old_service_owner_join_claimed"] == false
            && reattested["old_quarantine_cleared_in_process"] == false,
        "recovery authorization lineage changed"
    );

    let consumed = &reattested["preceding_exit"];
    let expected_sources: Vec<Value> = [
        (RESTORE_ID, poison.restore_evidence_sha256.as_str()),
        (RESTART_ID, poison.restart_evidence_sha256.as_str()),
        (EXIT_ID, poison.exit_evidence_sha256.as_str()),
    ].into_iter().map(|(case, digest)| {
        source(&format!("scheduler/{case}/evidence.json"), digest, "")
    }).collect();
    ensure!(
        *consumed == json!({
            "sources": expected_sources,
            "historical_observation": old.observation,
            "historical_manifest_sha256": sha(&old.manifest_prefix),
            "historical_next_forward_id": old.next_forward,
            "historical_request_ids": old.seen_request_ids,
            "consumed_exit_evidence_sha256": poison.exit_evidence_sha256,
            "consumed_ledger_sha256": sha(&poison.ledger_prefix),
            "consumed_orchestration_sha256": sha(&poison.orchestration_prefix)
        }),
        "recovery consumed-exit summary changed"
    );

    exact_extension(
        &old_frames,
        &ready_frames,
        &[
            frame("driver_to_orchestrator", request),
            frame("orchestrator_to_driver", accepted),
            frame("orchestrator_to_driver", ready_envelope),
        ],
    )?;
    ensure!(
        reattested["orchestration_prefix_sha256"] == sha(&ready.orchestration_prefix),
        "reattestation transcript commitment changed"
    );

    let root = format!("scheduler/{READY_ID}");
    files.pretty(&format!("{root}/request.json"), request)?;
    files.pretty(&format!("{root}/accepted.json"), accepted)?;
    files.pretty(&format!("{root}/ready.json"), ready_envelope)?;
    files.exact(&format!("{root}/old_absence_rechecked.json"), &absence_bytes)?;

    let binding_path = format!("{root}/launch_binding.json");
    let binding_bytes = files.committed(&binding_path, &launch.binding_sha256)?;
    let binding: Value = serde_json::from_slice(&binding_bytes)?;
    ensure!(
        reattested["launch_binding_path"] == binding_path
            && reattested["launch_binding_sha256"] == launch.binding_sha256
            && reattested["launch_binding"] == binding
            && binding["schema"] == "rustv1.accept_m4_recovery_launch_binding.v1"
            && binding["authenticated_ready"] == *ready_envelope
            && binding["policy_sha256"] == loaded.policy_sha256
            && binding["startup"] == launch.startup
            && binding["process_start_ticks"] == launch.process.start_ticks
            && binding["boot_id"] == launch.process.boot_id
            && binding["running_binary_sha256"] == launch.process.executable_sha256
            && binding["launch_performed_by"] == "external_supervisor"
            && binding["process_creation_after_old_exit_independently_measured"] == false
            && launch.process.pid == launch.pid
            && launch.process.executable_sha256 == loaded.artifacts["serve_m3"].sha256
            && crate::contract::digest_valid(text(&binding, "challenge")?),
        "recovery launch binding changed"
    );

    let ready_fields = &ready_envelope["fields"];
    ensure!(
        ready_fields["instance"] == launch.instance
            && ready_fields["pid"] == launch.pid
            && ready_fields["start_utc"] == launch.start_utc
            && Path::new(text(ready_fields, "manifest")?).canonicalize()? == launch.manifest
            && Path::new(text(ready_fields, "attestation")?).canonicalize()? == launch.attestation
            && launch.instance != old.launch.instance
            && launch.instance != core.initial_launch.instance
            && launch.pid != old.launch.pid
            && launch.pid != core.initial_launch.pid,
        "recovery retained launch identity changed"
    );

    // Recheck the historical path-isolation prerequisites without observing
    // either exited process.
    let (manifest_path, attestation_path) = recovery_paths(old, ready_envelope)?;
    ensure!(
        manifest_path == launch.manifest && attestation_path == launch.attestation,
        "recovery artifact isolation changed"
    );

    let attestation_bytes = external(&launch.attestation, JSON_LIMIT)?;
    ensure!(
        sha(&attestation_bytes) == text(&binding, "attestation_sha256")?,
        "recovery external attestation commitment changed"
    );
    files.exact(&format!("{root}/launch_attestation.json"), &attestation_bytes)?;
    files.exact(
        &format!("{root}/startup_executor_manifest.jsonl"),
        &launch.initial_manifest,
    )?;

    let attestation: Value = serde_json::from_slice(&attestation_bytes)?;
    ensure!(
        attestation["schema"] == "rustv1.server_launch_attestation.v1"
            && attestation["instance_id"] == launch.instance
            && attestation["pid"] == launch.pid
            && attestation["start_utc"] == launch.start_utc
            && attestation["settings_sha256"] == loaded.policy_sha256
            && attestation["binary_sha256"] == loaded.artifacts["serve_m3"].sha256
            && attestation["audit_sha256"] == loaded.artifacts["audit"].sha256
            && attestation["startup_check_sha256"] == loaded.artifacts["startup_check"].sha256
            && attestation["execution_policy"] == BATCHED_EXECUTION_POLICY
            && attestation["http_contract_id"] == BATCHED_HTTP_CONTRACT
            && attestation["batching"] == serde_json::to_value(&loaded.bundle.shapes.batching)?,
        "recovery attestation policy binding"
    );
    for name in ["rustc_version", "nvcc_version", "cuda_compute_cap", "target_features"] {
        ensure!(
            !text(&attestation["build_identity"], name)?.is_empty(),
            "empty recovery build identity"
        );
    }

    let attested = &binding["authenticated_observation"];
    let mut plain = attested.clone();
    ensure!(
        plain.as_object_mut().context("recovery attested observation")?
            .remove("attestation").is_some()
            && plain == launch.initial_observation
            && ready_observation(&plain, launch)? == 0,
        "recovery startup observation changed"
    );
    let runtime_attestation = &attested["attestation"];
    let mut base = runtime_attestation.clone();
    let object = base.as_object_mut().context("recovery runtime attestation")?;
    for name in ["http_address", "control_address", "startup", "capabilities"] {
        ensure!(object.remove(name).is_some(), "missing runtime attestation field");
    }
    ensure!(
        base == attestation
            && serde_json::to_vec_pretty(&base)? == attestation_bytes
            && core::check_startup(
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
            && plain["startup_output_sha256"] == launch.startup["startup_output_sha256"],
        "recovery startup/runtime/base attestation binding"
    );

    let expected_http = core::base_address(&core.base_url)?;
    let actual_http: SocketAddr = text(runtime_attestation, "http_address")?.parse()?;
    let actual_control: SocketAddr = text(runtime_attestation, "control_address")?.parse()?;
    let expected_control: SocketAddr = text(&loaded.policy, "launch_control_address")?.parse()?;
    ensure!(
        actual_control == expected_control
            && actual_http.port() == expected_http.port()
            && (actual_http.ip() == expected_http.ip() || actual_http.ip().is_unspecified())
            && reattested["initial_observation"] == launch.initial_observation
            && reattested["startup_manifest_sha256"] == sha(&launch.initial_manifest)
            && reattested["next_forward_id"] == launch.startup["next_forward_id"]
            && reattested["replacement_request_ids"] == json!([])
            && reattested["replacement_logical_requests"] == 0
            && reattested["exact_http_recovery_executed"] == false,
        "recovery saved endpoint/startup boundary"
    );

    // Bind the later cursors, without representing this as independent
    // reconciliation of the 44 saved HTTP request/response intervals.
    ensure!(
        recovered["reattestation_evidence_sha256"] == ready.reattestation_evidence_sha256
            && recovered["launch_binding_sha256"] == launch.binding_sha256
            && recovered["orchestration_prefix_sha256"] == sha(&recovery.orchestration_prefix)
            && recovered["final_observation"] == recovery.observation
            && recovered["final_manifest_sha256"] == sha(&recovery.manifest_prefix)
            && recovered["final_manifest_bytes"] == recovery.manifest_prefix.len()
            && recovered["next_forward_id"] == recovery.next_forward
            && recovered["replacement_request_ids"] == json!(recovery.seen_request_ids)
            && recovered["old_historical_observation"] == old.observation
            && recovered["old_service_owner_join_claimed"] == false
            && recovered["old_quarantine_cleared_in_process"] == false
            && recovery.manifest_prefix.starts_with(&launch.initial_manifest),
        "saved recovery cursor lineage changed"
    );
    ready_observation(&recovery.observation, launch)?;
    quiescence(&recovery.observation, &state.quiesced_observation)?;
    check_final_exit(&state.drain_request, &state.authenticated_exit)?;
    unique_nonce(
        &ready_frames,
        text(&state.drain_request["fields"], "drain_request_id")?,
    )?;

    let quiesced_bytes = serde_json::to_vec(&state.quiesced_observation)?;
    let fields = &state.drain_request["fields"];
    ensure!(
        fields["instance"] == launch.instance
            && fields["pid"] == launch.pid
            && fields["process_start_ticks"] == launch.process.start_ticks
            && fields["boot_id"] == launch.process.boot_id
            && fields["running_binary_sha256"] == launch.process.executable_sha256
            && fields["policy_sha256"] == loaded.policy_sha256
            && fields["launch_binding_sha256"] == launch.binding_sha256
            && fields["recovery_evidence_sha256"] == recovery.recovery_evidence_sha256
            && fields["manifest_sha256"] == sha(&recovery.manifest_prefix)
            && fields["quiesced_observation_sha256"] == sha(&quiesced_bytes)
            && fields["preceding_ledger_sha256"] == sha(&recovery.ledger_prefix)
            && state.process_absence == historical_absence(&launch.process),
        "final drain target/prerequisites changed"
    );
    exact_extension(
        &ready_frames,
        &final_frames,
        &[
            frame("driver_to_orchestrator", &state.drain_request),
            frame("orchestrator_to_driver", &state.authenticated_exit),
        ],
    )?;

    ensure!(
        drained["recovery_evidence_sha256"] == recovery.recovery_evidence_sha256
            && drained["launch_binding_sha256"] == launch.binding_sha256
            && drained["recovery_observation_preserved"] == recovery.observation
            && drained["recovery_observation_is_historical"] == true
            && drained["quiesced_observation"] == state.quiesced_observation
            && drained["quiesced_observation_sha256"] == sha(&quiesced_bytes)
            && drained["request"] == state.drain_request
            && drained["authenticated_exit"] == state.authenticated_exit
            && drained["independent_process_absence"] == state.process_absence
            && drained["preceding_orchestration_sha256"] == sha(&recovery.orchestration_prefix)
            && drained["orchestration_prefix_sha256"] == sha(&transcript)
            && drained["manifest_unchanged_sha256"] == sha(&recovery.manifest_prefix)
            && drained["manifest_bytes"] == recovery.manifest_prefix.len()
            && drained["executor_next_forward_id_unchanged"] == recovery.next_forward
            && drained["exposed_request_ids_unchanged"] == json!(recovery.seen_request_ids)
            && drained["supervisor_reported_process_reaped"] == true
            && drained["supervisor_reap_independently_awaited_by_driver"] == false
            && drained["replacement_service_owner_join_independently_observed"] == false
            && drained["replacement_post_exit_observation_fabricated"] == false
            && drained["old_historical_observation"] == old.observation
            && drained["old_service_owner_join_claimed"] == false
            && drained["old_quarantine_cleared_in_process"] == false
            && drained["replacement_launch_authorized"] == false,
        "final drain history changed or was relabelled"
    );

    let drain_root = format!("scheduler/{DRAIN_ID}");
    files.exact(&format!("{drain_root}/quiesced_observation.json"), &quiesced_bytes)?;
    files.pretty(&format!("{drain_root}/request.json"), &state.drain_request)?;
    files.pretty(
        &format!("{drain_root}/authenticated_exit.json"),
        &state.authenticated_exit,
    )?;
    files.pretty(&format!("{drain_root}/process_absence.json"), &state.process_absence)?;
    files.exact(
        &format!("{drain_root}/executor_manifest_snapshot.jsonl"),
        &recovery.manifest_prefix,
    )?;

    let old_manifest = external(&old.launch.manifest, MANIFEST_LIMIT)?;
    let recovery_manifest = external(&launch.manifest, MANIFEST_LIMIT)?;
    ensure!(
        old_manifest == old.manifest_prefix
            && recovery_manifest == recovery.manifest_prefix,
        "external final manifests differ from retained historical cursors"
    );

    files.stable()?;
    loaded.recheck()?;
    ensure!(
        external(&old.launch.manifest, MANIFEST_LIMIT)? == old_manifest
            && external(&launch.manifest, MANIFEST_LIMIT)? == recovery_manifest
            && external(&launch.attestation, JSON_LIMIT)? == attestation_bytes
            && bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)? == ledger
            && bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)? == transcript,
        "lineage inputs changed during reconciliation"
    );

    Ok(json!({
        "schema": "rustv1.accept_m4_recovery_lineage_reconciliation.v1",
        "boundary": LINEAGE_BOUNDARY,
        "policy_sha256": loaded.policy_sha256,
        "preceding_ledger_sha256": sha(&ledger),
        "preceding_initial_reconciliation_sha256":
            sha(&serde_json::to_vec(&continuation.initial_launch_reconciliation)?),
        "orchestration_sha256": sha(&transcript),
        "authenticated_transcript_frames_checked": final_frames.len(),
        "restart_semantic_positions": [6, 7, 8],
        "recovery_launch_semantic_positions": [9, 10, 11],
        "final_drain_semantic_positions": [12, 13],
        "whole_transcript_semantics_reconciled": false,
        "recovery_launch_binding_sha256": launch.binding_sha256,
        "recovery_attestation_sha256": sha(&attestation_bytes),
        "recovery_startup_manifest_sha256": sha(&launch.initial_manifest),
        "recovery_final_manifest_sha256": sha(&recovery_manifest),
        "recovery_final_manifest_bound_to_retained_cursor": true,
        "recovery_http_intervals_independently_reconciled_here": false,
        "historical_poisoned_observation": old.observation,
        "historical_poisoned_observation_preserved": true,
        "historical_recovery_observation_preserved": true,
        "old_service_owner_join_claimed": false,
        "replacement_service_owner_join_independently_observed": false,
        "supervisor_reap_independently_awaited_by_driver": false,
        "process_creation_after_old_exit_independently_measured": false,
        "historical_absence_records_checked_without_remeasurement": true,
        "verified_output_sha256": files.hashes,
        "complete_output_directory_coverage": false,
        "destructive_children_reconciled": false,
        "complete_poisoned_launch_artifact_set_reconciled": false,
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
    fn lineage_extension_rejects_extra_missing_and_reordered_frames() {
        // Format fixtures only. Production callers authenticate every frame
        // with verify_transcript before invoking exact_extension.
        let previous = vec![json!({"historical": 1})];
        let suffix = vec![json!({"request": 2}), json!({"ack": 3})];
        let mut current = previous.clone();
        current.extend(suffix.clone());
        assert!(exact_extension(&previous, &current, &suffix).is_ok());

        let mut extra = current.clone();
        extra.push(json!({"unexpected": 4}));
        assert!(exact_extension(&previous, &extra, &suffix).is_err());
        assert!(exact_extension(&previous, &current[..2], &suffix).is_err());

        let mut reordered = current.clone();
        reordered.swap(1, 2);
        assert!(exact_extension(&previous, &reordered, &suffix).is_err());

        let mut changed_prefix = current;
        changed_prefix[0] = json!({"historical": 99});
        assert!(exact_extension(&previous, &changed_prefix, &suffix).is_err());
    }

    #[test]
    fn historical_poison_cursor_cannot_be_relabelled_as_stopped_or_joined() {
        let observation = json!({
            "instance": "old",
            "pid": 42,
            "lifecycle": "poisoned",
            "worker_alive": true,
            "ready": false,
            "forward_calls": 9,
            "startup_output_sha256": sha(b"startup")
        });
        let manifest = b"{}\n";
        let ids = BTreeSet::from([7u64]);
        let exit = json!({
            "old_pre_exit_observation": observation,
            "old_observation_is_historical": true,
            "manifest_unchanged_sha256": sha(manifest),
            "executor_next_forward_id_unchanged": 3,
            "exposed_request_ids_unchanged": ids,
            "service_owner_join_claimed": false,
            "quarantine_cleared_in_process": false,
            "replacement_launch_authorized": false
        });
        assert!(historical_cursor(&exit, &observation, manifest, 3, &ids).is_ok());

        let mut stopped = observation.clone();
        stopped["lifecycle"] = json!("stopped");
        stopped["worker_alive"] = json!(false);
        assert!(historical_cursor(&exit, &stopped, manifest, 3, &ids).is_err());

        let mut joined = exit;
        joined["service_owner_join_claimed"] = json!(true);
        assert!(historical_cursor(&joined, &observation, manifest, 3, &ids).is_err());
    }

    #[test]
    fn lineage_nonce_rejects_prior_authorization_identity() {
        let nonce = sha(b"authorization");
        let frames = vec![frame(
            "driver_to_orchestrator",
            &json!({"fields": {"restart_request_id": nonce}}),
        )];
        assert!(unique_nonce(&frames, &nonce).is_err());
        assert!(unique_nonce(&frames, &sha(b"fresh")).is_ok());
        assert!(unique_nonce(&frames, "not-a-digest").is_err());
    }
}
