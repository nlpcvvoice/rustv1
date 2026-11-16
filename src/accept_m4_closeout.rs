// Included at:
// acceptance_scheduler::saved_initial::saved_coverage::closeout
//
// Offline closeout. No network operations, process observations,
// owner joins, child reaps, or additional orchestration frames.
//
// Only run(), gated by cuda, can publish overall production acceptance.
// Tests below exercise validators, not production acceptance.

use super::*;

const FINAL_CASE: &str = "final-artifact-and-manifest-reconciliation";
const FINAL_DIRECTORY: &str =
    "scheduler/final-artifact-and-manifest-reconciliation";
const FINAL_EVIDENCE: &str =
    "scheduler/final-artifact-and-manifest-reconciliation/evidence.json";

const SCHEDULER_MARKER: &str = "acceptance_scheduler_manifest.json";
const LIFECYCLE_MARKER: &str = "acceptance_lifecycle_manifest.json";
const OVERALL_MARKER: &str = "acceptance_m4_manifest.json";

/// Redacted completion receipt. Contains no continuation or credentials.
pub struct AcceptanceCompletion {
    pub completion_manifest_path: PathBuf,
    pub completion_manifest_sha256: String,
    pub policy_sha256: String,
    pub ledger: Value,
}

fn retained(c: &OutputCoverageContinuation) -> &FinalDrainState {
    super::super::saved_scheduler::retained(
        &c.scheduler_artifacts.independent_artifacts,
    )
}

fn retained_mut(c: &mut OutputCoverageContinuation) -> &mut FinalDrainState {
    &mut c.scheduler_artifacts.independent_artifacts
        .poisoned_launch.destructive_owners.recovery_http
        .recovery_lineage.initial_artifacts.imported_artifacts
        .local_artifacts.final_drain.state
}

fn ordered_plan(required: &policy::Required) -> Result<Vec<(String, String)>> {
    required.validate()?;
    let plan: Vec<_> = required.phases.iter()
        .flat_map(|(phase, cases)| {
            cases.iter().map(move |case| (phase.clone(), case.clone()))
        })
        .collect();

    ensure!(
        plan.last() == Some(&(policy::LIFECYCLE.to_owned(), FINAL_CASE.to_owned())),
        "closeout requires the final reconciliation case last"
    );
    Ok(plan)
}

/// Exact global order, schema, execution status and successful coverage.
/// Phase/overall markers are deliberately absent from this plan.
fn ledger_rows(bytes: &[u8], plan: &[(String, String)]) -> Result<Vec<Value>> {
    let records = core::jsonl(bytes)?;
    ensure!(records.len() == plan.len(), "closeout ledger coverage");
    for (record, (phase, case)) in records.iter().zip(plan) {
        closed(record, &[
            "acceptance_phase", "case_id", "status", "pass", "details",
        ])?;
        ensure!(
            record["acceptance_phase"] == phase.as_str()
                && record["case_id"] == case.as_str()
                && record["status"] == "executed"
                && record["pass"] == true
                && record["details"].is_object(),
            "closeout ledger order/status/schema"
        );
    }
    Ok(records)
}

/// Preserve exact preceding bytes, including their original serialization.
/// Require precisely the expected new row, with no trailing extra row.
fn final_extension(previous: &[u8], current: &[u8], expected: &Value) -> Result<()> {
    core::jsonl(previous)?;
    ensure!(
        current.starts_with(previous),
        "closeout append rewrote preceding ledger bytes"
    );
    let suffix = &current[previous.len()..];
    let rows = core::jsonl(suffix)?;
    ensure!(
        rows == vec![expected.clone()],
        "closeout did not append exactly the expected final row"
    );
    Ok(())
}

fn inventory_of(value: &Value) -> Result<BTreeMap<String, String>> {
    let mut inventory = BTreeMap::new();
    merge(&mut inventory, value)?;
    Ok(inventory)
}

fn add_new(
    inventory: &mut BTreeMap<String, String>,
    relative: &str,
    digest: String,
) -> Result<()> {
    safe_relative(relative)?;
    ensure!(
        crate::contract::digest_valid(&digest)
            && !inventory.contains_key(relative),
        "closeout artifact collision or invalid digest"
    );
    inventory.insert(relative.to_owned(), digest);
    Ok(())
}

fn absent(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
        Ok(_) => anyhow::bail!("closeout destination already exists: {}", path.display()),
    }
}

fn write_pretty(out: &Path, relative: &str, value: &Value) -> Result<String> {
    safe_relative(relative)?;
    let bytes = serde_json::to_vec_pretty(value)?;
    ensure!(bytes.len() <= JSON_LIMIT, "closeout JSON exceeds bound");
    write_new(&out.join(relative), &bytes)?;
    ensure!(
        bounded(&output_path(out, relative)?, JSON_LIMIT)? == bytes,
        "closeout artifact readback differs"
    );
    Ok(sha(&bytes))
}

/// These are preserved scope limitations, not excluded required cases.
///
/// The complete original reconciliation objects and producer files remain
/// authoritative for their more detailed historical classifications.
fn limitations() -> Value {
    json!({
        "historical_observations_relabelled": false,
        "unjoined_owner_closures_relabelled": false,
        "diagnostic_hashes_are_original_historical_commitments": false,
        "http_marker_original_digest_recovered": false,
        "http_challenge_original_value_independently_authenticated": false,
        "handoff_original_path_spelling_independently_authenticated": false,
        "independent_http_connection_joins_claimed": false,
        "independent_saved_response_header_revalidation_claimed": false,
        "family53_driver_binary_producer_field_present": false,
        "external_service_owner_joins_independently_observed": false,
        "supervisor_reaps_independently_awaited_by_driver": false,
        "external_process_creation_order_independently_measured": false,
        "isolated_padded_capacity_rejection_proven": false,
        "isolated_attention_capacity_rejection_proven": false,
        "independent_queue_slot_rejection_predicate_proven": false,
        "general_quiesced_backlog_assembly_suppression_proven": false,
        "arbitrary_batch_shape_bit_invariance_claimed": false,
        "manual_clock_witnesses_are_wall_clock_measurements": false,
        "coordinator_runtime_drop_counted_as_owner_join": false
    })
}

/// Preserve all nine prerequisite summaries and combined coverage as
/// historical objects. Do not change their original false completion flags.
fn prerequisite_objects(c: &OutputCoverageContinuation) -> Vec<Value> {
    let scheduler = &c.scheduler_artifacts;
    let independent = &scheduler.independent_artifacts;
    let poison = &independent.poisoned_launch;
    let owners = &poison.destructive_owners;
    let http = &owners.recovery_http;
    let lineage = &http.recovery_lineage;
    let initial = &lineage.initial_artifacts;
    let imports = &initial.imported_artifacts;
    let local = &imports.local_artifacts;

    vec![
        local.local_reconciliation.clone(),
        imports.import_reconciliation.clone(),
        initial.initial_launch_reconciliation.clone(),
        lineage.recovery_lineage_reconciliation.clone(),
        http.recovery_http_reconciliation.clone(),
        owners.destructive_owners_reconciliation.clone(),
        poison.poisoned_launch_reconciliation.clone(),
        independent.independent_reconciliation.clone(),
        scheduler.scheduler_reconciliation.clone(),
        c.output_coverage_reconciliation.clone(),
    ]
}

/// Closing reads use the already-validated inventory, never directory-derived
/// expected filenames or newly selected historical commitments.
///
/// The caller supplies the exact ledger bytes expected at this stage.
fn closing_checks(
    c: &OutputCoverageContinuation,
    inventory: &BTreeMap<String, String>,
    ledger: &[u8],
) -> Result<()> {
    let state = retained(c);
    let core_state = &state.recovery.ready.poison_exit.old.core;
    let out = &core_state.out;
    core_state.loaded.recheck()?;

    ensure!(
        bounded(&output_path(out, "accept_records.jsonl")?, MANIFEST_LIMIT)?
            == ledger
            && bounded(&output_path(out, "orchestration.jsonl")?, JSON_LIMIT)?
                == state.orchestration_prefix,
        "closeout ledger/transcript changed"
    );

    coverage(out, inventory)?;
    stable(out, inventory)?;

    let external = c.output_coverage_reconciliation[
        "verified_external_launch_artifacts"
    ].as_array().context("closeout external artifact inventory")?;
    ensure!(external.len() == 6, "closeout external launch artifact coverage");
    for entry in external {
        closed(entry, &["path", "sha256"])?;
        let path = Path::new(text(entry, "path")?);
        ensure!(
            external_entry(path, text(entry, "sha256")?, MANIFEST_LIMIT)? == *entry,
            "closeout external artifact changed"
        );
    }

    core_state.loaded.recheck()?;
    stable(out, inventory)?;
    coverage(out, inventory)?;
    ensure!(
        bounded(&output_path(out, "accept_records.jsonl")?, MANIFEST_LIMIT)?
            == ledger
            && bounded(&output_path(out, "orchestration.jsonl")?, JSON_LIMIT)?
                == state.orchestration_prefix,
        "closeout final ledger/transcript stability"
    );
    Ok(())
}

fn phase_ids(required: &policy::Required, phase: &str) -> Result<Vec<String>> {
    let matches: Vec<_> = required.phases.iter()
        .filter(|(name, _)| name == phase)
        .collect();
    ensure!(matches.len() == 1, "closeout phase declaration");
    Ok(matches[0].1.clone())
}

#[allow(clippy::too_many_arguments)]
fn phase_marker(
    phase: &str,
    required: &policy::Required,
    records: &[Value],
    policy_digest: &str,
    ledger_digest: &str,
    evidence_digest: &str,
    inventory: &BTreeMap<String, String>,
    scheduler_marker_digest: Option<&str>,
) -> Result<Value> {
    ensure!(
        matches!(phase, policy::SCHEDULER | policy::LIFECYCLE)
            && (phase == policy::LIFECYCLE) == scheduler_marker_digest.is_some(),
        "closeout phase marker order"
    );
    let ids = phase_ids(required, phase)?;
    let completed: Vec<String> = records.iter()
        .filter(|row| row["acceptance_phase"] == phase)
        .map(|row| Ok(text(row, "case_id")?.to_owned()))
        .collect::<Result<_>>()?;
    ensure!(completed == ids, "closeout phase marker case coverage");

    Ok(json!({
        "schema": if phase == policy::SCHEDULER {
            "rustv1.accept_m4_scheduler_manifest.v1"
        } else {
            "rustv1.accept_m4_lifecycle_manifest.v1"
        },
        "acceptance_phase": phase,
        "policy_sha256": policy_digest,
        "global_records_sha256": ledger_digest,
        "required_case_ids": ids,
        "completed_case_ids": completed,
        "pending_required_cases_in_scope": [],
        "exceptions": [],
        "final_reconciliation": {
            "path": FINAL_EVIDENCE,
            "sha256": evidence_digest
        },
        "preceding_scheduler_marker": scheduler_marker_digest.map(|digest| {
            json!({"path": SCHEDULER_MARKER, "sha256": digest})
        }),
        "output_sha256_before_marker": inventory,
        "marker_excluded_from_its_own_inventory": true,
        "marker_is_required_case": false,
        "historical_limitations": limitations(),
        "final_artifact_reconciliation_complete": true,
        "scheduler_phase_complete": true,
        "lifecycle_phase_complete": phase == policy::LIFECYCLE,
        "global_ledger_finish_performed": false,
        "post_exit_control_calls": 0,
        "process_observations": 0,
        "owner_joins_performed": 0,
        "child_reaps_performed": 0,
        "pass": true,
        "m4_accepted": false
    }))
}

fn finish_proof(
    proof: &Value,
    policy_digest: &str,
    inventory: &BTreeMap<String, String>,
    count: usize,
) -> Result<()> {
    let expected = json!({
        "policy_sha256": policy_digest,
        "required_cases_sha256": inventory.get("required_cases.json")
            .context("closeout required declaration digest")?,
        "records_sha256": inventory.get("accept_records.jsonl")
            .context("closeout final ledger digest")?,
        "required_case_count": count,
        "completed_case_count": count,
        "pending_required_cases": [],
        "exceptions": [],
        "pass": true
    });
    ensure!(*proof == expected, "Ledger::finish readback proof differs");
    Ok(())
}

/// Production closeout is unavailable in CPU-only builds.
///
/// This function is synchronous and consumes the original continuation.
/// It does not create a runtime or move the retained orchestration socket
/// onto another runtime.
#[cfg(feature = "cuda")]
pub(crate) fn run(
    mut c: OutputCoverageContinuation,
) -> Result<AcceptanceCompletion> {
    // Last complete invocation of the pre-append reconciliation chain.
    ensure!(
        super::reconcile(&c.scheduler_artifacts)?
            == c.output_coverage_reconciliation,
        "output coverage continuation changed before closeout"
    );

    let state = retained(&c);
    let core_state = &state.recovery.ready.poison_exit.old.core;
    let loaded = &core_state.loaded;
    let out = core_state.out.clone();
    let required = loaded.required.clone();
    let plan = ordered_plan(&required)?;
    let previous = state.ledger_prefix.clone();
    let policy_digest = loaded.policy_sha256.clone();
    let driver_digest = loaded.artifacts["accept_m4"].sha256.clone();
    let binding_summary = loaded.binding_summary();

    ensure!(
        core_state.ledger.pending() == vec![json!({
            "acceptance_phase": policy::LIFECYCLE,
            "case_id": FINAL_CASE
        })],
        "closeout called at wrong ledger boundary"
    );
    ledger_rows(&previous, &plan[..plan.len() - 1])?;

    let mut inventory = inventory_of(&c.output_coverage_reconciliation)?;
    ensure!(
        inventory.get("accept_records.jsonl") == Some(&sha(&previous)),
        "closeout preceding ledger inventory"
    );
    for relative in [
        FINAL_DIRECTORY, SCHEDULER_MARKER, LIFECYCLE_MARKER, OVERALL_MARKER,
    ] {
        absent(&out.join(relative))?;
    }
    closing_checks(&c, &inventory, &previous)?;

    // 1. Final reconciliation evidence, then exactly one final ledger pass.
    //
    // This artifact is produced before its ledger row. Its embedded coverage
    // is explicitly the pre-append historical boundary, not a self-hash.
    let evidence = json!({
        "schema": "rustv1.accept_m4_final_reconciliation.v1",
        "case_id": FINAL_CASE,
        "policy_sha256": policy_digest,
        "driver_binary_sha256": driver_digest,
        "preceding_ledger_sha256": sha(&previous),
        "preceding_ledger_bytes": previous.len(),
        "preceding_record_count": plan.len() - 1,
        "required_case_count": plan.len(),
        "orchestration_sha256": sha(&state.orchestration_prefix),
        "prerequisite_reconciliations": prerequisite_objects(&c),
        "historical_prerequisite_objects_preserved_unchanged": true,
        "historical_limitations": limitations(),
        "coverage_scope": "all_outputs_before_this_evidence_and_final_ledger_append",
        "subsequent_closeout_outputs": [
            FINAL_EVIDENCE,
            SCHEDULER_MARKER,
            LIFECYCLE_MARKER,
            OVERALL_MARKER
        ],
        "subsequent_ledger_change": "exactly_one_final_reconciliation_row",
        "whole_transcript_semantics_reconciled": true,
        "all_artifact_prerequisites_reconciled": true,
        "final_artifact_reconciliation_complete": true,
        "ledger_case_append_included_in_this_artifact": false,
        "post_exit_control_calls": 0,
        "process_observations": 0,
        "owner_joins_performed": 0,
        "child_reaps_performed": 0,
        "scheduler_phase_complete": false,
        "lifecycle_phase_complete": false,
        "m4_accepted": false
    });

    private_directory(&out.join(FINAL_DIRECTORY))?;
    let evidence_digest = write_pretty(&out, FINAL_EVIDENCE, &evidence)?;
    add_new(&mut inventory, FINAL_EVIDENCE, evidence_digest.clone())?;
    closing_checks(&c, &inventory, &previous)?;

    let details = json!({
        "evidence_path": FINAL_EVIDENCE,
        "evidence_sha256": evidence_digest,
        "preceding_ledger_sha256": sha(&previous),
        "policy_sha256": policy_digest
    });
    let expected_row = json!({
        "acceptance_phase": policy::LIFECYCLE,
        "case_id": FINAL_CASE,
        "status": "executed",
        "pass": true,
        "details": details
    });

    retained_mut(&mut c).recovery.ready.poison_exit.old.core.ledger.append(
        policy::LIFECYCLE, FINAL_CASE, true, details, true,
    )?;

    let ledger = bounded(
        &output_path(&out, "accept_records.jsonl")?, MANIFEST_LIMIT,
    )?;
    final_extension(&previous, &ledger, &expected_row)?;
    let records = ledger_rows(&ledger, &plan)?;
    ensure!(
        retained(&c).recovery.ready.poison_exit.old.core.ledger.pending().is_empty(),
        "closeout final append left required cases pending"
    );

    let ledger_digest = sha(&ledger);
    ensure!(
        inventory.insert("accept_records.jsonl".to_owned(), ledger_digest.clone())
            == Some(sha(&previous)),
        "closeout ledger inventory transition"
    );
    closing_checks(&c, &inventory, &ledger)?;

    // 2. Scheduler marker. Not a required ledger case.
    let scheduler = phase_marker(
        policy::SCHEDULER, &required, &records, &policy_digest,
        &ledger_digest, &evidence_digest, &inventory, None,
    )?;
    let scheduler_digest = write_pretty(&out, SCHEDULER_MARKER, &scheduler)?;
    add_new(&mut inventory, SCHEDULER_MARKER, scheduler_digest.clone())?;
    closing_checks(&c, &inventory, &ledger)?;

    // 3. Lifecycle marker, bound to the preceding scheduler marker.
    let lifecycle = phase_marker(
        policy::LIFECYCLE, &required, &records, &policy_digest,
        &ledger_digest, &evidence_digest, &inventory,
        Some(&scheduler_digest),
    )?;
    let lifecycle_digest = write_pretty(&out, LIFECYCLE_MARKER, &lifecycle)?;
    add_new(&mut inventory, LIFECYCLE_MARKER, lifecycle_digest.clone())?;
    closing_checks(&c, &inventory, &ledger)?;

    // 4. Actual global Ledger::finish(), not a reconstructed substitute.
    let finished =
        retained_mut(&mut c).recovery.ready.poison_exit.old.core.ledger.finish()?;
    finish_proof(&finished, &policy_digest, &inventory, plan.len())?;
    ledger_rows(&ledger, &plan)?;
    closing_checks(&c, &inventory, &ledger)?;

    // 5. Overall marker is the last new output artifact.
    //
    // Its inventory contains every preceding output, including both phase
    // markers and the extended ledger. It excludes only itself.
    let overall = json!({
        "schema": "rustv1.accept_m4_manifest.v1",
        "acceptance_phase": "full_acceptance",
        "producer": {
            "component": "accept_m4",
            "command": "acceptance-full",
            "binary_sha256": driver_digest
        },
        "binding_summary": binding_summary,
        "policy_sha256": policy_digest,
        "required": required.as_json(),
        "ledger": finished,
        "final_reconciliation": {
            "path": FINAL_EVIDENCE,
            "sha256": evidence_digest
        },
        "phase_markers": {
            "scheduler": {
                "path": SCHEDULER_MARKER,
                "sha256": scheduler_digest
            },
            "lifecycle": {
                "path": LIFECYCLE_MARKER,
                "sha256": lifecycle_digest
            }
        },
        "output_sha256": inventory,
        "output_inventory_excludes_only": [OVERALL_MARKER],
        "verified_external_launch_artifacts":
            c.output_coverage_reconciliation["verified_external_launch_artifacts"],
        "historical_limitations": limitations(),
        "required_cases_excluded": [],
        "pending_required_cases": [],
        "exceptions": [],
        "markers_are_required_cases": false,
        "whole_transcript_semantics_reconciled": true,
        "final_artifact_reconciliation_complete": true,
        "scheduler_phase_complete": true,
        "lifecycle_phase_complete": true,
        "global_ledger_finish_performed": true,
        "post_exit_control_calls": 0,
        "process_observations": 0,
        "owner_joins_performed": 0,
        "child_reaps_performed": 0,
        "pass": true,
        "m4_accepted": true
    });
    let overall_digest = write_pretty(&out, OVERALL_MARKER, &overall)?;
    add_new(&mut inventory, OVERALL_MARKER, overall_digest.clone())?;

    // Read-only checks after publication; no further output artifacts.
    closing_checks(&c, &inventory, &ledger)?;

    Ok(AcceptanceCompletion {
        completion_manifest_path: out.join(OVERALL_MARKER),
        completion_manifest_sha256: overall_digest,
        policy_sha256: policy_digest,
        ledger: finished,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(rows: &[Value]) -> Vec<u8> {
        let mut result = Vec::new();
        for row in rows {
            serde_json::to_writer(&mut result, row).unwrap();
            result.push(b'\n');
        }
        result
    }

    fn row(phase: &str, case: &str) -> Value {
        json!({
            "acceptance_phase": phase,
            "case_id": case,
            "status": "executed",
            "pass": true,
            "details": {}
        })
    }

    #[test]
    fn closeout_ledger_requires_full_order_and_exact_single_append() {
        // Disposable format fixtures; not acceptance execution evidence.
        let first = row("scheduler", "probe");
        let last = row(policy::LIFECYCLE, FINAL_CASE);
        let previous = bytes(&[first.clone()]);
        let current = bytes(&[first.clone(), last.clone()]);
        let plan = vec![
            ("scheduler".to_owned(), "probe".to_owned()),
            (policy::LIFECYCLE.to_owned(), FINAL_CASE.to_owned()),
        ];
        ledger_rows(&current, &plan).unwrap();
        final_extension(&previous, &current, &last).unwrap();

        assert!(ledger_rows(&previous, &plan).is_err());
        assert!(ledger_rows(&bytes(&[last.clone(), first]), &plan).is_err());

        let mut failed = last.clone();
        failed["pass"] = json!(false);
        let mut changed = previous.clone();
        changed.extend(bytes(&[failed]));
        assert!(final_extension(&previous, &changed, &last).is_err());

        let mut extra = current.clone();
        extra.extend(bytes(&[last.clone()]));
        assert!(final_extension(&previous, &extra, &last).is_err());

        let mut reserialized_prefix = b" {\"different\":true}\n".to_vec();
        reserialized_prefix.extend(bytes(&[last.clone()]));
        assert!(final_extension(&previous, &reserialized_prefix, &last).is_err());

        assert!(final_extension(
            &previous, &current[..current.len() - 1], &last,
        ).is_err());
    }

    #[test]
    fn closeout_markers_are_not_cases_and_require_scheduler_before_lifecycle() {
        let required = policy::Required {
            phases: vec![
                (policy::SCHEDULER.to_owned(), vec!["probe".to_owned()]),
                (policy::LIFECYCLE.to_owned(), vec![FINAL_CASE.to_owned()]),
            ],
        };
        let records = vec![
            row(policy::SCHEDULER, "probe"),
            row(policy::LIFECYCLE, FINAL_CASE),
        ];
        let digest = sha(b"fixture");
        let inventory = BTreeMap::from([
            ("accept_records.jsonl".to_owned(), digest.clone()),
        ]);
        let scheduler = phase_marker(
            policy::SCHEDULER, &required, &records,
            &digest, &digest, &digest, &inventory, None,
        ).unwrap();
        assert_eq!(scheduler["m4_accepted"], false);
        assert_eq!(scheduler["marker_is_required_case"], false);
        assert_eq!(scheduler["global_ledger_finish_performed"], false);
        assert_eq!(required.count(), 2);

        assert!(phase_marker(
            policy::LIFECYCLE, &required, &records,
            &digest, &digest, &digest, &inventory, None,
        ).is_err());
        let lifecycle = phase_marker(
            policy::LIFECYCLE, &required, &records,
            &digest, &digest, &digest, &inventory, Some(&digest),
        ).unwrap();
        assert_eq!(lifecycle["m4_accepted"], false);
        assert_eq!(lifecycle["lifecycle_phase_complete"], true);
        assert_eq!(required.count(), 2);
    }

    #[test]
    fn closeout_finish_proof_rejects_pending_exclusions_and_rebased_hashes() {
        let policy_digest = sha(b"policy");
        let inventory = BTreeMap::from([
            ("required_cases.json".to_owned(), sha(b"required")),
            ("accept_records.jsonl".to_owned(), sha(b"records")),
        ]);
        let proof = json!({
            "policy_sha256": policy_digest,
            "required_cases_sha256": inventory["required_cases.json"],
            "records_sha256": inventory["accept_records.jsonl"],
            "required_case_count": 2,
            "completed_case_count": 2,
            "pending_required_cases": [],
            "exceptions": [],
            "pass": true
        });
        finish_proof(&proof, &policy_digest, &inventory, 2).unwrap();

        for (field, replacement) in [
            ("pending_required_cases", json!(["missing"])),
            ("exceptions", json!(["CUDA excluded"])),
            ("completed_case_count", json!(1)),
            ("records_sha256", json!(sha(b"rebased"))),
            ("pass", json!(false)),
        ] {
            let mut changed = proof.clone();
            changed[field] = replacement;
            assert!(finish_proof(
                &changed, &policy_digest, &inventory, 2,
            ).is_err());
        }
    }
}
