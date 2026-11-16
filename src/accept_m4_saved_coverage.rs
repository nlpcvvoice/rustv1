// Included at acceptance_scheduler::saved_initial::saved_coverage.
//
// Combined offline output coverage and closing byte stability.
//
// No expected filename is discovered from directory enumeration.
// Historical diagnostic digests remain inventory/stability values.
// This boundary does not append the final ledger case.

use super::*;

const COVERAGE_BOUNDARY: &str =
    "ALL-ARTIFACTS-TRANSCRIPT-AND-OUTPUT-COVERAGE-RECONCILED";

fn merge(
    inventory: &mut BTreeMap<String, String>,
    source: &Value,
) -> Result<()> {
    let entries = source["verified_output_sha256"]
        .as_object().context("preceding reconciliation inventory")?;
    ensure!(!entries.is_empty(), "empty preceding reconciliation inventory");
    for (relative, digest) in entries {
        safe_relative(relative)?;
        let digest = digest.as_str().context("inventory digest string")?;
        ensure!(
            crate::contract::digest_valid(digest),
            "invalid reconciliation inventory digest"
        );
        if let Some(previous) = inventory.get(relative) {
            ensure!(
                previous == digest,
                "overlapping reconciliation inventories disagree"
            );
        } else {
            inventory.insert(relative.clone(), digest.to_owned());
        }
    }
    Ok(())
}

/// Build every expected directory from the independently validated file set.
fn directories(inventory: &BTreeMap<String, String>) -> Result<BTreeSet<String>> {
    let mut result = BTreeSet::from([String::new()]);
    for relative in inventory.keys() {
        safe_relative(relative)?;
        let mut parent = Path::new(relative).parent();
        while let Some(path) = parent {
            let name = path.to_str().context("non-UTF8 expected directory")?;
            if name.is_empty() {
                break;
            }
            ensure!(
                !inventory.contains_key(name),
                "inventory path is both file and directory"
            );
            result.insert(name.to_owned());
            parent = path.parent();
        }
    }
    Ok(result)
}

/// Compare each directory's immediate entry names and types.
///
/// Unknown directories are rejected before traversal. Thus an unexpected
/// directory cannot supply its own recursively discovered expected contents.
fn coverage(out: &Path, inventory: &BTreeMap<String, String>) -> Result<()> {
    ensure!(!inventory.is_empty(), "empty combined inventory");
    let expected_directories = directories(inventory)?;
    regular_directory(out)?;

    for relative in &expected_directories {
        let root = if relative.is_empty() {
            out.to_owned()
        } else {
            out.join(relative)
        };
        regular_directory(&root)?;
        let mut expected = BTreeMap::<String, bool>::new();

        for file in inventory.keys() {
            let path = Path::new(file);
            let parent = path.parent().context("inventory file parent")?;
            if parent == Path::new(relative) {
                let name = path.file_name().and_then(|n| n.to_str())
                    .context("inventory filename")?;
                ensure!(
                    expected.insert(name.to_owned(), false).is_none(),
                    "duplicate expected entry"
                );
            }
        }
        for directory in &expected_directories {
            if directory.is_empty() {
                continue;
            }
            let path = Path::new(directory);
            if path.parent() == Some(Path::new(relative)) {
                let name = path.file_name().and_then(|n| n.to_str())
                    .context("expected directory name")?;
                ensure!(
                    expected.insert(name.to_owned(), true).is_none(),
                    "file/directory entry collision"
                );
            }
        }

        let mut actual = BTreeMap::new();
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            ensure!(
                kind.is_file() || kind.is_dir(),
                "symlink or special file in acceptance output"
            );
            let name = entry.file_name().into_string()
                .map_err(|_| anyhow::anyhow!("non-UTF8 output entry"))?;
            ensure!(
                actual.insert(name, kind.is_dir()).is_none(),
                "duplicate output entry"
            );
        }
        ensure!(
            actual == expected,
            "combined output coverage mismatch in {}",
            root.display()
        );
    }
    Ok(())
}

fn stable(out: &Path, inventory: &BTreeMap<String, String>) -> Result<()> {
    for (relative, digest) in inventory {
        // Validate ancestors and final type on every closing read.
        let path = output_path(out, relative)?;
        ensure!(
            sha(&bounded(&path, MANIFEST_LIMIT)?) == *digest,
            "combined output artifact changed: {relative}"
        );
    }
    Ok(())
}

fn external_entry(path: &Path, digest: &str, limit: usize) -> Result<Value> {
    ensure!(
        crate::contract::digest_valid(digest)
            && sha(&regular_external_bytes(path, limit)?) == digest,
        "closing external launch artifact changed"
    );
    Ok(json!({"path": path, "sha256": digest}))
}

pub(crate) fn reconcile(
    continuation: &SchedulerArtifactsContinuation,
) -> Result<Value> {
    let independent = &continuation.independent_artifacts;
    ensure!(
        super::saved_scheduler::reconcile(independent)?
            == continuation.scheduler_reconciliation,
        "preceding scheduler reconciliation changed"
    );

    let poison = &independent.poisoned_launch;
    let owners = &poison.destructive_owners;
    let recovery_http = &owners.recovery_http;
    let lineage = &recovery_http.recovery_lineage;
    let initial = &lineage.initial_artifacts;
    let imports = &initial.imported_artifacts;
    let local = &imports.local_artifacts;
    let state = &local.final_drain.state;
    let old = &state.recovery.ready.poison_exit.old;
    let core_state = &old.core;
    let loaded = &core_state.loaded;
    let out = &core_state.out;
    loaded.recheck()?;

    ensure!(
        core_state.ledger.pending() == vec![json!({
            "acceptance_phase": policy::LIFECYCLE,
            "case_id": "final-artifact-and-manifest-reconciliation"
        })],
        "combined coverage called at wrong ledger boundary"
    );
    let mut inventory = BTreeMap::new();
    let sources = [
        &local.local_reconciliation,
        &imports.import_reconciliation,
        &initial.initial_launch_reconciliation,
        &lineage.recovery_lineage_reconciliation,
        &recovery_http.recovery_http_reconciliation,
        &owners.destructive_owners_reconciliation,
        &poison.poisoned_launch_reconciliation,
        &independent.independent_reconciliation,
        &continuation.scheduler_reconciliation,
    ];
    for source in sources {
        ensure!(
            source["policy_sha256"] == loaded.policy_sha256
                && source["preceding_ledger_sha256"] == sha(&state.ledger_prefix)
                && source["m4_accepted"] == false,
            "preceding reconciliation policy/ledger/scope mismatch"
        );
        merge(&mut inventory, source)?;
    }

    // These two files are immutable at this boundary, but the ledger will
    // deliberately receive its final case in the subsequent closeout.
    expect_bytes(
        out, "accept_records.jsonl", &state.ledger_prefix,
        MANIFEST_LIMIT, &mut inventory,
    )?;
    expect_bytes(
        out, "orchestration.jsonl", &state.orchestration_prefix,
        JSON_LIMIT, &mut inventory,
    )?;

    coverage(out, &inventory)?;
    stable(out, &inventory)?;

    let initial_launch = &core_state.initial_launch;
    let recovery_launch = &state.recovery.ready.replacement;
    let initial_summary = &initial.initial_launch_reconciliation["initial_launch"];
    let scheduler_summary = &continuation.scheduler_reconciliation;
    let lineage_summary = &lineage.recovery_lineage_reconciliation;
    let external_specs = [
        (
            &initial_launch.attestation,
            text(initial_summary, "attestation_sha256")?,
            JSON_LIMIT,
        ),
        (
            &initial_launch.manifest,
            text(initial_summary, "final_manifest_sha256")?,
            MANIFEST_LIMIT,
        ),
        (
            &old.launch.attestation,
            text(scheduler_summary, "scheduler_attestation_sha256")?,
            JSON_LIMIT,
        ),
        (
            &old.launch.manifest,
            text(&scheduler_summary["coalescing"], "manifest_sha256")?,
            MANIFEST_LIMIT,
        ),
        (
            &recovery_launch.attestation,
            text(lineage_summary, "recovery_attestation_sha256")?,
            JSON_LIMIT,
        ),
        (
            &recovery_launch.manifest,
            text(lineage_summary, "recovery_final_manifest_sha256")?,
            MANIFEST_LIMIT,
        ),
    ];
    let mut external = Vec::new();
    for (path, digest, limit) in external_specs {
        external.push(external_entry(path, digest, limit)?);
    }

    // Repeat the full offline validation chain against original commitments.
    // Do not invoke live-boundary consumers or remeasure process absence.
    ensure!(
        super::saved_scheduler::reconcile(independent)?
            == continuation.scheduler_reconciliation,
        "preceding reconciliations changed during combined coverage"
    );
    loaded.recheck()?;
    stable(out, &inventory)?;
    coverage(out, &inventory)?;

    for ((path, digest, limit), expected) in external_specs.into_iter().zip(&external) {
        ensure!(
            external_entry(path, digest, limit)? == *expected,
            "external closing inventory changed"
        );
    }
    ensure!(
        bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)?
            == state.ledger_prefix
            && bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)?
                == state.orchestration_prefix,
        "closing ledger/transcript changed"
    );

    let source_digests: Vec<Value> = sources.into_iter().map(|source| {
        Ok(json!({
            "schema": text(source, "schema")?,
            "boundary": text(source, "boundary")?,
            "reconciliation_sha256": sha(&serde_json::to_vec(source)?)
        }))
    }).collect::<Result<_>>()?;
    let diagnostic_files = owners.destructive_owners_reconciliation[
        "parent_diagnostics_without_original_byte_commitments"
    ].as_array().context("parent diagnostic classification")?;
    let mut diagnostics = diagnostic_files.clone();
    diagnostics.push(json!("independent/service_log.jsonl"));

    Ok(json!({
        "schema": "rustv1.accept_m4_combined_output_reconciliation.v1",
        "boundary": COVERAGE_BOUNDARY,
        "policy_sha256": loaded.policy_sha256,
        "preceding_ledger_sha256": sha(&state.ledger_prefix),
        "orchestration_sha256": sha(&state.orchestration_prefix),
        "sources": source_digests,
        "verified_output_sha256": inventory,
        "verified_external_launch_artifacts": external,
        "output_file_count": inventory.len(),
        "output_directory_count_including_root": directories(&inventory)?.len(),
        "expected_file_set_source": "union_of_independently_reconciled_producer_files",
        "directory_enumeration_used_as_expected_file_source": false,
        "complete_output_directory_coverage": true,
        "unexpected_empty_directories_rejected": true,
        "symlinks_and_special_files_rejected": true,
        "overlapping_inventory_digests_checked": true,
        "closing_output_byte_stability_checked": true,
        "closing_external_launch_byte_stability_checked": true,
        "closing_policy_and_bound_inputs_rechecked": true,
        "whole_transcript_semantics_reconciled": true,
        "all_artifact_prerequisites_reconciled": true,
        "diagnostic_files_without_original_byte_commitments": diagnostics,
        "diagnostic_hashes_are_original_historical_commitments": false,
        "http_marker_original_digest_recovered": false,
        "historical_observations_relabelled": false,
        "unjoined_owner_closures_relabelled": false,
        "independent_connection_joins_manufactured": false,
        "post_exit_control_calls": 0,
        "process_observations": 0,
        "owner_joins_performed": 0,
        "child_reaps_performed": 0,
        "ledger_rows_appended": 0,
        "final_artifact_reconciliation_ledger_case_appended": false,
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
    fn combined_inventory_requires_agreement_and_rejects_path_collisions() {
        let digest = sha(b"original");
        let source = json!({
            "verified_output_sha256": {"a/b": digest}
        });
        let mut inventory = BTreeMap::new();
        merge(&mut inventory, &source).unwrap();
        merge(&mut inventory, &source).unwrap();
        assert_eq!(inventory.len(), 1);

        let changed = json!({
            "verified_output_sha256": {"a/b": sha(b"changed")}
        });
        assert!(merge(&mut inventory, &changed).is_err());
        let traversal = json!({
            "verified_output_sha256": {"../outside": digest}
        });
        assert!(merge(&mut inventory, &traversal).is_err());

        inventory.insert("a".into(), digest);
        assert!(directories(&inventory).is_err());
    }

    #[test]
    fn combined_coverage_rejects_missing_extra_empty_directory_symlink_and_mutation() {
        let root = std::env::temp_dir().join(format!(
            "rv1-combined-coverage-{}-{}",
            std::process::id(), core::random_label().unwrap()
        ));
        private_directory(&root).unwrap();
        let result: Result<()> = (|| {
            private_directory(&root.join("scheduler"))?;
            private_directory(&root.join("scheduler/probe"))?;
            write_new(&root.join("ledger.jsonl"), b"{}\n")?;
            write_new(&root.join("scheduler/probe/evidence.json"), b"{}")?;
            let inventory = BTreeMap::from([
                ("ledger.jsonl".to_owned(), sha(b"{}\n")),
                ("scheduler/probe/evidence.json".to_owned(), sha(b"{}")),
            ]);
            coverage(&root, &inventory)?;
            stable(&root, &inventory)?;

            let path = root.join("scheduler/probe/evidence.json");
            let hidden = root.join("scheduler/probe/hidden");
            fs::rename(&path, &hidden)?;
            let missing = coverage(&root, &inventory);
            fs::rename(&hidden, &path)?;
            ensure!(missing.is_err(), "missing file accepted");

            let extra = root.join("extra");
            write_new(&extra, b"x")?;
            let additional = coverage(&root, &inventory);
            fs::remove_file(&extra)?;
            ensure!(additional.is_err(), "extra root file accepted");

            let empty = root.join("scheduler/empty");
            private_directory(&empty)?;
            let directory = coverage(&root, &inventory);
            fs::remove_dir(&empty)?;
            ensure!(directory.is_err(), "unexpected empty directory accepted");

            #[cfg(unix)]
            {
                let link = root.join("scheduler/linked");
                std::os::unix::fs::symlink(&path, &link)?;
                let linked = coverage(&root, &inventory);
                fs::remove_file(&link)?;
                ensure!(linked.is_err(), "symlink accepted");

                let original_dir = root.join("scheduler/probe");
                let moved_dir = root.join("saved-probe");
                fs::rename(&original_dir, &moved_dir)?;
                std::os::unix::fs::symlink(&moved_dir, &original_dir)?;
                let ancestor_link = stable(&root, &inventory);
                fs::remove_file(&original_dir)?;
                fs::rename(&moved_dir, &original_dir)?;
                ensure!(ancestor_link.is_err(), "symlink ancestor accepted");
            }

            fs::write(&path, b"{\"changed\":true}")?;
            let changed = stable(&root, &inventory);
            fs::write(&path, b"{}")?;
            ensure!(changed.is_err(), "closing mutation accepted");

            stable(&root, &inventory)?;
            coverage(&root, &inventory)
        })();
        if result.is_ok() {
            fs::remove_dir_all(&root).unwrap();
        }
        result.unwrap_or_else(|error| {
            panic!(
                "combined coverage failed; retained {}: {error:#}",
                root.display()
            )
        });
    }
}

#[cfg(feature = "scheduler-test-seam")]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) mod closeout {
    include!("accept_m4_closeout.rs");
}
