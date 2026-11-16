use super::*;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Component,
};

const RECEIPT: &str = "cpu_harness_receipt.json";
const PLAN: &str = "cpu_harness_plan.json";
const RECORDS: &str = "cpu_harness_records.jsonl";

const LOCAL_CASES: [&str; 13] = [
    "family-54-physical-b-capacity",
    "family-54-padded-token-capacity",
    "family-54-attention-cell-capacity",
    "family-54-fifo-no-split-no-bypass",
    "family-55-queue-saturation",
    "family-55-queued-expiry-zero-execution",
    "family-55-batching-wait-bound",
    "family-56-invalid-last-isolation",
    "family-56-pad-and-oov-token-gates",
    "family-57-quiesce-assembled-work",
    "family-57-drain-running-work",
    "family-58-physical-failure-fanout",
    "family-58-running-expiry-retirement",
];

const LIFECYCLE_CASES: [&str; 10] = [
    "replacement-launch-after-independent-join",
    "family-58-live-policy-drift-poison",
    "family-58-terminal-liveness-readiness-refusal",
    "family-58-restore-does-not-recover",
    "family-58-authenticated-supervised-restart",
    "family-58-old-process-exit",
    "family-58-replacement-reattestation",
    "family-58-exact-recovery",
    "final-authenticated-drain-and-exit",
    "final-artifact-and-manifest-reconciliation",
];

fn classification(phase: &str, case: &str) -> Result<(&'static str, Option<&'static str>)> {
    match phase {
        policy::PREPASS => Ok((
            "validated_external_evidence_not_executed",
            Some("requires-cuda-producer"),
        )),
        policy::POLICY => {
            ensure!(
                matches!(case, "finalized-policy-bindings" | "prepass-evidence-consumed"),
                "unclassified policy case"
            );
            Ok(("production_validator_executed", None))
        }
        policy::LAUNCH | policy::EXIT => Ok((
            "excluded",
            Some("requires-real-supervised-production-launch"),
        )),
        policy::HTTP => Ok((
            "excluded",
            Some("requires-real-http-producer"),
        )),
        policy::INDEPENDENT | policy::JOIN => Ok((
            "excluded",
            Some("requires-cuda"),
        )),
        policy::SCHEDULER if LOCAL_CASES.contains(&case) => {
            Ok(("local_projection_executed", None))
        }
        policy::SCHEDULER => match case {
            "family-53-live-cross-request-coalescing" => {
                Ok(("excluded", Some("requires-real-http-producer")))
            }
            "scheduler-owners-joined" => {
                Ok(("excluded", Some("requires-complete-production-owner-inventory")))
            }
            _ => anyhow::bail!("unclassified scheduler case"),
        },
        policy::LIFECYCLE => {
            ensure!(LIFECYCLE_CASES.contains(&case), "unclassified lifecycle case");
            let reason = match case {
                "family-58-live-policy-drift-poison" | "family-58-exact-recovery" => {
                    "requires-cuda"
                }
                "final-artifact-and-manifest-reconciliation" => {
                    "requires-complete-production-evidence-and-cuda-closeout"
                }
                _ => "requires-real-production-launch-lineage",
            };
            Ok(("excluded", Some(reason)))
        }
        _ => anyhow::bail!("unclassified required phase"),
    }
}

fn plan(required: &policy::Required) -> Result<Vec<Value>> {
    required.validate()?;
    let scheduler = required.phases.iter()
        .find(|(phase, _)| phase == policy::SCHEDULER)
        .context("missing scheduler phase")?;
    let expected_scheduler: Vec<String> =
        std::iter::once("family-53-live-cross-request-coalescing")
            .chain(LOCAL_CASES)
            .chain(std::iter::once("scheduler-owners-joined"))
            .map(str::to_owned)
            .collect();
    ensure!(
        scheduler.1 == expected_scheduler,
        "CPU projection scheduler inventory changed"
    );
    let lifecycle = required.phases.iter()
        .find(|(phase, _)| phase == policy::LIFECYCLE)
        .context("missing lifecycle phase")?;
    ensure!(
        lifecycle.1 == LIFECYCLE_CASES.map(str::to_owned),
        "CPU projection lifecycle inventory changed"
    );

    required.phases.iter().flat_map(|(phase, cases)| {
        cases.iter().map(move |case| {
            let (disposition, reason) = classification(phase, case)?;
            Ok(json!({
                "acceptance_phase": phase,
                "case_id": case,
                "disposition": disposition,
                "reason_class": reason,
                "production_acceptance_case_satisfied": false
            }))
        })
    }).collect()
}

fn connection_index(records: &[Value]) -> Result<BTreeMap<u64, Value>> {
    let mut result = BTreeMap::new();
    for record in records {
        ensure!(
            record["join_awaited"] == true
                && matches!(
                    record["join_outcome"].as_str(),
                    Some("completed" | "cancelled-after-abort-and-awaited")
                ),
            "CPU probe connection not joined"
        );
        ensure!(
            result.insert(uint(record, "connection_id")?, record.clone()).is_none(),
            "duplicate CPU probe connection"
        );
    }
    Ok(result)
}

pub(super) async fn local_receipt(
    root: &Path,
    relative_root: &str,
    evidence: &Value,
) -> Result<Value> {
    let files = super::saved_local::verify_probe(root, evidence)?;
    let instance = text(evidence, "instance")?;
    let mut responses: Vec<&Value> = evidence["responses"]
        .as_array().context("CPU probe responses")?.iter().collect();
    if let Some(rejections) = evidence.get("rejections") {
        responses.extend(rejections.as_array().context("CPU rejections")?);
    }
    if let Some(refusal) = evidence["witness"]
        .get("admission_refusal")
        .filter(|value| !value.is_null())
    {
        responses.push(refusal);
    }
    let expected: Vec<Value> = responses.iter()
        .map(|response| response["headers"]["_connection_owner"].clone())
        .collect();
    let actual = http::join_connections(instance).await?;
    ensure!(
        actual.iter().all(|record| record["instance"] == instance)
            && connection_index(&expected)? == connection_index(&actual)?,
        "CPU probe actual connection inventory differs from responses"
    );
    ensure!(
        super::saved_local::verify_probe(root, evidence)? == files,
        "CPU local probe changed during connection closure"
    );
    Ok(json!({
        "relative_root": relative_root,
        "instance": instance,
        "evidence_sha256": file_sha(&root.join("evidence.json"))?,
        "output_sha256": files,
        "http_connections": actual,
        "external_live_server_isolation_checked": false,
        "numerical_model_execution": false
    }))
}

fn safe_relative(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && Path::new(name).components()
                .all(|part| matches!(part, Component::Normal(_))),
        "unsafe CPU output inventory path"
    );
    Ok(())
}

fn merge_files(
    inventory: &mut BTreeMap<String, String>,
    prefix: &str,
    files: &Value,
) -> Result<()> {
    for (name, digest) in files.as_object().context("CPU file inventory")? {
        let relative = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        safe_relative(&relative)?;
        let digest = digest.as_str().context("CPU inventory digest")?;
        ensure!(
            crate::contract::digest_valid(digest)
                && inventory.insert(relative, digest.to_owned()).is_none(),
            "CPU inventory collision or invalid digest"
        );
    }
    Ok(())
}

fn verify_tree(root: &Path, inventory: &BTreeMap<String, String>) -> Result<()> {
    let mut directories = BTreeSet::from([String::new()]);
    for name in inventory.keys() {
        safe_relative(name)?;
        let mut parent = Path::new(name).parent();
        while let Some(path) = parent {
            let name = path.to_str().context("CPU directory UTF-8")?;
            if name.is_empty() {
                break;
            }
            ensure!(!inventory.contains_key(name), "CPU file/directory collision");
            directories.insert(name.to_owned());
            parent = path.parent();
        }
    }

    for directory in &directories {
        let path = root.join(directory);
        ensure!(
            fs::symlink_metadata(&path)?.file_type().is_dir(),
            "CPU output directory is not a non-symlink directory"
        );
        let mut expected = BTreeMap::new();
        for (names, is_directory) in [
            (inventory.keys().collect::<Vec<_>>(), false),
            (directories.iter().filter(|name| !name.is_empty()).collect(), true),
        ] {
            for name in names {
                let child = Path::new(name);
                if child.parent() == Some(Path::new(directory)) {
                    let basename = child.file_name().and_then(|name| name.to_str())
                        .context("CPU basename")?;
                    ensure!(
                        expected.insert(basename.to_owned(), is_directory).is_none(),
                        "duplicate CPU output entry"
                    );
                }
            }
        }
        let mut actual = BTreeMap::new();
        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            ensure!(kind.is_file() || kind.is_dir(), "CPU symlink or special file");
            let name = entry.file_name().into_string()
                .map_err(|_| anyhow::anyhow!("CPU filename UTF-8"))?;
            ensure!(actual.insert(name, kind.is_dir()).is_none(), "duplicate entry");
        }
        ensure!(actual == expected, "CPU output coverage mismatch");
    }
    for (name, digest) in inventory {
        ensure!(file_sha(&root.join(name))? == *digest, "CPU output changed: {name}");
    }
    Ok(())
}

fn record_file(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

async fn local_case(loaded: &Loaded, root: &Path, case: &str) -> Result<Value> {
    match case {
        "family-54-physical-b-capacity" => {
            let evidence = super::local(loaded, root).await?;
            save_json(&root.join("evidence.json"), &evidence)?;
            Ok(json!({
                "probes": [local_receipt(root, "", &evidence).await?]
            }))
        }
        "family-54-padded-token-capacity"
        | "family-54-attention-cell-capacity"
        | "family-54-fifo-no-split-no-bypass" => {
            super::capacity::cpu_case(loaded, root, case).await
        }
        "family-55-queue-saturation"
        | "family-55-queued-expiry-zero-execution"
        | "family-55-batching-wait-bound" => {
            super::family55::cpu_case(loaded, root, case).await
        }
        "family-56-invalid-last-isolation"
        | "family-56-pad-and-oov-token-gates"
        | "family-57-quiesce-assembled-work" => {
            super::family56_57::cpu_case(loaded, root, case).await
        }
        "family-57-drain-running-work"
        | "family-58-physical-failure-fanout"
        | "family-58-running-expiry-retirement" => {
            super::family57_58::cpu_case(loaded, root, case).await
        }
        _ => anyhow::bail!("unknown CPU local case"),
    }
}

async fn execute(
    loaded: &Loaded,
    out: &Path,
    planned: &[Value],
    inventory: &mut BTreeMap<String, String>,
) -> Result<Vec<Value>> {
    let mut file = record_file(&out.join(RECORDS))?;
    let mut expected_bytes = Vec::new();
    let mut records = Vec::new();

    for entry in planned {
        let phase = text(entry, "acceptance_phase")?;
        let case = text(entry, "case_id")?;
        let disposition = text(entry, "disposition")?;
        let details = match disposition {
            "validated_external_evidence_not_executed" => {
                let source = loaded.prepass_records.iter()
                    .find(|record| record["case_id"] == case)
                    .context("missing policy-consumed prepass record")?;
                json!({
                    "producer_record_sha256": sha(&serde_json::to_vec(source)?),
                    "digest_scope": "serialized_loaded_record_not_original_jsonl_bytes",
                    "producer_evidence_sha256":
                        loaded.artifacts["batch_in_process_evidence"].sha256,
                    "numerical_execution_performed_by_harness": false
                })
            }
            "production_validator_executed" => {
                loaded.recheck()?;
                loaded.binding_summary()
            }
            "local_projection_executed" => {
                ensure!(phase == policy::SCHEDULER, "CPU projection phase");
                loaded.recheck()?;
                envelope(loaded)?;
                let details = local_case(loaded, &out.join(case), case).await?;
                if let Some(files) = details.get("output_sha256") {
                    merge_files(inventory, case, files)?;
                } else {
                    for probe in details["probes"].as_array().context("CPU probes")? {
                        let relative = text(probe, "relative_root")?;
                        let prefix = if relative.is_empty() {
                            case.to_owned()
                        } else {
                            format!("{case}/{relative}")
                        };
                        merge_files(inventory, &prefix, &probe["output_sha256"])?;
                    }
                }
                loaded.recheck()?;
                details
            }
            "excluded" => json!({"execution_performed": false}),
            _ => anyhow::bail!("unknown CPU disposition"),
        };
        let record = json!({
            "kind": "cpu_projection_case",
            "plan": entry,
            "details": details,
            "production_acceptance_case_satisfied": false
        });
        let mut bytes = serde_json::to_vec(&record)?;
        bytes.push(b'\n');
        file.write_all(&bytes)?;
        file.sync_data()?;
        expected_bytes.extend_from_slice(&bytes);
        records.push(record);
    }
    file.sync_all()?;
    ensure!(
        bounded(&out.join(RECORDS), MANIFEST_LIMIT)? == expected_bytes
            && core::jsonl(&expected_bytes)? == records,
        "CPU record readback mismatch"
    );
    inventory.insert(RECORDS.to_owned(), sha(&expected_bytes));
    Ok(records)
}

pub(crate) fn run(
    model: &Path,
    input: &Path,
    input_sha256: &str,
    policy_path: &Path,
    out: &Path,
) -> Result<Value> {
    let loaded = policy::load(model, input, input_sha256, policy_path)?;
    let planned = plan(&loaded.required)?;
    let exclusions: Vec<Value> = planned.iter()
        .filter(|entry| entry["reason_class"].is_string())
        .cloned().collect();

    private_directory(out)?;
    let out = out.canonicalize()?;
    let declaration = json!({
        "schema": "rustv1.accept_m4_cpu_projection_plan.v1",
        "policy_sha256": loaded.policy_sha256,
        "required": loaded.required.as_json(),
        "cases": planned,
        "exclusions": exclusions,
        "exclusion_scope": "production_case_execution_in_this_harness",
        "production_acceptance": false
    });
    let plan_digest = save_json(&out.join(PLAN), &declaration)?;
    let mut inventory = BTreeMap::from([(PLAN.to_owned(), plan_digest)]);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let result = runtime.block_on(execute(&loaded, &out, &planned, &mut inventory));
    runtime.shutdown_timeout(Duration::from_secs(1));
    let records = result?;

    ensure!(
        records.len() == loaded.required.count()
            && records.iter().filter(|record| {
                record["plan"]["disposition"] == "local_projection_executed"
            }).count() == LOCAL_CASES.len(),
        "CPU projection coverage mismatch"
    );
    loaded.recheck()?;
    verify_tree(&out, &inventory)?;

    let receipt = json!({
        "schema": "rustv1.accept_m4_cpu_projection_receipt.v1",
        "accepted": false,
        "production_acceptance": false,
        "harness_checks_passed": true,
        "scope": "policy_loaded_cpu_projection_not_production_full_acceptance",
        "binding_summary": loaded.binding_summary(),
        "required_case_count": loaded.required.count(),
        "classified_case_count": records.len(),
        "local_projection_case_count": LOCAL_CASES.len(),
        "production_acceptance_cases_satisfied": 0,
        "exclusions": exclusions,
        "exclusion_scope": "production_case_execution_in_this_harness",
        "numerical_execution_performed_by_harness": false,
        "external_live_server_isolation_checked": false,
        "global_acceptance_ledger_created": false,
        "global_ledger_finish_performed": false,
        "production_completion_marker_written": false,
        "runtime_drop_counted_as_owner_join": false,
        "output_sha256": inventory,
        "output_inventory_excludes_only": [RECEIPT]
    });
    let digest = save_json(&out.join(RECEIPT), &receipt)?;
    inventory.insert(RECEIPT.to_owned(), digest.clone());
    loaded.recheck()?;
    verify_tree(&out, &inventory)?;
    ensure!(
        value(&out.join(RECEIPT))? == receipt,
        "CPU receipt readback differs"
    );
    Ok(json!({
        "kind": "cpu_harness_done",
        "accepted": false,
        "production_acceptance": false,
        "harness_checks_passed": true,
        "receipt_path": out.join(RECEIPT),
        "receipt_sha256": digest,
        "records_sha256": inventory[RECORDS],
        "policy_sha256": loaded.policy_sha256
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn required() -> policy::Required {
        policy::Required {
            phases: vec![
                (
                    policy::POLICY.to_owned(),
                    vec![
                        "finalized-policy-bindings".to_owned(),
                        "prepass-evidence-consumed".to_owned(),
                    ],
                ),
                (
                    policy::SCHEDULER.to_owned(),
                    std::iter::once("family-53-live-cross-request-coalescing")
                        .chain(LOCAL_CASES)
                        .chain(std::iter::once("scheduler-owners-joined"))
                        .map(str::to_owned)
                        .collect(),
                ),
                (
                    policy::LIFECYCLE.to_owned(),
                    LIFECYCLE_CASES.map(str::to_owned).to_vec(),
                ),
            ],
        }
    }

    #[test]
    fn cpu_projection_classifies_every_case_without_acceptance() {
        let required = required();
        let entries = plan(&required).unwrap();
        assert_eq!(entries.len(), required.count());
        assert_eq!(
            entries.iter().filter(|entry| {
                entry["disposition"] == "local_projection_executed"
            }).count(),
            13
        );
        assert_eq!(
            entries.iter().filter(|entry| entry["disposition"] == "excluded").count(),
            12
        );
        assert!(entries.iter().all(|entry| {
            entry["production_acceptance_case_satisfied"] == false
                && entry.get("m4_accepted").is_none()
        }));
        assert!(classification("unknown", "case").is_err());
        assert!(classification(policy::SCHEDULER, "unknown").is_err());
    }

    #[test]
    fn cpu_projection_rejects_changed_required_inventory() {
        let mut changed = required();
        changed.phases[1].1.swap(1, 2);
        assert!(plan(&changed).is_err());
        let mut changed = required();
        changed.phases[2].1.push("new-case".to_owned());
        assert!(plan(&changed).is_err());
    }

    #[test]
    fn cpu_projection_connection_inventory_rejects_unjoined_and_duplicates() {
        let joined = json!({
            "connection_id": 1,
            "join_awaited": true,
            "join_outcome": "completed"
        });
        assert!(connection_index(std::slice::from_ref(&joined)).is_ok());
        assert!(connection_index(&[joined.clone(), joined.clone()]).is_err());
        let mut unjoined = joined;
        unjoined["join_awaited"] = json!(false);
        assert!(connection_index(&[unjoined]).is_err());
    }

    #[test]
    fn cpu_projection_tree_rejects_extra_missing_and_changed_files() {
        let root = std::env::temp_dir().join(format!(
            "rv1-cpu-tree-{}-{}",
            std::process::id(),
            core::random_label().unwrap()
        ));
        private_directory(&root).unwrap();
        private_directory(&root.join("probe")).unwrap();
        write_new(&root.join("probe/evidence.json"), b"{}").unwrap();
        let inventory = BTreeMap::from([
            ("probe/evidence.json".to_owned(), sha(b"{}")),
        ]);
        verify_tree(&root, &inventory).unwrap();
        write_new(&root.join("extra"), b"x").unwrap();
        assert!(verify_tree(&root, &inventory).is_err());
        fs::remove_file(root.join("extra")).unwrap();
        fs::write(root.join("probe/evidence.json"), b"changed").unwrap();
        assert!(verify_tree(&root, &inventory).is_err());
        fs::remove_file(root.join("probe/evidence.json")).unwrap();
        assert!(verify_tree(&root, &inventory).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
