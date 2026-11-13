// Included at:
// acceptance_scheduler::lifecycle_prefix::restart_prefix::reattestation::
// exact_recovery::final_drain::saved_lineage::saved_http
//
// Offline reconciliation only.
//
// Do not call consume_recovery, consume_ready, live_cursor, unchanged,
// check_live, prove_absent, join_connections, or any control/orchestration
// exchange here.
//
// Coverage is deliberately scoped to three output subtrees:
// - scheduler/family-58-replacement-reattestation
// - scheduler/family-58-exact-recovery
// - scheduler/final-authenticated-drain-and-exit
//
// This is not combined acceptance-output coverage.

use super::*;

const HTTP_BOUNDARY: &str = "RECOVERY-HTTP-ARTIFACTS-RECONCILED";

/// Validate interval bounds and commitments before slicing.
///
/// No newly selected manifest baseline is accepted. The caller supplies the
/// retained final manifest and advances expected_start from retained startup.
fn interval_bytes<'a>(
    manifest: &'a [u8],
    commitment: &Value,
    expected_start: usize,
) -> Result<&'a [u8]> {
    let start = usize::try_from(uint(commitment, "start_byte")?)?;
    let end = usize::try_from(uint(commitment, "end_byte")?)?;
    ensure!(
        start == expected_start && start < end && end <= manifest.len(),
        "recovery HTTP interval gap, overlap, empty interval or invalid bounds"
    );
    ensure!(
        commitment["prefix_before_sha256"] == sha(&manifest[..start])
            && commitment["prefix_after_sha256"] == sha(&manifest[..end])
            && commitment["interval_sha256"] == sha(&manifest[start..end]),
        "recovery HTTP interval commitment changed"
    );
    Ok(&manifest[start..end])
}

/// Validate the saved response's correlation and actual connection-join
/// record. This performs no connection registry lookup or owner join.
fn response_binding(
    response: &Value,
    instance: &str,
    case: &str,
    request_id: u64,
    request_digest: &str,
) -> Result<Value> {
    closed(
        response,
        &[
            "case_id",
            "http_status",
            "request_id",
            "headers",
            "response_path",
            "response_sha256",
        ],
    )?;
    ensure!(
        response["case_id"] == case
            && response["http_status"] == 200
            && response["request_id"] == request_id
            && response["response_path"] == format!("{case}.response.bin")
            && crate::contract::digest_valid(text(response, "response_sha256")?),
        "recovery saved response identity/status/path"
    );

    let headers = &response["headers"];
    closed(
        headers,
        &[
            "x-rv1-instance-id",
            "x-acceptance-case-id",
            "x-rv1-request-id",
            "content-type",
            "_connection_owner",
        ],
    )?;
    ensure!(
        headers["x-rv1-instance-id"] == instance
            && headers["x-acceptance-case-id"] == case
            && headers["x-rv1-request-id"] == request_id
            && text(headers, "content-type")?
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("application/json"),
        "recovery saved HTTP correlation/media"
    );

    let connection = &headers["_connection_owner"];
    ensure!(
        connection["instance"] == instance
            && connection["owner_pid"] == std::process::id()
            && connection["case_id"] == case
            && connection["method"] == "POST"
            && connection["path"] == "/v1/embeddings"
            && connection["request_sha256"] == request_digest
            && connection["join_awaited"] == true
            && matches!(
                connection["join_outcome"].as_str(),
                Some("completed" | "cancelled-after-abort-and-awaited")
            ),
        "recovery saved connection ownership/request/join binding"
    );
    uint(connection, "connection_id")?;
    Ok(connection.clone())
}

/// Exact flat-directory coverage.
///
/// Files::path checks all ancestor components as non-symlink directories.
/// The evidence file is mandatory in each subtree covered here.
fn flat_coverage(files: &Files<'_>, relative: &str) -> Result<()> {
    files.path(&format!("{relative}/evidence.json"))?;
    let root = files.root.join(relative);
    ensure!(
        fs::symlink_metadata(&root)?.file_type().is_dir(),
        "recovery output subtree is not a non-symlink directory"
    );

    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        ensure!(
            entry.file_type()?.is_file(),
            "recovery output subtree contains a symlink or non-file"
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 recovery output filename"))?;
        ensure!(actual.insert(name), "duplicate recovery directory entry");
    }

    let prefix = format!("{relative}/");
    let mut expected = BTreeSet::new();
    for path in files.hashes.keys() {
        if let Some(name) = path.strip_prefix(&prefix) {
            ensure!(
                Path::new(name).components().count() == 1
                    && matches!(
                        Path::new(name).components().next(),
                        Some(Component::Normal(_))
                    ),
                "non-flat recovery inventory entry"
            );
            ensure!(
                expected.insert(name.to_owned()),
                "duplicate recovery inventory filename"
            );
        }
    }
    ensure!(
        actual == expected,
        "missing or unaccounted recovery subtree artifact: {relative}"
    );
    Ok(())
}

/// Import only the preceding lineage inventory's entries belonging to a
/// named subtree. Each entry is read again through its verified digest.
///
/// These digests are reconciliation inventory values, not invented original
/// byte commitments for sidecars whose producer retained only JSON values.
fn import_lineage_subtree(
    files: &mut Files<'_>,
    lineage: &Value,
    relative: &str,
) -> Result<()> {
    let inventory = lineage["verified_output_sha256"]
        .as_object()
        .context("preceding recovery-lineage inventory")?;
    let prefix = format!("{relative}/");
    let mut count = 0usize;
    for (path, digest) in inventory {
        if path.starts_with(&prefix) {
            let digest = digest.as_str().context("lineage inventory digest")?;
            ensure!(
                crate::contract::digest_valid(digest),
                "invalid lineage inventory digest"
            );
            ensure!(
                sha(&files.read(path, MANIFEST_LIMIT)?) == digest,
                "preceding lineage subtree artifact changed"
            );
            count += 1;
        }
    }
    ensure!(count > 0, "missing preceding lineage subtree inventory");
    Ok(())
}

pub(crate) fn reconcile(
    continuation: &crate::acceptance_scheduler::RecoveryLaunchLineageContinuation,
) -> Result<Value> {
    // Revalidate all preceding offline prerequisites at their original
    // commitments, without invoking any live boundary consumer.
    ensure!(
        super::reconcile(&continuation.initial_artifacts)?
            == continuation.recovery_lineage_reconciliation,
        "preceding recovery-launch lineage reconciliation changed"
    );

    let state = &continuation
        .initial_artifacts
        .imported_artifacts
        .local_artifacts
        .final_drain
        .state;
    let recovery = &state.recovery;
    let ready = &recovery.ready;
    let old = &ready.poison_exit.old;
    let core_state = &old.core;
    let loaded = &core_state.loaded;
    let launch = &ready.replacement;
    let out = &core_state.out;

    loaded.recheck()?;
    ensure!(
        core_state.ledger.pending() == vec![json!({
            "acceptance_phase": policy::LIFECYCLE,
            "case_id": FINAL_ID
        })],
        "recovery HTTP reconciliation called at wrong ledger boundary"
    );

    let ledger = bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)?;
    let transcript = bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)?;
    ensure!(
        ledger == state.ledger_prefix
            && transcript == state.orchestration_prefix,
        "final retained ledger/transcript changed"
    );

    let mut files = Files::new(out)?;
    let saved = read_case(
        &mut files,
        &ledger,
        RECOVERY_ID,
        &recovery.recovery_evidence_sha256,
        loaded,
        launch,
    )?;
    let drained = read_case(
        &mut files,
        &ledger,
        DRAIN_ID,
        &state.drain_evidence_sha256,
        loaded,
        launch,
    )?;

    let root = format!("scheduler/{RECOVERY_ID}");
    let ready_root = format!("scheduler/{READY_ID}");
    let drain_root = format!("scheduler/{DRAIN_ID}");

    let bundle = &loaded.bundle;
    ensure!(
        bundle.fixtures.len() == 22
            && bundle.singleton.len() == 22
            && bundle.reference.len() == 22,
        "recovery frozen fixture coverage"
    );
    let requests = saved["requests"]
        .as_array()
        .context("saved recovery request array")?;
    ensure!(requests.len() == 44, "saved recovery request coverage");

    manifest_suffix(&launch.initial_manifest, &recovery.manifest_prefix)?;
    let manifest = &recovery.manifest_prefix;
    let mut cursor = launch.initial_manifest.len();
    let mut next = uint(&launch.startup, "next_forward_id")?;
    let mut observation = launch.initial_observation.clone();
    ensure!(
        ready_observation(&observation, launch)? == 0,
        "recovery startup baseline is not fresh"
    );

    let mut ids = BTreeSet::new();
    let mut last_id = None;
    let mut connection_ids = BTreeSet::new();
    let mut connections = Vec::new();
    let mut checked_requests = Vec::new();
    let mut previous_float: Option<Vec<Vec<f32>>> = None;

    for (position, proof) in requests.iter().enumerate() {
        let fixture = position / 2;
        let encoding = if position % 2 == 0 {
            previous_float = None;
            "float"
        } else {
            "base64"
        };
        let case = format!("sched58-recovery-{fixture}-{encoding}");
        let request_name = format!("{case}.request.bin");
        let response_name = format!("{case}.response.bin");
        let interval_name = format!("{case}.executor_interval.jsonl");

        let request_bytes = serde_json::to_vec(&json!({
            "model": MODEL_ID,
            "input": [bundle.fixtures[fixture].text],
            "encoding_format": encoding
        }))?;
        let request_digest = sha(&request_bytes);

        ensure!(
            proof["case_id"] == case
                && proof["fixture_index"] == fixture
                && proof["fixture_id"] == bundle.fixtures[fixture].id
                && proof["encoding"] == encoding
                && proof["request_path"] == request_name
                && proof["request_sha256"] == request_digest,
            "recovery request plan/order/commitment changed"
        );
        files.exact(&format!("{root}/{request_name}"), &request_bytes)?;

        let id = uint(proof, "request_id")?;
        ensure!(
            id > 0
                && ids.insert(id)
                && last_id.is_none_or(|previous| id > previous),
            "recovery request ID reused or reordered"
        );

        let response = &proof["response"];
        let connection = response_binding(
            response,
            &launch.instance,
            &case,
            id,
            &request_digest,
        )?;
        ensure!(
            connection_ids.insert(uint(&connection, "connection_id")?),
            "recovery connection identity reused"
        );

        let response_bytes = files.committed(
            &format!("{root}/{response_name}"),
            text(response, "response_sha256")?,
        )?;
        files.pretty(
            &format!("{root}/{case}.response.json"),
            response,
        )?;

        let reply = http::Reply {
            status: u16::try_from(uint(response, "http_status")?)?,
            request_id: Some(id),
            headers: response["headers"].clone(),
            bytes: response_bytes,
        };
        let (output, usage) =
            http::decode_success(&reply, bundle, &[fixture], encoding)?;
        let output_digest = exact_singleton(
            &output,
            &bundle.singleton[fixture],
            &bundle.fixtures[fixture].golden_sha256,
        )?;

        if encoding == "base64" {
            crate::contract::exact(
                &output,
                previous_float.as_ref().context("missing preceding float output")?,
            )?;
        }

        let interval = &proof["manifest_interval"];
        ensure!(
            interval["path"] == interval_name,
            "recovery executor interval filename changed"
        );
        let suffix = interval_bytes(manifest, interval, cursor)?;
        ensure!(
            core::jsonl(suffix)?.len() == 2,
            "serialized recovery request must have exactly one physical pair"
        );
        files.exact(&format!("{root}/{interval_name}"), suffix)?;

        let completed = BTreeMap::from([(
            id,
            CompletedRequest {
                case: case.clone(),
                fixtures: vec![fixture],
                output: output.clone(),
            },
        )]);
        let physical = reconcile_wave(suffix, next, &completed, bundle)?;
        ensure!(
            physical.forwards.len() == 1
                && physical.forwards[0]["B"] == 1
                && physical.forwards[0]["distinct_request_ids"] == json!([id])
                && physical.witnesses.is_empty()
                && physical.next_forward
                    == next.checked_add(1).context("recovery forward overflow")?,
            "recovery physical B1 interval cardinality"
        );

        let after = &proof["observation_after"];
        ready_observation(after, launch)?;
        logical_delta(&observation, after, 1)?;
        let end = usize::try_from(uint(interval, "end_byte")?)?;

        // Reconstruct the complete producer proof, including fields that
        // distinguish real exact-bit checks from summary declarations.
        let expected = json!({
            "case_id": case,
            "fixture_index": fixture,
            "fixture_id": bundle.fixtures[fixture].id,
            "encoding": encoding,
            "request_id": id,
            "request_path": request_name,
            "request_sha256": request_digest,
            "response": response,
            "usage": usage,
            "output_f32le_sha256": output_digest,
            "frozen_singleton_sha256": bundle.fixtures[fixture].golden_sha256,
            "exact_f32_bits": true,
            "prior_float_output_checked_exactly": encoding == "base64",
            "observation_before": observation,
            "observation_after": after,
            "logical_backend_request_delta": 1,
            "manifest_interval": {
                "path": interval_name,
                "start_byte": cursor,
                "end_byte": end,
                "prefix_before_sha256": sha(&manifest[..cursor]),
                "prefix_after_sha256": sha(&manifest[..end]),
                "interval_sha256": sha(suffix)
            },
            "physical_forwards": physical.forwards,
            "unmatched_physical_records": 0
        });
        ensure!(*proof == expected, "recovery complete request proof changed");
        files.pretty(&format!("{root}/{case}.evidence.json"), &expected)?;
        checked_requests.push(expected);

        if encoding == "float" {
            previous_float = Some(output);
        }
        connections.push(connection);
        cursor = end;
        next = physical.next_forward;
        observation = after.clone();
        last_id = Some(id);
    }

    let saved_connections = saved["http_connections"]
        .as_array()
        .context("saved recovery connection inventory")?;
    check_connections(&launch.instance, &connections, saved_connections)?;
    ensure!(
        cursor == manifest.len()
            && next == recovery.next_forward
            && next == uint(&launch.startup, "next_forward_id")?
                .checked_add(44)
                .context("recovery forward count overflow")?
            && ids == recovery.seen_request_ids
            && ids.len() == 44
            && connection_ids.len() == 44
            && observation == recovery.observation,
        "recovery final cursor or complete interval coverage changed"
    );
    logical_delta(&launch.initial_observation, &observation, 44)?;

    // Reconstruct the complete parent evidence. Its historical flags remain
    // historical: final drain had not occurred at this producer boundary.
    let expected_saved = json!({
        "schema": "rustv1.accept_m4_exact_recovery.v1",
        "case_id": RECOVERY_ID,
        "instance": launch.instance,
        "pid": launch.pid,
        "policy_sha256": loaded.policy_sha256,
        "driver_binary_sha256": loaded.artifacts["accept_m4"].sha256,
        "preceding_ledger_sha256": sha(&ready.ledger_prefix),
        "reattestation_evidence_path": format!("scheduler/{READY_ID}/evidence.json"),
        "reattestation_evidence_sha256": ready.reattestation_evidence_sha256,
        "launch_binding_sha256": launch.binding_sha256,
        "orchestration_prefix_sha256": sha(&ready.orchestration_prefix),
        "initial_observation": launch.initial_observation,
        "startup_manifest_sha256": sha(&launch.initial_manifest),
        "requests": checked_requests,
        "http_connections": saved_connections,
        "final_observation": observation,
        "final_manifest_sha256": sha(manifest),
        "final_manifest_bytes": manifest.len(),
        "next_forward_id": next,
        "replacement_request_ids": ids,
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
        "old_exit_evidence_sha256": ready.poison_exit.exit_evidence_sha256,
        "old_service_owner_join_claimed": false,
        "old_quarantine_cleared_in_process": false,
        "replacement_left_ready": true,
        "final_drain_executed": false,
        "lifecycle_phase_complete": false,
        "m4_accepted": false
    });
    ensure!(
        saved == expected_saved,
        "complete recovery parent evidence changed"
    );
    files.pretty(&format!("{root}/evidence.json"), &expected_saved)?;
    files.pretty(
        &format!("{root}/http_connections.json"),
        &json!(saved_connections),
    )?;
    files.exact(
        &format!("{root}/executor_manifest_snapshot.jsonl"),
        manifest,
    )?;
    files.pretty(
        &format!("{root}/final_observation.json"),
        &recovery.observation,
    )?;

    // Final drain does not create another recovery HTTP connection. Preserve
    // the saved joined inventory rather than performing new joins after exit.
    ensure!(
        drained["http_connections"] == saved["http_connections"],
        "final-drain connection inventory differs from recovery"
    );
    files.pretty(
        &format!("{drain_root}/http_connections.json"),
        &drained["http_connections"],
    )?;

    // The preceding lineage pass already checked the other files in these
    // two subtrees. Recheck their inventory digests and close their exact
    // directory coverage here.
    for subtree in [&ready_root, &drain_root] {
        import_lineage_subtree(
            &mut files,
            &continuation.recovery_lineage_reconciliation,
            subtree,
        )?;
    }
    for subtree in [&ready_root, &root, &drain_root] {
        flat_coverage(&files, subtree)?;
    }

    ensure!(
        external(&launch.manifest, MANIFEST_LIMIT)? == *manifest,
        "external recovery final manifest changed"
    );

    // Revalidate the preceding boundary at the end as well. This is offline
    // artifact validation, not a repeat of any acceptance execution.
    ensure!(
        super::reconcile(&continuation.initial_artifacts)?
            == continuation.recovery_lineage_reconciliation,
        "preceding lineage changed during recovery HTTP reconciliation"
    );
    loaded.recheck()?;
    files.stable()?;
    for subtree in [&ready_root, &root, &drain_root] {
        flat_coverage(&files, subtree)?;
    }
    ensure!(
        external(&launch.manifest, MANIFEST_LIMIT)? == *manifest
            && bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)? == ledger
            && bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)? == transcript,
        "recovery HTTP reconciliation inputs changed"
    );

    Ok(json!({
        "schema": "rustv1.accept_m4_recovery_http_reconciliation.v1",
        "boundary": HTTP_BOUNDARY,
        "policy_sha256": loaded.policy_sha256,
        "preceding_ledger_sha256": sha(&ledger),
        "orchestration_sha256": sha(&transcript),
        "preceding_recovery_lineage_reconciliation_sha256":
            sha(&serde_json::to_vec(
                &continuation.recovery_lineage_reconciliation
            )?),
        "recovery_evidence_sha256": recovery.recovery_evidence_sha256,
        "recovery_launch_binding_sha256": launch.binding_sha256,
        "startup_manifest_sha256": sha(&launch.initial_manifest),
        "final_manifest_sha256": sha(manifest),
        "final_manifest_bytes": manifest.len(),
        "next_forward_id": next,
        "reconciled_http_requests": 44,
        "reconciled_serving_physical_forwards": 44,
        "physical_b1_exact_f32_bits_checked": true,
        "float_base64_exact_pairs_checked": 22,
        "saved_connection_joins_checked": 44,
        "final_drain_connection_inventory_matches_recovery": true,
        "unmatched_serving_manifest_bytes": 0,
        "exact_output_subtrees": [ready_root, root, drain_root],
        "scoped_output_directory_coverage_complete": true,
        "verified_output_sha256": files.hashes,
        "complete_output_directory_coverage": false,
        "historical_poisoned_observation_preserved": true,
        "historical_recovery_observation_preserved": true,
        "old_service_owner_join_claimed": false,
        "replacement_service_owner_join_independently_observed": false,
        "supervisor_reap_independently_awaited_by_driver": false,
        "post_exit_control_calls": 0,
        "process_observations": 0,
        "owner_joins_performed": 0,
        "ledger_rows_appended": 0,
        "destructive_children_reconciled": false,
        "complete_poisoned_launch_artifact_set_reconciled": false,
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
    fn recovery_intervals_reject_gap_overlap_empty_and_changed_commitment() {
        // Format fixtures only; no execution evidence is synthesized.
        let manifest = b"startup\nbegin\nend\n";
        let start = b"startup\n".len();
        let commitment = json!({
            "start_byte": start,
            "end_byte": manifest.len(),
            "prefix_before_sha256": sha(&manifest[..start]),
            "prefix_after_sha256": sha(manifest),
            "interval_sha256": sha(&manifest[start..])
        });

        assert_eq!(
            interval_bytes(manifest, &commitment, start).unwrap(),
            b"begin\nend\n"
        );
        assert!(interval_bytes(manifest, &commitment, start - 1).is_err());
        assert!(interval_bytes(manifest, &commitment, start + 1).is_err());

        let mut empty = commitment.clone();
        empty["end_byte"] = json!(start);
        assert!(interval_bytes(manifest, &empty, start).is_err());

        let mut oversized = commitment.clone();
        oversized["end_byte"] = json!(manifest.len() + 1);
        assert!(interval_bytes(manifest, &oversized, start).is_err());

        let mut changed = manifest.to_vec();
        changed[start] = b'X';
        assert!(interval_bytes(&changed, &commitment, start).is_err());

        let mut wrong_digest = commitment;
        wrong_digest["prefix_after_sha256"] = json!(sha(b"other"));
        assert!(interval_bytes(manifest, &wrong_digest, start).is_err());
    }

    #[test]
    fn recovery_response_binding_rejects_identity_and_join_relabelling() {
        let request_digest = sha(b"request");
        let response = json!({
            "case_id": "case",
            "http_status": 200,
            "request_id": 7,
            "headers": {
                "x-rv1-instance-id": "recovery",
                "x-acceptance-case-id": "case",
                "x-rv1-request-id": 7,
                "content-type": "application/json",
                "_connection_owner": {
                    "instance": "recovery",
                    "owner_pid": std::process::id(),
                    "case_id": "case",
                    "method": "POST",
                    "path": "/v1/embeddings",
                    "request_sha256": request_digest,
                    "connection_id": 12,
                    "join_awaited": true,
                    "join_outcome": "completed"
                }
            },
            "response_path": "case.response.bin",
            "response_sha256": sha(b"response")
        });
        assert!(response_binding(
            &response, "recovery", "case", 7, &request_digest
        ).is_ok());

        let mut wrong_instance = response.clone();
        wrong_instance["headers"]["x-rv1-instance-id"] = json!("old");
        assert!(response_binding(
            &wrong_instance, "recovery", "case", 7, &request_digest
        ).is_err());

        let mut wrong_id = response.clone();
        wrong_id["headers"]["x-rv1-request-id"] = Value::Null;
        assert!(response_binding(
            &wrong_id, "recovery", "case", 7, &request_digest
        ).is_err());

        let mut unjoined = response.clone();
        unjoined["headers"]["_connection_owner"]["join_awaited"] = json!(false);
        assert!(response_binding(
            &unjoined, "recovery", "case", 7, &request_digest
        ).is_err());

        let mut panicked = response.clone();
        panicked["headers"]["_connection_owner"]["join_outcome"] = json!("panicked");
        assert!(response_binding(
            &panicked, "recovery", "case", 7, &request_digest
        ).is_err());

        assert!(response_binding(
            &response, "recovery", "case", 7, &sha(b"different-request")
        ).is_err());

        let mut extra = response;
        extra["new_historical_claim"] = json!(true);
        assert!(response_binding(
            &extra, "recovery", "case", 7, &request_digest
        ).is_err());
    }

    #[test]
    fn recovery_directory_coverage_rejects_extra_missing_and_nonfiles() {
        let root = std::env::temp_dir().join(format!(
            "rv1-recovery-coverage-{}-{}",
            std::process::id(),
            core::random_label().unwrap()
        ));
        private_directory(&root).unwrap();
        let subtree = root.join("recovery");
        private_directory(&subtree).unwrap();

        let result: Result<()> = (|| {
            write_new(&subtree.join("evidence.json"), b"{}")?;
            let mut files = Files::new(&root)?;
            files.exact("recovery/evidence.json", b"{}")?;
            flat_coverage(&files, "recovery")?;

            let extra = subtree.join("unexpected.bin");
            write_new(&extra, b"unexpected")?;
            ensure!(
                flat_coverage(&files, "recovery").is_err(),
                "extra recovery artifact accepted"
            );
            fs::remove_file(&extra)?;

            let evidence = subtree.join("evidence.json");
            let hidden = subtree.join("hidden.json");
            fs::rename(&evidence, &hidden)?;
            let missing = flat_coverage(&files, "recovery");
            fs::rename(&hidden, &evidence)?;
            ensure!(missing.is_err(), "missing recovery evidence accepted");

            let directory = subtree.join("unexpected-directory");
            private_directory(&directory)?;
            ensure!(
                flat_coverage(&files, "recovery").is_err(),
                "unexpected recovery subdirectory accepted"
            );
            fs::remove_dir(&directory)?;

            #[cfg(unix)]
            {
                let link = subtree.join("linked-evidence.json");
                std::os::unix::fs::symlink(&evidence, &link)?;
                let linked = flat_coverage(&files, "recovery");
                fs::remove_file(&link)?;
                ensure!(linked.is_err(), "recovery symlink accepted");
            }

            fs::write(&evidence, b"{\"changed\":true}")?;
            let changed = files.stable();
            fs::write(&evidence, b"{}")?;
            ensure!(changed.is_err(), "changed recovery artifact accepted");

            files.stable()?;
            flat_coverage(&files, "recovery")?;
            Ok(())
        })();

        if result.is_ok() {
            fs::remove_dir_all(&root).unwrap();
        }
        result.unwrap_or_else(|error| {
            panic!(
                "recovery directory test failed; retained {}: {error:#}",
                root.display()
            )
        });
    }
}
