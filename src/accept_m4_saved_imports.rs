// Included at acceptance_scheduler::saved_imports.
//
// Offline reconciliation of:
// - the frozen required-case declaration;
// - imported numerical-prepass records and their policy-bound source;
// - the two policy-loading ledger rows;
// - imported HTTP records;
// - saved HTTP request/response files and complete executor intervals.
//
// This is not complete cross-phase or final-artifact reconciliation.
// In particular, it does not authenticate a newly computed digest of the
// HTTP completion marker as an original historical commitment.
//
// No network calls, process observations, owner joins, or ledger writes.

use super::*;
use std::path::Component;

const BOUNDARY: &str = "CROSS-PHASE-IMPORTS-RECONCILED";

fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    v[key]
        .as_array()
        .with_context(|| format!("missing saved-import array: {key}"))
}

fn normal_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    ensure!(
        path.components().count() == 1
            && matches!(path.components().next(), Some(Component::Normal(_))),
        "unsafe saved-import filename"
    );
    Ok(())
}

fn regular_directory(path: &Path) -> Result<()> {
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_dir(),
        "saved-import root is not a regular non-symlink directory"
    );
    Ok(())
}

/// Every returned digest is calculated only after validating either:
/// - an original byte commitment, or
/// - exact expected bytes reconstructed from retained, bound state.
///
/// Returned inventory digests are not described as new historical evidence.
fn read_expected(
    root: &Path,
    name: &str,
    expected_digest: &str,
    limit: usize,
    inventory: &mut BTreeMap<String, String>,
    prefix: &str,
) -> Result<Vec<u8>> {
    normal_name(name)?;
    ensure!(
        crate::contract::digest_valid(expected_digest),
        "invalid saved-import digest"
    );
    regular_directory(root)?;
    let path = root.join(name);
    ensure!(
        fs::symlink_metadata(&path)?.file_type().is_file(),
        "saved-import artifact is not a regular non-symlink file"
    );
    let bytes = bounded(&path, limit)?;
    ensure!(
        sha(&bytes) == expected_digest,
        "saved-import artifact digest mismatch: {name}"
    );
    let relative = if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}/{name}")
    };
    ensure!(
        inventory
            .insert(relative, expected_digest.to_owned())
            .is_none(),
        "duplicate saved-import inventory entry"
    );
    Ok(bytes)
}

fn phase_rows<'a>(records: &'a [Value], phase: &str) -> Vec<&'a Value> {
    records
        .iter()
        .filter(|record| record["acceptance_phase"] == phase)
        .collect()
}

fn required_ids<'a>(loaded: &'a Loaded, phase: &str) -> Result<&'a Vec<String>> {
    let phases: Vec<_> = loaded.required.phases
        .iter()
        .filter(|(name, _)| name == phase)
        .collect();
    ensure!(phases.len() == 1, "missing/duplicate required phase");
    Ok(&phases[0].1)
}

/// Validate the import wrapper against the exact original producer bytes.
///
/// This does not replace source_records_sha256 with the digest of a
/// reserialized producer array.
fn imported(
    global: &[Value],
    phase: &str,
    required: &[String],
    source_bytes: &[u8],
) -> Result<Vec<Value>> {
    let source = core::jsonl(source_bytes)?;
    let rows = phase_rows(global, phase);
    ensure!(
        source.len() == required.len() && rows.len() == required.len(),
        "imported phase coverage"
    );
    let digest = sha(source_bytes);
    for ((row, producer), case) in rows.iter().zip(&source).zip(required) {
        closed(
            row,
            &["acceptance_phase", "case_id", "status", "pass", "details"],
        )?;
        closed(
            &row["details"],
            &["source_records_sha256", "producer_record"],
        )?;
        ensure!(
            row["case_id"] == case.as_str()
                && row["status"] == "executed"
                && row["pass"] == true
                && row["details"]["source_records_sha256"] == digest
                && row["details"]["producer_record"] == *producer
                && producer["case_id"] == case.as_str()
                && producer["status"] == "executed"
                && producer["pass"] == true
                && producer["details"].is_object()
                && producer.get("acceptance_phase")
                    .is_none_or(|value| value == phase),
            "imported phase identity/status/source binding"
        );
    }
    Ok(source)
}

/// Validate bounds before slicing. Empty intervals are permitted only for
/// callers that separately establish zero execution.
fn interval<'a>(
    manifest: &'a [u8],
    commitment: &Value,
    cursor: usize,
) -> Result<&'a [u8]> {
    let start = usize::try_from(uint(commitment, "start_byte")?)?;
    let end = usize::try_from(uint(commitment, "end_byte")?)?;
    ensure!(
        start == cursor && start <= end && end <= manifest.len(),
        "saved HTTP interval gap, overlap or invalid bounds"
    );
    ensure!(
        commitment["prefix_before_sha256"] == sha(&manifest[..start])
            && commitment["prefix_after_sha256"] == sha(&manifest[..end])
            && commitment["interval_sha256"] == sha(&manifest[start..end]),
        "saved HTTP interval commitment mismatch"
    );
    Ok(&manifest[start..end])
}

fn connection(
    response: &Value,
    instance: &str,
    case: &str,
    request_digest: &str,
    request_id: Option<u64>,
) -> Result<()> {
    let headers = &response["headers"];
    let owner = &headers["_connection_owner"];
    ensure!(
        headers["x-rv1-instance-id"] == instance
            && headers["x-acceptance-case-id"] == case
            && headers["x-rv1-request-id"] == json!(request_id)
            && owner["instance"] == instance
            && owner["owner_pid"] == std::process::id()
            && owner["case_id"] == case
            && owner["method"] == "POST"
            && owner["path"] == "/v1/embeddings"
            && owner["request_sha256"] == request_digest
            && owner["join_awaited"] == true
            && matches!(
                owner["join_outcome"].as_str(),
                Some("completed" | "cancelled-after-abort-and-awaited")
            ),
        "saved HTTP correlation/connection join binding"
    );
    uint(owner, "connection_id")?;
    Ok(())
}

/// Recompute the HTTP producer's numerical row evidence.
///
/// Its B1 candidate-selection rule differs from scheduler::reconcile_wave:
/// the HTTP producer still selects shape_specific/canonical candidates at B1.
/// Preserve that historical distinction rather than relabelling its rows.
fn http_rows(
    execution: &Value,
    indices: &[usize],
    output: &[Vec<f32>],
    bundle: &Bundle,
    forward: u64,
) -> Result<Vec<String>> {
    let rows = array(execution, "rows")?;
    ensure!(
        rows.len() == indices.len() && output.len() == indices.len(),
        "saved HTTP numerical row coverage"
    );
    let mut hashes = Vec::new();
    for (position, ((row, fixture_index), actual)) in
        rows.iter().zip(indices).zip(output).enumerate()
    {
        let fixture = &bundle.fixtures[*fixture_index];
        let digest = sha(&row_bytes(actual));
        let (candidate, selection): (&[f32], &str) =
            match bundle.candidates.get(&(indices.to_vec(), position)) {
                Some(candidate) => (candidate.as_slice(), "shape_specific"),
                None => (
                    bundle.canonical(*fixture_index, position % 2)?,
                    "canonical_same_fixture_position_modulo_two",
                ),
            };
        let expected = json!({
            "fixture_index": fixture_index,
            "fixture_id": fixture.id,
            "request_row": position,
            "physical_row": position,
            "forward_id": forward,
            "valid_length": fixture.ids.len(),
            "token_sha256": fixture.ids_sha256,
            "output_f32le_sha256": digest,
            "singleton_comparison":
                compare(actual, &bundle.singleton[*fixture_index])?,
            "independent_reference_comparison":
                compare(actual, &bundle.reference[*fixture_index])?,
            "candidate_comparison": compare(actual, candidate)?,
            "candidate_selection": selection
        });
        ensure!(*row == expected, "saved HTTP numerical row changed");
        hashes.push(digest);
    }
    Ok(hashes)
}

fn http_artifacts(
    state: &FinalDrainState,
    source: &[Value],
    inventory: &mut BTreeMap<String, String>,
) -> Result<Value> {
    let old = &state.recovery.ready.poison_exit.old;
    let loaded = &old.core.loaded;
    let launch = &old.core.initial_launch;
    let root = old.core.out.join("http");
    regular_directory(&root)?;

    ensure!(!source.is_empty(), "empty HTTP source");
    let first = &source[0]["details"];
    let last = &source.last().context("last HTTP record")?["details"];
    ensure!(
        first["manifest_interval"]["start_byte"] == launch.initial_manifest.len()
            && first["manifest_interval"]["prefix_before_sha256"]
                == sha(&launch.initial_manifest),
        "HTTP startup prefix differs from retained initial launch"
    );

    let initial = read_expected(
        &root,
        "initial_executor_manifest.jsonl",
        &sha(&launch.initial_manifest),
        MANIFEST_LIMIT,
        inventory,
        "http",
    )?;
    ensure!(
        initial == launch.initial_manifest,
        "HTTP initial snapshot bytes changed"
    );
    let manifest = read_expected(
        &root,
        "executor_manifest_snapshot.jsonl",
        text(&last["manifest_interval"], "prefix_after_sha256")?,
        MANIFEST_LIMIT,
        inventory,
        "http",
    )?;
    core::jsonl(&manifest)?;
    ensure!(
        manifest.starts_with(&initial),
        "HTTP whole snapshot does not extend its startup"
    );

    // final_observation.json has no separate original byte digest in the
    // global import. Compare its JSON value with the digest-bound last row.
    let observation_path = root.join("final_observation.json");
    ensure!(
        fs::symlink_metadata(&observation_path)?.file_type().is_file(),
        "HTTP final observation is not a regular file"
    );
    let observation_bytes = bounded(&observation_path, JSON_LIMIT)?;
    ensure!(
        serde_json::from_slice::<Value>(&observation_bytes)?
            == last["observation_after"],
        "HTTP final observation differs from imported source"
    );
    ensure!(
        inventory.insert(
            "http/final_observation.json".to_owned(),
            sha(&observation_bytes),
        ).is_none(),
        "duplicate HTTP final observation inventory"
    );

    let mut cursor = initial.len();
    let mut next = uint(&launch.startup, "next_forward_id")?;
    let mut observation = launch.initial_observation.clone();
    let mut last_id = None;
    let mut ids = BTreeSet::new();
    let mut connection_ids = BTreeSet::new();
    let mut layouts = BTreeMap::<Vec<usize>, Vec<String>>::new();
    let mut positives = 0usize;
    let mut rejections = 0usize;

    for (position, record) in source.iter().enumerate() {
        let case = text(record, "case_id")?;
        ensure!(label_valid(case), "invalid HTTP source case label");
        let details = &record["details"];
        let request = &details["request"];
        let response = &details["response"];
        let execution = &details["execution"];
        let closed_admission = case == "closed-admission";

        ensure!(
            details["instance"] == launch.instance
                && details["pid"] == launch.pid
                && request["path"] == format!("{case}.request.bin")
                && response["path"] == format!("{case}.response.bin"),
            "HTTP source launch/file association"
        );

        // The one historical quiescence transition occurs immediately before
        // the final closed-admission probe. It is not a post-exit observation.
        if closed_admission {
            ensure!(
                position + 1 == source.len() && observation["ready"] == true,
                "closed-admission placement"
            );
            observation["ready"] = json!(false);
        }
        ensure!(
            details["observation_before"] == observation,
            "HTTP observation chain gap"
        );
        let after = &details["observation_after"];
        closed(
            after,
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
            after["instance"] == launch.instance
                && after["pid"] == launch.pid
                && after["lifecycle"] == "ready"
                && after["worker_alive"] == true
                && after["ready"] == !closed_admission
                && after["startup_output_sha256"]
                    == launch.initial_observation["startup_output_sha256"],
            "HTTP historical observation identity/lifecycle"
        );

        let request_bytes = read_expected(
            &root,
            text(request, "path")?,
            text(request, "sha256")?,
            JSON_LIMIT,
            inventory,
            "http",
        )?;
        let response_bytes = read_expected(
            &root,
            text(response, "path")?,
            text(response, "sha256")?,
            JSON_LIMIT,
            inventory,
            "http",
        )?;

        let request_id = if details["server_request_id"].is_null() {
            None
        } else {
            Some(uint(details, "server_request_id")?)
        };
        if let Some(id) = request_id {
            ensure!(
                id > 0
                    && last_id.is_none_or(|previous| id > previous)
                    && ids.insert(id),
                "saved HTTP request IDs reused or reordered"
            );
            last_id = Some(id);
        }
        ensure!(
            request_id.is_none() == closed_admission,
            "saved HTTP admission/request-ID association"
        );
        connection(
            response,
            &launch.instance,
            case,
            text(request, "sha256")?,
            request_id,
        )?;
        ensure!(
            connection_ids.insert(uint(
                &response["headers"]["_connection_owner"],
                "connection_id",
            )?),
            "saved HTTP connection identity reused"
        );

        let suffix = interval(&manifest, &details["manifest_interval"], cursor)?;
        let status = u16::try_from(uint(response, "http_status")?)?;
        if status == 200 {
            ensure!(!closed_admission, "closed admission succeeded");
            let rows = array(execution, "rows")?;
            ensure!((1..=8).contains(&rows.len()), "HTTP success row count");
            let indices: Vec<usize> = rows.iter()
                .map(|row| {
                    let index = usize::try_from(uint(row, "fixture_index")?)?;
                    ensure!(index < loaded.bundle.fixtures.len(), "fixture index");
                    Ok(index)
                })
                .collect::<Result<_>>()?;
            let encoding = text(&response["validation"], "encoding")?;
            ensure!(
                matches!(encoding, "float" | "base64"),
                "HTTP saved success encoding"
            );

            let body: Value = serde_json::from_slice(&request_bytes)?;
            let texts: Vec<&str> = indices.iter()
                .map(|index| loaded.bundle.fixtures[*index].text.as_str())
                .collect();
            let input_matches = body["input"] == json!(texts)
                || (texts.len() == 1 && body["input"] == texts[0]);
            let actual_encoding = body.get("encoding_format")
                .filter(|value| !value.is_null())
                .map(|value| value.as_str().context("request encoding"))
                .transpose()?
                .unwrap_or("float");
            ensure!(
                body["model"] == MODEL_ID
                    && input_matches
                    && actual_encoding == encoding,
                "saved success request/fixture/encoding association"
            );

            let reply = http::Reply {
                status,
                request_id,
                headers: response["headers"].clone(),
                bytes: response_bytes,
            };
            let (output, usage) =
                http::decode_success(&reply, &loaded.bundle, &indices, encoding)?;
            ensure!(
                response["validation"]["usage"] == usage,
                "saved HTTP usage changed"
            );
            logical_delta(&observation, after, 1)?;
            ensure!(
                details["logical_backend_request_delta"] == 1
                    && core::jsonl(suffix)?.len() == 2,
                "serialized HTTP logical/physical interval count"
            );

            let completed = BTreeMap::from([(
                request_id.context("successful HTTP request ID")?,
                CompletedRequest {
                    case: case.to_owned(),
                    fixtures: indices.clone(),
                    output: output.clone(),
                },
            )]);
            let physical = reconcile_wave(suffix, next, &completed, &loaded.bundle)?;
            let width = indices.iter()
                .map(|index| loaded.bundle.fixtures[*index].ids.len())
                .max()
                .context("HTTP width")?;
            let valid: usize = indices.iter()
                .map(|index| loaded.bundle.fixtures[*index].ids.len())
                .sum();
            ensure!(
                physical.forwards.len() == 1
                    && physical.witnesses.is_empty()
                    && execution["route"] == "IndependentExecutor::forward_batch"
                    && execution["physical_forwards"] == 1
                    && execution["forward_id"] == next
                    && execution["B"] == indices.len()
                    && execution["S"] == width
                    && execution["valid_tokens"] == valid
                    && execution["padding_tokens"] == indices.len() * width - valid
                    && execution["association_source"] == "executor_rows",
                "saved HTTP execution summary/physical association"
            );
            let hashes = http_rows(execution, &indices, &output, &loaded.bundle, next)?;
            let repeated = if let Some(previous) = layouts.get(&indices) {
                ensure!(*previous == hashes, "HTTP identical-layout repeat changed");
                true
            } else {
                false
            };
            ensure!(
                execution["previous_identical_layout_checked_exactly"] == repeated,
                "HTTP repeat-history summary changed"
            );
            layouts.insert(indices, hashes);
            next = physical.next_forward;
            positives += 1;
        } else {
            ensure!(
                matches!(status, 400 | 413 | 415 | 503)
                    && suffix.is_empty()
                    && *after == observation
                    && details["logical_backend_request_delta"] == 0
                    && execution["physical_forwards"] == 0
                    && execution["logical_backend_request_delta"] == 0
                    && execution["zero_execution_proof"]["observations_identical"] == true
                    && execution["zero_execution_proof"]["manifest_bytes_identical"] == true
                    && execution["physical_capacity_gate_proven"] == false
                    && serde_json::from_slice::<Value>(&response_bytes)?
                        == response["validation"],
                "saved HTTP rejection/zero-execution binding"
            );
            // The original producer record retains the exact gate and error
            // envelope. Do not relabel schema rejection as token/OOV coverage.
            ensure!(
                response["validation"]["error"].is_object()
                    && response["validation"].as_object()
                        .is_some_and(|object| object.len() == 1)
                    && execution["rejection_gate"].is_string()
                    && execution["association_source"]
                        == if closed_admission {
                            "null-request-id-observation-and-byte-invariance"
                        } else {
                            "http-request-id-observation-and-byte-invariance"
                        },
                "saved HTTP rejection scope"
            );
            rejections += 1;
        }

        cursor = usize::try_from(uint(&details["manifest_interval"], "end_byte")?)?;
        observation = after.clone();
    }

    ensure!(
        cursor == manifest.len()
            && observation == last["observation_after"]
            && observation["ready"] == false
            && positives > 0
            && rejections > 0
            && next == uint(&launch.startup, "next_forward_id")?
                .checked_add(u64::try_from(positives)?)
                .context("HTTP forward count overflow")?,
        "HTTP complete saved interval coverage"
    );

    Ok(json!({
        "source_records": source.len(),
        "successful_requests": positives,
        "rejected_requests": rejections,
        "serving_physical_forwards": positives,
        "startup_manifest_sha256": sha(&initial),
        "final_manifest_sha256": sha(&manifest),
        "final_manifest_bytes": manifest.len(),
        "next_forward_id": next,
        "saved_connection_records_checked": connection_ids.len(),
        "unmatched_serving_manifest_bytes": 0,
        "historical_observations_relabelled": false
    }))
}

pub(super) fn reconcile(
    continuation: &LocalArtifactsContinuation,
) -> Result<Value> {
    let state = &continuation.final_drain.state;
    let old = &state.recovery.ready.poison_exit.old;
    let loaded = &old.core.loaded;
    let out = &old.core.out;

    // Revalidate the previous offline boundary against its original
    // commitments. This is not a ledger rewind or a rerun of any probe.
    ensure!(
        physical_b::saved_local::reconcile(state)?
            == continuation.local_reconciliation,
        "retained local-artifact reconciliation changed"
    );
    loaded.recheck()?;
    let ledger = bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)?;
    ensure!(
        ledger == state.ledger_prefix,
        "saved-import reconciliation ledger changed"
    );
    let global = core::jsonl(&ledger)?;
    let mut inventory = BTreeMap::new();

    let required_bytes = serde_json::to_vec_pretty(&json!({
        "policy_sha256": loaded.policy_sha256,
        "required": loaded.required.as_json()
    }))?;
    let saved_required = read_expected(
        out,
        "required_cases.json",
        &sha(&required_bytes),
        JSON_LIMIT,
        &mut inventory,
        "",
    )?;
    ensure!(
        saved_required == required_bytes,
        "required declaration differs from frozen policy-loaded plan"
    );

    // Loaded::recheck authenticates this producer artifact and its companion
    // bytes through the original policy/loading commitments.
    let numerical_path = &loaded.paths["batch_in_process_evidence"];
    let numerical_bytes = bounded(numerical_path, JSON_LIMIT)?;
    ensure!(
        sha(&numerical_bytes) == loaded.artifacts["batch_in_process_evidence"].sha256,
        "numerical evidence policy commitment changed"
    );
    let numerical: Value = serde_json::from_slice(&numerical_bytes)?;
    ensure!(
        numerical["records"]["path"] == "accept_records.jsonl",
        "numerical companion path changed"
    );
    let numerical_root = numerical_path.parent().context("numerical evidence parent")?;
    let numerical_records_path = numerical_root.join("accept_records.jsonl");
    let numerical_records = bounded(&numerical_records_path, MANIFEST_LIMIT)?;
    ensure!(
        sha(&numerical_records) == text(&numerical["records"], "sha256")?,
        "numerical producer record commitment changed"
    );
    let prepass = imported(
        &global,
        policy::PREPASS,
        required_ids(loaded, policy::PREPASS)?,
        &numerical_records,
    )?;
    ensure!(
        prepass == loaded.prepass_records
            && numerical["records"]["count"] == prepass.len(),
        "imported numerical records differ from policy-consumed records"
    );

    let policy_rows = phase_rows(&global, policy::POLICY);
    ensure!(policy_rows.len() == 2, "policy ledger coverage");
    ensure!(
        policy_rows[0]["case_id"] == "finalized-policy-bindings"
            && policy_rows[0]["details"] == loaded.binding_summary()
            && policy_rows[1]["case_id"] == "prepass-evidence-consumed"
            && policy_rows[1]["details"] == json!({
                "evidence_sha256":
                    loaded.artifacts["batch_in_process_evidence"].sha256,
                "source_records_sha256": sha(&numerical_records),
                "records": loaded.prepass_records.len(),
                "consumer": "accept_m4_policy::load"
            }),
        "policy-loading ledger binding changed"
    );

    let http_rows = phase_rows(&global, policy::HTTP);
    let http_source_digest = text(
        &http_rows.first().context("missing imported HTTP phase")?["details"],
        "source_records_sha256",
    )?;
    let http_bytes = read_expected(
        &out.join("http"),
        "accept_records.jsonl",
        http_source_digest,
        MANIFEST_LIMIT,
        &mut inventory,
        "http",
    )?;
    let http_source = imported(
        &global,
        policy::HTTP,
        required_ids(loaded, policy::HTTP)?,
        &http_bytes,
    )?;
    let http_summary = http_artifacts(state, &http_source, &mut inventory)?;

    for (relative, digest) in &inventory {
        let path = out.join(relative);
        ensure!(
            fs::symlink_metadata(&path)?.file_type().is_file()
                && file_sha(&path)? == *digest,
            "saved-import artifact changed during reconciliation"
        );
    }
    loaded.recheck()?;
    ensure!(
        bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)? == ledger
            && bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)?
                == state.orchestration_prefix,
        "ledger/transcript changed during saved-import reconciliation"
    );

    Ok(json!({
        "schema": "rustv1.accept_m4_import_reconciliation.v1",
        "boundary": BOUNDARY,
        "policy_sha256": loaded.policy_sha256,
        "preceding_ledger_sha256": sha(&ledger),
        "local_reconciliation_sha256":
            sha(&serde_json::to_vec(&continuation.local_reconciliation)?),
        "numerical_evidence_sha256":
            loaded.artifacts["batch_in_process_evidence"].sha256,
        "numerical_source_records_sha256": sha(&numerical_records),
        "numerical_imported_records": prepass.len(),
        "policy_loading_records_checked": 2,
        "http_source_records_sha256": sha(&http_bytes),
        "http": http_summary,
        "verified_output_sha256": inventory,
        "http_completion_marker_reconciled": false,
        "all_cross_phase_artifacts_reconciled": false,
        "external_launches_reconciled": false,
        "destructive_children_reconciled": false,
        "historical_poisoned_launch_reconciled": false,
        "authenticated_transcript_semantics_reconciled": false,
        "post_exit_control_calls": 0,
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

    fn source_bytes(record: &Value) -> Vec<u8> {
        let mut bytes = serde_json::to_vec(record).unwrap();
        bytes.push(b'\n');
        bytes
    }

    #[test]
    fn imports_reject_rebased_source_and_changed_embedded_record() {
        // Disposable format fixtures, not acceptance execution evidence.
        let producer = json!({
            "case_id": "case",
            "status": "executed",
            "pass": true,
            "details": {"historical": "original"}
        });
        let bytes = source_bytes(&producer);
        let global = vec![json!({
            "acceptance_phase": "phase",
            "case_id": "case",
            "status": "executed",
            "pass": true,
            "details": {
                "source_records_sha256": sha(&bytes),
                "producer_record": producer
            }
        })];
        let required = vec!["case".to_owned()];
        assert!(imported(&global, "phase", &required, &bytes).is_ok());

        let mut changed = producer.clone();
        changed["details"]["historical"] = json!("rewritten");
        let changed_bytes = source_bytes(&changed);
        assert!(imported(&global, "phase", &required, &changed_bytes).is_err());

        let mut rebased = global.clone();
        rebased[0]["details"]["source_records_sha256"] = json!(sha(&changed_bytes));
        assert!(imported(&rebased, "phase", &required, &changed_bytes).is_err());

        let mut embedded = global.clone();
        embedded[0]["details"]["producer_record"] = changed;
        assert!(imported(&embedded, "phase", &required, &bytes).is_err());

        let duplicate = vec![global[0].clone(), global[0].clone()];
        assert!(imported(&duplicate, "phase", &required, &bytes).is_err());
    }

    #[test]
    fn saved_http_intervals_reject_gaps_overlap_and_changed_bytes() {
        let bytes = b"startup\nserving\n";
        let start = b"startup\n".len();
        let commitment = json!({
            "start_byte": start,
            "end_byte": bytes.len(),
            "prefix_before_sha256": sha(&bytes[..start]),
            "prefix_after_sha256": sha(bytes),
            "interval_sha256": sha(&bytes[start..])
        });
        assert_eq!(
            interval(bytes, &commitment, start).unwrap(),
            b"serving\n"
        );
        assert!(interval(bytes, &commitment, start - 1).is_err());
        assert!(interval(bytes, &commitment, start + 1).is_err());

        let mut changed = bytes.to_vec();
        changed[start] = b'X';
        assert!(interval(&changed, &commitment, start).is_err());

        let mut oversized = commitment.clone();
        oversized["end_byte"] = json!(bytes.len() + 1);
        assert!(interval(bytes, &oversized, start).is_err());

        let mut reversed = commitment;
        reversed["end_byte"] = json!(start - 1);
        assert!(interval(bytes, &reversed, start).is_err());
    }
}
