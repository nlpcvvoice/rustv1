// Included at acceptance_scheduler::saved_initial::saved_independent.
//
// Offline independent-mode artifact and complete manifest reconciliation.
//
// No Service creation, CUDA execution, control/HTTP request, process
// observation, owner join, orchestration exchange, or ledger append.
//
// Historical limitation:
// the independent producer used its own AbortConnection client. It retained
// Service-owner and HTTP-server joins, but no individual connection-task
// join records. Do not invent them.
//
// service_log.jsonl has no original producer byte commitment. Its contents
// are checked against committed request outcomes, and its current digest is
// an inventory/stability value, not an original historical commitment.

use super::*;

const INDEPENDENT_BOUNDARY: &str =
    "INDEPENDENT-ARTIFACTS-AND-MANIFEST-RECONCILED";
const STARTUP_CASE: &str = "independent-startup-reconciled";
const JOIN_CASE: &str = "independent-owner-shutdown-and-join";

enum Expected {
    Success(Vec<usize>),
    Rejection(&'static str),
}

struct Planned {
    id: String,
    body: Value,
    expected: Expected,
}

fn positive(loaded: &Loaded, id: String, indices: &[usize]) -> Result<Planned> {
    ensure!(
        label_valid(&id)
            && (1..=8).contains(&indices.len())
            && indices.iter().all(|i| *i < loaded.bundle.fixtures.len()),
        "invalid saved-independent positive plan"
    );
    Ok(Planned {
        id,
        body: json!({
            "model": MODEL_ID,
            "input": indices.iter()
                .map(|i| loaded.bundle.fixtures[*i].text.as_str())
                .collect::<Vec<_>>(),
            "encoding_format": "float"
        }),
        expected: Expected::Success(indices.to_vec()),
    })
}

/// Reconstruct the producer plan from policy-loaded mappings, not from
/// evidence rows or directory enumeration.
fn plan(loaded: &Loaded) -> Result<Vec<Planned>> {
    ensure!(
        loaded.mappings.len() == 46
            && loaded.bundle.fixtures.len() == 22,
        "saved-independent mapping/fixture coverage"
    );
    let mut cases = Vec::new();
    for mapping in &loaded.mappings[..41] {
        for (position, indices) in mapping.requests.iter().enumerate() {
            cases.push(positive(
                loaded,
                format!("mapped-{}-{position}", mapping.m2_group_id),
                indices,
            )?);
        }
    }

    let short = loaded.bundle.fixtures[0].text.as_str();
    let pad = loaded.policy["pad_text"].as_str();
    let rejects = [
        ("invalid-empty-batch", json!([]), "empty_batch"),
        ("invalid-last-empty", json!([short, ""]), "empty_sequence"),
        (
            if pad.is_some() {
                "invalid-last-pad"
            } else {
                "invalid-last-empty-pad-substitute"
            },
            json!([short, pad.unwrap_or("")]),
            if pad.is_some() {
                "pad_token"
            } else {
                "empty_sequence"
            },
        ),
        (
            "invalid-last-overlength",
            json!([short, text(&loaded.policy, "over_length_text")?]),
            "over_length",
        ),
        ("invalid-token-array", json!([1, 2]), "invalid_request"),
    ];
    for (offset, (id, input, code)) in rejects.into_iter().enumerate() {
        cases.push(Planned {
            id: id.to_owned(),
            body: json!({"model": MODEL_ID, "input": input}),
            expected: Expected::Rejection(code),
        });
        let mapping = &loaded.mappings[41 + offset];
        ensure!(
            mapping.m2_group_id == 42 + offset
                && mapping.requests.len() == 1,
            "saved-independent recovery placement"
        );
        cases.push(positive(
            loaded,
            format!("mapped-{}-0", mapping.m2_group_id),
            &mapping.requests[0],
        )?);
    }

    let expected = policy::mapping_case_ids(&loaded.mappings, pad.is_some());
    ensure!(
        cases.iter().map(|case| case.id.as_str()).collect::<Vec<_>>()
            == expected.iter().map(String::as_str).collect::<Vec<_>>(),
        "saved-independent plan differs from frozen mapping order"
    );
    let unique: BTreeSet<_> = cases.iter().map(|case| case.id.as_str()).collect();
    ensure!(unique.len() == cases.len(), "duplicate independent plan case");
    Ok(cases)
}

fn rejection_body(code: &str) -> Result<Value> {
    let (message, param) = match code {
        "empty_batch" => ("Input batch is empty", Some("input")),
        "empty_sequence" => ("Input sequence is empty", Some("input")),
        "pad_token" => ("Input contains a padding token", Some("input")),
        "over_length" => ("Input exceeds the sequence limit", Some("input")),
        "invalid_request" => ("Invalid request", None),
        _ => anyhow::bail!("unplanned independent rejection"),
    };
    Ok(json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": code
        }
    }))
}

/// Bounds and all three byte commitments are checked before slicing.
///
/// Empty intervals are allowed only for separately checked rejections.
fn interval<'a>(
    manifest: &'a [u8],
    commitment: &Value,
    cursor: usize,
) -> Result<&'a [u8]> {
    closed(commitment, &[
        "start_byte",
        "end_byte",
        "prefix_before_sha256",
        "prefix_after_sha256",
        "interval_sha256",
    ])?;
    let start = usize::try_from(uint(commitment, "start_byte")?)?;
    let end = usize::try_from(uint(commitment, "end_byte")?)?;
    ensure!(
        start == cursor && start <= end && end <= manifest.len(),
        "independent interval gap, overlap or invalid bounds"
    );
    ensure!(
        commitment["prefix_before_sha256"] == sha(&manifest[..start])
            && commitment["prefix_after_sha256"] == sha(&manifest[..end])
            && commitment["interval_sha256"] == sha(&manifest[start..end]),
        "independent interval byte commitment changed"
    );
    Ok(&manifest[start..end])
}

fn observation(
    value: &Value,
    instance: &str,
    pid: u32,
    startup_digest: &Value,
) -> Result<u64> {
    closed(value, &[
        "instance",
        "pid",
        "lifecycle",
        "worker_alive",
        "ready",
        "forward_calls",
        "startup_output_sha256",
    ])?;
    ensure!(
        value["instance"] == instance
            && value["pid"] == pid
            && value["lifecycle"] == "ready"
            && value["worker_alive"] == true
            && value["ready"] == true
            && value["startup_output_sha256"] == *startup_digest,
        "independent historical observation identity/readiness"
    );
    uint(value, "forward_calls")
}

/// The original independent Reply did not retain headers. Decode only the
/// saved body/status here; do not synthesize headers or connection evidence.
fn outputs(
    bytes: &[u8],
    indices: &[usize],
    loaded: &Loaded,
) -> Result<Vec<Vec<f32>>> {
    let value: Value = serde_json::from_slice(bytes)?;
    closed(&value, &["object", "data", "model", "usage"])?;
    let total: usize = indices.iter()
        .map(|i| loaded.bundle.fixtures[*i].ids.len())
        .sum();
    ensure!(
        value["object"] == "list"
            && value["model"] == MODEL_ID
            && value["usage"] == json!({
                "prompt_tokens": total,
                "total_tokens": total
            }),
        "independent saved success object/model/usage"
    );
    let data = value["data"].as_array().context("independent response rows")?;
    ensure!(data.len() == indices.len(), "independent response cardinality");
    let mut rows = Vec::new();
    for (position, (item, fixture)) in data.iter().zip(indices).enumerate() {
        closed(item, &["object", "index", "embedding"])?;
        ensure!(
            item["object"] == "embedding"
                && item["index"] == position
                && item["embedding"].is_array(),
            "independent response row association"
        );
        let row: Vec<f32> = serde_json::from_value(item["embedding"].clone())?;
        crate::contract::exact(
            std::slice::from_ref(&row),
            std::slice::from_ref(&loaded.bundle.singleton[*fixture]),
        )?;
        ensure!(
            sha(&row_bytes(&row)) == loaded.bundle.fixtures[*fixture].golden_sha256,
            "independent response differs from frozen singleton commitment"
        );
        compare(&row, &loaded.bundle.reference[*fixture])?;
        rows.push(row);
    }
    Ok(rows)
}

/// Check the uncommitted diagnostic log against committed outcomes.
///
/// This establishes semantic consistency and exact coverage, not original
/// elapsed-time authenticity or an original log-byte commitment.
fn completion_log(
    bytes: &[u8],
    outcomes: &[(u64, u16, Option<&str>)],
) -> Result<()> {
    let logs = core::jsonl(bytes)?;
    ensure!(logs.len() == outcomes.len(), "independent completion-log coverage");
    for (log, (id, status, code)) in logs.iter().zip(outcomes) {
        closed(log, &[
            "event", "request_id", "route", "status", "code", "elapsed_ms",
        ])?;
        let elapsed = log["elapsed_ms"].as_f64().context("completion elapsed_ms")?;
        ensure!(
            log["event"] == "request_complete"
                && log["request_id"] == *id
                && log["route"] == "embeddings"
                && log["status"] == *status
                && log["code"] == json!(code)
                && elapsed.is_finite()
                && elapsed >= 0.0,
            "independent diagnostic completion differs from committed outcome"
        );
    }
    Ok(())
}

/// Exact flat coverage from independently visited producer filenames.
fn independent_coverage(
    out: &Path,
    inventory: &BTreeMap<String, String>,
) -> Result<()> {
    let root = out.join("independent");
    regular_directory(&root)?;
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        ensure!(
            entry.file_type()?.is_file(),
            "independent subtree contains symlink or non-file"
        );
        let name = entry.file_name().into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 independent filename"))?;
        ensure!(actual.insert(name), "duplicate independent filename");
    }

    let mut expected = BTreeSet::new();
    for path in inventory.keys() {
        if let Some(name) = path.strip_prefix("independent/") {
            ensure!(
                Path::new(name).components().count() == 1
                    && matches!(
                        Path::new(name).components().next(),
                        Some(Component::Normal(_))
                    ),
                "non-flat independent inventory"
            );
            ensure!(expected.insert(name.to_owned()), "duplicate inventory name");
        }
    }
    ensure!(
        actual == expected,
        "missing or unaccounted independent artifact"
    );
    Ok(())
}

fn stable(out: &Path, inventory: &BTreeMap<String, String>) -> Result<()> {
    for (relative, digest) in inventory {
        ensure!(
            file_sha(&output_path(out, relative)?)? == *digest,
            "independent reconciliation artifact changed"
        );
    }
    Ok(())
}

pub(crate) fn reconcile(
    continuation: &crate::acceptance_scheduler::PoisonedLaunchArtifactsContinuation,
) -> Result<Value> {
    let preceding = &continuation.destructive_owners;
    ensure!(
        crate::acceptance_scheduler::lifecycle_prefix::saved_owners::
            saved_poison::reconcile(preceding)?
            == continuation.poisoned_launch_reconciliation,
        "preceding poison reconciliation changed"
    );

    let state = &preceding.recovery_http.recovery_lineage.initial_artifacts
        .imported_artifacts.local_artifacts.final_drain.state;
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
        "independent reconciliation called at wrong ledger boundary"
    );
    let ledger = bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)?;
    let transcript = bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)?;
    ensure!(
        ledger == state.ledger_prefix && transcript == state.orchestration_prefix,
        "retained independent reconciliation inputs changed"
    );
    let records = core::jsonl(&ledger)?;
    let cases = plan(loaded)?;
    let rows = phase_rows(&records, policy::INDEPENDENT);
    let expected_ids: Vec<String> = std::iter::once(STARTUP_CASE.to_owned())
        .chain(cases.iter().map(|case| case.id.clone()))
        .collect();
    ensure!(
        required_ids(loaded, policy::INDEPENDENT)? == expected_ids
            && rows.len() == expected_ids.len(),
        "independent required/actual phase coverage"
    );
    for (row, id) in rows.iter().zip(&expected_ids) {
        closed(row, &[
            "acceptance_phase", "case_id", "status", "pass", "details",
        ])?;
        ensure!(
            row["case_id"] == id.as_str()
                && row["status"] == "executed"
                && row["pass"] == true
                && row["details"].is_object(),
            "independent ledger order/status/schema"
        );
    }
    let joins = phase_rows(&records, policy::JOIN);
    ensure!(
        required_ids(loaded, policy::JOIN)? == vec![JOIN_CASE.to_owned()]
            && joins.len() == 1
            && joins[0]["case_id"] == JOIN_CASE
            && joins[0]["status"] == "executed"
            && joins[0]["pass"] == true
            && joins[0]["details"] == core_state.independent_join,
        "independent owner-join ledger/retained binding"
    );

    let startup_details = &rows[0]["details"];
    closed(startup_details, &["owner_binding_sha256", "startup", "observation"])?;
    let binding_digest = text(startup_details, "owner_binding_sha256")?;
    let mut inventory = BTreeMap::new();
    let binding_bytes = read_committed(
        out, "independent/owner_binding.json", binding_digest,
        JSON_LIMIT, &mut inventory,
    )?;
    let binding: Value = serde_json::from_slice(&binding_bytes)?;
    let instance = text(&binding, "instance")?;
    let pid = u32::try_from(uint(&binding, "pid")?)?;
    let address: SocketAddr = text(&binding, "http_address")?.parse()?;
    ensure!(
        label_valid(instance)
            && instance.starts_with("independent-")
            && crate::contract::digest_valid(
                instance.strip_prefix("independent-").context("independent prefix")?
            )
            && pid == std::process::id()
            && instance != core_state.initial_launch.instance
            && instance != old.launch.instance
            && instance != state.recovery.ready.replacement.instance
            && address.ip().is_loopback()
            && address.port() != 0
            && text(&binding, "http_address")? == address.to_string(),
        "independent owner identity/endpoint"
    );

    // initial_exit.json was reconstructed by the preceding initial pass.
    // Its exact bytes, not a reserialized alternative, bind owner creation.
    let exit_bytes = read_output(
        out, "initial_exit.json", JSON_LIMIT, &mut inventory,
    )?;
    ensure!(
        exit_bytes == serde_json::to_vec_pretty(&core_state.initial_exit)?,
        "initial exit producer bytes changed"
    );
    let expected_binding = json!({
        "schema": "rustv1.accept_m4_independent_owner.v1",
        "instance": instance,
        "pid": pid,
        "policy_sha256": loaded.policy_sha256,
        "driver_binary_sha256": loaded.artifacts["accept_m4"].sha256,
        "startup_check_sha256": loaded.artifacts["startup_check"].sha256,
        "tokenizer_sha256": loaded.artifacts["tokenizer"].sha256,
        "audit_identity": loaded.bundle.audit_identity,
        "previous_process_exit_evidence_sha256": sha(&exit_bytes),
        "execution_policy": "independent_sequence_v1",
        "batching": null,
        "backend_forward": "IndependentExecutor::forward_serving",
        "http_address": address.to_string()
    });
    ensure!(
        binding == expected_binding
            && binding_bytes == serde_json::to_vec_pretty(&expected_binding)?,
        "complete independent owner binding changed"
    );

    let join = &core_state.independent_join;
    ensure!(
        join["owner_binding_sha256"] == binding_digest,
        "independent startup/join owner binding differs"
    );
    let manifest = read_committed(
        out, "independent/executor_manifest.jsonl",
        text(join, "executor_manifest_sha256")?,
        MANIFEST_LIMIT, &mut inventory,
    )?;
    let manifest_records = core::jsonl(&manifest)?;
    let startup_size =
        usize::try_from(uint(&startup_details["startup"], "manifest_bytes")?)?;
    ensure!(
        startup_size > 0 && startup_size <= manifest.len(),
        "independent startup byte boundary"
    );
    let startup = core::check_startup(
        &manifest[..startup_size], loaded, binding_digest, false,
    )?;
    ensure!(
        startup_details["startup"] == startup,
        "complete independent startup reconciliation changed"
    );
    let initial_observation = &startup_details["observation"];
    ensure!(
        observation(
            initial_observation, instance, pid, &startup["startup_output_sha256"],
        )? == 0,
        "independent startup logical count"
    );

    let mut cursor = startup_size;
    let mut next = uint(&startup, "next_forward_id")?;
    let startup_forwards = next;
    let mut current_observation = initial_observation.clone();
    let mut last_request = 0u64;
    let mut successes = 0usize;
    let mut rejections = 0usize;
    let mut serving_forwards = 0u64;
    let mut outcomes = Vec::<(u64, u16, Option<&str>)>::new();

    for (row, case) in rows[1..].iter().zip(&cases) {
        let details = &row["details"];
        let request_relative = format!("independent/{}.request.bin", case.id);
        let response_relative = format!("independent/{}.response.bin", case.id);
        let request = read_committed(
            out, &request_relative, text(details, "request_sha256")?,
            JSON_LIMIT, &mut inventory,
        )?;
        ensure!(
            request == serde_json::to_vec(&case.body)?,
            "independent request differs from reconstructed plan"
        );
        let batched_request = read_output(
            out, &format!("http/{}.request.bin", case.id),
            JSON_LIMIT, &mut inventory,
        )?;
        ensure!(
            request == batched_request,
            "independent replay differs from original batched request bytes"
        );
        let response = read_committed(
            out, &response_relative, text(details, "response_sha256")?,
            JSON_LIMIT, &mut inventory,
        )?;
        let id = uint(details, "server_request_id")?;
        ensure!(
            id > last_request,
            "independent request ID reused, zero or reordered"
        );
        let suffix = interval(&manifest, &details["manifest_interval"], cursor)?;
        let end = usize::try_from(uint(&details["manifest_interval"], "end_byte")?)?;
        let after = &details["observation_after"];
        observation(after, instance, pid, &startup["startup_output_sha256"])?;

        let (status, code, execution) = match &case.expected {
            Expected::Rejection(code) => {
                ensure!(
                    suffix.is_empty()
                        && *after == current_observation
                        && serde_json::from_slice::<Value>(&response)?
                            == rejection_body(code)?,
                    "independent rejection envelope or zero-execution changed"
                );
                rejections += 1;
                (400u16, Some(*code), json!({
                    "physical_forwards": 0,
                    "logical_backend_request_delta": 0,
                    "observations_identical": true,
                    "manifest_bytes_identical": true,
                    "code": code,
                    "pad_token_coverage": *code == "pad_token",
                    "oov_token_coverage": false
                }))
            }
            Expected::Success(indices) => {
                logical_delta(&current_observation, after, 1)?;
                let output = outputs(&response, indices, loaded)?;
                let tokens: Vec<Vec<u32>> = indices.iter()
                    .map(|i| loaded.bundle.fixtures[*i].ids.clone())
                    .collect();
                let hashes: Vec<String> =
                    output.iter().map(|row| sha(&row_bytes(row))).collect();
                let interval_records = core::jsonl(suffix)?;
                let mut physical = core::reconcile_saved_serving_pairs(
                    &interval_records, next, &tokens, &hashes, loaded,
                )?;
                for (position, (physical, fixture)) in
                    physical.iter_mut().zip(indices).enumerate()
                {
                    physical["fixture_index"] = json!(fixture);
                    physical["fixture_id"] = json!(loaded.bundle.fixtures[*fixture].id);
                    physical["response_row"] = json!(position);
                    physical["singleton_exact_f32_bits"] = json!(true);
                    physical["independent_reference_comparison"] =
                        compare(&output[position], &loaded.bundle.reference[*fixture])?;
                }
                let count = u64::try_from(indices.len())?;
                next = next.checked_add(count).context("independent forward overflow")?;
                serving_forwards = serving_forwards.checked_add(count)
                    .context("independent serving count overflow")?;
                successes += 1;
                (200u16, None, json!({
                    "route": "IndependentExecutor::forward_serving",
                    "physical_forwards": indices.len(),
                    "logical_backend_request_delta": 1,
                    "rows": physical,
                    "association_source":
                        "serialized-http-request-observation-and-exclusive-manifest-interval",
                    "executor_emitted_server_request_ids": false
                }))
            }
        };

        let expected_details = json!({
            "instance": instance,
            "pid": pid,
            "server_request_id": id,
            "request_sha256": sha(&request),
            "request_bytes_identical_to_batched_phase": true,
            "response_sha256": sha(&response),
            "http_status": status,
            "observation_before": current_observation,
            "observation_after": after,
            "manifest_interval": {
                "start_byte": cursor,
                "end_byte": end,
                "prefix_before_sha256": sha(&manifest[..cursor]),
                "prefix_after_sha256": sha(&manifest[..end]),
                "interval_sha256": sha(suffix)
            },
            "execution": execution
        });
        ensure!(
            *details == expected_details,
            "complete independent producer details changed"
        );
        outcomes.push((id, status, code));
        current_observation = after.clone();
        cursor = end;
        last_request = id;
    }

    let expected_record_count = usize::try_from(next)?
        .checked_mul(2).and_then(|count| count.checked_add(1))
        .context("independent manifest record-count overflow")?;
    ensure!(
        cursor == manifest.len()
            && manifest_records.len() == expected_record_count
            && next == startup_forwards.checked_add(serving_forwards)
                .context("independent total forward overflow")?
            && successes > 0
            && rejections == 5
            && successes + rejections == cases.len(),
        "independent complete manifest/request coverage"
    );
    logical_delta(initial_observation, &current_observation, successes)?;

    let expected_join = json!({
        "instance": instance,
        "pid": pid,
        "owner_binding_sha256": binding_digest,
        "service_shutdown_returned_ok": true,
        "owner_thread_joined": true,
        "http_server_joined": true,
        "lifecycle": "stopped",
        "ready": false,
        "final_pre_shutdown_observation": current_observation,
        "executor_manifest_sha256": sha(&manifest),
        "executor_manifest_bytes": manifest.len(),
        "physical_forwards_including_startup": next,
        "startup_physical_forwards": startup_forwards,
        "unmatched_physical_records": 0,
        "later_cuda_launch_permitted_only_after_this_join": true
    });
    ensure!(
        *join == expected_join,
        "complete historical independent join proof changed"
    );
    expect_pretty(
        out, "independent/owner_join.json", &expected_join, &mut inventory,
    )?;

    let log_bytes = read_output(
        out, "independent/service_log.jsonl", MANIFEST_LIMIT, &mut inventory,
    )?;
    completion_log(&log_bytes, &outcomes)?;
    independent_coverage(out, &inventory)?;

    // Repeat the prior offline validation chain, not any live consumer.
    ensure!(
        crate::acceptance_scheduler::lifecycle_prefix::saved_owners::
            saved_poison::reconcile(preceding)?
            == continuation.poisoned_launch_reconciliation,
        "preceding artifacts changed during independent reconciliation"
    );
    loaded.recheck()?;
    stable(out, &inventory)?;
    independent_coverage(out, &inventory)?;
    ensure!(
        bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)? == ledger
            && bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)? == transcript,
        "independent reconciliation ledger/transcript changed"
    );

    Ok(json!({
        "schema": "rustv1.accept_m4_independent_artifact_reconciliation.v1",
        "boundary": INDEPENDENT_BOUNDARY,
        "policy_sha256": loaded.policy_sha256,
        "preceding_ledger_sha256": sha(&ledger),
        "orchestration_sha256": sha(&transcript),
        "preceding_poison_reconciliation_sha256":
            sha(&serde_json::to_vec(&continuation.poisoned_launch_reconciliation)?),
        "instance": instance,
        "pid": pid,
        "owner_binding_sha256": binding_digest,
        "owner_join_inventory_sha256": inventory["independent/owner_join.json"],
        "startup_manifest_sha256": sha(&manifest[..startup_size]),
        "final_manifest_sha256": sha(&manifest),
        "final_manifest_bytes": manifest.len(),
        "startup_physical_forwards": startup_forwards,
        "serving_physical_forwards": serving_forwards,
        "physical_forwards_including_startup": next,
        "successful_logical_requests": successes,
        "rejected_logical_requests": rejections,
        "mapped_request_count": cases.len(),
        "complete_independent_manifest_coverage": true,
        "unmatched_manifest_bytes": 0,
        "unmatched_physical_records": 0,
        "exact_output_subtrees": ["independent"],
        "independent_mode_artifacts_reconciled": true,
        "request_bytes_equal_batched_phase": true,
        "physical_b1_exact_f32_bits_checked": true,
        "executor_emitted_server_request_ids": false,
        "association_source":
            "serialized-http-request-observation-and-exclusive-manifest-interval",
        "historical_service_owner_join_preserved": true,
        "historical_http_server_join_preserved": true,
        "individual_http_connection_join_evidence_retained": false,
        "individual_http_connection_joins_claimed": false,
        "saved_response_headers_retained": false,
        "saved_response_header_revalidation_claimed": false,
        "historical_observations_relabelled": false,
        "diagnostic_files_without_original_byte_commitments": [
            "independent/service_log.jsonl"
        ],
        "diagnostic_log_semantic_coverage_checked": true,
        "diagnostic_elapsed_times_historically_authenticated": false,
        "diagnostic_hashes_are_original_historical_commitments": false,
        "verified_output_sha256": inventory,
        "scheduler_launch_and_coalescing_reconciliation_complete": false,
        "complete_scheduler_manifest_reconciled_here": false,
        "whole_transcript_semantics_reconciled": false,
        "complete_output_directory_coverage": false,
        "post_exit_control_calls": 0,
        "process_observations": 0,
        "owner_joins_performed": 0,
        "child_reaps_performed": 0,
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
    fn independent_intervals_reject_gaps_overlap_mutation_and_extra_fields() {
        let bytes = b"startup\nbegin\nend\n";
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
            b"begin\nend\n"
        );
        assert!(interval(bytes, &commitment, start - 1).is_err());
        assert!(interval(bytes, &commitment, start + 1).is_err());

        let mut changed = bytes.to_vec();
        changed[start] = b'X';
        assert!(interval(&changed, &commitment, start).is_err());

        let mut oversized = commitment.clone();
        oversized["end_byte"] = json!(bytes.len() + 1);
        assert!(interval(bytes, &oversized, start).is_err());

        let mut reversed = commitment.clone();
        reversed["end_byte"] = json!(start - 1);
        assert!(interval(bytes, &reversed, start).is_err());

        let empty = json!({
            "start_byte": start,
            "end_byte": start,
            "prefix_before_sha256": sha(&bytes[..start]),
            "prefix_after_sha256": sha(&bytes[..start]),
            "interval_sha256": sha(b"")
        });
        assert!(interval(bytes, &empty, start).unwrap().is_empty());

        let mut extra = commitment;
        extra["invented_request_id"] = json!(1);
        assert!(interval(bytes, &extra, start).is_err());
    }

    fn log_bytes(records: &[Value]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for record in records {
            serde_json::to_writer(&mut bytes, record).unwrap();
            bytes.push(b'\n');
        }
        bytes
    }

    #[test]
    fn independent_diagnostic_log_requires_exact_committed_outcomes() {
        // Disposable format fixtures, not acceptance execution evidence.
        let records = vec![
            json!({
                "event": "request_complete",
                "request_id": 1,
                "route": "embeddings",
                "status": 200,
                "code": null,
                "elapsed_ms": 1.5
            }),
            json!({
                "event": "request_complete",
                "request_id": 2,
                "route": "embeddings",
                "status": 400,
                "code": "empty_batch",
                "elapsed_ms": 0.0
            }),
        ];
        let expected = [(1u64, 200u16, None), (2, 400, Some("empty_batch"))];
        assert!(completion_log(&log_bytes(&records), &expected).is_ok());

        let mut duplicate = records.clone();
        duplicate.push(records[1].clone());
        assert!(completion_log(&log_bytes(&duplicate), &expected).is_err());
        assert!(completion_log(&log_bytes(&records[..1]), &expected).is_err());

        let mut reordered = records.clone();
        reordered.swap(0, 1);
        assert!(completion_log(&log_bytes(&reordered), &expected).is_err());

        let mut changed = records.clone();
        changed[1]["code"] = json!("pad_token");
        assert!(completion_log(&log_bytes(&changed), &expected).is_err());

        let mut negative = records.clone();
        negative[0]["elapsed_ms"] = json!(-1.0);
        assert!(completion_log(&log_bytes(&negative), &expected).is_err());

        let mut extra = records;
        extra[0]["owner_thread_joined"] = json!(true);
        assert!(completion_log(&log_bytes(&extra), &expected).is_err());
    }

    #[test]
    fn independent_observation_preserves_pre_shutdown_history() {
        let digest = json!(sha(b"startup"));
        let before = json!({
            "instance": "independent",
            "pid": 42,
            "lifecycle": "ready",
            "worker_alive": true,
            "ready": true,
            "forward_calls": 3,
            "startup_output_sha256": digest
        });
        assert_eq!(observation(&before, "independent", 42, &digest).unwrap(), 3);

        let mut stopped = before.clone();
        stopped["lifecycle"] = json!("stopped");
        stopped["worker_alive"] = json!(false);
        stopped["ready"] = json!(false);
        assert!(observation(&stopped, "independent", 42, &digest).is_err());

        let mut invented = before;
        invented["connection_tasks_joined"] = json!(true);
        assert!(observation(&invented, "independent", 42, &digest).is_err());
    }

    #[test]
    fn independent_files_reject_missing_extra_symlinks_and_closing_mutation() {
        let root = std::env::temp_dir().join(format!(
            "rv1-independent-files-{}-{}",
            std::process::id(),
            core::random_label().unwrap()
        ));
        private_directory(&root).unwrap();
        let subtree = root.join("independent");
        private_directory(&subtree).unwrap();

        let result: Result<()> = (|| {
            let path = subtree.join("owner_join.json");
            let original = b"{\"historical\":true}";
            write_new(&path, original)?;
            let mut inventory = BTreeMap::new();
            read_committed(
                &root, "independent/owner_join.json", &sha(original),
                JSON_LIMIT, &mut inventory,
            )?;
            independent_coverage(&root, &inventory)?;
            stable(&root, &inventory)?;

            let hidden = subtree.join("hidden");
            fs::rename(&path, &hidden)?;
            let missing = independent_coverage(&root, &inventory);
            fs::rename(&hidden, &path)?;
            ensure!(missing.is_err(), "missing independent artifact accepted");

            let extra = subtree.join("extra");
            write_new(&extra, b"x")?;
            let additional = independent_coverage(&root, &inventory);
            fs::remove_file(&extra)?;
            ensure!(additional.is_err(), "extra independent artifact accepted");

            let directory = subtree.join("unexpected-directory");
            private_directory(&directory)?;
            let nonfile = independent_coverage(&root, &inventory);
            fs::remove_dir(&directory)?;
            ensure!(nonfile.is_err(), "independent non-file accepted");

            #[cfg(unix)]
            {
                let link = subtree.join("linked");
                std::os::unix::fs::symlink(&path, &link)?;
                let linked = independent_coverage(&root, &inventory);
                fs::remove_file(&link)?;
                ensure!(linked.is_err(), "independent symlink accepted");
            }

            fs::write(&path, b"{\"historical\":false}")?;
            let changed = stable(&root, &inventory);
            fs::write(&path, original)?;
            ensure!(changed.is_err(), "closing independent mutation accepted");

            stable(&root, &inventory)?;
            independent_coverage(&root, &inventory)
        })();

        if result.is_ok() {
            fs::remove_dir_all(&root).unwrap();
        }
        result.unwrap_or_else(|error| {
            panic!(
                "independent file checks failed; retained {}: {error:#}",
                root.display()
            )
        });
    }
}
