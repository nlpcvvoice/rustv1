// Included inside acceptance_scheduler::lifecycle_prefix::saved_owners.
//
// Offline only. Reconstruct historical ownership evidence without joining,
// observing, launching, stopping, or reaping anything.
//
// Parent stdout/stderr lacked original producer commitments. Their current
// inventory hashes are stability checks, not historical authentication.

use super::*;

const BOUNDARY: &str = "DESTRUCTIVE-CHILDREN-AND-OWNERS-RECONCILED";
const FINAL_CASE: &str = "final-artifact-and-manifest-reconciliation";
const CHILD_CASES: [&str; 3] = [
    "family-57-drain-running-work",
    "family-58-physical-failure-fanout",
    "family-58-running-expiry-retirement",
];

struct Files<'a> {
    root: &'a Path,
    hashes: BTreeMap<String, String>,
}

impl<'a> Files<'a> {
    fn new(root: &'a Path) -> Result<Self> {
        ensure!(
            fs::symlink_metadata(root)?.file_type().is_dir(),
            "owner reconciliation root is not a non-symlink directory"
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
                && parts.iter().all(|p| matches!(p, Component::Normal(_))),
            "unsafe owner artifact path"
        );
        let mut path = self.root.to_owned();
        for (index, part) in parts.iter().enumerate() {
            let Component::Normal(name) = part else {
                anyhow::bail!("invalid owner artifact component");
            };
            path.push(name);
            let kind = fs::symlink_metadata(&path)?.file_type();
            ensure!(
                if index + 1 == parts.len() {
                    kind.is_file()
                } else {
                    kind.is_dir()
                },
                "symlink or unexpected owner artifact type"
            );
        }
        Ok(path)
    }

    fn read(&mut self, relative: &str) -> Result<Vec<u8>> {
        let bytes = bounded(&self.path(relative)?, MANIFEST_LIMIT)?;
        let digest = sha(&bytes);
        if let Some(previous) = self.hashes.get(relative) {
            ensure!(*previous == digest, "owner artifact changed between reads");
        } else {
            self.hashes.insert(relative.to_owned(), digest);
        }
        Ok(bytes)
    }

    fn committed(&mut self, relative: &str, digest: &str) -> Result<Vec<u8>> {
        ensure!(
            crate::contract::digest_valid(digest),
            "invalid owner artifact commitment"
        );
        let bytes = self.read(relative)?;
        ensure!(sha(&bytes) == digest, "owner artifact commitment: {relative}");
        Ok(bytes)
    }

    fn exact(&mut self, relative: &str, expected: &[u8]) -> Result<()> {
        ensure!(
            self.read(relative)? == expected,
            "owner artifact bytes differ: {relative}"
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
                "owner artifact changed during reconciliation"
            );
        }
        Ok(())
    }
}

fn jsonl_bytes(records: &[Value]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for record in records {
        serde_json::to_writer(&mut bytes, record)?;
        bytes.push(b'\n');
    }
    Ok(bytes)
}

fn ledger_prefix(bytes: &[u8], index: usize) -> Result<&[u8]> {
    let records = core::jsonl(bytes)?;
    ensure!(index <= records.len(), "owner ledger prefix index");
    let end: usize = bytes
        .split_inclusive(|byte| *byte == b'\n')
        .take(index)
        .map(<[u8]>::len)
        .sum();
    Ok(&bytes[..end])
}

/// Exact flat coverage using independently visited expected filenames.
fn flat(files: &Files<'_>, relative: &str) -> Result<()> {
    let root = if relative.is_empty() {
        files.root.to_owned()
    } else {
        files.root.join(relative)
    };
    ensure!(
        fs::symlink_metadata(&root)?.file_type().is_dir(),
        "owner subtree type"
    );
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        ensure!(entry.file_type()?.is_file(), "non-file owner subtree entry");
        let name = entry.file_name().into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 owner filename"))?;
        ensure!(actual.insert(name), "duplicate owner filename");
    }
    let prefix = if relative.is_empty() {
        String::new()
    } else {
        format!("{relative}/")
    };
    let mut expected = BTreeSet::new();
    for path in files.hashes.keys() {
        if let Some(name) = path.strip_prefix(&prefix) {
            ensure!(
                Path::new(name).components().count() == 1,
                "non-flat owner inventory"
            );
            expected.insert(name.to_owned());
        }
    }
    ensure!(actual == expected, "missing or extra owner subtree artifact");
    Ok(())
}

fn case_directory(files: &Files<'_>) -> Result<()> {
    let mut actual_files = BTreeSet::new();
    let mut directories = BTreeSet::new();
    for entry in fs::read_dir(files.root)? {
        let entry = entry?;
        let name = entry.file_name().into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 child parent filename"))?;
        if entry.file_type()?.is_file() {
            actual_files.insert(name);
        } else {
            ensure!(entry.file_type()?.is_dir(), "symlink in child parent");
            directories.insert(name);
        }
    }
    ensure!(
        directories == BTreeSet::from(["child".to_owned()]),
        "destructive parent directory coverage"
    );
    let expected: BTreeSet<String> = files.hashes.keys()
        .filter(|name| !name.contains('/'))
        .cloned()
        .collect();
    ensure!(actual_files == expected, "destructive parent file coverage");
    flat(files, "child")
}

fn closure(case: &str, value: &Value) -> Result<()> {
    let failure = case == CHILD_CASES[1];
    ensure!(CHILD_CASES.contains(&case), "unknown destructive closure case");
    closed(
        value,
        &[
            "service_shutdown_returned_ok",
            "owner_thread_joined",
            "shutdown_error",
            "http_server_joined",
            "client_tasks_drained",
            "http_connections",
            "lifecycle",
            "ready",
            "quarantine_cleared",
            "drop_counted_as_join",
            "shutdown_timeout_counted_as_join",
            "process_exit_required",
        ],
    )?;
    ensure!(
        value["service_shutdown_returned_ok"] == !failure
            && value["owner_thread_joined"] == !failure
            && value["shutdown_error"]
                == if failure { json!("deadline_exceeded") } else { Value::Null }
            && value["http_server_joined"] == true
            && value["client_tasks_drained"] == true
            && value["ready"] == false
            && value["quarantine_cleared"] == false
            && value["drop_counted_as_join"] == false
            && value["shutdown_timeout_counted_as_join"] == false
            && value["process_exit_required"] == failure
            && value["lifecycle"] == match case {
                "family-57-drain-running-work" => "stopped",
                "family-58-physical-failure-fanout" => "poisoned",
                _ => "retired",
            },
        "historical child closure changed or relabelled"
    );
    array(value, "http_connections")?;
    Ok(())
}

fn vector(token: u32) -> Vec<f32> {
    let mut row = vec![0.0; 2048];
    row[token as usize % 2048] = 1.0;
    row
}

fn child_bytes(
    files: &mut Files<'_>,
    hashes: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<Vec<u8>> {
    ensure!(
        Path::new(name).components().count() == 1
            && matches!(Path::new(name).components().next(), Some(Component::Normal(_))),
        "unsafe child filename"
    );
    let digest = hashes.get(name)
        .and_then(Value::as_str)
        .context("missing original child file commitment")?;
    files.committed(&format!("child/{name}"), digest)
}

fn child_pretty(
    files: &mut Files<'_>,
    hashes: &serde_json::Map<String, Value>,
    name: &str,
    expected: &Value,
) -> Result<()> {
    ensure!(
        child_bytes(files, hashes, name)? == serde_json::to_vec_pretty(expected)?,
        "child sidecar differs from bound value: {name}"
    );
    Ok(())
}

fn error_body(code: &str) -> Result<(u16, Value)> {
    let (status, message, kind) = match code {
        "inference_failed" => (500, "Inference failed", "api_error"),
        "deadline_exceeded" => (504, "Request deadline exceeded", "api_error"),
        "executor_poisoned" => (
            503,
            "Executor is retired; restart required",
            "service_unavailable_error",
        ),
        _ => anyhow::bail!("unexpected destructive response code"),
    };
    Ok((status, json!({
        "error": {
            "message": message,
            "type": kind,
            "param": null,
            "code": code
        }
    })))
}

/// Verify the child-owned files and the parent's spec/reap sidecars.
///
/// `child` must come from a preceding ledger-committed parent evidence object.
/// In the CPU harness it comes directly from spawn_probe after actual reap.
fn child_bundle(
    files: &mut Files<'_>,
    case: &str,
    child: &Value,
    limits: &Value,
    binary: &str,
    forbidden_pids: &[u32],
) -> Result<()> {
    ensure!(CHILD_CASES.contains(&case), "unknown destructive case");
    closed(child, &["exit", "identity", "evidence", "child_output_sha256"])?;
    let exit = &child["exit"];
    let identity = &child["identity"];
    let evidence = &child["evidence"];
    let probe = &evidence["probe"];
    let pid = u32::try_from(uint(exit, "pid")?)?;
    let parent_pid = std::process::id();
    let child_dir = files.root.join("child").canonicalize()?;

    ensure!(
        pid > 0 && pid != parent_pid && !forbidden_pids.contains(&pid),
        "child process identity/isolation"
    );
    let expected_exit = json!({
        "pid": pid,
        "parent_pid": parent_pid,
        "exit_code": 0,
        "success": true,
        "parent_watchdog_fired": false,
        "child_process_reaped": true,
        "process_drop_counted_as_reap": false,
        "evidence_directory": child_dir,
        "binary_sha256": binary
    });
    ensure!(*exit == expected_exit, "saved parent reap evidence changed");
    files.pretty("child_exit.json", &expected_exit)?;

    let nonce = text(evidence, "nonce")?;
    ensure!(crate::contract::digest_valid(nonce), "child nonce format");
    let expected_identity = json!({
        "pid": pid,
        "parent_pid": parent_pid,
        "nonce": nonce,
        "case_id": case,
        "evidence_directory": child_dir,
        "binary_sha256": binary
    });
    ensure!(*identity == expected_identity, "child identity changed");
    ensure!(
        *evidence == json!({
            "schema": "rustv1.accept_m4_destructive_child.v1",
            "case_id": case,
            "nonce": nonce,
            "pid": pid,
            "parent_pid": parent_pid,
            "evidence_directory": child_dir,
            "probe": probe,
            "child_checks_passed": true
        }),
        "child wrapper identity/schema"
    );
    files.pretty("child_spec.json", &json!({
        "case": case,
        "out": child_dir,
        "nonce": nonce,
        "limits": limits,
        "parent_pid": parent_pid
    }))?;

    let hashes = child["child_output_sha256"].as_object()
        .context("original child output inventory")?;
    child_pretty(files, hashes, "child_identity.json", identity)?;
    child_pretty(files, hashes, "evidence.json", evidence)?;

    let failure = case == CHILD_CASES[1];
    let expiry = case == CHILD_CASES[2];
    ensure!(
        probe["case_id"] == case
            && probe["pid"] == pid
            && label_valid(text(probe, "instance")?)
            && probe["batching"]
                == serde_json::to_value(rustv1::model::BatchedExecution::default())?
            && probe["semantic_clock"] == "manual_microseconds"
            && probe["request_timeout_us"] == 60_000_000
            && probe["request_timeout_shortened"] == false
            && probe["wall_watchdogs_are_semantic_time"] == false
            && probe["production_source_sha256"] == sha(include_bytes!("http_serve.rs"))
            && probe["production_scheduler_used"] == true
            && probe["backend"] == "synthetic_scheduler_probe"
            && probe["numerical_model_evidence"] == false
            && probe["process_termination_is_quarantine_cleanup"] == failure,
        "child probe identity/scope"
    );
    core::base_address(text(probe, "http_base")?)?;
    closure(case, &probe["cleanup"])?;
    child_pretty(files, hashes, "owner_closure.json", &probe["cleanup"])?;

    let mut semantic = probe.clone();
    let object = semantic.as_object_mut().context("child probe object")?;
    ensure!(
        object.remove("cleanup").is_some()
            && object.remove("process_termination_is_quarantine_cleanup").is_some(),
        "child cleanup projection"
    );
    child_pretty(files, hashes, "semantic_evidence.json", &semantic)?;

    let before = &probe["observation_before"];
    let instance = text(probe, "instance")?;
    let startup = json!({
        "kind": "startup",
        "tokens": [[100]],
        "output_sha256": [sha(&row_bytes(&vector(100)))]
    });
    ensure!(
        *before == json!({
            "instance": instance,
            "pid": pid,
            "lifecycle": "ready",
            "worker_alive": true,
            "ready": true,
            "forward_calls": 0,
            "startup_output_sha256": sha(&row_bytes(&vector(100)))
        }),
        "child startup observation"
    );
    let after = &probe["observation_after"];
    let expected_after = if case == CHILD_CASES[0] {
        json!({
            "lifecycle": "stopped",
            "ready": false,
            "owner_already_joined": true
        })
    } else {
        let mut expected = before.clone();
        expected["lifecycle"] = json!(if failure { "poisoned" } else { "retired" });
        expected["ready"] = json!(false);
        expected["forward_calls"] = json!(3);
        expected
    };
    ensure!(*after == expected_after, "historical child observation relabelled");
    child_pretty(files, hashes, "observation_before.json", before)?;
    child_pretty(files, hashes, "observation_after.json", after)?;

    let r = &probe["reconciliation"];
    let control = uint(r, "control_request_id")?;
    let members = array(r, "physical_failure_or_drain_members")?;
    ensure!(members.len() == 2, "destructive member coverage");
    let first = members[0].as_u64().context("first destructive request")?;
    let second = members[1].as_u64().context("second destructive request")?;
    let ids = [control, first, second];
    ensure!(
        ids.iter().all(|id| *id > 0)
            && ids.into_iter().collect::<BTreeSet<_>>().len() == 3
            && r["physical_partition"] == json!([[control], [first, second]])
            && r["physical_rows"] == json!([1, 3])
            && r["logical_requests"] == 3
            && r["physical_forwards"] == 2,
        "child physical partition"
    );
    for name in [
        "unmatched_insertions",
        "unmatched_physical_rows",
        "missing_responses",
        "duplicate_responses",
        "duplicate_production_completions",
    ] {
        ensure!(r[name] == 0, "child reconciliation nonzero counter: {name}");
    }

    let responses = r["responses"].as_object().context("child responses")?;
    ensure!(responses.len() == 3, "child response coverage");
    let tokens: [&[u32]; 3] = [&[90], &[100, 101], &[102]];
    let mut connections = Vec::new();
    let mut input_rows = Vec::new();
    let mut output_rows = Vec::new();

    for (position, (id, tokens)) in ids.iter().zip(tokens).enumerate() {
        let response = responses.get(&id.to_string()).context("child response ID")?;
        let request_case = format!("sched5758-request-{}", tokens[0]);
        let body = json!({
            "model": MODEL_ID,
            "input": tokens.iter().map(|t| format!("token-{t}")).collect::<Vec<_>>(),
            "encoding_format": "float"
        });
        let request = serde_json::to_vec(&body)?;
        ensure!(
            child_bytes(files, hashes, &format!("{request_case}.request.bin"))? == request,
            "destructive request plan changed"
        );
        let response_bytes =
            child_bytes(files, hashes, &format!("{request_case}.response.bin"))?;
        let response_body: Value = serde_json::from_slice(&response_bytes)?;
        let code = if position == 0 || case == CHILD_CASES[0] {
            None
        } else if failure {
            Some("inference_failed")
        } else if position == 1 {
            Some("deadline_exceeded")
        } else {
            Some("executor_poisoned")
        };

        let output_hashes = if let Some(code) = code {
            let (status, expected) = error_body(code)?;
            ensure!(
                response["http_status"] == status && response_body == expected,
                "child error envelope/status"
            );
            Value::Null
        } else {
            ensure!(response["http_status"] == 200, "child success status");
            closed(&response_body, &["object", "data", "model", "usage"])?;
            ensure!(
                response_body["object"] == "list"
                    && response_body["model"] == MODEL_ID
                    && response_body["usage"] == json!({
                        "prompt_tokens": tokens.len(),
                        "total_tokens": tokens.len()
                    }),
                "child success schema/usage"
            );
            let data = array(&response_body, "data")?;
            ensure!(data.len() == tokens.len(), "child output row coverage");
            let mut outputs = Vec::new();
            for (row_index, item) in data.iter().enumerate() {
                closed(item, &["object", "index", "embedding"])?;
                ensure!(
                    item["object"] == "embedding" && item["index"] == row_index,
                    "child output association"
                );
                outputs.push(serde_json::from_value::<Vec<f32>>(item["embedding"].clone())?);
            }
            let expected: Vec<Vec<f32>> = tokens.iter().map(|t| vector(*t)).collect();
            crate::contract::exact(&outputs, &expected)?;
            json!(outputs.iter().map(|row| sha(&row_bytes(row))).collect::<Vec<_>>())
        };

        let headers = &response["headers"];
        closed(headers, &[
            "x-rv1-instance-id",
            "x-acceptance-case-id",
            "x-rv1-request-id",
            "content-type",
            "_connection_owner",
        ])?;
        ensure!(
            headers["x-rv1-instance-id"] == instance
                && headers["x-acceptance-case-id"] == request_case
                && headers["x-rv1-request-id"] == *id
                && text(headers, "content-type")?.split(';').next()
                    .unwrap_or("").trim().eq_ignore_ascii_case("application/json"),
            "child saved response correlation"
        );
        let connection = &headers["_connection_owner"];
        ensure!(
            connection["instance"] == instance
                && connection["owner_pid"] == pid
                && connection["case_id"] == request_case
                && connection["method"] == "POST"
                && connection["path"] == "/v1/embeddings"
                && connection["request_sha256"] == sha(&request),
            "child connection/request binding"
        );
        connections.push(connection.clone());

        ensure!(
            *response == json!({
                "request_id": id,
                "case_id": request_case,
                "http_status": response["http_status"],
                "code": code,
                "headers": headers,
                "request_sha256": sha(&request),
                "response_sha256": sha(&response_bytes),
                "output_sha256": output_hashes
            }),
            "complete child response evidence changed"
        );
        child_pretty(files, hashes, &format!("response-{id}.json"), response)?;

        let mut starts = Vec::new();
        let mut returned = Vec::new();
        for (request_row, token) in tokens.iter().enumerate() {
            let physical_row = if position == 2 { 2 } else { request_row };
            let row = json!({
                "physical_row": physical_row,
                "request_id": id,
                "request_row": request_row,
                "case_id": request_case,
                "valid_length": 1,
                "token_sha256": crate::contract::ids_sha(&[*token])
            });
            let output = if failure && *token == 101 {
                vec![0.0; 2048]
            } else {
                vector(*token)
            };
            let mut completed = row.clone();
            completed["output_sha256"] = json!(sha(&row_bytes(&output)));
            starts.push(row);
            returned.push(completed);
        }
        input_rows.push(starts);
        output_rows.push(returned);
    }
    check_connections(instance, &connections, array(&probe["cleanup"], "http_connections")?)?;
    ensure!(
        r["outside_batch_control_unchanged"] == responses[&control.to_string()],
        "outside-batch control changed"
    );

    let mut records = vec![startup];
    for (forward, positions) in [vec![0usize], vec![1, 2]].into_iter().enumerate() {
        let rows: Vec<Value> = positions.iter()
            .flat_map(|p| input_rows[*p].clone()).collect();
        let returned: Vec<Value> = positions.iter()
            .flat_map(|p| output_rows[*p].clone()).collect();
        let b = rows.len();
        records.push(json!({
            "kind": "begin",
            "forward_id": forward,
            "B": b,
            "S": 1,
            "valid_tokens": b,
            "padding_tokens": 0,
            "input_rows": b,
            "association_rows": b,
            "rows": rows
        }));
        records.push(json!({
            "kind": "returned_rows",
            "forward_id": forward,
            "rows": returned,
            "injected_invalid_output_rows":
                if failure && forward == 1 { vec![1usize] } else { vec![] }
        }));
    }
    let manifest = jsonl_bytes(&records)?;
    ensure!(
        r["backend_records"] == json!(records)
            && r["backend_manifest_sha256"] == sha(&manifest)
            && child_bytes(files, hashes, "backend_manifest.jsonl")? == manifest
            && child_bytes(files, hashes, "running_manifest.jsonl")?
                == jsonl_bytes(&records[..4])?,
        "complete child backend/running-manifest coverage"
    );

    let events = array(r, "scheduler_events")?;
    for (index, event) in events.iter().enumerate() {
        ensure!(
            event["seq"] == index
                && matches!(event["kind"].as_str(), Some(
                    "inserted" | "checkpoint" | "assembly" | "assembly_receive" | "launch"
                )),
            "child scheduler sequence/kind"
        );
    }
    let inserted: Vec<&Value> = events.iter()
        .filter(|event| event["kind"] == "inserted").collect();
    let launches: Vec<&Value> = events.iter()
        .filter(|event| event["kind"] == "launch").collect();
    ensure!(inserted.len() == 3 && launches.len() == 2, "child scheduler coverage");
    for (position, event) in inserted.iter().enumerate() {
        let now = match position {
            0 => 0,
            1 => 1000,
            _ if expiry => 1100,
            _ => 1000,
        };
        ensure!(
            event["request_id"] == ids[position]
                && event["case_id"] == format!("sched5758-request-{}", tokens[position][0])
                && event["B"] == tokens[position].len()
                && event["S"] == 1
                && event["inserted_us"] == now
                && event["now_us"] == now
                && event["deadline_us"] == now + 60_000_000u64,
            "child insertion plan/timing"
        );
    }
    for (index, members) in [json!([control]), json!([first, second])]
        .into_iter().enumerate()
    {
        ensure!(
            launches[index]["request_ids"] == members
                && launches[index]["B"] == if index == 0 { 1 } else { 3 }
                && launches[index]["S"] == 1
                && launches[index]["now_us"] == if index == 0 { 1000 } else { 2000 }
                && launches[index]["boundary_us"].is_null(),
            "child launch association/timing"
        );
    }
    let assembled = &probe["assembled"];
    ensure!(
        assembled["kind"] == "checkpoint"
            && assembled["stage"] == "assembled"
            && assembled["request_ids"] == json!([first, second])
            && assembled["B"] == 3
            && assembled["S"] == 1
            && assembled["now_us"] == 2000
            && assembled["boundary_us"] == 2000
            && events.get(usize::try_from(uint(assembled, "seq")?)?) == Some(assembled),
        "child assembled witness"
    );
    // All three requests were inserted and both launches occurred before
    // the destructive backend gate was opened. No later scheduler work exists.
    child_pretty(files, hashes, "running_events.json", &json!(events))?;
    child_pretty(files, hashes, "final_scheduler_events.json", &json!(events))?;

    let witness = if case == CHILD_CASES[0] {
        json!({
            "shutdown_initial_poll_pending": true,
            "running_batch_held_when_drain_started": true,
            "production_shutdown_completed": true,
            "owner_thread_joined_by_measured_shutdown": true,
            "outcome": "running_work_completed_before_owner_join"
        })
    } else if failure {
        json!({
            "fault": "one_zero_norm_physical_output_row",
            "fault_physical_row": 1,
            "fault_request_id": first,
            "fault_request_row": 1,
            "production_failure_gate": "http_serve::run_batch -> validate_output",
            "backend_implements_fanout": false,
            "backend_implements_poison": false,
            "queued_requests_outside_failed_batch": 0,
            "queued_requests_survive_poison_claimed": false
        })
    } else {
        child_pretty(files, hashes, "retired_while_backend_held.json", &json!({
            "now_us": 60_001_000u64,
            "lifecycle": "retired",
            "ready": false,
            "backend_manifest_sha256": sha(&jsonl_bytes(&records[..4])?),
            "responses": responses
        }))?;
        json!({
            "running_deadline_expiry": "retire_executor",
            "expired_request_id": first,
            "expired_deadline_us": 60_001_000u64,
            "other_member_request_id": second,
            "other_member_deadline_us": 60_001_100u64,
            "retired_before_backend_return": true,
            "http_success_after_expiry": false,
            "production_error_codes": ["deadline_exceeded", "executor_poisoned"]
        })
    };
    ensure!(probe["witness"] == witness, "historical destructive witness changed");

    let log_bytes = child_bytes(files, hashes, "service_log.jsonl")?;
    ensure!(
        r["production_completion_log_sha256"] == sha(&log_bytes),
        "child completion log commitment"
    );
    let logs = core::jsonl(&log_bytes)?;
    ensure!(logs.len() == 3, "child completion log count");
    let mut logged = BTreeSet::new();
    for log in logs {
        closed(&log, &["event", "request_id", "route", "status", "code", "elapsed_ms"])?;
        let id = uint(&log, "request_id")?;
        let response = responses.get(&id.to_string()).context("unknown child completion")?;
        let elapsed = log["elapsed_ms"].as_f64().context("completion elapsed time")?;
        ensure!(
            logged.insert(id)
                && log["event"] == "request_complete"
                && log["route"] == "embeddings"
                && log["status"] == response["http_status"]
                && log["code"] == response["code"]
                && elapsed.is_finite()
                && elapsed >= 0.0,
            "child completion log/response association"
        );
    }
    ensure!(logged == ids.into_iter().collect(), "child completion coverage");

    // The expected set comes from the producer filenames visited above,
    // not merely from iterating the supplied hash table.
    let visited: BTreeSet<String> = files.hashes.keys()
        .filter_map(|path| path.strip_prefix("child/").map(str::to_owned))
        .collect();
    ensure!(
        visited == hashes.keys().cloned().collect(),
        "unrecognized or missing child artifact commitment"
    );
    flat(files, "child")?;

    // Producer did not retain commitments for these parent diagnostics.
    // Do not use their contents as execution/reap proof.
    files.read("child_stdout.log")?;
    files.read("child_stderr.log")?;
    Ok(())
}

fn bound_case(
    files: &mut Files<'_>,
    row: &Value,
    relative: &str,
    loaded: &Loaded,
) -> Result<(Value, String)> {
    ensure!(
        row["status"] == "executed"
            && row["pass"] == true
            && row["details"]["evidence_path"] == relative,
        "owner source ledger binding"
    );
    let digest = text(&row["details"], "evidence_sha256")?.to_owned();
    let bytes = files.committed(relative, &digest)?;
    let evidence: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        evidence["case_id"] == row["case_id"]
            && evidence["policy_sha256"] == loaded.policy_sha256
            && evidence["driver_binary_sha256"] == loaded.artifacts["accept_m4"].sha256,
        "owner source policy/driver/case binding"
    );
    Ok((evidence, digest))
}

/// Rebuild the exact saved inventory using only saved connection records.
///
/// The local_inventory/collect_requests/owner helpers are pure constructors.
/// Their historical classifications are reused unchanged.
fn rebuild_inventory(
    old: &SchedulerState,
    sources: &[(String, String, Value)],
    saved: &Value,
    preceding_ledger: &[u8],
) -> Result<Value> {
    ensure!(sources.len() == 14, "owner source coverage");
    let mut owners = Vec::new();
    let mut expected_connections = BTreeMap::<String, Vec<Value>>::new();
    let mut child_connections = BTreeMap::<String, Vec<Value>>::new();
    let mut source_table = Vec::new();
    let coalescing = &sources[0].2;
    let mut request_ids = BTreeSet::new();

    for (position, (path, digest, evidence)) in sources.iter().enumerate() {
        let case = text(evidence, "case_id")?;
        source_table.push(source(path, digest, ""));
        if position == 0 {
            ensure!(
                digest == &old.coalescing_evidence_sha256
                    && evidence["instance"] == old.launch.instance
                    && evidence["pid"] == old.launch.pid,
                "owner coalescing source binding"
            );
            for (index, wave) in array(evidence, "waves")?.iter().enumerate() {
                let responses = array(wave, "responses")?;
                for response in responses {
                    ensure!(
                        request_ids.insert(uint(response, "request_id")?),
                        "duplicate coalescing owner request ID"
                    );
                }
                collect_requests(
                    &mut owners,
                    &mut expected_connections,
                    &old.launch.instance,
                    u64::from(std::process::id()),
                    responses,
                    true,
                    &source(path, digest, &format!("/waves/{index}/responses")),
                )?;
            }
        } else if position == 1 {
            local_inventory(
                &mut owners, &mut expected_connections, evidence,
                &source(path, digest, ""), true, false, false,
            )?;
        } else if position <= 10 {
            let names = probe_names(case)?;
            let probes = array(evidence, "probes")?;
            ensure!(names.len() == probes.len(), "owner local probe count");
            for (index, (name, probe)) in names.iter().zip(probes).enumerate() {
                ensure!(probe["probe"] == *name, "owner local probe order");
                local_inventory(
                    &mut owners, &mut expected_connections, probe,
                    &source(path, digest, &format!("/probes/{index}")),
                    true, false, false,
                )?;
            }
        } else {
            ensure!(case == CHILD_CASES[position - 11], "owner child order");
            let child = &evidence["child"];
            let probe = &child["evidence"]["probe"];
            let exit = &child["exit"];
            let instance = text(probe, "instance")?;
            let pid = uint(probe, "pid")?;
            local_inventory(
                &mut owners, &mut expected_connections, probe,
                &source(path, digest, "/child/evidence/probe"),
                false, case == CHILD_CASES[1], true,
            )?;
            ensure!(
                child_connections.insert(
                    instance.to_owned(),
                    array(&probe["cleanup"], "http_connections")?.clone(),
                ).is_none(),
                "child owner instance reused"
            );
            for kind in ["child_watchdog_thread", "child_tokio_runtime_threads"] {
                owner(
                    &mut owners, instance, pid, kind, kind,
                    "terminated-with-reaped-child-not-joined",
                    &source(path, digest, "/child/exit"), exit.clone(),
                );
            }
            owner(
                &mut owners, instance, pid, "child_process", "owned-child", "reaped",
                &source(path, digest, "/child/exit"), exit.clone(),
            );
        }
    }

    let saved_connections = array(saved, "http_connections")?;
    let mut connection_evidence = Vec::new();
    for (instance, expected) in &expected_connections {
        let actual: Vec<Value> = saved_connections.iter()
            .filter(|record| record["instance"] == instance.as_str())
            .cloned().collect();
        check_connections(instance, expected, &actual)?;
        if let Some(child) = child_connections.remove(instance) {
            ensure!(actual == child, "child closure/owner connection order changed");
        }
        // Original registry traversal is ascending connection_id.
        let mut previous = None;
        for record in &actual {
            let id = uint(record, "connection_id")?;
            ensure!(
                previous.is_none_or(|old| id > old),
                "saved connection registry order changed"
            );
            previous = Some(id);
            owner(
                &mut owners,
                instance,
                uint(record, "owner_pid")?,
                "http_connection_task",
                &id.to_string(),
                text(record, "join_outcome")?,
                &json!({"evidence_origin": "preceding_digest_bound_response_headers"}),
                record.clone(),
            );
        }
        connection_evidence.extend(actual);
    }
    ensure!(
        child_connections.is_empty()
            && connection_evidence == *saved_connections,
        "unmatched or reordered saved owner connections"
    );

    let mut identities = BTreeSet::new();
    for entry in &owners {
        ensure!(
            identities.insert((
                uint(entry, "pid")?,
                text(entry, "instance")?.to_owned(),
                text(entry, "owner_kind")?.to_owned(),
                text(entry, "owner_identity")?.to_owned(),
            )),
            "duplicate reconstructed owner identity"
        );
    }

    Ok(json!({
        "schema": "rustv1.accept_m4_scheduler_owner_inventory.v1",
        "case_id": OWNERS,
        "preceding_ledger_sha256": sha(preceding_ledger),
        "preceding_scheduler_case_count": sources.len(),
        "sources": source_table,
        "owners": owners,
        "http_connections": connection_evidence,
        "retained_outside_local_closure_scope": [
            {
                "kind": "coordinator_runtime",
                "disposition": "returned-in-continuation-not-joined"
            },
            {
                "kind": "authenticated_orchestration_connection",
                "disposition": "returned-in-continuation"
            },
            {
                "kind": "replacement_server",
                "instance": old.launch.instance,
                "pid": old.launch.pid,
                "disposition": "preserved-live-for-lifecycle-cases"
            }
        ],
        "child_local_closures_preserved": true,
        "parent_reap_is_distinct_containment_boundary": true,
        "summary_booleans_used_as_owner_coverage": false,
        "live_observation": coalescing["final_observation"],
        "live_manifest_sha256": coalescing["final_manifest_sha256"],
        "next_forward": coalescing["next_forward_id"],
        "request_ids": request_ids,
        "policy_sha256": old.core.loaded.policy_sha256,
        "driver_binary_sha256": old.core.loaded.artifacts["accept_m4"].sha256
    }))
}

pub(crate) fn reconcile(
    continuation: &crate::acceptance_scheduler::RecoveryHttpArtifactsContinuation,
) -> Result<Value> {
    ensure!(
        restart_prefix::reattestation::exact_recovery::final_drain::
            saved_lineage::saved_http::reconcile(&continuation.recovery_lineage)?
            == continuation.recovery_http_reconciliation,
        "preceding recovery HTTP reconciliation changed"
    );

    let state = &continuation.recovery_lineage.initial_artifacts
        .imported_artifacts.local_artifacts.final_drain.state;
    let old = &state.recovery.ready.poison_exit.old;
    let loaded = &old.core.loaded;
    let out = &old.core.out;
    loaded.recheck()?;
    ensure!(
        old.core.ledger.pending() == vec![json!({
            "acceptance_phase": policy::LIFECYCLE,
            "case_id": FINAL_CASE
        })],
        "owner reconciliation called at wrong ledger boundary"
    );
    let ledger = bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)?;
    let transcript = bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)?;
    ensure!(
        ledger == state.ledger_prefix && transcript == state.orchestration_prefix,
        "retained owner reconciliation inputs changed"
    );
    let records = core::jsonl(&ledger)?;
    let scheduler: Vec<(usize, &Value)> = records.iter().enumerate()
        .filter(|(_, record)| record["acceptance_phase"] == policy::SCHEDULER)
        .collect();
    let required = &loaded.required.phases.iter()
        .find(|(phase, _)| phase == policy::SCHEDULER)
        .context("required scheduler phase")?.1;
    ensure!(
        scheduler.len() == 15
            && required.len() == 15
            && required.last().map(String::as_str) == Some(OWNERS),
        "complete scheduler source ledger coverage"
    );

    let mut files = Files::new(out)?;
    let mut sources = Vec::new();
    for (position, ((_, row), case)) in scheduler.iter().zip(required).enumerate() {
        ensure!(row["case_id"] == case.as_str(), "scheduler source order");
        let relative = match position {
            0 => "scheduler/family-53.evidence.json".to_owned(),
            1 => "scheduler/family-54-physical-b/evidence.json".to_owned(),
            _ => format!("scheduler/{case}/evidence.json"),
        };
        let (evidence, digest) = bound_case(&mut files, row, &relative, loaded)?;
        if position < 14 {
            sources.push((relative, digest, evidence));
        }
    }

    let coalescing = &sources[0].2;
    ensure!(
        coalescing["final_manifest_sha256"] == sha(&old.manifest_prefix)
            && coalescing["next_forward_id"] == old.next_forward,
        "local/destructive work changed historical physical cursor"
    );
    let mut destructive_summaries = Vec::new();
    let mut diagnostics = Vec::new();

    for (index, case) in CHILD_CASES.iter().enumerate() {
        let (relative, digest, evidence) = &sources[11 + index];
        let preserved = &evidence["live_reconciliation_unchanged"];
        let mut ids = BTreeSet::new();
        for wave in array(coalescing, "waves")? {
            for response in array(wave, "responses")? {
                ensure!(ids.insert(uint(response, "request_id")?), "coalescing duplicate ID");
            }
        }
        let expected_preserved = json!({
            "observation": coalescing["final_observation"],
            "manifest_sha256": sha(&old.manifest_prefix),
            "manifest_bytes": old.manifest_prefix.len(),
            "next_forward": old.next_forward,
            "request_ids": ids
        });
        ensure!(
            *preserved == expected_preserved
                && evidence["schema"] == "rustv1.accept_m4_destructive_case.v1"
                && evidence["live_instance"] == old.launch.instance
                && evidence["live_pid"] == old.launch.pid
                && evidence["child_evidence_directory_separate"] == true
                && evidence["child_process_reaped_before_ledger_pass"] == true
                && evidence["production_scheduler_used"] == true
                && evidence["backend"] == "synthetic_scheduler_probe"
                && evidence["numerical_model_evidence"] == false,
            "destructive parent isolation/scope"
        );

        let root_relative = format!("scheduler/{case}");
        // Validates ancestors before constructing the scoped file inventory.
        files.path(relative)?;
        let root = out.join(&root_relative);
        let mut child_files = Files::new(&root)?;
        child_files.committed("evidence.json", digest)?;
        child_files.pretty("live_before.json", preserved)?;
        child_files.pretty("live_after_check.json", &json!({
            "unchanged": true,
            "preserved": preserved,
            "error": null
        }))?;
        child_bundle(
            &mut child_files,
            case,
            &evidence["child"],
            &loaded.policy["limits"],
            &loaded.artifacts["accept_m4"].sha256,
            &[
                old.launch.pid,
                old.core.initial_launch.pid,
                state.recovery.ready.replacement.pid,
            ],
        )?;
        case_directory(&child_files)?;
        child_files.stable()?;
        case_directory(&child_files)?;

        for (name, digest) in &child_files.hashes {
            files.committed(&format!("{root_relative}/{name}"), digest)?;
        }
        for name in ["child_stdout.log", "child_stderr.log"] {
            diagnostics.push(format!("{root_relative}/{name}"));
        }
        destructive_summaries.push(json!({
            "case_id": case,
            "parent_evidence_sha256": digest,
            "pid": evidence["child"]["exit"]["pid"],
            "instance": evidence["child"]["evidence"]["probe"]["instance"],
            "saved_parent_reap_checked": true,
            "owner_thread_joined":
                evidence["child"]["evidence"]["probe"]["cleanup"]["owner_thread_joined"],
            "quarantine_cleared": false,
            "exact_subtree_coverage": true
        }));
    }

    let (owner_index, owner_row) = scheduler[14];
    let owner_path = format!("scheduler/{OWNERS}/evidence.json");
    let (saved, owner_digest) = bound_case(&mut files, owner_row, &owner_path, loaded)?;
    ensure!(
        owner_row["details"]["instance"] == old.launch.instance
            && owner_row["details"]["pid"] == old.launch.pid,
        "owner inventory ledger launch identity"
    );
    let expected = rebuild_inventory(
        old, &sources, &saved, ledger_prefix(&ledger, owner_index)?,
    )?;
    ensure!(saved == expected, "complete saved scheduler owner inventory changed");
    files.pretty(&owner_path, &expected)?;
    flat(&files, &format!("scheduler/{OWNERS}"))?;

    // Repeat the preceding offline chain, not any live boundary consumer.
    ensure!(
        restart_prefix::reattestation::exact_recovery::final_drain::
            saved_lineage::saved_http::reconcile(&continuation.recovery_lineage)?
            == continuation.recovery_http_reconciliation,
        "preceding recovery HTTP artifacts changed during owner reconciliation"
    );
    loaded.recheck()?;
    files.stable()?;
    flat(&files, &format!("scheduler/{OWNERS}"))?;
    for case in CHILD_CASES {
        let root_relative = format!("scheduler/{case}");
        let root = out.join(&root_relative);
        let scoped = Files {
            root: &root,
            hashes: files.hashes.iter()
                .filter_map(|(path, digest)| {
                    path.strip_prefix(&format!("{root_relative}/"))
                        .map(|name| (name.to_owned(), digest.clone()))
                }).collect(),
        };
        case_directory(&scoped)?;
    }
    ensure!(
        bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)? == ledger
            && bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)? == transcript,
        "owner reconciliation ledger/transcript changed"
    );

    Ok(json!({
        "schema": "rustv1.accept_m4_destructive_owner_reconciliation.v1",
        "boundary": BOUNDARY,
        "policy_sha256": loaded.policy_sha256,
        "preceding_ledger_sha256": sha(&ledger),
        "orchestration_sha256": sha(&transcript),
        "preceding_recovery_http_reconciliation_sha256":
            sha(&serde_json::to_vec(&continuation.recovery_http_reconciliation)?),
        "destructive_children": destructive_summaries,
        "destructive_children_reconciled": true,
        "saved_scheduler_owner_inventory_sha256": owner_digest,
        "saved_scheduler_owner_inventory_reconstructed_exactly": true,
        "saved_owner_entries_checked": array(&saved, "owners")?.len(),
        "saved_connection_entries_checked": array(&saved, "http_connections")?.len(),
        "original_unjoined_child_owner_closure_preserved": true,
        "parent_reap_distinct_from_owner_join": true,
        "parent_diagnostics_without_original_byte_commitments": diagnostics,
        "parent_diagnostics_used_as_execution_or_reap_proof": false,
        "verified_output_sha256": files.hashes,
        "complete_output_directory_coverage": false,
        "complete_poisoned_launch_artifact_set_reconciled": false,
        "independent_mode_artifacts_reconciled": false,
        "complete_scheduler_manifest_reconciled_here": false,
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

/// Exercise the same offline child validator on actual, already-reaped
/// disposable harness output. This does not synthesize execution evidence.
#[cfg(test)]
pub(crate) fn child_mutation_checks(
    root: &Path,
    case: &str,
    child: &Value,
    limits: &Value,
) -> Result<()> {
    let binary = file_sha(&std::env::current_exe()?)?;
    let verify = || -> Result<()> {
        let mut files = Files::new(root)?;
        child_bundle(&mut files, case, child, limits, &binary, &[])?;
        files.stable()
    };
    verify()?;

    let closure_path = root.join("child/owner_closure.json");
    let original = bounded(&closure_path, JSON_LIMIT)?;
    fs::write(&closure_path, b"{}")?;
    let changed = verify();
    fs::write(&closure_path, &original)?;
    ensure!(changed.is_err(), "changed historical closure accepted");

    let hidden = root.join("child/owner_closure.saved");
    fs::rename(&closure_path, &hidden)?;
    let missing = verify();
    fs::rename(&hidden, &closure_path)?;
    ensure!(missing.is_err(), "missing child closure accepted");

    let extra = root.join("child/unaccounted.bin");
    write_new(&extra, b"unexpected")?;
    let extra_result = verify();
    fs::remove_file(&extra)?;
    ensure!(extra_result.is_err(), "extra child file accepted");

    let exit_path = root.join("child_exit.json");
    let original_exit = bounded(&exit_path, JSON_LIMIT)?;
    let mut changed_exit = child["exit"].clone();
    changed_exit["child_process_reaped"] = json!(false);
    fs::write(&exit_path, serde_json::to_vec_pretty(&changed_exit)?)?;
    let changed_reap = verify();
    fs::write(&exit_path, &original_exit)?;
    ensure!(changed_reap.is_err(), "changed parent reap accepted");

    verify()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn original_closure(case: &str) -> Value {
        let failure = case == CHILD_CASES[1];
        json!({
            "service_shutdown_returned_ok": !failure,
            "owner_thread_joined": !failure,
            "shutdown_error": if failure { Some("deadline_exceeded") } else { None },
            "http_server_joined": true,
            "client_tasks_drained": true,
            "http_connections": [],
            "lifecycle": match case {
                "family-57-drain-running-work" => "stopped",
                "family-58-physical-failure-fanout" => "poisoned",
                _ => "retired"
            },
            "ready": false,
            "quarantine_cleared": false,
            "drop_counted_as_join": false,
            "shutdown_timeout_counted_as_join": false,
            "process_exit_required": failure
        })
    }

    #[test]
    fn destructive_closures_preserve_unjoined_poison_and_joined_retirement() {
        for case in CHILD_CASES {
            assert!(closure(case, &original_closure(case)).is_ok());
        }
        let mut joined = original_closure(CHILD_CASES[1]);
        joined["owner_thread_joined"] = json!(true);
        assert!(closure(CHILD_CASES[1], &joined).is_err());

        let mut stopped = original_closure(CHILD_CASES[1]);
        stopped["lifecycle"] = json!("stopped");
        assert!(closure(CHILD_CASES[1], &stopped).is_err());

        let mut cleared = original_closure(CHILD_CASES[1]);
        cleared["quarantine_cleared"] = json!(true);
        assert!(closure(CHILD_CASES[1], &cleared).is_err());

        let mut retired = original_closure(CHILD_CASES[2]);
        retired["lifecycle"] = json!("stopped");
        assert!(closure(CHILD_CASES[2], &retired).is_err());
    }

    #[test]
    fn owner_prefix_preserves_original_jsonl_bytes() {
        let bytes = b"{ \"a\": 1 }\n{\"b\":2}\n";
        assert_eq!(ledger_prefix(bytes, 1).unwrap(), b"{ \"a\": 1 }\n");
        assert_eq!(ledger_prefix(bytes, 2).unwrap(), bytes);
        assert!(ledger_prefix(bytes, 3).is_err());
        assert!(ledger_prefix(b"{}\n\n", 1).is_err());
        assert!(ledger_prefix(b"{}", 1).is_err());
    }

    #[test]
    fn owner_paths_and_coverage_reject_traversal_symlinks_and_extra_files() {
        let root = std::env::temp_dir().join(format!(
            "rv1-owner-coverage-{}-{}",
            std::process::id(),
            core::random_label().unwrap()
        ));
        private_directory(&root).unwrap();

        let result: Result<()> = (|| {
            write_new(&root.join("evidence.json"), b"{}")?;
            let mut files = Files::new(&root)?;
            files.exact("evidence.json", b"{}")?;
            flat(&files, "")?;
            ensure!(files.path("../evidence.json").is_err(), "traversal accepted");
            ensure!(files.path("/tmp/evidence.json").is_err(), "absolute path accepted");

            let extra = root.join("extra");
            write_new(&extra, b"x")?;
            ensure!(flat(&files, "").is_err(), "extra file accepted");
            fs::remove_file(&extra)?;

            #[cfg(unix)]
            {
                let link = root.join("link");
                std::os::unix::fs::symlink(root.join("evidence.json"), &link)?;
                ensure!(files.path("link").is_err(), "symlink accepted");
                ensure!(flat(&files, "").is_err(), "symlink coverage accepted");
                fs::remove_file(&link)?;
            }

            fs::write(root.join("evidence.json"), b"{\"changed\":true}")?;
            ensure!(files.stable().is_err(), "changed artifact accepted");
            fs::write(root.join("evidence.json"), b"{}")?;
            files.stable()?;
            flat(&files, "")
        })();

        if result.is_ok() {
            fs::remove_dir_all(&root).unwrap();
        }
        result.unwrap_or_else(|error| {
            panic!("owner coverage failed; retained {}: {error:#}", root.display())
        });
    }
}
