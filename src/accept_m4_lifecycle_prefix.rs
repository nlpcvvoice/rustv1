use super::*;
use std::path::Component;

const OWNERS: &str = "scheduler-owners-joined";
const BINDING: &str = "replacement-launch-after-independent-join";
const DRIFT: &str = "family-58-live-policy-drift-poison";
const TERMINAL: &str = "family-58-terminal-liveness-readiness-refusal";
const NEXT: &str = "family-58-restore-does-not-recover";

fn array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    value[key]
        .as_array()
        .with_context(|| format!("missing array: {key}"))
}

fn under(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "unsafe evidence path"
    );
    let root = root.canonicalize()?;
    let path = root.join(relative).canonicalize()?;
    ensure!(
        path.starts_with(root) && path.is_file(),
        "evidence escapes root"
    );
    Ok(path)
}

async fn unchanged(state: &SchedulerState) -> Result<()> {
    state.launch.process.check_live()?;
    ensure!(
        state.core.control.call("observe", None).await? == state.observation
            && bounded(&state.launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
        "unreconciled replacement observation/manifest activity"
    );
    Ok(())
}

fn source(path: &str, digest: &str, pointer: &str) -> Value {
    json!({
        "evidence_path": path,
        "evidence_sha256": digest,
        "json_pointer": pointer
    })
}

#[allow(clippy::too_many_arguments)]
fn owner(
    inventory: &mut Vec<Value>,
    instance: &str,
    pid: u64,
    kind: &str,
    identity: &str,
    outcome: &str,
    source: &Value,
    proof: Value,
) {
    inventory.push(json!({
        "instance": instance,
        "pid": pid,
        "owner_kind": kind,
        "owner_identity": identity,
        "outcome": outcome,
        "source": source,
        "proof": proof
    }));
}

fn response_values(value: &Value) -> Result<Vec<Value>> {
    if let Some(values) = value.as_array() {
        Ok(values.clone())
    } else if let Some(values) = value.as_object() {
        Ok(values.values().cloned().collect())
    } else {
        anyhow::bail!("missing response inventory")
    }
}

fn collect_requests(
    inventory: &mut Vec<Value>,
    expected_connections: &mut BTreeMap<String, Vec<Value>>,
    instance: &str,
    pid: u64,
    responses: &[Value],
    spawned: bool,
    source: &Value,
) -> Result<()> {
    for response in responses {
        let case = text(response, "case_id")?;
        let connection = &response["headers"]["_connection_owner"];
        ensure!(
            connection.is_object()
                && connection["instance"] == instance
                && connection["owner_pid"] == pid
                && connection["case_id"] == case
                && connection["method"] == "POST"
                && connection["path"] == "/v1/embeddings"
                && connection["join_awaited"] == true
                && connection["join_outcome"] != "panicked",
            "response lacks a matching actual connection join"
        );
        if let Some(digest) = response.get("request_sha256") {
            ensure!(
                connection["request_sha256"] == *digest,
                "connection/request bytes binding"
            );
        }

        expected_connections
            .entry(instance.to_owned())
            .or_default()
            .push(connection.clone());

        owner(
            inventory,
            instance,
            pid,
            if spawned {
                "client_request_task"
            } else {
                "inline_request_future"
            },
            case,
            if spawned {
                "completed-and-awaited"
            } else {
                "completed-inline"
            },
            source,
            json!({
                "case_id": case,
                "request_id": response["request_id"],
                "response_sha256": response["response_sha256"],
                "http_status": response["http_status"],
                "evidence_origin": if spawned {
                    "preceding_runner_join_next_result"
                } else {
                    "preceding_runner_direct_await_result"
                }
            }),
        );
    }
    Ok(())
}

fn local_inventory(
    inventory: &mut Vec<Value>,
    expected_connections: &mut BTreeMap<String, Vec<Value>>,
    probe: &Value,
    source: &Value,
    pump: bool,
    failure: bool,
    child: bool,
) -> Result<()> {
    let instance = text(probe, "instance")?;
    let pid = uint(probe, "pid")?;
    let closure = &probe["cleanup"];
    ensure!(
        closure.is_object()
            && closure["http_server_joined"] == true
            && closure["ready"] == false
            && closure["quarantine_cleared"] == false
            && closure["drop_counted_as_join"] == false,
        "local closure mismatch"
    );

    if failure {
        ensure!(
            child
                && closure["owner_thread_joined"] == false
                && closure["service_shutdown_returned_ok"] == false
                && closure["shutdown_error"] == "deadline_exceeded",
            "physical failure closure must remain unjoined/quarantined"
        );
    } else {
        ensure!(
            closure["owner_thread_joined"] == true
                && closure["service_shutdown_returned_ok"] == true,
            "missing individual Service owner join"
        );
    }

    owner(
        inventory,
        instance,
        pid,
        "service_owner_thread",
        "service-owner",
        if failure {
            "unjoined-contained-by-child-reap"
        } else {
            "joined"
        },
        source,
        closure.clone(),
    );
    owner(
        inventory,
        instance,
        pid,
        "http_server_task",
        "http-server",
        "completed-and-awaited",
        source,
        closure.clone(),
    );

    if pump {
        ensure!(
            closure["cleanup_clock_task_joined"] == true,
            "missing cleanup clock join"
        );
        owner(
            inventory,
            instance,
            pid,
            "cleanup_clock_task",
            "cleanup-clock",
            "completed-and-awaited",
            source,
            closure.clone(),
        );
    }
    if closure.get("client_tasks_drained").is_some() {
        ensure!(
            closure["client_tasks_drained"] == true,
            "request task set not drained"
        );
    }

    let responses = if child {
        response_values(&probe["reconciliation"]["responses"])?
    } else {
        response_values(&probe["responses"])?
    };
    collect_requests(
        inventory,
        expected_connections,
        instance,
        pid,
        &responses,
        true,
        source,
    )?;

    if let Some(rejections) = probe.get("rejections") {
        collect_requests(
            inventory,
            expected_connections,
            instance,
            pid,
            rejections.as_array().context("rejection array")?,
            false,
            source,
        )?;
    }
    if let Some(refusal) = probe["witness"]
        .get("admission_refusal")
        .filter(|value| !value.is_null())
    {
        collect_requests(
            inventory,
            expected_connections,
            instance,
            pid,
            std::slice::from_ref(refusal),
            false,
            source,
        )?;
    }
    Ok(())
}

fn check_connections(instance: &str, expected: &[Value], actual: &[Value]) -> Result<()> {
    let index = |values: &[Value]| -> Result<BTreeMap<u64, Value>> {
        let mut result = BTreeMap::new();
        for value in values {
            ensure!(
                value["instance"] == instance
                    && value["join_awaited"] == true
                    && matches!(
                        value["join_outcome"].as_str(),
                        Some("completed" | "cancelled-after-abort-and-awaited")
                    ),
                "unjoined or panicked connection"
            );
            let id = uint(value, "connection_id")?;
            ensure!(
                result.insert(id, value.clone()).is_none(),
                "duplicate connection owner"
            );
        }
        Ok(result)
    };
    ensure!(
        index(expected)? == index(actual)?,
        "connection inventory differs from preceding response evidence"
    );
    Ok(())
}

fn probe_names(case: &str) -> Result<&'static [&'static str]> {
    Ok(match case {
        "family-54-padded-token-capacity" => &["padded-below", "padded-at"],
        "family-54-attention-cell-capacity" => &["attention-below", "attention-at"],
        "family-54-fifo-no-split-no-bypass" => &["fifo"],
        "family-55-queue-saturation" => &["saturation"],
        "family-55-queued-expiry-zero-execution" => &["queued-expiry"],
        "family-55-batching-wait-bound" => &["wait-dominated", "deadline-dominated"],
        "family-56-invalid-last-isolation" => &["invalid-last"],
        "family-56-pad-and-oov-token-gates" => &["pad", "oov"],
        "family-57-quiesce-assembled-work" => &["quiesce"],
        _ => anyhow::bail!("unexpected local scheduler case"),
    })
}

fn validate_child_files(root: &Path, child: &Value) -> Result<()> {
    let files = child["child_output_sha256"]
        .as_object()
        .context("child file hashes")?;
    let actual: BTreeSet<String> = fs::read_dir(root)?
        .map(|entry| {
            let entry = entry?;
            ensure!(entry.file_type()?.is_file(), "non-file child evidence");
            entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("non-UTF8 child filename"))
        })
        .collect::<Result<_>>()?;
    ensure!(
        actual == files.keys().cloned().collect(),
        "child evidence file coverage changed"
    );
    for (name, digest) in files {
        ensure!(
            file_sha(&under(root, name)?)? == digest.as_str().context("child digest")?,
            "child evidence digest changed"
        );
    }
    ensure!(
        value(&root.join("evidence.json"))? == child["evidence"]
            && value(&root.join("child_identity.json"))? == child["identity"]
            && value(&root.join("owner_closure.json"))? == child["evidence"]["probe"]["cleanup"],
        "child-local evidence changed"
    );
    Ok(())
}

async fn close_owners(state: &SchedulerState) -> Result<Value> {
    unchanged(state).await?;
    let ledger_bytes = bounded(&state.core.out.join("accept_records.jsonl"), MANIFEST_LIMIT)?;
    let ledger = core::jsonl(&ledger_bytes)?;
    let scheduler_rows: Vec<&Value> = ledger
        .iter()
        .filter(|record| record["acceptance_phase"] == policy::SCHEDULER)
        .collect();
    let required = &state
        .core
        .loaded
        .required
        .phases
        .iter()
        .find(|(phase, _)| phase == policy::SCHEDULER)
        .context("scheduler required phase")?
        .1;
    ensure!(
        required.last().map(String::as_str) == Some(OWNERS)
            && scheduler_rows.len() + 1 == required.len()
            && scheduler_rows.len() == 14,
        "preceding scheduler case coverage"
    );

    let mut inventory = Vec::new();
    let mut expected_connections = BTreeMap::<String, Vec<Value>>::new();
    let mut child_connections = BTreeMap::<String, Vec<Value>>::new();
    let mut sources = Vec::new();

    for (position, (record, case)) in scheduler_rows.iter().zip(required).enumerate() {
        ensure!(
            record["case_id"] == *case && record["status"] == "executed" && record["pass"] == true,
            "preceding scheduler ledger order/status"
        );
        let path = match position {
            0 => "scheduler/family-53.evidence.json".to_owned(),
            1 => "scheduler/family-54-physical-b/evidence.json".to_owned(),
            _ => format!("scheduler/{case}/evidence.json"),
        };
        let digest = text(&record["details"], "evidence_sha256")?;
        ensure!(
            record["details"]["evidence_path"] == path,
            "unexpected case evidence path"
        );
        let resolved = under(&state.core.out, &path)?;
        ensure!(
            file_sha(&resolved)? == digest,
            "scheduler ledger/evidence digest mismatch"
        );
        let evidence = value(&resolved)?;
        ensure!(
            evidence["case_id"] == *case,
            "scheduler evidence case mismatch"
        );
        sources.push(source(&path, digest, ""));

        if position == 0 {
            ensure!(
                digest == state.coalescing_evidence_sha256
                    && evidence["instance"] == state.launch.instance
                    && evidence["pid"] == state.launch.pid,
                "retained coalescing binding"
            );
            for (wave, evidence) in array(&evidence, "waves")?.iter().enumerate() {
                let src = source(&path, digest, &format!("/waves/{wave}/responses"));
                collect_requests(
                    &mut inventory,
                    &mut expected_connections,
                    &state.launch.instance,
                    u64::from(std::process::id()),
                    array(evidence, "responses")?,
                    true,
                    &src,
                )?;
            }
        } else if position == 1 {
            local_inventory(
                &mut inventory,
                &mut expected_connections,
                &evidence,
                &source(&path, digest, ""),
                true,
                false,
                false,
            )?;
        } else if position <= 10 {
            let names = probe_names(case)?;
            let probes = array(&evidence, "probes")?;
            let bindings = array(&evidence, "probe_bindings")?;
            ensure!(
                names.len() == probes.len() && probes.len() == bindings.len(),
                "local probe inventory cardinality"
            );
            let parent = resolved.parent().context("case evidence parent")?;
            for (index, ((name, probe), binding)) in
                names.iter().zip(probes).zip(bindings).enumerate()
            {
                let relative = format!("{name}/evidence.json");
                ensure!(binding["path"] == relative, "probe path/order mismatch");
                let probe_path = under(parent, &relative)?;
                ensure!(
                    file_sha(&probe_path)? == text(binding, "sha256")?
                        && value(&probe_path)? == *probe,
                    "embedded/file probe digest binding"
                );
                local_inventory(
                    &mut inventory,
                    &mut expected_connections,
                    probe,
                    &source(&path, digest, &format!("/probes/{index}")),
                    true,
                    false,
                    false,
                )?;
            }
        } else {
            let child = &evidence["child"];
            let exit = &child["exit"];
            let identity = &child["identity"];
            let child_evidence = &child["evidence"];
            let probe = &child_evidence["probe"];
            let pid = uint(exit, "pid")?;
            let instance = text(probe, "instance")?;
            ensure!(
                exit["child_process_reaped"] == true
                    && exit["process_drop_counted_as_reap"] == false
                    && exit["parent_watchdog_fired"] == false
                    && exit["success"] == true
                    && exit["exit_code"] == 0
                    && exit["parent_pid"] == std::process::id()
                    && pid != u64::from(std::process::id())
                    && pid != u64::from(state.launch.pid)
                    && child_evidence["pid"] == pid
                    && probe["pid"] == pid
                    && identity["pid"] == pid
                    && identity["nonce"] == child_evidence["nonce"]
                    && child_evidence["case_id"] == *case
                    && child_evidence["child_checks_passed"] == true,
                "parent/child containment identity"
            );
            let parent = resolved.parent().context("child case parent")?;
            let child_dir = parent.join("child").canonicalize()?;
            ensure!(
                exit["evidence_directory"] == json!(child_dir)
                    && child_evidence["evidence_directory"] == json!(child_dir)
                    && value(&parent.join("child_exit.json"))? == *exit,
                "parent reap evidence path binding"
            );
            validate_child_files(&child_dir, child)?;
            let src = source(&path, digest, "/child/evidence/probe");
            local_inventory(
                &mut inventory,
                &mut expected_connections,
                probe,
                &src,
                false,
                case == "family-58-physical-failure-fanout",
                true,
            )?;
            ensure!(
                child_connections
                    .insert(
                        instance.to_owned(),
                        array(&probe["cleanup"], "http_connections")?.clone(),
                    )
                    .is_none(),
                "child instance reused"
            );

            // These entries deliberately do NOT say joined.
            for kind in ["child_watchdog_thread", "child_tokio_runtime_threads"] {
                owner(
                    &mut inventory,
                    instance,
                    pid,
                    kind,
                    kind,
                    "terminated-with-reaped-child-not-joined",
                    &source(&path, digest, "/child/exit"),
                    exit.clone(),
                );
            }
            owner(
                &mut inventory,
                instance,
                pid,
                "child_process",
                "owned-child",
                "reaped",
                &source(&path, digest, "/child/exit"),
                exit.clone(),
            );
        }
    }

    let mut connection_evidence = Vec::new();
    for (instance, expected) in &expected_connections {
        let actual = if let Some(records) = child_connections.remove(instance) {
            records
        } else {
            http::join_connections(instance).await?
        };
        check_connections(instance, expected, &actual)?;
        for record in &actual {
            owner(
                &mut inventory,
                instance,
                uint(record, "owner_pid")?,
                "http_connection_task",
                &uint(record, "connection_id")?.to_string(),
                text(record, "join_outcome")?,
                &json!({"evidence_origin": "preceding_digest_bound_response_headers"}),
                record.clone(),
            );
        }
        connection_evidence.extend(actual);
    }
    ensure!(
        child_connections.is_empty(),
        "unmatched child connection inventory"
    );

    let mut identities = BTreeSet::new();
    for entry in &inventory {
        let key = (
            uint(entry, "pid")?,
            text(entry, "instance")?.to_owned(),
            text(entry, "owner_kind")?.to_owned(),
            text(entry, "owner_identity")?.to_owned(),
        );
        ensure!(identities.insert(key), "duplicate owner inventory identity");
    }
    unchanged(state).await?;

    Ok(json!({
        "schema": "rustv1.accept_m4_scheduler_owner_inventory.v1",
        "case_id": OWNERS,
        "preceding_ledger_sha256": sha(&ledger_bytes),
        "preceding_scheduler_case_count": scheduler_rows.len(),
        "sources": sources,
        "owners": inventory,
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
                "instance": state.launch.instance,
                "pid": state.launch.pid,
                "disposition": "preserved-live-for-lifecycle-cases"
            }
        ],
        "child_local_closures_preserved": true,
        "parent_reap_is_distinct_containment_boundary": true,
        "summary_booleans_used_as_owner_coverage": false,
        "live_observation": state.observation,
        "live_manifest_sha256": sha(&state.manifest_prefix),
        "next_forward": state.next_forward,
        "request_ids": state.seen_request_ids
    }))
}

fn decode_hex(value: &str) -> Result<Vec<u8>> {
    ensure!(crate::contract::digest_valid(value), "invalid HMAC hex");
    (0..value.len())
        .step_by(2)
        .map(|offset| Ok(u8::from_str_radix(&value[offset..offset + 2], 16)?))
        .collect()
}

fn verify_transcript(bytes: &[u8], policy_value: &Value) -> Result<Vec<Value>> {
    let records = core::jsonl(bytes)?;
    let mut sent = 0u64;
    let mut received = 0u64;
    for record in &records {
        closed(record, &["direction", "envelope"])?;
        let (next, secret) = match text(record, "direction")? {
            "driver_to_orchestrator" => (&mut sent, "orch_secret_server_to_orchestrator"),
            "orchestrator_to_driver" => (&mut received, "orch_secret_orchestrator_to_server"),
            _ => anyhow::bail!("unknown transcript direction"),
        };
        let envelope = &record["envelope"];
        closed(envelope, &["kind", "seq", "fields", "mac"])?;
        ensure!(
            envelope["kind"].is_string()
                && envelope["fields"].is_object()
                && uint(envelope, "seq")? == *next,
            "transcript sequence/schema"
        );
        *next = next
            .checked_add(1)
            .context("transcript sequence overflow")?;
        let payload = json!({
            "kind": envelope["kind"],
            "seq": envelope["seq"],
            "fields": envelope["fields"]
        });
        let key = ring::hmac::Key::new(
            ring::hmac::HMAC_SHA256,
            &decode_hex(text(policy_value, secret)?)?,
        );
        ring::hmac::verify(
            &key,
            core::canonical_json(&payload)?.as_bytes(),
            &decode_hex(text(envelope, "mac")?)?,
        )
        .map_err(|_| anyhow::anyhow!("saved transcript MAC mismatch"))?;
    }
    Ok(records)
}

async fn consume_binding(state: &SchedulerState) -> Result<Value> {
    unchanged(state).await?;
    state.core.orchestration.sync()?;
    let path = state.directory.join("launch_binding.json");
    let bytes = bounded(&path, JSON_LIMIT)?;
    ensure!(
        sha(&bytes) == state.launch.binding_sha256,
        "replacement binding changed"
    );
    let binding: Value = serde_json::from_slice(&bytes)?;
    let request = &binding["authenticated_launch_request"];
    let accepted = &binding["authenticated_launch_accepted"];
    let ready = &binding["authenticated_ready"];

    ensure!(
        binding["schema"] == "rustv1.accept_m4_scheduler_launch_binding.v1"
            && binding["launch_authorization_after_independent_join"] == true
            && binding["process_creation_after_join_independently_measured"] == false
            && binding["launch_performed_by"] == "external_supervisor"
            && binding["policy_sha256"] == state.core.loaded.policy_sha256
            && request["kind"] == "orch_request_launch"
            && accepted["kind"] == "orch_launch_accepted"
            && ready["kind"] == "orch_ready",
        "saved replacement binding semantics"
    );
    closed(
        &request["fields"],
        &[
            "launch_request_id",
            "reason",
            "previous_instance",
            "previous_pid",
            "policy_sha256",
            "owner_closure_sha256",
            "old_absence_sha256",
        ],
    )?;
    closed(
        &accepted["fields"],
        &["launch_request_id", "request_seq", "previous_instance"],
    )?;
    closed(
        &ready["fields"],
        &[
            "launch_request_id",
            "instance",
            "pid",
            "start_utc",
            "manifest",
            "attestation",
        ],
    )?;

    let join_path = state.core.out.join("independent/owner_join.json");
    let join_sha = file_sha(&join_path)?;
    ensure!(
        value(&join_path)? == state.core.independent_join
            && state.core.independent_join["owner_thread_joined"] == true
            && state.core.independent_join["service_shutdown_returned_ok"] == true
            && state.core.independent_join["http_server_joined"] == true
            && binding["independent_join_sha256"] == join_sha,
        "saved independent join binding"
    );

    let request_id = text(&request["fields"], "launch_request_id")?;
    ensure!(
        label_valid(request_id)
            && request["fields"]["reason"] == "after_independent_join"
            && request["fields"]["previous_instance"] == state.core.initial_launch.instance
            && request["fields"]["previous_pid"] == state.core.initial_launch.pid
            && request["fields"]["policy_sha256"] == state.core.loaded.policy_sha256
            && request["fields"]["owner_closure_sha256"] == join_sha
            && request["fields"]["old_absence_sha256"]
                == sha(&serde_json::to_vec(
                    &binding["old_process_absence_rechecked"]
                )?)
            && binding["old_process_absence_rechecked"] == state.core.initial_exit["process_exit"]
            && binding["old_process_absence_rechecked"]["old_process_absent"] == true
            && accepted["fields"]["launch_request_id"] == request_id
            && accepted["fields"]["request_seq"] == request["seq"]
            && accepted["fields"]["previous_instance"] == state.core.initial_launch.instance
            && ready["fields"]["launch_request_id"] == request_id,
        "saved launch request/acknowledgement/absence binding"
    );
    ensure!(
        ready["fields"]["instance"] == state.launch.instance
            && ready["fields"]["pid"] == state.launch.pid
            && ready["fields"]["start_utc"] == state.launch.start_utc
            && Path::new(text(&ready["fields"], "manifest")?).canonicalize()?
                == state.launch.manifest
            && Path::new(text(&ready["fields"], "attestation")?).canonicalize()?
                == state.launch.attestation
            && binding["process_start_ticks"] == state.launch.process.start_ticks
            && binding["boot_id"] == state.launch.process.boot_id
            && binding["running_binary_sha256"] == state.launch.process.executable_sha256,
        "ready envelope/retained Launch identity mismatch"
    );

    ensure!(
        bounded(
            &state.directory.join("startup_executor_manifest.jsonl"),
            MANIFEST_LIMIT
        )? == state.launch.initial_manifest
            && binding["startup"] == state.launch.startup
            && file_sha(&state.launch.attestation)? == text(&binding, "attestation_sha256")?
            && file_sha(&state.directory.join("launch_attestation.json"))?
                == text(&binding, "attestation_sha256")?,
        "replacement startup/attestation bytes changed"
    );
    let mut initial = binding["authenticated_observation"].clone();
    ensure!(
        initial
            .as_object_mut()
            .context("attested observation")?
            .remove("attestation")
            .is_some()
            && initial == state.launch.initial_observation,
        "replacement initial observation binding"
    );

    let transcript_path = state.core.out.join("orchestration.jsonl");
    let transcript_bytes = bounded(&transcript_path, JSON_LIMIT)?;
    let transcript = verify_transcript(&transcript_bytes, &state.core.loaded.policy)?;
    let locate = |envelope: &Value, direction: &str| -> Result<usize> {
        let positions: Vec<usize> = transcript
            .iter()
            .enumerate()
            .filter(|(_, row)| row["direction"] == direction && row["envelope"] == *envelope)
            .map(|(position, _)| position)
            .collect();
        ensure!(
            positions.len() == 1,
            "missing/duplicate saved handshake envelope"
        );
        Ok(positions[0])
    };
    let request_position = locate(request, "driver_to_orchestrator")?;
    let accepted_position = locate(accepted, "orchestrator_to_driver")?;
    let ready_position = locate(ready, "orchestrator_to_driver")?;
    ensure!(
        request_position < accepted_position
            && accepted_position < ready_position
            && transcript
                .iter()
                .filter(|row| row["envelope"]["kind"] == "orch_request_launch")
                .count()
                == 1
            && transcript
                .iter()
                .filter(|row| row["envelope"]["kind"] == "orch_launch_accepted")
                .count()
                == 1,
        "replacement handshake order/cardinality"
    );
    ensure!(
        bounded(&transcript_path, JSON_LIMIT)? == transcript_bytes,
        "transcript changed during binding consumption"
    );
    unchanged(state).await?;

    Ok(json!({
        "schema": "rustv1.accept_m4_consumed_replacement_binding.v1",
        "case_id": BINDING,
        "launch_binding_path": "scheduler/launch_binding.json",
        "launch_binding_sha256": sha(&bytes),
        "independent_join_sha256": join_sha,
        "orchestration_transcript_sha256": sha(&transcript_bytes),
        "transcript_positions": {
            "request": request_position,
            "acknowledgement": accepted_position,
            "ready": ready_position
        },
        "request": request,
        "acknowledgement": accepted,
        "ready": ready,
        "instance": state.launch.instance,
        "pid": state.launch.pid,
        "launch_authorization_after_independent_join": true,
        "process_creation_after_join_independently_measured": false,
        "additional_launch_requested": false,
        "additional_supervisor_message_received": false
    }))
}

fn client(state: &SchedulerState) -> Result<Http> {
    Http::new(
        &state.core.base_url,
        &state.launch.instance,
        serde_json::from_value(state.core.loaded.policy["limits"].clone())?,
    )
}

fn trigger_body(state: &SchedulerState) -> Value {
    json!({
        "model": MODEL_ID,
        "input": [state.core.loaded.bundle.fixtures[0].text],
        "encoding_format": "float"
    })
}

fn production_error(reply: &http::Reply, code: &str) -> Result<()> {
    let (status, message, kind) = match code {
        "inference_failed" => (500, "Inference failed", "api_error"),
        "executor_poisoned" => (
            503,
            "Executor is retired; restart required",
            "service_unavailable_error",
        ),
        _ => anyhow::bail!("unplanned lifecycle error"),
    };
    ensure!(
        reply.status == status
            && serde_json::from_slice::<Value>(&reply.bytes)?
                == json!({
                    "error": {
                        "message": message,
                        "type": kind,
                        "param": null,
                        "code": code
                    }
                }),
        "production error status/envelope mismatch"
    );
    Ok(())
}

fn save_reply(out: &Path, name: &str, reply: &http::Reply) -> Result<Value> {
    let body = format!("{name}.response.bin");
    write_new(&out.join(&body), &reply.bytes)?;
    let evidence = json!({
        "case_id": name,
        "http_status": reply.status,
        "request_id": reply.request_id,
        "headers": reply.headers,
        "response_path": body,
        "response_sha256": sha(&reply.bytes)
    });
    save_json(&out.join(format!("{name}.response.json")), &evidence)?;
    Ok(evidence)
}

fn poisoned(before: &Value) -> Result<Value> {
    let mut expected = before.clone();
    expected["ready"] = json!(false);
    expected["lifecycle"] = json!("poisoned");
    expected["forward_calls"] = json!(uint(before, "forward_calls")?
        .checked_add(1)
        .context("logical counter overflow")?);
    Ok(expected)
}

async fn drift(state: &mut SchedulerState, out: &Path) -> Result<Value> {
    unchanged(state).await?;
    ready_observation(&state.observation, &state.launch)?;
    let before = state.observation.clone();
    let original_ids = state.seen_request_ids.clone();
    save_json(&out.join("observation_before.json"), &before)?;
    write_new(&out.join("manifest_before.jsonl"), &state.manifest_prefix)?;

    let quiesced = state.core.control.call("quiesce", None).await?;
    save_json(&out.join("quiesced.json"), &quiesced)?;
    let mut expected_quiesced = before.clone();
    expected_quiesced["ready"] = json!(false);
    ensure!(
        quiesced == expected_quiesced
            && state.core.control.call("observe", None).await? == quiesced
            && bounded(&state.launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
        "quiescence changed the preserved execution interval"
    );

    let armed = state.core.control.call("inject_policy_drift", None).await?;
    save_json(&out.join("armed.json"), &armed)?;
    ensure!(
        armed == before
            && bounded(&state.launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
        "drift arm changed execution/identity or failed to open one-request admission"
    );

    let body = trigger_body(state);
    let request = serde_json::to_vec(&body)?;
    write_new(&out.join("sched58-drift-trigger.request.bin"), &request)?;
    let reply = client(state)?
        .send("sched58-drift-trigger", &Wire::json(body)?)
        .await?;
    let response = save_reply(out, "sched58-drift-trigger", &reply)?;
    production_error(&reply, "inference_failed")?;
    let id = reply
        .request_id
        .context("admitted drift trigger missing request ID")?;
    ensure!(
        !original_ids.contains(&id),
        "drift trigger reused a live request ID"
    );

    let after = state.core.control.call("observe", None).await?;
    save_json(&out.join("poisoned.json"), &after)?;
    let manifest = bounded(&state.launch.manifest, MANIFEST_LIMIT)?;
    write_new(&out.join("manifest_after.jsonl"), &manifest)?;
    ensure!(
        after == poisoned(&before)? && manifest == state.manifest_prefix,
        "drift poison/logical count/zero physical-manifest delta mismatch"
    );
    state.launch.process.check_live()?;

    // The Service logical counter advances; math.check fails before executor
    // forward-ID allocation and physical_forward_begin. No fabricated pair.
    state.observation = after.clone();
    ensure!(
        state.seen_request_ids.insert(id),
        "drift request ID collision"
    );
    unchanged(state).await?;

    Ok(json!({
        "schema": "rustv1.accept_m4_live_policy_drift.v1",
        "case_id": DRIFT,
        "instance": state.launch.instance,
        "pid": state.launch.pid,
        "launch_binding_sha256": state.launch.binding_sha256,
        "observation_before": before,
        "quiesced": quiesced,
        "armed": armed,
        "response": response,
        "observation_after": after,
        "submitted_trigger_requests": 1,
        "fixture_index": 0,
        "fixture_token_sha256": state.core.loaded.bundle.fixtures[0].ids_sha256,
        "request_sha256": sha(&request),
        "logical_backend_request_delta": 1,
        "executor_manifest_interval_bytes": 0,
        "executor_manifest_sha256": sha(&manifest),
        "executor_next_forward_id_unchanged": state.next_forward,
        "physical_begin_records_added": 0,
        "physical_end_records_added": 0,
        "physical_forward_pair_invented": false,
        "failure_boundary": "IndependentExecutor::forward_batch_validated: math.check before forward ID",
        "request_ids_before": original_ids,
        "request_ids_after": state.seen_request_ids
    }))
}

async fn terminal(state: &SchedulerState, out: &Path) -> Result<Value> {
    unchanged(state).await?;
    ensure!(
        state.observation["lifecycle"] == "poisoned"
            && state.observation["worker_alive"] == true
            && state.observation["ready"] == false,
        "terminal probe requires the preserved poisoned replacement"
    );
    let client = client(state)?;

    let live = client.get_health("sched58-livez", "/health/livez").await?;
    let live_evidence = save_reply(out, "sched58-livez", &live)?;
    ensure!(
        live.status == 200 && live.bytes.is_empty() && live.request_id.is_none(),
        "real livez route did not return empty 200"
    );

    let ready = client
        .get_health("sched58-readyz", "/health/readyz")
        .await?;
    let ready_evidence = save_reply(out, "sched58-readyz", &ready)?;
    production_error(&ready, "executor_poisoned")?;
    ensure!(
        ready.request_id.is_none(),
        "readyz exposed embeddings request ID"
    );

    let body = trigger_body(state);
    write_new(
        &out.join("sched58-terminal-refusal.request.bin"),
        &serde_json::to_vec(&body)?,
    )?;
    let refusal = client
        .send("sched58-terminal-refusal", &Wire::json(body)?)
        .await?;
    let refusal_evidence = save_reply(out, "sched58-terminal-refusal", &refusal)?;
    production_error(&refusal, "executor_poisoned")?;
    ensure!(
        refusal.request_id.is_none(),
        "terminal admission refusal unexpectedly exposed a request ID"
    );
    unchanged(state).await?;

    Ok(json!({
        "schema": "rustv1.accept_m4_terminal_http_routes.v1",
        "case_id": TERMINAL,
        "instance": state.launch.instance,
        "pid": state.launch.pid,
        "livez": {
            "method": "GET", "path": "/health/livez", "response": live_evidence
        },
        "readyz": {
            "method": "GET", "path": "/health/readyz", "response": ready_evidence,
            "production_path": "readyz -> App::is_ready -> BackendHandle::ready"
        },
        "subsequent_embeddings_refusal": refusal_evidence,
        "observation_unchanged": state.observation,
        "manifest_unchanged_sha256": sha(&state.manifest_prefix),
        "executor_next_forward_id_unchanged": state.next_forward,
        "exposed_request_ids_unchanged": state.seen_request_ids,
        "harness_defined_readiness_predicate": false
    }))
}

async fn execute(mut state: SchedulerState) -> Result<SchedulerState> {
    for (phase, case) in [
        (policy::SCHEDULER, OWNERS),
        (policy::LIFECYCLE, BINDING),
        (policy::LIFECYCLE, DRIFT),
        (policy::LIFECYCLE, TERMINAL),
    ] {
        ensure!(
            state.core.ledger.pending().first()
                == Some(&json!({
                    "acceptance_phase": phase, "case_id": case
                })),
            "owner/lifecycle prefix called at wrong ledger position"
        );
        let out = state.directory.join(case);
        let work: Result<Value> = async {
            private_directory(&out)?;
            state.core.loaded.recheck()?;
            let mut evidence = match case {
                OWNERS => close_owners(&state).await?,
                BINDING => consume_binding(&state).await?,
                DRIFT => drift(&mut state, &out).await?,
                TERMINAL => terminal(&state, &out).await?,
                _ => unreachable!(),
            };
            state.core.loaded.recheck()?;
            state.core.orchestration.sync()?;
            evidence["policy_sha256"] = json!(state.core.loaded.policy_sha256);
            evidence["driver_binary_sha256"] =
                json!(state.core.loaded.artifacts["accept_m4"].sha256);
            let digest = save_json(&out.join("evidence.json"), &evidence)?;
            Ok(json!({
                "evidence_path": format!("scheduler/{case}/evidence.json"),
                "evidence_sha256": digest,
                "instance": state.launch.instance,
                "pid": state.launch.pid
            }))
        }
        .await;

        match work {
            Ok(details) => state.core.ledger.append(phase, case, true, details, true)?,
            Err(error) => {
                let _ = save_json(
                    &out.join("failure.json"),
                    &json!({
                        "error": format!("{error:#}")
                    }),
                );
                let _ = state.core.ledger.append(
                    phase,
                    case,
                    false,
                    json!({
                        "error": format!("{error:#}"),
                        "retained_evidence_path": format!("scheduler/{case}")
                    }),
                    true,
                );
                return Err(error);
            }
        }
    }
    ensure!(
        state.core.ledger.pending().first()
            == Some(&json!({
                "acceptance_phase": policy::LIFECYCLE, "case_id": NEXT
            })),
        "unexpected post-readiness boundary"
    );
    Ok(state)
}

pub(super) fn run(continuation: SchedulerContinuation) -> Result<SchedulerContinuation> {
    let SchedulerContinuation { runtime, state } = continuation;
    let retained = state.directory.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.block_on(execute(state))
    }));
    match result {
        Ok(result) => Ok(SchedulerContinuation {
            runtime,
            state: result?,
        }),
        Err(payload) => {
            let _ = save_json(
                &retained.join("owner_lifecycle_prefix_panic.json"),
                &json!({
                    "panic": true,
                    "retained_evidence_directory": retained,
                    "completion_claimed": false
                }),
            );
            eprintln!(
                "owner/lifecycle prefix panic; evidence retained at {}",
                retained.display()
            );
            std::panic::resume_unwind(payload)
        }
    }
}

#[cfg(test)]
#[path = "accept_m4_lifecycle_prefix_tests.rs"]
mod tests;
