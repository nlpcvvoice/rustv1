// Included inside acceptance_scheduler::physical_b::saved_local.
//
// Offline validation of saved local scheduler probes.
// No Service creation, control exchange, process observation, ledger append,
// phase completion, or overall acceptance occurs here.

use super::*;
use std::path::Component;

const RECONCILIATION: &str = "final-artifact-and-manifest-reconciliation";

const CASES: &[(&str, &[&str])] = &[
    ("family-54-padded-token-capacity", &["padded-below", "padded-at"]),
    (
        "family-54-attention-cell-capacity",
        &["attention-below", "attention-at"],
    ),
    ("family-54-fifo-no-split-no-bypass", &["fifo"]),
    ("family-55-queue-saturation", &["saturation"]),
    ("family-55-queued-expiry-zero-execution", &["queued-expiry"]),
    (
        "family-55-batching-wait-bound",
        &["wait-dominated", "deadline-dominated"],
    ),
    ("family-56-invalid-last-isolation", &["invalid-last"]),
    ("family-56-pad-and-oov-token-gates", &["pad", "oov"]),
    ("family-57-quiesce-assembled-work", &["quiesce"]),
];

fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .with_context(|| format!("missing saved-artifact string: {key}"))
}

fn values<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    v[key]
        .as_array()
        .with_context(|| format!("missing saved-artifact array: {key}"))
}

fn normal_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    ensure!(
        path.components().count() == 1
            && matches!(path.components().next(), Some(Component::Normal(_))),
        "unsafe local artifact filename"
    );
    Ok(())
}

fn record_bytes(records: &[Value]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for record in records {
        serde_json::to_writer(&mut bytes, record)?;
        bytes.push(b'\n');
    }
    Ok(bytes)
}

/// Exact directory coverage, rejecting symlinks and unexpected entry types.
fn directory_entries(root: &Path, directories: bool) -> Result<BTreeSet<String>> {
    ensure!(
        fs::symlink_metadata(root)?.file_type().is_dir(),
        "local evidence root is not a regular directory"
    );
    let mut names = BTreeSet::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        ensure!(
            kind.is_file() || (directories && kind.is_dir()),
            "symlink or unsupported local evidence entry"
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 local evidence filename"))?;
        normal_name(&name)?;
        ensure!(names.insert(name), "duplicate directory entry");
    }
    Ok(names)
}

struct Files<'a> {
    root: &'a Path,
    hashes: BTreeMap<String, String>,
}

impl<'a> Files<'a> {
    fn new(root: &'a Path) -> Result<Self> {
        directory_entries(root, false)?;
        Ok(Self {
            root,
            hashes: BTreeMap::new(),
        })
    }

    fn read(&mut self, name: &str, limit: usize) -> Result<Vec<u8>> {
        normal_name(name)?;
        ensure!(
            !self.hashes.contains_key(name),
            "artifact registered more than once"
        );
        let path = self.root.join(name);
        ensure!(
            fs::symlink_metadata(&path)?.file_type().is_file(),
            "saved artifact is not a regular non-symlink file"
        );
        let bytes = bounded(&path, limit)?;
        self.hashes.insert(name.to_owned(), sha(&bytes));
        Ok(bytes)
    }

    fn json(&mut self, name: &str, expected: &Value) -> Result<()> {
        let bytes = self.read(name, JSON_LIMIT)?;
        ensure!(
            serde_json::from_slice::<Value>(&bytes)? == *expected,
            "saved JSON differs from committed evidence: {name}"
        );
        Ok(())
    }

    fn bytes(&mut self, name: &str, expected: &[u8]) -> Result<()> {
        ensure!(
            self.read(name, MANIFEST_LIMIT)? == expected,
            "saved bytes differ from committed evidence: {name}"
        );
        Ok(())
    }

    fn digest(&mut self, name: &str, expected: &str) -> Result<()> {
        ensure!(
            crate::contract::digest_valid(expected),
            "invalid saved artifact commitment"
        );
        let bytes = self.read(name, MANIFEST_LIMIT)?;
        ensure!(
            sha(&bytes) == expected,
            "saved artifact digest mismatch: {name}"
        );
        Ok(())
    }

    fn finish(self) -> Result<BTreeMap<String, String>> {
        ensure!(
            directory_entries(self.root, false)?
                == self.hashes.keys().cloned().collect(),
            "missing or unaccounted local artifact"
        );
        for (name, digest) in &self.hashes {
            let path = self.root.join(name);
            ensure!(
                fs::symlink_metadata(&path)?.file_type().is_file()
                    && file_sha(&path)? == *digest,
                "local artifact changed during validation"
            );
        }
        Ok(self.hashes)
    }
}

fn prefix_through(events: &[Value], event: &Value) -> Result<Value> {
    let index = usize::try_from(uint(event, "seq")?)?;
    ensure!(
        events.get(index) == Some(event),
        "saved boundary event is absent from final scheduler events"
    );
    Ok(json!(&events[..=index]))
}

fn before_first_assembly(events: &[Value]) -> Result<Value> {
    let index = events
        .iter()
        .position(|event| event["kind"] == "assembly")
        .context("missing first assembly after held interval")?;
    Ok(json!(&events[..index]))
}

fn request_files(
    files: &mut Files<'_>,
    probe: &Value,
    responses: &[&Value],
) -> Result<()> {
    let instance = string(probe, "instance")?;
    let pid = uint(probe, "pid")?;
    let mut cases = BTreeSet::new();

    for response in responses {
        let case = string(response, "case_id")?;
        ensure!(
            label_valid(case) && cases.insert(case.to_owned()),
            "invalid or duplicate local response case"
        );
        let connection = &response["headers"]["_connection_owner"];
        ensure!(
            connection["instance"] == instance
                && connection["owner_pid"] == pid
                && connection["case_id"] == case
                && connection["method"] == "POST"
                && connection["path"] == "/v1/embeddings"
                && connection["join_awaited"] == true
                && matches!(
                    connection["join_outcome"].as_str(),
                    Some("completed" | "cancelled-after-abort-and-awaited")
                ),
            "saved local response connection identity/join"
        );

        // Physical-B omits a top-level request digest. Its digest-bound
        // connection record supplies the actual transmitted request digest.
        let request_digest = string(connection, "request_sha256")?;
        if let Some(digest) = response.get("request_sha256") {
            ensure!(
                digest == request_digest,
                "response/connection request commitments disagree"
            );
        }
        files.digest(&format!("{case}.request.bin"), request_digest)?;
        files.digest(
            &format!("{case}.response.bin"),
            string(response, "response_sha256")?,
        )?;
    }
    Ok(())
}

/// Validate every file produced by one successful local probe.
///
/// The caller must authenticate `probe` through its parent ledger commitment.
/// JSON sidecars without an original byte digest are compared structurally
/// with the digest-bound embedded value; they are not assigned invented
/// historical byte commitments.
pub(super) fn verify_probe(
    root: &Path,
    probe: &Value,
) -> Result<BTreeMap<String, String>> {
    let schema = string(probe, "schema")?;
    let physical_b = schema == "rustv1.accept_m4_controlled_b_capacity.v1";
    let capacity = schema == "rustv1.accept_m4_controlled_capacity_probe.v1";
    let family55 = schema == "rustv1.accept_m4_family55_probe.v1";
    let family5657 = schema == "rustv1.accept_m4_family56_57_probe.v1";
    ensure!(
        physical_b || capacity || family55 || family5657,
        "unsupported saved local probe schema"
    );

    let name = if physical_b {
        ensure!(probe["case_id"] == CASE, "physical-B case identity");
        "physical-b"
    } else {
        string(probe, "probe")?
    };
    ensure!(
        (physical_b && name == "physical-b")
            || (capacity
                && matches!(
                    name,
                    "padded-below" | "padded-at"
                        | "attention-below" | "attention-at" | "fifo"
                ))
            || (family55
                && matches!(
                    name,
                    "saturation" | "queued-expiry"
                        | "wait-dominated" | "deadline-dominated"
                ))
            || (family5657 && matches!(name, "invalid-last" | "pad" | "oov" | "quiesce")),
        "local probe name/schema mismatch"
    );

    let closure = &probe["cleanup"];
    ensure!(
        closure["service_shutdown_returned_ok"] == true
            && closure["owner_thread_joined"] == true
            && closure["http_server_joined"] == true
            && closure["cleanup_clock_task_joined"] == true
            && closure["lifecycle"] == "stopped"
            && closure["ready"] == false
            && closure["quarantine_cleared"] == false
            && closure["drop_counted_as_join"] == false,
        "successful local probe lacks its required closure"
    );

    let events = values(probe, "scheduler_events")?;
    for (index, event) in events.iter().enumerate() {
        ensure!(event["seq"] == index, "local scheduler event sequence gap");
    }
    let records = values(probe, "backend_records")?;
    ensure!(!records.is_empty(), "local startup record missing");
    let manifest = record_bytes(records)?;
    ensure!(
        probe["backend_manifest_sha256"] == sha(&manifest),
        "embedded local backend records/manifest commitment disagree"
    );

    let mut files = Files::new(root)?;
    files.json("evidence.json", probe)?;
    files.bytes("controlled_backend_manifest.jsonl", &manifest)?;
    files.json("scheduler_events.json", &json!(events))?;
    if let Some(digest) = probe.get("scheduler_events_sha256") {
        ensure!(
            digest.as_str() == Some(files.hashes["scheduler_events.json"].as_str()),
            "scheduler event byte commitment changed"
        );
    }

    let mut responses: Vec<&Value> = values(probe, "responses")?.iter().collect();
    if family5657 {
        responses.extend(values(probe, "rejections")?.iter());
    }
    if family55 {
        if let Some(refusal) = probe["witness"]
            .get("admission_refusal")
            .filter(|value| !value.is_null())
        {
            responses.push(refusal);
        }
    }
    request_files(&mut files, probe, &responses)?;

    if family55 || family5657 {
        ensure!(
            probe["all_local_owners_joined"] == true
                && closure["client_tasks_drained"] == true
                && closure["work_error"].is_null()
                && closure["shutdown_error"].is_null()
                && closure["transport_error"].is_null()
                && closure["clock_error"].is_null()
                && probe["backend_manifest_bytes"] == manifest.len(),
            "local diagnostic closure/manifest size"
        );
        let initial = record_bytes(&records[..1])?;
        ensure!(
            probe["manifest_before_sha256"] == sha(&initial),
            "startup-only manifest commitment"
        );
        files.bytes("manifest_before.jsonl", &initial)?;
        files.json("observation_before.json", &probe["observation_before"])?;
        files.json("observation_after.json", &probe["observation_after"])?;
        files.json("owner_closure.json", closure)?;
        files.json("backend_records.json", &json!(records))?;

        if family55 {
            match name {
                "saturation" | "queued-expiry" => {
                    files.json("held_events.json", &before_first_assembly(events)?)?;
                }
                "wait-dominated" | "deadline-dominated" => {
                    files.json(
                        "below_bound_events.json",
                        &prefix_through(events, &probe["witness"]["below_bound_receive"])?,
                    )?;
                }
                _ => unreachable!(),
            }
        } else {
            files.json("tokenizer_returns.json", &probe["tokenizer_returns"])?;
            files.json("responses.json", &probe["responses"])?;
            files.json("rejections.json", &probe["rejections"])?;

            if name != "quiesce" {
                files.json("rejection.json", &probe["witness"]["rejection"])?;
                let held = if name == "invalid-last" {
                    before_first_assembly(events)?
                } else {
                    ensure!(events.is_empty(), "token-only probe entered scheduler");
                    json!([])
                };
                files.json("rejection_events.json", &held)?;
                files.bytes("manifest_after_rejection.jsonl", &initial)?;
            }
        }
    }

    files.finish()
}

fn bound_evidence(
    out: &Path,
    records: &[Value],
    case: &str,
    relative: &str,
) -> Result<(Value, String)> {
    let matches: Vec<&Value> = records
        .iter()
        .filter(|record| {
            record["acceptance_phase"] == policy::SCHEDULER
                && record["case_id"] == case
        })
        .collect();
    ensure!(matches.len() == 1, "missing/duplicate local scheduler ledger row");
    let details = &matches[0]["details"];
    ensure!(
        details["evidence_path"] == relative,
        "local scheduler evidence path changed"
    );
    let digest = string(details, "evidence_sha256")?;
    ensure!(
        crate::contract::digest_valid(digest),
        "invalid local scheduler evidence digest"
    );
    let bytes = bounded(&out.join(relative), JSON_LIMIT)?;
    ensure!(sha(&bytes) == digest, "local scheduler evidence changed");
    let evidence: Value = serde_json::from_slice(&bytes)?;
    ensure!(evidence["case_id"] == case, "local scheduler evidence case");
    Ok((evidence, digest.to_owned()))
}

fn add_inventory(
    inventory: &mut BTreeMap<String, String>,
    prefix: &str,
    files: BTreeMap<String, String>,
) -> Result<()> {
    for (name, digest) in files {
        ensure!(
            inventory.insert(format!("{prefix}/{name}"), digest).is_none(),
            "duplicate local artifact inventory path"
        );
    }
    Ok(())
}

pub(crate) fn reconcile(state: &FinalDrainState) -> Result<Value> {
    let old = &state.recovery.ready.poison_exit.old;
    let loaded = &old.core.loaded;
    let out = &old.core.out;
    loaded.recheck()?;

    ensure!(
        old.core.ledger.pending() == vec![json!({
            "acceptance_phase": policy::LIFECYCLE,
            "case_id": RECONCILIATION
        })],
        "local reconciliation called at wrong ledger boundary"
    );
    let ledger = bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)?;
    ensure!(ledger == state.ledger_prefix, "final-drain ledger prefix changed");
    let records = core::jsonl(&ledger)?;
    let required: Vec<(&String, &String)> = loaded.required.phases
        .iter()
        .flat_map(|(phase, ids)| ids.iter().map(move |id| (phase, id)))
        .collect();
    ensure!(
        records.len() + 1 == required.len(),
        "preceding ledger coverage"
    );
    for (record, (phase, case)) in records.iter().zip(&required) {
        closed(
            record,
            &["acceptance_phase", "case_id", "status", "pass", "details"],
        )?;
        ensure!(
            record["acceptance_phase"] == phase.as_str()
                && record["case_id"] == case.as_str()
                && record["status"] == "executed"
                && record["pass"] == true
                && record["details"].is_object(),
            "preceding ledger order/status/schema"
        );
    }

    // Byte stability only at this segment. Full authenticated-transcript
    // semantic reconciliation remains required before the global case passes.
    old.core.orchestration.sync()?;
    ensure!(
        bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)?
            == state.orchestration_prefix,
        "final-drain transcript bytes changed"
    );

    let mut inventory = BTreeMap::new();
    let mut sources = Vec::new();

    let physical_root = "scheduler/family-54-physical-b";
    let physical_path = format!("{physical_root}/evidence.json");
    let (physical, digest) = bound_evidence(out, &records, CASE, &physical_path)?;
    ensure!(
        physical["policy_sha256"] == loaded.policy_sha256
            && physical["driver_binary_sha256"] == loaded.artifacts["accept_m4"].sha256,
        "physical-B policy/driver binding"
    );
    add_inventory(
        &mut inventory,
        physical_root,
        verify_probe(&out.join(physical_root), &physical)?,
    )?;
    sources.push(json!({
        "case_id": CASE,
        "evidence_path": physical_path,
        "evidence_sha256": digest
    }));

    for &(case, names) in CASES {
        let relative_root = format!("scheduler/{case}");
        let relative = format!("{relative_root}/evidence.json");
        let root = out.join(&relative_root);
        let (parent, digest) = bound_evidence(out, &records, case, &relative)?;
        ensure!(
            parent["policy_sha256"] == loaded.policy_sha256
                && parent["driver_binary_sha256"] == loaded.artifacts["accept_m4"].sha256,
            "local parent policy/driver binding"
        );
        let probes = values(&parent, "probes")?;
        let bindings = values(&parent, "probe_bindings")?;
        ensure!(
            probes.len() == names.len() && bindings.len() == names.len(),
            "local probe coverage"
        );
        let expected_entries: BTreeSet<String> = names.iter()
            .map(|name| (*name).to_owned())
            .chain(std::iter::once("evidence.json".to_owned()))
            .collect();
        ensure!(
            directory_entries(&root, true)? == expected_entries
                && fs::symlink_metadata(root.join("evidence.json"))?
                    .file_type().is_file(),
            "local parent directory coverage/type"
        );
        ensure!(
            inventory.insert(relative.clone(), digest.clone()).is_none(),
            "duplicate parent artifact"
        );

        for ((name, probe), binding) in names.iter().zip(probes).zip(bindings) {
            let probe_relative = format!("{name}/evidence.json");
            ensure!(
                probe["probe"] == *name
                    && binding["path"] == probe_relative,
                "local probe identity/order/path"
            );
            let probe_root = root.join(name);
            let probe_bytes = bounded(&probe_root.join("evidence.json"), JSON_LIMIT)?;
            ensure!(
                sha(&probe_bytes) == string(binding, "sha256")?
                    && serde_json::from_slice::<Value>(&probe_bytes)? == *probe,
                "embedded probe/file/digest binding"
            );
            add_inventory(
                &mut inventory,
                &format!("{relative_root}/{name}"),
                verify_probe(&probe_root, probe)?,
            )?;
        }
        sources.push(json!({
            "case_id": case,
            "evidence_path": relative,
            "evidence_sha256": digest
        }));
    }

    // Detect mutation during the traversal; do not establish new evidence
    // baselines for previously committed files.
    for (relative, digest) in &inventory {
        ensure!(
            file_sha(&out.join(relative))? == *digest,
            "local artifact changed during reconciliation"
        );
    }
    loaded.recheck()?;
    ensure!(
        bounded(&out.join("accept_records.jsonl"), MANIFEST_LIMIT)? == ledger
            && bounded(&out.join("orchestration.jsonl"), JSON_LIMIT)?
                == state.orchestration_prefix,
        "ledger/transcript changed during local reconciliation"
    );

    Ok(json!({
        "schema": "rustv1.accept_m4_local_artifact_reconciliation.v1",
        "boundary": "LOCAL-PROBE-ARTIFACTS-RECONCILED",
        "policy_sha256": loaded.policy_sha256,
        "preceding_ledger_sha256": sha(&ledger),
        "orchestration_bytes_sha256": sha(&state.orchestration_prefix),
        "sources": sources,
        "verified_output_sha256": inventory,
        "covered_scheduler_cases": 10,
        "covered_local_instances": 14,
        "post_exit_control_calls": 0,
        "ledger_rows_appended": 0,
        "final_artifact_reconciliation_complete": false,
        "scheduler_phase_complete": false,
        "lifecycle_phase_complete": false,
        "m4_accepted": false
    }))
}

#[cfg(test)]
pub(super) fn mutation_checks(root: &Path, probe: &Value) -> Result<()> {
    verify_probe(root, probe)?;

    // These are disposable test outputs, never a production ledger/history.
    let path = root.join("scheduler_events.json");
    let original = bounded(&path, JSON_LIMIT)?;
    fs::write(&path, b"[]")?;
    let changed = verify_probe(root, probe);
    fs::write(&path, &original)?;

    // Empty scheduler traces are valid for the pad/OOV probes, so use a
    // structurally different value rather than assuming [] changes them.
    if original == b"[]" {
        fs::write(&path, b"{}")?;
        let changed_empty = verify_probe(root, probe);
        fs::write(&path, &original)?;
        ensure!(changed_empty.is_err(), "changed empty trace was accepted");
    } else if serde_json::from_slice::<Value>(&original)? == json!([]) {
        fs::write(&path, b"{}")?;
        let changed_empty = verify_probe(root, probe);
        fs::write(&path, &original)?;
        ensure!(changed_empty.is_err(), "changed empty trace was accepted");
    } else {
        ensure!(changed.is_err(), "changed scheduler trace was accepted");
    }

    let hidden = root.join("scheduler_events.saved");
    fs::rename(&path, &hidden)?;
    let missing = verify_probe(root, probe);
    fs::rename(&hidden, &path)?;
    ensure!(missing.is_err(), "missing scheduler trace was accepted");

    let extra = root.join("unexpected.bin");
    write_new(&extra, b"unexpected")?;
    let unexpected = verify_probe(root, probe);
    fs::remove_file(extra)?;
    ensure!(unexpected.is_err(), "unaccounted artifact was accepted");

    verify_probe(root, probe)?;
    Ok(())
}
