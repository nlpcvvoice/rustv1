//! M4 scheduler prefix: fresh supervised launch and live coalescing proof.
//!
//! This is a complete family-53 runner, not the complete scheduler phase.
//! It appends precisely the first required scheduler case.
//!
//! The external supervisor must launch the replacement only after the
//! independent owner has joined. This module receives the existing
//! authenticated orch_ready exchange; it invents no launch/control command.
//! Receiving orch_ready after the join is NOT represented as an independent
//! measurement of the replacement's process-creation time.
//!
//! The existing HTTP client is reused. Concurrent HTTP submission is only
//! stimulus: coalescing is established exclusively from reconciled executor
//! rows plus the logical observation delta.
//!
//! Failed-to-coalesce waves are retained and fully reconciled. A bounded
//! sequence of such waves is not an executor retry or an OOM retry.
//!
//! No CUDA owner is created here. This module also compiles without CUDA.
//! No phase-completion or overall-acceptance manifest is written here.

use crate::{
    acceptance_core::{self as core, Continuation, CoreState, Launch, ProcessIdentity},
    acceptance_http::{self as http, Http, Limits, Wire},
    acceptance_policy::{self as policy, Loaded},
    contract::{bounded, compare, file_sha, label_valid, row_bytes, sha, value, write_new, Bundle},
};
use anyhow::{ensure, Context, Result};
use rustv1::{
    http_serve::MODEL_ID,
    model::{BATCHED_EXECUTION_POLICY, BATCHED_HTTP_CONTRACT},
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
};
use tokio::task::JoinSet;

const JSON_LIMIT: usize = 32 * 1024 * 1024;
const MANIFEST_LIMIT: usize = 256 * 1024 * 1024;

const FIRST_CASE: &str = "family-53-live-cross-request-coalescing";
const NEXT_CASE: &str = "family-54-physical-b-capacity";

const WAVE_REQUESTS: usize = 8;
const MAX_WAVES: usize = 8;

/// Runtime ownership remains outside the state, as in core::Continuation.
pub struct SchedulerContinuation {
    pub runtime: tokio::runtime::Runtime,
    pub state: SchedulerState,
}

/// Continuation cursor for the remaining scheduler and lifecycle work.
///
/// Contains credentials indirectly through CoreState. Do not derive Debug
/// or Serialize.
///
/// Every byte of manifest_prefix has been reconciled, including startup.
/// next_forward is the next expected physical forward ID.
/// seen_request_ids contains HTTP IDs from this replacement instance only.
pub struct SchedulerState {
    pub core: CoreState,
    pub launch: Launch,
    pub directory: PathBuf,
    pub manifest_prefix: Vec<u8>,
    pub observation: Value,
    pub next_forward: u64,
    pub seen_request_ids: BTreeSet<u64>,
    pub coalescing_evidence_sha256: String,
}

fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .with_context(|| format!("missing string: {key}"))
}

fn uint(v: &Value, key: &str) -> Result<u64> {
    v[key]
        .as_u64()
        .with_context(|| format!("missing unsigned integer: {key}"))
}

fn closed(v: &Value, keys: &[&str]) -> Result<()> {
    let object = v.as_object().context("expected object")?;
    ensure!(
        object.len() == keys.len() && keys.iter().all(|key| object.contains_key(*key)),
        "missing or unexpected fields"
    );
    Ok(())
}

fn private_directory(path: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(path)
        .context("scheduler output directory must be new")?;
    Ok(())
}

fn save_json(path: &Path, v: &Value) -> Result<String> {
    let bytes = serde_json::to_vec_pretty(v)?;
    write_new(path, &bytes)?;
    Ok(sha(&bytes))
}

fn envelope(loaded: &Loaded) -> Result<()> {
    let batching = &loaded.bundle.shapes.batching;
    ensure!(
        batching.max_batch_size == 32
            && batching.max_padded_tokens == 32_768
            && batching.max_attention_cells == 33_554_432
            && batching.max_wait_us == 1_000
            && loaded.bundle.engine.max_len == 1_024
            && loaded.bundle.engine.wait_us == 1_000,
        "scheduler qualification envelope changed"
    );
    batching.validate_engine(&loaded.bundle.engine)?;
    Ok(())
}

fn ready_observation(v: &Value, launch: &Launch) -> Result<u64> {
    closed(
        v,
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
        v["instance"] == launch.instance
            && v["pid"] == launch.pid
            && v["lifecycle"] == "ready"
            && v["worker_alive"] == true
            && v["ready"] == true
            && v["startup_output_sha256"] == launch.initial_observation["startup_output_sha256"],
        "live scheduler observation identity/readiness"
    );
    uint(v, "forward_calls")
}

fn logical_delta(before: &Value, after: &Value, requests: usize) -> Result<()> {
    let expected_calls = uint(before, "forward_calls")?
        .checked_add(u64::try_from(requests)?)
        .context("logical request counter overflow")?;
    let mut expected = before.clone();
    expected["forward_calls"] = json!(expected_calls);
    ensure!(
        *after == expected,
        "wave must change only forward_calls, by logical request count"
    );
    Ok(())
}

fn math(record: &Value, bundle: &Bundle) -> Result<()> {
    let m = &record["math"];
    ensure!(
        m["audit_identity"] == bundle.audit_identity
            && m["candle_core_version"] == "0.9.2"
            && m["reduced_precision"] == json!({"f32": false, "f16": false, "bf16": false}),
        "scheduler executor math binding"
    );
    Ok(())
}

fn new_evidence_path(raw: &str, state: &CoreState, handoff: &policy::Handoff) -> Result<PathBuf> {
    let path = Path::new(raw).canonicalize()?;
    ensure!(path.is_file(), "launch evidence must be a file");
    ensure!(
        path != state.loaded.policy_path
            && path != handoff.prepass_input_path
            && path != state.initial_launch.manifest
            && path != state.initial_launch.attestation
            && !state.loaded.paths.values().any(|p| p == &path)
            && !state.loaded.bundle.paths.values().any(|p| p == &path)
            && !path.starts_with(&state.out),
        "new launch evidence aliases existing/bound coordinator evidence"
    );
    Ok(path)
}

/// Bind the supervisor's already-launched fresh replacement.
///
/// The supervisor remains responsible for the no-overlap launch ordering.
/// We verify the preceding join evidence, old process absence, new process
/// identity, fresh authenticated attestation, and complete startup interval.
async fn bind_replacement(state: &mut CoreState, out: &Path) -> Result<Launch> {
    state.loaded.recheck()?;
    envelope(&state.loaded)?;

    ensure!(
        state.independent_join["owner_thread_joined"] == true
            && state.independent_join["service_shutdown_returned_ok"] == true
            && state.independent_join["http_server_joined"] == true
            && state.independent_join["lifecycle"] == "stopped"
            && state.independent_join["ready"] == false,
        "independent owner join prerequisite"
    );

    let join_path = state.out.join("independent/owner_join.json");
    ensure!(
        value(&join_path)? == state.independent_join,
        "independent join artifact changed"
    );
    let join_sha = file_sha(&join_path)?;

    let handoff = policy::Handoff::read(&state.out.join("http_handoff.json"))?;
    ensure!(
        handoff.instance == state.initial_launch.instance
            && handoff.pid == state.initial_launch.pid
            && handoff.model_root == state.loaded.bundle.model_dir
            && handoff.prepass_input_sha256 == state.loaded.bundle.input_sha256
            && file_sha(&handoff.prepass_input_path)? == state.loaded.bundle.input_sha256,
        "original handoff binding"
    );

    // Recheck absence, rather than interpreting a terminal observation as exit.
    let old_absence = state.initial_launch.process.prove_absent().await?;
    let ready = state.orchestration.receive().await?;
    ensure!(
        ready["kind"] == "orch_ready",
        "expected replacement orch_ready"
    );

    let fields = &ready["fields"];
    let instance = text(fields, "instance")?.to_owned();
    let pid = u32::try_from(uint(fields, "pid")?)?;
    let start_utc = text(fields, "start_utc")?.to_owned();
    ensure!(
        label_valid(&instance)
            && instance != state.initial_launch.instance
            && pid != state.initial_launch.pid
            && !start_utc.is_empty()
            && start_utc.len() <= 80,
        "replacement launch identity"
    );

    let manifest = new_evidence_path(text(fields, "manifest")?, state, &handoff)?;
    let attestation = new_evidence_path(text(fields, "attestation")?, state, &handoff)?;
    ensure!(manifest != attestation, "manifest/attestation alias");

    let process = ProcessIdentity::capture(pid, &state.loaded.artifacts["serve_m3"].sha256)?;
    let challenge = core::random_label()?;
    let attested = state.control.call("attest", Some(&challenge)).await?;

    closed(
        &attested,
        &[
            "instance",
            "pid",
            "lifecycle",
            "worker_alive",
            "ready",
            "forward_calls",
            "startup_output_sha256",
            "attestation",
        ],
    )?;
    ensure!(
        attested["instance"] == instance
            && attested["pid"] == pid
            && attested["lifecycle"] == "ready"
            && attested["worker_alive"] == true
            && attested["ready"] == true
            && attested["forward_calls"] == 0,
        "replacement is not a fresh ready instance"
    );

    let saved_bytes = bounded(&attestation, JSON_LIMIT)?;
    let saved: Value = serde_json::from_slice(&saved_bytes)?;
    ensure!(
        saved["schema"] == "rustv1.server_launch_attestation.v1"
            && saved["instance_id"] == instance
            && saved["pid"] == pid
            && saved["start_utc"] == start_utc
            && saved["settings_sha256"] == state.loaded.policy_sha256
            && saved["binary_sha256"] == state.loaded.artifacts["serve_m3"].sha256
            && saved["audit_sha256"] == state.loaded.artifacts["audit"].sha256
            && saved["startup_check_sha256"] == state.loaded.artifacts["startup_check"].sha256
            && saved["execution_policy"] == BATCHED_EXECUTION_POLICY
            && saved["http_contract_id"] == BATCHED_HTTP_CONTRACT
            && saved["batching"] == serde_json::to_value(&state.loaded.bundle.shapes.batching)?,
        "replacement saved attestation binding"
    );
    for key in [
        "rustc_version",
        "nvcc_version",
        "cuda_compute_cap",
        "target_features",
    ] {
        ensure!(
            !text(&saved["build_identity"], key)?.is_empty(),
            "empty replacement build identity"
        );
    }

    let runtime_attestation = &attested["attestation"];
    let mut reconstructed = runtime_attestation.clone();
    let object = reconstructed
        .as_object_mut()
        .context("runtime attestation object")?;
    for key in ["http_address", "control_address", "startup", "capabilities"] {
        ensure!(
            object.remove(key).is_some(),
            "missing runtime attestation field"
        );
    }
    ensure!(
        reconstructed == saved && serde_json::to_vec_pretty(&saved)? == saved_bytes,
        "replacement runtime/saved attestation mismatch"
    );

    let expected_http = core::base_address(&state.base_url)?;
    let actual_http: SocketAddr = text(runtime_attestation, "http_address")?.parse()?;
    let actual_control: SocketAddr = text(runtime_attestation, "control_address")?.parse()?;
    let expected_control: SocketAddr =
        text(&state.loaded.policy, "launch_control_address")?.parse()?;
    ensure!(
        actual_control == expected_control
            && actual_http.port() == expected_http.port()
            && (actual_http.ip() == expected_http.ip() || actual_http.ip().is_unspecified()),
        "replacement attested endpoint mismatch"
    );

    let initial_manifest = bounded(&manifest, MANIFEST_LIMIT)?;
    let startup = core::check_startup(&initial_manifest, &state.loaded, &sha(&saved_bytes), true)?;
    ensure!(
        runtime_attestation["startup"]
            == json!({
                "completed": true,
                "comparison": "exact_f32_bits",
                "output_f32le_sha256": startup["startup_output_sha256"]
            })
            && attested["startup_output_sha256"] == startup["startup_output_sha256"],
        "replacement startup output binding"
    );

    let initial_observation = state.control.call("observe", None).await?;
    let mut plain = attested.clone();
    plain
        .as_object_mut()
        .context("attested observation object")?
        .remove("attestation");
    ensure!(
        initial_observation == plain
            && bounded(&manifest, MANIFEST_LIMIT)? == initial_manifest
            && bounded(&attestation, JSON_LIMIT)? == saved_bytes,
        "replacement activity or evidence mutation during binding"
    );
    process.check_live()?;

    write_new(
        &out.join("startup_executor_manifest.jsonl"),
        &initial_manifest,
    )?;
    write_new(&out.join("launch_attestation.json"), &saved_bytes)?;
    let binding_sha256 = save_json(
        &out.join("launch_binding.json"),
        &json!({
            "schema": "rustv1.accept_m4_scheduler_launch_binding.v1",
            "authenticated_ready": ready,
            "challenge": challenge,
            "authenticated_observation": attested,
            "policy_sha256": state.loaded.policy_sha256,
            "attestation_sha256": sha(&saved_bytes),
            "startup": startup,
            "independent_join_sha256": join_sha,
            "old_process_absence_rechecked": old_absence,
            "process_start_ticks": process.start_ticks,
            "boot_id": process.boot_id,
            "running_binary_sha256": process.executable_sha256,
            "launch_performed_by": "external_supervisor",
            "process_creation_after_join_independently_measured": false
        }),
    )?;
    state.orchestration.sync()?;

    Ok(Launch {
        instance,
        pid,
        manifest,
        attestation,
        start_utc,
        process,
        initial_manifest,
        initial_observation,
        startup,
        binding_sha256,
    })
}

#[derive(Clone)]
struct RequestPlan {
    case: String,
    fixtures: Vec<usize>,
    bytes: Vec<u8>,
}

struct CompletedRequest {
    case: String,
    fixtures: Vec<usize>,
    output: Vec<Vec<f32>>,
}

struct Reconciled {
    next_forward: u64,
    forwards: Vec<Value>,
    witnesses: Vec<Value>,
}

/// Reconcile an entire wave, not just the forward selected as its witness.
///
/// Request IDs are not sorted to infer FIFO insertion order: HTTP allocation
/// precedes tokenization and is not a queue-insertion sequence number.
fn reconcile_wave(
    suffix: &[u8],
    first_forward: u64,
    requests: &BTreeMap<u64, CompletedRequest>,
    bundle: &Bundle,
) -> Result<Reconciled> {
    let records = core::jsonl(suffix)?;
    ensure!(
        !requests.is_empty() && records.len().is_multiple_of(2),
        "wave physical pair cardinality"
    );

    let mut remaining: BTreeSet<u64> = requests.keys().copied().collect();
    let mut forwards = Vec::new();
    let mut witnesses = Vec::new();
    let mut next_forward = first_forward;

    for pair in records.as_chunks::<2>().0 {
        let begin = &pair[0];
        let end = &pair[1];
        ensure!(
            begin["kind"] == "physical_forward_begin"
                && end["kind"] == "physical_forward_end"
                && end["status"] == "ok",
            "wave forward kind/status"
        );

        let starts = begin["rows"].as_array().context("begin rows")?;
        let ends = end["rows"].as_array().context("end rows")?;
        let b = starts.len();
        ensure!(
            (1..=32).contains(&b) && ends.len() == b,
            "physical row cardinality"
        );

        // Derive the expected physical layout from complete request groups.
        // A request may occur in one forward only, with all its rows intact.
        let mut members = Vec::new();
        let mut physical_fixtures = Vec::new();
        let mut physical_outputs: Vec<&Vec<f32>> = Vec::new();
        let mut physical_request_rows = Vec::new();
        let mut cursor = 0usize;

        while cursor < b {
            let request_id = uint(&starts[cursor], "request_id")?;
            let request = requests
                .get(&request_id)
                .context("manifest contains an unsubmitted request ID")?;
            ensure!(
                remaining.remove(&request_id),
                "request split, repeated, or executed in multiple forwards"
            );
            ensure!(
                !request.fixtures.is_empty()
                    && request.fixtures.len() <= 8
                    && request.output.len() == request.fixtures.len()
                    && request.fixtures.len() <= b - cursor,
                "manifest splits a logical HTTP request"
            );

            for (request_row, fixture_index) in request.fixtures.iter().enumerate() {
                let physical_row = cursor + request_row;
                let fixture = &bundle.fixtures[*fixture_index];
                let expected = json!({
                    "physical_row": physical_row,
                    "request_id": request_id,
                    "request_row": request_row,
                    "case_id": request.case,
                    "valid_length": fixture.ids.len(),
                    "token_sha256": fixture.ids_sha256
                });
                ensure!(
                    starts[physical_row] == expected,
                    "physical/request/input/token association"
                );
                let mut completed = expected;
                completed["output_sha256"] = json!(sha(&row_bytes(&request.output[request_row])));
                ensure!(
                    ends[physical_row] == completed,
                    "physical/request/output association"
                );
                physical_fixtures.push(*fixture_index);
                physical_outputs.push(&request.output[request_row]);
                physical_request_rows.push((request_id, request_row));
            }
            cursor += request.fixtures.len();
            members.push(request_id);
        }

        let s = physical_fixtures
            .iter()
            .map(|i| bundle.fixtures[*i].ids.len())
            .max()
            .context("empty physical layout")?;
        let valid: usize = physical_fixtures
            .iter()
            .map(|i| bundle.fixtures[*i].ids.len())
            .sum();

        ensure!(
            s <= 1024
                && bundle.shapes.batching.fits(b, s)
                && b * s <= 32_768
                && b * s * s <= 33_554_432,
            "physical forward exceeds qualification envelope"
        );
        for record in [begin, end] {
            ensure!(
                record["forward_id"] == next_forward
                    && record["execution_policy"] == BATCHED_EXECUTION_POLICY
                    && record["profile"] == "F"
                    && record["B"] == b
                    && record["S"] == s
                    && record["projection_rows"] == b * s
                    && record["attention_cells_before_heads"] == b * s * s
                    && record["valid_tokens"] == valid,
                "physical identity/geometry/accounting mismatch"
            );
            math(record, bundle)?;
        }
        ensure!(
            begin["padding_tokens"] == b * s - valid,
            "physical padding accounting"
        );

        let mut rows = Vec::new();
        for (position, fixture_index) in physical_fixtures.iter().enumerate() {
            let output = physical_outputs[position];
            let fixture = &bundle.fixtures[*fixture_index];
            let hash = sha(&row_bytes(output));
            let singleton = compare(output, &bundle.singleton[*fixture_index])?;
            let reference = compare(output, &bundle.reference[*fixture_index])?;
            let (candidate, selection): (&[f32], &str) = if b == 1 {
                ensure!(hash == fixture.golden_sha256, "physical B1 exact bits");
                (&bundle.singleton[*fixture_index], "frozen_singleton")
            } else if let Some(candidate) = bundle
                .candidates
                .get(&(physical_fixtures.clone(), position))
            {
                (candidate, "shape_specific")
            } else {
                (
                    bundle.canonical(*fixture_index, position % 2)?,
                    "canonical_same_fixture_position_modulo_two",
                )
            };
            rows.push(json!({
                "physical_row": position,
                "request_id": physical_request_rows[position].0,
                "request_row": physical_request_rows[position].1,
                "fixture_index": fixture_index,
                "fixture_id": fixture.id,
                "output_f32le_sha256": hash,
                "singleton_comparison": singleton,
                "independent_reference_comparison": reference,
                "candidate_comparison": compare(output, candidate)?,
                "candidate_selection": selection
            }));
        }

        let proof = json!({
            "forward_id": next_forward,
            "B": b,
            "S": s,
            "projection_rows": b * s,
            "attention_cells_before_heads": b * s * s,
            "valid_tokens": valid,
            "padding_tokens": b * s - valid,
            "distinct_request_ids": members,
            "logical_requests_in_physical_forward": members.len(),
            "rows": rows
        });
        if members.len() >= 2 {
            witnesses.push(proof.clone());
        }
        forwards.push(proof);
        next_forward = next_forward.checked_add(1).context("forward ID overflow")?;
    }

    ensure!(
        remaining.is_empty(),
        "successful HTTP requests absent from executor manifest"
    );
    Ok(Reconciled {
        next_forward,
        forwards,
        witnesses,
    })
}

fn wave_plan(bundle: &Bundle, wave: usize) -> Result<Vec<RequestPlan>> {
    // Short heterogeneous fixtures keep stimulus modest. Each HTTP request
    // remains a distinct logical request, even when its input repeats.
    let fixtures = [10usize, 11, 12, 10, 11, 12, 0, 1];
    ensure!(fixtures.len() == WAVE_REQUESTS, "internal wave cardinality");

    fixtures
        .into_iter()
        .enumerate()
        .map(|(position, fixture)| {
            let case = format!("sched53-wave-{wave}-request-{position}");
            ensure!(label_valid(&case), "wave case label");
            let body = json!({
                "model": MODEL_ID,
                "input": [bundle.fixtures[fixture].text],
                "encoding_format": "float"
            });
            Ok(RequestPlan {
                case,
                fixtures: vec![fixture],
                bytes: serde_json::to_vec(&body)?,
            })
        })
        .collect()
}

/// Execute one bounded wave and preserve all returned evidence.
///
/// Every task uses the unchanged HTTP client's connect/header/body deadlines.
/// JoinSet bounds concurrency and aborts outstanding tasks if this function
/// is cancelled. Cancellation is not treated as server-work cancellation.
async fn wave(state: &mut SchedulerState, index: usize) -> Result<Value> {
    state.launch.process.check_live()?;
    let before = state.core.control.call("observe", None).await?;
    ready_observation(&before, &state.launch)?;
    ensure!(
        before == state.observation
            && bounded(&state.launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
        "unaccounted activity before scheduler wave"
    );

    let plans = wave_plan(&state.core.loaded.bundle, index)?;
    let mut tasks = JoinSet::new();

    for plan in &plans {
        write_new(
            &state.directory.join(format!("{}.request.bin", plan.case)),
            &plan.bytes,
        )?;
        let limits: Limits = serde_json::from_value(state.core.loaded.policy["limits"].clone())?;
        let client = Http::new(&state.core.base_url, &state.launch.instance, limits)?;
        let plan = plan.clone();
        tasks.spawn(async move {
            let body: Value = serde_json::from_slice(&plan.bytes)?;
            let wire = Wire::json(body)?;
            let reply = client.send(&plan.case, &wire).await?;
            Ok::<_, anyhow::Error>((plan, reply))
        });
    }

    let mut requests = BTreeMap::new();
    let mut response_evidence = Vec::new();
    let mut first_error: Option<anyhow::Error> = None;

    // Do not stop collecting the other bounded requests upon one failure.
    while let Some(joined) = tasks.join_next().await {
        let result: Result<()> = async {
            let (plan, reply) = joined.context("scheduler HTTP task failed")??;
            write_new(
                &state.directory.join(format!("{}.response.bin", plan.case)),
                &reply.bytes,
            )?;
            let id = reply
                .request_id
                .context("successful wave response missing ID")?;
            ensure!(
                state.seen_request_ids.insert(id),
                "server request ID reused across scheduler waves"
            );
            let (output, usage) =
                http::decode_success(&reply, &state.core.loaded.bundle, &plan.fixtures, "float")?;
            response_evidence.push(json!({
                "case_id": plan.case,
                "request_id": id,
                "request_sha256": sha(&plan.bytes),
                "response_sha256": sha(&reply.bytes),
                "http_status": reply.status,
                "headers": reply.headers,
                "usage": usage
            }));
            ensure!(
                requests
                    .insert(
                        id,
                        CompletedRequest {
                            case: plan.case,
                            fixtures: plan.fixtures,
                            output,
                        }
                    )
                    .is_none(),
                "duplicate wave request ID"
            );
            Ok(())
        }
        .await;
        if let Err(error) = result {
            if first_error.is_none() {
                first_error = Some(error);
            }
        }
    }
    if let Some(error) = first_error {
        return Err(error);
    }
    ensure!(requests.len() == WAVE_REQUESTS, "incomplete HTTP wave");

    // No observe command is inserted while launching the wave: control
    // messages are scheduler batch boundaries. This observation follows all
    // complete HTTP responses and closes the execution interval.
    let after = state.core.control.call("observe", None).await?;
    ready_observation(&after, &state.launch)?;
    logical_delta(&before, &after, requests.len())?;

    let bytes = bounded(&state.launch.manifest, MANIFEST_LIMIT)?;
    ensure!(
        bytes.starts_with(&state.manifest_prefix),
        "scheduler manifest rewritten or truncated"
    );
    let start = state.manifest_prefix.len();
    let reconciled = reconcile_wave(
        &bytes[start..],
        state.next_forward,
        &requests,
        &state.core.loaded.bundle,
    )?;

    let interval_name = format!("wave-{index}.executor_interval.jsonl");
    write_new(&state.directory.join(&interval_name), &bytes[start..])?;
    let evidence = json!({
        "wave": index,
        "instance": state.launch.instance,
        "pid": state.launch.pid,
        "submitted_logical_requests": requests.len(),
        "logical_backend_request_delta": requests.len(),
        "observation_before": before,
        "observation_after": after,
        "responses": response_evidence,
        "manifest_interval": {
            "path": interval_name,
            "start_byte": start,
            "end_byte": bytes.len(),
            "prefix_before_sha256": sha(&state.manifest_prefix),
            "prefix_after_sha256": sha(&bytes),
            "interval_sha256": sha(&bytes[start..])
        },
        "physical_forwards": reconciled.forwards,
        "coalescing_witnesses": reconciled.witnesses,
        "unmatched_physical_records": 0,
        "request_id_order_used_as_fifo_insertion_proof": false
    });
    save_json(
        &state.directory.join(format!("wave-{index}.json")),
        &evidence,
    )?;

    state.manifest_prefix = bytes;
    state.observation = after;
    state.next_forward = reconciled.next_forward;
    Ok(evidence)
}

async fn execute(core_state: CoreState) -> Result<SchedulerState> {
    let mut core_state = core_state;
    let pending = core_state.ledger.pending();
    ensure!(
        pending.first()
            == Some(&json!({
                "acceptance_phase": policy::SCHEDULER,
                "case_id": FIRST_CASE
            })),
        "scheduler prefix called at wrong global ledger position"
    );

    let work: Result<SchedulerState> = async {
        let directory = core_state.out.join("scheduler");
        private_directory(&directory)?;
        let launch = bind_replacement(&mut core_state, &directory).await?;
        let next_forward = uint(&launch.startup, "next_forward_id")?;
        let manifest_prefix = launch.initial_manifest.clone();
        let observation = launch.initial_observation.clone();

        Ok(SchedulerState {
            core: core_state,
            launch,
            directory,
            manifest_prefix,
            observation,
            next_forward,
            seen_request_ids: BTreeSet::new(),
            coalescing_evidence_sha256: String::new(),
        })
    }
    .await;

    // Binding failures cannot be appended here after core_state has moved
    // into the construction future. They return failure, leaving the first
    // required scheduler case pending; no completion is claimed.
    let mut state = work?;

    let result: Result<Value> = async {
        let mut waves = Vec::new();
        let mut witnessed = false;
        for index in 0..MAX_WAVES {
            let evidence = wave(&mut state, index).await?;
            let witnesses = evidence["coalescing_witnesses"]
                .as_array()
                .context("wave witness array")?;
            witnessed |= !witnesses.is_empty();
            waves.push(evidence);
            if witnessed {
                break;
            }
        }
        ensure!(
            witnessed,
            "no reconciled cross-request physical forward within bounded waves"
        );

        state.core.loaded.recheck()?;
        state.launch.process.check_live()?;
        ensure!(
            state.core.control.call("observe", None).await? == state.observation
                && bounded(&state.launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
            "unreconciled activity while closing family 53"
        );
        write_new(
            &state.directory.join("family-53.executor_snapshot.jsonl"),
            &state.manifest_prefix,
        )?;

        let evidence = json!({
            "schema": "rustv1.accept_m4_live_coalescing.v1",
            "case_id": FIRST_CASE,
            "instance": state.launch.instance,
            "pid": state.launch.pid,
            "launch_binding_sha256": state.launch.binding_sha256,
            "policy_sha256": state.core.loaded.policy_sha256,
            "batching": state.core.loaded.bundle.shapes.batching,
            "maximum_sequence_length": 1024,
            "maximum_waves": MAX_WAVES,
            "maximum_concurrent_http_requests": WAVE_REQUESTS,
            "waves": waves,
            "final_observation": state.observation,
            "final_manifest_sha256": sha(&state.manifest_prefix),
            "next_forward_id": state.next_forward,
            "unmatched_physical_records": 0,
            "coalescing_proven_by": [
                "logical_request_observation_delta",
                "complete_executor_manifest_delta",
                "one_physical_forward_with_multiple_distinct_request_ids"
            ],
            "fifo_insertion_order_proven": false,
            "batching_wait_bound_measured": false,
            "destructive_restart_proven": false,
            "scheduler_phase_complete": false,
            "m4_accepted": false
        });
        state.coalescing_evidence_sha256 =
            save_json(&state.directory.join("family-53.evidence.json"), &evidence)?;
        state.core.orchestration.sync()?;
        Ok(json!({
            "evidence_path": "scheduler/family-53.evidence.json",
            "evidence_sha256": state.coalescing_evidence_sha256,
            "launch_binding_sha256": state.launch.binding_sha256,
            "instance": state.launch.instance,
            "pid": state.launch.pid,
            "unmatched_physical_records": 0
        }))
    }
    .await;

    match result {
        Ok(details) => {
            state
                .core
                .ledger
                .append(policy::SCHEDULER, FIRST_CASE, true, details, true)?;
        }
        Err(error) => {
            let _ = state.core.ledger.append(
                policy::SCHEDULER,
                FIRST_CASE,
                false,
                json!({"error": format!("{error:#}")}),
                true,
            );
            return Err(error);
        }
    }

    ensure!(
        state.core.ledger.pending().first()
            == Some(&json!({
                "acceptance_phase": policy::SCHEDULER,
                "case_id": NEXT_CASE
            })),
        "unexpected scheduler continuation ledger position"
    );
    Ok(state)
}

/// Run on the runtime returned by core::run_prefix.
///
/// The supervisor must arrange the next orch_ready for a new batched server,
/// launched only after the independent owner join. There is no automatic
/// control-socket launch, resume, shutdown, or restart command.
///
/// On error the run is incomplete. The external supervisor must clean up
/// its process; dropping this continuation is not process-exit proof.
pub fn run_coalescing(continuation: Continuation) -> Result<SchedulerContinuation> {
    let Continuation { runtime, state } = continuation;
    let state = runtime.block_on(execute(state))?;
    Ok(SchedulerContinuation { runtime, state })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_delta_counts_requests_not_physical_forwards() {
        let before = json!({
            "instance": "test",
            "forward_calls": 4,
            "ready": true
        });
        let after = json!({
            "instance": "test",
            "forward_calls": 12,
            "ready": true
        });
        assert!(logical_delta(&before, &after, 8).is_ok());
        assert!(logical_delta(&before, &after, 1).is_err());

        let mut changed = after.clone();
        changed["ready"] = json!(false);
        assert!(logical_delta(&before, &changed, 8).is_err());
    }

    #[test]
    fn logical_counter_overflow_is_rejected() {
        let before = json!({"forward_calls": u64::MAX});
        assert!(logical_delta(&before, &before, 1).is_err());
    }

    #[test]
    fn fixed_live_wave_is_bounded_below_admission_capacity() {
        assert_eq!(WAVE_REQUESTS, 8);
        assert_eq!(MAX_WAVES, 8);
        const { assert!(WAVE_REQUESTS < 64) };
    }

    #[test]
    fn closed_objects_reject_extra_fields() {
        assert!(closed(&json!({"a": 1}), &["a"]).is_ok());
        assert!(closed(&json!({"a": 1, "b": 2}), &["a"]).is_err());
    }
}
