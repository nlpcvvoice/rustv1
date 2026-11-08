//! M4 scheduler prefix: fresh supervised launch and live coalescing proof.
//!
//! This is a complete family-53 runner, not the complete scheduler phase.
//! It appends precisely the first required scheduler case.
//!
//! The driver sends the authenticated `orch_request_launch` authorization
//! only after validating the independent owner join and the old process's
//! absence. The supervisor must not launch a replacement before receiving
//! that authorization, then replies with the authenticated `orch_ready`
//! exchange. Receiving orch_ready after the join is NOT represented as an
//! independent measurement of the replacement's process-creation time.
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
    let launch_request_id = core::random_label()?;
    let launch_request = state
        .orchestration
        .send(
            "orch_request_launch",
            json!({
                "launch_request_id": launch_request_id,
                "reason": "after_independent_join",
                "previous_instance": state.initial_launch.instance,
                "previous_pid": state.initial_launch.pid,
                "policy_sha256": state.loaded.policy_sha256,
                "owner_closure_sha256": join_sha,
                "old_absence_sha256": sha(&serde_json::to_vec(&old_absence)?)
            }),
        )
        .await?;

    let launch_accepted = state.orchestration.receive().await?;
    ensure!(
        launch_accepted["kind"] == "orch_launch_accepted",
        "expected launch authorization acknowledgement"
    );
    closed(
        &launch_accepted["fields"],
        &["launch_request_id", "request_seq", "previous_instance"],
    )?;
    ensure!(
        launch_accepted["fields"]["launch_request_id"] == launch_request_id
            && launch_accepted["fields"]["request_seq"] == launch_request["seq"]
            && launch_accepted["fields"]["previous_instance"] == state.initial_launch.instance,
        "launch acknowledgement binding"
    );

    let ready = state
        .orchestration
        .receive_with_budget(std::time::Duration::from_secs(660))
        .await?;
    ensure!(ready["kind"] == "orch_ready", "replacement launch failed");
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
    ensure!(
        ready["fields"]["launch_request_id"] == launch_request_id,
        "replacement ready belongs to another launch request"
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
            "authenticated_launch_request": launch_request,
            "authenticated_launch_accepted": launch_accepted,
            "launch_authorization_after_independent_join": true,
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
/// Run the family-53 coalescing case. The driver sends `orch_request_launch`
/// after validating the independent join and old-process absence; the
/// supervisor must not launch a replacement before that authorization.
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
#[cfg(feature = "scheduler-test-seam")]
mod physical_b {
    use super::*;
    use crate::contract::{exact, ids_sha};
    use rustv1::{
        http_serve::{
            deterministic::SchedulerControl, validate_output, App, Backend, BoundedListener, Code,
            Config, ControlCommand, Lifecycle, Logger, Service, StartupCheck, TextTokenizer,
        },
        model::{BatchedExecution, ExecutionRow},
    };
    use std::{
        sync::{Arc, Mutex, MutexGuard},
        time::Duration,
    };
    use tokio::{
        net::TcpListener,
        sync::oneshot,
        time::{sleep, timeout},
    };

    const CASE: &str = "family-54-physical-b-capacity";
    const FOLLOWING: &str = "family-54-padded-token-capacity";
    const WATCHDOG: Duration = Duration::from_secs(30);

    fn service_result<T>(result: std::result::Result<T, Code>) -> Result<T> {
        result.map_err(|code| anyhow::anyhow!("controlled Service: {code:?}"))
    }

    fn guard<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn vector(token: u32) -> Vec<f32> {
        let mut row = vec![0.0; 2048];
        row[token as usize % 2048] = 1.0;
        row
    }

    struct Tokenizer;

    impl TextTokenizer for Tokenizer {
        fn encode(&self, text: &str) -> std::result::Result<Vec<u32>, Code> {
            let token = text
                .strip_prefix("token-")
                .ok_or(Code::InvalidRequest)?
                .parse::<u32>()
                .map_err(|_| Code::InvalidRequest)?;
            // The production validate_tokens gate remains authoritative.
            Ok(vec![token])
        }
    }

    #[derive(Default)]
    struct Recording {
        records: Vec<Value>,
        next_forward: u64,
    }

    struct ControlledBackend {
        recording: Arc<Mutex<Recording>>,
    }

    impl Backend for ControlledBackend {
        fn forward(&mut self, batches: &[Vec<u32>]) -> std::result::Result<Vec<Vec<f32>>, Code> {
            let output: Vec<Vec<f32>> = batches.iter().map(|ids| vector(ids[0])).collect();
            guard(&self.recording).records.push(json!({
                "kind": "controlled_startup",
                "backend": "synthetic_scheduler_probe",
                "tokens": batches,
                "output_sha256": output.iter()
                    .map(|row| sha(&row_bytes(row))).collect::<Vec<_>>()
            }));
            Ok(output)
        }

        fn forward_batch(
            &mut self,
            batches: &[Vec<u32>],
            associations: &[ExecutionRow],
        ) -> std::result::Result<Vec<Vec<f32>>, Code> {
            // Record what the production scheduler actually supplied.
            // Do not impose the expected batch layout inside the backend.
            let b = batches.len();
            let s = batches.iter().map(Vec::len).max().unwrap_or(0);
            let valid = batches.iter().map(Vec::len).sum::<usize>();
            let output: Vec<Vec<f32>> = batches.iter().map(|ids| vector(ids[0])).collect();

            let mut recording = guard(&self.recording);
            let id = recording.next_forward;
            recording.next_forward += 1;

            let members: Vec<Value> = batches
                .iter()
                .zip(associations)
                .enumerate()
                .map(|(physical_row, (ids, association))| {
                    json!({
                        "physical_row": physical_row,
                        "request_id": association.request_id,
                        "request_row": association.request_row,
                        "case_id": association.case_id,
                        "valid_length": ids.len(),
                        "token_sha256": ids_sha(ids)
                    })
                })
                .collect();

            let begin = json!({
                "kind": "controlled_backend_forward_begin",
                "backend": "synthetic_scheduler_probe",
                "forward_id": id,
                "B": b,
                "S": s,
                "projection_rows": b * s,
                "attention_cells_before_heads": b * s * s,
                "valid_tokens": valid,
                "padding_tokens": b * s - valid,
                "input_rows": batches.len(),
                "association_rows": associations.len(),
                "rows": members
            });
            recording.records.push(begin.clone());

            let mut end = begin;
            end["kind"] = json!("controlled_backend_forward_end");
            end["status"] = json!("ok");
            end["rows"] = json!(members
                .iter()
                .zip(&output)
                .map(|(member, row)| {
                    let mut member = member.clone();
                    member["output_sha256"] = json!(sha(&row_bytes(row)));
                    member
                })
                .collect::<Vec<_>>());
            recording.records.push(end);
            Ok(output)
        }

        fn inject_policy_drift(&mut self) -> std::result::Result<(), Code> {
            Err(Code::InvalidRequest)
        }

        fn restore_policy(&mut self) -> std::result::Result<(), Code> {
            Err(Code::InvalidRequest)
        }
    }

    #[derive(Clone)]
    struct Plan {
        case: String,
        tokens: Vec<u32>,
        body: Value,
    }

    fn plans() -> Vec<Plan> {
        let mut token = 100u32;
        [8usize, 8, 8, 8, 1]
            .into_iter()
            .enumerate()
            .map(|(position, count)| {
                let tokens: Vec<u32> = (token..token + count as u32).collect();
                token += count as u32;
                Plan {
                    case: format!("sched54-b-request-{position}"),
                    body: json!({
                        "model": MODEL_ID,
                        "input": tokens.iter()
                            .map(|token| format!("token-{token}"))
                            .collect::<Vec<_>>(),
                        "encoding_format": "float"
                    }),
                    tokens,
                }
            })
            .collect()
    }

    async fn find_event(
        control: &SchedulerControl,
        predicate: impl Fn(&Value) -> bool,
    ) -> Result<Value> {
        timeout(WATCHDOG, async {
            let mut index = 0;
            loop {
                let event = service_result(control.event(index).await)?;
                if predicate(&event) {
                    return Ok(event);
                }
                index += 1;
            }
        })
        .await
        .context("scheduler event watchdog; not an assembly-time witness")?
    }

    async fn checkpoint(control: &SchedulerControl, stage: &str, first: u64) -> Result<Value> {
        find_event(control, |event| {
            event["kind"] == "checkpoint"
                && event["stage"] == stage
                && event["request_ids"][0] == first
        })
        .await
    }

    fn release(control: &SchedulerControl, event: &Value) -> Result<()> {
        service_result(control.release(usize::try_from(uint(event, "seq")?)?))
    }

    async fn observe(app: &App) -> Result<Value> {
        Ok(serde_json::to_value(service_result(
            app.control(ControlCommand::Observe).await,
        )?)?)
    }

    fn decode(reply: &http::Reply, plan: &Plan) -> Result<Vec<Vec<f32>>> {
        ensure!(reply.status == 200, "controlled request failed");
        let body: Value = serde_json::from_slice(&reply.bytes)?;
        closed(&body, &["object", "data", "model", "usage"])?;
        ensure!(
            body["object"] == "list"
                && body["model"] == MODEL_ID
                && body["usage"]
                    == json!({
                        "prompt_tokens": plan.tokens.len(),
                        "total_tokens": plan.tokens.len()
                    }),
            "controlled response schema/usage"
        );
        let data = body["data"]
            .as_array()
            .context("controlled response rows")?;
        ensure!(data.len() == plan.tokens.len(), "response row count");
        let mut output = Vec::new();
        for (position, item) in data.iter().enumerate() {
            closed(item, &["object", "index", "embedding"])?;
            ensure!(
                item["object"] == "embedding" && item["index"] == position,
                "controlled output association"
            );
            output.push(serde_json::from_value::<Vec<f32>>(
                item["embedding"].clone(),
            )?);
        }
        service_result(validate_output(&output, plan.tokens.len()))?;
        let expected: Vec<Vec<f32>> = plan.tokens.iter().map(|token| vector(*token)).collect();
        exact(&output, &expected)?;
        Ok(output)
    }

    fn reconcile(
        records: &[Value],
        events: &[Value],
        insertion: &[Value],
        requests: &BTreeMap<u64, (Plan, Vec<Vec<f32>>)>,
    ) -> Result<Value> {
        ensure!(
            insertion.len() == 5 && requests.len() == 5,
            "request coverage"
        );
        ensure!(records.len() == 5, "unexpected controlled backend calls");
        ensure!(
            records[0]
                == json!({
                    "kind": "controlled_startup",
                    "backend": "synthetic_scheduler_probe",
                    "tokens": [[100]],
                    "output_sha256": [sha(&row_bytes(&vector(100)))]
                }),
            "controlled startup reconciliation"
        );

        let ids: Vec<u64> = insertion
            .iter()
            .map(|event| uint(event, "request_id"))
            .collect::<Result<_>>()?;
        ensure!(
            ids.iter().copied().collect::<BTreeSet<_>>().len() == 5,
            "duplicate acknowledged request ID"
        );
        let all_inserted: Vec<&Value> = events
            .iter()
            .filter(|event| event["kind"] == "inserted")
            .collect();
        ensure!(
            all_inserted.len() == 5 && all_inserted.iter().zip(insertion).all(|(a, b)| *a == b),
            "unmatched or reordered insertion events"
        );
        for (position, event) in insertion.iter().enumerate() {
            ensure!(
                event["inserted_us"] == 0
                    && event["deadline_us"] == 60_000_000
                    && event["S"] == 1
                    && event["B"] == if position == 4 { 1 } else { 8 },
                "insertion geometry/time"
            );
        }

        let launches: Vec<&Value> = events
            .iter()
            .filter(|event| event["kind"] == "launch")
            .collect();
        ensure!(launches.len() == 2, "unexpected launch decisions");
        let expected_members = [&ids[..4], &ids[4..]];
        let expected_b = [32usize, 1];

        let mut seen = BTreeSet::new();
        for forward in 0..2 {
            let members = expected_members[forward];
            let b = expected_b[forward];
            let launch = launches[forward];
            ensure!(
                launch["request_ids"] == json!(members)
                    && launch["B"] == b
                    && launch["S"] == 1
                    && launch["now_us"] == if forward == 0 { 0 } else { 1000 },
                "production launch differs from deterministic plan"
            );

            let begin = &records[1 + forward * 2];
            let end = &records[2 + forward * 2];
            ensure!(
                begin["kind"] == "controlled_backend_forward_begin"
                    && end["kind"] == "controlled_backend_forward_end"
                    && end["status"] == "ok",
                "controlled manifest pair"
            );
            let mut expected_rows = Vec::new();
            let mut completed_rows = Vec::new();
            for id in members {
                ensure!(seen.insert(*id), "request executed more than once");
                let (plan, output) = requests.get(id).context("unknown request")?;
                for (request_row, token) in plan.tokens.iter().enumerate() {
                    let row = json!({
                        "physical_row": expected_rows.len(),
                        "request_id": id,
                        "request_row": request_row,
                        "case_id": plan.case,
                        "valid_length": 1,
                        "token_sha256": ids_sha(&[*token])
                    });
                    let mut completed = row.clone();
                    completed["output_sha256"] = json!(sha(&row_bytes(&output[request_row])));
                    expected_rows.push(row);
                    completed_rows.push(completed);
                }
            }
            ensure!(expected_rows.len() == b, "physical row count");
            for record in [begin, end] {
                ensure!(
                    record["backend"] == "synthetic_scheduler_probe"
                        && record["forward_id"] == forward
                        && record["B"] == b
                        && record["S"] == 1
                        && record["projection_rows"] == b
                        && record["attention_cells_before_heads"] == b
                        && record["valid_tokens"] == b
                        && record["padding_tokens"] == 0
                        && record["input_rows"] == b
                        && record["association_rows"] == b,
                    "controlled manifest geometry/accounting"
                );
            }
            ensure!(
                begin["rows"] == json!(expected_rows) && end["rows"] == json!(completed_rows),
                "controlled manifest input/output association"
            );
        }
        ensure!(
            seen.len() == requests.len(),
            "unexecuted successful request"
        );

        let assemblies: Vec<&Value> = events
            .iter()
            .filter(|event| event["kind"] == "assembly")
            .collect();
        ensure!(assemblies.len() == 5, "assembly decision coverage");
        for (position, b) in [8usize, 16, 24, 32, 1].into_iter().enumerate() {
            let members = if position == 4 {
                &ids[4..]
            } else {
                &ids[..position + 1]
            };
            ensure!(
                assemblies[position]["B"] == b
                    && assemblies[position]["S"] == 1
                    && assemblies[position]["request_ids"] == json!(members)
                    && assemblies[position]["boundary_us"] == 1000,
                "production assembly progression/boundary"
            );
        }

        let policy = BatchedExecution::default();
        ensure!(
            policy.fits(32, 1)
                && !policy.fits(33, 1)
                && 33 < policy.max_padded_tokens
                && 33 < policy.max_attention_cells,
            "B capacity is not the isolated limiting dimension"
        );
        Ok(json!({
            "submitted_logical_requests": 5,
            "submitted_real_rows": 33,
            "physical_batches": [32, 1],
            "physical_widths": [1, 1],
            "first_batch_request_ids": ids[..4],
            "overflow_request_id": ids[4],
            "all_requests_intact": true,
            "dummy_rows": 0,
            "unmatched_backend_records": 0,
            "unmatched_insertions": 0,
            "numerical_model_evidence": false,
            "backend": "synthetic_scheduler_probe",
            "capacity_witness": "production_B_ceiling_with_33_acknowledged_real_rows"
        }))
    }

    async fn local(loaded: &Loaded, out: &Path) -> Result<Value> {
        private_directory(out)?;
        let control = SchedulerControl::new();
        let recording = Arc::new(Mutex::new(Recording::default()));
        let mut config = Config::production(64, 131072, 1024);
        config.batching = Some(loaded.bundle.shapes.batching.clone());
        config.scheduler_control = Some(control.clone());
        let instance = format!("sched54-{}", core::random_label()?);
        config.instance = Some(instance.clone());
        service_result(config.validate())?;

        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let listener = service_result(BoundedListener::new(listener, &config.transport))?;
        let backend_recording = recording.clone();
        let mut service = service_result(Service::spawn(
            config,
            Arc::new(Tokenizer),
            Logger::new(std::io::sink()),
            StartupCheck {
                tokens: vec![vec![100]],
                embeddings: vec![vector(100)],
            },
            move || {
                Ok(ControlledBackend {
                    recording: backend_recording,
                })
            },
        ))?;
        let app = service.app.clone();
        let router = app.router();
        let (stop, stopped) = oneshot::channel();
        let mut server = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
        });
        let mut tasks = JoinSet::new();

        // No ? outside this work block after creating the owners until cleanup.
        let work: Result<Value> = async {
            service_result(service.wait_ready().await)?;
            let before = observe(&app).await?;
            ensure!(
                before["instance"] == instance
                    && before["pid"] == std::process::id()
                    && before["lifecycle"] == "ready"
                    && before["ready"] == true
                    && before["worker_alive"] == true
                    && before["forward_calls"] == 0,
                "controlled initial observation"
            );

            let mut insertion = Vec::new();
            for plan in plans() {
                write_new(
                    &out.join(format!("{}.request.bin", plan.case)),
                    &serde_json::to_vec(&plan.body)?,
                )?;
                let limits: Limits = serde_json::from_value(loaded.policy["limits"].clone())?;
                let client = Http::new(&format!("http://{address}"), &instance, limits)?;
                let case = plan.case.clone();
                tasks.spawn(async move {
                    let wire = Wire::json(plan.body.clone())?;
                    let reply = client.send(&plan.case, &wire).await?;
                    Ok::<_, anyhow::Error>((plan, reply))
                });
                insertion.push(
                    find_event(&control, |event| {
                        event["kind"] == "inserted" && event["case_id"] == case
                    })
                    .await?,
                );
            }

            let ids: Vec<u64> = insertion
                .iter()
                .map(|event| uint(event, "request_id"))
                .collect::<Result<_>>()?;

            let first = checkpoint(&control, "assembly_start", ids[0]).await?;
            ensure!(
                guard(&recording).records.len() == 1,
                "backend ran before assembly release"
            );
            release(&control, &first)?;

            let assembled = checkpoint(&control, "assembled", ids[0]).await?;
            ensure!(
                assembled["request_ids"] == json!(&ids[..4])
                    && assembled["B"] == 32
                    && assembled["S"] == 1
                    && assembled["boundary_us"] == 1000
                    && assembled["now_us"] == 0,
                "B32 was not assembled at frozen time"
            );
            release(&control, &assembled)?;

            let second = checkpoint(&control, "assembly_start", ids[4]).await?;
            release(&control, &second)?;
            let second_seq = uint(&second, "seq")?;
            find_event(&control, |event| {
                event["kind"] == "assembly_receive"
                    && event["seq"].as_u64().is_some_and(|seq| seq > second_seq)
            })
            .await?;
            service_result(control.advance_to_us(1000))?;

            let tail = checkpoint(&control, "assembled", ids[4]).await?;
            ensure!(
                tail["request_ids"] == json!(&ids[4..])
                    && tail["B"] == 1
                    && tail["boundary_us"] == 1000
                    && tail["now_us"] == 1000,
                "overflow request assembly"
            );
            release(&control, &tail)?;

            let mut requests = BTreeMap::new();
            let mut responses = Vec::new();
            while let Some(joined) = tasks.join_next().await {
                let (plan, reply) = joined.context("controlled client task")??;
                write_new(
                    &out.join(format!("{}.response.bin", plan.case)),
                    &reply.bytes,
                )?;
                let id = reply.request_id.context("controlled response ID")?;
                let acknowledged = insertion
                    .iter()
                    .find(|event| event["case_id"] == plan.case)
                    .context("response without insertion")?;
                ensure!(acknowledged["request_id"] == id, "ack/HTTP ID mismatch");
                let output = decode(&reply, &plan)?;
                responses.push(json!({
                    "case_id": plan.case,
                    "request_id": id,
                    "http_status": reply.status,
                    "headers": reply.headers,
                    "response_sha256": sha(&reply.bytes)
                }));
                ensure!(
                    requests.insert(id, (plan, output)).is_none(),
                    "duplicate controlled response"
                );
            }

            let after = observe(&app).await?;
            logical_delta(&before, &after, 5)?;
            let events = service_result(control.snapshot())?;
            let records = guard(&recording).records.clone();
            let reconciliation = reconcile(&records, &events, &insertion, &requests)?;

            Ok(json!({
                "schema": "rustv1.accept_m4_controlled_b_capacity.v1",
                "case_id": CASE,
                "instance": instance,
                "pid": std::process::id(),
                "batching": loaded.bundle.shapes.batching,
                "max_sequence_tokens": 1024,
                "semantic_clock": "manual_microseconds",
                "watchdogs_are_capacity_witnesses": false,
                "observation_before": before,
                "observation_after": after,
                "responses": responses,
                "insertion_acknowledgements": insertion,
                "scheduler_events": events,
                "backend_records": records,
                "reconciliation": reconciliation
            }))
        }
        .await;

        // Cleanup is unconditional, including errors in event checks or I/O.
        // Admission is closed before allowing clock/progress to drain.
        app.begin_drain();
        control.open_for_cleanup();

        // A cleanup-only clock pump prevents already-admitted tokenization
        // from inserting a new job after the last manual clock advance.
        // Its timing is never used as capacity evidence.
        let cleanup_control = control.clone();
        let (pump_stop, mut pump_stopped) = oneshot::channel();
        let pump = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut pump_stopped => return Ok::<(), Code>(()),
                    _ = sleep(Duration::from_millis(1)) => {
                        cleanup_control.advance_by_us(1000)?;
                    }
                }
            }
        });

        tasks.abort_all();
        while tasks.join_next().await.is_some() {}

        let shutdown = service_result(service.shutdown().await);
        let _ = stop.send(());
        let transport = match timeout(WATCHDOG, &mut server).await {
            Ok(Ok(Ok(()))) => Ok(()),
            other => {
                server.abort();
                let _ = server.await;
                Err(anyhow::anyhow!("controlled HTTP cleanup failed: {other:?}"))
            }
        };
        let _ = pump_stop.send(());
        let pump_result = pump
            .await
            .context("cleanup clock task")
            .and_then(service_result);

        if let Err(error) = work {
            return Err(error).with_context(|| {
                format!(
                    "controlled case failed; owner_join_ok={}, http_join_ok={}, clock_join_ok={}",
                    shutdown.is_ok(),
                    transport.is_ok(),
                    pump_result.is_ok()
                )
            });
        }
        let mut evidence = work?;
        shutdown?;
        transport?;
        pump_result?;
        ensure!(
            app.lifecycle() == Lifecycle::Stopped && !app.is_ready(),
            "controlled owner did not stop after join"
        );

        let final_records = guard(&recording).records.clone();
        ensure!(
            evidence["backend_records"] == json!(final_records),
            "additional backend execution during cleanup"
        );
        let final_events = service_result(control.snapshot())?;
        ensure!(
            evidence["scheduler_events"] == json!(final_events),
            "additional scheduler activity during successful cleanup"
        );

        let mut manifest_bytes = Vec::new();
        for record in &final_records {
            serde_json::to_writer(&mut manifest_bytes, record)?;
            manifest_bytes.push(b'\n');
        }
        write_new(
            &out.join("controlled_backend_manifest.jsonl"),
            &manifest_bytes,
        )?;
        write_new(
            &out.join("scheduler_events.json"),
            &serde_json::to_vec_pretty(&final_events)?,
        )?;
        evidence["cleanup"] = json!({
            "service_shutdown_returned_ok": true,
            "owner_thread_joined": true,
            "http_server_joined": true,
            "cleanup_clock_task_joined": true,
            "lifecycle": "stopped",
            "ready": false,
            "quarantine_cleared": false,
            "drop_counted_as_join": false
        });
        evidence["backend_manifest_sha256"] = json!(sha(&manifest_bytes));
        evidence["scheduler_phase_complete"] = json!(false);
        evidence["m4_accepted"] = json!(false);
        Ok(evidence)
    }

    async fn execute(mut state: SchedulerState) -> Result<SchedulerState> {
        ensure!(
            state.core.ledger.pending().first()
                == Some(&json!({
                    "acceptance_phase": policy::SCHEDULER,
                    "case_id": CASE
                })),
            "physical-B runner called at wrong ledger position"
        );

        let result: Result<Value> = async {
            state.core.loaded.recheck()?;
            envelope(&state.core.loaded)?;
            state.launch.process.check_live()?;
            ensure!(
                state.core.control.call("observe", None).await? == state.observation
                    && bounded(&state.launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
                "unreconciled live activity before controlled case"
            );

            let out = state.directory.join("family-54-physical-b");
            let mut evidence = local(&state.core.loaded, &out).await?;

            // Do not discard or rebaseline the live family-53 cursor.
            state.launch.process.check_live()?;
            ensure!(
                state.core.control.call("observe", None).await? == state.observation
                    && bounded(&state.launch.manifest, MANIFEST_LIMIT)? == state.manifest_prefix,
                "live server changed during controlled case"
            );
            state.core.loaded.recheck()?;
            state.core.orchestration.sync()?;

            evidence["policy_sha256"] = json!(state.core.loaded.policy_sha256);
            evidence["driver_binary_sha256"] =
                json!(state.core.loaded.artifacts["accept_m4"].sha256);
            evidence["live_manifest_unchanged_sha256"] = json!(sha(&state.manifest_prefix));

            let evidence_sha = save_json(&out.join("evidence.json"), &evidence)?;
            Ok(json!({
                "evidence_path": "scheduler/family-54-physical-b/evidence.json",
                "evidence_sha256": evidence_sha,
                "backend": "synthetic_scheduler_probe",
                "production_scheduler_used": true,
                "physical_batches": [32, 1],
                "unmatched_backend_records": 0,
                "local_owner_joined": true,
                "local_http_server_joined": true,
                "numerical_model_evidence": false
            }))
        }
        .await;

        match result {
            Ok(details) => {
                state
                    .core
                    .ledger
                    .append(policy::SCHEDULER, CASE, true, details, true)?
            }
            Err(error) => {
                let _ = state.core.ledger.append(
                    policy::SCHEDULER,
                    CASE,
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
                    "case_id": FOLLOWING
                })),
            "unexpected post-capacity ledger position"
        );
        Ok(state)
    }

    pub(super) fn run(continuation: SchedulerContinuation) -> Result<SchedulerContinuation> {
        let SchedulerContinuation { runtime, state } = continuation;
        let state = runtime.block_on(execute(state))?;
        Ok(SchedulerContinuation { runtime, state })
    }

    pub(super) mod capacity {
        include!("accept_m4_capacity.rs");
    }

    pub(super) mod family55 {
        include!("accept_m4_family55.rs");
    }

    pub(super) mod family56_57 {
        include!("accept_m4_family56_57.rs");
    }

    pub(super) mod family57_58 {
        include!("accept_m4_family57_58.rs");
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn plan_crosses_only_the_b_ceiling() {
            let plans = plans();
            assert_eq!(
                plans
                    .iter()
                    .map(|plan| plan.tokens.len())
                    .collect::<Vec<_>>(),
                vec![8, 8, 8, 8, 1]
            );
            let tokens: BTreeSet<u32> = plans
                .iter()
                .flat_map(|plan| plan.tokens.iter().copied())
                .collect();
            assert_eq!(tokens.len(), 33);
            let policy = BatchedExecution::default();
            assert!(policy.fits(32, 1));
            assert!(!policy.fits(33, 1));
            for token in tokens {
                assert!(validate_output(&[vector(token)], 1).is_ok());
            }
        }
    }
}

/// Execute exactly family-54-physical-b-capacity.
///
/// No dispatcher completion arm and no global completion marker are added.
#[cfg(feature = "scheduler-test-seam")]
pub fn run_physical_b_capacity(
    continuation: SchedulerContinuation,
) -> Result<SchedulerContinuation> {
    physical_b::run(continuation)
}

/// Execute the remaining family-54 cases, in global ledger order.
///
/// Requires scheduler-test-seam. Uses the unchanged full qualification
/// envelope and the production Service scheduler.
///
/// Returns with family-55-queue-saturation pending. Does not finish the
/// scheduler phase or the overall acceptance ledger.
#[cfg(feature = "scheduler-test-seam")]
pub fn run_remaining_capacity(
    continuation: SchedulerContinuation,
) -> Result<SchedulerContinuation> {
    physical_b::capacity::run(continuation)
}

/// Execute the family-55 queue/expiry/wait cases, in global ledger order.
///
/// Requires scheduler-test-seam. Returns with
/// family-56-invalid-last-isolation pending; no completion markers.
#[cfg(feature = "scheduler-test-seam")]
pub fn run_queue_expiry_wait(continuation: SchedulerContinuation) -> Result<SchedulerContinuation> {
    physical_b::family55::run(continuation)
}

/// Execute the family-56 pair and family-57 assembled-work quiesce case.
///
/// Consumes the continuation returned by run_queue_expiry_wait.
/// Preserves its runtime and retained live-server reconciliation cursor.
///
/// Returns with family-57-drain-running-work pending.
#[cfg(feature = "scheduler-test-seam")]
pub fn run_invalid_tokens_quiesce(
    continuation: SchedulerContinuation,
) -> Result<SchedulerContinuation> {
    physical_b::family56_57::run(continuation)
}

/// Execute exactly the destructive family-57/58 trio.
///
/// Consumes the continuation returned by run_invalid_tokens_quiesce.
/// Destructive work executes only in reaped child processes.
///
/// Returns with scheduler-owners-joined pending.
#[cfg(feature = "scheduler-test-seam")]
pub fn run_drain_failure_expiry(
    continuation: SchedulerContinuation,
) -> Result<SchedulerContinuation> {
    physical_b::family57_58::run(continuation)
}

/// Private child-process protocol, not an acceptance-phase dispatcher.
///
/// This function terminates the child with a defined exit code.
#[cfg(feature = "scheduler-test-seam")]
pub fn scheduler_destructive_child(spec: &str) -> ! {
    physical_b::family57_58::child_entry(spec)
}

#[cfg(feature = "scheduler-test-seam")]
#[path = "accept_m4_lifecycle_prefix.rs"]
mod lifecycle_prefix;

/// Consume the continuation from run_drain_failure_expiry.
///
/// Executes scheduler-owners-joined and exactly the first three lifecycle
/// cases. Returns with family-58-restore-does-not-recover pending.
/// Keeps the same runtime and replacement Launch. Does not finish a ledger
/// or emit a phase/overall completion marker.
#[cfg(feature = "scheduler-test-seam")]
pub fn run_owners_binding_drift_readiness(
    continuation: SchedulerContinuation,
) -> Result<SchedulerContinuation> {
    lifecycle_prefix::run(continuation)
}

#[cfg(feature = "scheduler-test-seam")]
pub use lifecycle_prefix::restart_prefix::{PoisonExitContinuation, PoisonExitState};

/// Consume run_owners_binding_drift_readiness's continuation.
///
/// Executes exactly:
/// - family-58-restore-does-not-recover
/// - family-58-authenticated-supervised-restart
/// - family-58-old-process-exit
///
/// Returns with family-58-replacement-reattestation pending.
/// Replacement launch remains unauthorized until the next segment.
/// No phase or overall completion marker is emitted.
#[cfg(feature = "scheduler-test-seam")]
pub fn run_restore_restart_exit(
    continuation: SchedulerContinuation,
) -> Result<PoisonExitContinuation> {
    lifecycle_prefix::restart_prefix::run(continuation)
}

#[cfg(feature = "scheduler-test-seam")]
pub use lifecycle_prefix::restart_prefix::reattestation::{
    RecoveryReadyContinuation, RecoveryReadyState,
};

/// Execute exactly family-58-replacement-reattestation.
///
/// Consumes the poisoned-process exit continuation. Preserves the old
/// historical poisoned observation and reconciliation cursor unchanged.
///
/// Returns with family-58-exact-recovery pending. No completion markers.
#[cfg(feature = "scheduler-test-seam")]
pub fn run_replacement_reattestation(
    continuation: PoisonExitContinuation,
) -> Result<RecoveryReadyContinuation> {
    lifecycle_prefix::restart_prefix::reattestation::run(continuation)
}

#[cfg(feature = "scheduler-test-seam")]
pub use lifecycle_prefix::restart_prefix::reattestation::exact_recovery::{
    ExactRecoveryContinuation, ExactRecoveryState,
};

/// Execute exactly family-58-exact-recovery.
///
/// Consumes the reattested replacement's fresh startup baseline.
/// Keeps the exited scheduler's historical poisoned cursor unchanged.
///
/// Returns with final-authenticated-drain-and-exit pending.
/// No phase-completion or overall-completion marker is written.
#[cfg(feature = "scheduler-test-seam")]
pub fn run_exact_recovery(
    continuation: RecoveryReadyContinuation,
) -> Result<ExactRecoveryContinuation> {
    lifecycle_prefix::restart_prefix::reattestation::exact_recovery::run(continuation)
}
