//! Finalized M4 policy, numerical-evidence consumption and common ledger.
//!
//! This module performs no CUDA work and starts/stops no serving process.
//! It never treats declaration, artifact hashing or ledger registration as
//! execution of a required acceptance case.
//!
//! Call materialize only BEFORE launch. Call load for launch-bound policy.
//! All paths use the existing model/repo/policy root convention.
//!
//! Ledger use is single-writer. Phase runners validate their evidence before
//! append/import. Completion of the global ledger requires every registered
//! phase; completion of one phase is not overall M4 acceptance.

use crate::contract::{
    self, bounded, digest_valid, file_sha, ids_sha, label_valid, row_bytes, sha, unresolved, value,
    write_new, Artifact, Bundle,
};
use anyhow::{bail, ensure, Context, Result};
use rustv1::model::{BATCHED_EXECUTION_POLICY, BATCHED_HTTP_CONTRACT};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    net::SocketAddr,
    path::{Component, Path, PathBuf},
};

const JSON_LIMIT: usize = 32 * 1024 * 1024;
const RECORD_LIMIT: usize = 256 * 1024 * 1024;
const MODEL: &str = "nvidia/Nemotron-3-Embed-1B";

pub const PREPASS: &str = "controlled_numerical_prepass";
pub const POLICY: &str = "policy_loading";
pub const LAUNCH: &str = "initial_launch";
pub const HTTP: &str = "http_reconciliation";
pub const EXIT: &str = "batched_owner_exit";
pub const INDEPENDENT: &str = "independent_mapping_regression";
pub const JOIN: &str = "independent_owner_join";
pub const SCHEDULER: &str = "scheduler";
pub const LIFECYCLE: &str = "lifecycle";

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Handoff {
    pub schema: String,
    pub model_root: PathBuf,
    pub prepass_input_path: PathBuf,
    pub prepass_input_sha256: String,
    pub live_manifest_path: PathBuf,
    pub instance: String,
    pub pid: u32,
}

impl Handoff {
    pub fn read(path: &Path) -> Result<Self> {
        let handoff: Self = serde_json::from_slice(&bounded(path, JSON_LIMIT)?)?;
        ensure!(
            handoff.schema == "rustv1.accept_m4_handoff.v1"
                && handoff.pid > 0
                && label_valid(&handoff.instance)
                && digest_valid(&handoff.prepass_input_sha256)
                && handoff.model_root.is_absolute()
                && handoff.prepass_input_path.is_absolute()
                && handoff.live_manifest_path.is_absolute(),
            "invalid handoff"
        );
        Ok(handoff)
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mapping {
    pub m2_group_id: usize,
    pub adaptation: String,
    pub requests: Vec<Vec<usize>>,
}

pub struct Loaded {
    /// Exact finalized bytes, including secrets: never emit this field.
    pub bytes: Vec<u8>,
    /// Contains secrets: never serialize this whole value to an evidence log.
    pub policy: Value,
    pub policy_sha256: String,
    pub policy_path: PathBuf,
    pub paths: BTreeMap<String, PathBuf>,
    pub artifacts: BTreeMap<String, Artifact>,
    pub bundle: Bundle,
    pub mappings: Vec<Mapping>,
    pub prepass_records: Vec<Value>,
    pub required: Required,
    companion_hashes: BTreeMap<PathBuf, String>,
    input_path: PathBuf,
    executable_sha256: String,
}

fn closed(v: &Value, names: &[&str]) -> Result<()> {
    let o = v.as_object().context("expected object")?;
    ensure!(
        o.len() == names.len() && names.iter().all(|n| o.contains_key(*n)),
        "missing or unexpected object fields"
    );
    Ok(())
}

fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .with_context(|| format!("expected string: {key}"))
}

fn number(v: &Value, key: &str) -> Result<u64> {
    v[key]
        .as_u64()
        .with_context(|| format!("expected unsigned integer: {key}"))
}

fn checked_digest(v: &Value, key: &str) -> Result<String> {
    let s = text(v, key)?;
    ensure!(digest_valid(s), "invalid digest: {key}");
    Ok(s.to_owned())
}

fn path_under(root: &Path, relative: &str) -> Result<PathBuf> {
    let p = Path::new(relative);
    ensure!(
        !p.as_os_str().is_empty() && p.components().all(|c| matches!(c, Component::Normal(_))),
        "artifact path must be relative without traversal"
    );
    let root = root.canonicalize()?;
    let resolved = root.join(p).canonicalize()?;
    ensure!(
        resolved.starts_with(&root) && resolved.is_file(),
        "artifact escapes root or is not a file"
    );
    Ok(resolved)
}

/// Require precisely 0600, not merely absence of group/other permissions.
/// Metadata and bytes are obtained from the same opened file.
fn policy_bytes(path: &Path) -> Result<Vec<u8>> {
    #[cfg(not(unix))]
    {
        let _ = path;
        bail!("M4 policy permission validation requires Unix");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let named = fs::symlink_metadata(path)?;
        ensure!(
            named.file_type().is_file(),
            "policy must be a regular non-symlink file"
        );
        let mut f = File::open(path)?;
        let opened = f.metadata()?;
        ensure!(
            opened.is_file()
                && opened.permissions().mode() & 0o7777 == 0o600
                && named.dev() == opened.dev()
                && named.ino() == opened.ino(),
            "policy must be the checked regular 0600 file"
        );
        ensure!(opened.len() <= JSON_LIMIT as u64, "policy too large");
        let mut bytes = Vec::new();
        (&mut f)
            .take(JSON_LIMIT as u64 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= JSON_LIMIT, "policy grew beyond limit");
        let after = f.metadata()?;
        ensure!(
            after.len() == opened.len()
                && after.mtime() == opened.mtime()
                && after.mtime_nsec() == opened.mtime_nsec()
                && after.permissions().mode() & 0o7777 == 0o600,
            "policy changed while reading"
        );
        Ok(bytes)
    }
}

fn jsonl(bytes: &[u8]) -> Result<Vec<Value>> {
    ensure!(
        !bytes.is_empty() && bytes.ends_with(b"\n"),
        "empty/truncated JSONL"
    );
    bytes[..bytes.len() - 1]
        .split(|b| *b == b'\n')
        .map(|line| {
            ensure!(!line.is_empty(), "blank JSONL record");
            Ok(serde_json::from_slice(line)?)
        })
        .collect()
}

fn batching() -> Value {
    json!({
        "max_batch_size": 32,
        "max_padded_tokens": 32768,
        "max_attention_cells": 33554432,
        "max_wait_us": 1000
    })
}

fn materialization() -> Value {
    json!({
        "fixtures": {"artifact_role": "fixtures_exec", "count": 22},
        "mappings": {"artifact_role": "mappings_v1", "count": 46}
    })
}

fn scheduler_contract() -> Value {
    json!({
        "kind": "fifo_opportunistic_static",
        "single_owner": true,
        "overlapping_forwards": false,
        "max_http_inputs": 8,
        "split_http_request": false,
        "head_bypass": false,
        "dummy_batch_rows": false,
        "sequence_width_rounding": false,
        "wait_origin": "queue_insertion",
        "assembly_deadline": "minimum_of_oldest_batch_wait_and_member_request_deadlines",
        "control_messages_are_batch_boundaries": true,
        "queue_capacity_unit": "logical_requests",
        "scheduler_local_jobs_retain_admission": true,
        "expired_queued_requests_execute": false,
        "running_deadline_expiry": "retire_executor",
        "physical_failure": "fail_all_members_and_poison",
        "oom_retry": false,
        "chunked_prefill": false
    })
}

fn padding_contract() -> Value {
    json!({
        "side": "right",
        "pad_id": 11,
        "user_pad_ids": "reject",
        "physical_b": "real_sequence_count",
        "physical_s": "maximum_valid_length",
        "attention": "bidirectional",
        "mask_dtype": "F32",
        "valid_key_mask": 0,
        "invalid_key_mask": "negative_infinity",
        "mask_applies_to": "all_queries_in_every_layer",
        "padded_queries": "attend_valid_keys_then_discard",
        "position_ids": "zero_based_per_sequence",
        "pooling": "sum_valid_states_divide_by_valid_length",
        "normalization": "per_sequence_l2"
    })
}

fn precision_contract() -> Value {
    json!({
        "weights_and_projections": "BF16",
        "profile": "F",
        "gemm_reduced_precision_f32": false,
        "gemm_reduced_precision_f16": false,
        "gemm_reduced_precision_bf16": false,
        "physical_b1_comparison": "exact_f32_bits",
        "physical_b_gt_1_comparison": {
            "cosine_strictly_greater_than": 0.999,
            "max_abs_at_most": 0.01,
            "compare_directly_against": [
                "batched_candidate", "independent_reference", "frozen_singleton_candidate"
            ]
        },
        "all_outputs": {
            "width": 2048,
            "finite": true,
            "norm_computation": "FP64",
            "l2_norm_abs_error_at_most": 0.00001
        },
        "controlled_identical_batch_repeats": "exact_f32_bits",
        "arbitrary_batch_shape_bit_invariance": false,
        "batch_invariance": "direct_pairwise_numerical_gates"
    })
}

fn acceptance_contract() -> Value {
    let names = [
        "all_fixtures_b2_both_positions",
        "homogeneous_batch_size_sweep",
        "mixed_length_boundaries",
        "permutation_duplicate_association",
        "fixed_layout_repeats",
        "padding_width_invariance",
        "cross_request_coalescing",
        "capacity_boundaries",
        "queue_expiry_wait_bounds",
        "invalid_request_isolation",
        "quiesce_and_drain",
        "poison_and_restart",
    ];
    json!({
        "driver_role": "accept_m4",
        "legacy_mapping_groups": 46,
        "fixture_count": 22,
        "retain_fixture_indices": true,
        "retain_mapping_requests": true,
        "recovery_groups": [42, 43, 44, 45, 46],
        "recovery_placement": "immediately_after_corresponding_rejection",
        "batch_case_artifact_role": "batch_cases",
        "batch_families": names.iter().enumerate()
            .map(|(i, name)| json!({"id": 47 + i, "name": name})).collect::<Vec<_>>(),
        "physical_batch_proof": "executor_manifest",
        "http_concurrency_alone_is_batch_proof": false,
        "canonical_batched_golden_rows": 44,
        "canonical_batched_construction": "for_each_Fi_export_target_from_Fi_F06_and_F06_Fi",
        "shape_specific_golden_key": [
            "execution_policy", "ordered_token_digests", "valid_lengths",
            "B", "S", "physical_row"
        ],
        "runtime_placeholder_rejection": "recursive_exact_or_embedded_RUNTIME",
        "pending_required_cases_must_be_empty": true,
        "exceptions": []
    })
}

fn deployment_contract() -> Value {
    json!({
        "gpu": "NVIDIA B200",
        "route": "sm_100 native",
        "engine": "candle 0.9.2",
        "cudarc": "0.19.10",
        "rust_toolchain": "1.99.1",
        "cuda_toolkit": "12.8.1",
        "profile": "F",
        "execution_policy": BATCHED_EXECUTION_POLICY,
        "batching": batching(),
        "scheduler": scheduler_contract(),
        "padding": padding_contract(),
        "precision": precision_contract(),
        "startup": {
            "route": "forward_serving",
            "physical_b": 1,
            "physical_s": "valid_length",
            "comparison": "exact_f32_bits",
            "existing_startup_artifact_format": true,
            "launch_attestation_binding_required": true
        },
        "audit_requirements": {
            "active_artifact_role": "audit",
            "base_artifact_role": "audit_b1_base",
            "active_execution_policy": BATCHED_EXECUTION_POLICY,
            "maximum_physical_b": 32,
            "maximum_physical_s": 1024,
            "source_region_renewal_required": true,
            "batch_dependent_layout_review_required": true,
            "b1_only_audit_sufficient": false,
            "audit_identity_must_match_artifact": true,
            "audit_lockfile_binding_must_match": true,
            "audit_feature_tree_binding_must_match": true
        },
        "acceptance": acceptance_contract(),
        "materialization": materialization()
    })
}

fn check_policy_header(p: &Value) -> Result<()> {
    closed(
        p,
        &[
            "schema",
            "contract_id",
            "http_contract_id",
            "run_id",
            "model",
            "audit_identity",
            "deployment",
            "artifacts",
            "fixtures",
            "mappings",
            "limits",
            "launch_control_address",
            "orch_secret_server_to_orchestrator",
            "orch_secret_orchestrator_to_server",
            "over_length_text",
            "pad_text",
            "samples",
        ],
    )?;
    // This rejects embedded RUNTIME as well as the earlier placeholder forms.
    ensure!(!unresolved(p), "unresolved policy");
    ensure!(
        p["schema"] == "rustv1.acceptance_policy.v3"
            && p["contract_id"] == "b200-independent-F-v2"
            && p["http_contract_id"] == BATCHED_HTTP_CONTRACT
            && p["model"] == MODEL
            && p["samples"] == json!([])
            && p["deployment"] == deployment_contract()
            && label_valid(text(p, "run_id")?),
        "policy schema/contract/deployment"
    );
    let a = text(p, "orch_secret_server_to_orchestrator")?;
    let b = text(p, "orch_secret_orchestrator_to_server")?;
    ensure!(
        digest_valid(a) && digest_valid(b) && a != b,
        "directional orchestration keys must be distinct 32-byte lowercase hex"
    );
    let address: SocketAddr = text(p, "launch_control_address")?.parse()?;
    ensure!(
        address.ip().is_loopback() && address.port() != 0,
        "invalid launch control endpoint"
    );
    let l = &p["limits"];
    closed(
        l,
        &[
            "connect_ms",
            "headers_ms",
            "body_ms",
            "control_ms",
            "observation_ms",
            "server_request_ms",
            "response_bytes",
            "admission_capacity",
            "queue_capacity",
        ],
    )?;
    for key in [
        "connect_ms",
        "headers_ms",
        "body_ms",
        "control_ms",
        "observation_ms",
    ] {
        ensure!((1..=300_000).contains(&number(l, key)?), "invalid deadline");
    }
    ensure!(
        l["server_request_ms"] == 60_000
            && l["response_bytes"] == 1_048_576
            && l["admission_capacity"] == 64
            && l["queue_capacity"] == 64,
        "server envelope mismatch"
    );
    ensure!(
        p["fixtures"].is_array()
            && p["mappings"].is_array()
            && !text(p, "over_length_text")?.trim().is_empty()
            && (p["pad_text"].is_null() || p["pad_text"].is_string()),
        "invalid materialized arrays/probe fields"
    );
    Ok(())
}

fn resolve(
    p: &Value,
    model: &Path,
    policy_path: &Path,
) -> Result<(BTreeMap<String, Artifact>, BTreeMap<String, PathBuf>)> {
    let artifacts: BTreeMap<String, Artifact> = serde_json::from_value(p["artifacts"].clone())?;
    let repo = std::env::current_dir()?.canonicalize()?;
    let parent = policy_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut paths = BTreeMap::new();
    for (role, artifact) in &artifacts {
        ensure!(
            !role.is_empty() && digest_valid(&artifact.sha256),
            "artifact identity"
        );
        let root = match artifact.root.as_str() {
            "model" => model,
            "repo" => repo.as_path(),
            "policy" => parent,
            _ => bail!("unknown artifact root"),
        };
        let path = path_under(root, &artifact.path)?;
        ensure!(
            file_sha(&path)? == artifact.sha256,
            "artifact bytes mismatch: {role}"
        );
        paths.insert(role.clone(), path);
    }
    for role in [
        "config",
        "engine",
        "tokenizer",
        "cargo_lock",
        "feature_tree",
        "audit",
        "audit_b1_base",
        "reference_provenance",
        "corpus",
        "corpus_reference",
        "corpus_goldens",
        "fixture_goldens",
        "fixture_reference",
        "golden_index",
        "fixtures_exec",
        "mappings_v1",
        "batch_cases",
        "batch_goldens",
        "batch_golden_index",
        "in_process_evidence",
        "batch_in_process_evidence",
        "cpu_evidence",
        "serve_m3",
        "accept_m3",
        "accept_m4",
        "bench_m2_baseline",
        "benchmark_environment",
        "startup_check",
    ] {
        ensure!(paths.contains_key(role), "missing policy artifact: {role}");
    }
    Ok((artifacts, paths))
}

fn check_mappings(v: &Value) -> Result<Vec<Mapping>> {
    let mappings: Vec<Mapping> = serde_json::from_value(v.clone())?;
    ensure!(mappings.len() == 46, "46 mappings required");
    let mut covered = BTreeSet::new();
    for (position, m) in mappings.iter().enumerate() {
        ensure!(
            m.m2_group_id == position + 1
                && !m.adaptation.trim().is_empty()
                && !m.requests.is_empty(),
            "mapping identity/order"
        );
        if position >= 41 {
            ensure!(
                m.requests.len() == 1,
                "recovery mapping must be one request"
            );
        }
        for request in &m.requests {
            ensure!(
                (1..=8).contains(&request.len()) && request.iter().all(|i| *i < 22),
                "mapping request layout"
            );
            covered.extend(request.iter().copied());
        }
    }
    ensure!(covered.len() == 22, "all fixtures must be mapped");
    Ok(mappings)
}

fn check_probes(p: &Value, b: &Bundle) -> Result<()> {
    let tokenizer = tokenizers::Tokenizer::from_bytes(&bounded(b.path("tokenizer")?, JSON_LIMIT)?)
        .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
    let encode = |s: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?
            .get_ids()
            .to_vec())
    };
    let over = encode(text(p, "over_length_text")?)?;
    ensure!(
        over.len() > 1024 && over.iter().all(|id| *id < 131072 && *id != 11),
        "overlength probe does not exercise the intended gate"
    );
    if let Some(pad) = p["pad_text"].as_str() {
        ensure!(
            !pad.trim().is_empty() && encode(pad)? == vec![11],
            "pad_text must encode to exactly [11], or be null"
        );
    }
    Ok(())
}

#[derive(Clone)]
struct Probe {
    id: String,
    indices: Vec<usize>,
    s: usize,
    kind: ProbeKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProbeKind {
    Independent,
    Batch,
    Padding,
    RepeatSummary,
    PaddingSummary,
    Reconciliation,
}

fn probe(id: String, indices: Vec<usize>, s: usize, kind: ProbeKind) -> Probe {
    Probe {
        id,
        indices,
        s,
        kind,
    }
}

fn prepass_plan(b: &Bundle) -> Vec<Probe> {
    let mut plan = Vec::new();
    for i in 0..22 {
        plan.push(probe(
            format!("singleton-{i}"),
            vec![i],
            b.fixtures[i].ids.len(),
            ProbeKind::Independent,
        ));
    }
    let mut layouts = Vec::new();
    for i in 0..22 {
        layouts.push((format!("canonical-47-{i}-0"), vec![i, 6]));
        layouts.push((format!("canonical-47-{i}-1"), vec![6, i]));
    }
    layouts.extend(
        b.shapes
            .cases
            .iter()
            .map(|s| (s.case_id.clone(), s.fixtures.clone())),
    );
    for (id, indices) in layouts {
        let s = indices
            .iter()
            .map(|i| b.fixtures[*i].ids.len())
            .max()
            .unwrap_or(0);
        for suffix in ["first", "immediate"] {
            plan.push(probe(
                format!("{id}-{suffix}"),
                indices.clone(),
                s,
                ProbeKind::Batch,
            ));
        }
        plan.push(probe(
            format!("{id}-interleave-short"),
            vec![0],
            1,
            ProbeKind::Batch,
        ));
        plan.push(probe(
            format!("{id}-interleave-long"),
            vec![6],
            1024,
            ProbeKind::Batch,
        ));
        plan.push(probe(
            format!("{id}-later"),
            indices.clone(),
            s,
            ProbeKind::Batch,
        ));
        plan.push(probe(
            format!("repeat-{id}"),
            indices,
            s,
            ProbeKind::RepeatSummary,
        ));
    }
    plan.push(probe(
        "prepass-manifest-reconciliation".into(),
        vec![],
        0,
        ProbeKind::Reconciliation,
    ));
    for i in 0..22 {
        let length = b.fixtures[i].ids.len();
        if length == 1024 {
            continue;
        }
        for (suffix, indices, s) in [
            ("natural", vec![i, i], length),
            ("wide", vec![i, i], 1024),
            ("immediate", vec![i, i], 1024),
            ("interleave", vec![0, 0], 1),
            ("later", vec![i, i], 1024),
        ] {
            plan.push(probe(
                format!("padding-52-{i}-{suffix}"),
                indices,
                s,
                ProbeKind::Padding,
            ));
        }
        plan.push(probe(
            format!("padding-52-{i}"),
            vec![i, i],
            length,
            ProbeKind::PaddingSummary,
        ));
    }
    plan
}

fn comparison(v: &Value) -> Result<()> {
    closed(v, &["cosine", "max_abs"])?;
    let c = v["cosine"].as_f64().context("cosine")?;
    let a = v["max_abs"].as_f64().context("max_abs")?;
    ensure!(
        c.is_finite() && c > 0.999 && a.is_finite() && (0.0..=0.01).contains(&a),
        "numerical evidence gate"
    );
    Ok(())
}

fn math(record: &Value, b: &Bundle) -> Result<()> {
    let m = &record["math"];
    ensure!(
        m["audit_identity"] == b.audit_identity
            && m["candle_core_version"] == "0.9.2"
            && m["reduced_precision"] == json!({"f32": false, "f16": false, "bf16": false}),
        "manifest math binding"
    );
    Ok(())
}

fn output_hashes(record: &Value) -> Result<Vec<String>> {
    record["details"]["rows"]
        .as_array()
        .context("record rows")?
        .iter()
        .map(|row| checked_digest(row, "output_f32le_sha256"))
        .collect()
}

fn check_forward(
    begin: &Value,
    end: &Value,
    p: &Probe,
    hashes: &[String],
    b: &Bundle,
    forward: usize,
    request: usize,
) -> Result<()> {
    let padding = p.kind == ProbeKind::Padding;
    ensure!(
        begin["kind"]
            == if padding {
                "controlled_padding_begin"
            } else {
                "physical_forward_begin"
            }
            && end["kind"]
                == if padding {
                    "controlled_padding_end"
                } else {
                    "physical_forward_end"
                }
            && end["status"] == "ok",
        "manifest pair kind/status"
    );
    for r in [begin, end] {
        ensure!(
            r["B"] == p.indices.len() && r["S"] == p.s,
            "manifest geometry"
        );
        math(r, b)?;
        if padding {
            ensure!(
                r["diagnostic_id"] == forward
                    && r["case_id"] == p.id
                    && r["route"] == "Model::forward_with_padded_s"
                    && r["serving_route"] == false
                    && r["projection_rows"] == p.indices.len() * p.s,
                "padding manifest identity"
            );
        } else {
            ensure!(
                r["forward_id"] == forward && r["profile"] == "F",
                "forward identity"
            );
        }
    }
    ensure!(
        begin["projection_rows"] == p.indices.len() * p.s,
        "projection rows"
    );
    if p.kind == ProbeKind::Independent {
        let f = &b.fixtures[p.indices[0]];
        ensure!(
            begin["logical_request_row"] == p.indices[0]
                && begin["token_sha256"] == f.ids_sha256
                && end["token_sha256"] == f.ids_sha256
                && end["output_sha256"] == hashes[0]
                && begin.get("rows").is_none()
                && end.get("rows").is_none(),
            "independent singleton manifest"
        );
        return Ok(());
    }
    let valid: usize = p.indices.iter().map(|i| b.fixtures[*i].ids.len()).sum();
    if !padding {
        for r in [begin, end] {
            ensure!(
                r["execution_policy"] == BATCHED_EXECUTION_POLICY
                    && r["projection_rows"] == p.indices.len() * p.s
                    && r["attention_cells_before_heads"] == p.indices.len() * p.s * p.s
                    && r["valid_tokens"] == valid,
                "batch accounting"
            );
        }
        ensure!(
            begin["padding_tokens"] == p.indices.len() * p.s - valid,
            "padding count"
        );
    }
    let starts = begin["rows"].as_array().context("begin rows")?;
    let ends = end["rows"].as_array().context("end rows")?;
    ensure!(
        starts.len() == hashes.len() && ends.len() == hashes.len(),
        "manifest row count"
    );
    for (position, i) in p.indices.iter().enumerate() {
        let f = &b.fixtures[*i];
        let member = if padding {
            json!({
                "physical_row": position, "fixture_index": i,
                "valid_length": f.ids.len(), "token_sha256": f.ids_sha256
            })
        } else {
            json!({
                "physical_row": position, "request_id": request,
                "request_row": position, "case_id": p.id,
                "valid_length": f.ids.len(), "token_sha256": f.ids_sha256
            })
        };
        ensure!(starts[position] == member, "begin row association");
        let mut completed = member;
        completed["output_sha256"] = json!(hashes[position]);
        ensure!(ends[position] == completed, "end row association");
    }
    Ok(())
}

fn consume_evidence(
    evidence_path: &Path,
    b: &Bundle,
) -> Result<(Vec<Value>, BTreeMap<PathBuf, String>)> {
    let e = value(evidence_path)?;
    closed(
        &e,
        &[
            "schema",
            "producer",
            "scope",
            "execution_policy",
            "profile",
            "width",
            "input_sha256",
            "artifact_sha256",
            "audit_identity",
            "batching",
            "reference_row_order",
            "device_ordinal",
            "build_identity",
            "environment",
            "completed_families",
            "frozen_singleton_rows_checked",
            "canonical_target_cases",
            "declared_shape_cases",
            "padding_fixture_indices",
            "executor_physical_forwards",
            "padding_diagnostic_forwards",
            "direct_pairwise_comparisons",
            "records",
            "executor_manifest",
            "padding_manifest",
            "candidate_selection",
            "gates",
            "pending_required_cases_in_scope",
            "exceptions",
            "pass",
            "http_certified",
            "m4_accepted",
        ],
    )?;
    ensure!(!unresolved(&e), "unresolved numerical evidence");
    let hashes: BTreeMap<&str, &str> = b
        .artifacts
        .iter()
        .map(|(k, a)| (k.as_str(), a.sha256.as_str()))
        .collect();
    let padding_indices: Vec<usize> = (0..22)
        .filter(|i| b.fixtures[*i].ids.len() < 1024)
        .collect();
    ensure!(
        e["schema"] == "rustv1.batch_in_process_evidence.v1"
            && e["scope"] == "controlled_numerical_families_47_through_52"
            && e["execution_policy"] == BATCHED_EXECUTION_POLICY
            && e["profile"] == "F"
            && e["width"] == 2048
            && e["input_sha256"] == b.input_sha256
            && e["artifact_sha256"] == json!(hashes)
            && e["audit_identity"] == b.audit_identity
            && e["batching"] == batching()
            && e["reference_row_order"] == "fixture_index"
            && e["device_ordinal"] == 0
            && e["completed_families"] == json!([47, 48, 49, 50, 51, 52])
            && e["frozen_singleton_rows_checked"] == 22
            && e["canonical_target_cases"] == 44
            && e["declared_shape_cases"] == b.shapes.cases.len()
            && e["padding_fixture_indices"] == json!(padding_indices)
            && e["pending_required_cases_in_scope"] == json!([])
            && e["exceptions"] == json!([])
            && e["pass"] == true
            && e["http_certified"] == false
            && e["m4_accepted"] == false,
        "numerical evidence scope/bindings"
    );
    closed(&e["producer"], &["component", "command", "binary_sha256"])?;
    ensure!(
        e["producer"]["component"] == "accept_m4" && e["producer"]["command"] == "prepass",
        "wrong evidence producer"
    );
    checked_digest(&e["producer"], "binary_sha256")?;
    // Producer binary may precede this driver revision. Its digest is evidence
    // provenance, not a claim that it equals the current acceptance executable.
    closed(
        &e["build_identity"],
        &[
            "rustc_version",
            "nvcc_version",
            "cuda_compute_cap",
            "target_features",
        ],
    )?;
    for k in [
        "rustc_version",
        "nvcc_version",
        "cuda_compute_cap",
        "target_features",
    ] {
        ensure!(
            !text(&e["build_identity"], k)?.is_empty(),
            "empty build identity"
        );
    }
    closed(
        &e["environment"],
        &[
            "CUDA_VISIBLE_DEVICES",
            "CUBLAS_WORKSPACE_CONFIG",
            "NVIDIA_TF32_OVERRIDE",
        ],
    )?;
    for k in [
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "NVIDIA_TF32_OVERRIDE",
    ] {
        ensure!(
            e["environment"][k].is_null() || e["environment"][k].is_string(),
            "invalid environment value"
        );
    }
    ensure!(
        e["candidate_selection"]
            == json!({
                "declared_shapes": "shape_specific",
                "canonical_targets": "shape_specific",
                "unretained_canonical_companions": "canonical_same_fixture_same_position",
                "padding_diagnostics": "canonical_same_fixture_position_modulo_two"
            }),
        "candidate-selection contract"
    );
    let gates = [
        "all_outputs_width_finite_fp64_norm",
        "physical_b1_exact",
        "actual_to_candidate_direct",
        "actual_to_singleton_direct",
        "actual_to_independent_reference_direct",
        "identical_layout_immediate_exact",
        "identical_layout_interleaved_exact",
        "direct_pairwise_invariance",
        "executor_manifest_reconciled",
    ];
    closed(&e["gates"], &gates)?;
    ensure!(
        gates.iter().all(|k| e["gates"][*k] == true),
        "failed numerical gate"
    );

    let parent = evidence_path.parent().context("evidence parent")?;
    let mut companions = BTreeMap::new();
    let mut contents = BTreeMap::new();
    for (key, filename) in [
        ("records", "accept_records.jsonl"),
        ("executor_manifest", "executor_manifest.jsonl"),
        ("padding_manifest", "padding_manifest.jsonl"),
    ] {
        if key == "records" {
            closed(&e[key], &["path", "sha256", "count"])?;
        } else {
            closed(&e[key], &["path", "sha256"])?;
        }
        ensure!(e[key]["path"] == filename, "producer companion filename");
        let expected = checked_digest(&e[key], "sha256")?;
        let path = path_under(parent, filename)?;
        let bytes = bounded(&path, RECORD_LIMIT)?;
        ensure!(sha(&bytes) == expected, "companion hash: {key}");
        companions.insert(path, expected);
        contents.insert(key, jsonl(&bytes)?);
    }
    let records = contents.remove("records").context("records")?;
    let executor = contents.remove("executor_manifest").context("executor")?;
    let padding = contents.remove("padding_manifest").context("padding")?;
    let plan = prepass_plan(b);
    ensure!(
        records.len() == plan.len() && e["records"]["count"] == records.len(),
        "numerical record coverage"
    );
    let start = executor.first().context("executor start")?;
    ensure!(
        start["kind"] == "executor_start"
            && start["execution_policy"] == BATCHED_EXECUTION_POLICY
            && start["profile"] == "F"
            && start["batching"] == batching(),
        "numerical executor start"
    );
    math(start, b)?;

    let mut forward = 0usize;
    let mut diagnostic = 0usize;
    let mut request = 1usize;
    let mut history = [0u64; 22];
    let mut pairwise = 0u64;
    let mut outputs: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut seen = BTreeSet::new();

    for (record, p) in records.iter().zip(&plan) {
        closed(record, &["case_id", "status", "pass", "details"])?;
        ensure!(
            record["case_id"] == p.id
                && record["status"] == "executed"
                && record["pass"] == true
                && seen.insert(p.id.clone()),
            "missing, reordered, duplicate or failed numerical case"
        );
        let d = &record["details"];
        match p.kind {
            ProbeKind::RepeatSummary => {
                closed(
                    d,
                    &[
                        "family",
                        "immediate_exact",
                        "after_other_shapes_exact",
                        "ordered_fixture_indices",
                    ],
                )?;
                ensure!(
                    d["family"] == 51
                        && d["immediate_exact"] == true
                        && d["after_other_shapes_exact"] == true
                        && d["ordered_fixture_indices"] == json!(p.indices),
                    "repeat summary"
                );
                let base = p.id.strip_prefix("repeat-").context("repeat ID")?;
                let first = outputs.get(&format!("{base}-first"));
                ensure!(
                    first.is_some()
                        && first == outputs.get(&format!("{base}-immediate"))
                        && first == outputs.get(&format!("{base}-later")),
                    "repeat digest mismatch"
                );
                continue;
            }
            ProbeKind::PaddingSummary => {
                closed(
                    d,
                    &[
                        "family",
                        "fixture_index",
                        "B",
                        "widths",
                        "direct_width_comparisons",
                        "immediate_exact",
                        "after_other_shape_exact",
                        "serving_route",
                    ],
                )?;
                ensure!(
                    d["family"] == 52
                        && d["fixture_index"] == p.indices[0]
                        && d["B"] == 2
                        && d["widths"] == json!([p.s, 1024])
                        && d["immediate_exact"] == true
                        && d["after_other_shape_exact"] == true
                        && d["serving_route"] == false,
                    "padding summary"
                );
                let comparisons = d["direct_width_comparisons"]
                    .as_array()
                    .context("width comparisons")?;
                ensure!(comparisons.len() == 2, "width comparison count");
                for c in comparisons {
                    comparison(c)?;
                }
                let wide = outputs.get(&format!("{}-wide", p.id));
                ensure!(
                    wide.is_some()
                        && wide == outputs.get(&format!("{}-immediate", p.id))
                        && wide == outputs.get(&format!("{}-later", p.id)),
                    "padding repeat digest mismatch"
                );
                continue;
            }
            ProbeKind::Reconciliation => {
                closed(
                    d,
                    &[
                        "physical_forwards",
                        "unmatched_records",
                        "executor_manifest_sha256",
                    ],
                )?;
                ensure!(
                    d["physical_forwards"] == forward
                        && d["unmatched_records"] == 0
                        && d["executor_manifest_sha256"] == e["executor_manifest"]["sha256"],
                    "reconciliation record"
                );
                continue;
            }
            _ => {}
        }

        closed(d, &["route", "B", "S", "rows"])?;
        let route = match p.kind {
            ProbeKind::Independent => "IndependentExecutor::forward_serving",
            ProbeKind::Batch => "IndependentExecutor::forward_batch",
            ProbeKind::Padding => "Model::forward_with_padded_s",
            _ => unreachable!(),
        };
        ensure!(
            d["route"] == route && d["B"] == p.indices.len() && d["S"] == p.s,
            "numerical record geometry/route"
        );
        let rows = d["rows"].as_array().context("numerical rows")?;
        ensure!(rows.len() == p.indices.len(), "numerical row cardinality");
        for (position, (row, i)) in rows.iter().zip(&p.indices).enumerate() {
            closed(
                row,
                &[
                    "position",
                    "fixture_id",
                    "fixture_index",
                    "output_f32le_sha256",
                    "singleton_comparison",
                    "independent_reference_comparison",
                    "candidate_comparison",
                    "candidate_selection",
                    "direct_prior_row_comparisons",
                ],
            )?;
            ensure!(
                row["position"] == position
                    && row["fixture_index"] == *i
                    && row["fixture_id"] == b.fixtures[*i].id
                    && row["direct_prior_row_comparisons"] == history[*i],
                "numerical row identity/history"
            );
            for k in [
                "singleton_comparison",
                "independent_reference_comparison",
                "candidate_comparison",
            ] {
                comparison(&row[k])?;
            }
            let selection = if p.kind == ProbeKind::Independent || p.indices.len() == 1 {
                "frozen_singleton"
            } else if p.kind == ProbeKind::Padding {
                "canonical_padding_diagnostic"
            } else if b.candidates.contains_key(&(p.indices.clone(), position)) {
                "shape_specific"
            } else {
                "canonical_companion"
            };
            ensure!(
                row["candidate_selection"] == selection,
                "candidate selection"
            );
            let hash = checked_digest(row, "output_f32le_sha256")?;
            if p.indices.len() == 1 {
                ensure!(hash == b.fixtures[*i].golden_sha256, "B1 exact digest");
            }
            pairwise = pairwise
                .checked_add(history[*i])
                .context("pairwise overflow")?;
            history[*i] = history[*i].checked_add(1).context("history overflow")?;
        }
        let hashes = output_hashes(record)?;
        if p.kind == ProbeKind::Padding {
            let begin = padding
                .get(2 * diagnostic)
                .context("missing padding begin")?;
            let end = padding
                .get(2 * diagnostic + 1)
                .context("missing padding end")?;
            check_forward(begin, end, p, &hashes, b, diagnostic, 0)?;
            diagnostic += 1;
        } else {
            let begin = executor
                .get(1 + 2 * forward)
                .context("missing forward begin")?;
            let end = executor
                .get(2 + 2 * forward)
                .context("missing forward end")?;
            check_forward(begin, end, p, &hashes, b, forward, request)?;
            forward += 1;
            if p.kind == ProbeKind::Batch {
                request += 1;
            }
        }
        outputs.insert(p.id.clone(), hashes);
    }
    ensure!(
        executor.len() == 1 + 2 * forward
            && padding.len() == 2 * diagnostic
            && e["executor_physical_forwards"] == forward
            && e["padding_diagnostic_forwards"] == diagnostic
            && e["direct_pairwise_comparisons"] == pairwise,
        "unmatched manifests or numerical counter mismatch"
    );
    Ok((records, companions))
}

/// The rejected request is immediately followed by its mapped recovery.
/// These IDs exactly match the delivered HTTP module.
pub fn mapping_case_ids(mappings: &[Mapping], pad_available: bool) -> Vec<String> {
    let mut ids = Vec::new();
    for m in &mappings[..41] {
        for request in 0..m.requests.len() {
            ids.push(format!("mapped-{}-{request}", m.m2_group_id));
        }
    }
    for (offset, rejection) in [
        "invalid-empty-batch",
        "invalid-last-empty",
        if pad_available {
            "invalid-last-pad"
        } else {
            "invalid-last-empty-pad-substitute"
        },
        "invalid-last-overlength",
        "invalid-token-array",
    ]
    .iter()
    .enumerate()
    {
        ids.push((*rejection).to_owned());
        ids.push(format!("mapped-{}-0", 42 + offset));
    }
    ids
}

fn http_case_ids(mappings: &[Mapping], pad_available: bool) -> Vec<String> {
    let mut ids = mapping_case_ids(mappings, pad_available);
    for i in 0..22 {
        ids.push(format!("base64-singleton-{i}"));
    }
    for name in ["mixed", "eight-long"] {
        for encoding in ["float", "base64"] {
            ids.push(format!("{name}-{encoding}"));
        }
    }
    ids.extend(
        [
            "scalar-default",
            "scalar-null-encoding",
            "chunked-valid",
            "invalid-empty-text",
            "invalid-whitespace",
            "invalid-null-input",
            "invalid-mixed-input-types",
            "invalid-model",
            "invalid-encoding-string",
            "invalid-encoding-number",
            "invalid-encoding-object",
            "invalid-encoding-array",
            "invalid-encoding-boolean",
            "invalid-unknown-field",
            "http-input-count-9",
            "http-input-count-33",
            "invalid-malformed-json",
            "invalid-media",
            "invalid-content-encoding",
            "payload-too-large-length",
            "payload-too-large-chunked",
            "final-open-recovery",
            "closed-admission",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    ids
}

#[derive(Clone)]
pub struct Required {
    /// Phase order is also the required completion order.
    pub phases: Vec<(String, Vec<String>)>,
}

impl Required {
    fn build(b: &Bundle, mappings: &[Mapping], pad_available: bool) -> Result<Self> {
        let strings = |s: &[&str]| s.iter().map(|s| (*s).to_owned()).collect();
        let required = Self {
            phases: vec![
                (
                    PREPASS.into(),
                    prepass_plan(b).into_iter().map(|p| p.id).collect(),
                ),
                (
                    POLICY.into(),
                    strings(&["finalized-policy-bindings", "prepass-evidence-consumed"]),
                ),
                (
                    LAUNCH.into(),
                    strings(&[
                        "authenticated-orchestration-ready",
                        "fresh-launch-attestation",
                        "startup-manifest-reconciliation",
                    ]),
                ),
                (HTTP.into(), http_case_ids(mappings, pad_available)),
                (
                    EXIT.into(),
                    strings(&[
                        "authenticated-supervised-shutdown",
                        "old-process-exit-proven",
                    ]),
                ),
                (INDEPENDENT.into(), {
                    let mut ids = strings(&["independent-startup-reconciled"]);
                    ids.extend(mapping_case_ids(mappings, pad_available));
                    ids
                }),
                (
                    JOIN.into(),
                    strings(&["independent-owner-shutdown-and-join"]),
                ),
                (
                    SCHEDULER.into(),
                    strings(&[
                        "family-53-live-cross-request-coalescing",
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
                        "scheduler-owners-joined",
                    ]),
                ),
                (
                    LIFECYCLE.into(),
                    strings(&[
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
                    ]),
                ),
            ],
        };
        required.validate()?;
        Ok(required)
    }

    pub fn validate(&self) -> Result<()> {
        let mut phases = BTreeSet::new();
        ensure!(!self.phases.is_empty(), "empty required ledger");
        for (phase, ids) in &self.phases {
            ensure!(
                phases.insert(phase) && !ids.is_empty(),
                "duplicate/empty phase"
            );
            let mut cases = BTreeSet::new();
            for id in ids {
                ensure!(
                    !id.is_empty()
                        && id.len() <= 160
                        && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
                        && cases.insert(id),
                    "invalid/duplicate required case"
                );
            }
        }
        Ok(())
    }

    pub fn as_json(&self) -> Value {
        json!({
            "schema": "rustv1.accept_m4_required_cases.v1",
            "phases": self.phases.iter().map(|(phase, ids)| json!({
                "acceptance_phase": phase, "required_case_ids": ids
            })).collect::<Vec<_>>()
        })
    }

    pub fn count(&self) -> usize {
        self.phases.iter().map(|(_, ids)| ids.len()).sum()
    }
}

fn prepare(
    model_dir: &Path,
    input_path: &Path,
    expected_input_sha: &str,
    policy_path: &Path,
    allow_empty_materialization: bool,
) -> Result<Loaded> {
    let bytes = policy_bytes(policy_path)?;
    let mut policy: Value = serde_json::from_slice(&bytes)?;
    check_policy_header(&policy)?;
    let bundle = contract::load(model_dir, input_path, expected_input_sha)?;
    ensure!(
        bundle.engine.max_len == 1024
            && bundle.engine.wait_us == 1000
            && serde_json::to_value(&bundle.shapes.batching)? == batching()
            && policy["audit_identity"] == bundle.audit_identity,
        "qualification envelope/audit identity"
    );
    let (artifacts, paths) = resolve(&policy, model_dir, policy_path)?;

    // Every shared role binds both identity and actual resolved file.
    for (role, a) in &bundle.artifacts {
        if let Some(p) = artifacts.get(role) {
            ensure!(
                p.sha256 == a.sha256 && paths.get(role) == bundle.paths.get(role),
                "policy/prepass path or digest mismatch: {role}"
            );
        }
    }
    for role in [
        "config",
        "engine",
        "tokenizer",
        "cargo_lock",
        "feature_tree",
        "audit",
        "reference_provenance",
        "fixtures_exec",
        "fixture_goldens",
        "fixture_reference",
        "batch_goldens",
        "batch_golden_index",
        "mappings_v1",
    ] {
        let a = bundle
            .artifacts
            .get(role)
            .context("required prepass role")?;
        ensure!(
            artifacts[role].sha256 == a.sha256 && paths.get(role) == bundle.paths.get(role),
            "missing or mismatched cross-phase binding: {role}"
        );
    }
    for (role, a) in &bundle.artifacts {
        if role.starts_with("weight:") {
            ensure!(
                artifacts.get(role).is_some_and(|p| p.sha256 == a.sha256)
                    && paths.get(role) == bundle.paths.get(role),
                "policy checkpoint binding: {role}"
            );
        }
    }
    ensure!(
        artifacts["audit"].sha256 != artifacts["audit_b1_base"].sha256,
        "B1-only audit cannot be the active batched audit"
    );
    let executable_sha256 = file_sha(&std::env::current_exe()?)?;
    ensure!(
        artifacts["accept_m4"].sha256 == executable_sha256,
        "running acceptance executable differs from policy"
    );

    for (field, role, count) in [
        ("fixtures", "fixtures_exec", 22),
        ("mappings", "mappings_v1", 46),
    ] {
        let array = value(&paths[role])?;
        ensure!(
            array.as_array().is_some_and(|a| a.len() == count),
            "materialization count"
        );
        if allow_empty_materialization && policy[field] == json!([]) {
            policy[field] = array;
        } else {
            ensure!(
                policy[field] == array,
                "materialized array mismatch: {field}"
            );
        }
    }
    let mappings = check_mappings(&policy["mappings"])?;
    check_probes(&policy, &bundle)?;

    let check: rustv1::http_serve::StartupCheck =
        serde_json::from_slice(&bounded(&paths["startup_check"], JSON_LIMIT)?)?;
    ensure!(
        !check.tokens.is_empty()
            && check.tokens.len() <= 4096
            && check.tokens.len() == check.embeddings.len(),
        "startup check cardinality"
    );
    for (tokens, row) in check.tokens.iter().zip(&check.embeddings) {
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= 1024
                && tokens.iter().all(|id| *id < 131072 && *id != 11),
            "invalid startup tokens"
        );
        contract::norm(row)?;
        // For known fixture rows the startup artifact must retain frozen bits.
        for fixture in &bundle.fixtures {
            if tokens == &fixture.ids {
                ensure!(
                    sha(&row_bytes(row)) == fixture.golden_sha256
                        && ids_sha(tokens) == fixture.ids_sha256,
                    "startup fixture exact-bit binding"
                );
            }
        }
    }

    let (prepass_records, companion_hashes) =
        consume_evidence(&paths["batch_in_process_evidence"], &bundle)?;
    let required = Required::build(&bundle, &mappings, policy["pad_text"].is_string())?;
    Ok(Loaded {
        policy_sha256: sha(&bytes),
        bytes,
        policy,
        policy_path: policy_path.canonicalize()?,
        paths,
        artifacts,
        bundle,
        mappings,
        prepass_records,
        required,
        companion_hashes,
        input_path: input_path.canonicalize()?,
        executable_sha256,
    })
}

/// Load an already-finalized policy. Does not alter any file or policy value.
pub fn load(
    model_dir: &Path,
    input_path: &Path,
    expected_input_sha: &str,
    policy_path: &Path,
) -> Result<Loaded> {
    let loaded = prepare(
        model_dir,
        input_path,
        expected_input_sha,
        policy_path,
        false,
    )?;
    loaded.recheck()?;
    Ok(loaded)
}

/// Validate and materialize a prelaunch policy to a NEW file in the same
/// directory. Keeping the directory preserves all policy-root bindings.
/// All non-materialization placeholders must already have been finalized.
pub fn materialize(
    model_dir: &Path,
    input_path: &Path,
    expected_input_sha: &str,
    source: &Path,
    destination: &Path,
) -> Result<String> {
    let loaded = prepare(model_dir, input_path, expected_input_sha, source, true)?;
    let parent = |p: &Path| -> Result<PathBuf> {
        Ok(p.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .canonicalize()?)
    };
    ensure!(
        parent(source)? == parent(destination)?,
        "materialization must preserve policy root"
    );
    ensure!(
        source != destination,
        "materialization may not overwrite source"
    );
    loaded.recheck()?;
    let bytes = serde_json::to_vec_pretty(&loaded.policy)?;
    write_new(destination, &bytes)?;
    // Read through the same 0600 checker; a restrictive umask that removes
    // owner permissions is rejected instead of silently weakening validation.
    ensure!(
        policy_bytes(destination)? == bytes,
        "materialized policy reread"
    );
    Ok(sha(&bytes))
}

impl Loaded {
    pub fn recheck(&self) -> Result<()> {
        ensure!(
            policy_bytes(&self.policy_path)? == self.bytes
                && file_sha(&self.input_path)? == self.bundle.input_sha256
                && file_sha(&std::env::current_exe()?)? == self.executable_sha256,
            "policy/input/executable changed"
        );
        self.bundle.recheck_bindings()?;
        for (role, artifact) in &self.artifacts {
            ensure!(
                file_sha(&self.paths[role])? == artifact.sha256,
                "policy artifact changed: {role}"
            );
        }
        for (path, expected) in &self.companion_hashes {
            ensure!(file_sha(path)? == *expected, "numerical companion changed");
        }
        Ok(())
    }

    /// Redacted binding summary suitable for evidence.
    pub fn binding_summary(&self) -> Value {
        let artifacts: BTreeMap<&str, &str> = self
            .artifacts
            .iter()
            .map(|(role, a)| (role.as_str(), a.sha256.as_str()))
            .collect();
        json!({
            "policy_sha256": self.policy_sha256,
            "prepass_input_sha256": self.bundle.input_sha256,
            "artifact_sha256": artifacts,
            "run_id": self.policy["run_id"],
            "audit_identity": self.bundle.audit_identity,
            "required_case_count": self.required.count(),
            "samples_empty": true,
            "materialized_fixture_count": 22,
            "materialized_mapping_count": 46
        })
    }
}

/// Append-only, single-writer global ledger. A failed record is terminal.
/// Required cases are frozen before the first completion is appended.
pub struct Ledger {
    file: File,
    path: PathBuf,
    required: Required,
    phase: usize,
    position: usize,
    count: usize,
    failed: bool,
    policy_sha256: String,
    skeleton_sha256: String,
}

impl Ledger {
    /// `out` must already be a newly created, private evidence directory.
    pub fn create(out: &Path, loaded: &Loaded) -> Result<Self> {
        loaded.required.validate()?;
        let skeleton = serde_json::to_vec_pretty(&json!({
            "policy_sha256": loaded.policy_sha256,
            "required": loaded.required.as_json()
        }))?;
        write_new(&out.join("required_cases.json"), &skeleton)?;
        let path = out.join("accept_records.jsonl");
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&path)?;
        Ok(Self {
            file,
            path,
            required: loaded.required.clone(),
            phase: 0,
            position: 0,
            count: 0,
            failed: false,
            policy_sha256: loaded.policy_sha256.clone(),
            skeleton_sha256: sha(&skeleton),
        })
    }

    fn expected(&self) -> Result<(&str, &str)> {
        let (phase, ids) = self
            .required
            .phases
            .get(self.phase)
            .context("ledger already complete")?;
        Ok((phase, ids.get(self.position).context("ledger position")?))
    }

    /// `echo=false` is for importing an unchanged producer that already emitted
    /// this case to the same run's stdout. It must not be used to hide cases.
    pub fn append(
        &mut self,
        phase: &str,
        id: &str,
        pass: bool,
        details: Value,
        echo: bool,
    ) -> Result<()> {
        ensure!(!self.failed, "ledger is terminal after failure");
        let (expected_phase, expected_id) = self.expected()?;
        ensure!(
            phase == expected_phase && id == expected_id,
            "unexpected, duplicate or out-of-order case"
        );
        let record = json!({
            "acceptance_phase": phase,
            "case_id": id,
            "status": "executed",
            "pass": pass,
            "details": details
        });
        serde_json::to_writer(&mut self.file, &record)?;
        self.file.write_all(b"\n")?;
        self.file.sync_data()?;
        if echo {
            println!("{record}");
        }
        self.count += 1;
        if !pass {
            self.failed = true;
            bail!("case failed: {phase}/{id}");
        }
        self.position += 1;
        if self.position == self.required.phases[self.phase].1.len() {
            self.phase += 1;
            self.position = 0;
        }
        Ok(())
    }

    /// Import a complete phase's already-validated producer records.
    /// The independent expected sequence remains authoritative.
    pub fn import(
        &mut self,
        phase: &str,
        records: &[Value],
        source_sha256: &str,
        echo: bool,
    ) -> Result<()> {
        ensure!(digest_valid(source_sha256), "source digest");
        let (expected_phase, ids) = self
            .required
            .phases
            .get(self.phase)
            .context("phase complete")?;
        ensure!(
            expected_phase == phase && self.position == 0 && ids.len() == records.len(),
            "phase import coverage"
        );
        // Validate the entire import before appending any member.
        for (record, id) in records.iter().zip(ids) {
            ensure!(
                record["case_id"] == *id
                    && record["status"] == "executed"
                    && record["pass"] == true
                    && record["details"].is_object()
                    && record.get("acceptance_phase").is_none_or(|v| v == phase),
                "invalid imported phase record"
            );
        }
        for record in records {
            let id = text(record, "case_id")?;
            self.append(
                phase,
                id,
                true,
                json!({
                    "source_records_sha256": source_sha256,
                    "producer_record": record
                }),
                echo,
            )?;
        }
        Ok(())
    }

    pub fn pending(&self) -> Vec<Value> {
        let mut pending = Vec::new();
        for (index, (phase, ids)) in self.required.phases.iter().enumerate().skip(self.phase) {
            let offset = if index == self.phase {
                self.position
            } else {
                0
            };
            for id in &ids[offset..] {
                pending.push(json!({"acceptance_phase": phase, "case_id": id}));
            }
        }
        pending
    }

    /// Readback proof only. This method creates no completion manifest.
    /// The coordinator must write its manifest LAST, after its final binding,
    /// process-exit and evidence-output checks.
    pub fn finish(&mut self) -> Result<Value> {
        ensure!(
            !self.failed
                && self.phase == self.required.phases.len()
                && self.position == 0
                && self.count == self.required.count(),
            "global required-case ledger is incomplete"
        );
        self.file.sync_all()?;
        let bytes = bounded(&self.path, RECORD_LIMIT)?;
        let records = jsonl(&bytes)?;
        ensure!(
            records.len() == self.required.count(),
            "ledger readback count"
        );
        let expected = self
            .required
            .phases
            .iter()
            .flat_map(|(phase, ids)| ids.iter().map(move |id| (phase, id)));
        for (record, (phase, id)) in records.iter().zip(expected) {
            closed(
                record,
                &["acceptance_phase", "case_id", "status", "pass", "details"],
            )?;
            ensure!(
                record["acceptance_phase"] == *phase
                    && record["case_id"] == *id
                    && record["status"] == "executed"
                    && record["pass"] == true,
                "ledger readback mismatch"
            );
        }
        Ok(json!({
            "policy_sha256": self.policy_sha256,
            "required_cases_sha256": self.skeleton_sha256,
            "records_sha256": sha(&bytes),
            "required_case_count": self.required.count(),
            "completed_case_count": records.len(),
            "pending_required_cases": [],
            "exceptions": [],
            "pass": true
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_schema_rejects_missing_and_extra() {
        assert!(closed(&json!({"a": 1}), &["a"]).is_ok());
        assert!(closed(&json!({}), &["a"]).is_err());
        assert!(closed(&json!({"a": 1, "b": 2}), &["a"]).is_err());
    }

    #[test]
    fn comparison_is_strict_at_cosine_boundary() {
        assert!(comparison(&json!({"cosine": 0.999, "max_abs": 0.0})).is_err());
        assert!(comparison(&json!({"cosine": 1.0, "max_abs": 0.01})).is_ok());
        assert!(comparison(&json!({"cosine": 1.0, "max_abs": 0.010001})).is_err());
        assert!(comparison(&json!({"cosine": 1.0, "max_abs": -0.1})).is_err());
    }

    #[test]
    fn jsonl_is_not_permissive_about_truncation() {
        assert!(jsonl(b"{}\n").is_ok());
        assert!(jsonl(b"{}").is_err());
        assert!(jsonl(b"{}\n\n").is_err());
        assert!(jsonl(b"").is_err());
    }

    #[test]
    fn required_ledger_rejects_duplicate_cases() {
        let required = Required {
            phases: vec![("phase".into(), vec!["case".into(), "case".into()])],
        };
        assert!(required.validate().is_err());
    }

    #[test]
    fn recovery_order_is_immediate_and_fallback_is_named() {
        let mappings: Vec<Mapping> = (1..=46)
            .map(|i| Mapping {
                m2_group_id: i,
                adaptation: "test".into(),
                requests: vec![vec![0]],
            })
            .collect();
        let ids = mapping_case_ids(&mappings, false);
        let position = ids
            .iter()
            .position(|id| id == "invalid-last-empty-pad-substitute")
            .unwrap();
        assert_eq!(ids[position + 1], "mapped-44-0");
        assert!(!ids.iter().any(|id| id == "invalid-last-pad"));
        assert_eq!(ids.last().unwrap(), "mapped-46-0");
    }

    #[test]
    fn envelope_and_materialization_are_exact() {
        assert_eq!(batching()["max_attention_cells"], 33_554_432);
        assert_eq!(batching()["max_wait_us"], 1000);
        assert_eq!(materialization()["fixtures"]["count"], 22);
        assert_eq!(materialization()["mappings"]["count"], 46);
        assert!(materialization().get("instruction").is_none());
    }
}
