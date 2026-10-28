use crate::contract::{
    bounded, compare, exact, file_sha, ids_sha, norm, row_bytes, sha, write_new, Bundle, WIDTH,
};
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device};
use rustv1::{
    cuda_math::MathModePin,
    model::{ExecutionRow, IndependentExecutor, Model, BATCHED_EXECUTION_POLICY},
    precision::PrecisionProfile,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::Write,
    path::Path,
};

struct ExpectedForward {
    indices: Vec<usize>,
    associations: Option<Vec<ExecutionRow>>,
    hashes: Vec<String>,
}

struct Records {
    file: File,
    count: usize,
}

impl Records {
    fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            file: OpenOptions::new().write(true).create_new(true).open(path)?,
            count: 0,
        })
    }

    fn pass(&mut self, case: &str, details: Value) -> Result<()> {
        let record = json!({
            "case_id": case,
            "status": "executed",
            "pass": true,
            "details": details
        });
        serde_json::to_writer(&mut self.file, &record)?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        self.count += 1;
        println!("{record}");
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }
}

struct Runner<'a> {
    bundle: &'a Bundle,
    records: Records,
    expected: Vec<ExpectedForward>,
    next_request: u64,
    history: BTreeMap<usize, Vec<Vec<f32>>>,
    pairwise_comparisons: u64,
}

impl Runner<'_> {
    fn inspect(
        &mut self,
        case: &str,
        indices: &[usize],
        s: usize,
        output: &[Vec<f32>],
        candidate_mode: &str,
        route: &str,
    ) -> Result<()> {
        ensure!(output.len() == indices.len(), "output cardinality");
        let mut rows = Vec::new();
        for (position, (&fixture_index, row)) in indices.iter().zip(output).enumerate() {
            norm(row)?;
            let fixture = &self.bundle.fixtures[fixture_index];
            let singleton = compare(row, &self.bundle.singleton[fixture_index])?;
            let reference = compare(row, &self.bundle.reference[fixture_index])?;
            if indices.len() == 1 && s == fixture.ids.len() {
                ensure!(
                    sha(&row_bytes(row)) == fixture.golden_sha256,
                    "physical B1 exact singleton mismatch"
                );
            }
            let (candidate, selection) = match candidate_mode {
                "shape" => (self.bundle.candidate(indices, position)?, "shape_specific"),
                "canonical-pair" => {
                    match self.bundle.candidates.get(&(indices.to_vec(), position)) {
                        Some(candidate) => (candidate.as_slice(), "shape_specific"),
                        None => (
                            self.bundle.canonical(fixture_index, position % 2)?,
                            "canonical_companion",
                        ),
                    }
                }
                "canonical" => (
                    self.bundle.canonical(fixture_index, position % 2)?,
                    "canonical_padding_diagnostic",
                ),
                "singleton" => (
                    self.bundle.singleton[fixture_index].as_slice(),
                    "frozen_singleton",
                ),
                _ => anyhow::bail!("invalid internal candidate mode"),
            };
            let candidate_comparison = compare(row, candidate)?;
            let prior = self.history.entry(fixture_index).or_default();
            for other in prior.iter() {
                compare(row, other)?;
                self.pairwise_comparisons = self
                    .pairwise_comparisons
                    .checked_add(1)
                    .context("pairwise counter overflow")?;
            }
            let compared_prior_rows = prior.len();
            prior.push(row.clone());
            rows.push(json!({
                "position": position,
                "fixture_id": fixture.id,
                "fixture_index": fixture_index,
                "output_f32le_sha256": sha(&row_bytes(row)),
                "singleton_comparison": singleton,
                "independent_reference_comparison": reference,
                "candidate_comparison": candidate_comparison,
                "candidate_selection": selection,
                "direct_prior_row_comparisons": compared_prior_rows
            }));
        }
        self.records.pass(
            case,
            json!({
                "route": route,
                "B": indices.len(),
                "S": s,
                "rows": rows
            }),
        )
    }

    fn batch(
        &mut self,
        executor: &mut IndependentExecutor,
        case: &str,
        indices: &[usize],
        mode: &str,
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(case.len() <= 128, "execution case label too long");
        let tokens: Vec<Vec<u32>> = indices
            .iter()
            .map(|&i| self.bundle.fixtures[i].ids.clone())
            .collect();
        let s = tokens.iter().map(Vec::len).max().context("empty batch")?;
        let request = self.next_request;
        self.next_request = request.checked_add(1).context("request counter overflow")?;
        let associations: Vec<ExecutionRow> = indices
            .iter()
            .enumerate()
            .map(|(request_row, _)| ExecutionRow {
                request_id: request,
                request_row,
                case_id: Some(case.to_owned()),
            })
            .collect();
        let output = executor.forward_batch(&tokens, &associations)?;
        self.expected.push(ExpectedForward {
            indices: indices.to_vec(),
            associations: Some(associations),
            hashes: output.iter().map(|row| sha(&row_bytes(row))).collect(),
        });
        self.inspect(
            case,
            indices,
            s,
            &output,
            mode,
            "IndependentExecutor::forward_batch",
        )?;
        Ok(output)
    }

    fn repeated(
        &mut self,
        executor: &mut IndependentExecutor,
        case: &str,
        indices: &[usize],
        mode: &str,
    ) -> Result<()> {
        let first = self.batch(executor, &format!("{case}-first"), indices, mode)?;
        let immediate = self.batch(executor, &format!("{case}-immediate"), indices, mode)?;
        exact(&first, &immediate)?;
        self.batch(
            executor,
            &format!("{case}-interleave-short"),
            &[0],
            "singleton",
        )?;
        self.batch(
            executor,
            &format!("{case}-interleave-long"),
            &[6],
            "singleton",
        )?;
        let later = self.batch(executor, &format!("{case}-later"), indices, mode)?;
        exact(&first, &later)?;
        self.records.pass(
            &format!("repeat-{case}"),
            json!({
                "family": 51,
                "immediate_exact": true,
                "after_other_shapes_exact": true,
                "ordered_fixture_indices": indices
            }),
        )
    }
}

fn check_math(value: &Value, bundle: &Bundle) -> Result<()> {
    ensure!(
        value["audit_identity"] == bundle.audit_identity
            && value["candle_core_version"] == "0.9.2"
            && value["reduced_precision"]["f32"] == false
            && value["reduced_precision"]["f16"] == false
            && value["reduced_precision"]["bf16"] == false,
        "manifest math identity/control mismatch"
    );
    Ok(())
}

fn reconcile(path: &Path, bundle: &Bundle, expected: &[ExpectedForward]) -> Result<usize> {
    let bytes = bounded(path, 256 * 1024 * 1024)?;
    ensure!(bytes.ends_with(b"\n"), "truncated executor manifest");
    let lines: Vec<&[u8]> = bytes
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .collect();
    ensure!(
        lines.len() == 1 + expected.len() * 2,
        "unexpected/missing physical forwards"
    );
    let start: Value = serde_json::from_slice(lines[0])?;
    ensure!(
        start["kind"] == "executor_start"
            && start["execution_policy"] == BATCHED_EXECUTION_POLICY
            && start["profile"] == "F"
            && start["batching"] == serde_json::to_value(&bundle.shapes.batching)?,
        "executor start binding"
    );
    check_math(&start["math"], bundle)?;

    for (forward_id, expected) in expected.iter().enumerate() {
        let begin: Value = serde_json::from_slice(lines[1 + 2 * forward_id])?;
        let end: Value = serde_json::from_slice(lines[2 + 2 * forward_id])?;
        let b = expected.indices.len();
        let lengths: Vec<usize> = expected
            .indices
            .iter()
            .map(|&i| bundle.fixtures[i].ids.len())
            .collect();
        let s = *lengths.iter().max().context("empty expected forward")?;
        for record in [&begin, &end] {
            ensure!(
                record["forward_id"] == forward_id
                    && record["B"] == b
                    && record["S"] == s
                    && record["profile"] == "F",
                "forward geometry/identity mismatch"
            );
            check_math(&record["math"], bundle)?;
        }
        ensure!(
            begin["kind"] == "physical_forward_begin"
                && end["kind"] == "physical_forward_end"
                && end["status"] == "ok",
            "forward begin/end mismatch"
        );
        ensure!(begin["projection_rows"] == b * s, "projection rows");
        match &expected.associations {
            None => {
                ensure!(b == 1, "independent forward must have B1");
                let fixture = &bundle.fixtures[expected.indices[0]];
                ensure!(
                    begin["token_sha256"] == fixture.ids_sha256
                        && end["token_sha256"] == fixture.ids_sha256
                        && end["output_sha256"] == expected.hashes[0]
                        && begin.get("rows").is_none()
                        && end.get("rows").is_none(),
                    "independent manifest row"
                );
            }
            Some(associations) => {
                ensure!(
                    begin["execution_policy"] == BATCHED_EXECUTION_POLICY
                        && end["execution_policy"] == BATCHED_EXECUTION_POLICY
                        && end["projection_rows"] == b * s
                        && begin["attention_cells_before_heads"] == b * s * s
                        && end["attention_cells_before_heads"] == b * s * s
                        && begin["valid_tokens"] == lengths.iter().sum::<usize>()
                        && end["valid_tokens"] == lengths.iter().sum::<usize>()
                        && begin["padding_tokens"] == b * s - lengths.iter().sum::<usize>(),
                    "batched manifest accounting"
                );
                let begin_rows = begin["rows"].as_array().context("begin rows")?;
                let end_rows = end["rows"].as_array().context("end rows")?;
                ensure!(
                    begin_rows.len() == b && end_rows.len() == b,
                    "manifest row count"
                );
                for position in 0..b {
                    let fixture = &bundle.fixtures[expected.indices[position]];
                    let association = &associations[position];
                    let member = json!({
                        "physical_row": position,
                        "request_id": association.request_id,
                        "request_row": association.request_row,
                        "case_id": association.case_id,
                        "valid_length": fixture.ids.len(),
                        "token_sha256": fixture.ids_sha256
                    });
                    ensure!(begin_rows[position] == member, "begin association");
                    let mut completed = member;
                    completed["output_sha256"] = json!(expected.hashes[position]);
                    ensure!(end_rows[position] == completed, "end association/output");
                }
            }
        }
    }
    Ok(expected.len())
}

#[allow(clippy::too_many_arguments)]
fn diagnostic(
    model: &Model,
    pin: &MathModePin,
    manifest: &mut File,
    runner: &mut Runner<'_>,
    case: &str,
    indices: &[usize],
    s: usize,
    next_id: &mut u64,
) -> Result<Vec<Vec<f32>>> {
    ensure!(
        runner.bundle.shapes.batching.fits(indices.len(), s),
        "diagnostic geometry exceeds qualification capacity"
    );
    let tokens: Vec<Vec<u32>> = indices
        .iter()
        .map(|&i| runner.bundle.fixtures[i].ids.clone())
        .collect();
    ensure!(
        tokens.iter().all(|ids| ids.len() <= s),
        "diagnostic S too small"
    );
    let id = *next_id;
    *next_id = id.checked_add(1).context("diagnostic counter overflow")?;
    let math = serde_json::to_value(pin.check()?)?;
    check_math(&math, runner.bundle)?;
    let members: Vec<Value> = tokens
        .iter()
        .enumerate()
        .map(|(position, ids)| {
            json!({
                "physical_row": position,
                "fixture_index": indices[position],
                "valid_length": ids.len(),
                "token_sha256": ids_sha(ids)
            })
        })
        .collect();
    let begin = json!({
        "kind": "controlled_padding_begin",
        "route": "Model::forward_with_padded_s",
        "diagnostic_id": id,
        "case_id": case,
        "B": indices.len(),
        "S": s,
        "projection_rows": indices.len() * s,
        "rows": members,
        "math": math,
        "serving_route": false
    });
    serde_json::to_writer(&mut *manifest, &begin)?;
    manifest.write_all(b"\n")?;
    manifest.flush()?;

    let output = model.forward_with_padded_s(&tokens, s)?;
    let math = serde_json::to_value(pin.check()?)?;
    check_math(&math, runner.bundle)?;
    ensure!(output.len() == indices.len(), "diagnostic row count");
    let mut rows = members;
    for (member, row) in rows.iter_mut().zip(&output) {
        norm(row)?;
        member["output_sha256"] = json!(sha(&row_bytes(row)));
    }
    serde_json::to_writer(
        &mut *manifest,
        &json!({
            "kind": "controlled_padding_end",
            "route": "Model::forward_with_padded_s",
            "diagnostic_id": id,
            "case_id": case,
            "B": indices.len(),
            "S": s,
            "projection_rows": indices.len() * s,
            "rows": rows,
            "math": math,
            "status": "ok",
            "serving_route": false
        }),
    )?;
    manifest.write_all(b"\n")?;
    manifest.flush()?;
    runner.inspect(
        case,
        indices,
        s,
        &output,
        "canonical",
        "Model::forward_with_padded_s",
    )?;
    Ok(output)
}

pub fn run(bundle: Bundle, out: &Path) -> Result<String> {
    let binary_before = file_sha(&std::env::current_exe()?)?;
    fs::create_dir(out).context("OUT_DIR must not already exist")?;
    let records_path = out.join("accept_records.jsonl");
    let executor_path = out.join("executor_manifest.jsonl");
    let diagnostic_path = out.join("padding_manifest.jsonl");
    let mut runner = Runner {
        bundle: &bundle,
        records: Records::new(&records_path)?,
        expected: Vec::new(),
        next_request: 1,
        history: BTreeMap::new(),
        pairwise_comparisons: 0,
    };

    {
        let mut executor = rustv1::m2_route::open_batched_with_attestation(
            &bundle.model_dir,
            &executor_path,
            0,
            &bundle.audit_identity,
            None,
            bundle.shapes.batching.clone(),
        )?;
        let tokens: Vec<Vec<u32>> = bundle.fixtures.iter().map(|f| f.ids.clone()).collect();
        let output = executor.forward_serving(&tokens)?;
        exact(&output, &bundle.singleton)?;
        for (i, row) in output.iter().enumerate() {
            runner.expected.push(ExpectedForward {
                indices: vec![i],
                associations: None,
                hashes: vec![sha(&row_bytes(row))],
            });
            runner.inspect(
                &format!("singleton-{i}"),
                &[i],
                bundle.fixtures[i].ids.len(),
                std::slice::from_ref(row),
                "singleton",
                "IndependentExecutor::forward_serving",
            )?;
        }

        for fixture in 0..22 {
            runner.repeated(
                &mut executor,
                &format!("canonical-47-{fixture}-0"),
                &[fixture, 6],
                "canonical-pair",
            )?;
            runner.repeated(
                &mut executor,
                &format!("canonical-47-{fixture}-1"),
                &[6, fixture],
                "canonical-pair",
            )?;
        }
        for shape in &bundle.shapes.cases {
            runner.repeated(&mut executor, &shape.case_id, &shape.fixtures, "shape")?;
        }
        ensure!(!executor.is_poisoned(), "pre-pass executor poisoned");
    }

    let reconciled = reconcile(&executor_path, &bundle, &runner.expected)?;
    runner.records.pass(
        "prepass-manifest-reconciliation",
        json!({
            "physical_forwards": reconciled,
            "unmatched_records": 0,
            "executor_manifest_sha256": file_sha(&executor_path)?
        }),
    )?;

    let mut diagnostics = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&diagnostic_path)?;
    let mut diagnostic_count = 0u64;
    let mut padding_fixtures = Vec::new();
    {
        // The executor above is dropped before this second policy owner exists.
        // Pin precedes CUDA device/model construction, as in m2_route.
        let pin = MathModePin::pin(&bundle.audit_identity)?;
        let model = Model::load(
            &bundle.model_dir,
            Device::new_cuda(0)?,
            bundle.engine.clone(),
            DType::BF16,
        )?
        .with_precision(PrecisionProfile::F);
        check_math(&serde_json::to_value(pin.check()?)?, &bundle)?;

        for fixture in 0..22 {
            let length = bundle.fixtures[fixture].ids.len();
            if length == 1024 {
                continue;
            }
            let indices = [fixture, fixture];
            let natural = diagnostic(
                &model,
                &pin,
                &mut diagnostics,
                &mut runner,
                &format!("padding-52-{fixture}-natural"),
                &indices,
                length,
                &mut diagnostic_count,
            )?;
            let padded = diagnostic(
                &model,
                &pin,
                &mut diagnostics,
                &mut runner,
                &format!("padding-52-{fixture}-wide"),
                &indices,
                1024,
                &mut diagnostic_count,
            )?;
            let immediate = diagnostic(
                &model,
                &pin,
                &mut diagnostics,
                &mut runner,
                &format!("padding-52-{fixture}-immediate"),
                &indices,
                1024,
                &mut diagnostic_count,
            )?;
            exact(&padded, &immediate)?;
            diagnostic(
                &model,
                &pin,
                &mut diagnostics,
                &mut runner,
                &format!("padding-52-{fixture}-interleave"),
                &[0, 0],
                1,
                &mut diagnostic_count,
            )?;
            let later = diagnostic(
                &model,
                &pin,
                &mut diagnostics,
                &mut runner,
                &format!("padding-52-{fixture}-later"),
                &indices,
                1024,
                &mut diagnostic_count,
            )?;
            exact(&padded, &later)?;
            let pairwise: Vec<Value> = natural
                .iter()
                .zip(&padded)
                .map(|(a, b)| compare(a, b))
                .collect::<Result<_>>()?;
            runner.records.pass(
                &format!("padding-52-{fixture}"),
                json!({
                    "family": 52,
                    "fixture_index": fixture,
                    "B": 2,
                    "widths": [length, 1024],
                    "direct_width_comparisons": pairwise,
                    "immediate_exact": true,
                    "after_other_shape_exact": true,
                    "serving_route": false
                }),
            )?;
            padding_fixtures.push(fixture);
        }
        ensure!(!padding_fixtures.is_empty(), "no padding-width diagnostics");
        check_math(&serde_json::to_value(pin.check()?)?, &bundle)?;
    }
    diagnostics.sync_all()?;
    runner.records.finish()?;

    bundle.recheck_bindings()?;
    ensure!(
        binary_before == file_sha(&std::env::current_exe()?)?,
        "driver binary changed during execution"
    );

    let artifact_hashes: BTreeMap<&str, &str> = bundle
        .artifacts
        .iter()
        .map(|(role, artifact)| (role.as_str(), artifact.sha256.as_str()))
        .collect();
    let evidence = json!({
        "schema": "rustv1.batch_in_process_evidence.v1",
        "producer": {
            "component": "accept_m4",
            "command": "prepass",
            "binary_sha256": binary_before
        },
        "scope": "controlled_numerical_families_47_through_52",
        "execution_policy": BATCHED_EXECUTION_POLICY,
        "profile": "F",
        "width": WIDTH,
        "input_sha256": bundle.input_sha256,
        "artifact_sha256": artifact_hashes,
        "audit_identity": bundle.audit_identity,
        "batching": bundle.shapes.batching,
        "reference_row_order": "fixture_index",
        "device_ordinal": 0,
        "build_identity": {
            "rustc_version": env!("RV1_BUILD_RUSTC_VERSION"),
            "nvcc_version": env!("RV1_BUILD_NVCC_VERSION"),
            "cuda_compute_cap": env!("RV1_BUILD_CUDA_COMPUTE_CAP"),
            "target_features": env!("RV1_BUILD_TARGET_FEATURES")
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": std::env::var("CUDA_VISIBLE_DEVICES").ok(),
            "CUBLAS_WORKSPACE_CONFIG": std::env::var("CUBLAS_WORKSPACE_CONFIG").ok(),
            "NVIDIA_TF32_OVERRIDE": std::env::var("NVIDIA_TF32_OVERRIDE").ok()
        },
        "completed_families": [47, 48, 49, 50, 51, 52],
        "frozen_singleton_rows_checked": 22,
        "canonical_target_cases": 44,
        "declared_shape_cases": bundle.shapes.cases.len(),
        "padding_fixture_indices": padding_fixtures,
        "executor_physical_forwards": reconciled,
        "padding_diagnostic_forwards": diagnostic_count,
        "direct_pairwise_comparisons": runner.pairwise_comparisons,
        "records": {
            "path": "accept_records.jsonl",
            "sha256": file_sha(&records_path)?,
            "count": runner.records.count
        },
        "executor_manifest": {
            "path": "executor_manifest.jsonl",
            "sha256": file_sha(&executor_path)?
        },
        "padding_manifest": {
            "path": "padding_manifest.jsonl",
            "sha256": file_sha(&diagnostic_path)?
        },
        "candidate_selection": {
            "declared_shapes": "shape_specific",
            "canonical_targets": "shape_specific",
            "unretained_canonical_companions": "canonical_same_fixture_same_position",
            "padding_diagnostics": "canonical_same_fixture_position_modulo_two"
        },
        "gates": {
            "all_outputs_width_finite_fp64_norm": true,
            "physical_b1_exact": true,
            "actual_to_candidate_direct": true,
            "actual_to_singleton_direct": true,
            "actual_to_independent_reference_direct": true,
            "identical_layout_immediate_exact": true,
            "identical_layout_interleaved_exact": true,
            "direct_pairwise_invariance": true,
            "executor_manifest_reconciled": true
        },
        "pending_required_cases_in_scope": [],
        "exceptions": [],
        "pass": true,
        "http_certified": false,
        "m4_accepted": false
    });
    let bytes = serde_json::to_vec_pretty(&evidence)?;
    // Completion marker is written only after every gate and binding recheck.
    write_new(&out.join("batch_in_process_evidence.json"), &bytes)?;
    Ok(sha(&bytes))
}
