//! Batch-route candidate golden exporter.
//!
//! Usage:
//!   golden_export_m4 MODEL_DIR FIXTURES_JSON SHAPES_JSON OUT_DIR AUDIT_IDENTITY
//!
//! FIXTURES_JSON is the existing 22-entry executable fixture array:
//!   id, text, ids, ids_sha256, golden_row, golden_sha256
//!
//! SHAPES_JSON:
//! {
//!   "schema": "rustv1.batch_shapes.v1",
//!   "batching": {
//!     "max_batch_size": 32,
//!     "max_padded_tokens": 32768,
//!     "max_attention_cells": 33554432,
//!     "max_wait_us": 1000
//!   },
//!   "cases": [
//!     {"case_id":"homogeneous-short-b4","fixtures":[0,0,0,0],"S":1},
//!     {"case_id":"mixed-boundaries","fixtures":[0,1,2,3,4,5,6],"S":1024}
//!   ]
//! }
//!
//! `fixtures` contains zero-based fixture indices in physical row order.
//! Each declared case exports every physical row.
//! S must equal the maximum valid length. No bucket rounding, dummy rows,
//! input truncation, singleton-loop fallback or capacity retry is allowed.
//! Enlarged-S padding diagnostics are NOT this route and are rejected here.
//!
//! Additionally, always exports exactly 44 canonical target rows:
//!   [Fi,F06], retaining physical row 0;
//!   [F06,Fi], retaining physical row 1;
//! for each fixture Fi in fixture order.
//!
//! Before batch export, forward_serving checks every fixture against its
//! frozen singleton SHA256. Existing singleton matrices are never modified.
//! Every batch is repeated immediately and must match exactly. Every batch
//! output is also compared numerically directly against its checked singleton.
//! These are export safeguards, NOT full M4 acceptance: this tool has no
//! independent-reference input and does not certify HTTP or scheduler behavior.
//!
//! OUT_DIR must not already exist. Output:
//!   export_provenance.json
//!   executor_manifest.jsonl
//!   batch_goldens.f32             packed row-major little-endian F32
//!   batch_golden_index.json       written LAST; export completion marker
//!
//! A failed export leaves an incomplete directory for diagnosis. Do not
//! materialize a policy from that directory. There is no overwrite/resume mode.
//!
//! Audit identity uses the existing convention:
//!   sha256:<64 lowercase hex>:<repo-relative audit path>
//!
//! This auto-discovered Cargo binary needs no Cargo.toml changes. Without
//! --features cuda it builds a diagnostic entry point that exits nonzero.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("golden_export_m4 requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() {
    if let Err(error) = cuda::run() {
        eprintln!("golden_export_m4 failed: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use anyhow::{ensure, Context, Result};
    use rustv1::model::{BatchedExecution, EngineConfig, ExecutionRow, BATCHED_EXECUTION_POLICY};
    use serde::Deserialize;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs::{self, File, OpenOptions},
        io::{Read, Write},
        path::{Component, Path, PathBuf},
    };

    const WIDTH: usize = 2048;
    const MAX_JSON: usize = 8 * 1024 * 1024;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fixture {
        id: String,
        text: String,
        ids: Vec<u32>,
        ids_sha256: String,
        golden_row: usize,
        golden_sha256: String,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Shape {
        case_id: String,
        fixtures: Vec<usize>,
        #[serde(rename = "S")]
        s: usize,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Plan {
        schema: String,
        batching: BatchedExecution,
        cases: Vec<Shape>,
    }

    fn sha(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn file_sha(path: &Path) -> Result<String> {
        let mut file = File::open(path)?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 128 * 1024];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        Ok(format!("{:x}", hash.finalize()))
    }

    fn read_json_bytes(path: &Path) -> Result<Vec<u8>> {
        let file = File::open(path)?;
        ensure!(file.metadata()?.len() <= MAX_JSON as u64, "JSON too large");
        let mut bytes = Vec::new();
        file.take(MAX_JSON as u64 + 1).read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= MAX_JSON, "JSON grew beyond limit");
        Ok(bytes)
    }

    fn digest_valid(s: &str) -> bool {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    fn label_valid(s: &str) -> bool {
        !s.is_empty() && s.len() <= 128 && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
    }

    fn unresolved(value: &Value) -> bool {
        match value {
            Value::String(s) => [
                "RUNTIME",
                "FILLED_BY_INTEGRATOR",
                "<SHA256:",
                "PENDING_ROUNDTRIP",
            ]
            .iter()
            .any(|needle| s.contains(needle)),
            Value::Array(values) => values.iter().any(unresolved),
            Value::Object(values) => values
                .iter()
                .any(|(key, value)| key.contains("RUNTIME") || unresolved(value)),
            _ => false,
        }
    }

    fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    }

    fn row_bytes(row: &[f32]) -> Vec<u8> {
        row.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn ids_sha(ids: &[u32]) -> String {
        sha(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
    }

    fn norm(row: &[f32]) -> Result<f64> {
        ensure!(row.len() == WIDTH, "embedding width");
        ensure!(row.iter().all(|x| x.is_finite()), "nonfinite embedding");
        let n = row
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt();
        ensure!(
            n.is_finite() && (n - 1.0).abs() <= 1e-5,
            "FP64 norm gate: {n}"
        );
        Ok(n)
    }

    fn numerical(actual: &[f32], expected: &[f32]) -> Result<Value> {
        let an = norm(actual)?;
        let en = norm(expected)?;
        let dot = actual
            .iter()
            .zip(expected)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>();
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .fold(0.0f64, f64::max);
        let cosine = dot / (an * en);
        ensure!(
            cosine.is_finite() && cosine > 0.999 && max_abs <= 0.01,
            "singleton comparison: cosine={cosine}, max_abs={max_abs}"
        );
        Ok(json!({"cosine": cosine, "max_abs": max_abs}))
    }

    fn exact(actual: &[Vec<f32>], expected: &[Vec<f32>]) -> Result<()> {
        ensure!(actual.len() == expected.len(), "repeat row count");
        for (a, b) in actual.iter().zip(expected) {
            norm(a)?;
            norm(b)?;
            ensure!(
                a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                "identical-layout repeat differs"
            );
        }
        Ok(())
    }

    fn verify_audit(
        identity: &str,
        batching: &BatchedExecution,
        engine: &EngineConfig,
    ) -> Result<(PathBuf, String)> {
        let mut parts = identity.splitn(3, ':');
        ensure!(parts.next() == Some("sha256"), "audit identity prefix");
        let expected = parts.next().context("audit digest")?;
        let relative = parts.next().context("audit path")?;
        ensure!(digest_valid(expected), "audit digest format");

        let relative_path = Path::new(relative);
        ensure!(
            !relative_path.as_os_str().is_empty()
                && relative_path
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
            "audit path must be repo-relative without traversal"
        );
        let root = std::env::current_dir()?.canonicalize()?;
        let path = root.join(relative_path).canonicalize()?;
        ensure!(path.starts_with(&root), "audit escapes repository");

        let bytes = read_json_bytes(&path)?;
        ensure!(sha(&bytes) == expected, "audit digest mismatch");
        let audit: Value = serde_json::from_slice(&bytes)?;
        ensure!(!unresolved(&audit), "unresolved audit placeholders");
        ensure!(
            audit["schema"] == "rustv1.cuda_math_source_audit.v1"
                && audit["status"] == "accepted"
                && audit["signed"] == true
                && audit["verification"]["mechanical_region_boundary_verification_completed"]
                    == true
                && audit["contract_id"] == "b200-independent-F-v2"
                && audit["intended_path"] == relative
                && audit["scope"]["execution_policy"] == BATCHED_EXECUTION_POLICY
                && audit["scope"]["precision_profile"] == "F"
                && audit["scope"]["maximum_physical_b"]
                    .as_u64()
                    .is_some_and(|b| b >= batching.max_batch_size as u64)
                && audit["scope"]["sequence_length_max"]
                    .as_u64()
                    .is_some_and(|s| s >= engine.max_len as u64)
                && audit["scope"]["physical_sequence_batch_size"] != 1,
            "audit does not cover the configured batched route"
        );
        ensure!(
            audit["cargo_lock_sha256"] == file_sha(&root.join("Cargo.lock"))?,
            "audit lockfile binding"
        );
        // Trust is the externally designated audit digest. `signed: true`
        // is an artifact field, not cryptographic signature verification here.
        Ok((path, expected.to_owned()))
    }

    fn validate_shape(
        shape: &Shape,
        fixtures: &[Fixture],
        policy: &BatchedExecution,
    ) -> Result<()> {
        ensure!(label_valid(&shape.case_id), "invalid case ID");
        ensure!(
            !shape.case_id.starts_with("canonical-"),
            "canonical- prefix is reserved"
        );
        ensure!(
            !shape.fixtures.is_empty() && shape.fixtures.len() <= 32,
            "invalid physical B"
        );
        let mut maximum = 0;
        for &index in &shape.fixtures {
            let fixture = fixtures.get(index).context("shape fixture index")?;
            maximum = maximum.max(fixture.ids.len());
        }
        ensure!(
            shape.s == maximum,
            "{}: serving-route S must equal maximum valid length",
            shape.case_id
        );
        ensure!(
            policy.fits(shape.fixtures.len(), shape.s),
            "{}: shape exceeds capacity",
            shape.case_id
        );
        Ok(())
    }

    struct Export {
        matrix: File,
        matrix_hash: Sha256,
        rows: Vec<Value>,
        next_request: u64,
    }

    impl Export {
        fn batch(
            &mut self,
            executor: &mut rustv1::model::IndependentExecutor,
            fixtures: &[Fixture],
            singleton: &[Vec<f32>],
            case_id: &str,
            indices: &[usize],
            retain: &[usize],
        ) -> Result<()> {
            let sequences: Vec<Vec<u32>> =
                indices.iter().map(|&i| fixtures[i].ids.clone()).collect();
            let lengths: Vec<usize> = sequences.iter().map(Vec::len).collect();
            let digests: Vec<String> = sequences.iter().map(|ids| ids_sha(ids)).collect();
            let s = *lengths.iter().max().context("empty shape")?;

            let first_id = self.next_request;
            let second_id = first_id
                .checked_add(1)
                .context("request counter overflow")?;
            self.next_request = second_id
                .checked_add(1)
                .context("request counter overflow")?;

            let associations = |request_id| {
                (0..indices.len())
                    .map(|request_row| ExecutionRow {
                        request_id,
                        request_row,
                        case_id: Some(case_id.to_owned()),
                    })
                    .collect::<Vec<_>>()
            };

            let first = executor.forward_batch(&sequences, &associations(first_id))?;
            let second = executor.forward_batch(&sequences, &associations(second_id))?;
            ensure!(first.len() == indices.len(), "batch row count");
            exact(&first, &second).with_context(|| format!("repeat {case_id}"))?;

            let mut comparisons = Vec::with_capacity(first.len());
            for (row, &fixture_index) in first.iter().zip(indices) {
                let comparison = numerical(row, &singleton[fixture_index])
                    .with_context(|| format!("case {case_id}, fixture {fixture_index}"))?;
                if indices.len() == 1 {
                    ensure!(
                        sha(&row_bytes(row)) == fixtures[fixture_index].golden_sha256,
                        "physical B=1 must retain frozen exact bits"
                    );
                }
                comparisons.push(comparison);
            }

            for &physical_row in retain {
                let row = first.get(physical_row).context("retained row index")?;
                let fixture_index = indices[physical_row];
                let bytes = row_bytes(row);
                let matrix_row = self.rows.len();
                self.matrix.write_all(&bytes)?;
                self.matrix_hash.update(&bytes);
                self.rows.push(json!({
                    "matrix_row": matrix_row,
                    "byte_offset": matrix_row * WIDTH * 4,
                    "byte_length": WIDTH * 4,
                    "case_id": case_id,
                    "fixture_index": fixture_index,
                    "fixture_id": fixtures[fixture_index].id,
                    "execution_policy": BATCHED_EXECUTION_POLICY,
                    "route": "IndependentExecutor::forward_batch",
                    "ordered_fixture_indices": indices,
                    "ordered_token_digests": digests,
                    "valid_lengths": lengths,
                    "B": indices.len(),
                    "S": s,
                    "physical_row": physical_row,
                    "output_sha256": sha(&bytes),
                    "singleton_comparison": comparisons[physical_row],
                    "identical_layout_repeat_exact": true,
                    "manifest_request_ids": [first_id, second_id]
                }));
            }
            Ok(())
        }
    }

    pub fn run() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        ensure!(
            args.len() == 6,
            "usage: golden_export_m4 MODEL_DIR FIXTURES_JSON SHAPES_JSON OUT_DIR AUDIT_IDENTITY"
        );
        let model_dir = Path::new(&args[1]);
        let out_dir = Path::new(&args[4]);
        let fixture_bytes = read_json_bytes(Path::new(&args[2]))?;
        let shape_bytes = read_json_bytes(Path::new(&args[3]))?;
        let plan_value: Value = serde_json::from_slice(&shape_bytes)?;
        ensure!(
            !unresolved(&plan_value),
            "unresolved shape-plan placeholders"
        );
        let fixtures: Vec<Fixture> = serde_json::from_slice(&fixture_bytes)?;
        let plan: Plan = serde_json::from_value(plan_value)?;

        ensure!(fixtures.len() == 22, "expected exactly 22 fixtures");
        ensure!(plan.schema == "rustv1.batch_shapes.v1", "shape-plan schema");
        ensure!(plan.cases.len() <= 4096, "too many declared shapes");

        let engine_bytes = read_json_bytes(&model_dir.join("engine.json"))?;
        let engine: EngineConfig = serde_json::from_slice(&engine_bytes)?;
        plan.batching.validate_engine(&engine)?;
        ensure!(
            engine.max_len == 1024,
            "qualification requires max_len=1024"
        );
        ensure!(fixtures[0].ids.len() == 1, "F00 must be the short fixture");
        ensure!(fixtures[6].ids.len() == 1024, "F06 must have length 1024");
        ensure!(plan.batching.fits(2, 1024), "canonical B=2/S=1024 must fit");

        let tokenizer_bytes = read_json_bytes(&model_dir.join("tokenizer.json"))?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        ensure!(
            tokenizer.get_padding().is_none() && tokenizer.get_truncation().is_none(),
            "tokenizer padding/truncation must be disabled"
        );

        let mut fixture_names = BTreeSet::new();
        for fixture in &fixtures {
            ensure!(fixture_names.insert(&fixture.id), "duplicate fixture ID");
            ensure!(!fixture.text.trim().is_empty(), "empty fixture text");
            ensure!(
                !fixture.ids.is_empty()
                    && fixture.ids.len() <= 1024
                    && fixture.ids.iter().all(|&id| id < 131072 && id != 11),
                "invalid fixture tokens"
            );
            ensure!(
                digest_valid(&fixture.ids_sha256) && ids_sha(&fixture.ids) == fixture.ids_sha256,
                "fixture token digest"
            );
            ensure!(
                digest_valid(&fixture.golden_sha256),
                "singleton digest format"
            );
            let encoded = tokenizer
                .encode(fixture.text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
            ensure!(
                encoded.get_ids() == fixture.ids.as_slice(),
                "fixture {} text/token mismatch",
                fixture.id
            );
        }

        let mut case_names = BTreeSet::new();
        for shape in &plan.cases {
            validate_shape(shape, &fixtures, &plan.batching)?;
            ensure!(case_names.insert(&shape.case_id), "duplicate case ID");
        }

        let (_, audit_sha) = verify_audit(&args[5], &plan.batching, &engine)?;

        let mut model_hashes = BTreeMap::new();
        let mut found_weights = false;
        for entry in fs::read_dir(model_dir)? {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("non-UTF8 model filename"))?;
            if matches!(
                name.as_str(),
                "config.json" | "engine.json" | "tokenizer.json"
            ) || name.ends_with(".safetensors")
                || name.ends_with(".safetensors.index.json")
            {
                found_weights |= name.ends_with(".safetensors");
                model_hashes.insert(name, file_sha(&entry.path())?);
            }
        }
        ensure!(found_weights, "missing checkpoint weights");
        for required in ["config.json", "engine.json", "tokenizer.json"] {
            ensure!(model_hashes.contains_key(required), "missing {required}");
        }

        let provenance = json!({
            "schema": "rustv1.batch_golden_export_provenance.v1",
            "execution_policy": BATCHED_EXECUTION_POLICY,
            "profile": "F",
            "binary_sha256": file_sha(&std::env::current_exe()?)?,
            "cargo_lock_sha256": file_sha(Path::new("Cargo.lock"))?,
            "fixtures_sha256": sha(&fixture_bytes),
            "shape_plan_sha256": sha(&shape_bytes),
            "model_files": model_hashes,
            "audit_identity": args[5],
            "audit_sha256": audit_sha,
            "batching": plan.batching,
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
            "candidate_only": true,
            "independent_reference_checked": false,
            "http_certified": false
        });
        let provenance_bytes = serde_json::to_vec_pretty(&provenance)?;

        // No existing output is overwritten. Absence of the final index means
        // the export did not finish, even when the matrix contains some rows.
        fs::create_dir(out_dir).context("OUT_DIR must not already exist")?;
        write_new(&out_dir.join("export_provenance.json"), &provenance_bytes)?;
        let manifest_path = out_dir.join("executor_manifest.jsonl");

        let mut executor = rustv1::m2_route::open_batched_with_attestation(
            model_dir,
            &manifest_path,
            0,
            &args[5],
            None,
            plan.batching.clone(),
        )?;

        let tokens: Vec<Vec<u32>> = fixtures.iter().map(|f| f.ids.clone()).collect();
        let singleton = executor.forward_serving(&tokens)?;
        ensure!(singleton.len() == fixtures.len(), "singleton cardinality");
        for (fixture, row) in fixtures.iter().zip(&singleton) {
            norm(row)?;
            ensure!(
                sha(&row_bytes(row)) == fixture.golden_sha256,
                "frozen singleton mismatch for {} (source matrix row {})",
                fixture.id,
                fixture.golden_row
            );
        }

        let matrix_path = out_dir.join("batch_goldens.f32");
        let mut export = Export {
            matrix: OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&matrix_path)?,
            matrix_hash: Sha256::new(),
            rows: Vec::new(),
            next_request: 1,
        };

        for fixture_index in 0..fixtures.len() {
            export.batch(
                &mut executor,
                &fixtures,
                &singleton,
                &format!("canonical-47-{fixture_index}-0"),
                &[fixture_index, 6],
                &[0],
            )?;
            export.batch(
                &mut executor,
                &fixtures,
                &singleton,
                &format!("canonical-47-{fixture_index}-1"),
                &[6, fixture_index],
                &[1],
            )?;
        }
        ensure!(export.rows.len() == 44, "canonical row count");

        for shape in &plan.cases {
            let retain: Vec<usize> = (0..shape.fixtures.len()).collect();
            export.batch(
                &mut executor,
                &fixtures,
                &singleton,
                &shape.case_id,
                &shape.fixtures,
                &retain,
            )?;
        }

        ensure!(!executor.is_poisoned(), "export executor poisoned");
        drop(executor);
        export.matrix.sync_all()?;

        let matrix_sha256 = format!("{:x}", export.matrix_hash.finalize());
        let row_count = export.rows.len();
        let index = json!({
            "schema": "rustv1.batch_candidate_goldens.v1",
            "execution_policy": BATCHED_EXECUTION_POLICY,
            "profile": "F",
            "width": WIDTH,
            "dtype": "F32",
            "byte_order": "little",
            "matrix_file": "batch_goldens.f32",
            "matrix_sha256": matrix_sha256,
            "matrix_bytes": row_count * WIDTH * 4,
            "row_count": row_count,
            "canonical_row_count": 44,
            "declared_case_count": plan.cases.len(),
            "fixtures_sha256": sha(&fixture_bytes),
            "shape_plan_sha256": sha(&shape_bytes),
            "export_provenance_sha256": sha(&provenance_bytes),
            "executor_manifest_sha256": file_sha(&manifest_path)?,
            "singleton_exact_sha256_checked": true,
            "singleton_files_modified": false,
            "independent_reference_checked": false,
            "candidate_only": true,
            "http_certified": false,
            "rows": export.rows
        });
        let index_bytes = serde_json::to_vec_pretty(&index)?;
        write_new(&out_dir.join("batch_golden_index.json"), &index_bytes)?;

        println!(
            "{}",
            json!({
                "kind": "batch_golden_export_complete",
                "pass": true,
                "candidate_only": true,
                "rows": row_count,
                "canonical_rows": 44,
                "matrix_sha256": matrix_sha256,
                "index_sha256": sha(&index_bytes)
            })
        );
        Ok(())
    }
}
