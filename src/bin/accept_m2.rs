use anyhow::{ensure, Context, Result};
use rustv1::model::IndependentExecutor;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Component, Path, PathBuf},
};

const WIDTH: usize = 2048;
const SAMPLES: usize = 22;
const AUDIT_PATH: &str =
    "audits/cuda_math/candle-0.9.2_cudarc-0.19.10.v1.json";

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_sha256(path: &Path) -> Result<String> {
    Ok(sha256(&fs::read(path)
        .with_context(|| format!("read {}", path.display()))?))
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(
        &fs::read(path)
            .with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))
}

fn emit(value: Value) {
    println!("{value}");
}

/// One record per executed test. Details are nested so callers cannot
/// accidentally replace kind/sample/pass.
fn test<T>(
    kind: &str,
    sample: Value,
    body: impl FnOnce() -> Result<(T, Value)>,
) -> Result<T> {
    match body() {
        Ok((output, details)) => {
            emit(json!({
                "kind": kind,
                "sample": sample,
                "pass": true,
                "details": details
            }));
            Ok(output)
        }
        Err(error) => {
            emit(json!({
                "kind": kind,
                "sample": sample,
                "pass": false,
                "error": format!("{error:#}")
            }));
            Err(error)
        }
    }
}

fn validate_row(row: &[f32], label: &str) -> Result<f64> {
    ensure!(row.len() == WIDTH, "{label}: wrong embedding width");
    ensure!(
        row.iter().all(|x| x.is_finite()),
        "{label}: nonfinite embedding"
    );
    let norm = row.iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        norm.is_finite() && (norm - 1.0).abs() <= 1e-5,
        "{label}: invalid FP64 norm {norm}"
    );
    Ok(norm)
}

// Verbatim helper from consultation #12.
fn require_bit_identity(
    actual: &[f32],
    expected: &[f32],
    label: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        actual.len() == 2048 && expected.len() == 2048,
        "{label}: wrong embedding width"
    );
    anyhow::ensure!(
        actual.iter().chain(expected).all(|x| x.is_finite()),
        "{label}: nonfinite embedding"
    );
    anyhow::ensure!(
        actual.iter().zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{label}: independent-sequence output changed bits"
    );
    Ok(())
}

fn compare(
    actual: &[f32],
    expected: &[f32],
    label: &str,
    cosine_floor: f64,
    max_abs_limit: f64,
    require_bits: bool,
) -> Result<Value> {
    let actual_norm = validate_row(actual, label)?;
    let expected_norm = validate_row(expected, label)?;

    // validate_row has checked both widths; zip cannot truncate a mismatch.
    let mut dot = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&a, &b) in actual.iter().zip(expected) {
        let a = f64::from(a);
        let b = f64::from(b);
        dot += a * b;
        max_abs = max_abs.max((a - b).abs());
    }

    let cosine = dot / (actual_norm * expected_norm);
    ensure!(
        cosine.is_finite() && max_abs.is_finite(),
        "{label}: nonfinite comparison metrics"
    );
    ensure!(
        cosine > cosine_floor && max_abs <= max_abs_limit,
        "{label}: cosine={cosine:.12}, max_abs={max_abs:.12}; \
         require cosine>{cosine_floor}, max_abs<={max_abs_limit}"
    );

    if require_bits {
        require_bit_identity(actual, expected, label)?;
    }

    Ok(json!({
        "cosine": cosine,
        "max_abs": max_abs,
        "actual_norm_fp64": actual_norm,
        "expected_norm_fp64": expected_norm,
        "bit_identity_required": require_bits,
        "bit_identical": actual.iter().zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    }))
}

fn load_reference(path: &Path) -> Result<Vec<Vec<f32>>> {
    let bytes = fs::read(path)?;
    ensure!(
        bytes.len() == SAMPLES * WIDTH * 4,
        "reference must contain exactly 22*2048 little-endian f32 values"
    );

    let values: Vec<f32> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let rows: Vec<Vec<f32>> = values.chunks_exact(WIDTH)
        .map(|row| row.to_vec())
        .collect();

    ensure!(rows.len() == SAMPLES, "reference cardinality");
    for (i, row) in rows.iter().enumerate() {
        validate_row(row, &format!("reference {i}"))?;
    }
    Ok(rows)
}

/// Verify the artifact identity supplied by the environment.
///
/// This checks the preserved artifact's bytes, schema identifier, dependency
/// versions and Cargo.lock binding. It does not independently establish that
/// the human source-audit conclusions are correct, or rehash registry sources.
fn audit_identity() -> Result<(String, Value)> {
    let identity = std::env::var("RV1_CUDA_MATH_AUDIT")
        .context("RV1_CUDA_MATH_AUDIT is required")?;
    let mut fields = identity.splitn(3, ':');
    ensure!(fields.next() == Some("sha256"), "invalid audit scheme");
    let digest = fields.next().context("missing audit digest")?;
    let relative = fields.next().context("missing audit path")?;

    ensure!(
        digest.len() == 64
            && digest.bytes().all(|b|
                b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "audit digest must be 64 lowercase hexadecimal characters"
    );
    ensure!(relative == AUDIT_PATH, "unexpected audit artifact path");

    let path = Path::new(relative);
    ensure!(
        !path.is_absolute()
            && path.components().all(|c| matches!(c, Component::Normal(_))),
        "audit path must be repository-relative without traversal"
    );

    let bytes = fs::read(path)?;
    ensure!(sha256(&bytes) == digest, "audit artifact SHA-256 mismatch");
    let audit: Value = serde_json::from_slice(&bytes)?;

    ensure!(
        audit["schema"].as_str() == Some("rustv1.cuda_math_source_audit.v1"),
        "wrong audit schema"
    );
    ensure!(
        audit["crates"]["candle-core"]["version"].as_str() == Some("0.9.2"),
        "audit candle-core version mismatch"
    );
    ensure!(
        audit["crates"]["cudarc"]["version"].as_str() == Some("0.19.10"),
        "audit cudarc version mismatch"
    );

    let lock_hash = file_sha256(Path::new("Cargo.lock"))?;
    ensure!(
        audit["cargo_lock_sha256"].as_str() == Some(lock_hash.as_str()),
        "audit Cargo.lock binding mismatch"
    );

    Ok((identity.clone(), json!({
        "identity": identity,
        "artifact_sha256": digest,
        "cargo_lock_sha256": lock_hash,
        "source_audit_conclusions": "caller_attested_not_rederived_by_driver"
    })))
}

fn invariant_request(
    executor: &mut IndependentExecutor,
    prompts: &[Vec<u32>],
    baselines: &[Vec<f32>],
    indices: &[usize],
    kind: &str,
    cosine_floor: f64,
) -> Result<()> {
    test(kind, json!(indices), || {
        ensure!(!indices.is_empty(), "empty invariant test");
        let sequences: Vec<Vec<u32>> = indices.iter()
            .map(|&i| prompts[i].clone())
            .collect();

        let output = executor.forward_serving(&sequences)?;
        ensure!(
            output.len() == indices.len(),
            "{kind}: wrong output cardinality"
        );

        let mut rows = Vec::with_capacity(indices.len());
        for (position, &sample) in indices.iter().enumerate() {
            let metrics = compare(
                &output[position],
                &baselines[sample],
                &format!("{kind}: position {position}, sample {sample}"),
                cosine_floor,
                0.005,
                true,
            )?;
            rows.push(json!({
                "position": position,
                "sample": sample,
                "metrics": metrics
            }));
        }

        Ok(((), json!({
            "logical_input_count": sequences.len(),
            "physical_execution": "one_B1_forward_per_sequence",
            "rows": rows
        })))
    })
}

fn invalid_then_recover(
    executor: &mut IndependentExecutor,
    manifest_path: &Path,
    prompts: &[Vec<u32>],
    baselines: &[Vec<f32>],
    valid_sample: usize,
    name: &str,
    invalid: Vec<Vec<u32>>,
) -> Result<()> {
    test("invalid_request", json!(name), || {
        let before = fs::read(manifest_path)?;
        let error = executor.forward_serving(&invalid)
            .err()
            .context("invalid request unexpectedly succeeded")?;

        ensure!(
            !executor.is_poisoned(),
            "{name}: input validation poisoned the executor"
        );
        ensure!(
            fs::read(manifest_path)? == before,
            "{name}: invalid request launched or recorded physical work"
        );

        Ok(((), json!({
            "rejection": format!("{error:#}"),
            "poisoned": false,
            "manifest_unchanged": true
        })))
    })?;

    invariant_request(
        executor,
        prompts,
        baselines,
        &[valid_sample],
        &format!("recovery_after_{name}"),
        0.9999,
    )
}

/// Public so parity_run can execute exactly the same acceptance suite.
/// The caller retains its existing artifact/exception binding checks.
/// This function emits test records but not the final "done" record.
pub fn run_suite(
    model_dir: &Path,
    manifest_path: &Path,
    policy: &Value,
) -> Result<()> {
    let prompts_path = Path::new("results/prompts.json");
    let reference_path = Path::new("results/ref_single.f32");

    let (prompts, reference, short, long, vocab_size, audit) =
        test("preflight", Value::Null, || {
            // Exact empty-list assertion required by consultation #12.
            anyhow::ensure!(
                policy["samples"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!(
                        "policy samples must be an array"
                    ))?
                    .is_empty(),
                "M2 independent-F permits no active exceptions"
            );

            let prompts: Vec<Vec<u32>> =
                serde_json::from_slice(&fs::read(prompts_path)?)?;
            ensure!(prompts.len() == SAMPLES, "expected exactly 22 prompts");

            let engine = read_json(&model_dir.join("engine.json"))?;
            ensure!(
                engine["max_len"].as_u64() == Some(1024),
                "this corpus requires engine.max_len=1024"
            );

            let config = read_json(&model_dir.join("config.json"))?;
            let vocab_size = u32::try_from(
                config["vocab_size"].as_u64().context("config vocab_size")?
            ).context("vocab_size cannot be represented as an OOV u32")?;
            ensure!(vocab_size > 11, "invalid vocabulary size");

            for (i, prompt) in prompts.iter().enumerate() {
                ensure!(
                    !prompt.is_empty() && prompt.len() <= 1024,
                    "sample {i}: invalid length"
                );
                ensure!(
                    prompt.iter().all(|&id| id != 11 && id < vocab_size),
                    "sample {i}: invalid token"
                );
            }

            let short = prompts.iter().enumerate()
                .min_by_key(|(_, p)| p.len())
                .map(|(i, _)| i)
                .context("missing short sample")?;
            ensure!(prompts[short].len() <= 16, "missing short sample");
            let long = prompts.iter()
                .position(|p| p.len() == 1024)
                .context("missing length-1024 sample")?;

            let reference = load_reference(reference_path)?;
            let (audit, audit_details) = audit_identity()?;

            let details = json!({
                "scope": "M2_in_process_token_ID_API",
                "http_certified": false,
                "sample_count": SAMPLES,
                "short_sample": short,
                "length_1024_sample": long,
                "profile": "F",
                "exception_samples": [],
                "prompts_sha256": file_sha256(prompts_path)?,
                "reference_sha256": file_sha256(reference_path)?,
                "config_sha256": file_sha256(&model_dir.join("config.json"))?,
                "engine_sha256": file_sha256(&model_dir.join("engine.json"))?,
                "audit": audit_details
            });

            Ok((
                (prompts, reference, short, long, vocab_size, audit),
                details,
            ))
        })?;

    // Exactly one live executor is created for all eight closure groups.
    let mut executor = test("open_route", Value::Null, || {
        let executor = rustv1::m2_route::open(
            model_dir,
            manifest_path,
            0,
            &audit,
        )?;
        Ok((executor, json!({
            "device_ordinal": 0,
            "route": "rustv1::m2_route::open",
            "manifest_path": manifest_path
        })))
    })?;

    // Group 1: genuine singleton baselines, all 22 against the frozen reference.
    let mut baselines = Vec::with_capacity(SAMPLES);
    for i in 0..SAMPLES {
        let row = test("singleton_reference", json!(i), || {
            let output = executor.forward_serving(
                std::slice::from_ref(&prompts[i])
            )?;
            ensure!(output.len() == 1, "singleton cardinality");
            let metrics = compare(
                &output[0],
                &reference[i],
                &format!("singleton {i}"),
                0.999,
                0.01,
                false,
            )?;
            Ok((output[0].clone(), metrics))
        })?;
        baselines.push(row);
    }

    // Group 2: cover every sample in multi-input logical requests.
    // These are NOT padded GPU batches. #12 admits logical input arrays
    // independently of the physical B=1 execution shape.
    let natural: Vec<usize> = (0..SAMPLES).collect();
    for group in natural.chunks(4) {
        invariant_request(
            &mut executor, &prompts, &baselines, group,
            "multi_input", 0.9999,
        )?;
    }
    invariant_request(
        &mut executor, &prompts, &baselines, &[short, long],
        "multi_input_short_1024", 0.9999,
    )?;

    // Group 3: reverse and deterministic Fisher-Yates shuffle.
    let reversed: Vec<usize> = natural.iter().copied().rev().collect();
    invariant_request(
        &mut executor, &prompts, &baselines, &reversed,
        "reversed", 0.9999,
    )?;

    let mut shuffled = natural.clone();
    let mut state = 0x5256_314d_3253_4855u64;
    for i in (1..shuffled.len()).rev() {
        state = state.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state % ((i + 1) as u64)) as usize;
        shuffled.swap(i, j);
    }
    invariant_request(
        &mut executor, &prompts, &baselines, &shuffled,
        "deterministic_shuffle", 0.9999,
    )?;

    // Group 4: duplicates must equal both each other and their singleton.
    invariant_request(
        &mut executor, &prompts, &baselines,
        &[short, long, short, long],
        "duplicates", 0.9999,
    )?;

    // Group 5: repeat exactly the same input array and physical shapes.
    // All runs are checked against the singleton bits, hence transitively
    // against each other, retaining the stricter repeated-execution cosine.
    for repeat in 0..3 {
        invariant_request(
            &mut executor, &prompts, &baselines, &[short, long],
            &format!("repeat_identical_shape_{repeat}"), 0.999999,
        )?;
    }

    // Group 6: changing length across separate logical requests.
    for (step, sample) in [short, long, short, long, short, long]
        .into_iter().enumerate()
    {
        invariant_request(
            &mut executor, &prompts, &baselines, &[sample],
            &format!("alternating_length_{step}"), 0.9999,
        )?;
    }

    // Group 7: each invalid request is followed immediately by a valid one.
    // A valid prefix additionally proves whole-request prevalidation.
    let valid = prompts[short].clone();
    let ordinary_id = valid[0];
    let invalid_cases = vec![
        ("empty_batch", Vec::<Vec<u32>>::new()),
        ("empty_sequence", vec![valid.clone(), Vec::new()]),
        ("pad_token", vec![valid.clone(), vec![11]]),
        ("over_length", vec![valid.clone(), vec![ordinary_id; 1025]]),
        ("out_of_vocabulary", vec![valid.clone(), vec![vocab_size]]),
    ];

    for (name, invalid) in invalid_cases {
        invalid_then_recover(
            &mut executor,
            manifest_path,
            &prompts,
            &baselines,
            short,
            name,
            invalid,
        )?;
    }

    // Group 8 MUST be last: this deliberately destroys executor usability.
    test("operational_error_poison", json!(short), || {
        use candle_core::cuda_backend::{
            gemm_reduced_precision_f32,
            set_gemm_reduced_precision_f32,
        };

        ensure!(!executor.is_poisoned(), "executor already poisoned");
        ensure!(
            !gemm_reduced_precision_f32(),
            "unexpected pre-injection F32 setting"
        );

        let before = fs::read(manifest_path)?;

        struct RestoreF32;
        impl Drop for RestoreF32 {
            fn drop(&mut self) {
                candle_core::cuda_backend::set_gemm_reduced_precision_f32(false);
            }
        }

        let error = {
            let _restore = RestoreF32;
            set_gemm_reduced_precision_f32(true);

            // Deterministic under the required single-thread/no-other-writers
            // contract: the setter changes the exact process-global flag read
            // synchronously by MathModePin::check. forward_serving marks the
            // executor poisoned BEFORE that check. No GPU fault, allocation
            // failure, race, or actual reduced-precision GEMM is needed.
            executor.forward_serving(std::slice::from_ref(&valid))
                .err()
                .context("injected policy drift unexpectedly succeeded")?
        }; // Restore the flag, but do NOT clear either sticky failure state.

        ensure!(
            format!("{error:#}").contains("Candle GEMM policy drift"),
            "wrong injected failure: {error:#}"
        );
        ensure!(executor.is_poisoned(), "operational error did not poison");
        ensure!(
            !gemm_reduced_precision_f32(),
            "injection cleanup failed"
        );
        ensure!(
            fs::read(manifest_path)? == before,
            "policy drift should be detected before physical-forward begin"
        );

        Ok(((), json!({
            "error": format!("{error:#}"),
            "poisoned": true,
            "global_flag_restored": true,
            "manifest_unchanged": true
        })))
    })?;

    test("poisoned_request_refused", json!(short), || {
        let before = fs::read(manifest_path)?;
        let error = executor.forward_serving(std::slice::from_ref(&valid))
            .err()
            .context("poisoned executor accepted a request")?;
        ensure!(
            format!("{error:#}").contains("executor is poisoned"),
            "subsequent request did not fail at the poison guard: {error:#}"
        );
        ensure!(executor.is_poisoned(), "poison state cleared");
        ensure!(
            fs::read(manifest_path)? == before,
            "poisoned request performed physical work"
        );

        Ok(((), json!({
            "error": format!("{error:#}"),
            "poisoned": true,
            "manifest_unchanged": true
        })))
    })?;

    // Success here includes the EXPECTED final poisoned state.
    drop(executor);
    Ok(())
}

fn entry() -> Result<()> {
    let (model_dir, manifest_path, policy) =
        test("arguments_and_policy", Value::Null, || {
            let args: Vec<String> = std::env::args().collect();
            ensure!(
                args.len() == 4,
                "usage: accept_m2 MODEL_DIR NEW_EXECUTOR_MANIFEST_JSONL \
                 M2_EXCEPTION_POLICY_JSON"
            );

            let model_dir = PathBuf::from(&args[1]);
            let manifest_path = PathBuf::from(&args[2]);
            let policy_path = PathBuf::from(&args[3]);
            let policy = read_json(&policy_path)?;

            Ok((
                (model_dir, manifest_path, policy),
                json!({
                    "policy_path": policy_path,
                    "policy_file_sha256": file_sha256(&policy_path)?
                }),
            ))
        })?;

    run_suite(&model_dir, &manifest_path, &policy)
}

fn main() {
    let result = entry();
    emit(json!({
        "kind": "done",
        "sample": null,
        "pass": result.is_ok(),
        "accepted": result.is_ok()
    }));

    if let Err(error) = result {
        eprintln!("M2 acceptance failed: {error:#}");
        std::process::exit(1);
    }
}
