// src/bin/bench_m2_baseline.rs
//
// Serial token-ID API benchmark: K=1, C=1.
// Includes executor manifest durability and CPU-resident output retrieval.
// Excludes model load, tokenization, and HTTP.
//
// Reuse the CPU-only artifact/golden validator. This module's main function
// is an ordinary unused function here; it is not a second process entry point.
#[allow(dead_code)]
#[path = "accept_m3.rs"]
mod acceptance;

use anyhow::{ensure, Context, Result};
use rustv1::m2_route;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const WARMUP_PASSES: usize = 10;
const MIN_DURATION: Duration = Duration::from_secs(60);
const MIN_PASSES: usize = 100;
const RUNS: usize = 3;

fn emit(value: Value) {
    println!("{value}");
}

fn percentile(sorted: &[f64], fraction: f64) -> Result<f64> {
    ensure!(!sorted.is_empty(), "no latency observations");
    ensure!(
        sorted.iter().all(|x| x.is_finite() && *x >= 0.0),
        "invalid latency"
    );
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Ok(sorted[index])
}

fn create_new(path: &Path) -> Result<File> {
    Ok(OpenOptions::new().write(true).create_new(true).open(path)?)
}

fn write_record(file: &mut File, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn validate_output(output: &[Vec<f32>], golden: &[f32]) -> Result<()> {
    ensure!(output.len() == 1, "singleton output cardinality");
    acceptance::exact_row(&output[0], golden)
}

fn unique_directory(parent: &Path) -> Result<PathBuf> {
    fs::create_dir_all(parent)?;
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = parent.join(format!("m2-bench-{timestamp}-{}", std::process::id()));
    // No fallback name and no retry: collision fails closed.
    fs::create_dir(&path)?;
    Ok(path)
}

fn run() -> Result<Value> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 5,
        "usage: bench_m2_baseline MODEL_DIR PROMPTS_JSON MANIFEST_DIR AUDIT_IDENTITY"
    );

    let policy_path =
        PathBuf::from(std::env::var("RV1_M3_POLICY").context("RV1_M3_POLICY is required")?);
    let environment_path =
        PathBuf::from(std::env::var("RV1_BENCH_ENV").context("RV1_BENCH_ENV is required")?);
    let model_dir = PathBuf::from(&args[1]);

    // All identity, corpus, audit, reference, and golden checks precede CUDA.
    let bundle = acceptance::load_bundle(&model_dir, &policy_path, "bench_m2_baseline")?;
    ensure!(
        args[4] == bundle.policy.audit_identity,
        "CLI audit identity mismatch"
    );
    ensure!(
        acceptance::file_sha256(Path::new(&args[2]))? == bundle.policy.artifacts["corpus"].sha256,
        "CLI corpus is not the frozen corpus"
    );
    ensure!(
        environment_path.canonicalize()? == bundle.paths["benchmark_environment"].canonicalize()?,
        "benchmark environment path mismatch"
    );
    let environment: Value = serde_json::from_slice(&fs::read(&environment_path)?)?;
    ensure!(
        environment["schema"] == "rustv1.benchmark_environment.v1",
        "environment schema"
    );
    ensure!(
        environment["deployment"] == bundle.policy.deployment,
        "benchmark deployment"
    );
    ensure!(
        environment["approved_for_measurement"] == true,
        "measurement not approved"
    );
    ensure!(
        environment["no_competing_gpu_workload"] == true,
        "exclusive workload required"
    );
    ensure!(
        environment["profiler_enabled"] == false,
        "profiler must be off"
    );
    ensure!(
        environment["result_cache_enabled"] == false,
        "result cache must be off"
    );
    for field in [
        "gpu_observations",
        "cpu_configuration",
        "manifest_storage",
        "runtime_library_identity",
        "clock_power_temperature_observations",
    ] {
        ensure!(
            !environment[field].is_null(),
            "missing environment field {field}"
        );
    }

    let directory = unique_directory(Path::new(&args[3]))?;
    let manifest_path = directory.join("executor.jsonl");
    let observations_path = directory.join("observations.jsonl");
    let summaries_path = directory.join("summaries.jsonl");
    let metadata_path = directory.join("metadata.json");

    // The directory is newly created. Do not precreate the executor's file:
    // its own create-new policy, if present, must remain effective.
    ensure!(!manifest_path.exists(), "manifest unexpectedly exists");
    let mut observations = create_new(&observations_path)?;
    let mut summaries = create_new(&summaries_path)?;
    let mut metadata = create_new(&metadata_path)?;

    let metadata_value = json!({
        "schema":"rustv1.m2_baseline_metadata.v1",
        "policy_sha256":bundle.policy_sha256,
        "contract_id":bundle.policy.contract_id,
        "audit_identity":bundle.policy.audit_identity,
        "deployment":bundle.policy.deployment,
        "corpus_sha256":bundle.policy.artifacts["corpus"].sha256,
        "goldens_sha256":bundle.policy.artifacts["corpus_goldens"].sha256,
        "benchmark_environment_sha256":bundle.policy.artifacts["benchmark_environment"].sha256,
        "executable_sha256":acceptance::file_sha256(&std::env::current_exe()?)?,
        "scope":"serial_token_ID_API_CPU_resident_outputs_with_executor_durability",
        "logical_request_size":1,
        "physical_sequence_batch_size":1,
        "concurrency":1,
        "includes_tokenization":false,
        "includes_http":false,
        "warmup_passes":WARMUP_PASSES,
        "timed_runs":RUNS,
        "minimum_run_seconds":MIN_DURATION.as_secs(),
        "minimum_complete_passes":MIN_PASSES,
        "environment":environment
    });
    serde_json::to_writer_pretty(&mut metadata, &metadata_value)?;
    metadata.write_all(b"\n")?;
    metadata.sync_all()?;

    emit(json!({
        "kind":"preflight","pass":true,
        "evidence_directory":directory,
        "metadata_sha256":acceptance::file_sha256(&metadata_path)?
    }));

    // Exactly one executor. No instrumentation changes to its sync policy.
    let mut executor = m2_route::open(&model_dir, &manifest_path, 0, &bundle.policy.audit_identity)
        .context("opening designated M2 executor")?;

    for pass in 0..WARMUP_PASSES {
        for (sample, prompt) in bundle.corpus.iter().enumerate() {
            let output = executor.forward_serving(std::slice::from_ref(prompt))?;
            validate_output(&output, &bundle.corpus_goldens[sample])
                .with_context(|| format!("warmup pass {pass}, sample {sample}"))?;
        }
    }
    emit(json!({
        "kind":"warmup","pass":true,
        "complete_passes":WARMUP_PASSES,
        "completed_requests":WARMUP_PASSES * bundle.corpus.len()
    }));

    let mut rates = Vec::with_capacity(RUNS);
    let mut p99s = Vec::with_capacity(RUNS);
    let mut run_summaries = Vec::with_capacity(RUNS);

    for run_index in 1..=RUNS {
        let mut latency_ms = Vec::new();
        let mut per_length: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        let mut completed_passes = 0usize;
        let mut completed_tokens = 0usize;
        let mut completed_requests = 0usize;
        let mut execution_seconds = 0.0f64;
        let mut validation_seconds = 0.0f64;
        let mut observation_write_seconds = 0.0f64;

        let run_start = Instant::now();
        while run_start.elapsed() < MIN_DURATION || completed_passes < MIN_PASSES {
            for (sample, prompt) in bundle.corpus.iter().enumerate() {
                let execution_start = Instant::now();
                let result = executor.forward_serving(std::slice::from_ref(prompt));
                let execution = execution_start.elapsed().as_secs_f64();
                execution_seconds += execution;

                let validation_start = Instant::now();
                let checked = match result {
                    Ok(output) => validate_output(&output, &bundle.corpus_goldens[sample]),
                    Err(error) => Err(error),
                };
                let validation = validation_start.elapsed().as_secs_f64();
                validation_seconds += validation;

                let observation = json!({
                    "run":run_index,
                    "pass_index":completed_passes,
                    "sample":sample,
                    "tokens":prompt.len(),
                    "request_latency_ms":execution * 1000.0,
                    "validation_ms":validation * 1000.0,
                    "pass":checked.is_ok(),
                    "error":checked.as_ref().err().map(|e| format!("{e:#}"))
                });
                let write_start = Instant::now();
                write_record(&mut observations, &observation)?;
                observation_write_seconds += write_start.elapsed().as_secs_f64();

                // Persist the failing observation before returning.
                if let Err(error) = checked {
                    observations.sync_all()?;
                    write_record(
                        &mut summaries,
                        &json!({
                            "kind":"run","run":run_index,"pass":false,
                            "completed_requests":completed_requests,
                            "errors":1,"rejections":0,"cancellations":0,"timeouts":0,
                            "reason":"forward_or_output_validation_failure"
                        }),
                    )?;
                    summaries.sync_all()?;
                    return Err(error).context("benchmark correctness failure");
                }

                let milliseconds = execution * 1000.0;
                latency_ms.push(milliseconds);
                per_length
                    .entry(prompt.len())
                    .or_default()
                    .push(milliseconds);
                completed_requests += 1;
                completed_tokens += prompt.len();
            }
            completed_passes += 1;
        }

        // The wall denominator includes validation, bookkeeping, and raw
        // observation writes, but excludes post-run sorting and summary work.
        let wall_seconds = run_start.elapsed().as_secs_f64();
        ensure!(
            wall_seconds >= MIN_DURATION.as_secs_f64(),
            "duration minimum"
        );
        ensure!(completed_passes >= MIN_PASSES, "pass minimum");
        ensure!(
            completed_requests == completed_passes * bundle.corpus.len(),
            "incomplete final pass"
        );

        latency_ms.sort_by(f64::total_cmp);
        let p50 = percentile(&latency_ms, 0.50)?;
        let p95 = percentile(&latency_ms, 0.95)?;
        let p99 = percentile(&latency_ms, 0.99)?;
        let request_rate = completed_requests as f64 / wall_seconds;
        ensure!(
            request_rate.is_finite() && request_rate > 0.0,
            "invalid request rate"
        );
        ensure!(p99 > 0.0, "invalid p99");

        let mut length_summaries = Vec::new();
        for (length, mut times) in per_length {
            times.sort_by(f64::total_cmp);
            length_summaries.push(json!({
                "tokens":length,
                "count":times.len(),
                "p50_ms":percentile(&times,0.50)?,
                "p95_ms":percentile(&times,0.95)?,
                "p99_ms":percentile(&times,0.99)?
            }));
        }

        let accounted = execution_seconds + validation_seconds + observation_write_seconds;
        let summary = json!({
            "kind":"run","run":run_index,"pass":true,
            "completed_passes":completed_passes,
            "completed_requests":completed_requests,
            "completed_sequences":completed_requests,
            "completed_tokens":completed_tokens,
            "wall_seconds":wall_seconds,
            "successful_requests_per_second":request_rate,
            "successful_sequences_per_second":request_rate,
            "actual_tokens_per_second":completed_tokens as f64 / wall_seconds,
            "latency_sample_count":latency_ms.len(),
            "request_latency_p50_ms":p50,
            "request_latency_p95_ms":p95,
            "request_latency_p99_ms":p99,
            "percentile_definition":"empirical_nearest_rank",
            "execution_seconds":execution_seconds,
            "output_validation_seconds":validation_seconds,
            "observation_write_seconds":observation_write_seconds,
            "other_harness_seconds":(wall_seconds-accounted).max(0.0),
            "errors":0,"rejections":0,"cancellations":0,"timeouts":0,
            "per_length":length_summaries
        });

        observations.sync_all()?;
        write_record(&mut summaries, &summary)?;
        summaries.sync_all()?;
        emit(summary.clone());
        run_summaries.push(summary);
        rates.push(request_rate);
        p99s.push(p99);
    }

    drop(executor);
    observations.sync_all()?;
    summaries.sync_all()?;

    let rate_ratio = rates.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        / rates.iter().copied().fold(f64::INFINITY, f64::min);
    let p99_ratio = p99s.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        / p99s.iter().copied().fold(f64::INFINITY, f64::min);

    let result = json!({
        "kind":"baseline_summary",
        "pass":rate_ratio <= 1.10 && p99_ratio <= 1.20,
        "all_three_runs_completed":run_summaries.len() == RUNS,
        "throughput_max_min_ratio":rate_ratio,
        "throughput_ratio_limit":1.10,
        "p99_max_min_ratio":p99_ratio,
        "p99_ratio_limit":1.20,
        "metadata_sha256":acceptance::file_sha256(&metadata_path)?,
        "observations_sha256":acceptance::file_sha256(&observations_path)?,
        "summaries_sha256":acceptance::file_sha256(&summaries_path)?,
        "executor_manifest_sha256":acceptance::file_sha256(&manifest_path)?,
        "evidence_directory":directory
    });
    emit(result.clone());
    ensure!(rate_ratio <= 1.10, "throughput stability gate failed");
    ensure!(p99_ratio <= 1.20, "p99 stability gate failed");
    Ok(result)
}

fn main() {
    match run() {
        Ok(summary) => emit(json!({
            "kind":"done","accepted":true,"pass":true,"summary":summary
        })),
        Err(error) => {
            emit(json!({
                "kind":"done","accepted":false,"pass":false,
                "error":format!("{error:#}")
            }));
            eprintln!("baseline failed: {error:#}");
            std::process::exit(1);
        }
    }
}
