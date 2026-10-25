use anyhow::{ensure, Context, Result};
use rustv1::m2_route;
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const WIDTH: usize = 2048;
const REQUIRED_WARMUP_PASSES: usize = 10;
const MIN_TIMED_DURATION: Duration = Duration::from_secs(60);
const MIN_TIMED_PASSES: usize = 100;
const TIMED_RUNS: usize = 3;

fn load_prompts(path: &Path) -> Result<Vec<Vec<u32>>> {
    let bytes = fs::read(path)?;
    let prompts: Vec<Vec<u32>> = serde_json::from_slice(&bytes)?;
    ensure!(
        prompts.len() == 22,
        "corpus must contain exactly 22 samples"
    );
    for (i, p) in prompts.iter().enumerate() {
        ensure!(
            !p.is_empty() && p.len() <= 1024,
            "sample {i} invalid length"
        );
    }
    Ok(prompts)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 5,
        "usage: bench_m2_baseline MODEL_DIR PROMPTS_JSON MANIFEST_DIR AUDIT_IDENTITY"
    );

    let model_dir = PathBuf::from(&args[1]);
    let prompts_path = PathBuf::from(&args[2]);
    let manifest_dir = PathBuf::from(&args[3]);
    let audit_identity = args[4].clone();

    let prompts = load_prompts(&prompts_path)?;
    let _total_corpus_tokens: usize = prompts.iter().map(Vec::len).sum();

    println!("initializing designated M2 executor...");
    let manifest_path = manifest_dir.join("m2_benchmark_manifest.jsonl");
    let mut executor = m2_route::open(&model_dir, &manifest_path, 0, &audit_identity)
        .context("failed opening M2 executor")?;

    println!("starting {REQUIRED_WARMUP_PASSES} untimed warm-up passes (C=1)...");
    for _ in 0..REQUIRED_WARMUP_PASSES {
        for prompt in &prompts {
            let res = executor.forward_serving(std::slice::from_ref(prompt))?;
            ensure!(res.len() == 1 && res[0].len() == WIDTH, "invalid output");
        }
    }
    println!("warm-up complete. Beginning {TIMED_RUNS} timed runs...");

    for run_idx in 1..=TIMED_RUNS {
        let mut sample_latencies_ms = Vec::new();
        let mut length_latencies: std::collections::BTreeMap<usize, Vec<f64>> =
            std::collections::BTreeMap::new();
        let mut completed_passes = 0usize;
        let mut completed_tokens = 0usize;
        let mut completed_samples = 0usize;

        let run_start = Instant::now();

        while run_start.elapsed() < MIN_TIMED_DURATION || completed_passes < MIN_TIMED_PASSES {
            for prompt in &prompts {
                let s_len = prompt.len();
                let start = Instant::now();
                let out = executor.forward_serving(std::slice::from_ref(prompt))?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                // Host validation (CPU-resident embeddings check)
                ensure!(out[0].len() == WIDTH, "row validation failed");
                ensure!(out[0][0].is_finite(), "non-finite result returned");

                sample_latencies_ms.push(elapsed_ms);
                length_latencies.entry(s_len).or_default().push(elapsed_ms);
                completed_samples += 1;
                completed_tokens += s_len;
            }
            completed_passes += 1;
        }

        let total_wall_secs = run_start.elapsed().as_secs_f64();
        sample_latencies_ms.sort_by(f64::total_cmp);

        let samples_per_sec = completed_samples as f64 / total_wall_secs;
        let tokens_per_sec = completed_tokens as f64 / total_wall_secs;

        println!("--- RUN {run_idx} RESULTS ---");
        println!("passes:            {completed_passes}");
        println!("wall time:         {total_wall_secs:.3} s");
        println!("throughput:        {samples_per_sec:.2} samples/sec");
        println!("tokens/sec:        {tokens_per_sec:.2} tokens/sec");
        println!(
            "latencies (ms):    p50={:.2}  p95={:.2}  p99={:.2}",
            percentile(&sample_latencies_ms, 0.50),
            percentile(&sample_latencies_ms, 0.95),
            percentile(&sample_latencies_ms, 0.99),
        );
        println!("per-length breakdown (p50 ms):");
        for (len, mut times) in length_latencies {
            times.sort_by(f64::total_cmp);
            println!(
                "  L={:<4} (count={}): p50={:.2} ms",
                len,
                times.len(),
                percentile(&times, 0.50)
            );
        }
    }

    println!("all baseline benchmark runs completed successfully.");
    Ok(())
}
