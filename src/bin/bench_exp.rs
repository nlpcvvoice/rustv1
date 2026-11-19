// Exploratory batched-throughput sweep. NOT a certifying run:
// no policy, no startup check, no attestation binding. The v2 audit
// identity is recorded as historical B=1 provenance only; batched source
// qualification and numerical acceptance are pending v3.
//
// Usage: bench_exp <MODEL_DIR> <FIXTURES_JSON> <OUT_JSON> <AUDIT_ID>

#[cfg(not(feature = "cuda"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("bench_exp requires the cuda feature")
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use anyhow::ensure;
    use serde_json::json;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 5,
        "usage: bench_exp MODEL_DIR FIXTURES OUT AUDIT_ID"
    );
    let model_dir = Path::new(&args[1]);
    let fixtures_path = &args[2];
    let out_path = &args[3];

    let audit_identity = args[4].as_str();
    println!("EXPERIMENT non_certifying=true audit_provenance=v2_B1");

    let fixtures: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(fixtures_path)?)?;
    let mut pools: Vec<Vec<u32>> = Vec::new();
    for f in &fixtures {
        let ids: Vec<u32> = serde_json::from_value(f["ids"].clone())?;
        ensure!(!ids.is_empty(), "empty fixture ids");
        pools.push(ids);
    }
    println!(
        "EXPERIMENT fixtures={} pool_max_len={}",
        pools.len(),
        pools.iter().map(Vec::len).max().unwrap()
    );

    let batching = rustv1::model::BatchedExecution::default();
    let manifest = model_dir.join("bench_exp_executor_manifest.jsonl");
    let mut executor = rustv1::m2_route::open_batched_with_attestation(
        model_dir,
        &manifest,
        0,
        audit_identity,
        None,
        batching.clone(),
    )?;
    println!(
        "EXPERIMENT executor_open max_b={} max_padded={}",
        batching.max_batch_size, batching.max_padded_tokens
    );

    let mut results = Vec::new();
    for &s in &[32usize, 128, 512, 1024] {
        for &b in &[1usize, 2, 4, 8, 16, 32] {
            if b * s > batching.max_padded_tokens || b * s * s > batching.max_attention_cells {
                println!("SWEEP b={b} s={s} SKIP ceilings");
                continue;
            }
            let mut sequences: Vec<Vec<u32>> = Vec::with_capacity(b);
            for i in 0..b {
                let src = &pools[i % pools.len()];
                let mut seq = vec![1u32; s];
                let n = src.len().min(s);
                seq[..n].copy_from_slice(&src[..n]);
                sequences.push(seq);
            }
            let assoc: Vec<rustv1::model::ExecutionRow> = (0..b)
                .map(|i| rustv1::model::ExecutionRow {
                    request_id: i as u64,
                    request_row: 0,
                    case_id: Some("exp".to_owned()),
                })
                .collect();

            for _ in 0..2 {
                executor.forward_batch(&sequences, &assoc)?;
            }
            let mut times = Vec::new();
            for _ in 0..10 {
                let t0 = Instant::now();
                let out = executor.forward_batch(&sequences, &assoc)?;
                times.push(t0.elapsed().as_secs_f64());
                ensure!(out.len() == b, "batch output cardinality");
            }
            times.sort_by(|a, b2| a.partial_cmp(b2).unwrap());
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let rec = json!({
                "b": b,
                "s": s,
                "padded_tokens": b * s,
                "median_ms": (times[times.len() / 2] * 1000.0).round() / 1000.0,
                "min_ms": (times[0] * 1000.0).round() / 1000.0,
                "max_ms": (times[times.len() - 1] * 1000.0).round() / 1000.0,
                "mean_ms": (mean * 1000.0).round() / 1000.0,
                "tokens_per_s": (b * s) as f64 / mean,
            });
            println!("SWEEP {}", serde_json::to_string(&rec)?);
            results.push(rec);
        }
    }

    let doc = json!({
        "experiment": true,
        "certifying": false,
        "audit_provenance": audit_identity,
        "note": "batched qualification pending v3; v2 covers B=1 only",
        "results": results,
    });
    std::fs::write(out_path, serde_json::to_vec_pretty(&doc)?)?;
    println!("EXPERIMENT_RC 0");
    Ok(())
}
