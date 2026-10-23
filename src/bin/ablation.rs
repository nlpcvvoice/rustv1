// M2 ablation and forensics (ASTRO6 #9 deliverable integration).
// One profile per process: ablation MODEL_DIR PROFILE(A..F)
// Emits JSON lines consumable by the stage-comparison protocol.

use anyhow::{ensure, Result};
use candle_core::{DType, Device};
use rustv1::model::{EngineConfig, Model};
use rustv1::precision::{self, PrecisionProfile};
use serde_json::json;
use std::path::PathBuf;

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() % 4 == 0, "reference byte count not divisible by 4");
    let mut v = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(v)
}

// Same serving-budget grouping as parity_run (grouped order).
fn budget_groups(prompts: &[Vec<u32>], order: &[usize],
                 max_seqs: usize, max_tokens: usize) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut tokens = 0usize;
    let mut max_len = 0usize;
    for &i in order {
        let l = prompts[i].len();
        let next_tokens = tokens + l;
        let next_padded = (cur.len() + 1) * max_len.max(l);
        if !cur.is_empty()
            && (cur.len() + 1 > max_seqs
                || next_tokens > max_tokens
                || next_padded > max_tokens)
        {
            groups.push(std::mem::take(&mut cur));
            tokens = 0;
            max_len = 0;
        }
        cur.push(i);
        tokens += l;
        max_len = max_len.max(l);
    }
    if !cur.is_empty() { groups.push(cur); }
    groups
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 3, "usage: ablation MODEL_DIR PROFILE(A..F)");
    let dir = PathBuf::from(&args[1]);
    let profile: PrecisionProfile = args[2].parse()?;

    let prompts: Vec<Vec<u32>> =
        serde_json::from_slice(&std::fs::read("results/prompts.json")?)?;
    let reference = read_f32("results/ref_single.f32")?;
    ensure!(reference.len() == prompts.len() * 2048, "reference size");
    let engine: EngineConfig =
        serde_json::from_slice(&std::fs::read(dir.join("engine.json"))?)?;

    ensure!(candle_core::utils::cuda_is_available(), "CUDA required");
    let dev = Device::new_cuda(0)?;
    let model = Model::load(&dir, dev.clone(), engine.clone(), DType::BF16)?
        .with_precision(profile);
    println!("{}", json!({"kind": "start", "profile": format!("{profile:?}"),
        "device": format!("{dev:?}"), "dtype": "bf16"}));

    // 1. Bit-pattern conversion audit (identical FP32 inputs CPU vs CUDA).
    precision::conversion_audit(&dev)?;

    // 2. Singleton comparison against the frozen L4 reference.
    let mut singles = Vec::with_capacity(prompts.len());
    for (i, p) in prompts.iter().enumerate() {
        let out = model.forward(&[p.clone()])?;
        ensure!(out.len() == 1, "singleton cardinality");
        singles.push(out[0].clone());
        let (cos, max_abs, comp, l2) =
            precision::metrics(&out[0], &reference[i * 2048..(i + 1) * 2048])?;
        println!("{}", json!({"kind": "singleton_ref",
            "profile": format!("{profile:?}"), "sample": i,
            "token_sha256": precision::token_digest(p),
            "len": p.len(), "cosine": cos, "max_abs": max_abs,
            "max_error_component": comp, "l2": l2}));
    }

    // 3. Batch invariance forensics on the acceptance groups.
    let order: Vec<usize> = (0..prompts.len()).collect();
    let groups =
        budget_groups(&prompts, &order, engine.max_seqs, engine.max_tokens);
    println!("{}", json!({"kind": "forensics_groups",
        "sizes": groups.iter().map(Vec::len).collect::<Vec<_>>()}));
    precision::batch_forensics(&prompts, &groups,
        |seqs, s| model.forward_observed(seqs, s))?;

    // 4. Pooled-mean sensitivity for the failure and diagnostic samples.
    // The vLLM pre-L2 mean is not in the saved reference; the reference
    // direction is rescaled to our executed mean norm (cosine is
    // scale-invariant; the norm comparison is therefore not informative
    // and is labeled as such in the output).
    for &i in &[0usize, 21, 5, 19] {
        let states =
            model.forward_states(&[prompts[i].clone()])?[0].clone();
        let mean = model.pooled_mean(&[prompts[i].clone()])?
            .to_vec2::<f32>()?[0].clone();
        let out = singles[i].clone();
        let ref_out = reference[i * 2048..(i + 1) * 2048].to_vec();
        let mn: f64 = mean.iter()
            .map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let ref_mean: Vec<f32> =
            ref_out.iter().map(|v| (*v as f64 * mn) as f32).collect();
        precision::pooled_sensitivity(i, &states, &mean, &out,
            &ref_mean, &ref_out)?;
    }

    println!("{}", json!({"kind": "done", "profile": format!("{profile:?}")}));
    Ok(())
}
