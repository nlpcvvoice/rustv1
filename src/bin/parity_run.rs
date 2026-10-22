// M1 acceptance run: load the frozen BF16 checkpoint as an FP32 CPU
// oracle, compare the expanded corpus against saved vLLM embeddings,
// then run self-consistency suites: batch invariance, reordered batch,
// duplicate requests, invalid-input rejection, and a stage-level
// bidirectionality test.
//
// Frozen pass criteria (per ASTRO6 #4, fixed before evaluation):
//   reference comparison: dim 2048, finite, |norm-1| <= 1e-5,
//                         cosine > 0.999, max_abs <= 0.01
//   batch invariance:    cosine > 0.9999, max_abs <= 0.005
//     (relaxed from the proposed 0.99999/0.001 because the captured vLLM
//      0.25.0 reference itself shows batch sensitivity of cosine 0.99992
//      and max_abs 0.0026; evidence in reference meta.json)

use anyhow::{ensure, Result};
use candle_core::Device;
use rustv1::model::{EngineConfig, Model};
use std::path::PathBuf;

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path).unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn compare(emb: &[f32], reference: &[f32]) -> (f64, f64, f64) {
    let dot: f64 = emb.iter().zip(reference).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = emb.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = reference.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let max_abs = emb.iter().zip(reference)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max);
    (dot / (na * nb), max_abs, (na - 1.0).abs())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 4, "usage: parity_run MODEL_DIR PROMPTS_JSON REFERENCE_F32");

    let dir = PathBuf::from(&args[1]);
    let prompts: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let reference = read_f32(&args[3]);
    ensure!(reference.len() == prompts.len() * 2048,
        "reference {} != {} x 2048", reference.len(), prompts.len());

    let engine: EngineConfig =
        serde_json::from_slice(&std::fs::read(dir.join("engine.json"))?)?;
    let model = Model::load(&dir, Device::Cpu, engine, candle_core::DType::F32)?;

    println!("loaded; running {} prompts (lengths {:?})",
        prompts.len(), prompts.iter().map(Vec::len).collect::<Vec<_>>());
    let started = std::time::Instant::now();
    let out = model.forward(&prompts)?;
    println!("forward done in {:.1}s", started.elapsed().as_secs_f64());

    let mut pass = true;
    for (i, emb) in out.iter().enumerate() {
        let r = &reference[i * 2048..(i + 1) * 2048];
        ensure!(emb.len() == 2048 && emb.iter().all(|v| v.is_finite()),
            "sample {i}: bad shape/finiteness");
        let (cosine, max_abs, norm_dev) = compare(emb, r);
        let ok = norm_dev <= 1e-5 && cosine > 0.999 && max_abs <= 0.01;
        pass &= ok;
        println!("sample {i}: len={} cosine={cosine:.6} max_abs={max_abs:.5} \
                 norm_dev={norm_dev:.2e} {}",
            prompts[i].len(), if ok { "PASS" } else { "FAIL" });
    }

    // Batch invariance: singletons (already computed above) versus one
    // mixed batch and a reversed batch.
    let idx: Vec<usize> = (0..prompts.len()).collect();
    let mut total = 0usize;
    let mut worst_cos = 1.0f64;
    let mut worst_abs = 0.0f64;

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_tokens = 0usize;
    for &i in &idx {
        let l = prompts[i].len();
        if cur_tokens + l > engine.max_tokens || cur.len() + 1 > engine.max_seqs {
            if !cur.is_empty() { groups.push(std::mem::take(&mut cur)); }
            cur_tokens = 0;
        }
        cur.push(i);
        cur_tokens += l;
    }
    if !cur.is_empty() { groups.push(cur); }

    for g in 0..groups.len() {
        let seqs: Vec<_> = groups[g].iter().map(|&i| prompts[i].clone()).collect();
        let outs = model.forward(&seqs)?;
        for (k, &i) in groups[g].iter().enumerate() {
            let (cosine, max_abs, _) = compare(&outs[k], &out[i]);
            worst_cos = worst_cos.min(cosine);
            worst_abs = worst_abs.max(max_abs);
            total += 1;
        }
    }
    let mut rev: Vec<usize> = idx.clone();
    rev.reverse();
    let seqs: Vec<_> = rev.iter().map(|&i| prompts[i].clone()).collect();
    let outs = model.forward(&seqs)?;
    for (k, &i) in rev.iter().enumerate() {
        let (cosine, max_abs, _) = compare(&outs[k], &out[i]);
        worst_cos = worst_cos.min(cosine);
        worst_abs = worst_abs.max(max_abs);
        total += 1;
    }
    // Duplicate request in one batch.
    let dup = vec![prompts[0].clone(), prompts[0].clone()];
    let outs = model.forward(&dup)?;
    let (cosine, max_abs, _) = compare(&outs[0], &outs[1]);
    println!("batch invariance: n={total} worst_cosine={worst_cos:.8} \
             worst_max_abs={worst_abs:.6} dup_cosine={cosine:.8} dup_max_abs={max_abs:.6}");
    let inv_ok = worst_cos > 0.9999 && worst_abs <= 0.005
        && cosine > 0.9999 && max_abs <= 0.005;
    pass &= inv_ok;
    println!("batch-invariance: {}", if inv_ok { "PASS" } else { "FAIL" });

    // Invalid inputs must be rejected.
    let bad_empty = model.forward(&[vec![]]).is_err();
    let bad_pad = model.forward(&[vec![11, 11]]).is_err();
    let bad_long = model.forward(&[vec![5u32; engine.max_len + 1]]).is_err();
    let bad_oov = model.forward(&[vec![cfg_vocab_oov()]]).is_err();
    let neg_ok = bad_empty && bad_pad && bad_long && bad_oov;
    println!("invalid-inputs: empty={bad_empty} pad={bad_pad} long={bad_long} \
             oov={bad_oov} -> {}", if neg_ok { "PASS" } else { "FAIL" });
    pass &= neg_ok;

    // Stage-level bidirectionality: changing a later token must change an
    // earlier token's post-norm state.
    let base = vec![12107u32, 1541, 1058, 52528, 52528, 1541];
    let mut altered = base.clone();
    *altered.last_mut().unwrap() = 9990;
    let states = model.forward_states(&[base.clone(), altered])?;
    let d0: f64 = states[0][0].iter().zip(&states[1][0])
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);
    let dlast: f64 = states[0][base.len() - 1].iter()
        .zip(&states[1][base.len() - 1])
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);
    let bidir_ok = d0 > 1e-4 && dlast > 1e-4;
    println!("bidirectionality: token0_maxdiff={d0:.5} lasttoken_maxdiff={dlast:.5} \
             -> {}", if bidir_ok { "PASS" } else { "FAIL" });
    pass &= bidir_ok;

    if pass {
        println!("M1-ACCEPTANCE: PASS");
    } else {
        println!("M1-ACCEPTANCE: FAIL");
        std::process::exit(1);
    }
    Ok(())
}

fn cfg_vocab_oov() -> u32 { 131_072 }
