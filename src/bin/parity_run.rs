// End-to-end parity run: load the frozen BF16 checkpoint on CPU, forward the
// captured reference prompts (raw token IDs), and compare against the saved
// vLLM embeddings. Pass criteria (frozen before evaluation, per ASTRO6 #4):
// dim 2048, all finite, |norm-1| <= 1e-5, cosine > 0.999, max_abs <= 0.01.

use anyhow::{ensure, Result};
use candle_core::Device;
use rustv1::model::{EngineConfig, Model};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 4, "usage: parity_run MODEL_DIR PROMPTS_JSON REFERENCE_F32");

    let dir = PathBuf::from(&args[1]);
    let prompts: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let ref_bytes = std::fs::read(&args[3])?;
    ensure!(ref_bytes.len() % 2048 == 0, "reference length not a multiple of 2048");
    let nref = ref_bytes.len() / (2048 * 4);
    let reference: Vec<f32> = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let engine: EngineConfig =
        serde_json::from_slice(&std::fs::read(dir.join("engine.json"))?)?;
    // FP32 CPU mathematical oracle per ASTRO6 #4: same BF16 checkpoint
    // values widened losslessly to FP32. Candle CPU does not support BF16
    // matmul; BF16 parity is a GPU (M2) concern.
    let model = Model::load(&dir, Device::Cpu, engine, candle_core::DType::F32)?;

    println!("loaded; running {} prompts (lengths {:?})",
        prompts.len(), prompts.iter().map(Vec::len).collect::<Vec<_>>());
    let started = std::time::Instant::now();
    let out = model.forward(&prompts)?;
    println!("forward done in {:.1}s", started.elapsed().as_secs_f64());

    ensure!(out.len() == nref, "output count {} != reference count {nref}", out.len());
    let mut pass = true;

    for (i, emb) in out.iter().enumerate() {
        let r = &reference[i * 2048..(i + 1) * 2048];
        ensure!(emb.len() == 2048, "sample {i}: dim {} != 2048", emb.len());
        let finite = emb.iter().all(|v| v.is_finite());
        let dot: f64 = emb.iter().zip(r).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = emb.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = r.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (na * nb);
        let max_abs = emb.iter().zip(r)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0f64, f64::max);
        let norm_dev = (na - 1.0).abs();

        let ok = finite
            && norm_dev <= 1e-5
            && cosine > 0.999
            && max_abs <= 0.01;
        pass &= ok;
        println!(
            "sample {i}: len={} finite={finite} cosine={cosine:.6} \
             max_abs={max_abs:.5} norm_dev={norm_dev:.2e} {}",
            prompts[i].len(),
            if ok { "PASS" } else { "FAIL" }
        );
    }

    if pass {
        println!("M1-CORE: PASS");
    } else {
        println!("M1-CORE: FAIL");
        std::process::exit(1);
    }
    Ok(())
}
