// M2 bring-up smoke (ASTRO6 #6/#7 spec): exercise the exact operator
// shapes and layouts this model uses on a real CUDA device in BF16,
// compare against deterministic CPU references, then load the real
// model with explicit CUDA + BF16 selection (no silent fallback).
//
// Build: cargo build --release --locked --features cuda
// Run:   gpu_smoke MODEL_DIR

use anyhow::{ensure, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use rustv1::model::{EngineConfig, Model};
use std::path::PathBuf;

fn cmp(label: &str, gpu: &Tensor, cpu: &Tensor, max_abs_gate: f64) -> Result<()> {
    let a = gpu.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    ensure!(a.len() == b.len(), "{label}: len {} vs {}", a.len(), b.len());
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = dot / (na * nb);
    let max_abs = a.iter().zip(&b).map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0f64, f64::max);
    ensure!(cosine.is_finite() && max_abs.is_finite(), "{label}: non-finite metrics");
    ensure!(cosine > 0.9999, "{label}: cosine {cosine:.6} <= 0.9999");
    ensure!(max_abs <= max_abs_gate, "{label}: max_abs {max_abs:.4} > {max_abs_gate}");
    println!("op {label}: cosine={cosine:.6} max_abs={max_abs:.5} PASS");
    Ok(())
}

fn randn(rows: usize, cols: usize, seed: u64, dev: &Device) -> Result<Tensor> {
    let mut s = seed;
    let mut v = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
        let u = u.max(1e-9);
        let mag = (-(2.0 * u.ln())).sqrt();
        v.push(mag * (6.283185307179586 * u).cos() * 0.02);
    }
    Ok(Tensor::from_vec(v, (rows, cols), dev)?)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 2, "usage: gpu_smoke MODEL_DIR");
    let dir = PathBuf::from(&args[1]);

    // Explicit device selection: no silent CPU fallback.
    ensure!(cuda_is_available(), "CUDA backend not compiled in \
            (build with --features cuda)");
    let dev = Device::new_cuda(0)?;
    println!("device: {:?}", dev);

    let cpu = Device::Cpu;

    // Projection GEMMs at the exact widths this model uses.
    for (name, n, k, m_rows) in [
        ("q_proj 2048->3072", 3072usize, 2048usize, 7usize),
        ("k_proj 2048->1024", 1024, 2048, 7),
        ("o_proj 3072->2048", 2048, 3072, 7),
        ("gate_proj 2048->6144", 6144, 2048, 7),
        ("down_proj 6144->2048", 2048, 6144, 256),
    ] {
        let w = randn(n, k, 0xA11CE + n as u64, &cpu)?.to_dtype(DType::BF16)?;
        let x = randn(m_rows, k, 0xBEEF + n as u64, &cpu)?.to_dtype(DType::BF16)?;
        // CPU reference in FP32 over the same BF16-rounded inputs
        // (candle CPU has no BF16 matmul).
        let ref_out = x.to_dtype(DType::F32)?
            .broadcast_matmul(&w.to_dtype(DType::F32)?.t()?)?;
        let out = x.to_device(&dev)?.matmul(&w.t()?.to_device(&dev)?)?;
        cmp(name, &out, &ref_out, 0.05)?;
    }

    // Batched attention GEMMs with head_dim 128 (b=2, h=24, s=257 odd).
    {
        let (b, h, s, d) = (2usize, 24usize, 257usize, 128usize);
        let q = randn(b * h * s, d, 7, &cpu)?.to_dtype(DType::BF16)?
            .reshape((b, h, s, d))?;
        let k = randn(b * h * s, d, 8, &cpu)?.to_dtype(DType::BF16)?
            .reshape((b, h, s, d))?;
        let v = randn(b * h * s, d, 9, &cpu)?.to_dtype(DType::BF16)?
            .reshape((b, h, s, d))?;
        let scores_ref = q.to_dtype(DType::F32)?.broadcast_matmul(
            &k.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?)?;
        let probs_ref = candle_nn::ops::softmax(&scores_ref
            .affine(0.08838834764831845, 0.0)?, 3)?
            .to_dtype(DType::BF16)?;
        let ctx_ref = probs_ref.to_dtype(DType::F32)?.broadcast_matmul(
            &v.to_dtype(DType::F32)?)?;

        let (qg, kg, vg) = (q.to_device(&dev)?, k.to_device(&dev)?, v.to_device(&dev)?);
        let scores = qg.broadcast_matmul(&kg.transpose(2, 3)?.contiguous()?)?;
        let probs = candle_nn::ops::softmax(&scores.to_dtype(DType::F32)?
            .affine(0.08838834764831845, 0.0)?, 3)?
            .to_dtype(DType::BF16)?;
        let ctx = probs.broadcast_matmul(&vg)?;
        cmp("attention QK^T (b2 h24 s257 d128)", &scores, &scores_ref, 0.1)?;
        cmp("attention PV", &ctx, &ctx_ref, 0.05)?;

        // KV repetition (GQA 8 -> 24 heads) and head ordering.
        let kv = randn(b * 8 * s, d, 10, &cpu)?.to_dtype(DType::BF16)?
            .reshape((b, 8, s, d))?;
        let rep_ref = kv.repeat((0, 3))?;
        let rep = kv.to_device(&dev)?.repeat((0, 3))?;
        cmp("GQA kv repeat 8->24", &rep, &rep_ref, 1e-6)?;
    }

    // Embedding lookup (BF16) and D2H.
    {
        let table = randn(131072, 2048, 11, &cpu)?.to_dtype(DType::BF16)?;
        let ids = Tensor::new(&[1u32, 2, 3, 11, 131071], &cpu)?;
        let ref_out = table.i(&ids)?;
        let out = table.to_device(&dev)?.i(&ids.to_device(&dev)?)?;
        cmp("embedding gather bf16", &out, &ref_out, 1e-6)?;
    }

    // FP32 reductions and softmax.
    {
        let x = randn(256, 2048, 12, &cpu)?;
        let sum_ref = x.sum(1)?;
        let sum = x.to_device(&dev)?.sum(1)?;
        cmp("fp32 row sum", &sum, &sum_ref, 1e-3)?;
        let sm_ref = candle_nn::ops::softmax(&x, 1)?;
        let sm = candle_nn::ops::softmax(&x.to_device(&dev)?, 1)?;
        cmp("fp32 softmax", &sm, &sm_ref, 1e-5)?;
    }

    // BF16 <-> FP32 conversion boundaries.
    {
        let x = randn(64, 64, 13, &cpu)?;
        let rt_ref = x.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
        let rt = x.to_device(&dev)?.to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?;
        cmp("bf16<->f32 roundtrip", &rt, &rt_ref, 1e-6)?;
    }

    // Real model: explicit CUDA + BF16, no fallback.
    let engine: EngineConfig =
        serde_json::from_slice(&std::fs::read(dir.join("engine.json"))?)?;
    let model = Model::load(&dir, dev.clone(), engine.clone(), DType::BF16)?;
    println!("model loaded on CUDA in BF16");

    let prompts: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(
        "results/prompts.json"))?;
    for &i in &[0usize, 1, 10, 2] {
        let out = model.forward(&[prompts[i].clone()])?;
        ensure!(out.len() == 1 && out[0].len() == 2048, "smoke sample {i}: shape");
        ensure!(out[0].iter().all(|v| v.is_finite()), "smoke sample {i}: non-finite");
        let n: f64 = out[0].iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        ensure!((n - 1.0).abs() <= 1e-5, "smoke sample {i}: norm {n}");
        println!("model smoke sample {i} (len {}): finite, norm_dev={:.2e} PASS",
            prompts[i].len(), (n - 1.0).abs());
    }

    // Maximum admitted padded shape: B=4 x S=1024 = 4096 tokens.
    let t0 = std::time::Instant::now();
    let seq = vec![7u32; 1024];
    let out = model.forward(&[seq.clone(), seq.clone(), seq.clone(), seq])?;
    ensure!(out.len() == 4, "max-shape batch rows");
    for r in &out {
        ensure!(r.len() == 2048 && r.iter().all(|v| v.is_finite()),
            "max-shape: bad row");
    }
    println!("max-shape B=4 S=1024: PASS in {:.1}s", t0.elapsed().as_secs_f64());

    println!("GPU-SMOKE: PASS");
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_is_available() -> bool {
    candle_core::cuda_is_available().unwrap_or(false)
}

#[cfg(not(feature = "cuda"))]
fn cuda_is_available() -> bool {
    false
}
