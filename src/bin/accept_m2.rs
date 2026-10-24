// M2 acceptance via the independent-sequence executor (ASTRO6 #10,
// ruling Q2c): every sequence runs as a physical B=1, S=valid_len
// forward. Gates: singleton parity against the frozen L4 reference,
// bit identity across grouping/reorder/duplicate, empty exception
// list. accept_m2 MODEL_DIR

use anyhow::{bail, ensure, Context, Result};
use candle_core::Device;
use rustv1::cuda_math::MathModePin;
use rustv1::model::{EngineConfig, IndependentExecutor};
use rustv1::precision;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() % 4 == 0, "reference byte count not divisible by 4");
    let mut v = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(v)
}

fn row_sha(row: &[f32]) -> String {
    let bytes: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
    format!("{:x}", Sha256::digest(&bytes))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 2, "usage: accept_m2 MODEL_DIR");
    let dir = PathBuf::from(&args[1]);

    let prompts: Vec<Vec<u32>> =
        serde_json::from_slice(&std::fs::read("results/prompts.json")?)?;
    let reference = read_f32("results/ref_single.f32")?;
    ensure!(reference.len() == prompts.len() * 2048, "reference size");
    let engine: EngineConfig =
        serde_json::from_slice(&std::fs::read(dir.join("engine.json"))?)?;

    let exceptions: serde_json::Value =
        serde_json::from_slice(&std::fs::read(
            "results/exception_policy_m2.json")?)?;
    let listed = exceptions["samples"].as_array()
        .context("exception policy missing samples array")?;
    ensure!(listed.is_empty(), "M2 exception list must be empty");

    ensure!(candle_core::utils::cuda_is_available(), "CUDA required");
    let dev = Device::new_cuda(0)?;
    let cuda_dev = match &dev {
        Device::Cuda(d) => d.clone(),
        _ => bail!("CUDA device expected"),
    };

    // Candle's live cuBLAS handle via the public accessor; the pin
    // records version, stream, math and atomics mode around every
    // physical forward in the manifest.
    let blas = cuda_dev.cublas_handle();
    let handle = *blas.handle();
    let audit = "candle-core-0.9.2 cublas_handle() + cudarc-0.17.8 \
                 handle(); backend excerpts audited 2026-10-23"
        .to_string();
    let math = unsafe {
        MathModePin::from_audited_candle_handle(
            handle as *mut std::ffi::c_void, audit)?
    };

    let mut exec = unsafe {
        IndependentExecutor::load(&dir, dev.clone(), engine, math,
            Path::new("m2_acceptance_manifest.jsonl"))?
    };
    println!("{}", json!({"kind": "start", "device": format!("{dev:?}"),
        "policy": "independent_sequence_v1", "profile": "F"}));

    // 1. Singleton parity against the frozen L4 reference.
    let mut sha = Vec::with_capacity(prompts.len());
    let mut failures = 0usize;
    for (i, p) in prompts.iter().enumerate() {
        let rows = exec.forward_serving(&[p.clone()])?;
        ensure!(rows.len() == 1, "singleton cardinality");
        let (cos, max_abs, comp, l2) =
            precision::metrics(&rows[0], &reference[i * 2048..(i + 1) * 2048])?;
        let pass = cos > 0.999 && max_abs <= 0.01;
        if !pass { failures += 1; }
        sha.push(row_sha(&rows[0]));
        println!("{}", json!({"kind": "singleton_ref", "sample": i,
            "token_sha256": precision::token_digest(p), "len": p.len(),
            "cosine": cos, "max_abs": max_abs, "max_error_component": comp,
            "l2": l2, "pass": pass, "output_sha256": sha[i]}));
    }

    // 2. Bit identity: one grouped request, reversed order, duplicates.
    let all: Vec<Vec<u32>> = prompts.clone();
    let rows = exec.forward_serving(&all)?;
    let grouped: Vec<String> = rows.iter().map(|r| row_sha(r)).collect();
    let grouped_ok = grouped == sha;
    println!("{}", json!({"kind": "bit_identity_grouped",
        "count": grouped.len(), "identical_to_singles": grouped_ok}));

    let rev: Vec<Vec<u32>> = prompts.iter().rev().cloned().collect();
    let rows = exec.forward_serving(&rev)?;
    let rev_sha: Vec<String> = rows.iter().map(|r| row_sha(r)).collect();
    let rev_ok = rev_sha == sha.iter().rev().cloned().collect::<Vec<_>>();
    println!("{}", json!({"kind": "bit_identity_reversed",
        "count": rev_sha.len(), "identical_to_singles": rev_ok}));

    let dup = vec![prompts[0].clone(), prompts[0].clone(), prompts[0].clone()];
    let rows = exec.forward_serving(&dup)?;
    let dup_ok = rows.iter().all(|r| row_sha(r) == sha[0]);
    println!("{}", json!({"kind": "bit_identity_duplicate",
        "count": rows.len(), "identical_to_singleton": dup_ok}));

    let accepted = failures == 0 && grouped_ok && rev_ok && dup_ok;
    println!("{}", json!({"kind": "done", "accepted": accepted,
        "singleton_failures": failures,
        "manifest": "m2_acceptance_manifest.jsonl"}));
    ensure!(accepted, "M2 acceptance gates failed");
    Ok(())
}
