// Stage-level dumps for the Q2 stage-comparison protocol (ASTRO6 #9).
// Usage: stage_dump MODEL_DIR PROFILE SAMPLES_CSV [STAGES_CSV|all]
// Emits JSON lines: one per stage, FP32-widened, base64 f32-LE blobs,
// keyed by sample/layer/stage to pair with vLLM capture_rv1 records.

use anyhow::{ensure, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use candle_core::{DType, Device};
use rustv1::model::{EngineConfig, Model};
use rustv1::precision::PrecisionProfile;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const DEFAULT_STAGES: &str = "embed_tokens,input_layernorm.normalized,\
qkv.pre_rope,q.post_rope,k.post_rope,attention.pre_o_proj,o_proj,\
residual.post_attention,post_attention_layernorm.normalized,mlp.gate_up,\
mlp.silu_mul,mlp.down,residual.post_down,final_norm";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() >= 4,
        "usage: stage_dump MODEL_DIR PROFILE SAMPLES_CSV [STAGES_CSV|all]");
    let dir = PathBuf::from(&args[1]);
    let profile: PrecisionProfile = args[2].parse()?;
    let samples: Vec<usize> = args[3].split(',')
        .map(|x| x.trim().parse::<usize>())
        .collect::<std::result::Result<_, _>>()?;
    let stages_spec = args.get(4).map(String::as_str).unwrap_or(DEFAULT_STAGES);
    let all = stages_spec == "all";
    let stages: Vec<String> = if all { vec![] } else {
        stages_spec.split(',').map(|x| x.trim().to_string()).collect()
    };

    let prompts: Vec<Vec<u32>> =
        serde_json::from_slice(&std::fs::read("results/prompts.json")?)?;
    let engine: EngineConfig =
        serde_json::from_slice(&std::fs::read(dir.join("engine.json"))?)?;

    ensure!(candle_core::utils::cuda_is_available(), "CUDA required");
    let dev = Device::new_cuda(0)?;
    let model = Model::load(&dir, dev.clone(), engine, DType::BF16)?
        .with_precision(profile);
    println!("{}", json!({"kind": "stage_dump_start",
        "profile": format!("{profile:?}"), "device": format!("{dev:?}"),
        "stages": if all { "all".to_string() } else { stages.join(",") }}));

    for &si in &samples {
        let seq = vec![prompts[si].clone()];
        let mut collected: Vec<(isize, String, Vec<u8>, Vec<usize>)> =
            Vec::new();
        {
            let mut sink = |li: isize, name: &str, t: &candle_core::Tensor| {
                if !all && !stages.iter().any(|s| s == name) { return; }
                let f = t.to_dtype(DType::F32).and_then(|t| {
                    let shape = t.dims().to_vec();
                    t.flatten_all().map(|t| (t, shape))
                });
                if let Ok((t, shape)) = f {
                    if let Ok(v) = t.to_vec1::<f32>() {
                        let mut bytes =
                            Vec::with_capacity(v.len() * 4);
                        for x in v {
                            bytes.extend_from_slice(
                                &x.to_le_bytes());
                        }
                        collected.push((li, name.to_string(), bytes, shape));
                    }
                }
            };
            let states = model.run_decoder_dump(
                &seq, seq[0].len(), &mut sink)?;
            // Pooled stages computed from the dumped final states.
            let mean = states.i(0)?.sum(0)?.affine(
                1.0 / seq[0].len() as f64, 0.0)?;
            let mut bytes = Vec::with_capacity(2048 * 4);
            for x in mean.to_vec1::<f32>()? {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            collected.push((-1, "pooled.pre_l2".into(), bytes, vec![2048]));
        }
        for (li, name, bytes, shape) in &collected {
            let sha = format!("{:x}", Sha256::digest(bytes));
            println!("{}", json!({"kind": "stage", "sample_index": si,
                "token_sha256": rustv1::precision::token_digest(&prompts[si]),
                "layer_index": li, "stage_name": name, "shape": shape,
                "original_dtype": "bf16-executed-f32-widened",
                "encoding": "f32-le-base64", "npy_sha256": sha,
                "blob": STANDARD.encode(bytes)}));
        }
        let out = model.forward(&[prompts[si].clone()])?;
        let bytes: Vec<u8> = out[0].iter()
            .flat_map(|x| x.to_le_bytes()).collect();
        println!("{}", json!({"kind": "stage", "sample_index": si,
            "layer_index": -1, "stage_name": "final_embedding",
            "shape": [2048], "original_dtype": "f32",
            "encoding": "f32-le-base64",
            "npy_sha256": format!("{:x}", Sha256::digest(&bytes)),
            "blob": STANDARD.encode(&bytes)}));
    }
    println!("{}", json!({"kind": "stage_dump_done"}));
    Ok(())
}
