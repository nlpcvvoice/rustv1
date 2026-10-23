use anyhow::{ensure, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use candle_core::{DType, Device, IndexOp, Tensor};
use rustv1::model::{EngineConfig, Model};
use rustv1::precision::{token_digest, PrecisionProfile};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    io::{self, Write},
    path::PathBuf,
};

fn emit_tensor(
    output: &mut dyn Write,
    sample: usize,
    digest: &str,
    layer: isize,
    name: &str,
    tensor: &Tensor,
) -> Result<()> {
    let dtype = format!("{:?}", tensor.dtype());
    let values = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    ensure!(values.iter().all(|x| x.is_finite()), "{name}: nonfinite stage");
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    serde_json::to_writer(&mut *output, &json!({
        "kind": "stage",
        "evidence_class": "observed_execution",
        "sample_index": sample,
        "token_sha256": digest,
        "layer_index": layer,
        "stage_name": name,
        "shape": tensor.dims(),
        "original_dtype": dtype,
        "encoding": "f32-le-base64",
        "npy_sha256": format!("{:x}", Sha256::digest(&bytes)),
        "blob": STANDARD.encode(&bytes)
    }))?;
    output.write_all(b"\n")?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        (4..=6).contains(&args.len()),
        "usage: stage_dump MODEL_DIR F SAMPLES_CSV [STAGES_CSV|all] [LAYERS_CSV|all]"
    );
    let dir = PathBuf::from(&args[1]);
    let profile: PrecisionProfile = args[2].parse()?;
    ensure!(profile == PrecisionProfile::F, "M2 dump requires F");
    let samples: Vec<usize> = args[3].split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;

    let stage_spec = args.get(4).map(String::as_str).unwrap_or("all");
    let layer_spec = args.get(5).map(String::as_str).unwrap_or("-1,0");
    let stages: BTreeSet<&str> = stage_spec.split(',').collect();
    let layers: BTreeSet<isize> = if layer_spec == "all" {
        BTreeSet::new()
    } else {
        layer_spec.split(',').map(str::parse)
            .collect::<std::result::Result<_, _>>()?
    };

    let prompts: Vec<Vec<u32>> =
        serde_json::from_slice(&std::fs::read("results/prompts.json")?)?;
    let engine: EngineConfig =
        serde_json::from_slice(&std::fs::read(dir.join("engine.json"))?)?;
    let device = Device::new_cuda(0)?;
    let model = Model::load(&dir, device, engine, DType::BF16)?
        .with_precision(profile);

    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());

    for sample in samples {
        let ids = prompts.get(sample)
            .ok_or_else(|| anyhow::anyhow!("sample index out of range"))?;
        let digest = token_digest(ids);

        serde_json::to_writer(&mut output, &json!({
            "kind": "input",
            "sample_index": sample,
            "token_sha256": digest,
            "length": ids.len(),
            "token_ids": ids
        }))?;
        output.write_all(b"\n")?;

        let states = {
            let mut sink = |layer: isize, name: &str, tensor: &Tensor| -> Result<()> {
                if layer_spec != "all" && !layers.contains(&layer) {
                    return Ok(());
                }
                if stage_spec != "all" && !stages.contains(name) {
                    return Ok(());
                }
                emit_tensor(&mut output, sample, &digest, layer, name, tensor)
            };
            model.run_decoder_dump(std::slice::from_ref(ids), ids.len(), &mut sink)?
        };

        // Exactly the pooling arithmetic of Model::forward, on this execution's
        // states. There is no second decoder invocation.
        let pooled = states.i(0)?.sum(0)?
            .affine(1.0 / ids.len() as f64, 0.0)?
            .unsqueeze(0)?;
        let denominator = pooled.sqr()?.sum_keepdim(1)?.sqrt()?
            .clamp(1e-12, f64::INFINITY)?;
        let embedding = pooled.broadcast_div(&denominator)?;

        emit_tensor(
            &mut output, sample, &digest, -1, "pooled.pre_l2", &pooled.i(0)?,
        )?;
        emit_tensor(
            &mut output, sample, &digest, -1, "final_embedding", &embedding.i(0)?,
        )?;
        output.flush()?;
    }
    Ok(())
}
