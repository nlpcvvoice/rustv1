use anyhow::{bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use candle_core::{DType, Device, Tensor};
use rustv1::model::{EngineConfig, Model};
use rustv1::precision::{comparison_metrics, PrecisionProfile};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

fn object(line: &str) -> Result<Option<Value>> {
    let line = line.trim();
    let line = line.strip_prefix("RV1 ").unwrap_or(line);
    if !line.starts_with('{') {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(line)?))
}

fn read_records(path: &str, digest: &str, layer: i64) -> Result<BTreeMap<String, Value>> {
    // First pass obtains explicit sample-index-to-token-digest bindings.
    let mut identities = BTreeMap::<u64, String>::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let Some(v) = object(&line?)? else { continue };
        if let (Some(index), Some(hash)) = (v["sample_index"].as_u64(), v["token_sha256"].as_str())
        {
            if let Some(old) = identities.insert(index, hash.to_owned()) {
                ensure!(old == hash, "sample index has conflicting token identities");
            }
        }
    }

    let mut result = BTreeMap::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let Some(v) = object(&line?)? else { continue };
        if !matches!(v["kind"].as_str(), Some("stage" | "tensor")) {
            continue;
        }

        let hash = v["token_sha256"]
            .as_str()
            .or_else(|| {
                v["sample_index"]
                    .as_u64()
                    .and_then(|i| identities.get(&i).map(String::as_str))
            })
            .context("tensor lacks a bound token digest")?;

        if hash != digest || v["layer_index"].as_i64() != Some(layer) {
            continue;
        }

        let name = v["stage_name"].as_str().context("stage_name")?.to_owned();
        ensure!(
            result.insert(name.clone(), v).is_none(),
            "ambiguous duplicate stage {name}; fix capture attribution"
        );
    }
    ensure!(!result.is_empty(), "no matching stages");
    Ok(result)
}

fn shape(v: &Value) -> Result<Vec<usize>> {
    v["shape"]
        .as_array()
        .context("shape")?
        .iter()
        .map(|n| {
            let n = n.as_u64().context("noninteger shape")?;
            usize::try_from(n).context("shape overflow")
        })
        .collect()
}

fn decode(v: &Value) -> Result<(Vec<f32>, Vec<usize>)> {
    let raw = STANDARD.decode(v["blob"].as_str().context("missing raw blob")?)?;
    let expected_hash = v["npy_sha256"].as_str().context("missing payload hash")?;
    ensure!(
        format!("{:x}", Sha256::digest(&raw)) == expected_hash,
        "payload hash mismatch"
    );
    let shape = shape(v)?;
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d).context("shape overflow"))?;

    let bytes: &[u8] = match v["encoding"].as_str().context("encoding")? {
        "f32-le-base64" => &raw,
        "npy-f32-le-base64" => {
            ensure!(raw.len() >= 10 && &raw[..6] == b"\x93NUMPY", "invalid npy");
            let (header_start, header_len) = match raw[6] {
                1 => (10usize, u16::from_le_bytes(raw[8..10].try_into()?) as usize),
                2 | 3 => {
                    ensure!(raw.len() >= 12, "truncated npy");
                    (12usize, u32::from_le_bytes(raw[8..12].try_into()?) as usize)
                }
                version => bail!("unsupported npy version {version}"),
            };
            let end = header_start
                .checked_add(header_len)
                .context("npy overflow")?;
            ensure!(end <= raw.len(), "truncated npy header");
            let header = std::str::from_utf8(&raw[header_start..end])?;
            ensure!(
                header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\""),
                "npy must be little-endian f32"
            );
            ensure!(
                header.contains("'fortran_order': False")
                    || header.contains("\"fortran_order\": False"),
                "Fortran npy unsupported"
            );

            let tail = header
                .split("'shape':")
                .nth(1)
                .or_else(|| header.split("\"shape\":").nth(1))
                .context("npy shape")?;
            let start = tail.find('(').context("npy shape tuple")?;
            let end_tuple = tail[start..].find(')').context("npy shape tuple")? + start;
            let npy_shape: Vec<usize> = tail[start + 1..end_tuple]
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()?;
            ensure!(npy_shape == shape, "npy/record shape mismatch");
            &raw[end..]
        }
        other => bail!("unsupported encoding {other}"),
    };

    ensure!(
        bytes.len() == count.checked_mul(4).context("size overflow")?,
        "payload length mismatch"
    );
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect();
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "nonfinite captured tensor"
    );
    Ok((values, shape))
}

fn tensor(
    records: &BTreeMap<String, Value>,
    name: &str,
    heads: Option<usize>,
    device: &Device,
    bf16: bool,
) -> Result<Tensor> {
    let v = records
        .get(name)
        .with_context(|| format!("missing stage {name}"))?;
    let (values, shape) = decode(v)?;

    if bf16 {
        ensure!(
            values.iter().all(|x| x.to_bits() & 0xffff == 0),
            "{name}: input is not losslessly BF16-representable"
        );
    }

    let x = Tensor::from_vec(values, shape.as_slice(), device)?;
    let x = if let Some(heads) = heads {
        match shape.as_slice() {
            [1, h, _, 128] if *h == heads => x,
            [s, h, 128] if *h == heads => x.reshape((1, *s, heads, 128))?.transpose(1, 2)?,
            [s, width] if *width == heads * 128 => {
                x.reshape((1, *s, heads, 128))?.transpose(1, 2)?
            }
            _ => bail!("{name}: unsupported head layout {shape:?}"),
        }
    } else {
        match shape.as_slice() {
            [1, _, _] => x,
            [s, width] => x.reshape((1, *s, *width))?,
            _ => bail!("{name}: unsupported token layout {shape:?}"),
        }
    };

    Ok(if bf16 { x.to_dtype(DType::BF16)? } else { x })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 6,
        "usage: replay_common_input MODEL_DIR CAPTURE_JSONL TOKEN_SHA256 LAYER OP"
    );
    let dir = PathBuf::from(&args[1]);
    let digest = &args[3];
    ensure!(
        digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid digest"
    );
    let layer: i64 = args[4].parse()?;
    let op = args[5].as_str();

    let record_layer = if op == "final_rms" { -1 } else { layer };
    ensure!(
        op == "final_rms" || (0..16).contains(&layer),
        "invalid layer"
    );
    let records = read_records(&args[2], digest, record_layer)?;

    let engine: EngineConfig = serde_json::from_slice(&std::fs::read(dir.join("engine.json"))?)?;
    let audit_identity = std::env::var("RV1_CUDA_MATH_AUDIT").map_err(|_| {
        anyhow::anyhow!("RV1_CUDA_MATH_AUDIT must identify the locked-source audit artifact")
    })?;
    let math = rustv1::cuda_math::MathModePin::pin(audit_identity)?;
    let device = Device::new_cuda(0)?;
    let model =
        Model::load(&dir, device.clone(), engine, DType::BF16)?.with_precision(PrecisionProfile::F);

    let get = |name: &str, heads: Option<usize>| tensor(&records, name, heads, &device, true);

    let (inputs, output_stage, output_heads) = match op {
        "input_rms" | "post_attention_rms" | "final_rms" => {
            let prefix = match op {
                "input_rms" => "input_layernorm",
                "post_attention_rms" => "post_attention_layernorm",
                _ => "norm",
            };
            let mut inputs = vec![get(&format!("{prefix}.hidden_in"), None)?];
            let residual = format!("{prefix}.residual_in");
            if records.contains_key(&residual) {
                inputs.push(get(&residual, None)?);
            }
            let output = if op == "final_rms" && !records.contains_key("norm.normalized") {
                "final_norm".to_owned()
            } else {
                format!("{prefix}.normalized")
            };
            (inputs, output, None)
        }
        "qkv" => (
            vec![get("input_layernorm.normalized", None)?],
            "qkv.pre_rope".to_owned(),
            None,
        ),
        "rope_q" => (
            vec![get("q.pre_rope", Some(24))?],
            "q.post_rope".to_owned(),
            Some(24),
        ),
        "rope_k" => (
            vec![get("k.pre_rope", Some(8))?],
            "k.post_rope".to_owned(),
            Some(8),
        ),
        "attention" => (
            vec![
                get("q.attention_input", Some(24))?,
                get("k.attention_input", Some(8))?,
                get("v.attention_input", Some(8))?,
            ],
            "attention.pre_o_proj".to_owned(),
            None,
        ),
        "o_proj" => (
            vec![get("attention.pre_o_proj", None)?],
            "o_proj".to_owned(),
            None,
        ),
        "gate_up" => (
            vec![get("post_attention_layernorm.normalized", None)?],
            "mlp.gate_up".to_owned(),
            None,
        ),
        "silu_mul" => (
            vec![get("mlp.gate_up", None)?],
            "mlp.silu_mul".to_owned(),
            None,
        ),
        "down" => (
            vec![get("mlp.silu_mul", None)?],
            "mlp.down".to_owned(),
            None,
        ),
        other => bail!("unsupported operator {other}"),
    };

    if matches!(op, "rope_q" | "rope_k") {
        let positions = records
            .get("positions")
            .context("missing rotary positions")?;
        let (positions, _) = decode(positions)?;
        let s = inputs[0].dim(2)?;
        ensure!(
            positions.len() == s && positions.iter().enumerate().all(|(i, &p)| p == i as f32),
            "this Rust rotary path requires positions 0..S"
        );
    }

    let reference = tensor(&records, &output_stage, output_heads, &device, false)?;
    let reference_values = reference.flatten_all()?.to_vec1::<f32>()?;

    math.check()?;
    for (implementation, result) in model.replay_common_input(op, layer.max(0) as usize, &inputs)? {
        ensure!(
            result.dims() == reference.dims(),
            "replay output shape mismatch"
        );
        let values = result
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        println!(
            "{}",
            json!({
                "kind": "operator_comparison",
                "evidence_class": "common_input_replay",
                "reference_evidence_class": "observed_execution",
                "token_sha256": digest,
                "layer_index": record_layer,
                "operator": op,
                "implementation": implementation,
                "reference_stage": output_stage,
                "shape": result.dims(),
                "math_mode": math.check()?,
                "norm_semantics": "Rust_separate_rounded_add_then_RMSNorm",
                "rotary_cache": "Rust_executed_cache",
                "metrics": comparison_metrics(&values, &reference_values)?
            })
        );
    }
    Ok(())
}
