use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

#[derive(Clone, Deserialize)]
pub struct EngineConfig {
    pub causal: bool,
    pub rope_layout: String,
    pub max_len: usize,
    pub max_tokens: usize,
    pub max_seqs: usize,
    pub wait_us: u64,
}

#[derive(Deserialize)]
struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,

    // Required: hidden_size / num_attention_heads is invalid here.
    head_dim: usize,

    vocab_size: usize,
    rms_norm_eps: f64,
    hidden_act: String,
    max_position_embeddings: usize,

    // Frozen checkpoint contract: bidirectional attention, mean pooling.
    is_causal: bool,
    pooling: String,
    pad_token_id: u32,

    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: Option<Value>,
    #[serde(default)]
    rope_scaling: Option<Value>,

    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    ada_rms_norm_t_cond: bool,
    #[serde(default)]
    llama_4_scaling: Option<Value>,
}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if !index.exists() {
        let p = dir.join("model.safetensors");
        ensure!(p.is_file(), "missing model.safetensors");
        return Ok(vec![p]);
    }

    let index: Index = serde_json::from_slice(&fs::read(index)?)?;
    let names: BTreeSet<_> = index.weight_map.into_values().collect();

    names.into_iter().map(|name| {
        let p = Path::new(&name);
        ensure!(
            p.components().count() == 1
                && matches!(p.components().next(), Some(Component::Normal(_))),
            "unsafe shard path: {name}"
        );
        let p = dir.join(p);
        ensure!(p.is_file(), "missing shard: {}", p.display());
        Ok(p)
    }).collect()
}

fn theta(c: &Config) -> Result<f64> {
    let mut result = c.rope_theta;

    for params in [&c.rope_scaling, &c.rope_parameters]
        .into_iter()
        .flatten()
        .filter(|p| !p.is_null())
    {
        let kind = params.get("rope_type")
            .or_else(|| params.get("type"))
            .and_then(Value::as_str);

        ensure!(
            kind == Some("default"),
            "unsupported RoPE configuration: {params}; implement and validate it"
        );

        if let Some(v) = params.get("rope_theta").and_then(Value::as_f64) {
            result = Some(v);
        }
    }

    let t = result.context("missing rope_theta")?;
    ensure!(t.is_finite() && t > 0.0, "invalid rope_theta");
    Ok(t)
}

fn rms(x: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let denom = xf.sqr()?
        .mean_keepdim(D::Minus1)?
        .affine(1.0, eps)?
        .sqrt()?;

    xf.broadcast_div(&denom)?
        .to_dtype(DType::BF16)?
        .broadcast_mul(weight)
}

// Flatten explicitly, so each projection is a conventional 2-D GEMM.
fn linear3(l: &Linear, x: &Tensor) -> candle_core::Result<Tensor> {
    let (b, s, h) = x.dims3()?;
    let y = l.forward(&x.reshape((b * s, h))?)?;
    let n = y.dim(1)?;
    y.reshape((b, s, n))
}

fn no_bias(vb: VarBuilder<'_>, name: &str, out: usize, input: usize)
    -> candle_core::Result<Linear>
{
    Ok(Linear::new(
        vb.get((out, input), &format!("{name}.weight"))?,
        None,
    ))
}

struct Layer {
    norm1: Tensor,
    norm2: Tensor,
    qkv: Linear,
    out: Linear,
    gate_up: Linear,
    down: Linear,
}

pub struct Model {
    device: Device,
    cfg: Config,
    engine: EngineConfig,
    embed: Embedding,
    layers: Vec<Layer>,
    final_norm: Tensor,
    cos: Tensor,
    sin: Tensor,
}

impl Model {
    pub fn load(dir: &Path, device: Device, engine: EngineConfig) -> Result<Self> {
        let cfg: Config =
            serde_json::from_slice(&fs::read(dir.join("config.json"))?)?;

        ensure!(
            (cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim)
                == (2048, 24, 128),
            "unexpected head geometry: this checkpoint has head_dim=128 \
             while hidden_size / num_attention_heads = 85.33"
        );
        ensure!(
            (cfg.num_hidden_layers, cfg.intermediate_size, cfg.vocab_size)
                == (16, 6144, 131072),
            "unexpected model topology"
        );
        ensure!(cfg.head_dim % 2 == 0, "invalid head_dim");
        ensure!(cfg.hidden_act == "silu", "only SiLU is implemented");
        ensure!(!cfg.is_causal, "this executor requires bidirectional attention");
        ensure!(!engine.causal, "engine.causal must be false");
        ensure!(cfg.pooling == "avg", "this executor requires avg pooling");
        ensure!(cfg.pad_token_id == 11, "unexpected padding token");
        ensure!(!cfg.attention_bias && !cfg.mlp_bias, "bias unsupported");
        ensure!(!cfg.ada_rms_norm_t_cond, "adaptive norm unsupported");
        ensure!(
            cfg.llama_4_scaling.as_ref().map_or(true, Value::is_null),
            "llama_4_scaling requires an implementation"
        );
        ensure!(engine.rope_layout == "split_half", "unsupported RoPE layout");
        ensure!(
            engine.max_len > 0 && engine.max_len <= cfg.max_position_embeddings,
            "invalid serving length"
        );

        let rope_theta = theta(&cfg)?;
        let files = weight_files(dir)?;

        // SAFETY: model files must remain immutable while the mmap is live.
        // Deploy the model directory read-only.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &files, DType::BF16, &device
            )?
        };
        let vb = vb.pp("model");

        let embed = candle_nn::embedding(
            cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens")
        )?;

        let q = cfg.num_attention_heads * cfg.head_dim;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

        for idx in 0..cfg.num_hidden_layers {
            let l = vb.pp(format!("layers.{idx}"));
            let a = l.pp("self_attn");
            let m = l.pp("mlp");

            let wq = a.get((q, h), "q_proj.weight")?;
            let wk = a.get((kv, h), "k_proj.weight")?;
            let wv = a.get((kv, h), "v_proj.weight")?;
            let wg = m.get((i, h), "gate_proj.weight")?;
            let wu = m.get((i, h), "up_proj.weight")?;

            layers.push(Layer {
                norm1: l.get(h, "input_layernorm.weight")?,
                norm2: l.get(h, "post_attention_layernorm.weight")?,
                qkv: Linear::new(Tensor::cat(&[&wq, &wk, &wv], 0)?, None),
                out: no_bias(a, "o_proj", h, q)?,
                gate_up: Linear::new(Tensor::cat(&[&wg, &wu], 0)?, None),
                down: no_bias(m, "down_proj", h, i)?,
            });
        }

        let final_norm = vb.get(h, "norm.weight")?;

        let half = cfg.head_dim / 2;
        let mut cos = Vec::with_capacity(engine.max_len * half);
        let mut sin = Vec::with_capacity(engine.max_len * half);
        for p in 0..engine.max_len {
            for j in 0..half {
                let inv = rope_theta.powf(-(2 * j) as f64 / cfg.head_dim as f64);
                let angle = p as f64 * inv;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }

        // FP32 RoPE arithmetic, cast back to BF16 afterward.
        let cos = Tensor::from_vec(cos, (engine.max_len, half), &device)?;
        let sin = Tensor::from_vec(sin, (engine.max_len, half), &device)?;

        Ok(Self {
            device, cfg, engine, embed, layers, final_norm, cos, sin,
        })
    }

    fn rope(&self, x: &Tensor, s: usize) -> candle_core::Result<Tensor> {
        let half = self.cfg.head_dim / 2;
        let x = x.to_dtype(DType::F32)?;
        let a = x.narrow(3, 0, half)?;
        let b = x.narrow(3, half, half)?;

        let cos = self.cos.narrow(0, 0, s)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, 0, s)?.unsqueeze(0)?.unsqueeze(0)?;

        let left = (a.broadcast_mul(&cos)? - b.broadcast_mul(&sin)?)?;
        let right = (b.broadcast_mul(&cos)? + a.broadcast_mul(&sin)?)?;

        Tensor::cat(&[&left, &right], 3)?.to_dtype(DType::BF16)
    }

    fn repeat_kv(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, nkv, s, d) = x.dims4()?;
        let groups = self.cfg.num_attention_heads / nkv;
        x.unsqueeze(2)?
            .broadcast_as((b, nkv, groups, s, d))?
            .contiguous()?
            .reshape((b, nkv * groups, s, d))
    }

    pub fn forward(&self, sequences: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        ensure!(!sequences.is_empty(), "empty batch");
        let b = sequences.len();
        let s = sequences.iter().map(Vec::len).max().unwrap();
        ensure!(s <= self.engine.max_len, "sequence too long");
        ensure!(sequences.iter().all(|x| !x.is_empty()), "empty sequence");

        let mut ids = vec![self.cfg.pad_token_id; b * s];
        for (row, seq) in sequences.iter().enumerate() {
            ensure!(
                seq.iter().all(|&id| (id as usize) < self.cfg.vocab_size),
                "token ID outside vocabulary"
            );
            ensure!(
                seq.iter().all(|&id| id != self.cfg.pad_token_id),
                "input sequences must be unpadded and must not contain pad tokens"
            );
            ids[row * s..row * s + seq.len()].copy_from_slice(seq);
        }
        let ids = Tensor::from_vec(ids, (b, s), &self.device)?;
        let mut x = self.embed.forward(&ids)?;

        // Correctness path only: O(B*S*S) mask and attention scores.
        // Bidirectional: only padding is masked. Every padded query still
        // has valid keys, avoiding all-masked rows.
        let mut mask = vec![0f32; b * s * s];
        for (row, seq) in sequences.iter().enumerate() {
            for query in 0..s {
                for key in 0..s {
                    if key >= seq.len() {
                        mask[(row * s + query) * s + key] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let mask = Tensor::from_vec(mask, (b, 1, s, s), &self.device)?;

        let nq = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let d = self.cfg.head_dim;
        let q_width = nq * d;
        let kv_width = nkv * d;

        for layer in &self.layers {
            let n = rms(&x, &layer.norm1, self.cfg.rms_norm_eps)?;
            let qkv = linear3(&layer.qkv, &n)?;

            let q = qkv.narrow(2, 0, q_width)?
                .reshape((b, s, nq, d))?
                .transpose(1, 2)?;
            let k = qkv.narrow(2, q_width, kv_width)?
                .reshape((b, s, nkv, d))?
                .transpose(1, 2)?;
            let v = qkv.narrow(2, q_width + kv_width, kv_width)?
                .reshape((b, s, nkv, d))?
                .transpose(1, 2)?;

            let q = self.rope(&q, s)?.contiguous()?;
            let k = self.repeat_kv(&self.rope(&k, s)?)?;
            let v = self.repeat_kv(&v)?;

            let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?
                .to_dtype(DType::F32)?
                .affine(1.0 / (d as f64).sqrt(), 0.0)?
                .broadcast_add(&mask)?;

            let probabilities = candle_nn::ops::softmax_last_dim(&scores)?
                .to_dtype(DType::BF16)?;

            let attention = probabilities.matmul(&v)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, q_width))?;

            x = (&x + &linear3(&layer.out, &attention)?)?;

            let n = rms(&x, &layer.norm2, self.cfg.rms_norm_eps)?;
            let gu = linear3(&layer.gate_up, &n)?;
            let gate = gu.narrow(2, 0, self.cfg.intermediate_size)?;
            let up = gu.narrow(
                2, self.cfg.intermediate_size, self.cfg.intermediate_size
            )?;
            let activated = candle_nn::Activation::Silu.forward(&gate)?;
            let mlp = (&activated * &up)?;
            x = (&x + &linear3(&layer.down, &mlp)?)?;
        }

        // Final RMSNorm is part of the decoder, before the embedding pooler.
        // It cannot be moved after an average.
        let x = rms(&x, &self.final_norm, self.cfg.rms_norm_eps)?
            .to_dtype(DType::F32)?;

        let mut rows = Vec::with_capacity(b);
        for (row, seq) in sequences.iter().enumerate() {
            let valid = x.i(row)?.narrow(0, 0, seq.len())?;
            let mean = valid
                .sum(0)?
                .affine(1.0 / seq.len() as f64, 0.0)?;
            rows.push(mean.unsqueeze(0)?);
        }
        let pooled = Tensor::cat(&rows, 0)?;
        let norm = pooled.sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?;

        Ok(pooled.broadcast_div(&norm)?.to_vec2::<f32>()?)
    }
}
