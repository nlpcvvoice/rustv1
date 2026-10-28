use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

pub const BATCHED_EXECUTION_POLICY: &str = "batched_execution_v2";
pub const BATCHED_HTTP_CONTRACT: &str = "m4-http-batched-F-v1";
pub const MAX_PHYSICAL_BATCH: usize = 32;
pub const MAX_PADDED_TOKENS: usize = 32_768;
pub const MAX_ATTENTION_CELLS: usize = 33_554_432;

#[derive(Clone, Deserialize)]
pub struct EngineConfig {
    pub causal: bool,
    pub rope_layout: String,
    pub max_len: usize,
    pub max_tokens: usize,
    pub max_seqs: usize,
    pub wait_us: u64,
}

/// Immutable limits for the explicit batched route.
///
/// Padded tokens count B*S. Attention cells count B*S*S, before heads.
/// These are qualification ceilings, not estimates of available GPU memory.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BatchedExecution {
    pub max_batch_size: usize,
    pub max_padded_tokens: usize,
    pub max_attention_cells: usize,
    pub max_wait_us: u64,
}

impl Default for BatchedExecution {
    fn default() -> Self {
        Self {
            max_batch_size: MAX_PHYSICAL_BATCH,
            max_padded_tokens: MAX_PADDED_TOKENS,
            max_attention_cells: MAX_ATTENTION_CELLS,
            max_wait_us: 1_000,
        }
    }
}

impl BatchedExecution {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (2..=MAX_PHYSICAL_BATCH).contains(&self.max_batch_size),
            "batched max_batch_size must be in 2..=32"
        );
        ensure!(
            (1..=MAX_PADDED_TOKENS).contains(&self.max_padded_tokens),
            "invalid padded-token budget"
        );
        ensure!(
            (1..=MAX_ATTENTION_CELLS).contains(&self.max_attention_cells),
            "invalid attention-cell budget"
        );
        ensure!(self.max_wait_us <= 2_000, "batch wait exceeds 2000 us");
        Ok(())
    }

    pub fn validate_engine(&self, engine: &EngineConfig) -> Result<()> {
        self.validate()?;
        ensure!(
            engine.max_len > 0 && engine.max_len <= 1024,
            "invalid engine max_len"
        );
        ensure!(
            self.max_batch_size <= engine.max_seqs,
            "batch limit exceeds engine.max_seqs"
        );
        ensure!(
            self.max_padded_tokens <= engine.max_tokens,
            "padded-token limit exceeds engine.max_tokens"
        );
        ensure!(
            self.max_padded_tokens >= engine.max_len,
            "one maximum-length sequence must fit"
        );
        let one_attention = engine
            .max_len
            .checked_mul(engine.max_len)
            .context("attention size overflow")?;
        ensure!(
            self.max_attention_cells >= one_attention,
            "one maximum-length attention matrix must fit"
        );
        ensure!(
            self.max_wait_us == engine.wait_us,
            "batch wait differs from engine.wait_us"
        );
        Ok(())
    }

    pub fn fits(&self, b: usize, s: usize) -> bool {
        if b == 0 || b > self.max_batch_size || s == 0 || s > 1024 {
            return false;
        }
        let Some(tokens) = b.checked_mul(s) else {
            return false;
        };
        let Some(cells) = tokens.checked_mul(s) else {
            return false;
        };
        tokens <= self.max_padded_tokens && cells <= self.max_attention_cells
    }
}

/// Server-assigned request identity plus the original input position.
///
/// request_id is unique within the server instance. case_id is only a
/// correlation label; it is not an authorization or uniqueness mechanism.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionRow {
    pub request_id: u64,
    pub request_row: usize,
    pub case_id: Option<String>,
}

impl ExecutionRow {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn validate(&self) -> Result<()> {
        if let Some(case) = &self.case_id {
            ensure!(
                !case.is_empty()
                    && case.len() <= 128
                    && case.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'),
                "invalid execution case ID"
            );
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    hidden_act: String,
    max_position_embeddings: usize,
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

fn semantic_config_check(raw: &Value) -> Result<()> {
    let obj = raw.as_object().context("config.json must be an object")?;
    const CONSUMED: &[&str] = &[
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "rms_norm_eps",
        "hidden_act",
        "max_position_embeddings",
        "is_causal",
        "pooling",
        "pad_token_id",
        "rope_theta",
        "rope_parameters",
        "rope_scaling",
        "attention_bias",
        "mlp_bias",
        "ada_rms_norm_t_cond",
        "llama_4_scaling",
    ];
    for (key, value) in obj {
        if CONSUMED.contains(&key.as_str()) {
            continue;
        }
        match key.as_str() {
            "sliding_window" => ensure!(
                value.is_null(),
                "sliding_window={value} changes attention semantics"
            ),
            "attention_dropout" => ensure!(
                value.is_null() || value.as_f64() == Some(0.0),
                "nonzero attention_dropout is unsupported"
            ),
            "bos_token_id" => ensure!(value.as_u64() == Some(1), "unexpected bos_token_id {value}"),
            "eos_token_id" => ensure!(value.as_u64() == Some(2), "unexpected eos_token_id {value}"),
            "dtype" => ensure!(
                value.as_str() == Some("bfloat16"),
                "unexpected dtype {value}"
            ),
            "use_cache"
            | "tie_word_embeddings"
            | "initializer_range"
            | "architectures"
            | "model_type"
            | "nemo_version"
            | "transformers_version" => {}
            other => bail!("unknown config key '{other}': semantic manifest is not closed"),
        }
    }
    for (field, known) in [
        (
            "rope_parameters",
            &[
                "rope_type",
                "type",
                "rope_theta",
                "factor",
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "apply_yarn_scaling",
                "attn_factor",
                "llama_4_scaling_beta",
                "mscale",
                "mscale_all_dim",
            ][..],
        ),
        (
            "llama_4_scaling",
            &["beta", "original_max_position_embeddings"][..],
        ),
    ] {
        if let Some(v) = obj.get(field) {
            if let Some(inner) = v.as_object() {
                for (key, value) in inner {
                    ensure!(
                        known.contains(&key.as_str()),
                        "unknown {field}.{key}: nested semantic manifest is not closed"
                    );
                    if field == "rope_parameters" {
                        match key.as_str() {
                            "llama_4_scaling_beta" => ensure!(
                                value.as_f64() == Some(0.1),
                                "unexpected rope_parameters.llama_4_scaling_beta {value}"
                            ),
                            "mscale" | "mscale_all_dim" => ensure!(
                                value.as_f64() == Some(1.0),
                                "non-neutral {field}.{key}={value} while apply_yarn_scaling=false"
                            ),
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    Ok(())
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

    names
        .into_iter()
        .map(|name| {
            let p = Path::new(&name);
            ensure!(
                p.components().count() == 1
                    && matches!(p.components().next(), Some(Component::Normal(_))),
                "unsafe shard path: {name}"
            );
            let p = dir.join(p);
            ensure!(p.is_file(), "missing shard: {}", p.display());
            Ok(p)
        })
        .collect()
}

fn theta(c: &Config) -> Result<f64> {
    let mut found = 0usize;
    if let Some(t) = c.rope_theta {
        found += 1;
        ensure!(t == 1_000_000.0, "unexpected top-level rope_theta {t}");
    }
    for params in [&c.rope_scaling, &c.rope_parameters]
        .into_iter()
        .flatten()
        .filter(|p| !p.is_null())
    {
        if let Some(v) = params.get("rope_theta").filter(|v| !v.is_null()) {
            found += 1;
            let t = v
                .as_f64()
                .context("every declared rope_theta must be numeric")?;
            ensure!(t == 1_000_000.0, "unexpected nested rope_theta {t}");
        }
    }
    ensure!(found > 0, "at least one rope_theta declaration is required");
    Ok(1_000_000.0)
}

pub struct YarnParams {
    pub factor: f64,
    pub original_max: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

fn yarn_params(c: &Config) -> Result<YarnParams> {
    ensure!(
        c.rope_scaling.as_ref().is_none_or(Value::is_null),
        "rope_scaling must be absent for the frozen contract"
    );
    let params = c
        .rope_parameters
        .as_ref()
        .filter(|p| !p.is_null() && p.is_object())
        .context("frozen contract requires an object rope_parameters")?;

    let kind = match params.get("rope_type") {
        Some(v) => Some(v.as_str().context("rope_type must be a string")?),
        None => None,
    };
    let alias = match params.get("type") {
        Some(v) => Some(v.as_str().context("type must be a string")?),
        None => None,
    };
    if let (Some(a), Some(b)) = (kind, alias) {
        ensure!(a == b, "conflicting rope_type/type discriminators");
    }
    let kind = kind
        .or(alias)
        .context("rope_parameters missing discriminator")?;

    match kind {
        "yarn" => {}
        other => bail!("frozen contract requires rope_type yarn, got {other:?}"),
    }

    ensure!(
        params.get("apply_yarn_scaling").and_then(Value::as_bool) == Some(false),
        "apply_yarn_scaling=true is unsupported by the frozen contract"
    );
    let attn_factor = match params.get("attn_factor") {
        None => 1.0,
        Some(v) => v.as_f64().context("attn_factor must be numeric")?,
    };
    ensure!(attn_factor == 1.0, "non-unit attn_factor unsupported");
    let extrapolation_factor = match params.get("extrapolation_factor") {
        None => 1.0,
        Some(v) => v.as_f64().context("extrapolation_factor must be numeric")?,
    };
    ensure!(
        extrapolation_factor == 1.0,
        "non-unit extrapolation_factor unsupported"
    );
    let truncate = match params.get("truncate") {
        None => true,
        Some(v) => v.as_bool().context("truncate must be boolean")?,
    };
    ensure!(truncate, "truncate=false unsupported");
    let y = YarnParams {
        factor: params
            .get("factor")
            .and_then(Value::as_f64)
            .context("yarn missing factor")?,
        original_max: params
            .get("original_max_position_embeddings")
            .and_then(Value::as_f64)
            .context("yarn missing original_max_position_embeddings")?,
        beta_fast: params
            .get("beta_fast")
            .and_then(Value::as_f64)
            .context("yarn missing beta_fast")?,
        beta_slow: params
            .get("beta_slow")
            .and_then(Value::as_f64)
            .context("yarn missing beta_slow")?,
    };
    ensure!(
        y.factor == 16.0 && y.original_max == 16_384.0 && y.beta_fast == 32.0 && y.beta_slow == 1.0,
        "YaRN parameters differ from the frozen contract"
    );
    Ok(y)
}

fn expected_manifest() -> BTreeMap<String, (Vec<usize>, &'static str)> {
    let mut m = BTreeMap::new();
    m.insert("embed_tokens.weight".into(), (vec![131072, 2048], "BF16"));
    m.insert("norm.weight".into(), (vec![2048], "BF16"));
    for l in 0..16 {
        let shapes: [(&str, Vec<usize>); 9] = [
            ("input_layernorm.weight", vec![2048]),
            ("post_attention_layernorm.weight", vec![2048]),
            ("self_attn.q_proj.weight", vec![3072, 2048]),
            ("self_attn.k_proj.weight", vec![1024, 2048]),
            ("self_attn.v_proj.weight", vec![1024, 2048]),
            ("self_attn.o_proj.weight", vec![2048, 3072]),
            ("mlp.gate_proj.weight", vec![6144, 2048]),
            ("mlp.up_proj.weight", vec![6144, 2048]),
            ("mlp.down_proj.weight", vec![2048, 6144]),
        ];
        for (name, shape) in shapes {
            m.insert(format!("layers.{l}.{name}"), (shape, "BF16"));
        }
    }
    m
}

fn validate_manifest(files: &[PathBuf]) -> Result<()> {
    let expected = expected_manifest();
    let mut seen: BTreeMap<String, (Vec<usize>, String)> = BTreeMap::new();

    for file in files {
        let bytes = fs::read(file)?;
        ensure!(bytes.len() > 8, "truncated safetensors: {}", file.display());
        let n = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
        let end = 8usize.checked_add(n).context("header size overflow")?;
        ensure!(end <= bytes.len(), "truncated header: {}", file.display());
        let header: BTreeMap<String, Value> = serde_json::from_slice(&bytes[8..end])?;

        for (name, entry) in header {
            if name == "__metadata__" {
                continue;
            }
            let shape = entry
                .get("shape")
                .and_then(Value::as_array)
                .context("tensor missing shape")?
                .iter()
                .map(|v| v.as_u64().and_then(|v| usize::try_from(v).ok()))
                .collect::<Option<Vec<_>>>()
                .context("malformed shape")?;
            let dtype = entry
                .get("dtype")
                .and_then(Value::as_str)
                .context("tensor missing dtype")?
                .to_string();
            ensure!(
                seen.insert(name.clone(), (shape, dtype)).is_none(),
                "duplicate tensor across shards: {name}"
            );
        }
    }

    ensure!(
        seen.len() == expected.len(),
        "manifest size {} != expected {}",
        seen.len(),
        expected.len()
    );
    for (name, (shape, dtype)) in &expected {
        let (got_shape, got_dtype) = seen
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            got_shape == shape,
            "shape mismatch {name}: {got_shape:?} != {shape:?}"
        );
        ensure!(
            got_dtype == dtype,
            "dtype mismatch {name}: {got_dtype} != {dtype}"
        );
    }
    Ok(())
}

fn linear3(l: &Linear, x: &Tensor) -> candle_core::Result<Tensor> {
    let (b, s, h) = x.dims3()?;
    let y = l.forward(&x.reshape((b * s, h))?)?;
    let n = y.dim(1)?;
    y.reshape((b, s, n))
}

fn no_bias(
    vb: VarBuilder<'_>,
    name: &str,
    out: usize,
    input: usize,
) -> candle_core::Result<Linear> {
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

pub fn build_rope_tables(
    theta: f64,
    yarn: Option<&YarnParams>,
    head_dim: usize,
    max_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let inv_freq: Vec<f64> = match yarn {
        None => (0..half)
            .map(|j| theta.powf(-((2 * j) as f64) / head_dim as f64))
            .collect(),
        Some(y) => crate::yarn::standard_yarn_inv_freq(
            head_dim,
            theta,
            y.factor,
            y.original_max,
            y.beta_fast,
            y.beta_slow,
        )
        .into_iter()
        .map(f64::from)
        .collect(),
    };

    let mut cos = Vec::with_capacity(max_len * half);
    let mut sin = Vec::with_capacity(max_len * half);
    for p in 0..max_len {
        for &inv in &inv_freq {
            let angle = p as f64 * inv;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

pub type StageSink<'a> = dyn FnMut(isize, &str, &Tensor) -> Result<()> + 'a;

fn emit_stage(
    sink: &mut Option<&mut StageSink<'_>>,
    layer: isize,
    name: &str,
    tensor: &Tensor,
) -> Result<()> {
    if let Some(callback) = sink.as_mut() {
        callback(layer, name, tensor)?;
    }
    Ok(())
}

pub struct Model {
    device: Device,
    dtype: DType,
    cfg: Config,
    engine: EngineConfig,
    embed: Embedding,
    layers: Vec<Layer>,
    final_norm: Tensor,
    cos: Tensor,
    sin: Tensor,
    llama4: Option<(f64, f64)>,
    precision: crate::precision::PrecisionProfile,
}

impl Model {
    pub fn load(dir: &Path, device: Device, engine: EngineConfig, dtype: DType) -> Result<Self> {
        let config_bytes = fs::read(dir.join("config.json"))?;
        semantic_config_check(&serde_json::from_slice(&config_bytes)?)?;
        let cfg: Config = serde_json::from_slice(&config_bytes)?;

        ensure!(
            (cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim) == (2048, 24, 128),
            "unexpected head geometry"
        );
        ensure!(
            (cfg.num_hidden_layers, cfg.intermediate_size, cfg.vocab_size) == (16, 6144, 131072),
            "unexpected model topology"
        );
        ensure!(cfg.head_dim.is_multiple_of(2), "invalid head_dim");
        ensure!(cfg.num_key_value_heads == 8, "unexpected KV head count");
        ensure!(
            cfg.num_attention_heads
                .is_multiple_of(cfg.num_key_value_heads),
            "invalid GQA geometry"
        );
        ensure!(cfg.rms_norm_eps == 1e-5, "unexpected rms_norm_eps");
        ensure!(cfg.hidden_act == "silu", "only SiLU is implemented");
        ensure!(
            !cfg.is_causal,
            "this executor requires bidirectional attention"
        );
        ensure!(!engine.causal, "engine.causal must be false");
        ensure!(cfg.pooling == "avg", "this executor requires avg pooling");
        ensure!(cfg.pad_token_id == 11, "unexpected padding token");
        ensure!(!cfg.attention_bias && !cfg.mlp_bias, "bias unsupported");
        ensure!(!cfg.ada_rms_norm_t_cond, "adaptive norm unsupported");
        let yarn = yarn_params(&cfg)?;
        let l4 = cfg
            .llama_4_scaling
            .as_ref()
            .filter(|v| !v.is_null() && v.is_object())
            .context("frozen contract requires llama_4_scaling")?;
        ensure!(
            l4.get("beta").and_then(Value::as_f64) == Some(0.1)
                && l4
                    .get("original_max_position_embeddings")
                    .and_then(Value::as_f64)
                    == Some(16384.0),
            "llama_4_scaling differs from the frozen contract"
        );
        let llama4 = Some((
            l4.get("beta").and_then(Value::as_f64).unwrap(),
            l4.get("original_max_position_embeddings")
                .and_then(Value::as_f64)
                .unwrap(),
        ));
        ensure!(
            engine.rope_layout == "split_half",
            "unsupported RoPE layout"
        );
        ensure!(
            engine.max_len > 0 && engine.max_len <= cfg.max_position_embeddings,
            "invalid serving length"
        );
        ensure!(engine.max_len <= 1024, "validates at most 1024 tokens");

        let rope_theta = theta(&cfg)?;
        let files = weight_files(dir)?;
        validate_manifest(&files)?;

        // SAFETY: checkpoint files must remain immutable while mappings live.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

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
        let (cos_v, sin_v) =
            build_rope_tables(rope_theta, Some(&yarn), cfg.head_dim, engine.max_len);
        let half = cfg.head_dim / 2;
        let cos = Tensor::from_vec(cos_v, (engine.max_len, half), &device)?;
        let sin = Tensor::from_vec(sin_v, (engine.max_len, half), &device)?;

        Ok(Self {
            device,
            dtype,
            cfg,
            engine,
            embed,
            layers,
            final_norm,
            cos,
            sin,
            llama4,
            precision: crate::precision::PrecisionProfile::A,
        })
    }

    pub fn with_precision(mut self, profile: crate::precision::PrecisionProfile) -> Self {
        self.precision = profile;
        self
    }

    pub fn replay_common_input(
        &self,
        operator: &str,
        layer_index: usize,
        inputs: &[Tensor],
    ) -> Result<Vec<(String, Tensor)>> {
        use crate::precision::PrecisionProfile;

        ensure!(
            self.precision == PrecisionProfile::F,
            "replay requires profile F"
        );
        ensure!(
            self.dtype == DType::BF16,
            "replay requires BF16 model storage"
        );
        ensure!(!inputs.is_empty(), "missing replay input");

        let layer = if operator == "final_rms" {
            None
        } else {
            Some(self.layers.get(layer_index).context("layer out of range")?)
        };

        let one = || -> Result<&Tensor> {
            ensure!(inputs.len() == 1, "{operator}: expected one input");
            Ok(&inputs[0])
        };

        let norm_input = || -> Result<Tensor> {
            ensure!(
                inputs.len() == 1 || inputs.len() == 2,
                "{operator}: expected hidden, optionally residual"
            );
            ensure!(inputs[0].dtype() == DType::BF16, "norm input must be BF16");
            if inputs.len() == 2 {
                ensure!(inputs[1].dtype() == DType::BF16, "residual must be BF16");
                Ok((&inputs[0] + &inputs[1])?)
            } else {
                Ok(inputs[0].clone())
            }
        };

        let result = match operator {
            "input_rms" | "post_attention_rms" | "final_rms" => {
                let x = norm_input()?;
                let weight = match operator {
                    "input_rms" => &layer.unwrap().norm1,
                    "post_attention_rms" => &layer.unwrap().norm2,
                    _ => &self.final_norm,
                };
                crate::precision::rms(&x, weight, self.cfg.rms_norm_eps, PrecisionProfile::F)?
            }
            "qkv" => linear3(&layer.unwrap().qkv, one()?)?,
            "o_proj" => linear3(&layer.unwrap().out, one()?)?,
            "gate_up" => linear3(&layer.unwrap().gate_up, one()?)?,
            "down" => linear3(&layer.unwrap().down, one()?)?,
            "silu_mul" => {
                let gu = one()?;
                let width = self.cfg.intermediate_size;
                ensure!(gu.dim(2)? == 2 * width, "wrong gate/up width");
                crate::precision::silu_mul(
                    &gu.narrow(2, 0, width)?,
                    &gu.narrow(2, width, width)?,
                    PrecisionProfile::F,
                )?
            }
            "rope_q" | "rope_k" => {
                let x = one()?;
                let (b, heads, s, d) = x.dims4()?;
                let expected_heads = if operator == "rope_q" {
                    self.cfg.num_attention_heads
                } else {
                    self.cfg.num_key_value_heads
                };
                ensure!(
                    b == 1 && heads == expected_heads && d == self.cfg.head_dim,
                    "wrong rotary geometry"
                );
                ensure!(s > 0 && s <= self.engine.max_len, "wrong rotary length");
                self.rope(x, s)?
            }
            "attention" => {
                ensure!(inputs.len() == 3, "attention requires Q, K, V");
                let q = &inputs[0];
                let k = &inputs[1];
                let v = &inputs[2];
                let (b, heads, s, d) = q.dims4()?;
                ensure!(
                    b == 1 && heads == self.cfg.num_attention_heads && d == self.cfg.head_dim,
                    "wrong Q geometry"
                );
                ensure!(
                    k.dims() == [1, self.cfg.num_key_value_heads, s, d] && v.dims() == k.dims(),
                    "K/V must be captured before GQA repetition"
                );
                let k = self.repeat_kv(k)?;
                let v = self.repeat_kv(v)?;
                let mask = Tensor::zeros((1, 1, s, s), DType::F32, &self.device)?;
                let actual = crate::precision::attention(q, &k, &v, &mask, PrecisionProfile::F)?;
                let oracle = crate::precision::attention(
                    &q.to_dtype(DType::F32)?,
                    &k.to_dtype(DType::F32)?,
                    &v.to_dtype(DType::F32)?,
                    &mask,
                    PrecisionProfile::F,
                )?;
                let flatten = |x: Tensor| -> candle_core::Result<Tensor> {
                    x.transpose(1, 2)?.contiguous()?.reshape((1, s, heads * d))
                };
                return Ok(vec![
                    ("profile_F".to_owned(), flatten(actual)?),
                    ("dense_fp32_oracle".to_owned(), flatten(oracle)?),
                ]);
            }
            other => bail!("unsupported replay operator {other}"),
        };

        Ok(vec![("profile_F".to_owned(), result)])
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

        Tensor::cat(&[&left, &right], 3)?.to_dtype(self.dtype)
    }

    fn repeat_kv(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, nkv, s, d) = x.dims4()?;
        let groups = self.cfg.num_attention_heads / nkv;
        x.unsqueeze(2)?
            .broadcast_as((b, nkv, groups, s, d))?
            .contiguous()?
            .reshape((b, nkv * groups, s, d))
    }

    fn run_decoder(&self, sequences: &[Vec<u32>]) -> Result<Tensor> {
        ensure!(!sequences.is_empty(), "empty batch");
        let s = sequences.iter().map(Vec::len).max().unwrap();
        self.run_decoder_padded(sequences, s, None)
    }

    pub fn run_decoder_dump(
        &self,
        sequences: &[Vec<u32>],
        padded_s: usize,
        sink: &mut StageSink<'_>,
    ) -> Result<Tensor> {
        self.run_decoder_padded(sequences, padded_s, Some(sink))
    }

    /// Right padding; all queries mask invalid keys in every layer.
    /// Padded query outputs are discarded, not all-masked.
    ///
    /// The arithmetic below is retained for the B=1 exact-bit route.
    /// Serving capacity checks are performed by IndependentExecutor.
    fn run_decoder_padded(
        &self,
        sequences: &[Vec<u32>],
        padded_s: usize,
        mut dump: Option<&mut StageSink<'_>>,
    ) -> Result<Tensor> {
        ensure!(!sequences.is_empty(), "empty batch");
        let b = sequences.len();
        let s = padded_s;
        ensure!(s > 0 && s <= self.engine.max_len, "invalid sequence width");
        ensure!(sequences.iter().all(|x| !x.is_empty()), "empty sequence");
        ensure!(sequences.iter().all(|x| x.len() <= s), "padded_s too small");
        let padded_tokens = b.checked_mul(s).context("padded size overflow")?;
        let attention_cells = padded_tokens.checked_mul(s).context("mask size overflow")?;

        let mut ids = vec![self.cfg.pad_token_id; padded_tokens];
        for (row, seq) in sequences.iter().enumerate() {
            ensure!(
                seq.iter().all(|&id| (id as usize) < self.cfg.vocab_size),
                "token ID outside vocabulary"
            );
            ensure!(
                seq.iter().all(|&id| id != self.cfg.pad_token_id),
                "input sequences must not contain padding tokens"
            );
            ids[row * s..row * s + seq.len()].copy_from_slice(seq);
        }

        let ids = Tensor::from_vec(ids, (b, s), &self.device)?;
        let mut x = self.embed.forward(&ids)?;
        emit_stage(&mut dump, -1, "embed_tokens", &x)?;

        let mut mask = vec![0f32; attention_cells];
        for (row, seq) in sequences.iter().enumerate() {
            for query in 0..s {
                for key in seq.len()..s {
                    mask[(row * s + query) * s + key] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (b, 1, s, s), &self.device)?;

        let nq = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let d = self.cfg.head_dim;
        let q_width = nq * d;
        let kv_width = nkv * d;

        let mut previous_branch: Option<Tensor> = None;
        let mut previous_residual: Option<Tensor> = None;

        for (li, layer) in self.layers.iter().enumerate() {
            let li = li as isize;

            match (&previous_branch, &previous_residual) {
                (Some(branch), Some(residual)) => {
                    emit_stage(&mut dump, li, "input_layernorm.hidden_in", branch)?;
                    emit_stage(&mut dump, li, "input_layernorm.residual_in", residual)?;
                }
                (None, None) => {
                    emit_stage(&mut dump, li, "input_layernorm.hidden_in", &x)?;
                }
                _ => bail!("inconsistent residual observation state"),
            }

            let n = crate::precision::rms(&x, &layer.norm1, self.cfg.rms_norm_eps, self.precision)?;
            emit_stage(&mut dump, li, "input_layernorm.normalized", &n)?;
            emit_stage(&mut dump, li, "input_layernorm.residual", &x)?;

            let qkv = linear3(&layer.qkv, &n)?;
            emit_stage(&mut dump, li, "qkv.pre_rope", &qkv)?;

            let q = qkv
                .narrow(2, 0, q_width)?
                .reshape((b, s, nq, d))?
                .transpose(1, 2)?;
            let k = qkv
                .narrow(2, q_width, kv_width)?
                .reshape((b, s, nkv, d))?
                .transpose(1, 2)?;
            let v = qkv
                .narrow(2, q_width + kv_width, kv_width)?
                .reshape((b, s, nkv, d))?
                .transpose(1, 2)?;

            emit_stage(&mut dump, li, "q.pre_rope", &q)?;
            emit_stage(&mut dump, li, "k.pre_rope", &k)?;
            emit_stage(&mut dump, li, "v.pre_rope", &v)?;

            if dump.is_some() {
                let positions = Tensor::from_vec(
                    (0..s).map(|p| p as u32).collect::<Vec<_>>(),
                    s,
                    &self.device,
                )?;
                let cos = self.cos.narrow(0, 0, s)?;
                let sin = self.sin.narrow(0, 0, s)?;
                let cache = Tensor::cat(&[&cos, &sin], 1)?;
                emit_stage(&mut dump, li, "positions", &positions)?;
                emit_stage(&mut dump, li, "rope.cos_sin_cache", &cache)?;
            }

            let q = self.rope(&q, s)?.contiguous()?;
            let k = self.rope(&k, s)?;
            emit_stage(&mut dump, li, "q.post_rope", &q)?;
            emit_stage(&mut dump, li, "k.post_rope", &k)?;

            let q = match self.llama4 {
                Some((beta, original_max)) => {
                    let scales: Vec<f32> = (0..s)
                        .map(|p| crate::yarn::llama4_attn_scale(beta, original_max, p) as f32)
                        .collect();
                    let scale = Tensor::from_vec(scales, (1, 1, s, 1), &self.device)?;
                    q.to_dtype(DType::F32)?
                        .broadcast_mul(&scale)?
                        .to_dtype(self.dtype)?
                }
                None => q,
            };

            emit_stage(&mut dump, li, "q.attention_input", &q)?;
            emit_stage(&mut dump, li, "k.attention_input", &k)?;
            emit_stage(&mut dump, li, "v.attention_input", &v)?;

            let k = self.repeat_kv(&k)?;
            let v = self.repeat_kv(&v)?;
            let context = crate::precision::attention(&q, &k, &v, &mask, self.precision)?;
            let attention = context
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, q_width))?;

            emit_stage(&mut dump, li, "attention.pre_o_proj", &attention)?;
            let o = linear3(&layer.out, &attention)?;
            emit_stage(&mut dump, li, "o_proj", &o)?;

            emit_stage(&mut dump, li, "post_attention_layernorm.hidden_in", &o)?;
            emit_stage(&mut dump, li, "post_attention_layernorm.residual_in", &x)?;
            x = (&x + &o)?;
            emit_stage(&mut dump, li, "residual.post_attention", &x)?;

            let n = crate::precision::rms(&x, &layer.norm2, self.cfg.rms_norm_eps, self.precision)?;
            emit_stage(&mut dump, li, "post_attention_layernorm.normalized", &n)?;
            emit_stage(&mut dump, li, "post_attention_layernorm.residual", &x)?;

            let gu = linear3(&layer.gate_up, &n)?;
            emit_stage(&mut dump, li, "mlp.gate_up", &gu)?;
            let gate = gu.narrow(2, 0, self.cfg.intermediate_size)?;
            let up = gu.narrow(2, self.cfg.intermediate_size, self.cfg.intermediate_size)?;
            let mlp = crate::precision::silu_mul(&gate, &up, self.precision)?;
            emit_stage(&mut dump, li, "mlp.silu_mul", &mlp)?;
            let down = linear3(&layer.down, &mlp)?;
            emit_stage(&mut dump, li, "mlp.down", &down)?;

            previous_residual = Some(x.clone());
            previous_branch = Some(down.clone());
            x = (&x + &down)?;
            emit_stage(&mut dump, li, "residual.post_down", &x)?;
        }

        emit_stage(
            &mut dump,
            -1,
            "norm.hidden_in",
            previous_branch.as_ref().context("missing final branch")?,
        )?;
        emit_stage(
            &mut dump,
            -1,
            "norm.residual_in",
            previous_residual
                .as_ref()
                .context("missing final residual")?,
        )?;

        let normalized =
            crate::precision::rms(&x, &self.final_norm, self.cfg.rms_norm_eps, self.precision)?;
        emit_stage(&mut dump, -1, "norm.residual", &x)?;
        emit_stage(&mut dump, -1, "norm.normalized", &normalized)?;

        let states = normalized.to_dtype(DType::F32)?;
        emit_stage(&mut dump, -1, "final_norm", &states)?;
        Ok(states)
    }

    pub fn forward(&self, sequences: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        let pooled = self.pooled_mean(sequences)?;
        let norm = pooled
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?;
        Ok(pooled.broadcast_div(&norm)?.to_vec2::<f32>()?)
    }

    pub fn pooled_mean(&self, sequences: &[Vec<u32>]) -> Result<Tensor> {
        let x = self.run_decoder(sequences)?;
        let mut rows = Vec::with_capacity(sequences.len());
        for (row, seq) in sequences.iter().enumerate() {
            let valid = x.i(row)?.narrow(0, 0, seq.len())?;
            let mean = valid.sum(0)?.affine(1.0 / seq.len() as f64, 0.0)?;
            rows.push(mean.unsqueeze(0)?);
        }
        Ok(Tensor::cat(&rows, 0)?)
    }

    pub fn forward_with_padded_s(
        &self,
        sequences: &[Vec<u32>],
        padded_s: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let x = self.run_decoder_padded(sequences, padded_s, None)?;
        let mut rows = Vec::with_capacity(sequences.len());
        for (row, seq) in sequences.iter().enumerate() {
            let valid = x.i(row)?.narrow(0, 0, seq.len())?;
            let mean = valid.sum(0)?.affine(1.0 / seq.len() as f64, 0.0)?;
            rows.push(mean.unsqueeze(0)?);
        }
        let pooled = Tensor::cat(&rows, 0)?;
        let norm = pooled
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?;
        Ok(pooled.broadcast_div(&norm)?.to_vec2::<f32>()?)
    }

    pub fn forward_observed(
        &self,
        sequences: &[Vec<u32>],
        padded_s: usize,
    ) -> Result<crate::precision::ObservedRun> {
        let embeddings = self.forward_with_padded_s(sequences, padded_s)?;
        Ok(crate::precision::ObservedRun {
            padded_b: sequences.len(),
            padded_s,
            projection_rows: sequences.len() * padded_s,
            embeddings,
        })
    }

    pub fn forward_states(&self, sequences: &[Vec<u32>]) -> Result<Vec<Vec<Vec<f32>>>> {
        let x = self.run_decoder(sequences)?;
        let mut out = Vec::with_capacity(sequences.len());
        for (row, seq) in sequences.iter().enumerate() {
            out.push(x.i(row)?.narrow(0, 0, seq.len())?.to_vec2::<f32>()?);
        }
        Ok(out)
    }
}

#[cfg(feature = "cuda")]
pub struct IndependentExecutor {
    model: Model,
    math: crate::cuda_math::MathModePin,
    manifest: std::fs::File,
    next_forward: u64,
    poisoned: bool,
    batching: Option<BatchedExecution>,
    _thread_affinity: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(feature = "cuda")]
impl IndependentExecutor {
    pub fn load(
        dir: &Path,
        device: Device,
        engine: EngineConfig,
        math: crate::cuda_math::MathModePin,
        manifest_path: &Path,
    ) -> Result<Self> {
        Self::load_with_attestation(dir, device, engine, math, manifest_path, None)
    }

    pub fn load_with_attestation(
        dir: &Path,
        device: Device,
        engine: EngineConfig,
        math: crate::cuda_math::MathModePin,
        manifest_path: &Path,
        attestation_sha256: Option<&str>,
    ) -> Result<Self> {
        Self::load_configured(
            dir,
            device,
            engine,
            math,
            manifest_path,
            attestation_sha256,
            None,
        )
    }

    pub fn load_batched_with_attestation(
        dir: &Path,
        device: Device,
        engine: EngineConfig,
        math: crate::cuda_math::MathModePin,
        manifest_path: &Path,
        attestation_sha256: Option<&str>,
        batching: BatchedExecution,
    ) -> Result<Self> {
        batching.validate_engine(&engine)?;
        Self::load_configured(
            dir,
            device,
            engine,
            math,
            manifest_path,
            attestation_sha256,
            Some(batching),
        )
    }

    fn load_configured(
        dir: &Path,
        device: Device,
        engine: EngineConfig,
        math: crate::cuda_math::MathModePin,
        manifest_path: &Path,
        attestation_sha256: Option<&str>,
        batching: Option<BatchedExecution>,
    ) -> Result<Self> {
        use std::io::Write;

        ensure!(matches!(&device, Device::Cuda(_)), "CUDA required");
        ensure!(engine.max_seqs > 0, "invalid max_seqs");
        ensure!(engine.max_len > 0, "invalid max_len");
        ensure!(engine.max_tokens >= engine.max_len, "one sequence must fit");
        if let Some(policy) = &batching {
            policy.validate_engine(&engine)?;
        }
        math.check()?;

        let model = Model::load(dir, device, engine, DType::BF16)?
            .with_precision(crate::precision::PrecisionProfile::F);

        let mut manifest = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(manifest_path)?;

        let mut start_record = serde_json::json!({
            "kind": "executor_start",
            "execution_policy": "independent_sequence_v1",
            "serving_route_scope": "in_process_token_ID_API",
            "http_certified": false,
            "profile": "F",
            "weights_and_projections": "BF16",
            "norm_activation_attention": "widened_as_defined_by_F",
            "physical_B": 1,
            "physical_S": "valid_token_count",
            "stream_policy": "single_owner_no_overlap",
            "math": math.check()?
        });

        if let Some(policy) = &batching {
            start_record["execution_policy"] = serde_json::json!(BATCHED_EXECUTION_POLICY);
            start_record["physical_B"] = serde_json::json!({
                "minimum": 1,
                "maximum": policy.max_batch_size,
                "dummy_rows": false
            });
            start_record["physical_S"] = serde_json::json!("maximum_valid_length");
            start_record["batching"] = serde_json::to_value(policy)?;
            start_record["masking"] = serde_json::json!({
                "padding": "right",
                "pad_id": 11,
                "attention": "bidirectional",
                "invalid_key_score": "negative_infinity",
                "padded_queries": "attend_valid_keys_then_discard",
                "positions": "zero_based_per_sequence",
                "pooling": "valid_tokens_only"
            });
            start_record["independent_compatibility_route"] = serde_json::json!("forward_serving");
        }

        if let Some(att_sha) = attestation_sha256 {
            start_record["attestation_sha256"] = serde_json::json!(att_sha);
        }

        serde_json::to_writer(&mut manifest, &start_record)?;
        manifest.write_all(b"\n")?;
        manifest.flush()?;
        manifest.sync_data()?;

        Ok(Self {
            model,
            math,
            manifest,
            next_forward: 0,
            poisoned: false,
            batching,
            _thread_affinity: std::marker::PhantomData,
        })
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn batching(&self) -> Option<&BatchedExecution> {
        self.batching.as_ref()
    }

    fn record(&mut self, value: &serde_json::Value) -> Result<()> {
        use std::io::Write;

        serde_json::to_writer(&mut self.manifest, value)?;
        self.manifest.write_all(b"\n")?;
        self.manifest.flush()?;
        self.manifest.sync_data()?;
        Ok(())
    }

    fn validate_sequences(&self, sequences: &[Vec<u32>]) -> Result<()> {
        ensure!(!sequences.is_empty(), "empty request");
        for sequence in sequences {
            ensure!(
                !sequence.is_empty() && sequence.len() <= self.model.engine.max_len,
                "invalid sequence length"
            );
            ensure!(
                sequence.iter().all(|&id| {
                    id != self.model.cfg.pad_token_id && (id as usize) < self.model.cfg.vocab_size
                }),
                "padding or out-of-vocabulary token"
            );
        }
        Ok(())
    }

    /// Compatibility route: always one physical B=1 forward per sequence.
    ///
    /// This remains the route for existing startup checks, M2 acceptance
    /// and singleton golden export. Opening with a batching policy does
    /// not change this method's execution semantics.
    pub fn forward_serving(&mut self, sequences: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        ensure!(!self.poisoned, "executor is poisoned; restart the process");
        self.validate_sequences(sequences)?;

        self.poisoned = true;
        let result = self.forward_validated(sequences);
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    /// Execute exactly one physical batch, without splitting or retry.
    ///
    /// The scheduler assembles complete, already-validated requests. This
    /// boundary validates again before any execution. B=1 uses the same
    /// Model::forward arithmetic as the compatibility route, with S=L.
    pub fn forward_batch(
        &mut self,
        sequences: &[Vec<u32>],
        associations: &[ExecutionRow],
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(!self.poisoned, "executor is poisoned; restart the process");
        self.validate_sequences(sequences)?;
        let policy = self
            .batching
            .as_ref()
            .context("executor was not opened for batched execution")?;
        ensure!(
            associations.len() == sequences.len(),
            "association cardinality mismatch"
        );

        let mut positions = BTreeSet::new();
        let mut cases: BTreeMap<u64, &Option<String>> = BTreeMap::new();
        for association in associations {
            association.validate()?;
            ensure!(
                positions.insert((association.request_id, association.request_row)),
                "duplicate request/input association"
            );
            if let Some(previous) = cases.insert(association.request_id, &association.case_id) {
                ensure!(
                    previous == &association.case_id,
                    "inconsistent case ID for one request"
                );
            }
        }

        let s = sequences
            .iter()
            .map(Vec::len)
            .max()
            .context("empty batch")?;
        ensure!(
            policy.fits(sequences.len(), s),
            "physical batch exceeds configured capacity"
        );

        self.poisoned = true;
        let result = self.forward_batch_validated(sequences, associations, s);
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    fn forward_batch_validated(
        &mut self,
        sequences: &[Vec<u32>],
        associations: &[ExecutionRow],
        s: usize,
    ) -> Result<Vec<Vec<f32>>> {
        use sha2::{Digest, Sha256};

        let b = sequences.len();
        let projection_rows = b.checked_mul(s).context("projection size overflow")?;
        let attention_cells = projection_rows
            .checked_mul(s)
            .context("attention size overflow")?;
        let valid_tokens = sequences.iter().try_fold(0usize, |total, sequence| {
            total
                .checked_add(sequence.len())
                .context("valid token count overflow")
        })?;
        let math = self.math.check()?;
        let forward_id = self.next_forward;
        self.next_forward = self
            .next_forward
            .checked_add(1)
            .context("forward counter overflow")?;

        let members: Vec<Value> = sequences
            .iter()
            .zip(associations)
            .enumerate()
            .map(|(physical_row, (sequence, association))| {
                serde_json::json!({
                    "physical_row": physical_row,
                    "request_id": association.request_id,
                    "request_row": association.request_row,
                    "case_id": association.case_id,
                    "valid_length": sequence.len(),
                    "token_sha256": crate::precision::token_digest(sequence)
                })
            })
            .collect();

        self.record(&serde_json::json!({
            "kind": "physical_forward_begin",
            "execution_policy": BATCHED_EXECUTION_POLICY,
            "forward_id": forward_id,
            "B": b,
            "S": s,
            "projection_rows": projection_rows,
            "attention_cells_before_heads": attention_cells,
            "valid_tokens": valid_tokens,
            "padding_tokens": projection_rows - valid_tokens,
            "profile": "F",
            "rows": members,
            "math": math
        }))?;

        let result = (|| -> Result<Vec<Vec<f32>>> {
            // One call with physical B rows. No singleton loop here.
            // For B=1 this is the existing unpadded numerical path.
            let output = self.model.forward(sequences)?;
            ensure!(output.len() == b, "physical batch output cardinality");
            for row in &output {
                ensure!(row.len() == 2048, "embedding width");
                ensure!(row.iter().all(|x| x.is_finite()), "nonfinite embedding");
                let norm = row
                    .iter()
                    .map(|&x| f64::from(x).powi(2))
                    .sum::<f64>()
                    .sqrt();
                ensure!(
                    norm.is_finite() && (norm - 1.0).abs() <= 1e-5,
                    "embedding norm {norm}"
                );
            }
            // Host retrieval in Model::forward completes the output path.
            self.math.check()?;
            Ok(output)
        })();

        let output = match result {
            Ok(output) => output,
            Err(error) => {
                let _ = self.record(&serde_json::json!({
                    "kind": "physical_forward_end",
                    "execution_policy": BATCHED_EXECUTION_POLICY,
                    "forward_id": forward_id,
                    "B": b,
                    "S": s,
                    "profile": "F",
                    "rows": members,
                    "status": "error",
                    "error": format!("{error:#}")
                }));
                return Err(error);
            }
        };

        let output_rows: Vec<Value> = output
            .iter()
            .zip(&members)
            .map(|(row, member)| {
                let bytes: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
                let mut member = member.clone();
                member["output_sha256"] =
                    serde_json::json!(format!("{:x}", Sha256::digest(&bytes)));
                member
            })
            .collect();
        let math = self.math.check()?;
        self.record(&serde_json::json!({
            "kind": "physical_forward_end",
            "execution_policy": BATCHED_EXECUTION_POLICY,
            "forward_id": forward_id,
            "B": b,
            "S": s,
            "projection_rows": projection_rows,
            "attention_cells_before_heads": attention_cells,
            "valid_tokens": valid_tokens,
            "profile": "F",
            "status": "ok",
            "rows": output_rows,
            "math": math
        }))?;

        Ok(output)
    }

    fn forward_validated(&mut self, sequences: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        let mut rows = Vec::with_capacity(sequences.len());

        for (request_row, sequence) in sequences.iter().enumerate() {
            let math = self.math.check()?;
            let forward_id = self.next_forward;
            self.next_forward = self
                .next_forward
                .checked_add(1)
                .context("forward counter overflow")?;

            let digest = crate::precision::token_digest(sequence);

            self.record(&serde_json::json!({
                "kind": "physical_forward_begin",
                "forward_id": forward_id,
                "logical_request_row": request_row,
                "token_sha256": digest,
                "B": 1,
                "S": sequence.len(),
                "projection_rows": sequence.len(),
                "profile": "F",
                "math": math
            }))?;

            let result = (|| -> Result<Vec<f32>> {
                let output = self.model.forward(std::slice::from_ref(sequence))?;
                ensure!(output.len() == 1, "physical forward cardinality");

                let row = output
                    .into_iter()
                    .next()
                    .context("missing physical forward output")?;
                ensure!(row.len() == 2048, "embedding width");
                ensure!(row.iter().all(|x| x.is_finite()), "nonfinite embedding");

                let norm = row
                    .iter()
                    .map(|&x| f64::from(x).powi(2))
                    .sum::<f64>()
                    .sqrt();

                ensure!(
                    norm.is_finite() && (norm - 1.0).abs() <= 1e-5,
                    "embedding norm {norm}"
                );

                self.math.check()?;
                Ok(row)
            })();

            let row = match result {
                Ok(row) => row,
                Err(error) => {
                    let _ = self.record(&serde_json::json!({
                        "kind": "physical_forward_end",
                        "forward_id": forward_id,
                        "token_sha256": digest,
                        "B": 1,
                        "S": sequence.len(),
                        "profile": "F",
                        "status": "error",
                        "error": format!("{error:#}")
                    }));
                    return Err(error);
                }
            };

            use sha2::{Digest, Sha256};
            let bytes: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();

            let math = self.math.check()?;
            self.record(&serde_json::json!({
                "kind": "physical_forward_end",
                "forward_id": forward_id,
                "token_sha256": digest,
                "B": 1,
                "S": sequence.len(),
                "profile": "F",
                "status": "ok",
                "output_sha256": format!("{:x}", Sha256::digest(&bytes)),
                "math": math
            }))?;

            rows.push(row);
        }

        Ok(rows)
    }
}

#[cfg(test)]
mod guard_tests {
    use super::*;

    const BASE: &str = r#""hidden_size":2048,"intermediate_size":6144,
        "num_hidden_layers":16,"num_attention_heads":24,"num_key_value_heads":8,
        "head_dim":128,"vocab_size":131072,"rms_norm_eps":1e-5,
        "hidden_act":"silu","max_position_embeddings":262144,
        "is_causal":false,"pooling":"avg","pad_token_id":11"#;

    const YARN: &str = r#""rope_parameters":{"rope_type":"yarn","type":"yarn",
        "rope_theta":1000000.0,"factor":16.0,
        "original_max_position_embeddings":16384,
        "beta_fast":32.0,"beta_slow":1.0,"apply_yarn_scaling":false}"#;

    fn cfg(extra: &str) -> Config {
        let inner = if extra.is_empty() {
            BASE.to_string()
        } else {
            format!("{BASE},{extra}")
        };
        serde_json::from_str(&format!("{{{inner}}}")).unwrap()
    }

    #[test]
    fn unknown_config_key_rejected() {
        let raw: Value =
            serde_json::from_str(&format!("{{{BASE},\"mystery_switch\":true}}")).unwrap();
        assert!(semantic_config_check(&raw).is_err());
        let raw: Value =
            serde_json::from_str(&format!("{{{BASE},\"sliding_window\":4096}}")).unwrap();
        assert!(semantic_config_check(&raw).is_err());
        let raw: Value =
            serde_json::from_str(&format!("{{{BASE},{YARN},\"mystery_nested\":1}}")).unwrap();
        assert!(semantic_config_check(&raw).is_err());
        let raw: Value = serde_json::from_str(&format!(
            "{{{BASE},\"rope_parameters\":{{\"rope_type\":\"yarn\",\
             \"rope_theta\":1000000.0,\"factor\":16.0,\
             \"original_max_position_embeddings\":16384,\"beta_fast\":32.0,\
             \"beta_slow\":1.0,\"apply_yarn_scaling\":false,\
             \"mystery_inner\":true}}}}"
        ))
        .unwrap();
        assert!(semantic_config_check(&raw).is_err());
        let raw: Value = serde_json::from_str(&format!(
            "{{{BASE},{YARN},\"sliding_window\":null,\"model_type\":\"ministral3\"}}"
        ))
        .unwrap();
        assert!(semantic_config_check(&raw).is_ok());
    }

    #[test]
    fn missing_rope_parameters_rejected() {
        assert!(yarn_params(&cfg("")).is_err());
    }

    #[test]
    fn malformed_alias_rejected() {
        assert!(yarn_params(&cfg(
            r#""rope_parameters":{"rope_type":123,"type":"yarn","factor":16.0,
               "original_max_position_embeddings":16384,"beta_fast":32.0,
               "beta_slow":1.0,"apply_yarn_scaling":false}"#
        ))
        .is_err());
    }

    #[test]
    fn default_rope_type_rejected_for_frozen_contract() {
        assert!(yarn_params(&cfg(
            r#""rope_parameters":{"rope_type":"default","rope_theta":1000000.0}"#
        ))
        .is_err());
    }

    #[test]
    fn malformed_attn_factor_rejected() {
        assert!(yarn_params(&cfg(
            r#""rope_parameters":{"rope_type":"yarn","rope_theta":1000000.0,
               "factor":16.0,"original_max_position_embeddings":16384,
               "beta_fast":32.0,"beta_slow":1.0,"apply_yarn_scaling":false,
               "attn_factor":"x"}"#
        ))
        .is_err());
    }

    #[test]
    fn missing_theta_rejected_and_wrong_theta_rejected() {
        assert!(theta(&cfg("")).is_err());
        assert!(theta(&cfg(YARN)).is_ok());
        assert!(theta(&cfg(
            r#""rope_parameters":{"rope_type":"yarn","rope_theta":500000.0,
               "factor":16.0,"original_max_position_embeddings":16384,
               "beta_fast":32.0,"beta_slow":1.0,"apply_yarn_scaling":false}"#
        ))
        .is_err());
    }

    #[test]
    fn manifest_has_146_entries() {
        assert_eq!(expected_manifest().len(), 146);
    }

    fn batch_engine() -> EngineConfig {
        EngineConfig {
            causal: false,
            rope_layout: "split_half".into(),
            max_len: 1024,
            max_tokens: 32768,
            max_seqs: 32,
            wait_us: 1000,
        }
    }

    #[test]
    fn batched_default_envelope() {
        let policy = BatchedExecution::default();
        assert!(policy.validate_engine(&batch_engine()).is_ok());
        assert!(policy.fits(1, 1));
        assert!(policy.fits(1, 1024));
        assert!(policy.fits(32, 1024));
        assert!(!policy.fits(33, 1));
        assert!(!policy.fits(1, 1025));
        assert!(!policy.fits(0, 1));
        assert!(!policy.fits(1, 0));
        assert!(!policy.fits(usize::MAX, usize::MAX));
    }

    #[test]
    fn padded_budget_is_not_valid_token_budget() {
        let policy = BatchedExecution {
            max_batch_size: 32,
            max_padded_tokens: 2048,
            max_attention_cells: MAX_ATTENTION_CELLS,
            max_wait_us: 1000,
        };
        assert!(policy.fits(2, 1024));
        assert!(!policy.fits(3, 1024));
        assert!(policy.fits(8, 256));
    }

    #[test]
    fn attention_budget_is_independent() {
        let policy = BatchedExecution {
            max_batch_size: 32,
            max_padded_tokens: MAX_PADDED_TOKENS,
            max_attention_cells: 2 * 1024 * 1024,
            max_wait_us: 1000,
        };
        assert!(policy.fits(2, 1024));
        assert!(!policy.fits(3, 1024));
        assert!(policy.fits(32, 256));
    }

    #[test]
    fn engine_limits_are_not_silently_overridden() {
        let policy = BatchedExecution::default();
        let mut engine = batch_engine();
        engine.max_seqs = 8;
        assert!(policy.validate_engine(&engine).is_err());
        engine = batch_engine();
        engine.max_tokens = 8192;
        assert!(policy.validate_engine(&engine).is_err());
        engine = batch_engine();
        engine.wait_us = 0;
        assert!(policy.validate_engine(&engine).is_err());
    }

    #[test]
    fn batching_policy_rejects_unknown_fields() {
        let raw = r#"{
            "max_batch_size":32,
            "max_padded_tokens":32768,
            "max_attention_cells":33554432,
            "max_wait_us":1000,
            "silent_fallback":true
        }"#;
        assert!(serde_json::from_str::<BatchedExecution>(raw).is_err());
    }

    #[test]
    fn execution_case_id_validation() {
        let mut row = ExecutionRow {
            request_id: 1,
            request_row: 0,
            case_id: Some("batch-47-0".into()),
        };
        assert!(row.validate().is_ok());
        row.case_id = Some("bad\nid".into());
        assert!(row.validate().is_err());
        row.case_id = None;
        assert!(row.validate().is_ok());
    }
}
