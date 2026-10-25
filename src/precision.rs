use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::Module;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionProfile {
    A,
    B,
    C,
    D,
    E,
    F,
}

impl FromStr for PrecisionProfile {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_uppercase().as_str() {
            "A" => Ok(Self::A),
            "B" => Ok(Self::B),
            "C" => Ok(Self::C),
            "D" => Ok(Self::D),
            "E" => Ok(Self::E),
            "F" => Ok(Self::F),
            _ => anyhow::bail!("precision profile must be A, B, C, D, E, or F"),
        }
    }
}

impl PrecisionProfile {
    fn single_norm(self) -> bool {
        matches!(self, Self::B | Self::F)
    }

    fn single_activation(self) -> bool {
        matches!(self, Self::C | Self::F)
    }

    fn wide_qk(self) -> bool {
        matches!(self, Self::D | Self::E | Self::F)
    }

    fn wide_pv(self) -> bool {
        matches!(self, Self::E | Self::F)
    }
}

pub fn rms(
    x: &Tensor,
    weight: &Tensor,
    eps: f64,
    profile: PrecisionProfile,
) -> candle_core::Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let denom = xf
        .sqr()?
        .mean_keepdim(D::Minus1)?
        .affine(1.0, eps)?
        .sqrt()?;
    let normalized = xf.broadcast_div(&denom)?;

    if profile.single_norm() {
        normalized
            .broadcast_mul(&weight.to_dtype(DType::F32)?)?
            .to_dtype(dtype)
    } else {
        normalized
            .to_dtype(dtype)?
            .broadcast_mul(&weight.to_dtype(dtype)?)
    }
}

pub fn silu_mul(
    gate: &Tensor,
    up: &Tensor,
    profile: PrecisionProfile,
) -> candle_core::Result<Tensor> {
    if profile.single_activation() {
        let g = gate.to_dtype(DType::F32)?;
        let u = up.to_dtype(DType::F32)?;
        (candle_nn::Activation::Silu.forward(&g)? * u)?.to_dtype(gate.dtype())
    } else {
        candle_nn::Activation::Silu.forward(gate)? * up
    }
}

/// Assumption: q/k/v are [B,H,S,D] with matching H.
/// The existing executor performs its current KV-repeat before this call.
/// mask is FP32 [B,1,S,S], zero or -infinity.
/// This is diagnostic dense attention, not the M3 fused implementation.
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    profile: PrecisionProfile,
) -> candle_core::Result<Tensor> {
    let dtype = q.dtype();
    let d = q.dim(3)?;
    let scores = if profile.wide_qk() {
        q.to_dtype(DType::F32)?
            .matmul(&k.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?)?
    } else {
        q.matmul(&k.transpose(2, 3)?.contiguous()?)?
            .to_dtype(DType::F32)?
    };

    let scores = scores
        .affine(1.0 / (d as f64).sqrt(), 0.0)?
        .broadcast_add(mask)?;
    let p = candle_nn::ops::softmax_last_dim(&scores)?;

    if profile.wide_pv() {
        p.matmul(&v.to_dtype(DType::F32)?)?.to_dtype(dtype)
    } else {
        p.to_dtype(dtype)?.matmul(v)
    }
}

pub fn token_digest(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn finite_vector(x: &[f32], label: &str) -> Result<()> {
    ensure!(x.len() == 2048, "{label}: expected 2048 components");
    ensure!(x.iter().all(|v| v.is_finite()), "{label}: nonfinite");
    Ok(())
}

fn embedding(x: &[f32], label: &str) -> Result<()> {
    finite_vector(x, label)?;
    let norm = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    ensure!(
        norm > 0.0 && (norm - 1.0).abs() <= 1e-5,
        "{label}: invalid norm {norm}"
    );
    Ok(())
}

pub fn metrics(a: &[f32], b: &[f32]) -> Result<(f64, f64, usize, f64)> {
    ensure!(
        a.len() == b.len() && !a.is_empty(),
        "metric dimension mismatch"
    );
    ensure!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "nonfinite metric input"
    );

    let (mut dot, mut aa, mut bb, mut sq) = (0.0, 0.0, 0.0, 0.0);
    let (mut max_abs, mut index) = (0.0, 0);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        aa += x * x;
        bb += y * y;
        let e = (x - y).abs();
        sq += e * e;
        if e > max_abs {
            max_abs = e;
            index = i;
        }
    }
    ensure!(aa > 0.0 && bb > 0.0, "zero norm");
    let cosine = dot / (aa.sqrt() * bb.sqrt());
    ensure!(cosine.is_finite() && max_abs.is_finite(), "invalid metric");
    Ok((cosine, max_abs, index, sq.sqrt()))
}

/// Return these from the actual executed path, not from requested shapes.
pub struct ObservedRun {
    pub embeddings: Vec<Vec<f32>>,
    pub padded_b: usize,
    pub padded_s: usize,
    pub projection_rows: usize,
}

fn execute<F>(
    run: &mut F,
    corpus: &[Vec<u32>],
    ids: &[usize],
    padded_s: usize,
    label: &str,
) -> Result<ObservedRun>
where
    F: FnMut(&[Vec<u32>], usize) -> Result<ObservedRun>,
{
    ensure!(!ids.is_empty() && ids.len() <= 8, "invalid B");
    ensure!(padded_s <= 1024 && ids.len() * padded_s <= 4096, "budget");
    let sequences: Vec<_> = ids.iter().map(|&i| corpus[i].clone()).collect();
    ensure!(
        sequences
            .iter()
            .all(|x| !x.is_empty() && x.len() <= padded_s),
        "invalid lengths"
    );

    let observed = run(&sequences, padded_s)?;
    ensure!(
        observed.embeddings.len() == ids.len(),
        "wrong output cardinality"
    );
    ensure!(observed.padded_b == ids.len(), "B override not executed");
    ensure!(observed.padded_s == padded_s, "S override not executed");
    ensure!(
        observed.projection_rows == ids.len() * padded_s,
        "unexpected projection row count"
    );

    for row in &observed.embeddings {
        embedding(row, label)?;
    }
    println!(
        "{}",
        json!({
            "kind": "executed_shape", "label": label,
            "batch_ids": ids,
            "lengths": sequences.iter().map(Vec::len).collect::<Vec<_>>(),
            "B": observed.padded_b, "S": observed.padded_s,
            "projection_rows": observed.projection_rows,
        })
    );
    Ok(observed)
}

fn pair(
    label: &str,
    corpus: &[Vec<u32>],
    ids: &[usize],
    row: usize,
    singleton: &ObservedRun,
    batch: &ObservedRun,
) -> Result<(f64, f64)> {
    let i = ids[row];
    let (cosine, max_abs, component, l2) =
        metrics(&singleton.embeddings[0], &batch.embeddings[row])?;
    println!(
        "{}",
        json!({
            "kind": "invariance_pair", "label": label,
            "sample_index": i, "token_sha256": token_digest(&corpus[i]),
            "sample_5": i == 5,
            "singleton": {
                "B": singleton.padded_b, "S": singleton.padded_s,
                "projection_rows": singleton.projection_rows,
            },
            "batch_ids": ids,
            "batch_lengths": ids.iter().map(|&j| corpus[j].len()).collect::<Vec<_>>(),
            "B": batch.padded_b, "S": batch.padded_s,
            "projection_rows": batch.projection_rows,
            "cosine": cosine, "max_abs": max_abs,
            "max_error_component": component, "l2": l2,
        })
    );
    Ok((cosine, max_abs))
}

/// Assumption: callback's second argument is an explicit padded-S override.
/// It MUST alter the model allocation/mask/projection shape, not append valid
/// tokens to a request. execute() checks the returned observed shape.
///
/// groups are the exact acceptance comparison groups, including ordering.
/// No exception is applied here, including to sample 5.
pub fn batch_forensics<F>(corpus: &[Vec<u32>], groups: &[Vec<usize>], mut run: F) -> Result<()>
where
    F: FnMut(&[Vec<u32>], usize) -> Result<ObservedRun>,
{
    ensure!(!corpus.is_empty() && !groups.is_empty(), "empty experiment");
    for seq in corpus {
        ensure!(!seq.is_empty() && seq.len() <= 1024, "invalid corpus");
    }

    let mut singles = Vec::with_capacity(corpus.len());
    for i in 0..corpus.len() {
        singles.push(execute(
            &mut run,
            corpus,
            &[i],
            corpus[i].len(),
            "singleton",
        )?);
    }

    let mut worst_cos = (f64::INFINITY, 0usize, 0usize);
    let mut worst_abs = (-1.0f64, 0usize, 0usize);

    for (g, ids) in groups.iter().enumerate() {
        ensure!(
            !ids.is_empty() && ids.iter().all(|&i| i < corpus.len()),
            "invalid group"
        );
        let s = ids.iter().map(|&i| corpus[i].len()).max().unwrap();
        let output = execute(&mut run, corpus, ids, s, "acceptance_group")?;
        for (row, &i) in ids.iter().enumerate() {
            let (c, a) = pair("acceptance_group", corpus, ids, row, &singles[i], &output)?;
            if c < worst_cos.0 {
                worst_cos = (c, i, g);
            }
            if a > worst_abs.0 {
                worst_abs = (a, i, g);
            }
        }
    }

    println!(
        "{}",
        json!({
            "kind": "worst_identities",
            "cosine": {"value": worst_cos.0, "sample": worst_cos.1, "group": worst_cos.2},
            "max_abs": {"value": worst_abs.0, "sample": worst_abs.1, "group": worst_abs.2},
        })
    );

    let mut targets = vec![(worst_cos.1, worst_cos.2), (worst_abs.1, worst_abs.2)];
    targets.sort_unstable();
    targets.dedup();

    for (target, group) in targets {
        let n = corpus[target].len();
        let fixed_s = groups[group]
            .iter()
            .map(|&i| corpus[i].len())
            .max()
            .unwrap();

        // Effect 1: vary B, fixed padded S, duplicate identical target inputs.
        for b in [1usize, 2, 4, 8] {
            if b * fixed_s > 4096 {
                continue;
            }
            let ids = vec![target; b];
            let output = execute(&mut run, corpus, &ids, fixed_s, "vary_B_fixed_S")?;
            for row in 0..b {
                pair(
                    "vary_B_fixed_S",
                    corpus,
                    &ids,
                    row,
                    &singles[target],
                    &output,
                )?;
            }
        }

        // Effect 2: fixed B and request composition; vary padding only.
        let ids = vec![target, target];
        let mut sizes = vec![n, (n + 1).min(1024), n.max(257), 1024];
        sizes.sort_unstable();
        sizes.dedup();
        for s in sizes {
            let output = execute(&mut run, corpus, &ids, s, "vary_S_fixed_B")?;
            for row in 0..2 {
                pair(
                    "vary_S_fixed_B",
                    corpus,
                    &ids,
                    row,
                    &singles[target],
                    &output,
                )?;
            }
        }

        // Effect 3: identical composition/shape, different order.
        let original = &groups[group];
        let a = execute(&mut run, corpus, original, fixed_s, "order_original")?;
        let mut reversed = original.clone();
        reversed.reverse();
        let b = execute(&mut run, corpus, &reversed, fixed_s, "order_reversed")?;
        for row in 0..original.len() {
            let reversed_row = original.len() - 1 - row;
            let (c, e, component, l2) = metrics(&a.embeddings[row], &b.embeddings[reversed_row])?;
            println!(
                "{}",
                json!({
                    "kind": "ordering_pair", "sample": original[row],
                    "token_sha256": token_digest(&corpus[original[row]]),
                    "original_ids": original, "reversed_ids": reversed,
                    "B": a.padded_b, "S": a.padded_s,
                    "projection_rows": a.projection_rows,
                    "cosine": c, "max_abs": e, "l2": l2,
                    "max_error_component": component,
                })
            );
        }

        // Effect 4: repeat the same shape after an intervening different shape.
        let before = execute(&mut run, corpus, original, fixed_s, "state_before")?;
        let intervening_s = if fixed_s == 1024 { n } else { 1024 };
        execute(
            &mut run,
            corpus,
            &[target],
            intervening_s,
            "state_intervening",
        )?;
        let after = execute(&mut run, corpus, original, fixed_s, "state_after")?;
        for (row, &i) in original.iter().enumerate() {
            let (c, e, component, l2) = metrics(&before.embeddings[row], &after.embeddings[row])?;
            println!(
                "{}",
                json!({
                    "kind": "state_pair", "sample": i,
                    "token_sha256": token_digest(&corpus[i]),
                    "batch_ids": original,
                    "B": before.padded_b, "S": before.padded_s,
                    "projection_rows": before.projection_rows,
                    "cosine": c, "max_abs": e, "l2": l2,
                    "max_error_component": component,
                    "bit_equal": before.embeddings[row].iter()
                        .zip(&after.embeddings[row])
                        .all(|(x,y)| x.to_bits() == y.to_bits()),
                })
            );
        }
    }
    Ok(())
}

/// Identical FP32 bit patterns are transferred to CUDA before conversion.
/// Read BF16 values directly as half::bf16; do not infer bits from tolerances.
pub fn conversion_audit(cuda: &Device) -> Result<()> {
    ensure!(
        matches!(cuda, Device::Cuda(_)),
        "conversion audit requires CUDA"
    );

    let mut bits = vec![
        0x00000000, 0x80000000, 0x00000001, 0x007fffff, 0x00800000, 0x3f800000, 0xbf800000,
        0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x7fa12345, 0x3ca3d709, 0x3ca3d70a,
        0x3ca3d70b, // near +0.02
        0xbca3d709, 0xbca3d70a, 0xbca3d70b, // near -0.02
    ];

    for upper in 0x3ca0u32..=0x3ca8 {
        for low in [0u32, 0x7fff, 0x8000, 0x8001, 0xffff] {
            bits.push((upper << 16) | low);
            bits.push(0x80000000 | (upper << 16) | low);
        }
    }
    // Halfway ties around 1, with both even and odd retained BF16 LSBs.
    for upper in [0x3f80u32, 0x3f81, 0xbf80, 0xbf81] {
        for low in [0x7fffu32, 0x8000, 0x8001] {
            bits.push((upper << 16) | low);
        }
    }

    let values: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
    let cpu = Tensor::from_vec(values, bits.len(), &Device::Cpu)?;
    let gpu = cpu.to_device(cuda)?;

    let returned = gpu.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    ensure!(
        returned.iter().zip(&bits).all(|(x, b)| x.to_bits() == *b),
        "FP32 transfer altered input bits"
    );

    let c = cpu.to_dtype(DType::BF16)?.to_vec1::<half::bf16>()?;
    let g = gpu
        .to_dtype(DType::BF16)?
        .to_device(&Device::Cpu)?
        .to_vec1::<half::bf16>()?;

    let mut finite_mismatches = 0usize;
    let mut nan_mismatches = 0usize;
    for ((&input, cpu), gpu) in bits.iter().zip(c).zip(g) {
        let nan = f32::from_bits(input).is_nan();
        let different = cpu.to_bits() != gpu.to_bits();
        if different {
            if nan {
                nan_mismatches += 1;
            } else {
                finite_mismatches += 1;
            }
        }
        let rne = if nan {
            None
        } else {
            Some((input.wrapping_add(0x7fff + ((input >> 16) & 1)) >> 16) as u16)
        };
        println!(
            "{}",
            json!({
                "kind": "conversion",
                "f32_bits": format!("{input:08x}"),
                "cpu_bf16_bits": format!("{:04x}", cpu.to_bits()),
                "cuda_bf16_bits": format!("{:04x}", gpu.to_bits()),
                "rne_expected": rne.map(|x| format!("{x:04x}")),
                "nan": nan, "equal_bits": !different,
            })
        );
    }
    println!(
        "{}",
        json!({
            "kind": "conversion_summary",
            "non_nan_mismatches": finite_mismatches,
            "nan_payload_mismatches": nan_mismatches,
        })
    );
    ensure!(
        finite_mismatches == 0,
        "CPU/CUDA non-NaN conversion mismatch"
    );
    Ok(())
}

/// Supply executed means and outputs, not means recomputed by this helper.
/// states are the executed post-final-RMSNorm valid token states.
pub fn pooled_sensitivity(
    sample: usize,
    states: &[Vec<f32>],
    mean: &[f32],
    output: &[f32],
    reference_mean: &[f32],
    reference_output: &[f32],
) -> Result<()> {
    ensure!(!states.is_empty(), "empty token states");
    for state in states {
        finite_vector(state, "token state")?;
    }
    finite_vector(mean, "mean")?;
    finite_vector(reference_mean, "reference mean")?;
    embedding(output, "output")?;
    embedding(reference_output, "reference output")?;

    let norm = |x: &[f32]| x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    let mean_norm = norm(mean);
    let token_norm_mean = states.iter().map(|x| norm(x)).sum::<f64>() / states.len() as f64;

    let (pre_cos, pre_max, pre_index, pre_l2) = metrics(mean, reference_mean)?;
    let (post_cos, post_max, post_index, post_l2) = metrics(output, reference_output)?;
    println!(
        "{}",
        json!({
            "kind": "pooled_sensitivity", "sample": sample,
            "tokens": states.len(),
            "pooled_mean_norm": mean_norm,
            "reference_pooled_mean_norm": norm(reference_mean),
            "mean_token_state_norm": token_norm_mean,
            "cancellation_ratio": mean_norm / token_norm_mean,
            "pre_l2": {
                "cosine": pre_cos, "max_abs": pre_max,
                "l2_error": pre_l2, "max_error_component": pre_index,
            },
            "post_l2": {
                "cosine": post_cos, "max_abs": post_max,
                "l2_error": post_l2, "max_error_component": post_index,
            },
        })
    );
    Ok(())
}

pub fn comparison_metrics(a: &[f32], b: &[f32]) -> anyhow::Result<serde_json::Value> {
    use std::collections::BTreeMap;

    anyhow::ensure!(
        !a.is_empty() && a.len() == b.len(),
        "comparison shape mismatch"
    );
    anyhow::ensure!(
        a.iter().chain(b).all(|x| x.is_finite()),
        "nonfinite comparison input"
    );

    fn ordered32(x: f32) -> i64 {
        if x == 0.0 {
            return 0;
        }
        let bits = x.to_bits();
        let magnitude = i64::from(bits & 0x7fff_ffff);
        if bits >> 31 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    fn ordered16(x: f32) -> i64 {
        if x == 0.0 {
            return 0;
        }
        let bits = (x.to_bits() >> 16) as u16;
        let magnitude = i64::from(bits & 0x7fff);
        if bits >> 15 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    fn histogram(distances: &[u64]) -> serde_json::Value {
        let mut bins: BTreeMap<String, usize> = BTreeMap::new();
        for &d in distances {
            let key = match d {
                0 => "0",
                1 => "1",
                2 => "2",
                3..=4 => "3-4",
                5..=8 => "5-8",
                9..=16 => "9-16",
                17..=32 => "17-32",
                33..=64 => "33-64",
                65..=256 => "65-256",
                _ => ">256",
            };
            *bins.entry(key.to_owned()).or_default() += 1;
        }
        let mut sorted = distances.to_vec();
        sorted.sort_unstable();
        let p99_index = (sorted.len() * 99)
            .div_ceil(100)
            .saturating_sub(1)
            .min(sorted.len() - 1);
        serde_json::json!({
            "histogram": bins,
            "p99": sorted[p99_index],
            "max": sorted[sorted.len() - 1],
            "signed_zero_canonicalized": true
        })
    }

    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    let mut dot = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut differing = 0usize;
    let mut distances32 = Vec::with_capacity(a.len());

    for (&x, &y) in a.iter().zip(b) {
        let xf = f64::from(x);
        let yf = f64::from(y);
        aa += xf * xf;
        bb += yf * yf;
        dot += xf * yf;
        squared_error += (xf - yf).powi(2);
        max_abs = max_abs.max((xf - yf).abs());
        differing += usize::from(x.to_bits() != y.to_bits());
        distances32.push(ordered32(x).abs_diff(ordered32(y)));
    }

    let cosine = if aa > 0.0 && bb > 0.0 {
        Some(dot / (aa.sqrt() * bb.sqrt()))
    } else if aa == 0.0 && bb == 0.0 {
        Some(1.0)
    } else {
        None
    };

    let bf16_exact = a.iter().chain(b).all(|x| x.to_bits() & 0xffff == 0);
    let bf16_steps = if bf16_exact {
        let distances: Vec<u64> = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| ordered16(x).abs_diff(ordered16(y)))
            .collect();
        Some(histogram(&distances))
    } else {
        None
    };

    Ok(serde_json::json!({
        "components": a.len(),
        "cosine": cosine,
        "max_abs": max_abs,
        "l2": squared_error.sqrt(),
        "relative_l2_to_reference": if bb > 0.0 {
            Some((squared_error / bb).sqrt())
        } else {
            None
        },
        "differing_component_count": differing,
        "bit_identical": differing == 0,
        "f32_steps": histogram(&distances32),
        "bf16_steps": bf16_steps
    }))
}
