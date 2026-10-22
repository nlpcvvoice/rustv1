// Standard YaRN frequency blend. Mathematically equivalent to the vLLM
// v0.25.0 reference (YaRNScalingRotaryEmbedding::_compute_inv_freq with
// extrapolation_factor=1 and yarn_find_correction_range(truncate=true));
// floating-point equivalence is NOT established (reference builds positive
// powers and the blend in FP32; this helper blends in FP64 and rounds once).
// Wired into the loader for rope_type="yarn". The reference uses attn_factor
// as the cache amplitude when apply_yarn_scaling=false; this port supports
// only attn_factor=1, enforced by the loader.

/// Standard YaRN frequency blend.
pub fn standard_yarn_inv_freq(
    dim: usize,
    theta: f64,
    factor: f64,
    original_context: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> Vec<f32> {
    let correction_dim = |rotations: f64| {
        dim as f64
            * (original_context / (rotations * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * theta.ln())
    };

    let low = correction_dim(beta_fast).floor().max(0.0);
    let high = correction_dim(beta_slow).ceil().min((dim - 1) as f64);
    let denominator = if low == high {
        0.001
    } else {
        high - low
    };

    (0..dim / 2)
        .map(|j| {
            let extrapolated = theta.powf(-((2 * j) as f64) / dim as f64);
            let interpolated = extrapolated / factor;
            let ramp = ((j as f64 - low) / denominator).clamp(0.0, 1.0);

            // Low-frequency-index dimensions retain unscaled frequencies.
            // High-frequency-index dimensions use interpolation.
            (extrapolated * (1.0 - ramp) + interpolated * ramp) as f32
        })
        .collect()
}

/// llama-4 attention scale, shared by Model::forward and the tests.
/// Matches vLLM mistral.py: 1 + beta * ln(1 + floor(p / original_max)).
pub fn llama4_attn_scale(beta: f64, original_max: f64, position: usize) -> f64 {
    1.0 + beta * (1.0 + ((position as f64) / original_max).floor()).ln()
}

#[cfg(test)]
mod yarn_tests {
    use super::*;

    #[test]
    fn standard_yarn_transition() {
        let inv = standard_yarn_inv_freq(128, 1_000_000.0, 16.0, 16_384.0, 32.0, 1.0);
        assert_eq!(inv.len(), 64);

        // The standard correction boundaries are floor(...)=20, ceil(...)=37.
        for j in 0..64 {
            let plain = 1_000_000_f64.powf(-((2 * j) as f64) / 128.0);
            let ratio = inv[j] as f64 / plain;

            if j <= 20 {
                assert!((ratio - 1.0).abs() < 1e-6);
            } else if j >= 37 {
                assert!((ratio - 1.0 / 16.0).abs() < 1e-6);
            } else {
                assert!(ratio > 1.0 / 16.0 && ratio < 1.0);
            }
        }
    }

    #[test]
    fn llama4_scale_identity_below_boundary() {
        // Exercises the shared helper used by Model::forward.
        let (beta, original_max) = (0.1_f64, 16_384.0);
        for p in 0..1024 {
            assert_eq!(llama4_attn_scale(beta, original_max, p), 1.0);
        }
        // Exact values at and beyond the boundary.
        assert_eq!(llama4_attn_scale(beta, original_max, 16_383), 1.0);
        assert!(
            (llama4_attn_scale(beta, original_max, 16_384)
                - (1.0 + 0.1 * 2.0f64.ln())).abs() < 1e-12
        );
        assert!(
            (llama4_attn_scale(beta, original_max, 32_767)
                - (1.0 + 0.1 * 2.0f64.ln())).abs() < 1e-12
        );
        assert!(
            (llama4_attn_scale(beta, original_max, 32_768)
                - (1.0 + 0.1 * 3.0f64.ln())).abs() < 1e-12
        );
    }
}
