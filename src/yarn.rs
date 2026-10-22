// Standard YaRN frequency blend. Implemented and unit-tested, but NOT wired
// into the model loader: the semantics of `apply_yarn_scaling=false` in the
// reference implementation must be verified against the installed source
// before this table replaces plain RoPE frequencies.
#![allow(dead_code)]

/// Standard YaRN frequency blend for:
/// dim=128, theta=1e6, factor=16, original_context=16384,
/// beta_fast=32, beta_slow=1.
///
/// This implements the mathematical blend, not the architecture-specific
/// semantics of `apply_yarn_scaling`.
pub fn standard_yarn_inv_freq() -> Vec<f32> {
    const DIM: usize = 128;
    const THETA: f64 = 1_000_000.0;
    const FACTOR: f64 = 16.0;
    const ORIGINAL: f64 = 16_384.0;

    let correction_dim = |rotations: f64| {
        DIM as f64
            * (ORIGINAL / (rotations * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * THETA.ln())
    };

    let low = correction_dim(32.0).floor().max(0.0);
    let high = correction_dim(1.0).ceil().min((DIM - 1) as f64);
    let denominator = if low == high {
        0.001
    } else {
        high - low
    };

    (0..DIM / 2)
        .map(|j| {
            let extrapolated = THETA.powf(-((2 * j) as f64) / DIM as f64);
            let interpolated = extrapolated / FACTOR;
            let ramp = ((j as f64 - low) / denominator).clamp(0.0, 1.0);

            // Low-frequency-index dimensions retain unscaled frequencies.
            // High-frequency-index dimensions use interpolation.
            (extrapolated * (1.0 - ramp) + interpolated * ramp) as f32
        })
        .collect()
}

#[cfg(test)]
mod yarn_tests {
    use super::*;

    #[test]
    fn standard_yarn_transition() {
        let inv = standard_yarn_inv_freq();
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
}
