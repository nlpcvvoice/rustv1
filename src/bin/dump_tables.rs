// Dump the exact RoPE/YaRN/llama-4 tables that Model::load would build for
// the frozen contract, as raw little-endian f32, for parity comparison
// against the vLLM reference formulas executed in torch.
//
// Layout: inv_freq[64] | cos[1024*64] | sin[1024*64] | llama4_scales[1024]
// plus boundary scales appended: llama4_scales at 16383,16384,32767,32768.

use rustv1::model::{build_rope_tables, YarnParams};
use rustv1::yarn::{llama4_attn_scale, standard_yarn_inv_freq};

fn main() {
    let out = std::env::args().nth(1).expect("usage: dump_tables FILE");

    let y = YarnParams {
        factor: 16.0,
        original_max: 16_384.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };

    let inv = standard_yarn_inv_freq(128, 1_000_000.0, 16.0, 16_384.0, 32.0, 1.0);
    let (cos, sin) = build_rope_tables(1_000_000.0, Some(&y), 128, 1024);
    let mut scales: Vec<f32> = (0..1024)
        .map(|p| llama4_attn_scale(0.1, 16_384.0, p) as f32)
        .collect();
    for p in [16_383usize, 16_384, 32_767, 32_768] {
        scales.push(llama4_attn_scale(0.1, 16_384.0, p) as f32);
    }

    let mut bytes = Vec::new();
    for v in inv
        .iter()
        .chain(cos.iter())
        .chain(sin.iter())
        .chain(scales.iter())
    {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out, &bytes).expect("write");
    eprintln!(
        "inv={} cos={} sin={} scales={}",
        inv.len(),
        cos.len(),
        sin.len(),
        scales.len()
    );
}
