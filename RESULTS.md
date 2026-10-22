# Results

## M1 — numerical parity with vLLM (accepted)

Reference: vLLM 0.25.0, `nvidia/Nemotron-3-Embed-1B-BF16`, L4,
`enforce_eager`, prefix caching off, mean pooling + L2 normalize.
Implementation: this repo, Candle, CPU, FP32 oracle (BF16 weights
losslessly widened; Candle CPU has no BF16 matmul). BF16 compute on CUDA
is M2.

Frozen gates: dim 2048, all finite, |norm-1| <= 1e-5, cosine > 0.999,
max_abs <= 0.01. Batch invariance gates (cosine > 0.9999, max_abs <= 0.005)
were set from the measured vLLM self-sensitivity (0.99992 / 0.0026), not
guessed.

| Suite | Result |
|---|---|
| Table parity (RoPE inv_freq / cos-sin / llama_4_scaling vs vLLM formulas) | inv_freq max_rel 1.2e-7; cos/sin max_abs 8.3e-5; llama_4_scaling bit-exact |
| M1-CORE, 3 prompts (len 255/256/1024) | cosine 0.999983 / 0.999986 / 0.999982 |
| Expanded corpus, 17 prompts (len 1-1024, boundaries, EOS, repeats, natural text) | 17/17 PASS; worst cosine 0.999697; max_abs <= 0.0092; norm_dev <= 5.2e-7 |
| Batch invariance (single vs grouped vs reversed vs duplicate, n=34) | bit-exact |
| Invalid inputs (empty / pad id / over-length / out-of-vocab) | all rejected |
| Bidirectionality (stage level: later-token edit changes earlier-token state) | confirmed |

Raw outputs: `results/m1_core_results.txt`,
`results/m1_acceptance_results.txt`, `results/table_parity_results.txt`.

Note on degenerate inputs: a 4-token sequence repeated to length 1023 is
unstable in vLLM itself (single-vs-batch cosine 0.99885). Such samples are
kept as diagnostics, never as gates. The corpus sample was replaced with a
diverse 1023-token text (cosine 0.999991).

## Reproduce

```
pip install vllm==0.25.0 tokenizers numpy   # reference capture only
modal run tools/capture_expanded.py         # Modal; regenerates corpus + reference (base64 tar in log)
cargo run --release --bin parity_run -- <MODEL_DIR> results/prompts.json results/ref_single.f32
```

`<MODEL_DIR>` is a local copy of the Hugging Face checkpoint with
`engine.json.example` copied to it as `engine.json`.
