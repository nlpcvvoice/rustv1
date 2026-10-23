# Results

## M1 — numerical parity with vLLM

Reference: vLLM 0.25.0, `nvidia/Nemotron-3-Embed-1B-BF16`, L4,
`enforce_eager`, prefix caching off, mean pooling + L2 normalize.
Implementation: this repo, Candle, CPU, FP32 oracle (BF16 weights
losslessly widened; Candle CPU has no BF16 matmul). BF16 compute on CUDA
is M2.

Claim (externally reviewed and approved wording):

> M1 passes the frozen token-ID embedding acceptance suite using CPU FP32
> arithmetic over losslessly widened BF16 checkpoint weights, against
> saved vLLM 0.25.0 BF16 outputs. The suite contains 22 samples spanning
> lengths 1–1024: 21 pass the frozen cross-engine numerical gates, and
> one exact stress sample is retained under exception policy v2 (bound to
> the sample by a SHA-256 digest over its little-endian token ids). All
> samples pass the applicable structural and batch-invariance checks.

This is not CUDA certification, not bit-exact parity, and not general
model-quality validation.

Frozen gates: dim 2048, all finite, |norm-1| <= 1e-5 (FP64), cosine >
0.999, max_abs <= 0.01. Batch invariance gates (cosine > 0.9999, max_abs
<= 0.005) were set from measured vLLM self-sensitivity (0.99992 / 0.0026),
not guessed.

| Suite (corrected harness, commit 6a6748a, `--locked` build) | Result |
|---|---|
| Table parity (RoPE inv_freq / cos-sin / llama_4_scaling vs vLLM formulas) | inv_freq max_rel 1.2e-7; cos/sin max_abs 8.3e-5; llama_4_scaling bit-exact |
| Gating corpus, 21 singleton prompts (len 1-1024: boundaries, EOS, repeats incl. two distinct repeated ids, varied natural text, special-token literals, punctuation) | 21/21 PASS; worst cosine 0.999660; max_abs <= 0.0092 |
| Diagnostic stress case (4 ids repeated to 1023) | cosine 0.997910 — reported, not gated; the vLLM reference itself is batch-unstable on this input (single-vs-mixed cosine 0.99885) |
| Batch invariance vs singleton baselines (grouped, reversed, deterministically shuffled under serving budgets; duplicates) n=68 | worst cosine 1.00000000, max_abs 1e-6 |
| Repeated identical-shape execution | bit-identical (to_bits) |
| Invalid inputs (empty batch / empty sequence / pad id / over-length / out-of-vocab) | all rejected |
| Tokenizer equality (Rust `tokenizers` vs reference, 7 cases: multilingual, whitespace, punctuation, special-token literals) | 7/7 exact |
| Bidirectionality (stage level: later-token edit changes earlier-token state) | confirmed |

Harness properties (per external review): singleton baselines computed
explicitly; batch suites respect serving budgets on both actual and padded
tokens; every compared row and the reference file validated fail-closed
before any metric reduction; unknown `config.json` keys fail closed
(semantic manifest); artifact hashes below tie the run to exact inputs.

Cross-capture note: vLLM singleton embeddings are not bit-identical
across captures (observed max_abs <= 1.2e-08, cosine 1.00000000 — f32
rounding-level kernel nondeterminism).

Raw outputs: `results/m1_acceptance_v3_results.txt` (current, locked),
`results/cargo_test_results.txt` (9/9, `cargo test --locked`),
`results/m1_core_results.txt`, `results/table_parity_results.txt`.
Artifact SHA-256 manifest is printed by the acceptance run itself
(checkpoint, config, tokenizer, engine, corpus, reference, lockfile).

## Reproduce

```
pip install vllm==0.25.0 tokenizers numpy   # reference capture only
modal run tools/capture_expanded.py         # regenerates corpus + reference (base64 tar in log)
cargo run --release --locked --bin parity_run -- <MODEL_DIR> \
    results/prompts.json results/ref_single.f32 <MODEL_DIR>/tokenizer.json \
    results/tokenization_cases.json results/exception_policy.json
```

`<MODEL_DIR>` is a local copy of the Hugging Face checkpoint with
`engine.json.example` copied to it as `engine.json`.
