# rustv1

Experimental Rust embedding server for `nvidia/Nemotron-3-Embed-1B` (NVFP4, B200-class GPUs).
Goal: close the long-input-sequence throughput gap measured in
[embedding-serving-bench](https://github.com/nlpcvvoice/embedding-serving-bench)
at ISL=1024, by building a small, fully owned inference stack.

Status: **correctness bring-up in progress**. The loader is fail-closed: it
refuses to run until every model-contract detail is verified against the
reference implementation. See [Contract](#contract) and [Open items](#open-items).

## Design

| Layer | Choice | Rationale |
|---|---|---|
| Serving | axum HTTP `/v1/embeddings` | OpenAI-compatible surface, bounded admission |
| Batching | whole-sequence FIFO with bounded coalescing | deterministic padded-token budgets |
| Executor | single model thread, candle tensors | correctness oracle first; cuBLASLt / fused kernels / CUDA graphs later |
| Tokenizer | `tokenizers` crate, no hidden truncation/padding | server owns padding |

## Contract (frozen from the checkpoint `config.json`)

| Property | Value | Enforcement |
|---|---|---|
| `head_dim` | 128 (hidden 2048 / 24 heads = 85.33 is **not** head_dim) | assert at load |
| `is_causal` | `false` (bidirectional) | assert; no triangular mask |
| `pooling` | `avg` over valid tokens, after final RMSNorm, then L2 | assert; fixed order |
| `pad_token_id` | 11; server-owned; client inputs containing 11 are rejected | assert + request validation |
| `rope_parameters` | `rope_type: yarn`, factor 16, original 16384, `apply_yarn_scaling: false` | loader refuses until verified |
| `llama_4_scaling` | present (beta 0.1) | loader refuses until verified |
| `rms_norm_eps` | 1e-5 | used |
| `tie_word_embeddings` | true (embed only; no lm_head in embedding path) | — |

## Correctness plan

1. Capture reference outputs from vLLM 0.25.0 (`tools/capture_reference.py`):
   tokenizer facts, deterministic 255/256/1024-token prompts, pooled embeddings.
2. Extract the installed reference implementation of YaRN / llama-4 scaling
   (the script harvests the relevant source files) instead of guessing formulas.
3. Wire RoPE only after comparing `inv_freq` / cos-sin tables at positions
   0, 1, 255, 1023 against the reference.
4. Per-sample cosine similarity > 0.999 vs reference on CPU, then GPU.
5. Throughput sweep with `src/bin/bench.rs` against the vLLM baseline.

## Open items

| Item | Blocking? |
|---|---|
| Reference YaRN branch semantics (`apply_yarn_scaling=false`) | yes |
| `llama_4_scaling` exact formula (position vs length, boundary, rounding) | yes |
| NVFP4 tensor inventory + scale convention (global scale vs reciprocal) | after BF16 parity |
| Fused attention / cuBLASLt / CUDA graphs | after parity |

## License

MIT
