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

---

## M2 — Audited CUDA execution and L4 acceptance

**Status: L4 acceptance passed; M2 L4 closure signed off.**  
**B200 final acceptance: pending.**

| Identity | Value |
|---|---|
| Acceptance commit | `ea0446f` |
| Release tag | `v0.4.0-m2` — annotated at the acceptance commit |
| Acceptance executable | `accept_m2` |
| Audited candle-core version | `0.9.2` |
| Audited cudarc version | `0.19.10` |
| Audit artifact commit | `b637193` |
| Audit artifact | `audits/cuda_math/candle-0.9.2_cudarc-0.19.10.v1.json` |
| Acceptance records | `results/m2_acceptance_l4_v2_records.txt` |
| Physical-forward manifest | `results/m2_executor_manifest_l4.jsonl` |

### Execution contract

M2 establishes an audited CUDA execution route with the following contract:

- The process-global math-policy pin is established before the execution
  CUDA device is created. The CUDA availability check in `parity_run`
  creates no pre-pin CUDA device.
- The execution route requires the expected audit identity. Missing or
  mismatched audit identity fails closed.
- Reduced-precision GEMM policy is disabled for `f32`, `f16`, and `bf16`
  throughout the accepted physical forwards.
- Valid requests satisfy the numerical acceptance criteria against the
  frozen L4 reference.
- The tested batching, ordering, duplicate-input, and length variations
  produce outputs bit-identical to their singleton baselines.
- Invalid requests are rejected without poisoning the executor or adding
  physical-forward manifest records.
- Detected Candle GEMM policy drift fails closed, poisons the executor,
  and prevents subsequent execution.
- The accepted exceptions list is empty.

Numerical agreement and bit identity are distinct checks: singleton
outputs are compared numerically with the frozen reference, while
batching and ordering invariants are checked bit-for-bit against the
accepted singleton outputs. These results apply to the tested suite and
accepted configuration.

### Audit chain

| Check | Evidence / result |
|---|---|
| Source-region verification | 100/100 examined regions mechanically verified against exact locked bytes; 0 problems |
| Audit coverage corrections | Includes the grouped GEMM entry and corrected `set_stream` position |
| Provenance | Repository revision and RFC3339 timestamp recorded |
| Dependency binding | `Cargo.lock` SHA-256 binding and four crate lock checksums recorded |
| Feature-tree preservation | `audits/cuda_math/cargo_tree_cuda.txt`, SHA-256-bound |
| Runtime identity checks | `audit_identity()` enforces artifact hash, schema ID, candle-core `0.9.2`, cudarc `0.19.10`, and `Cargo.lock` binding |
| Missing or mismatched identity | Fails closed; exercised in earlier runs |
| Acceptance-run identity | Every physical-forward record carries the audit identity |

The signed artifact is a versioned, dependency-bound audit. Changes to
audited dependency bytes, the lockfile binding, or relevant feature
selection require renewed validation rather than inheriting this result.

### Acceptance matrix

| Area | Result |
|---|---|
| Overall acceptance | `accepted=true`; 0 failed records out of 57 |
| Group 1: singleton/reference comparison | 22/22 passed against the frozen L4 reference |
| Worst singleton cosine similarity | `0.999724` |
| Maximum singleton `max_abs` | `0.006184` |
| Groups 2–6: batching and ordering invariants | 46 invariant groups passed; all rows bit-identical to singleton baselines |
| Group 7: invalid requests | All five invalid-request classes rejected |
| Group 7: recovery | Executor remained usable; immediate recovery singletons were bit-identical |
| Group 8: injected policy drift | Expected error, executor poisoning, and subsequent poison-guard refusal |
| Physical-forward policy | 111 records; every record reports `math.reduced_precision = {f32:false, f16:false, bf16:false}` |
| Physical-forward audit attribution | Audit identity present in all 111 records |
| Exceptions | Empty list enforced by preflight |
| CUDA tokenizer equality | 7/7 passed in the `parity_run` CUDA branch |
| M1 CPU regression | `parity_run` configuration `cpu,f32,A`: `M1-ACCEPTANCE PASS` |
| CUDA build check | `cargo check --features cuda`: exit code 0 |
| Tests | `cargo test`: exit code 0 |
| Formatting | Clean; 0-byte diff |

The 57 acceptance records, 46 invariant groups, and 111 physical-forward
records are different accounting units.

### Closure groups

| Closure group | Covered cases | Outcome |
|---|---|---|
| Audit and identity | Locked-source verification, provenance, feature tree, runtime audit binding | Closed |
| Singleton accuracy | 22 singleton comparisons against the frozen L4 reference | Closed |
| Batching and ordering | Multi-input, short plus 1024-token input, reversed order, deterministic shuffle, duplicates, three identical shapes, alternating lengths | Closed |
| Request validation | Empty batch, empty sequence, pad token, over-length input, out-of-vocabulary input | Closed |
| Fail-closed execution | Injected Candle GEMM policy drift, poisoning, global-flag restoration, post-poison refusal | Closed |
| Integration regression | Tokenizer equality and M1 CPU acceptance | Closed |
| Local hygiene | CUDA check, tests, formatting | Closed |
| Hosted CI publication | Workflow added; push pending | Outstanding publication task |
| B200 final acceptance | Separate execution against this accepted state | Pending |

### Fail-closed evidence

| Trigger | Required behavior | Observed result |
|---|---|---|
| Missing or mismatched audit identity | Refuse execution | Exercised in earlier runs |
| `empty_batch` | Reject without poisoning or adding forward records | Passed |
| `empty_sequence` | Reject without poisoning or adding forward records | Passed |
| `pad_token` | Reject without poisoning or adding forward records | Passed |
| `over_length` | Reject without poisoning or adding forward records | Passed |
| `out_of_vocabulary` | Reject without poisoning or adding forward records | Passed |
| Injected Candle GEMM policy drift | Return policy-drift error; poison executor; restore global flag; leave manifest unchanged | Passed; error contains `Candle GEMM policy drift` |
| Request after poisoning | Refuse at poison guard | Passed |

Immediate recovery singletons after invalid requests remained
bit-identical to baseline. This distinguishes ordinary request rejection
from a policy-integrity failure, which deliberately makes the executor
unusable.

### Reproduction

Use the accepted commit, its locked dependencies, its signed audit
artifact, and the same frozen L4 reference.

Repository and local-check commands:

```sh
git checkout --detach ea0446f

cargo check --locked --features cuda
cargo test --locked
cargo fmt --all -- --check

cargo build --locked --features cuda --bin accept_m2
```

Construct the audit environment value using the complete artifact digest,
not the abbreviated digest shown in review summaries:

```sh
AUDIT_PATH=audits/cuda_math/candle-0.9.2_cudarc-0.19.10.v1.json
AUDIT_SHA256="$(sha256sum "$AUDIT_PATH" | cut -d ' ' -f 1)"

export RV1_CUDA_MATH_AUDIT="sha256:${AUDIT_SHA256}:${AUDIT_PATH}"
```

Computing this digest does not replace verification of the accepted
artifact identity or the driver's runtime identity checks.

Run `accept_m2` with the exact launch arguments, model inputs, and frozen
reference selection used by the accepted R1 invocation. Run `parity_run`
with the accepted CUDA tokenizer suite and the `cpu,f32,A` regression
configuration.

**Reproduction-command completion note:** the sign-off package does not
include the literal `accept_m2` or `parity_run` launch commands. Insert
those recorded commands here before describing this section as a
copy-and-paste reproduction recipe; do not infer CLI flags from the
configuration labels above.

Expected acceptance result:

- `accepted=true`, with no failed acceptance records;
- no exceptions;
- all required invariant and validation checks passing;
- audit identity present on every physical-forward record;
- reduced-precision flags false for all three recorded dtypes;
- successful policy-drift poisoning and post-poison refusal.

### Scope

This section records completed L4 acceptance. It does not claim completed
B200 acceptance, successful hosted CI execution, or cross-device bit
identity.
