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

## M2-B200 — native sm_100 route with independent vLLM@B200 reference (contract b200-independent-F-v2)

Completed 2026-10-25 on Modal B200 (ephemeral, no persistent storage).

| Stage | Evidence | Result |
|---|---|---|
| Qualification probe (B200 identity, capability 10.0, engine version, singleton repeatability, norm gates) | `results/b200_qualprobe_run7.txt`, `results/b200_qualification/qualification.json` | pass=true |
| Independent reference capture (vLLM 0.25.0, 22 frozen inputs, 3 passes) | `results/b200_capture_summary.txt`, `results/ref_single_b200.f32` (sha256 37e99147…) | success=true |
| Acceptance suite (prebuilt sm_100 binary vs B200 goldens; 22 singletons, 8 closure groups, 5 recovery groups, exception groups) | `results/m2_acceptance_b200_v2_records.txt` | ACCEPT_RC 0, accepted=true |

Gate outcomes: 22/22 singletons within cos>0.999 and max_abs<=0.01
(min cosine 0.99961, max abs 0.00616); all closure/recovery/exception
records pass; audit identity on every physical-forward record; reduced-
precision flags false; poison and refusal groups pass.

Honest non-determinism record: vLLM re-passes are not bit-identical
(min cosine 1.000000000, max abs jitter 3.4e-08), consistent with
flash-attn scheduling; gates are orders of magnitude above this jitter.

Audit binding: interim artifact v1.1 (`audits/cuda_math/candle-0.9.2_cudarc-0.19.10.v1.json`,
integrator-produced, signed=false) binds the current Cargo.lock and main
source bytes per auditor response #26; auditor-signed v2 is pending.

This section supersedes the L4 scope note above for the B200 route.
L4 evidence is retained unchanged as the historical record that motivated
this contract (sm_89 PTX-JIT numeric drift, see `m2_ablation_analysis.md`).

## M3 preparation — candidate golden export and served-fixture domain

The M3 HTTP acceptance scope is frozen by the 46-entry mapping array
(`results/golden_export_b200/mappings_v1.json`, delivered verbatim by
external architecture consultation #30 and mechanically verified on
receipt: 46 entries, chunk coverage 0..21 for the reverse and shuffled
groups, frozen wrapping-u64 permutation byte-for-byte).

Served fixtures (`results/golden_export_b200/fixtures_exec.json`): 22
entries, each with text, token ids, `ids_sha256`, `golden_row`, and
`golden_sha256`. Nineteen are exact-id fixtures whose text re-tokenizes
under the production tokenizer to exactly the frozen corpus token
arrays; three (slots 2, 4, 5; lengths 255, 257, 1023) are
length-preserving substitutes carrying their own candidate goldens.

Candidate goldens were exported on B200 (sm_100 native) by
`src/bin/golden_export.rs` through the designated M2 token-ID route:
25 rows x 2048 f32 (`golden_export.golden.f32`, matrix SHA-256
`817c460fc1891a49b43eae043bfc6c45d2afb20e889422097b65591876107cc6`),
per-row digests in `golden_export.golden_rows.json`. Run: 130 s on one
B200, zero billed cost; inputs and substitute texts committed for
provenance. Pad-token coverage input (`"<pad>"` -> id 11) verified
against the production tokenizer path.

## M3 — HTTP serving acceptance (contract m3-http-independent-F-v1)

Completed 2026-10-27 on Modal B200 (ephemeral, no persistent storage,
billed cost $0).

| Stage | Evidence | Result |
|---|---|---|
| Auditor-signed CUDA math source audit v2 | `audits/cuda_math/candle-0.9.2_cudarc-0.19.10.v2.json` (sha256 `c42efd53…`) | status=accepted, signed=true (attributed review sign-off), mechanical=true |
| Mechanical region-boundary re-verification at integrated HEAD 39c9b31 | `audits/cuda_math/mechanical_reverification_report_final_39c9b31.json` (ALL_PASS 102/102, sha256 `cdfa17f3…`), plus evidence supplement and archive-digest reconciliation | all_pass=true |
| HTTP acceptance run 1 + destructive policy-drift + supervised restart run 2 | `results/m3/accept_records.txt` (64 case records, 64 pass / 0 fail), `results/m3/executor_manifest_run1.jsonl` (227 records), `results/m3/executor_manifest_run2.jsonl` (7 records), `results/m3/att_run{1,2}.json` | ACCEPT_RC 0, `{"accepted":true,"pass":true}` |

Gate outcomes: all 22 served fixtures over HTTP matched the frozen
candidate goldens by exact output SHA-256 (not cosine tolerance);
invalid-request, exception, recovery, quiesce, policy-drift poison
(500 + post-poison refusal), and supervised-restart lifecycle cases
all pass; every physical-forward manifest record carries the audit
identity and profile-F math block; both run attestations record the
same binary SHA-256.

Integration fixes made during M3 acceptance (post-audit, outside the
audited source set): unify server instance identity with the launch
attestation (5ca5a89); quiesce admission before arming policy drift
in the driver (2ba851d). Both were cross-component consistency bugs
in the HTTP stack, caught by the acceptance gates themselves.

Scope note: HTTP certification is served-fixture numerical
acceptance. Throughput/latency baselining is a separate pending
M3-bench measurement.

## M3-bench — B=1 throughput/latency baseline (B200, same audited bytes)

Closed-loop same-host client, 22-fixture mixed tier (avg ISL ~215) and
ISL-1024 tier, 60 s measured per level, 0 errors at every level.
Captured 2026-10-27 (mixed tier complete; long tier C=1 only —
saturation already established, sweep intentionally stopped).

| Concurrency | req/s | tokens/s | p50 ms | p90 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 78 | 16.7k | 7.4 | 39.0 | 39.9 |
| 2 | 82 | 17.7k | 17.6 | 44.8 | 55.0 |
| 4 | 83 | 17.8k | 42.6 | 72.2 | 89.0 |
| 8 | 82 | 17.7k | 93.9 | 124.1 | 149.7 |
| 16 | 82 | 17.7k | 192.5 | 222.2 | 248.6 |
| 32 | 83 | 17.7k | 388.7 | 425.7 | 468.9 |
| 64 | 83 | 17.7k | 774.0 | 824.4 | 888.8 |

ISL-1024, C=1: 25.8 req/s, 26.4k tokens/s, p50 38.6 ms.

Reading: throughput saturates at ~83 req/s from C=2 upward — the
contract-mandated serial B=1 executor is the ceiling; added
concurrency only queues latency. This is the measured baseline the
batching work is measured against.

## M4 (in progress) — server-side batching integration

Batched executor, bounded FIFO batch scheduler, and server wiring
authored by the external architect (#32a / #32a-continuation rounds)
were integrated verbatim at 48a632c; rustfmt/clippy 1.99.1
compliance applied via the cifix loop; one integration fix at 5b9e1ad
(`Atomic::fetch_update` -> `try_update` rename; `ExecutionRow::validate`
gated for CPU-only builds). All gates green: Modal m3verify
(FMT/CPU_CHECK/CPU_TEST/CUDA_CHECK/CLIPPY) and GitHub CI (6 jobs).

Driver-side components integrated since (all via the same
external-architect rounds, verbatim + fmt/clippy fixes, every step
gated by Modal m3verify and GitHub CI):

| Component | Commit | Role |
|---|---|---|
| `golden_export_m4` | (v32b round) | batch-capable golden export: batched goldens + executor manifest + provenance, audit identity bound at runtime |
| `accept_m4` thin bin + contract + prepass module | c1eb284 | in-process pre-pass: 15-role batch evidence consumed by finalize; materialization contract (fixtures_exec 22, mappings_v1 46) |
| acceptance-http module | 05b668c, 2ca6996 | live serialized HTTP-to-executor reconciliation: per-case connection isolation, observation/manifest prefix invariants, zero-execution proof for rejections, four-way association for positives, handoff v1 contract |

Not yet accepted: the remaining driver phases (finalized-policy
loading, the 46 mappings in both modes, scheduler/lifecycle
orchestration families) are still in request (round #32d onward), and
an audit v3 renewal is mandatory before any M4 acceptance run —
`serve_m3` rejects an audit artifact identical to the B1 base, and
audit v2 explicitly covers physical B=1 only.
