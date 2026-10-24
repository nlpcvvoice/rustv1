# RustV1

RustV1 is a model–hardware co-design reference implementation for Nemotron-class BF16 embedding inference using Candle and CUDA. It emphasizes an explicit precision contract, independently executed sequences, reproducible acceptance evidence, and fail-closed inference. The accepted CUDA route uses physical batch size one and validates numerical agreement with a frozen reference separately from exact output-bit identity across independent execution sequences.

## Repository layout

| Path | Purpose |
|---|---|
| `Cargo.toml`, `Cargo.lock` | Package configuration and locked dependencies |
| `src/lib.rs` | Library entry point and feature-gated exports |
| `src/precision.rs` | Precision profiles, attention boundaries, and comparison helpers |
| `src/cuda_math.rs` | Process-global CUDA math-mode control and validation |
| `src/m2_route.rs` | M2 CUDA route integration |
| `src/bin/parity_run.rs` | Parity driver; CUDA execution uses the M2 acceptance suite |
| `src/bin/accept_m2.rs` | Standalone M2 acceptance driver |
| `src/bin/replay_common_input.rs` | Common-input replay diagnostics |
| `src/bin/stage_dump.rs` | Intermediate-stage diagnostics |
| `audits/cuda_math/` | Versioned CUDA math source-audit artifacts |
| `.github/workflows/ci.yml` | Formatting, CPU tests, and CUDA compile/lint checks |

## Build

Run commands from the package root containing `Cargo.toml`.

CPU development checks:

```sh
cargo fmt --check
cargo test --locked
cargo check --locked
cargo build --release --locked --bin parity_run
```

CUDA builds require the CUDA development toolkit and a compatible Rust toolchain:

```sh
cargo check --locked --features cuda --bins
cargo clippy --locked --features cuda -- -D warnings
cargo build --release --locked --features cuda --bin parity_run --bin accept_m2
```

Compile-only CI uses a CUDA 12.4.1 development image with explicit compute
capability 89 code generation. It does not require a GPU and does not establish
runtime correctness.

Actual CUDA execution requires a compatible GPU, driver, and runtime libraries.
B200 execution requires its own supported toolkit and build configuration; the
compile-only compute-capability-89 job is not B200 validation.

## Run

Model weights are not stored in this repository; `<MODEL_DIR>` is a local
directory containing `config.json`, `engine.json`, `model.safetensors`, and
`tokenizer.json` for the frozen checkpoint.

CPU parity (FP32 oracle, profile A):

```sh
cargo run --release --locked --bin parity_run -- \
  <MODEL_DIR> results/prompts.json results/ref_single.f32 \
  <MODEL_DIR>/tokenizer.json results/tokenization_cases.json \
  results/exception_policy.json cpu,f32 A
```

CUDA parity (profile F; runs the shared M2 acceptance suite, then tokenizer
equality):

```sh
cargo run --release --locked --features cuda --bin parity_run -- \
  <MODEL_DIR> results/prompts.json results/ref_single.f32 \
  <MODEL_DIR>/tokenizer.json results/tokenization_cases.json \
  results/exception_policy_m2.json cuda,bf16 F
```

Standalone CUDA M2 acceptance (requires the validated audit artifact selected
via `RV1_CUDA_MATH_AUDIT`, see below):

```sh
export RV1_CUDA_MATH_AUDIT="sha256:<64-lowercase-hex-of-artifact-bytes>:audits/cuda_math/candle-0.9.2_cudarc-0.17.8.v1.json"
cargo run --release --locked --features cuda --bin accept_m2 -- \
  <MODEL_DIR> m2_executor_manifest.jsonl results/exception_policy_m2.json
```

CPU-only builds and parity runs are useful development checks. They do not pass
the CUDA M2 gate. A CPU-only invocation of `accept_m2`, if the binary exposes
one, must report that CUDA acceptance is unavailable rather than report success.

The CUDA branch of `parity_run` invokes the shared M2 acceptance suite and then
checks tokenizer equality. The CPU route remains separate.

## Frozen M2 contract

- Exactly 22 frozen samples.
- Comparison against the immutable L4 reference.
- Per sample: cosine similarity strictly greater than `0.999`.
- Per sample: maximum absolute component error at most `0.01`.
- Exact output-bit identity across the required independent-sequence tests.
- Empty exception list.
- PrecisionProfile F.
- Physical execution batch size `B=1`.
- Sequence length at most 1024 tokens.

A logical request containing several sequences does not authorize a physical
multi-sequence tensor batch.

Numerical agreement with the reference and bit identity across repeated or
reordered independent executions are separate requirements. Passing one does
not imply the other. Cross-device bit identity is not assumed.

## CUDA math audit

The versioned source-audit artifact is:

```text
audits/cuda_math/candle-0.9.2_cudarc-0.17.8.v1.json
```

Select the completed, validated artifact before starting CUDA execution. The
environment value binds the artifact path to the SHA-256 of its exact bytes:

```sh
AUDIT=audits/cuda_math/candle-0.9.2_cudarc-0.17.8.v1.json
export RV1_CUDA_MATH_AUDIT="sha256:$(sha256sum "$AUDIT" | cut -d' ' -f1):$AUDIT"
```

Setting this variable does not approve an artifact. Missing, incomplete,
mismatched, or rejected audit evidence must fail closed. Draft artifacts with
unresolved fields are not acceptance evidence.

The final audit binds source findings to exact source hashes, locked
dependencies, resolved features, and the audited repository revision. It
documents the requested compute types and dispatch paths; it does not prove
the internal cuBLAS algorithm or replace GPU acceptance tests.

The application pins the required process-global math settings before CUDA
device creation and checks them around model execution. Other code must not
mutate these settings or the executor's cuBLAS configuration.

A poisoned executor must not be reused. Recovery requires a fresh process and
the normal initialization and validation sequence.

## Acceptance evidence

Archive the following for each accepted configuration:

- Code revision and dependency/build provenance.
- Completed CUDA math audit and verification evidence.
- GPU, driver, CUDA, and cuBLAS identities.
- Frozen input, tokenizer, model, and reference identities.
- Raw acceptance logs and output hashes.
- All closure-group results and executed-shape records.
- Numerical comparison and bit-identity results.
- Failure-path and poison-persistence results.
- Throughput and latency measurements with explicit workload and concurrency.

CI compilation is not GPU acceptance. A passing source audit is not numerical
acceptance.

## Milestone boundaries

M2 retains the frozen BF16/profile-F contract and independent-sequence execution.

M3 NVFP4 inference and HTTP serving are separate design work. They must not
silently change M2 precision, batching, error behavior, or acceptance thresholds.
