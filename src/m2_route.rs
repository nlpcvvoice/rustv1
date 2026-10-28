use anyhow::Result;
use candle_core::Device;
use std::path::Path;

use crate::{
    cuda_math::MathModePin,
    model::{BatchedExecution, EngineConfig, IndependentExecutor},
};

/// Compatibility route: forward_serving executes physical B=1.
pub fn open(
    model_dir: &Path,
    manifest_path: &Path,
    device_ordinal: usize,
    audit_identity: &str,
) -> Result<IndependentExecutor> {
    open_with_attestation(
        model_dir,
        manifest_path,
        device_ordinal,
        audit_identity,
        None,
    )
}

/// Compatibility route with launch-attestation digest binding.
///
/// Pin acquisition precedes CUDA device and model initialization.
/// The caller remains responsible for authenticating the audit identity
/// and attestation artifact before opening this route.
pub fn open_with_attestation(
    model_dir: &Path,
    manifest_path: &Path,
    device_ordinal: usize,
    audit_identity: &str,
    attestation_sha256: Option<&str>,
) -> Result<IndependentExecutor> {
    let math = MathModePin::pin(audit_identity)?;

    let engine: EngineConfig =
        serde_json::from_slice(&std::fs::read(model_dir.join("engine.json"))?)?;

    let device = Device::new_cuda(device_ordinal)?;

    IndependentExecutor::load_with_attestation(
        model_dir,
        device,
        engine,
        math,
        manifest_path,
        attestation_sha256,
    )
}

/// Explicit batched route. No existing caller is silently migrated.
///
/// The supplied policy must fit the bound engine configuration.
/// forward_serving remains available for exact-bit singleton startup
/// checks; forward_batch performs one physical batch per call.
///
/// The B=1-only source audit is not sufficient evidence for this route.
pub fn open_batched_with_attestation(
    model_dir: &Path,
    manifest_path: &Path,
    device_ordinal: usize,
    audit_identity: &str,
    attestation_sha256: Option<&str>,
    batching: BatchedExecution,
) -> Result<IndependentExecutor> {
    batching.validate()?;
    let math = MathModePin::pin(audit_identity)?;

    let engine: EngineConfig =
        serde_json::from_slice(&std::fs::read(model_dir.join("engine.json"))?)?;
    batching.validate_engine(&engine)?;

    let device = Device::new_cuda(device_ordinal)?;

    IndependentExecutor::load_batched_with_attestation(
        model_dir,
        device,
        engine,
        math,
        manifest_path,
        attestation_sha256,
        batching,
    )
}
