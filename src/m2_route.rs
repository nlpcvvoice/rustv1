// src/m2_route.rs

use anyhow::Result;
use candle_core::Device;
use std::path::Path;

use crate::{
    cuda_math::MathModePin,
    model::{EngineConfig, IndependentExecutor},
};

/// Open the designated M2 in-process token-ID route (backwards compatible).
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

/// Open the designated M2 route with optional B1 launch attestation SHA256 binding.
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
