use anyhow::Result;
use candle_core::Device;
use std::path::Path;

use crate::{
    cuda_math::MathModePin,
    model::{EngineConfig, IndependentExecutor},
};

/// Open the designated M2 in-process token-ID route.
///
/// Call on the thread that will own and use the executor. Do not move the
/// executor to another thread; it is intentionally !Send + !Sync.
pub fn open(
    model_dir: &Path,
    manifest_path: &Path,
    device_ordinal: usize,
    audit_identity: &str,
) -> Result<IndependentExecutor> {
    // Pin before device/model initialization or warmup.
    let math = MathModePin::pin(audit_identity)?;

    let engine: EngineConfig =
        serde_json::from_slice(&std::fs::read(model_dir.join("engine.json"))?)?;

    let device = Device::new_cuda(device_ordinal)?;

    IndependentExecutor::load(model_dir, device, engine, math, manifest_path)
}
