use anyhow::{ensure, Result};
use candle_core::cuda_backend::{
    gemm_reduced_precision_bf16,
    gemm_reduced_precision_f16,
    gemm_reduced_precision_f32,
    set_gemm_reduced_precision_bf16,
    set_gemm_reduced_precision_f16,
    set_gemm_reduced_precision_f32,
};
use serde_json::{json, Value};
use std::{
    cell::Cell,
    marker::PhantomData,
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
    thread::{self, ThreadId},
};

// This prevents duplicate owners through THIS module. It cannot interpose on
// calls to Candle's public setters made elsewhere in the process.
static PIN_ACTIVE: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy)]
struct Settings {
    f32: bool,
    f16: bool,
    bf16: bool,
}

impl Settings {
    fn read() -> Self {
        Self {
            f32: gemm_reduced_precision_f32(),
            f16: gemm_reduced_precision_f16(),
            bf16: gemm_reduced_precision_bf16(),
        }
    }

    fn all_disabled(self) -> bool {
        !self.f32 && !self.f16 && !self.bf16
    }

    fn json(self) -> Value {
        json!({
            "f32": self.f32,
            "f16": self.f16,
            "bf16": self.bf16
        })
    }
}

/// Process-wide Candle CUDA GEMM policy owner.
///
/// Operational contract:
/// - Acquire before model loading, warmup, or any GPU work.
/// - No other code may change Candle's reduced-precision settings.
/// - No other executor may overlap this executor's device/stream work.
/// - Retain this object for the entire execution lifetime.
///
/// These are operational restrictions, not Rust memory-safety preconditions.
/// Therefore construction is safe and does not accept a raw CUDA handle.
pub struct MathModePin {
    audit_identity: String,
    original: Settings,
    owner: ThreadId,
    drifted: Cell<bool>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl MathModePin {
    pub fn pin(audit_identity: impl Into<String>) -> Result<Self> {
        let audit_identity = audit_identity.into();
        ensure!(
            !audit_identity.trim().is_empty(),
            "missing locked-source audit identity"
        );

        ensure!(
            PIN_ACTIVE
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            "a process-global MathModePin already exists"
        );

        // Construct the guard before changing settings so any later error
        // releases PIN_ACTIVE via Drop.
        let pin = Self {
            audit_identity,
            original: Settings::read(),
            owner: thread::current().id(),
            drifted: Cell::new(false),
            _not_send_sync: PhantomData,
        };

        // These are three separate atomic stores, not a transactional update.
        // The startup-before-work contract is therefore essential.
        set_gemm_reduced_precision_f32(false);
        set_gemm_reduced_precision_f16(false);
        set_gemm_reduced_precision_bf16(false);

        pin.check()?;
        Ok(pin)
    }

    pub fn check(&self) -> Result<Value> {
        ensure!(
            thread::current().id() == self.owner,
            "math policy owner thread changed"
        );
        ensure!(
            !self.drifted.get(),
            "math policy previously drifted; restart the process"
        );

        let observed = Settings::read();
        if !observed.all_disabled() {
            self.drifted.set(true);
            anyhow::bail!(
                "Candle GEMM policy drift: f32={}, f16={}, bf16={}; restart required",
                observed.f32,
                observed.f16,
                observed.bf16
            );
        }

        Ok(json!({
            "policy": "candle_cuda_gemm_no_reduced_precision_v1",
            "verification": "source_audit_plus_process_global_pin",
            "audit_identity": self.audit_identity,
            "candle_core_version": "0.9.2",
            "control_scope": "process_global_Candle_CUDA_dispatch",
            "reduced_precision": observed.json(),
            "original_reduced_precision": self.original.json(),
            "audited_compute_type": {
                "f32_strided_batched": "CUBLAS_COMPUTE_32F",
                "bf16_strided_batched": "CUBLAS_COMPUTE_32F"
            },
            "profile_F_mapping": {
                "bf16_projections": "MM_BF16_REDUCED_PRECISION",
                "widened_f32_QK": "MM_F32_REDUCED_PRECISION",
                "widened_f32_PV": "MM_F32_REDUCED_PRECISION",
                "f16": "not_used"
            },
            "ownership": "one_policy_owner_no_other_setter_calls",
            "check_semantics": "snapshot_with_sticky_detected_drift",
            "raw_handle_attributes_queried": false,
            "cublas_atomics_mode_pinned": false,
            "internal_kernel_selection_verified": false,
            "NVIDIA_TF32_OVERRIDE": std::env::var_os("NVIDIA_TF32_OVERRIDE")
                .map(|v| v.to_string_lossy().into_owned()),
            "CUBLAS_WORKSPACE_CONFIG": std::env::var_os("CUBLAS_WORKSPACE_CONFIG")
                .map(|v| v.to_string_lossy().into_owned())
        }))
    }
}

impl Drop for MathModePin {
    fn drop(&mut self) {
        // Deliberately leave all settings unchanged. Restoring old settings
        // would introduce an implicit global policy transition.
        PIN_ACTIVE.store(false, Ordering::Release);
    }
}
