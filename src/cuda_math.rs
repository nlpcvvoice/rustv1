use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{
    ffi::c_void,
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
    thread::{self, ThreadId},
};

#[link(name = "cublas")]
unsafe extern "C" {
    fn cublasGetVersion_v2(handle: *mut c_void, version: *mut i32) -> i32;
    fn cublasGetStream_v2(handle: *mut c_void, stream: *mut *mut c_void) -> i32;
    fn cublasGetMathMode(handle: *mut c_void, mode: *mut i32) -> i32;
    fn cublasSetMathMode(handle: *mut c_void, mode: i32) -> i32;
    fn cublasGetAtomicsMode(handle: *mut c_void, mode: *mut i32) -> i32;
    fn cublasSetAtomicsMode(handle: *mut c_void, mode: i32) -> i32;
}

const CUBLAS_DEFAULT_MATH: i32 = 0;
const CUBLAS_ATOMICS_NOT_ALLOWED: i32 = 0;

fn status(code: i32, operation: &str) -> Result<()> {
    ensure!(code == 0, "{operation}: cuBLAS status {code}");
    Ok(())
}

pub struct MathModePin {
    handle: NonNull<c_void>,
    stream: *mut c_void,
    version: i32,
    original_math_mode: i32,
    original_atomics_mode: i32,
    audit_identity: String,
    owner: ThreadId,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl MathModePin {
    /// # Safety
    ///
    /// - handle is Candle's LIVE cuBLAS handle, not a newly created handle.
    /// - handle remains alive for this object's entire lifetime.
    /// - this thread owns the handle and its stream exclusively.
    /// - the locked backend call path was audited: it does not override math
    ///   mode per operation, use FAST_TF32 compute types, or route these FP32
    ///   matmuls through a separately configured cuBLASLt handle.
    /// - audit_identity identifies the audited backend sources/build.
    ///
    /// This function pins DEFAULT_MATH and disallows atomics. If those differ
    /// from the accepted F run, this is a changed configuration requiring a
    /// fresh unchanged-threshold acceptance run, not retrospective certification.
    pub unsafe fn from_audited_candle_handle(
        handle: *mut c_void,
        audit_identity: String,
    ) -> Result<Self> {
        ensure!(!audit_identity.trim().is_empty(), "missing backend audit identity");
        let handle = NonNull::new(handle).ok_or_else(|| anyhow::anyhow!("null handle"))?;

        let mut version = 0;
        let mut stream = std::ptr::null_mut();
        let mut original_math_mode = 0;
        let mut original_atomics_mode = 0;

        unsafe {
            status(cublasGetVersion_v2(handle.as_ptr(), &mut version), "get version")?;
            status(cublasGetStream_v2(handle.as_ptr(), &mut stream), "get stream")?;
            status(cublasGetMathMode(handle.as_ptr(), &mut original_math_mode), "get math")?;
            status(
                cublasGetAtomicsMode(handle.as_ptr(), &mut original_atomics_mode),
                "get atomics",
            )?;
            status(
                cublasSetMathMode(handle.as_ptr(), CUBLAS_DEFAULT_MATH),
                "set math",
            )?;
            status(
                cublasSetAtomicsMode(handle.as_ptr(), CUBLAS_ATOMICS_NOT_ALLOWED),
                "set atomics",
            )?;
        }

        let result = Self {
            handle,
            stream,
            version,
            original_math_mode,
            original_atomics_mode,
            audit_identity,
            owner: thread::current().id(),
            _not_send_sync: PhantomData,
        };
        result.check()?;
        Ok(result)
    }

    pub fn check(&self) -> Result<Value> {
        ensure!(thread::current().id() == self.owner, "cuBLAS owner thread changed");
        let mut stream = std::ptr::null_mut();
        let mut math = -1;
        let mut atomics = -1;
        unsafe {
            status(cublasGetStream_v2(self.handle.as_ptr(), &mut stream), "get stream")?;
            status(cublasGetMathMode(self.handle.as_ptr(), &mut math), "get math")?;
            status(cublasGetAtomicsMode(self.handle.as_ptr(), &mut atomics), "get atomics")?;
        }
        ensure!(stream == self.stream, "Candle cuBLAS stream changed");
        ensure!(math == CUBLAS_DEFAULT_MATH, "cuBLAS math mode changed: {math}");
        ensure!(atomics == CUBLAS_ATOMICS_NOT_ALLOWED,
            "cuBLAS atomics mode changed: {atomics}");

        Ok(json!({
            "cublas_version": self.version,
            "handle_identity": format!("{:p}", self.handle.as_ptr()),
            "stream_identity": format!("{:p}", stream),
            "math_mode": math,
            "math_mode_name": "CUBLAS_DEFAULT_MATH",
            "atomics_mode": atomics,
            "original_math_mode": self.original_math_mode,
            "original_atomics_mode": self.original_atomics_mode,
            "fp32_call_path_audit": self.audit_identity,
            "tf32_claim_basis": "handle_query_plus_locked_call_path_audit",
            "NVIDIA_TF32_OVERRIDE": std::env::var("NVIDIA_TF32_OVERRIDE").ok(),
            "CUBLAS_WORKSPACE_CONFIG": std::env::var("CUBLAS_WORKSPACE_CONFIG").ok()
        }))
    }
}
