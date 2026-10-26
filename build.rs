// build.rs

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=RV1_CUDA_COMPUTE_CAP");

    // 1. Rustc version
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let rustc_version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "rustc 1.83.0".to_string());

    // 2. NVCC version
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let nvcc_version = Command::new(nvcc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|s| {
            s.lines()
                .find(|l| l.contains("release"))
                .map(|l| l.trim().to_string())
        })
        .unwrap_or_else(|| "Cuda compilation tools, release 12.8".to_string());

    // 3. Compute capability
    let compute_cap =
        std::env::var("RV1_CUDA_COMPUTE_CAP").unwrap_or_else(|_| "sm_100".to_string());

    // 4. Target features
    let target_features =
        std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_else(|_| "fxsr,sse,sse2".to_string());

    println!("cargo:rustc-env=RV1_BUILD_RUSTC_VERSION={rustc_version}");
    println!("cargo:rustc-env=RV1_BUILD_NVCC_VERSION={nvcc_version}");
    println!("cargo:rustc-env=RV1_BUILD_CUDA_COMPUTE_CAP={compute_cap}");
    println!("cargo:rustc-env=RV1_BUILD_TARGET_FEATURES={target_features}");
}
