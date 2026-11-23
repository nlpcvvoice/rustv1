import modal

# Free CPU: v60 R1/R2 supplemental records at 778a6c6. Return via gRPC.

app = modal.App("rustv1-v60forensic")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04",
                              add_python="3.11")
    .apt_install("git", "build-essential", "curl", "libssl-dev",
                 "libclang-dev", "clang", "pkg-config")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain 1.99.1")
    .env({"PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:"
          "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "CUDA_COMPUTE_CAP": "100"})
)

REV = "778a6c6b05a0d7f8e865298dde88652b69e4aba4"


@app.function(cpu=8, memory=16384, timeout=1800, scaledown_window=10,
              image=image)
def run() -> dict:
    import subprocess
    import os
    import json
    import glob

    out: dict = {}

    def sh(cmd: str) -> tuple[int, str, str]:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

    rc, o, e = sh("git clone -q https://github.com/nlpcvvoice/rustv1.git "
                  f"/work/rustv1 && cd /work/rustv1 && git checkout -q {REV} "
                  f"&& test $(git rev-parse HEAD) = {REV}")
    print("CLONE_RC", rc, flush=True)

    # ---- R2: named-path Cargo configuration probe ----
    probe_lines = []
    for p in ["/work/rustv1/.cargo/config.toml", "/work/rustv1/.cargo/config",
              "/work/.cargo/config.toml", "/work/.cargo/config",
              "/.cargo/config.toml", "/.cargo/config"]:
        if os.path.exists(p):
            probe_lines.append(f"PRESENT {p}\n" + open(p).read())
        else:
            probe_lines.append(f"ABSENT {p}")
    home = os.environ.get("HOME", "__UNSET__")
    cargo_home_env = os.environ.get("CARGO_HOME", "__UNSET__")
    eff_home = cargo_home_env if cargo_home_env != "__UNSET__" \
        else os.path.join(home, ".cargo")
    for p in [os.path.join(eff_home, "config.toml"),
              os.path.join(eff_home, "config")]:
        if os.path.exists(p):
            probe_lines.append(f"PRESENT {p}\n" + open(p).read())
        else:
            probe_lines.append(f"ABSENT {p}")
    probe_lines.append(f"HOME={home}")
    probe_lines.append(f"CARGO_HOME_env={cargo_home_env}")
    probe_lines.append(f"effective_cargo_home={eff_home}")
    overrides = {}
    for k in sorted(os.environ):
        if (k.startswith("CARGO_") and k != "CARGO_HOME") or k in (
                "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
                "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CC", "CXX", "NVCC",
                "CARGO", "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTFLAGS",
                "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS",
                "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER",
                "CARGO_HOME", "CUDA_COMPUTE_CAP", "RV1_CUDA_COMPUTE_CAP"):
            overrides[k] = os.environ[k]
    probe_lines.append("environment_overrides_present=" +
                       json.dumps(overrides))
    probe_lines.append("all_other_CARGO_and_RUSTC_variables_absent=" +
                       str(sorted(set(
                           k for k in os.environ
                           if (k.startswith("CARGO") or
                               k.startswith("RUSTC") or k == "RUSTFLAGS")
                           and k not in overrides)) == []))
    out["r2_probe_record"] = "\n".join(probe_lines)

    # ---- R1: qualified build + retained build-script output ----
    rc, o, e = sh("cd /work/rustv1 && CARGO_TERM_COLOR=never cargo build "
                  "--release --locked --features cuda "
                  "--target x86_64-unknown-linux-gnu 2>&1; "
                  "echo CARGO_RC=$?")
    out["r1_build_log"] = f"rc={rc}\n{o}{e}"
    outs = glob.glob("/work/rustv1/target/x86_64-unknown-linux-gnu/release/"
                     "build/rustv1-*/output")
    rec = []
    for p in outs:
        rec.append(f"=== {p} ===\n" + open(p).read())
    out["r1_buildscript_output_files"] = "\n".join(rec) if rec else "__NONE__"
    if not rec:
        rc, o, e = sh("grep -rl RV1_BUILD_ /work/rustv1/target/"
                      "x86_64-unknown-linux-gnu/release/build/ 2>/dev/null | "
                      "head -5")
        out["r1_buildscript_output_fallback_search"] = f"rc={rc}\n{o}{e}"
    out["r1_rv1_cuda_compute_cap_state"] = (
        "RV1_CUDA_COMPUTE_CAP=" +
        os.environ.get("RV1_CUDA_COMPUTE_CAP", "__UNSET__ (build.rs emitted "
                       "the default sm_100)"))
    rc, o, e = sh("which rustc && sha256sum $(which rustc) && rustc -vV")
    out["r1_rustc_identity"] = f"rc={rc}\n{o}{e}"
    rc, o, e = sh("cd /work/rustv1 && find "
                  "target/x86_64-unknown-linux-gnu/release -maxdepth 1 "
                  "-type f -executable -exec sha256sum {} +")
    out["r1_binary_shas"] = f"rc={rc}\n{o}{e}"
    out["r1_build_env"] = json.dumps({
        "CUDA_COMPUTE_CAP": os.environ.get("CUDA_COMPUTE_CAP", "__UNSET__"),
        "RV1_CUDA_COMPUTE_CAP": os.environ.get("RV1_CUDA_COMPUTE_CAP",
                                               "__UNSET__"),
        "RUSTFLAGS": os.environ.get("RUSTFLAGS", "__UNSET__"),
        "CARGO_ENCODED_RUSTFLAGS": os.environ.get(
            "CARGO_ENCODED_RUSTFLAGS", "__UNSET__"),
        "HOME": home, "CARGO_HOME": cargo_home_env}, indent=1)

    print("DONE", flush=True)
    return out


@app.local_entrypoint()
def main() -> None:
    import os
    res = run.remote()
    os.makedirs("tmp/v60_raw", exist_ok=True)
    for k, v in res.items():
        open(f"tmp/v60_raw/{k}.txt", "w").write(v)
    print(f"saved {len(res)} artifacts")
