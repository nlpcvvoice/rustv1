import modal

# Free CPU: v61 rustc effective-resolution record at 778a6c6. gRPC return.

app = modal.App("rustv1-v61rustcident")

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


@app.function(cpu=2, memory=8192, timeout=600, scaledown_window=10,
              image=image)
def run() -> dict:
    import subprocess

    out: dict = {}

    def sh(cmd: str) -> tuple[int, str, str]:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

    rc, o, e = sh("git clone -q https://github.com/nlpcvvoice/rustv1.git "
                  f"/work/rustv1 && cd /work/rustv1 && git checkout -q {REV} "
                  f"&& test $(git rev-parse HEAD) = {REV}")
    print("CLONE_RC", rc, flush=True)

    rc, o, e = sh("cd /work/rustv1 && pwd && "
                  "rustup show active-toolchain && rustup which rustc")
    out["resolution"] = f"rc={rc}\n{o}{e}"
    lines = o.splitlines()
    rustc_path = lines[-1].strip() if len(lines) >= 3 else ""
    if rustc_path.startswith("/"):
        rc, o2, e2 = sh(f"sha256sum {rustc_path} && {rustc_path} -vV && "
                        f"sha256sum /root/.cargo/bin/rustc && "
                        f"test -L {rustc_path} && echo RESOLVED_IS_SYMLINK "
                        f"|| echo RESOLVED_NOT_SYMLINK; "
                        f"readlink -f {rustc_path}")
        out["resolved_identity"] = f"rc={rc}\n{o2}{e2}"
        out["resolved_path"] = rustc_path
    print("DONE", flush=True)
    return out


@app.local_entrypoint()
def main() -> None:
    import os
    res = run.remote()
    os.makedirs("tmp/v61_raw", exist_ok=True)
    for k, v in res.items():
        open(f"tmp/v61_raw/{k}.txt", "w").write(v)
    print(f"saved {len(res)} artifacts")
