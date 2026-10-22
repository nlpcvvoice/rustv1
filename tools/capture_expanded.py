import modal

# Expanded M1 corpus: build 17 prompts (length boundaries, repeated tokens,
# EOS cases, natural text), capture vLLM 0.25.0 BF16 singleton embeddings,
# and measure vLLM's own batch sensitivity. Outputs echoed as base64 tar.

app = modal.App("rustv1-expanded-capture")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.25.0", "tokenizers", "numpy")
    .apt_install("git", "tar")
)


@app.function(gpu="L4:1", cpu=8, memory=32768, timeout=2400,
              scaledown_window=10, image=image)
def capture() -> None:
    import base64
    import json
    import subprocess

    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    from vllm import LLM
    import importlib.metadata
    import numpy as np

    assert importlib.metadata.version("vllm") == "0.25.0"
    model = snapshot_download("nvidia/Nemotron-3-Embed-1B-BF16")

    tok = Tokenizer.from_file(model + "/tokenizer.json")
    tok.no_truncation(); tok.no_padding()
    seedA = tok.encode("passage: hello", add_special_tokens=False).ids
    seedB = tok.encode("passage: the quick brown fox jumps over",
                       add_special_tokens=False).ids

    def synth(seed, length):
        return (seed * ((length + len(seed) - 1) // len(seed)))[:length]

    prompts = []
    for length in [1, 2, 255, 256, 257, 1023, 1024]:
        prompts.append(synth(seedA if length % 2 else seedB, length))
    prompts.append([40] * 10)
    prompts.append([40] * 512)
    prompts.append([2])
    prompts.append(seedA + [2])
    prompts.append(seedB[:10] + [2] + seedB[10:20])
    for text in [
        "passage: hello world",
        "query: what is Rust?",
        "passage: 你好，世界",
        "passage:   spaces   and... punctuation!!!",
        "a",
    ]:
        prompts.append(tok.encode(text, add_special_tokens=False).ids)

    for i, p in enumerate(prompts):
        assert p and 11 not in p and len(p) <= 1024 and max(p) < 131072, i
    print("prompts:", len(prompts), "lengths:", [len(p) for p in prompts], flush=True)

    llm = LLM(model=model, runner="pooling", dtype="bfloat16",
              tensor_parallel_size=1, max_model_len=1024, max_num_seqs=8,
              max_num_batched_tokens=4096, enforce_eager=True,
              enable_prefix_caching=False)

    def embed(batch):
        res = llm.embed([{"prompt_token_ids": p} for p in batch], use_tqdm=False)
        return np.array([r.outputs.embedding for r in res], dtype=np.float32)

    singles = [embed([p]) for p in prompts]
    single = np.concatenate(singles, axis=0)
    assert single.shape == (len(prompts), 2048)

    tri = [0, 2, 6]
    b_tri = embed([prompts[i] for i in tri])
    b_rev = embed([prompts[i] for i in reversed(tri)])
    b_dup = embed([prompts[0], prompts[0]])

    def sens(a, b):
        c = (a @ b.T) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        d = np.abs(a - b)
        return float(np.diag(c).min()), float(d.max())

    meta = {
        "lengths": [len(p) for p in prompts],
        "vllm_batch_sensitivity": {
            "tri_vs_single": sens(b_tri, np.concatenate([singles[i] for i in tri])),
            "rev_vs_single": sens(b_rev, np.concatenate([singles[i] for i in reversed(tri)])),
            "dup_internal": sens(b_dup[:1], b_dup[1:]),
        },
    }
    print(json.dumps(meta, indent=1), flush=True)

    import pathlib
    out = pathlib.Path("/work/expanded"); out.mkdir(parents=True)
    (out / "prompts.json").write_text(json.dumps(prompts))
    single.tofile(out / "ref_single.f32")
    b_tri.tofile(out / "ref_batch_tri.f32")
    b_rev.tofile(out / "ref_batch_rev.f32")
    b_dup.tofile(out / "ref_batch_dup.f32")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    subprocess.run(["tar", "czf", "/work/exp.tgz", "-C", "/work", "expanded"], check=True)
    blob = base64.b64encode(open("/work/exp.tgz", "rb").read()).decode()
    print(f"===B64BEGIN len={len(blob)}===")
    for i in range(0, len(blob), 1500):
        print(blob[i:i + 1500], flush=True)
    print("===B64END===")
