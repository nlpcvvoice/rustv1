import modal

# Expanded M1 corpus v2 (ASTRO6 #6 test-matrix gaps): 22 prompts =
# previous 17 + second repeated-token ID (short/long) + genuinely varied
# 1023 text + special-token-literal + whitespace/punctuation cases.
# Captures vLLM 0.25.0 BF16 singleton embeddings, vLLM batch sensitivity,
# and reference tokenization cases for Rust tokenizer equality.

app = modal.App("rustv1-expanded-capture-v2")

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
    import pathlib

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

    # New per ASTRO6 #6: second repeated-token ID, short and long.
    prompts.append([918] * 7)
    prompts.append([918] * 300)

    # Genuinely varied long input (distinct sentences, truncated to 1023).
    sentences = [
        "The capital of France is Paris. ",
        "The Andromeda galaxy spans roughly one hundred thousand light years. ",
        "Wheat yields rose three percent in the northern counties. ",
        "Quarterly churn improved to two point one percent after the pilot. ",
        "Rust ownership rules prevent data races at compile time. ",
        "The Ganges delta floods seasonally each monsoon. ",
        "Bond yields climbed after the central bank held rates steady. ",
        "Attention weighs every token against every other token. ",
        "Olive oil production dropped in Andalusia this year. ",
        "The compiler rejected the borrow that outlived its owner. ",
        "Hurricane tracks shifted northward over the decade. ",
        "Sample variance divides the sum of squares by n minus one. ",
        "The port of Rotterdam handled record container volume. ",
        "Gradient descent steps shrink near a well conditioned minimum. ",
        "Lunar tides affect estuary salinity twice a day. ",
        "The audit found three misclassified refunds per thousand. ",
        "Monte Carlo error decreases like one over root n. ",
        "The orchard planted dwarf rootstock along the ridge. ",
        "Precision and recall trade off under threshold changes. ",
        "Comet tails always point away from the sun. ",
        "Soil moisture sensors triggered irrigation twice in July. ",
    ]
    varied = "".join(sentences * 5)
    s_var = tok.encode(varied, add_special_tokens=False).ids[:1023]
    s_var = [12 if t == 11 else t for t in s_var]
    assert len(s_var) == 1023 and 11 not in s_var
    prompts.append(s_var)

    # Special-token-literal and whitespace/punctuation cases.
    prompts.append(tok.encode("passage: <s> hello", add_special_tokens=False).ids)
    prompts.append(tok.encode("...  !! ?? ,, --  ", add_special_tokens=False).ids)

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

    tok_cases = [
        {"text": t, "ids": tok.encode(t, add_special_tokens=False).ids}
        for t in [
            "passage: hello",
            "query: what is Rust?",
            "passage: 你好，世界",
            "passage:   spaces   and... punctuation!!!",
            "passage: <s> hello",
            "a",
            "passage: a.b,c!!",
        ]
    ]

    out = pathlib.Path("/work/expanded"); out.mkdir(parents=True)
    (out / "prompts.json").write_text(json.dumps(prompts))
    single.tofile(out / "ref_single.f32")
    b_tri.tofile(out / "ref_batch_tri.f32")
    b_rev.tofile(out / "ref_batch_rev.f32")
    b_dup.tofile(out / "ref_batch_dup.f32")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "tokenization_cases.json").write_text(json.dumps(tok_cases))

    subprocess.run(["tar", "czf", "/work/exp.tgz", "-C", "/work", "expanded"], check=True)
    blob = base64.b64encode(open("/work/exp.tgz", "rb").read()).decode()
    print(f"===B64BEGIN len={len(blob)}===")
    for i in range(0, len(blob), 1500):
        print(blob[i:i + 1500], flush=True)
    print("===B64END===")
