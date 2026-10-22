#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
import vllm
from tokenizers import Tokenizer
from vllm import LLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="reference")
    args = ap.parse_args()

    version = importlib.metadata.version("vllm")
    if version != "0.25.0":
        raise RuntimeError(f"Expected vLLM 0.25.0, found {version}")

    model = Path(args.model).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok_path = model / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    tok.no_truncation()
    tok.no_padding()

    text = "passage: hello"
    plain = tok.encode(text, add_special_tokens=False)
    special = tok.encode(text, add_special_tokens=True)
    assert plain.ids == special.ids, "Unexpected special-token insertion"
    assert plain.ids, "Unexpected empty encoding"
    assert 11 not in plain.ids

    tokenization = {
        "text": text,
        "ids": plain.ids,
        "tokens": plain.tokens,
        "add_special_tokens_true_ids": special.ids,
        "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest(),
    }
    print(json.dumps(tokenization, indent=2, ensure_ascii=False))
    (out / "tokenization.json").write_text(
        json.dumps(tokenization, indent=2, ensure_ascii=False)
    )

    lengths = [255, 256, 1024]

    # Synthetic token prompts: deterministic, exact lengths, no padding/EOS
    # appended by this script. They need not be encodings of natural sentences.
    seed = plain.ids
    prompts = [
        (seed * ((length + len(seed) - 1) // len(seed)))[:length]
        for length in lengths
    ]

    ids = torch.full((3, 1024), 11, dtype=torch.long)
    mask = torch.zeros((3, 1024), dtype=torch.bool)
    for row, prompt in enumerate(prompts):
        ids[row, :len(prompt)] = torch.tensor(prompt)
        mask[row, :len(prompt)] = True

    torch.save(
        {
            "input_ids": ids,
            "attention_mask": mask,
            "lengths": torch.tensor(lengths),
        },
        out / "inputs.pt",
    )
    (out / "prompts.json").write_text(json.dumps(prompts))

    llm = LLM(
        model=str(model),
        runner="pooling",
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=3,
        max_num_batched_tokens=sum(lengths),
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    results = llm.embed(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        use_tqdm=False,
    )
    embeddings = torch.tensor(
        [result.outputs.embedding for result in results],
        dtype=torch.float32,
    )
    assert embeddings.shape == (3, 2048)

    torch.save(embeddings, out / "vllm_pooling_output.pt")
    print("Output L2 norms:", embeddings.norm(dim=-1).tolist())

    # Capture installed implementation evidence rather than assuming module
    # paths or inventing worker hook APIs.
    source_root = Path(vllm.__file__).resolve().parent
    source_out = out / "vllm_sources"
    source_out.mkdir(exist_ok=True)

    needles = (
        "llama_4_scaling",
        "apply_yarn_scaling",
        "weight_global_scale",
        "input_global_scale",
    )
    manifest = []
    for path in source_root.rglob("*.py"):
        source = path.read_text(errors="replace")
        if "ministral3" in path.name.lower() or any(
            needle in source for needle in needles
        ):
            relative = path.relative_to(source_root)
            target = source_out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
            manifest.append(str(relative))

    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
