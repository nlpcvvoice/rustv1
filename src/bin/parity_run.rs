// M1 acceptance run (harness corrected per ASTRO6 #6):
//   - singleton baselines computed explicitly, one forward per prompt
//   - batch suites respect the serving budgets: max_seqs, actual tokens,
//     and padded tokens (B x max sequence length)
//   - every compared output and the reference file are validated
//     fail-closed before any metric is reduced
//   - invariance: grouped, reversed, deterministically shuffled,
//     duplicated, and repeated identical-shape executions
//   - empty batch and invalid inputs must be rejected
//   - stage-level bidirectionality
//   - sha256 manifest of every artifact, tied to this revision
//
// Frozen pass criteria (fixed before evaluation):
//   reference comparison: dim 2048, finite, |norm-1| <= 1e-5 (FP64),
//                         cosine > 0.999, max_abs <= 0.01
//   batch invariance:     cosine > 0.9999, max_abs <= 0.005
//     (set from measured vLLM 0.25.0 self-sensitivity 0.99992 / 0.0026)
//   repeated identical shape: cosine > 0.999999, bit identity reported

use anyhow::{bail, ensure, Context, Result};
use candle_core::Device;
use rustv1::model::{EngineConfig, Model};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::PathBuf;

fn sha256_file(path: &std::path::Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() % 4 == 0,
        "reference file size {} is not a multiple of 4", bytes.len());
    Ok(bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn norm_f64(v: &[f32]) -> f64 {
    v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt()
}

fn validate_row(v: &[f32], label: &str) -> Result<()> {
    ensure!(v.len() == 2048, "{label}: dim {} != 2048", v.len());
    ensure!(v.iter().all(|x| x.is_finite()), "{label}: non-finite component");
    let n = norm_f64(v);
    ensure!(n.is_finite() && n > 0.0, "{label}: invalid norm {n}");
    ensure!((n - 1.0).abs() <= 1e-5, "{label}: |norm-1| = {:.2e} > 1e-5",
        (n - 1.0).abs());
    Ok(())
}

fn compare(emb: &[f32], reference: &[f32]) -> Result<(f64, f64)> {
    ensure!(emb.len() == reference.len(),
        "compare: dimension mismatch {} vs {}", emb.len(), reference.len());
    let dot: f64 = emb.iter().zip(reference)
        .map(|(a, b)| *a as f64 * *b as f64).sum();
    let na = norm_f64(emb);
    let nb = norm_f64(reference);
    let cosine = dot / (na * nb);
    let max_abs = emb.iter().zip(reference)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max);
    ensure!(cosine.is_finite() && max_abs.is_finite(),
        "non-finite comparison metric");
    Ok((cosine, max_abs))
}

fn bit_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 7 || args.len() == 8,
        "usage: parity_run MODEL_DIR PROMPTS_JSON REFERENCE_F32 \
         TOKENIZER_JSON TOKENIZATION_JSON EXCEPTION_POLICY_JSON \
         [DEVICE,DTYPE  e.g. cpu,f32 | cuda,bf16]");
    println!("acceptance command: {}", args[1..].join(" "));

    // Explicit device and dtype selection; no silent fallback.
    let (device_name, dtype_name) = if args.len() == 8 {
        let spec = args[7].clone();
        let mut it = spec.split(',');
        (it.next().unwrap_or("").to_string(),
         it.next().unwrap_or("").to_string())
    } else {
        ("cpu".to_string(), "f32".to_string())
    };
    let device = match device_name.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => {
            ensure!(candle_core::utils::cuda_is_available(),
                "device 'cuda' requested but CUDA backend not compiled in");
            Device::new_cuda(0)?
        }
        other => bail!("unknown device '{other}': use cpu or cuda"),
    };
    let dtype = match dtype_name.as_str() {
        "f32" => candle_core::DType::F32,
        "bf16" => candle_core::DType::BF16,
        other => bail!("unknown dtype '{other}': use f32 or bf16"),
    };
    println!("execution target: device={device:?} dtype={dtype:?}");

    let dir = PathBuf::from(&args[1]);
    let prompts: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let reference = read_f32(&args[3])?;
    ensure!(reference.len() == prompts.len() * 2048,
        "reference {} != {} x 2048", reference.len(), prompts.len());

    // Artifact manifest ties this acceptance to exact inputs.
    for (name, p) in [
        ("model.safetensors", dir.join("model.safetensors")),
        ("config.json", dir.join("config.json")),
        ("tokenizer.json", PathBuf::from(&args[4])),
        ("engine.json", dir.join("engine.json")),
        ("prompts.json", PathBuf::from(&args[2])),
        ("reference.f32", PathBuf::from(&args[3])),
        ("tokenization_cases.json", PathBuf::from(&args[5])),
        ("exception_policy.json", PathBuf::from(&args[6])),
        ("Cargo.lock", PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock")),
    ] {
        println!("sha256 {name} = {}", sha256_file(&p)?);
    }

    // Reference rows validated fail-closed before any comparison.
    for i in 0..prompts.len() {
        validate_row(&reference[i * 2048..(i + 1) * 2048], &format!("ref {i}"))?;
    }

    let engine: EngineConfig =
        serde_json::from_slice(&std::fs::read(dir.join("engine.json"))?)?;
    let model = Model::load(&dir, device, engine.clone(), dtype)?;
    println!("loaded; {} prompts, lengths {:?}", prompts.len(),
        prompts.iter().map(Vec::len).collect::<Vec<_>>());

    let mut pass = true;

    // Singleton baselines: one forward per prompt, explicitly.
    let started = std::time::Instant::now();
    let mut single: Vec<Vec<f32>> = Vec::with_capacity(prompts.len());
    for (i, p) in prompts.iter().enumerate() {
        let out = model.forward(&[p.clone()])?;
        ensure!(out.len() == 1, "singleton {i}: got {} rows", out.len());
        validate_row(&out[0], &format!("singleton {i}"))?;
        single.push(out.into_iter().next().unwrap());
    }
    println!("singletons done in {:.1}s", started.elapsed().as_secs_f64());

    // Enforced named, versioned exception policy (ASTRO6 #7): each
    // exception must match its index and the digest of its token
    // sequence, defined as SHA-256 over consecutive little-endian u32
    // ids. Duplicate, out-of-range, or digest-mismatched indices are
    // rejected: the exemption cannot be detached from the approved
    // sample identity.
    let policy: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&args[6])?)?;
    ensure!(policy["version"].as_u64() == Some(2),
        "exception policy version must be 2");
    let mut diagnostics: Vec<usize> = Vec::new();
    for s in policy["samples"].as_array().context("policy: samples array")? {
        let idx = s["index"].as_u64().context("policy: index")? as usize;
        ensure!(idx < prompts.len(), "policy: index {idx} out of range");
        ensure!(!diagnostics.contains(&idx), "policy: duplicate index {idx}");
        let want = s["ids_sha256"].as_str().context("policy: ids_sha256")?;
        let mut h = Sha256::new();
        for t in &prompts[idx] { h.update(t.to_le_bytes()); }
        let got = format!("{:x}", h.finalize());
        ensure!(got == want,
            "policy: sample {idx} token digest mismatch (got {got}); \
             the corpus no longer matches the approved exception identity");
        println!("exception v{} enforced: sample {idx} bound to digest {got}",
            policy["version"].as_u64().unwrap());
        diagnostics.push(idx);
    }

    for (i, emb) in single.iter().enumerate() {
        let (cosine, max_abs) =
            compare(emb, &reference[i * 2048..(i + 1) * 2048])?;
        let ok = cosine > 0.999 && max_abs <= 0.01;
        if diagnostics.contains(&i) {
            println!("sample {i}: len={} cosine={cosine:.6} max_abs={max_abs:.5} \
                     DIAGNOSTIC (not gated; see exception_policy.json)",
                prompts[i].len());
        } else {
            pass &= ok;
            println!("sample {i}: len={} cosine={cosine:.6} max_abs={max_abs:.5} {}",
                prompts[i].len(), if ok { "PASS" } else { "FAIL" });
        }
    }

    // Batch suites under serving budgets: actual tokens and padded
    // B x max_len must both fit max_tokens; at most max_seqs per batch.
    fn budget_groups(prompts: &[Vec<u32>], order: &[usize],
                     max_seqs: usize, max_tokens: usize) -> Vec<Vec<usize>> {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let mut tokens = 0usize;
        let mut max_len = 0usize;
        for &i in order {
            let l = prompts[i].len();
            let next_tokens = tokens + l;
            let next_padded = (cur.len() + 1) * max_len.max(l);
            if !cur.is_empty()
                && (cur.len() + 1 > max_seqs
                    || next_tokens > max_tokens
                    || next_padded > max_tokens)
            {
                groups.push(std::mem::take(&mut cur));
                tokens = 0;
                max_len = 0;
            }
            cur.push(i);
            tokens += l;
            max_len = max_len.max(l);
        }
        if !cur.is_empty() { groups.push(cur); }
        groups
    }

    fn check_batch(model: &Model, prompts: &[Vec<u32>], group: &[usize],
                   single: &[Vec<f32>], label: &str,
                   worst_cos: &mut f64, worst_abs: &mut f64,
                   n: &mut usize) -> Result<()> {
        let seqs: Vec<_> = group.iter().map(|&i| prompts[i].clone()).collect();
        let outs = model.forward(&seqs)?;
        ensure!(outs.len() == seqs.len(),
            "{label}: got {} rows for {} sequences", outs.len(), seqs.len());
        for (k, &i) in group.iter().enumerate() {
            validate_row(&outs[k], &format!("{label} seq {i}"))?;
            let (cosine, max_abs) = compare(&outs[k], &single[i])?;
            *worst_cos = worst_cos.min(cosine);
            *worst_abs = worst_abs.max(max_abs);
            *n += 1;
        }
        Ok(())
    }

    let idx: Vec<usize> = (0..prompts.len()).collect();
    let mut rev = idx.clone();
    rev.reverse();
    let mut shuffled = idx.clone();
    {   // deterministic Fisher-Yates (LCG, fixed seed)
        let mut s = 0x5EEDu64;
        for k in (1..shuffled.len()).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (s >> 33) as usize % (k + 1);
            shuffled.swap(k, j);
        }
    }

    let mut worst_cos = 1.0f64;
    let mut worst_abs = 0.0f64;
    let mut n = 0usize;
    for (name, order) in [("grouped", &idx), ("reversed", &rev), ("shuffled", &shuffled)] {
        let groups = budget_groups(&prompts, order, engine.max_seqs, engine.max_tokens);
        println!("batch suite {name}: {} groups, sizes {:?}", groups.len(),
            groups.iter().map(Vec::len).collect::<Vec<_>>());
        for g in &groups {
            check_batch(&model, &prompts, g, &single, name,
                &mut worst_cos, &mut worst_abs, &mut n)?;
        }
    }

    // Duplicates in one batch: against each other and the singleton.
    let dup = vec![prompts[0].clone(), prompts[0].clone()];
    let outs = model.forward(&dup)?;
    ensure!(outs.len() == 2, "duplicate batch: {} rows", outs.len());
    for r in &outs { validate_row(r, "duplicate")?; }
    let (dup_cos, dup_abs) = compare(&outs[0], &outs[1])?;
    let (ds_cos, ds_abs) = compare(&outs[1], &single[0])?;
    worst_cos = worst_cos.min(dup_cos).min(ds_cos);
    worst_abs = worst_abs.max(dup_abs).max(ds_abs);
    n += 2;

    // Repeated identical-shape execution: regression baseline, tight gate.
    let g0 = budget_groups(&prompts, &idx, engine.max_seqs, engine.max_tokens)
        .into_iter().next().unwrap();
    let seqs: Vec<_> = g0.iter().map(|&i| prompts[i].clone()).collect();
    let a = model.forward(&seqs)?;
    let b = model.forward(&seqs)?;
    ensure!(a.len() == g0.len(), "repeat A: {} rows for {} sequences",
        a.len(), g0.len());
    ensure!(b.len() == g0.len(), "repeat B: {} rows for {} sequences",
        b.len(), g0.len());
    let mut rep_worst_cos = 1.0f64;
    let mut bits_same = true;
    for k in 0..g0.len() {
        validate_row(&a[k], &format!("repeat A {k}"))?;
        validate_row(&b[k], &format!("repeat B {k}"))?;
        let (c, _) = compare(&a[k], &b[k])?;
        rep_worst_cos = rep_worst_cos.min(c);
        bits_same &= bit_identical(&a[k], &b[k]);
    }
    println!("repeated-shape: worst_cosine={rep_worst_cos:.10} bit_identical={bits_same}");
    pass &= rep_worst_cos > 0.999999;

    println!("batch invariance: n={n} worst_cosine={worst_cos:.8} worst_max_abs={worst_abs:.6}");
    let inv_ok = worst_cos > 0.9999 && worst_abs <= 0.005;
    pass &= inv_ok;
    println!("batch-invariance: {}", if inv_ok { "PASS" } else { "FAIL" });

    // Invalid inputs must be rejected.
    let bad_batch = model.forward(&[]).is_err();
    let bad_empty = model.forward(&[vec![]]).is_err();
    let bad_pad = model.forward(&[vec![11, 11]]).is_err();
    let bad_long = model.forward(&[vec![5u32; engine.max_len + 1]]).is_err();
    let bad_oov = model.forward(&[vec![131_072u32]]).is_err();
    let neg_ok = bad_batch && bad_empty && bad_pad && bad_long && bad_oov;
    println!("invalid-inputs: empty_batch={bad_batch} empty_seq={bad_empty} \
             pad={bad_pad} long={bad_long} oov={bad_oov} -> {}",
        if neg_ok { "PASS" } else { "FAIL" });
    pass &= neg_ok;

    // Tokenizer equality vs the reference tokenizer (HF `tokenizers`).
    let tokenizer = tokenizers::Tokenizer::from_file(&args[4])
        .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(&args[5])?)?;
    let mut tok_ok = true;
    for case in &cases {
        let text = case["text"].as_str().unwrap();
        let expected: Vec<u32> = case["ids"].as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let got: Vec<u32> = tokenizer.encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids().to_vec();
        if got != expected {
            tok_ok = false;
            println!("tokenizer MISMATCH {text:?}: rust={got:?} ref={expected:?}");
        }
    }
    println!("tokenizer-equality: {} cases -> {}", cases.len(),
        if tok_ok { "PASS" } else { "FAIL" });
    pass &= tok_ok;

    // Stage-level bidirectionality: changing a later token must change an
    // earlier token's post-norm state.
    let base = vec![12107u32, 1541, 1058, 52528, 52528, 1541];
    let mut altered = base.clone();
    *altered.last_mut().unwrap() = 9990;
    let states = model.forward_states(&[base.clone(), altered])?;
    ensure!(states.len() == 2, "bidirectionality: {} sequences", states.len());
    ensure!(states[0].len() == base.len() && states[1].len() == base.len(),
        "bidirectionality: state lengths {:?} != {}",
        states.iter().map(Vec::len).collect::<Vec<_>>(), base.len());
    for (si, seq) in states.iter().enumerate() {
        for (ti, row) in seq.iter().enumerate() {
            ensure!(row.len() == 2048,
                "bidirectionality: state {si}/{ti} dim {} != 2048", row.len());
            ensure!(row.iter().all(|v| v.is_finite()),
                "bidirectionality: non-finite state {si}/{ti}");
        }
    }
    let d0: f64 = states[0][0].iter().zip(&states[1][0])
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);
    let dlast: f64 = states[0][base.len() - 1].iter()
        .zip(&states[1][base.len() - 1])
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);
    ensure!(d0.is_finite() && dlast.is_finite(),
        "bidirectionality: non-finite difference metric");
    let bidir_ok = d0 > 1e-4 && dlast > 1e-4;
    println!("bidirectionality: token0_maxdiff={d0:.5} lasttoken_maxdiff={dlast:.5} \
             -> {}", if bidir_ok { "PASS" } else { "FAIL" });
    pass &= bidir_ok;

    if pass {
        println!("M1-ACCEPTANCE: PASS");
    } else {
        println!("M1-ACCEPTANCE: FAIL");
        std::process::exit(1);
    }
    Ok(())
}
