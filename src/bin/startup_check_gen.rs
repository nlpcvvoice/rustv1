// src/bin/startup_check_gen.rs
//
// Generate a serve_m3 startup-check artifact from the same executor route
// the server worker uses (m2_route::open + forward_serving), on the same
// device and binary build. Output is bit-exact by construction.
use anyhow::Result;
use serde_json::json;
use std::{path::PathBuf, process};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: startup_check_gen MODEL_DIR TOKENS_JSON OUT_JSON AUDIT_IDENTITY");
        process::exit(2);
    }
    if let Err(error) = run(&args[1], &args[2], &args[3], &args[4]) {
        eprintln!("startup_check_gen failed: {error:#}");
        process::exit(1);
    }
}

fn run(model_dir: &str, tokens_path: &str, out_path: &str, audit_identity: &str) -> Result<()> {
    let tokens: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(tokens_path)?)?;
    let mut executor = rustv1::m2_route::open(
        &PathBuf::from(model_dir),
        &PathBuf::from("startup_check_gen_manifest.jsonl"),
        0,
        audit_identity,
    )?;
    let embeddings = executor.forward_serving(&tokens)?;
    std::fs::write(
        out_path,
        serde_json::to_vec(&json!({"tokens": tokens, "embeddings": embeddings}))?,
    )?;
    println!("GEN_RC 0 {} rows {}", out_path, embeddings.len());
    Ok(())
}
