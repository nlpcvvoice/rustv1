// src/bin/golden_export.rs
//
// Export candidate singleton goldens from the designated M2 in-process
// token-ID route (m2_route::open + forward_serving), on the same device
// and binary build as acceptance. Emits a packed little-endian f32 matrix
// plus a per-row index with ids/output digests. ids_sha256 follows the
// frozen convention: SHA256 over concatenated u32 little-endian bytes.
use anyhow::{ensure, Result};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{path::PathBuf, process};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: golden_export MODEL_DIR INPUTS_JSON OUT_PREFIX AUDIT_IDENTITY");
        process::exit(2);
    }
    if let Err(error) = run(&args[1], &args[2], &args[3], &args[4]) {
        eprintln!("golden_export failed: {error:#}");
        process::exit(1);
    }
}

fn run(model_dir: &str, inputs_path: &str, out_prefix: &str, audit_identity: &str) -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Input {
        id: String,
        source: String,
        ids: Vec<u32>,
    }
    let inputs: Vec<Input> = serde_json::from_slice(&std::fs::read(inputs_path)?)?;
    ensure!(!inputs.is_empty(), "empty inputs");

    let mut executor = rustv1::m2_route::open(
        &PathBuf::from(model_dir),
        &PathBuf::from("golden_export_manifest.jsonl"),
        0,
        audit_identity,
    )?;

    let tokens: Vec<Vec<u32>> = inputs.iter().map(|i| i.ids.clone()).collect();
    let embeddings = executor.forward_serving(&tokens)?;
    ensure!(embeddings.len() == inputs.len(), "row count mismatch");
    let width = embeddings[0].len();
    for row in &embeddings {
        ensure!(row.len() == width, "ragged output width");
    }

    let f32_path = format!("{out_prefix}.golden.f32");
    let mut buffer = Vec::with_capacity(embeddings.len() * width * 4);
    for row in &embeddings {
        for v in row {
            buffer.extend_from_slice(&v.to_le_bytes());
        }
    }
    let mut matrix_sha = Sha256::new();
    matrix_sha.update(&buffer);
    let matrix_sha = format!("{:x}", matrix_sha.finalize());
    std::fs::write(&f32_path, &buffer)?;

    let rows: Vec<serde_json::Value> = inputs
        .iter()
        .zip(embeddings.iter())
        .map(|(input, emb)| {
            let mut ids_sha = Sha256::new();
            for id in &input.ids {
                ids_sha.update(id.to_le_bytes());
            }
            let mut out_sha = Sha256::new();
            for v in emb {
                out_sha.update(v.to_le_bytes());
            }
            json!({
                "id": input.id,
                "source": input.source,
                "ids_len": input.ids.len(),
                "ids_sha256": format!("{:x}", ids_sha.finalize()),
                "output_sha256": format!("{:x}", out_sha.finalize()),
            })
        })
        .collect();
    let index_path = format!("{out_prefix}.golden_rows.json");
    std::fs::write(
        &index_path,
        serde_json::to_vec_pretty(&json!({
            "schema": "rustv1.candidate_goldens.v1",
            "rows": rows,
            "width": width,
            "matrix_sha256": matrix_sha,
        }))?,
    )?;

    println!(
        "EXPORT_RC 0 rows {} width {} matrix_sha256 {}",
        embeddings.len(),
        width,
        matrix_sha
    );
    Ok(())
}
