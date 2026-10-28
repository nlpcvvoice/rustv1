//! M4 controlled numerical pre-pass.
//!
//! This delivery implements only:
//!   accept_m4 prepass MODEL_DIR PREPASS_JSON PREPASS_SHA256 OUT_DIR
//!
//! It does not implement or claim complete HTTP/scheduler/lifecycle acceptance.
//! Successful execution produces pre-finalize numerical evidence and exits 0.
//! Overall M4 acceptance is always false in this command's terminal record.

#[cfg(feature = "cuda")]
#[path = "../accept_m4_contract.rs"]
mod contract;

#[cfg(feature = "cuda")]
#[path = "../accept_m4_prepass.rs"]
mod prepass;

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("accept_m4 prepass requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 || args[1] != "prepass" {
        eprintln!("usage: accept_m4 prepass MODEL_DIR PREPASS_JSON PREPASS_SHA256 OUT_DIR");
        std::process::exit(2);
    }

    let result = contract::load(
        std::path::Path::new(&args[2]),
        std::path::Path::new(&args[3]),
        &args[4],
    )
    .and_then(|bundle| prepass::run(bundle, std::path::Path::new(&args[5])));

    match result {
        Ok(evidence_sha256) => {
            println!(
                "{}",
                serde_json::json!({
                    "kind": "done",
                    "scope": "controlled_numerical_prepass",
                    "accepted": false,
                    "pass": true,
                    "batch_in_process_evidence_sha256": evidence_sha256
                })
            );
        }
        Err(error) => {
            println!(
                "{}",
                serde_json::json!({
                    "case_id": "prepass-failure",
                    "status": "executed",
                    "pass": false,
                    "error": format!("{error:#}")
                })
            );
            println!(
                "{}",
                serde_json::json!({
                    "kind": "done",
                    "scope": "controlled_numerical_prepass",
                    "accepted": false,
                    "pass": false
                })
            );
            std::process::exit(1);
        }
    }
}
