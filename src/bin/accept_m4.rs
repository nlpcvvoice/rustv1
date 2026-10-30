//! M4 phase dispatcher.
//!
//! prepass MODEL_DIR PREPASS_JSON PREPASS_SHA256 OUT_DIR
//! acceptance-http POLICY_JSON BASE_URL CONTROL_SOCKET OUT_DIR HANDOFF_JSON
//!
//! Neither command reports overall M4 acceptance.

#[path = "../accept_m4_contract.rs"]
mod contract;

#[cfg(feature = "cuda")]
#[path = "../accept_m4_prepass.rs"]
mod prepass;

#[path = "../accept_m4_http.rs"]
mod acceptance_http;

use anyhow::{bail, Result};
use std::path::Path;

fn dispatch(args: &[String]) -> Result<(&'static str, String)> {
    match args.get(1).map(String::as_str) {
        Some("prepass") if args.len() == 6 => {
            #[cfg(feature = "cuda")]
            {
                let bundle = contract::load(Path::new(&args[2]), Path::new(&args[3]), &args[4])?;
                let digest = prepass::run(bundle, Path::new(&args[5]))?;
                Ok(("controlled_numerical_prepass", digest))
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("prepass requires --features cuda")
            }
        }
        Some("acceptance-http") if args.len() == 7 => {
            let digest = acceptance_http::run(
                Path::new(&args[2]),
                &args[3],
                &args[4],
                Path::new(&args[5]),
                Path::new(&args[6]),
            )?;
            Ok(("http_reconciliation", digest))
        }
        _ => bail!(
            "usage:\n\
             accept_m4 prepass MODEL_DIR PREPASS_JSON PREPASS_SHA256 OUT_DIR\n\
             accept_m4 acceptance-http POLICY_JSON BASE_URL CONTROL_SOCKET OUT_DIR HANDOFF_JSON"
        ),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let phase = if args.get(1).map(String::as_str) == Some("acceptance-http") {
        "http_reconciliation"
    } else {
        "controlled_numerical_prepass"
    };

    match dispatch(&args) {
        Ok((scope, digest)) => {
            println!(
                "{}",
                serde_json::json!({
                    "kind": "done",
                    "acceptance_phase": scope,
                    "accepted": false,
                    "m4_accepted": false,
                    "pass": true,
                    "completion_manifest_sha256": digest
                })
            );
        }
        Err(error) => {
            println!(
                "{}",
                serde_json::json!({
                    "kind": "done",
                    "acceptance_phase": phase,
                    "accepted": false,
                    "m4_accepted": false,
                    "pass": false,
                    "error": format!("{error:#}")
                })
            );
            std::process::exit(1);
        }
    }
}
