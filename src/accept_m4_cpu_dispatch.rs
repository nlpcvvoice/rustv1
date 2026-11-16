use std::{io::Write, path::Path};

fn emit(value: &serde_json::Value) -> std::io::Result<()> {
    let mut bytes = serde_json::to_vec(value).map_err(std::io::Error::other)?;
    bytes.push(b'\n');
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    output.write_all(&bytes)?;
    output.flush()
}

pub(crate) fn main(args: &[String]) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> anyhow::Result<serde_json::Value> {
            anyhow::ensure!(
                args.len() == 7 && args.get(1).map(String::as_str) == Some("cpu-harness"),
                "usage: accept_m4 cpu-harness MODEL_DIR PREPASS_JSON \
                 PREPASS_SHA256 POLICY_JSON OUT_DIR"
            );
            crate::acceptance_scheduler::run_cpu_harness(
                Path::new(&args[2]),
                Path::new(&args[3]),
                &args[4],
                Path::new(&args[5]),
                Path::new(&args[6]),
            )
        },
    ));
    let (code, value) = match result {
        Ok(Ok(value)) => (0, value),
        Ok(Err(error)) => (
            1,
            serde_json::json!({
                "kind": "cpu_harness_done",
                "accepted": false,
                "production_acceptance": false,
                "harness_checks_passed": false,
                "error": format!("{error:#}")
            }),
        ),
        Err(_) => (
            101,
            serde_json::json!({
                "kind": "cpu_harness_done",
                "accepted": false,
                "production_acceptance": false,
                "harness_checks_passed": false,
                "error": "CPU harness panicked"
            }),
        ),
    };
    if emit(&value).is_err() {
        74
    } else {
        code
    }
}
