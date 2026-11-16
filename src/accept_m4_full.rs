// Feature-gated acceptance driver dispatcher.
// No serving behavior changes and no supervisor protocol additions.

use std::io::Write;

const PHASE: &str = "full_acceptance";

fn syntax(args: &[String]) -> bool {
    args.len() == 8 && args.get(1).map(String::as_str) == Some("acceptance-full")
}

fn emit(value: &serde_json::Value) -> std::io::Result<()> {
    let mut bytes = serde_json::to_vec(value).map_err(std::io::Error::other)?;
    bytes.push(b'\n');
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    output.write_all(&bytes)?;
    output.flush()
}

fn failure(code: i32, category: &str, error: &str) -> i32 {
    let value = serde_json::json!({
        "kind": "done",
        "acceptance_phase": PHASE,
        "command": "acceptance-full",
        "accepted": false,
        "m4_accepted": false,
        "pass": false,
        "exit_code": code,
        "failure_category": category,
        "error": error
    });
    if emit(&value).is_err() {
        74
    } else {
        code
    }
}

#[cfg(feature = "cuda")]
fn execute(args: &[String]) -> anyhow::Result<crate::acceptance_scheduler::AcceptanceCompletion> {
    use crate::{acceptance_core as core, acceptance_scheduler as scheduler};
    use std::path::Path;

    let c = core::run_prefix(
        Path::new(&args[2]),
        Path::new(&args[3]),
        &args[4],
        Path::new(&args[5]),
        &args[6],
        Path::new(&args[7]),
    )?;

    let c = scheduler::run_coalescing(c)?;
    let c = scheduler::run_physical_b_capacity(c)?;
    let c = scheduler::run_remaining_capacity(c)?;
    let c = scheduler::run_queue_expiry_wait(c)?;
    let c = scheduler::run_invalid_tokens_quiesce(c)?;
    let c = scheduler::run_drain_failure_expiry(c)?;
    let c = scheduler::run_owners_binding_drift_readiness(c)?;
    let c = scheduler::run_restore_restart_exit(c)?;
    let c = scheduler::run_replacement_reattestation(c)?;
    let c = scheduler::run_exact_recovery(c)?;
    let c = scheduler::run_final_drain(c)?;

    let c = scheduler::reconcile_local_artifacts(c)?;
    let c = scheduler::reconcile_imported_artifacts(c)?;
    let c = scheduler::reconcile_initial_launch_artifacts(c)?;
    let c = scheduler::reconcile_recovery_launch_lineage(c)?;
    let c = scheduler::reconcile_recovery_http_artifacts(c)?;
    let c = scheduler::reconcile_destructive_children_and_owners(c)?;
    let c = scheduler::reconcile_poisoned_launch_artifacts(c)?;
    let c = scheduler::reconcile_independent_artifacts(c)?;
    let c = scheduler::reconcile_scheduler_artifacts(c)?;
    let c = scheduler::reconcile_output_coverage(c)?;

    scheduler::finish_acceptance(c)
}

/// Returns the process exit code. Caller must exit with this value.
pub(crate) fn main(args: &[String]) -> i32 {
    if !syntax(args) {
        return failure(
            2,
            "usage",
            "usage: accept_m4 acceptance-full MODEL_DIR PREPASS_JSON \
             PREPASS_SHA256 POLICY_JSON BASE_URL OUT_DIR",
        );
    }

    #[cfg(not(feature = "cuda"))]
    {
        failure(
            2,
            "build_configuration",
            "acceptance-full requires both cuda and scheduler-test-seam",
        )
    }

    #[cfg(feature = "cuda")]
    {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(args)));
        match result {
            Ok(Ok(completion)) => {
                let value = serde_json::json!({
                    "kind": "done",
                    "acceptance_phase": PHASE,
                    "command": "acceptance-full",
                    "accepted": true,
                    "m4_accepted": true,
                    "pass": true,
                    "exit_code": 0,
                    "policy_sha256": completion.policy_sha256,
                    "completion_manifest_path": completion.completion_manifest_path,
                    "completion_manifest_sha256":
                        completion.completion_manifest_sha256,
                    "records_sha256": completion.ledger["records_sha256"],
                    "required_case_count": completion.ledger["required_case_count"],
                    "completed_case_count": completion.ledger["completed_case_count"],
                    "pending_required_cases": [],
                    "required_cases_excluded": [],
                    "exceptions": []
                });
                if emit(&value).is_err() {
                    74
                } else {
                    0
                }
            }
            Ok(Err(error)) => failure(1, "acceptance_failure", &format!("{error:#}")),
            Err(_) => failure(101, "panic", "acceptance driver panicked"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_dispatcher_requires_exact_positional_argv() {
        let args: Vec<String> = [
            "accept_m4",
            "acceptance-full",
            "/model",
            "/input.json",
            "input-sha",
            "/policy.json",
            "http://127.0.0.1:9700",
            "/new-output",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();

        assert!(syntax(&args));
        assert!(!syntax(&args[..7]));

        let mut extra = args.clone();
        extra.push("--skip-cuda".to_owned());
        assert!(!syntax(&extra));

        let mut other = args;
        other[1] = "acceptance-http".to_owned();
        assert!(!syntax(&other));
    }
}
