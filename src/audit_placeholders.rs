use serde_json::Value;

// Shared policy/audit placeholder semantics. This is not a substitute for
// schema validation, required fields, digest validation, or artifact binding.
pub fn unresolved(value: &Value) -> bool {
    match value {
        Value::String(text) => {
            text == "RUNTIME"
                || ["FILLED_BY_INTEGRATOR", "<SHA256:", "PENDING_ROUNDTRIP"]
                    .iter()
                    .any(|marker| text.contains(marker))
        }
        Value::Array(values) => values.iter().any(unresolved),
        Value::Object(values) => values
            .iter()
            .any(|(key, value)| key == "RUNTIME" || unresolved(value)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::unresolved;
    use serde_json::json;

    #[test]
    fn rejects_exact_sentinels_recursively() {
        for value in [
            json!("RUNTIME"),
            json!({"RUNTIME": 1}),
            json!({"x": [{"y": "RUNTIME"}]}),
        ] {
            assert!(unresolved(&value));
        }
    }

    #[test]
    fn allows_descriptions_and_audit_identifiers() {
        for text in [
            "RUNTIME_NUMERICAL_ACCEPTANCE",
            "recursive_exact_or_embedded_RUNTIME",
            "Replace every RUNTIME value before launch.",
        ] {
            assert!(!unresolved(&json!({"description": text})));
        }
    }

    #[test]
    fn retains_reserved_value_substring_rejection() {
        for marker in ["FILLED_BY_INTEGRATOR", "<SHA256:", "PENDING_ROUNDTRIP"] {
            assert!(unresolved(&json!({"x": [format!("before{marker}after")]})));
        }
        assert!(!unresolved(&json!({"FILLED_BY_INTEGRATOR": 1})));
        assert!(!unresolved(&json!(" RUNTIME ")));
    }
}
