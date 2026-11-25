use serde_json::Value;

/// RFC 6901 JSON Pointer component.
fn escape(component: &str) -> String {
    component.replace('~', "~0").replace('/', "~1")
}

fn child(path: &str, component: &str) -> String {
    format!("{path}/{}", escape(component))
}

/// Return the first differing JSON Pointer.
/// The empty string denotes the document root.
///
/// Numeric equality here deliberately uses serde_json::Value equality.
/// This is diagnostic only; acceptance uses exact serialized bytes.
pub(crate) fn first_json_difference(expected: &Value, actual: &Value) -> Option<String> {
    fn walk(expected: &Value, actual: &Value, path: &str) -> Option<String> {
        match (expected, actual) {
            (Value::Object(left), Value::Object(right)) => {
                // Deterministic regardless of serde_json/preserve_order.
                let keys: std::collections::BTreeSet<&str> = left
                    .keys()
                    .chain(right.keys())
                    .map(String::as_str)
                    .collect();

                for key in keys {
                    let next = child(path, key);
                    match (left.get(key), right.get(key)) {
                        (Some(a), Some(b)) => {
                            if let Some(found) = walk(a, b, &next) {
                                return Some(found);
                            }
                        }
                        _ => return Some(next),
                    }
                }
                None
            }
            (Value::Array(left), Value::Array(right)) => {
                for (index, (a, b)) in left.iter().zip(right).enumerate() {
                    let next = child(path, &index.to_string());
                    if let Some(found) = walk(a, b, &next) {
                        return Some(found);
                    }
                }
                if left.len() != right.len() {
                    Some(child(path, &left.len().min(right.len()).to_string()))
                } else {
                    None
                }
            }
            _ if expected == actual => None,
            _ => Some(path.to_owned()),
        }
    }

    walk(expected, actual, "")
}

/// Construct the persistence commitment before touching the filesystem.
pub(crate) fn encode_proof(proof: &Value) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec_pretty(proof)
}

/// Verify the execution-time serialized commitment.
///
/// Parsing is used ONLY to explain a byte mismatch. Parsed values are never
/// adopted as replacement expectations, and no tolerance is used.
pub(crate) fn verify_proof_bytes(
    case_id: &str,
    expected: &[u8],
    actual: &[u8],
) -> Result<(), String> {
    if expected == actual {
        return Ok(());
    }

    let byte_offset = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a != b)
        .unwrap_or(expected.len().min(actual.len()));

    let detail = match (
        serde_json::from_slice::<Value>(expected),
        serde_json::from_slice::<Value>(actual),
    ) {
        (Ok(left), Ok(right)) => match first_json_difference(&left, &right) {
            Some(pointer) => format!("field_path={pointer:?}"),
            None => "field_path=\"$serialization\" \
                 (different bytes; parsed JSON values compare equal)"
                .to_owned(),
        },
        (Err(_), _) => "field_path=\"$serialization\" \
             (execution-time serialization cannot be parsed)"
            .to_owned(),
        (_, Err(_)) => "field_path=\"$serialization\" \
             (saved evidence is invalid JSON)"
            .to_owned(),
    };

    Err(format!(
        "saved recovery proof changed: case_id={case_id} \
         {detail} byte_offset={byte_offset} \
         expected_bytes={} actual_bytes={}",
        expected.len(),
        actual.len(),
    ))
}
