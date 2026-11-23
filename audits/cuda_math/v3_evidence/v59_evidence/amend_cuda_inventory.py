#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import re

BASE = "603613720ef3c4667eb12509d6de11346cc4b4a8"
CURRENT = "63db92d873c61ed9f7dac0345aba837ff48a8a7f"
BASE_INVENTORY_SHA = (
    "3ac30e62c0cc960b10858681b31d14c765cbbfc9d6ab590a2cdf4c94b67acc1f"
)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def lines(data):
    text = data.decode("utf-8")
    if not text:
        return 0
    return len(text.split("\n")) - int(text.endswith("\n"))

def replace_once(data, old, new):
    old, new = old.encode("utf-8"), new.encode("utf-8")
    require(data.count(old) == 1, "replacement preimage not unique")
    return data.replace(old, new, 1)

MODULE = '''mod audit_placeholders {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/audit_placeholders.rs"
    ));
}

'''

M3_OLD = '''fn has_placeholder(v: &Value) -> bool {
    match v {
        Value::String(s) => {
            s.contains("FILLED_BY_INTEGRATOR")
                || s.contains("<SHA256:")
                || s.contains("PENDING_ROUNDTRIP")
        }
        Value::Array(a) => a.iter().any(has_placeholder),
        Value::Object(o) => o.values().any(has_placeholder),
        _ => false,
    }
}
'''
M3_NEW = MODULE + '''fn has_placeholder(v: &Value) -> bool {
    audit_placeholders::unresolved(v)
}
'''

CONTRACT_OLD = '''pub fn unresolved(value: &Value) -> bool {
    match value {
        Value::String(text) => [
            "RUNTIME",
            "FILLED_BY_INTEGRATOR",
            "<SHA256:",
            "PENDING_ROUNDTRIP",
        ]
        .iter()
        .any(|needle| text.contains(needle)),
        Value::Array(values) => values.iter().any(unresolved),
        Value::Object(values) => values
            .iter()
            .any(|(key, value)| key.contains("RUNTIME") || unresolved(value)),
        _ => false,
    }
}
'''
CONTRACT_NEW = MODULE + '''pub fn unresolved(value: &Value) -> bool {
    audit_placeholders::unresolved(value)
}
'''

# id, repo path, supplied preimage SHA-256, current LF line count,
# post-patch LF line count, region label, current anchors
RECORDS = [
    (
        "repo_serve_m3", "src/bin/serve_m3.rs",
        "7756694aafef4bd9893b5341b66fed1a494b78ce5a552f050b1867e438eb1c71",
        750, 750, "whole-file production entry point and audit loader",
        [
            "struct CudaBackend", "fn placeholder", "fn artifact_path",
            "fn load(", "async fn run(", "open_batched_with_attestation",
            "Service::spawn", "mod audit_placeholders",
            '"/src/audit_placeholders.rs"',
            "audit_placeholders::unresolved(value)"
        ]
    ),
    (
        "repo_golden_export_m4", "src/bin/golden_export_m4.rs",
        "37293a24d3f9aa0c1f653200f5086db377ffe08f3d77f7c72256b591c73c32ef",
        654, 654, "whole-file exporter audit gate and execution entry point",
        [
            "fn unresolved", "fn verify_audit", "fn validate_shape",
            "struct Export", "fn batch(", "pub fn run(",
            "open_batched_with_attestation", "mod audit_placeholders",
            '"/src/audit_placeholders.rs"',
            "audit_placeholders::unresolved(value)"
        ]
    ),
    (
        "repo_audit_placeholders", "src/audit_placeholders.rs",
        "f7756e6f5a43e2942a0ae809e99ac1dc4aba6d6507bafd983d092f2f6fb67d20",
        56, 56, "whole-file shared placeholder scanner and regression tests",
        [
            "pub fn unresolved", 'text == "RUNTIME"', 'key == "RUNTIME"',
            "Value::Array", "Value::Object",
            "FILLED_BY_INTEGRATOR", "<SHA256:", "PENDING_ROUNDTRIP",
            "fn rejects_exact_sentinels_recursively",
            "fn allows_descriptions_and_audit_identifiers",
            "fn retains_reserved_value_substring_rejection"
        ]
    ),
    (
        "repo_accept_m3", "src/bin/accept_m3.rs",
        "3ee476c36d0d4c3d150780bded79c9aafcd5449eb012e20ca3c2b71603bad269",
        1513, 1511, "whole-file legacy acceptance loader integration review",
        [
            "fn has_placeholder", "pub fn load_bundle",
            "!has_placeholder(&policy_val)", "!has_placeholder(&audit)",
            "fn artifact_path", "fn required_path",
            "running executable hash mismatch"
        ]
    ),
    (
        "repo_accept_m4_contract", "src/accept_m4_contract.rs",
        "e55a8ab3034b54177e89812869f101d14389784c63d0644b6fc4a7b832dfe2f2",
        640, 633, "whole-file prepass contract and audit binding review",
        [
            "pub fn unresolved", "pub fn load(", "fn load_candidates",
            "!unresolved(&input_value)", "!unresolved(&audit)",
            "active audit scope/bindings", "batch_export_provenance",
            "batch_export_manifest", "batch_shapes"
        ]
    ),
    (
        "repo_accept_m4_policy", "src/accept_m4_policy.rs",
        "2ec91c644c0313243eccd76215023eacb5c881d9bf90caeb3ea2053076716d7e",
        1921, 1921, "whole-file policy and evidence-consumer integration review",
        [
            "use crate::contract::{", "sha, unresolved, value",
            "fn acceptance_contract", "fn deployment_contract",
            "fn check_policy_header", "!unresolved(p)",
            'p["deployment"] == deployment_contract()',
            "fn consume_evidence", "fn prepare(", "pub fn materialize(",
            "pub fn recheck", "pub struct Ledger"
        ]
    )
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--post-loader-revision")
    args = parser.parse_args()

    package = pathlib.Path(args.package).resolve(strict=True)
    future = args.post_loader_revision
    if future is not None:
        require(bool(re.fullmatch(r"[0-9a-f]{40}", future)), "full revision")
        require(future not in (BASE, CURRENT), "new frozen revision required")

    raw = (package / "cuda_regions_6036137.json").read_bytes()
    require(sha(raw) == BASE_INVENTORY_SHA, "baseline inventory digest")
    inventory = json.loads(raw)
    require(inventory["repository_revision"] == BASE, "baseline revision")
    require(inventory["schema"] == "rustv1.region_inventory.v3.1", "schema")
    require(len(inventory["files"]) == 20, "baseline file count")
    require(sum(len(f["regions"]) for f in inventory["files"]) == 88,
            "baseline region count")

    by_id = {f["id"]: f for f in inventory["files"]}
    require(len(by_id) == 20, "duplicate baseline ID")
    expected_sources = {}
    provenance = []

    for ident, path, digest, before_n, after_n, label, anchors in RECORDS:
        attachment = "sources/" + path.replace("/", "__")
        data = (package / attachment).read_bytes()
        require(sha(data) == digest, "preimage digest: " + path)
        require(lines(data) == before_n, "preimage line count: " + path)
        anchors = list(anchors)

        if future:
            if ident == "repo_accept_m3":
                data = replace_once(data, M3_OLD, M3_NEW)
                anchors += [
                    "mod audit_placeholders", '"/src/audit_placeholders.rs"',
                    "audit_placeholders::unresolved(v)"
                ]
            elif ident == "repo_accept_m4_contract":
                data = replace_once(data, CONTRACT_OLD, CONTRACT_NEW)
                anchors += [
                    "mod audit_placeholders", '"/src/audit_placeholders.rs"',
                    "audit_placeholders::unresolved(value)"
                ]
            elif ident == "repo_accept_m4_policy":
                data = replace_once(
                    data,
                    '"runtime_placeholder_rejection": '
                    '"recursive_exact_or_embedded_RUNTIME"',
                    '"runtime_placeholder_rejection": '
                    '"recursive_exact_RUNTIME_and_reserved_value_substrings"'
                )
                data = replace_once(
                    data,
                    "// This rejects embedded RUNTIME as well as the earlier placeholder forms.",
                    "// Reject exact RUNTIME keys/values and reserved value substrings."
                )
                anchors.append(
                    "recursive_exact_RUNTIME_and_reserved_value_substrings"
                )

        count = after_n if future else before_n
        require(lines(data) == count, "result line count: " + path)
        text = data.decode("utf-8")
        require(all(a in text for a in anchors), "missing anchor: " + path)
        result_digest = sha(data)
        entry = {
            "id": ident,
            "classification": "new",
            "root": "repo",
            "path": path,
            "sha256": result_digest,
            "regions": [[1, count, label, anchors]]
        }
        if ident in by_id:
            by_id[ident].clear()
            by_id[ident].update(entry)
        else:
            inventory["files"].append(entry)

        expected_sources[path] = data
        provenance.append({
            "path": path,
            "package_path": attachment,
            "preimage_sha256": digest,
            "expected_sha256": result_digest,
            "first_line": 1,
            "last_line": count
        })

    inventory["repository_revision"] = future or CURRENT
    require(len(inventory["files"]) == 24, "amended file count")
    require(sum(len(f["regions"]) for f in inventory["files"]) == 92,
            "amended region count")

    # A new output directory prevents overwriting earlier evidence.
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=False, exist_ok=False)
    for path, data in expected_sources.items():
        dest = out / "expected_sources" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    inventory_bytes = (
        json.dumps(inventory, indent=2) + "\n"
    ).encode("utf-8")
    (out / "inventory.json").write_bytes(inventory_bytes)
    record = {
        "basis": "bound supplied preimages; approved replacements in memory",
        "checkout_used_to_generate_expectations": False,
        "repository_revision": future or CURRENT,
        "baseline_inventory_sha256": BASE_INVENTORY_SHA,
        "generator_sha256": sha(pathlib.Path(__file__).read_bytes()),
        "inventory_sha256": sha(inventory_bytes),
        "post_loader_patch": bool(future),
        "files": provenance
    }
    (out / "amendment_provenance.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )

if __name__ == "__main__":
    main()
