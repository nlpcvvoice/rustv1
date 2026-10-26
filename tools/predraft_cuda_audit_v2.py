#!/usr/bin/env python3
"""Prepare an unsigned CUDA audit v2 renewal at integration commit 39c9b31.

Run from the repository root with Python 3.11+.

This tool computes provenance and records historical/current source differences.
It does not perform source-region verification, feature-resolution review,
schema validation, or auditor signing.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib


SCHEMA = "rustv1.cuda_math_source_audit.v1"
EXPECTED_HEAD = "39c9b31"
REPORT_SHA256 = (
    "af934f6db4d3643f7181b1da18cf3c936"
    "7bd31bef0a98a165d9fffaff52323db"
)
FINAL_PATH = "audits/cuda_math/candle-0.9.2_cudarc-0.19.10.v2.json"
DRAFT_PATH = "audits/cuda_math/candle-0.9.2_cudarc-0.19.10.v2.predraft.json"
DROP_SYMBOL = "<MathModePin as Drop>::drop"

EXPECTED_CRATES = {
    "candle-core": (
        "0.9.2",
        "c15b675b80d994b2eadb20a4bbe434eabeb454eac3ee5e2b4cf6f147ee9be091",
    ),
    "candle-nn": (
        "0.9.2",
        "3045fa9e7aef8567d209a27d56b692f60b96f4d0569f4c3011f8ca6715c65e03",
    ),
    "cudarc": (
        "0.19.10",
        "7359bf1de037ddada72729c3acec9b040f3468400c6e9e740b01ba0cd2e89562",
    ),
    "candle-kernels": (
        "0.9.2",
        "b8455f84bd810047c7c41216683c1020c915a9f8a740b3b0eabdd4fb2fbaa660",
    ),
    "ring": (
        "0.17.14",
        "a4689e6c2294d81e88dc6261c768b63bc4fcdb852be6d1352498b114f61383b7",
    ),
}

REPO_SOURCES = {
    "repo_precision": "src/precision.rs",
    "repo_cuda_math": "src/cuda_math.rs",
    "repo_m2_route": "src/m2_route.rs",
    "repo_model": "src/model.rs",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(*args):
    return subprocess.check_output(["git", *args])


def git_text(*args):
    return git_bytes(*args).decode("utf-8").strip()


def unique_by_id(items):
    result = {}
    for item in items:
        identity = item["id"]
        require(identity not in result, f"duplicate source id: {identity}")
        result[identity] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1",
        default="audits/cuda_math/candle-0.9.2_cudarc-0.19.10.v1.json",
    )
    parser.add_argument(
        "--report",
        default="tmp/mechanical_reverification_report.json",
    )
    parser.add_argument("--lock", default="Cargo.lock")
    parser.add_argument("--out", default=DRAFT_PATH)
    args = parser.parse_args()

    root = Path(git_text("rev-parse", "--show-toplevel")).resolve()
    require(Path.cwd().resolve() == root, "run from repository root")

    head = git_text("rev-parse", "--verify", "HEAD^{commit}")
    expected = git_text(
        "rev-parse", "--verify", f"{EXPECTED_HEAD}^{{commit}}"
    )
    require(re.fullmatch(r"[0-9a-f]{40}", head), "full Git SHA required")
    require(head == expected, "not integration commit 39c9b31")
    require(
        not git_text("diff", "--name-only", "HEAD", "--"),
        "tracked checkout differs from HEAD",
    )

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out = out.resolve()
    require(out.is_relative_to(root), "output must be inside repository")
    out_relative = out.relative_to(root).as_posix()
    require(out_relative != FINAL_PATH, "refusing to write final audit path")

    base_bytes = Path(args.v1).read_bytes()
    report_bytes = Path(args.report).read_bytes()
    lock_bytes = Path(args.lock).read_bytes()
    require(sha(report_bytes) == REPORT_SHA256, "report digest mismatch")
    require(
        lock_bytes == git_bytes("show", f"{head}:Cargo.lock"),
        "supplied lock differs from committed Cargo.lock",
    )

    old = json.loads(base_bytes)
    report = json.loads(report_bytes)
    lock = tomllib.loads(lock_bytes.decode("utf-8"))

    require(old["schema"] == SCHEMA, "unexpected audit schema")
    require(old["artifact_revision"] == "v1", "unexpected base revision")
    require(
        report["schema"]
        == "rustv1.mechanical_region_boundary_reverification.v1",
        "unexpected report schema",
    )
    require(report["all_pass"] is False, "unexpected historical report")

    for name, (version, checksum) in EXPECTED_CRATES.items():
        matches = [
            package for package in lock["package"]
            if package["name"] == name and package["version"] == version
        ]
        require(len(matches) == 1, f"missing or ambiguous package: {name}")
        require(
            matches[0].get("source")
            == "registry+https://github.com/rust-lang/crates.io-index",
            f"unexpected package source: {name}",
        )
        require(
            matches[0].get("checksum") == checksum,
            f"unexpected package checksum: {name}",
        )
        if name != "ring":
            require(
                old["crates"][name]["version"] == version
                and old["crates"][name]["cargo_lock_checksum"] == checksum,
                f"historical crate binding differs: {name}",
            )

    sources = unique_by_id(old["source_files"])
    reported = unique_by_id(report["files"])
    require(set(sources) == set(reported), "report/source inventory mismatch")
    require(len(sources) == 14, "unexpected historical source count")

    failures = []
    region_count = 0
    for identity, source in sources.items():
        evidence = reported[identity]
        require(
            evidence["file_sha256_match"] is True
            and evidence["file_sha256"] == source["sha256"]
            and evidence["file_sha256_expected"] == source["sha256"],
            f"historical source binding differs: {identity}",
        )
        actual_regions = evidence["regions"]
        expected_regions = source["examined_regions"]
        require(
            len(actual_regions) == len(expected_regions),
            f"historical region count differs: {identity}",
        )
        for actual, recorded in zip(actual_regions, expected_regions):
            require(
                all(
                    actual[key] == recorded[key]
                    for key in ("symbol", "first_line", "last_line")
                ),
                f"historical region differs: {identity}",
            )
            region_count += 1
            if actual["pass"] is not True:
                failures.append((identity, actual["symbol"]))

    require(region_count == 100, "unexpected historical region total")
    require(
        failures == [("repo_cuda_math", DROP_SYMBOL)],
        "unexpected historical failures",
    )

    current_sources = []
    current_bytes = {}
    for identity, path in REPO_SOURCES.items():
        committed = git_bytes("show", f"{head}:{path}")
        require(Path(path).read_bytes() == committed, f"dirty source: {path}")
        current_bytes[identity] = committed
        current_sources.append({
            "id": identity,
            "path": path,
            "repository_revision": head,
            "sha256": sha(committed),
            "historical_sha256": sources[identity]["sha256"],
            "matches_historical_source": sha(committed) == sources[identity]["sha256"],
        })

    require(
        sha(current_bytes["repo_cuda_math"]) == sources["repo_cuda_math"]["sha256"],
        "cuda_math changed; supplied boundary correction requires new review",
    )
    lines = current_bytes["repo_cuda_math"].decode("utf-8").splitlines()
    expected_drop = [
        "impl Drop for MathModePin {",
        "    fn drop(&mut self) {",
        "        // Deliberately leave all settings unchanged. Restoring old settings",
        "        // would introduce an implicit global policy transition.",
        "        PIN_ACTIVE.store(false, Ordering::Release);",
        "    }",
        "}",
    ]
    require(lines[153:160] == expected_drop, "Drop boundary text differs")

    draft = copy.deepcopy(old)
    draft_sources = unique_by_id(draft["source_files"])
    drop_regions = [
        region
        for region in draft_sources["repo_cuda_math"]["examined_regions"]
        if region["symbol"] == DROP_SYMBOL
    ]
    require(len(drop_regions) == 1, "ambiguous Drop region")
    drop_regions[0]["first_line"] = 155
    drop_regions[0]["last_line"] = 159

    now = datetime.now(timezone.utc).isoformat()
    changed = [
        source["id"] for source in current_sources
        if not source["matches_historical_source"]
    ]
    draft.update({
        "artifact_revision": "v2",
        "intended_path": out_relative,
        "status": "unsigned_predraft",
        "status_reason": (
            "Current commit and lock bindings computed. Historical report "
            "bound without changing its failed result. Drop boundary corrected. "
            "Current region verification, source renewal, feature review and "
            "auditor sign-off remain incomplete."
        ),
        "signed": False,
        "audited_at_utc": None,
        "generated_at_utc": now,
        "auditor": "Unsigned renewal generator; no renewed auditor sign-off.",
        "repository_revision": head,
        "cargo_lock_sha256": sha(lock_bytes),
    })

    for key, value in list(draft["verification"].items()):
        if isinstance(value, bool):
            draft["verification"][key] = False
    draft["verification"]["auditor_attribution_note"] = (
        "Historical findings remain inherited claims. This run computes "
        "provenance and checks the exact Drop excerpt only; it does not "
        "complete mechanical verification of all current regions."
    )

    draft["schema_compatibility"].update({
        "required_fields_populated": False,
        "structurally_reviewed_against_supplied_schema": False,
        "validated": False,
        "reason": "Unsigned renewal; final schema validation not performed.",
    })

    draft["cargo_tree_features"]["note"] = (
        "Historical evidence only; not regenerated against the current lock."
    )
    draft["crates"]["cudarc"]["note"] = (
        "The audited direct package is 0.19.10. The separately locked 0.17.8 "
        "package occurs through ug-cuda in the lockfile. Current enabled-graph "
        "status is not established by this generator."
    )

    draft["provenance"] = {
        "repository_revision": head,
        "generated_at_utc": now,
        "cargo_lock": {"path": "Cargo.lock", "sha256": sha(lock_bytes)},
        "base_audit": {"path": args.v1, "sha256": sha(base_bytes)},
        "historical_provenance": old.get("provenance"),
        "historical_mechanical_report": {
            "path": args.report,
            "sha256": REPORT_SHA256,
            "all_pass": False,
            "scope": "Historical source hashes and recorded regions, not current HEAD.",
        },
        "current_repository_sources": current_sources,
        "changed_repository_source_ids": changed,
        "boundary_correction": {
            "source_file_id": "repo_cuda_math",
            "symbol": DROP_SYMBOL,
            "old_first_line": 158,
            "old_last_line": 162,
            "new_first_line": 155,
            "new_last_line": 159,
            "basis": "Exact enclosing impl excerpt checked at current HEAD.",
        },
        "source_hash_note": (
            "source_files retains historical revisions and hashes. Current "
            "hashes are recorded separately above, not substituted beneath "
            "unverified historical regions."
        ),
    }

    draft["conclusions"]["ready_for_acceptance_gate"] = False
    draft["conclusions"]["runtime_acceptance_completed_by_this_audit"] = False
    draft["conclusions"]["qualification"] = (
        "Inherited source conclusions are not renewed for the current commit. "
        "Historical prose citations may be stale. This artifact cannot clear "
        "the acceptance gate."
    )
    draft["blocking_items"] = [
        "CURRENT_SOURCE_REGIONS_AND_FINDINGS",
        "CURRENT_MECHANICAL_VERIFICATION",
        "CURRENT_FEATURE_TREE_AND_RESOLUTION_REVIEW",
        "FINAL_SCHEMA_VALIDATION_AND_AUDITOR_SIGNOFF",
    ]
    draft["blocking_item_resolution"] = [
        {"id": item, "status": "OPEN"} for item in draft["blocking_items"]
    ]
    draft["operational_restrictions"].insert(
        0,
        "Do not publish this unsigned predraft as an accepted audit.",
    )
    draft["renewal_target_path"] = FINAL_PATH

    raw = (
        json.dumps(draft, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as destination:
        destination.write(raw)

    print(json.dumps({
        "output": out_relative,
        "artifact_sha256": sha(raw),
        "repository_revision": head,
        "cargo_lock_sha256": sha(lock_bytes),
        "changed_repository_source_ids": changed,
        "status": draft["status"],
        "signed": False,
        "ready_for_acceptance_gate": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()