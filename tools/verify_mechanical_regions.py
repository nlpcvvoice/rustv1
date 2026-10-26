#!/usr/bin/env python3
"""Mechanical region-boundary re-verification for the cuda_math audit.

For every source_file in the audit artifact:
  1. fetch the exact crate archive from crates.io and verify the crate
     sha256 against the audit's crates section;
  2. verify each source file sha256 against the audit record;
  3. verify region boundaries: 1 <= first_line <= last_line <= line_count;
  4. verify the region symbol appears within [first_line, last_line]
     (matching the final path segment of the recorded symbol).

Emits a JSON report with per-file/per-region results and a report
sha256 for binding into the audit v2 predraft.

Usage:
  python3 tools/verify_mechanical_regions.py \
    --audit audits/cuda_math/candle-0.9.2_cudarc-0.19.10.v1.json \
    --out tmp/mechanical_reverification_report.json
"""
import argparse
import hashlib
import io
import json
import re
import tarfile
import time
import urllib.request


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_crate(name: str, version: str) -> bytes:
    url = f"https://static.crates.io/crates/{name}/{name}-{version}.crate"
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    audit = json.load(open(args.audit))
    crates = audit.get("crates", {})
    crate_hashes = {
        name: info.get("cargo_lock_checksum")
        for name, info in crates.items()
    }
    repo_rev = None
    for sf in audit["source_files"]:
        if sf["owner"] == "rustv1":
            repo_rev = sf["version_or_revision"]
            break

    archives = {}
    files = {}
    results = []
    all_pass = True

    for sf in audit["source_files"]:
        owner, ver, path = sf["owner"], sf["version_or_revision"], sf["path"]
        if owner == "rustv1" and path.startswith("rustv1/"):
            path = path[len("rustv1/"):]
        fres = {"id": sf["id"], "owner": owner, "version": ver, "path": path,
                "regions": [], "file_sha256_match": False}
        key = (owner, ver)
        if owner == "rustv1":
            import subprocess
            blob = subprocess.run(
                ["git", "-C", "NVIDIA/github/rustv1", "show", f"{ver}:{path}"],
                capture_output=True).stdout or None
            if blob is None:
                fres["error"] = "path not found at pinned revision"
                fres["pass"] = False
                all_pass = False
                results.append(fres)
                continue
            fres["source"] = f"git rev {ver}"
        else:
            if key not in archives:
                raw = fetch_crate(owner, ver)
                archives[key] = raw
                crate_sha = crate_hashes.get(owner)
                fres["crate_sha256"] = sha256_bytes(raw)
                fres["crate_sha256_expected"] = crate_sha
                fres["crate_sha256_match"] = (crate_sha == sha256_bytes(raw))
                if crate_sha and not fres["crate_sha256_match"]:
                    all_pass = False
                tf = tarfile.open(fileobj=io.BytesIO(raw))
                prefix = f"{owner}-{ver}/"
                for m in tf.getmembers():
                    if m.name.startswith(prefix):
                        files[(key, m.name[len(prefix):])] = tf.extractfile(m).read()
            else:
                fres["crate_sha256_match"] = True
            blob = files.get((key, path))
            if blob is None:
                fres["error"] = "file not found in crate archive"
                fres["pass"] = False
                all_pass = False
                results.append(fres)
                continue
        fsha = sha256_bytes(blob)
        fres["file_sha256"] = fsha
        fres["file_sha256_expected"] = sf["sha256"]
        fres["file_sha256_match"] = (fsha == sf["sha256"])
        if not fres["file_sha256_match"]:
            all_pass = False

        lines = blob.decode("utf-8", errors="replace").splitlines()
        fres["line_count"] = len(lines)
        for reg in sf["examined_regions"]:
            fl, ll, sym = reg["first_line"], reg["last_line"], reg["symbol"]
            sym_key = re.split(r"[,<( ]", sym.split(">::")[-1].split("::")[-1])[0]
            sym_key = sym_key.strip()
            if not sym_key:
                sym_key = sym.split()[0]
            in_bounds = 1 <= fl <= ll <= len(lines)
            window = "\n".join(lines[fl - 1:ll]) if in_bounds else ""
            sym_found = sym_key in window if in_bounds else False
            ok = in_bounds and sym_found
            if not ok:
                all_pass = False
            fres["regions"].append({"symbol": sym, "first_line": fl,
                                    "last_line": ll, "in_bounds": in_bounds,
                                    "symbol_in_region": sym_found,
                                    "pass": ok})
        fres["pass"] = (fres["file_sha256_match"]
                        and all(r["pass"] for r in fres["regions"]))
        results.append(fres)

    report = {
        "schema": "rustv1.mechanical_region_boundary_reverification.v1",
        "audit_artifact": audit.get("intended_path"),
        "audit_revision": audit.get("artifact_revision"),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
        "tool": "tools/verify_mechanical_regions.py",
        "all_pass": all_pass,
        "files": results,
    }
    body = json.dumps(report, indent=1, sort_keys=True)
    open(args.out, "w").write(body)
    print("REPORT_SHA256", hashlib.sha256(body.encode()).hexdigest())
    print("ALL_PASS", all_pass)
    for f in results:
        bad = [r for r in f["regions"] if not r["pass"]]
        print(f"{f['id']} {f['path']}: file_sha={f['file_sha256_match']} "
              f"regions={len(f['regions'])} failed={len(bad)}")


if __name__ == "__main__":
    main()
