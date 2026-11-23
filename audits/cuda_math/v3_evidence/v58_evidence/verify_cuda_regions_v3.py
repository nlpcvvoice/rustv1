#!/usr/bin/env python3
"""Bounds, explicit-anchor and byte-identity checks; not a Rust parser."""
import argparse
import hashlib
import json
import pathlib
import platform
import re
import subprocess
import sys

HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def relative_path(text):
    require(isinstance(text, str) and text, "empty/non-string path")
    require("\\" not in text and "\x00" not in text, "invalid path")
    pieces = text.split("/")
    require(all(p not in ("", ".", "..") for p in pieces), "unsafe path")
    require(not text.startswith("/"), "absolute path")
    return pathlib.PurePosixPath(text)


def local_regular(root, relative):
    parts = relative_path(relative).parts
    path = root
    for part in parts:
        path = path / part
        require(not path.is_symlink(), "symlink not allowed: " + str(path))
    require(path.is_file(), "not a regular file: " + str(path))
    return path


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        result.returncode == 0,
        "git failed: " + result.stderr.decode("utf-8", errors="replace"),
    )
    return result.stdout


def lines_of(data):
    # LF-delimited source lines; terminal LF does not add an empty line.
    text = data.decode("utf-8", errors="strict")
    if not text:
        return []
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--crate-root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo = pathlib.Path(args.repo).resolve(strict=True)
    crate_root = pathlib.Path(args.crate_root).resolve(strict=True)
    inventory_path = pathlib.Path(args.inventory).resolve(strict=True)
    verifier_path = pathlib.Path(__file__).resolve(strict=True)
    inventory_bytes = inventory_path.read_bytes()
    verifier_bytes = verifier_path.read_bytes()
    inventory = json.loads(
        inventory_bytes.decode("utf-8"), object_pairs_hook=unique_object
    )

    require(inventory["schema"] == "rustv1.region_inventory.v3.1", "schema")
    revision = inventory["repository_revision"]
    require(bool(HEX40.fullmatch(revision)), "full revision required")
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    require(head == revision, "HEAD differs from inventory revision")
    require(
        not git(repo, "diff", "--name-only", "HEAD", "--"),
        "tracked working-tree/index modifications",
    )

    # The aliases are explicit, not basename or suffix guessing.
    aliases = inventory["repo_path_aliases"]
    for old, new in aliases.items():
        relative_path(old)
        relative_path(new)

    report = {
        "schema": "rustv1.region_report.v3.1",
        "repository_revision": revision,
        "inventory_sha256": sha(inventory_bytes),
        "verifier_sha256": sha(verifier_bytes),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "repo_root": str(repo),
        "crate_root": str(crate_root),
        "line_semantics": "one-based inclusive, LF-delimited",
        "anchor_semantics": "literal substring, all anchors required",
        "rust_syntax_parsed": False,
        "semantic_audit_completed_by_verifier": False,
        "repo_path_aliases": aliases,
        "files": [],
        "all_pass": True,
    }

    ids = set()
    locations = set()
    region_count = 0
    passing_regions = 0
    passing_hashes = 0

    require(inventory["files"], "empty inventory")
    for entry in inventory["files"]:
        file_id = entry["id"]
        require(file_id not in ids, "duplicate file id: " + file_id)
        ids.add(file_id)
        require(
            entry["classification"] in ("inherited", "renewed", "new", "binding"),
            "classification",
        )
        expected = entry["sha256"]
        require(bool(HEX64.fullmatch(expected)), "invalid expected SHA-256")

        root_kind = entry["root"]
        path = entry["path"]
        if root_kind == "repo":
            path = aliases.get(path, path)
        else:
            require(root_kind == "crate", "unknown source root")
        relative_path(path)
        location = (root_kind, path)
        require(location not in locations, "duplicate normalized location")
        locations.add(location)

        item = {
            "id": file_id,
            "classification": entry["classification"],
            "root": root_kind,
            "declared_path": entry["path"],
            "normalized_path": path,
            "expected_sha256": expected,
            "regions": [],
        }
        try:
            root = repo if root_kind == "repo" else crate_root
            disk_bytes = local_regular(root, path).read_bytes()
            if root_kind == "repo":
                tree = git(repo, "ls-tree", revision, "--", path)
                require(
                    tree.startswith(b"100644 blob ")
                    or tree.startswith(b"100755 blob "),
                    "repository entry is not an ordinary tracked blob",
                )
                data = git(repo, "show", revision + ":" + path)
                require(disk_bytes == data, "working file differs from Git blob")
                item["working_copy_matches_blob"] = True
            else:
                data = disk_bytes

            actual = sha(data)
            item["actual_sha256"] = actual
            item["sha256_match"] = actual == expected
            passing_hashes += int(actual == expected)
            lines = lines_of(data)
            item["line_count"] = len(lines)
            regions = entry["regions"]
            require(
                entry["classification"] == "binding" or bool(regions),
                "source file has no regions",
            )

            for number, region in enumerate(regions):
                first, last, symbol, anchors = region
                require(
                    type(first) is int and type(last) is int,
                    "bounds must be integers",
                )
                require(isinstance(symbol, str) and symbol, "missing symbol")
                require(
                    isinstance(anchors, list)
                    and anchors
                    and all(isinstance(a, str) and a for a in anchors),
                    "missing/invalid anchors",
                )
                in_bounds = 1 <= first <= last <= len(lines)
                excerpt = "\n".join(lines[first - 1:last]) if in_bounds else ""
                anchor_results = [
                    {"text": anchor, "present": anchor in excerpt}
                    for anchor in anchors
                ]
                passed = (
                    item["sha256_match"]
                    and in_bounds
                    and all(a["present"] for a in anchor_results)
                )
                item["regions"].append({
                    "index": number,
                    "symbol": symbol,
                    "first_line": first,
                    "last_line": last,
                    "in_bounds": in_bounds,
                    "anchors": anchor_results,
                    "pass": passed,
                })
                region_count += 1
                passing_regions += int(passed)

            item["pass"] = item["sha256_match"] and all(
                r["pass"] for r in item["regions"]
            )
        except Exception as error:
            item["pass"] = False
            item["error"] = str(error)

        report["files"].append(item)
        report["all_pass"] &= item["pass"]

    require(
        inventory_path.read_bytes() == inventory_bytes,
        "inventory changed during verification",
    )
    require(
        verifier_path.read_bytes() == verifier_bytes,
        "verifier changed during verification",
    )
    require(
        git(repo, "rev-parse", "HEAD").decode().strip() == revision,
        "HEAD changed during verification",
    )

    report["file_count"] = len(report["files"])
    report["passing_file_hash_comparisons"] = passing_hashes
    report["region_count"] = region_count
    report["passing_region_checks"] = passing_regions
    output = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # Refuse to overwrite a prior report.
    with open(args.out, "xb") as handle:
        handle.write(output)
    sys.stdout.buffer.write(output)
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("VERIFIER ERROR: " + str(error), file=sys.stderr)
        sys.exit(2)
