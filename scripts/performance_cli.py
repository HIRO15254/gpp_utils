#!/usr/bin/env python3
"""Compare real ``gpp run`` binaries without reusing a production data root.

Each measured child gets a fresh temporary ``--root``. Timing is wall-clock
time around that child process; correctness comes from normalized saved results,
not from its stdout. Run this driver serially after builds have finished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import tempfile
import time
import statistics
from pathlib import Path


SEED_FILE = re.compile(r"seed_(\d+)\.json$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_result(value: object) -> object:
    """Match examples/bench_batch.rs: retain every non-timing result field."""
    if not isinstance(value, dict):
        raise ValueError("saved result must be a JSON object")
    value = value.copy()
    value.pop("attempt_id", None)
    value.pop("elapsed_ms", None)
    diagnostics = value.get("diagnostics")
    if isinstance(diagnostics, dict):
        diagnostics = diagnostics.copy()
        diagnostics.pop("search_ms", None)
        diagnostics.pop("measurement_ms", None)
        value["diagnostics"] = diagnostics
    return value


def canonical_dump(value: object) -> bytes:
    """Serialize JSON values with sorted keys and exact IEEE-754 float bits.

    ``json.loads`` creates Python ``float`` values using correctly rounded
    binary64 conversion. Encoding their bits rather than decimal text prevents
    Python/Rust shortest-decimal formatting choices from changing a signature.
    """
    if value is None:
        return b"n"
    if value is True:
        return b"b1"
    if value is False:
        return b"b0"
    if isinstance(value, int):
        return b"i" + str(value).encode("ascii") + b";"
    if isinstance(value, float):
        return b"f" + struct.pack("<Q", struct.unpack("<Q", struct.pack("<d", value))[0])
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return b"s" + len(encoded).to_bytes(8, "little") + encoded
    if isinstance(value, list):
        return b"a" + len(value).to_bytes(8, "little") + b"".join(canonical_dump(item) for item in value)
    if isinstance(value, dict):
        parts = [b"o", len(value).to_bytes(8, "little")]
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError("JSON object key is not a string")
            parts.extend((canonical_dump(key), canonical_dump(value[key])))
        return b"".join(parts)
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def result_signature(root: Path, summary: dict) -> dict:
    expected = summary.get("completed")
    if not isinstance(expected, int) or expected <= 0:
        raise ValueError("gpp summary has no completed jobs")
    for field in ("reused", "failed", "cancelled", "not_started"):
        if summary.get(field) != 0:
            raise ValueError(f"gpp summary {field} is not zero")
    if summary.get("deadline_reached") is not False:
        raise ValueError("unexpected deadline during an unrestricted comparison")
    rows = []
    runs = root / "runs"
    if not runs.is_dir():
        raise ValueError("gpp did not create runs/")
    for path in sorted(runs.glob("*/seed_*.json")):
        match = SEED_FILE.fullmatch(path.name)
        if not match:
            raise ValueError(f"unexpected result filename: {path}")
        condition_id = path.parent.name
        if len(condition_id) != 64 or any(char not in "0123456789abcdef" for char in condition_id):
            raise ValueError(f"invalid condition directory: {path.parent}")
        seed = int(match.group(1))
        value = normalize_result(json.loads(path.read_text(encoding="utf-8")))
        rows.append((condition_id, seed, value, path))
    keys = [(condition_id, seed) for condition_id, seed, _, _ in rows]
    if len(rows) != expected:
        raise ValueError(f"saved result count {len(rows)} differs from completed {expected}")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate condition/seed result")
    incomplete = list(runs.glob("*/seed_*.incomplete.json"))
    if incomplete:
        raise ValueError(f"incomplete result marker remains: {incomplete[0]}")
    digest = hashlib.sha256()
    digest.update(len(rows).to_bytes(8, "little"))
    run_hashes = []
    for condition_id, seed, value, path in sorted(rows):
        encoded = canonical_dump(value)
        result_hash = hashlib.sha256(encoded).hexdigest()
        digest.update(condition_id.encode("ascii"))
        digest.update(seed.to_bytes(8, "little"))
        digest.update(encoded)
        run_hashes.append({"condition_id": condition_id, "seed": seed,
                           "result_sha256": result_hash, "path": str(path.relative_to(root))})
    return {"completed": len(rows), "signature": digest.hexdigest(), "runs": run_hashes}


def parse_summary(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("gpp --json must emit exactly one summary line")
    summary = json.loads(lines[0])
    if not isinstance(summary, dict):
        raise ValueError("gpp summary is not a JSON object")
    return summary


def invoke(binary: Path, spec: Path, threads: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="gpp-performance-cli-") as temporary:
        root = Path(temporary) / "root"
        command = [str(binary), "run", str(spec), "--root", str(root), "--threads", str(threads),
                   "--rounds", "--json"]
        started = time.monotonic()
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        elapsed_ns = int((time.monotonic() - started) * 1_000_000_000)
        record = {"command": command, "returncode": completed.returncode,
                  "elapsed_ns": elapsed_ns, "stdout": completed.stdout, "stderr": completed.stderr}
        if completed.returncode:
            raise RuntimeError(f"gpp run failed:\n{completed.stderr}\n{completed.stdout}")
        summary = parse_summary(completed.stdout)
        record["summary"] = summary
        record["result"] = result_signature(root, summary)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-binary", type=Path, required=True)
    parser.add_argument("--after-binary", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True,
                        help="for example cases/batch-A-cold-flip.json or batch-A-eo-flat.json")
    parser.add_argument("--outprefix", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2, help="ABBA/BAAB pairs")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.repeats <= 0 or args.threads <= 0:
        raise SystemExit("--repeats and --threads must be positive")
    before = args.before_binary.resolve(strict=True)
    after = args.after_binary.resolve(strict=True)
    spec = args.spec.resolve(strict=True)
    output = Path(f"{args.outprefix}.json")
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = {"before": [], "after": []}
    warmups = {}
    for label, binary in (("before", before), ("after", after)):
        warmups[label] = invoke(binary, spec, args.threads)
    for pair in range(args.repeats):
        order = [("before", before), ("after", after)]
        if pair % 2:
            order.reverse()
        order = order + list(reversed(order))  # ABBA, then BAAB.
        for label, binary in order:
            record = invoke(binary, spec, args.threads)
            record["pair"] = pair
            record["sample"] = len(samples[label])
            samples[label].append(record)
    signatures = {label: {sample["result"]["signature"] for sample in values}
                  for label, values in samples.items()}
    signatures["before"].add(warmups["before"]["result"]["signature"])
    signatures["after"].add(warmups["after"]["result"]["signature"])
    if any(len(values) != 1 for values in signatures.values()) or signatures["before"] != signatures["after"]:
        raise RuntimeError(f"normalized result mismatch: {signatures}")
    before_times = [sample["elapsed_ns"] for sample in samples["before"]]
    after_times = [sample["elapsed_ns"] for sample in samples["after"]]
    payload = {
        "schema_version": 1,
        "full_spec": str(spec),
        "spec_sha256": sha256(spec),
        "binaries": {
            "before": {"path": str(before), "sha256": sha256(before)},
            "after": {"path": str(after), "sha256": sha256(after)},
        },
        "threads": args.threads,
        "repeats": args.repeats,
        "warmups": warmups,
        "samples": samples,
        "result_signature": next(iter(signatures["before"])),
        "timing": {
            "before_median_ns": statistics.median(before_times),
            "after_median_ns": statistics.median(after_times),
            "speedup_before_div_after": None,
        },
        "signature_algorithm": (
            "SHA-256 over completed-count, sorted condition_id/seed, and every normalized result; "
            "normalization removes only attempt_id, elapsed_ms, diagnostics.search_ms, and diagnostics.measurement_ms; "
            "canonical JSON encodes sorted keys and f64 IEEE-754 bits."
        ),
    }
    payload["timing"]["speedup_before_div_after"] = (
        payload["timing"]["before_median_ns"] / payload["timing"]["after_median_ns"]
        if payload["timing"]["after_median_ns"] else None
    )
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "signature": payload["result_signature"],
                      "speedup": payload["timing"]["speedup_before_div_after"]}))


if __name__ == "__main__":
    main()
