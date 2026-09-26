#!/usr/bin/env python3
"""Freeze or build isolated performance candidates without changing the checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRAS = ["examples/bench_recovery.rs", "examples/bench_batch.rs", "examples/verify_performance_cases.rs",
          "src/smoothing/test_reference.rs", "src/experiment/performance_probe.rs"]


def tracked():
    return subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")[:-1]


def copy_files(source, destination):
    paths = set(tracked()) | set(EXTRAS)
    for relative in paths:
        src, dst = source / relative, destination / relative
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Cargo's source freshness check uses timestamps. Keep the write's
            # current mtime when switching variants in the same build directory.
            shutil.copyfile(src, dst)
        elif dst.is_file():
            dst.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["snapshot", "build"])
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--kind", choices=["recovery", "batch", "all"], default="recovery")
    parser.add_argument("--worker", default="", help="isolated build directory for parallel builds")
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    if args.action == "snapshot":
        target = work / args.tag
        if target.exists():
            raise ValueError(f"snapshot already exists: {target}")
        copy_files(source, target)
        return
    build_root = work / args.worker if args.worker else work
    target = build_root / "build-source"
    copy_files(source, target)
    for overlay in args.overlay:
        for path in overlay.rglob("*.rs"):
            relative = path.relative_to(overlay)
            # Solver snapshots intentionally contain just the owned filename.
            if len(relative.parts) == 1:
                relative = Path("src/solvers") / relative
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    output = work / "bin" / args.tag
    output.mkdir(parents=True, exist_ok=True)
    command = ["cargo", "build", "--release", "--locked"]
    if args.kind in ("recovery", "all"):
        command += ["--example", "bench_recovery"]
    if args.kind in ("batch", "all"):
        command += ["--example", "bench_batch"]
    command += ["--target-dir", str(build_root / "build-target")]
    with (output / "build.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=target, stdout=log, stderr=subprocess.STDOUT, check=True)
    built_names = {"recovery": ["bench_recovery"], "batch": ["bench_batch"],
                   "all": ["bench_recovery", "bench_batch"]}[args.kind]
    latest_input = max(path.stat().st_mtime_ns for path in target.rglob("*.rs"))
    for name in built_names:
        binary = build_root / "build-target/release/examples" / f"{name}.exe"
        if binary.exists():
            if binary.stat().st_mtime_ns < latest_input:
                raise RuntimeError(f"Cargo did not refresh the candidate binary: {binary}")
            shutil.copy2(binary, output / binary.name)
    source_hashes = {str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in target.rglob("*.rs")}
    (output / "build.json").write_text(json.dumps({"command": command,
        "base_source": str(source),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "overlays": [str(path.resolve()) for path in args.overlay],
        "source_sha256": source_hashes}, indent=2), encoding="utf-8")
    if args.probe:
        command = ["cargo", "test", "--release", "--locked", "--lib", "--no-run",
                   "--message-format=json", "--target-dir", str(build_root / "build-target")]
        with (output / "probe-build.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=target, text=True, stdout=subprocess.PIPE,
                                       stderr=log, check=True)
        for line in completed.stdout.splitlines():
            item = json.loads(line)
            if item.get("reason") == "compiler-artifact" and item.get("executable"):
                shutil.copy2(item["executable"], output / "performance_probe.exe")
    print(output, flush=True)


if __name__ == "__main__":
    main()
