#!/usr/bin/env python3
"""Run recovery comparison binaries sequentially and verify their signatures."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_cases(args):
    cases = json.loads((args.cases / "manifest.json").read_text(encoding="utf-8"))
    cases = [case for case in cases if (args.extended or not case["name"].startswith("grid-"))
             and (not args.select or re.search(args.select, case["name"]))]
    if not cases:
        raise ValueError("selection matched no cases")
    return cases


def command_for(binary, spec, args, *, repeats, warmup, fixture=None):
    command = [str(binary), "--spec", str(spec), "--mode", args.mode,
               "--repeats", str(repeats), "--warmup", str(warmup),
               "--iterations", str(args.iterations), "--threads", str(args.threads)]
    if args.phase:
        command += ["--phase", args.phase]
    if fixture:
        command += ["--fixture", str(fixture)]
    return command


def invoke(binary, spec, case, args, *, repeats, warmup, fixture=None):
    command = command_for(binary, spec, args, repeats=repeats, warmup=warmup, fixture=fixture)
    before = digest(fixture) if fixture and fixture.exists() else None
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    after = digest(fixture) if fixture and fixture.exists() else None
    record = {"case": case["name"], "command": command, "returncode": completed.returncode,
              "seconds": time.monotonic() - started, "stderr": completed.stderr,
              "spec_sha256": digest(spec)}
    if fixture:
        record.update({"fixture": str(fixture), "fixture_sha256_before": before,
                       "fixture_sha256_after": after})
    if completed.returncode:
        raise RuntimeError(f"{case['name']} failed:\n{completed.stderr}\n{completed.stdout}")
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()], record


def write_sidecar(out, binary, commands):
    out.with_suffix(".commands.json").write_text(
        json.dumps({"binary_sha256": digest(binary), "commands": commands}, indent=2),
        encoding="utf-8")


def run(args):
    binary = args.binary.resolve()
    cases = selected_cases(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.fixtures:
        args.fixtures.mkdir(parents=True, exist_ok=True)
    commands = []
    with args.out.open("w", encoding="utf-8") as output:
        for i, case in enumerate(cases, 1):
            spec = (args.cases / case["spec"]).resolve()
            fixture = ((args.fixtures / f"{case['name']}.json").resolve()
                       if args.fixtures else None)
            rows, record = invoke(binary, spec, case, args, repeats=args.repeats,
                                  warmup=args.warmup, fixture=fixture)
            commands.append(record)
            write_sidecar(args.out, binary, commands)
            for row in rows:
                row["binary"] = args.tag
                output.write(json.dumps(row) + "\n")
            output.flush()
            elapsed = sum(row["elapsed_ns"] for row in rows) / 1e9
            print(f"[{i}/{len(cases)}] {case['name']}: {elapsed:.3f}s measured", flush=True)


def load_rows(path):
    grouped = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["case"], row["phase"], row["iterations"])
        rep = row["repeat"]
        if rep in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate repeat {rep} for {key} in {path}")
        grouped[key][rep] = row
    return grouped


def sidecar_specs(path):
    sidecar = path.with_suffix(".commands.json")
    if not sidecar.is_file():
        raise ValueError(f"missing command sidecar: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    specs = {}
    for record in payload.get("commands", []):
        case, spec = record.get("case"), record.get("spec_sha256")
        if not case:
            command = record.get("command")
            if not isinstance(command, list):
                raise ValueError(f"sidecar lacks a command list: {sidecar}")
            try:
                index = command.index("--spec")
                spec_path = command[index + 1]
            except (ValueError, IndexError):
                raise ValueError(f"sidecar command lacks a --spec path: {sidecar}") from None
            if not isinstance(spec_path, str) or not spec_path:
                raise ValueError(f"sidecar has malformed --spec path: {sidecar}")
            case = Path(spec_path).stem
        if not isinstance(case, str) or not case or not isinstance(spec, str) or not spec:
            raise ValueError(f"sidecar lacks case/spec hash: {sidecar}")
        if record.get("returncode") != 0:
            raise ValueError(f"sidecar records failed command for {case}: {sidecar}")
        previous = specs.setdefault(case, spec)
        if previous != spec:
            raise ValueError(f"conflicting spec hashes for {case}: {sidecar}")
    if not specs:
        raise ValueError(f"sidecar has no command records: {sidecar}")
    return specs


def summarize_paths(before_path, after_path):
    before, after = load_rows(before_path), load_rows(after_path)
    if set(before) != set(after):
        raise ValueError(f"case mismatch: missing={set(before)-set(after)}, extra={set(after)-set(before)}")
    before_specs, after_specs = sidecar_specs(before_path), sidecar_specs(after_path)
    output_cases = {key[0] for key in before}
    if output_cases != set(before_specs) or output_cases != set(after_specs):
        raise ValueError("output cases and command-sidecar cases differ")
    if before_specs != after_specs:
        raise ValueError("before/after spec SHA-256 mappings differ")
    summary = []
    for key, left in before.items():
        right = after[key]
        if set(left) != set(right):
            raise ValueError(f"repeat mismatch {key}: before={set(left)}, after={set(right)}")
        signatures = {row["signature"] for row in [*left.values(), *right.values()]}
        if len(signatures) != 1:
            raise ValueError(f"RESULT MISMATCH {key}: {signatures}")
        base = statistics.median(row["elapsed_ns"] for row in left.values())
        candidate = statistics.median(row["elapsed_ns"] for row in right.values())
        summary.append({"case": key[0], "phase": key[1], "iterations": key[2],
                        "before_ns": base, "after_ns": candidate,
                        "speedup": base / candidate if candidate else None,
                        "signature": next(iter(signatures)), "matched": True,
                        "before_samples": len(left), "after_samples": len(right)})
    return summary


def summarize(args):
    summary = summarize_paths(args.before, args.after)
    text = json.dumps(summary, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)


def paired(args):
    if args.repeats <= 0 or args.iterations <= 0 or args.threads <= 0 or args.warmup < 0:
        raise ValueError("paired repeats, iterations, and threads must be positive; warmup must be non-negative")
    if args.mode == "io":
        if not args.fixtures or not args.fixtures.is_dir():
            raise ValueError("paired io mode requires an existing --fixtures directory")
    elif args.fixtures:
        raise ValueError("--fixtures is only permitted for paired io mode")
    cases = selected_cases(args)
    before_binary, after_binary = args.before_binary.resolve(), args.after_binary.resolve()
    before_out = args.out.with_name(f"{args.out.name}.before.jsonl")
    after_out = args.out.with_name(f"{args.out.name}.after.jsonl")
    summary_out = args.out.with_name(f"{args.out.name}.summary.json")
    outputs = [before_out, after_out, summary_out,
               before_out.with_suffix(".commands.json"), after_out.with_suffix(".commands.json")]
    if any(path.exists() for path in outputs):
        raise ValueError("paired output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    commands = {"before": [], "after": []}
    rows = {"before": [], "after": []}
    repeats = {"before": 0, "after": 0}
    for i, case in enumerate(cases, 1):
        spec = (args.cases / case["spec"]).resolve()
        fixture = ((args.fixtures / f"{case['name']}.json").resolve()
                   if args.fixtures else None)
        fixture_sha256 = digest(fixture) if fixture else None

        def require_unchanged_fixture(record):
            if fixture and (record["fixture_sha256_before"] != fixture_sha256
                            or record["fixture_sha256_after"] != fixture_sha256):
                raise ValueError(f"fixture changed or differed between binaries: {fixture}")
        # Keep warm-up outside the paired samples so every emitted invocation has
        # exactly one measured iteration and its position is explicit.
        for label, binary in [("before", before_binary), ("after", after_binary)]:
            _, record = invoke(binary, spec, case, args, repeats=1, warmup=args.warmup,
                               fixture=fixture)
            require_unchanged_fixture(record)
            record["kind"] = "warmup"
            commands[label].append(record)
        for pair in range(args.repeats):
            order = [("before", before_binary), ("after", after_binary)]
            if pair % 2:
                order.reverse()
            # ABBA on even pairs, BAAB on odd pairs.
            order = order + list(reversed(order))
            for label, binary in order:
                emitted, record = invoke(binary, spec, case, args, repeats=1, warmup=0,
                                         fixture=fixture)
                require_unchanged_fixture(record)
                record.update({"kind": "sample", "pair": pair})
                commands[label].append(record)
                for row in emitted:
                    row["binary"] = label
                    row["repeat"] = repeats[label]
                    rows[label].append(row)
                repeats[label] += 1
        print(f"[{i}/{len(cases)}] {case['name']}: {args.repeats * 2} paired samples/binary",
              flush=True)
    for label, out, binary in [("before", before_out, before_binary),
                               ("after", after_out, after_binary)]:
        with out.open("w", encoding="utf-8") as output:
            for row in rows[label]:
                output.write(json.dumps(row) + "\n")
        write_sidecar(out, binary, commands[label])
    summary_out.write_text(json.dumps(summarize_paths(before_out, after_out), indent=2) + "\n",
                           encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    launch = sub.add_parser("run")
    launch.add_argument("--binary", type=Path, required=True)
    launch.add_argument("--tag", required=True)
    launch.add_argument("--cases", type=Path, required=True)
    launch.add_argument("--out", type=Path, required=True)
    launch.add_argument("--mode", choices=["run", "kernel", "io"], default="run")
    launch.add_argument("--phase")
    launch.add_argument("--select")
    launch.add_argument("--extended", action="store_true")
    launch.add_argument("--repeats", type=int, default=3)
    launch.add_argument("--warmup", type=int, default=1)
    launch.add_argument("--iterations", type=int, default=1000)
    launch.add_argument("--threads", type=int, default=1)
    launch.add_argument("--fixtures", type=Path)
    compare = sub.add_parser("summarize")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--out", type=Path)
    paired_run = sub.add_parser("paired")
    paired_run.add_argument("--before-binary", type=Path, required=True)
    paired_run.add_argument("--after-binary", type=Path, required=True)
    paired_run.add_argument("--cases", type=Path, required=True)
    paired_run.add_argument("--out", type=Path, required=True, help="output filename prefix")
    paired_run.add_argument("--mode", choices=["run", "kernel", "io"], default="run")
    paired_run.add_argument("--phase")
    paired_run.add_argument("--select")
    paired_run.add_argument("--extended", action="store_true")
    paired_run.add_argument("--repeats", type=int, default=5, help="ABBA/BAAB pairs per case")
    paired_run.add_argument("--warmup", type=int, default=1)
    paired_run.add_argument("--iterations", type=int, default=1000)
    paired_run.add_argument("--threads", type=int, default=1)
    paired_run.add_argument("--fixtures", type=Path)
    args = parser.parse_args()
    {"run": run, "summarize": summarize, "paired": paired}[args.action](args)


if __name__ == "__main__":
    main()
