#!/usr/bin/env python3
"""Build, train, and rebuild gpp_utils with rustc profile-guided optimization.

The workflow follows the rustc book's Cargo guidance: both builds use the same
explicit host target, absolute profile paths, and isolated target directories.
Ambient Rust flags are rejected so the recorded flags are the complete flags.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_inputs(source: Path) -> list[Path]:
    roots = [
        "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "build.rs", ".cargo", "src",
        "examples",
    ]
    files: list[Path] = []
    for name in roots:
        path = source / name
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
    return sorted(set(files), key=lambda path: path.relative_to(source).as_posix())


def tree_digest(source: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def selected_cases(cases: Path, pattern: str | None) -> list[dict]:
    manifest_path = cases / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expression = re.compile(pattern) if pattern else None
    selected = [
        case
        for case in manifest
        if not case["name"].startswith("grid-")
        and (expression is None or expression.search(case["name"]))
    ]
    if not selected:
        raise ValueError("selection matched no non-grid recovery cases")
    if pattern is None and len(selected) != 15:
        raise ValueError(f"expected 15 default recovery cases, found {len(selected)}")
    return selected


class Recorder:
    def __init__(self, source: Path, work: Path, metadata: dict):
        self.source = source
        self.work = work
        self.metadata = metadata
        self.log = work / "pgo.log"
        self.meta = work / "metadata.json"
        self._save()

    def _save(self) -> None:
        self.meta.write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")

    def command(
        self,
        label: str,
        command: list[str],
        *,
        env_overrides: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("RUSTFLAGS", None)
        environment.pop("CARGO_ENCODED_RUSTFLAGS", None)
        environment.update(env_overrides or {})
        started = time.time()
        completed = subprocess.run(
            command,
            cwd=cwd or self.source,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        record = {
            "label": label,
            "command": command,
            "cwd": str((cwd or self.source).resolve()),
            "environment": env_overrides or {},
            "started_unix": started,
            "seconds": time.time() - started,
            "returncode": completed.returncode,
        }
        self.metadata["commands"].append(record)
        self._save()
        with self.log.open("a", encoding="utf-8") as output:
            output.write(f"\n=== {label} ===\n")
            output.write(f"command: {json.dumps(command)}\n")
            output.write(completed.stdout)
            output.write(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{label} failed; see {self.log}")
        return completed


def tool_output(command: list[str], source: Path) -> str:
    return subprocess.run(
        command,
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="frozen source directory")
    parser.add_argument("--work", type=Path, required=True, help="new output directory")
    parser.add_argument("--cases", type=Path, required=True, help="recovery case directory")
    parser.add_argument("--select", help="optional regex selecting non-grid case names")
    parser.add_argument(
        "--base-rustflags",
        default="",
        help="explicit flags prepended identically to generate/use builds",
    )
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    cases = args.cases.resolve(strict=True)
    compare = Path(__file__).resolve().with_name("performance_compare.py")
    if not compare.is_file():
        raise FileNotFoundError(f"missing comparison driver: {compare}")
    work = args.work.resolve()
    if work.exists():
        raise FileExistsError(f"refusing existing --work directory: {work}")
    if any(character.isspace() for character in str(work)):
        raise ValueError("--work must not contain whitespace because rustc profile paths use RUSTFLAGS")
    if not (source / "Cargo.lock").is_file():
        raise FileNotFoundError(f"missing frozen Cargo.lock under {source}")
    if "profile-generate" in args.base_rustflags or "profile-use" in args.base_rustflags:
        raise ValueError("--base-rustflags must not contain PGO profile flags")
    ambient = {
        key: value
        for key, value in os.environ.items()
        if key in {"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"}
        or (key.startswith("CARGO_TARGET_") and key.endswith("_RUSTFLAGS"))
    }
    if ambient:
        names = ", ".join(sorted(ambient))
        raise RuntimeError(
            f"ambient Rust flags are not reproducible ({names}); unset them and use "
            "--base-rustflags explicitly"
        )

    chosen = selected_cases(cases, args.select)
    inputs = source_inputs(source)
    rustc_verbose = tool_output(["rustc", "-vV"], source)
    host_line = next(line for line in rustc_verbose.splitlines() if line.startswith("host: "))
    target = host_line.removeprefix("host: ").strip()
    sysroot = Path(tool_output(["rustc", "--print", "sysroot"], source)).resolve()
    executable = ".exe" if os.name == "nt" else ""
    llvm_candidates = [
        sysroot / "lib" / "rustlib" / target / "bin" / f"llvm-profdata{executable}",
        sysroot / "bin" / f"llvm-profdata{executable}",
    ]
    llvm_profdata = next((path for path in llvm_candidates if path.is_file()), None)
    if llvm_profdata is None:
        locations = ", ".join(str(path) for path in llvm_candidates)
        raise FileNotFoundError(
            "llvm-profdata was not found (install llvm-tools-preview for this toolchain); "
            f"checked {locations}"
        )

    work.mkdir(parents=True)
    profile_dir = (work / "profiles").resolve()
    generate_target = (work / "generate-target").resolve()
    use_target = (work / "use-target").resolve()
    profile_dir.mkdir()
    source_hashes = {
        path.relative_to(source).as_posix(): sha256(path)
        for path in inputs
    }
    case_hashes = {"manifest.json": sha256(cases / "manifest.json")}
    for case in chosen:
        relative = Path(case["spec"])
        case_hashes[relative.as_posix()] = sha256(cases / relative)
    metadata = {
        "schema_version": 1,
        "source": str(source),
        "work": str(work),
        "cases": str(cases),
        "selection_regex": args.select,
        "selected_cases": [case["name"] for case in chosen],
        "target_triple": target,
        "rustc_verbose": rustc_verbose,
        "cargo_version_verbose": tool_output(["cargo", "-vV"], source),
        "sysroot": str(sysroot),
        "llvm_profdata": str(llvm_profdata),
        "llvm_profdata_version": tool_output([str(llvm_profdata), "--version"], source),
        "base_rustflags": args.base_rustflags,
        "source_tree_sha256": tree_digest(source, inputs),
        "source_inputs_sha256": source_hashes,
        "case_inputs_sha256": case_hashes,
        "comparison_driver": str(compare),
        "comparison_driver_sha256": sha256(compare),
        "pgo_driver": str(Path(__file__).resolve()),
        "pgo_driver_sha256": sha256(Path(__file__).resolve()),
        "commands": [],
    }
    recorder = Recorder(source, work, metadata)

    def flags(pgo: str) -> str:
        return " ".join(part for part in [args.base_rustflags.strip(), pgo] if part)

    build_common = [
        "cargo", "build", "--release", "--locked", "--bin", "gpp",
        "--example", "bench_recovery", "--example", "bench_batch", "--target", target,
    ]
    normal_target = (work / "normal-target").resolve()
    recorder.command(
        "build-normal",
        build_common + ["--target-dir", str(normal_target)],
        env_overrides={"RUSTFLAGS": args.base_rustflags.strip()},
    )
    normal_binaries = [
        normal_target / target / "release" / f"gpp{executable}",
        normal_target / target / "release" / "examples" / f"bench_recovery{executable}",
        normal_target / target / "release" / "examples" / f"bench_batch{executable}",
    ]
    missing = [str(path) for path in normal_binaries if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"normal binaries missing: {', '.join(missing)}")
    metadata["normal_binaries_sha256"] = {
        str(path): sha256(path) for path in normal_binaries
    }
    recorder._save()
    generate_flags = flags(f"-Cprofile-generate={profile_dir}")
    recorder.command(
        "build-generate",
        build_common + ["--target-dir", str(generate_target)],
        env_overrides={"RUSTFLAGS": generate_flags},
    )
    instrumented = [
        generate_target / target / "release" / f"gpp{executable}",
        generate_target / target / "release" / "examples" / f"bench_recovery{executable}",
        generate_target / target / "release" / "examples" / f"bench_batch{executable}",
    ]
    missing = [str(path) for path in instrumented if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"instrumented binaries missing: {', '.join(missing)}")
    metadata["instrumented_binaries_sha256"] = {
        str(path): sha256(path) for path in instrumented
    }
    recorder._save()
    bench = instrumented[1]
    training = [
        sys.executable, str(compare), "run", "--binary", str(bench), "--tag", "pgo-train",
        "--cases", str(cases), "--out", str(work / "training.jsonl"), "--mode", "run",
        "--repeats", "1", "--warmup", "0",
    ]
    if args.select:
        training += ["--select", args.select]
    recorder.command(
        "train-recovery",
        training,
        env_overrides={"LLVM_PROFILE_FILE": str(profile_dir / "%p.profraw")},
    )
    raw_profiles = sorted(profile_dir.glob("*.profraw"))
    if not raw_profiles:
        raise RuntimeError(f"training produced no .profraw files under {profile_dir}")
    metadata["profile_raw_sha256"] = {
        path.name: sha256(path) for path in raw_profiles
    }
    training_sidecar = (work / "training.commands.json").resolve()
    if not training_sidecar.is_file():
        raise FileNotFoundError(f"training command sidecar missing: {training_sidecar}")
    metadata["training_sidecar_sha256"] = sha256(training_sidecar)
    recorder._save()
    profdata = (work / "merged.profdata").resolve()
    recorder.command(
        "merge-profile",
        [str(llvm_profdata), "merge", "-o", str(profdata), *map(str, raw_profiles)],
    )
    metadata["profile_data_sha256"] = sha256(profdata)
    recorder._save()
    use_flags = flags(
        f"-Cprofile-use={profdata} -Cllvm-args=-pgo-warn-missing-function"
    )
    recorder.command(
        "build-use",
        build_common + ["--target-dir", str(use_target)],
        env_overrides={"RUSTFLAGS": use_flags},
    )
    optimized = [
        use_target / target / "release" / f"gpp{executable}",
        use_target / target / "release" / "examples" / f"bench_recovery{executable}",
        use_target / target / "release" / "examples" / f"bench_batch{executable}",
    ]
    missing = [str(path) for path in optimized if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"optimized binaries missing: {', '.join(missing)}")
    metadata["optimized_binaries_sha256"] = {
        str(path): sha256(path) for path in optimized
    }
    metadata["completed_unix"] = time.time()
    recorder._save()
    print(recorder.meta)


if __name__ == "__main__":
    main()
