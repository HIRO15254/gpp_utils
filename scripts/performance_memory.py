#!/usr/bin/env python3
"""Record Windows child-process memory peaks without using its timing as a CPU benchmark.

Pass the measured command after ``--``. This tool runs that command serially
with ``shell=False`` and writes captured stdout/stderr plus memory counters to
one JSON file. Its polling overhead means it must not be used as a source of
CPU-time measurements for the child.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


class ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("page_fault_count", wintypes.DWORD),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
        ("private_usage", ctypes.c_size_t),
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def executable(command: list[str]) -> Path:
    candidate = Path(command[0])
    if candidate.is_file():
        return candidate.resolve()
    found = shutil.which(command[0])
    if not found:
        raise FileNotFoundError(f"cannot find child executable: {command[0]}")
    return Path(found).resolve()


def memory_reader():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.K32GetProcessMemoryInfo
    function.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    function.restype = wintypes.BOOL

    def read(handle: int) -> ProcessMemoryCountersEx:
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not function(wintypes.HANDLE(handle), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters

    return read


def measure(command: list[str], interval_seconds: float, read_memory) -> dict:
    with tempfile.TemporaryDirectory(prefix="gpp-performance-memory-") as temporary:
        directory = Path(temporary)
        stdout_path = directory / "stdout.txt"
        stderr_path = directory / "stderr.txt"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            child = subprocess.Popen(command, shell=False, stdout=stdout, stderr=stderr)
            samples = []

            def sample() -> None:
                counters = read_memory(child._handle)
                samples.append({
                    "peak_working_set_bytes": counters.peak_working_set_size,
                    "peak_pagefile_usage_bytes": counters.peak_pagefile_usage,
                    "private_usage_bytes": counters.private_usage,
                })

            try:
                sample()
                while child.poll() is None:
                    time.sleep(interval_seconds)
                    sample()
                sample()
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
                raise
        return {
            "returncode": child.returncode,
            "stdout": stdout_path.read_text(encoding="utf-8", errors="replace"),
            "stderr": stderr_path.read_text(encoding="utf-8", errors="replace"),
            "memory": {
                "peak_working_set_bytes": max(item["peak_working_set_bytes"] for item in samples),
                "peak_pagefile_usage_bytes": max(item["peak_pagefile_usage_bytes"] for item in samples),
                "max_observed_private_usage_bytes": max(item["private_usage_bytes"] for item in samples),
                "sample_count": len(samples),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--interval-ms", type=float, default=5.0,
                        help="memory polling interval, from 5 to 10 milliseconds")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="child command; put it after --")
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if os.name != "nt":
        raise SystemExit("performance_memory.py requires Windows")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if not 5.0 <= args.interval_ms <= 10.0:
        raise SystemExit("--interval-ms must be between 5 and 10")
    if not args.command:
        raise SystemExit("a child command is required after --")
    binary = executable(args.command)
    read_memory = memory_reader()
    results = [measure(args.command, args.interval_ms / 1000.0, read_memory)
               for _ in range(args.repeats)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema_version": 1,
        "platform": "Windows",
        "full_command": args.command,
        "binary": {"path": str(binary), "sha256": sha256(binary)},
        "repeats": results,
        "measurement_note": (
            "These are memory-polling measurements. Do not use this tool's polling "
            "or wall-clock behavior as the measured command's CPU-time result."
        ),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
