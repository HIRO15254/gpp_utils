#!/usr/bin/env python3
"""Run the repository's required, cross-platform Rust verification commands."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


COMMANDS = (
    ("cargo", "fmt", "--all", "--", "--check"),
    ("cargo", "clippy", "--all-targets", "--all-features", "--locked", "--", "-D", "warnings"),
    ("cargo", "test", "--all-targets", "--locked"),
    ("cargo", "test", "--doc", "--locked"),
    ("cargo", "test", "--release", "--locked", "--lib", "solvers::engine::exact_tests"),
    ("cargo", "build", "--release", "--all-targets", "--locked"),
    ("cargo", "doc", "--no-deps", "--locked"),
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    for command in COMMANDS:
        print("+", " ".join(command), flush=True)
        environment = None
        if command[1] == "doc":
            environment = os.environ.copy()
            existing = environment.get("RUSTDOCFLAGS", "").strip()
            environment["RUSTDOCFLAGS"] = f"{existing} -D warnings".strip()
        completed = subprocess.run(command, check=False, cwd=root, env=environment)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
