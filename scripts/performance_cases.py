#!/usr/bin/env python3
"""Select unchanged recovery-v1 conditions for implementation benchmarks.

Scientific conditions (including float bit patterns, budget and measurements)
come from the checked-in generator output. Kernel benchmarks repeatedly evaluate
states from those runs; their repetition counts are not experiment step budgets.
This script never writes to the production data root.
"""

from __future__ import annotations

import argparse
import copy
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "A": "baseline_v1.toml",
    "B": "recovery_v1_B_smoothing.toml",
    "C": "recovery_v1_C_fitness.toml",
}


def select(stage, graph, neighborhood, solver_kind, parameter_index, extra, seed):
    source = ROOT / "experiments_sa_eo" / SOURCES[stage]
    spec = tomllib.loads(source.read_text(encoding="utf-8"))
    kind, nodes, degree = graph
    original_graph = next(g for g in spec["graphs"] if g["kind"] == kind)
    assert nodes in original_graph["node_counts"]
    assert degree in original_graph["expected_degrees"]
    assert seed in spec["run_seeds"] and neighborhood in spec["neighborhoods"]
    spec["graphs"] = [{"kind": kind, "node_counts": [nodes],
                       "expected_degrees": [degree], "seeds": original_graph["seeds"]}]
    spec["run_seeds"] = [seed]
    spec["neighborhoods"] = [neighborhood]
    solver = copy.deepcopy(next(s for s in spec["solvers"] if s["kind"] == solver_kind))
    axis = "temperatures" if solver_kind == "sa" else "taus"
    solver[axis] = [solver[axis][parameter_index]]
    if solver_kind == "sa":
        smoothing_kind, k = extra
        smoothing = copy.deepcopy(next(s for s in solver["smoothing"] if s["kind"] == smoothing_kind))
        if k is not None:
            assert k in smoothing["ks"]
            smoothing["ks"] = [k]
        solver["smoothing"] = [smoothing]
    else:
        solver["fitnesses"] = [solver["fitnesses"][extra]]
    spec["solvers"] = [solver]
    spec["name"] = "performance-recovery-subset"
    return spec


def cases(extended=False):
    # Temperature/tau are selected by position, never reconstructed numerically.
    definitions = [
        ("A-cold-flip", "A", ("random", 124, 5.0), "flip", "sa", 0, ("none", None), 0),
        ("A-hot-swap", "A", ("random", 500, 20.0), "swap", "sa", 80, ("none", None), 0),
        ("A-mid-flip", "A", ("geometric", 500, 5.0), "flip", "sa", 30, ("none", None), 1),
        ("A-mid-swap", "A", ("geometric", 250, 10.0), "swap", "sa", 30, ("none", None), 1),
        ("A-eo-flip", "A", ("random", 500, 20.0), "flip", "eo", 30, 0, 0),
        ("A-eo-swap", "A", ("geometric", 500, 5.0), "swap", "eo", 30, 0, 0),
        ("A-eo-flat", "A", ("random", 124, 5.0), "swap", "eo", 0, 0, 1),
        ("A-eo-greedy", "A", ("geometric", 250, 10.0), "flip", "eo", 60, 0, 1),
        ("B-random-k1", "B", ("random", 124, 5.0), "flip", "sa", 30, ("random_k_average", 1), 0),
        ("B-random-k32", "B", ("geometric", 250, 10.0), "flip", "sa", 0, ("random_k_average", 32), 1),
        ("B-all-average", "B", ("random", 500, 20.0), "flip", "sa", 80, ("all_average", None), 0),
        ("C-mul-zero", "C", ("random", 500, 20.0), "flip", "eo", 30, 1, 0),
        ("C-mul-half", "C", ("geometric", 500, 5.0), "flip", "eo", 60, 6, 1),
        ("C-add-zero", "C", ("random", 124, 5.0), "flip", "eo", 0, 11, 0),
        ("C-add-high", "C", ("geometric", 250, 10.0), "flip", "eo", 30, 23, 1),
    ]
    if extended:
        for kind in ["random", "geometric"]:
            for n in [124, 250, 500]:
                for d in [5.0, 10.0, 20.0]:
                    for neighborhood in ["flip", "swap"]:
                        for solver in ["sa", "eo"]:
                            name = f"grid-{kind}-{n}-{d:g}-{neighborhood}-{solver}"
                            extra = ("none", None) if solver == "sa" else 0
                            definitions.append((name, "A", (kind, n, d), neighborhood,
                                                solver, 30, extra, 0))
    return [(name, stage, select(stage, graph, neighborhood, solver, index, extra, seed))
            for name, stage, graph, neighborhood, solver, index, extra, seed in definitions]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, stage, spec in cases(args.extended):
        path = args.out / f"{name}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        manifest.append({"name": name, "stage": stage, "source": SOURCES[stage], "spec": path.name})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} unchanged recovery conditions to {args.out}")


if __name__ == "__main__":
    main()
