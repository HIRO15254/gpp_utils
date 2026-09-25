"""旧実験履歴（data/graphs, data/results_sa_eo）を新形式 data/v1 へ変換する。

背景
----
2026-09-16 に取り込んだ上流リファクタ（51577f9..7b37ef1）で保存形式が変わった。

旧形式:
    data/results_sa_eo/<graph_id>/<config_name>/seed_<s>.json
    （graph_spec + config + seed + final_partition + records を各 run に複製）

新形式:
    data/v1/graphs/<graph_id>.json
    data/v1/batches/<batch_id>/experiment.json
    data/v1/runs/<condition_id>/seed_<s>.json
    （条件は experiment.json に 1 回だけ。run は分割プール + その参照のみ）

変換できるもの・できないものの判定根拠は docs/convert_v1.md に記録した。
本スクリプトは「できるもの」だけを実行する。

2026-09-25 にベースラインを再定義した（Theta -1.50..+2.50、tau 0..3.00、各組 s0 のみ、10^6）。
新しいベースラインの仕様 baseline_v1.toml は make_baseline_v1.py が生成する。本スクリプトの
格子は再定義前の旧ベースライン（旧 data/results_sa_eo と同じ条件）で、グラフ変換と旧条件の
対応表にだけ使う。

サブコマンド:
    graphs  data/graphs/*.json を data/v1/graphs/<graph_id>.json へ変換する
    map     旧ディレクトリ名 → 新 condition_id の対応表 CSV を書き出す
    all     上記 2 つを順に実行する

使い方:
    PYTHONUTF8=1 python experiments_sa_eo/convert_to_v1.py all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPP = ROOT / "target" / "release" / "gpp.exe"
OLD_GRAPH_DIR = ROOT / "data" / "graphs"
NEW_ROOT = ROOT / "data" / "v1"
MAP_PATH = ROOT / "experiments_sa_eo" / "condition_map_v1.csv"

# --- 旧ベースライン格子（旧 experiments_sa_eo/grid.py の定義をそのまま移す） ---
# grid.py は上流で削除された。旧 data/results_sa_eo の条件はここを単一の真実の源とする。
# 現行のベースラインは make_baseline_v1.py（2026-09-25 再定義）。
LOG10_ITERATIONS = 7
MAX_STEPS = 10**LOG10_ITERATIONS
ALPHA = 0.05
NS = [124, 250, 500]
DS = [5.0, 10.0, 20.0]
INSTANCE_SEEDS = [0, 1, 2, 3]
RUN_SEEDS = list(range(32))
MAX_BASIN_STEPS = 10_000


def thetas() -> list[float]:
    """SA の温度 Theta = log10(T)。-1.50 〜 +1.50 の 0.05 刻み = 61 点。"""
    return [round(v / 100.0, 2) for v in range(-150, 151, 5)]


def taus() -> list[float]:
    """旧 EO の指数 tau。0.00 〜 1.70 の 0.05 刻み（35 点）+ 上裾 1.85, 2.00 = 37 点。"""
    return [round(v / 100.0, 2) for v in range(0, 171, 5)] + [1.85, 2.00]


def taus_v1() -> list[float]:
    """新形式で表せる tau。

    2026-09-25 のアルゴリズム v2 から新実装は tau >= 0 を受け付ける（plan.rs:
    "tau must be finite and non-negative"）ので、旧格子の 37 点すべてを表せる。
    それ以前の新実装は tau > 0 を要求し、tau = 0.00 だけは作れなかった。
    """
    return list(taus())


def old_graph_id(kind: str, n: int, d: float, seed: int) -> str:
    """旧 GraphSpec::id() と同じ命名（旧ストアのディレクトリ名）。"""
    prefix = "random" if kind == "random" else "geom"
    d_s = str(int(d)) if float(d).is_integer() else str(d).replace(".", "p")
    return f"{prefix}_n{n}_d{d_s}_s{seed}"


def _fmt_theta(theta: float) -> str:
    if float(theta).is_integer():
        return f"th{int(theta):+d}"
    return f"th{theta:+.2f}".replace(".", "p")


def _fmt_num(x: float) -> str:
    """旧 Rust の Display + '.'→'p' と同じ整形。1.0→"1", 1.05→"1p05"。"""
    if float(x).is_integer():
        return str(int(x))
    return repr(float(x)).replace(".", "p")


def old_condition_name(family: str, param: float) -> str:
    """旧ストアの条件ディレクトリ名。"""
    if family == "flipSA":
        return f"{_fmt_theta(param)}_iter{LOG10_ITERATIONS}_none"
    if family == "swapSA":
        return f"saswap_{_fmt_theta(param)}_iter{LOG10_ITERATIONS}"
    if family == "swapEO":
        return f"eo_iter{LOG10_ITERATIONS}_tau{_fmt_num(param)}"
    if family == "flipEO":
        return f"eoflipmulalpha_iter{LOG10_ITERATIONS}_tau{_fmt_num(param)}_a1"
    raise ValueError(family)


# --- spec -------------------------------------------------------------------


def _floats(xs) -> str:
    return ", ".join(repr(float(x)) for x in xs)


def write_spec(path: Path, run_seeds: list[int] | None = None) -> Path:
    """旧ベースライン格子を新形式の ExperimentSpec（TOML）として書き出す。

    旧 4 系列はすべて新形式の (neighborhood, solver) で表せる（docs/convert_v1.md
    の対応表）。ルートの neighborhoods x solvers の直積がそのまま旧格子になるので、
    条件グループ（[[conditions]]）は使わない。アルゴリズム v2 から tau = 0.00 も
    表せるので、旧格子と同じ 196 条件/グラフになる。

    対応表を作るための一時ファイルにだけ使う。現行ベースラインの baseline_v1.toml を
    上書きしないよう、書き出し先は必ず指定させる。
    """
    temperatures = [10.0**t for t in thetas()]
    seeds = RUN_SEEDS if run_seeds is None else run_seeds
    lines = [
        "# SA/EO x Flip/Swap ベースライン格子（旧 iter7 実験と同じ条件）を新形式で表したもの。",
        "# 旧 experiments_sa_eo/grid.py + CLAUDE.md のベースライン定義から機械変換した。",
        "# 生成元: experiments_sa_eo/convert_to_v1.py spec",
        "#",
        "# 旧 4 系列との対応:",
        '#   flipSA = neighborhood "flip" x solver sa   温度 T = 10^Theta',
        '#   swapSA = neighborhood "swap" x solver sa',
        '#   swapEO = neighborhood "swap" x solver eo   fitness "default" (= g/deg)',
        '#   flipEO = neighborhood "flip" x solver eo   旧 EoFlipMulAlpha{tau, alpha=1.0} は',
        "#                                              alpha=1 で lambda1 が常に 1 になり g/deg と同一",
        "# 近傍 2 種 x (SA 61 温度 + EO 37 tau) = 196 条件/グラフ（旧格子と同じ）。",
        "# tau = 0.00（一様ランダム選択の極限）はアルゴリズム v2 から表せる。",
        "#",
        "# 注意: 新実装は乱数導出仕様が旧実装と異なる（rng: sha256-mt19937-64-v1、",
        "# 旧実装は実行シードを MT19937-64 へ直接投入）。ここで回した結果は旧",
        "# data/results_sa_eo の run とは別軌跡であり、run 単位で混ぜてはいけない。",
        "",
        "schema_version = 1",
        'name = "sa-eo-flip-swap-baseline"',
        f"run_seeds = [{', '.join(str(s) for s in seeds)}]",
        'neighborhoods = ["flip", "swap"]',
        "",
        "[problem]",
        f"alpha = {ALPHA!r}",
        "",
        "[budget]",
        f"max_steps = {MAX_STEPS}",
        "",
        "[measurement]",
        "# 旧実装と同じ計測点（0, 1..9, 10..90, ..., 10^7 の 65 点）になる。",
        'schedule = "logarithmic"',
        "# 旧 records の basin_real_from_real / basin_real_from_best に対応する。",
        'basin = "real"',
        "best_basin = true",
        "# 旧実装のベイスン山登りは打ち切りなしだが、この規模では常に局所最適へ到達する",
        "# （目的関数値が数百なので改善ステップ数はそれ以下）。既定値のままで同等。",
        f"max_basin_steps = {MAX_BASIN_STEPS}",
        "diagnostics = false",
        "",
        "[[graphs]]",
        'kind = "random"',
        f"node_counts = [{', '.join(str(n) for n in NS)}]",
        f"expected_degrees = [{_floats(DS)}]",
        f"seeds = [{', '.join(str(s) for s in INSTANCE_SEEDS)}]",
        "",
        "[[graphs]]",
        'kind = "geometric"',
        f"node_counts = [{', '.join(str(n) for n in NS)}]",
        f"expected_degrees = [{_floats(DS)}]",
        f"seeds = [{', '.join(str(s) for s in INSTANCE_SEEDS)}]",
        "",
        "# Theta = -1.50 .. +1.50（0.05 刻み、61 点）を T = 10^Theta にしたもの。",
        "[[solvers]]",
        'kind = "sa"',
        f"temperatures = [{_floats(temperatures)}]",
        'smoothing = [{ kind = "none" }]',
        "",
        "# tau = 0.00 .. 1.70（0.05 刻み）+ 1.85, 2.00 の 37 点。",
        "[[solvers]]",
        'kind = "eo"',
        f"taus = [{_floats(taus_v1())}]",
        'fitnesses = [{ kind = "default", params = {} }]',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


# --- graphs -----------------------------------------------------------------


def content_hash(node_count: int, edges) -> str:
    """新実装 Graph::content_hash() と同じ値を計算する。"""
    h = hashlib.sha256()
    h.update(b"gpp-graph-v1\0")
    h.update(struct.pack("<Q", node_count))
    for a, b in edges:
        h.update(struct.pack("<QQ", a, b))
    return h.hexdigest()


def load_plan(spec: Path) -> dict:
    if not GPP.exists():
        sys.exit(f"{GPP} が無い。cargo build --release を先に実行する。")
    out = subprocess.run(
        [str(GPP), "plan", str(spec), "--json"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return json.loads(out.stdout)


def load_condition_plan() -> dict:
    """condition_id / graph_id の一覧だけを得る。

    condition_id は実行シードに依存しないので、実行シード 1 本だけの複製を計画して
    ジョブ数を 1/32（446,976 → 13,968）に落とす。フル格子の計画 JSON は数百 MB に
    なるため、ここで使ってはいけない。
    """
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "plan_probe.toml"
        write_spec(probe, run_seeds=[RUN_SEEDS[0]])
        return load_plan(probe)


def graph_ids(plan: dict) -> dict:
    """旧グラフ ID（geom_n124_d10_s0 形式）→ 新 graph_id。"""
    out = {}
    for job in plan["jobs"]:
        g = job["condition"]["graph"]
        key = old_graph_id(g["kind"], g["node_count"], g["expected_degree"], g["seed"])
        out[key] = job["graph_id"]
    return out


def convert_graphs(plan: dict, dest_root: Path = NEW_ROOT):
    """旧グラフ JSON を新形式へ変換する。

    旧形式は隣接リスト + 幾何グラフの座標を持つ。新仕様は正規化した辺リストだけを
    保存し、座標は保存しない（必要なら生成条件から再生成する）。
    """
    dest = dest_root / "graphs"
    dest.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for old_id, new_id in sorted(graph_ids(plan).items()):
        src = OLD_GRAPH_DIR / f"{old_id}.json"
        if not src.exists():
            print(f"  skip {old_id}: 旧ファイルが無い")
            skipped += 1
            continue
        old = json.loads(src.read_text(encoding="utf-8"))
        adjacency = old["adjacency_list"]
        edges = sorted(
            {(min(u, v), max(u, v)) for u, nbrs in enumerate(adjacency) for v in nbrs}
        )
        if old["edge_count"] != len(edges):
            sys.exit(f"{old_id}: edge_count={old['edge_count']} だが正規化辺数は {len(edges)}")
        out = {
            "schema_version": 1,
            "node_count": len(adjacency),
            "edges": [[a, b] for a, b in edges],
            "content_hash": content_hash(len(adjacency), edges),
        }
        # 新実装の atomic::write_json と同じ整形（serde_json to_vec_pretty + 末尾改行）
        # にして、変換したファイルと gpp が生成したファイルがバイト一致するようにする。
        (dest / f"{new_id}.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        written += 1
    return written, skipped


# --- map --------------------------------------------------------------------


def classify(condition: dict):
    """新 condition から (旧系列名, スイープ対象パラメータ) を求める。"""
    nb = condition["neighborhood"]
    solver = condition["solver"]
    if solver["kind"] == "sa":
        theta = round(math.log10(solver["temperature"]), 2)
        return ("flipSA" if nb == "flip" else "swapSA"), theta
    if solver["kind"] == "eo":
        return ("flipEO" if nb == "flip" else "swapEO"), solver["tau"]
    raise ValueError(solver["kind"])


def write_map(plan: dict, path: Path = MAP_PATH) -> int:
    """旧ストアのパス → 新 condition_id の対応表。

    旧 run を新 run として再利用してはいけない（乱数導出が違う）。この表は
    同じ条件の旧結果と新結果を突き合わせるためのもの。
    """
    rows = []
    for job in plan["jobs"]:
        if job["seed"] != RUN_SEEDS[0]:
            continue  # condition_id は実行シードに依存しない
        cond = job["condition"]
        g = cond["graph"]
        family, param = classify(cond)
        old_g = old_graph_id(g["kind"], g["node_count"], g["expected_degree"], g["seed"])
        rows.append(
            {
                "family": family,
                "param": param,
                "graph_kind": g["kind"],
                "node_count": g["node_count"],
                "expected_degree": g["expected_degree"],
                "graph_seed": g["seed"],
                "old_graph_id": old_g,
                "old_condition_dir": f"data/results_sa_eo/{old_g}/{old_condition_name(family, param)}",
                "new_graph_id": job["graph_id"],
                "new_condition_id": job["condition_id"],
            }
        )
    rows.sort(key=lambda r: (r["family"], r["param"], r["old_graph_id"]))
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["graphs", "map", "all"])
    ap.add_argument("--root", type=Path, default=NEW_ROOT, help="新形式の保存ルート")
    args = ap.parse_args()

    plan = load_condition_plan()
    print(f"plan     conditions={len(plan['jobs'])}（実行シード 1 本ぶんの計画）")
    if args.command in ("graphs", "all"):
        written, skipped = convert_graphs(plan, args.root)
        rel = args.root.relative_to(ROOT)
        print(f"graphs -> {rel}/graphs  変換 {written} 件 / 未変換 {skipped} 件")
    if args.command in ("map", "all"):
        n = write_map(plan)
        print(f"map    -> {MAP_PATH.relative_to(ROOT)}  {n} 条件")


if __name__ == "__main__":
    main()
