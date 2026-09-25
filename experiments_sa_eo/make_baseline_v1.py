"""SA/EO x flip/swap ベースラインの実験仕様 baseline_v1.toml を書き出す。

ベースライン条件の単一の真実の源。今後の計算はすべてこのベースラインを基準とする
（2026-09-25 決定。運用の決まりは AGENTS.md「実験のベースライン」）。2026-09-25 に次の条件へ再定義した。
    Theta   -1.50 .. +2.50（0.05 刻み、81 点）、温度 T = 10^Theta
    tau      0.00 .. 3.00（0.05 刻み、61 点）、適応度 default（g/deg）
    グラフ   random / geometric x n 124, 250, 500 x 平均次数 5, 10, 20 の 18 組、各組インスタンス 1 つ（s0）
    実行シード 32 個、10^6 ステップ
近傍 2 種 x (SA 81 温度 + EO 61 tau) = 284 条件/グラフ、18 グラフ x 32 シードで 163,584 ジョブ。

再定義前のベースライン（Theta -1.50..+1.50、tau 0..1.70 と 1.85/2.00、4 インスタンス、10^7）の定義と、
旧 data/results_sa_eo との対応表は convert_to_v1.py に残してある。

使い方:
    PYTHONUTF8=1 python experiments_sa_eo/make_baseline_v1.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "experiments_sa_eo" / "baseline_v1.toml"

MAX_STEPS = 10**6
ALPHA = 0.05
NS = [124, 250, 500]
DS = [5.0, 10.0, 20.0]
INSTANCE_SEEDS = [0]
RUN_SEEDS = list(range(32))
MAX_BASIN_STEPS = 10_000


def thetas() -> list[float]:
    """SA の温度 Theta = log10(T)。-1.50 〜 +2.50 の 0.05 刻み = 81 点。"""
    return [round(v / 100.0, 2) for v in range(-150, 251, 5)]


def temperature(theta: float) -> float:
    """旧 grid.py・convert_to_v1.py と同じ式。同じ Theta から同じ温度（同じ条件 ID）になる。"""
    return 10.0 ** round(theta, 2)


def taus() -> list[float]:
    """EO の指数 tau。0.00 〜 3.00 の 0.05 刻み = 61 点。"""
    return [round(v / 100.0, 2) for v in range(0, 301, 5)]


def _floats(xs) -> str:
    return ", ".join(repr(float(x)) for x in xs)


def _ints(xs) -> str:
    return ", ".join(str(x) for x in xs)


def write_spec(path: Path = SPEC_PATH) -> Path:
    th, ta = thetas(), taus()
    graph_block = [
        f"node_counts = [{_ints(NS)}]",
        f"expected_degrees = [{_floats(DS)}]",
        f"seeds = [{_ints(INSTANCE_SEEDS)}]",
    ]
    lines = [
        "# SA/EO x Flip/Swap ベースライン（2026-09-25 再定義）。",
        "# 今後の計算はすべてこのベースラインを基準とする（AGENTS.md「実験のベースライン」）。",
        "# 生成元: experiments_sa_eo/make_baseline_v1.py（条件を変えるときはこちらを直して再生成する）",
        "#",
        "# 4 系列:",
        '#   flipSA = neighborhood "flip" x solver sa   温度 T = 10^Theta',
        '#   swapSA = neighborhood "swap" x solver sa',
        '#   flipEO = neighborhood "flip" x solver eo   fitness "default" (= g/deg)',
        '#   swapEO = neighborhood "swap" x solver eo',
        f"# 近傍 2 種 x (SA {len(th)} 温度 + EO {len(ta)} tau) = {2 * (len(th) + len(ta))} 条件/グラフ。",
        "#",
        "# 再定義前（Theta -1.50..+1.50、tau 0..1.70 と 1.85/2.00、4 インスタンス、10^7）とは",
        "# 条件 ID が別になる。旧 data/results_sa_eo の run とも乱数導出が違うので、run 単位で混ぜない。",
        "",
        "schema_version = 1",
        'name = "sa-eo-flip-swap-baseline"',
        f"run_seeds = [{_ints(RUN_SEEDS)}]",
        'neighborhoods = ["flip", "swap"]',
        "",
        "[problem]",
        f"alpha = {ALPHA!r}",
        "",
        "[budget]",
        f"max_steps = {MAX_STEPS}",
        "",
        "[measurement]",
        "# 計測点は 0, 1..9, 10..90, ..., 10^6 の 56 点。",
        'schedule = "logarithmic"',
        "# 現行解からのベイスン値（basin_real）と暫定解からのベイスン値（basin_best）を測る。",
        'basin = "real"',
        "best_basin = true",
        f"max_basin_steps = {MAX_BASIN_STEPS}",
        "diagnostics = false",
        "",
        "[[graphs]]",
        'kind = "random"',
        *graph_block,
        "",
        "[[graphs]]",
        'kind = "geometric"',
        *graph_block,
        "",
        f"# Theta = {th[0]:+.2f} .. {th[-1]:+.2f}（0.05 刻み、{len(th)} 点）を T = 10^Theta にしたもの。",
        "[[solvers]]",
        'kind = "sa"',
        f"temperatures = [{_floats(temperature(t) for t in th)}]",
        'smoothing = [{ kind = "none" }]',
        "",
        f"# tau = {ta[0]:.2f} .. {ta[-1]:.2f}（0.05 刻み、{len(ta)} 点）。",
        "[[solvers]]",
        'kind = "eo"',
        f"taus = [{_floats(ta)}]",
        'fitnesses = [{ kind = "default", params = {} }]',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


if __name__ == "__main__":
    out = write_spec()
    n_cond = 2 * (len(thetas()) + len(taus()))
    n_graphs = 2 * len(NS) * len(DS) * len(INSTANCE_SEEDS)
    print(f"{out.relative_to(ROOT)}: {n_cond} 条件/グラフ x {n_graphs} グラフ x {len(RUN_SEEDS)} シード = "
          f"{n_cond * n_graphs * len(RUN_SEEDS):,} ジョブ")
