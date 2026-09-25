"""復旧 run 計画 v1 の段階 C（適応度）と B（平滑化）の実験仕様を書き出す。

計画の全体（実行順 A → C → B、規模、含めないもの）は RECOVERY_PLAN_v1.md にある。
グラフ・ステップ数・実行シード・計測条件・Theta と tau の格子は、make_baseline_v1.py の定数と式をそのまま使う。
そのため C の default 列は A（baseline_v1.toml）の flipEO と、B の平滑化なし列は A の flipSA と条件 ID が一致し、
gpp が A の結果を再利用する。
    C  recovery_v1_C_fitness.toml    flip x EO、tau 61 点 x 適応度 24 種
                                     （default、乗算 alpha 0.0..0.9 の 10 点、加算 beta 13 点）
    B  recovery_v1_B_smoothing.toml  flip x SA、Theta 81 点 x 平滑化 8 種
                                     （none、random_k K=1, 2, 4, 8, 16, 32、all_average）

使い方:
    PYTHONUTF8=1 python experiments_sa_eo/make_recovery_v1.py
"""

from __future__ import annotations

from pathlib import Path

import make_baseline_v1 as base

HERE = Path(__file__).resolve().parent
C_PATH = HERE / "recovery_v1_C_fitness.toml"
B_PATH = HERE / "recovery_v1_B_smoothing.toml"

# 乗算 alpha=1.0 は default と同じ順位になるので入れない。
MUL_ALPHAS = [round(v / 10.0, 1) for v in range(0, 10)]
# 旧スクリーニングの 1.25..32 に、0.0（lambda0 の情報がなくなる端）と 0.5（0<beta<1 の代表）を足したもの。
ADD_BETAS = [0.0, 0.5, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]
RANDOM_KS = [1, 2, 4, 8, 16, 32]


def _common(name: str, comment: list[str]) -> list[str]:
    graph_block = [
        f"node_counts = [{base._ints(base.NS)}]",
        f"expected_degrees = [{base._floats(base.DS)}]",
        f"seeds = [{base._ints(base.INSTANCE_SEEDS)}]",
    ]
    return [
        *comment,
        "# 計画: experiments_sa_eo/RECOVERY_PLAN_v1.md（実行順 A → C → B）",
        "# 生成元: experiments_sa_eo/make_recovery_v1.py（グラフ・ステップ・シード・計測・格子はベースラインの定数と式を使う）",
        "",
        "schema_version = 1",
        f'name = "{name}"',
        f"run_seeds = [{base._ints(base.RUN_SEEDS)}]",
        'neighborhoods = ["flip"]',
        "",
        "[problem]",
        f"alpha = {base.ALPHA!r}",
        "",
        "[budget]",
        f"max_steps = {base.MAX_STEPS}",
        "",
        "[measurement]",
        "# ベースラインと同じ計測。0, 1..9, 10..90, ..., 10^6 の 56 点で、現行解と暫定解からのベイスン値を測る。",
        'schedule = "logarithmic"',
        'basin = "real"',
        "best_basin = true",
        f"max_basin_steps = {base.MAX_BASIN_STEPS}",
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
    ]


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def write_specs(c_path: Path = C_PATH, b_path: Path = B_PATH) -> list[Path]:
    ta, th = base.taus(), base.thetas()
    fitnesses = ['{ kind = "default", params = {} }']
    fitnesses += [f'{{ kind = "multiplicative", params = {{ alpha = {a!r} }} }}' for a in MUL_ALPHAS]
    fitnesses += [f'{{ kind = "additive", params = {{ beta = {b!r} }} }}' for b in ADD_BETAS]
    c = _common(
        "recovery-v1-C-fitness",
        [
            "# 復旧 run 計画 v1 の段階 C: flipEO の適応度（前期まとめテーマ 2。旧 data/results_sa_eo_screen の作り直し）。",
            "# default 列はベースラインの flipEO と同一条件なので、段階 A の結果が再利用される。",
            "# 乗算 alpha=1.0 は default と同じ順位。加算 beta は 0<beta<1 で同一アルゴリズムなので 0.5 を代表、0.0 は退化端。",
        ],
    ) + [
        f"# tau = {ta[0]:.2f} .. {ta[-1]:.2f}（0.05 刻み、{len(ta)} 点）。ベースラインと同じ式で生成。",
        "[[solvers]]",
        'kind = "eo"',
        f"taus = [{base._floats(ta)}]",
        "fitnesses = [",
        *[f"  {f}," for f in fitnesses],
        "]",
    ]
    b = _common(
        "recovery-v1-B-smoothing",
        [
            "# 復旧 run 計画 v1 の段階 B: 平滑化 SA（前期まとめテーマ 1。旧 iter6 の作り直し）。",
            "# none 列はベースラインの flipSA と同一条件なので、段階 A の結果が再利用される。",
            "# all_average は重み w=1 の全近傍平均。重み付き平均が温度を変えた平滑化なし SA と同じになる理論の確認用。",
        ],
    ) + [
        f"# Theta = {th[0]:+.2f} .. {th[-1]:+.2f}（0.05 刻み、{len(th)} 点）を T = 10^Theta にしたもの。ベースラインと同じ式で生成。",
        "[[solvers]]",
        'kind = "sa"',
        f"temperatures = [{base._floats(base.temperature(t) for t in th)}]",
        "[[solvers.smoothing]]",
        'kind = "none"',
        "[[solvers.smoothing]]",
        'kind = "random_k_average"',
        f"ks = [{base._ints(RANDOM_KS)}]",
        "[[solvers.smoothing]]",
        'kind = "all_average"',
    ]
    return [_write(c_path, c), _write(b_path, b)]


def job_counts() -> dict[str, int]:
    graphs_x_seeds = 2 * len(base.NS) * len(base.DS) * len(base.INSTANCE_SEEDS) * len(base.RUN_SEEDS)
    return {
        "C": (1 + len(MUL_ALPHAS) + len(ADD_BETAS)) * len(base.taus()) * graphs_x_seeds,
        "B": (2 + len(RANDOM_KS)) * len(base.thetas()) * graphs_x_seeds,
    }


if __name__ == "__main__":
    paths = write_specs()
    counts = job_counts()
    for stage, path in zip(["C", "B"], paths):
        print(f"{path.relative_to(base.ROOT)}: {counts[stage]:,} ジョブ")
