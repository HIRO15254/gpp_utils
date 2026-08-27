"""SA/EO x Flip/Swap 比較実験（iter7）のグリッド定義。単一の真実の源。

**今後の実験のベースライン**でもある（リポジトリ直下 CLAUDE.md 参照）。
本実験 + 追試1/4/5 で埋めた完全グリッドを表す。

ここだけを見れば「何を回すか」が全部わかるようにしてある。
`make_batches.py` / `calibrate.py` / `scan_state.py` / `analyze.py` はすべてここを import する。

用語:
  - グラフ組 (combo)  : (kind, n, d) の 18 通り
  - インスタンス      : 各組の生成シード s0..s3（4 本）→ 実グラフは 72 本
  - 実行シード        : ソルバーの乱数シード 0..31（32 本）
  - 1 条件あたり試行数 = 4 インスタンス x 32 実行シード = 128
"""

LOG10_ITERATIONS = 7

KINDS = ["Random", "Geometric"]
NS = [124, 250, 500]
DS = [5.0, 10.0, 20.0]
INSTANCE_SEEDS = [0, 1, 2, 3]

RUN_SEEDS = list(range(32))

STORE = "data/results_sa_eo"
CALIB_STORE = "data/results_sa_eo_calib"
GRAPH_DIR = "data/graphs"


def graphs():
    """全 72 グラフの GraphSpec（cli がそのまま食える形）。"""
    out = []
    for kind in KINDS:
        for n in NS:
            for d in DS:
                for s in INSTANCE_SEEDS:
                    out.append({"kind": kind, "n": n, "d": d, "seed": s})
    return out


def graph_id(g):
    """Rust 側 `GraphSpec::id()` と同じ命名（`data/results/<id>/` のディレクトリ名）。"""
    prefix = "random" if g["kind"] == "Random" else "geom"
    d = g["d"]
    d_s = str(int(d)) if float(d).is_integer() else str(d).replace(".", "p")
    return f"{prefix}_n{g['n']}_d{d_s}_s{g['seed']}"


def thetas():
    """SA の温度 Θ = log10(T)。-1.50 〜 +1.50 の 0.05 刻み = 61 点。

    本実験は +0.80 まで、追試5（2026-08-26）で +1.50 まで延長。両端で応答曲線が
    悪化に転じることを確認済み（最適点は全組で内部）。
    浮動小数の累積誤差を避けるため整数から作る。
    """
    return [round(v / 100.0, 2) for v in range(-150, 151, 5)]


def taus():
    """EO のべき乗則指数 τ。0.00 〜 1.70 の 0.05 刻み（35 点）+ 粗い上裾 1.85, 2.00 = 37 点。

    本実験は 0.60/0.80/0.95 + 1.00〜1.70、追試1/4/5 で 0.00〜0.95 を 0.05 刻みに
    埋めた。swapEO / flipEO 共通。τ=0 は一様ランダム選択の極限。
    """
    return [round(v / 100.0, 2) for v in range(0, 171, 5)] + [1.85, 2.00]


# --- RunConfig ビルダー -------------------------------------------------------
# `name` は表示用。実際の保存先ディレクトリは Rust の `RunConfig::id()` が決める。


def _fmt_theta(theta):
    if float(theta).is_integer():
        return f"th{int(theta):+d}"
    return f"th{theta:+.2f}".replace(".", "p")


def _fmt_num(x):
    """Rust の `{}`（Display）+ `.`→`p` と同じ整形。1.0→"1", 1.05→"1p05"。"""
    if float(x).is_integer():
        return str(int(x))
    return repr(float(x)).replace(".", "p")


def cfg_flip_sa(theta, log10=LOG10_ITERATIONS):
    return {
        "name": f"{_fmt_theta(theta)}_iter{log10}_none",
        "theta": theta,
        "log10_iterations": log10,
        "smoothing": "None",
        "solver": "Sa",
    }


def cfg_swap_sa(theta, log10=LOG10_ITERATIONS):
    return {
        "name": f"saswap_{_fmt_theta(theta)}_iter{log10}",
        "theta": theta,
        "log10_iterations": log10,
        "smoothing": "None",
        "solver": "SaSwap",
    }


def cfg_swap_eo(tau, log10=LOG10_ITERATIONS):
    return {
        "name": f"eo_iter{log10}_tau{_fmt_num(tau)}",
        "theta": None,
        "log10_iterations": log10,
        "smoothing": "None",
        "solver": {"Eo": {"tau": tau}},
    }


def cfg_flip_eo(tau, log10=LOG10_ITERATIONS):
    """元論文の適応度 λ = g/deg をフリップ近傍に適用したもの（α=1 で λ1 ≡ 1）。"""
    return {
        "name": f"eoflipmulalpha_iter{log10}_tau{_fmt_num(tau)}_a1",
        "theta": None,
        "log10_iterations": log10,
        "smoothing": "None",
        "solver": {"EoFlipMulAlpha": {"tau": tau, "alpha": 1.0}},
    }


def sa_configs(log10=LOG10_ITERATIONS):
    return [cfg_flip_sa(t, log10) for t in thetas()] + [
        cfg_swap_sa(t, log10) for t in thetas()
    ]


def eo_configs(log10=LOG10_ITERATIONS):
    return [cfg_swap_eo(t, log10) for t in taus()] + [
        cfg_flip_eo(t, log10) for t in taus()
    ]


def family_of(cfg):
    """保存された config オブジェクトから系列名を決める。

    **ディレクトリ名の正規表現には依存しない**（旧実験でそれが事故の元になった）。
    """
    solver = cfg["solver"]
    if solver == "Sa":
        return "flipSA"
    if solver == "SaSwap":
        return "swapSA"
    if isinstance(solver, dict):
        if "Eo" in solver:
            return "swapEO"
        if "EoFlipMulAlpha" in solver:
            return "flipEO"
    return None


def param_of(cfg):
    """系列のスイープ対象パラメータ値（SA なら Θ、EO なら τ）。"""
    fam = family_of(cfg)
    if fam in ("flipSA", "swapSA"):
        return cfg["theta"]
    solver = cfg["solver"]
    key = "Eo" if "Eo" in solver else "EoFlipMulAlpha"
    return solver[key]["tau"]


if __name__ == "__main__":
    gs, th, ta = graphs(), thetas(), taus()
    print(f"グラフ組: {len(KINDS) * len(NS) * len(DS)} 組 x {len(INSTANCE_SEEDS)} インスタンス = {len(gs)} グラフ")
    print(f"Θ: {len(th)} 点  {th[0]} .. {th[-1]}")
    print(f"τ: {len(ta)} 点  {ta}")
    print(f"実行シード: {len(RUN_SEEDS)} 本 → 1 条件 {len(INSTANCE_SEEDS) * len(RUN_SEEDS)} 試行")
    print(f"1 ラウンドのジョブ数: {len(gs) * (len(sa_configs()) + len(eo_configs()))}")
    print(f"全ラウンド合計:       {len(gs) * (len(sa_configs()) + len(eo_configs())) * len(RUN_SEEDS)}")
