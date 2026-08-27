# -*- coding: utf-8 -*-
"""追試（2026-08-25）のバッチ生成。`grid.py`（本実験の真実の源）は変更しない。

追試1: flipEO の τ グリッドを下方へ延長（τ < 0.6）。
  本実験では random 系 6 組で τ* = 0.6（グリッド下端）に張り付き、端で曲線がまだ
  下降中だった。τ を下げる = 選択が一様に近づく = 片側崩壊が弱まる、という機序が
  あるので、真の最適はグリッド外にあると見込まれる。τ=0 は一様ランダム選択の極限。
  → 本実験と同じストア `data/results_sa_eo` に追加（応答曲線が地続きになる）。

追試3: flipEO の**現行解**の不均衡 |T|-|F| を実測する。
  本実験では現行解スコアからの下界推定しかできていない。`save_states` は
  EoFlip 系ソルバーでのみ働く（`execute_with_states`）ので flipEO は対象。
  → 汚染回避のため別ストア `data/results_sa_eo_states` に出す。

追試2（SA の Θ を +0.80 より上へ延長）は**実施しない**。応答曲線を確認したところ
geom d=20 の 3 組はいずれもグリッド上端で既に悪化に転じており（例 geom_n500_d20
flipSA: +0.75 で 162.2 → +0.80 で 167.3、swapSA: 162.7 → 174.4）、最適点は
グリッド内部にある。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import grid  # noqa: E402

HERE = Path(__file__).parent
BATCH_DIR = HERE / "batches_followup"
STATES_STORE = "data/results_sa_eo_states"

# --- 追試1 -------------------------------------------------------------------
TAUS_LOW = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# --- 追試3 -------------------------------------------------------------------
# 崩壊が強い組・弱い組を混ぜる（本実験の下界推定: random_n500_d20 76% /
# random_n250_d10 74% / geom_n500_d10 62% / geom_n500_d5 19%）。
STATES_COMBOS = [
    ("Random", 250, 10.0),
    ("Random", 500, 20.0),
    ("Geometric", 500, 10.0),
    ("Geometric", 500, 5.0),
]
STATES_TAUS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.0]
STATES_SEEDS = 8


def write_followup1():
    """1 ラウンド = 実行シード 1 本。どこで止めても全条件が同じ試行数になる。"""
    graphs = grid.graphs()
    configs = [grid.cfg_flip_eo(t) for t in TAUS_LOW]
    BATCH_DIR.mkdir(exist_ok=True)
    index = []
    for k in grid.RUN_SEEDS:
        spec = {
            "graphs": graphs,
            "configs": configs,
            "seed_start": k,
            "seed_count": 1,
            "save_states": False,
        }
        f = BATCH_DIR / f"f1_round{k:02d}.json"
        f.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        index.append({"round": k, "file": str(f), "jobs": len(graphs) * len(configs)})
    (BATCH_DIR / "f1_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    n = len(graphs) * len(configs) * len(grid.RUN_SEEDS)
    print(f"追試1: {len(graphs)} グラフ x {len(configs)} τ x {len(grid.RUN_SEEDS)} シード "
          f"= {n} run / {len(index)} ラウンド -> {BATCH_DIR}")
    print(f"  τ = {TAUS_LOW}")


def write_followup3():
    graphs = [{"kind": k, "n": n, "d": d, "seed": 0} for k, n, d in STATES_COMBOS]
    configs = [grid.cfg_flip_eo(t) for t in STATES_TAUS]
    spec = {
        "graphs": graphs,
        "configs": configs,
        "seed_start": 0,
        "seed_count": STATES_SEEDS,
        "save_states": True,
    }
    f = BATCH_DIR / "f3_states.json"
    BATCH_DIR.mkdir(exist_ok=True)
    f.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    n = len(graphs) * len(configs) * STATES_SEEDS
    print(f"追試3: {len(graphs)} グラフ x {len(configs)} τ x {STATES_SEEDS} シード "
          f"= {n} run (save_states) -> {f}")
    print(f"  ストア: {STATES_STORE}")


def write_reprocheck():
    """既存 run が bit 一致で再現するかの検査用（別ストアへ 1 本）。"""
    spec = {
        "graphs": [{"kind": "Random", "n": 250, "d": 10.0, "seed": 0}],
        "configs": [grid.cfg_flip_eo(1.3), grid.cfg_swap_eo(1.3), grid.cfg_flip_sa(-0.20)],
        "seed_start": 0,
        "seed_count": 1,
        "save_states": False,
    }
    BATCH_DIR.mkdir(exist_ok=True)
    f = BATCH_DIR / "reprocheck.json"
    f.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"再現性チェック -> {f}")


if __name__ == "__main__":
    write_reprocheck()
    write_followup1()
    write_followup3()


# --- 追試4（案a）: τ の粗い区間 (0.60, 0.95) を 0.05 刻みに埋める ------------
# 当初計画の τ グリッドは「コア 1.00〜1.70 を 0.05 刻み、両裾は粗く」であり、
# 0.60 と 0.95 の間には 0.80 の 1 点しか存在しなかった。Flip-EO は 18 問題例中
# 12 例で最適 τ がこの粗い区間側に落ちており、特に geometric D=20 では谷が
# 1 点でしか通っていない（geom (500,20): τ=0.60 で 239.9、0.80 で 174.9、
# 0.95 で 280.7）。この区間を両系列とも 0.05 刻みに揃える。
TAUS_FILL = [0.65, 0.70, 0.75, 0.85, 0.90]


def write_followup4():
    graphs = grid.graphs()
    configs = ([grid.cfg_flip_eo(t) for t in TAUS_FILL]
               + [grid.cfg_swap_eo(t) for t in TAUS_FILL])
    BATCH_DIR.mkdir(exist_ok=True)
    index = []
    for k in grid.RUN_SEEDS:
        spec = {
            "graphs": graphs,
            "configs": configs,
            "seed_start": k,
            "seed_count": 1,
            "save_states": False,
        }
        f = BATCH_DIR / f"f4_round{k:02d}.json"
        f.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        index.append({"round": k, "file": str(f), "jobs": len(graphs) * len(configs)})
    (BATCH_DIR / "f4_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    n = len(graphs) * len(configs) * len(grid.RUN_SEEDS)
    print(f"追試4: {len(graphs)} グラフ x {len(configs)} 条件 x {len(grid.RUN_SEEDS)} シード "
          f"= {n} run / {len(index)} ラウンド")
    print(f"  τ = {TAUS_FILL}（Flip-EO と Swap-EO の両方）")


# --- 追試5（2026-08-26）: 両端の 0.05 刻み延長 ---------------------------------
# EO 側: τ を 0.05 刻みで 0.00 まで下げる。
#   swapEO は本実験で τ<0.60 が全く無い（0.00〜0.55 の 12 点を追加）。
#   flipEO は追試1で 0.0〜0.5 を 0.1 刻みで持っているので、間の 6 点だけ埋める。
# SA 側: Θ を 0.05 刻みで +0.85〜+1.50 に延長（14 点 x flipSA/swapSA）。
#   追試2 は「geom d=20 は上端で既に悪化に転じている」として見送ったが、
#   上昇の勾配が緩い（geom_n500_d20 flipSA: 162.2 → 167.3）ので上側の枝を
#   きちんと描くために延長する。SA は 1 run 1〜2 s なので安い。
# 出力は本実験と同じストア `data/results_sa_eo`（応答曲線が地続きになる）。
TAUS_EXT_SWAP = [round(v / 100.0, 2) for v in range(0, 56, 5)]      # 0.00..0.55 (12)
TAUS_EXT_FLIP = [round(v / 100.0, 2) for v in range(5, 56, 10)]     # 0.05,0.15,..,0.55 (6)
THETAS_EXT = [round(v / 100.0, 2) for v in range(85, 151, 5)]       # +0.85..+1.50 (14)


def write_followup5():
    graphs = grid.graphs()
    sa = ([grid.cfg_flip_sa(t) for t in THETAS_EXT]
          + [grid.cfg_swap_sa(t) for t in THETAS_EXT])
    eo = ([grid.cfg_swap_eo(t) for t in TAUS_EXT_SWAP]
          + [grid.cfg_flip_eo(t) for t in TAUS_EXT_FLIP])
    BATCH_DIR.mkdir(exist_ok=True)
    index = []
    for k in grid.RUN_SEEDS:
        for tag, cfgs in (("SA", sa), ("EO", eo)):
            spec = {
                "graphs": graphs,
                "configs": cfgs,
                "seed_start": k,
                "seed_count": 1,
                "save_states": False,
            }
            f = BATCH_DIR / f"f5_round{k:02d}__{tag}.json"
            f.write_text(json.dumps(spec, indent=1), encoding="utf-8")
            index.append({"round": k, "tag": tag, "file": str(f).replace("\\", "/"),
                          "jobs": len(graphs) * len(cfgs)})
    (BATCH_DIR / "f5_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    n_sa = len(graphs) * len(sa) * len(grid.RUN_SEEDS)
    n_eo = len(graphs) * len(eo) * len(grid.RUN_SEEDS)
    print(f"追試5: SA {len(sa)} 条件 ({n_sa} run) + EO {len(eo)} 条件 ({n_eo} run) "
          f"= {n_sa + n_eo} run / {len(grid.RUN_SEEDS)} ラウンド x 2 ファイル")
    print(f"  Θ = {THETAS_EXT}")
    print(f"  τ swapEO = {TAUS_EXT_SWAP}")
    print(f"  τ flipEO = {TAUS_EXT_FLIP}")
