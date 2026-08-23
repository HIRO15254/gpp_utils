"""`data/results_sa_eo` のカバレッジ走査。ラウンド完了ごとに欠損が無いか確認する。

系列とパラメータは**保存された `config` オブジェクトから導出**する
（ディレクトリ名の正規表現には依存しない。旧 iter8 実験ではそれが事故の元になった）。

使い方:
    PYTHONUTF8=1 python experiments_sa_eo/scan_state.py
    PYTHONUTF8=1 python experiments_sa_eo/scan_state.py --detail   # 欠損を全部列挙
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import grid  # noqa: E402


def scan(store):
    """(graph_id, family, param) -> 存在する seed の集合。"""
    seen = defaultdict(set)
    bad_cfg = set()
    for cfg_dir in sorted(Path(store).glob("*/*")):
        if not cfg_dir.is_dir():
            continue
        files = [f for f in cfg_dir.glob("seed_*.json") if not f.name.endswith("_states.json")]
        if not files:
            continue
        # config は同一ディレクトリ内で共通なので 1 本だけ読む。
        d = json.loads(files[0].read_text(encoding="utf-8"))
        cfg = d["config"]
        fam = grid.family_of(cfg)
        if fam is None or cfg["log10_iterations"] != grid.LOG10_ITERATIONS:
            bad_cfg.add(cfg_dir.name)
            continue
        key = (cfg_dir.parent.name, fam, grid.param_of(cfg))
        for f in files:
            seen[key].add(int(f.stem.split("_")[1]))
    return seen, bad_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=grid.STORE)
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    if not Path(args.store).exists():
        sys.exit(f"{args.store} がありません（まだ 1 ラウンドも流していない）")

    seen, bad_cfg = scan(args.store)
    expect_keys = set()
    for g in grid.graphs():
        gid = grid.graph_id(g)
        for t in grid.thetas():
            expect_keys.add((gid, "flipSA", t))
            expect_keys.add((gid, "swapSA", t))
        for t in grid.taus():
            expect_keys.add((gid, "swapEO", t))
            expect_keys.add((gid, "flipEO", t))

    # 系列ごとの完了ラウンド数 = 全条件が持っている seed の共通集合の大きさ
    per_fam = defaultdict(list)
    missing = []
    for key in sorted(expect_keys):
        got = seen.get(key, set())
        per_fam[key[1]].append(len(got))
        if args.detail:
            gap = set(range(max(got) + 1 if got else 0)) - got
            if gap:
                missing.append((key, sorted(gap)))

    print(f"ストア: {args.store}")
    print(f"期待条件数: {len(expect_keys)}（72 グラフ x (Θ47x2 + τ20x2)）")
    print()
    print(f"{'family':<8} {'条件数':>6} {'min seeds':>10} {'max seeds':>10} {'完了試行/条件':>14}")
    for fam in ("flipSA", "swapSA", "swapEO", "flipEO"):
        counts = per_fam[fam]
        if not counts:
            print(f"{fam:<8} {'-':>6}")
            continue
        # 4 インスタンスぶん揃って初めて「+4 試行」なので、最小値がそのまま安全側の指標。
        print(f"{fam:<8} {len(counts):>6} {min(counts):>10} {max(counts):>10} "
              f"{min(counts) * 1:>14}")

    all_counts = [c for v in per_fam.values() for c in v]
    if all_counts:
        complete_rounds = min(all_counts)
        print()
        print(f"全系列そろって完了しているラウンド数: {complete_rounds} "
              f"→ 1 条件あたり {complete_rounds * len(grid.INSTANCE_SEEDS)} 試行")
        if min(all_counts) != max(all_counts):
            print(f"注意: 条件によって seed 数が {min(all_counts)}..{max(all_counts)} とばらついている"
                  f"（進行中のラウンドがあるなら正常）")

    if bad_cfg:
        print()
        print(f"iter{grid.LOG10_ITERATIONS} 以外 / 未知の config ディレクトリ {len(bad_cfg)} 件:"
              f" {sorted(bad_cfg)[:5]}")

    if args.detail and missing:
        print()
        print(f"連番の穴がある条件 {len(missing)} 件:")
        for key, gap in missing[:40]:
            print(f"  {key} -> 欠損 seed {gap[:10]}")


if __name__ == "__main__":
    main()
