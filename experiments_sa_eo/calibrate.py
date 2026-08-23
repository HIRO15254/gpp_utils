"""本実行前の較正ベンチ。14 スレッド実負荷でのラウンド所要時間を実測から予測する。

やること:
 1. 18 グラフ組の代表（各組 s0）x 4 系列 x 数シードを **10^6 ステップ**で回す
    （本番の 1/10。1 系列 1 パラメータ点だけ）
 2. 各ジョブの `elapsed_ms`（= 14 並列下の実測値、帯域競合込み）を 10 倍して
    10^7 換算コストにし、本番グリッド（Θ 47 点 / τ 20 点 x 4 インスタンス）で総和を取る
 3. 1 ラウンド所要時間・全 32 ラウンドの見込み・48h に収まるラウンド数を出す
 4. **Flip EO（α=1）の崩壊チェック** — 元論文の λ=g/deg はフリップ近傍でバランスを
    引き戻す力を持たないので、探索解が片側集合へ崩壊しうる。`basin_diff_from_real` /
    `basin_diff_from_best` の実測を出す

使い方:
    PYTHONUTF8=1 python experiments_sa_eo/calibrate.py --threads 14
    PYTHONUTF8=1 python experiments_sa_eo/calibrate.py --threads 14 --analyze-only
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import grid  # noqa: E402

HERE = Path(__file__).parent
BATCH = HERE / "calibrate.json"
CALIB_LOG10 = 6
CALIB_SEEDS = 4
# 較正で使う代表パラメータ（応答曲線の中央付近）
CALIB_THETA = -0.30
CALIB_TAU = 1.30


def calib_graphs():
    """18 組の代表 1 本ずつ（インスタンス s0）。"""
    return [g for g in grid.graphs() if g["seed"] == 0]


def write_batch():
    spec = {
        "graphs": calib_graphs(),
        "configs": [
            grid.cfg_flip_sa(CALIB_THETA, CALIB_LOG10),
            grid.cfg_swap_sa(CALIB_THETA, CALIB_LOG10),
            grid.cfg_swap_eo(CALIB_TAU, CALIB_LOG10),
            grid.cfg_flip_eo(CALIB_TAU, CALIB_LOG10),
        ],
        "seed_start": 0,
        "seed_count": CALIB_SEEDS,
        "save_states": False,
    }
    BATCH.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return spec


def run(threads):
    cmd = [
        "./target/release/cli.exe",
        "--batch", str(BATCH).replace("\\", "/"),
        "--out", grid.CALIB_STORE,
        "--graphs", grid.GRAPH_DIR,
        "--threads", str(threads),
        "--overwrite",
    ]
    print("$ " + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    wall = time.time() - t0
    if r.returncode != 0:
        sys.exit(f"cli が失敗しました (exit {r.returncode})")
    print(f"較正ラン完了: 実時間 {wall:.1f}s")
    return wall


def load_results():
    """(graph_id, family) -> 10^7 換算の 1 seed あたり秒数（14 並列下の実測から）。"""
    store = Path(grid.CALIB_STORE)
    per = {}
    records = {}
    for f in store.glob("*/*/seed_*.json"):
        if f.name.endswith("_states.json"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        fam = grid.family_of(d["config"])
        if fam is None:
            continue
        gid = f.parent.parent.name
        per.setdefault((gid, fam), []).append(d["elapsed_ms"] / 1000.0)
        records.setdefault(fam, []).append((gid, d["records"]))
    cost = {k: (sum(v) / len(v)) * 10.0 for k, v in per.items()}  # 10^6 -> 10^7
    return cost, records, {k: len(v) for k, v in per.items()}


def project(cost, threads):
    """本番グリッドでの 1 ラウンド所要時間（実時間・時間）。

    `elapsed_ms` は「その並列度での 1 ジョブの実時間」なので、総和は**スレッド秒**になる。
    実時間に直すにはスレッド数で割る（帯域競合は elapsed_ms 側に既に織り込み済み）。
    """
    n_inst = len(grid.INSTANCE_SEEDS)
    n_theta = len(grid.thetas())
    n_tau = len(grid.taus())

    sa_s = eo_s = 0.0
    missing = []
    for g in calib_graphs():
        gid = grid.graph_id(g)
        for fam, npts in (("flipSA", n_theta), ("swapSA", n_theta),
                          ("swapEO", n_tau), ("flipEO", n_tau)):
            c = cost.get((gid, fam))
            if c is None:
                missing.append((gid, fam))
                continue
            # 1 ラウンド = 実行シード 1 本 x 4 インスタンス x 全パラメータ点
            tot = c * n_inst * npts
            if fam.endswith("SA"):
                sa_s += tot
            else:
                eo_s += tot
    if missing:
        print(f"警告: 較正結果が欠けている組合せ {len(missing)} 件: {missing[:5]}")
    return sa_s / threads / 3600.0, eo_s / threads / 3600.0


def collapse_report(records):
    """Flip EO（α=1）が片側集合へ崩壊していないかを見る。"""
    print()
    print("=== Flip EO (λ=g/deg, α=1) の崩壊チェック ===")
    for fam in ("flipEO", "flipSA"):
        rows = records.get(fam, [])
        if not rows:
            continue
        agg = {}
        for gid, recs in rows:
            n = int(gid.split("_n")[1].split("_")[0])
            md = max(abs(r.get("basin_diff_from_real", 0)) for r in recs)
            mb = max(abs(r.get("basin_diff_from_best", 0)) for r in recs)
            cur = agg.get(gid, (0, 0, n))
            agg[gid] = (max(cur[0], md), max(cur[1], mb), n)
        worst = sorted(((md / n, gid, md, mb, n)
                        for gid, (md, mb, n) in agg.items()), reverse=True)
        print(f"[{fam}] ベイスンの |A|-|B| 最大値 / N（上位 5 グラフ）")
        for frac, gid, md, mb, n in worst[:5]:
            flag = "  <-- 崩壊の疑い" if frac > 0.5 else ""
            print(f"   {gid:<22} 探索解ベイスン {md:>4} / {n}  最良解ベイスン {mb:>4}"
                  f"  ({frac:.1%}){flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--analyze-only", action="store_true",
                    help="既存の較正結果だけを解析する（再実行しない）")
    args = ap.parse_args()

    spec = write_batch()
    if not args.analyze_only:
        n_jobs = len(spec["graphs"]) * len(spec["configs"]) * spec["seed_count"]
        print(f"較正バッチ: {n_jobs} ジョブ（10^{CALIB_LOG10} ステップ, {args.threads} スレッド）")
        run(args.threads)

    cost, records, counts = load_results()
    if not cost:
        sys.exit(f"{grid.CALIB_STORE} に結果がありません")

    print()
    print("=== 10^7 換算の 1 seed あたり秒数（14 並列下の実測 x10）===")
    print(f"{'graph':<22} {'flipSA':>8} {'swapSA':>8} {'swapEO':>8} {'flipEO':>8}")
    for g in calib_graphs():
        gid = grid.graph_id(g)
        row = [cost.get((gid, f)) for f in ("flipSA", "swapSA", "swapEO", "flipEO")]
        print(f"{gid:<22} " + " ".join(f"{v:8.1f}" if v else "     n/a" for v in row))

    sa_h, eo_h = project(cost, args.threads)
    round_h = sa_h + eo_h
    n_rounds = len(grid.RUN_SEEDS)
    print()
    print("=== 予測 ===")
    print(f"1 ラウンド（実時間, {args.threads} スレッド）: "
          f"SA {sa_h:.2f}h + EO {eo_h:.2f}h = {round_h:.2f}h")
    print(f"全 {n_rounds} ラウンド: {round_h * n_rounds:.1f}h "
          f"(1 条件あたり {len(grid.INSTANCE_SEEDS) * n_rounds} 試行)")
    for budget in (44.0, 46.0):
        fit = int(budget // round_h)
        print(f"  {budget:.0f}h 以内に完走できるラウンド数: {min(fit, n_rounds)} "
              f"(= 1 条件あたり {min(fit, n_rounds) * len(grid.INSTANCE_SEEDS)} 試行)")

    collapse_report(records)


if __name__ == "__main__":
    main()
