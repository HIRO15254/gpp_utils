"""予備実験の集計: smoothing_pilot run の出力(TSV)から、グラフ×手法ごとの最適Θでの暫定解ベイスン値を求める。

最適Θの選択による楽観的な偏りを避けるため、シードを2分割して交差検証する(0-3で選んで4-7で評価し、逆も行って平均する)。
--summary を付けると、グラフ×手法×Θごとの平均・標準誤差・受理率をTSVで書き出す。

複数の出力(例: シードを追加した試走)をカンマ区切りで渡すと、まとめて集計する。

使い方:
    python analyze.py <pilot.tsv[,more.tsv]> [checkpoint(既定 1000000)] [--window lo,hi] [--variants a,b,...]
    python analyze.py <pilot.tsv[,more.tsv]> --summary <out.tsv>
"""

import collections
import csv
import math
import sys

ORDER = ["none", "rk1", "rk4", "nsm_rho1", "nsm_rho0.5", "nsm_erosion", "hk_tau0.5", "hk_tau2", "hk_sched"]


def load(paths, window=None, variants=None, graphs=None):
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path in paths.split(","):
        for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
            theta = float(r["theta"])
            if window and not (window[0] - 1e-9 <= theta <= window[1] + 1e-9):
                continue
            if variants and r["variant"] not in variants:
                continue
            if graphs and not any(g in r["graph"] for g in graphs):
                continue
            rows[r["step"]][(r["graph"], r["variant"])].setdefault(theta, {})[int(r["seed"])] = r
    return rows


def graph_key(g):
    kind, n, d = g.split("_")
    return (kind != "random", int(n[1:]), float(d[1:]))


def mean(xs):
    return sum(xs) / len(xs)


def se(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def values(th_map, t, seeds, col="basin_best"):
    return [float(th_map[t][s][col]) for s in seeds if s in th_map[t]]


def cv_best(th_map, seeds):
    half = len(seeds) // 2
    folds = [seeds[:half], seeds[half:]]
    out = []
    for a, b in [(0, 1), (1, 0)]:
        t = min(th_map, key=lambda x: mean(values(th_map, x, folds[a])))
        out.append(mean(values(th_map, t, folds[b])))
    return mean(out)


def table(rows, step):
    data = rows[step]
    graphs = sorted({g for g, _ in data}, key=graph_key)
    variants = [v for v in ORDER if any((g, v) in data for g in graphs)]
    seeds = sorted({s for m in data.values() for t in m.values() for s in t})
    print(f"# checkpoint={step}: 交差検証した最適Θでの暫定解ベイスン値の平均(括弧内は全シードで選んだ最適Θ)")
    print("graph".ljust(20) + "".join(v.rjust(17) for v in variants))
    rel = collections.defaultdict(dict)
    for g in graphs:
        base = cv_best(data[(g, "none")], seeds)
        cells = []
        for v in variants:
            th_map = data[(g, v)]
            cv = cv_best(th_map, seeds)
            t = min(th_map, key=lambda x: mean(values(th_map, x, seeds)))
            rel[v][g] = (cv / base - 1.0) * 100.0
            cells.append(f"{cv:8.1f}({t:+.1f})".rjust(17))
        print(g.ljust(20) + "".join(cells))
    print("\n# noneに対する相対差の平均[%](負が改善)")
    for fam in ["random", "geometric"]:
        gs = [g for g in graphs if g.startswith(fam)]
        if gs:
            print(fam.ljust(20) + "".join(f"{mean([rel[v][g] for g in gs]):+.2f}%".rjust(17) for v in variants))


def summary(rows, out):
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("graph\tvariant\ttheta\tseeds\tbasin_best_1e5\tbasin_best_1e6\tse_basin_best_1e6\tbest_real_1e6\tacc_rate_1e6\n")
        d6, d5 = rows["1000000"], rows["100000"]
        for g, v in sorted(d6, key=lambda k: (graph_key(k[0]), ORDER.index(k[1]) if k[1] in ORDER else 99)):
            for t in sorted(d6[(g, v)]):
                seeds = sorted(d6[(g, v)][t])
                b6 = values(d6[(g, v)], t, seeds)
                b5 = values(d5[(g, v)], t, seeds)
                real = values(d6[(g, v)], t, seeds, "best_real")
                acc = values(d6[(g, v)], t, seeds, "acc_rate")
                f.write(f"{g}\t{v}\t{t:.2f}\t{len(seeds)}\t{mean(b5):.3f}\t{mean(b6):.3f}\t{se(b6):.3f}\t{mean(real):.3f}\t{mean(acc):.5f}\n")


def option(name):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return None


if __name__ == "__main__":
    window = option("--window")
    window = tuple(map(float, window.split(","))) if window else None
    variants = option("--variants")
    variants = variants.split(",") if variants else None
    graphs = option("--graphs")
    graphs = graphs.split(",") if graphs else None
    data = load(sys.argv[1], window, variants, graphs)
    if option("--summary"):
        summary(data, option("--summary"))
    else:
        step = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "1000000"
        table(data, step)
