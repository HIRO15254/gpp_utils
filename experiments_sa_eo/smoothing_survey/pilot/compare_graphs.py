"""予備実験のグラフが gpp の保存グラフと一致するかを確かめる。

gpp でベースラインの18グラフを生成し(例: max_steps=1 の実験を data ルートへ流す)、その graphs/*.json の辺リストと、
smoothing_pilot graphs が書き出した *.edges を突き合わせる。

使い方: python compare_graphs.py <gpp_root>/graphs <pilot_graphs_dir>
"""

import glob
import json
import os
import sys


def main(gpp_graphs: str, pilot_graphs: str) -> int:
    pilot = {}
    for path in glob.glob(os.path.join(pilot_graphs, "*.edges")):
        lines = open(path, encoding="utf-8").read().split("\n")
        edges = sorted(tuple(map(int, line.split())) for line in lines[1:] if line.strip())
        pilot[tuple(edges)] = os.path.basename(path)
    matched = 0
    files = sorted(glob.glob(os.path.join(gpp_graphs, "*.json")))
    for path in files:
        g = json.load(open(path, encoding="utf-8"))
        edges = tuple(sorted(tuple(e) for e in g["edges"]))
        name = pilot.get(edges)
        matched += name is not None
        print(f"{os.path.basename(path)[:16]}  n={g['node_count']}  m={len(edges)}  {name or 'NO MATCH'}")
    print(f"一致 {matched} / {len(files)}")
    return 0 if matched == len(files) == len(pilot) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
