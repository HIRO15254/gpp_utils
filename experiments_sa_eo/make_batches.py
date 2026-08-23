"""ラウンド用バッチ定義を生成する。

1 ラウンド = 実行シードを 1 本、72 グラフ x 全パラメータ点 x 4 系列に追加する
（1 条件あたり +4 試行）。ラウンドごとに SA / EO の 2 ファイルに分けるのは

  - SA が安い（1 ラウンド 0.3h）ので先に流し切って早期に応答曲線を得るため
  - 途中で時間切れになったときの切り分けを簡単にするため

`cli --batch` は既存 seed をスキップするので、同じファイルを何度流しても冪等。

使い方:
    PYTHONUTF8=1 python experiments_sa_eo/make_batches.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import grid  # noqa: E402

OUT = Path(__file__).parent / "batches"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gs = grid.graphs()
    sa = grid.sa_configs()
    eo = grid.eo_configs()

    index = []
    for k in grid.RUN_SEEDS:
        for tag, cfgs in (("SA", sa), ("EO", eo)):
            spec = {
                "graphs": gs,
                "configs": cfgs,
                "seed_start": k,
                "seed_count": 1,
                "save_states": False,
            }
            p = OUT / f"round{k:02d}__{tag}.json"
            p.write_text(json.dumps(spec, indent=1), encoding="utf-8")
            index.append({"round": k, "tag": tag, "file": str(p).replace("\\", "/"),
                          "jobs": len(gs) * len(cfgs)})

    (OUT / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    total = sum(e["jobs"] for e in index)
    print(f"{len(index)} バッチを {OUT} に生成（{len(grid.RUN_SEEDS)} ラウンド x 2）")
    print(f"1 ラウンド: SA {len(gs) * len(sa)} + EO {len(gs) * len(eo)} = "
          f"{len(gs) * (len(sa) + len(eo))} ジョブ")
    print(f"全ラウンド合計: {total} ジョブ")


if __name__ == "__main__":
    main()
