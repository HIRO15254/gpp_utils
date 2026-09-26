"""予備実験の前処理: smoothing_pilot graphs が書き出した辺リストから、各グラフのラプラシアンを固有分解して保存する。

出力 <outdir>/<graph>.bin は little-endian f64 で、固有値 n 個の後に固有ベクトル行列 U(行優先、U[i*n+k] は
k 番目の固有ベクトルの i 成分)が続く。smoothing_pilot run が熱核平滑化の重みを作るときに読む。

使い方: python eig.py <graphs_dir> <outdir>
"""

import glob
import os
import sys

import numpy as np


def main(graph_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(graph_dir, "*.edges"))):
        lines = open(path, encoding="utf-8").read().split("\n")
        n, _ = map(int, lines[0].split())
        a = np.zeros((n, n))
        for line in lines[1:]:
            if line.strip():
                i, j = map(int, line.split())
                a[i, j] = a[j, i] = 1.0
        lap = np.diag(a.sum(1)) - a
        w, u = np.linalg.eigh(lap)
        name = os.path.basename(path)[: -len(".edges")]
        np.concatenate([w, u.reshape(-1)]).astype("<f8").tofile(os.path.join(out_dir, f"{name}.bin"))
        print(f"{name}: lambda2={w[1]:.4f} lambda_max={w[-1]:.2f} components={int((w < 1e-9).sum())}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
