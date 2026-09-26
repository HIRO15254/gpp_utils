"""命題1の補足: Flipの半径3の球面平均、独立ランダム反転の期待値、Swapの距離2クラスの平均が一次式になることを全列挙で確認する。

使い方: python check_affine2.py
"""
import itertools, math, random
import numpy as np
from check_affine import rand_graph, H

def flip_r3(n=10, p=0.4, alpha=0.05, seed=7):
    E = rand_graph(n, p, seed); C = len(E) / 2 + alpha * n
    f3 = (math.comb(n - 2, 3) - 2 * math.comb(n - 2, 2) + math.comb(n - 2, 1)) / math.comb(n, 3)
    worst = 0.0
    for x in range(0, 1 << n, 7):
        b = [(x >> i) & 1 for i in range(n)]; h = H(b, E, alpha)[0]; tot = 0.0
        for R in itertools.combinations(range(n), 3):
            b2 = b.copy()
            for r in R: b2[r] ^= 1
            tot += H(b2, E, alpha)[0]
        worst = max(worst, abs(tot / math.comb(n, 3) - (C + f3 * (h - C))))
    print(f"flip r=3 sphere: factor={f3:.6f}, max dev={worst:.2e}")

def noise(n=9, p=0.4, alpha=0.05, q=0.13, seed=8):
    E = rand_graph(n, p, seed); C = len(E) / 2 + alpha * n
    worst = 0.0
    for x in range(0, 1 << n, 5):
        b = [(x >> i) & 1 for i in range(n)]; h = H(b, E, alpha)[0]; tot = 0.0
        for y in range(1 << n):
            k = bin(y).count('1'); w = q ** k * (1 - q) ** (n - k)
            b2 = [b[i] ^ ((y >> i) & 1) for i in range(n)]
            tot += w * H(b2, E, alpha)[0]
        worst = max(worst, abs(tot - (C + (1 - 2 * q) ** 2 * (h - C))))
    print(f"independent flip noise q={q}: factor={(1-2*q)**2:.6f}, max dev={worst:.2e}")

def swap_d2(n=10, p=0.4, alpha=0.05, seed=9):
    E = rand_graph(n, p, seed); m = n // 2
    rows = []
    for A in itertools.combinations(range(n), m):
        b = [0] * n
        for i in A: b[i] = 1
        As = [i for i in range(n) if b[i]]; Bs = [i for i in range(n) if not b[i]]
        vals = []
        for a1, a2 in itertools.combinations(As, 2):
            for c1, c2 in itertools.combinations(Bs, 2):
                b2 = b.copy()
                for z in (a1, a2, c1, c2): b2[z] ^= 1
                vals.append(H(b2, E, alpha)[0])
        rows.append((H(b, E, alpha)[0], np.mean(vals)))
    X = np.array(rows); A_ = np.vstack([X[:, 0], np.ones(len(X))]).T
    coef, res, *_ = np.linalg.lstsq(A_, X[:, 1], rcond=None)
    print(f"swap distance-2 class: fitted slope={coef[0]:.6f} const={coef[1]:.4f}, max residual={np.abs(A_ @ coef - X[:, 1]).max():.2e}")

flip_r3(); noise(); swap_d2()
