"""提案AのSwap版の式の確認: 等分割 s について、交換近傍のボルツマン和
    sum_{a in A, b in B} exp(-(H(s^{ab}) - H(s))/lam)
が Q_A Q_B + sum_{カット辺(a,b)} exp(-(g_a+g_b)/lam) (exp(-2/lam) - 1) に等しいことを、全等分割で確かめる。
g_v = d_v - 2 c_v はカット部分の反転利得(等分割では罰金は変わらない)。補正項は g_a+g_b の値ごとの整数
ヒストグラムで表せることも同時に確かめる。

使い方: python check_swap_softmin.py
"""

import itertools
import math

import numpy as np

from hk_landscape import erdos, geometric


def main():
    worst = 0.0
    for kind in ("geo", "er"):
        for seed in range(4):
            n = 12
            rng = np.random.default_rng(500 + seed)
            A = geometric(n, 4.0, rng) if kind == "geo" else erdos(n, 4.0, rng)
            for lam in (0.3, 1.0, 3.0):
                for comb in itertools.combinations(range(n), n // 2):
                    s = -np.ones(n)
                    s[list(comb)] = 1
                    cut = 0.25 * (A.sum() - s @ A @ s)
                    ina = [i for i in range(n) if s[i] > 0]
                    inb = [i for i in range(n) if s[i] < 0]
                    # 直接計算
                    direct = 0.0
                    for a in ina:
                        for b in inb:
                            t = s.copy()
                            t[a], t[b] = -t[a], -t[b]
                            direct += math.exp(-((0.25 * (A.sum() - t @ A @ t)) - cut) / lam)
                    # 式
                    g = s * (A @ s)  # d_v - 2 c_v
                    qa = sum(math.exp(-g[a] / lam) for a in ina)
                    qb = sum(math.exp(-g[b] / lam) for b in inb)
                    hist = {}
                    for a in ina:
                        for b in inb:
                            if A[a, b]:
                                k = int(round(g[a] + g[b]))
                                hist[k] = hist.get(k, 0) + 1
                    corr = sum(c * math.exp(-k / lam) for k, c in hist.items()) * (math.exp(-2 / lam) - 1)
                    worst = max(worst, abs(direct - (qa * qb + corr)) / direct)
    print(f"Swap soft-min formula: max relative deviation = {worst:.2e}")


if __name__ == "__main__":
    main()
