"""命題3: gppのSA + random_k_average(Flip)をengine.rsと同じ手順で動かし、経験分布をpseudo-marginalの理論分布
pi_K(s) ~ E_S[exp(-mean_{w in S} H(s^w)/T)] と比べる。K=1では影の解 y = s^w が exp(-H/T) に従うことも確かめる。

使い方: python check_pseudo_marginal.py [ステップ数(既定 2000000)]
"""
import itertools, math, random, sys
import numpy as np

def build(n, p, alpha, seed):
    rng = random.Random(seed)
    E = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    Ht = np.zeros(1 << n)
    for x in range(1 << n):
        cut = sum(1 for i, j in E if ((x >> i) & 1) != ((x >> j) & 1))
        m = 2 * bin(x).count("1") - n
        Ht[x] = cut + alpha * m * m
    return E, Ht

def theory(n, Ht, T, K):
    w = np.zeros(1 << n)
    subsets = list(itertools.combinations(range(n), K))
    for x in range(1 << n):
        nb = [Ht[x ^ (1 << v)] for v in range(n)]
        w[x] = np.mean([math.exp(-sum(nb[i] for i in S) / (K * T)) for S in subsets])
    return w / w.sum()

def boltz(E, T):
    w = np.exp(-(E - E.min()) / T); return w / w.sum()

def simulate(n, Ht, T, K, steps, seed):
    rng = random.Random(seed)
    nbr = [[Ht[x ^ (1 << v)] for v in range(n)] for x in range(1 << n)]
    s = rng.randrange(1 << n)
    idx = rng.sample(range(n), K)
    cur = sum(nbr[s][i] for i in idx) / K
    shadow_w = idx[0]
    hist = np.zeros(1 << n); hist_y = np.zeros(1 << n)
    for _ in range(steps):
        v = rng.randrange(n)
        s2 = s ^ (1 << v)
        idx = rng.sample(range(n), K)
        est = sum(nbr[s2][i] for i in idx) / K
        d = est - cur
        if d < 0 or rng.random() < math.exp(-d / T):
            s, cur, shadow_w = s2, est, idx[0]
        hist[s] += 1
        hist_y[s ^ (1 << shadow_w)] += 1
    return hist / steps, hist_y / steps

def tv(p, q): return 0.5 * np.abs(p - q).sum()

if __name__ == "__main__":
    n, p, alpha = 8, 0.45, 0.05
    E, Ht = build(n, p, alpha, seed=5)
    print(f"n={n} |E|={len(E)} Hmin={Ht.min()}")
    fmean = np.array([np.mean([Ht[x ^ (1 << v)] for v in range(n)]) for x in range(1 << n)])
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    for T in (0.7, 1.5):
        for K in (1, 2, 4):
            th = theory(n, Ht, T, K)
            emp, emp_y = simulate(n, Ht, T, K, steps, seed=11 + K)
            line = (f"T={T} K={K}: TV(emp, pseudo-marginal)={tv(emp, th):.4f}  "
                    f"TV(emp, exp(-mean/T))={tv(emp, boltz(fmean, T)):.4f}  TV(emp, exp(-H/T))={tv(emp, boltz(Ht, T)):.4f}")
            if K == 1:
                line += f"  | shadow y=s^w: TV(y, exp(-H/T))={tv(emp_y, boltz(Ht, T)):.4f}"
            print(line, flush=True)
