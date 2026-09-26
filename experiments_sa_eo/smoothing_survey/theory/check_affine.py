"""命題1(Flip・Swapの近傍平均が一次式になること)と命題2(重みなしグラフでのGu-Huang平滑化の退化)を全列挙で確認する。

使い方: python check_affine.py
"""
import itertools, math, random
import numpy as np

def rand_graph(n, p, seed):
    rng = random.Random(seed)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]

def H(bits, edges, alpha):
    cut = sum(1 for i, j in edges if bits[i] != bits[j])
    m = sum(1 if b else -1 for b in bits)
    return cut + alpha * m * m, cut, m

def check_flip(n=10, p=0.35, alpha=0.05, seed=1):
    E = rand_graph(n, p, seed); ne = len(E)
    worst = 0.0; worst2 = 0.0
    fac2 = (math.comb(n - 2, 2) - 2 * (n - 2) + 1) / math.comb(n, 2)
    for x in range(1 << n):
        b = [(x >> i) & 1 for i in range(n)]
        h, cut, m = H(b, E, alpha)
        avg = 0.0
        for v in range(n):
            b2 = b.copy(); b2[v] ^= 1
            avg += H(b2, E, alpha)[0]
        avg /= n
        pred = (1 - 4 / n) * h + 2 * ne / n + 4 * alpha
        worst = max(worst, abs(avg - pred))
        # distance-2 sphere (all pairs)
        avg2 = 0.0
        for a, c in itertools.combinations(range(n), 2):
            b2 = b.copy(); b2[a] ^= 1; b2[c] ^= 1
            avg2 += H(b2, E, alpha)[0]
        avg2 /= math.comb(n, 2)
        C = ne / 2 + alpha * n   # mean of H over all states
        pred2 = C + fac2 * (h - C)
        worst2 = max(worst2, abs(avg2 - pred2))
    print(f"flip n={n} |E|={ne}: max|avg1 - ((1-4/n)H + 2|E|/n + 4a)| = {worst:.2e}; "
          f"max|avg2 - (C + {fac2:.6f}(H-C))| = {worst2:.2e}")

def check_swap(n=10, p=0.35, alpha=0.05, seed=2):
    E = rand_graph(n, p, seed); ne = len(E); m = n // 2
    slope = 1 - 4 / m + 2 / m ** 2
    const = 2 * ne / m
    worst = 0.0
    for A in itertools.combinations(range(n), m):
        b = [0] * n
        for i in A: b[i] = 1
        h = H(b, E, alpha)[0]
        As = [i for i in range(n) if b[i]]; Bs = [i for i in range(n) if not b[i]]
        avg = 0.0
        for a in As:
            for c in Bs:
                b2 = b.copy(); b2[a] ^= 1; b2[c] ^= 1
                avg += H(b2, E, alpha)[0]
        avg /= m * m
        worst = max(worst, abs(avg - (slope * h + const)))
    print(f"swap n={n} |E|={ne}: max|avg - ((1-8/n+8/n^2)H + 2|E|/(n/2))| = {worst:.2e}  (slope={slope:.6f})")

def check_gu_huang(n=10, p=0.35, alpha=0.05, a=3.0, seed=3):
    E = set(rand_graph(n, p, seed)); ne = len(E)
    dens = ne / math.comb(n, 2)
    w_edge = dens + (1 - dens) ** a      # Gu-Huang on 0/1 data, mean = density
    w_non = dens - dens ** a
    c_a = (1 - dens) ** a + dens ** a
    alpha_eff = alpha - (dens - dens ** a) / 4
    diffs = []
    for x in range(1 << n):
        b = [(x >> i) & 1 for i in range(n)]
        mval = sum(1 if t else -1 for t in b)
        smooth = sum((w_edge if (i, j) in E else w_non) for i, j in itertools.combinations(range(n), 2) if b[i] != b[j])
        smooth += alpha * mval * mval
        cut = sum(1 for i, j in E if b[i] != b[j])
        diffs.append(smooth - (c_a * cut + alpha_eff * mval * mval))
    diffs = np.array(diffs)
    print(f"Gu-Huang a={a}: H_a - (c_a*cut + alpha_eff*M^2) is constant? spread={diffs.max()-diffs.min():.2e}, "
          f"c_a={c_a:.4f}, alpha_eff={alpha_eff:.5f} => MA(T) on H_a == MA(T/c_a) on cut + (alpha_eff/c_a) M^2")

if __name__ == "__main__":
    check_flip(); check_swap(); check_gu_huang()
    for n in (124, 250, 500):
        f = 1 - 4 / n; s = 1 - 8 / n + 8 / n ** 2
        print(f"n={n}: flip all_average dTheta={-math.log10(f):.4f}, swap all_average dTheta={-math.log10(s):.4f}")
