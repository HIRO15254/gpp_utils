"""命題4(6)の確認: エロージョン F_0(s) = min_{y in N[s]} H(y) (Flip + 罰金) について、全列挙で次を確かめる。
(a) F_0 の局所最小解(より小さい近傍を持たない解) s の読み出し y*(s) = argmin_{N[s]} H は H の局所最小解である。
(b) ハミング距離2・3の H の局所最小解の組には、F_0 が max(H(y1), H(y2)) を超えない経路がある。
    (距離4以上では成り立たない例があることも数える。)

使い方: python check_erosion.py
"""

import itertools

import numpy as np

from hk_landscape import erdos, geometric


def tables(A, alpha):
    n = len(A)
    N = 1 << n
    bits = ((np.arange(N)[:, None] >> np.arange(n)[None]) & 1).astype(float)
    S = 2 * bits - 1
    H = 0.25 * (A.sum() - np.einsum("ki,ij,kj->k", S, A, S)) + alpha * S.sum(1) ** 2
    idx = np.arange(N)
    nb = np.stack([H[idx ^ (1 << v)] for v in range(n)], 1)
    F0 = np.minimum(H, nb.min(1))
    return H, F0


def path_ok(y1, y2, F0, bound, n):
    """y1 から y2 へ差のビットを1つずつ反転する全順序の経路のうち、F0 <= bound を保つものがあるか。"""
    diff = [b for b in range(n) if (y1 ^ y2) >> b & 1]
    for order in itertools.permutations(diff):
        s, ok = y1, True
        for b in order:
            s ^= 1 << b
            if F0[s] > bound + 1e-9:
                ok = False
                break
        if ok:
            return True
    return False


def main():
    alpha, tol = 0.05, 1e-9
    for kind in ("geo", "er"):
        for seed in range(6):
            n = 12
            rng = np.random.default_rng(300 + seed)
            A = geometric(n, 4.0, rng) if kind == "geo" else erdos(n, 4.0, rng)
            H, F0 = tables(A, alpha)
            N = len(H)
            idx = np.arange(N)
            nbH = np.stack([H[idx ^ (1 << v)] for v in range(n)], 1)
            nbF = np.stack([F0[idx ^ (1 << v)] for v in range(n)], 1)
            h_min = np.where(nbH.min(1) >= H - tol)[0]          # H の局所最小解(非厳密)
            f_min = np.where(nbF.min(1) >= F0 - tol)[0]          # F0 の局所最小解(非厳密)
            # (a)
            bad_a = 0
            for s in f_min:
                cand = [s] + [s ^ (1 << v) for v in range(n)]
                y = min(cand, key=lambda x: H[x])
                bad_a += not (nbH[y].min() >= H[y] - tol)
            # (b)
            hm = set(h_min.tolist())
            counts = {2: [0, 0], 3: [0, 0], 4: [0, 0]}
            for y1 in h_min:
                for d in (2, 3, 4):
                    for flips in itertools.combinations(range(n), d):
                        y2 = y1
                        for b in flips:
                            y2 ^= 1 << b
                        if y2 <= y1 or y2 not in hm:
                            continue
                        ok = path_ok(int(y1), int(y2), F0, max(H[y1], H[y2]), n)
                        counts[d][0] += 1
                        counts[d][1] += ok
            print(f"[{kind} seed={seed}] F0-minima={len(f_min)} readout-not-H-min={bad_a} | "
                  + " ".join(f"d={d}: {c[1]}/{c[0]} pairs barrier-free" for d, c in counts.items()))


if __name__ == "__main__":
    main()
