"""提案B(熱核スペクトル平滑化)の小規模全列挙: 等分割上(Swap)の厳密局所最小解の数、最適解が保たれるか、
スペクトル二分割との差、大きいtから小さいtへの継続(continuation)の成功率を調べる。landscape2.pyからも使う。

使い方: python hk_landscape.py [頂点数(既定16)]
"""
import itertools, math, sys
import numpy as np
from scipy.linalg import expm, eigh

def geometric(n, d, rng):
    pts = rng.random((n, 2)); r2 = d / (n * math.pi)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if ((pts[i] - pts[j]) ** 2).sum() <= r2: A[i, j] = A[j, i] = 1
    return A

def erdos(n, d, rng):
    p = d / (n - 1); A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p: A[i, j] = A[j, i] = 1
    return A

def hk_weights(A, t):
    if t == 0: return A.copy()
    L = np.diag(A.sum(1)) - A
    K = expm(-t * L) / t
    np.fill_diagonal(K, 0.0)
    return K * (A.sum() / K.sum())      # keep total weight = 2|E| (sum over ordered pairs)

def balanced_states(n):
    S = []
    for comb in itertools.combinations(range(1, n), n // 2 - 1):   # vertex 0 fixed in A (label symmetry)
        s = -np.ones(n); s[0] = 1; s[list(comb)] = 1; S.append(s)
    return np.array(S)

def energies(S, W):
    return 0.25 * (W.sum() - np.einsum('ki,ij,kj->k', S, W, S))

def swap_min_delta(S, W, chunk=4096):
    out = np.empty(len(S))
    for c0 in range(0, len(S), chunk):
        s = S[c0:c0 + chunk]; h = s @ W; g = s * h           # g_i = flip gain
        D = g[:, :, None] + g[:, None, :] + 2 * W[None]      # swap gain for (i, j)
        mask = (s[:, :, None] > 0) & (s[:, None, :] < 0)     # i in A, j in B
        D = np.where(mask, D, np.inf)
        out[c0:c0 + chunk] = D.reshape(len(s), -1).min(1)
    return out

def descend(s, W):
    """steepest-descent (swap) on energy defined by W, deterministic tie-break (first index)."""
    s = s.copy()
    while True:
        h = s @ W; g = s * h
        A_ = np.where(s > 0)[0]; B_ = np.where(s < 0)[0]
        D = g[A_][:, None] + g[B_][None, :] + 2 * W[np.ix_(A_, B_)]
        k = np.argmin(D)
        if D.flat[k] >= -1e-12: return s
        i, j = A_[k // len(B_)], B_[k % len(B_)]
        s[i], s[j] = -s[i], -s[j]

def canon(s): return tuple(s if s[0] > 0 else -s)

def run(kind, n, d, seed, ts):
    rng = np.random.default_rng(seed)
    A = geometric(n, d, rng) if kind == 'geo' else erdos(n, d, rng)
    if A.sum() == 0: return None
    S = balanced_states(n)
    E0 = energies(S, A); opt = E0.min(); opt_idx = set(np.where(np.abs(E0 - opt) < 1e-9)[0])
    L = np.diag(A.sum(1)) - A; w, U = eigh(L); f = U[:, 1]
    spec = -np.ones(n); spec[np.argsort(-f)[: n // 2]] = 1
    spec_cut = energies(spec[None], A)[0]
    rows = []
    for t in ts:
        W = hk_weights(A, t); Et = energies(S, W)
        md = swap_min_delta(S, W); nmin = int((md > 1e-9).sum())
        best = np.where(np.abs(Et - Et.min()) < 1e-9)[0]
        rows.append((t, nmin, any(b in opt_idx for b in best), E0[best[0]] - opt))
    # continuation: start from argmin at largest t, descend at each smaller t, finish on t=0
    s = S[np.argmin(energies(S, hk_weights(A, ts[-1])))].copy()
    for t in reversed(ts):
        s = descend(s, hk_weights(A, t))
    cont_gap = energies(s[None], A)[0] - opt
    # plain HC from random starts on the exact cut (fraction reaching optimum), for reference
    hc_hits = 0; trials = 50
    for k in range(trials):
        s0 = S[rng.integers(len(S))]
        hc_hits += abs(energies(descend(s0, A)[None], A)[0] - opt) < 1e-9
    return dict(edges=int(A.sum() / 2), opt=opt, spec_gap=spec_cut - opt, rows=rows, cont_gap=cont_gap, hc_rate=hc_hits / trials)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    ts = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    for kind in ('geo', 'er'):
        agg_nmin = np.zeros(len(ts)); agg_keep = np.zeros(len(ts)); res = []
        for seed in range(12):
            r = run(kind, n, 5.0, seed, ts)
            if r is None: continue
            res.append(r)
            agg_nmin += [x[1] for x in r['rows']]; agg_keep += [x[2] for x in r['rows']]
        k = len(res)
        print(f"\n[{kind}] n={n} d=5, instances={k}")
        print("  t        : " + " ".join(f"{t:6.2f}" for t in ts))
        print("  mean #min: " + " ".join(f"{v / k:6.1f}" for v in agg_nmin))
        print("  opt kept : " + " ".join(f"{v / k:6.2f}" for v in agg_keep))
        print(f"  spectral-bisection gap to optimum (mean): {np.mean([r['spec_gap'] for r in res]):.2f}; "
              f"continuation reaches optimum: {np.mean([r['cont_gap'] < 1e-9 for r in res]):.2f} "
              f"(mean gap {np.mean([r['cont_gap'] for r in res]):.2f}); random-start swap-HC success {np.mean([r['hc_rate'] for r in res]):.2f}")
