"""台地を考慮した地形統計(小規模全列挙)。
(1) swap: 熱核平滑化H_tの台地込み局所最小解の数と最適解の保存。
(2) flip: 近傍ソフトミンF_{lam,kappa}とエロージョンF_0について、台地込み局所最小解の数と、
    全初期解からの最急降下(Fで下った後にHで下る)が最適解に到達する割合。

使い方: python landscape2.py swap 16 / python landscape2.py flip 16
"""
import itertools, math, sys
from collections import deque
import numpy as np
from scipy.linalg import expm
from hk_landscape import geometric, erdos, hk_weights

TOL = 1e-9

def plateau_minima(E, nbrs):
    """E: energies (N,), nbrs(i) -> iterable of neighbour indices. Count connected equal-energy components
    with no strictly lower neighbour."""
    N = len(E); seen = np.zeros(N, bool); count = 0
    has_lower = np.zeros(N, bool)
    for i in range(N):
        for j in nbrs(i):
            if E[j] < E[i] - TOL: has_lower[i] = True; break
    for i in range(N):
        if seen[i] or has_lower[i]: continue
        # BFS over equal-energy neighbours
        comp_ok = True; q = deque([i]); seen[i] = True
        while q:
            k = q.popleft()
            if has_lower[k]: comp_ok = False
            for j in nbrs(k):
                if not seen[j] and abs(E[j] - E[k]) <= TOL:
                    seen[j] = True; q.append(j)
        count += comp_ok
    return count

# ---------- (1) swap, heat kernel ----------
def swap_space(n):
    masks = []
    for comb in itertools.combinations(range(1, n), n // 2 - 1):
        m = 1
        for c in comb: m |= 1 << c
        masks.append(m)
    idx = {m: k for k, m in enumerate(masks)}
    full = (1 << n) - 1
    def nb(k, cache={}):
        if k in cache: return cache[k]
        m = masks[k]; A = [i for i in range(n) if m >> i & 1]; B = [i for i in range(n) if not m >> i & 1]
        out = []
        for i in A:
            for j in B:
                m2 = m ^ (1 << i) ^ (1 << j)
                if not m2 & 1: m2 = full ^ m2
                out.append(idx[m2])
        cache[k] = out; return out
    S = np.array([[1.0 if m >> i & 1 else -1.0 for i in range(n)] for m in masks])
    return S, nb

def energies(S, W): return 0.25 * (W.sum() - np.einsum('ki,ij,kj->k', S, W, S))

def run_swap(n, seeds, ts):
    for kind in ('geo', 'er'):
        S, nb = swap_space(n)
        tab = np.zeros((len(seeds), len(ts))); keep = np.zeros((len(seeds), len(ts)))
        for a, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            A = geometric(n, 5.0, rng) if kind == 'geo' else erdos(n, 5.0, rng)
            E0 = energies(S, A); opt = set(np.where(E0 <= E0.min() + TOL)[0])
            for b, t in enumerate(ts):
                Et = energies(S, hk_weights(A, t))
                tab[a, b] = plateau_minima(Et, nb)
                keep[a, b] = any(k in opt for k in np.where(Et <= Et.min() + TOL)[0])
        print(f"[swap/{kind}] n={n}: t=" + " ".join(f"{t:5.2f}" for t in ts))
        print(f"   mean #plateau-minima: " + " ".join(f"{v:5.1f}" for v in tab.mean(0)))
        print(f"   optimum kept        : " + " ".join(f"{v:5.2f}" for v in keep.mean(0)))

# ---------- (2) flip + penalty, neighbourhood soft-min ----------
def flip_tables(A, alpha):
    n = len(A); N = 1 << n
    bits = ((np.arange(N)[:, None] >> np.arange(n)[None]) & 1).astype(float)
    S = 2 * bits - 1
    cut = 0.25 * (A.sum() - np.einsum('ki,ij,kj->k', S, A, S))
    M = S.sum(1)
    return cut + alpha * M * M

def soft_min(H, n, lam, kappa):
    N = len(H); idx = np.arange(N)
    nbE = np.stack([H[idx ^ (1 << v)] for v in range(n)], 1)          # (N, n)
    if lam == 0:   # erosion (kappa>0)
        return np.minimum(H, nbE.min(1))
    m = np.minimum(H, nbE.min(1))
    z = np.exp(-(H - m) / lam) + kappa * np.exp(-(nbE - m[:, None]) / lam).sum(1)
    return m - lam * np.log(z)

def descent_endpoints(F, n):
    N = len(F); idx = np.arange(N)
    nbF = np.stack([F[idx ^ (1 << v)] for v in range(n)], 1)
    best_v = nbF.argmin(1); best = nbF[idx, best_v]
    nxt = np.where(best < F - TOL, idx ^ (1 << best_v), idx)
    for _ in range(4 * n):            # pointer jumping to the fixed point
        nxt = nxt[nxt]
    return nxt

def run_flip(n, seeds, settings, alpha=0.05):
    for kind in ('geo', 'er'):
        res = {name: [] for name, _, _ in settings}; resmin = {name: [] for name, _, _ in settings}
        for seed in seeds:
            rng = np.random.default_rng(100 + seed)
            A = geometric(n, 5.0, rng) if kind == 'geo' else erdos(n, 5.0, rng)
            H = flip_tables(A, alpha); opt = H.min()
            endH = descent_endpoints(H, n)
            nbfun = lambda i: [i ^ (1 << v) for v in range(n)]
            for name, lam, kappa in settings:
                F = H if name == 'none' else soft_min(H, n, lam, kappa)
                end = descent_endpoints(F, n)
                end = endH[end]                  # then descend on the real objective
                res[name].append(float(np.mean(H[end] <= opt + TOL)))
                resmin[name].append(plateau_minima(F, nbfun))
        print(f"[flip/{kind}] n={n} alpha={alpha}: fraction of all 2^n starts reaching the optimum / mean #plateau minima")
        for name, _, _ in settings:
            print(f"   {name:22s}: success {np.mean(res[name]):.3f}   minima {np.mean(resmin[name]):7.1f}")

if __name__ == "__main__":
    which = sys.argv[1]
    if which == 'swap':
        run_swap(int(sys.argv[2]), range(8), [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
    else:
        run_flip(int(sys.argv[2]), range(8), [('none', None, None), ('erosion lam->0', 0, 1.0),
                 ('softmin lam=0.5 k=1', 0.5, 1.0), ('softmin lam=1 k=1', 1.0, 1.0), ('softmin lam=0.5 k=0.1', 0.5, 0.1)])
