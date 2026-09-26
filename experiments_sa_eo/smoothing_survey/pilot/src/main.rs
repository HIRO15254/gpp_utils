//! Pilot prototype (NOT part of gpp_utils): fixed-temperature Metropolis on GPP (flip neighbourhood,
//! H = cut + alpha*M^2) with several smoothing variants, to sanity-check proposals of the survey.
use rand::Rng;
use rand_mt::Mt64;
use rayon::prelude::*;
use std::io::{Read, Write};
use std::sync::Arc;

const ALPHA: f64 = 0.05;
const MAX_BASIN_STEPS: usize = 10_000;

#[derive(Clone)]
struct Graph {
    name: String,
    n: usize,
    m: usize,
    adj: Vec<Vec<u32>>,
    deg: Vec<i32>,
    maxdeg: i32,
}

fn build(name: String, n: usize, edges: &[(usize, usize)]) -> Graph {
    let mut adj = vec![Vec::new(); n];
    for &(a, b) in edges {
        adj[a].push(b as u32);
        adj[b].push(a as u32);
    }
    let deg: Vec<i32> = adj.iter().map(|x| x.len() as i32).collect();
    let maxdeg = *deg.iter().max().unwrap_or(&0);
    Graph {
        name,
        n,
        m: edges.len(),
        adj,
        deg,
        maxdeg,
    }
}

/// Exact copy of gpp_utils::graph_partition::Graph::generate (rand 0.8.5 + rand_mt 4.2.2).
fn generate(kind: &str, n: usize, d: f64, seed: u64) -> Graph {
    let mut rng = Mt64::new(seed);
    let mut edges = Vec::new();
    match kind {
        "random" => {
            let p = d / (n - 1) as f64;
            for a in 0..n {
                for b in a + 1..n {
                    if rng.gen::<f64>() < p {
                        edges.push((a, b));
                    }
                }
            }
        }
        "geometric" => {
            let mut pts: Vec<(f64, f64)> = Vec::with_capacity(n);
            for _ in 0..n {
                pts.push((rng.gen(), rng.gen()));
            }
            let th2 = d / (n as f64 * std::f64::consts::PI);
            for a in 0..n {
                for b in a + 1..n {
                    let dx = pts[a].0 - pts[b].0;
                    let dy = pts[a].1 - pts[b].1;
                    if dx * dx + dy * dy <= th2 {
                        edges.push((a, b));
                    }
                }
            }
        }
        _ => panic!("unknown kind"),
    }
    build(format!("{kind}_n{n}_d{d}"), n, &edges)
}

fn baseline_graphs() -> Vec<Graph> {
    let mut out = Vec::new();
    for kind in ["random", "geometric"] {
        for n in [124usize, 250, 500] {
            for d in [5.0f64, 10.0, 20.0] {
                out.push(generate(kind, n, d, 0));
            }
        }
    }
    out
}

#[derive(Clone)]
struct St {
    s: Vec<i8>,
    cuts: Vec<i32>,
    cut: i64,
    mm: i64,
}

impl St {
    fn new(g: &Graph, s: Vec<i8>) -> St {
        let mut cuts = vec![0; g.n];
        let mut cut = 0;
        for a in 0..g.n {
            for &b in &g.adj[a] {
                let b = b as usize;
                if a < b && s[a] != s[b] {
                    cuts[a] += 1;
                    cuts[b] += 1;
                    cut += 1;
                }
            }
        }
        let mm = s.iter().map(|&x| x as i64).sum();
        St { s, cuts, cut, mm }
    }
    #[inline]
    fn h(&self) -> f64 {
        self.cut as f64 + ALPHA * (self.mm * self.mm) as f64
    }
    #[inline]
    fn dcut(&self, g: &Graph, v: usize) -> i32 {
        g.deg[v] - 2 * self.cuts[v]
    }
    #[inline]
    fn gain(&self, g: &Graph, v: usize) -> f64 {
        let mm2 = self.mm - 2 * self.s[v] as i64;
        self.dcut(g, v) as f64 + ALPHA * ((mm2 * mm2 - self.mm * self.mm) as f64)
    }
    #[inline]
    fn flip(&mut self, g: &Graph, v: usize) {
        let sv = self.s[v];
        for &u in &g.adj[v] {
            let u = u as usize;
            if self.s[u] != sv {
                self.cuts[u] -= 1
            } else {
                self.cuts[u] += 1
            }
        }
        self.cut += (g.deg[v] - 2 * self.cuts[v]) as i64;
        self.cuts[v] = g.deg[v] - self.cuts[v];
        self.mm -= 2 * sv as i64;
        self.s[v] = -sv;
    }
}

/// Integer bucket counts of single-flip cut gains per side (n-fold-way / FM-style).
#[derive(Clone)]
struct Buckets {
    off: i32,
    cnt: [Vec<u32>; 2],
}
#[inline]
fn side(x: i8) -> usize {
    if x > 0 {
        0
    } else {
        1
    }
}
impl Buckets {
    fn new(g: &Graph, st: &St) -> Buckets {
        let off = g.maxdeg;
        let mut b = Buckets {
            off,
            cnt: [
                vec![0; (2 * off + 1) as usize],
                vec![0; (2 * off + 1) as usize],
            ],
        };
        for v in 0..g.n {
            b.add(g, st, v);
        }
        b
    }
    #[inline]
    fn add(&mut self, g: &Graph, st: &St, v: usize) {
        self.cnt[side(st.s[v])][(st.dcut(g, v) + self.off) as usize] += 1;
    }
    #[inline]
    fn remove(&mut self, g: &Graph, st: &St, v: usize) {
        self.cnt[side(st.s[v])][(st.dcut(g, v) + self.off) as usize] -= 1;
    }
}
fn flip_b(st: &mut St, b: &mut Buckets, g: &Graph, v: usize) {
    b.remove(g, st, v);
    for &u in &g.adj[v] {
        b.remove(g, st, u as usize);
    }
    st.flip(g, v);
    b.add(g, st, v);
    for &u in &g.adj[v] {
        b.add(g, st, u as usize);
    }
}

struct Nsm {
    lam: f64,
    ln_kappa: f64,
    etab: Vec<f64>,
}
impl Nsm {
    fn new(lam: f64, kappa: f64, off: i32) -> Nsm {
        let etab = (0..=(2 * off) as usize)
            .map(|k| {
                if lam > 0.0 {
                    (-(k as f64) / lam).exp()
                } else {
                    0.0
                }
            })
            .collect();
        Nsm {
            lam,
            ln_kappa: kappa.ln(),
            etab,
        }
    }
}
#[inline]
fn side_offset(side: usize, mm: i64) -> f64 {
    let sigma: i64 = if side == 0 { 1 } else { -1 };
    let mm2 = mm - 2 * sigma;
    ALPHA * ((mm2 * mm2 - mm * mm) as f64)
}
/// F = -lam*log( e^{-H/lam} + kappa * sum_w e^{-H(s^w)/lam} );  lam==0 -> erosion min(H, min_w H(s^w)).
/// Returns (F, gmin) where gmin is the smallest single-flip gain.
fn nsm_f(st: &St, b: &Buckets, p: &Nsm) -> (f64, f64) {
    let h = st.h();
    let mut gmin = f64::INFINITY;
    let mut cmin = [usize::MAX; 2];
    for sd in 0..2 {
        if let Some(i) = b.cnt[sd].iter().position(|&c| c > 0) {
            cmin[sd] = i;
            let gs = (i as i32 - b.off) as f64 + side_offset(sd, st.mm);
            if gs < gmin {
                gmin = gs;
            }
        }
    }
    if p.lam == 0.0 {
        return (h + gmin.min(0.0), gmin);
    }
    let mut z = 0.0;
    for sd in 0..2 {
        if cmin[sd] == usize::MAX {
            continue;
        }
        let gs = (cmin[sd] as i32 - b.off) as f64 + side_offset(sd, st.mm);
        let mut acc = 0.0;
        for (k, &c) in b.cnt[sd][cmin[sd]..].iter().enumerate() {
            if c > 0 {
                acc += c as f64 * p.etab[k];
            }
        }
        z += (-(gs - gmin) / p.lam).exp() * acc;
    }
    let x = p.ln_kappa - gmin / p.lam + z.ln();
    let sp = if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    };
    (h - p.lam * sp, gmin)
}

/// Steepest-descent hill climbing on the real objective (flip), lowest index on ties.
fn basin(g: &Graph, s: &[i8]) -> f64 {
    let mut st = St::new(g, s.to_vec());
    for _ in 0..MAX_BASIN_STEPS {
        let mut best = -1e-9;
        let mut bv = usize::MAX;
        for v in 0..g.n {
            let gv = st.gain(g, v);
            if gv < best {
                best = gv;
                bv = v;
            }
        }
        if bv == usize::MAX {
            break;
        }
        st.flip(g, bv);
    }
    st.h()
}

#[derive(Clone, Debug)]
enum Variant {
    None,
    RandomK(usize),
    Nsm { rho: f64, kappa: f64 }, // lam = rho * T (rho = 0 -> erosion)
    HkStatic(usize),              // index into per-graph matrices
    HkSched,                      // uses the schedule matrices
}

fn variant_list() -> Vec<(String, Variant)> {
    vec![
        ("none".into(), Variant::None),
        ("rk1".into(), Variant::RandomK(1)),
        ("rk4".into(), Variant::RandomK(4)),
        (
            "nsm_rho1".into(),
            Variant::Nsm {
                rho: 1.0,
                kappa: 1.0,
            },
        ),
        (
            "nsm_rho0.5".into(),
            Variant::Nsm {
                rho: 0.5,
                kappa: 1.0,
            },
        ),
        (
            "nsm_erosion".into(),
            Variant::Nsm {
                rho: 0.0,
                kappa: 1.0,
            },
        ),
        ("hk_tau0.5".into(), Variant::HkStatic(0)),
        ("hk_tau2".into(), Variant::HkStatic(1)),
        ("hk_sched".into(), Variant::HkSched),
    ]
}
const HK_STATIC_TAUS: [f64; 2] = [0.5, 2.0];
const HK_SCHED_TAUS: [f64; 9] = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0];
const HK_SCHED_STEPS: [u64; 9] = [
    100_000, 100_000, 100_000, 100_000, 100_000, 100_000, 100_000, 100_000, 200_000,
];

struct Eig {
    lam: Vec<f64>,
    u: Vec<f64>, // row-major, u[i*n+k] = component i of eigenvector k
}
fn read_eig(path: &str, n: usize) -> Eig {
    let mut f = std::fs::File::open(path).unwrap_or_else(|_| panic!("missing {path}"));
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let vals: Vec<f64> = buf
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals.len(), n + n * n, "eig size mismatch for {path}");
    Eig {
        lam: vals[..n].to_vec(),
        u: vals[n..].to_vec(),
    }
}
/// Normalised heat-kernel weights: offdiag(exp(-tL)) with t = tau / mean_degree, scaled to total 2m.
/// tau == 0 returns the dense adjacency matrix.
fn hk_matrix(g: &Graph, e: &Eig, tau: f64) -> Vec<f64> {
    let n = g.n;
    let mut w = vec![0.0; n * n];
    if tau == 0.0 {
        for a in 0..n {
            for &b in &g.adj[a] {
                w[a * n + b as usize] = 1.0;
            }
        }
        return w;
    }
    let dbar = 2.0 * g.m as f64 / n as f64;
    let t = tau / dbar;
    let sq: Vec<f64> = e
        .lam
        .iter()
        .map(|&l| (-t * l.max(0.0) * 0.5).exp())
        .collect();
    let mut x = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            x[i * n + k] = e.u[i * n + k] * sq[k];
        }
    }
    let mut total = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let (ri, rj) = (&x[i * n..(i + 1) * n], &x[j * n..(j + 1) * n]);
            let v: f64 = ri.iter().zip(rj).map(|(a, b)| a * b).sum();
            let v = v.max(0.0); // heat kernel is entrywise non-negative; clip round-off
            w[i * n + j] = v;
            w[j * n + i] = v;
            total += 2.0 * v;
        }
    }
    let scale = 2.0 * g.m as f64 / total;
    for v in w.iter_mut() {
        *v *= scale;
    }
    w
}

struct Out {
    step: u64,
    best_real: f64,
    basin_best: f64,
    best_readout: f64,
    acc: f64,
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn run(
    g: &Graph,
    var: &Variant,
    t: f64,
    seed: u64,
    checkpoints: &[u64],
    hk_static: &[Arc<Vec<f64>>],
    hk_sched: &[Arc<Vec<f64>>],
    init_seed: Option<u64>,
) -> Vec<Out> {
    let mut rng = Mt64::new(seed);
    let n = g.n;
    // With `init_seed`, the initial partition depends only on (graph, run seed), as in gpp
    // (common random numbers across variants and temperatures).
    let s0: Vec<i8> = match init_seed {
        Some(x) => {
            let mut r0 = Mt64::new(x);
            (0..n)
                .map(|_| if r0.gen::<bool>() { 1 } else { -1 })
                .collect()
        }
        None => (0..n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect(),
    };
    let mut st = St::new(g, s0);
    let mut best_real = st.h();
    let mut best_state = st.s.clone();
    let mut best_readout = f64::NAN;
    let mut accepted: u64 = 0;
    let mut outs = Vec::new();
    let total = *checkpoints.last().unwrap();
    let mut ck = 0usize;
    macro_rules! record {
        ($step:expr) => {
            while ck < checkpoints.len() && checkpoints[ck] == $step {
                outs.push(Out {
                    step: $step,
                    best_real,
                    basin_best: basin(g, &best_state),
                    best_readout,
                    acc: accepted as f64 / ($step.max(1)) as f64,
                });
                ck += 1;
            }
        };
    }
    macro_rules! improve {
        () => {
            let hr = st.h();
            if hr < best_real {
                best_real = hr;
                best_state.copy_from_slice(&st.s);
            }
        };
    }
    record!(0u64);
    match var {
        Variant::None => {
            for step in 1..=total {
                let v = rng.gen_range(0..n);
                let delta = st.gain(g, v);
                if delta < 0.0 || (t > 0.0 && rng.gen::<f64>() < (-delta / t).exp()) {
                    st.flip(g, v);
                    accepted += 1;
                    improve!();
                }
                record!(step);
            }
        }
        Variant::RandomK(k) => {
            let k = *k;
            assert!(
                k >= 1 && k <= n,
                "random_k needs 1 <= K <= n (distance-1 samples only)"
            );
            let mut pick = vec![0usize; k];
            let est = |st: &St, rng: &mut Mt64, pick: &mut Vec<usize>| -> f64 {
                let mut c = 0;
                while c < k {
                    let w = rng.gen_range(0..n);
                    if !pick[..c].contains(&w) {
                        pick[c] = w;
                        c += 1;
                    }
                }
                st.h() + pick.iter().map(|&w| st.gain(g, w)).sum::<f64>() / k as f64
            };
            let mut cur = est(&st, &mut rng, &mut pick);
            for step in 1..=total {
                let v = rng.gen_range(0..n);
                st.flip(g, v);
                let e = est(&st, &mut rng, &mut pick);
                let delta = e - cur;
                if delta < 0.0 || (t > 0.0 && rng.gen::<f64>() < (-delta / t).exp()) {
                    cur = e;
                    accepted += 1;
                    improve!();
                } else {
                    st.flip(g, v);
                }
                record!(step);
            }
        }
        Variant::Nsm { rho, kappa } => {
            let p = Nsm::new(rho * t, *kappa, g.maxdeg);
            let mut b = Buckets::new(g, &st);
            let (mut fcur, gm) = nsm_f(&st, &b, &p);
            best_readout = st.h() + gm.min(0.0);
            for step in 1..=total {
                let v = rng.gen_range(0..n);
                flip_b(&mut st, &mut b, g, v);
                let (fnew, gm) = nsm_f(&st, &b, &p);
                let delta = fnew - fcur;
                if delta < 0.0 || (t > 0.0 && rng.gen::<f64>() < (-delta / t).exp()) {
                    fcur = fnew;
                    accepted += 1;
                    improve!();
                    let rd = st.h() + gm.min(0.0);
                    if rd < best_readout {
                        best_readout = rd;
                    }
                } else {
                    flip_b(&mut st, &mut b, g, v);
                }
                record!(step);
            }
        }
        Variant::HkStatic(_) | Variant::HkSched => {
            let (mats, lens): (Vec<Arc<Vec<f64>>>, Vec<u64>) = match var {
                Variant::HkStatic(i) => (vec![hk_static[*i].clone()], vec![total]),
                _ => (hk_sched.to_vec(), HK_SCHED_STEPS.to_vec()),
            };
            let mut hf = vec![0.0; n];
            let mut step: u64 = 0;
            for (w, len) in mats.iter().zip(lens) {
                for i in 0..n {
                    hf[i] = (0..n).map(|j| w[i * n + j] * st.s[j] as f64).sum();
                }
                for _ in 0..len {
                    step += 1;
                    let v = rng.gen_range(0..n);
                    let sv = st.s[v] as f64;
                    let mm2 = st.mm - 2 * st.s[v] as i64;
                    let delta = sv * hf[v] + ALPHA * ((mm2 * mm2 - st.mm * st.mm) as f64);
                    if delta < 0.0 || (t > 0.0 && rng.gen::<f64>() < (-delta / t).exp()) {
                        let row = &w[v * n..(v + 1) * n];
                        for i in 0..n {
                            hf[i] -= 2.0 * sv * row[i];
                        }
                        st.flip(g, v);
                        accepted += 1;
                        improve!();
                    }
                    record!(step);
                }
            }
        }
    }
    outs
}

fn selftest() {
    let mut rng = Mt64::new(12345);
    for trial in 0..30 {
        let n = 8 + trial % 7;
        let g = generate(
            if trial % 2 == 0 {
                "random"
            } else {
                "geometric"
            },
            n,
            3.0 + (trial % 4) as f64,
            trial as u64,
        );
        let s0: Vec<i8> = (0..n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();
        let mut st = St::new(&g, s0);
        let mut b = Buckets::new(&g, &st);
        for (lam, kappa) in [(0.0, 1.0), (0.3, 1.0), (1.7, 0.2), (5.0, 3.0)] {
            let p = Nsm::new(lam, kappa, g.maxdeg);
            for _ in 0..50 {
                let v = rng.gen_range(0..n);
                flip_b(&mut st, &mut b, &g, v);
                // brute force
                let h = st.h();
                let nb: Vec<f64> = (0..n)
                    .map(|w| {
                        let mut c = st.clone();
                        c.flip(&g, w);
                        c.h()
                    })
                    .collect();
                let brute = if lam == 0.0 {
                    nb.iter().cloned().fold(h, f64::min)
                } else {
                    let m = nb.iter().cloned().fold(h, f64::min);
                    let z = (-(h - m) / lam).exp()
                        + kappa * nb.iter().map(|&x| (-(x - m) / lam).exp()).sum::<f64>();
                    m - lam * z.ln()
                };
                let (f, _) = nsm_f(&st, &b, &p);
                assert!(
                    (f - brute).abs() < 1e-9 * (1.0 + brute.abs()),
                    "nsm mismatch n={n} lam={lam}: {f} vs {brute}"
                );
                let fresh = Buckets::new(&g, &st);
                assert_eq!(fresh.cnt, b.cnt, "bucket drift");
                let full = St::new(&g, st.s.clone());
                assert_eq!((full.cut, full.mm), (st.cut, st.mm));
                assert_eq!(full.cuts, st.cuts);
            }
        }
    }
    // heat-kernel gain vs direct energy difference with a random symmetric W
    for trial in 0..10 {
        let n = 12 + trial;
        let g = generate("random", n, 4.0, 100 + trial as u64);
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let x: f64 = rng.gen();
                w[i * n + j] = x;
                w[j * n + i] = x;
            }
        }
        let energy = |s: &[i8]| -> f64 {
            let mut e = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    if s[i] != s[j] {
                        e += w[i * n + j];
                    }
                }
            }
            let mm: i64 = s.iter().map(|&x| x as i64).sum();
            e + ALPHA * (mm * mm) as f64
        };
        let s: Vec<i8> = (0..n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();
        let st = St::new(&g, s.clone());
        let hf: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| w[i * n + j] * s[j] as f64).sum())
            .collect();
        for v in 0..n {
            let mut s2 = s.clone();
            s2[v] = -s2[v];
            let mm2 = st.mm - 2 * s[v] as i64;
            let d = s[v] as f64 * hf[v] + ALPHA * ((mm2 * mm2 - st.mm * st.mm) as f64);
            assert!(
                (d - (energy(&s2) - energy(&s))).abs() < 1e-9,
                "hk gain mismatch"
            );
        }
    }
    println!("selftest ok");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("selftest") => selftest(),
        Some("graphs") => {
            let dir = &args[2];
            std::fs::create_dir_all(dir).unwrap();
            for g in baseline_graphs() {
                let mut f = std::fs::File::create(format!("{dir}/{}.edges", g.name)).unwrap();
                writeln!(f, "{} {}", g.n, g.m).unwrap();
                for a in 0..g.n {
                    for &b in &g.adj[a] {
                        if (a as u32) < b {
                            writeln!(f, "{a} {b}").unwrap();
                        }
                    }
                }
                println!("{}\tn={}\tm={}\tmean_deg={:.3}\tmax_deg={}", g.name, g.n, g.m, 2.0 * g.m as f64 / g.n as f64, g.maxdeg);
            }
        }
        Some("run") => {
            // run <eigdir> <graph-filter substrings comma> <variants comma|all> <theta_lo> <theta_hi> <theta_step_x100> <seeds>
            let eigdir = &args[2];
            let gfilter: Vec<&str> = args[3].split(',').collect();
            let vfilter: Vec<&str> = args[4].split(',').collect();
            let (lo, hi, st100): (i64, i64, i64) = (args[5].parse().unwrap(), args[6].parse().unwrap(), args[7].parse().unwrap());
            let nseeds: u64 = args[8].parse().unwrap();
            let seed_offset: u64 = args.get(9).map(|x| x.parse().unwrap()).unwrap_or(0);
            let crn = args.get(10).map(|x| x == "crn").unwrap_or(false);
            let checkpoints = [0u64, 1_000, 10_000, 100_000, 1_000_000];
            let variants: Vec<(usize, String, Variant)> = variant_list()
                .into_iter()
                .enumerate()
                .filter(|(_, (name, _))| vfilter.contains(&"all") || vfilter.contains(&name.as_str()))
                .map(|(i, (a, b))| (i, a, b))
                .collect();
            let thetas: Vec<i64> = (0..).map(|k| lo + k * st100).take_while(|&x| x <= hi).collect(); // theta*100
            println!("graph\tvariant\ttheta\tseed\tstep\tbest_real\tbasin_best\tbest_readout\tacc_rate");
            for (gi, g) in baseline_graphs().into_iter().enumerate() {
                if !gfilter.iter().any(|f| *f == "all" || g.name.contains(f)) {
                    continue;
                }
                let need_hk = variants.iter().any(|(_, _, v)| matches!(v, Variant::HkStatic(_) | Variant::HkSched));
                let (hk_static, hk_sched) = if need_hk {
                    let e = read_eig(&format!("{eigdir}/{}.bin", g.name), g.n);
                    let a: Vec<Arc<Vec<f64>>> = HK_STATIC_TAUS.iter().map(|&tau| Arc::new(hk_matrix(&g, &e, tau))).collect();
                    let b: Vec<Arc<Vec<f64>>> = HK_SCHED_TAUS.iter().map(|&tau| Arc::new(hk_matrix(&g, &e, tau))).collect();
                    (a, b)
                } else {
                    (vec![], vec![])
                };
                let mut tasks = Vec::new();
                for (vi, vname, var) in &variants {
                    for &th in &thetas {
                        for seed in seed_offset..seed_offset + nseeds {
                            tasks.push((*vi, vname.clone(), var.clone(), th, seed));
                        }
                    }
                }
                let lines: Vec<String> = tasks
                    .par_iter()
                    .map(|(vi, vname, var, th, seed)| {
                        let t = 10f64.powf(*th as f64 / 100.0);
                        let rs = mix(mix(gi as u64) ^ mix(1000 + *vi as u64) ^ mix(100_000 + (*th + 1000) as u64) ^ mix(10_000_000 + *seed));
                        let init = crn.then(|| mix(mix(gi as u64) ^ mix(20_000_000 + *seed)));
                        let outs = run(&g, var, t, rs, &checkpoints, &hk_static, &hk_sched, init);
                        outs.iter()
                            .map(|o| format!("{}\t{}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{:.5}", g.name, vname, *th as f64 / 100.0, seed, o.step, o.best_real, o.basin_best, o.best_readout, o.acc))
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .collect();
                let stdout = std::io::stdout();
                let mut lock = stdout.lock();
                for l in lines {
                    writeln!(lock, "{l}").unwrap();
                }
                lock.flush().unwrap();
                eprintln!("done {}", g.name);
            }
        }
        _ => eprintln!("usage: selftest | graphs <dir> | run <eigdir> <graphs> <variants> <lo100> <hi100> <step100> <nseeds> [seed_offset] [crn]"),
    }
}
