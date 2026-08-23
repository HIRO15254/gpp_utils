//! 収穫済み状態スナップショット（`seed_X_states.json`）へのオフラインプローブ。
//!
//! ストア内の各状態に対して全プローブ設定（[`EoFlipFitnessSpec`] のリスト）の λ を
//! 再計算し、設定ペアごとの順位・選択分布距離（[`crate::rank_metrics`]）を
//! step 帯（init / early / mid / late）別に集約して CSV に書き出す。
//!
//! 生の λ を全ランで出力するとギガバイト級になるため、集約（(条件, 収穫設定,
//! 収穫τ, 帯, ペア) ごとの平均・標準偏差・件数）を Rust 側で行う。
//! 頂点レベルの詳細（Q5 の属性回帰用）は限定したランのみ `vertices.csv` に出す。

use std::collections::BTreeMap;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use rayon::prelude::*;

use crate::file_utils::{ensure_dir_exists, load_json};
use crate::graph_partition::GraphPartitionProblem;
use crate::graph_spec::{GraphLibrary, GraphSpec};
use crate::rank_metrics::{
    bottom_m_jaccard, jensen_shannon, kendall_tau_b, midranks, selection_probs_from,
    shannon_entropy, sorted_order, tie_groups, total_variation,
};
use crate::run_config::EoFlipFitnessSpec;
use crate::run_executor::{
    build_power_law_cdf, degrees_of, eo_flip_lambdas, is_majority_side, state_context,
    unpack_bits_hex, RunStates,
};

/// プローブ対象の 1 設定。適応度式か、物差し用のランダム順位（決定的ハッシュ）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProbeSpec {
    Fitness(EoFlipFitnessSpec),
    /// (graph_seed, seed, step, vertex) の splitmix64 ハッシュを λ とする疑似設定。
    /// どの適応度とも無関係な順位の「τ_b ≈ 0 / JSD 上限」アンカー。
    RandomAnchor,
}

impl ProbeSpec {
    pub fn label(&self) -> String {
        match self {
            ProbeSpec::Fitness(f) => f.label(),
            ProbeSpec::RandomAnchor => "random_anchor".to_string(),
        }
    }
}

/// step → 帯番号。0: init（初期解）、1: early（10^0-2）、2: mid（10^3-4）、3: late（10^5-）。
pub const BAND_NAMES: [&str; 4] = ["init", "early", "mid", "late"];

pub fn band_of(step: usize) -> usize {
    if step == 0 {
        0
    } else if step < 1_000 {
        1
    } else if step < 100_000 {
        2
    } else {
        3
    }
}

/// 頂点レベルダンプの限定条件。
#[derive(Debug, Clone)]
pub struct VertexDump {
    /// ダンプするステップ（states に存在するものだけ拾う）。
    pub steps: Vec<usize>,
    /// 収穫側設定のラベル（[`EoFlipFitnessSpec::label`]）がこれに一致するランのみ。
    pub src_label: String,
    /// このグラフ instance seed のみ。
    pub graph_seed: u64,
    /// このアルゴリズム seed のみ。
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct ProbeConfig {
    pub store_dir: PathBuf,
    pub graph_dir: PathBuf,
    pub out_dir: PathBuf,
    /// プローブ設定リスト（[`ProbeSpec::RandomAnchor`] は `include_random_anchor` で付加）。
    pub specs: Vec<EoFlipFitnessSpec>,
    /// 選択分布系指標（TV / JSD / エントロピー）を計算する τ のリスト。
    pub taus: Vec<f64>,
    /// bottom-m Jaccard の m リスト。
    pub jaccard_ms: Vec<usize>,
    /// 対象アルゴ seed（None = 全 seed）。
    pub seeds: Option<Vec<u64>>,
    pub include_random_anchor: bool,
    pub threads: usize,
    pub vertex_dump: Option<VertexDump>,
}

#[derive(Debug, Default)]
pub struct ProbeSummary {
    pub runs: usize,
    pub states: usize,
    pub groups: usize,
    pub rows_pairs: usize,
    pub rows_specs: usize,
    pub rows_vertices: usize,
    pub errors: Vec<String>,
}

/// splitmix64（決定的ハッシュ）。
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn anchor_lambda(graph_seed: u64, seed: u64, step: usize, vertex: usize) -> f64 {
    let mut h = splitmix64(graph_seed ^ 0xA5A5_0000_0000_0001);
    h = splitmix64(h ^ seed);
    h = splitmix64(h ^ step as u64);
    h = splitmix64(h ^ vertex as u64);
    // [0,1) の一様値。タイは実質発生しない。
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// 条件 id（インスタンス seed を除いたグラフ仕様）。例: `random_n250_d5`。
fn cond_of(spec: &GraphSpec) -> String {
    let id = spec.id();
    match id.rfind("_s") {
        Some(pos) => id[..pos].to_string(),
        None => id,
    }
}

fn fmt_p(x: f64) -> String {
    if x.fract().abs() < 1e-9 {
        format!("{}", x as i64)
    } else {
        format!("{}", x).replace('.', "p")
    }
}

/// 1 状態 × 1 設定の順位情報一式。
struct SpecEval {
    lambdas: Vec<f64>,
    order: Vec<usize>,
    groups: Vec<(usize, usize)>,
    /// taus と同順の選択確率分布。
    probs: Vec<Vec<f64>>,
}

/// 集約アキュムレータ（1 グループ = 1 (条件, 収穫設定, 収穫τ) 内で使用）。
struct PairAcc {
    /// [band][pair][metric] の (sum, sumsq)。metric 順: tau_b, tv×T, jsd×T, jacc×M。
    sum: Vec<f64>,
    sumsq: Vec<f64>,
    /// [band][pair]: tau_b が定義できた状態数。
    tau_b_n: Vec<usize>,
    /// [band][pair]: 全状態数（tau_b 以外の分母）。
    n: Vec<usize>,
}

struct SpecAcc {
    /// [band][spec]: (distinct λ 数, 最大同率群サイズ率, 縮退状態数, エントロピー×T, top1×T)
    distinct_sum: Vec<f64>,
    max_tie_frac_sum: Vec<f64>,
    degenerate: Vec<usize>,
    entropy_sum: Vec<f64>,
    top1_sum: Vec<f64>,
    n: Vec<usize>,
}

pub fn run_probe(cfg: &ProbeConfig) -> Result<ProbeSummary, String> {
    // ---- プローブ設定リスト確定 ----
    let mut specs: Vec<ProbeSpec> = cfg.specs.iter().map(|&f| ProbeSpec::Fitness(f)).collect();
    if cfg.include_random_anchor {
        specs.push(ProbeSpec::RandomAnchor);
    }
    let s_count = specs.len();
    if s_count < 2 {
        return Err("プローブ設定が 2 未満です".to_string());
    }
    let labels: Vec<String> = specs.iter().map(|s| s.label()).collect();
    let pairs: Vec<(usize, usize)> = (0..s_count)
        .flat_map(|i| ((i + 1)..s_count).map(move |j| (i, j)))
        .collect();
    let t_count = cfg.taus.len();
    let m_count = cfg.jaccard_ms.len();
    let n_metrics = 1 + 2 * t_count + m_count;

    // ---- states ファイル探索・読み込み ----
    let mut states_files: Vec<PathBuf> = Vec::new();
    let store = &cfg.store_dir;
    let rd = std::fs::read_dir(store).map_err(|e| format!("store 読み込み失敗 {}: {}", store.display(), e))?;
    for g_ent in rd.flatten() {
        if !g_ent.path().is_dir() {
            continue;
        }
        for c_ent in std::fs::read_dir(g_ent.path()).map_err(|e| e.to_string())?.flatten() {
            if !c_ent.path().is_dir() {
                continue;
            }
            for f_ent in std::fs::read_dir(c_ent.path()).map_err(|e| e.to_string())?.flatten() {
                let name = f_ent.file_name().to_string_lossy().to_string();
                if name.starts_with("seed_") && name.ends_with("_states.json") {
                    // seed フィルタ（ファイル名の数値部分で判定）。
                    if let Some(filter) = &cfg.seeds {
                        let num: Option<u64> = name
                            .trim_start_matches("seed_")
                            .trim_end_matches("_states.json")
                            .parse()
                            .ok();
                        match num {
                            Some(s) if filter.contains(&s) => {}
                            _ => continue,
                        }
                    }
                    states_files.push(f_ent.path());
                }
            }
        }
    }
    if states_files.is_empty() {
        return Err(format!("{} に seed_*_states.json が見つかりません", store.display()));
    }

    let mut summary = ProbeSummary::default();
    let mut all_states: Vec<RunStates> = Vec::with_capacity(states_files.len());
    for p in &states_files {
        match load_json::<RunStates>(p) {
            Ok(s) => all_states.push(s),
            Err(e) => summary.errors.push(format!("{}: {}", p.display(), e)),
        }
    }
    summary.runs = all_states.len();
    summary.states = all_states.iter().map(|s| s.snapshots.len()).sum();

    // ---- グラフ事前ロード ----
    let library = GraphLibrary::new(&cfg.graph_dir);
    let mut problems: BTreeMap<String, Arc<(GraphPartitionProblem, Vec<usize>)>> = BTreeMap::new();
    for st in &all_states {
        let gid = st.graph_spec.id();
        if !problems.contains_key(&gid) {
            let stored = library
                .load_or_generate(st.graph_spec)
                .map_err(|e| format!("グラフ {}: {}", gid, e))?;
            let prob = stored.problem();
            let degrees = degrees_of(&prob);
            problems.insert(gid, Arc::new((prob, degrees)));
        }
    }

    // ---- (条件, 収穫設定ラベル, 収穫τ) でグループ化 ----
    let mut groups_map: BTreeMap<(String, String, String), Vec<RunStates>> = BTreeMap::new();
    for st in all_states {
        let cond = cond_of(&st.graph_spec);
        let src = EoFlipFitnessSpec::from_solver(&st.config.solver)
            .map(|f| f.label())
            .unwrap_or_else(|| "non_eoflip".to_string());
        let tau = st
            .config
            .solver
            .tau()
            .map(fmt_p)
            .unwrap_or_else(|| "na".to_string());
        groups_map.entry((cond, src, tau)).or_default().push(st);
    }
    summary.groups = groups_map.len();

    // ---- 出力ライタ ----
    ensure_dir_exists(&cfg.out_dir).map_err(|e| format!("out dir: {}", e))?;
    let conds: Vec<String> = {
        let mut v: Vec<String> = groups_map.keys().map(|k| k.0.clone()).collect();
        v.dedup();
        v.sort();
        v.dedup();
        v
    };
    let mut pair_header = String::from("cond,src_setting,src_tau,band,n_runs,n_states,spec_a,spec_b,kendall_b_mean,kendall_b_std,frac_tau_b_undef");
    for t in &cfg.taus {
        pair_header.push_str(&format!(",tv_t{}", fmt_p(*t)));
    }
    for t in &cfg.taus {
        pair_header.push_str(&format!(",jsd_t{}", fmt_p(*t)));
    }
    for m in &cfg.jaccard_ms {
        pair_header.push_str(&format!(",jacc_m{}", m));
    }
    let mut spec_header = String::from("cond,src_setting,src_tau,band,n_states,spec,n_distinct_lambda_mean,max_tie_frac_mean,frac_degenerate");
    for t in &cfg.taus {
        spec_header.push_str(&format!(",sel_entropy_t{}", fmt_p(*t)));
    }
    for t in &cfg.taus {
        spec_header.push_str(&format!(",top1_mass_t{}", fmt_p(*t)));
    }

    let mut pair_writers: BTreeMap<String, Mutex<std::io::BufWriter<std::fs::File>>> =
        BTreeMap::new();
    let mut spec_writers: BTreeMap<String, Mutex<std::io::BufWriter<std::fs::File>>> =
        BTreeMap::new();
    for cond in &conds {
        let mut pw = std::io::BufWriter::new(
            std::fs::File::create(cfg.out_dir.join(format!("pairs_{}.csv", cond)))
                .map_err(|e| format!("pairs csv: {}", e))?,
        );
        writeln!(pw, "{}", pair_header).map_err(|e| e.to_string())?;
        pair_writers.insert(cond.clone(), Mutex::new(pw));
        let mut sw = std::io::BufWriter::new(
            std::fs::File::create(cfg.out_dir.join(format!("specs_{}.csv", cond)))
                .map_err(|e| format!("specs csv: {}", e))?,
        );
        writeln!(sw, "{}", spec_header).map_err(|e| e.to_string())?;
        spec_writers.insert(cond.clone(), Mutex::new(sw));
    }
    let vertex_writer = if cfg.vertex_dump.is_some() {
        let mut vw = std::io::BufWriter::new(
            std::fs::File::create(cfg.out_dir.join("vertices.csv"))
                .map_err(|e| format!("vertices csv: {}", e))?,
        );
        writeln!(
            vw,
            "cond,graph_seed,src_setting,src_tau,seed,step,vertex,deg,cuts,lambda0,in_true,is_majority,spec,lambda,rank_mid,tie_size,p_sel_first_tau"
        )
        .map_err(|e| e.to_string())?;
        Some(Mutex::new(vw))
    } else {
        None
    };

    // ---- 並列処理（グループ単位） ----
    let groups: Vec<((String, String, String), Vec<RunStates>)> = groups_map.into_iter().collect();
    let row_counts = Mutex::new((0usize, 0usize, 0usize)); // (pairs, specs, vertices)
    let errors = Mutex::new(Vec::<String>::new());

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.threads.max(1))
        .build()
        .map_err(|e| e.to_string())?;

    pool.install(|| {
        groups.par_iter().for_each(|((cond, src, src_tau), runs)| {
            let n_bands = BAND_NAMES.len();
            let mut pacc = PairAcc {
                sum: vec![0.0; n_bands * pairs.len() * n_metrics],
                sumsq: vec![0.0; n_bands * pairs.len() * n_metrics],
                tau_b_n: vec![0; n_bands * pairs.len()],
                n: vec![0; n_bands * pairs.len()],
            };
            let mut sacc = SpecAcc {
                distinct_sum: vec![0.0; n_bands * s_count],
                max_tie_frac_sum: vec![0.0; n_bands * s_count],
                degenerate: vec![0; n_bands * s_count],
                entropy_sum: vec![0.0; n_bands * s_count * t_count],
                top1_sum: vec![0.0; n_bands * s_count * t_count],
                n: vec![0; n_bands * s_count],
            };
            let mut vertex_rows: Vec<String> = Vec::new();

            for run in runs {
                let gid = run.graph_spec.id();
                let Some(pd) = problems.get(&gid) else {
                    errors.lock().unwrap().push(format!("{}: problem 未ロード", gid));
                    continue;
                };
                let (prob, degrees) = (&pd.0, &pd.1);
                let n = run.n;
                if n < 2 {
                    continue;
                }
                let cdfs: Vec<Vec<f64>> =
                    cfg.taus.iter().map(|&t| build_power_law_cdf(n, t)).collect();

                let dump_this_run = cfg.vertex_dump.as_ref().is_some_and(|d| {
                    d.src_label == *src && run.graph_spec.seed == d.graph_seed && run.seed == d.seed
                });

                for snap in &run.snapshots {
                    let current = match unpack_bits_hex(&snap.bits, n) {
                        Ok(p) => p,
                        Err(e) => {
                            errors
                                .lock()
                                .unwrap()
                                .push(format!("{} seed={} step={}: {}", gid, run.seed, snap.step, e));
                            continue;
                        }
                    };
                    let ctx = state_context(prob, &current);
                    let band = band_of(snap.step);

                    // 全設定の λ・順位・選択分布。
                    let evals: Vec<SpecEval> = specs
                        .iter()
                        .map(|sp| {
                            let mut lambdas = vec![0.0f64; n];
                            match sp {
                                ProbeSpec::Fitness(f) => {
                                    eo_flip_lambdas(f, &current, &ctx, degrees, &mut lambdas)
                                }
                                ProbeSpec::RandomAnchor => {
                                    for (v, l) in lambdas.iter_mut().enumerate() {
                                        *l = anchor_lambda(
                                            run.graph_spec.seed,
                                            run.seed,
                                            snap.step,
                                            v,
                                        );
                                    }
                                }
                            }
                            let order = sorted_order(&lambdas);
                            let groups = tie_groups(&lambdas, &order);
                            let probs = cdfs
                                .iter()
                                .map(|cum| selection_probs_from(&order, &groups, cum))
                                .collect();
                            SpecEval { lambdas, order, groups, probs }
                        })
                        .collect();

                    // 設定単体の統計。
                    for (si, ev) in evals.iter().enumerate() {
                        let idx = band * s_count + si;
                        sacc.n[idx] += 1;
                        sacc.distinct_sum[idx] += ev.groups.len() as f64;
                        let max_tie =
                            ev.groups.iter().map(|&(s, e)| e - s).max().unwrap_or(0);
                        sacc.max_tie_frac_sum[idx] += max_tie as f64 / n as f64;
                        if ev.groups.len() == 1 {
                            sacc.degenerate[idx] += 1;
                        }
                        for (ti, p) in ev.probs.iter().enumerate() {
                            let tidx = (band * s_count + si) * t_count + ti;
                            sacc.entropy_sum[tidx] += shannon_entropy(p);
                            sacc.top1_sum[tidx] +=
                                p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        }
                    }

                    // ペア指標。
                    for (pi, &(i, j)) in pairs.iter().enumerate() {
                        let base = (band * pairs.len() + pi) * n_metrics;
                        let nidx = band * pairs.len() + pi;
                        pacc.n[nidx] += 1;
                        if let Some(tb) = kendall_tau_b(&evals[i].lambdas, &evals[j].lambdas) {
                            pacc.tau_b_n[nidx] += 1;
                            pacc.sum[base] += tb;
                            pacc.sumsq[base] += tb * tb;
                        }
                        for ti in 0..t_count {
                            let tv = total_variation(&evals[i].probs[ti], &evals[j].probs[ti]);
                            let js = jensen_shannon(&evals[i].probs[ti], &evals[j].probs[ti]);
                            pacc.sum[base + 1 + ti] += tv;
                            pacc.sumsq[base + 1 + ti] += tv * tv;
                            pacc.sum[base + 1 + t_count + ti] += js;
                            pacc.sumsq[base + 1 + t_count + ti] += js * js;
                        }
                        for (mi, &m) in cfg.jaccard_ms.iter().enumerate() {
                            let jc = bottom_m_jaccard(
                                &evals[i].order,
                                &evals[i].groups,
                                &evals[j].order,
                                &evals[j].groups,
                                m,
                            );
                            pacc.sum[base + 1 + 2 * t_count + mi] += jc;
                            pacc.sumsq[base + 1 + 2 * t_count + mi] += jc * jc;
                        }
                    }

                    // 頂点レベルダンプ。
                    if dump_this_run
                        && cfg
                            .vertex_dump
                            .as_ref()
                            .is_some_and(|d| d.steps.contains(&snap.step))
                    {
                        for (si, ev) in evals.iter().enumerate() {
                            let ranks = midranks(&ev.order, &ev.groups);
                            let mut tie_size = vec![0usize; n];
                            for &(s, e) in &ev.groups {
                                for pos in s..e {
                                    tie_size[ev.order[pos]] = e - s;
                                }
                            }
                            for v in 0..n {
                                let deg = degrees[v];
                                let cuts = ctx.cuts_at[v];
                                let lambda0 = if deg == 0 {
                                    1.0
                                } else {
                                    (deg as f64 - cuts as f64) / deg as f64
                                };
                                vertex_rows.push(format!(
                                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                                    cond,
                                    run.graph_spec.seed,
                                    src,
                                    src_tau,
                                    run.seed,
                                    snap.step,
                                    v,
                                    deg,
                                    cuts,
                                    lambda0,
                                    current[v] as u8,
                                    is_majority_side(current[v], ctx.cur_t, ctx.cur_f) as u8,
                                    labels[si],
                                    ev.lambdas[v],
                                    ranks[v],
                                    tie_size[v],
                                    ev.probs.first().map(|p| p[v]).unwrap_or(f64::NAN),
                                ));
                            }
                        }
                    }
                }
            }

            // ---- グループの集約行を書き出し ----
            let n_runs = runs.len();
            let mut pair_lines = String::new();
            for band in 0..BAND_NAMES.len() {
                for (pi, &(i, j)) in pairs.iter().enumerate() {
                    let nidx = band * pairs.len() + pi;
                    let n_states = pacc.n[nidx];
                    if n_states == 0 {
                        continue;
                    }
                    let base = (band * pairs.len() + pi) * n_metrics;
                    let tb_n = pacc.tau_b_n[nidx];
                    let (tb_mean, tb_std) = if tb_n > 0 {
                        let mean = pacc.sum[base] / tb_n as f64;
                        let var = (pacc.sumsq[base] / tb_n as f64 - mean * mean).max(0.0);
                        (mean, var.sqrt())
                    } else {
                        (f64::NAN, f64::NAN)
                    };
                    let frac_undef = 1.0 - tb_n as f64 / n_states as f64;
                    let mut line = format!(
                        "{},{},{},{},{},{},{},{},{},{},{}",
                        cond,
                        src,
                        src_tau,
                        BAND_NAMES[band],
                        n_runs,
                        n_states,
                        labels[i],
                        labels[j],
                        tb_mean,
                        tb_std,
                        frac_undef
                    );
                    for k in 0..(2 * t_count + m_count) {
                        line.push_str(&format!(
                            ",{}",
                            pacc.sum[base + 1 + k] / n_states as f64
                        ));
                    }
                    pair_lines.push_str(&line);
                    pair_lines.push('\n');
                }
            }
            let mut spec_lines = String::new();
            for band in 0..BAND_NAMES.len() {
                for si in 0..s_count {
                    let idx = band * s_count + si;
                    let n_states = sacc.n[idx];
                    if n_states == 0 {
                        continue;
                    }
                    let mut line = format!(
                        "{},{},{},{},{},{},{},{},{}",
                        cond,
                        src,
                        src_tau,
                        BAND_NAMES[band],
                        n_states,
                        labels[si],
                        sacc.distinct_sum[idx] / n_states as f64,
                        sacc.max_tie_frac_sum[idx] / n_states as f64,
                        sacc.degenerate[idx] as f64 / n_states as f64
                    );
                    for ti in 0..t_count {
                        line.push_str(&format!(
                            ",{}",
                            sacc.entropy_sum[idx * t_count + ti] / n_states as f64
                        ));
                    }
                    for ti in 0..t_count {
                        line.push_str(&format!(
                            ",{}",
                            sacc.top1_sum[idx * t_count + ti] / n_states as f64
                        ));
                    }
                    spec_lines.push_str(&line);
                    spec_lines.push('\n');
                }
            }

            let pair_rows = pair_lines.lines().count();
            let spec_rows = spec_lines.lines().count();
            if let Some(w) = pair_writers.get(cond) {
                let mut g = w.lock().unwrap();
                let _ = g.write_all(pair_lines.as_bytes());
            }
            if let Some(w) = spec_writers.get(cond) {
                let mut g = w.lock().unwrap();
                let _ = g.write_all(spec_lines.as_bytes());
            }
            let v_rows = vertex_rows.len();
            if let Some(vw) = &vertex_writer {
                let mut g = vw.lock().unwrap();
                for r in &vertex_rows {
                    let _ = writeln!(g, "{}", r);
                }
            }
            let mut rc = row_counts.lock().unwrap();
            rc.0 += pair_rows;
            rc.1 += spec_rows;
            rc.2 += v_rows;
        });
    });

    for (_, w) in pair_writers {
        w.into_inner().unwrap().flush().map_err(|e| e.to_string())?;
    }
    for (_, w) in spec_writers {
        w.into_inner().unwrap().flush().map_err(|e| e.to_string())?;
    }
    if let Some(vw) = vertex_writer {
        vw.into_inner().unwrap().flush().map_err(|e| e.to_string())?;
    }

    let rc = row_counts.into_inner().unwrap();
    summary.rows_pairs = rc.0;
    summary.rows_specs = rc.1;
    summary.rows_vertices = rc.2;
    summary.errors.extend(errors.into_inner().unwrap());

    // ---- マニフェスト ----
    let manifest = serde_json::json!({
        "store_dir": cfg.store_dir.display().to_string(),
        "taus": cfg.taus,
        "jaccard_ms": cfg.jaccard_ms,
        "seeds": cfg.seeds,
        "specs": labels,
        "runs": summary.runs,
        "states": summary.states,
        "groups": summary.groups,
        "rows_pairs": summary.rows_pairs,
        "rows_specs": summary.rows_specs,
        "rows_vertices": summary.rows_vertices,
        "errors": summary.errors,
    });
    std::fs::write(
        cfg.out_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .map_err(|e| e.to_string())?;

    Ok(summary)
}

/// 収穫グリッドと同一の既定プローブ設定リスト（36 設定）。
pub fn default_probe_specs() -> Vec<EoFlipFitnessSpec> {
    let mut v = Vec::new();
    for a in [0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512] {
        v.push(EoFlipFitnessSpec::Legacy { alpha_eo: a, diff_exp: 2.0 });
    }
    for a in [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0] {
        v.push(EoFlipFitnessSpec::MulAlpha { alpha: a });
    }
    for b in [0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0] {
        v.push(EoFlipFitnessSpec::AddBeta { beta: b });
    }
    v.push(EoFlipFitnessSpec::MulGamma);
    v
}

/// 既定の τ リスト（プローブ側）。
pub fn default_probe_taus() -> Vec<f64> {
    vec![0.8, 1.1, 1.4, 1.7]
}

/// 既定の bottom-m リスト。
pub fn default_jaccard_ms() -> Vec<usize> {
    vec![1, 2, 4, 8, 16, 32, 64, 128]
}
