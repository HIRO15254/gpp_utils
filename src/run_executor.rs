//! 対数刻みでスナップショットを取りながら SA を実行する。
//!
//! 各スナップショットでは現在解と、スムージング空間および元空間それぞれで
//! 山登りを行ったベイスンの評価値を記録する。
//!
//! # Tier 2 specialized fast path
//!
//! `GraphPartitionProblem` の整数状態 `(cut_count, t_count, f_count)` を
//! ループ全体で保持し、スコア計算を delta_apply の連鎖で O(degree) に削減する。
//! Smoothing も同じ整数状態を経由して評価することで、N²クローン爆発を完全に解消する。
//!
//! 数値結果は元実装とビット完全一致（new_cut/new_diff を整数で再構成し、
//! 元と同じ式 `int as f64 + ALPHA * int as f64 * int as f64` に渡すため）。

use std::path::{Path, PathBuf};

use rand::Rng;
use rand_mt::Mt19937GenRand64;
use serde::{Deserialize, Serialize};

use crate::file_utils::{ensure_dir_exists, load_json, save_json};
use crate::graph_partition::{get_partition_sizes, GraphPartitionProblem, Partition};
use crate::graph_spec::GraphSpec;
use crate::optimization::Problem;
use crate::run_config::{RunConfig, SmoothingSpec};

/// 1 ステップ分の計測値。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// SA のステップ数（0 = 初期解、その後対数刻み）。
    pub step: usize,
    /// 現在解の現在（スムージング）空間での評価値。
    pub current_smoothed: f64,
    /// 現在解の元空間での評価値（=実スコア）。
    pub current_real: f64,
    /// 現在空間で山登り → そのベイスンの現在空間評価値。
    pub basin_smoothed_from_smoothed: f64,
    /// 現在空間で山登り → そのベイスンの元空間評価値。
    pub basin_real_from_smoothed: f64,
    /// 元空間で山登り → そのベイスンの現在空間評価値。
    pub basin_smoothed_from_real: f64,
    /// 元空間で山登り → そのベイスンの元空間評価値。
    pub basin_real_from_real: f64,
}

/// 1 シードあたりの実行結果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub graph_spec: GraphSpec,
    pub config: RunConfig,
    pub seed: u64,
    pub final_partition: Partition,
    pub records: Vec<StepRecord>,
    pub elapsed_ms: f64,
}

/// 対数刻みのステップを返す: 1, 2, ..., 9, 10, 20, ..., 90, 100, ...
pub fn logarithmic_steps(max_iter: usize) -> Vec<usize> {
    let mut v = Vec::new();
    if max_iter == 0 {
        return v;
    }
    let mut decade: usize = 1;
    while decade <= max_iter {
        for k in 1..=9 {
            let s = k * decade;
            if s > max_iter {
                break;
            }
            v.push(s);
        }
        if let Some(next) = decade.checked_mul(10) {
            decade = next;
        } else {
            break;
        }
    }
    if v.last().copied() != Some(max_iter) {
        v.push(max_iter);
    }
    v
}

// ============================================================================
// 整数状態追跡による山登り
// ============================================================================

/// 元空間での山登り（実スコアで下降）。
///
/// `cuts_at` を保持し `delta_apply_cached` で各候補を O(1) 評価するため、
/// 1 ステップあたり O(N)（従来は O(N · degree)）。
/// 戻り値にベイスンの `cuts_at` を含める。
fn hill_climb_real_fast(
    prob: &GraphPartitionProblem,
    start: &Partition,
    start_cuts: &[i32],
    start_cut: i32,
    start_t: usize,
    start_f: usize,
) -> (Partition, Vec<i32>, i32, usize, usize) {
    let mut current = start.clone();
    let mut cuts_at = start_cuts.to_vec();
    let (mut cur_cut, mut cur_t, mut cur_f) = (start_cut, start_t, start_f);
    let mut cur_score = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);

    loop {
        let n = current.len();
        let mut best_idx: Option<usize> = None;
        let mut best_score = cur_score;
        for i in 0..n {
            let s = prob
                .delta_apply_cached(&current, &cuts_at, i, cur_cut, cur_t, cur_f)
                .3;
            if s < best_score {
                best_score = s;
                best_idx = Some(i);
            }
        }
        match best_idx {
            Some(i) => {
                let (nc, nt, nf, ns) =
                    prob.delta_apply_cached(&current, &cuts_at, i, cur_cut, cur_t, cur_f);
                prob.flip_vertex(&mut current, &mut cuts_at, i);
                cur_cut = nc;
                cur_t = nt;
                cur_f = nf;
                cur_score = ns;
            }
            None => break,
        }
    }
    (current, cuts_at, cur_cut, cur_t, cur_f)
}

/// スムージング空間での山登り（smoothing closure を経由）。
/// 1 ステップあたり O(N · sm_cost)。`sm` は (partition, cut, t, f) → smoothed score。
///
/// 注意: `sm` は内部 RNG を進める可能性があるため、呼び出し回数を元実装と一致させること。
/// 元実装の hill_climb_smoothed は:
///   - ループ前: sm.score(prob, &current) を 1 回（初回 current_smoothed 取得）
///   - 各ステップ: 全 N 候補について sm.score(prob, n) を 1 回ずつ（合計 N 回）
///   ※ accept 後は best_score を current_smoothed に代入のみ（再評価しない）
fn hill_climb_smoothed_fast<F>(
    prob: &GraphPartitionProblem,
    start: &Partition,
    start_cuts: &[i32],
    start_cut: i32,
    start_t: usize,
    start_f: usize,
    sm: &mut F,
) -> (Partition, Vec<i32>, i32, usize, usize)
where
    F: FnMut(&Partition, &[i32], i32, usize, usize) -> f64,
{
    let mut current = start.clone();
    let mut cuts_at = start_cuts.to_vec();
    let (mut cur_cut, mut cur_t, mut cur_f) = (start_cut, start_t, start_f);
    let mut cur_smoothed = sm(&current, &cuts_at, cur_cut, cur_t, cur_f);

    loop {
        let n = current.len();
        let mut best_idx: Option<usize> = None;
        let mut best_smoothed = cur_smoothed;
        for i in 0..n {
            // 候補 = current^flip(i)。整数状態を算出後、(current, cuts_at) を
            // 一時的に flip して sm を呼び、unflip で戻す。
            let (nc, nt, nf, _) =
                prob.delta_apply_cached(&current, &cuts_at, i, cur_cut, cur_t, cur_f);
            prob.flip_vertex(&mut current, &mut cuts_at, i);
            let s = sm(&current, &cuts_at, nc, nt, nf);
            prob.flip_vertex(&mut current, &mut cuts_at, i); // unflip
            if s < best_smoothed {
                best_smoothed = s;
                best_idx = Some(i);
            }
        }
        match best_idx {
            Some(i) => {
                let (nc, nt, nf, _) =
                    prob.delta_apply_cached(&current, &cuts_at, i, cur_cut, cur_t, cur_f);
                prob.flip_vertex(&mut current, &mut cuts_at, i);
                cur_cut = nc;
                cur_t = nt;
                cur_f = nf;
                cur_smoothed = best_smoothed;
            }
            None => break,
        }
    }
    (current, cuts_at, cur_cut, cur_t, cur_f)
}

// ============================================================================
// スナップショット作成
// ============================================================================

fn make_snapshot_fast<F>(
    prob: &GraphPartitionProblem,
    current: &Partition,
    cuts_at: &[i32],
    cur_cut: i32,
    cur_t: usize,
    cur_f: usize,
    current_smoothed: f64,
    step: usize,
    sm: &mut F,
    no_smoothing: bool,
) -> StepRecord
where
    F: FnMut(&Partition, &[i32], i32, usize, usize) -> f64,
{
    let current_real = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);

    if no_smoothing {
        // smoothed-basin == real-basin → 1 回の HC で 4 フィールド埋め尽くし
        let (_, _, bc, bt, bf) =
            hill_climb_real_fast(prob, current, cuts_at, cur_cut, cur_t, cur_f);
        let basin_real = GraphPartitionProblem::score_from_state(bc, bt, bf);
        StepRecord {
            step,
            current_smoothed,
            current_real,
            basin_smoothed_from_smoothed: basin_real,
            basin_real_from_smoothed: basin_real,
            basin_smoothed_from_real: basin_real,
            basin_real_from_real: basin_real,
        }
    } else {
        // スムージング空間での山登り
        let (basin_sm_pt, basin_sm_cuts, bsmc, bsmt, bsmf) =
            hill_climb_smoothed_fast(prob, current, cuts_at, cur_cut, cur_t, cur_f, sm);
        let basin_smoothed_from_smoothed = sm(&basin_sm_pt, &basin_sm_cuts, bsmc, bsmt, bsmf);
        let basin_real_from_smoothed = GraphPartitionProblem::score_from_state(bsmc, bsmt, bsmf);

        // 元空間での山登り
        let (basin_re_pt, basin_re_cuts, brc, brt, brf) =
            hill_climb_real_fast(prob, current, cuts_at, cur_cut, cur_t, cur_f);
        let basin_smoothed_from_real = sm(&basin_re_pt, &basin_re_cuts, brc, brt, brf);
        let basin_real_from_real = GraphPartitionProblem::score_from_state(brc, brt, brf);

        StepRecord {
            step,
            current_smoothed,
            current_real,
            basin_smoothed_from_smoothed,
            basin_real_from_smoothed,
            basin_smoothed_from_real,
            basin_real_from_real,
        }
    }
}

// ============================================================================
// 共通 SA ループ
// ============================================================================

/// 共通 SA ループ。`sm` は smoothed score 評価クロージャ。
///
/// `sm` は呼ばれるたびに内部 RNG を進める可能性がある（RandomK 用）。
/// 呼び出しパターンは元実装と完全一致させる：
///   - ループ前: `sm(&current, ...)` を 1 回（初期 current_smoothed）
///   - 各 SA イテレーション: `sm(&current_with_flipped_idx, ...)` を 1 回
///   - スナップショット: `make_snapshot_fast` 内で smoothed HC + 2 回の sm 呼び出し
fn run_sa_generic<F>(
    prob: &GraphPartitionProblem,
    cfg: &RunConfig,
    seed: u64,
    no_smoothing: bool,
    mut sm: F,
) -> (Partition, Vec<StepRecord>)
where
    F: FnMut(&Partition, &[i32], i32, usize, usize) -> f64,
{
    let mut rng = Mt19937GenRand64::new(seed);
    let mut current: Partition = prob.random_solution(&mut rng);
    let mut cur_cut = prob.count_cut_edges(&current);
    let (mut cur_t, mut cur_f) = get_partition_sizes(&current);
    let mut cuts_at = prob.compute_cuts_at(&current);

    let mut current_smoothed = sm(&current, &cuts_at, cur_cut, cur_t, cur_f);

    let max_iter = cfg.iterations();
    let temperature = cfg.temperature();

    let snap_steps = logarithmic_steps(max_iter);
    let mut snap_iter = snap_steps.iter().copied().peekable();
    let mut records = Vec::with_capacity(snap_steps.len() + 1);

    // 初期スナップショット (step = 0)
    records.push(make_snapshot_fast(
        prob,
        &current,
        &cuts_at,
        cur_cut,
        cur_t,
        cur_f,
        current_smoothed,
        0,
        &mut sm,
        no_smoothing,
    ));

    let n = prob.neighbour_size();
    if n == 0 {
        return (current, records);
    }

    for it in 1..=max_iter {
        let idx = rng.gen_range(0..n);

        // 候補 c = current^flip(idx) の整数状態（フリップ前の cuts_at で算出）
        let (nc, nt, nf, _) =
            prob.delta_apply_cached(&current, &cuts_at, idx, cur_cut, cur_t, cur_f);

        // (current, cuts_at) を候補へ flip し、sm を評価する。
        prob.flip_vertex(&mut current, &mut cuts_at, idx);
        let neighbour_smoothed = sm(&current, &cuts_at, nc, nt, nf);

        let delta = neighbour_smoothed - current_smoothed;
        let accept = if delta < 0.0 {
            true
        } else if temperature > 0.0 {
            rng.r#gen::<f64>() < (-delta / temperature).exp()
        } else {
            false
        };
        if accept {
            // 候補状態をそのまま採用（current, cuts_at は flip 済み）。
            cur_cut = nc;
            cur_t = nt;
            cur_f = nf;
            current_smoothed = neighbour_smoothed;
        } else {
            // 不採用 → flip_vertex は対合なので同じ idx でもう一度呼んで戻す。
            prob.flip_vertex(&mut current, &mut cuts_at, idx);
        }

        // デバッグビルドでの整合性アサーション（drift 検出用）
        #[cfg(debug_assertions)]
        if it % 1000 == 0 {
            let recomputed_cut = prob.count_cut_edges(&current);
            let (recomputed_t, recomputed_f) = get_partition_sizes(&current);
            debug_assert_eq!(cur_cut, recomputed_cut, "cut drift at it={}", it);
            debug_assert_eq!(cur_t, recomputed_t, "t drift at it={}", it);
            debug_assert_eq!(cur_f, recomputed_f, "f drift at it={}", it);
            debug_assert_eq!(
                cuts_at,
                prob.compute_cuts_at(&current),
                "cuts_at drift at it={}",
                it
            );
        }

        if let Some(&want) = snap_iter.peek() {
            if it == want {
                records.push(make_snapshot_fast(
                    prob,
                    &current,
                    &cuts_at,
                    cur_cut,
                    cur_t,
                    cur_f,
                    current_smoothed,
                    it,
                    &mut sm,
                    no_smoothing,
                ));
                snap_iter.next();
            }
        }
    }

    (current, records)
}

// ============================================================================
// Smoothing 種別ごとの specialized SA
// ============================================================================

fn run_sa_none(
    prob: &GraphPartitionProblem,
    cfg: &RunConfig,
    seed: u64,
) -> (Partition, Vec<StepRecord>) {
    run_sa_generic(prob, cfg, seed, true, |_p, _cuts, c, t, f| {
        GraphPartitionProblem::score_from_state(c, t, f)
    })
}

fn run_sa_kavg(
    prob: &GraphPartitionProblem,
    k: usize,
    cfg: &RunConfig,
    seed: u64,
) -> (Partition, Vec<StepRecord>) {
    let n = prob.neighbour_size();
    let sample_count = k.min(n);
    run_sa_generic(prob, cfg, seed, false, move |p, cuts_at, c, t, f| {
        if n == 0 || sample_count == 0 {
            return GraphPartitionProblem::score_from_state(c, t, f);
        }
        // 元実装の `neighbours.iter().take(sample_count).map(|n| problem.score(n)).sum() / count`
        // と等価。インデックス 0..sample_count を順に評価し、左→右に逐次加算。
        let sum: f64 = (0..sample_count)
            .map(|j| prob.delta_apply_cached(p, cuts_at, j, c, t, f).3)
            .sum();
        sum / sample_count as f64
    })
}

fn run_sa_random_k(
    prob: &GraphPartitionProblem,
    k: usize,
    sm_seed: u64,
    cfg: &RunConfig,
    seed: u64,
) -> (Partition, Vec<StepRecord>) {
    let n = prob.neighbour_size();
    let mut sm_rng = Mt19937GenRand64::new(sm_seed);

    run_sa_generic(prob, cfg, seed, false, move |p, cuts_at, c, t, f| {
        if n == 0 {
            return GraphPartitionProblem::score_from_state(c, t, f);
        }

        if k <= n {
            // d1 から K 個ランダムサンプリング（Fisher-Yates）
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = sm_rng.gen_range(i..n);
                indices.swap(i, j);
            }
            let scores: Vec<f64> = indices[..k]
                .iter()
                .map(|&i| prob.delta_apply_cached(p, cuts_at, i, c, t, f).3)
                .collect();
            if scores.is_empty() {
                return GraphPartitionProblem::score_from_state(c, t, f);
            }
            scores.iter().sum::<f64>() / scores.len() as f64
        } else {
            // d2 フォールバック: d1 全部 + d2 から (k - n) 個サンプル
            // 元実装の d2 列挙順は (j, k_idx) で j < k_idx の昇順。
            let mut d2_pairs: Vec<(usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
            for j in 0..n {
                for k_idx in (j + 1)..n {
                    d2_pairs.push((j, k_idx));
                }
            }

            let needed = k - n;
            let take = needed.min(d2_pairs.len());
            for i in 0..take {
                let j = sm_rng.gen_range(i..d2_pairs.len());
                d2_pairs.swap(i, j);
            }

            // d1 の全スコア + 選ばれた d2 のスコア
            let mut scores: Vec<f64> = (0..n)
                .map(|i| prob.delta_apply_cached(p, cuts_at, i, c, t, f).3)
                .collect();
            for &(j, k_idx) in &d2_pairs[..take] {
                // p^flip(j)^flip(k_idx) のスコア
                let (jc, jt, jf, _) = prob.delta_apply_cached(p, cuts_at, j, c, t, f);
                // p[j] を一時 flip して delta_apply(_, k_idx, jc, jt, jf) を取る
                let mut p_clone = p.clone();
                p_clone[j] = !p_clone[j];
                let s = prob.delta_apply(&p_clone, k_idx, jc, jt, jf).3;
                scores.push(s);
            }

            if scores.is_empty() {
                return GraphPartitionProblem::score_from_state(c, t, f);
            }
            scores.iter().sum::<f64>() / scores.len() as f64
        }
    })
}

fn run_sa_weighted(
    prob: &GraphPartitionProblem,
    k: usize,
    cfg: &RunConfig,
    seed: u64,
) -> (Partition, Vec<StepRecord>) {
    let n = prob.neighbour_size();
    run_sa_generic(prob, cfg, seed, false, move |p, cuts_at, c, t, f| {
        if n == 0 {
            return GraphPartitionProblem::score_from_state(c, t, f);
        }
        let k_clamped = k.min(n) as f64;
        let weight = k_clamped / n as f64;
        // 元実装と同じ加算順序: 0..n を逐次加算 → / n
        let neighbour_avg = (0..n)
            .map(|i| prob.delta_apply_cached(p, cuts_at, i, c, t, f).3)
            .sum::<f64>()
            / n as f64;
        let current_score = GraphPartitionProblem::score_from_state(c, t, f);
        weight * neighbour_avg + (1.0 - weight) * current_score
    })
}

// ============================================================================
// 公開 API
// ============================================================================

/// 単一シードの実行を行い、結果を返す（保存はしない）。
pub fn execute(
    spec: GraphSpec,
    cfg: &RunConfig,
    prob: &GraphPartitionProblem,
    seed: u64,
) -> RunResult {
    let t0 = std::time::Instant::now();
    let sm_seed = seed.wrapping_add(0xDEAD_BEEF);
    let (final_p, records) = match cfg.smoothing {
        SmoothingSpec::None => run_sa_none(prob, cfg, seed),
        SmoothingSpec::KAverage(k) => run_sa_kavg(prob, k, cfg, seed),
        SmoothingSpec::RandomKAverage(k) => run_sa_random_k(prob, k, sm_seed, cfg, seed),
        SmoothingSpec::WeightedAverage(k) => run_sa_weighted(prob, k, cfg, seed),
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    RunResult {
        graph_spec: spec,
        config: cfg.clone(),
        seed,
        final_partition: final_p,
        records,
        elapsed_ms,
    }
}

/// 結果ストアの管理。
pub struct ResultStore {
    pub base_dir: PathBuf,
}

impl ResultStore {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    /// 結果ファイルのパス（`base/<graph_id>/<config_id>/seed_<seed>.json`）。
    pub fn path_for(&self, spec: &GraphSpec, cfg: &RunConfig, seed: u64) -> PathBuf {
        self.base_dir
            .join(spec.id())
            .join(cfg.id())
            .join(format!("seed_{}.json", seed))
    }

    pub fn exists(&self, spec: &GraphSpec, cfg: &RunConfig, seed: u64) -> bool {
        self.path_for(spec, cfg, seed).exists()
    }

    pub fn load(&self, spec: &GraphSpec, cfg: &RunConfig, seed: u64) -> Option<RunResult> {
        load_json::<RunResult>(&self.path_for(spec, cfg, seed)).ok()
    }

    pub fn save(&self, result: &RunResult) -> Result<(), String> {
        let p = self.path_for(&result.graph_spec, &result.config, result.seed);
        if let Some(parent) = p.parent() {
            ensure_dir_exists(parent).map_err(|e| format!("create dir: {}", e))?;
        }
        save_json(result, &p).map_err(|e| format!("save: {}", e))
    }

    /// gnuplot で扱いやすい TSV を出力する。
    /// 列: step, current_smoothed, current_real,
    ///     basin_smoothed_from_smoothed, basin_real_from_smoothed,
    ///     basin_smoothed_from_real, basin_real_from_real
    pub fn export_tsv(&self, result: &RunResult, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            ensure_dir_exists(parent).map_err(|e| format!("create dir: {}", e))?;
        }
        use std::io::Write;
        let mut f = std::fs::File::create(path).map_err(|e| format!("create: {}", e))?;
        writeln!(
            f,
            "# graph={} config={} seed={}",
            result.graph_spec.id(),
            result.config.id(),
            result.seed
        )
        .map_err(|e| format!("write: {}", e))?;
        writeln!(
            f,
            "# step\tcur_sm\tcur_real\tbasin_sm_from_sm\tbasin_real_from_sm\tbasin_sm_from_real\tbasin_real_from_real"
        )
        .map_err(|e| format!("write: {}", e))?;
        for r in &result.records {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                r.step,
                r.current_smoothed,
                r.current_real,
                r.basin_smoothed_from_smoothed,
                r.basin_real_from_smoothed,
                r.basin_smoothed_from_real,
                r.basin_real_from_real
            )
            .map_err(|e| format!("write: {}", e))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_steps_small() {
        let s = logarithmic_steps(100);
        assert_eq!(s[0], 1);
        assert_eq!(s.last().copied(), Some(100));
        assert!(s.contains(&9));
        assert!(s.contains(&10));
        assert!(s.contains(&50));
        assert!(s.contains(&90));
    }

    #[test]
    fn test_log_steps_appends_max_when_not_decade() {
        let s = logarithmic_steps(150);
        assert_eq!(s.last().copied(), Some(150));
        assert!(s.contains(&100));
    }

    #[test]
    fn test_execute_runs() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec {
            kind: GraphKind::Random,
            n: 30,
            d: 4.0,
            seed: 0,
        };
        let stored = StoredGraph::generate(spec);
        let prob = stored.problem();
        let mut cfg = RunConfig::new("t");
        cfg.log10_iterations = 2;
        cfg.theta = Some(0.0);
        let r = execute(spec, &cfg, &prob, 42);
        assert!(!r.records.is_empty());
        assert_eq!(r.records[0].step, 0);
    }
}
