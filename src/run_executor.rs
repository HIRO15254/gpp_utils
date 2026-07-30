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
//! スコア値は整数状態から `score_from_state`（= `score()` と同一の式）で
//! 再構成するため、浮動小数点演算レベルで決定論的に一致する。
//!
//! # ベイスン山登りのタイブレーク
//!
//! ベイスン評価の山登りで同スコアの近傍が複数あるときは、1 つを一様ランダムに
//! 選ぶ。タイブレーク用の乱数列は SA 本体・スムージングとは独立した専用 RNG
//! （`seed ^ TIEBREAK_SALT`）から取るため、SA の軌跡と `final_partition` には
//! 影響しない（`records` のベイスン値のみがタイブレークの影響を受ける）。

use std::path::{Path, PathBuf};

use rand::Rng;
use rand_mt::Mt19937GenRand64;
use serde::{Deserialize, Serialize};

use crate::file_utils::{ensure_dir_exists, load_json, save_json};
use crate::graph_partition::{get_partition_sizes, GraphPartitionProblem, Partition};
use crate::graph_spec::GraphSpec;
use crate::optimization::Problem;
use crate::run_config::{EoFlipFitnessSpec, RunConfig, SmoothingSpec, SolverSpec};

/// ベイスン山登りのタイブレーク用 RNG シードを、SA 本体のシードから派生させる際の塩。
/// SA・スムージングの乱数列と独立にすることで `final_partition` への影響を避ける。
const TIEBREAK_SALT: u64 = 0x7113_B4EA_C0DE_5EED;

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
///
/// 同スコアの近傍が複数あるときは `tie_rng` から reservoir sampling で
/// 1 つを一様選択する。
fn hill_climb_real_fast(
    prob: &GraphPartitionProblem,
    start: &Partition,
    start_cuts: &[i32],
    start_cut: i32,
    start_t: usize,
    start_f: usize,
    tie_rng: &mut Mt19937GenRand64,
) -> (Partition, Vec<i32>, i32, usize, usize) {
    let mut current = start.clone();
    let mut cuts_at = start_cuts.to_vec();
    let (mut cur_cut, mut cur_t, mut cur_f) = (start_cut, start_t, start_f);
    let mut cur_score = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);

    loop {
        let n = current.len();
        let mut best_idx: Option<usize> = None;
        let mut best_score = cur_score;
        let mut tie_count: u64 = 0;
        for i in 0..n {
            let s = prob
                .delta_apply_cached(&current, &cuts_at, i, cur_cut, cur_t, cur_f)
                .3;
            if s < best_score {
                best_score = s;
                best_idx = Some(i);
                tie_count = 1;
            } else if best_idx.is_some() && s == best_score {
                tie_count += 1;
                if tie_rng.gen_range(0..tie_count) == 0 {
                    best_idx = Some(i);
                }
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
///
/// 同スコアの近傍が複数あるときは `tie_rng` から reservoir sampling で
/// 1 つを一様選択する（`tie_rng` は `sm` の内部 RNG とは独立）。
fn hill_climb_smoothed_fast<F>(
    prob: &GraphPartitionProblem,
    start: &Partition,
    start_cuts: &[i32],
    start_cut: i32,
    start_t: usize,
    start_f: usize,
    sm: &mut F,
    tie_rng: &mut Mt19937GenRand64,
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
        let mut tie_count: u64 = 0;
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
                tie_count = 1;
            } else if best_idx.is_some() && s == best_smoothed {
                tie_count += 1;
                if tie_rng.gen_range(0..tie_count) == 0 {
                    best_idx = Some(i);
                }
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
    tie_rng: &mut Mt19937GenRand64,
) -> StepRecord
where
    F: FnMut(&Partition, &[i32], i32, usize, usize) -> f64,
{
    let current_real = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);

    if no_smoothing {
        // smoothed-basin == real-basin → 1 回の HC で 4 フィールド埋め尽くし
        let (_, _, bc, bt, bf) =
            hill_climb_real_fast(prob, current, cuts_at, cur_cut, cur_t, cur_f, tie_rng);
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
            hill_climb_smoothed_fast(prob, current, cuts_at, cur_cut, cur_t, cur_f, sm, tie_rng);
        let basin_smoothed_from_smoothed = sm(&basin_sm_pt, &basin_sm_cuts, bsmc, bsmt, bsmf);
        let basin_real_from_smoothed = GraphPartitionProblem::score_from_state(bsmc, bsmt, bsmf);

        // 元空間での山登り
        let (basin_re_pt, basin_re_cuts, brc, brt, brf) =
            hill_climb_real_fast(prob, current, cuts_at, cur_cut, cur_t, cur_f, tie_rng);
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
    // ベイスン山登りのタイブレーク専用 RNG。SA 本体・スムージングの乱数列とは
    // 独立なので、final_partition には影響しない。
    let mut tie_rng = Mt19937GenRand64::new(seed ^ TIEBREAK_SALT);
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
        &mut tie_rng,
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
                    &mut tie_rng,
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
    weight: f64,
    cfg: &RunConfig,
    seed: u64,
) -> (Partition, Vec<StepRecord>) {
    let n = prob.neighbour_size();
    // 重みは 0〜1 にクランプ（w=0: 平滑化なし相当、w=1: 全近傍平均）。
    let weight = weight.clamp(0.0, 1.0);
    run_sa_generic(prob, cfg, seed, false, move |p, cuts_at, c, t, f| {
        if n == 0 {
            return GraphPartitionProblem::score_from_state(c, t, f);
        }
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
// τ-Extremal Optimization（厳密バランスのスワップ版）
// ============================================================================

/// べき乗則 P(k) ∝ k^{-τ} の累積分布関数を構築する（1-indexed ランク）。
///
/// `cdf[k-1] = (Σ_{j=1}^{k} j^{-τ}) / (Σ_{j=1}^{n} j^{-τ})`。
/// `u ~ U(0,1)` を二分探索することでランク k を引ける。
pub(crate) fn build_power_law_cdf(n: usize, tau: f64) -> Vec<f64> {
    let mut cdf = Vec::with_capacity(n);
    let mut cumulative = 0.0;
    for k in 1..=n {
        cumulative += (k as f64).powf(-tau);
        cdf.push(cumulative);
    }
    let z = cumulative;
    for val in &mut cdf {
        *val /= z;
    }
    cdf
}

/// 厳密にバランスした初期分割を作る（`t = n - n/2` 個を `true`、残りを `false`）。
/// Fisher–Yates でシャッフルするため、`rng` 由来で再現可能。
fn balanced_init(n: usize, rng: &mut Mt19937GenRand64) -> Partition {
    let t = n - n / 2; // 偶数: N/2、奇数: ⌈N/2⌉
    let mut p = vec![false; n];
    for slot in p.iter_mut().take(t) {
        *slot = true;
    }
    // Fisher–Yates シャッフル。
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        p.swap(i, j);
    }
    p
}

/// スワップ近傍での最急降下山登り（厳密バランスを保ったまま、改善スワップが尽きるまで）。
///
/// 各ステップで全 (A,B) ペアのうち実スコアを最も下げるスワップ 1 つを選んで適用する
/// （[`hill_climb_real_fast`] のスワップ版）。同スコアのスワップが複数あるときは `tie_rng` から
/// reservoir sampling で一様選択する。バランスは不変なので到達点も `|A|=|B|`。
///
/// 候補スコアは v1 を一時フリップしてから `delta_apply_cached(v2)` で O(1) 評価する
/// （隣接ペアの二重計上も整数状態の連鎖で正しく処理される）。1 降下ステップは
/// O(|A|·|B|) = O(N²)。スコアは厳密バランス下で整数なので、改善スワップは毎回カットを
/// 1 以上下げ、有限ステップで停止する。
///
/// PHASE 2: Kernighan–Lin 風のゲインバケットで各頂点の最良スワップゲインを増分更新すれば、
/// 1 ステップを O(N·deg) 程度に削減できる。
fn hill_climb_swap_fast(
    prob: &GraphPartitionProblem,
    start: &Partition,
    start_cuts: &[i32],
    start_cut: i32,
    start_t: usize,
    start_f: usize,
    tie_rng: &mut Mt19937GenRand64,
) -> (Partition, Vec<i32>, i32, usize, usize) {
    let mut current = start.clone();
    let mut cuts_at = start_cuts.to_vec();
    let (mut cur_cut, mut cur_t, mut cur_f) = (start_cut, start_t, start_f);
    let mut cur_score = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
    let n = current.len();

    loop {
        let mut best: Option<(usize, usize)> = None;
        let mut best_score = cur_score;
        let mut tie_count: u64 = 0;

        for v1 in 0..n {
            if !current[v1] {
                continue; // v1 は A 集合（true 側）の代表のみ
            }
            let (c1, t1, f1, _) =
                prob.delta_apply_cached(&current, &cuts_at, v1, cur_cut, cur_t, cur_f);
            prob.flip_vertex(&mut current, &mut cuts_at, v1); // 一時フリップ
            for v2 in 0..n {
                if current[v2] || v2 == v1 {
                    continue; // v2 は B 集合（false 側、ただし v1 自身は除く）
                }
                let s = prob.delta_apply_cached(&current, &cuts_at, v2, c1, t1, f1).3;
                if s < best_score {
                    best_score = s;
                    best = Some((v1, v2));
                    tie_count = 1;
                } else if best.is_some() && s == best_score {
                    tie_count += 1;
                    if tie_rng.gen_range(0..tie_count) == 0 {
                        best = Some((v1, v2));
                    }
                }
            }
            prob.flip_vertex(&mut current, &mut cuts_at, v1); // アンフリップ
        }

        match best {
            Some((v1, v2)) => {
                let (c1, t1, f1, _) =
                    prob.delta_apply_cached(&current, &cuts_at, v1, cur_cut, cur_t, cur_f);
                prob.flip_vertex(&mut current, &mut cuts_at, v1);
                let (c2, t2, f2, ns) =
                    prob.delta_apply_cached(&current, &cuts_at, v2, c1, t1, f1);
                prob.flip_vertex(&mut current, &mut cuts_at, v2);
                cur_cut = c2;
                cur_t = t2;
                cur_f = f2;
                cur_score = ns;
            }
            None => break,
        }
    }
    (current, cuts_at, cur_cut, cur_t, cur_f)
}

/// スワップ近傍ソルバ（[`run_eo`] / [`run_sa_swap`]）用のスナップショット。
///
/// 平滑化なし（smoothed == real）。`current_*` は現在解の生スコア、`basin_*`（4 フィールド）は
/// **スワップ近傍の局所最適**（`hill_climb_swap_fast` で算出）。SA フリップ版の
/// `make_snapshot_fast(no_smoothing=true)` のスワップ版に相当する。
fn make_swap_snapshot(
    prob: &GraphPartitionProblem,
    current: &Partition,
    cuts_at: &[i32],
    cur_cut: i32,
    cur_t: usize,
    cur_f: usize,
    step: usize,
    tie_rng: &mut Mt19937GenRand64,
) -> StepRecord {
    let current_real = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
    let (_, _, bc, bt, bf) =
        hill_climb_swap_fast(prob, current, cuts_at, cur_cut, cur_t, cur_f, tie_rng);
    let basin = GraphPartitionProblem::score_from_state(bc, bt, bf);
    StepRecord {
        step,
        current_smoothed: current_real,
        current_real,
        basin_smoothed_from_smoothed: basin,
        basin_real_from_smoothed: basin,
        basin_smoothed_from_real: basin,
        basin_real_from_real: basin,
    }
}

/// スペック忠実な τ-EO（厳密バランスのスワップ版）の高速パス。
///
/// - **適応度** λ_v = g_v / deg_v = (deg_v − cuts_at[v]) / deg_v（孤立頂点は 1.0）。
///   λ が小さいほど「悪い」（誤った集合にいる疑いが濃い）。
/// - **統一ランク**: 全 N 頂点を λ 昇順にランク付けし、`P(k) ∝ k^{-τ}` でランク k1 を引く。
///   v1 の所属集合と反対側になるまで k2 を引き直し（再抽選上限超過時は反対集合から一様ランダム）、
///   v1 と v2 を**スワップ**する（2 連続フリップ）。これにより `|A| = |B|` が全ステップで厳密維持される。
/// - **無条件受理**: カットの増減にかかわらず常にスワップを適用する。
/// - **S_best 別途保存**: 最良解を `best` に保持し、`final_partition` として返す。
///
/// バランス維持時はペナルティ項が一定（偶数 N なら 0）なので、実スコアの最小化 ≡ カットの最小化。
fn run_eo(
    prob: &GraphPartitionProblem,
    cfg: &RunConfig,
    seed: u64,
    tau: f64,
) -> (Partition, Vec<StepRecord>) {
    let mut rng = Mt19937GenRand64::new(seed);
    let n = prob.neighbour_size();

    let mut current = balanced_init(n, &mut rng);
    let mut cur_cut = prob.count_cut_edges(&current);
    let (mut cur_t, mut cur_f) = get_partition_sizes(&current);
    let mut cuts_at = prob.compute_cuts_at(&current);

    let mut best = current.clone();
    let mut best_score = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);

    // 頂点次数は不変なので 1 度だけ取得する。
    let degrees: Vec<usize> = (0..n)
        .map(|v| prob.graph().adjacency_list[v].len())
        .collect();

    // ベイスン算出（スワップ降下）のタイブレーク専用 RNG。本体 rng とは独立なので
    // final_partition には影響せず、records のベイスン値のみがタイブレークの影響を受ける。
    let mut tie_rng = Mt19937GenRand64::new(seed ^ TIEBREAK_SALT);

    let max_iter = cfg.iterations();
    let snap_steps = logarithmic_steps(max_iter);
    let mut snap_iter = snap_steps.iter().copied().peekable();
    let mut records = Vec::with_capacity(snap_steps.len() + 1);

    // 初期スナップショット (step = 0)。ベイスンはスワップ近傍の局所最適。
    records.push(make_swap_snapshot(
        prob, &current, &cuts_at, cur_cut, cur_t, cur_f, 0, &mut tie_rng,
    ));

    // スワップには両集合に最低 1 頂点ずつ必要。
    if n < 2 {
        return (best, records);
    }

    let cdf = build_power_law_cdf(n, tau);
    // 反対集合の頂点が当たるまでの k2 再抽選上限。超過時は反対集合から一様ランダムにフォールバック。
    const MAX_RESELECT: usize = 50;

    for it in 1..=max_iter {
        // --- 適応度 λ を全頂点について計算（cuts_at から O(N)）---
        // PHASE 2: 毎ステップの全ソートは O(N log N)。順序統計木/ヒープで λ を保持し、
        // スワップで変化する v1,v2 とその隣接頂点の λ のみ局所更新すれば O(deg log N) にできる。
        let lambdas: Vec<f64> =
            (0..n).map(|v| swap_fitness(degrees[v], cuts_at[v])).collect();

        // λ 昇順ランク（order[0] = 最悪 = ランク 1）。
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| lambdas[a].partial_cmp(&lambdas[b]).unwrap());

        // --- 統一ランクから v1 を引く（同率群は平均化規則で等確率）---
        let (v1, _k1) = select_eo_rank(&lambdas, &order, &cdf, rng.r#gen::<f64>());
        let set1 = current[v1];

        // --- 反対集合から v2 を引く（再抽選上限 → 一様ランダムフォールバック）---
        let mut v2 = None;
        for _ in 0..MAX_RESELECT {
            let (cand, _k2) = select_eo_rank(&lambdas, &order, &cdf, rng.r#gen::<f64>());
            if current[cand] != set1 {
                v2 = Some(cand);
                break;
            }
        }
        let v2 = v2.unwrap_or_else(|| {
            // フォールバック: 反対集合から一様ランダム（両集合とも非空なので必ず終了）。
            loop {
                let cand = rng.gen_range(0..n);
                if current[cand] != set1 {
                    break cand;
                }
            }
        });

        // --- 無条件スワップ（2 連続フリップ）---
        // v1 をフリップ → 更新後の cuts_at で v2 をフリップ。隣接していても正しく処理される。
        let (c1, t1, f1, _) =
            prob.delta_apply_cached(&current, &cuts_at, v1, cur_cut, cur_t, cur_f);
        prob.flip_vertex(&mut current, &mut cuts_at, v1);
        let (c2, t2, f2, _) = prob.delta_apply_cached(&current, &cuts_at, v2, c1, t1, f1);
        prob.flip_vertex(&mut current, &mut cuts_at, v2);
        cur_cut = c2;
        cur_t = t2; // スワップ後は元の値に戻る（バランス維持）。
        cur_f = f2;

        let real_score = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
        if real_score < best_score {
            best_score = real_score;
            best = current.clone();
        }

        // デバッグビルドでの不変条件アサーション（バランス・カット・cuts_at の drift 検出）。
        #[cfg(debug_assertions)]
        if it % 1000 == 0 {
            debug_assert_eq!(cur_t, n - n / 2, "balance drift at it={}", it);
            debug_assert_eq!(
                cur_cut,
                prob.count_cut_edges(&current),
                "cut drift at it={}",
                it
            );
            let (rt, rf) = get_partition_sizes(&current);
            debug_assert_eq!((cur_t, cur_f), (rt, rf), "size drift at it={}", it);
            debug_assert_eq!(
                cuts_at,
                prob.compute_cuts_at(&current),
                "cuts_at drift at it={}",
                it
            );
        }

        if let Some(&want) = snap_iter.peek() {
            if it == want {
                records.push(make_swap_snapshot(
                    prob, &current, &cuts_at, cur_cut, cur_t, cur_f, it, &mut tie_rng,
                ));
                snap_iter.next();
            }
        }
    }

    (best, records)
}

/// スワップ版 EO の次数正規化適応度 `λ0 = g/deg = (deg - cuts)/deg`（孤立頂点は 1.0）。
/// `run_eo` の λ 計算そのものであり、フリップ近傍版の新適応度（`eo_flip_lambda_mul_alpha` 等）
/// でも「λ0 はスワップ版 EO の適応度と同じ」を表す共通部品として使う。
fn swap_fitness(deg: usize, cuts: i32) -> f64 {
    if deg == 0 {
        1.0
    } else {
        (deg as f64 - cuts as f64) / deg as f64
    }
}

/// τ-EO の頂点抽選（**同率群の重みを平均化する最終規則**）。
///
/// `order` は λ 昇順（先頭 = 最悪 = 順位 1）に並べた頂点列、`cum` は
/// [`build_power_law_cdf`] の出力（`cum[k] = Σ_{j≤k} (j+1)^{-τ}` を正規化したもの、
/// `cum[n-1] = 1`）、`u ∈ [0,1)` は一様乱数。乱数消費は 1 draw のみ。
///
/// # 規則
///
/// λ が bit 等値の頂点は「同率群」を成す。位置 `[s, e)` を占める同率群について:
///
/// ```text
/// 群の合計重み = Σ_{k=s}^{e-1} w_k        （w_k = (k+1)^{-τ} / Z）
/// 各メンバーの重み = (Σ_{k=s}^{e-1} w_k) / (e - s)
/// ```
///
/// つまり **同率群が受け取る選択頻度の合計は、その群が同率でなかった場合の合計に等しく、
/// 群内の各頂点は等頻度**になる。例えば λ が 1-2-2-2-5 位の並びなら、2〜4 位ぶんの
/// 重み `w₂+w₃+w₄` を同率の 3 頂点で等分する。
///
/// 同率が無ければ各群のサイズが 1 なので、素の `P(k) ∝ k^{-τ}` 抽選と厳密に一致する。
///
/// 戻り値は `(選択頂点, その昇順位置 0-indexed)`。
pub(crate) fn select_eo_rank(
    lambdas: &[f64],
    order: &[usize],
    cum: &[f64],
    u: f64,
) -> (usize, usize) {
    let n = order.len();
    debug_assert!(n > 0);
    let mut s = 0usize;
    while s < n {
        // 同値 λ の極大ブロック [s, e) を切り出す。
        let mut e = s + 1;
        while e < n && lambdas[order[e]] == lambdas[order[s]] {
            e += 1;
        }
        let lo = if s == 0 { 0.0 } else { cum[s - 1] };
        let hi = cum[e - 1];
        // 最終ブロックは丸め誤差で u ≥ cum[n-1] になっても受け止める。
        if u < hi || e == n {
            let m = e - s;
            let per = (hi - lo) / m as f64;
            let off = if per > 0.0 {
                (((u - lo).max(0.0) / per) as usize).min(m - 1)
            } else {
                0
            };
            let pos = s + off;
            return (order[pos], pos);
        }
        s = e;
    }
    // ブロック走査は必ず最終ブロックで return するので到達しない。
    unreachable!("select_eo_rank: ブロック走査が範囲を抜けた (n={}, u={})", n, u)
}

// ============================================================================
// τ-Extremal Optimization（フリップ近傍版・SA と同一ランドスケープ）
// ============================================================================

/// flip近傍版 EO の適応度 λ_eff を計算する。
///
/// バランスペナルティを「悪い辺 / 良い辺」として g/deg に織り込む対称版:
/// - `improvement = alpha_eo·(|diff|^p − |diff_after|^p)`（>0: flip でペナルティ減 / <0: 増）、`q = |improvement|`
/// - `improvement > 0`（多数派側）: `q` を全辺(deg)と悪い辺(b)に足す → `λ_eff = g/(deg+q)`（小さく＝選ばれやすく）
/// - `improvement < 0`（少数派側）: `q` を全辺(deg)と良い辺(g)に足す → `λ_eff = (g+q)/(deg+q)`（大きく＝守られる）
/// - `improvement = 0`: 変化なし → `λ_eff = g/deg`
/// - `deg+q = 0`（孤立かつ q=0）: `λ_eff = 1.0`
///
/// `deg = 0` かつ多数派側（q>0）なら `λ_eff = 0/(0+q) = 0` となり、カット0コストの
/// 自由な是正フリップとして最優先で選ばれる。
///
/// `alpha_eo`（係数）と `p`（diff の指数）は手選択の内部勾配だけを調整するハイパーパラメータで、
/// 目的関数 `score = cut + ALPHA·diff²` には影響しない。`p` は分数を取りうるため `|diff|` に対して
/// 適用する（`diff`/`diff_after` は符号付き）。既定 `alpha_eo = ALPHA(=0.05)`・`p = 2.0` では
/// `|diff|² = diff²` となり従来の `ALPHA·(diff² − diff_after²)` を byte 完全再現する。
fn eo_flip_lambda(deg: usize, cuts: i32, diff_after: i64, diff: i64, alpha_eo: f64, p: f64) -> f64 {
    let deg_f = deg as f64;
    let g = deg_f - cuts as f64;
    let improvement =
        alpha_eo * ((diff.abs() as f64).powf(p) - (diff_after.abs() as f64).powf(p));
    let q = improvement.abs();
    let deg_eff = deg_f + q;
    if deg_eff == 0.0 {
        1.0
    } else if improvement > 0.0 {
        g / deg_eff
    } else if improvement < 0.0 {
        (g + q) / deg_eff
    } else {
        g / deg_f
    }
}

/// 頂点が現在「多数派集合」（自集合のほうが反対集合より頂点数が多い）に属すかどうか。
/// 均衡時（`t == f`）はどちらの集合も多数派としない（`false` を返す）。
fn is_majority_side(in_true: bool, t: usize, f: usize) -> bool {
    if in_true {
        t > f
    } else {
        f > t
    }
}

/// フリップ近傍版 EO の適応度（乗算 α 版）: `λ = λ0 · λ1`。
///
/// `λ0` はスワップ版 EO と同じ [`swap_fitness`]。`λ1` は対象頂点が多数派なら `alpha`、
/// 少数派なら `1.0`（`is_majority` は [`is_majority_side`] で判定）。
fn eo_flip_lambda_mul_alpha(deg: usize, cuts: i32, is_majority: bool, alpha: f64) -> f64 {
    let lambda1 = if is_majority { alpha } else { 1.0 };
    swap_fitness(deg, cuts) * lambda1
}

/// フリップ近傍版 EO の適応度（加算 β 版）: `λ = β·λ0 + λ1`。
///
/// `λ0` は [`eo_flip_lambda_mul_alpha`] と同じ。`λ1` は多数派なら `0.0`、少数派なら `1.0`。
fn eo_flip_lambda_add_beta(deg: usize, cuts: i32, is_majority: bool, beta: f64) -> f64 {
    let lambda1 = if is_majority { 0.0 } else { 1.0 };
    beta * swap_fitness(deg, cuts) + lambda1
}

/// フリップ近傍版 EO の適応度（乗算 γ 版）: `λ = λ0 · λ1`。
///
/// `λ0` は [`eo_flip_lambda_mul_alpha`] と同じ。`λ1` は多数派なら `gamma`
/// （= 少数派集合の頂点数 / (N/2)、呼び出し側が毎ステップ算出して渡す）、少数派なら `1.0`。
fn eo_flip_lambda_mul_gamma(deg: usize, cuts: i32, is_majority: bool, gamma: f64) -> f64 {
    let lambda1 = if is_majority { gamma } else { 1.0 };
    swap_fitness(deg, cuts) * lambda1
}

/// 全頂点の EoFlip 適応度 λ を `out` に書き込む（長さは呼び出し側で `n` に揃える）。
///
/// [`run_eo_flip`] の本体ループと、フリップ選択トレースのシャドープローブ評価の
/// 双方が同じ実装を使う（順位付けの定義が一致することを保証する）。
fn compute_eo_flip_lambdas(
    fitness: &EoFlipFitnessSpec,
    current: &Partition,
    cuts_at: &[i32],
    degrees: &[usize],
    cur_t: usize,
    cur_f: usize,
    out: &mut [f64],
) {
    let n = current.len();
    let diff = cur_t as i64 - cur_f as i64;
    match *fitness {
        EoFlipFitnessSpec::Legacy { alpha_eo, diff_exp } => {
            for i in 0..n {
                let diff_after = if current[i] { diff - 2 } else { diff + 2 };
                out[i] =
                    eo_flip_lambda(degrees[i], cuts_at[i], diff_after, diff, alpha_eo, diff_exp);
            }
        }
        EoFlipFitnessSpec::MulAlpha { alpha } => {
            for i in 0..n {
                let is_majority = is_majority_side(current[i], cur_t, cur_f);
                out[i] = eo_flip_lambda_mul_alpha(degrees[i], cuts_at[i], is_majority, alpha);
            }
        }
        EoFlipFitnessSpec::AddBeta { beta } => {
            for i in 0..n {
                let is_majority = is_majority_side(current[i], cur_t, cur_f);
                out[i] = eo_flip_lambda_add_beta(degrees[i], cuts_at[i], is_majority, beta);
            }
        }
        EoFlipFitnessSpec::MulGamma => {
            let minority = cur_t.min(cur_f) as f64;
            let gamma = minority / (n as f64 / 2.0);
            for i in 0..n {
                let is_majority = is_majority_side(current[i], cur_t, cur_f);
                out[i] = eo_flip_lambda_mul_gamma(degrees[i], cuts_at[i], is_majority, gamma);
            }
        }
    }
}

// ============================================================================
// フリップ選択トレース（EoFlip 系専用の計装）
// ============================================================================

/// フリップ選択トレースの指定（バッチ単位）。
///
/// `probes` は「同一状態に対して順位付けを比較する」シャドープローブの適応度設定。
/// 各スナップショットステップで軌道を変えずに全プローブの λ を計算し、実際の
/// 選択との一致率・bottom-m Jaccard・Kendall τ_b を記録する（乱数は消費しない）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlipTraceSpec {
    #[serde(default)]
    pub probes: Vec<EoFlipFitnessSpec>,
}

/// 10 進ディケード（step ∈ [10^d, 10^{d+1}-1]）ごとの選択挙動カウンタ。
///
/// すべて「そのディケード内で実行されたステップ」に関する集計で、
/// `steps = maj+min+bal = dcut_neg+dcut_zero+dcut_pos = ddiff_dec+ddiff_eq+ddiff_inc`。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlipDecadeStats {
    /// このディケードで実行したステップ数。
    pub steps: u64,
    /// 選択頂点が多数派集合に属していた回数（[`is_majority_side`] 準拠）。
    pub maj: u64,
    /// 選択頂点が少数派集合に属していた回数（均衡時を除く）。
    pub min: u64,
    /// 均衡状態（t == f、多数派なし）で選択した回数。
    pub bal: u64,
    /// フリップでカットが減った / 不変 / 増えた回数。
    pub dcut_neg: u64,
    pub dcut_zero: u64,
    pub dcut_pos: u64,
    /// フリップで |diff| が減った / 不変 / 増えた回数（|diff| 不変は |±1|→|∓1| 型のみ）。
    pub ddiff_dec: u64,
    pub ddiff_eq: u64,
    pub ddiff_inc: u64,
    /// 直前（lag-1）/ 2 手前（lag-2）と同一頂点を再フリップした回数（振動検出）。
    pub lag1: u64,
    pub lag2: u64,
    /// 選択頂点と λ が完全一致（bit 等値）する頂点数 w の log2 ビン別ヒストグラム
    /// （bin = ⌈log2(w)⌉: w=1→0, 2→1, 3-4→2, 5-8→3, …。長さ 16 固定）。
    pub tie_hist: Vec<u64>,
    /// 選択頂点の属性和（平均は steps で割る）: 次数・カット辺数・λ0 = g/deg。
    pub deg_sum: u64,
    pub cuts_sum: i64,
    pub lambda0_sum: f64,
    /// Δcut の総和（平均カット押し上げ/押し下げ）。
    pub dcut_sum: i64,
    /// フリップ前 |diff| の総和（バランス軌道）。
    pub absdiff_sum: u64,
}

/// tie 幅ヒストグラムのビン数（w ≤ 2^15 = 32768 頂点まで対応）。
const TIE_HIST_BINS: usize = 16;

/// シャドープローブ 1 設定の全イベント。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlipProbeStats {
    /// [`EoFlipFitnessSpec::label`]。
    pub label: String,
    pub events: Vec<FlipProbeEvent>,
}

/// スナップショットステップ 1 回分のプローブ測定。
///
/// いずれも「実際に走っている適応度」の順位・選択との比較:
/// - `agree`: 同じ抽選ランク k のもとでプローブも同じ頂点を選んだか
/// - `jaccard8` / `jaccard32`: 昇順 bottom-8 / bottom-32 集合の Jaccard 係数
/// - `kendall_b`: λ ベクトル間の Kendall τ_b（tie 補正つき、O(N²) 計算）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlipProbeEvent {
    pub step: usize,
    pub agree: bool,
    pub jaccard8: f64,
    pub jaccard32: f64,
    pub kendall_b: f64,
}

/// フリップ選択トレースの出力（`seed_<seed>_trace.json`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlipSelectionTrace {
    pub graph_id: String,
    pub config_id: String,
    pub seed: u64,
    /// 頂点数。
    pub n: usize,
    pub max_iter: usize,
    /// この run 自身の適応度ラベル（[`EoFlipFitnessSpec::label`]）。
    pub fitness_label: String,
    pub tau: f64,
    /// ディケード別カウンタ（index d = step ∈ [10^d, 10^{d+1}-1]）。
    pub decades: Vec<FlipDecadeStats>,
    /// 抽選ランク k の全期間ヒストグラム（長さ n）。
    pub rank_hist: Vec<u64>,
    /// 頂点別フリップ回数の全期間ヒストグラム（長さ n）。集中度指標は分析側で算出。
    pub flip_counts: Vec<u64>,
    pub probes: Vec<FlipProbeStats>,
}

/// Kendall τ_b（tie 補正つき）。O(N²) の全ペア走査。
///
/// 分母 0（少なくとも一方が全 tie）のときは 0.0 を返す。
fn kendall_tau_b(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let (mut conc, mut disc, mut tie_x, mut tie_y) = (0u64, 0u64, 0u64, 0u64);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let sx = if dx > 0.0 { 1 } else if dx < 0.0 { -1 } else { 0 };
            let sy = if dy > 0.0 { 1 } else if dy < 0.0 { -1 } else { 0 };
            match (sx, sy) {
                (0, 0) => {}
                (0, _) => tie_x += 1,
                (_, 0) => tie_y += 1,
                _ if sx == sy => conc += 1,
                _ => disc += 1,
            }
        }
    }
    let denom = (((conc + disc + tie_x) as f64) * ((conc + disc + tie_y) as f64)).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (conc as f64 - disc as f64) / denom
    }
}

/// 昇順 bottom-m 集合同士の Jaccard 係数（m は n でクリップ）。
fn bottom_m_jaccard(order_a: &[usize], order_b: &[usize], m: usize) -> f64 {
    let n = order_a.len();
    let m = m.min(n);
    if m == 0 {
        return 1.0;
    }
    let set_a: std::collections::HashSet<usize> = order_a[..m].iter().copied().collect();
    let inter = order_b[..m].iter().filter(|v| set_a.contains(v)).count();
    // |A∪B| = 2m - |A∩B|
    inter as f64 / (2 * m - inter) as f64
}

/// 値を小数 6 桁へ丸める（トレース JSON の肥大化防止）。
fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// [`run_eo_flip`] 内で選択挙動を集計する内部コレクタ。
///
/// 乱数を一切消費しないため、トレースの有無で軌道は変わらない。
struct FlipTraceCollector<'a> {
    spec: &'a FlipTraceSpec,
    decades: Vec<FlipDecadeStats>,
    /// 現在のディケード index と次のディケード開始ステップ。
    decade_idx: usize,
    next_decade_start: usize,
    rank_hist: Vec<u64>,
    flip_counts: Vec<u64>,
    prev1: Option<usize>,
    prev2: Option<usize>,
    /// プローブ測定を行うステップ（`logarithmic_steps` と同一）。
    probe_steps: std::iter::Peekable<std::vec::IntoIter<usize>>,
    probe_events: Vec<Vec<FlipProbeEvent>>,
    /// プローブ用の再利用バッファ。
    probe_lambdas: Vec<f64>,
    probe_order: Vec<usize>,
}

impl<'a> FlipTraceCollector<'a> {
    fn new(spec: &'a FlipTraceSpec, n: usize, max_iter: usize) -> Self {
        let n_decades = {
            // ディケード数 = ⌊log10(max_iter)⌋ + 1（max_iter=10^6 なら d=0..6 の 7 個）。
            let mut d = 1;
            let mut thr = 10usize;
            while thr <= max_iter {
                d += 1;
                thr = match thr.checked_mul(10) {
                    Some(t) => t,
                    None => break,
                };
            }
            d
        };
        let mut decades = Vec::with_capacity(n_decades);
        for _ in 0..n_decades {
            decades.push(FlipDecadeStats { tie_hist: vec![0; TIE_HIST_BINS], ..Default::default() });
        }
        Self {
            spec,
            decades,
            decade_idx: 0,
            next_decade_start: 10,
            rank_hist: vec![0; n],
            flip_counts: vec![0; n],
            prev1: None,
            prev2: None,
            probe_steps: logarithmic_steps(max_iter).into_iter().peekable(),
            probe_events: vec![Vec::new(); spec.probes.len()],
            probe_lambdas: vec![0.0; n],
            probe_order: (0..n).collect(),
        }
    }

    /// 1 ステップ分のカウンタ更新（フリップ適用前の状態 + delta 計算結果を受け取る）。
    #[allow(clippy::too_many_arguments)]
    fn record_step(
        &mut self,
        it: usize,
        k: usize,
        idx: usize,
        lambdas: &[f64],
        current: &Partition,
        cuts_at: &[i32],
        degrees: &[usize],
        cur_cut: i32,
        cur_t: usize,
        cur_f: usize,
        new_cut: i32,
        new_t: usize,
        new_f: usize,
    ) {
        while it >= self.next_decade_start {
            self.decade_idx += 1;
            self.next_decade_start = self.next_decade_start.saturating_mul(10);
        }
        let d = &mut self.decades[self.decade_idx];
        d.steps += 1;

        if cur_t == cur_f {
            d.bal += 1;
        } else if is_majority_side(current[idx], cur_t, cur_f) {
            d.maj += 1;
        } else {
            d.min += 1;
        }

        let dcut = new_cut - cur_cut;
        match dcut.cmp(&0) {
            std::cmp::Ordering::Less => d.dcut_neg += 1,
            std::cmp::Ordering::Equal => d.dcut_zero += 1,
            std::cmp::Ordering::Greater => d.dcut_pos += 1,
        }
        d.dcut_sum += dcut as i64;

        let absdiff_before = (cur_t as i64 - cur_f as i64).unsigned_abs();
        let absdiff_after = (new_t as i64 - new_f as i64).unsigned_abs();
        match absdiff_after.cmp(&absdiff_before) {
            std::cmp::Ordering::Less => d.ddiff_dec += 1,
            std::cmp::Ordering::Equal => d.ddiff_eq += 1,
            std::cmp::Ordering::Greater => d.ddiff_inc += 1,
        }
        d.absdiff_sum += absdiff_before;

        if self.prev1 == Some(idx) {
            d.lag1 += 1;
        }
        if self.prev2 == Some(idx) {
            d.lag2 += 1;
        }
        self.prev2 = self.prev1;
        self.prev1 = Some(idx);

        d.deg_sum += degrees[idx] as u64;
        d.cuts_sum += cuts_at[idx] as i64;
        d.lambda0_sum += swap_fitness(degrees[idx], cuts_at[idx]);

        // tie 幅（選択頂点の λ と bit 等値の頂点数、自分含む ≥1）。
        let sel_lambda = lambdas[idx];
        let w = lambdas.iter().filter(|&&l| l == sel_lambda).count() as u64;
        let bin = (u64::BITS - (w - 1).leading_zeros()) as usize;
        d.tie_hist[bin.min(TIE_HIST_BINS - 1)] += 1;

        self.rank_hist[k] += 1;
        self.flip_counts[idx] += 1;
    }

    /// スナップショットステップならシャドープローブを評価する（フリップ適用前に呼ぶ）。
    ///
    /// `agree` は「本体と同じ乱数 `u` のもとでプローブも同じ頂点を選んだか」
    /// （プローブ側の λ 順序に対して同じ [`select_eo_rank`] を適用する）。
    #[allow(clippy::too_many_arguments)]
    fn maybe_probe(
        &mut self,
        it: usize,
        idx: usize,
        lambdas: &[f64],
        order: &[usize],
        current: &Partition,
        cuts_at: &[i32],
        degrees: &[usize],
        cur_t: usize,
        cur_f: usize,
        cdf: &[f64],
        u: f64,
    ) {
        match self.probe_steps.peek() {
            Some(&want) if it == want => {
                self.probe_steps.next();
            }
            _ => return,
        }
        for (pi, probe) in self.spec.probes.iter().enumerate() {
            compute_eo_flip_lambdas(
                probe,
                current,
                cuts_at,
                degrees,
                cur_t,
                cur_f,
                &mut self.probe_lambdas,
            );
            // 本体と同一の順位付け規約（昇順・安定ソート）。
            for (i, o) in self.probe_order.iter_mut().enumerate() {
                *o = i;
            }
            let pl = &self.probe_lambdas;
            self.probe_order
                .sort_by(|&a, &b| pl[a].partial_cmp(&pl[b]).unwrap());
            let (sel_p, _) = select_eo_rank(&self.probe_lambdas, &self.probe_order, cdf, u);
            self.probe_events[pi].push(FlipProbeEvent {
                step: it,
                agree: sel_p == idx,
                jaccard8: round6(bottom_m_jaccard(order, &self.probe_order, 8)),
                jaccard32: round6(bottom_m_jaccard(order, &self.probe_order, 32)),
                kendall_b: round6(kendall_tau_b(&self.probe_lambdas, lambdas)),
            });
        }
    }

    /// 集計結果を出力形式へ変換する（graph/config/seed は呼び出し側で埋める）。
    fn finish(self, n: usize, max_iter: usize, fitness_label: String, tau: f64) -> FlipSelectionTrace {
        FlipSelectionTrace {
            graph_id: String::new(),
            config_id: String::new(),
            seed: 0,
            n,
            max_iter,
            fitness_label,
            tau,
            decades: self.decades,
            rank_hist: self.rank_hist,
            flip_counts: self.flip_counts,
            probes: self
                .spec
                .probes
                .iter()
                .zip(self.probe_events)
                .map(|(p, events)| FlipProbeStats { label: p.label(), events })
                .collect(),
        }
    }
}

/// フリップ近傍版 τ-EO の高速パス。
///
/// SA（`run_sa_*`）と **同一の近傍（単一フリップ）・目的関数（cut + α·diff²）・初期化
/// （`random_solution`）・ベイスン算出（`make_snapshot_fast` + `hill_climb_real_fast`）**を共有し、
/// 違いは「内側の1手の選び方」だけ:
/// 全頂点を `fitness`（[`EoFlipFitnessSpec`]）で計算した適応度の昇順にランク付けし、
/// べき乗則 `P(k)∝k^{-τ}` で1頂点を引いて **無条件にフリップ**する（受理判定なし）。
/// 最良解 `best` を別途保存して返す。
///
/// バランスはペナルティ項で扱う（厳密制約ではない）ため、スワップ版 `run_eo` と異なり
/// 解は厳密 N/2 にはならない。これにより SA と StepRecord が1対1で比較できる。
///
/// `trace_spec` を渡すと選択挙動を集計した [`FlipSelectionTrace`] を追加で返す。
/// トレースは乱数を消費しないため、有無にかかわらず軌道・records はバイト一致する。
fn run_eo_flip(
    prob: &GraphPartitionProblem,
    cfg: &RunConfig,
    seed: u64,
    tau: f64,
    fitness: EoFlipFitnessSpec,
    trace_spec: Option<&FlipTraceSpec>,
) -> (Partition, Vec<StepRecord>, Option<FlipSelectionTrace>) {
    let mut rng = Mt19937GenRand64::new(seed);
    // ベイスン山登りのタイブレーク専用 RNG（SA と同一の派生）。
    let mut tie_rng = Mt19937GenRand64::new(seed ^ TIEBREAK_SALT);

    let mut current: Partition = prob.random_solution(&mut rng);
    let mut cur_cut = prob.count_cut_edges(&current);
    let (mut cur_t, mut cur_f) = get_partition_sizes(&current);
    let mut cuts_at = prob.compute_cuts_at(&current);

    let n = prob.neighbour_size();
    let degrees: Vec<usize> = (0..n)
        .map(|v| prob.graph().adjacency_list[v].len())
        .collect();

    let mut best = current.clone();
    let mut best_score = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);

    // プレーン EO: smoothed == real。SA の make_snapshot_fast を no_smoothing=true で使う。
    let mut sm = |_p: &Partition, _cuts: &[i32], c: i32, t: usize, f: usize| {
        GraphPartitionProblem::score_from_state(c, t, f)
    };

    let max_iter = cfg.iterations();
    let snap_steps = logarithmic_steps(max_iter);
    let mut snap_iter = snap_steps.iter().copied().peekable();
    let mut records = Vec::with_capacity(snap_steps.len() + 1);

    // 初期スナップショット (step = 0)。ベイスンは SA と同じ単一フリップ山登り。
    let cs0 = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
    records.push(make_snapshot_fast(
        prob, &current, &cuts_at, cur_cut, cur_t, cur_f, cs0, 0, &mut sm, true, &mut tie_rng,
    ));

    if n == 0 {
        return (best, records, None);
    }

    let cdf = build_power_law_cdf(n, tau);

    let mut tracer = trace_spec.map(|ts| FlipTraceCollector::new(ts, n, max_iter));
    let mut lambdas = vec![0.0f64; n];
    let mut order: Vec<usize> = (0..n).collect();

    for it in 1..=max_iter {
        // 各頂点の適応度（O(N)）。
        // PHASE 2: 毎ステップ全ソートは O(N log N)。順序統計木で適応度を保持し、
        // フリップで変化する頂点と隣接頂点だけ局所更新すれば落とせる。
        compute_eo_flip_lambdas(&fitness, &current, &cuts_at, &degrees, cur_t, cur_f, &mut lambdas);

        for (i, o) in order.iter_mut().enumerate() {
            *o = i;
        }
        order.sort_by(|&a, &b| lambdas[a].partial_cmp(&lambdas[b]).unwrap());

        // ランク抽選（乱数消費は 1 draw/step）。k は rank_hist 用の昇順位置。
        let u: f64 = rng.r#gen::<f64>();
        let (idx, k) = select_eo_rank(&lambdas, &order, &cdf, u);

        // 無条件フリップ（受理判定なし）。
        let (nc, nt, nf, _) =
            prob.delta_apply_cached(&current, &cuts_at, idx, cur_cut, cur_t, cur_f);

        // トレース（フリップ適用前の状態で記録。乱数は消費しない）。
        if let Some(tr) = tracer.as_mut() {
            tr.record_step(
                it, k, idx, &lambdas, &current, &cuts_at, &degrees, cur_cut, cur_t, cur_f, nc,
                nt, nf,
            );
            tr.maybe_probe(
                it, idx, &lambdas, &order, &current, &cuts_at, &degrees, cur_t, cur_f, &cdf, u,
            );
        }

        prob.flip_vertex(&mut current, &mut cuts_at, idx);
        cur_cut = nc;
        cur_t = nt;
        cur_f = nf;

        let real_score = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
        if real_score < best_score {
            best_score = real_score;
            best = current.clone();
        }

        #[cfg(debug_assertions)]
        if it % 1000 == 0 {
            debug_assert_eq!(
                cur_cut,
                prob.count_cut_edges(&current),
                "cut drift at it={}",
                it
            );
            let (rt, rf) = get_partition_sizes(&current);
            debug_assert_eq!((cur_t, cur_f), (rt, rf), "size drift at it={}", it);
            debug_assert_eq!(
                cuts_at,
                prob.compute_cuts_at(&current),
                "cuts_at drift at it={}",
                it
            );
        }

        if let Some(&want) = snap_iter.peek() {
            if it == want {
                let cs = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
                records.push(make_snapshot_fast(
                    prob, &current, &cuts_at, cur_cut, cur_t, cur_f, cs, it, &mut sm, true,
                    &mut tie_rng,
                ));
                snap_iter.next();
            }
        }
    }

    let trace = tracer.map(|tr| tr.finish(n, max_iter, fitness.label(), tau));
    (best, records, trace)
}

// ============================================================================
// SA（スワップ近傍・厳密バランス）
// ============================================================================

/// 固定温度 SA をスワップ近傍（厳密バランス）で実行する版。
///
/// スワップ版 EO（[`run_eo`]）と **同一の近傍（v1∈A ↔ v2∈B のスワップ）・厳密バランス
/// `|A|=|B|=N/2`・初期化（`balanced_init`）・ベイスン算出（`hill_climb_swap_fast` による
/// スワップ近傍の局所最適）**を共有し、違いは「1 手の選び方」だけ: ランダムに 1 スワップを
/// 提案し、**メトロポリス基準**で受理する（温度 `θ = RunConfig::theta`）。
/// `θ = None`（T=0）は改善スワップのみ受理する貪欲降下。
///
/// 厳密バランスなのでペナルティ項は一定（偶数 N なら 0）、実スコア = カット数。
/// `final_partition` には最良解 `S_best` を返す。`smoothing` は無視する。
fn run_sa_swap(
    prob: &GraphPartitionProblem,
    cfg: &RunConfig,
    seed: u64,
) -> (Partition, Vec<StepRecord>) {
    let mut rng = Mt19937GenRand64::new(seed);
    let n = prob.neighbour_size();

    let mut current = balanced_init(n, &mut rng);
    let mut cur_cut = prob.count_cut_edges(&current);
    let (mut cur_t, mut cur_f) = get_partition_sizes(&current);
    let mut cuts_at = prob.compute_cuts_at(&current);

    let mut best = current.clone();
    let mut best_score = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);

    // ベイスン算出（スワップ降下）のタイブレーク専用 RNG。本体 rng と独立。
    let mut tie_rng = Mt19937GenRand64::new(seed ^ TIEBREAK_SALT);

    let temperature = cfg.temperature();
    let max_iter = cfg.iterations();
    let snap_steps = logarithmic_steps(max_iter);
    let mut snap_iter = snap_steps.iter().copied().peekable();
    let mut records = Vec::with_capacity(snap_steps.len() + 1);

    records.push(make_swap_snapshot(
        prob, &current, &cuts_at, cur_cut, cur_t, cur_f, 0, &mut tie_rng,
    ));

    if n < 2 {
        return (best, records);
    }

    for it in 1..=max_iter {
        // ランダムスワップ: v1 を一様、v2 を反対集合から一様（バランスなので数回で当たる）。
        let v1 = rng.gen_range(0..n);
        let set1 = current[v1];
        let v2 = loop {
            let cand = rng.gen_range(0..n);
            if current[cand] != set1 {
                break cand;
            }
        };

        let cur_score = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
        // v1 をフリップ → 更新後 cuts_at で v2 のスワップ結果スコアを得る。
        let (c1, t1, f1, _) =
            prob.delta_apply_cached(&current, &cuts_at, v1, cur_cut, cur_t, cur_f);
        prob.flip_vertex(&mut current, &mut cuts_at, v1);
        let (c2, t2, f2, swap_score) =
            prob.delta_apply_cached(&current, &cuts_at, v2, c1, t1, f1);

        let delta = swap_score - cur_score;
        let accept = if delta < 0.0 {
            true
        } else if temperature > 0.0 {
            rng.r#gen::<f64>() < (-delta / temperature).exp()
        } else {
            false
        };

        if accept {
            prob.flip_vertex(&mut current, &mut cuts_at, v2); // スワップ完成
            cur_cut = c2;
            cur_t = t2;
            cur_f = f2;
            if swap_score < best_score {
                best_score = swap_score;
                best = current.clone();
            }
        } else {
            prob.flip_vertex(&mut current, &mut cuts_at, v1); // v1 を戻す（対合）
        }

        #[cfg(debug_assertions)]
        if it % 1000 == 0 {
            debug_assert_eq!(cur_t, n - n / 2, "balance drift at it={}", it);
            debug_assert_eq!(
                cur_cut,
                prob.count_cut_edges(&current),
                "cut drift at it={}",
                it
            );
            debug_assert_eq!(
                cuts_at,
                prob.compute_cuts_at(&current),
                "cuts_at drift at it={}",
                it
            );
        }

        if let Some(&want) = snap_iter.peek() {
            if it == want {
                records.push(make_swap_snapshot(
                    prob, &current, &cuts_at, cur_cut, cur_t, cur_f, it, &mut tie_rng,
                ));
                snap_iter.next();
            }
        }
    }

    (best, records)
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
    execute_traced(spec, cfg, prob, seed, None).0
}

/// [`execute`] のトレース対応版。`trace` を渡し、かつソルバーが EoFlip 系のときだけ
/// [`FlipSelectionTrace`] を追加で返す（それ以外は `None`）。
/// トレースは乱数を消費しないため `RunResult` は [`execute`] とバイト一致する。
pub fn execute_traced(
    spec: GraphSpec,
    cfg: &RunConfig,
    prob: &GraphPartitionProblem,
    seed: u64,
    trace: Option<&FlipTraceSpec>,
) -> (RunResult, Option<FlipSelectionTrace>) {
    let t0 = std::time::Instant::now();
    let sm_seed = seed.wrapping_add(0xDEAD_BEEF);
    let (final_p, records, flip_trace) = match cfg.solver {
        SolverSpec::Eo { tau } => {
            let (p, r) = run_eo(prob, cfg, seed, tau);
            (p, r, None)
        }
        SolverSpec::EoFlip { tau, .. }
        | SolverSpec::EoFlipMulAlpha { tau, .. }
        | SolverSpec::EoFlipAddBeta { tau, .. }
        | SolverSpec::EoFlipMulGamma { tau } => {
            let fitness = EoFlipFitnessSpec::from_solver(&cfg.solver)
                .expect("EoFlip 系 solver は必ず fitness を持つ");
            run_eo_flip(prob, cfg, seed, tau, fitness, trace)
        }
        SolverSpec::SaSwap => {
            let (p, r) = run_sa_swap(prob, cfg, seed);
            (p, r, None)
        }
        SolverSpec::Sa => {
            let (p, r) = match cfg.smoothing {
                SmoothingSpec::None => run_sa_none(prob, cfg, seed),
                SmoothingSpec::KAverage(k) => run_sa_kavg(prob, k, cfg, seed),
                SmoothingSpec::RandomKAverage(k) => run_sa_random_k(prob, k, sm_seed, cfg, seed),
                SmoothingSpec::WeightedAverage(w) => run_sa_weighted(prob, w, cfg, seed),
            };
            (p, r, None)
        }
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let result = RunResult {
        graph_spec: spec,
        config: cfg.clone(),
        seed,
        final_partition: final_p,
        records,
        elapsed_ms,
    };
    let flip_trace = flip_trace.map(|mut tr| {
        tr.graph_id = spec.id();
        tr.config_id = cfg.id();
        tr.seed = seed;
        tr
    });
    (result, flip_trace)
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

    /// フリップ選択トレースのパス（`base/<graph_id>/<config_id>/seed_<seed>_trace.json`）。
    pub fn trace_path_for(&self, spec: &GraphSpec, cfg: &RunConfig, seed: u64) -> PathBuf {
        self.base_dir
            .join(spec.id())
            .join(cfg.id())
            .join(format!("seed_{}_trace.json", seed))
    }

    pub fn trace_exists(&self, spec: &GraphSpec, cfg: &RunConfig, seed: u64) -> bool {
        self.trace_path_for(spec, cfg, seed).exists()
    }

    /// フリップ選択トレースを `seed_<seed>_trace.json` として保存する。
    pub fn save_trace(
        &self,
        spec: &GraphSpec,
        cfg: &RunConfig,
        trace: &FlipSelectionTrace,
    ) -> Result<(), String> {
        let p = self.trace_path_for(spec, cfg, trace.seed);
        if let Some(parent) = p.parent() {
            ensure_dir_exists(parent).map_err(|e| format!("create dir: {}", e))?;
        }
        save_json(trace, &p).map_err(|e| format!("save trace: {}", e))
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
    use crate::run_config::{DEFAULT_EO_FLIP_ALPHA, DEFAULT_EO_FLIP_DIFF_EXP};

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

    // ------------------------------------------------------------------
    // τ-EO テスト
    // ------------------------------------------------------------------

    fn eo_cfg(log10_iterations: u32, tau: f64) -> RunConfig {
        let mut cfg = RunConfig::new("eo");
        cfg.theta = None;
        cfg.smoothing = SmoothingSpec::None;
        cfg.log10_iterations = log10_iterations;
        cfg.solver = SolverSpec::Eo { tau };
        cfg
    }

    /// 偶数 N: 全ステップでバランス維持 ＋ 決定論 ＋ best ≤ 初期。
    /// log10_iter=4（10000 step）で debug ビルドの drift アサート（balance / cut / cuts_at）も通る。
    #[test]
    fn test_eo_balanced_and_deterministic() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec {
            kind: GraphKind::Random,
            n: 30,
            d: 4.0,
            seed: 0,
        };
        let prob = StoredGraph::generate(spec).problem();
        let cfg = eo_cfg(4, 1.4);

        let (p1, r1) = run_eo(&prob, &cfg, 42, 1.4);
        let (p2, r2) = run_eo(&prob, &cfg, 42, 1.4);
        assert_eq!(p1, p2, "EO は同一シードで決定論的であるべき");
        assert_eq!(r1.len(), r2.len());

        // 最終解は厳密にバランスしている。
        let (t, f) = get_partition_sizes(&p1);
        assert_eq!((t, f), (15, 15));

        // 各スナップショットで basin（スワップ近傍の局所最適）は current 以下、smoothed==real。
        for rec in &r1 {
            assert!(
                rec.basin_real_from_real <= rec.current_real + 1e-9,
                "basin は current 以下のはず: step={}, basin={}, current={}",
                rec.step, rec.basin_real_from_real, rec.current_real
            );
            assert!((rec.current_smoothed - rec.current_real).abs() < 1e-12);
        }

        assert_eq!(r1[0].step, 0);
    }

    /// 既知最適: パス P4 (0-1-2-3) のバランス分割の最小カットは 1（{0,1}|{2,3}）。
    #[test]
    fn test_eo_reaches_known_optimum_path4() {
        let mut graph = crate::graph_partition::Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        let prob = GraphPartitionProblem::new(graph);
        let cfg = eo_cfg(3, 1.4);

        let (best, records) = run_eo(&prob, &cfg, 7, 1.4);
        let best_cut = prob.count_cut_edges(&best);
        assert_eq!(best_cut, 1, "P4 のバランス最小カットは 1");
        // 記録された best-so-far も 1.0 に到達している。
        assert!((records.last().unwrap().basin_real_from_real - 1.0).abs() < 1e-9);
        // バランス維持。
        assert_eq!(get_partition_sizes(&best), (2, 2));
    }

    /// τ 依存性: 大きく異なる τ では探索軌跡（records）が一致しない。
    #[test]
    fn test_eo_tau_effect() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec {
            kind: GraphKind::Random,
            n: 60,
            d: 4.0,
            seed: 1,
        };
        let prob = StoredGraph::generate(spec).problem();

        let (_p_lo, r_lo) = run_eo(&prob, &eo_cfg(3, 1.05), 99, 1.05);
        let (_p_hi, r_hi) = run_eo(&prob, &eo_cfg(3, 1.9), 99, 1.9);

        let traj_lo: Vec<f64> = r_lo.iter().map(|r| r.current_real).collect();
        let traj_hi: Vec<f64> = r_hi.iter().map(|r| r.current_real).collect();
        assert_ne!(traj_lo, traj_hi, "τ を大きく変えれば軌跡は変わるはず");
    }

    /// 奇数 N で panic せず、サイズが ⌈N/2⌉ / ⌊N/2⌋ に固定される。
    #[test]
    fn test_eo_odd_n_no_panic() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec {
            kind: GraphKind::Random,
            n: 31,
            d: 4.0,
            seed: 2,
        };
        let prob = StoredGraph::generate(spec).problem();
        let cfg = eo_cfg(3, 1.4);
        let (best, _records) = run_eo(&prob, &cfg, 5, 1.4);
        let (t, f) = get_partition_sizes(&best);
        assert_eq!((t, f), (16, 15));
    }

    /// 孤立頂点（deg=0）があっても 0 除算せず動く（λ=1.0 として扱う）。
    #[test]
    fn test_eo_isolated_vertex() {
        // 6 頂点。頂点 5 は孤立。
        let mut graph = crate::graph_partition::Graph::new(6);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        let prob = GraphPartitionProblem::new(graph);
        let cfg = eo_cfg(3, 1.4);
        let (best, _records) = run_eo(&prob, &cfg, 3, 1.4);
        assert_eq!(get_partition_sizes(&best), (3, 3));
        // best のカットは妥当な範囲（≤ 初期）。
        assert!(prob.count_cut_edges(&best) <= 4);
    }

    // ------------------------------------------------------------------
    // τ-EO（flip 近傍版）テスト
    // ------------------------------------------------------------------

    fn eoflip_cfg(log10_iterations: u32, tau: f64) -> RunConfig {
        let mut cfg = RunConfig::new("eoflip");
        cfg.theta = None;
        cfg.smoothing = SmoothingSpec::None;
        cfg.log10_iterations = log10_iterations;
        cfg.solver = SolverSpec::EoFlip {
            tau,
            alpha_eo: DEFAULT_EO_FLIP_ALPHA,
            diff_exp: DEFAULT_EO_FLIP_DIFF_EXP,
        };
        cfg
    }

    /// eo_flip_lambda の方向性（多数派は λ_eff 低下、少数派は上昇）と balance時の挙動（既定 α/p）。
    #[test]
    fn test_eo_flip_lambda_direction() {
        let (a0, p0) = (DEFAULT_EO_FLIP_ALPHA, DEFAULT_EO_FLIP_DIFF_EXP);
        // diff=10（true集合が多数派）, deg=4, cuts=1 (g=3)
        // 多数派(true): diff_after=8 → improvement>0 → g/(deg+q)
        let maj = eo_flip_lambda(4, 1, 8, 10, a0, p0);
        // 少数派(false): diff_after=12 → improvement<0 → (g+q)/(deg+q)
        let min = eo_flip_lambda(4, 1, 12, 10, a0, p0);
        assert!(maj < min, "多数派の λ_eff は少数派より小さい（選ばれやすい）: maj={maj}, min={min}");

        // balance時 (diff=0): どちらも diff_after=±2 → improvement<0、同じ q=4α
        let a = eo_flip_lambda(4, 1, 2, 0, a0, p0);
        let b = eo_flip_lambda(4, 1, -2, 0, a0, p0);
        assert!((a - b).abs() < 1e-12, "balance時は左右対称");
    }

    /// 分数指数 p=0.5・カスタム α_eo の経路が有限で、多数派<少数派の向きと負 diff の abs 対称性を保つ。
    #[test]
    fn test_eo_flip_lambda_fractional_exp() {
        let (a, p) = (0.1_f64, 0.5_f64);
        // diff=10, deg=4, cuts=1 (g=3)
        let maj = eo_flip_lambda(4, 1, 8, 10, a, p); // 多数派: improvement>0
        let min = eo_flip_lambda(4, 1, 12, 10, a, p); // 少数派: improvement<0
        assert!(maj.is_finite() && min.is_finite(), "分数指数でも有限");
        assert!(maj < min, "分数指数でも多数派 λ_eff < 少数派: maj={maj}, min={min}");

        // balance時 (diff=0) の左右対称（|±2|^p が等しい）。
        let l = eo_flip_lambda(4, 1, 2, 0, a, p);
        let r = eo_flip_lambda(4, 1, -2, 0, a, p);
        assert!(l.is_finite() && (l - r).abs() < 1e-12, "負 diff も abs で対称: l={l}, r={r}");
    }

    #[test]
    fn test_is_majority_side() {
        // t > f: true 集合が多数派。
        assert!(is_majority_side(true, 5, 3));
        assert!(!is_majority_side(false, 5, 3));
        // f > t: false 集合が多数派。
        assert!(!is_majority_side(true, 3, 5));
        assert!(is_majority_side(false, 3, 5));
        // 均衡時: どちらも多数派ではない。
        assert!(!is_majority_side(true, 4, 4));
        assert!(!is_majority_side(false, 4, 4));
    }

    /// eo_flip_lambda_mul_alpha: 多数派は λ0*alpha、少数派は λ0*1、alpha<1 なら多数派が
    /// 選ばれやすくなる（λ が小さいほど選ばれやすい）方向を確認する。
    #[test]
    fn test_eo_flip_lambda_mul_alpha_direction() {
        // deg=4, cuts=1 → λ0 = 3/4 = 0.75
        let maj = eo_flip_lambda_mul_alpha(4, 1, true, 0.5);
        let min = eo_flip_lambda_mul_alpha(4, 1, false, 0.5);
        assert!((maj - 0.375).abs() < 1e-12);
        assert!((min - 0.75).abs() < 1e-12);
        assert!(maj < min, "alpha<1 では多数派の λ が少数派より小さい: maj={maj}, min={min}");

        // alpha=1（既定）なら多数派/少数派とも λ0 と一致（バイアスなし）。
        let neutral_maj = eo_flip_lambda_mul_alpha(4, 1, true, 1.0);
        let neutral_min = eo_flip_lambda_mul_alpha(4, 1, false, 1.0);
        assert!((neutral_maj - neutral_min).abs() < 1e-12);
        assert!((neutral_maj - 0.75).abs() < 1e-12);
    }

    /// eo_flip_lambda_add_beta: 多数派は β*λ0 + 0、少数派は β*λ0 + 1 で、
    /// 少数派の適応度が常に多数派以上になる（保護される）方向を確認する。
    #[test]
    fn test_eo_flip_lambda_add_beta_direction() {
        // deg=4, cuts=1 → λ0 = 0.75
        let maj = eo_flip_lambda_add_beta(4, 1, true, 1.0);
        let min = eo_flip_lambda_add_beta(4, 1, false, 1.0);
        assert!((maj - 0.75).abs() < 1e-12);
        assert!((min - 1.75).abs() < 1e-12);
        assert!(maj < min, "多数派の λ は少数派より小さい（選ばれやすい）: maj={maj}, min={min}");
    }

    /// eo_flip_lambda_mul_gamma: 均衡時は γ=1 となり多数派/少数派の λ1 が一致する
    /// （多数派側バイアスなしに退化）。不均衡時は γ<1 で多数派が選ばれやすくなる。
    #[test]
    fn test_eo_flip_lambda_mul_gamma_direction() {
        // deg=4, cuts=1 → λ0 = 0.75。不均衡（gamma=0.5）では多数派 < 少数派。
        let maj = eo_flip_lambda_mul_gamma(4, 1, true, 0.5);
        let min = eo_flip_lambda_mul_gamma(4, 1, false, 0.5);
        assert!((maj - 0.375).abs() < 1e-12);
        assert!((min - 0.75).abs() < 1e-12);
        assert!(maj < min, "gamma<1 では多数派の λ が少数派より小さい: maj={maj}, min={min}");

        // 均衡（gamma=1）では差がない。
        let bal_maj = eo_flip_lambda_mul_gamma(4, 1, true, 1.0);
        let bal_min = eo_flip_lambda_mul_gamma(4, 1, false, 1.0);
        assert!((bal_maj - bal_min).abs() < 1e-12);
    }

    /// flip版: 決定論 ＋ スナップショット ＋ ベイスンが現在解以下（局所最適）。
    #[test]
    fn test_eo_flip_runs_and_basin_le_current() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec { kind: GraphKind::Random, n: 40, d: 4.0, seed: 0 };
        let prob = StoredGraph::generate(spec).problem();
        let cfg = eoflip_cfg(4, 1.4);

        let legacy_fitness = || EoFlipFitnessSpec::Legacy {
            alpha_eo: DEFAULT_EO_FLIP_ALPHA,
            diff_exp: DEFAULT_EO_FLIP_DIFF_EXP,
        };
        let (p1, r1, _) = run_eo_flip(&prob, &cfg, 42, 1.4, legacy_fitness(), None);
        let (p2, _r2, _) = run_eo_flip(&prob, &cfg, 42, 1.4, legacy_fitness(), None);
        assert_eq!(p1, p2, "flip版も決定論的");
        assert_eq!(r1[0].step, 0);
        assert!(!r1.is_empty());

        // SA と同じベイスン算出: 単一フリップ山登りの局所最適は現在解以下。
        for rec in &r1 {
            assert!(
                rec.basin_real_from_real <= rec.current_real + 1e-9,
                "basin は current 以下のはず: step={}, basin={}, current={}",
                rec.step, rec.basin_real_from_real, rec.current_real
            );
            // プレーン EO は smoothed == real。
            assert!((rec.current_smoothed - rec.current_real).abs() < 1e-12);
        }
    }

    /// flip版: 退化（全頂点が片側へ collapse）しないこと。ペナルティによる復元力で
    /// 最良解はある程度バランスしている。
    #[test]
    fn test_eo_flip_no_degenerate_collapse() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec { kind: GraphKind::Random, n: 60, d: 4.0, seed: 3 };
        let prob = StoredGraph::generate(spec).problem();
        let cfg = eoflip_cfg(4, 1.4);
        let (best, _r, _) = run_eo_flip(
            &prob,
            &cfg,
            7,
            1.4,
            EoFlipFitnessSpec::Legacy {
                alpha_eo: DEFAULT_EO_FLIP_ALPHA,
                diff_exp: DEFAULT_EO_FLIP_DIFF_EXP,
            },
            None,
        );
        let (t, f) = get_partition_sizes(&best);
        // 完全片側 (0/60) には陥らない。ペナルティ α=0.05 下では概ね均衡近傍。
        let imbalance = (t as i64 - f as i64).abs();
        assert!(imbalance <= 20, "退化的に偏りすぎ: t={t}, f={f}");
        assert!(t > 0 && f > 0, "片側集合が空になってはいけない");
    }

    /// τ 依存性: 大きく異なる τ では軌跡が一致しない。
    #[test]
    fn test_eo_flip_tau_effect() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec { kind: GraphKind::Random, n: 60, d: 4.0, seed: 1 };
        let prob = StoredGraph::generate(spec).problem();
        let (_p1, r_lo, _) = run_eo_flip(
            &prob,
            &eoflip_cfg(3, 1.05),
            99,
            1.05,
            EoFlipFitnessSpec::Legacy {
                alpha_eo: DEFAULT_EO_FLIP_ALPHA,
                diff_exp: DEFAULT_EO_FLIP_DIFF_EXP,
            },
            None,
        );
        let (_p2, r_hi, _) = run_eo_flip(
            &prob,
            &eoflip_cfg(3, 1.9),
            99,
            1.9,
            EoFlipFitnessSpec::Legacy {
                alpha_eo: DEFAULT_EO_FLIP_ALPHA,
                diff_exp: DEFAULT_EO_FLIP_DIFF_EXP,
            },
            None,
        );
        let lo: Vec<f64> = r_lo.iter().map(|r| r.current_real).collect();
        let hi: Vec<f64> = r_hi.iter().map(|r| r.current_real).collect();
        assert_ne!(lo, hi);
    }

    /// execute() 経由でも EoFlip がディスパッチされる。
    #[test]
    fn test_execute_dispatches_eo_flip() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec { kind: GraphKind::Random, n: 20, d: 3.0, seed: 0 };
        let prob = StoredGraph::generate(spec).problem();
        let cfg = eoflip_cfg(2, 1.4);
        let r = execute(spec, &cfg, &prob, 42);
        assert!(!r.records.is_empty());
        assert_eq!(r.config.id(), "eoflip_iter2_tau1p4");
    }

    /// execute() 経由で新 3 適応度（MulAlpha/AddBeta/MulGamma）も動作し、決定論的で
    /// バランスが完全崩壊しないこと。
    #[test]
    fn test_execute_dispatches_eo_flip_new_variants() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec { kind: GraphKind::Random, n: 40, d: 4.0, seed: 0 };
        let prob = StoredGraph::generate(spec).problem();

        let mut cfg_mul_alpha = RunConfig::new("eoflip-mul-alpha");
        cfg_mul_alpha.theta = None;
        cfg_mul_alpha.smoothing = SmoothingSpec::None;
        cfg_mul_alpha.log10_iterations = 3;
        cfg_mul_alpha.solver = SolverSpec::EoFlipMulAlpha { tau: 1.4, alpha: 0.3 };

        let mut cfg_add_beta = RunConfig::new("eoflip-add-beta");
        cfg_add_beta.theta = None;
        cfg_add_beta.smoothing = SmoothingSpec::None;
        cfg_add_beta.log10_iterations = 3;
        cfg_add_beta.solver = SolverSpec::EoFlipAddBeta { tau: 1.4, beta: 1.0 };

        let mut cfg_mul_gamma = RunConfig::new("eoflip-mul-gamma");
        cfg_mul_gamma.theta = None;
        cfg_mul_gamma.smoothing = SmoothingSpec::None;
        cfg_mul_gamma.log10_iterations = 3;
        cfg_mul_gamma.solver = SolverSpec::EoFlipMulGamma { tau: 1.4 };

        for cfg in [&cfg_mul_alpha, &cfg_add_beta, &cfg_mul_gamma] {
            let r1 = execute(spec, cfg, &prob, 42);
            let r2 = execute(spec, cfg, &prob, 42);
            assert!(!r1.records.is_empty(), "records が空: id={}", cfg.id());
            assert_eq!(
                r1.final_partition, r2.final_partition,
                "同一シードで決定論的: id={}",
                cfg.id()
            );
            let (t, f) = get_partition_sizes(&r1.final_partition);
            assert!(t > 0 && f > 0, "片側集合が空になってはいけない: id={}", cfg.id());
        }

        assert_eq!(cfg_mul_alpha.id(), "eoflipmulalpha_iter3_tau1p4_a0p3");
        assert_eq!(cfg_add_beta.id(), "eoflipaddbeta_iter3_tau1p4_b1");
        assert_eq!(cfg_mul_gamma.id(), "eoflipmulgamma_iter3_tau1p4");
    }

    // ------------------------------------------------------------------
    // SA（スワップ近傍・厳密バランス）テスト
    // ------------------------------------------------------------------

    fn saswap_cfg(log10_iterations: u32, theta: Option<f64>) -> RunConfig {
        let mut cfg = RunConfig::new("saswap");
        cfg.theta = theta;
        cfg.smoothing = SmoothingSpec::None;
        cfg.log10_iterations = log10_iterations;
        cfg.solver = SolverSpec::SaSwap;
        cfg
    }

    /// 厳密バランス維持 ＋ 決定論 ＋ best ≤ 初期（偶数 N）。
    /// log10_iter=4 で debug ビルドの drift アサート（balance/cut/cuts_at）も通る。
    #[test]
    fn test_sa_swap_balanced_and_deterministic() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec { kind: GraphKind::Random, n: 30, d: 4.0, seed: 0 };
        let prob = StoredGraph::generate(spec).problem();
        let cfg = saswap_cfg(4, Some(0.0));

        let (p1, r1) = run_sa_swap(&prob, &cfg, 42);
        let (p2, _r2) = run_sa_swap(&prob, &cfg, 42);
        assert_eq!(p1, p2, "SaSwap は同一シードで決定論的");
        assert_eq!(get_partition_sizes(&p1), (15, 15), "厳密バランス維持");
        // 各スナップショットで basin（スワップ近傍の局所最適）は current 以下。
        for rec in &r1 {
            assert!(
                rec.basin_real_from_real <= rec.current_real + 1e-9,
                "basin は current 以下: step={}, basin={}, current={}",
                rec.step, rec.basin_real_from_real, rec.current_real
            );
        }
    }

    /// 貪欲（T=0）は現在解が単調非増加（reject は据え置き、accept は改善のみ）。
    #[test]
    fn test_sa_swap_greedy_monotone() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec { kind: GraphKind::Random, n: 40, d: 4.0, seed: 1 };
        let prob = StoredGraph::generate(spec).problem();
        let cfg = saswap_cfg(4, None); // T = 0 → 貪欲スワップ降下
        let (_best, records) = run_sa_swap(&prob, &cfg, 9);
        let cur: Vec<f64> = records.iter().map(|r| r.current_real).collect();
        for w in cur.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "T=0 では current は単調非増加: {:?}", w);
        }
    }

    /// execute() 経由でも SaSwap がディスパッチされ、厳密バランスが保たれる。
    #[test]
    fn test_execute_dispatches_sa_swap() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec { kind: GraphKind::Random, n: 20, d: 3.0, seed: 0 };
        let prob = StoredGraph::generate(spec).problem();
        let cfg = saswap_cfg(2, Some(0.0));
        let r = execute(spec, &cfg, &prob, 42);
        assert!(!r.records.is_empty());
        assert_eq!(get_partition_sizes(&r.final_partition), (10, 10));
        assert_eq!(r.config.id(), "saswap_th+0_iter2");
    }

    /// hill_climb_swap_fast: バランス保存 ＋ basin ≤ start ＋ 到達点が本当にスワップ局所最適
    /// （改善スワップが存在しない）であることを検証する。
    #[test]
    fn test_hill_climb_swap_fast_is_local_opt() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec { kind: GraphKind::Random, n: 30, d: 4.0, seed: 0 };
        let prob = StoredGraph::generate(spec).problem();

        for seed in 0..5u64 {
            let mut rng = Mt19937GenRand64::new(seed);
            let start = balanced_init(30, &mut rng);
            let start_cut = prob.count_cut_edges(&start);
            let (st, sf) = get_partition_sizes(&start);
            let cuts_at = prob.compute_cuts_at(&start);
            let mut tie = Mt19937GenRand64::new(seed ^ 0xABCD);

            let (basin, b_cuts, bc, bt, bf) =
                hill_climb_swap_fast(&prob, &start, &cuts_at, start_cut, st, sf, &mut tie);

            // バランス保存。
            assert_eq!(get_partition_sizes(&basin), (st, sf));
            // 整数状態が再計算と一致。
            assert_eq!(bc, prob.count_cut_edges(&basin));
            assert_eq!(b_cuts, prob.compute_cuts_at(&basin));
            // basin ≤ start。
            let start_score = GraphPartitionProblem::score_from_state(start_cut, st, sf);
            let basin_score = GraphPartitionProblem::score_from_state(bc, bt, bf);
            assert!(basin_score <= start_score + 1e-9);

            // 到達点に改善スワップが存在しないことを確認（全 A×B ペアを走査）。
            let n = basin.len();
            for v1 in 0..n {
                if !basin[v1] {
                    continue;
                }
                let mut p = basin.clone();
                let mut c = b_cuts.clone();
                let (c1, t1, f1, _) = prob.delta_apply_cached(&p, &c, v1, bc, bt, bf);
                prob.flip_vertex(&mut p, &mut c, v1);
                for v2 in 0..n {
                    if p[v2] || v2 == v1 {
                        continue;
                    }
                    let s = prob.delta_apply_cached(&p, &c, v2, c1, t1, f1).3;
                    assert!(
                        s >= basin_score - 1e-9,
                        "局所最適のはずが改善スワップが存在: seed={seed}, v1={v1}, v2={v2}, s={s}, basin={basin_score}"
                    );
                }
            }
        }
    }

    /// execute() 経由でも EO がディスパッチされ、結果が得られる。
    #[test]
    fn test_execute_dispatches_eo() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let spec = GraphSpec {
            kind: GraphKind::Random,
            n: 20,
            d: 3.0,
            seed: 0,
        };
        let prob = StoredGraph::generate(spec).problem();
        let cfg = eo_cfg(2, 1.4);
        let r = execute(spec, &cfg, &prob, 42);
        assert!(!r.records.is_empty());
        assert_eq!(r.records[0].step, 0);
        assert_eq!(get_partition_sizes(&r.final_partition), (10, 10));
        assert_eq!(r.config.id(), "eo_iter2_tau1p4");
    }

    // ------------------------------------------------------------------
    // フリップ選択トレース
    // ------------------------------------------------------------------

    /// kendall_tau_b: 完全一致 → 1、完全逆順 → -1、tie 込みの手計算例と一致。
    #[test]
    fn test_kendall_tau_b_basics() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y_same = [10.0, 20.0, 30.0, 40.0];
        let y_rev = [4.0, 3.0, 2.0, 1.0];
        assert!((kendall_tau_b(&x, &y_same) - 1.0).abs() < 1e-12);
        assert!((kendall_tau_b(&x, &y_rev) + 1.0).abs() < 1e-12);

        // tie 例: x = [1,2,2,3], y = [1,3,2,4]。
        // 全 6 ペア中 x-tie は (2,3) の 1 ペア（y は 3≠2 なので tie_x）。
        // 残り 5 ペアはすべて concordant → τ_b = 5 / sqrt(6*5) ≈ 0.9129。
        let xt = [1.0, 2.0, 2.0, 3.0];
        let yt = [1.0, 3.0, 2.0, 4.0];
        let expect = 5.0 / (6.0f64 * 5.0).sqrt();
        assert!((kendall_tau_b(&xt, &yt) - expect).abs() < 1e-12);

        // 片方が全 tie → 分母 0 → 0.0。
        let flat = [7.0, 7.0, 7.0, 7.0];
        assert_eq!(kendall_tau_b(&x, &flat), 0.0);
    }

    /// bottom_m_jaccard: 同一 order → 1、素集合 → 0。
    #[test]
    fn test_bottom_m_jaccard() {
        let a = [0usize, 1, 2, 3, 4, 5, 6, 7];
        let b = [4usize, 5, 6, 7, 0, 1, 2, 3];
        assert!((bottom_m_jaccard(&a, &a, 4) - 1.0).abs() < 1e-12);
        assert_eq!(bottom_m_jaccard(&a, &b, 4), 0.0);
        // m=8 は全体 → 集合として同一 → 1。
        assert!((bottom_m_jaccard(&a, &b, 8) - 1.0).abs() < 1e-12);
    }

    /// EoFlipFitnessSpec::label と from_solver の対応。
    #[test]
    fn test_eo_flip_fitness_spec_label() {
        assert_eq!(
            EoFlipFitnessSpec::Legacy { alpha_eo: 0.064, diff_exp: 2.0 }.label(),
            "legacy_a0p064_p2"
        );
        assert_eq!(EoFlipFitnessSpec::MulAlpha { alpha: 0.1 }.label(), "mulalpha_a0p1");
        assert_eq!(EoFlipFitnessSpec::AddBeta { beta: 1.0 }.label(), "addbeta_b1");
        assert_eq!(EoFlipFitnessSpec::MulGamma.label(), "mulgamma");
        assert_eq!(
            EoFlipFitnessSpec::from_solver(&SolverSpec::EoFlipMulGamma { tau: 1.2 }),
            Some(EoFlipFitnessSpec::MulGamma)
        );
        assert_eq!(EoFlipFitnessSpec::from_solver(&SolverSpec::Sa), None);
    }

    /// トレース: 軌道不変（トレース有無で records / best が一致）＋ カウンタ保存則
    /// ＋ 自己プローブは完全一致（agree 全 true, jaccard=1, τ_b=1）。
    #[test]
    fn test_eo_flip_trace_invariants() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let gspec = GraphSpec { kind: GraphKind::Random, n: 40, d: 4.0, seed: 2 };
        let prob = StoredGraph::generate(gspec).problem();
        let cfg = eoflip_cfg(3, 1.4);
        let fitness = EoFlipFitnessSpec::Legacy {
            alpha_eo: DEFAULT_EO_FLIP_ALPHA,
            diff_exp: DEFAULT_EO_FLIP_DIFF_EXP,
        };
        let trace_spec = FlipTraceSpec {
            probes: vec![
                fitness, // 自己プローブ
                EoFlipFitnessSpec::MulAlpha { alpha: 0.1 },
                EoFlipFitnessSpec::AddBeta { beta: 1.0 },
                EoFlipFitnessSpec::MulGamma,
            ],
        };

        let (p_plain, r_plain, t_plain) = run_eo_flip(&prob, &cfg, 42, 1.4, fitness, None);
        let (p_tr, r_tr, t_tr) =
            run_eo_flip(&prob, &cfg, 42, 1.4, fitness, Some(&trace_spec));
        assert!(t_plain.is_none());
        let trace = t_tr.expect("トレース指定ありなら Some");

        // 軌道不変。
        assert_eq!(p_plain, p_tr, "トレース有無で best が変わってはいけない");
        assert_eq!(r_plain.len(), r_tr.len());
        for (a, b) in r_plain.iter().zip(&r_tr) {
            assert_eq!(a.step, b.step);
            assert_eq!(a.current_real.to_bits(), b.current_real.to_bits());
            assert_eq!(a.basin_real_from_real.to_bits(), b.basin_real_from_real.to_bits());
        }

        // カウンタ保存則: 各分類の合計 = 総ステップ数。
        let max_iter = cfg.iterations() as u64;
        let steps: u64 = trace.decades.iter().map(|d| d.steps).sum();
        assert_eq!(steps, max_iter);
        let majminbal: u64 = trace.decades.iter().map(|d| d.maj + d.min + d.bal).sum();
        assert_eq!(majminbal, max_iter);
        let dcut: u64 =
            trace.decades.iter().map(|d| d.dcut_neg + d.dcut_zero + d.dcut_pos).sum();
        assert_eq!(dcut, max_iter);
        let ddiff: u64 =
            trace.decades.iter().map(|d| d.ddiff_dec + d.ddiff_eq + d.ddiff_inc).sum();
        assert_eq!(ddiff, max_iter);
        let ties: u64 = trace.decades.iter().map(|d| d.tie_hist.iter().sum::<u64>()).sum();
        assert_eq!(ties, max_iter);
        assert_eq!(trace.rank_hist.iter().sum::<u64>(), max_iter);
        assert_eq!(trace.flip_counts.iter().sum::<u64>(), max_iter);
        // ディケード数 = log10(1000)+1 = 4、最後のディケードは step=1000 の 1 個だけ。
        assert_eq!(trace.decades.len(), 4);
        assert_eq!(trace.decades[3].steps, 1);

        // フリップは毎ステップ diff を ±2 変えるので ddiff_eq は |±1|→|∓1| 型のみ。
        // n=40（偶数）では diff は常に偶数 → ddiff_eq = 0。
        let ddiff_eq: u64 = trace.decades.iter().map(|d| d.ddiff_eq).sum();
        assert_eq!(ddiff_eq, 0, "偶数 n では |diff| 不変はあり得ない");

        // 自己プローブ（probes[0]）は完全一致。
        let self_probe = &trace.probes[0];
        assert_eq!(self_probe.label, fitness.label());
        assert!(!self_probe.events.is_empty());
        for ev in &self_probe.events {
            assert!(ev.agree, "自己プローブは同じ頂点を選ぶはず (step={})", ev.step);
            assert!((ev.jaccard8 - 1.0).abs() < 1e-9);
            assert!((ev.jaccard32 - 1.0).abs() < 1e-9);
            assert!((ev.kendall_b - 1.0).abs() < 1e-9, "τ_b=1 のはず: {}", ev.kendall_b);
        }
        // プローブイベント数 = logarithmic_steps の個数。
        assert_eq!(self_probe.events.len(), logarithmic_steps(cfg.iterations()).len());

        // 異なる適応度のプローブは（一般に）順位が変わる → どれかで agree=false がある。
        let any_disagree = trace.probes[1..]
            .iter()
            .any(|p| p.events.iter().any(|e| !e.agree));
        assert!(any_disagree, "異なる適応度で選択が一度も変わらないのは不自然");
    }

    /// execute_traced: EoFlip 系のみトレースを返し、graph/config/seed が埋まる。
    /// 非 EoFlip 系（Eo）はトレース指定があっても None。
    #[test]
    fn test_execute_traced_dispatch() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let gspec = GraphSpec { kind: GraphKind::Random, n: 20, d: 3.0, seed: 0 };
        let prob = StoredGraph::generate(gspec).problem();
        let ts = FlipTraceSpec { probes: vec![EoFlipFitnessSpec::MulGamma] };

        let cfg = eoflip_cfg(2, 1.4);
        let (r, tr) = execute_traced(gspec, &cfg, &prob, 42, Some(&ts));
        let tr = tr.expect("EoFlip はトレースを返す");
        assert_eq!(tr.graph_id, gspec.id());
        assert_eq!(tr.config_id, cfg.id());
        assert_eq!(tr.seed, 42);
        assert_eq!(tr.n, 20);
        assert_eq!(tr.max_iter, 100);
        assert_eq!(tr.tau, 1.4);
        // execute() と結果一致（トレースは軌道に影響しない）。
        let r_plain = execute(gspec, &cfg, &prob, 42);
        assert_eq!(r.final_partition, r_plain.final_partition);

        let cfg_eo = eo_cfg(2, 1.4);
        let (_r2, tr2) = execute_traced(gspec, &cfg_eo, &prob, 42, Some(&ts));
        assert!(tr2.is_none(), "スワップ版 Eo はトレース対象外");
    }

    /// 同率が無いときは素の `P(k) ∝ k^{-τ}` 抽選（CDF 二分探索）と厳密一致する。
    #[test]
    fn test_select_eo_rank_no_ties_matches_plain_power_law() {
        let tau = 1.4;
        let n = 12;
        let cdf = build_power_law_cdf(n, tau);
        // 全て相異なる λ。order は昇順（ここでは頂点番号と一致させる）。
        let lambdas: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let order: Vec<usize> = (0..n).collect();

        let m = 20_000;
        for i in 0..m {
            let u = (i as f64 + 0.5) / m as f64;
            let (v, pos) = select_eo_rank(&lambdas, &order, &cdf, u);
            let want = match cdf.binary_search_by(|p| p.partial_cmp(&u).unwrap()) {
                Ok(k) => k,
                Err(k) => k.min(n - 1),
            };
            assert_eq!(pos, want, "u={u} で位置が旧 draw_rank と不一致");
            assert_eq!(v, order[pos]);
        }
    }

    /// 最終規則の核心（ユーザー指定の 1-2-2-2-5 例）:
    /// 同率群の合計頻度は「同率でなかった場合の合計」に等しく、群内は等頻度。
    #[test]
    fn test_select_eo_rank_tie_group_averaging() {
        let tau = 1.4;
        let n = 5;
        let cdf = build_power_law_cdf(n, tau);
        // 頂点 1,2,3 が同率 → 昇順位置 1,2,3（= 順位 2,3,4）を占める。
        let lambdas = [0.1, 0.5, 0.5, 0.5, 0.9];
        let order = vec![0usize, 1, 2, 3, 4];

        let w = |k: usize| (k as f64).powf(-tau);
        let z = (1..=n).map(w).sum::<f64>();
        let group = (w(2) + w(3) + w(4)) / z;
        let expect = [w(1) / z, group / 3.0, group / 3.0, group / 3.0, w(5) / z];

        let m = 200_000;
        let mut counts = [0u64; 5];
        for i in 0..m {
            let u = (i as f64 + 0.5) / m as f64;
            let (v, _) = select_eo_rank(&lambdas, &order, &cdf, u);
            counts[v] += 1;
        }
        for v in 0..n {
            let got = counts[v] as f64 / m as f64;
            assert!(
                (got - expect[v]).abs() < 1e-4,
                "P(v={v}) = {got} が解析値 {} と不一致",
                expect[v]
            );
        }
        // 群の合計は「同率でなかった場合の 2〜4 位ぶんの合計」に一致。
        let group_got = (counts[1] + counts[2] + counts[3]) as f64 / m as f64;
        assert!((group_got - group).abs() < 1e-4, "群合計 {group_got} ≠ {group}");
        // 群内は等分（格子 u では ±1 カウント以内）。
        assert!(counts[1].abs_diff(counts[2]) <= 1, "群内は等頻度のはず: {counts:?}");
        assert!(counts[2].abs_diff(counts[3]) <= 1, "群内は等頻度のはず: {counts:?}");
    }

    /// 任意の同率構成で「全体の合計 = 1」かつ「各群の合計 = 素の CDF 差分」。
    #[test]
    fn test_select_eo_rank_group_totals_preserved() {
        let tau = 1.1;
        // 群サイズ [2, 1, 3, 1] → n = 7、群境界の位置は 0,2,3,6。
        let sizes = [2usize, 1, 3, 1];
        let n: usize = sizes.iter().sum();
        let cdf = build_power_law_cdf(n, tau);
        let mut lambdas = Vec::with_capacity(n);
        for (gi, &sz) in sizes.iter().enumerate() {
            for _ in 0..sz {
                lambdas.push(gi as f64);
            }
        }
        let order: Vec<usize> = (0..n).collect();

        let m = 200_000;
        let mut counts = vec![0u64; n];
        for i in 0..m {
            let u = (i as f64 + 0.5) / m as f64;
            let (v, pos) = select_eo_rank(&lambdas, &order, &cdf, u);
            assert_eq!(v, order[pos]);
            counts[v] += 1;
        }
        assert_eq!(counts.iter().sum::<u64>(), m as u64, "全体で 1 になるはず");

        let mut s = 0usize;
        for &sz in &sizes {
            let e = s + sz;
            let lo = if s == 0 { 0.0 } else { cdf[s - 1] };
            let want = cdf[e - 1] - lo;
            let got = counts[s..e].iter().sum::<u64>() as f64 / m as f64;
            assert!(
                (got - want).abs() < 1e-4,
                "群 [{s},{e}) の合計 {got} が同率なし時の {want} と不一致"
            );
            // 群内は等分。
            for v in s..e {
                assert!(counts[v].abs_diff(counts[s]) <= 1, "群内は等頻度: {counts:?}");
            }
            s = e;
        }
    }

    /// tie が大量に出る設定（AddBeta β=0）でも決定論的で、トレース保存則・
    /// 自己プローブ一致が維持される。
    #[test]
    fn test_eo_flip_tie_heavy_runs() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let gspec = GraphSpec { kind: GraphKind::Random, n: 40, d: 4.0, seed: 2 };
        let prob = StoredGraph::generate(gspec).problem();
        let mut cfg = eoflip_cfg(3, 1.4);
        cfg.solver = SolverSpec::EoFlipAddBeta { tau: 1.4, beta: 0.0 }; // 全ステップ大量 tie
        let fitness = EoFlipFitnessSpec::AddBeta { beta: 0.0 };
        let ts = FlipTraceSpec { probes: vec![fitness] };

        let (p1, r1, t1) = run_eo_flip(&prob, &cfg, 42, 1.4, fitness, Some(&ts));
        let (p2, r2, _) = run_eo_flip(&prob, &cfg, 42, 1.4, fitness, None);
        assert_eq!(p1, p2, "決定論的（トレース有無でも不変）");
        assert_eq!(r1.len(), r2.len());

        let tr = t1.expect("trace");
        let max_iter = cfg.iterations() as u64;
        assert_eq!(tr.decades.iter().map(|d| d.steps).sum::<u64>(), max_iter);
        assert_eq!(tr.rank_hist.iter().sum::<u64>(), max_iter);
        // 自己プローブは同じ u で同じ頂点を選ぶ。
        for ev in &tr.probes[0].events {
            assert!(ev.agree, "自己プローブは同じ u で同じ頂点を選ぶ (step={})", ev.step);
        }
    }

    /// FlipSelectionTrace の serde round-trip。
    #[test]
    fn test_flip_trace_serde_roundtrip() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let gspec = GraphSpec { kind: GraphKind::Random, n: 20, d: 3.0, seed: 1 };
        let prob = StoredGraph::generate(gspec).problem();
        let cfg = eoflip_cfg(2, 1.4);
        let ts = FlipTraceSpec {
            probes: vec![
                EoFlipFitnessSpec::Legacy { alpha_eo: 0.064, diff_exp: 2.0 },
                EoFlipFitnessSpec::MulGamma,
            ],
        };
        let (_r, tr) = execute_traced(gspec, &cfg, &prob, 7, Some(&ts));
        let tr = tr.unwrap();
        let json = serde_json::to_string(&tr).unwrap();
        let back: FlipSelectionTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(back.config_id, tr.config_id);
        assert_eq!(back.rank_hist, tr.rank_hist);
        assert_eq!(back.probes.len(), 2);
        assert_eq!(back.probes[0].label, "legacy_a0p064_p2");
    }
}
