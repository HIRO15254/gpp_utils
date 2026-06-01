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
use crate::run_config::{RunConfig, SmoothingSpec, SolverSpec};

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
fn build_power_law_cdf(n: usize, tau: f64) -> Vec<f64> {
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

/// プレーン EO 用のスナップショット。平滑化を行わないので smoothed == real。
/// `current_*` は現在の生スコア、`basin_*`（4 フィールド）は best-so-far の m_best を表す。
fn make_eo_snapshot(step: usize, current_real: f64, best_so_far: f64) -> StepRecord {
    StepRecord {
        step,
        current_smoothed: current_real,
        current_real,
        basin_smoothed_from_smoothed: best_so_far,
        basin_real_from_smoothed: best_so_far,
        basin_smoothed_from_real: best_so_far,
        basin_real_from_real: best_so_far,
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

    let max_iter = cfg.iterations();
    let snap_steps = logarithmic_steps(max_iter);
    let mut snap_iter = snap_steps.iter().copied().peekable();
    let mut records = Vec::with_capacity(snap_steps.len() + 1);

    // 初期スナップショット (step = 0)。
    let initial_real = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
    records.push(make_eo_snapshot(0, initial_real, best_score));

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
        let lambdas: Vec<f64> = (0..n)
            .map(|v| {
                let deg = degrees[v];
                if deg == 0 {
                    1.0
                } else {
                    (deg as f64 - cuts_at[v] as f64) / deg as f64
                }
            })
            .collect();

        // λ 昇順ランク（order[0] = 最悪 = ランク 1）。
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| lambdas[a].partial_cmp(&lambdas[b]).unwrap());

        // --- 統一ランクから v1 を引く ---
        let k1 = draw_rank(&cdf, &mut rng, n);
        let v1 = order[k1];
        let set1 = current[v1];

        // --- 反対集合から v2 を引く（再抽選上限 → 一様ランダムフォールバック）---
        let mut v2 = None;
        for _ in 0..MAX_RESELECT {
            let k2 = draw_rank(&cdf, &mut rng, n);
            let cand = order[k2];
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
                records.push(make_eo_snapshot(it, real_score, best_score));
                snap_iter.next();
            }
        }
    }

    (best, records)
}

/// べき乗則 CDF から `u ~ U(0,1)` を二分探索してランク（0-indexed）を引く。
fn draw_rank(cdf: &[f64], rng: &mut Mt19937GenRand64, n: usize) -> usize {
    let u: f64 = rng.r#gen::<f64>();
    match cdf.binary_search_by(|probe| probe.partial_cmp(&u).unwrap()) {
        Ok(pos) => pos,
        Err(pos) => pos.min(n - 1),
    }
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
    let (final_p, records) = match cfg.solver {
        SolverSpec::Eo { tau } => run_eo(prob, cfg, seed, tau),
        SolverSpec::Sa => match cfg.smoothing {
            SmoothingSpec::None => run_sa_none(prob, cfg, seed),
            SmoothingSpec::KAverage(k) => run_sa_kavg(prob, k, cfg, seed),
            SmoothingSpec::RandomKAverage(k) => run_sa_random_k(prob, k, sm_seed, cfg, seed),
            SmoothingSpec::WeightedAverage(w) => run_sa_weighted(prob, w, cfg, seed),
        },
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

        // best（最終 basin）は初期カット（step0 current）以下。
        let initial = r1[0].current_real;
        let final_best = r1.last().unwrap().basin_real_from_real;
        assert!(final_best <= initial + 1e-9);

        // 返した best のカットは、整数状態から再構成した best_score と一致する。
        let (bt, bf) = get_partition_sizes(&p1);
        let best_cut = prob.count_cut_edges(&p1);
        let best_score = GraphPartitionProblem::score_from_state(best_cut, bt, bf);
        assert!((best_score - final_best).abs() < 1e-9);

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
}
