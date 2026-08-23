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
use crate::eo_rank_index::EoRankIndex;
use crate::graph_spec::GraphSpec;
use crate::optimization::Problem;
use crate::run_config::{EoFlipFitnessSpec, RunConfig, SmoothingSpec, SolverSpec};

/// ベイスン山登りのタイブレーク用 RNG シードを、SA 本体のシードから派生させる際の塩。
/// SA・スムージングの乱数列と独立にすることで `final_partition` への影響を避ける。
const TIEBREAK_SALT: u64 = 0x7113_B4EA_C0DE_5EED;

/// 「これまでの最良解」からのベイスン算出に使うタイブレーク RNG の塩。
///
/// [`TIEBREAK_SALT`] 由来の `tie_rng` から追加で乱数を引くと既存の `basin_*_from_real` が
/// 変わってしまう（回帰ベースラインが壊れる）ため、**独立した第 3 の乱数列**にする。
const BEST_TIEBREAK_SALT: u64 = 0x8E57_5EED_B4EA_C0DE;

/// 1 ステップ分の計測値。
///
/// `best_*` / `basin_diff_*` は後から追加したフィールドで、旧 JSON には存在しない。
/// `#[serde(default)]` により旧結果もそのまま読める（欠損は 0.0 / 0 になる）。
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
    /// **この時点までの最良解**（実スコア最小）の実スコア。step に対し単調非増加。
    #[serde(default)]
    pub best_real: f64,
    /// **この時点までの最良解から山登り**したベイスンの元空間評価値。
    ///
    /// 山登りはソルバー自身の近傍で行う（Flip 系 = 単一フリップ、Swap 系 = スワップ）。
    /// 最良解が更新されていない区間では値が一定になる（キャッシュ再利用）。
    #[serde(default)]
    pub basin_real_from_best: f64,
    /// 現在解のベイスンの集合サイズ差 `|A| − |B|`（true 側 − false 側、符号付き）。
    #[serde(default)]
    pub basin_diff_from_real: i64,
    /// 最良解のベイスンの集合サイズ差 `|A| − |B|`（true 側 − false 側、符号付き）。
    #[serde(default)]
    pub basin_diff_from_best: i64,
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

/// 対数刻みで採取した分割スナップショット。
///
/// `bits` は N ビットをバイト詰め（頂点 v → byte v/8 の bit v%8、LSB first）した
/// hex 文字列（長さ `2*ceil(n/8)`）。Python 側のデコードは:
/// `np.unpackbits(np.frombuffer(bytes.fromhex(s), np.uint8), bitorder='little')[:n]`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// 採取ステップ（0 = 初期解、その後 [`logarithmic_steps`] の各点）。
    pub step: usize,
    /// 分割ビット列の hex 表現。
    pub bits: String,
}

/// 1 run 分の状態スナップショット列（`seed_X_states.json`）。
///
/// 自己記述的（graph_spec / config / seed を内包）なので、オフラインプローブは
/// このファイルだけで λ 再計算に必要な情報を復元できる。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStates {
    pub graph_spec: GraphSpec,
    pub config: RunConfig,
    pub seed: u64,
    /// 頂点数（`bits` のパディングビット解釈に必要）。
    pub n: usize,
    pub snapshots: Vec<StateSnapshot>,
}

/// 分割をバイト詰め（LSB first）の hex 文字列にする。[`StateSnapshot::bits`] 参照。
pub fn pack_bits_hex(p: &[bool]) -> String {
    let nbytes = (p.len() + 7) / 8;
    let mut out = String::with_capacity(nbytes * 2);
    for byte_idx in 0..nbytes {
        let mut b: u8 = 0;
        for bit in 0..8 {
            let v = byte_idx * 8 + bit;
            if v < p.len() && p[v] {
                b |= 1 << bit;
            }
        }
        out.push_str(&format!("{:02x}", b));
    }
    out
}

/// [`pack_bits_hex`] の逆変換。長さ不一致・非 hex 文字・非ゼロのパディングビットはエラー。
pub fn unpack_bits_hex(hex: &str, n: usize) -> Result<Partition, String> {
    let nbytes = (n + 7) / 8;
    if hex.len() != nbytes * 2 {
        return Err(format!(
            "bits 長が不正: n={} なら {} 文字のはずが {} 文字",
            n,
            nbytes * 2,
            hex.len()
        ));
    }
    let mut p = vec![false; n];
    for byte_idx in 0..nbytes {
        let b = u8::from_str_radix(&hex[byte_idx * 2..byte_idx * 2 + 2], 16)
            .map_err(|e| format!("非 hex 文字 (byte {}): {}", byte_idx, e))?;
        for bit in 0..8 {
            let v = byte_idx * 8 + bit;
            let set = (b >> bit) & 1 == 1;
            if v < n {
                p[v] = set;
            } else if set {
                return Err(format!("パディングビットが非ゼロ (byte {}, bit {})", byte_idx, bit));
            }
        }
    }
    Ok(p)
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
// 「これまでの最良解」の追跡
// ============================================================================

/// その時点までに現れた**実スコア最小の解**と、そのベイスン評価値のキャッシュ。
///
/// 4 ソルバーすべてで共有する。ベイスンはスナップショット時にしか要らないので、
/// 最良解が更新された回数（`version`）を見て前回値をそのまま返す。これにより
/// 計算が減るだけでなく、`basin_real_from_best` が「最良解が更新された時だけ動く
/// 階段関数」になり解釈しやすくなる。
struct BestTracker {
    /// 最良解そのもの（`final_partition` として返す用途も兼ねる）。
    partition: Partition,
    /// 最良解の実スコア。
    score: f64,
    /// 最良解が更新されるたびに増える版番号。
    version: u64,
    /// `cached_*` を計算した時点の版番号（`u64::MAX` = 未計算）。
    cached_version: u64,
    cached_basin: f64,
    cached_diff: i64,
    /// 最良解ベイスン専用のタイブレーク RNG（本体・既存 `tie_rng` と独立）。
    tie_rng: Mt19937GenRand64,
}

impl BestTracker {
    fn new(seed: u64, partition: &Partition, score: f64) -> Self {
        Self {
            partition: partition.clone(),
            score,
            version: 0,
            cached_version: u64::MAX,
            cached_basin: 0.0,
            cached_diff: 0,
            tie_rng: Mt19937GenRand64::new(seed ^ BEST_TIEBREAK_SALT),
        }
    }

    /// 実スコアが**厳密に**改善していれば最良解を差し替える。
    #[inline]
    fn offer(&mut self, score: f64, partition: &Partition) {
        if score < self.score {
            self.score = score;
            self.partition.copy_from_slice(partition);
            self.version += 1;
        }
    }

    /// フリップ近傍での最良解ベイスン（評価値, 集合サイズ差）。
    fn basin_flip(&mut self, prob: &GraphPartitionProblem) -> (f64, i64) {
        if self.cached_version == self.version {
            return (self.cached_basin, self.cached_diff);
        }
        let (cuts_at, cut, t, f) = self.derive_state(prob);
        let (_, _, bc, bt, bf) =
            hill_climb_real_fast(prob, &self.partition, &cuts_at, cut, t, f, &mut self.tie_rng);
        self.store(bc, bt, bf)
    }

    /// スワップ近傍での最良解ベイスン（評価値, 集合サイズ差）。
    fn basin_swap(&mut self, prob: &GraphPartitionProblem) -> (f64, i64) {
        if self.cached_version == self.version {
            return (self.cached_basin, self.cached_diff);
        }
        let (cuts_at, cut, t, f) = self.derive_state(prob);
        let (_, _, bc, bt, bf) =
            hill_climb_swap_fast(prob, &self.partition, &cuts_at, cut, t, f, &mut self.tie_rng);
        self.store(bc, bt, bf)
    }

    /// 最良解から山登りに必要な派生状態を作る（O(E)。スナップショット時にしか呼ばれないので、
    /// 改善のたびに `cuts_at` を clone するより安い）。
    fn derive_state(&self, prob: &GraphPartitionProblem) -> (Vec<i32>, i32, usize, usize) {
        let cuts_at = prob.compute_cuts_at(&self.partition);
        let cut = prob.count_cut_edges(&self.partition);
        let (t, f) = get_partition_sizes(&self.partition);
        (cuts_at, cut, t, f)
    }

    fn store(&mut self, bc: i32, bt: usize, bf: usize) -> (f64, i64) {
        self.cached_basin = GraphPartitionProblem::score_from_state(bc, bt, bf);
        self.cached_diff = bt as i64 - bf as i64;
        self.cached_version = self.version;
        (self.cached_basin, self.cached_diff)
    }
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
    best: &mut BestTracker,
) -> StepRecord
where
    F: FnMut(&Partition, &[i32], i32, usize, usize) -> f64,
{
    let current_real = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
    // 最良解側は独立した RNG を使うので、既存フィールドの値には一切影響しない。
    let (basin_real_from_best, basin_diff_from_best) = best.basin_flip(prob);
    let best_real = best.score;

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
            best_real,
            basin_real_from_best,
            basin_diff_from_real: bt as i64 - bf as i64,
            basin_diff_from_best,
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
            best_real,
            basin_real_from_best,
            basin_diff_from_real: brt as i64 - brf as i64,
            basin_diff_from_best,
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

    // 最良解の追跡は **実スコア基準**（平滑化スコアではない）。`final_partition` の意味は
    // 変えず（従来どおり `current` を返す）、records の `best_*` 系だけに使う。
    let mut best = BestTracker::new(
        seed,
        &current,
        GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f),
    );

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
        &mut best,
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
            // 実スコアが動くのは受理時だけ。
            best.offer(
                GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f),
                &current,
            );
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
                    &mut best,
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
/// # 候補スコアの整数化
///
/// `a(v) = deg(v) − 2·cuts_at[v]` と置くと、v1∈A・v2∈B をスワップした後のカット数は
///
/// ```text
/// new_cut = cur_cut + a(v1) + a(v2) + 2·[v1 と v2 が隣接]
/// ```
///
/// （v1 が A→B に移ると辺 (v1,v2) はカットでなくなるので `cuts_at[v2]` が 1 減る、
/// という補正が `+2·adj`）。**スワップはバランスを変えない**のでペナルティ項
/// `ALPHA·diff²` は全候補で共通であり、実スコア `new_cut as f64 + ALPHA·diff²` の大小比較は
/// `new_cut`（i32）の大小比較と**厳密に一致**する。よって内側ループは整数演算だけで済み、
/// `delta_apply_cached` の呼び出し・隣接リストへのポインタ追跡・f64 演算をすべて省ける。
/// 走査順（v1 昇順 × v2 昇順）と `tie_rng` の消費列は元実装と同一なので、
/// **到達点はビット完全一致**する。
///
/// 1 降下ステップは O(|A|·|B|) = O(N²) のまま。スコアは厳密バランス下で整数なので、
/// 改善スワップは毎回カットを 1 以上下げ、有限ステップで停止する。
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
    let n = current.len();

    let adj = &prob.graph().adjacency_list;
    // ループ外に確保して降下ステップごとに使い回す。
    let mut a: Vec<i32> = vec![0; n];
    let mut b_side: Vec<u32> = Vec::with_capacity(n);
    // 隣接判定は「世代スタンプ」方式（毎回クリアしない）。
    let mut adj_mark: Vec<u32> = vec![0; n];
    let mut stamp: u32 = 0;

    loop {
        let mut best: Option<(usize, usize)> = None;
        let mut best_cut = cur_cut;
        let mut tie_count: u64 = 0;

        for v in 0..n {
            a[v] = adj[v].len() as i32 - 2 * cuts_at[v];
        }
        b_side.clear();
        b_side.extend((0..n).filter(|&v| !current[v]).map(|v| v as u32));

        for v1 in 0..n {
            if !current[v1] {
                continue; // v1 は A 集合（true 側）の代表のみ
            }
            stamp += 1;
            for &u in &adj[v1] {
                adj_mark[u] = stamp;
            }
            let base = cur_cut + a[v1];
            for &v2u in &b_side {
                let v2 = v2u as usize;
                let nc = base + a[v2] + if adj_mark[v2] == stamp { 2 } else { 0 };
                if nc < best_cut {
                    best_cut = nc;
                    best = Some((v1, v2));
                    tie_count = 1;
                } else if best.is_some() && nc == best_cut {
                    tie_count += 1;
                    if tie_rng.gen_range(0..tie_count) == 0 {
                        best = Some((v1, v2));
                    }
                }
            }
        }

        match best {
            Some((v1, v2)) => {
                let (c1, t1, f1, _) =
                    prob.delta_apply_cached(&current, &cuts_at, v1, cur_cut, cur_t, cur_f);
                prob.flip_vertex(&mut current, &mut cuts_at, v1);
                let (c2, t2, f2, _) =
                    prob.delta_apply_cached(&current, &cuts_at, v2, c1, t1, f1);
                prob.flip_vertex(&mut current, &mut cuts_at, v2);
                cur_cut = c2;
                cur_t = t2;
                cur_f = f2;
                debug_assert_eq!(cur_cut, best_cut, "整数デルタ式と実適用のカットが不一致");
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
    best: &mut BestTracker,
) -> StepRecord {
    let current_real = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
    // 最良解側は独立した RNG を使うので、既存フィールドの値には一切影響しない。
    let (basin_real_from_best, basin_diff_from_best) = best.basin_swap(prob);
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
        best_real: best.score,
        basin_real_from_best,
        basin_diff_from_real: bt as i64 - bf as i64,
        basin_diff_from_best,
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
    run_eo_impl(prob, cfg, seed, tau, true)
}

/// [`run_eo`] の本体。`use_index = false` で従来の全体ソート経路に切り替える
/// （差分更新索引との一致を検証する回帰テスト専用）。
fn run_eo_impl(
    prob: &GraphPartitionProblem,
    cfg: &RunConfig,
    seed: u64,
    tau: f64,
    use_index: bool,
) -> (Partition, Vec<StepRecord>) {
    let mut rng = Mt19937GenRand64::new(seed);
    let n = prob.neighbour_size();

    let mut current = balanced_init(n, &mut rng);
    let mut cur_cut = prob.count_cut_edges(&current);
    let (mut cur_t, mut cur_f) = get_partition_sizes(&current);
    let mut cuts_at = prob.compute_cuts_at(&current);

    let mut best = BestTracker::new(
        seed,
        &current,
        GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f),
    );

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
        prob, &current, &cuts_at, cur_cut, cur_t, cur_f, 0, &mut tie_rng, &mut best,
    ));

    // スワップには両集合に最低 1 頂点ずつ必要。
    if n < 2 {
        return (best.partition, records);
    }

    let cdf = build_power_law_cdf(n, tau);
    // 反対集合の頂点が当たるまでの k2 再抽選上限。超過時は反対集合から一様ランダムにフォールバック。
    const MAX_RESELECT: usize = 50;

    // λ = g/deg は cuts_at だけの関数なので、スワップで変化するのは v1,v2 とその隣接だけ。
    // 差分更新索引（[`EoRankIndex`]）はこれを O(deg log M) で反映し、全体ソートを不要にする。
    // 返り値は全体ソート版と厳密一致する（`use_index = false` の経路と回帰テストで照合）。
    let mut index = if use_index {
        Some(EoRankIndex::new(&degrees, &cuts_at))
    } else {
        None
    };
    // ソート経路用のバッファ（毎ステップ確保しないようループ外に置く）。
    let mut lambdas = vec![0.0f64; n];
    let mut order: Vec<usize> = (0..n).collect();

    for it in 1..=max_iter {
        // --- 適応度 λ の昇順ランクを用意（order[0] = 最悪 = ランク 1）---
        if index.is_none() {
            for (v, lam) in lambdas.iter_mut().enumerate() {
                *lam = swap_fitness(degrees[v], cuts_at[v]);
            }
            for (i, o) in order.iter_mut().enumerate() {
                *o = i;
            }
            order.sort_by(|&a, &b| lambdas[a].partial_cmp(&lambdas[b]).unwrap());
        }

        // --- 統一ランクから v1 を引く（同率群は平均化規則で等確率）---
        let (v1, _k1) = match &index {
            Some(ix) => ix.select(&cdf, rng.r#gen::<f64>()),
            None => select_eo_rank(&lambdas, &order, &cdf, rng.r#gen::<f64>()),
        };
        let set1 = current[v1];

        // --- 反対集合から v2 を引く（再抽選上限 → 一様ランダムフォールバック）---
        let mut v2 = None;
        for _ in 0..MAX_RESELECT {
            let (cand, _k2) = match &index {
                Some(ix) => ix.select(&cdf, rng.r#gen::<f64>()),
                None => select_eo_rank(&lambdas, &order, &cdf, rng.r#gen::<f64>()),
            };
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
        if let Some(ix) = index.as_mut() {
            update_index_after_flip(prob, &cuts_at, &degrees, v1, ix);
        }
        let (c2, t2, f2, _) = prob.delta_apply_cached(&current, &cuts_at, v2, c1, t1, f1);
        prob.flip_vertex(&mut current, &mut cuts_at, v2);
        if let Some(ix) = index.as_mut() {
            update_index_after_flip(prob, &cuts_at, &degrees, v2, ix);
        }
        cur_cut = c2;
        cur_t = t2; // スワップ後は元の値に戻る（バランス維持）。
        cur_f = f2;

        let real_score = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
        best.offer(real_score, &current);

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
            if let Some(ix) = index.as_ref() {
                debug_assert_eq!(
                    ix.debug_verify(&degrees, &cuts_at),
                    Ok(()),
                    "EoRankIndex drift at it={}",
                    it
                );
            }
        }

        if let Some(&want) = snap_iter.peek() {
            if it == want {
                records.push(make_swap_snapshot(
                    prob, &current, &cuts_at, cur_cut, cur_t, cur_f, it, &mut tie_rng,
                    &mut best,
                ));
                snap_iter.next();
            }
        }
    }

    (best.partition, records)
}

/// スワップ版 EO の次数正規化適応度 `λ0 = g/deg = (deg - cuts)/deg`（孤立頂点は 1.0）。
/// `run_eo` の λ 計算そのものであり、フリップ近傍版の新適応度（`eo_flip_lambda_mul_alpha` 等）
/// でも「λ0 はスワップ版 EO の適応度と同じ」を表す共通部品として使う。
pub(crate) fn swap_fitness(deg: usize, cuts: i32) -> f64 {
    if deg == 0 {
        1.0
    } else {
        (deg as f64 - cuts as f64) / deg as f64
    }
}

/// 頂点 `idx` のフリップ**後**に、λ が変化した頂点（`idx` とその隣接）だけを索引へ反映する。
///
/// `cuts_at` は既にフリップ後の値であること（[`GraphPartitionProblem::flip_vertex`] 実行済み）。
/// 隣接リストに重複があっても [`EoRankIndex::update_vertex`] は冪等なので安全。
#[inline]
fn update_index_after_flip(
    prob: &GraphPartitionProblem,
    cuts_at: &[i32],
    degrees: &[usize],
    idx: usize,
    index: &mut EoRankIndex,
) {
    index.update_vertex(idx, degrees[idx], cuts_at[idx]);
    for &u in &prob.graph().adjacency_list[idx] {
        index.update_vertex(u, degrees[u], cuts_at[u]);
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
pub fn is_majority_side(in_true: bool, t: usize, f: usize) -> bool {
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

/// 保存済み分割から λ 計算に必要な派生状態（cuts_at / 集合サイズ）を組み立てる（O(E)）。
///
/// [`eo_flip_lambdas`] と組で使い、走行時と同じ入力を復元する。呼び出し側が
/// `cuts_at`・`cur_t`・`cur_f` を個別に用意すると不整合の余地があるため、この型で束ねる。
pub struct StateContext {
    pub cuts_at: Vec<i32>,
    pub cur_t: usize,
    pub cur_f: usize,
}

/// 分割から [`StateContext`] を構築する。
pub fn state_context(prob: &GraphPartitionProblem, current: &Partition) -> StateContext {
    let (cur_t, cur_f) = get_partition_sizes(current);
    StateContext {
        cuts_at: prob.compute_cuts_at(current),
        cur_t,
        cur_f,
    }
}

/// 全頂点の次数リスト（[`run_eo_flip`] 内部と同じ定義）。
pub fn degrees_of(prob: &GraphPartitionProblem) -> Vec<usize> {
    let n = prob.neighbour_size();
    (0..n)
        .map(|v| prob.graph().adjacency_list[v].len())
        .collect()
}

/// 保存済み状態に対する EoFlip 適応度 λ のオフライン再計算。
///
/// 走行時の [`compute_eo_flip_lambdas`] に委譲するため、同一状態なら結果は
/// **ビット単位で一致**する。`out` の長さは頂点数に揃えること。
pub fn eo_flip_lambdas(
    fitness: &EoFlipFitnessSpec,
    current: &Partition,
    ctx: &StateContext,
    degrees: &[usize],
    out: &mut [f64],
) {
    compute_eo_flip_lambdas(fitness, current, &ctx.cuts_at, degrees, ctx.cur_t, ctx.cur_f, out);
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
fn run_eo_flip(
    prob: &GraphPartitionProblem,
    cfg: &RunConfig,
    seed: u64,
    tau: f64,
    fitness: EoFlipFitnessSpec,
    states_out: Option<&mut Vec<StateSnapshot>>,
) -> (Partition, Vec<StepRecord>) {
    // 差分更新索引は λ が (deg, cuts) だけで決まる場合にしか使えない。
    // MulAlpha の α=1 は λ1 ≡ 1 で `swap_fitness * 1.0`（f64 恒等）なので該当する。
    let use_index = matches!(fitness, EoFlipFitnessSpec::MulAlpha { alpha } if alpha == 1.0);
    run_eo_flip_impl(prob, cfg, seed, tau, fitness, states_out, use_index)
}

/// [`run_eo_flip`] の本体。`use_index = false` で従来の全体ソート経路に切り替える
/// （差分更新索引との一致を検証する回帰テスト専用）。
#[allow(clippy::too_many_arguments)]
fn run_eo_flip_impl(
    prob: &GraphPartitionProblem,
    cfg: &RunConfig,
    seed: u64,
    tau: f64,
    fitness: EoFlipFitnessSpec,
    mut states_out: Option<&mut Vec<StateSnapshot>>,
    use_index: bool,
) -> (Partition, Vec<StepRecord>) {
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

    let mut best = BestTracker::new(
        seed,
        &current,
        GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f),
    );

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
        &mut best,
    ));
    // 状態収穫は records と同じタイミングで、乱数を一切消費せずに行う
    // （軌道は保存有無でビット単位に不変）。
    if let Some(out) = states_out.as_deref_mut() {
        out.push(StateSnapshot {
            step: 0,
            bits: pack_bits_hex(&current),
        });
    }

    if n == 0 {
        return (best.partition, records);
    }

    let cdf = build_power_law_cdf(n, tau);

    // λ = g/deg（MulAlpha α=1）のときはフリップした頂点とその隣接だけ λ が変わるので、
    // 全体ソートを差分更新索引に置き換える。返り値はソート版と厳密一致する。
    let mut index = if use_index {
        Some(EoRankIndex::new(&degrees, &cuts_at))
    } else {
        None
    };
    let mut lambdas = vec![0.0f64; n];
    let mut order: Vec<usize> = (0..n).collect();

    for it in 1..=max_iter {
        // 各頂点の適応度（O(N)）→ 昇順ランク（O(N log N)）。索引経路では両方とも不要。
        if index.is_none() {
            compute_eo_flip_lambdas(
                &fitness, &current, &cuts_at, &degrees, cur_t, cur_f, &mut lambdas,
            );

            for (i, o) in order.iter_mut().enumerate() {
                *o = i;
            }
            order.sort_by(|&a, &b| lambdas[a].partial_cmp(&lambdas[b]).unwrap());
        }

        // ランク抽選（乱数消費は 1 draw/step）。
        let u: f64 = rng.r#gen::<f64>();
        let (idx, _k) = match &index {
            Some(ix) => ix.select(&cdf, u),
            None => select_eo_rank(&lambdas, &order, &cdf, u),
        };

        // 無条件フリップ（受理判定なし）。
        let (nc, nt, nf, _) =
            prob.delta_apply_cached(&current, &cuts_at, idx, cur_cut, cur_t, cur_f);

        prob.flip_vertex(&mut current, &mut cuts_at, idx);
        if let Some(ix) = index.as_mut() {
            update_index_after_flip(prob, &cuts_at, &degrees, idx, ix);
        }
        cur_cut = nc;
        cur_t = nt;
        cur_f = nf;

        let real_score = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
        best.offer(real_score, &current);

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
            if let Some(ix) = index.as_ref() {
                debug_assert_eq!(
                    ix.debug_verify(&degrees, &cuts_at),
                    Ok(()),
                    "EoRankIndex drift at it={}",
                    it
                );
            }
        }

        if let Some(&want) = snap_iter.peek() {
            if it == want {
                let cs = GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f);
                records.push(make_snapshot_fast(
                    prob, &current, &cuts_at, cur_cut, cur_t, cur_f, cs, it, &mut sm, true,
                    &mut tie_rng, &mut best,
                ));
                if let Some(out) = states_out.as_deref_mut() {
                    out.push(StateSnapshot {
                        step: it,
                        bits: pack_bits_hex(&current),
                    });
                }
                snap_iter.next();
            }
        }
    }

    (best.partition, records)
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

    let mut best = BestTracker::new(
        seed,
        &current,
        GraphPartitionProblem::score_from_state(cur_cut, cur_t, cur_f),
    );

    // ベイスン算出（スワップ降下）のタイブレーク専用 RNG。本体 rng と独立。
    let mut tie_rng = Mt19937GenRand64::new(seed ^ TIEBREAK_SALT);

    let temperature = cfg.temperature();
    let max_iter = cfg.iterations();
    let snap_steps = logarithmic_steps(max_iter);
    let mut snap_iter = snap_steps.iter().copied().peekable();
    let mut records = Vec::with_capacity(snap_steps.len() + 1);

    records.push(make_swap_snapshot(
        prob, &current, &cuts_at, cur_cut, cur_t, cur_f, 0, &mut tie_rng, &mut best,
    ));

    if n < 2 {
        return (best.partition, records);
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
            best.offer(swap_score, &current);
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
                    &mut best,
                ));
                snap_iter.next();
            }
        }
    }

    (best.partition, records)
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
    execute_with_states(spec, cfg, prob, seed, false).0
}

/// [`execute`] の状態収穫つき版。`collect_states = true` かつ EoFlip 系ソルバーのときのみ
/// [`RunStates`] を返す（他ソルバーは常に `None`）。収穫は乱数を消費しないため、
/// `collect_states` の有無で `RunResult` はビット単位に一致する（`elapsed_ms` を除く）。
pub fn execute_with_states(
    spec: GraphSpec,
    cfg: &RunConfig,
    prob: &GraphPartitionProblem,
    seed: u64,
    collect_states: bool,
) -> (RunResult, Option<RunStates>) {
    let t0 = std::time::Instant::now();
    let sm_seed = seed.wrapping_add(0xDEAD_BEEF);
    let mut states: Option<Vec<StateSnapshot>> = None;
    let (final_p, records) = match cfg.solver {
        SolverSpec::Eo { tau } => run_eo(prob, cfg, seed, tau),
        SolverSpec::EoFlip { tau, .. }
        | SolverSpec::EoFlipMulAlpha { tau, .. }
        | SolverSpec::EoFlipAddBeta { tau, .. }
        | SolverSpec::EoFlipMulGamma { tau } => {
            let fitness = EoFlipFitnessSpec::from_solver(&cfg.solver)
                .expect("EoFlip 系 solver は必ず fitness を持つ");
            if collect_states {
                states = Some(Vec::new());
            }
            run_eo_flip(prob, cfg, seed, tau, fitness, states.as_mut())
        }
        SolverSpec::SaSwap => run_sa_swap(prob, cfg, seed),
        SolverSpec::Sa => match cfg.smoothing {
            SmoothingSpec::None => run_sa_none(prob, cfg, seed),
            SmoothingSpec::KAverage(k) => run_sa_kavg(prob, k, cfg, seed),
            SmoothingSpec::RandomKAverage(k) => run_sa_random_k(prob, k, sm_seed, cfg, seed),
            SmoothingSpec::WeightedAverage(w) => run_sa_weighted(prob, w, cfg, seed),
        },
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
    let run_states = states.map(|snapshots| RunStates {
        graph_spec: spec,
        config: cfg.clone(),
        seed,
        n: prob.neighbour_size(),
        snapshots,
    });
    (result, run_states)
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

    /// 状態スナップショットファイルのパス（`base/<graph_id>/<config_id>/seed_<seed>_states.json`）。
    pub fn states_path_for(&self, spec: &GraphSpec, cfg: &RunConfig, seed: u64) -> PathBuf {
        self.base_dir
            .join(spec.id())
            .join(cfg.id())
            .join(format!("seed_{}_states.json", seed))
    }

    pub fn states_exist(&self, spec: &GraphSpec, cfg: &RunConfig, seed: u64) -> bool {
        self.states_path_for(spec, cfg, seed).exists()
    }

    /// 状態スナップショットを保存する。ファイル数が多いため pretty ではなく compact JSON。
    pub fn save_states(&self, states: &RunStates) -> Result<(), String> {
        let p = self.states_path_for(&states.graph_spec, &states.config, states.seed);
        if let Some(parent) = p.parent() {
            ensure_dir_exists(parent).map_err(|e| format!("create dir: {}", e))?;
        }
        let json = serde_json::to_string(states).map_err(|e| format!("serialize: {}", e))?;
        std::fs::write(&p, json).map_err(|e| format!("save states: {}", e))
    }

    pub fn load_states(&self, spec: &GraphSpec, cfg: &RunConfig, seed: u64) -> Option<RunStates> {
        load_json::<RunStates>(&self.states_path_for(spec, cfg, seed)).ok()
    }

    /// gnuplot で扱いやすい TSV を出力する。
    /// 列: step, current_smoothed, current_real,
    ///     basin_smoothed_from_smoothed, basin_real_from_smoothed,
    ///     basin_smoothed_from_real, basin_real_from_real,
    ///     best_real, basin_real_from_best, basin_diff_from_real, basin_diff_from_best
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
            "# step\tcur_sm\tcur_real\tbasin_sm_from_sm\tbasin_real_from_sm\tbasin_sm_from_real\tbasin_real_from_real\tbest_real\tbasin_real_from_best\tbasin_diff_from_real\tbasin_diff_from_best"
        )
        .map_err(|e| format!("write: {}", e))?;
        for r in &result.records {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                r.step,
                r.current_smoothed,
                r.current_real,
                r.basin_smoothed_from_smoothed,
                r.basin_real_from_smoothed,
                r.basin_smoothed_from_real,
                r.basin_real_from_real,
                r.best_real,
                r.basin_real_from_best,
                r.basin_diff_from_real,
                r.basin_diff_from_best
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
        let (p1, r1) = run_eo_flip(&prob, &cfg, 42, 1.4, legacy_fitness(), None);
        let (p2, _r2) = run_eo_flip(&prob, &cfg, 42, 1.4, legacy_fitness(), None);
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
        let (best, _r) = run_eo_flip(
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
        let (_p1, r_lo) = run_eo_flip(
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
        let (_p2, r_hi) = run_eo_flip(
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

    /// EoFlipFitnessSpec::from_solver の対応。
    #[test]
    fn test_eo_flip_fitness_spec_from_solver() {
        assert_eq!(
            EoFlipFitnessSpec::from_solver(&SolverSpec::EoFlipMulGamma { tau: 1.2 }),
            Some(EoFlipFitnessSpec::MulGamma)
        );
        assert_eq!(EoFlipFitnessSpec::from_solver(&SolverSpec::Sa), None);
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

    /// tie が大量に出る設定（AddBeta β=0）でも決定論的（同じ seed なら同じ軌道）。
    #[test]
    fn test_eo_flip_tie_heavy_runs() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let gspec = GraphSpec { kind: GraphKind::Random, n: 40, d: 4.0, seed: 2 };
        let prob = StoredGraph::generate(gspec).problem();
        let mut cfg = eoflip_cfg(3, 1.4);
        cfg.solver = SolverSpec::EoFlipAddBeta { tau: 1.4, beta: 0.0 }; // 全ステップ大量 tie
        let fitness = EoFlipFitnessSpec::AddBeta { beta: 0.0 };

        let (p1, r1) = run_eo_flip(&prob, &cfg, 42, 1.4, fitness, None);
        let (p2, r2) = run_eo_flip(&prob, &cfg, 42, 1.4, fitness, None);
        assert_eq!(p1, p2, "決定論的");
        assert_eq!(r1.len(), r2.len());
    }

    // ------------------------------------------------------------------
    // 状態スナップショット（save_states / オフライン λ）
    // ------------------------------------------------------------------

    #[test]
    fn test_pack_unpack_bits_roundtrip() {
        let mut rng = Mt19937GenRand64::new(7);
        for n in [0usize, 1, 3, 7, 8, 9, 250, 251] {
            let p: Vec<bool> = (0..n).map(|_| rng.r#gen::<bool>()).collect();
            let hex = pack_bits_hex(&p);
            assert_eq!(hex.len(), 2 * ((n + 7) / 8), "n={n}");
            let back = unpack_bits_hex(&hex, n).unwrap();
            assert_eq!(back, p, "n={n}");
        }
    }

    #[test]
    fn test_unpack_bits_rejects_bad_input() {
        // 長さ不一致
        assert!(unpack_bits_hex("00", 250).is_err());
        // 非 hex 文字
        assert!(unpack_bits_hex("zz", 8).is_err());
        // 非ゼロのパディングビット（n=4 なら上位 4bit は 0 のはず）
        assert!(unpack_bits_hex("f0", 4).is_err());
        assert!(unpack_bits_hex("0f", 4).is_ok());
    }

    /// states 収集: step 列は [0] ++ logarithmic_steps、各スナップは records と整合。
    #[test]
    fn test_run_eo_flip_states_steps_and_scores() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let gspec = GraphSpec { kind: GraphKind::Random, n: 30, d: 4.0, seed: 5 };
        let prob = StoredGraph::generate(gspec).problem();
        let cfg = eoflip_cfg(3, 1.4);
        let fitness = EoFlipFitnessSpec::MulGamma;

        let mut states = Vec::new();
        let (_best, records) = run_eo_flip(&prob, &cfg, 11, 1.4, fitness, Some(&mut states));

        let mut want_steps = vec![0usize];
        want_steps.extend(logarithmic_steps(cfg.iterations()));
        assert_eq!(
            states.iter().map(|s| s.step).collect::<Vec<_>>(),
            want_steps
        );
        assert_eq!(states.len(), records.len());

        for (snap, rec) in states.iter().zip(records.iter()) {
            assert_eq!(snap.step, rec.step);
            let p = unpack_bits_hex(&snap.bits, 30).unwrap();
            let cut = prob.count_cut_edges(&p);
            let (t, f) = get_partition_sizes(&p);
            let score = GraphPartitionProblem::score_from_state(cut, t, f);
            // 完全一致（score_from_state は走行時と同一の式・同一の整数状態）。
            assert_eq!(score, rec.current_real, "step={}", snap.step);
        }
    }

    /// states 収集の有無で軌道（final_partition・全 StepRecord）がビット単位に不変。
    #[test]
    fn test_states_collection_does_not_change_trajectory() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let gspec = GraphSpec { kind: GraphKind::Random, n: 40, d: 5.0, seed: 9 };
        let prob = StoredGraph::generate(gspec).problem();
        let cfg = eoflip_cfg(3, 1.1);
        let fitness = EoFlipFitnessSpec::MulAlpha { alpha: 0.3 };

        let (p_plain, r_plain) = run_eo_flip(&prob, &cfg, 123, 1.1, fitness, None);
        let mut states = Vec::new();
        let (p_traced, r_traced) =
            run_eo_flip(&prob, &cfg, 123, 1.1, fitness, Some(&mut states));

        assert_eq!(p_plain, p_traced);
        assert_eq!(r_plain.len(), r_traced.len());
        for (a, b) in r_plain.iter().zip(r_traced.iter()) {
            assert_eq!(a.step, b.step);
            assert_eq!(a.current_real, b.current_real);
            assert_eq!(a.current_smoothed, b.current_smoothed);
            assert_eq!(a.basin_real_from_real, b.basin_real_from_real);
            assert_eq!(a.basin_smoothed_from_smoothed, b.basin_smoothed_from_smoothed);
            assert_eq!(a.basin_real_from_smoothed, b.basin_real_from_smoothed);
            assert_eq!(a.basin_smoothed_from_real, b.basin_smoothed_from_real);
        }
        assert!(!states.is_empty());
    }

    /// 最終スナップショットは current（best とは限らない）。records の最終値と一致する。
    #[test]
    fn test_last_snapshot_is_current_not_best() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let gspec = GraphSpec { kind: GraphKind::Random, n: 40, d: 5.0, seed: 2 };
        let prob = StoredGraph::generate(gspec).problem();
        let cfg = eoflip_cfg(4, 1.4);
        let fitness = EoFlipFitnessSpec::Legacy {
            alpha_eo: DEFAULT_EO_FLIP_ALPHA,
            diff_exp: DEFAULT_EO_FLIP_DIFF_EXP,
        };

        let mut states = Vec::new();
        let (best, records) = run_eo_flip(&prob, &cfg, 3, 1.4, fitness, Some(&mut states));
        let last = states.last().unwrap();
        let p = unpack_bits_hex(&last.bits, 40).unwrap();
        let cut = prob.count_cut_edges(&p);
        let (t, f) = get_partition_sizes(&p);
        let score = GraphPartitionProblem::score_from_state(cut, t, f);
        assert_eq!(score, records.last().unwrap().current_real);
        // EO は無条件受理なので通常 current(最終) != best。最低限、best のスコア以上。
        let best_cut = prob.count_cut_edges(&best);
        let (bt, bf) = get_partition_sizes(&best);
        let best_score = GraphPartitionProblem::score_from_state(best_cut, bt, bf);
        assert!(score >= best_score);
    }

    /// オフライン λ 再計算が走行時と同一状態でビット一致する。
    #[test]
    fn test_offline_lambdas_bit_match() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let gspec = GraphSpec { kind: GraphKind::Geometric, n: 50, d: 6.0, seed: 4 };
        let prob = StoredGraph::generate(gspec).problem();
        let cfg = eoflip_cfg(3, 1.4);
        let degrees = degrees_of(&prob);

        let probe_specs = [
            EoFlipFitnessSpec::Legacy { alpha_eo: 0.064, diff_exp: 2.0 },
            EoFlipFitnessSpec::MulAlpha { alpha: 0.1 },
            EoFlipFitnessSpec::AddBeta { beta: 0.0 },
            EoFlipFitnessSpec::MulGamma,
        ];

        let mut states = Vec::new();
        let _ = run_eo_flip(
            &prob,
            &cfg,
            21,
            1.4,
            EoFlipFitnessSpec::MulGamma,
            Some(&mut states),
        );

        for snap in &states {
            let p = unpack_bits_hex(&snap.bits, 50).unwrap();
            let ctx = state_context(&prob, &p);
            for fs in &probe_specs {
                // オフライン API 経由と、走行時内部関数の直呼びが完全一致。
                let mut a = vec![0.0f64; 50];
                let mut b = vec![0.0f64; 50];
                eo_flip_lambdas(fs, &p, &ctx, &degrees, &mut a);
                compute_eo_flip_lambdas(fs, &p, &ctx.cuts_at, &degrees, ctx.cur_t, ctx.cur_f, &mut b);
                assert_eq!(a, b, "spec={fs:?} step={}", snap.step);
            }
        }
    }

    /// execute_with_states: EoFlip 系のみ Some、他は None。RunStates のメタ情報も整合。
    #[test]
    fn test_execute_with_states_dispatch() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let gspec = GraphSpec { kind: GraphKind::Random, n: 24, d: 3.0, seed: 1 };
        let prob = StoredGraph::generate(gspec).problem();

        let mut cfg = eoflip_cfg(2, 1.4);
        cfg.solver = SolverSpec::EoFlipMulGamma { tau: 1.4 };
        let (res, states) = execute_with_states(gspec, &cfg, &prob, 5, true);
        let st = states.expect("EoFlip 系は states を返す");
        assert_eq!(st.n, 24);
        assert_eq!(st.seed, 5);
        assert_eq!(st.graph_spec, gspec);
        assert_eq!(st.snapshots.len(), res.records.len());
        // collect_states=false なら None。
        let (_res2, none_states) = execute_with_states(gspec, &cfg, &prob, 5, false);
        assert!(none_states.is_none());

        // 非 EoFlip 系は collect_states=true でも None。
        let sa_cfg = RunConfig::new("sa");
        let (_r, s) = execute_with_states(gspec, &sa_cfg, &prob, 5, true);
        assert!(s.is_none());
    }

    /// ResultStore の states 保存/読込の往復。
    #[test]
    fn test_result_store_states_roundtrip() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        let gspec = GraphSpec { kind: GraphKind::Random, n: 24, d: 3.0, seed: 1 };
        let prob = StoredGraph::generate(gspec).problem();
        let mut cfg = eoflip_cfg(2, 1.4);
        cfg.solver = SolverSpec::EoFlipMulGamma { tau: 1.4 };
        let (_res, states) = execute_with_states(gspec, &cfg, &prob, 8, true);
        let st = states.unwrap();

        let dir = std::env::temp_dir().join(format!("gpp_states_test_{}", std::process::id()));
        let store = ResultStore::new(&dir);
        assert!(!store.states_exist(&gspec, &cfg, 8));
        store.save_states(&st).unwrap();
        assert!(store.states_exist(&gspec, &cfg, 8));
        let loaded = store.load_states(&gspec, &cfg, 8).unwrap();
        assert_eq!(loaded.n, st.n);
        assert_eq!(loaded.seed, st.seed);
        assert_eq!(
            loaded.snapshots.iter().map(|s| (&s.bits, s.step)).collect::<Vec<_>>(),
            st.snapshots.iter().map(|s| (&s.bits, s.step)).collect::<Vec<_>>()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------
    // 新指標（best_real / basin_real_from_best / basin_diff_*）
    // ------------------------------------------------------------------

    /// 4 ソルバーすべてで新指標の不変条件が成り立つ:
    /// (a) `best_real` は step に対し単調非増加
    /// (b) `basin_real_from_best ≤ best_real`（山登りは悪化しない）
    /// (c) `best_real` が変わらない区間では `basin_real_from_best` も一定（キャッシュ再利用）
    /// (d) `basin_real_from_best ≥` 全 step の `basin_real_from_real` の最小値ではない
    ///     （最良解のベイスンが常に最良ベイスンとは限らない）ことは検査しない
    #[test]
    fn test_best_metrics_invariants() {
        use crate::graph_spec::{GraphKind, StoredGraph};

        let spec = GraphSpec { kind: GraphKind::Random, n: 60, d: 5.0, seed: 4 };
        let prob = StoredGraph::generate(spec).problem();

        let mut cfgs: Vec<RunConfig> = Vec::new();
        // flip SA
        let mut c = RunConfig::new("sa");
        c.theta = Some(-0.3);
        c.smoothing = SmoothingSpec::None;
        c.log10_iterations = 4;
        c.solver = SolverSpec::Sa;
        cfgs.push(c);
        // swap SA
        let mut c = RunConfig::new("saswap");
        c.theta = Some(-0.3);
        c.smoothing = SmoothingSpec::None;
        c.log10_iterations = 4;
        c.solver = SolverSpec::SaSwap;
        cfgs.push(c);
        // swap EO / flip EO（元論文の λ=g/deg）
        cfgs.push(eo_cfg(4, 1.4));
        let mut c = RunConfig::new("eoflipmulalpha");
        c.theta = None;
        c.smoothing = SmoothingSpec::None;
        c.log10_iterations = 4;
        c.solver = SolverSpec::EoFlipMulAlpha { tau: 1.4, alpha: 1.0 };
        cfgs.push(c);

        for cfg in &cfgs {
            let res = execute(spec, cfg, &prob, 11);
            let id = cfg.id();
            let mut prev: Option<&StepRecord> = None;
            for r in &res.records {
                assert!(
                    r.basin_real_from_best <= r.best_real,
                    "{id} step {}: ベイスン {} > 最良解 {}",
                    r.step,
                    r.basin_real_from_best,
                    r.best_real
                );
                if let Some(p) = prev {
                    assert!(
                        r.best_real <= p.best_real,
                        "{id} step {}: best_real が悪化 ({} -> {})",
                        r.step,
                        p.best_real,
                        r.best_real
                    );
                    if r.best_real == p.best_real {
                        assert_eq!(
                            r.basin_real_from_best.to_bits(),
                            p.basin_real_from_best.to_bits(),
                            "{id} step {}: 最良解が同じなのにベイスンが変わった",
                            r.step
                        );
                        assert_eq!(
                            r.basin_diff_from_best, p.basin_diff_from_best,
                            "{id} step {}: 最良解が同じなのに diff が変わった",
                            r.step
                        );
                    }
                }
                prev = Some(r);
            }
            // 最終レコードの best_real は全 current_real の最小値と一致するはず
            // （current_real は対数刻みの抜き取りなので「以下」でしか照合できない）。
            let min_seen = res
                .records
                .iter()
                .map(|r| r.current_real)
                .fold(f64::INFINITY, f64::min);
            let last = res.records.last().expect("records 非空");
            assert!(
                last.best_real <= min_seen,
                "{id}: best_real {} > 観測された最小 current_real {}",
                last.best_real,
                min_seen
            );
        }
    }

    /// スワップ系は厳密バランスなので、ベイスンの集合サイズ差は常に 0（偶数 N）。
    /// フリップ系は不均衡を取りうる。
    #[test]
    fn test_basin_diff_zero_for_swap_solvers() {
        use crate::graph_spec::{GraphKind, StoredGraph};

        let spec = GraphSpec { kind: GraphKind::Random, n: 60, d: 5.0, seed: 5 };
        let prob = StoredGraph::generate(spec).problem();
        for cfg in [eo_cfg(4, 1.4), {
            let mut c = RunConfig::new("saswap");
            c.theta = Some(-0.3);
            c.smoothing = SmoothingSpec::None;
            c.log10_iterations = 4;
            c.solver = SolverSpec::SaSwap;
            c
        }] {
            let res = execute(spec, &cfg, &prob, 2);
            for r in &res.records {
                assert_eq!(r.basin_diff_from_real, 0, "{}: step {}", cfg.id(), r.step);
                assert_eq!(r.basin_diff_from_best, 0, "{}: step {}", cfg.id(), r.step);
            }
        }
    }

    /// 旧 JSON（新フィールドなし）が `#[serde(default)]` で読めること。
    #[test]
    fn test_step_record_deserializes_legacy_json() {
        let legacy = r#"{
            "step": 10,
            "current_smoothed": 1.0,
            "current_real": 2.0,
            "basin_smoothed_from_smoothed": 3.0,
            "basin_real_from_smoothed": 4.0,
            "basin_smoothed_from_real": 5.0,
            "basin_real_from_real": 6.0
        }"#;
        let r: StepRecord = serde_json::from_str(legacy).expect("旧 JSON が読めること");
        assert_eq!(r.step, 10);
        assert_eq!(r.basin_real_from_real, 6.0);
        assert_eq!(r.best_real, 0.0);
        assert_eq!(r.basin_real_from_best, 0.0);
        assert_eq!(r.basin_diff_from_real, 0);
        assert_eq!(r.basin_diff_from_best, 0);
    }

    // ------------------------------------------------------------------
    // 差分更新索引（EoRankIndex）とソート版のビット一致回帰
    // ------------------------------------------------------------------

    /// `StepRecord` 列を f64 のビットパターンまで含めて比較する。
    fn assert_records_bit_identical(a: &[StepRecord], b: &[StepRecord], ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: レコード数が不一致");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.step, y.step, "{ctx}: step 不一致 (rec {i})");
            let fields: [(&str, f64, f64); 6] = [
                ("current_smoothed", x.current_smoothed, y.current_smoothed),
                ("current_real", x.current_real, y.current_real),
                (
                    "basin_smoothed_from_smoothed",
                    x.basin_smoothed_from_smoothed,
                    y.basin_smoothed_from_smoothed,
                ),
                (
                    "basin_real_from_smoothed",
                    x.basin_real_from_smoothed,
                    y.basin_real_from_smoothed,
                ),
                (
                    "basin_smoothed_from_real",
                    x.basin_smoothed_from_real,
                    y.basin_smoothed_from_real,
                ),
                ("basin_real_from_real", x.basin_real_from_real, y.basin_real_from_real),
            ];
            for (name, xv, yv) in fields {
                assert_eq!(
                    xv.to_bits(),
                    yv.to_bits(),
                    "{ctx}: {name} がビット不一致 (rec {i}, step {}): {xv} vs {yv}",
                    x.step
                );
            }
        }
    }

    /// スワップ版 EO: 索引経路とソート経路が `final_partition`・`records` ともビット完全一致する。
    #[test]
    fn test_eo_index_matches_sort_path() {
        use crate::graph_spec::{GraphKind, StoredGraph};

        for (kind, n, d) in [
            (GraphKind::Random, 64usize, 5.0f64),
            (GraphKind::Random, 96, 12.0),
            (GraphKind::Geometric, 80, 8.0),
        ] {
            let prob = StoredGraph::generate(GraphSpec { kind, n, d, seed: 1 }).problem();
            for tau in [0.6_f64, 1.3, 2.0] {
                let cfg = eo_cfg(4, tau);
                for seed in [0u64, 7, 12345] {
                    let (best_ix, rec_ix) = run_eo_impl(&prob, &cfg, seed, tau, true);
                    let (best_sort, rec_sort) = run_eo_impl(&prob, &cfg, seed, tau, false);
                    let ctx = format!("{kind:?} n={n} d={d} tau={tau} seed={seed}");
                    assert_eq!(best_ix, best_sort, "{ctx}: final_partition 不一致");
                    assert_records_bit_identical(&rec_ix, &rec_sort, &ctx);
                }
            }
        }
    }

    /// フリップ版 EO（MulAlpha α=1 = 元論文の λ=g/deg）も同様にビット完全一致する。
    #[test]
    fn test_eo_flip_mulalpha1_index_matches_sort_path() {
        use crate::graph_spec::{GraphKind, StoredGraph};

        for (kind, n, d) in [
            (GraphKind::Random, 64usize, 5.0f64),
            (GraphKind::Random, 96, 12.0),
            (GraphKind::Geometric, 80, 8.0),
        ] {
            let prob = StoredGraph::generate(GraphSpec { kind, n, d, seed: 2 }).problem();
            for tau in [0.6_f64, 1.3, 2.0] {
                let mut cfg = RunConfig::new("eoflipmulalpha");
                cfg.theta = None;
                cfg.smoothing = SmoothingSpec::None;
                cfg.log10_iterations = 4;
                cfg.solver = SolverSpec::EoFlipMulAlpha { tau, alpha: 1.0 };
                let fitness = EoFlipFitnessSpec::MulAlpha { alpha: 1.0 };
                for seed in [0u64, 7, 12345] {
                    let (best_ix, rec_ix) =
                        run_eo_flip_impl(&prob, &cfg, seed, tau, fitness, None, true);
                    let (best_sort, rec_sort) =
                        run_eo_flip_impl(&prob, &cfg, seed, tau, fitness, None, false);
                    let ctx = format!("{kind:?} n={n} d={d} tau={tau} seed={seed}");
                    assert_eq!(best_ix, best_sort, "{ctx}: final_partition 不一致");
                    assert_records_bit_identical(&rec_ix, &rec_sort, &ctx);
                }
            }
        }
    }

    /// 孤立頂点（deg=0）を含むグラフでも索引経路が成立する。
    #[test]
    fn test_eo_index_with_isolated_vertex() {
        let mut graph = crate::graph_partition::Graph::new(8);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        // 頂点 6, 7 は孤立。
        let prob = GraphPartitionProblem::new(graph);
        let cfg = eo_cfg(4, 1.4);
        let (best_ix, rec_ix) = run_eo_impl(&prob, &cfg, 3, 1.4, true);
        let (best_sort, rec_sort) = run_eo_impl(&prob, &cfg, 3, 1.4, false);
        assert_eq!(best_ix, best_sort);
        assert_records_bit_identical(&rec_ix, &rec_sort, "isolated");
    }

    /// 差分更新索引の速度比を実測する。通常のテスト実行では走らせない。
    ///
    /// `cargo test --release -- --ignored --nocapture bench_eo_index_speedup`
    #[test]
    #[ignore = "ベンチマーク（数分かかる）"]
    fn bench_eo_index_speedup() {
        use crate::graph_spec::{GraphKind, StoredGraph};
        use std::time::Instant;

        let tau = 1.3;
        println!("\n{:<26} {:>10} {:>10} {:>8}", "condition", "sort[ms]", "index[ms]", "speedup");
        for (kind, n, d) in [
            (GraphKind::Random, 124usize, 5.0f64),
            (GraphKind::Random, 250, 5.0),
            (GraphKind::Random, 250, 20.0),
            (GraphKind::Random, 500, 5.0),
            (GraphKind::Random, 500, 20.0),
            (GraphKind::Geometric, 500, 20.0),
        ] {
            let prob = StoredGraph::generate(GraphSpec { kind, n, d, seed: 0 }).problem();
            // スワップ版 EO
            let cfg = eo_cfg(6, tau);
            let t0 = Instant::now();
            let _ = run_eo_impl(&prob, &cfg, 0, tau, false);
            let sort_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            let _ = run_eo_impl(&prob, &cfg, 0, tau, true);
            let index_ms = t1.elapsed().as_secs_f64() * 1e3;
            println!(
                "{:<26} {:>10.0} {:>10.0} {:>7.2}x",
                format!("swapEO {kind:?} n{n} d{d}"),
                sort_ms,
                index_ms,
                sort_ms / index_ms
            );

            // フリップ版 EO（MulAlpha α=1 = 元論文の λ=g/deg）
            let mut fcfg = RunConfig::new("bench");
            fcfg.theta = None;
            fcfg.smoothing = SmoothingSpec::None;
            fcfg.log10_iterations = 6;
            fcfg.solver = SolverSpec::EoFlipMulAlpha { tau, alpha: 1.0 };
            let fitness = EoFlipFitnessSpec::MulAlpha { alpha: 1.0 };
            let t0 = Instant::now();
            let _ = run_eo_flip_impl(&prob, &fcfg, 0, tau, fitness, None, false);
            let sort_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            let _ = run_eo_flip_impl(&prob, &fcfg, 0, tau, fitness, None, true);
            let index_ms = t1.elapsed().as_secs_f64() * 1e3;
            println!(
                "{:<26} {:>10.0} {:>10.0} {:>7.2}x",
                format!("flipEO {kind:?} n{n} d{d}"),
                sort_ms,
                index_ms,
                sort_ms / index_ms
            );
        }
    }

    /// α≠1 の MulAlpha は索引経路を使わない（λ が多数派/少数派に依存するため）。
    #[test]
    fn test_eo_flip_non_unit_alpha_uses_sort_path() {
        use crate::graph_spec::{GraphKind, StoredGraph};

        let prob = StoredGraph::generate(GraphSpec {
            kind: GraphKind::Random,
            n: 64,
            d: 5.0,
            seed: 3,
        })
        .problem();
        let tau = 1.3;
        let mut cfg = RunConfig::new("eoflipmulalpha");
        cfg.theta = None;
        cfg.smoothing = SmoothingSpec::None;
        cfg.log10_iterations = 4;
        cfg.solver = SolverSpec::EoFlipMulAlpha { tau, alpha: 0.5 };
        let fitness = EoFlipFitnessSpec::MulAlpha { alpha: 0.5 };
        // 公開経路（run_eo_flip）は α≠1 でソート経路にフォールバックするので、
        // 明示的にソート経路を指定した結果と一致する。
        let (best_auto, rec_auto) = run_eo_flip(&prob, &cfg, 5, tau, fitness, None);
        let (best_sort, rec_sort) =
            run_eo_flip_impl(&prob, &cfg, 5, tau, fitness, None, false);
        assert_eq!(best_auto, best_sort);
        assert_records_bit_identical(&rec_auto, &rec_sort, "alpha=0.5");
    }
}
