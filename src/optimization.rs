//! 解最適化フレームワーク
//!
//! 問題、スコア計算、探索戦略を独立したトレイトで定義し、
//! 任意の組み合わせで最適化実験を実行できる構造。

use rand_mt::Mt19937GenRand64;
use serde::{Deserialize, Serialize};

/// 最適化問題の定義。
///
/// 解の基本操作（スコア計算、近傍生成、初期化）のみを定義。
/// スコア計算方法は [`Smoothing`] トレイトで差し替え可能。
pub trait Problem<S: Clone>: Send + Sync {
    /// 解のスコアを計算する（実問題のスコア）。
    fn score(&self, solution: &S) -> f64;

    /// 解の全近傍を生成する。
    fn neighbour(&self, solution: &S) -> Vec<S>;

    /// ランダムな解を生成する。
    fn random_solution(&self, rng: &mut Mt19937GenRand64) -> S;

    /// 近傍サイズ（最適化用、デフォルトは全近傍の長さ）。
    fn neighbour_size(&self) -> usize {
        // デフォルト実装は呼び出せないため、実装側で override することを推奨
        usize::MAX
    }

    /// 移動 `move_idx` を適用した近傍のスコアを返す。
    ///
    /// デフォルト実装は `neighbour(s)[move_idx]` を score して返すため、
    /// 全近傍を作成するコストがかかる。具象型でオーバーライドすることで
    /// クローン爆発を解消できる。
    fn score_at_move(&self, solution: &S, move_idx: usize) -> f64 {
        let n = self.neighbour(solution);
        self.score(&n[move_idx])
    }

    /// 移動 `move_idx` を `solution` に in-place 適用する。
    ///
    /// デフォルト実装は `neighbour(s)[move_idx]` で置き換える。
    fn apply_move(&self, solution: &mut S, move_idx: usize) {
        let n = self.neighbour(solution);
        *solution = n.into_iter().nth(move_idx).expect("invalid move_idx");
    }
}

/// スコア計算方法の差し替え層。
///
/// 同じ問題に対して異なるスコア評価方法を提供する。
/// 例えば、実スコア、K-近傍平均、連続緩和など。
pub trait Smoothing<S: Clone>: Send + Sync {
    /// 問題のスコアを評価（平滑化）。
    fn score(&self, problem: &dyn Problem<S>, solution: &S) -> f64;
}

/// ソルバーの実行統計。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverStats {
    /// 完了した反復回数（ステップ数）。
    pub iterations_completed: usize,
    /// 初期解のスコア。
    pub initial_score: f64,
    /// 最終解のスコア。
    pub final_score: f64,
    /// 探索中に見つけた最良スコア。
    pub best_score: f64,
    /// 受け入れられた移動回数。
    pub accepted_moves: usize,
    /// 拒否された移動回数。
    pub rejected_moves: usize,
    /// スコア履歴 [(反復, 最良実スコア)]。
    pub score_history: Vec<(usize, f64)>,
    /// 平滑化スコア履歴 [(反復, 最良平滑化スコア)]（平滑化なしの場合は score_history と同じ）。
    pub smoothed_score_history: Vec<(usize, f64)>,
}

/// 探索戦略（ソルバー）。
///
/// 任意の [`Problem`] と [`Smoothing`] の組み合わせで最適化を実行する。
pub trait Solver: Send + Sync {
    /// 最適化を実行する。
    ///
    /// # Arguments
    /// - `problem`: 最適化問題
    /// - `smoothing`: スコア計算方法
    /// - `initial`: 初期解
    /// - `seed`: 乱数生成用シード。
    fn solve<S: Clone>(
        &self,
        problem: &dyn Problem<S>,
        smoothing: &dyn Smoothing<S>,
        initial: S,
        seed: u64,
    ) -> (S, SolverStats);
}
