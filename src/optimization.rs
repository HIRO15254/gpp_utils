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
    /// - `seed`: 乱数生成用シード。ソルバーは別途平滑化用シード（seed ^ 0xAAAAAAAAAAAAAAAA）を使用する。
    fn solve<S: Clone>(
        &self,
        problem: &dyn Problem<S>,
        smoothing: &dyn Smoothing<S>,
        initial: S,
        seed: u64,
    ) -> (S, SolverStats);
}

// ============================================================================
// Smoothing の基本実装
// ============================================================================

/// スムージングなし（元のスコアをそのまま使用）。
#[derive(Debug, Clone)]
pub struct NoSmoothing;

impl<S: Clone> Smoothing<S> for NoSmoothing {
    fn score(&self, problem: &dyn Problem<S>, solution: &S) -> f64 {
        problem.score(solution)
    }
}

/// K-近傍平均によるスムージング。
///
/// ランダムに選んだ K 個の近傍のスコアを平均して、
/// 平滑化されたスコアを計算する。
#[derive(Debug, Clone)]
pub struct KAveragingSmoothing {
    /// サンプリングする近傍数。
    pub k: usize,
}

impl KAveragingSmoothing {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl<S: Clone> Smoothing<S> for KAveragingSmoothing {
    fn score(&self, problem: &dyn Problem<S>, solution: &S) -> f64 {
        let neighbours = problem.neighbour(solution);
        if neighbours.is_empty() {
            return problem.score(solution);
        }

        let sample_count = self.k.min(neighbours.len());
        let sum: f64 = neighbours
            .iter()
            .take(sample_count)
            .map(|n| problem.score(n))
            .sum();
        sum / sample_count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct DummyProblem;

    impl Problem<i32> for DummyProblem {
        fn score(&self, solution: &i32) -> f64 {
            (solution * solution) as f64
        }

        fn neighbour(&self, solution: &i32) -> Vec<i32> {
            vec![solution - 1, *solution, solution + 1]
        }

        fn random_solution(&self, _rng: &mut Mt19937GenRand64) -> i32 {
            0
        }

        fn neighbour_size(&self) -> usize {
            3
        }
    }

    #[test]
    fn test_no_smoothing_equals_real_score() {
        let problem = DummyProblem;
        let smoothing = NoSmoothing;
        let solution = 5i32;

        assert_eq!(
            smoothing.score(&problem, &solution),
            problem.score(&solution)
        );
    }

    #[test]
    fn test_k_averaging_smoothing() {
        let problem = DummyProblem;
        let smoothing = KAveragingSmoothing::new(2);
        let solution = 5i32;

        // neighbours = [4, 5, 6]
        // scores = [16, 25, 36]
        // average of first 2: (16 + 25) / 2 = 20.5
        let score = smoothing.score(&problem, &solution);
        assert!((score - 20.5).abs() < 1e-10);
    }

}
