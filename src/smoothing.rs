//! スコア計算方法の差し替え実装。
//!
//! 同じ問題に対して異なるスコア評価方法（平滑化戦略）を提供する。

use crate::optimization::{Problem, Smoothing};
use rand::Rng;
use rand_mt::Mt19937GenRand64;

/// スムージングなし（元のスコアをそのまま使用）。
///
/// これは optimization.rs の NoSmoothing と同じ実装ですが、
/// 別モジュールで再度定義することで、独立した使用が可能です。
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

/// 決定論的な全近傍平均スムージング。
///
/// ランダムサンプリングではなく、すべての近傍のスコアを平均する。
/// 計算コストは高いが、より安定した評価が得られる。
#[derive(Debug, Clone)]
pub struct AllNeighbourAveragingSmoothing;

impl<S: Clone> Smoothing<S> for AllNeighbourAveragingSmoothing {
    fn score(&self, problem: &dyn Problem<S>, solution: &S) -> f64 {
        let neighbours = problem.neighbour(solution);
        if neighbours.is_empty() {
            return problem.score(solution);
        }

        let sum: f64 = neighbours.iter().map(|n| problem.score(n)).sum();
        sum / neighbours.len() as f64
    }
}

/// 複数のスムージングレベルを順序付きで提供する。
///
/// space_id に応じて異なる K 値を使用した K-近傍平均を提供し、
/// 滑らかな空間から元の空間への段階的な移行を実現する。
#[derive(Debug, Clone)]
pub struct MultiLevelSmoothing {
    /// スムージング空間の数。
    space_count: usize,
    /// 最も平滑化された空間での最大 K 値。
    max_k: usize,
}

impl MultiLevelSmoothing {
    /// 新しい multi-level smoothing を作成する。
    ///
    /// # Arguments
    /// - `space_count`: スムージング空間の数（2 以上推奨）
    /// - `max_k`: 空間 0 でのサンプリング数（大きいほど平滑）
    pub fn new(space_count: usize, max_k: usize) -> Self {
        Self { space_count, max_k }
    }

    /// 空間インデックスに対応する K 値を計算する。
    fn k_for_space(&self, space_id: usize) -> usize {
        if self.space_count <= 1 || space_id >= self.space_count - 1 {
            return 1;
        }
        let t = space_id as f64 / (self.space_count - 1) as f64;
        let k = self.max_k as f64 * (1.0 - t) + 1.0 * t;
        k.round().max(1.0) as usize
    }

    /// 指定した空間でのスムージングを実行する。
    pub fn score_in_space<S: Clone>(
        &self,
        space_id: usize,
        problem: &dyn Problem<S>,
        solution: &S,
    ) -> f64 {
        let k = self.k_for_space(space_id);
        if k <= 1 {
            return problem.score(solution);
        }

        let neighbours = problem.neighbour(solution);
        if neighbours.is_empty() {
            return problem.score(solution);
        }

        let sample_count = k.min(neighbours.len());
        let sum: f64 = neighbours
            .iter()
            .take(sample_count)
            .map(|n| problem.score(n))
            .sum();
        sum / sample_count as f64
    }

    /// スムージング空間の数を返す。
    pub fn space_count(&self) -> usize {
        self.space_count
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
    fn test_no_smoothing() {
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

    #[test]
    fn test_all_neighbour_averaging() {
        let problem = DummyProblem;
        let smoothing = AllNeighbourAveragingSmoothing;
        let solution = 5i32;

        // neighbours = [4, 5, 6]
        // scores = [16, 25, 36]
        // average: (16 + 25 + 36) / 3 = 77 / 3 ≈ 25.667
        let score = smoothing.score(&problem, &solution);
        assert!((score - (77.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_multi_level_smoothing_k_values() {
        let multi = MultiLevelSmoothing::new(5, 50);
        // space_id=0: k = 50 * (1 - 0) + 1 * 0 = 50
        assert_eq!(multi.k_for_space(0), 50);
        // space_id=1: t = 1/4 = 0.25; k = 50 * 0.75 + 1 * 0.25 = 37.75 → 38
        assert_eq!(multi.k_for_space(1), 38);
        // space_id=4: space_id >= space_count - 1, so k = 1
        assert_eq!(multi.k_for_space(4), 1);
    }

    #[test]
    fn test_multi_level_smoothing_space_count() {
        let multi = MultiLevelSmoothing::new(10, 100);
        assert_eq!(multi.space_count(), 10);
    }

    #[test]
    fn test_multi_level_smoothing_score_in_space() {
        let problem = DummyProblem;
        let multi = MultiLevelSmoothing::new(5, 50);
        let solution = 5i32;

        // space_id=0: K=50, but only 3 neighbours available
        // score = (16 + 25 + 36) / 3 ≈ 25.667
        let score0 = multi.score_in_space(0, &problem, &solution);
        assert!((score0 - (77.0 / 3.0)).abs() < 1e-10);

        // space_id=4: K=1 (original)
        // score = 25.0
        let score4 = multi.score_in_space(4, &problem, &solution);
        assert!((score4 - 25.0).abs() < 1e-10);
    }
}
