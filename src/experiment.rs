//! 複数平滑化 × 複数ソルバーの実験フレームワーク。
//!
//! 異なるスムージング戦略とソルバーの組み合わせを体系的に実行し、
//! 結果を集計・分析する。

use crate::optimization::{Problem, Smoothing, SolverStats};
use serde::{Deserialize, Serialize};

/// 実験の結果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRunResult {
    /// 問題の名前。
    pub problem_name: String,
    /// スムージング戦略の名前。
    pub smoothing_name: String,
    /// ソルバーの名前。
    pub solver_name: String,
    /// ソルバーの統計情報。
    pub stats: SolverStats,
}

/// ベイスン評価器。
///
/// 与えられた解から出発して hill climbing により局所最適解に到達し、
/// そのスコアを返す。異なるスムージング戦略での basin の深さを比較するのに使用。
pub struct BasinEvaluator;

impl BasinEvaluator {
    /// 指定したスムージング戦略でのベイスン値を評価する。
    ///
    /// # Arguments
    /// - `problem`: 最適化問題
    /// - `smoothing`: スコア計算戦略
    /// - `solution`: 評価対象の解
    ///
    /// # Returns
    /// basin に到達したときのスコア（実スコア）。
    pub fn evaluate<S: Clone>(
        problem: &dyn Problem<S>,
        smoothing: &dyn Smoothing<S>,
        solution: &S,
    ) -> f64 {
        let mut current = solution.clone();
        let mut current_smoothed = smoothing.score(problem, &current);

        loop {
            let neighbours = problem.neighbour(&current);
            if neighbours.is_empty() {
                break;
            }

            let best_neighbour_opt = neighbours
                .iter()
                .map(|n| {
                    let smoothed = smoothing.score(problem, n);
                    (n.clone(), smoothed)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            match best_neighbour_opt {
                Some((best_neighbour, best_neighbour_smoothed)) => {
                    if best_neighbour_smoothed < current_smoothed {
                        current = best_neighbour;
                        current_smoothed = best_neighbour_smoothed;
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }

        // 実スコアで評価（smoothing ではなく）
        problem.score(&current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_partition::{Graph, GraphPartitionProblem};
    use crate::smoothing::NoSmoothing;

    #[test]
    fn test_basin_evaluator_simple() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let solution = vec![true, false, true, false];

        let basin_score = BasinEvaluator::evaluate(&problem, &NoSmoothing, &solution);

        // Basin score should be finite
        assert!(basin_score.is_finite());
        // Should be <= initial score
        assert!(basin_score <= problem.score(&solution) + 1e-10);
    }

    #[test]
    fn test_basin_evaluator_consistency() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        let problem = GraphPartitionProblem::new(graph);
        let solution = vec![true, false, true];

        // Evaluate same solution twice
        let basin1 = BasinEvaluator::evaluate(&problem, &NoSmoothing, &solution);
        let basin2 = BasinEvaluator::evaluate(&problem, &NoSmoothing, &solution);

        // Should be deterministic
        assert_eq!(basin1, basin2);
    }
}
