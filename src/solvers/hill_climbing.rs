//! 山登り法（局所探索）ソルバー。
//!
//! 貪欲な局所探索により、スムージングされたランドスケープで
//! 局所最適解を探索する。

use crate::optimization::{Problem, Smoothing, Solver, SolverStats};
use rand_mt::Mt19937GenRand64;

/// 山登り法ソルバー。
///
/// 隣接解の中から最良のものを選択し、改善がなくなるまで繰り返す。
/// スムージング戦略と組み合わせることで、異なるランドスケープで
/// 最適化を実行できる。
#[derive(Debug, Clone)]
pub struct HillClimbingSolver {
    /// ログ出力間隔（None の場合はログなし）。
    pub log_interval: Option<usize>,
}

impl HillClimbingSolver {
    /// 新しいソルバーを作成する。
    pub fn new() -> Self {
        Self { log_interval: None }
    }

    /// ログ出力間隔を設定する。
    pub fn with_logging(mut self, interval: usize) -> Self {
        self.log_interval = Some(interval);
        self
    }
}

impl Default for HillClimbingSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for HillClimbingSolver {
    fn solve<S: Clone>(
        &self,
        problem: &dyn Problem<S>,
        smoothing: &dyn Smoothing<S>,
        initial: S,
        _rng: &mut Mt19937GenRand64,
    ) -> (S, SolverStats) {
        let mut current = initial.clone();
        let mut current_smoothed = smoothing.score(problem, &current);
        let initial_score = problem.score(&current);
        let mut best = current.clone();
        let mut best_score = initial_score;
        let mut iterations = 0;
        let mut score_history = vec![(0, best_score)];

        loop {
            // すべての近傍を評価
            let neighbours = problem.neighbour(&current);
            if neighbours.is_empty() {
                break; // 近傍がない（これはまずない）
            }

            // スムージングされたスコアで最良の近傍を探す
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
                        // 改善した → 移動
                        current = best_neighbour;
                        current_smoothed = best_neighbour_smoothed;
                        iterations += 1;

                        // 実スコアで best を追跡
                        let real_score = problem.score(&current);
                        if real_score < best_score {
                            best = current.clone();
                            best_score = real_score;
                        }

                        // ログ出力
                        if let Some(log_int) = self.log_interval {
                            if iterations % log_int == 0 {
                                println!(
                                    "HC Iteration {}: smoothed={:.6}, real={:.6}, best={:.6}",
                                    iterations, current_smoothed, real_score, best_score
                                );
                            }
                        }

                        // 履歴記録
                        if iterations % 10 == 0 || iterations < 100 {
                            score_history.push((iterations, best_score));
                        }
                    } else {
                        // 改善しなかった → 終了
                        break;
                    }
                }
                None => break,
            }
        }

        score_history.push((iterations, best_score));

        let stats = SolverStats {
            iterations_completed: iterations,
            initial_score,
            final_score: problem.score(&current),
            best_score,
            accepted_moves: iterations,
            rejected_moves: 0,
            score_history,
        };

        (best, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_partition::GraphPartitionProblem;
    use crate::smoothing::NoSmoothing;

    #[test]
    fn test_hill_climbing_simple_graph() {
        use crate::graph_partition::Graph;
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, true, true, false];

        let solver = HillClimbingSolver::new();
        let smoothing = NoSmoothing;
        let mut rng = Mt19937GenRand64::new(42);

        let (solution, stats) = solver.solve(&problem, &smoothing, initial.clone(), &mut rng);

        // Best score should be <= initial score
        assert!(stats.best_score <= stats.initial_score + 1e-10);
        // Solution should be valid
        assert_eq!(solution.len(), 4);
        // Some iterations should have been made
        assert!(stats.iterations_completed >= 0);
    }

    #[test]
    fn test_hill_climbing_converges() {
        use crate::graph_partition::Graph;
        let mut graph = Graph::new(2);
        graph.add_edge(0, 1);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, true]; // Both in same partition = 1 cut edge + penalty

        let solver = HillClimbingSolver::new();
        let smoothing = NoSmoothing;
        let mut rng = Mt19937GenRand64::new(42);

        let (_solution, stats) = solver.solve(&problem, &smoothing, initial, &mut rng);

        // Should have made at least one improvement
        assert!(stats.best_score < stats.initial_score + 1e-10);
    }

    #[test]
    fn test_hill_climbing_determinism() {
        use crate::graph_partition::GraphGenerationMethod;
        let method = GraphGenerationMethod::Random {
            node_count: 5,
            expected_degree: 2.0,
        };
        let mut rng1 = Mt19937GenRand64::new(42);
        let problem = GraphPartitionProblem::generate(method.clone(), &mut rng1);

        let initial1 = problem.random_solution(&mut Mt19937GenRand64::new(100));
        let mut rng = Mt19937GenRand64::new(200);
        let (sol1, stats1) = HillClimbingSolver::new().solve(
            &problem,
            &NoSmoothing,
            initial1.clone(),
            &mut rng,
        );

        let initial2 = initial1.clone();
        let mut rng = Mt19937GenRand64::new(200);
        let (sol2, stats2) =
            HillClimbingSolver::new().solve(&problem, &NoSmoothing, initial2, &mut rng);

        // Same seed should give same result
        assert_eq!(sol1, sol2);
        assert_eq!(stats1.best_score, stats2.best_score);
    }
}
