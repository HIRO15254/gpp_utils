//! 焼きなまし法（Simulated Annealing）ソルバー。
//!
//! スムージング戦略と組み合わせて、平滑化されたランドスケープ上で
//! メトロポリス法による確率的探索を行う。

use crate::optimization::{Problem, Smoothing, Solver, SolverStats};
use rand::Rng;
use rand_mt::Mt19937GenRand64;

/// 焼きなまし法ソルバー。
///
/// ランダムに選んだ近傍のスムージングスコアを評価し、メトロポリス基準で
/// 受理/棄却を決定する。温度 T が高いほど悪化を許容しやすい。
#[derive(Debug, Clone)]
pub struct SimulatedAnnealingSolver {
    /// 固定温度（Metropolis 判定に使用）。
    pub temperature: f64,
    /// 最大反復回数。
    pub max_iterations: usize,
    /// ログ出力間隔（None の場合はログなし）。
    pub log_interval: Option<usize>,
}

impl SimulatedAnnealingSolver {
    /// 新しいソルバーを作成する。
    pub fn new(temperature: f64, max_iterations: usize) -> Self {
        Self { temperature, max_iterations, log_interval: None }
    }

    /// ログ出力間隔を設定する。
    pub fn with_logging(mut self, interval: usize) -> Self {
        self.log_interval = Some(interval);
        self
    }
}

impl Default for SimulatedAnnealingSolver {
    fn default() -> Self {
        Self::new(10.0, 50_000)
    }
}

impl Solver for SimulatedAnnealingSolver {
    fn solve<S: Clone>(
        &self,
        problem: &dyn Problem<S>,
        smoothing: &dyn Smoothing<S>,
        initial: S,
        seed: u64,
    ) -> (S, SolverStats) {
        let mut rng = Mt19937GenRand64::new(seed);
        let mut _smoothing_rng = Mt19937GenRand64::new(seed ^ 0xAAAAAAAAAAAAAAAA);

        let mut current = initial.clone();
        let mut current_smoothed = smoothing.score(problem, &current);
        let initial_score = problem.score(&current);
        let mut best = current.clone();
        let mut best_score = initial_score;
        let mut best_smoothed = current_smoothed;

        let mut accepted = 0usize;
        let mut rejected = 0usize;
        let mut score_history = vec![(0, best_score)];
        let mut smoothed_score_history = vec![(0, best_smoothed)];
        let record_interval = (self.max_iterations / 100).max(1);

        for iteration in 0..self.max_iterations {
            let neighbours = problem.neighbour(&current);
            if neighbours.is_empty() {
                break;
            }

            let idx = rng.gen_range(0..neighbours.len());
            let neighbour_smoothed = smoothing.score(problem, &neighbours[idx]);

            let delta = neighbour_smoothed - current_smoothed;
            let accept = if delta < 0.0 {
                true
            } else if self.temperature > 0.0 {
                rng.r#gen::<f64>() < (-delta / self.temperature).exp()
            } else {
                false
            };

            if accept {
                current = neighbours[idx].clone();
                current_smoothed = neighbour_smoothed;
                accepted += 1;

                let real_score = problem.score(&current);
                if real_score < best_score {
                    best = current.clone();
                    best_score = real_score;
                    best_smoothed = current_smoothed;
                }
            } else {
                rejected += 1;
            }

            if iteration % record_interval == 0 {
                score_history.push((iteration, best_score));
                smoothed_score_history.push((iteration, best_smoothed));
            }

            if let Some(log_int) = self.log_interval {
                if iteration % log_int == 0 {
                    println!(
                        "SA Iteration {}: T={:.4}, smoothed={:.6}, best={:.6}",
                        iteration, self.temperature, current_smoothed, best_score
                    );
                }
            }
        }

        score_history.push((self.max_iterations, best_score));
        smoothed_score_history.push((self.max_iterations, best_smoothed));

        let stats = SolverStats {
            iterations_completed: self.max_iterations,
            initial_score,
            final_score: problem.score(&current),
            best_score,
            accepted_moves: accepted,
            rejected_moves: rejected,
            score_history,
            smoothed_score_history,
        };

        (best, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_partition::GraphPartitionProblem;
    use crate::smoothing::NoSmoothing;
    use crate::graph_partition::Graph;

    #[test]
    fn test_sa_simple_graph() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, true, false, false];
        let solver = SimulatedAnnealingSolver::new(5.0, 1_000);
        let smoothing = NoSmoothing;

        let (_sol, stats) = solver.solve(&problem, &smoothing, initial.clone(), 42);

        assert!(stats.best_score <= stats.initial_score + 1e-10);
        assert_eq!(stats.iterations_completed, 1_000);
        assert!(stats.accepted_moves + stats.rejected_moves > 0);
        assert!(!stats.smoothed_score_history.is_empty());
    }

    #[test]
    fn test_sa_determinism() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, false, true];
        let solver = SimulatedAnnealingSolver::new(3.0, 500);
        let smoothing = NoSmoothing;

        let (sol1, stats1) = solver.solve(&problem, &smoothing, initial.clone(), 99);
        let (sol2, stats2) = solver.solve(&problem, &smoothing, initial.clone(), 99);

        assert_eq!(sol1, sol2);
        assert_eq!(stats1.best_score, stats2.best_score);
    }
}
