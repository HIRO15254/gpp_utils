use crate::optimization::OptimizationProblem;
use rand::Rng;
use rand_mt::Mt19937GenRand64;

/// 近傍平均平滑化 SA の設定。
#[derive(Debug, Clone)]
pub struct SmoothedSAConfig {
    /// SA 温度（固定）。
    pub temperature: f64,
    /// 最大反復回数。
    pub max_iterations: usize,
    /// 初期サンプル数 K（大きいほど平滑）。
    pub k_init: usize,
    /// 最終サンプル数 K（1 で通常の SA と等価）。
    pub k_final: usize,
    /// ログ出力間隔。
    pub log_interval: Option<usize>,
}

impl Default for SmoothedSAConfig {
    fn default() -> Self {
        Self {
            temperature: 10.0,
            max_iterations: 100_000,
            k_init: 50,
            k_final: 1,
            log_interval: None,
        }
    }
}

impl SmoothedSAConfig {
    pub fn new(temperature: f64, max_iterations: usize, k_init: usize, k_final: usize) -> Self {
        Self {
            temperature,
            max_iterations,
            k_init,
            k_final,
            log_interval: None,
        }
    }

    pub fn with_logging(mut self, interval: usize) -> Self {
        self.log_interval = Some(interval);
        self
    }
}

/// 近傍平均平滑化 SA の実行統計。
#[derive(Debug, Clone)]
pub struct SmoothedSAStats {
    pub iterations_completed: usize,
    pub initial_score: f64,
    pub final_score: f64,
    pub best_score: f64,
    pub accepted_moves: usize,
    pub rejected_moves: usize,
    pub score_history: Vec<(usize, f64)>,
}

/// 近傍平均平滑化 SA ソルバー。
///
/// スコアを近傍 K 個の平均で評価し、K を徐々に減衰させることで
/// 序盤は滑らかなランドスケープで大域探索、終盤は元のランドスケープで精密探索する。
///
/// # Example
/// ```
/// use gpp_utils::smoothed_sa::{SmoothedSASolver, SmoothedSAConfig};
/// use gpp_utils::graph_partition::{Graph, GraphPartitionProblem};
/// use gpp_utils::optimization::OptimizationProblem;
/// use rand_mt::Mt19937GenRand64;
///
/// let mut graph = Graph::new(4);
/// graph.add_edge(0, 1);
/// graph.add_edge(1, 2);
/// graph.add_edge(2, 3);
/// let problem = GraphPartitionProblem::new(graph);
///
/// let config = SmoothedSAConfig::new(10.0, 5000, 20, 1);
/// let solver = SmoothedSASolver::new(config);
///
/// let initial = vec![true, false, true, false];
/// let mut rng = Mt19937GenRand64::new(42);
/// let (solution, stats) = solver.solve(&problem, initial, &mut rng);
/// ```
pub struct SmoothedSASolver {
    config: SmoothedSAConfig,
}

impl SmoothedSASolver {
    pub fn new(config: SmoothedSAConfig) -> Self {
        Self { config }
    }

    /// 平滑化スコアを計算する。K 個の近傍のスコアの平均を返す。
    fn smoothed_score<P, S: Clone, G>(
        &self,
        problem: &dyn OptimizationProblem<P, S, G>,
        solution: &S,
        score: f64,
        k: usize,
        rng: &mut Mt19937GenRand64,
    ) -> f64 {
        let mut sum = 0.0;
        for _ in 0..k {
            let (_, neighbour_score) = problem.random_neighbour(solution, score, rng);
            sum += neighbour_score;
        }
        sum / k as f64
    }

    /// 指定した問題を近傍平均平滑化 SA で解く。
    pub fn solve<P, S: Clone, G>(
        &self,
        problem: &dyn OptimizationProblem<P, S, G>,
        initial_solution: S,
        rng: &mut Mt19937GenRand64,
    ) -> (S, SmoothedSAStats) {
        let t = self.config.temperature;

        let mut current_solution = initial_solution;
        let mut current_score = problem.score(&current_solution);
        let initial_score = current_score;
        let mut best_solution = current_solution.clone();
        let mut best_score = current_score;

        let mut accepted = 0usize;
        let mut rejected = 0usize;

        let mut score_history = Vec::new();
        let record_interval = (self.config.max_iterations / 100).max(1);

        // 初期の平滑化スコアを計算
        let k0 = self.config.k_init;
        let mut current_smoothed = self.smoothed_score(problem, &current_solution, current_score, k0, rng);

        for iteration in 0..self.config.max_iterations {
            // K の線形補間（切り上げ）
            let progress = iteration as f64 / self.config.max_iterations as f64;
            let k = (self.config.k_init as f64
                + (self.config.k_final as f64 - self.config.k_init as f64) * progress)
                .ceil() as usize;
            let k = k.max(1);

            // ランダム近傍を生成
            let (neighbour_solution, neighbour_score) =
                problem.random_neighbour(&current_solution, current_score, rng);

            // 近傍の平滑化スコアを計算
            let neighbour_smoothed =
                self.smoothed_score(problem, &neighbour_solution, neighbour_score, k, rng);

            // メトロポリス判定（平滑化スコアで判定）
            let accept = if neighbour_smoothed < current_smoothed {
                true
            } else if t > 0.0 {
                let delta = current_smoothed - neighbour_smoothed;
                let probability = (delta / t).exp();
                rng.r#gen::<f64>() < probability
            } else {
                false
            };

            if accept {
                current_solution = neighbour_solution;
                current_score = neighbour_score;
                current_smoothed = neighbour_smoothed;
                accepted += 1;

                // 真のスコアで最良解を追跡
                if current_score < best_score {
                    best_solution = current_solution.clone();
                    best_score = current_score;
                }
            } else {
                rejected += 1;
            }

            if iteration % record_interval == 0 {
                score_history.push((iteration, best_score));
            }

            if let Some(log_interval) = self.config.log_interval {
                if iteration % log_interval == 0 {
                    println!(
                        "SmoothedSA Iteration {}: K={}, Smoothed={:.6}, Real={:.6}, Best={:.6}",
                        iteration, k, current_smoothed, current_score, best_score
                    );
                }
            }
        }

        score_history.push((self.config.max_iterations, best_score));

        let stats = SmoothedSAStats {
            iterations_completed: self.config.max_iterations,
            initial_score,
            final_score: current_score,
            best_score,
            accepted_moves: accepted,
            rejected_moves: rejected,
            score_history,
        };

        (best_solution, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_partition::{Graph, GraphPartitionProblem};
    use crate::optimization::OptimizationProblem;

    #[test]
    fn test_solver_small_graph() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, true, false];

        let config = SmoothedSAConfig::new(10.0, 2000, 10, 1);
        let solver = SmoothedSASolver::new(config);

        let mut rng = Mt19937GenRand64::new(42);
        let (_solution, stats) = solver.solve(&problem, initial, &mut rng);

        assert!(stats.best_score <= stats.initial_score + 1e-10);
        assert_eq!(stats.iterations_completed, 2000);
        assert!(stats.accepted_moves + stats.rejected_moves > 0);
    }

    #[test]
    fn test_determinism() {
        let mut graph = Graph::new(6);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, true, false, true, false];

        let config = SmoothedSAConfig::new(5.0, 500, 10, 1);
        let solver = SmoothedSASolver::new(config);

        let mut rng1 = Mt19937GenRand64::new(123);
        let (sol1, stats1) = solver.solve(&problem, initial.clone(), &mut rng1);

        let mut rng2 = Mt19937GenRand64::new(123);
        let (sol2, stats2) = solver.solve(&problem, initial, &mut rng2);

        assert_eq!(sol1, sol2);
        assert_eq!(stats1.best_score, stats2.best_score);
    }

    #[test]
    fn test_k1_runs() {
        // K=1 で通常の SA に近い動作
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, false, true];

        let config = SmoothedSAConfig::new(5.0, 500, 1, 1);
        let solver = SmoothedSASolver::new(config);

        let mut rng = Mt19937GenRand64::new(42);
        let (_solution, stats) = solver.solve(&problem, initial, &mut rng);

        assert_eq!(stats.iterations_completed, 500);
        assert!(stats.accepted_moves + stats.rejected_moves > 0);
    }

    #[test]
    fn test_stats_correctness() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, false, true];
        let initial_score = problem.score(&initial);

        let config = SmoothedSAConfig::new(5.0, 100, 5, 1);
        let solver = SmoothedSASolver::new(config);

        let mut rng = Mt19937GenRand64::new(777);
        let (solution, stats) = solver.solve(&problem, initial, &mut rng);

        assert_eq!(stats.initial_score, initial_score);
        assert_eq!(stats.iterations_completed, 100);
        assert!(stats.best_score <= stats.initial_score + 1e-10);
        assert!((stats.best_score - problem.score(&solution)).abs() < 1e-10);
    }
}
