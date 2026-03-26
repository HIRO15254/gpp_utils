use crate::optimization::ContinuousRelaxationProblem;
use rand::Rng;
use rand_mt::Mt19937GenRand64;

/// 連続緩和 SA の設定。
#[derive(Debug, Clone)]
pub struct ContinuousRelaxationConfig {
    /// SA 温度（固定）。
    pub temperature: f64,
    /// 最大反復回数。
    pub max_iterations: usize,
    /// 初期 β（小さい = 滑らか）。
    pub beta_init: f64,
    /// 最終 β（大きい = ほぼ離散）。
    pub beta_final: f64,
    /// 離散スコア評価の間隔。
    pub eval_interval: usize,
    /// ログ出力間隔。
    pub log_interval: Option<usize>,
}

impl Default for ContinuousRelaxationConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            max_iterations: 100_000,
            beta_init: 1.0,
            beta_final: 50.0,
            eval_interval: 100,
            log_interval: None,
        }
    }
}

impl ContinuousRelaxationConfig {
    pub fn new(
        temperature: f64,
        max_iterations: usize,
        beta_init: f64,
        beta_final: f64,
    ) -> Self {
        Self {
            temperature,
            max_iterations,
            beta_init,
            beta_final,
            eval_interval: 100,
            log_interval: None,
        }
    }

    pub fn with_eval_interval(mut self, interval: usize) -> Self {
        self.eval_interval = interval;
        self
    }

    pub fn with_logging(mut self, interval: usize) -> Self {
        self.log_interval = Some(interval);
        self
    }
}

/// 連続緩和 SA の実行統計。
#[derive(Debug, Clone)]
pub struct ContinuousRelaxationStats {
    pub iterations_completed: usize,
    pub initial_score: f64,
    pub final_score: f64,
    pub best_score: f64,
    pub final_beta: f64,
    pub accepted_moves: usize,
    pub rejected_moves: usize,
    pub score_history: Vec<(usize, f64)>,
}

/// 連続緩和 SA ソルバー。
///
/// 二値変数を [0, 1] の連続変数に緩和し、連続空間上で SA を実行する。
/// β を徐々に大きくすることで摂動幅を狭め、二値解に収束させる。
///
/// # Example
/// ```
/// use gpp_utils::continuous_relaxation::{ContinuousRelaxationSolver, ContinuousRelaxationConfig};
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
/// let config = ContinuousRelaxationConfig::new(1.0, 5000, 1.0, 50.0);
/// let solver = ContinuousRelaxationSolver::new(config);
///
/// let mut rng = Mt19937GenRand64::new(42);
/// let (partition, stats) = solver.solve(&problem, &mut rng);
/// ```
pub struct ContinuousRelaxationSolver {
    config: ContinuousRelaxationConfig,
}

impl ContinuousRelaxationSolver {
    pub fn new(config: ContinuousRelaxationConfig) -> Self {
        Self { config }
    }

    /// 指定した問題を連続緩和 SA で解く。
    pub fn solve<P>(
        &self,
        problem: &dyn ContinuousRelaxationProblem<P>,
        rng: &mut Mt19937GenRand64,
    ) -> (Vec<bool>, ContinuousRelaxationStats) {
        let n = problem.dimension();
        let t = self.config.temperature;
        let beta_ratio = self.config.beta_final / self.config.beta_init;

        // 連続解をランダムに初期化 [0, 1]
        let mut current: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>()).collect();
        let mut current_cont_score = problem.continuous_score(&current);

        // 離散化して初期真スコアを取得
        let initial_discrete = problem.discretize(&current);
        let initial_score = problem.discrete_score(&initial_discrete);
        let mut best_partition = initial_discrete;
        let mut best_score = initial_score;

        let mut accepted = 0usize;
        let mut rejected = 0usize;

        let mut score_history = Vec::new();
        let record_interval = (self.config.max_iterations / 100).max(1);

        for iteration in 0..self.config.max_iterations {
            // β の指数的増加
            let progress = iteration as f64 / self.config.max_iterations as f64;
            let beta = self.config.beta_init * beta_ratio.powf(progress);

            // 近傍: ランダムな要素を摂動
            let i = rng.gen_range(0..n);
            let perturbation = (rng.r#gen::<f64>() - 0.5) * 2.0 / beta;
            let mut new = current.clone();
            new[i] = (new[i] + perturbation).clamp(0.0, 1.0);

            let new_cont_score = problem.continuous_score(&new);

            // メトロポリス判定（連続スコアで判定）
            let delta = new_cont_score - current_cont_score;
            let accept = if delta < 0.0 {
                true
            } else if t > 0.0 {
                rng.r#gen::<f64>() < (-delta / t).exp()
            } else {
                false
            };

            if accept {
                current = new;
                current_cont_score = new_cont_score;
                accepted += 1;
            } else {
                rejected += 1;
            }

            // 一定間隔で離散スコアを評価して最良解を更新
            if iteration % self.config.eval_interval == 0 || iteration == self.config.max_iterations - 1
            {
                let partition = problem.discretize(&current);
                let score = problem.discrete_score(&partition);
                if score < best_score {
                    best_partition = partition;
                    best_score = score;
                }
            }

            if iteration % record_interval == 0 {
                score_history.push((iteration, best_score));
            }

            if let Some(log_interval) = self.config.log_interval {
                if iteration % log_interval == 0 {
                    println!(
                        "ContinuousRelaxation Iteration {}: beta={:.2}, ContScore={:.6}, Best={:.6}",
                        iteration, beta, current_cont_score, best_score
                    );
                }
            }
        }

        // 最終評価
        let final_partition = problem.discretize(&current);
        let final_score = problem.discrete_score(&final_partition);
        if final_score < best_score {
            best_partition = final_partition;
            best_score = final_score;
        }

        let final_beta = self.config.beta_init * beta_ratio.powf(1.0);

        score_history.push((self.config.max_iterations, best_score));

        let stats = ContinuousRelaxationStats {
            iterations_completed: self.config.max_iterations,
            initial_score,
            final_score,
            best_score,
            final_beta,
            accepted_moves: accepted,
            rejected_moves: rejected,
            score_history,
        };

        (best_partition, stats)
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

        let config = ContinuousRelaxationConfig::new(1.0, 5000, 1.0, 50.0)
            .with_eval_interval(50);
        let solver = ContinuousRelaxationSolver::new(config);

        let mut rng = Mt19937GenRand64::new(42);
        let (partition, stats) = solver.solve(&problem, &mut rng);

        assert_eq!(partition.len(), 4);
        assert_eq!(stats.iterations_completed, 5000);
        assert!(stats.accepted_moves + stats.rejected_moves > 0);
        assert!(stats.best_score.is_finite());
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

        let config = ContinuousRelaxationConfig::new(1.0, 1000, 1.0, 30.0);
        let solver = ContinuousRelaxationSolver::new(config);

        let mut rng1 = Mt19937GenRand64::new(123);
        let (sol1, stats1) = solver.solve(&problem, &mut rng1);

        let mut rng2 = Mt19937GenRand64::new(123);
        let (sol2, stats2) = solver.solve(&problem, &mut rng2);

        assert_eq!(sol1, sol2);
        assert_eq!(stats1.best_score, stats2.best_score);
        assert_eq!(stats1.accepted_moves, stats2.accepted_moves);
    }

    #[test]
    fn test_high_beta_near_discrete() {
        // β が大きい場合、摂動幅が小さく離散的な動きに近い
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);

        let config = ContinuousRelaxationConfig::new(0.5, 2000, 50.0, 100.0)
            .with_eval_interval(10);
        let solver = ContinuousRelaxationSolver::new(config);

        let mut rng = Mt19937GenRand64::new(42);
        let (partition, stats) = solver.solve(&problem, &mut rng);

        assert_eq!(partition.len(), 4);
        assert!(stats.best_score.is_finite());
        assert!((stats.final_beta - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_correctness() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);

        let config = ContinuousRelaxationConfig::new(1.0, 100, 1.0, 20.0)
            .with_eval_interval(10);
        let solver = ContinuousRelaxationSolver::new(config);

        let mut rng = Mt19937GenRand64::new(777);
        let (partition, stats) = solver.solve(&problem, &mut rng);

        assert_eq!(stats.iterations_completed, 100);
        assert!(stats.best_score.is_finite());
        assert!(stats.best_score <= problem.score(&partition) + 1e-10);
        assert!((stats.final_beta - 20.0).abs() < 1e-6);
    }
}
