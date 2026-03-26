use crate::optimization::ExtremalOptimizationProblem;
use rand::Rng;
use rand_mt::Mt19937GenRand64;

/// τ-EO ソルバーの設定。
#[derive(Debug, Clone)]
pub struct ExtremalOptimizationConfig {
    /// べき乗則指数 τ。`None` の場合は 1 + 1/ln(n) を使用する。
    pub tau: Option<f64>,
    /// 最大反復回数。
    pub max_iterations: usize,
    /// ログ出力間隔。
    pub log_interval: Option<usize>,
}

impl Default for ExtremalOptimizationConfig {
    fn default() -> Self {
        Self {
            tau: None,
            max_iterations: 100_000,
            log_interval: None,
        }
    }
}

impl ExtremalOptimizationConfig {
    pub fn new(tau: Option<f64>, max_iterations: usize) -> Self {
        Self {
            tau,
            max_iterations,
            log_interval: None,
        }
    }

    pub fn with_logging(mut self, interval: usize) -> Self {
        self.log_interval = Some(interval);
        self
    }
}

/// τ-EO ソルバーの実行統計。
#[derive(Debug, Clone)]
pub struct ExtremalOptimizationStats {
    pub iterations_completed: usize,
    pub tau_used: f64,
    pub initial_score: f64,
    pub final_score: f64,
    pub best_score: f64,
    pub score_history: Vec<(usize, f64)>,
}

/// τ-Extremal Optimization ソルバー。
///
/// 解の各構成要素に適応度を割り当て、適応度の低い要素をべき乗則確率で
/// 選択・変更するメタヒューリスティクス。
///
/// # Example
/// ```
/// use gpp_utils::extremal_optimization::{ExtremalOptimizationSolver, ExtremalOptimizationConfig};
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
/// let config = ExtremalOptimizationConfig::new(None, 1000);
/// let solver = ExtremalOptimizationSolver::new(config);
///
/// let initial_solution = vec![true, false, true, false];
/// let mut rng = Mt19937GenRand64::new(42);
/// let (solution, stats) = solver.solve(&problem, initial_solution, &mut rng);
/// ```
pub struct ExtremalOptimizationSolver {
    config: ExtremalOptimizationConfig,
}

impl ExtremalOptimizationSolver {
    pub fn new(config: ExtremalOptimizationConfig) -> Self {
        Self { config }
    }

    /// 指定した問題を τ-EO で解く。
    pub fn solve<P, S: Clone, G>(
        &self,
        problem: &dyn ExtremalOptimizationProblem<P, S, G>,
        initial_solution: S,
        rng: &mut Mt19937GenRand64,
    ) -> (S, ExtremalOptimizationStats) {
        let n = problem.component_count();

        // n <= 1 ではフリップの余地がない
        if n <= 1 {
            let score = problem.score(&initial_solution);
            return (
                initial_solution,
                ExtremalOptimizationStats {
                    iterations_completed: 0,
                    tau_used: 0.0,
                    initial_score: score,
                    final_score: score,
                    best_score: score,
                    score_history: vec![(0, score)],
                },
            );
        }

        let tau = self.config.tau.unwrap_or(1.0 + 1.0 / (n as f64).ln());

        // べき乗則 CDF を事前計算
        let cdf = build_power_law_cdf(n, tau);

        let mut current_solution = initial_solution;
        let mut current_score = problem.score(&current_solution);
        let initial_score = current_score;
        let mut best_solution = current_solution.clone();
        let mut best_score = current_score;

        let mut score_history = Vec::new();
        let record_interval = (self.config.max_iterations / 100).max(1);

        for iteration in 0..self.config.max_iterations {
            // 全構成要素の適応度を計算
            let fitness = problem.component_fitness(&current_solution);

            // 適応度昇順にランク付け（インデックス 0 = 最悪）
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| fitness[a].partial_cmp(&fitness[b]).unwrap());

            // べき乗則分布からランクを選択
            let u: f64 = rng.r#gen::<f64>();
            let rank = match cdf.binary_search_by(|probe| probe.partial_cmp(&u).unwrap()) {
                Ok(pos) => pos,
                Err(pos) => pos.min(n - 1),
            };

            let component_id = indices[rank];

            // 変異を適用（常に受理）
            let (new_solution, new_score) =
                problem.neighbour(component_id, &current_solution, current_score);
            current_solution = new_solution;
            current_score = new_score;

            if current_score < best_score {
                best_solution = current_solution.clone();
                best_score = current_score;
            }

            if iteration % record_interval == 0 {
                score_history.push((iteration, best_score));
            }

            if let Some(log_interval) = self.config.log_interval {
                if iteration % log_interval == 0 {
                    println!(
                        "EO Iteration {}: Current={:.6}, Best={:.6}",
                        iteration, current_score, best_score
                    );
                }
            }
        }

        score_history.push((self.config.max_iterations, best_score));

        let stats = ExtremalOptimizationStats {
            iterations_completed: self.config.max_iterations,
            tau_used: tau,
            initial_score,
            final_score: current_score,
            best_score,
            score_history,
        };

        (best_solution, stats)
    }
}

/// べき乗則 P(k) ∝ k^{-τ} の累積分布関数を構築する（1-indexed ランク）。
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_partition::{Graph, GraphPartitionProblem};
    use crate::optimization::OptimizationProblem;

    #[test]
    fn test_power_law_cdf() {
        let cdf = build_power_law_cdf(10, 1.5);
        assert_eq!(cdf.len(), 10);
        // 単調増加
        for i in 1..cdf.len() {
            assert!(cdf[i] > cdf[i - 1]);
        }
        // 最後は 1.0
        assert!((cdf[9] - 1.0).abs() < 1e-10);
        // τ が大きいほどランク 1 に確率が集中
        let cdf_high_tau = build_power_law_cdf(10, 10.0);
        assert!(cdf_high_tau[0] > 0.99);
    }

    #[test]
    fn test_solver_small_graph() {
        // パスグラフ 0-1-2-3
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, true, false];

        let config = ExtremalOptimizationConfig::new(None, 5000);
        let solver = ExtremalOptimizationSolver::new(config);

        let mut rng = Mt19937GenRand64::new(42);
        let (_solution, stats) = solver.solve(&problem, initial, &mut rng);

        // 最良スコアは初期スコア以下
        assert!(stats.best_score <= stats.initial_score + 1e-10);
        assert_eq!(stats.iterations_completed, 5000);
    }

    #[test]
    fn test_determinism() {
        let mut graph = Graph::new(6);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        graph.add_edge(5, 0);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, true, false, true, false];

        let config = ExtremalOptimizationConfig::new(Some(1.5), 1000);
        let solver = ExtremalOptimizationSolver::new(config);

        let mut rng1 = Mt19937GenRand64::new(123);
        let (sol1, stats1) = solver.solve(&problem, initial.clone(), &mut rng1);

        let mut rng2 = Mt19937GenRand64::new(123);
        let (sol2, stats2) = solver.solve(&problem, initial, &mut rng2);

        assert_eq!(sol1, sol2);
        assert_eq!(stats1.best_score, stats2.best_score);
    }

    #[test]
    fn test_stats_correctness() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, false, true];
        let initial_score = problem.score(&initial);

        let config = ExtremalOptimizationConfig::new(Some(2.0), 100);
        let solver = ExtremalOptimizationSolver::new(config);

        let mut rng = Mt19937GenRand64::new(777);
        let (solution, stats) = solver.solve(&problem, initial, &mut rng);

        assert_eq!(stats.initial_score, initial_score);
        assert_eq!(stats.tau_used, 2.0);
        assert_eq!(stats.iterations_completed, 100);
        assert!(stats.best_score <= stats.initial_score + 1e-10);
        assert!((stats.best_score - problem.score(&solution)).abs() < 1e-10);
    }

    #[test]
    fn test_edge_case_single_node() {
        let graph = Graph::new(1);
        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true];

        let config = ExtremalOptimizationConfig::new(None, 100);
        let solver = ExtremalOptimizationSolver::new(config);

        let mut rng = Mt19937GenRand64::new(42);
        let (solution, stats) = solver.solve(&problem, initial.clone(), &mut rng);

        assert_eq!(solution, initial);
        assert_eq!(stats.iterations_completed, 0);
    }
}
