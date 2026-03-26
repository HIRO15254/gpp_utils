use crate::optimization::ReplicaCouplingProblem;
use rand::Rng;
use rand_mt::Mt19937GenRand64;

/// SQA ソルバーの設定。
#[derive(Debug, Clone)]
pub struct SQAConfig {
    /// レプリカ数 P。
    pub num_replicas: usize,
    /// 温度 T（固定）。
    pub temperature: f64,
    /// 横磁場の初期値 Γ_init。
    pub gamma_init: f64,
    /// 横磁場の最終値 Γ_final。
    pub gamma_final: f64,
    /// モンテカルロステップ数。1ステップ = P × n 回のフリップ試行。
    pub max_steps: usize,
    /// ログ出力間隔。
    pub log_interval: Option<usize>,
}

impl Default for SQAConfig {
    fn default() -> Self {
        Self {
            num_replicas: 16,
            temperature: 0.1,
            gamma_init: 5.0,
            gamma_final: 0.01,
            max_steps: 1000,
            log_interval: None,
        }
    }
}

impl SQAConfig {
    pub fn new(
        num_replicas: usize,
        temperature: f64,
        gamma_init: f64,
        gamma_final: f64,
        max_steps: usize,
    ) -> Self {
        Self {
            num_replicas,
            temperature,
            gamma_init,
            gamma_final,
            max_steps,
            log_interval: None,
        }
    }

    pub fn with_logging(mut self, interval: usize) -> Self {
        self.log_interval = Some(interval);
        self
    }
}

/// SQA ソルバーの実行統計。
#[derive(Debug, Clone)]
pub struct SQAStats {
    pub steps_completed: usize,
    pub initial_score: f64,
    pub final_score: f64,
    pub best_score: f64,
    pub final_gamma: f64,
    pub accepted_moves: usize,
    pub rejected_moves: usize,
    pub score_history: Vec<(usize, f64)>,
}

/// Simulated Quantum Annealing ソルバー。
///
/// 鈴木-トロッター分解に基づき、P 枚のレプリカを用いて
/// 量子トンネル効果を古典的に模擬するメタヒューリスティクス。
///
/// # Example
/// ```
/// use gpp_utils::quantum_annealing::{SQASolver, SQAConfig};
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
/// let config = SQAConfig::new(8, 0.1, 5.0, 0.01, 500);
/// let solver = SQASolver::new(config);
///
/// let initial_solution = vec![true, false, true, false];
/// let mut rng = Mt19937GenRand64::new(42);
/// let (solution, stats) = solver.solve(&problem, initial_solution, &mut rng);
/// ```
pub struct SQASolver {
    config: SQAConfig,
}

impl SQASolver {
    pub fn new(config: SQAConfig) -> Self {
        Self { config }
    }

    /// 指定した問題を SQA で解く。
    pub fn solve<P, S: Clone, G>(
        &self,
        problem: &dyn ReplicaCouplingProblem<P, S, G>,
        initial_solution: S,
        rng: &mut Mt19937GenRand64,
    ) -> (S, SQAStats) {
        let p = self.config.num_replicas;
        let n = problem.neighbour_size();
        let t = self.config.temperature;

        // レプリカの初期化
        let mut replicas: Vec<S> = Vec::with_capacity(p);
        let mut scores: Vec<f64> = Vec::with_capacity(p);

        let initial_score = problem.score(&initial_solution);
        let mut best_solution = initial_solution.clone();
        let mut best_score = initial_score;

        for _ in 0..p {
            let sol = problem.create_random_solution(rng);
            let score = problem.score(&sol);
            if score < best_score {
                best_solution = sol.clone();
                best_score = score;
            }
            replicas.push(sol);
            scores.push(score);
        }

        let mut accepted = 0usize;
        let mut rejected = 0usize;
        let mut current_gamma = self.config.gamma_init;

        let gamma_ratio = self.config.gamma_final / self.config.gamma_init;

        let mut score_history = Vec::new();
        let record_interval = (self.config.max_steps / 100).max(1);

        for step in 0..self.config.max_steps {
            // Γ の指数的減衰
            let progress = step as f64 / self.config.max_steps as f64;
            current_gamma = self.config.gamma_init * gamma_ratio.powf(progress);

            // レプリカ間結合強度 J⊥
            // J⊥ = -(PT/2) × ln(tanh(Γ/(PT)))
            let pt = p as f64 * t;
            let j_perp = if pt > 0.0 && current_gamma > 0.0 {
                let arg = current_gamma / pt;
                -(pt / 2.0) * arg.tanh().ln()
            } else {
                f64::INFINITY // Γ=0 → レプリカ完全結合
            };

            // 1ステップ = P × n 回のフリップ試行
            for _ in 0..(p * n) {
                let k = rng.gen_range(0..p);
                let i = rng.gen_range(0..n);

                // 問題由来のエネルギー変化（Δ評価）
                let (new_sol, new_score) = problem.neighbour(i, &replicas[k], scores[k]);
                let delta_problem = (new_score - scores[k]) / p as f64;

                // レプリカ結合エネルギーの変化
                let prev = if k == 0 { p - 1 } else { k - 1 };
                let next = if k == p - 1 { 0 } else { k + 1 };

                // フリップ前: 隣接レプリカとの一致数
                let equal_before_prev =
                    problem.components_equal(&replicas[k], &replicas[prev], i);
                let equal_before_next =
                    problem.components_equal(&replicas[k], &replicas[next], i);

                // フリップ後: 一致が反転する（同→異、異→同）
                // 一致 → -J⊥（結合エネルギー低い）、不一致 → +J⊥（結合エネルギー高い）
                // ΔE_coupling = Σ (フリップ後の結合E - フリップ前の結合E)
                let mut delta_coupling = 0.0;
                if equal_before_prev {
                    delta_coupling += 2.0 * j_perp; // 一致→不一致: +J⊥ - (-J⊥) = 2J⊥
                } else {
                    delta_coupling -= 2.0 * j_perp; // 不一致→一致: -J⊥ - (+J⊥) = -2J⊥
                }
                if equal_before_next {
                    delta_coupling += 2.0 * j_perp;
                } else {
                    delta_coupling -= 2.0 * j_perp;
                }

                let delta_total = delta_problem + delta_coupling;

                // メトロポリス判定
                let accept = if delta_total < 0.0 {
                    true
                } else if t > 0.0 {
                    rng.r#gen::<f64>() < (-delta_total / t).exp()
                } else {
                    false
                };

                if accept {
                    replicas[k] = new_sol;
                    scores[k] = new_score;
                    accepted += 1;

                    if new_score < best_score {
                        best_solution = replicas[k].clone();
                        best_score = new_score;
                    }
                } else {
                    rejected += 1;
                }
            }

            if step % record_interval == 0 {
                score_history.push((step, best_score));
            }

            if let Some(log_interval) = self.config.log_interval {
                if step % log_interval == 0 {
                    let avg_score = scores.iter().sum::<f64>() / p as f64;
                    println!(
                        "SQA Step {}: Gamma={:.6}, AvgScore={:.6}, Best={:.6}",
                        step, current_gamma, avg_score, best_score
                    );
                }
            }
        }

        score_history.push((self.config.max_steps, best_score));

        // 最終 Γ
        let final_gamma =
            self.config.gamma_init * gamma_ratio.powf(1.0);

        let final_score = scores
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        let stats = SQAStats {
            steps_completed: self.config.max_steps,
            initial_score,
            final_score,
            best_score,
            final_gamma,
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

        let config = SQAConfig::new(8, 0.1, 5.0, 0.01, 200);
        let solver = SQASolver::new(config);

        let mut rng = Mt19937GenRand64::new(42);
        let (_solution, stats) = solver.solve(&problem, initial, &mut rng);

        assert!(stats.best_score <= stats.initial_score + 1e-10);
        assert_eq!(stats.steps_completed, 200);
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
        graph.add_edge(5, 0);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, true, false, true, false];

        let config = SQAConfig::new(4, 0.1, 3.0, 0.01, 100);
        let solver = SQASolver::new(config);

        let mut rng1 = Mt19937GenRand64::new(123);
        let (sol1, stats1) = solver.solve(&problem, initial.clone(), &mut rng1);

        let mut rng2 = Mt19937GenRand64::new(123);
        let (sol2, stats2) = solver.solve(&problem, initial, &mut rng2);

        assert_eq!(sol1, sol2);
        assert_eq!(stats1.best_score, stats2.best_score);
        assert_eq!(stats1.accepted_moves, stats2.accepted_moves);
    }

    #[test]
    fn test_stats_correctness() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, false, true];
        let initial_score = problem.score(&initial);

        let config = SQAConfig::new(4, 0.1, 5.0, 0.01, 50);
        let solver = SQASolver::new(config);

        let mut rng = Mt19937GenRand64::new(777);
        let (solution, stats) = solver.solve(&problem, initial, &mut rng);

        assert_eq!(stats.initial_score, initial_score);
        assert_eq!(stats.steps_completed, 50);
        assert!(stats.best_score <= stats.initial_score + 1e-10);
        assert!((stats.best_score - problem.score(&solution)).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_decay() {
        let config = SQAConfig::new(4, 0.1, 5.0, 0.01, 100);

        let mut graph = Graph::new(2);
        graph.add_edge(0, 1);
        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false];

        let solver = SQASolver::new(config);
        let mut rng = Mt19937GenRand64::new(42);
        let (_solution, stats) = solver.solve(&problem, initial, &mut rng);

        // final_gamma は gamma_final に近い値になるはず
        assert!((stats.final_gamma - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_high_gamma_explores() {
        // Γ が大きい場合はレプリカがバラバラに動くので受理率が高い
        let config_high = SQAConfig::new(4, 0.1, 100.0, 50.0, 50);
        let config_low = SQAConfig::new(4, 0.1, 0.001, 0.0001, 50);

        let mut graph = Graph::new(6);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, true, true, false, false, false];

        let solver_high = SQASolver::new(config_high);
        let mut rng = Mt19937GenRand64::new(42);
        let (_, stats_high) = solver_high.solve(&problem, initial.clone(), &mut rng);

        let solver_low = SQASolver::new(config_low);
        let mut rng = Mt19937GenRand64::new(42);
        let (_, stats_low) = solver_low.solve(&problem, initial, &mut rng);

        // 高 Γ の方が受理率が高い（レプリカ結合が弱いため）
        let rate_high =
            stats_high.accepted_moves as f64 / (stats_high.accepted_moves + stats_high.rejected_moves) as f64;
        let rate_low =
            stats_low.accepted_moves as f64 / (stats_low.accepted_moves + stats_low.rejected_moves) as f64;
        assert!(
            rate_high > rate_low,
            "High Gamma acceptance rate ({:.4}) should be > Low Gamma ({:.4})",
            rate_high,
            rate_low
        );
    }
}
