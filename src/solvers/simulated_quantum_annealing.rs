//! Simulated Quantum Annealing（SQA）ソルバー。
//!
//! 鈴木-トロッター分解に基づき P 枚のレプリカを用いて
//! 量子トンネル効果を古典的に模擬するメタヒューリスティクス。
//!
//! このソルバーはレプリカ間結合のために `Vec<bool>` 型の解を直接操作するため、
//! 汎用 `Solver` トレイトではなく専用の `solve()` メソッドを提供する。

use crate::optimization::{Problem, SolverStats};
use rand::Rng;
use rand_mt::Mt19937GenRand64;

/// SQA ソルバー。
#[derive(Debug, Clone)]
pub struct SimulatedQuantumAnnealingSolver {
    /// レプリカ数 P。
    pub num_replicas: usize,
    /// 熱的温度 T（固定）。
    pub temperature: f64,
    /// 横磁場の初期値 Γ_init。
    pub gamma_init: f64,
    /// 横磁場の最終値 Γ_final。
    pub gamma_final: f64,
    /// モンテカルロステップ数（1 ステップ = P × n 回のフリップ試行）。
    pub max_steps: usize,
    /// ログ出力間隔（None の場合はログなし）。
    pub log_interval: Option<usize>,
}

impl SimulatedQuantumAnnealingSolver {
    /// 新しいソルバーを作成する。
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

    /// ログ出力間隔を設定する。
    pub fn with_logging(mut self, interval: usize) -> Self {
        self.log_interval = Some(interval);
        self
    }

    /// `Vec<bool>` 型の問題を SQA で解く。
    ///
    /// レプリカは `problem.random_solution()` で初期化される。
    /// 横磁場 Γ は `gamma_init` から `gamma_final` へ指数的に減衰する。
    pub fn solve(
        &self,
        problem: &dyn Problem<Vec<bool>>,
        initial: Vec<bool>,
        rng: &mut Mt19937GenRand64,
    ) -> (Vec<bool>, SolverStats) {
        let p = self.num_replicas;
        let n = initial.len();
        let t = self.temperature;

        let initial_score = problem.score(&initial);
        let mut best = initial.clone();
        let mut best_score = initial_score;

        // レプリカをランダム初期化
        let mut replicas: Vec<Vec<bool>> = (0..p)
            .map(|_| problem.random_solution(rng))
            .collect();
        let mut scores: Vec<f64> = replicas.iter().map(|r| problem.score(r)).collect();

        for (s, r) in scores.iter().zip(replicas.iter()) {
            if *s < best_score {
                best_score = *s;
                best = r.clone();
            }
        }

        let gamma_ratio = if self.gamma_init > 0.0 {
            self.gamma_final / self.gamma_init
        } else {
            1.0
        };

        let mut accepted = 0usize;
        let mut rejected = 0usize;
        let mut score_history = vec![(0, best_score)];
        let record_interval = (self.max_steps / 100).max(1);

        for step in 0..self.max_steps {
            let progress = step as f64 / self.max_steps as f64;
            let gamma = self.gamma_init * gamma_ratio.powf(progress);

            // レプリカ間結合強度 J⊥ = -(PT/2) × ln(tanh(Γ/(PT)))
            let pt = p as f64 * t;
            let j_perp = if pt > 0.0 && gamma > 0.0 {
                let arg = gamma / pt;
                -(pt / 2.0) * arg.tanh().ln()
            } else {
                f64::INFINITY
            };

            // 1ステップ = P × n 回のフリップ試行
            for _ in 0..(p * n) {
                let k = rng.gen_range(0..p);
                let i = rng.gen_range(0..n);

                // レプリカ k のビット i をフリップ
                let mut new_sol = replicas[k].clone();
                new_sol[i] = !new_sol[i];
                let new_score = problem.score(&new_sol);

                // 問題由来のエネルギー変化（P で割って各レプリカの寄与に換算）
                let delta_problem = (new_score - scores[k]) / p as f64;

                // 隣接レプリカとのフリップ前後の一致変化によるエネルギー変化
                let prev_k = if k == 0 { p - 1 } else { k - 1 };
                let next_k = if k == p - 1 { 0 } else { k + 1 };

                let equal_prev = replicas[k][i] == replicas[prev_k][i];
                let equal_next = replicas[k][i] == replicas[next_k][i];

                let mut delta_coupling = 0.0;
                // 一致 → 不一致: +2J⊥、不一致 → 一致: -2J⊥
                if equal_prev { delta_coupling += 2.0 * j_perp; } else { delta_coupling -= 2.0 * j_perp; }
                if equal_next { delta_coupling += 2.0 * j_perp; } else { delta_coupling -= 2.0 * j_perp; }

                let delta_total = delta_problem + delta_coupling;

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
                        best = replicas[k].clone();
                        best_score = new_score;
                    }
                } else {
                    rejected += 1;
                }
            }

            if step % record_interval == 0 {
                score_history.push((step, best_score));
            }

            if let Some(log_int) = self.log_interval {
                if step % log_int == 0 {
                    let avg = scores.iter().sum::<f64>() / p as f64;
                    println!(
                        "SQA Step {}: gamma={:.4}, avg={:.6}, best={:.6}",
                        step, gamma, avg, best_score
                    );
                }
            }
        }

        score_history.push((self.max_steps, best_score));

        let final_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);

        let stats = SolverStats {
            iterations_completed: self.max_steps,
            initial_score,
            final_score,
            best_score,
            accepted_moves: accepted,
            rejected_moves: rejected,
            score_history,
        };

        (best, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_partition::{Graph, GraphPartitionProblem};

    #[test]
    fn test_sqa_simple_graph() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, true, false, false];
        let solver = SimulatedQuantumAnnealingSolver::new(4, 0.1, 5.0, 0.01, 100);
        let mut rng = Mt19937GenRand64::new(42);

        let (_sol, stats) = solver.solve(&problem, initial, &mut rng);

        assert!(stats.best_score.is_finite());
        assert_eq!(stats.iterations_completed, 100);
        assert!(stats.accepted_moves + stats.rejected_moves > 0);
    }

    #[test]
    fn test_sqa_determinism() {
        let mut graph = Graph::new(6);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, true, false, true, false];
        let solver = SimulatedQuantumAnnealingSolver::new(4, 0.1, 3.0, 0.01, 50);

        let (sol1, stats1) = solver.solve(&problem, initial.clone(), &mut Mt19937GenRand64::new(77));
        let (sol2, stats2) = solver.solve(&problem, initial, &mut Mt19937GenRand64::new(77));

        assert_eq!(sol1, sol2);
        assert_eq!(stats1.best_score, stats2.best_score);
    }
}
