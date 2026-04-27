//! τ-Extremal Optimization ソルバー。
//!
//! 各近傍のスムージングスコアを「適応度」として使用し、
//! べき乗則確率で低適応度の構成要素を選択・変更する。

use crate::optimization::{Problem, Smoothing, Solver, SolverStats};
use rand::Rng;
use rand_mt::Mt19937GenRand64;

/// τ-EO ソルバー。
///
/// `problem.neighbour()` が返す近傍リストの順序を構成要素のインデックスとして扱う。
/// `GraphPartitionProblem` では近傍 i = ノード i をフリップした解に対応する。
#[derive(Debug, Clone)]
pub struct ExtremalOptimizationSolver {
    /// べき乗則指数 τ。`None` の場合は 1 + 1/ln(n) を使用。
    pub tau: Option<f64>,
    /// 最大反復回数。
    pub max_iterations: usize,
    /// ログ出力間隔（None の場合はログなし）。
    pub log_interval: Option<usize>,
}

impl ExtremalOptimizationSolver {
    /// 新しいソルバーを作成する。
    pub fn new(tau: Option<f64>, max_iterations: usize) -> Self {
        Self { tau, max_iterations, log_interval: None }
    }

    /// ログ出力間隔を設定する。
    pub fn with_logging(mut self, interval: usize) -> Self {
        self.log_interval = Some(interval);
        self
    }
}

impl Default for ExtremalOptimizationSolver {
    fn default() -> Self {
        Self::new(None, 50_000)
    }
}

impl Solver for ExtremalOptimizationSolver {
    fn solve<S: Clone>(
        &self,
        problem: &dyn Problem<S>,
        smoothing: &dyn Smoothing<S>,
        initial: S,
        rng: &mut Mt19937GenRand64,
    ) -> (S, SolverStats) {
        let initial_score = problem.score(&initial);

        // n = 近傍数（= 構成要素数）を初期解から取得
        let init_neighbours = problem.neighbour(&initial);
        let n = init_neighbours.len();

        if n == 0 {
            return (
                initial,
                SolverStats {
                    iterations_completed: 0,
                    initial_score,
                    final_score: initial_score,
                    best_score: initial_score,
                    accepted_moves: 0,
                    rejected_moves: 0,
                    score_history: vec![(0, initial_score)],
                },
            );
        }

        let tau = self.tau.unwrap_or(1.0 + 1.0 / (n as f64).ln());
        let cdf = build_power_law_cdf(n, tau);

        let mut current = initial;
        let mut current_smoothed = smoothing.score(problem, &current);
        let mut best = current.clone();
        let mut best_score = initial_score;

        let mut score_history = vec![(0, best_score)];
        let record_interval = (self.max_iterations / 100).max(1);

        for iteration in 0..self.max_iterations {
            let neighbours = problem.neighbour(&current);
            let n = neighbours.len();
            if n == 0 {
                break;
            }

            // 適応度 = スムージングスコアのデルタ（小さい = 悪適応 = フリップ対象）
            let fitness: Vec<f64> = neighbours
                .iter()
                .map(|nb| smoothing.score(problem, nb) - current_smoothed)
                .collect();

            // 適応度昇順でランク付け（インデックス 0 = 最悪）
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| fitness[a].partial_cmp(&fitness[b]).unwrap());

            // べき乗則でランクを選択
            let u: f64 = rng.r#gen::<f64>();
            let rank = match cdf.binary_search_by(|probe| probe.partial_cmp(&u).unwrap()) {
                Ok(pos) => pos,
                Err(pos) => pos.min(n - 1),
            };

            let component_id = indices[rank];
            current = neighbours[component_id].clone();
            current_smoothed = smoothing.score(problem, &current);

            let real_score = problem.score(&current);
            if real_score < best_score {
                best = current.clone();
                best_score = real_score;
            }

            if iteration % record_interval == 0 {
                score_history.push((iteration, best_score));
            }

            if let Some(log_int) = self.log_interval {
                if iteration % log_int == 0 {
                    println!(
                        "EO Iteration {}: tau={:.3}, real={:.6}, best={:.6}",
                        iteration, tau, real_score, best_score
                    );
                }
            }
        }

        score_history.push((self.max_iterations, best_score));

        let stats = SolverStats {
            iterations_completed: self.max_iterations,
            initial_score,
            final_score: problem.score(&current),
            best_score,
            accepted_moves: self.max_iterations, // EO は常に受理
            rejected_moves: 0,
            score_history,
        };

        (best, stats)
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
    use crate::smoothing::NoSmoothing;

    #[test]
    fn test_power_law_cdf() {
        let cdf = build_power_law_cdf(10, 1.5);
        assert_eq!(cdf.len(), 10);
        for i in 1..cdf.len() {
            assert!(cdf[i] > cdf[i - 1]);
        }
        assert!((cdf[9] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eo_simple_graph() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, true, false];
        let solver = ExtremalOptimizationSolver::new(None, 2_000);
        let smoothing = NoSmoothing;
        let mut rng = Mt19937GenRand64::new(42);

        let (_sol, stats) = solver.solve(&problem, &smoothing, initial, &mut rng);

        assert!(stats.best_score <= stats.initial_score + 1e-10);
        assert_eq!(stats.iterations_completed, 2_000);
    }

    #[test]
    fn test_eo_determinism() {
        let mut graph = Graph::new(6);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        graph.add_edge(5, 0);

        let problem = GraphPartitionProblem::new(graph);
        let initial = vec![true, false, true, false, true, false];
        let solver = ExtremalOptimizationSolver::new(Some(1.5), 500);
        let smoothing = NoSmoothing;

        let (sol1, stats1) = solver.solve(&problem, &smoothing, initial.clone(), &mut Mt19937GenRand64::new(123));
        let (sol2, stats2) = solver.solve(&problem, &smoothing, initial, &mut Mt19937GenRand64::new(123));

        assert_eq!(sol1, sol2);
        assert_eq!(stats1.best_score, stats2.best_score);
    }
}
