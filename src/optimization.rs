use rand::Rng;
use rand_mt::Mt19937GenRand64;

pub trait OptimizationProblem<P, S: Clone, G> {
    fn new(problem: P) -> Self where Self: Sized;
    
    fn generate_problem(generation_method: G, rng: &mut Mt19937GenRand64) -> P where Self: Sized;

    fn score(&self, solution: &S) -> f64;

    fn neighbour_size(&self) -> usize;

    fn neighbour(
        &self,
        neighbour_id: usize,
        current_solution: &S,
        current_score: f64,
    ) -> (S, f64);

    fn create_random_solution(&self, rng: &mut Mt19937GenRand64) -> S;

    fn random_neighbour(
        &self,
        current_solution: &S,
        current_score: f64,
        rng: &mut Mt19937GenRand64,
    ) -> (S, f64) {
        let neighbor_count = self.neighbour_size();
        let random_id = rng.gen_range(0..neighbor_count);
        self.neighbour(random_id, current_solution, current_score)
    }

    fn basin(&self, initial_solution: &S) -> S {
        let (solution, _) = self.basin_with_distance(initial_solution);
        solution
    }

    fn basin_with_distance(&self, initial_solution: &S) -> (S, usize) {
        let mut current_solution = initial_solution.clone();
        let mut current_score = self.score(&current_solution);
        let mut improved = true;
        let mut distance = 0;

        while improved {
            improved = false;
            let neighbor_count = self.neighbour_size();
            
            if let Some((best_neighbour_solution, best_neighbour_score)) = (0..neighbor_count)
                .map(|neighbor_id| self.neighbour(neighbor_id, &current_solution, current_score))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) 
            {
                if best_neighbour_score < current_score {
                    current_solution = best_neighbour_solution;
                    current_score = best_neighbour_score;
                    improved = true;
                    distance += 1;
                }
            }            
        }

        (current_solution, distance)
    }
}

/// Extremal Optimization に必要な追加インターフェース。
///
/// 解の各構成要素に適応度を割り当て、最も適応度の低い要素を
/// 優先的に変更する EO アルゴリズムで使用する。
///
/// 構成要素 i は `neighbour(i, ...)` による変異操作に対応する必要がある。
pub trait ExtremalOptimizationProblem<P, S: Clone, G>: OptimizationProblem<P, S, G> {
    /// 解の構成要素数。
    fn component_count(&self) -> usize;

    /// 各構成要素の適応度を計算する。値が大きいほど良い配置を示す。
    ///
    /// 返り値の長さは `component_count()` に等しい。
    fn component_fitness(&self, solution: &S) -> Vec<f64>;
}

/// Simulated Quantum Annealing (SQA) に必要な追加インターフェース。
///
/// レプリカ間結合エネルギーの計算のため、2つの解の特定構成要素が
/// 同じ値を持つかどうかを判定する機能を提供する。
pub trait ReplicaCouplingProblem<P, S: Clone, G>: OptimizationProblem<P, S, G> {
    /// 構成要素 `component` が2つの解で同じ値を持つか判定する。
    fn components_equal(&self, sol_a: &S, sol_b: &S, component: usize) -> bool;
}

/// 連続緩和 SA に必要なインターフェース。
///
/// 二値変数を [0, 1] の連続変数に緩和し、連続空間上で SA を実行するために使用する。
pub trait ContinuousRelaxationProblem<P> {
    /// 連続変数の次元数（構成要素数）。
    fn dimension(&self) -> usize;

    /// 連続解のスコアを計算する（滑らかな目的関数）。
    fn continuous_score(&self, solution: &[f64]) -> f64;

    /// 連続解を離散解（`Vec<bool>`）に変換する。
    fn discretize(&self, solution: &[f64]) -> Vec<bool>;

    /// 離散解の真のスコアを計算する（最良解追跡用）。
    fn discrete_score(&self, partition: &[bool]) -> f64;
}
