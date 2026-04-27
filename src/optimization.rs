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

/// 解空間平滑化に必要な追加インターフェース。
///
/// 複数のスムージングレベル（解空間）でのスコア評価を提供し、
/// 平滑化された空間から元の解空間へと段階的に移行する最適化手法で使用する。
///
/// インデックス 0 が最も平滑化された空間、`space_count() - 1` が元の解空間（スムージングなし）。
///
/// デフォルト実装では `random_neighbour` を K 回サンプリングした平均スコアを
/// 各空間のスムージングスコアとして使用する。K は空間インデックスに応じて
/// `max_smoothing_samples()` から 1 に線形減少する。
/// 問題固有の滑らかな目的関数がある場合は `smoothed_score_in_space` をオーバーライドできる。
pub trait SolutionSpaceSmoothingProblem<P, S: Clone, G>: OptimizationProblem<P, S, G> {
    /// スムージング空間の数。
    fn space_count(&self) -> usize;

    /// 最も平滑化された空間（`space_id = 0`）でのサンプリング数 K の最大値。
    fn max_smoothing_samples(&self) -> usize {
        50
    }

    /// 空間インデックス `space_id` に対応するサンプリング数 K を返す。
    ///
    /// `space_id = 0` のとき `max_smoothing_samples()`、
    /// `space_id = space_count() - 1` のとき 1 に線形補間する。
    fn samples_for_space(&self, space_id: usize) -> usize {
        let count = self.space_count();
        if count <= 1 || space_id >= count - 1 {
            return 1;
        }
        let t = space_id as f64 / (count - 1) as f64;
        let max_k = self.max_smoothing_samples() as f64;
        (max_k * (1.0 - t) + t).round().max(1.0) as usize
    }

    /// 指定した解空間でのスムージングスコアを計算する。
    ///
    /// デフォルト実装は `samples_for_space(space_id)` 個の近傍スコアの平均を返す。
    fn smoothed_score_in_space(
        &self,
        space_id: usize,
        solution: &S,
        current_score: f64,
        rng: &mut Mt19937GenRand64,
    ) -> f64 {
        let k = self.samples_for_space(space_id);
        if k <= 1 {
            return current_score;
        }
        let sum: f64 = (0..k)
            .map(|_| self.random_neighbour(solution, current_score, rng).1)
            .sum();
        sum / k as f64
    }
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
