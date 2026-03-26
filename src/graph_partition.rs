//! グラフ分割問題 (Graph Partitioning Problem) の定義と実装。
//!
//! グラフの頂点集合を2つの部分集合に分割し、カットエッジ数とバランスペナルティの
//! 合計を最小化する組合せ最適化問題を扱う。
//!
//! # 目的関数
//!
//! スコア = カットエッジ数 + α × (|V₁| − |V₂|)²
//!
//! - **カットエッジ**: 異なるパーティションに属する端点を持つ辺の数
//! - **バランスペナルティ**: 2つのパーティションのサイズ差の二乗に比例（係数 [`ALPHA`]）
//!
//! # グラフ生成方式
//!
//! - [`GraphGenerationMethod::Random`] — Erdős–Rényi モデル G(n, p)。各辺を独立に確率 p で生成する。
//! - [`GraphGenerationMethod::Geometric`] — 幾何グラフ。[0,1]² 上にランダムに配置した頂点間の距離が閾値以下なら辺を張る。
//!
//! # 近傍構造
//!
//! 1つの頂点のパーティション割り当てを反転する操作（フリップ）を近傍とする。
//! 近傍サイズは頂点数 n に等しく、[`OptimizationProblem::neighbour`] ではスコアの
//! 差分計算（Δ評価）により O(deg(v)) で新スコアを求める。

use crate::optimization::{
    ContinuousRelaxationProblem, ExtremalOptimizationProblem, OptimizationProblem,
    ReplicaCouplingProblem,
};
use rand::Rng;
use rand_mt::Mt19937GenRand64;

/// バランスペナルティの係数。
///
/// 目的関数に α × (|V₁| − |V₂|)² の項を加えることで、
/// 極端に偏った分割を抑制する。
pub const ALPHA: f64 = 0.05;

/// パーティションの表現。`true` / `false` で頂点を2群に分類する。
pub type Partition = Vec<bool>;

/// グラフの生成方式を指定する列挙型。
#[derive(Debug, Clone)]
pub enum GraphGenerationMethod {
    /// Erdős–Rényi ランダムグラフ G(n, p)。
    /// `expected_degree` から辺確率 p = expected_degree / (n − 1) を算出する。
    Random { node_count: usize, expected_degree: f64 },
    /// 幾何ランダムグラフ。
    /// [0,1]² 上に一様ランダムに頂点を配置し、距離が閾値以下の頂点対に辺を張る。
    Geometric { node_count: usize, expected_degree: f64 },
}

/// 無向・重みなしグラフの隣接リスト表現。
#[derive(Debug, Clone)]
pub struct Graph {
    /// 各頂点の隣接頂点リスト。
    pub adjacency_list: Vec<Vec<usize>>,
    /// 頂点数。
    pub node_count: usize,
}

impl Graph {
    /// 指定した頂点数で辺のない空グラフを生成する。
    pub fn new(node_count: usize) -> Self {
        Self {
            adjacency_list: vec![Vec::new(); node_count],
            node_count,
        }
    }

    /// 無向辺を追加する。自己ループおよび範囲外の頂点は無視される。
    pub fn add_edge(&mut self, from: usize, to: usize) {
        if from < self.node_count && to < self.node_count && from != to {
            self.adjacency_list[from].push(to);
            self.adjacency_list[to].push(from);
        }
    }

    /// 指定した頂点の隣接頂点リストを返す。
    pub fn get_neighbors(&self, node: usize) -> &Vec<usize> {
        &self.adjacency_list[node]
    }
}

/// パーティション内の `true` / `false` の頂点数を返す。
pub fn get_partition_sizes(partition: &Partition) -> (usize, usize) {
    let true_count = partition.iter().filter(|&&x| x).count();
    let false_count = partition.len() - true_count;
    (true_count, false_count)
}

/// グラフ分割問題のインスタンス。
///
/// [`OptimizationProblem`] トレイトを実装しており、焼きなまし法などの
/// 汎用ソルバーで解くことができる。
pub struct GraphPartitionProblem {
    graph: Graph,
}

impl GraphPartitionProblem {
    /// 幾何グラフを生成し、グラフと頂点座標の両方を返す。
    ///
    /// 座標は可視化や再現性のためにファイルに保存する用途を想定している。
    pub fn generate_geometric_with_coords(
        node_count: usize,
        expected_degree: f64,
        rng: &mut Mt19937GenRand64,
    ) -> (Graph, Vec<(f64, f64)>) {
        let mut graph = Graph::new(node_count);

        // Generate random points in [0, 1] x [0, 1]
        let mut points = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let x = rng.r#gen::<f64>();
            let y = rng.r#gen::<f64>();
            points.push((x, y));
        }

        // Calculate threshold distance for expected degree
        let pi = std::f64::consts::PI;
        let threshold = (expected_degree / (node_count as f64 * pi)).sqrt();

        // Create edges between points within threshold distance
        for u in 0..node_count {
            for v in (u + 1)..node_count {
                let dx = points[u].0 - points[v].0;
                let dy = points[u].1 - points[v].1;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist <= threshold {
                    graph.add_edge(u, v);
                }
            }
        }

        (graph, points)
    }
}

impl OptimizationProblem<Graph, Partition, GraphGenerationMethod> for GraphPartitionProblem {
    fn new(graph: Graph) -> Self {
        Self { graph }
    }

    fn generate_problem(generation_method: GraphGenerationMethod, rng: &mut Mt19937GenRand64) -> Graph {
        match generation_method {
            GraphGenerationMethod::Random { node_count, expected_degree } => {
                let mut graph = Graph::new(node_count);

                let edge_probability = if node_count > 1 {
                    expected_degree / (node_count - 1) as f64
                } else {
                    0.0
                };

                for i in 0..node_count {
                    for j in (i + 1)..node_count {
                        if rng.r#gen::<f64>() < edge_probability {
                            graph.add_edge(i, j);
                        }
                    }
                }

                graph
            }
            GraphGenerationMethod::Geometric { node_count, expected_degree } => {
                // Use the dedicated function and discard coordinates
                let (graph, _coords) = GraphPartitionProblem::generate_geometric_with_coords(
                    node_count,
                    expected_degree,
                    rng,
                );
                graph
            }
        }
    }

    fn score(&self, partition: &Partition) -> f64 {
        let mut cut_edges = 0;

        for node in 0..self.graph.node_count {
            for &neighbor in &self.graph.adjacency_list[node] {
                if partition[node] != partition[neighbor] {
                    cut_edges += 1;
                }
            }
        }
        cut_edges /= 2; // Each edge is counted twice

        // Calculate balance penalty
        let (true_count, false_count) = get_partition_sizes(partition);
        let diff = (true_count as i64 - false_count as i64).abs() as f64;
        let penalty = ALPHA * diff * diff;

        cut_edges as f64 + penalty
    }

    fn neighbour_size(&self) -> usize {
        self.graph.node_count
    }

    fn neighbour(
        &self,
        neighbour_id: usize,
        current_solution: &Partition,
        current_score: f64,
    ) -> (Partition, f64) {
        if neighbour_id >= self.graph.node_count {
            panic!("Neighbour ID out of bounds");
        }
        let mut new_partition = current_solution.clone();
        new_partition[neighbour_id] = !new_partition[neighbour_id];

        // Calculate efficient score delta
        let v1_size = current_solution
            .iter()
            .filter(|&&x| x != new_partition[neighbour_id])
            .count();
        let v2_size = self.graph.node_count - v1_size;

        let added_edges = self.graph.adjacency_list[neighbour_id]
            .iter()
            .filter(|&&neighbor| new_partition[neighbour_id] != new_partition[neighbor])
            .count();

        let degree_i = self.graph.adjacency_list[neighbour_id].len();
        let removed_edges = degree_i - added_edges;

        let score_delta = added_edges as f64
            - removed_edges as f64
            - ALPHA * 4.0 * (v1_size as f64 - v2_size as f64 - 1.0);

        let new_score = current_score + score_delta;

        (new_partition, new_score)
    }

    fn create_random_solution(&self, rng: &mut Mt19937GenRand64) -> Partition {
        (0..self.graph.node_count).map(|_| rng.r#gen::<bool>()).collect()
    }
}

impl ExtremalOptimizationProblem<Graph, Partition, GraphGenerationMethod> for GraphPartitionProblem {
    fn component_count(&self) -> usize {
        self.graph.node_count
    }

    fn component_fitness(&self, solution: &Partition) -> Vec<f64> {
        (0..self.graph.node_count)
            .map(|i| {
                let deg = self.graph.adjacency_list[i].len();
                if deg == 0 {
                    return 0.0;
                }
                let internal = self.graph.adjacency_list[i]
                    .iter()
                    .filter(|&&j| solution[j] == solution[i])
                    .count();
                let external = deg - internal;
                (internal as f64 - external as f64) / deg as f64
            })
            .collect()
    }
}

impl ReplicaCouplingProblem<Graph, Partition, GraphGenerationMethod> for GraphPartitionProblem {
    fn components_equal(&self, sol_a: &Partition, sol_b: &Partition, component: usize) -> bool {
        sol_a[component] == sol_b[component]
    }
}

impl ContinuousRelaxationProblem<Graph> for GraphPartitionProblem {
    fn dimension(&self) -> usize {
        self.graph.node_count
    }

    fn continuous_score(&self, solution: &[f64]) -> f64 {
        // カットエッジの連続版: Σ_{(u,v)∈E} |x_u - x_v|
        let mut cut_cost = 0.0;
        for u in 0..self.graph.node_count {
            for &v in &self.graph.adjacency_list[u] {
                if u < v {
                    cut_cost += (solution[u] - solution[v]).abs();
                }
            }
        }

        // バランスペナルティ: ALPHA × (Σx_i - n/2)²
        let sum: f64 = solution.iter().sum();
        let diff = sum - self.graph.node_count as f64 / 2.0;
        let penalty = ALPHA * diff * diff;

        cut_cost + penalty
    }

    fn discretize(&self, solution: &[f64]) -> Vec<bool> {
        solution.iter().map(|&x| x >= 0.5).collect()
    }

    fn discrete_score(&self, partition: &[bool]) -> f64 {
        self.score(&partition.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::OptimizationProblem;

    #[test]
    fn test_graph_creation() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        assert_eq!(graph.node_count, 4);
        assert_eq!(graph.get_neighbors(0), &vec![1]);
        assert_eq!(graph.get_neighbors(1), &vec![0, 2]);
        assert_eq!(graph.get_neighbors(2), &vec![1, 3]);
        assert_eq!(graph.get_neighbors(3), &vec![2]);
    }

    #[test]
    fn test_partition_sizes() {
        let partition = vec![true, false, true, false, true];
        let (true_count, false_count) = get_partition_sizes(&partition);
        assert_eq!(true_count, 3);
        assert_eq!(false_count, 2);
    }

    #[test]
    fn test_score_calculation() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);

        // Perfect partition: [true, false, true, false] - should have 3 cut edges
        // Edge 0-1: true-false (cut), Edge 1-2: false-true (cut), Edge 2-3: true-false (cut)
        let partition = vec![true, false, true, false];
        let score = problem.score(&partition);

        // 3 cut edges + penalty for imbalance
        let expected_penalty = ALPHA * 0.0; // balanced partition
        assert_eq!(score, 3.0 + expected_penalty);
    }

    #[test]
    fn test_imbalanced_partition_penalty() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);

        let problem = GraphPartitionProblem::new(graph);

        // Highly imbalanced: [true, true, true, false] - 3 vs 1
        let partition = vec![true, true, true, false];
        let score = problem.score(&partition);

        let cut_edges = 0.0; // Edge 0-1: true-true (not cut)
        let penalty = ALPHA * 2.0 * 2.0; // |3-1|^2 * ALPHA
        assert_eq!(score, cut_edges + penalty);
    }

    #[test]
    fn test_random_solution_generation() {
        let graph = Graph::new(10);
        let problem = GraphPartitionProblem::new(graph);
        let mut rng = Mt19937GenRand64::new(42);
        let solution = problem.create_random_solution(&mut rng);

        assert_eq!(solution.len(), 10);
        
        // With same seed, should be deterministic
        let graph2 = Graph::new(10);
        let problem2 = GraphPartitionProblem::new(graph2);
        let mut rng2 = Mt19937GenRand64::new(42);
        let solution2 = problem2.create_random_solution(&mut rng2);
        assert_eq!(solution, solution2);
    }

    #[test]
    fn test_neighbour_generation() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        let problem = GraphPartitionProblem::new(graph);
        let partition = vec![true, false, true];
        let score = problem.score(&partition);

        // Test flipping node 1
        let (new_partition, new_score) = problem.neighbour(1, &partition, score);
        assert_eq!(new_partition, vec![true, true, true]);
        let expected_score = problem.score(&new_partition);
        assert!(
            (new_score - expected_score).abs() < 1e-10,
            "Score mismatch: {} vs {}",
            new_score,
            expected_score
        );
    }

    #[test]
    fn test_efficient_neighbour_delta() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let partition = vec![true, false, true, false];
        let current_score = problem.score(&partition);

        // Test efficient delta calculation using neighbour method (flipping node 1)
        let (new_partition, new_score) = problem.neighbour(1, &partition, current_score);

        // Verify the score matches the actual calculation
        let expected_score = problem.score(&new_partition);
        assert!(
            (new_score - expected_score).abs() < 1e-10,
            "Delta calculation mismatch: {} vs {}",
            new_score,
            expected_score
        );
    }

    #[test]
    fn test_optimization_trait_methods() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        let problem = GraphPartitionProblem::new(graph);

        assert_eq!(problem.neighbour_size(), 3);

        let partition = vec![true, false, true];
        let score = problem.score(&partition);

        // Test basin method from trait
        let optimized = problem.basin(&partition);
        let optimized_score = problem.score(&optimized);

        // Basin should find a local optimum (score should be <= original)
        assert!(optimized_score <= score || (optimized_score - score).abs() < 1e-10);
    }

    #[test]
    fn test_generate_random_instance() {
        // Test with high expected degree - should generate many edges
        let method1 = GraphGenerationMethod::Random { node_count: 5, expected_degree: 3.0 };
        let mut rng1 = Mt19937GenRand64::new(42);
        let graph1 = GraphPartitionProblem::generate_problem(method1, &mut rng1);
        assert_eq!(graph1.node_count, 5);
        
        // Count edges and calculate average degree
        let total_degree1: usize = graph1.adjacency_list.iter().map(|adj| adj.len()).sum();
        let avg_degree1 = total_degree1 as f64 / graph1.node_count as f64;
        
        // Test with low expected degree - should generate fewer edges
        let method2 = GraphGenerationMethod::Random { node_count: 5, expected_degree: 1.0 };
        let mut rng2 = Mt19937GenRand64::new(42);
        let graph2 = GraphPartitionProblem::generate_problem(method2, &mut rng2);
        let total_degree2: usize = graph2.adjacency_list.iter().map(|adj| adj.len()).sum();
        let avg_degree2 = total_degree2 as f64 / graph2.node_count as f64;
        
        // Higher expected degree should generally produce higher average degree
        assert!(avg_degree1 >= avg_degree2);
        
        // Test determinism - same seed should produce same graph
        let method3 = GraphGenerationMethod::Random { node_count: 5, expected_degree: 2.0 };
        let mut rng3 = Mt19937GenRand64::new(42);
        let mut rng4 = Mt19937GenRand64::new(42);
        let graph3 = GraphPartitionProblem::generate_problem(method3.clone(), &mut rng3);
        let graph4 = GraphPartitionProblem::generate_problem(method3.clone(), &mut rng4);

        assert_eq!(graph3.adjacency_list, graph4.adjacency_list);
        
        // Test that average degree is approximately expected (with some tolerance for randomness)
        let total_degree3: usize = graph3.adjacency_list.iter().map(|adj| adj.len()).sum();
        let avg_degree3 = total_degree3 as f64 / graph3.node_count as f64;
        
        // Should be reasonably close to expected degree (within reasonable variance)
        assert!((avg_degree3 - 2.0).abs() < 1.5, "Average degree {} too far from expected 2.0", avg_degree3);
    }

    #[test]
    fn test_generate_problem_trait() {
        // Test the trait method for generating random problems
        let method = GraphGenerationMethod::Random { 
            node_count: 5, 
            expected_degree: 2.0 
        };
        let mut rng = Mt19937GenRand64::new(123);
        let graph = GraphPartitionProblem::generate_problem(method.clone(), &mut rng);
        assert_eq!(graph.node_count, 5);
        
        // Test determinism
        let mut rng1 = Mt19937GenRand64::new(123);
        let mut rng2 = Mt19937GenRand64::new(123);
        let graph1 = GraphPartitionProblem::generate_problem(method.clone(), &mut rng1);
        let graph2 = GraphPartitionProblem::generate_problem(method.clone(), &mut rng2);
        
        assert_eq!(graph1.adjacency_list, graph2.adjacency_list);
    }

    #[test]
    fn test_generation_method_variants() {
        // Test different generation parameters
        let method_small = GraphGenerationMethod::Random { 
            node_count: 3, 
            expected_degree: 1.0 
        };
        let method_large = GraphGenerationMethod::Random { 
            node_count: 6, 
            expected_degree: 3.0 
        };
        
        let mut rng = Mt19937GenRand64::new(456);
        let graph_small = GraphPartitionProblem::generate_problem(method_small, &mut rng);
        
        // Reset RNG with different seed for fair comparison
        let mut rng = Mt19937GenRand64::new(789);
        let graph_large = GraphPartitionProblem::generate_problem(method_large, &mut rng);
        
        assert_eq!(graph_small.node_count, 3);
        assert_eq!(graph_large.node_count, 6);
        
        // Calculate average degrees
        let avg_degree_small = if graph_small.node_count > 0 {
            graph_small.adjacency_list.iter().map(|adj| adj.len()).sum::<usize>() as f64 / graph_small.node_count as f64
        } else {
            0.0
        };
        let avg_degree_large = if graph_large.node_count > 0 {
            graph_large.adjacency_list.iter().map(|adj| adj.len()).sum::<usize>() as f64 / graph_large.node_count as f64
        } else {
            0.0
        };
        
        // Verify that larger expected degree generally produces higher average degree
        // (with some tolerance for randomness)
        assert!(avg_degree_large >= avg_degree_small - 1.0,
            "Large graph avg degree {} should be >= small graph avg degree {} - 1.0",
            avg_degree_large, avg_degree_small);
    }

    #[test]
    fn test_component_count() {
        let graph = Graph::new(7);
        let problem = GraphPartitionProblem::new(graph);
        assert_eq!(
            ExtremalOptimizationProblem::component_count(&problem),
            7
        );
    }

    #[test]
    fn test_component_fitness_path_graph() {
        // パスグラフ 0-1-2-3, パーティション [true, true, false, false]
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        let problem = GraphPartitionProblem::new(graph);
        let partition = vec![true, true, false, false];
        let fitness = problem.component_fitness(&partition);

        assert_eq!(fitness.len(), 4);
        // 頂点 0: 隣接=[1(同)], fitness = (1-0)/1 = 1.0
        assert!((fitness[0] - 1.0).abs() < 1e-10);
        // 頂点 1: 隣接=[0(同), 2(異)], fitness = (1-1)/2 = 0.0
        assert!((fitness[1] - 0.0).abs() < 1e-10);
        // 頂点 2: 隣接=[1(異), 3(同)], fitness = (1-1)/2 = 0.0
        assert!((fitness[2] - 0.0).abs() < 1e-10);
        // 頂点 3: 隣接=[2(同)], fitness = (1-0)/1 = 1.0
        assert!((fitness[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_component_fitness_isolated_vertex() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        // 頂点 2 は孤立

        let problem = GraphPartitionProblem::new(graph);
        let partition = vec![true, false, true];
        let fitness = problem.component_fitness(&partition);

        // 頂点 0: 隣接=[1(異)], fitness = (0-1)/1 = -1.0
        assert!((fitness[0] - (-1.0)).abs() < 1e-10);
        // 頂点 2: 孤立, fitness = 0.0
        assert!((fitness[2] - 0.0).abs() < 1e-10);
    }
}
