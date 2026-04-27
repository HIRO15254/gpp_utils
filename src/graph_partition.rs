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
//! 近傍サイズは頂点数 n に等しい。

use crate::optimization::Problem;
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
/// [`Problem<Vec<bool>>`] トレイトを実装しており、汎用ソルバーで解くことができる。
#[derive(Clone)]
pub struct GraphPartitionProblem {
    graph: Graph,
}

impl GraphPartitionProblem {
    /// グラフからインスタンスを生成する。
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    /// 指定した生成方式でグラフを生成し、問題インスタンスを返す。
    pub fn generate(method: GraphGenerationMethod, rng: &mut Mt19937GenRand64) -> Self {
        let graph = match method {
            GraphGenerationMethod::Random { node_count, expected_degree } => {
                Self::generate_random_graph(node_count, expected_degree, rng)
            }
            GraphGenerationMethod::Geometric { node_count, expected_degree } => {
                Self::generate_geometric_graph(node_count, expected_degree, rng).0
            }
        };
        Self { graph }
    }

    /// 幾何グラフを生成し、頂点座標と共に問題インスタンスを返す（GUI可視化用）。
    pub fn generate_geometric_with_coords(
        node_count: usize,
        expected_degree: f64,
        rng: &mut Mt19937GenRand64,
    ) -> (Self, Vec<(f64, f64)>) {
        let (graph, coords) = Self::generate_geometric_graph(node_count, expected_degree, rng);
        (Self { graph }, coords)
    }

    /// 内部グラフへの参照を返す（可視化用）。
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Erdős–Rényi ランダムグラフを生成する。
    fn generate_random_graph(
        node_count: usize,
        expected_degree: f64,
        rng: &mut Mt19937GenRand64,
    ) -> Graph {
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

    /// 幾何ランダムグラフを生成する。
    fn generate_geometric_graph(
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

impl Problem<Partition> for GraphPartitionProblem {
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

    fn neighbour(&self, partition: &Partition) -> Vec<Partition> {
        let mut neighbours = Vec::with_capacity(self.graph.node_count);

        for i in 0..self.graph.node_count {
            let mut new_partition = partition.clone();
            new_partition[i] = !new_partition[i];
            neighbours.push(new_partition);
        }

        neighbours
    }

    fn random_solution(&self, rng: &mut Mt19937GenRand64) -> Partition {
        (0..self.graph.node_count)
            .map(|_| rng.r#gen::<bool>())
            .collect()
    }

    fn neighbour_size(&self) -> usize {
        self.graph.node_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let solution = problem.random_solution(&mut rng);

        assert_eq!(solution.len(), 10);

        // With same seed, should be deterministic
        let graph2 = Graph::new(10);
        let problem2 = GraphPartitionProblem::new(graph2);
        let mut rng2 = Mt19937GenRand64::new(42);
        let solution2 = problem2.random_solution(&mut rng2);
        assert_eq!(solution, solution2);
    }

    #[test]
    fn test_neighbour_generation() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        let problem = GraphPartitionProblem::new(graph);
        let partition = vec![true, false, true];

        let neighbours = problem.neighbour(&partition);
        assert_eq!(neighbours.len(), 3);
        assert_eq!(neighbours[0], vec![false, false, true]);
        assert_eq!(neighbours[1], vec![true, true, true]);
        assert_eq!(neighbours[2], vec![true, false, false]);
    }

    #[test]
    fn test_neighbour_size() {
        let graph = Graph::new(7);
        let problem = GraphPartitionProblem::new(graph);
        assert_eq!(problem.neighbour_size(), 7);
    }

    #[test]
    fn test_generate_random_instance() {
        let method = GraphGenerationMethod::Random {
            node_count: 5,
            expected_degree: 2.0,
        };
        let mut rng = Mt19937GenRand64::new(42);
        let problem = GraphPartitionProblem::generate(method, &mut rng);

        assert_eq!(problem.neighbour_size(), 5);

        // Verify it's a valid partition
        let partition = problem.random_solution(&mut rng);
        assert_eq!(partition.len(), 5);
        let score = problem.score(&partition);
        assert!(score.is_finite());
    }

    #[test]
    fn test_determinism() {
        let method = GraphGenerationMethod::Random {
            node_count: 5,
            expected_degree: 2.0,
        };

        let mut rng1 = Mt19937GenRand64::new(42);
        let problem1 = GraphPartitionProblem::generate(method.clone(), &mut rng1);

        let mut rng2 = Mt19937GenRand64::new(42);
        let problem2 = GraphPartitionProblem::generate(method, &mut rng2);

        let mut rng = Mt19937GenRand64::new(123);
        let partition = problem1.random_solution(&mut rng);

        assert_eq!(
            problem1.score(&partition),
            problem2.score(&partition),
            "Same seed should produce same graph"
        );
    }
}
