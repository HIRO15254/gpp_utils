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

    /// 頂点 `i` をフリップしたときのスコア差分を返す（O(deg(i))）。
    /// `sizes` は現在の `(|V_T|, |V_F|)`。SA や HC で増分計算するために使う。
    pub fn flip_delta_with_sizes(
        &self,
        partition: &Partition,
        i: usize,
        sizes: (usize, usize),
    ) -> f64 {
        let bi = partition[i];
        let mut cut_now: i64 = 0;
        let mut deg: i64 = 0;
        for &v in &self.graph.adjacency_list[i] {
            deg += 1;
            if partition[v] != bi {
                cut_now += 1;
            }
        }
        // フリップ後のカット = deg - cut_now、よってカット差分 = deg - 2*cut_now
        let cut_delta = deg - 2 * cut_now;

        let t = sizes.0 as i64;
        let f = sizes.1 as i64;
        let diff_now = t - f;
        let diff_after = if bi { diff_now - 2 } else { diff_now + 2 };
        let pen_delta =
            ALPHA * ((diff_after * diff_after) as f64 - (diff_now * diff_now) as f64);
        cut_delta as f64 + pen_delta
    }

    /// `flip_delta_with_sizes` のサイズ自動計算版（O(n + deg(i))）。
    pub fn flip_delta(&self, partition: &Partition, i: usize) -> f64 {
        let sizes = get_partition_sizes(partition);
        self.flip_delta_with_sizes(partition, i, sizes)
    }

    /// 指定 partition の cut 数を i32 で計算する（O(E)）。
    ///
    /// SA / 山登りループの初期化や、デバッグビルドでの整合性検証に用いる。
    pub fn count_cut_edges(&self, partition: &Partition) -> i32 {
        let mut cut_edges: i32 = 0;
        for node in 0..self.graph.node_count {
            for &neighbor in &self.graph.adjacency_list[node] {
                if partition[node] != partition[neighbor] {
                    cut_edges += 1;
                }
            }
        }
        cut_edges / 2
    }

    /// 頂点 `v` をフリップした場合の整数状態とスコアを O(degree(v)) で計算する。
    ///
    /// 呼び出し側は現在の整数状態 `(cur_cut, cur_t, cur_f)` を保持しておく。
    /// 戻り値: `(new_cut, new_t, new_f, new_real_score)`。
    ///
    /// `new_real_score` は元の [`GraphPartitionProblem::score`] と
    /// **完全に同じ演算順序** (`int as f64 + ALPHA * int as f64 * int as f64`)
    /// で計算され、ビット完全一致を保証する。
    pub fn delta_apply(
        &self,
        partition: &Partition,
        v: usize,
        cur_cut: i32,
        cur_t: usize,
        cur_f: usize,
    ) -> (i32, usize, usize, f64) {
        let mut cuts_at_v: i32 = 0;
        for &u in &self.graph.adjacency_list[v] {
            if partition[v] != partition[u] {
                cuts_at_v += 1;
            }
        }
        let degree = self.graph.adjacency_list[v].len() as i32;
        let new_cut = cur_cut + degree - 2 * cuts_at_v;
        let (new_t, new_f) = if partition[v] {
            (cur_t - 1, cur_f + 1)
        } else {
            (cur_t + 1, cur_f - 1)
        };
        let diff = (new_t as i64 - new_f as i64).abs() as f64;
        let new_score = new_cut as f64 + ALPHA * diff * diff;
        (new_cut, new_t, new_f, new_score)
    }

    /// 整数状態とスコアから元の `score()` 形式の f64 を生成する。
    ///
    /// SA / 山登りループ内で `current_real` 等を計算する用途。
    /// `score()` と完全に同じ演算順序で評価する。
    pub fn score_from_state(cut: i32, t: usize, f: usize) -> f64 {
        let diff = (t as i64 - f as i64).abs() as f64;
        cut as f64 + ALPHA * diff * diff
    }

    /// 各頂点 `v` について「`v` に接続する辺のうちパーティションをまたぐ本数」
    /// (`cuts_at[v]`) を計算する（O(E)）。
    ///
    /// この配列を保持しておけば [`delta_apply_cached`](Self::delta_apply_cached) で
    /// フリップ差分を O(1) で評価でき、[`flip_vertex`](Self::flip_vertex) で
    /// O(degree) 増分更新できる。
    pub fn compute_cuts_at(&self, partition: &Partition) -> Vec<i32> {
        let mut cuts_at = vec![0i32; self.graph.node_count];
        for v in 0..self.graph.node_count {
            let mut c: i32 = 0;
            for &u in &self.graph.adjacency_list[v] {
                if partition[v] != partition[u] {
                    c += 1;
                }
            }
            cuts_at[v] = c;
        }
        cuts_at
    }

    /// 頂点 `idx` をフリップし、`partition` と `cuts_at` を同時に O(degree) で更新する。
    ///
    /// `cuts_at` は呼び出し時点で `partition` と整合している前提。
    /// 同じ `idx` で 2 回呼べば元に戻る（対合）。
    pub fn flip_vertex(&self, partition: &mut [bool], cuts_at: &mut [i32], idx: usize) {
        let bi = partition[idx];
        for &u in &self.graph.adjacency_list[idx] {
            // 辺 (idx, u) のクロス状態はフリップで反転する。
            if partition[u] != bi {
                cuts_at[u] -= 1;
            } else {
                cuts_at[u] += 1;
            }
        }
        let degree = self.graph.adjacency_list[idx].len() as i32;
        // idx の全辺がクロス状態を反転するので cuts_at[idx] = degree - cuts_at[idx]。
        cuts_at[idx] = degree - cuts_at[idx];
        partition[idx] = !partition[idx];
    }

    /// [`delta_apply`](Self::delta_apply) の O(1) 版。
    ///
    /// `cuts_at[v]` を走査せずに用いる以外は `delta_apply` と**完全に同じ演算**で、
    /// `cuts_at` が `partition` と整合していればビット完全一致の結果を返す。
    pub fn delta_apply_cached(
        &self,
        partition: &[bool],
        cuts_at: &[i32],
        v: usize,
        cur_cut: i32,
        cur_t: usize,
        cur_f: usize,
    ) -> (i32, usize, usize, f64) {
        let cuts_at_v = cuts_at[v];
        let degree = self.graph.adjacency_list[v].len() as i32;
        let new_cut = cur_cut + degree - 2 * cuts_at_v;
        let (new_t, new_f) = if partition[v] {
            (cur_t - 1, cur_f + 1)
        } else {
            (cur_t + 1, cur_f - 1)
        };
        let diff = (new_t as i64 - new_f as i64).abs() as f64;
        let new_score = new_cut as f64 + ALPHA * diff * diff;
        (new_cut, new_t, new_f, new_score)
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

    fn apply_move(&self, partition: &mut Partition, move_idx: usize) {
        partition[move_idx] = !partition[move_idx];
    }

    fn score_at_move(&self, partition: &Partition, move_idx: usize) -> f64 {
        // 一時クローン + flip + score（O(N + E)）。
        // N²クローン爆発（呼び出し元の `neighbour()` 経由）を解消する用途。
        // 元の `score()` と同じ演算経路を辿るためビット完全一致。
        let mut p = partition.clone();
        p[move_idx] = !p[move_idx];
        self.score(&p)
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

    /// 検証 1: `delta_apply` のビット完全一致テスト。
    /// new_real_score が `score(&flipped_partition)` と f64 ビット単位で一致することを保証。
    #[test]
    fn test_delta_apply_bitwise_equality() {
        for graph_seed in 0..5u64 {
            let method = GraphGenerationMethod::Random {
                node_count: 30,
                expected_degree: 4.0,
            };
            let mut rng = Mt19937GenRand64::new(graph_seed);
            let problem = GraphPartitionProblem::generate(method, &mut rng);
            let n = problem.graph.node_count;

            for trial in 0..50u64 {
                let mut prng = Mt19937GenRand64::new(graph_seed * 1000 + trial);
                let partition: Partition = problem.random_solution(&mut prng);
                let cur_cut = problem.count_cut_edges(&partition);
                let (cur_t, cur_f) = get_partition_sizes(&partition);

                for v in 0..n {
                    let (_nc, _nt, _nf, ns) =
                        problem.delta_apply(&partition, v, cur_cut, cur_t, cur_f);
                    let mut flipped = partition.clone();
                    flipped[v] = !flipped[v];
                    let direct = problem.score(&flipped);
                    assert_eq!(
                        ns.to_bits(),
                        direct.to_bits(),
                        "delta_apply mismatch: graph_seed={}, trial={}, v={}, ns={}, direct={}",
                        graph_seed, trial, v, ns, direct
                    );
                }
            }
        }
    }

    /// `compute_cuts_at` + `delta_apply_cached` が `delta_apply` と
    /// ビット完全一致することを検証する。
    #[test]
    fn test_delta_apply_cached_matches_delta_apply() {
        for graph_seed in 0..5u64 {
            let method = GraphGenerationMethod::Random {
                node_count: 30,
                expected_degree: 4.0,
            };
            let mut rng = Mt19937GenRand64::new(graph_seed);
            let problem = GraphPartitionProblem::generate(method, &mut rng);
            let n = problem.graph.node_count;

            for trial in 0..50u64 {
                let mut prng = Mt19937GenRand64::new(graph_seed * 1000 + trial);
                let partition: Partition = problem.random_solution(&mut prng);
                let cur_cut = problem.count_cut_edges(&partition);
                let (cur_t, cur_f) = get_partition_sizes(&partition);
                let cuts_at = problem.compute_cuts_at(&partition);

                for v in 0..n {
                    let (nc, nt, nf, ns) =
                        problem.delta_apply(&partition, v, cur_cut, cur_t, cur_f);
                    let (cc, ct, cf, cs) = problem
                        .delta_apply_cached(&partition, &cuts_at, v, cur_cut, cur_t, cur_f);
                    assert_eq!((nc, nt, nf), (cc, ct, cf));
                    assert_eq!(
                        ns.to_bits(),
                        cs.to_bits(),
                        "delta_apply_cached mismatch: graph_seed={}, trial={}, v={}",
                        graph_seed, trial, v
                    );
                }
            }
        }
    }

    /// `flip_vertex` が `cuts_at` を正しく増分更新し（full recompute と一致）、
    /// 同じ頂点で 2 回呼ぶと元に戻る（対合）ことを検証する。
    #[test]
    fn test_flip_vertex_maintains_cuts_at() {
        for graph_seed in 0..5u64 {
            let method = GraphGenerationMethod::Random {
                node_count: 30,
                expected_degree: 4.0,
            };
            let mut rng = Mt19937GenRand64::new(graph_seed);
            let problem = GraphPartitionProblem::generate(method, &mut rng);
            let n = problem.graph.node_count;

            for trial in 0..50u64 {
                let mut prng = Mt19937GenRand64::new(graph_seed * 1000 + trial);
                let partition: Partition = problem.random_solution(&mut prng);

                // 各頂点をフリップ → 整合性確認 → 再フリップで原状復帰（対合）。
                let mut p = partition.clone();
                let mut cuts_at = problem.compute_cuts_at(&p);
                for v in 0..n {
                    problem.flip_vertex(&mut p, &mut cuts_at, v);
                    assert_eq!(
                        cuts_at,
                        problem.compute_cuts_at(&p),
                        "cuts_at drift after flip: graph_seed={}, trial={}, v={}",
                        graph_seed, trial, v
                    );
                    problem.flip_vertex(&mut p, &mut cuts_at, v);
                }
                assert_eq!(p, partition, "flip_vertex is not involutive");

                // 連鎖フリップ: 0..n を順に適用しても整合する。
                let mut p2 = partition.clone();
                let mut c2 = problem.compute_cuts_at(&p2);
                for v in 0..n {
                    problem.flip_vertex(&mut p2, &mut c2, v);
                    assert_eq!(
                        c2,
                        problem.compute_cuts_at(&p2),
                        "cuts_at drift in chained flip: graph_seed={}, trial={}, v={}",
                        graph_seed, trial, v
                    );
                }
            }
        }
    }

    /// 検証 2: `score_at_move` のビット完全一致テスト。
    /// `score(&neighbour(s)[i])` と f64 ビット単位で一致することを保証。
    #[test]
    fn test_score_at_move_bitwise_equality() {
        let method = GraphGenerationMethod::Random {
            node_count: 30,
            expected_degree: 4.0,
        };
        let mut rng = Mt19937GenRand64::new(7);
        let problem = GraphPartitionProblem::generate(method, &mut rng);
        let n = problem.graph.node_count;

        for trial in 0..20u64 {
            let mut prng = Mt19937GenRand64::new(trial);
            let partition: Partition = problem.random_solution(&mut prng);
            let neighbours = problem.neighbour(&partition);
            for i in 0..n {
                let via = problem.score_at_move(&partition, i);
                let direct = problem.score(&neighbours[i]);
                assert_eq!(
                    via.to_bits(),
                    direct.to_bits(),
                    "score_at_move mismatch: trial={}, i={}, via={}, direct={}",
                    trial, i, via, direct
                );
            }
        }
    }

    /// 検証 3: 二重フリップ (delta_apply の連鎖) のビット完全一致テスト。
    /// `c = current^flip(idx)` の k 番目近傍について、
    /// `delta_apply(&current_after_flip_idx, k, candidate_state).3` が
    /// `score(&current^flip(idx)^flip(k))` と f64 ビット単位で一致することを保証。
    #[test]
    fn test_double_flip_bitwise_equality() {
        let method = GraphGenerationMethod::Random {
            node_count: 30,
            expected_degree: 4.0,
        };
        let mut rng = Mt19937GenRand64::new(11);
        let problem = GraphPartitionProblem::generate(method, &mut rng);
        let n = problem.graph.node_count;

        for trial in 0..20u64 {
            let mut prng = Mt19937GenRand64::new(trial);
            let partition: Partition = problem.random_solution(&mut prng);
            let cur_cut = problem.count_cut_edges(&partition);
            let (cur_t, cur_f) = get_partition_sizes(&partition);

            for idx in (0..n).step_by(3) {
                // 候補 c の整数状態
                let (nc, nt, nf, _) = problem.delta_apply(&partition, idx, cur_cut, cur_t, cur_f);
                // current を一時 flip (= c に変身)
                let mut current_flipped = partition.clone();
                current_flipped[idx] = !current_flipped[idx];

                for k in 0..n {
                    let via = problem.delta_apply(&current_flipped, k, nc, nt, nf).3;
                    let mut double_flipped = current_flipped.clone();
                    double_flipped[k] = !double_flipped[k];
                    let direct = problem.score(&double_flipped);
                    assert_eq!(
                        via.to_bits(),
                        direct.to_bits(),
                        "double-flip mismatch: trial={}, idx={}, k={}, via={}, direct={}",
                        trial, idx, k, via, direct
                    );
                }
            }
        }
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

    #[test]
    fn test_flip_delta_matches_full_recompute() {
        let method = GraphGenerationMethod::Random {
            node_count: 30,
            expected_degree: 4.0,
        };
        let mut rng = Mt19937GenRand64::new(7);
        let problem = GraphPartitionProblem::generate(method, &mut rng);
        let partition = problem.random_solution(&mut rng);
        let base = problem.score(&partition);
        for i in 0..partition.len() {
            let mut flipped = partition.clone();
            flipped[i] = !flipped[i];
            let direct = problem.score(&flipped) - base;
            let delta = problem.flip_delta(&partition, i);
            assert!(
                (direct - delta).abs() < 1e-9,
                "vertex {}: full={} delta={}",
                i,
                direct,
                delta
            );
        }
    }
}
