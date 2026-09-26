use crate::error::{Error, Result};
use crate::experiment::config::{GraphKind, GraphSpec};
use crate::optimization::CancellationToken;
use rand::Rng;
use rand_mt::Mt19937GenRand64;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::sync::OnceLock;

/// Largest vertex count for which [`Graph`] keeps an adjacency bit matrix
/// (`node_count * ceil(node_count / 64)` words, at most 8 MiB).
const MATRIX_MAX_NODES: usize = 8192;
const _: () = assert!(MATRIX_MAX_NODES * MATRIX_MAX_NODES.div_ceil(64) * 8 == 8 << 20);

#[derive(Clone)]
/// A normalized, immutable undirected graph.
///
/// Build a new graph with [`Self::from_edges`] when changing its topology. The
/// edge list and adjacency lists must always describe the same graph.
///
/// ```compile_fail
/// use gpp_utils::graph_partition::Graph;
/// let mut graph = Graph::from_edges(4, vec![]).unwrap();
/// graph.edges.push([0, 1]); // topology is private
/// ```
///
/// ```compile_fail
/// use gpp_utils::graph_partition::Graph;
/// let mut graph = Graph::from_edges(4, vec![]).unwrap();
/// graph.node_count = 2; // size cannot diverge from adjacency
/// ```
pub struct Graph {
    node_count: usize,
    edges: Vec<[usize; 2]>,
    adjacency: Vec<Vec<usize>>,
    /// Derived from `edges` for adjacency tests: bit `b % 64` of word
    /// `a * row_words + b / 64` is set exactly when `a` and `b` are adjacent.
    /// Empty (and `row_words == 0`) above [`MATRIX_MAX_NODES`] vertices. Not
    /// part of the stored graph or its content hash.
    matrix: Vec<u64>,
    row_words: usize,
    /// [`Self::content_hash`], computed on first use from `node_count` and
    /// `edges`, which never change.
    content_hash: OnceLock<String>,
}

/// The same output as the derived implementation before the derived fields
/// were added.
impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("node_count", &self.node_count)
            .field("edges", &self.edges)
            .field("adjacency", &self.adjacency)
            .finish()
    }
}

impl Graph {
    /// Validate endpoints, reject loops/duplicate edges and build canonical
    /// edges plus matching adjacency lists. Endpoints are zero based.
    pub fn from_edges(node_count: usize, edges: Vec<[usize; 2]>) -> Result<Self> {
        let mut normalized = BTreeSet::new();
        for [a, b] in edges {
            if a >= node_count || b >= node_count {
                return Err(Error::msg("edge endpoint out of range"));
            }
            if a == b {
                return Err(Error::msg("self loops are not allowed"));
            }
            let edge = if a < b { [a, b] } else { [b, a] };
            if !normalized.insert(edge) {
                return Err(Error::msg("duplicate edge"));
            }
        }
        let edges: Vec<_> = normalized.into_iter().collect();
        let mut adjacency = vec![Vec::new(); node_count];
        for &[a, b] in &edges {
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
        let row_words = if node_count <= MATRIX_MAX_NODES {
            node_count.div_ceil(64)
        } else {
            0
        };
        let mut matrix = vec![0u64; node_count * row_words];
        if row_words > 0 {
            for &[a, b] in &edges {
                matrix[a * row_words + b / 64] |= 1 << (b % 64);
                matrix[b * row_words + a / 64] |= 1 << (a % 64);
            }
        }
        Ok(Self {
            node_count,
            edges,
            adjacency,
            matrix,
            row_words,
            content_hash: OnceLock::new(),
        })
    }

    pub fn generate(spec: &GraphSpec, cancel: &CancellationToken) -> Result<Self> {
        if spec.node_count < 2
            || !spec.expected_degree.is_finite()
            || spec.expected_degree < 0.0
            || spec.expected_degree > (spec.node_count - 1) as f64
        {
            return Err(Error::msg("invalid graph specification"));
        }
        cancel.check()?;
        let mut iterations = 0usize;
        let mut rng = Mt19937GenRand64::new(spec.seed);
        let mut edges = Vec::new();
        match spec.kind {
            GraphKind::Random => {
                let p = spec.expected_degree / (spec.node_count - 1) as f64;
                for a in 0..spec.node_count {
                    for b in a + 1..spec.node_count {
                        if iterations & 1023 == 0 {
                            cancel.check()?;
                        }
                        iterations += 1;
                        if rng.r#gen::<f64>() < p {
                            edges.push([a, b]);
                        }
                    }
                }
            }
            GraphKind::Geometric => {
                let mut points: Vec<(f64, f64)> = Vec::with_capacity(spec.node_count);
                for i in 0..spec.node_count {
                    if i & 1023 == 0 {
                        cancel.check()?;
                    }
                    points.push((rng.r#gen(), rng.r#gen()));
                }
                let threshold2 =
                    spec.expected_degree / (spec.node_count as f64 * std::f64::consts::PI);
                for a in 0..spec.node_count {
                    for b in a + 1..spec.node_count {
                        if iterations & 1023 == 0 {
                            cancel.check()?;
                        }
                        iterations += 1;
                        let dx = points[a].0 - points[b].0;
                        let dy = points[a].1 - points[b].1;
                        if dx * dx + dy * dy <= threshold2 {
                            edges.push([a, b]);
                        }
                    }
                }
            }
        }
        cancel.check()?;
        Self::from_edges(spec.node_count, edges)
    }

    /// Number of vertices, numbered `0..node_count()`.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Canonical edges in lexicographic order, each with its smaller endpoint first.
    pub fn edges(&self) -> &[[usize; 2]] {
        &self.edges
    }

    /// Sorted adjacent vertex IDs. Panics if `vertex` is out of range.
    pub fn neighbors(&self, vertex: usize) -> &[usize] {
        &self.adjacency[vertex]
    }
    pub fn degree(&self, vertex: usize) -> usize {
        self.adjacency[vertex].len()
    }
    /// Whether `a` and `b` are adjacent: `self.neighbors(a).binary_search(&b).is_ok()`,
    /// answered from the adjacency matrix when the graph has one. Panics if
    /// `a` is out of range.
    #[inline]
    pub(crate) fn has_edge(&self, a: usize, b: usize) -> bool {
        if self.row_words == 0 {
            return self.adjacency[a].binary_search(&b).is_ok();
        }
        assert!(a < self.node_count, "vertex out of range");
        b < self.node_count && self.matrix[a * self.row_words + b / 64] >> (b % 64) & 1 == 1
    }
    pub fn score(&self, partition: &[bool], alpha: f64) -> f64 {
        let cut = self
            .edges
            .iter()
            .filter(|&&[a, b]| partition[a] != partition[b])
            .count();
        let a = partition.iter().filter(|&&x| x).count();
        let d = a as i64 - (partition.len() - a) as i64;
        cut as f64 + alpha * d as f64 * d as f64
    }
    /// SHA-256 of the vertex count and canonical edges, as lowercase hex.
    /// Computed once per graph value and then returned from the cache.
    pub fn content_hash(&self) -> String {
        self.content_hash
            .get_or_init(|| {
                let mut h = Sha256::new();
                h.update(b"gpp-graph-v1\0");
                h.update((self.node_count as u64).to_le_bytes());
                for &[a, b] in &self.edges {
                    h.update((a as u64).to_le_bytes());
                    h.update((b as u64).to_le_bytes());
                }
                h.finalize().iter().map(|b| format!("{b:02x}")).collect()
            })
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_edges(n: usize, p: f64, seed: u64) -> Vec<[usize; 2]> {
        let mut rng = Mt19937GenRand64::new(seed);
        let mut edges = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                if rng.r#gen::<f64>() < p {
                    edges.push([a, b]);
                }
            }
        }
        edges
    }

    /// `has_edge` agrees with the edge list for every pair, on both sides of
    /// the 64-bit word boundaries and of [`MATRIX_MAX_NODES`] (where it falls
    /// back to searching the adjacency list), and is false for any `b` past
    /// the last vertex.
    #[test]
    fn has_edge_matches_the_edge_list() {
        let mut cases = vec![
            (1, vec![]),
            (2, vec![[0, 1]]),
            (5, vec![]),
            (20, (1..20).map(|v| [0, v]).collect()),
        ];
        for (n, p, seed) in [
            (63, 0.2, 1),
            (64, 0.2, 2),
            (65, 0.2, 3),
            (127, 0.1, 4),
            (128, 0.1, 5),
            (129, 0.9, 6),
            (300, 0.05, 7),
        ] {
            cases.push((n, random_edges(n, p, seed)));
        }
        for n in [MATRIX_MAX_NODES, MATRIX_MAX_NODES + 1] {
            // Three random neighbours per vertex (a pairwise draw would take
            // too long here), plus edges touching the last vertices and word
            // ends.
            let mut rng = Mt19937GenRand64::new(n as u64);
            let mut edges: Vec<[usize; 2]> = (0..n)
                .flat_map(|a| {
                    let b: [usize; 3] = std::array::from_fn(|_| rng.gen_range(0..n));
                    b.map(|b| [a.min(b), a.max(b)])
                })
                .filter(|&[a, b]| a != b)
                .collect();
            edges.extend([[0, n - 1], [63, 64], [n - 65, n - 64], [n - 2, n - 1]]);
            edges.sort();
            edges.dedup();
            cases.push((n, edges));
        }
        for (n, edges) in cases {
            let graph = Graph::from_edges(n, edges).unwrap();
            assert_eq!(graph.row_words > 0, n <= MATRIX_MAX_NODES, "n = {n}");
            let set: BTreeSet<[usize; 2]> = graph.edges().iter().copied().collect();
            // Every row of small graphs; for large ones, the rows at both ends
            // and a stride of the rest.
            let rows: Vec<usize> = if n <= 2100 {
                (0..n).collect()
            } else {
                (0..130)
                    .chain(n - 130..n)
                    .chain((130..n - 130).step_by(17))
                    .collect()
            };
            for a in rows {
                let bs: Vec<usize> = if n <= 300 {
                    (0..n).collect()
                } else {
                    // Every neighbor, the word ends and a stride of the rest.
                    let mut bs = graph.neighbors(a).to_vec();
                    bs.extend((0..n).step_by(7 + a % 5));
                    bs.extend([0, 63, 64, n - 64, n - 1]);
                    bs
                };
                for b in bs {
                    let expected = set.contains(&[a.min(b), a.max(b)]);
                    assert_eq!(graph.has_edge(a, b), expected, "n = {n}: {a}-{b}");
                    assert_eq!(
                        graph.neighbors(a).binary_search(&b).is_ok(),
                        expected,
                        "n = {n}: {a}-{b}"
                    );
                }
                let past = n.div_ceil(64) * 64;
                for b in [n, n + 1, n + 63, past, past + 1, past + 64, usize::MAX] {
                    assert!(!graph.has_edge(a, b), "n = {n}: {a}-{b}");
                }
            }
        }
    }

    /// The cached hash equals a fresh SHA-256 of the documented layout on every
    /// call and clone, and the stored hash of a generated graph.
    #[test]
    fn cached_content_hash_matches_recomputation() {
        let reference = |graph: &Graph| -> String {
            let mut h = Sha256::new();
            h.update(b"gpp-graph-v1\0");
            h.update((graph.node_count() as u64).to_le_bytes());
            for &[a, b] in graph.edges() {
                h.update((a as u64).to_le_bytes());
                h.update((b as u64).to_le_bytes());
            }
            h.finalize().iter().map(|b| format!("{b:02x}")).collect()
        };
        let graphs = [
            Graph::from_edges(0, vec![]).unwrap(),
            Graph::from_edges(4, vec![]).unwrap(),
            Graph::from_edges(4, vec![[3, 2], [1, 0]]).unwrap(),
            Graph::from_edges(3000, random_edges(3000, 0.001, 9)).unwrap(),
        ];
        for graph in &graphs {
            let expected = reference(graph);
            let copy = graph.clone();
            assert_eq!(graph.content_hash(), expected);
            assert_eq!(graph.content_hash(), expected);
            assert_eq!(copy.content_hash(), expected);
            assert_eq!(graph.clone().content_hash(), expected);
        }
        assert_ne!(graphs[1].content_hash(), graphs[2].content_hash());
        // Stored by commit e4b6a1c for random, n = 124, degree 10, seed 0.
        let spec = GraphSpec {
            kind: GraphKind::Random,
            node_count: 124,
            expected_degree: 10.0,
            seed: 0,
        };
        let generated = Graph::generate(&spec, &CancellationToken::new()).unwrap();
        assert_eq!(generated.edges().len(), 623);
        assert_eq!(
            generated.content_hash(),
            "5bc69eb0b2893f383cca18a46d6047566fbefb517d3063d023b2b6c4f998ec8f"
        );
    }

    /// `Debug` prints the topology as the derived implementation did.
    #[test]
    fn debug_output_shows_only_the_topology() {
        let graph = Graph::from_edges(3, vec![[1, 0]]).unwrap();
        graph.content_hash();
        assert_eq!(
            format!("{graph:?}"),
            "Graph { node_count: 3, edges: [[0, 1]], adjacency: [[1], [0], []] }"
        );
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn adjacency_accelerator_matches_edges_and_sparse_fallback() {
        for n in [2, 63, 64, 65, 124, 500, 8193] {
            let edges: Vec<_> = (1..n).map(|v| [v - 1, v]).collect();
            let graph = Graph::from_edges(n, edges.clone()).unwrap();
            assert_eq!(graph.row_words == 0, n == 8193);
            for a in 0..n {
                for b in [0, a.saturating_sub(1), a, (a + 1).min(n - 1), n - 1] {
                    assert_eq!(graph.has_edge(a, b), a.abs_diff(b) == 1);
                }
            }
        }
    }

    #[test]
    fn cached_hash_keeps_original_encoding_across_clone_and_threads() {
        let graph = Graph::from_edges(4, vec![[3, 1], [0, 2]]).unwrap();
        let mut h = Sha256::new();
        h.update(b"gpp-graph-v1\0");
        h.update(4u64.to_le_bytes());
        for value in [0u64, 2, 1, 3] {
            h.update(value.to_le_bytes());
        }
        let expected = format!("{:x}", h.finalize());
        let cloned_before = graph.clone();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| assert_eq!(graph.content_hash(), expected));
            }
        });
        assert_eq!(cloned_before.content_hash(), expected);
        assert_eq!(graph.clone().content_hash(), expected);
    }
}
