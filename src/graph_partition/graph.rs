use crate::error::{Error, Result};
use crate::experiment::config::{GraphKind, GraphSpec};
use crate::optimization::CancellationToken;
use rand::Rng;
use rand_mt::Mt19937GenRand64;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
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
        Ok(Self {
            node_count,
            edges,
            adjacency,
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
    pub fn content_hash(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"gpp-graph-v1\0");
        h.update((self.node_count as u64).to_le_bytes());
        for &[a, b] in &self.edges {
            h.update((a as u64).to_le_bytes());
            h.update((b as u64).to_le_bytes());
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }
}
