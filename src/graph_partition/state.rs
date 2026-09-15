use super::Graph;
use crate::error::{Error, Result};

#[derive(Clone, Debug)]
pub struct PartitionState {
    partition: Vec<bool>,
    cut_edges: i64,
    size_a: usize,
    cuts_at: Vec<i64>,
}

impl PartitionState {
    pub fn new(graph: &Graph, partition: Vec<bool>) -> Result<Self> {
        if partition.len() != graph.node_count {
            return Err(Error::msg("partition length does not match graph"));
        }
        let mut cuts_at = vec![0; graph.node_count];
        let mut cut_edges = 0;
        for &[a, b] in &graph.edges {
            if partition[a] != partition[b] {
                cuts_at[a] += 1;
                cuts_at[b] += 1;
                cut_edges += 1;
            }
        }
        let size_a = partition.iter().filter(|&&x| x).count();
        Ok(Self {
            partition,
            cut_edges,
            size_a,
            cuts_at,
        })
    }
    pub fn partition(&self) -> &[bool] {
        &self.partition
    }
    pub fn cut_edges(&self) -> usize {
        self.cut_edges as usize
    }
    pub fn size_a(&self) -> usize {
        self.size_a
    }
    pub fn size_b(&self) -> usize {
        self.partition.len() - self.size_a
    }
    pub fn score(&self, alpha: f64) -> f64 {
        let d = self.size_a as i64 - self.size_b() as i64;
        self.cut_edges as f64 + alpha * d as f64 * d as f64
    }
    pub fn flip_score(&self, graph: &Graph, v: usize, alpha: f64) -> f64 {
        let cut = self.cut_edges + graph.degree(v) as i64 - 2 * self.cuts_at[v];
        let a = if self.partition[v] {
            self.size_a - 1
        } else {
            self.size_a + 1
        };
        let d = a as i64 - (self.partition.len() - a) as i64;
        cut as f64 + alpha * d as f64 * d as f64
    }
    pub fn apply_flip(&mut self, graph: &Graph, v: usize) {
        let old = self.partition[v];
        let old_cuts = self.cuts_at[v];
        for &u in graph.neighbors(v) {
            if self.partition[u] != old {
                self.cuts_at[u] -= 1
            } else {
                self.cuts_at[u] += 1
            }
        }
        self.cut_edges += graph.degree(v) as i64 - 2 * old_cuts;
        self.cuts_at[v] = graph.degree(v) as i64 - old_cuts;
        self.partition[v] = !old;
        if old {
            self.size_a -= 1
        } else {
            self.size_a += 1
        }
    }
    pub fn swap_score(&self, graph: &Graph, a: usize, b: usize, alpha: f64) -> f64 {
        if self.partition[a] == self.partition[b] || a == b {
            return self.score(alpha);
        }
        let adjacent = graph.neighbors(a).binary_search(&b).is_ok() as i64;
        let delta = graph.degree(a) as i64 - 2 * self.cuts_at[a] + graph.degree(b) as i64
            - 2 * self.cuts_at[b]
            + 2 * adjacent;
        let d = self.size_a as i64 - self.size_b() as i64;
        (self.cut_edges + delta) as f64 + alpha * d as f64 * d as f64
    }
    pub fn apply_swap(&mut self, graph: &Graph, a: usize, b: usize) {
        debug_assert_ne!(self.partition[a], self.partition[b]);
        self.apply_flip(graph, a);
        self.apply_flip(graph, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deltas_match() {
        let g = Graph::from_edges(4, vec![[0, 1], [1, 2], [2, 3], [0, 3]]).unwrap();
        for bits in 0..16 {
            let p = (0..4).map(|i| bits & (1 << i) != 0).collect();
            let s = PartitionState::new(&g, p).unwrap();
            for v in 0..4 {
                let mut n = s.clone();
                let expected = s.flip_score(&g, v, 0.05);
                n.apply_flip(&g, v);
                assert_eq!(expected, n.score(0.05));
            }
            for a in 0..4 {
                for b in 0..4 {
                    if s.partition[a] != s.partition[b] {
                        let mut n = s.clone();
                        let expected = s.swap_score(&g, a, b, 0.05);
                        n.apply_swap(&g, a, b);
                        assert_eq!(expected, n.score(0.05));
                    }
                }
            }
        }
    }
}
