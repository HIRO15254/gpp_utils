use super::Graph;
use crate::error::{Error, Result};

#[derive(Clone, Debug)]
/// Job-local partition and incremental objective cache.
///
/// All evaluation and update calls must use the same immutable graph passed to
/// [`Self::new`]. Build a new state after rebuilding a graph's topology. Vertices
/// passed to low-level move methods must be in range; swap endpoints must belong
/// to different groups. The experiment runner validates and maintains these
/// preconditions for callers using the high-level API.
pub struct PartitionState {
    partition: Vec<bool>,
    cut_edges: i64,
    size_a: usize,
    cuts_at: Vec<i64>,
}

impl PartitionState {
    pub fn new(graph: &Graph, partition: Vec<bool>) -> Result<Self> {
        if partition.len() != graph.node_count() {
            return Err(Error::msg("partition length does not match graph"));
        }
        let mut cuts_at = vec![0; graph.node_count()];
        let mut cut_edges = 0;
        for &[a, b] in graph.edges() {
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
    pub(crate) fn cuts_at(&self) -> &[i64] {
        &self.cuts_at
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
    /// Score after flipping `v`. `super::descent` relies on this exact
    /// expression `(cut + gain) as f64 + alpha * d as f64 * d as f64`.
    #[inline]
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
        let partition = &self.partition;
        // Same length as `partition`, so one bounds check covers both.
        let cuts_at = &mut self.cuts_at[..partition.len()];
        for &u in graph.neighbors(v) {
            // The edge to `u` becomes cut (+1) when `u` is on the side that
            // `v` leaves and uncut (-1) otherwise; branch-free.
            cuts_at[u] += 2 * i64::from(partition[u] == old) - 1;
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
    /// Score after swapping `a` and `b`; `super::descent` bounds it with the
    /// same expression without the adjacency term.
    #[inline]
    pub fn swap_score(&self, graph: &Graph, a: usize, b: usize, alpha: f64) -> f64 {
        if self.partition[a] == self.partition[b] || a == b {
            return self.score(alpha);
        }
        self.swap_score_across(graph, a, b, alpha)
    }
    /// [`Self::swap_score`] of vertices on different sides (hence `a != b`),
    /// which the caller guarantees; the same expression without the check.
    #[inline(always)]
    pub(crate) fn swap_score_across(&self, graph: &Graph, a: usize, b: usize, alpha: f64) -> f64 {
        debug_assert_ne!(self.partition[a], self.partition[b]);
        let adjacent = graph.has_edge(a, b) as i64;
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

    use rand::Rng;
    use rand_mt::Mt19937GenRand64;

    /// Graphs with isolated vertices, stars, cliques, paths, the bit-matrix
    /// word boundaries (63, 64, 65, 128, 129 vertices) and random graphs of
    /// several densities with up to a few hundred vertices.
    fn reference_graphs() -> Vec<Graph> {
        let complete = |n: usize| {
            (0..n)
                .flat_map(|a| (a + 1..n).map(move |b| [a, b]))
                .collect::<Vec<_>>()
        };
        let random = |n: usize, p: f64, seed: u64| {
            let mut rng = Mt19937GenRand64::new(seed);
            let mut edges = Vec::new();
            for a in 0..n {
                for b in a + 1..n {
                    if rng.r#gen::<f64>() < p {
                        edges.push([a, b]);
                    }
                }
            }
            Graph::from_edges(n, edges).unwrap()
        };
        let mut graphs = vec![
            Graph::from_edges(2, vec![]).unwrap(),
            Graph::from_edges(9, vec![]).unwrap(),
            Graph::from_edges(2, vec![[0, 1]]).unwrap(),
            Graph::from_edges(17, (1..17).map(|v| [0, v]).collect()).unwrap(),
            Graph::from_edges(130, (0..129).map(|v| [129, v]).collect()).unwrap(),
            Graph::from_edges(12, complete(12)).unwrap(),
            Graph::from_edges(41, complete(41)).unwrap(),
            Graph::from_edges(66, (0..65).map(|v| [v, v + 1]).collect()).unwrap(),
        ];
        // A clique, a star and isolated vertices in one graph.
        let mut mixed = complete(6);
        mixed.extend((7..20).map(|v| [6, v]));
        graphs.push(Graph::from_edges(24, mixed).unwrap());
        for (n, p, seed) in [
            (63, 0.1, 1),
            (64, 0.2, 2),
            (65, 0.05, 3),
            (128, 0.08, 4),
            (129, 0.5, 5),
            (200, 0.02, 6),
            (300, 0.03, 7),
            (301, 0.01, 8),
        ] {
            graphs.push(random(n, p, seed));
        }
        graphs
    }

    /// Cut counts, cut edges and group-A size recomputed from the edge list.
    fn naive_counts(graph: &Graph, partition: &[bool]) -> (Vec<i64>, i64, usize) {
        let mut cuts_at = vec![0; partition.len()];
        let mut cut_edges = 0;
        for &[a, b] in graph.edges() {
            if partition[a] != partition[b] {
                cuts_at[a] += 1;
                cuts_at[b] += 1;
                cut_edges += 1;
            }
        }
        (cuts_at, cut_edges, partition.iter().filter(|&&x| x).count())
    }

    fn assert_state_matches_reference(graph: &Graph, state: &PartitionState, context: &str) {
        let (cuts_at, cut_edges, size_a) = naive_counts(graph, &state.partition);
        assert_eq!(state.cuts_at, cuts_at, "cuts_at: {context}");
        assert_eq!(state.cut_edges, cut_edges, "cut_edges: {context}");
        assert_eq!(state.size_a, size_a, "size_a: {context}");
        let fresh = PartitionState::new(graph, state.partition.clone()).unwrap();
        assert_eq!(state.partition, fresh.partition, "{context}");
        assert_eq!(state.cuts_at, fresh.cuts_at, "{context}");
        assert_eq!(state.cut_edges, fresh.cut_edges, "{context}");
        assert_eq!(state.size_a, fresh.size_a, "{context}");
    }

    /// `flip_score(v)` has the bits of applying the flip and of the edge-list
    /// score of the flipped partition.
    fn assert_flip_score(graph: &Graph, state: &PartitionState, v: usize, alpha: f64) {
        let predicted = state.flip_score(graph, v, alpha);
        let mut applied = state.clone();
        applied.apply_flip(graph, v);
        let mut partition = state.partition.clone();
        partition[v] = !partition[v];
        assert_eq!(
            predicted.to_bits(),
            applied.score(alpha).to_bits(),
            "flip {v}"
        );
        assert_eq!(
            predicted.to_bits(),
            graph.score(&partition, alpha).to_bits(),
            "flip {v}"
        );
    }

    /// `swap_score(a, b)` (and `swap_score_across` for different sides) has the
    /// bits of applying the swap and of the edge-list score of the swapped
    /// partition; on one side or for `a == b` it is the current score.
    fn assert_swap_score(graph: &Graph, state: &PartitionState, a: usize, b: usize, alpha: f64) {
        let predicted = state.swap_score(graph, a, b, alpha);
        if state.partition[a] == state.partition[b] {
            assert_eq!(predicted.to_bits(), state.score(alpha).to_bits());
            return;
        }
        assert_eq!(
            predicted.to_bits(),
            state.swap_score_across(graph, a, b, alpha).to_bits()
        );
        let mut applied = state.clone();
        applied.apply_swap(graph, a, b);
        let mut partition = state.partition.clone();
        partition.swap(a, b);
        assert_eq!(
            predicted.to_bits(),
            applied.score(alpha).to_bits(),
            "swap {a} {b}"
        );
        assert_eq!(
            predicted.to_bits(),
            graph.score(&partition, alpha).to_bits(),
            "swap {a} {b}"
        );
    }

    /// Independent reference for the incremental state: random flip and swap
    /// sequences compared after every move with the edge list and with a
    /// state rebuilt from the partition, and every predicted move score
    /// compared with the applied move and the edge-list score (bits).
    #[test]
    fn random_moves_match_rebuilt_state_and_edge_list_scores() {
        let alphas = [0.05, 0.0, -0.0, 0.125, 3.0, 1e300];
        let mut rng = Mt19937GenRand64::new(0x5eed);
        let (mut flips, mut swaps, mut adjacent_swaps) = (0u64, 0u64, 0u64);
        for (index, graph) in reference_graphs().iter().enumerate() {
            let n = graph.node_count();
            for start in 0..3 {
                let mut partition: Vec<bool> = (0..n).map(|_| rng.r#gen()).collect();
                if start == 1 {
                    // Balanced, as Swap runs start.
                    partition = (0..n).map(|v| v < n / 2).collect();
                    for i in (1..n).rev() {
                        partition.swap(i, rng.gen_range(0..=i));
                    }
                } else if start == 2 {
                    partition.fill(true);
                }
                let mut state = PartitionState::new(graph, partition).unwrap();
                let exhaustive = n <= 24;
                for step in 0..300 {
                    let context = format!("graph {index} start {start} step {step}");
                    assert_state_matches_reference(graph, &state, &context);
                    let alpha = alphas[step % alphas.len()];
                    if exhaustive {
                        for v in 0..n {
                            assert_flip_score(graph, &state, v, alpha);
                        }
                        for a in 0..n {
                            for b in 0..n {
                                assert_swap_score(graph, &state, a, b, alpha);
                            }
                        }
                    } else {
                        for _ in 0..40 {
                            assert_flip_score(graph, &state, rng.gen_range(0..n), alpha);
                            let (a, b) = (rng.gen_range(0..n), rng.gen_range(0..n));
                            assert_swap_score(graph, &state, a, b, alpha);
                        }
                    }
                    let side_a: Vec<usize> = (0..n).filter(|&v| state.partition[v]).collect();
                    let side_b: Vec<usize> = (0..n).filter(|&v| !state.partition[v]).collect();
                    if side_a.is_empty() || side_b.is_empty() || rng.gen_range(0..2) == 0 {
                        let v = rng.gen_range(0..n);
                        assert_flip_score(graph, &state, v, alpha);
                        state.apply_flip(graph, v);
                        flips += 1;
                    } else {
                        let a = side_a[rng.gen_range(0..side_a.len())];
                        // Prefer adjacent pairs half of the time.
                        let b = match graph.neighbors(a).iter().find(|&&u| !state.partition[u]) {
                            Some(&u) if rng.gen_range(0..2) == 0 => u,
                            _ => side_b[rng.gen_range(0..side_b.len())],
                        };
                        adjacent_swaps += u64::from(graph.neighbors(a).contains(&b));
                        assert_swap_score(graph, &state, a, b, alpha);
                        state.apply_swap(graph, a, b);
                        swaps += 1;
                    }
                }
                assert_state_matches_reference(graph, &state, "end");
            }
        }
        assert!(flips > 1000 && swaps > 1000 && adjacent_swaps > 300);
    }
}
