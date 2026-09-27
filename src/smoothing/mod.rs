#![allow(clippy::too_many_arguments)]

use crate::error::{Error, Result};
use crate::experiment::config::{Neighborhood, SmoothingSpec};
use crate::graph_partition::{Graph, Move, PartitionState};
use crate::optimization::CancellationToken;
use rand::Rng;
use rand_mt::Mt19937GenRand64;
use std::cell::RefCell;
use std::collections::BTreeSet;

pub fn moves(state: &PartitionState, neighborhood: Neighborhood) -> Vec<Move> {
    moves_cancellable(state, neighborhood, &CancellationToken::default())
        .expect("a fresh cancellation token cannot be cancelled")
}

pub fn moves_cancellable(
    state: &PartitionState,
    neighborhood: Neighborhood,
    cancel: &CancellationToken,
) -> Result<Vec<Move>> {
    match neighborhood {
        Neighborhood::Flip => Ok((0..state.partition().len()).map(Move::Flip).collect()),
        Neighborhood::Swap => {
            let side_a: Vec<_> = state
                .partition()
                .iter()
                .enumerate()
                .filter_map(|(v, &side)| side.then_some(v))
                .collect();
            let side_b: Vec<_> = state
                .partition()
                .iter()
                .enumerate()
                .filter_map(|(v, &side)| (!side).then_some(v))
                .collect();
            let mut out = Vec::with_capacity(side_a.len() * side_b.len());
            for &a in &side_a {
                for &b in &side_b {
                    if out.len() & 1023 == 0 {
                        cancel.check()?;
                    }
                    out.push(Move::Swap(a, b));
                }
            }
            Ok(out)
        }
    }
}
pub fn apply(state: &mut PartitionState, graph: &Graph, mv: Move) {
    match mv {
        Move::Flip(v) => state.apply_flip(graph, v),
        Move::Swap(a, b) => state.apply_swap(graph, a, b),
    }
}
pub fn move_score(state: &PartitionState, graph: &Graph, mv: Move, alpha: f64) -> f64 {
    match mv {
        Move::Flip(v) => state.flip_score(graph, v, alpha),
        Move::Swap(a, b) => state.swap_score(graph, a, b, alpha),
    }
}

pub fn max_random_k(n: usize, neighborhood: Neighborhood) -> u128 {
    match neighborhood {
        Neighborhood::Flip => n as u128 + (n as u128 * (n.saturating_sub(1)) as u128) / 2,
        Neighborhood::Swap => {
            let h = n / 2;
            let m = (h * h) as u128;
            let c = (h * (h.saturating_sub(1)) / 2) as u128;
            m + c * c
        }
    }
}

pub fn validate(spec: &SmoothingSpec, n: usize, neighborhood: Neighborhood) -> Result<()> {
    match spec {
        SmoothingSpec::RandomKAverage { k } if *k == 0 => {
            Err(Error::msg("random_k_average k must be at least 1"))
        }
        SmoothingSpec::RandomKAverage { k } if *k as u128 > max_random_k(n, neighborhood) => Err(
            Error::msg("random_k_average k exceeds distance-one plus distance-two neighborhood"),
        ),
        _ => Ok(()),
    }
}

/// Smoothed evaluation of `state` (`docs/algorithms.md`, section 3).
///
/// The distance-one neighbors are numbered in the order of [`moves`]: Flip
/// neighbor `v` flips vertex `v`, and Swap neighbor `i` swaps `A[i / |B|]` with
/// `B[i % |B|]`, where `A` and `B` list the vertices of each side in increasing
/// order. `random_k_average` takes the first `min(k, M)` entries of a partial
/// Fisher-Yates shuffle of `0..M` (step `i` draws `gen_range(i..M)`) in
/// shuffled order and, if `k > M`, then the distance-two neighbors whose
/// canonical ordinals Floyd's algorithm samples, in increasing ordinal order.
///
/// Neither the move list nor candidate states are materialized. The shuffle is
/// replayed on a reused identity permutation with the same draws, and every
/// score comes from the same integer cut and size counts and the same
/// floating-point expression as [`move_score`] or [`PartitionState::score`] of
/// the candidate, added in the same order.
pub fn evaluate(
    state: &PartitionState,
    graph: &Graph,
    alpha: f64,
    neighborhood: Neighborhood,
    spec: &SmoothingSpec,
    rng: Option<&mut Mt19937GenRand64>,
    cancel: &CancellationToken,
    evaluations: &mut u64,
) -> Result<f64> {
    validate(spec, graph.node_count(), neighborhood)?;
    let real = state.score(alpha);
    let k = match *spec {
        SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 } => {
            *evaluations += 1;
            return Ok(real);
        }
        SmoothingSpec::RandomKAverage { k } => Some(k),
        SmoothingSpec::AllAverage | SmoothingSpec::WeightedAverage { .. } => None,
    };
    let average = Average {
        n: graph.node_count(),
        neighborhood,
        spec,
        k,
        real,
    };
    SCRATCH.with(|scratch| {
        let Scratch { shuffle, swap } = &mut *scratch.borrow_mut();
        match neighborhood {
            Neighborhood::Flip => {
                let mut neighbors = FlipNeighbors::new(state, graph, alpha);
                average.evaluate(&mut neighbors, rng, cancel, evaluations, shuffle)
            }
            Neighborhood::Swap => {
                let mut neighbors = SwapNeighbors::new(state, graph, alpha, swap);
                if neighbors.count() > 0 {
                    // `moves_cancellable` checks before it lists the first swap.
                    cancel.check()?;
                }
                average.evaluate(&mut neighbors, rng, cancel, evaluations, shuffle)
            }
        }
    })
}

/// Buffers reused by [`evaluate`] on this thread. Only the identity
/// permutation of [`Shuffle`] carries over between calls.
struct Scratch {
    shuffle: Shuffle,
    swap: SwapScratch,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = const {
        RefCell::new(Scratch {
            shuffle: Shuffle {
                identity: Vec::new(),
                chosen: Vec::new(),
            },
            swap: SwapScratch {
                side_a: Vec::new(),
                side_b: Vec::new(),
                slot: Vec::new(),
                gain: Vec::new(),
            },
        })
    };
}

/// Replays a partial Fisher-Yates shuffle of `0..m` in `O(take)` time.
/// Longest identity permutation (and draw buffer) kept between calls, in
/// entries (32 MiB of `usize`). Longer ones are rebuilt on each call, as the
/// previous version rebuilt its index vector, so a huge Swap neighborhood does
/// not pin its buffer to the thread for the thread's lifetime.
/// Tests use a small limit so both the retained and the rebuilt path run.
const RETAINED_ENTRIES: usize = if cfg!(test) { 1 << 12 } else { 1 << 22 };

struct Shuffle {
    /// The identity permutation `0..identity.len()` between calls.
    identity: Vec<usize>,
    /// The draws, then the chosen entries.
    chosen: Vec<usize>,
}

impl Shuffle {
    /// The first `take` entries of `0..m` after the shuffle steps
    /// `i = 0..take`, where step `i` swaps entries `i` and `rng.gen_range(i..m)`.
    /// Makes exactly these draws and checks cancellation before draw `i` when
    /// `i % 1024 == 0`.
    fn first(
        &mut self,
        rng: &mut Mt19937GenRand64,
        m: usize,
        take: usize,
        cancel: &CancellationToken,
    ) -> Result<&[usize]> {
        let Self { identity, chosen } = self;
        // The draws do not depend on the entries, so they come first; an
        // early return leaves the identity untouched.
        if chosen.capacity() > RETAINED_ENTRIES && take <= RETAINED_ENTRIES {
            *chosen = Vec::new();
        }
        chosen.clear();
        for i in 0..take {
            if i & 1023 == 0 {
                cancel.check()?;
            }
            chosen.push(rng.gen_range(i..m));
        }
        if identity.len() < m {
            let len = identity.len();
            identity.extend(len..m);
        }
        for (i, slot) in chosen.iter_mut().enumerate() {
            identity.swap(i, *slot);
            // Later steps only touch entries after `i`, so entry `i` is final.
            *slot = identity[i];
        }
        // Restore the identity. The steps changed the first `take` entries and
        // the drawn entries after them. The first step that draws such an entry
        // `j` moves its value `j` to a final place, so the latter are exactly
        // the chosen values of at least `take`.
        for (i, &value) in chosen.iter().enumerate() {
            identity[i] = i;
            identity[value] = value;
        }
        if identity.len() > RETAINED_ENTRIES {
            *identity = Vec::new();
        }
        Ok(chosen)
    }
}

/// Distance-one and distance-two neighbors of a state in canonical order.
trait Neighbors {
    /// Number `M` of distance-one neighbors.
    fn count(&self) -> usize;
    /// Called before the scores of a random sample of `samples` neighbors.
    fn prepare(&mut self, _samples: usize) {}
    /// [`move_score`] of distance-one neighbor `ordinal`.
    fn score(&self, ordinal: usize) -> f64;
    /// Sum of all distance-one scores, added to `0.0` in canonical order.
    fn sum(&mut self, cancel: &CancellationToken) -> Result<f64>;
    /// Objective of distance-two neighbor `ordinal`.
    fn distance_two_score(&self, ordinal: usize) -> f64;
}

/// The averaging rule of [`evaluate`] for a nontrivial specification.
struct Average<'a> {
    n: usize,
    neighborhood: Neighborhood,
    spec: &'a SmoothingSpec,
    /// `Some(k)` for `random_k_average`.
    k: Option<usize>,
    real: f64,
}

impl Average<'_> {
    fn evaluate(
        &self,
        neighbors: &mut impl Neighbors,
        rng: Option<&mut Mt19937GenRand64>,
        cancel: &CancellationToken,
        evaluations: &mut u64,
        shuffle: &mut Shuffle,
    ) -> Result<f64> {
        let m = neighbors.count();
        if m == 0 {
            *evaluations += 1;
            return Ok(self.real);
        }
        let (total, count) = match self.k {
            None => (neighbors.sum(cancel)?, m),
            Some(k) => {
                let r = rng.ok_or_else(|| Error::msg("random smoothing requires RNG"))?;
                let take = k.min(m);
                neighbors.prepare(k);
                let mut total = 0.0;
                for (q, &ordinal) in shuffle.first(r, m, take, cancel)?.iter().enumerate() {
                    if q & 1023 == 0 {
                        cancel.check()?;
                    }
                    total += neighbors.score(ordinal);
                }
                if k > take {
                    let needed = k - take;
                    let distance_two =
                        (max_random_k(self.n, self.neighborhood) as usize).saturating_sub(m);
                    // Floyd's algorithm samples exact canonical ordinals without replacement.
                    let mut ordinals = BTreeSet::new();
                    for j in distance_two - needed..distance_two {
                        if j & 1023 == 0 {
                            cancel.check()?;
                        }
                        let candidate = r.gen_range(0..=j);
                        if !ordinals.insert(candidate) {
                            ordinals.insert(j);
                        }
                    }
                    for (q, ordinal) in ordinals.into_iter().enumerate() {
                        if q & 1023 == 0 {
                            cancel.check()?;
                        }
                        total += neighbors.distance_two_score(ordinal);
                    }
                }
                (total, k)
            }
        };
        *evaluations += count as u64;
        let avg = total / count as f64;
        Ok(match *self.spec {
            SmoothingSpec::WeightedAverage { k } => {
                let w = k.min(m) as f64 / m as f64;
                w * avg + (1.0 - w) * self.real
            }
            _ => avg,
        })
    }
}

/// Flip neighbor `v` flips vertex `v`.
struct FlipNeighbors<'a> {
    state: &'a PartitionState,
    graph: &'a Graph,
    alpha: f64,
    cut_edges: i64,
    /// The balance terms of [`PartitionState::flip_score`] for a vertex on
    /// side B (index 0) and side A (index 1). The side sizes after a flip
    /// depend only on the side of the vertex, so each term is computed once by
    /// the same expression; the term of an empty side is never used.
    balance: [f64; 2],
}

impl<'a> FlipNeighbors<'a> {
    fn new(state: &'a PartitionState, graph: &'a Graph, alpha: f64) -> Self {
        let n = state.partition().len();
        let size_a = state.size_a();
        let balance = |a: usize| {
            let d = a as i64 - (n - a) as i64;
            alpha * d as f64 * d as f64
        };
        let from_b = if size_a < n {
            balance(size_a + 1)
        } else {
            f64::NAN
        };
        let from_a = size_a.checked_sub(1).map_or(f64::NAN, balance);
        Self {
            state,
            graph,
            alpha,
            cut_edges: state.cut_edges() as i64,
            balance: [from_b, from_a],
        }
    }

    /// [`PartitionState::flip_score`] of `v`, which is on `side` and has
    /// `cuts_at[v] == cuts_v`.
    fn flip_score(&self, v: usize, side: bool, cuts_v: i64) -> f64 {
        let cut = self.cut_edges + self.graph.degree(v) as i64 - 2 * cuts_v;
        cut as f64 + self.balance[usize::from(side)]
    }
}

impl Neighbors for FlipNeighbors<'_> {
    fn count(&self) -> usize {
        self.state.partition().len()
    }

    fn score(&self, v: usize) -> f64 {
        self.flip_score(v, self.state.partition()[v], self.state.cuts_at()[v])
    }

    fn sum(&mut self, cancel: &CancellationToken) -> Result<f64> {
        let sides = self.state.partition().chunks(1024);
        let cuts = self.state.cuts_at().chunks(1024);
        let mut total = 0.0;
        for (chunk, (sides, cuts)) in sides.zip(cuts).enumerate() {
            cancel.check()?;
            for (v, (&side, &cuts_v)) in (chunk * 1024..).zip(sides.iter().zip(cuts)) {
                total += self.flip_score(v, side, cuts_v);
            }
        }
        Ok(total)
    }

    fn distance_two_score(&self, ordinal: usize) -> f64 {
        let (a, b) = nth_pair(self.graph.node_count(), ordinal);
        score_after_flips(self.state, self.graph, self.alpha, &[a, b])
    }
}

struct SwapScratch {
    /// Side-A (`true`) vertices in increasing order.
    side_a: Vec<usize>,
    /// Side-B (`false`) vertices in increasing order.
    side_b: Vec<usize>,
    /// For [`Neighbors::sum`]: index of each vertex in `side_b`, or
    /// `side_b.len()` for side-A vertices.
    slot: Vec<usize>,
    /// For [`Neighbors::sum`]: `degree - 2 * cuts_at` per `side_b` entry and
    /// one unused trailing slot.
    gain: Vec<i64>,
}

impl SwapScratch {
    /// Makes `side_a` and `side_b` list the vertices of each side of
    /// `partition` in increasing order, as [`moves_cancellable`] does.
    fn list_sides(&mut self, partition: &[bool]) {
        // Branch-free: every vertex is written to both lists, and only the
        // list of its side advances.
        let Self { side_a, side_b, .. } = self;
        side_a.resize(partition.len(), 0);
        side_b.resize(partition.len(), 0);
        let (mut len_a, mut len_b) = (0, 0);
        for (v, &side) in partition.iter().enumerate() {
            side_a[len_a] = v;
            side_b[len_b] = v;
            len_a += usize::from(side);
            len_b += usize::from(!side);
        }
        side_a.truncate(len_a);
        side_b.truncate(len_b);
    }
}

/// Swap neighbor `i` swaps `A[i / |B|]` with `B[i % |B|]`, where `A` and `B`
/// list the vertices of each side in increasing order.
struct SwapNeighbors<'a> {
    state: &'a PartitionState,
    graph: &'a Graph,
    alpha: f64,
    scratch: &'a mut SwapScratch,
    /// Whether `scratch` lists the sides of `state`; otherwise the vertices of
    /// a sample are selected by their rank on their side.
    listed: bool,
}

impl<'a> SwapNeighbors<'a> {
    fn new(
        state: &'a PartitionState,
        graph: &'a Graph,
        alpha: f64,
        scratch: &'a mut SwapScratch,
    ) -> Self {
        Self {
            state,
            graph,
            alpha,
            scratch,
            listed: false,
        }
    }

    fn list_sides(&mut self) {
        if !self.listed {
            self.scratch.list_sides(self.state.partition());
            self.listed = true;
        }
    }
}

/// The vertex with `rank` (from 0) among the vertices on `side`, in increasing
/// order; panics if there is none.
fn select(partition: &[bool], side: bool, mut rank: usize) -> usize {
    // Number of vertices on `side` among 8 to 64 sides: the sides are read as
    // bytes 0 and 1 of 64-bit words, whose byte sums stay below 256.
    let count = |sides: &[bool]| {
        let mut bytes = 0u64;
        for &word in sides.as_chunks::<8>().0 {
            bytes += u64::from_ne_bytes(word.map(u8::from));
        }
        let on_a = (bytes.wrapping_mul(0x0101_0101_0101_0101) >> 56) as usize;
        if side { on_a } else { sides.len() - on_a }
    };
    // Skip whole blocks of 64 and then of 8 sides before the target, then scan.
    let mut start = 0;
    for block in partition.as_chunks::<64>().0 {
        let count = count(block);
        if rank < count {
            break;
        }
        rank -= count;
        start += 64;
    }
    for block in partition[start..].as_chunks::<8>().0 {
        let count = count(block);
        if rank < count {
            break;
        }
        rank -= count;
        start += 8;
    }
    for (v, &x) in (start..).zip(&partition[start..]) {
        let hit = usize::from(x == side);
        if rank < hit {
            return v;
        }
        rank -= hit;
    }
    unreachable!("rank below the side size")
}

impl Neighbors for SwapNeighbors<'_> {
    fn count(&self) -> usize {
        self.state.size_a() * self.state.size_b()
    }

    fn prepare(&mut self, samples: usize) {
        // Selecting both vertices of a sample by rank took about as long as
        // listing the sides of 128 vertices, so only a few samples of a large
        // state are selected.
        if samples.saturating_mul(128) > self.state.partition().len() {
            self.list_sides();
        }
    }

    fn score(&self, ordinal: usize) -> f64 {
        let size_b = self.state.size_b();
        let (a, b) = if self.listed {
            let SwapScratch { side_a, side_b, .. } = &*self.scratch;
            (side_a[ordinal / size_b], side_b[ordinal % size_b])
        } else {
            let partition = self.state.partition();
            (
                select(partition, true, ordinal / size_b),
                select(partition, false, ordinal % size_b),
            )
        };
        self.state.swap_score(self.graph, a, b, self.alpha)
    }

    fn sum(&mut self, cancel: &CancellationToken) -> Result<f64> {
        self.list_sides();
        // `swap_score(a, b)` is `(cut_edges + delta) as f64 + balance`, where the
        // balance term does not depend on the pair and the integer `delta` is
        // `gain(a) + gain(b) + 2 * adjacent` with `gain(v) = degree(v) - 2 * cuts_at(v)`.
        let (state, graph, alpha) = (self.state, self.graph, self.alpha);
        let SwapScratch {
            side_a,
            side_b,
            slot,
            gain,
        } = &mut *self.scratch;
        let cuts = state.cuts_at();
        slot.clear();
        slot.resize(state.partition().len(), side_b.len());
        gain.clear();
        for (i, &b) in side_b.iter().enumerate() {
            slot[b] = i;
            gain.push(graph.degree(b) as i64 - 2 * cuts[b]);
        }
        gain.push(0);
        let d = state.size_a() as i64 - state.size_b() as i64;
        let balance = alpha * d as f64 * d as f64;
        let cut_edges = state.cut_edges() as i64;
        let mut total = 0.0;
        for &a in side_a.iter() {
            cancel.check()?;
            let base = cut_edges + graph.degree(a) as i64 - 2 * cuts[a];
            // Temporarily add `2 * adjacent` to the gains of the neighbors of `a`.
            for &u in graph.neighbors(a) {
                gain[slot[u]] += 2;
            }
            for &gain_b in &gain[..side_b.len()] {
                total += (base + gain_b) as f64 + balance;
            }
            for &u in graph.neighbors(a) {
                gain[slot[u]] -= 2;
            }
        }
        Ok(total)
    }

    fn distance_two_score(&self, ordinal: usize) -> f64 {
        assert!(
            self.listed,
            "distance-two samples follow all distance-one samples"
        );
        let SwapScratch { side_a, side_b, .. } = &*self.scratch;
        let combinations = side_a.len() * (side_a.len() - 1) / 2;
        let (ai, aj) = nth_pair(side_a.len(), ordinal / combinations);
        let (bi, bj) = nth_pair(side_b.len(), ordinal % combinations);
        let flips = [side_a[ai], side_b[bi], side_a[aj], side_b[bj]];
        score_after_flips(self.state, self.graph, self.alpha, &flips)
    }
}

/// [`PartitionState::score`] after [`PartitionState::apply_flip`] of the
/// distinct `vertices` in order, from the same integer updates without a copy.
fn score_after_flips(state: &PartitionState, graph: &Graph, alpha: f64, vertices: &[usize]) -> f64 {
    let partition = state.partition();
    let cuts = state.cuts_at();
    let mut cut_edges = state.cut_edges() as i64;
    let mut size_a = state.size_a();
    for (t, &v) in vertices.iter().enumerate() {
        // `cuts_at[v]` when `v` is flipped: flipping an earlier neighbor `u`
        // moved the edge `(u, v)` into or out of the cut.
        let mut cuts_v = cuts[v];
        for &u in &vertices[..t] {
            if graph.neighbors(u).binary_search(&v).is_ok() {
                if partition[v] != partition[u] {
                    cuts_v -= 1;
                } else {
                    cuts_v += 1;
                }
            }
        }
        cut_edges += graph.degree(v) as i64 - 2 * cuts_v;
        if partition[v] {
            size_a -= 1;
        } else {
            size_a += 1;
        }
    }
    let d = size_a as i64 - (partition.len() - size_a) as i64;
    cut_edges as f64 + alpha * d as f64 * d as f64
}

/// The pair `(a, b)`, `a < b < n`, at position `ordinal` of the lexicographic
/// order; panics if there is none.
fn nth_pair(n: usize, ordinal: usize) -> (usize, usize) {
    if n < 2 || ordinal >= n * (n - 1) / 2 {
        unreachable!("validated distance-two ordinal")
    }
    // Row `a` lists `(a, a + 1..n)` from position `a * (2n - a - 1) / 2`.
    let start = |a: usize| a * (2 * n - a - 1) / 2;
    // The row is the last one starting at or before `ordinal`: start(lo) <=
    // ordinal < start(hi), where start(n - 1) is the pair count.
    let (mut lo, mut hi) = (0, n - 1);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if start(mid) <= ordinal {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo, lo + 1 + ordinal - start(lo))
}

/// Test-only view of the reusable scratch: (identity length, identity intact,
/// RefCell free), used to check that no call leaves the permutation scrambled.
#[cfg(test)]
pub(crate) fn scratch_status() -> (usize, bool, bool) {
    SCRATCH.with(|scratch| match scratch.try_borrow() {
        Ok(s) => (
            s.shuffle.identity.len(),
            s.shuffle.identity.iter().enumerate().all(|(i, &v)| i == v),
            true,
        ),
        Err(_) => (0, false, false),
    })
}

// The smoothing module as of 2fc65d9, frozen for the comparisons below (the
// same code as `solvers::test_reference::smoothing_e4b6a1c`).
#[cfg(test)]
#[allow(dead_code)]
mod test_reference;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_is_retained_up_to_the_limit_and_rebuilt_beyond_it() {
        let cancel = CancellationToken::default();
        for (n, retained) in [(100, true), (140, false), (100, true)] {
            let edges = (0..n - 1).map(|v| [v, v + 1]).collect();
            let graph = Graph::from_edges(n, edges).unwrap();
            let state = PartitionState::new(&graph, (0..n).map(|v| v % 2 == 0).collect()).unwrap();
            let m = n / 2 * (n / 2);
            assert_eq!(m <= RETAINED_ENTRIES, retained);
            let mut rng = Mt19937GenRand64::new(3);
            let mut evaluations = 0;
            evaluate(
                &state,
                &graph,
                0.05,
                Neighborhood::Swap,
                &SmoothingSpec::RandomKAverage { k: 7 },
                Some(&mut rng),
                &cancel,
                &mut evaluations,
            )
            .unwrap();
            assert_eq!(evaluations, 7);
            let (len, intact, free) = scratch_status();
            assert!(intact && free);
            assert_eq!(len >= m, retained, "n {n}: identity length {len}");
            assert!(len <= RETAINED_ENTRIES);
        }
    }
    use rand_mt::Mt19937GenRand64;

    #[test]
    fn full_random_k_covers_every_flip_neighbor_at_distances_one_and_two() {
        let graph = Graph::from_edges(4, vec![[0, 1], [1, 2], [2, 3]]).unwrap();
        let state = PartitionState::new(&graph, vec![true, false, true, false]).unwrap();
        let expected = {
            let mut sum = 0.0;
            let mut count = 0;
            for a in 0..4 {
                sum += state.flip_score(&graph, a, 0.2);
                count += 1;
                for b in a + 1..4 {
                    let mut candidate = state.clone();
                    candidate.apply_flip(&graph, a);
                    candidate.apply_flip(&graph, b);
                    sum += candidate.score(0.2);
                    count += 1;
                }
            }
            sum / count as f64
        };
        let mut rng = Mt19937GenRand64::new(7);
        let mut evaluations = 0;
        let actual = evaluate(
            &state,
            &graph,
            0.2,
            Neighborhood::Flip,
            &SmoothingSpec::RandomKAverage { k: 10 },
            Some(&mut rng),
            &CancellationToken::default(),
            &mut evaluations,
        )
        .unwrap();
        assert!((actual - expected).abs() < 1e-12);
        assert_eq!(evaluations, 10);
    }

    #[test]
    fn full_random_k_swap_has_four_distance_one_and_one_distance_two_states() {
        let graph = Graph::from_edges(4, vec![[0, 1], [1, 2], [2, 3], [0, 3]]).unwrap();
        let state = PartitionState::new(&graph, vec![true, true, false, false]).unwrap();
        let mut rng = Mt19937GenRand64::new(9);
        let mut evaluations = 0;
        evaluate(
            &state,
            &graph,
            0.2,
            Neighborhood::Swap,
            &SmoothingSpec::RandomKAverage { k: 5 },
            Some(&mut rng),
            &CancellationToken::default(),
            &mut evaluations,
        )
        .unwrap();
        assert_eq!(evaluations, 5);
    }

    #[test]
    fn select_finds_every_rank_on_both_sides() {
        let mut rng = Mt19937GenRand64::new(3);
        let sizes = (0..=80).chain([127, 128, 129, 200, 255, 256, 257, 500, 513]);
        for n in sizes {
            for density in [0.0, 0.1, 0.5, 0.9, 1.0] {
                let partition: Vec<bool> = (0..n).map(|_| rng.gen_bool(density)).collect();
                for side in [true, false] {
                    let vertices = (0..n).filter(|&v| partition[v] == side);
                    for (rank, v) in vertices.enumerate() {
                        assert_eq!(select(&partition, side, rank), v, "{n} {side} {rank}");
                    }
                }
            }
        }
    }

    /// The row-by-row enumeration of `nth_pair` at e4b6a1c.
    fn nth_pair_by_rows(n: usize, mut ordinal: usize) -> (usize, usize) {
        for a in 0..n - 1 {
            let row = n - a - 1;
            if ordinal < row {
                return (a, a + 1 + ordinal);
            }
            ordinal -= row;
        }
        unreachable!("validated distance-two ordinal")
    }

    #[test]
    fn nth_pair_matches_the_lexicographic_order_and_rejects_other_ordinals() {
        for n in 0..=70usize {
            let pairs = (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b)));
            let mut count = 0;
            for (ordinal, pair) in pairs.enumerate() {
                assert_eq!(nth_pair(n, ordinal), pair);
                assert_eq!(nth_pair_by_rows(n, ordinal), pair);
                count += 1;
            }
            assert!(std::panic::catch_unwind(|| nth_pair(n, count)).is_err());
            if n > 0 {
                assert!(std::panic::catch_unwind(|| nth_pair_by_rows(n, count)).is_err());
            }
        }
        let mut rng = Mt19937GenRand64::new(5);
        for n in [100, 499, 500, 1000, 4097] {
            let pairs = n * (n - 1) / 2;
            let ordinals = (0..500).map(|_| rng.gen_range(0..pairs));
            for ordinal in ordinals.chain([0, 1, n - 2, n - 1, pairs - 2, pairs - 1]) {
                assert_eq!(nth_pair(n, ordinal), nth_pair_by_rows(n, ordinal));
            }
        }
    }

    #[test]
    fn all_smoothing_matches_frozen_full_reference() {
        let isolated_64 = Graph::from_edges(64, vec![]).unwrap();
        let complete_8 = Graph::from_edges(
            8,
            (0..8)
                .flat_map(|a| (a + 1..8).map(move |b| [a, b]))
                .collect(),
        )
        .unwrap();
        for graph in [&isolated_64, &complete_8] {
            let n = graph.node_count();
            for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
                let partition = match neighborhood {
                    Neighborhood::Flip => (0..n).map(|v| v % 3 == 0 || v % 7 == 0).collect(),
                    Neighborhood::Swap => (0..n).map(|v| v < n / 2).collect(),
                };
                let state = PartitionState::new(graph, partition).unwrap();
                let first_count = match neighborhood {
                    Neighborhood::Flip => n,
                    Neighborhood::Swap => n * n / 4,
                };
                let maximum = max_random_k(n, neighborhood) as usize;
                let mut specs = vec![
                    SmoothingSpec::None,
                    SmoothingSpec::WeightedAverage { k: 0 },
                    SmoothingSpec::WeightedAverage { k: 3 },
                    SmoothingSpec::RandomKAverage { k: 1 },
                    SmoothingSpec::RandomKAverage {
                        k: (first_count + 3).min(maximum),
                    },
                ];
                // Exercise full distance-one + distance-two enumeration without
                // making the n=64 Swap case dominate the unit-test runtime.
                if n == 8 {
                    specs.push(SmoothingSpec::RandomKAverage { k: maximum });
                }
                for spec in specs {
                    for seed in [0, 1, 0x9e37_79b9_7f4a_7c15] {
                        let mut actual_rng = Mt19937GenRand64::new(seed);
                        let mut frozen_rng = Mt19937GenRand64::new(seed);
                        let mut actual_evals = 0;
                        let mut frozen_evals = 0;
                        let actual = evaluate(
                            &state,
                            graph,
                            0.05,
                            neighborhood,
                            &spec,
                            Some(&mut actual_rng),
                            &CancellationToken::default(),
                            &mut actual_evals,
                        )
                        .unwrap();
                        let frozen = super::test_reference::evaluate(
                            &state,
                            graph,
                            0.05,
                            neighborhood,
                            &spec,
                            Some(&mut frozen_rng),
                            &CancellationToken::default(),
                            &mut frozen_evals,
                        )
                        .unwrap();
                        assert_eq!(
                            actual.to_bits(),
                            frozen.to_bits(),
                            "bits: n={n} neighborhood={neighborhood:?} spec={spec:?} seed={seed}"
                        );
                        assert_eq!(
                            actual_evals, frozen_evals,
                            "evaluations: n={n} neighborhood={neighborhood:?} spec={spec:?} seed={seed}"
                        );
                        for draw in 0..624 {
                            assert_eq!(
                                actual_rng.next_u64(),
                                frozen_rng.next_u64(),
                                "rng draw {draw}: n={n} neighborhood={neighborhood:?} spec={spec:?} seed={seed}"
                            );
                        }
                    }
                }
            }
        }
    }

    // Frozen, deliberately independent distance-one evaluator from 2fc65d9.
    // Keep the materialized nested-loop order here when production changes.
    fn frozen_distance_one(
        state: &PartitionState,
        graph: &Graph,
        alpha: f64,
        neighborhood: Neighborhood,
        spec: &SmoothingSpec,
        rng: Option<&mut Mt19937GenRand64>,
        evaluations: &mut u64,
    ) -> f64 {
        let first: Vec<Move> = match neighborhood {
            Neighborhood::Flip => (0..state.partition().len()).map(Move::Flip).collect(),
            Neighborhood::Swap => {
                let side_a = state
                    .partition()
                    .iter()
                    .enumerate()
                    .filter_map(|(v, &x)| x.then_some(v))
                    .collect::<Vec<_>>();
                let side_b = state
                    .partition()
                    .iter()
                    .enumerate()
                    .filter_map(|(v, &x)| (!x).then_some(v))
                    .collect::<Vec<_>>();
                side_a
                    .iter()
                    .flat_map(|&a| side_b.iter().map(move |&b| Move::Swap(a, b)))
                    .collect()
            }
        };
        let k = match spec {
            SmoothingSpec::RandomKAverage { k } => Some(*k),
            _ => None,
        };
        let take = k.map_or(first.len(), |x| x.min(first.len()));
        let mut indices = (0..first.len()).collect::<Vec<_>>();
        if k.is_some() {
            let rng = rng.unwrap();
            for i in 0..take {
                let j = rng.gen_range(i..indices.len());
                indices.swap(i, j);
            }
        }
        let mut total = 0.0;
        for &i in &indices[..take] {
            total += move_score(state, graph, first[i], alpha);
        }
        *evaluations += take as u64;
        let average = total / take as f64;
        match spec {
            SmoothingSpec::WeightedAverage { k } => {
                let w = (*k).min(first.len()) as f64 / first.len() as f64;
                w * average + (1.0 - w) * state.score(alpha)
            }
            _ => average,
        }
    }

    #[test]
    fn optimized_distance_one_matches_frozen_bits_rng_and_counters() {
        let graph = Graph::from_edges(
            8,
            vec![
                [0, 1],
                [0, 4],
                [1, 2],
                [1, 6],
                [2, 3],
                [2, 5],
                [3, 4],
                [3, 7],
                [4, 5],
                [5, 6],
                [6, 7],
            ],
        )
        .unwrap();
        let state = PartitionState::new(
            &graph,
            vec![true, false, true, false, true, false, false, true],
        )
        .unwrap();
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            for spec in [
                SmoothingSpec::WeightedAverage { k: 3 },
                SmoothingSpec::RandomKAverage { k: 3 },
            ] {
                let mut actual_rng = Mt19937GenRand64::new(0x1020_3040_5060_7080);
                let mut frozen_rng = actual_rng.clone();
                let mut actual_evals = 0;
                let mut frozen_evals = 0;
                let actual = evaluate(
                    &state,
                    &graph,
                    0.05,
                    neighborhood,
                    &spec,
                    Some(&mut actual_rng),
                    &CancellationToken::default(),
                    &mut actual_evals,
                )
                .unwrap();
                let frozen = frozen_distance_one(
                    &state,
                    &graph,
                    0.05,
                    neighborhood,
                    &spec,
                    Some(&mut frozen_rng),
                    &mut frozen_evals,
                );
                assert_eq!(actual.to_bits(), frozen.to_bits());
                assert_eq!(actual_evals, frozen_evals);
                for _ in 0..624 {
                    assert_eq!(actual_rng.next_u64(), frozen_rng.next_u64());
                }
            }
        }
    }
}
