//! EO vertex selection, algorithm `v2` (averaged tie blocks).
//!
//! Every step ranks all vertices in the *canonical order* `(lambda, side, v)`:
//! `lambda` is the fitness compared with [`f64::partial_cmp`] (`-0.0 == 0.0`),
//! `side = partition[v]` with `false < true`, and the vertex ID decides the
//! rest. A *block* is a maximal run of equal `lambda` (`==`). Rank `r` has the
//! power-law weight `r^-tau` ([`power_law_cdf`]) and every member of a block
//! receives the block's average weight, so ties never consume random numbers.
//!
//! The first vertex uses one uniform draw `u`: the rank position of `u` in the
//! cumulative weights selects the block and the residual of `u` inside the
//! block selects the member. Swap draws a second `u2` and selects from the side
//! opposite to the first vertex, weighting each block by its averaged weight
//! restricted to its opposite-side members. When every such weight underflows to
//! zero (huge `tau`), it falls back to the lowest block with an opposite-side
//! member. No selection uses the tie RNG.
//!
//! Built-in fitness definitions are ranked by the incremental [`BuiltinIndex`];
//! any other definition is evaluated and sorted every step. Both rankings feed
//! the same selection arithmetic and select bit-identical vertices.

use crate::error::{Error, Result};
use crate::experiment::config::Neighborhood;
use crate::fitness::{BuiltinFitness, EngineFitness, VertexFitness, is_majority, lambda0};
use crate::graph_partition::{Graph, Move, PartitionState};
use rand::Rng;
use rand_mt::Mt19937GenRand64;
use std::cmp::Ordering;

const EMPTY_DISTRIBUTION: &str = "empty EO conditional rank distribution";

/// Normalized cumulative power-law weights of ranks `1..=n`.
///
/// `cum[k] = (1^-tau + ... + (k + 1)^-tau) / Z` accumulated left to right, so
/// `tau = 0` is uniform and a huge `tau` puts all weight on rank 1. Every term
/// is finite and the first is exactly 1, so no finite `tau >= 0` produces NaN.
pub(super) fn power_law_cdf(n: usize, tau: f64) -> Vec<f64> {
    let mut cum = Vec::with_capacity(n);
    let mut cumulative = 0.0;
    for j in 1..=n {
        cumulative += (j as f64).powf(-tau);
        cum.push(cumulative);
    }
    let z = cumulative;
    for c in cum.iter_mut() {
        *c /= z;
    }
    cum
}

/// Zero-based rank position selected by `u`; requires a non-empty `cum`.
#[inline]
fn rank_position(cum: &[f64], u: f64) -> usize {
    cum.partition_point(|&c| c <= u).min(cum.len() - 1)
}

/// Offset of the first selection inside the block `[start, end)`.
#[inline]
fn block_offset(cum: &[f64], start: usize, end: usize, u: f64) -> usize {
    let lo = if start == 0 { 0.0 } else { cum[start - 1] };
    let hi = cum[end - 1];
    let m = end - start;
    let per = (hi - lo) / m as f64;
    if per > 0.0 {
        (((u - lo).max(0.0) / per) as usize).min(m - 1)
    } else {
        0
    }
}

/// A block that contains at least one vertex of the conditioned side.
#[derive(Clone, Copy, Debug)]
struct Eligible {
    /// Bucket ID (index ranking) or first position (sorted ranking).
    block: usize,
    /// Block size `c`.
    size: usize,
    /// Members on the conditioned side, `c_o > 0`.
    opposite: usize,
    /// Block weight `W`.
    weight: f64,
    /// Conditioned share `x = W * c_o / c`.
    share: f64,
}

/// Choose `(eligible block, j)` for the second swap vertex.
///
/// Blocks are scanned in canonical order with a running sum; the last block is
/// the fallback of the scan, reached with the sum of all earlier shares.
fn conditional_choice(eligible: &[Eligible], total: f64, u2: f64) -> (usize, usize) {
    if total > 0.0 {
        let target = u2 * total;
        let mut acc = 0.0;
        let mut i = 0;
        while i + 1 < eligible.len() {
            let next = acc + eligible[i].share;
            if target < next {
                break;
            }
            acc = next;
            i += 1;
        }
        let block = eligible[i];
        let per = block.weight / block.size as f64;
        let j = if per > 0.0 {
            (((target - acc).max(0.0) / per) as usize).min(block.opposite - 1)
        } else {
            0
        };
        (i, j)
    } else {
        let block = eligible[0];
        (
            0,
            ((u2 * block.opposite as f64) as usize).min(block.opposite - 1),
        )
    }
}

/// Per-job EO selection state: the fixed weights and the vertex ranking.
pub(super) struct Eo {
    cum: Vec<f64>,
    eligible: Vec<Eligible>,
    ranker: Ranker,
}

enum Ranker {
    Index(BuiltinIndex),
    Sorted(SortedRanker),
}

impl Eo {
    /// Prepare EO for `state`; adds the fitness values computed to build the
    /// incremental index (`n` for built-ins, none otherwise).
    pub(super) fn new(
        fitness: EngineFitness,
        graph: &Graph,
        state: &PartitionState,
        neighborhood: Neighborhood,
        tau: f64,
        fitness_values: &mut u64,
    ) -> Result<Self> {
        let ranker = match fitness {
            EngineFitness::Builtin(kind) => {
                *fitness_values += graph.node_count() as u64;
                Ranker::Index(BuiltinIndex::new(kind, graph, state, neighborhood)?)
            }
            EngineFitness::Custom(fitness) => Ranker::Sorted(SortedRanker {
                fitness,
                keys: Vec::new(),
            }),
        };
        Ok(Self {
            cum: power_law_cdf(graph.node_count(), tau),
            eligible: Vec::new(),
            ranker,
        })
    }

    /// Select one move, drawing `u` (and `u2` for swap) from `rng`.
    ///
    /// Custom fitness is evaluated first and counted in `fitness_values`.
    pub(super) fn select(
        &mut self,
        graph: &Graph,
        state: &PartitionState,
        neighborhood: Neighborhood,
        rng: &mut Mt19937GenRand64,
        fitness_values: &mut u64,
    ) -> Result<Move> {
        if let Ranker::Sorted(sorted) = &mut self.ranker {
            sorted.rank(graph, state, fitness_values)?;
        }
        if state.partition().is_empty() {
            return Err(Error::msg(EMPTY_DISTRIBUTION));
        }
        let first = self.first(state, rng.r#gen::<f64>());
        Ok(match neighborhood {
            Neighborhood::Flip => Move::Flip(first),
            Neighborhood::Swap => {
                let opposite = !state.partition()[first];
                let total = self.conditional_blocks(state, opposite)?;
                Move::Swap(
                    first,
                    self.second(state, opposite, total, rng.r#gen::<f64>()),
                )
            }
        })
    }

    /// Update the ranking after `mv` was applied to `state`.
    ///
    /// Adds the re-ranked vertex count to `fitness_values`: `1 + deg(v)` for a
    /// flip and `2 + deg(a) + deg(b)` for a swap on the incremental index.
    pub(super) fn applied(
        &mut self,
        graph: &Graph,
        state: &PartitionState,
        mv: Move,
        fitness_values: &mut u64,
    ) {
        if let Ranker::Index(index) = &mut self.ranker {
            *fitness_values += index.refresh_move(graph, state, mv);
        }
    }

    /// First vertex for the uniform value `u`.
    fn first(&self, state: &PartitionState, u: f64) -> usize {
        let cum = &self.cum;
        let position = rank_position(cum, u);
        match &self.ranker {
            Ranker::Index(index) => {
                let ranking = index.ranking(state);
                let (bucket, start) = ranking.locate(position);
                let end = start + ranking.size(bucket);
                ranking.member(bucket, block_offset(cum, start, end, u))
            }
            Ranker::Sorted(sorted) => {
                let (start, end) = sorted.block_around(position);
                sorted.vertex(start + block_offset(cum, start, end, u))
            }
        }
    }

    /// Collect the blocks with `opposite`-side members; returns their total share.
    fn conditional_blocks(&mut self, state: &PartitionState, opposite: bool) -> Result<f64> {
        let cum = &self.cum;
        let eligible = &mut self.eligible;
        eligible.clear();
        let mut total = 0.0;
        let mut push = |block: usize, start: usize, size: usize, count: usize| {
            let lo = if start == 0 { 0.0 } else { cum[start - 1] };
            let weight = cum[start + size - 1] - lo;
            let share = weight * count as f64 / size as f64;
            total += share;
            eligible.push(Eligible {
                block,
                size,
                opposite: count,
                weight,
                share,
            });
        };
        match &self.ranker {
            Ranker::Index(index) => {
                let ranking = index.ranking(state);
                let mut start = 0;
                for (bucket, &counts) in ranking.counts.iter().enumerate() {
                    let size = (counts[0] + counts[1]) as usize;
                    if size == 0 {
                        continue;
                    }
                    let count = counts[usize::from(opposite)] as usize;
                    if count > 0 {
                        push(bucket, start, size, count);
                    }
                    start += size;
                }
            }
            Ranker::Sorted(sorted) => {
                let keys = &sorted.keys;
                let mut start = 0;
                while start < keys.len() {
                    let value = key_value(keys[start]);
                    let mut end = start;
                    let mut count = 0;
                    while end < keys.len() && key_value(keys[end]) == value {
                        count += usize::from(key_side(keys[end]) == opposite);
                        end += 1;
                    }
                    if count > 0 {
                        push(start, start, end - start, count);
                    }
                    start = end;
                }
            }
        }
        if self.eligible.is_empty() {
            Err(Error::msg(EMPTY_DISTRIBUTION))
        } else {
            Ok(total)
        }
    }

    /// Second swap vertex for `u2`, after [`Self::conditional_blocks`].
    fn second(&self, state: &PartitionState, opposite: bool, total: f64, u2: f64) -> usize {
        let (i, j) = conditional_choice(&self.eligible, total, u2);
        let block = self.eligible[i];
        match &self.ranker {
            Ranker::Index(index) => {
                index.ranking(state).members[2 * block.block + usize::from(opposite)].select(j)
            }
            Ranker::Sorted(sorted) => {
                // Inside a block the `false` side precedes the `true` side.
                let first = if opposite {
                    block.block + block.size - block.opposite
                } else {
                    block.block
                };
                let vertex = sorted.vertex(first + j);
                debug_assert_eq!(state.partition()[vertex], opposite);
                vertex
            }
        }
    }
}

/// Canonical sort key of a vertex with finite fitness `value`.
///
/// The high 64 bits map `value` monotonically onto `u64` with `-0.0` and `0.0`
/// merged, so key order is `f64::partial_cmp` order and equal high halves mean
/// `==`. Bit 63 holds the side and the low bits the vertex, which makes the key
/// unique and its integer order the canonical order.
#[inline]
fn canonical_key(value: f64, side: bool, vertex: usize) -> u128 {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    let ordered = if bits >> 63 == 1 {
        !bits
    } else {
        bits | (1 << 63)
    };
    (u128::from(ordered) << 64) | (u128::from(side) << 63) | vertex as u128
}

#[inline]
fn key_value(key: u128) -> u64 {
    (key >> 64) as u64
}

#[inline]
fn key_side(key: u128) -> bool {
    (key >> 63) & 1 == 1
}

#[inline]
fn key_vertex(key: u128) -> usize {
    (key as u64 & (u64::MAX >> 1)) as usize
}

/// Ranking of a caller-supplied fitness, rebuilt from `values()` every step.
struct SortedRanker {
    fitness: Box<dyn VertexFitness>,
    /// Canonical keys in ascending (canonical) order.
    keys: Vec<u128>,
}

impl SortedRanker {
    fn rank(
        &mut self,
        graph: &Graph,
        state: &PartitionState,
        fitness_values: &mut u64,
    ) -> Result<()> {
        let values = self.fitness.values(graph, state)?;
        *fitness_values += values.len() as u64;
        if values.len() != graph.node_count() || values.iter().any(|x| !x.is_finite()) {
            return Err(Error::msg("fitness returned invalid values"));
        }
        self.keys.clear();
        self.keys.extend(
            values
                .iter()
                .zip(state.partition())
                .enumerate()
                .map(|(v, (&value, &side))| canonical_key(value, side, v)),
        );
        // Keys are unique, so an unstable sort is deterministic.
        self.keys.sort_unstable();
        Ok(())
    }

    #[inline]
    fn vertex(&self, position: usize) -> usize {
        key_vertex(self.keys[position])
    }

    /// The block `[start, end)` containing sorted position `position`.
    fn block_around(&self, position: usize) -> (usize, usize) {
        let value = key_value(self.keys[position]);
        let mut start = position;
        while start > 0 && key_value(self.keys[start - 1]) == value {
            start -= 1;
        }
        let mut end = position + 1;
        while end < self.keys.len() && key_value(self.keys[end]) == value {
            end += 1;
        }
        (start, end)
    }
}

/// Group-size relation that decides which vertices are majority.
const SIZE_STATES: usize = 3;

/// `0`: A is larger, `1`: B is larger, `2`: equal sizes.
#[inline]
fn size_state(size_a: usize, size_b: usize) -> usize {
    match size_a.cmp(&size_b) {
        Ordering::Greater => 0,
        Ordering::Less => 1,
        Ordering::Equal => 2,
    }
}

/// Majority flags `[side false, side true]` of a size state.
fn majority_flags(size_state: usize) -> [bool; 2] {
    let (size_a, size_b) = [(1, 0), (0, 1), (0, 0)][size_state];
    [
        is_majority(false, size_a, size_b),
        is_majority(true, size_a, size_b),
    ]
}

/// Incremental canonical ranking of a built-in fitness.
///
/// A built-in value is `kind.lambda(lambda0, majority)`. `lambda0` takes one of
/// the precomputed *slots* (distinct `lambda0(degree, cuts)` values of the
/// degrees present) and majority depends only on the side and the group-size
/// state, so each maintained state maps `(side, slot)` to a *bucket* (distinct
/// value, ascending). A non-empty bucket is exactly one block of the canonical
/// order. Moves re-rank only the moved vertices and their neighbors.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct BuiltinIndex {
    kind: BuiltinFitness,
    /// Distinct `lambda0` values, ascending.
    slot_value: Vec<f64>,
    /// `slot_lut[lut_base[v] + cuts]` is the slot of `lambda0(degree(v), cuts)`.
    lut_base: Vec<usize>,
    slot_lut: Vec<u32>,
    /// Slot and side each vertex is currently ranked with.
    slot_of: Vec<u32>,
    side_of: Vec<bool>,
    /// Maintained size states.
    rankings: Vec<Ranking>,
    /// Ranking used for each size state.
    ranking_of_state: [usize; SIZE_STATES],
}

/// Canonical ranking of one group-size state.
#[derive(Clone, Debug, PartialEq)]
struct Ranking {
    /// `bucket_of[side * slots + slot]`.
    bucket_of: Vec<u32>,
    /// `members[2 * bucket + side]`, ascending vertex IDs.
    members: Vec<MemberSet>,
    /// `counts[bucket][side]`.
    counts: Vec<[u32; 2]>,
    /// 1-indexed Fenwick tree of bucket sizes.
    fenwick: Vec<u32>,
    /// Highest power of two not above the bucket count (at least 1).
    top_bit: usize,
    vertex_count: usize,
}

const BITSET_MEMBER_THRESHOLD: usize = 32;
/// Avoid one graph-sized allocation per populated bucket on large graphs.
const MAX_BITSET_VERTICES: usize = 4096;

/// Ordered bucket membership. Small buckets retain the compact vector path;
/// larger buckets use a bitset so relocation never shifts a long vector.
#[derive(Clone, Debug, PartialEq)]
enum MemberSet {
    Small(Vec<u32>),
    Bits { words: Vec<u64>, len: usize },
}

impl MemberSet {
    fn new() -> Self {
        Self::Small(Vec::new())
    }

    fn len(&self) -> usize {
        match self {
            Self::Small(values) => values.len(),
            Self::Bits { len, .. } => *len,
        }
    }

    fn select(&self, mut rank: usize) -> usize {
        match self {
            Self::Small(values) => values[rank] as usize,
            Self::Bits { words, .. } => {
                for (wi, &word) in words.iter().enumerate() {
                    let count = word.count_ones() as usize;
                    if rank < count {
                        let mut remaining = word;
                        for _ in 0..rank {
                            remaining &= remaining - 1;
                        }
                        return wi * 64 + remaining.trailing_zeros() as usize;
                    }
                    rank -= count;
                }
                unreachable!("member rank is within the recorded length")
            }
        }
    }

    fn insert(&mut self, v: u32, vertex_count: usize) {
        match self {
            Self::Small(values) => {
                let at = values.partition_point(|&x| x < v);
                values.insert(at, v);
                if values.len() > BITSET_MEMBER_THRESHOLD && vertex_count <= MAX_BITSET_VERTICES {
                    let mut words = vec![0u64; vertex_count.div_ceil(64)];
                    for &member in values.iter() {
                        words[member as usize / 64] |= 1 << (member as usize % 64);
                    }
                    let len = values.len();
                    *self = Self::Bits { words, len };
                }
            }
            Self::Bits { words, len } => {
                let bit = 1u64 << (v as usize % 64);
                let word = &mut words[v as usize / 64];
                debug_assert_eq!(*word & bit, 0);
                *word |= bit;
                *len += 1;
            }
        }
    }

    fn remove(&mut self, v: u32) {
        match self {
            Self::Small(values) => {
                let at = values
                    .binary_search(&v)
                    .expect("an indexed vertex is ranked in its recorded bucket");
                values.remove(at);
            }
            Self::Bits { words, len } => {
                let bit = 1u64 << (v as usize % 64);
                let word = &mut words[v as usize / 64];
                debug_assert_ne!(*word & bit, 0);
                *word &= !bit;
                *len -= 1;
                if *len <= BITSET_MEMBER_THRESHOLD {
                    let new_len = *len;
                    let values = (0..new_len)
                        .map(|rank| {
                            let mut rest = rank;
                            for (wi, &word) in words.iter().enumerate() {
                                let count = word.count_ones() as usize;
                                if rest < count {
                                    let mut remaining = word;
                                    for _ in 0..rest {
                                        remaining &= remaining - 1;
                                    }
                                    return (wi * 64 + remaining.trailing_zeros() as usize) as u32;
                                }
                                rest -= count;
                            }
                            unreachable!()
                        })
                        .collect();
                    *self = Self::Small(values);
                }
            }
        }
    }
}

impl BuiltinIndex {
    fn new(
        kind: BuiltinFitness,
        graph: &Graph,
        state: &PartitionState,
        neighborhood: Neighborhood,
    ) -> Result<Self> {
        let n = graph.node_count();
        if u32::try_from(n).is_err() {
            return Err(Error::msg("graph is too large for the EO index"));
        }
        let max_degree = (0..n).map(|v| graph.degree(v)).max().unwrap_or(0);
        let mut present = vec![false; max_degree + 1];
        for v in 0..n {
            present[graph.degree(v)] = true;
        }
        let degrees: Vec<usize> = (0..=max_degree).filter(|&d| present[d]).collect();
        let mut slot_value: Vec<f64> = degrees
            .iter()
            .flat_map(|&d| (0..=d).map(move |cuts| lambda0(d, cuts as i64)))
            .collect();
        slot_value.sort_by(f64::total_cmp);
        slot_value.dedup();
        let mut degree_offset = vec![usize::MAX; max_degree + 1];
        let mut slot_lut = Vec::new();
        for &d in &degrees {
            degree_offset[d] = slot_lut.len();
            for cuts in 0..=d {
                let value = lambda0(d, cuts as i64);
                slot_lut.push(slot_value.partition_point(|&x| x < value) as u32);
            }
        }
        let lut_base: Vec<usize> = (0..n).map(|v| degree_offset[graph.degree(v)]).collect();
        let slot_of: Vec<u32> = (0..n)
            .map(|v| slot_lut[lut_base[v] + state.cuts_at()[v] as usize])
            .collect();
        let side_of = state.partition().to_vec();
        // Swap never changes the group sizes; majority-independent values rank
        // identically in every state.
        let states = match neighborhood {
            Neighborhood::Swap => vec![size_state(state.size_a(), state.size_b())],
            Neighborhood::Flip if kind.depends_on_majority() => (0..SIZE_STATES).collect(),
            Neighborhood::Flip => vec![size_state(0, 0)],
        };
        let rankings = states
            .iter()
            .map(|&s| Ranking::new(kind, &slot_value, majority_flags(s), &slot_of, &side_of))
            .collect();
        let ranking_of_state = if states.len() == SIZE_STATES {
            [0, 1, 2]
        } else {
            [0; SIZE_STATES]
        };
        Ok(Self {
            kind,
            slot_value,
            lut_base,
            slot_lut,
            slot_of,
            side_of,
            rankings,
            ranking_of_state,
        })
    }

    /// Ranking for the current group sizes.
    #[inline]
    fn ranking(&self, state: &PartitionState) -> &Ranking {
        &self.rankings[self.ranking_of_state[size_state(state.size_a(), state.size_b())]]
    }

    /// Re-rank the vertices whose fitness `mv` can change; returns their count.
    fn refresh_move(&mut self, graph: &Graph, state: &PartitionState, mv: Move) -> u64 {
        match mv {
            Move::Flip(v) => {
                self.refresh_around(graph, state, v);
                1 + graph.degree(v) as u64
            }
            Move::Swap(a, b) => {
                self.refresh_around(graph, state, a);
                self.refresh_around(graph, state, b);
                2 + graph.degree(a) as u64 + graph.degree(b) as u64
            }
        }
    }

    fn refresh_around(&mut self, graph: &Graph, state: &PartitionState, v: usize) {
        self.refresh(state, v);
        for &u in graph.neighbors(v) {
            self.refresh(state, u);
        }
    }

    /// Move `v` to its current `(side, slot)` in every maintained ranking.
    #[inline]
    fn refresh(&mut self, state: &PartitionState, v: usize) {
        let side = state.partition()[v];
        let slot = self.slot_lut[self.lut_base[v] + state.cuts_at()[v] as usize];
        let (old_side, old_slot) = (self.side_of[v], self.slot_of[v]);
        if side == old_side && slot == old_slot {
            return;
        }
        let slots = self.slot_value.len();
        let old_key = usize::from(old_side) * slots + old_slot as usize;
        let key = usize::from(side) * slots + slot as usize;
        for ranking in &mut self.rankings {
            ranking.relocate(v as u32, old_side, old_key, side, key);
        }
        self.side_of[v] = side;
        self.slot_of[v] = slot;
    }
}

impl Ranking {
    fn new(
        kind: BuiltinFitness,
        slot_value: &[f64],
        majority: [bool; 2],
        slot_of: &[u32],
        side_of: &[bool],
    ) -> Self {
        let slots = slot_value.len();
        let value_of: Vec<f64> = majority
            .iter()
            .flat_map(|&majority| slot_value.iter().map(move |&l0| kind.lambda(l0, majority)))
            .collect();
        let mut distinct = value_of.clone();
        distinct.sort_by(f64::total_cmp);
        distinct.dedup();
        let bucket_of: Vec<u32> = value_of
            .iter()
            .map(|&x| distinct.partition_point(|&y| y < x) as u32)
            .collect();
        let buckets = distinct.len();
        let mut members = (0..2 * buckets)
            .map(|_| MemberSet::new())
            .collect::<Vec<_>>();
        let mut counts = vec![[0u32; 2]; buckets];
        for (v, (&slot, &side)) in slot_of.iter().zip(side_of).enumerate() {
            let side = usize::from(side);
            let bucket = bucket_of[side * slots + slot as usize] as usize;
            members[2 * bucket + side].insert(v as u32, slot_of.len());
            counts[bucket][side] += 1;
        }
        let mut fenwick = vec![0u32; buckets + 1];
        for (bucket, count) in counts.iter().enumerate() {
            fenwick[bucket + 1] = count[0] + count[1];
        }
        for i in 1..=buckets {
            let parent = i + i.isolate_lowest_one();
            if parent <= buckets {
                fenwick[parent] += fenwick[i];
            }
        }
        let mut top_bit = 1;
        while top_bit * 2 <= buckets {
            top_bit *= 2;
        }
        Self {
            bucket_of,
            members,
            counts,
            fenwick,
            top_bit,
            vertex_count: slot_of.len(),
        }
    }

    #[inline]
    fn size(&self, bucket: usize) -> usize {
        (self.counts[bucket][0] + self.counts[bucket][1]) as usize
    }

    /// Bucket containing canonical position `position < n` and its first position.
    #[inline]
    fn locate(&self, position: usize) -> (usize, usize) {
        let buckets = self.counts.len();
        let mut index = 0;
        let mut rest = position as u32;
        let mut bit = self.top_bit;
        while bit > 0 {
            let next = index + bit;
            if next <= buckets && self.fenwick[next] <= rest {
                index = next;
                rest -= self.fenwick[next];
            }
            bit >>= 1;
        }
        (index, position - rest as usize)
    }

    /// Member at `offset` of a block: `false` side ascending, then `true` side.
    #[inline]
    fn member(&self, bucket: usize, offset: usize) -> usize {
        let low = &self.members[2 * bucket];
        if offset < low.len() {
            low.select(offset)
        } else {
            self.members[2 * bucket + 1].select(offset - low.len())
        }
    }

    fn relocate(&mut self, v: u32, old_side: bool, old_key: usize, side: bool, key: usize) {
        let old_bucket = self.bucket_of[old_key] as usize;
        let bucket = self.bucket_of[key] as usize;
        if old_bucket == bucket && old_side == side {
            return;
        }
        let old_side_index = usize::from(old_side);
        self.members[2 * old_bucket + old_side_index].remove(v);
        self.counts[old_bucket][usize::from(old_side)] -= 1;
        let side_index = usize::from(side);
        self.members[2 * bucket + side_index].insert(v, self.vertex_count);
        self.counts[bucket][usize::from(side)] += 1;
        if old_bucket != bucket {
            self.fenwick_move(old_bucket, bucket);
        }
    }

    /// Move one count from bucket `from` to bucket `to`.
    #[inline]
    fn fenwick_move(&mut self, from: usize, to: usize) {
        let mut i = from + 1;
        while i < self.fenwick.len() {
            self.fenwick[i] -= 1;
            i += i.isolate_lowest_one();
        }
        let mut j = to + 1;
        while j < self.fenwick.len() {
            self.fenwick[j] += 1;
            j += j.isolate_lowest_one();
        }
    }
}

#[cfg(test)]
impl Eo {
    /// Whether this job ranks a built-in fitness incrementally.
    pub(super) fn is_indexed(&self) -> bool {
        matches!(self.ranker, Ranker::Index(_))
    }

    /// Assert that the incremental index equals a full rebuild from `state`.
    pub(super) fn assert_index_consistent(
        &self,
        graph: &Graph,
        state: &PartitionState,
        neighborhood: Neighborhood,
    ) {
        if let Ranker::Index(index) = &self.ranker {
            let rebuilt = BuiltinIndex::new(index.kind, graph, state, neighborhood).unwrap();
            assert!(
                *index == rebuilt,
                "incremental EO index differs from a rebuild"
            );
        }
    }
}

#[cfg(test)]
#[path = "eo_tests.rs"]
mod tests;
