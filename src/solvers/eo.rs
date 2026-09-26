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
#[derive(Clone, Copy, Debug, Default)]
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
    /// Scratch for the eligible blocks of a swap: one entry per vertex, the
    /// most blocks a ranking can have (empty for flip).
    eligible: Vec<Eligible>,
    ranker: Ranker,
}

enum Ranker {
    Index(BuiltinIndex),
    Sorted(SortedRanker),
}

/// The canonical order of the current state, as held by a [`Ranker`].
#[derive(Clone, Copy)]
enum Order<'a> {
    Index(Ranking<'a>),
    Sorted(&'a SortedRanker),
}

impl Ranker {
    /// The canonical order of `state`; a sorted ranker must be ranked first.
    #[inline]
    fn order(&self, state: &PartitionState) -> Order<'_> {
        match self {
            Ranker::Index(index) => Order::Index(index.ranking(state)),
            Ranker::Sorted(sorted) => Order::Sorted(sorted),
        }
    }
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
        let blocks = match neighborhood {
            Neighborhood::Flip => 0,
            Neighborhood::Swap => graph.node_count(),
        };
        Ok(Self {
            cum: power_law_cdf(graph.node_count(), tau),
            eligible: vec![Eligible::default(); blocks],
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
        let order = self.ranker.order(state);
        let first = order.first(&self.cum, rng.r#gen::<f64>());
        Ok(match neighborhood {
            Neighborhood::Flip => Move::Flip(first),
            Neighborhood::Swap => {
                let opposite = !state.partition()[first];
                let (count, total) =
                    order.conditional_blocks(&self.cum, &mut self.eligible, opposite)?;
                let second =
                    order.second(&self.eligible[..count], opposite, total, rng.r#gen::<f64>());
                debug_assert_eq!(state.partition()[second], opposite);
                Move::Swap(first, second)
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
}

impl Order<'_> {
    /// First vertex for the uniform value `u`.
    #[inline]
    fn first(self, cum: &[f64], u: f64) -> usize {
        let position = rank_position(cum, u);
        match self {
            Order::Index(ranking) => {
                let (bucket, start) = ranking.locate(position);
                let end = start + ranking.size(bucket);
                ranking.member(bucket, block_offset(cum, start, end, u))
            }
            Order::Sorted(sorted) => {
                let (start, end) = sorted.block_around(position);
                sorted.vertex(start + block_offset(cum, start, end, u))
            }
        }
    }

    /// Write the blocks with `opposite`-side members to the front of
    /// `eligible` (which holds at least one entry per block); returns their
    /// count and total share.
    ///
    /// Block `[start, end)` with `c_o` conditioned members has the weight
    /// `W = hi - lo` with `hi = cum[end - 1]` and `lo = cum[start - 1]` (0.0
    /// if `start == 0`) and the share `x = W * c_o / c`, summed in canonical
    /// order into the total.
    #[inline]
    fn conditional_blocks(
        self,
        cum: &[f64],
        eligible: &mut [Eligible],
        opposite: bool,
    ) -> Result<(usize, f64)> {
        let mut count = 0;
        let mut total = 0.0;
        match self {
            Order::Index(ranking) => {
                let side = usize::from(opposite);
                let (mut end, mut lo) = (0, 0.0);
                ranking.for_each_nonempty(|bucket, counts| {
                    let size = (counts[0] + counts[1]) as usize;
                    let members = counts[side] as usize;
                    end += size;
                    let hi = cum[end - 1];
                    // No branch on `members`: a block without conditioned
                    // members adds a share of exactly +0.0 (`W >= +0.0`),
                    // which leaves the total (+0.0 or positive) bit for bit
                    // unchanged, and its entry is overwritten or past `count`.
                    let weight = hi - lo;
                    let share = weight * members as f64 / size as f64;
                    total += share;
                    eligible[count] = Eligible {
                        block: bucket,
                        size,
                        opposite: members,
                        weight,
                        share,
                    };
                    count += usize::from(members > 0);
                    lo = hi;
                });
            }
            Order::Sorted(sorted) => {
                let keys = &sorted.keys;
                let mut start = 0;
                while start < keys.len() {
                    let value = key_value(keys[start]);
                    let mut end = start;
                    let mut members = 0;
                    while end < keys.len() && key_value(keys[end]) == value {
                        members += usize::from(key_side(keys[end]) == opposite);
                        end += 1;
                    }
                    if members > 0 {
                        let lo = if start == 0 { 0.0 } else { cum[start - 1] };
                        let size = end - start;
                        let weight = cum[end - 1] - lo;
                        let share = weight * members as f64 / size as f64;
                        total += share;
                        eligible[count] = Eligible {
                            block: start,
                            size,
                            opposite: members,
                            weight,
                            share,
                        };
                        count += 1;
                    }
                    start = end;
                }
            }
        }
        if count == 0 {
            Err(Error::msg(EMPTY_DISTRIBUTION))
        } else {
            Ok((count, total))
        }
    }

    /// Second swap vertex for `u2` from the `eligible` blocks of
    /// [`Self::conditional_blocks`].
    #[inline]
    fn second(self, eligible: &[Eligible], opposite: bool, total: f64, u2: f64) -> usize {
        let (i, j) = conditional_choice(eligible, total, u2);
        let block = eligible[i];
        match self {
            Order::Index(ranking) => ranking.select(2 * block.block + usize::from(opposite), j),
            Order::Sorted(sorted) => {
                // Inside a block the `false` side precedes the `true` side.
                let first = if opposite {
                    block.block + block.size - block.opposite
                } else {
                    block.block
                };
                sorted.vertex(first + j)
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

/// Buckets per [`Group`].
const GROUP: usize = 64;

/// Incremental canonical ranking of a built-in fitness.
///
/// A built-in value is `kind.lambda(lambda0, majority)`. `lambda0` takes one of
/// the precomputed *slots* (distinct `lambda0(degree, cuts)` values of the
/// degrees present) and majority depends only on the side and the group-size
/// state, so each maintained state maps the *key* `side * slots + slot` of a
/// vertex to a *bucket* (distinct value, ascending). A non-empty bucket is
/// exactly one block of the canonical order. Moves re-rank only the moved
/// vertices and their neighbors, in O(1) per vertex and maintained state.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct BuiltinIndex {
    kind: BuiltinFitness,
    /// `key_lut[lut_base[v] + 2 * cuts + side]` is the key of vertex `v` with
    /// `cuts` cut edges on `side`.
    lut_base: Vec<usize>,
    key_lut: Vec<u32>,
    /// Key each vertex is currently ranked with.
    key_of: Vec<u32>,
    /// Rankings of the maintained size states.
    rankings: Rankings,
    /// Ranking used for each size state.
    ranking_of_state: [usize; SIZE_STATES],
}

/// Canonical rankings of the maintained size states, in shared arrays.
///
/// Ranking `r` owns the buckets `base[r]..base[r + 1]`, its distinct values in
/// ascending order followed by empty padding up to a multiple of [`GROUP`].
/// The *cell* `2 * bucket + side` holds the members of one side of a bucket:
/// inside a block the `false` side precedes the `true` side, each in ascending
/// vertex order. Membership is a bitset per cell, so a move clears and sets one
/// bit per ranking and the `k`-th member of a cell is found by counting bits;
/// the bitsets take `16 * buckets * ceil(n / 64)` bytes. Cell counts, totals
/// and non-empty flags are kept per [`Group`], so a move updates O(1) entries
/// and [`Ranking::locate`] and [`Ranking::for_each_nonempty`] skip empty
/// buckets.
#[derive(Clone, Debug, PartialEq)]
struct Rankings {
    /// First bucket of each ranking, then the total bucket count.
    base: Vec<usize>,
    /// `cells[key * R + r]` is the cell of `key` in ranking `r` of `R`.
    cells: Vec<u32>,
    /// Words per cell bitset, `ceil(n / 64)`.
    words: usize,
    /// Bit `v % 64` of `bits[cell * words + v / 64]` is set iff `v` is in `cell`.
    bits: Vec<u64>,
    /// `groups[g]` holds the buckets `GROUP * g..GROUP * (g + 1)`.
    groups: Vec<Group>,
}

/// Cell counts, member total and non-empty flags of [`GROUP`] consecutive
/// buckets.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Group {
    /// Members of the group's buckets.
    size: u32,
    /// Bit `b` is set iff bucket `b` of the group has members.
    nonempty: u64,
    /// `counts[b][side]`: members of the cells of bucket `b` of the group.
    counts: [[u32; 2]; GROUP],
}

const EMPTY_GROUP: Group = Group {
    size: 0,
    nonempty: 0,
    counts: [[0; 2]; GROUP],
};

/// One ranking of [`Rankings`], with bucket and cell IDs relative to it.
#[derive(Clone, Copy)]
struct Ranking<'a> {
    words: usize,
    bits: &'a [u64],
    groups: &'a [Group],
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
        let slots = slot_value.len();
        if u32::try_from(2 * slots).is_err() {
            return Err(Error::msg("graph is too large for the EO index"));
        }
        let mut degree_offset = vec![usize::MAX; max_degree + 1];
        let mut key_lut = Vec::new();
        for &d in &degrees {
            degree_offset[d] = key_lut.len();
            for cuts in 0..=d {
                let value = lambda0(d, cuts as i64);
                let slot = slot_value.partition_point(|&x| x < value);
                key_lut.extend([slot as u32, (slots + slot) as u32]);
            }
        }
        let lut_base: Vec<usize> = (0..n).map(|v| degree_offset[graph.degree(v)]).collect();
        let (partition, cuts_at) = (state.partition(), state.cuts_at());
        let key_of: Vec<u32> = (0..n)
            .map(|v| key_lut[lut_base[v] + 2 * cuts_at[v] as usize + usize::from(partition[v])])
            .collect();
        // Swap never changes the group sizes; majority-independent values rank
        // identically in every state.
        let states = match neighborhood {
            Neighborhood::Swap => vec![size_state(state.size_a(), state.size_b())],
            Neighborhood::Flip if kind.depends_on_majority() => (0..SIZE_STATES).collect(),
            Neighborhood::Flip => vec![size_state(0, 0)],
        };
        let majority: Vec<[bool; 2]> = states.iter().map(|&s| majority_flags(s)).collect();
        let rankings = Rankings::new(kind, &slot_value, &majority, &key_of)?;
        let ranking_of_state = if states.len() == SIZE_STATES {
            [0, 1, 2]
        } else {
            [0; SIZE_STATES]
        };
        Ok(Self {
            kind,
            lut_base,
            key_lut,
            key_of,
            rankings,
            ranking_of_state,
        })
    }

    /// Ranking for the current group sizes.
    #[inline]
    fn ranking(&self, state: &PartitionState) -> Ranking<'_> {
        self.rankings
            .get(self.ranking_of_state[size_state(state.size_a(), state.size_b())])
    }

    /// Re-rank the vertices whose fitness `mv` can change; returns their count.
    fn refresh_move(&mut self, graph: &Graph, state: &PartitionState, mv: Move) -> u64 {
        match self.rankings.count() {
            1 => self.refresh_move_in::<1>(graph, state, mv),
            SIZE_STATES => self.refresh_move_in::<SIZE_STATES>(graph, state, mv),
            count => unreachable!("{count} maintained rankings"),
        }
    }

    /// [`Self::refresh_move`] with `R` maintained rankings.
    fn refresh_move_in<const R: usize>(
        &mut self,
        graph: &Graph,
        state: &PartitionState,
        mv: Move,
    ) -> u64 {
        match mv {
            Move::Flip(v) => {
                self.refresh_around::<R>(graph, state, v);
                1 + graph.degree(v) as u64
            }
            Move::Swap(a, b) => {
                self.refresh_around::<R>(graph, state, a);
                self.refresh_around::<R>(graph, state, b);
                2 + graph.degree(a) as u64 + graph.degree(b) as u64
            }
        }
    }

    #[inline(always)]
    fn refresh_around<const R: usize>(&mut self, graph: &Graph, state: &PartitionState, v: usize) {
        self.refresh::<R>(state, v);
        for &u in graph.neighbors(v) {
            self.refresh::<R>(state, u);
        }
    }

    /// Move `v` to its current key in every maintained ranking.
    #[inline(always)]
    fn refresh<const R: usize>(&mut self, state: &PartitionState, v: usize) {
        let side = usize::from(state.partition()[v]);
        let cuts = state.cuts_at()[v] as usize;
        let key = self.key_lut[self.lut_base[v] + 2 * cuts + side];
        let old_key = self.key_of[v];
        if key != old_key {
            self.key_of[v] = key;
            self.rankings
                .relocate::<R>(v, old_key as usize, key as usize);
        }
    }
}

impl Rankings {
    /// Rank the vertices with keys `key_of` in one state per `majority` flags.
    fn new(
        kind: BuiltinFitness,
        slot_value: &[f64],
        majority: &[[bool; 2]],
        key_of: &[u32],
    ) -> Result<Self> {
        let slots = slot_value.len();
        let count = majority.len();
        let mut base = vec![0];
        let mut cells = vec![0; 2 * slots * count];
        for (r, flags) in majority.iter().enumerate() {
            let value_of: Vec<f64> = flags
                .iter()
                .flat_map(|&majority| slot_value.iter().map(move |&l0| kind.lambda(l0, majority)))
                .collect();
            let mut distinct = value_of.clone();
            distinct.sort_by(f64::total_cmp);
            distinct.dedup();
            let first = base[r];
            for (key, &x) in value_of.iter().enumerate() {
                let bucket = first + distinct.partition_point(|&y| y < x);
                cells[key * count + r] = 2 * bucket + key / slots;
            }
            base.push(first + distinct.len().div_ceil(GROUP) * GROUP);
        }
        let buckets = base[count];
        if u32::try_from(2 * buckets).is_err() {
            return Err(Error::msg("graph is too large for the EO index"));
        }
        let words = key_of.len().div_ceil(64);
        let mut rankings = Self {
            base,
            cells: cells.into_iter().map(|cell| cell as u32).collect(),
            words,
            bits: vec![0; 2 * buckets * words],
            groups: vec![EMPTY_GROUP; buckets / GROUP],
        };
        for (v, &key) in key_of.iter().enumerate() {
            for r in 0..count {
                let cell = rankings.cells[key as usize * count + r] as usize;
                rankings.bits[cell * words + v / 64] |= 1 << (v % 64);
                let (bucket, side) = (cell / 2, cell % 2);
                let group = &mut rankings.groups[bucket / GROUP];
                group.counts[bucket % GROUP][side] += 1;
                group.size += 1;
                group.nonempty |= 1 << (bucket % GROUP);
            }
        }
        Ok(rankings)
    }

    /// Number of maintained rankings.
    #[inline]
    fn count(&self) -> usize {
        self.base.len() - 1
    }

    /// Ranking `r`.
    #[inline]
    fn get(&self, r: usize) -> Ranking<'_> {
        let (first, end) = (self.base[r], self.base[r + 1]);
        Ranking {
            words: self.words,
            bits: &self.bits[2 * first * self.words..2 * end * self.words],
            groups: &self.groups[first / GROUP..end / GROUP],
        }
    }

    /// Move `v` from the cells of `old_key` to the cells of `key` in all `R`
    /// rankings.
    #[inline(always)]
    fn relocate<const R: usize>(&mut self, v: usize, old_key: usize, key: usize) {
        let (rows, _) = self.cells.as_chunks::<R>();
        let (old_row, row) = (rows[old_key], rows[key]);
        let (word, bit) = (v / 64, 1u64 << (v % 64));
        for (&old_cell, &cell) in old_row.iter().zip(&row) {
            let (old_cell, cell) = (old_cell as usize, cell as usize);
            if old_cell == cell {
                continue;
            }
            debug_assert!(
                self.bits[old_cell * self.words + word] & bit != 0,
                "an indexed vertex is ranked in its recorded cell"
            );
            self.bits[old_cell * self.words + word] &= !bit;
            self.bits[cell * self.words + word] |= bit;
            let (old_bucket, old_side) = (old_cell / 2, old_cell % 2);
            let (bucket, side) = (cell / 2, cell % 2);
            let old_group = &mut self.groups[old_bucket / GROUP];
            let old_counts = &mut old_group.counts[old_bucket % GROUP];
            old_counts[old_side] -= 1;
            if old_bucket == bucket {
                old_counts[side] += 1;
                continue;
            }
            // The other side is loaded on its own: a wide load over the count
            // just stored would stall on store forwarding.
            let emptied = (old_counts[old_side] | old_counts[old_side ^ 1]) == 0;
            old_group.size -= 1;
            old_group.nonempty &= !(u64::from(emptied) << (old_bucket % GROUP));
            let group = &mut self.groups[bucket / GROUP];
            group.counts[bucket % GROUP][side] += 1;
            group.size += 1;
            group.nonempty |= 1 << (bucket % GROUP);
        }
    }
}

impl Ranking<'_> {
    /// Cell counts `[side false, side true]` of `bucket`.
    #[inline]
    fn counts(&self, bucket: usize) -> [u32; 2] {
        self.groups[bucket / GROUP].counts[bucket % GROUP]
    }

    #[inline]
    fn size(&self, bucket: usize) -> usize {
        let [low, high] = self.counts(bucket);
        (low + high) as usize
    }

    /// Bucket containing canonical position `position < n` and its first position.
    #[inline]
    fn locate(&self, position: usize) -> (usize, usize) {
        let mut rest = position;
        for (g, group) in self.groups.iter().enumerate() {
            if rest >= group.size as usize {
                rest -= group.size as usize;
                continue;
            }
            let mut mask = group.nonempty;
            loop {
                debug_assert!(mask != 0, "a group holds its total in non-empty buckets");
                let b = mask.trailing_zeros() as usize % GROUP;
                let [low, high] = group.counts[b];
                let size = (low + high) as usize;
                if rest < size {
                    return (GROUP * g + b, position - rest);
                }
                rest -= size;
                mask &= mask - 1;
            }
        }
        unreachable!("a located position is below the vertex count")
    }

    /// Call `f` with every non-empty bucket in ascending order and its counts.
    #[inline(always)]
    fn for_each_nonempty(&self, mut f: impl FnMut(usize, [u32; 2])) {
        for (g, group) in self.groups.iter().enumerate() {
            let mut mask = group.nonempty;
            while mask != 0 {
                let b = mask.trailing_zeros() as usize % GROUP;
                f(GROUP * g + b, group.counts[b]);
                mask &= mask - 1;
            }
        }
    }

    /// Member at `offset` of a block: `false` side ascending, then `true` side.
    #[inline]
    fn member(&self, bucket: usize, offset: usize) -> usize {
        let low = self.counts(bucket)[0] as usize;
        if offset < low {
            self.select(2 * bucket, offset)
        } else {
            self.select(2 * bucket + 1, offset - low)
        }
    }

    /// Member of `cell` with the `k`-th smallest vertex ID; requires `k` below
    /// the cell count.
    #[inline]
    fn select(&self, cell: usize, mut k: usize) -> usize {
        let words = &self.bits[cell * self.words..(cell + 1) * self.words];
        for (w, &word) in words.iter().enumerate() {
            let count = word.count_ones() as usize;
            if k < count {
                return 64 * w + select_in_word(word, k);
            }
            k -= count;
        }
        unreachable!("a selected member rank is below the cell count")
    }
}

/// Position of the `k`-th (zero-based) set bit of `word`; requires
/// `k < word.count_ones()`.
#[inline]
fn select_in_word(mut word: u64, k: usize) -> usize {
    for _ in 0..k {
        word &= word - 1;
    }
    word.trailing_zeros() as usize
}

#[cfg(test)]
impl Eo {
    /// Whether this job ranks a built-in fitness incrementally.
    pub(super) fn is_indexed(&self) -> bool {
        matches!(self.ranker, Ranker::Index(_))
    }

    /// First vertex for the uniform value `u`.
    fn first(&self, state: &PartitionState, u: f64) -> usize {
        self.ranker.order(state).first(&self.cum, u)
    }

    /// The blocks with `opposite`-side members and their total share (also
    /// for a flip job, which has no scratch of its own).
    fn conditional_blocks(
        &self,
        state: &PartitionState,
        opposite: bool,
    ) -> Result<(Vec<Eligible>, f64)> {
        let mut eligible = vec![Eligible::default(); state.partition().len()];
        let order = self.ranker.order(state);
        let (count, total) = order.conditional_blocks(&self.cum, &mut eligible, opposite)?;
        eligible.truncate(count);
        Ok((eligible, total))
    }

    /// Second swap vertex for `u2` from [`Self::conditional_blocks`].
    fn second(
        &self,
        state: &PartitionState,
        eligible: &[Eligible],
        opposite: bool,
        total: f64,
        u2: f64,
    ) -> usize {
        self.ranker
            .order(state)
            .second(eligible, opposite, total, u2)
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
