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
    /// Neighborhood of the job. It sizes `eligible`, and the incremental index
    /// keeps only the rankings that moves of this neighborhood use.
    neighborhood: Neighborhood,
    /// Scratch for the eligible blocks of a swap: one entry per vertex, the
    /// most blocks a ranking can have (empty for flip).
    eligible: Vec<Eligible>,
    ranker: Ranker,
}

// One per job and built once, so the size gap between the variants costs
// nothing, while boxing the index would add an indirection to every step.
#[allow(clippy::large_enum_variant)]
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
        // Validated by the plan and the runner. Finite `tau >= 0` makes every
        // weight finite and every block weight `hi - lo` +0.0 or positive,
        // which the branchless scan of `Order::conditional_blocks` needs.
        debug_assert!(
            tau.is_finite() && tau >= 0.0,
            "tau must be finite and non-negative, not {tau}"
        );
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
            neighborhood,
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
        assert_eq!(
            neighborhood, self.neighborhood,
            "EO selects moves of the neighborhood its job was built for"
        );
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
        assert!(
            matches!(
                (mv, self.neighborhood),
                (Move::Flip(_), Neighborhood::Flip) | (Move::Swap(..), Neighborhood::Swap)
            ),
            "EO applies moves of the neighborhood its job was built for"
        );
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

/// Mark bit of a row during [`Rerank::swap_overlapping`]; every row is below
/// it.
const MARK: u32 = 1 << 31;

/// Words of a closed-neighborhood signature (see [`BuiltinIndex::closed`]).
const SIGNATURE_WORDS: usize = 8;

/// Dense cell bitsets up to this many bytes are always used. Dense updates
/// are cheaper while the bitsets stay cache-friendly; far larger dense
/// bitsets (degrees in the hundreds or thousands) are slower than pooled ones.
const DENSE_BYTES: usize = 32 << 20;

/// Whether `R = count` rankings of `buckets` buckets over `n` vertices keep
/// their cell bitsets in a [`Pool`]. Dense bitsets, one per cell, take
/// `16 * buckets * ceil(n / 64)` bytes; pooled ones at most about
/// `8 * R * n * ceil(n / 64)`, since at most `n` cells of a ranking are
/// non-empty. Dense bitsets are faster to update and are kept unless they are
/// both above [`DENSE_BYTES`] and above that bound, so the bitsets never take
/// more than the larger of the two.
fn pooled_layout(buckets: usize, n: usize, count: usize) -> bool {
    let bitset = 8 * n.div_ceil(64);
    let dense = (2 * buckets).saturating_mul(bitset);
    let pooled = count.saturating_mul(n).saturating_mul(bitset);
    dense > DENSE_BYTES.max(pooled)
}

#[cfg(test)]
thread_local! {
    /// Test override of [`pooled_layout`] for the indexes built on this thread.
    static FORCE_POOLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// Test count of the rows [`Rerank::move_to`] moved on this thread.
    static MOVED_ROWS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Incremental canonical ranking of a built-in fitness.
///
/// A built-in value is `kind.lambda(lambda0, majority)`. `lambda0` takes one of
/// the precomputed *slots* (distinct `lambda0(degree, cuts)` values of the
/// degrees present) and majority depends only on the side and the group-size
/// state, so each maintained state maps the *key* `side * slots + slot` of a
/// vertex to a *bucket* (distinct value, ascending). A non-empty bucket is
/// exactly one block of the canonical order.
///
/// A vertex state `(degree, cuts, side)` is a *row* of the cell LUT, which
/// holds its cell in every maintained ranking. The rows of one degree are
/// consecutive, `first + 2 * cuts + side`, so a flip of `v` moves the row of
/// `v` to `2 * first + 2 * degree + 1 - row` and the row of each neighbor by
/// two, exactly as [`PartitionState`] updates the cut counts. A swap moves
/// every vertex it touches once, straight to its final row. Moves re-rank only
/// the moved vertices and their neighbors, in O(1) per vertex and maintained
/// state.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct BuiltinIndex {
    kind: BuiltinFitness,
    /// First LUT row of the degree of each vertex.
    first_row: Vec<u32>,
    /// Current LUT row of each vertex.
    row_of: Vec<u32>,
    /// With `R` maintained rankings, `cell_lut[row * R + r]` is the cell of
    /// `row` in ranking `r`.
    cell_lut: Vec<u32>,
    /// Rankings of the maintained size states.
    rankings: Rankings,
    /// Ranking used for each size state.
    ranking_of_state: [usize; SIZE_STATES],
    /// Swap jobs (empty for flip): the signature of the closed neighborhood
    /// `{v} ∪ N(v)` of each vertex `v`, 64 bytes with bit `u % 512` set for
    /// each member `u`. Disjoint signatures imply disjoint closed
    /// neighborhoods (and are exactly that for `n <= 512`).
    closed: Vec<[u64; SIGNATURE_WORDS]>,
}

/// Canonical rankings of the maintained size states, in shared arrays.
///
/// Ranking `r` owns the buckets `base[r]..base[r + 1]`, its distinct values in
/// ascending order followed by empty padding up to a multiple of [`GROUP`].
/// The *cell* `2 * bucket + side` holds the members of one side of a bucket:
/// inside a block the `false` side precedes the `true` side, each in ascending
/// vertex order. The members of a non-empty cell are a bitset of `ceil(n / 64)`
/// words in a [`Pool`] slot, so a move clears and sets one bit per ranking and
/// the `k`-th member of a cell is found by counting bits. The slot of a cell
/// is its cell ID when the pool is dense; a pooled layout
/// ([`pooled_layout`]) gives a cell a slot only while it is non-empty, so the
/// bitsets stay bounded by the vertex count however many buckets the degrees
/// create. Cell counts and slots, totals and non-empty flags are kept per
/// [`Group`], so a move updates O(1) entries and [`Ranking::locate`] and
/// [`Ranking::for_each_nonempty`] skip empty buckets.
#[derive(Clone, Debug)]
struct Rankings {
    /// First bucket of each ranking, then the total bucket count.
    base: Vec<usize>,
    /// `groups[g]` holds the buckets `GROUP * g..GROUP * (g + 1)`.
    groups: Vec<Group>,
    /// Bitsets of the non-empty cells.
    pool: Pool,
}

/// Cell counts and slots, member total and non-empty flags of [`GROUP`]
/// consecutive buckets.
#[derive(Clone, Copy, Debug)]
struct Group {
    /// Members of the group's buckets.
    size: u32,
    /// Bit `b` is set iff bucket `b` of the group has members.
    nonempty: u64,
    /// `counts[b][side]`: members of the cell `side` of bucket `b`.
    counts: [[u32; 2]; GROUP],
    /// `slots[b][side]`: the [`Pool`] slot of that cell in a pooled layout,
    /// while the cell is non-empty (unused when dense).
    slots: [[u32; 2]; GROUP],
}

const EMPTY_GROUP: Group = Group {
    size: 0,
    nonempty: 0,
    counts: [[0; 2]; GROUP],
    slots: [[0; 2]; GROUP],
};

/// Cell bitsets of `words` words each, one per slot.
///
/// A dense pool has one slot per cell, the cell ID. A pooled one keeps the
/// free slots on a stack: a cell takes the top slot when it gains its first
/// member and returns it when it loses its last one. A free slot's bitset is
/// all zero, so a cell that takes it holds exactly the members it adds. The
/// pool grows, doubling up to `limit` slots, only before a move whose
/// relocations could take more slots than are free; slots are never released
/// to the allocator.
#[derive(Clone, Debug)]
struct Pool {
    /// Whether cells take slots from the free stack.
    pooled: bool,
    /// Words per bitset, `ceil(n / 64)`.
    words: usize,
    /// Bit `v % 64` of `bits[slot * words + v / 64]` is set iff `v` is a
    /// member of the cell holding `slot`.
    bits: Vec<u64>,
    /// Pooled: the free slots are `free[1..=top]` (the top of the stack last)
    /// and `free.len()` is the slot count plus 2, so `free[top + 1]` is in
    /// bounds. Unused when dense.
    free: Vec<u32>,
    top: usize,
    /// Pooled: the most cells that can be non-empty at once,
    /// `min(R * n, cells)` (below `2^32`, so a slot fits in `u32`).
    limit: usize,
}

/// One ranking of [`Rankings`], with bucket and cell IDs relative to it.
#[derive(Clone, Copy)]
struct Ranking<'a> {
    words: usize,
    bits: &'a [u64],
    groups: &'a [Group],
    /// Dense pool: the slot (cell ID) of the ranking's first cell.
    dense_first: Option<usize>,
}

impl BuiltinIndex {
    fn new(
        kind: BuiltinFitness,
        graph: &Graph,
        state: &PartitionState,
        neighborhood: Neighborhood,
    ) -> Result<Self> {
        let too_large = || Error::msg("graph is too large for the EO index");
        let n = graph.node_count();
        if u32::try_from(n).is_err() {
            return Err(too_large());
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
            return Err(too_large());
        }
        // Swap never changes the group sizes; majority-independent values rank
        // identically in every state. An index keeps 1 or `SIZE_STATES`
        // rankings, which `refresh_move` relies on.
        let states = match neighborhood {
            Neighborhood::Swap => vec![size_state(state.size_a(), state.size_b())],
            Neighborhood::Flip if kind.depends_on_majority() => (0..SIZE_STATES).collect(),
            Neighborhood::Flip => vec![size_state(0, 0)],
        };
        let count = states.len();
        let majority: Vec<[bool; 2]> = states.iter().map(|&s| majority_flags(s)).collect();
        let (mut rankings, key_cells) = Rankings::new(kind, &slot_value, &majority, n)?;
        // The rows of each degree present: cut counts ascending, side `false`
        // before `true`.
        let mut degree_row = vec![0; max_degree + 1];
        let mut cell_lut = Vec::new();
        for &d in &degrees {
            degree_row[d] = cell_lut.len() / count;
            for cuts in 0..=d {
                let value = lambda0(d, cuts as i64);
                let slot = slot_value.partition_point(|&x| x < value);
                for key in [slot, slots + slot] {
                    cell_lut.extend_from_slice(&key_cells[key * count..(key + 1) * count]);
                }
            }
        }
        if cell_lut.len() / count > MARK as usize {
            return Err(too_large());
        }
        let first_row: Vec<u32> = (0..n).map(|v| degree_row[graph.degree(v)] as u32).collect();
        let (partition, cuts_at) = (state.partition(), state.cuts_at());
        let row_of: Vec<u32> = (0..n)
            .map(|v| first_row[v] + 2 * cuts_at[v] as u32 + u32::from(partition[v]))
            .collect();
        for (v, &row) in row_of.iter().enumerate() {
            let row = row as usize;
            for &cell in &cell_lut[row * count..(row + 1) * count] {
                rankings.insert(v, cell);
            }
        }
        let ranking_of_state = if count == SIZE_STATES {
            [0, 1, 2]
        } else {
            [0; SIZE_STATES]
        };
        let closed = match neighborhood {
            Neighborhood::Flip => Vec::new(),
            Neighborhood::Swap => (0..n)
                .map(|v| {
                    let mut signature = [0; SIGNATURE_WORDS];
                    for u in std::iter::once(v).chain(graph.neighbors(v).iter().copied()) {
                        let bit = u % (64 * SIGNATURE_WORDS);
                        signature[bit / 64] |= 1 << (bit % 64);
                    }
                    signature
                })
                .collect(),
        };
        Ok(Self {
            kind,
            first_row,
            row_of,
            cell_lut,
            rankings,
            ranking_of_state,
            closed,
        })
    }

    /// Ranking for the current group sizes.
    #[inline]
    fn ranking(&self, state: &PartitionState) -> Ranking<'_> {
        self.rankings
            .get(self.ranking_of_state[size_state(state.size_a(), state.size_b())])
    }

    /// Re-rank the vertices whose fitness `mv` can change after it was applied
    /// to `state`; returns their count.
    fn refresh_move(&mut self, graph: &Graph, state: &PartitionState, mv: Move) -> u64 {
        let touched = match (self.rankings.count(), self.rankings.pool.pooled) {
            (1, false) => self.refresh_move_in::<1, false>(graph, mv),
            (1, true) => self.refresh_move_in::<1, true>(graph, mv),
            (SIZE_STATES, false) => self.refresh_move_in::<SIZE_STATES, false>(graph, mv),
            (SIZE_STATES, true) => self.refresh_move_in::<SIZE_STATES, true>(graph, mv),
            (count, _) => {
                unreachable!("an EO index keeps 1 or {SIZE_STATES} rankings, not {count}")
            }
        };
        debug_assert!(
            self.rows_match(graph, state, mv),
            "the rows follow the cut counts of the state"
        );
        touched
    }

    /// [`Self::refresh_move`] with `R` maintained rankings.
    #[inline(always)]
    fn refresh_move_in<const R: usize, const POOLED: bool>(
        &mut self,
        graph: &Graph,
        mv: Move,
    ) -> u64 {
        match mv {
            Move::Flip(v) => self.refresh_flip::<R, POOLED>(graph, v),
            Move::Swap(a, b) => self.refresh_swap::<R, POOLED>(graph, a, b),
        }
    }

    /// Re-rank after `v` flipped. Flips and swaps have functions of their own,
    /// so that neither weighs on the register use of the other.
    #[inline(never)]
    fn refresh_flip<const R: usize, const POOLED: bool>(&mut self, graph: &Graph, v: usize) -> u64 {
        let touched = 1 + graph.degree(v);
        self.pass::<R, POOLED>(touched).flip(graph, v);
        touched as u64
    }

    /// Re-rank after `a` and `b` swapped, moving every touched vertex once.
    /// When the closed neighborhoods of `a` and `b` are disjoint (their
    /// signatures are), the flip of `a` and the flip of `b` touch disjoint
    /// vertex sets; otherwise [`Rerank::swap_overlapping`] merges them.
    #[inline(never)]
    fn refresh_swap<const R: usize, const POOLED: bool>(
        &mut self,
        graph: &Graph,
        a: usize,
        b: usize,
    ) -> u64 {
        let touched = 2 + graph.degree(a) + graph.degree(b);
        let (closed_a, closed_b) = (self.closed[a], self.closed[b]);
        let overlap = closed_a
            .iter()
            .zip(&closed_b)
            .fold(0, |acc, (x, y)| acc | (x & y));
        let mut pass = self.pass::<R, POOLED>(touched);
        if overlap == 0 {
            pass.flip(graph, a);
            pass.flip(graph, b);
        } else {
            pass.swap_overlapping(graph, a, b);
        }
        touched as u64
    }

    /// A re-ranking pass that relocates up to `touched` vertices.
    #[inline(always)]
    fn pass<const R: usize, const POOLED: bool>(
        &mut self,
        touched: usize,
    ) -> Rerank<'_, R, POOLED> {
        let Self {
            first_row,
            row_of,
            cell_lut,
            rankings,
            ..
        } = self;
        Rerank {
            first_row,
            row_of,
            lut: cell_lut.as_chunks().0,
            cells: rankings.relocator(touched),
        }
    }

    /// Whether the rows of the vertices `mv` touched match `state`.
    fn rows_match(&self, graph: &Graph, state: &PartitionState, mv: Move) -> bool {
        let (partition, cuts) = (state.partition(), state.cuts_at());
        let (a, b) = match mv {
            Move::Flip(v) => (v, v),
            Move::Swap(a, b) => (a, b),
        };
        [a, b]
            .into_iter()
            .flat_map(|v| std::iter::once(v).chain(graph.neighbors(v).iter().copied()))
            .all(|u| {
                self.row_of[u] == self.first_row[u] + 2 * cuts[u] as u32 + u32::from(partition[u])
            })
    }
}

/// One re-ranking pass over a [`BuiltinIndex`] with `R` maintained rankings.
struct Rerank<'a, const R: usize, const POOLED: bool> {
    first_row: &'a [u32],
    row_of: &'a mut [u32],
    lut: &'a [[u32; R]],
    cells: Relocator<'a>,
}

impl<const R: usize, const POOLED: bool> Rerank<'_, R, POOLED> {
    /// Re-rank `v` and its neighbors after `v` flipped.
    #[inline(always)]
    fn flip(&mut self, graph: &Graph, v: usize) {
        // `first + 2 * cuts + side` becomes `first + 2 * (degree - cuts) + 1 - side`.
        let neighbors = graph.neighbors(v);
        let row = self.row_of[v] as usize;
        let new = 2 * self.first_row[v] as usize + 2 * neighbors.len() + 1 - row;
        self.move_to(v, row, new);
        // A neighbor on the new side of `v` loses a cut edge, any other one
        // gains one; its side is the parity of its row.
        let side = new & 1;
        for &u in neighbors {
            let row = self.row_of[u] as usize;
            let new = row + 4 * ((row ^ side) & 1) - 2;
            self.move_to(u, row, new);
        }
    }

    /// Re-rank after `a` and `b` (on opposite sides) swapped, when their
    /// closed neighborhoods may overlap, moving every touched vertex once, to
    /// its final row as [`PartitionState::apply_swap`] leaves it: `a` and `b`
    /// to their flipped rows, two further when they are adjacent (their edge
    /// stays cut), a neighbor of only one of them as in that flip, and a
    /// common neighbor not at all (it loses one cut edge and gains another).
    /// Exact for disjoint closed neighborhoods too.
    ///
    /// Of `x` and `y`, the endpoints with `deg(x) >= deg(y)`, the neighbors
    /// of `y` and `y` itself are marked in the [`MARK`] bit of their rows
    /// first. The pass over the neighbors of `x` unmarks and keeps the marked
    /// ones (the common neighbors, and `y` when adjacent) and moves the
    /// others; the pass over the neighbors of `y` then moves the ones still
    /// marked. `x` is marked only when adjacent and is unmarked before that
    /// pass. Graphs are simple ([`Graph::from_edges`]), so a pass meets each
    /// vertex at most once.
    #[inline(always)]
    fn swap_overlapping(&mut self, graph: &Graph, a: usize, b: usize) {
        let (x, y) = if graph.degree(a) >= graph.degree(b) {
            (a, b)
        } else {
            (b, a)
        };
        let (neighbors_x, neighbors_y) = (graph.neighbors(x), graph.neighbors(y));
        let (row_x, row_y) = (self.row_of[x] as usize, self.row_of[y] as usize);
        // The sides `x` and `y` move to.
        let (side_x, side_y) = ((row_x & 1) ^ 1, (row_y & 1) ^ 1);
        for &u in neighbors_y {
            self.row_of[u] |= MARK;
        }
        self.row_of[y] |= MARK;
        for &u in neighbors_x {
            let row = self.row_of[u];
            if row & MARK != 0 {
                self.row_of[u] = row ^ MARK;
                continue;
            }
            let row = row as usize;
            self.move_to(u, row, row + 4 * ((row ^ side_x) & 1) - 2);
        }
        let adjacent = self.row_of[y] & MARK == 0;
        self.row_of[y] &= !MARK;
        if adjacent {
            self.row_of[x] ^= MARK;
        }
        for &u in neighbors_y {
            let row = self.row_of[u];
            if row & MARK == 0 {
                continue;
            }
            let row = (row ^ MARK) as usize;
            self.move_to(u, row, row + 4 * ((row ^ side_y) & 1) - 2);
        }
        let cut = 2 * usize::from(adjacent);
        for (v, row, degree) in [(x, row_x, neighbors_x.len()), (y, row_y, neighbors_y.len())] {
            let new = 2 * self.first_row[v] as usize + 2 * degree + 1 - row + cut;
            self.move_to(v, row, new);
        }
    }

    /// Move `u` from `row` to `new`.
    #[inline(always)]
    fn move_to(&mut self, u: usize, row: usize, new: usize) {
        #[cfg(test)]
        MOVED_ROWS.with(|moved| moved.set(moved.get() + 1));
        self.row_of[u] = new as u32;
        self.cells
            .relocate::<R, POOLED>(u, self.lut[row], self.lut[new]);
    }
}

/// Mutable view of [`Rankings`] for relocations whose slots the pool has
/// reserved; writes the free-stack top back when dropped.
struct Relocator<'a> {
    groups: &'a mut [Group],
    words: usize,
    bits: &'a mut [u64],
    free: &'a mut [u32],
    top: usize,
    pool_top: &'a mut usize,
    /// Relocations left of those the pool reserved slots for.
    #[cfg(debug_assertions)]
    budget: usize,
}

impl Drop for Relocator<'_> {
    fn drop(&mut self) {
        *self.pool_top = self.top;
    }
}

impl Relocator<'_> {
    /// Move `v` from the cells `old` to the cells `new` of the `R` rankings.
    /// `POOLED` rankings take and free slots; dense ones use the cell ID.
    #[inline(always)]
    fn relocate<const R: usize, const POOLED: bool>(
        &mut self,
        v: usize,
        old: [u32; R],
        new: [u32; R],
    ) {
        #[cfg(debug_assertions)]
        {
            assert!(
                self.budget > 0,
                "more relocations than the pool reserved slots for"
            );
            self.budget -= 1;
        }
        let (word, bit) = (v / 64, 1u64 << (v % 64));
        for r in 0..R {
            let (old_cell, cell) = (old[r], new[r]);
            if old_cell == cell {
                continue;
            }
            let (old_bucket, old_side) = ((old_cell / 2) as usize, (old_cell % 2) as usize);
            let (bucket, side) = ((cell / 2) as usize, (cell % 2) as usize);
            // Leave the old cell; its slot goes back when the cell empties.
            // The slot is written above the stack unconditionally and kept
            // by raising `top`, which avoids an unpredictable branch.
            let old_group = &mut self.groups[old_bucket / GROUP];
            let old_counts = &mut old_group.counts[old_bucket % GROUP];
            old_counts[old_side] -= 1;
            let old_slot = if POOLED {
                old_group.slots[old_bucket % GROUP][old_side]
            } else {
                old_cell
            };
            let old_word = old_slot as usize * self.words + word;
            debug_assert!(
                self.bits[old_word] & bit != 0,
                "an indexed vertex is ranked in its recorded cell"
            );
            self.bits[old_word] &= !bit;
            if POOLED {
                self.free[self.top + 1] = old_slot;
                self.top += usize::from(old_counts[old_side] == 0);
            }
            // The group totals and flags follow without a branch: a move
            // between the sides of one bucket clears its flag only until the
            // join below sets it again and leaves its total unchanged. A
            // bucket that empties had its flag set, so the exclusive or clears
            // it. The other side is loaded on its own: a wide load over the
            // count just stored would stall on store forwarding.
            let emptied = (old_counts[old_side] | old_counts[old_side ^ 1]) == 0;
            old_group.size -= 1;
            old_group.nonempty ^= u64::from(emptied) << (old_bucket % GROUP);
            // Join the new cell, taking the top free slot if it was empty.
            let group = &mut self.groups[bucket / GROUP];
            let slot = if POOLED {
                let taken = group.counts[bucket % GROUP][side] == 0;
                let candidate = self.free[self.top];
                self.top -= usize::from(taken);
                let join = &mut group.slots[bucket % GROUP][side];
                *join = if taken { candidate } else { *join };
                *join
            } else {
                cell
            };
            group.counts[bucket % GROUP][side] += 1;
            self.bits[slot as usize * self.words + word] |= bit;
            group.size += 1;
            group.nonempty |= 1 << (bucket % GROUP);
        }
    }
}

impl Rankings {
    /// Empty rankings of the keys of `slot_value`, one per `majority` flags,
    /// for `n` vertices; also returns `cells[key * R + r]`, the cell of `key`
    /// in ranking `r` of `R`.
    fn new(
        kind: BuiltinFitness,
        slot_value: &[f64],
        majority: &[[bool; 2]],
        n: usize,
    ) -> Result<(Self, Vec<u32>)> {
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
        let pooled = pooled_layout(buckets, n, count);
        #[cfg(test)]
        let pooled = FORCE_POOLED.with(std::cell::Cell::get).unwrap_or(pooled);
        let rankings = Self::empty(base, n, pooled);
        Ok((
            rankings,
            cells.into_iter().map(|cell| cell as u32).collect(),
        ))
    }

    /// Rankings with the buckets `base` and no members, for `n` vertices.
    fn empty(base: Vec<usize>, n: usize, pooled: bool) -> Self {
        let (count, buckets) = (base.len() - 1, base[base.len() - 1]);
        let words = n.div_ceil(64);
        Self {
            base,
            groups: vec![EMPTY_GROUP; buckets / GROUP],
            pool: Pool {
                pooled,
                words,
                bits: if pooled {
                    Vec::new()
                } else {
                    vec![0; 2 * buckets * words]
                },
                free: vec![0; 2],
                top: 0,
                limit: (count * n).min(2 * buckets),
            },
        }
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
            words: self.pool.words,
            bits: &self.pool.bits,
            groups: &self.groups[first / GROUP..end / GROUP],
            dense_first: (!self.pool.pooled).then_some(2 * first),
        }
    }

    /// Pool slot of the non-empty `cell`.
    #[inline]
    fn slot(&self, cell: usize) -> usize {
        if self.pool.pooled {
            self.groups[cell / 2 / GROUP].slots[cell / 2 % GROUP][cell % 2] as usize
        } else {
            cell
        }
    }

    /// Add `v` to `cell` (while building).
    fn insert(&mut self, v: usize, cell: u32) {
        if self.pool.pooled {
            self.pool.reserve(1);
        }
        let (bucket, side) = ((cell / 2) as usize, (cell % 2) as usize);
        let group = &mut self.groups[bucket / GROUP];
        let count = &mut group.counts[bucket % GROUP][side];
        if self.pool.pooled && *count == 0 {
            group.slots[bucket % GROUP][side] = self.pool.free[self.pool.top];
            self.pool.top -= 1;
        }
        *count += 1;
        group.size += 1;
        group.nonempty |= 1 << (bucket % GROUP);
        let slot = self.slot(cell as usize);
        let words = self.pool.words;
        self.pool.bits[slot * words + v / 64] |= 1 << (v % 64);
    }

    /// A [`Relocator`] for relocating up to `vertices` vertices in every
    /// ranking: each relocation takes at most one slot more than it frees.
    #[inline]
    fn relocator(&mut self, vertices: usize) -> Relocator<'_> {
        if self.pool.pooled {
            self.pool.reserve(self.count() * vertices);
        }
        Relocator {
            groups: &mut self.groups,
            words: self.pool.words,
            bits: &mut self.pool.bits,
            free: &mut self.pool.free,
            top: self.pool.top,
            pool_top: &mut self.pool.top,
            #[cfg(debug_assertions)]
            budget: vertices,
        }
    }
}

impl Pool {
    /// Make at least `needed` slots free, or as many as can still be taken:
    /// at most `limit` cells are non-empty at once.
    #[inline]
    fn reserve(&mut self, needed: usize) {
        let in_use = self.free.len() - 2 - self.top;
        let needed = needed.min(self.limit.saturating_sub(in_use));
        if self.top < needed {
            self.grow(in_use + needed);
        }
    }

    /// Grow to at least `slots` slots, doubling up to `limit`.
    #[cold]
    #[inline(never)]
    fn grow(&mut self, slots: usize) {
        let old = self.free.len() - 2;
        let new = (2 * old).max(16).min(self.limit).max(slots);
        self.bits.resize(new * self.words, 0);
        self.free.resize(new + 2, 0);
        // The lowest new slot ends on top of the stack.
        for (i, slot) in (old..new).rev().enumerate() {
            self.free[self.top + 1 + i] = slot as u32;
        }
        self.top += new - old;
    }

    /// The bitset in `slot`.
    #[inline]
    fn bitset(&self, slot: usize) -> &[u64] {
        &self.bits[slot * self.words..(slot + 1) * self.words]
    }
}

/// Logical equality: the same buckets, counts and members; which pool slot
/// holds a cell depends on the history of moves and is not compared. Two dense
/// pools compare all their bits, so stray bits of empty cells also differ.
impl PartialEq for Rankings {
    fn eq(&self, other: &Self) -> bool {
        if self.base != other.base
            || self.pool.words != other.pool.words
            || self.groups.len() != other.groups.len()
        {
            return false;
        }
        let counts = self
            .groups
            .iter()
            .zip(&other.groups)
            .all(|(a, b)| a.size == b.size && a.nonempty == b.nonempty && a.counts == b.counts);
        if !counts {
            return false;
        }
        if !self.pool.pooled && !other.pool.pooled {
            return self.pool.bits == other.pool.bits;
        }
        self.groups.iter().enumerate().all(|(g, group)| {
            let mut mask = group.nonempty;
            while mask != 0 {
                let b = mask.trailing_zeros() as usize;
                for side in 0..2 {
                    let cell = 2 * (GROUP * g + b) + side;
                    if group.counts[b][side] > 0
                        && self.pool.bitset(self.slot(cell)) != other.pool.bitset(other.slot(cell))
                    {
                        return false;
                    }
                }
                mask &= mask - 1;
            }
            true
        })
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

    /// Bitset of the members of `cell`; requires a non-empty cell.
    #[inline]
    fn members(&self, cell: usize) -> &[u64] {
        let slot = match self.dense_first {
            Some(first) => first + cell,
            None => self.groups[cell / 2 / GROUP].slots[cell / 2 % GROUP][cell % 2] as usize,
        };
        &self.bits[slot * self.words..(slot + 1) * self.words]
    }

    /// Member of `cell` with the `k`-th smallest vertex ID; requires `k` below
    /// the cell count.
    #[inline]
    fn select(&self, cell: usize, mut k: usize) -> usize {
        for (w, &word) in self.members(cell).iter().enumerate() {
            // Cells are mostly sparse; skip empty words before counting.
            if word == 0 {
                continue;
            }
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

    /// Whether this job keeps its cell bitsets in a pooled layout.
    pub(super) fn is_pooled(&self) -> bool {
        matches!(&self.ranker, Ranker::Index(index) if index.rankings.pool.pooled)
    }

    /// Assert that the incremental index equals a rebuild from `state` and
    /// that a pooled bitset pool is consistent.
    pub(super) fn assert_index_consistent(
        &self,
        graph: &Graph,
        state: &PartitionState,
        neighborhood: Neighborhood,
    ) {
        if let Ranker::Index(index) = &self.ranker {
            if index.rankings.pool.pooled {
                index.rankings.assert_pool_consistent();
            }
            // A full rebuild has the same parts that do not depend on the
            // state; `rebuilt` takes them from the index, which is cheaper on
            // graphs with many buckets.
            let rebuilt = if graph.node_count() <= 64 {
                BuiltinIndex::new(index.kind, graph, state, neighborhood).unwrap()
            } else {
                index.rebuilt(state)
            };
            assert!(
                *index == rebuilt,
                "incremental EO index differs from a rebuild"
            );
        }
    }
}

/// Run `f` with the EO indexes it builds on this thread in a pooled (`true`)
/// or dense layout, whatever [`pooled_layout`] would choose.
#[cfg(test)]
pub(super) fn with_layout<T>(pooled: bool, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_POOLED.with(|force| force.set(self.0));
        }
    }
    let _restore = Restore(FORCE_POOLED.with(|force| force.replace(Some(pooled))));
    f()
}

#[cfg(test)]
impl BuiltinIndex {
    /// The index of `state` built from scratch, reusing the parts that do not
    /// depend on the state: the kind, the rows of the degrees, the cell LUT,
    /// the buckets and the bitset layout.
    fn rebuilt(&self, state: &PartitionState) -> Self {
        let (partition, cuts) = (state.partition(), state.cuts_at());
        let row_of: Vec<u32> = (0..self.first_row.len())
            .map(|v| self.first_row[v] + 2 * cuts[v] as u32 + u32::from(partition[v]))
            .collect();
        let count = self.rankings.count();
        let mut rankings = Rankings::empty(
            self.rankings.base.clone(),
            row_of.len(),
            self.rankings.pool.pooled,
        );
        for (v, &row) in row_of.iter().enumerate() {
            let row = row as usize;
            for &cell in &self.cell_lut[row * count..(row + 1) * count] {
                rankings.insert(v, cell);
            }
        }
        Self {
            kind: self.kind,
            first_row: self.first_row.clone(),
            row_of,
            cell_lut: self.cell_lut.clone(),
            rankings,
            ranking_of_state: self.ranking_of_state,
            closed: self.closed.clone(),
        }
    }
}

#[cfg(test)]
impl Rankings {
    /// Move `v` from the cells `old` to the cells `new` of the `R` rankings.
    fn relocate_one<const R: usize>(&mut self, v: usize, old: [u32; R], new: [u32; R]) {
        if self.pool.pooled {
            self.relocator(1).relocate::<R, true>(v, old, new);
        } else {
            self.relocator(1).relocate::<R, false>(v, old, new);
        }
    }

    /// Assert the invariants of the groups and the bitset pool: group totals
    /// and flags match the counts, every non-empty cell holds its count of
    /// bits in its own slot, and the remaining bits are clear. A dense pool
    /// has one slot per cell, the cell ID; a pooled one has distinct slots for
    /// the non-empty cells and exactly the other slots on its free stack.
    fn assert_pool_consistent(&self) {
        let pool = &self.pool;
        let slots = pool.bits.len() / pool.words.max(1);
        assert_eq!(pool.bits.len(), slots * pool.words);
        let mut owner = vec![None; slots];
        for (g, group) in self.groups.iter().enumerate() {
            let mut size = 0;
            for (b, counts) in group.counts.iter().enumerate() {
                let count = counts[0] + counts[1];
                size += count;
                assert_eq!(
                    group.nonempty >> b & 1 == 1,
                    count > 0,
                    "group {g} bucket {b}"
                );
                for (side, &members) in counts.iter().enumerate() {
                    let id = 2 * (GROUP * g + b) + side;
                    if members > 0 || !pool.pooled {
                        let slot = self.slot(id);
                        let bits: u32 = pool.bitset(slot).iter().map(|w| w.count_ones()).sum();
                        assert_eq!(bits, members, "members of cell {id}");
                        let previous = owner[slot].replace(id);
                        assert_eq!(previous, None, "slot {slot} is shared");
                    }
                }
            }
            assert_eq!(group.size, size, "group {g}");
        }
        if pool.pooled {
            assert_eq!(pool.free.len(), slots + 2);
            assert!(slots <= pool.limit, "{slots} slots, limit {}", pool.limit);
            for &slot in &pool.free[1..=pool.top] {
                let slot = slot as usize;
                assert_eq!(owner[slot].replace(usize::MAX), None, "free slot {slot}");
                assert!(pool.bitset(slot).iter().all(|&w| w == 0));
            }
            assert!(
                owner.iter().all(Option::is_some),
                "every slot is taken or free"
            );
        } else {
            assert_eq!(slots, 2 * GROUP * self.groups.len());
        }
    }
}

#[cfg(test)]
#[path = "eo_tests.rs"]
mod tests;
