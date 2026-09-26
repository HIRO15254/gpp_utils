//! Exact best-improvement scan of the real objective.
//!
//! The basin hill climb of the runner and the real-objective path of HC share
//! one definition of a scan: start from `best = start` without a choice, visit
//! every move of [`crate::smoothing::moves`] in its canonical order (Flip:
//! `v = 0..n`; Swap: `a` over group A ascending, then `b` over group B
//! ascending), count one objective evaluation per move and apply
//!
//! ```text
//! if x < best { best = x; choice = Some(mv); ties = 1 }
//! else if choice.is_some() && x == best {
//!     ties += 1; if tie_rng.gen_range(0..ties) == 0 { choice = Some(mv) }
//! }
//! ```
//!
//! to the score `x` of each move. [`BestImprovement`] performs exactly this
//! scan without materializing the move list. A move with `x > best`, or with
//! `x >= best` while there is no choice yet, changes neither `best`, `choice`,
//! `ties` nor the tie RNG. Such moves are skipped when an integer bound proves
//! it; the other moves are scored with the expression of the
//! [`PartitionState`] methods in canonical order, so the result, the tie draws
//! and the evaluation count equal those of the full scan.
//!
//! A descent (one basin or one HC run) keeps per-vertex gains and gain-level
//! bitsets of its partition, updated in `O(degree)` per applied move, so a
//! scan reads one bitset word per 64 vertices (Flip) or per 64 vertices of a
//! row (Swap) instead of every vertex or every pair; see [`Tracker`].

use super::{Graph, Move, PartitionState};
use crate::error::{Error, Result};
use crate::experiment::config::Neighborhood;
use crate::optimization::CancellationToken;
use rand::Rng;
use rand_mt::Mt19937GenRand64;

/// Logical candidates (including skipped ones) between cancellation checks.
const CHECK_INTERVAL: usize = 1024;
const _: () = assert!(CHECK_INTERVAL.is_multiple_of(64));

/// Gains beyond `±LEVEL_CAP` share the outermost levels of [`Tracker`]; this
/// bounds its memory for graphs with hubs.
const LEVEL_CAP: i64 = 64;

/// What a scan does with a non-finite candidate score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NonFinite {
    /// Compare it by the rule like any other score (basin measurement).
    Compare,
    /// Fail with "non-finite search evaluation" at the first such candidate in
    /// canonical order, after counting its evaluation (HC).
    Reject,
}

fn non_finite() -> Error {
    Error::msg("non-finite search evaluation")
}

/// Running state of the best-improvement rule.
struct Rule {
    best: f64,
    choice: Option<Move>,
    ties: u64,
}

impl Rule {
    fn new(start: f64) -> Self {
        Self {
            best: start,
            choice: None,
            ties: 0,
        }
    }

    /// Apply the rule to candidate score `x`; returns whether `best` dropped.
    #[inline]
    fn consider(&mut self, x: f64, mv: Move, tie_rng: &mut Mt19937GenRand64) -> bool {
        if x < self.best {
            self.best = x;
            self.choice = Some(mv);
            self.ties = 1;
            true
        } else {
            if self.choice.is_some() && x == self.best {
                self.ties += 1;
                if tie_rng.gen_range(0..self.ties) == 0 {
                    self.choice = Some(mv);
                }
            }
            false
        }
    }

    /// Whether a candidate scoring `x`, or at least `x`, can change the state
    /// (`best`, `choice`, `ties` or the tie RNG). False for NaN.
    #[inline]
    fn may_change(&self, x: f64) -> bool {
        if self.choice.is_some() {
            x <= self.best
        } else {
            x < self.best
        }
    }

    fn outcome(self) -> Option<(Move, f64)> {
        self.choice.map(|mv| (mv, self.best))
    }
}

/// `f(s) = (cut + s) as f64 + penalty` over integers `s` in `-radius..=radius`.
///
/// Integer-to-float conversion and adding a fixed float are monotone, so `f`
/// is non-decreasing in `s`, and [`Rule::may_change`] holds on a prefix of
/// the range. [`Self::limit`] returns the end of that prefix.
struct Bound {
    cut: i64,
    penalty: f64,
    radius: i64,
}

impl Bound {
    /// `f(s)`: the bits of [`PartitionState::flip_score`] for `s = gain(v)`
    /// and of [`PartitionState::swap_score`] for
    /// `s = gain(a) + gain(b) + 2 * adjacent` under the matching penalty.
    #[inline(always)]
    fn score(&self, s: i64) -> f64 {
        (self.cut + s) as f64 + self.penalty
    }

    /// The largest `s` in `-radius..=radius` with `rule.may_change(f(s))`, or
    /// `-radius - 1` if there is none. Within the range, `s <= limit` holds
    /// exactly when `rule.may_change(f(s))` does.
    fn limit(&self, rule: &Rule) -> i64 {
        let admissible = |s: i64| rule.may_change(self.score(s));
        let (mut yes, mut no) = (-self.radius - 1, self.radius + 1);
        // Invariant: every `s <= yes` in range is admissible and no `s >= no`
        // is; the sentinels themselves are never evaluated. The first two
        // probes sit at the prefix end estimated in floating point (`f(s)`
        // is about `cut + s + penalty`); truncation instead of `floor` and
        // rounding can put it one off, which the second probe covers. Every
        // probe evaluates the exact predicate and the bisection finishes any
        // other case, so the estimate only saves probes.
        let guess = ((rule.best - self.penalty) as i64)
            .saturating_sub(self.cut)
            .clamp(-self.radius, self.radius);
        if admissible(guess) {
            yes = guess;
            if guess + 1 < no {
                if admissible(guess + 1) {
                    yes = guess + 1;
                } else {
                    no = guess + 1;
                }
            }
        } else {
            no = guess;
            if guess - 1 > yes {
                if admissible(guess - 1) {
                    yes = guess - 1;
                } else {
                    no = guess - 1;
                }
            }
        }
        while no - yes > 1 {
            let mid = yes + (no - yes) / 2;
            if admissible(mid) {
                yes = mid;
            } else {
                no = mid;
            }
        }
        yes
    }
}

/// Incremental data of the partition a descent works on.
///
/// With `gain(v) = degree(v) - 2 * cuts_at[v]` (the change of the cut size
/// when `v` flips) and `level(v) = clamp(gain(v), -cap - 1, cap + 1) + cap + 1`
/// for `cap = min(max degree, level cap)`, bit `v % 64` of
/// `levels[i * words + v / 64]` is set exactly when `level(v) <= i`. For
/// `-cap - 1 <= t <= cap` the set of level [`Self::level`]`(t)` is therefore
/// exactly `{v : gain(v) <= t}`, and for any `t` it contains that set (the
/// top level holds every vertex). The scans use it only to skip vertices
/// and compare the exact gain of every vertex they take from it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Tracker {
    /// Whether the fields describe a partition.
    active: bool,
    n: usize,
    words: usize,
    /// The largest degree; every gain lies in `-dmax..=dmax`.
    dmax: i64,
    cap: i64,
    gain: Vec<i64>,
    /// Bit `v % 64` of word `v / 64` is set when `v` is in group A.
    in_a: Vec<u64>,
    levels: Vec<u64>,
    /// Whether `count_b` is kept (Swap scans need it).
    counts: bool,
    /// `count_b[g + dmax]`: vertices of group B with gain `g`; empty unless
    /// `counts`.
    count_b: Vec<u32>,
}

impl Tracker {
    /// Rebuild all fields for `state`.
    fn track(&mut self, graph: &Graph, state: &PartitionState, level_cap: i64, counts: bool) {
        let partition = state.partition();
        let cuts = state.cuts_at();
        let n = partition.len();
        let words = n.div_ceil(64);
        self.active = true;
        self.n = n;
        self.words = words;
        let mut dmax = 0;
        self.gain.clear();
        self.gain.extend(cuts.iter().enumerate().map(|(v, &c)| {
            let degree = graph.degree(v) as i64;
            dmax = dmax.max(degree);
            degree - 2 * c
        }));
        self.dmax = dmax;
        self.cap = dmax.min(level_cap);
        self.counts = counts;
        self.count_b.clear();
        if counts {
            self.count_b.resize(2 * dmax as usize + 1, 0);
        }
        self.in_a.clear();
        self.in_a.resize(words, 0);
        self.levels.clear();
        self.levels.resize(self.level_count() * words, 0);
        let cap = self.cap;
        for (w, (sides, gains)) in partition.chunks(64).zip(self.gain.chunks(64)).enumerate() {
            let mut in_a_word = 0;
            for (j, (&in_a, &g)) in sides.iter().zip(gains).enumerate() {
                in_a_word |= u64::from(in_a) << j;
                if counts && !in_a {
                    self.count_b[(g + dmax) as usize] += 1;
                }
                let level = (g.clamp(-cap - 1, cap + 1) + cap + 1) as usize;
                self.levels[level * words + w] |= 1 << j;
            }
            self.in_a[w] = in_a_word;
        }
        // Each level so far holds the vertices exactly at it; accumulate.
        if words > 0 {
            for i in 1..self.level_count() {
                let (lower, upper) = self.levels.split_at_mut(i * words);
                for (word, below) in upper[..words].iter_mut().zip(&lower[(i - 1) * words..]) {
                    *word |= below;
                }
            }
        }
    }

    fn level_count(&self) -> usize {
        2 * self.cap as usize + 3
    }

    /// The level whose set contains `{v : gain(v) <= t}`, exactly for
    /// `-cap - 1 <= t <= cap` (see the type documentation).
    #[inline(always)]
    fn level(&self, t: i64) -> usize {
        (t.clamp(-self.cap - 1, self.cap + 1) + self.cap + 1) as usize
    }

    /// Word `w` of the set of level `level`.
    #[inline(always)]
    fn word(&self, level: usize, w: usize) -> u64 {
        self.levels[level * self.words + w]
    }

    /// The words of the set of level `level`.
    #[inline(always)]
    fn set(&self, level: usize) -> &[u64] {
        &self.levels[level * self.words..][..self.words]
    }

    /// The smallest gain in group B; requires `counts` and a nonempty group.
    fn min_gain_b(&self) -> i64 {
        let first = self.count_b.iter().position(|&c| c > 0);
        first.expect("group B is not empty") as i64 - self.dmax
    }

    /// Update the level sets for the gain of `v` changing from `g` to `h`.
    #[inline(always)]
    fn relevel(&mut self, v: usize, g: i64, h: i64) {
        let (from, to) = (self.level(g), self.level(h));
        let (words, bit) = (self.words, 1u64 << (v % 64));
        // `v` joins the sets of levels `to..from` or leaves those of
        // `from..to`.
        let mut i = from.min(to) * words + v / 64;
        let end = from.max(to) * words;
        if to < from {
            while i < end {
                self.levels[i] |= bit;
                i += words;
            }
        } else {
            while i < end {
                self.levels[i] &= !bit;
                i += words;
            }
        }
    }

    /// Record the flip of `v` (as [`PartitionState::apply_flip`] does).
    #[inline]
    fn flip(&mut self, graph: &Graph, v: usize) {
        let dmax = self.dmax;
        let (w, bit) = (v / 64, 1u64 << (v % 64));
        let was_in_a = self.in_a[w] & bit != 0;
        let g = self.gain[v];
        // `v` changes group and its cut edges become uncut and vice versa.
        if self.counts {
            if was_in_a {
                self.count_b[(dmax - g) as usize] += 1;
            } else {
                self.count_b[(g + dmax) as usize] -= 1;
            }
        }
        self.relevel(v, g, -g);
        self.gain[v] = -g;
        self.in_a[w] ^= bit;
        // No gain is clamped when `cap == dmax`: then `level(g) = g + cap + 1`,
        // and a gain change by 2 moves a vertex across exactly two levels.
        let unclamped = self.cap == dmax;
        let words = self.words;
        for &u in graph.neighbors(v) {
            let (wu, bit) = (u / 64, 1u64 << (u % 64));
            let in_a = self.in_a[wu] & bit != 0;
            let g = self.gain[u];
            // The edge to `u` becomes cut when `u` is in the group `v` left.
            let drop = in_a == was_in_a;
            let h = if drop { g - 2 } else { g + 2 };
            if self.counts && !in_a {
                self.count_b[(g + dmax) as usize] -= 1;
                self.count_b[(h + dmax) as usize] += 1;
            }
            if unclamped {
                // `u` joins the sets of levels `level(h)` and `level(h) + 1`
                // (drop) or leaves those of `level(g)` and `level(g) + 1`.
                let first = (g.min(h) + self.cap + 1) as usize * words + wu;
                let joined = if drop { bit } else { 0 };
                for i in [first, first + words] {
                    self.levels[i] = self.levels[i] & !bit | joined;
                }
            } else {
                self.relevel(u, g, h);
            }
            self.gain[u] = h;
        }
    }

    /// Panics unless the fields equal those rebuilt for `state`.
    #[cfg(debug_assertions)]
    fn assert_tracks(&self, graph: &Graph, state: &PartitionState, level_cap: i64) {
        let mut fresh = Tracker::default();
        fresh.track(graph, state, level_cap, self.counts);
        assert!(*self == fresh, "the descent data does not track the state");
    }
}

/// The cancellation checks of a Swap scan that visits only some rows.
///
/// The full scan checks before the rows (ranks in group A) `period - 1`,
/// `2 * period - 1`, ... with `period = ceil(CHECK_INTERVAL / row_len)`: before
/// the row with which the rows since the last check reach [`CHECK_INTERVAL`]
/// candidates. A scan that skips rows makes each such check before the next
/// row it visits or, after its last one, before counting.
struct RowChecks {
    period: usize,
    /// The first check row not yet covered by a check.
    next: usize,
}

impl RowChecks {
    fn new(row_len: usize) -> Self {
        let period = CHECK_INTERVAL.div_ceil(row_len);
        Self {
            period,
            next: period - 1,
        }
    }

    /// Whether to check before visiting the row of rank `rank` (ranks
    /// increase): true when a check row in `..=rank` is not yet covered.
    #[inline]
    fn due(&mut self, rank: usize) -> bool {
        if rank < self.next {
            return false;
        }
        // The first check row after `rank`.
        self.next = ((rank + 1) / self.period + 1) * self.period - 1;
        true
    }

    /// Whether to check after the last visited row of a scan with `rows` rows.
    fn due_at_end(&self, rows: usize) -> bool {
        self.next < rows
    }
}

/// Exact best-improvement scan of the real objective with reusable buffers.
///
/// One value serves any number of scans under a fixed neighborhood and
/// `alpha`. A descent (one basin or one HC run) calls [`Self::track`] once,
/// then [`Self::scan_tracked`] and, for each chosen move, [`Self::apply`].
#[derive(Clone, Debug)]
pub(crate) struct BestImprovement {
    neighborhood: Neighborhood,
    alpha: f64,
    non_finite: NonFinite,
    /// [`LEVEL_CAP`]; tests lower it to exercise clamped levels.
    level_cap: i64,
    tracker: Tracker,
    /// Candidates scored by the last scan.
    #[cfg(test)]
    scored: u64,
}

impl BestImprovement {
    pub(crate) fn new(neighborhood: Neighborhood, alpha: f64, non_finite: NonFinite) -> Self {
        Self {
            neighborhood,
            alpha,
            non_finite,
            level_cap: LEVEL_CAP,
            tracker: Tracker::default(),
            #[cfg(test)]
            scored: 0,
        }
    }

    /// Whether [`Self::track`] has been called.
    pub(crate) fn is_tracking(&self) -> bool {
        self.tracker.active
    }

    /// Start tracking `state` for [`Self::scan_tracked`] and [`Self::apply`].
    pub(crate) fn track(&mut self, graph: &Graph, state: &PartitionState) {
        let counts = self.neighborhood == Neighborhood::Swap;
        self.tracker.track(graph, state, self.level_cap, counts);
    }

    /// Apply `mv` to the tracked `state` (as [`crate::smoothing::apply`]) and
    /// to the tracked data.
    pub(crate) fn apply(&mut self, graph: &Graph, state: &mut PartitionState, mv: Move) {
        match mv {
            Move::Flip(v) => {
                state.apply_flip(graph, v);
                self.tracker.flip(graph, v);
            }
            Move::Swap(a, b) => {
                state.apply_swap(graph, a, b);
                self.tracker.flip(graph, a);
                self.tracker.flip(graph, b);
            }
        }
    }

    /// [`Self::track`] `state`, then [`Self::scan_tracked`]: one scan of any
    /// state (tests).
    #[cfg(test)]
    pub(crate) fn scan(
        &mut self,
        graph: &Graph,
        state: &PartitionState,
        start: f64,
        tie_rng: &mut Mt19937GenRand64,
        cancel: &CancellationToken,
        evaluations: &mut u64,
    ) -> Result<Option<(Move, f64)>> {
        self.track(graph, state);
        self.scan_tracked(graph, state, start, tie_rng, cancel, evaluations)
    }

    /// One scan of the whole neighborhood of the tracked `state`, starting
    /// from `best = start`. `state` must be the one passed to [`Self::track`],
    /// changed since then only by [`Self::apply`] (checked in debug builds).
    ///
    /// Returns the chosen move and its score, or `None` when no move improves
    /// on `start`. The result and the tie RNG draws equal those of the full
    /// scan described in the module documentation. `evaluations` advances by
    /// the neighborhood size (`n` for Flip, `|A| * |B|` for Swap) when the
    /// scan completes. Cancellation is checked first and then wherever the
    /// full scan checks it: before vertex `v` of a Flip scan when `v` is a
    /// positive multiple of [`CHECK_INTERVAL`], and before the rows of a Swap
    /// scan given by [`RowChecks`], where a check due at a skipped row is made
    /// before the next visited row or before counting.
    ///
    /// With `gain(v) = degree(v) - 2 * cuts_at[v]` and the balance penalty
    /// `alpha * d as f64 * d as f64` of the resulting group sizes, the score
    /// is `f(s) = (cut + s) as f64 + penalty` (see [`Bound`]) with
    /// `s = gain(v)` for `Flip(v)` and `s = gain(a) + gain(b) + 2 * adjacent`
    /// for `Swap(a, b)`. A Flip's penalty depends only on the side of `v`, and
    /// no swap changes it. Because `f` is non-decreasing, a Flip with
    /// `gain(v) > limit` and a Swap with `gain(a) + gain(b) > limit` (a lower
    /// bound of its `s`) cannot change the rule and are skipped; `limit` is
    /// recomputed whenever `best` drops. The remaining candidates are found
    /// from the gain-level bitsets in canonical order and scored with `f`,
    /// which has the bits of [`PartitionState::flip_score`] or
    /// [`PartitionState::swap_score`].
    pub(crate) fn scan_tracked(
        &mut self,
        graph: &Graph,
        state: &PartitionState,
        start: f64,
        tie_rng: &mut Mt19937GenRand64,
        cancel: &CancellationToken,
        evaluations: &mut u64,
    ) -> Result<Option<(Move, f64)>> {
        cancel.check()?;
        assert!(
            self.tracker.active && self.tracker.n == state.partition().len(),
            "scan_tracked needs the tracked state"
        );
        #[cfg(debug_assertions)]
        {
            self.tracker.assert_tracks(graph, state, self.level_cap);
        }
        #[cfg(test)]
        {
            self.scored = 0;
        }
        match self.neighborhood {
            Neighborhood::Flip => self.scan_flip(graph, state, start, tie_rng, cancel, evaluations),
            Neighborhood::Swap => self.scan_swap(graph, state, start, tie_rng, cancel, evaluations),
        }
    }

    fn scan_flip(
        &mut self,
        graph: &Graph,
        state: &PartitionState,
        start: f64,
        tie_rng: &mut Mt19937GenRand64,
        cancel: &CancellationToken,
        evaluations: &mut u64,
    ) -> Result<Option<(Move, f64)>> {
        let tracker = &self.tracker;
        let n = tracker.n;
        let size_a = state.size_a();
        // The penalty `PartitionState::flip_score` computes for a vertex of
        // group B (index 0) or A (index 1). A group without vertices never
        // uses its entry, which is then 0.
        let penalty = |size_after: Option<usize>| {
            size_after.map_or(0.0, |a| {
                let d = a as i64 - (n - a) as i64;
                self.alpha * d as f64 * d as f64
            })
        };
        let penalties = [
            penalty((size_a < n).then_some(size_a + 1)),
            penalty(size_a.checked_sub(1)),
        ];
        // The score is a finite integer conversion plus the penalty, so it is
        // non-finite exactly when the penalty is: with `Reject`, the full scan
        // fails at the first vertex whose group has a non-finite penalty.
        let stop = match self.non_finite {
            NonFinite::Reject if !penalties.iter().all(|p| p.is_finite()) => state
                .partition()
                .iter()
                .position(|&in_a| !penalties[usize::from(in_a)].is_finite())
                .unwrap_or(n),
            _ => n,
        };
        let cut = state.cut_edges() as i64;
        let bounds = penalties.map(|penalty| Bound {
            cut,
            penalty,
            radius: n as i64,
        });
        let mut rule = Rule::new(start);
        let mut limits = bounds.each_ref().map(|bound| bound.limit(&rule));
        let mut levels = limits.map(|limit| tracker.level(limit));
        for w in 0..stop.div_ceil(64) {
            let base = w * 64;
            if w != 0 && base.is_multiple_of(CHECK_INTERVAL) {
                cancel.check()?;
            }
            let in_a = tracker.in_a[w];
            let candidates = |levels: [usize; 2]| {
                tracker.word(levels[0], w) & !in_a | tracker.word(levels[1], w) & in_a
            };
            let mut bits = candidates(levels);
            if stop - base < 64 {
                bits &= (1u64 << (stop - base)) - 1;
            }
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let v = base + j;
                let side = (in_a >> j & 1) as usize;
                let gain = tracker.gain[v];
                if gain > limits[side] {
                    continue;
                }
                #[cfg(test)]
                {
                    self.scored += 1;
                }
                let x = bounds[side].score(gain);
                debug_assert_eq!(
                    x.to_bits(),
                    state.flip_score(graph, v, self.alpha).to_bits()
                );
                if rule.consider(x, Move::Flip(v), tie_rng) {
                    limits = bounds.each_ref().map(|bound| bound.limit(&rule));
                    levels = limits.map(|limit| tracker.level(limit));
                    bits &= candidates(levels);
                }
            }
        }
        if stop < n {
            *evaluations += stop as u64 + 1;
            return Err(non_finite());
        }
        *evaluations += n as u64;
        Ok(rule.outcome())
    }

    fn scan_swap(
        &mut self,
        graph: &Graph,
        state: &PartitionState,
        start: f64,
        tie_rng: &mut Mt19937GenRand64,
        cancel: &CancellationToken,
        evaluations: &mut u64,
    ) -> Result<Option<(Move, f64)>> {
        let tracker = &self.tracker;
        let rows = state.size_a();
        let row_len = state.size_b();
        let total = rows as u64 * row_len as u64;
        if total == 0 {
            return Ok(None);
        }
        // The expression of `PartitionState::swap_score`; swaps keep `d`.
        let d = state.size_a() as i64 - state.size_b() as i64;
        let penalty = self.alpha * d as f64 * d as f64;
        if self.non_finite == NonFinite::Reject && !penalty.is_finite() {
            // Every score is a finite integer conversion plus `penalty`, so the
            // full scan fails at its first candidate.
            *evaluations += 1;
            return Err(non_finite());
        }
        let bound = Bound {
            cut: state.cut_edges() as i64,
            penalty,
            radius: 2 * tracker.n as i64,
        };
        let mut rule = Rule::new(start);
        let mut limit = bound.limit(&rule);
        // Every candidate of the row of `a` has `s >= gain(a) + min_b`.
        let min_b = tracker.min_gain_b();
        let mut checks = RowChecks::new(row_len);
        let mut rank_base = 0;
        for w in 0..tracker.words {
            let in_a = tracker.in_a[w];
            let mut row_bits = tracker.word(tracker.level(limit - min_b), w) & in_a;
            while row_bits != 0 {
                let j = row_bits.trailing_zeros() as usize;
                row_bits &= row_bits - 1;
                let a = w * 64 + j;
                let ga = tracker.gain[a];
                if ga + min_b > limit {
                    continue;
                }
                let rank = rank_base + (in_a & ((1u64 << j) - 1)).count_ones() as usize;
                if checks.due(rank) {
                    cancel.check()?;
                }
                // The level set holding every `b` with `ga + gain(b) <= limit`.
                let mut set = tracker.set(tracker.level(limit - ga));
                for (wb, &in_a_b) in tracker.in_a.iter().enumerate() {
                    let mut bits = set[wb] & !in_a_b;
                    while bits != 0 {
                        let jb = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let b = wb * 64 + jb;
                        let gb = tracker.gain[b];
                        if ga + gb > limit {
                            continue;
                        }
                        #[cfg(test)]
                        {
                            self.scored += 1;
                        }
                        let x = bound.score(ga + gb + 2 * i64::from(graph.has_edge(a, b)));
                        debug_assert_eq!(
                            x.to_bits(),
                            state.swap_score(graph, a, b, self.alpha).to_bits()
                        );
                        if rule.consider(x, Move::Swap(a, b), tie_rng) {
                            limit = bound.limit(&rule);
                            set = tracker.set(tracker.level(limit - ga));
                            bits &= set[wb];
                        }
                    }
                }
                row_bits &= tracker.word(tracker.level(limit - min_b), w);
            }
            rank_base += in_a.count_ones() as usize;
        }
        if checks.due_at_end(rows) {
            cancel.check()?;
        }
        *evaluations += total;
        Ok(rule.outcome())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smoothing;
    use rand::seq::SliceRandom;

    /// The full scan with the configuration of `scan`, as the runner and HC
    /// wrote it before this module: every canonical move, one evaluation each,
    /// the rule on `move_score`, and for `Reject` the HC check after counting
    /// the evaluation.
    fn naive_scan(
        scan: &BestImprovement,
        graph: &Graph,
        state: &PartitionState,
        start: f64,
        tie_rng: &mut Mt19937GenRand64,
        evaluations: &mut u64,
    ) -> std::result::Result<Option<(Move, f64)>, ()> {
        let mut best = start;
        let mut choice = None;
        let mut ties = 0u64;
        for mv in smoothing::moves(state, scan.neighborhood) {
            *evaluations += 1;
            let x = smoothing::move_score(state, graph, mv, scan.alpha);
            if scan.non_finite == NonFinite::Reject && !x.is_finite() {
                return Err(());
            }
            if x < best {
                best = x;
                choice = Some(mv);
                ties = 1;
            } else if choice.is_some() && x == best {
                ties += 1;
                if tie_rng.gen_range(0..ties) == 0 {
                    choice = Some(mv);
                }
            }
        }
        Ok(choice.map(|mv| (mv, best)))
    }

    fn bits(outcome: Option<(Move, f64)>) -> Option<(Move, u64)> {
        outcome.map(|(mv, x)| (mv, x.to_bits()))
    }

    fn complete(n: usize) -> Graph {
        Graph::from_edges(
            n,
            (0..n)
                .flat_map(|a| (a + 1..n).map(move |b| [a, b]))
                .collect(),
        )
        .unwrap()
    }

    fn star(n: usize) -> Graph {
        Graph::from_edges(n, (1..n).map(|b| [0, b]).collect()).unwrap()
    }

    fn random_graph(n: usize, p: f64, seed: u64) -> Graph {
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
    }

    /// A clique on the first half, a path on most of the rest, isolated tail.
    fn mixed(n: usize) -> Graph {
        let half = n / 2;
        let mut edges: Vec<[usize; 2]> = (0..half)
            .flat_map(|a| (a + 1..half).map(move |b| [a, b]))
            .collect();
        edges.extend((half..n.saturating_sub(3)).map(|v| [v, v + 1]));
        Graph::from_edges(n, edges).unwrap()
    }

    /// Graphs with many ties (complete, isolated, star, mixed) and random ones.
    fn graphs() -> Vec<Graph> {
        let mut out = vec![
            complete(2),
            complete(8),
            complete(11),
            Graph::from_edges(10, vec![]).unwrap(),
            star(10),
            star(13),
            mixed(14),
        ];
        for (n, p, seed) in [
            (6, 0.5, 1),
            (10, 0.3, 2),
            (16, 0.2, 3),
            (16, 0.8, 4),
            (24, 0.15, 5),
            (31, 0.1, 6),
            (40, 0.25, 7),
        ] {
            out.push(random_graph(n, p, seed));
        }
        out
    }

    /// Random arbitrary and balanced (the Swap invariant) partitions, and the
    /// two one-sided extremes.
    fn partitions(n: usize, rng: &mut Mt19937GenRand64) -> Vec<Vec<bool>> {
        let mut out = vec![vec![true; n], vec![false; n]];
        for _ in 0..4 {
            out.push((0..n).map(|_| rng.r#gen()).collect());
            let mut balanced: Vec<bool> = (0..n).map(|v| v < n / 2).collect();
            balanced.shuffle(rng);
            out.push(balanced);
        }
        out
    }

    /// How often the scans took the branches the equivalence relies on.
    #[derive(Default)]
    struct Coverage {
        scans: u64,
        candidates: u64,
        scored: u64,
        partially_skipped: u64,
        tie_draws: u64,
        improvements: u64,
        errors: u64,
    }

    /// Run one scan and the naive scan from the same tie RNG and counter;
    /// compare the outcome bits, the evaluation count and the whole RNG state.
    fn assert_scan_matches(
        scan: &mut BestImprovement,
        graph: &Graph,
        state: &PartitionState,
        start: f64,
        tie_rng: &mut Mt19937GenRand64,
        evaluations: &mut u64,
        coverage: &mut Coverage,
    ) -> Option<(Move, f64)> {
        let mut naive_rng = tie_rng.clone();
        let mut naive_evaluations = *evaluations;
        let before = *evaluations;
        let untouched = tie_rng.clone();
        let fast = scan.scan(
            graph,
            state,
            start,
            tie_rng,
            &CancellationToken::new(),
            evaluations,
        );
        let naive = naive_scan(
            scan,
            graph,
            state,
            start,
            &mut naive_rng,
            &mut naive_evaluations,
        );
        let context = format!(
            "{:?} alpha={:e} start={start:e} {:?} partition={:?}",
            scan.neighborhood,
            scan.alpha,
            scan.non_finite,
            state.partition()
        );
        assert_eq!(*evaluations, naive_evaluations, "{context}");
        assert!(*tie_rng == naive_rng, "tie RNG differs: {context}");
        let candidates = *evaluations - before;
        coverage.scans += 1;
        coverage.candidates += candidates;
        coverage.scored += scan.scored;
        coverage.partially_skipped += u64::from(0 < scan.scored && scan.scored < candidates);
        coverage.tie_draws += u64::from(*tie_rng != untouched);
        match (fast, naive) {
            (Ok(fast), Ok(naive)) => {
                assert_eq!(bits(fast), bits(naive), "{context}");
                coverage.improvements += u64::from(fast.is_some());
                fast
            }
            (Err(error), Err(())) => {
                assert_eq!(error.to_string(), "non-finite search evaluation");
                coverage.errors += 1;
                None
            }
            (fast, naive) => panic!("{context}: {fast:?} vs {naive:?}"),
        }
    }

    #[test]
    fn bound_limit_is_the_end_of_the_admissible_prefix() {
        let mut rng = Mt19937GenRand64::new(11);
        let penalties = [0.0, -0.0, 0.05, 2.5, -3.0, 1.0e17, 1.0e300, f64::MAX];
        let non_finite = [f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
        for penalty in penalties.into_iter().chain(non_finite) {
            for radius in [0, 1, 2, 7, 40] {
                for cut in [0, 1, 5, 100] {
                    for _ in 0..20 {
                        let base = (cut + rng.gen_range(-radius - 3..=radius + 3)) as f64 + penalty;
                        let best = match rng.gen_range(0..4) {
                            0 => base,
                            1 => base + rng.r#gen::<f64>() - 0.5,
                            2 => f64::NAN,
                            _ => [f64::INFINITY, f64::NEG_INFINITY][rng.gen_range(0..2)],
                        };
                        for choice in [None, Some(Move::Flip(0))] {
                            let rule = Rule {
                                best,
                                choice,
                                ties: 1,
                            };
                            let bound = Bound {
                                cut,
                                penalty,
                                radius,
                            };
                            let expected = (-radius..=radius)
                                .rev()
                                .find(|&s| rule.may_change((cut + s) as f64 + penalty))
                                .unwrap_or(-radius - 1);
                            let limit = bound.limit(&rule);
                            assert_eq!(limit, expected, "{cut} {penalty:e} {best:e} {radius}");
                            for s in -radius..=radius {
                                assert_eq!(
                                    s <= limit,
                                    rule.may_change((cut + s) as f64 + penalty),
                                    "prefix {s}: {cut} {penalty:e} {best:e} {radius}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn single_scans_match_the_naive_full_scan() {
        let mut rng = Mt19937GenRand64::new(20260926);
        let mut coverage = [Coverage::default(), Coverage::default()];
        for graph in graphs() {
            for partition in partitions(graph.node_count(), &mut rng) {
                let state = PartitionState::new(&graph, partition).unwrap();
                for (index, neighborhood) in [Neighborhood::Flip, Neighborhood::Swap]
                    .into_iter()
                    .enumerate()
                {
                    for alpha in [0.0, 0.05, 1.0, -0.0, f64::MIN_POSITIVE, 0.3, 1.0e300] {
                        let mut scan =
                            BestImprovement::new(neighborhood, alpha, NonFinite::Compare);
                        let score = state.score(alpha);
                        // Starts other than the current score make the first
                        // candidates improve and then tie.
                        for start in [score, score - 1.0, score + 1.0, score - 2.5, score + 4.0] {
                            for seed in [0, 1, 99] {
                                assert_scan_matches(
                                    &mut scan,
                                    &graph,
                                    &state,
                                    start,
                                    &mut Mt19937GenRand64::new(seed),
                                    &mut 7,
                                    &mut coverage[index],
                                );
                            }
                        }
                    }
                }
            }
        }
        for coverage in coverage {
            assert!(coverage.scored * 2 < coverage.candidates);
            assert!(coverage.partially_skipped * 4 > coverage.scans);
            assert!(coverage.tie_draws * 4 > coverage.scans);
            assert!(coverage.improvements * 2 > coverage.scans);
        }
    }

    #[test]
    fn extreme_starts_and_non_finite_scores_match_the_naive_full_scan() {
        let mut rng = Mt19937GenRand64::new(5);
        let mut coverage = Coverage::default();
        for graph in [complete(8), star(9), random_graph(12, 0.4, 8), mixed(10)] {
            for partition in partitions(graph.node_count(), &mut rng) {
                let state = PartitionState::new(&graph, partition).unwrap();
                for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
                    for alpha in [f64::MAX, f64::INFINITY, f64::NAN, -1.0, 0.05, 1.0e308] {
                        for non_finite in [NonFinite::Compare, NonFinite::Reject] {
                            let mut scan = BestImprovement::new(neighborhood, alpha, non_finite);
                            for start in [
                                state.score(alpha),
                                f64::INFINITY,
                                f64::NEG_INFINITY,
                                f64::NAN,
                                0.0,
                            ] {
                                assert_scan_matches(
                                    &mut scan,
                                    &graph,
                                    &state,
                                    start,
                                    &mut Mt19937GenRand64::new(3),
                                    &mut 0,
                                    &mut coverage,
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(coverage.errors * 8 > coverage.scans);
        assert!(coverage.improvements * 16 > coverage.scans);
        assert!(coverage.tie_draws * 16 > coverage.scans);
    }

    /// Repeat scan and apply until no move is chosen, as the basin and HC do,
    /// comparing every step with the naive scan.
    #[test]
    fn whole_descents_match_the_naive_full_scan() {
        let mut rng = Mt19937GenRand64::new(77);
        let mut graphs = graphs();
        graphs.push(random_graph(64, 0.1, 9));
        graphs.push(random_graph(80, 0.05, 10));
        graphs.push(complete(20));
        let mut coverage = [Coverage::default(), Coverage::default()];
        let mut longest = 0;
        for graph in graphs {
            for partition in partitions(graph.node_count(), &mut rng) {
                for (index, neighborhood) in [Neighborhood::Flip, Neighborhood::Swap]
                    .into_iter()
                    .enumerate()
                {
                    for alpha in [0.0, 0.05, 1.0] {
                        for (seed, non_finite) in
                            [(0, NonFinite::Compare), (123, NonFinite::Reject)]
                        {
                            let mut state = PartitionState::new(&graph, partition.clone()).unwrap();
                            let mut tie_rng = Mt19937GenRand64::new(seed);
                            let mut evaluations = 1;
                            let mut current = state.score(alpha);
                            let mut scan = BestImprovement::new(neighborhood, alpha, non_finite);
                            for step in 0.. {
                                let Some((mv, best)) = assert_scan_matches(
                                    &mut scan,
                                    &graph,
                                    &state,
                                    current,
                                    &mut tie_rng,
                                    &mut evaluations,
                                    &mut coverage[index],
                                ) else {
                                    longest = longest.max(step);
                                    break;
                                };
                                smoothing::apply(&mut state, &graph, mv);
                                current = best;
                                assert_eq!(
                                    current.to_bits(),
                                    graph.score(state.partition(), alpha).to_bits()
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(longest >= 20);
        for coverage in coverage {
            assert!(coverage.scored * 4 < coverage.candidates);
            assert!(coverage.partially_skipped * 2 > coverage.scans);
            assert!(coverage.tie_draws * 4 > coverage.scans);
        }
    }

    #[test]
    fn cancelled_scan_fails_before_counting_or_drawing() {
        let graph = complete(6);
        let state =
            PartitionState::new(&graph, vec![true, false, true, false, true, false]).unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            let mut rng = Mt19937GenRand64::new(4);
            let untouched = rng.clone();
            let mut evaluations = 3;
            let mut scan = BestImprovement::new(neighborhood, 0.05, NonFinite::Compare);
            let start = state.score(0.05) + 1.0;
            assert!(
                scan.scan(&graph, &state, start, &mut rng, &cancel, &mut evaluations)
                    .is_err()
            );
            assert_eq!(evaluations, 3);
            assert!(rng == untouched);
        }
    }
}

// Independent-review tests: an oracle that recomputes `Graph::score` on copied
// partitions, adversarial alphas and starts, n > CHECK_INTERVAL, cancellation.
#[cfg(test)]
#[path = "descent_review_tests.rs"]
mod review_tests;

// Tracked descents (`track`, `scan_tracked`, `apply`) against a full scan on
// many graphs, level caps and alphas; cancellation; HC and the runner against
// the frozen 51577f9 engine and runner on larger graphs.
#[cfg(test)]
#[path = "descent_tracked_tests.rs"]
mod tracked_tests;
