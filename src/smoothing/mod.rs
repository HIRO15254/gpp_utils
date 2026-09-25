#![allow(clippy::too_many_arguments)]

use crate::error::{Error, Result};
use crate::experiment::config::{Neighborhood, SmoothingSpec};
use crate::graph_partition::{Graph, Move, PartitionState};
use crate::optimization::CancellationToken;
use rand::Rng;
use rand_mt::Mt19937GenRand64;

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

/// Canonically ordered neighborhood without materializing every move.
///
/// Swap owns only its two side arrays, so it remains valid while callers
/// temporarily apply and undo candidate moves on the source state.
pub(crate) struct MoveSequence {
    neighborhood: Neighborhood,
    node_count: usize,
    side_a: Vec<usize>,
    side_b: Vec<usize>,
}

impl MoveSequence {
    pub(crate) fn new(
        state: &PartitionState,
        neighborhood: Neighborhood,
        cancel: &CancellationToken,
    ) -> Result<Self> {
        let mut side_a = Vec::new();
        let mut side_b = Vec::new();
        if matches!(neighborhood, Neighborhood::Swap) {
            for (v, &side) in state.partition().iter().enumerate() {
                if v & 1023 == 0 {
                    cancel.check()?;
                }
                if side {
                    side_a.push(v);
                } else {
                    side_b.push(v);
                }
            }
        }
        Ok(Self {
            neighborhood,
            node_count: state.partition().len(),
            side_a,
            side_b,
        })
    }

    pub(crate) fn len(&self) -> usize {
        match self.neighborhood {
            Neighborhood::Flip => self.node_count,
            Neighborhood::Swap => self.side_a.len() * self.side_b.len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn get(&self, index: usize) -> Move {
        assert!(index < self.len(), "move index out of bounds");
        match self.neighborhood {
            Neighborhood::Flip => Move::Flip(index),
            Neighborhood::Swap => {
                let width = self.side_b.len();
                Move::Swap(self.side_a[index / width], self.side_b[index % width])
            }
        }
    }

    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = Move> + '_ {
        let neighborhood = self.neighborhood;
        let side_a = &self.side_a;
        let side_b = &self.side_b;
        let mut row = 0usize;
        let mut column = 0usize;
        (0..self.len()).map(move |index| match neighborhood {
            Neighborhood::Flip => Move::Flip(index),
            Neighborhood::Swap => {
                let mv = Move::Swap(side_a[row], side_b[column]);
                column += 1;
                if column == side_b.len() {
                    column = 0;
                    row += 1;
                }
                mv
            }
        })
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
    if matches!(spec, SmoothingSpec::None) | matches!(spec, SmoothingSpec::WeightedAverage { k: 0 })
    {
        *evaluations += 1;
        return Ok(real);
    }
    let first = MoveSequence::new(state, neighborhood, cancel)?;
    if first.is_empty() {
        *evaluations += 1;
        return Ok(real);
    }
    let plan = SmoothingPlan::new(&first, graph.node_count(), neighborhood, spec, rng, cancel)?;
    evaluate_prepared(
        state,
        graph,
        alpha,
        neighborhood,
        spec,
        &plan,
        &first,
        cancel,
        evaluations,
    )
}

pub(crate) enum SmoothingPlan {
    Canonical,
    Random {
        first_indices: Vec<usize>,
        distance_two_ordinals: Vec<usize>,
    },
}

impl SmoothingPlan {
    fn new(
        first: &MoveSequence,
        node_count: usize,
        neighborhood: Neighborhood,
        spec: &SmoothingSpec,
        rng: Option<&mut Mt19937GenRand64>,
        cancel: &CancellationToken,
    ) -> Result<Self> {
        let SmoothingSpec::RandomKAverage { k } = spec else {
            return Ok(Self::Canonical);
        };
        let first_take = (*k).min(first.len());
        let mut indices = (0..first.len()).collect::<Vec<_>>();
        let source = rng.ok_or_else(|| Error::msg("random smoothing requires RNG"))?;
        for i in 0..first_take {
            if i & 1023 == 0 {
                cancel.check()?;
            }
            let j = source.gen_range(i..indices.len());
            indices.swap(i, j);
        }
        indices.truncate(first_take);
        let needed = k.saturating_sub(first_take);
        let distance_two =
            (max_random_k(node_count, neighborhood) as usize).saturating_sub(first.len());
        let mut ordinals = std::collections::BTreeSet::new();
        for j in distance_two - needed..distance_two {
            if j & 1023 == 0 {
                cancel.check()?;
            }
            let candidate = source.gen_range(0..=j);
            if !ordinals.insert(candidate) {
                ordinals.insert(j);
            }
        }
        Ok(Self::Random {
            first_indices: indices,
            distance_two_ordinals: ordinals.into_iter().collect(),
        })
    }
}

pub(crate) fn plan(
    state: &PartitionState,
    graph: &Graph,
    neighborhood: Neighborhood,
    spec: &SmoothingSpec,
    rng: Option<&mut Mt19937GenRand64>,
    cancel: &CancellationToken,
) -> Result<SmoothingPlan> {
    validate(spec, graph.node_count(), neighborhood)?;
    let first = MoveSequence::new(state, neighborhood, cancel)?;
    SmoothingPlan::new(&first, graph.node_count(), neighborhood, spec, rng, cancel)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_with_plan(
    state: &PartitionState,
    graph: &Graph,
    alpha: f64,
    neighborhood: Neighborhood,
    spec: &SmoothingSpec,
    plan: &SmoothingPlan,
    cancel: &CancellationToken,
    evaluations: &mut u64,
) -> Result<f64> {
    validate(spec, graph.node_count(), neighborhood)?;
    let first = MoveSequence::new(state, neighborhood, cancel)?;
    evaluate_prepared(
        state,
        graph,
        alpha,
        neighborhood,
        spec,
        plan,
        &first,
        cancel,
        evaluations,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_prepared(
    state: &PartitionState,
    graph: &Graph,
    alpha: f64,
    neighborhood: Neighborhood,
    spec: &SmoothingSpec,
    plan: &SmoothingPlan,
    first: &MoveSequence,
    cancel: &CancellationToken,
    evaluations: &mut u64,
) -> Result<f64> {
    let real = state.score(alpha);
    if matches!(spec, SmoothingSpec::None) | matches!(spec, SmoothingSpec::WeightedAverage { k: 0 })
    {
        *evaluations += 1;
        return Ok(real);
    }
    if first.is_empty() {
        *evaluations += 1;
        return Ok(real);
    }
    let k = match spec {
        SmoothingSpec::RandomKAverage { k } => Some(*k),
        _ => None,
    };
    let first_take = k.map_or(first.len(), |x| x.min(first.len()));
    if let SmoothingPlan::Random {
        first_indices,
        distance_two_ordinals,
    } = plan
    {
        let mut total = 0.0;
        for (q, &i) in first_indices.iter().enumerate() {
            if q & 1023 == 0 {
                cancel.check()?
            }
            total += move_score(state, graph, first.get(i), alpha);
        }
        return evaluate_after_first(
            state,
            graph,
            alpha,
            neighborhood,
            spec,
            cancel,
            evaluations,
            first,
            total,
            first_take,
            distance_two_ordinals,
        );
    }
    let mut total = 0.0;
    for (q, mv) in first.iter().take(first_take).enumerate() {
        if q & 1023 == 0 {
            cancel.check()?;
        }
        total += move_score(state, graph, mv, alpha);
    }
    evaluate_after_first(
        state,
        graph,
        alpha,
        neighborhood,
        spec,
        cancel,
        evaluations,
        first,
        total,
        first_take,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_after_first(
    state: &PartitionState,
    graph: &Graph,
    alpha: f64,
    neighborhood: Neighborhood,
    spec: &SmoothingSpec,
    cancel: &CancellationToken,
    evaluations: &mut u64,
    first: &MoveSequence,
    mut total: f64,
    mut count: usize,
    distance_two_ordinals: &[usize],
) -> Result<f64> {
    let real = state.score(alpha);
    let k = match spec {
        SmoothingSpec::RandomKAverage { k } => Some(*k),
        _ => None,
    };
    if k.is_some_and(|k| k > count) {
        let mut scratch = state.clone();
        for (q, &ordinal) in distance_two_ordinals.iter().enumerate() {
            if q & 1023 == 0 {
                cancel.check()?;
            }
            let pair = distance_two_moves(graph.node_count(), neighborhood, ordinal, first);
            apply(&mut scratch, graph, pair.0);
            apply(&mut scratch, graph, pair.1);
            total += scratch.score(alpha);
            apply(&mut scratch, graph, pair.1);
            apply(&mut scratch, graph, pair.0);
            count += 1;
        }
    }
    *evaluations += count as u64;
    let avg = total / count as f64;
    Ok(match spec {
        SmoothingSpec::WeightedAverage { k } => {
            let w = (*k).min(first.len()) as f64 / first.len() as f64;
            w * avg + (1.0 - w) * real
        }
        _ => avg,
    })
}
fn distance_two_moves(
    node_count: usize,
    neighborhood: Neighborhood,
    ordinal: usize,
    sequence: &MoveSequence,
) -> (Move, Move) {
    match neighborhood {
        Neighborhood::Flip => {
            let (a, b) = nth_pair(node_count, ordinal);
            (Move::Flip(a), Move::Flip(b))
        }
        Neighborhood::Swap => {
            let side_a = &sequence.side_a;
            let side_b = &sequence.side_b;
            let combinations = side_a.len() * (side_a.len() - 1) / 2;
            let (ai, aj) = nth_pair(side_a.len(), ordinal / combinations);
            let (bi, bj) = nth_pair(side_b.len(), ordinal % combinations);
            (
                Move::Swap(side_a[ai], side_b[bi]),
                Move::Swap(side_a[aj], side_b[bj]),
            )
        }
    }
}

fn nth_pair(n: usize, ordinal: usize) -> (usize, usize) {
    debug_assert!(ordinal < n * n.saturating_sub(1) / 2);
    let prefix = |a: usize| a * (2 * n - a - 1) / 2;
    let mut low = 0usize;
    let mut high = n - 1;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if prefix(mid) <= ordinal {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    let a = low;
    (a, a + 1 + ordinal - prefix(a))
}

#[cfg(test)]
#[allow(dead_code)]
mod test_reference;

#[cfg(test)]
mod tests {
    use super::*;
    use rand_mt::Mt19937GenRand64;

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

    #[test]
    fn fixed_random_plan_matches_resampling_from_same_rng_for_each_state() {
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
        let cancel = CancellationToken::default();
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            let initial = if matches!(neighborhood, Neighborhood::Flip) {
                vec![true, false, true, false, true, false, false, true]
            } else {
                vec![true, true, true, true, false, false, false, false]
            };
            let base = PartitionState::new(&graph, initial).unwrap();
            let first_count = MoveSequence::new(&base, neighborhood, &cancel)
                .unwrap()
                .len();
            let spec = SmoothingSpec::RandomKAverage { k: first_count + 3 };
            let mut plan_rng = Mt19937GenRand64::new(0x8899_aabb_ccdd_eeff);
            let plan = plan(
                &base,
                &graph,
                neighborhood,
                &spec,
                Some(&mut plan_rng),
                &cancel,
            )
            .unwrap();
            let moves = MoveSequence::new(&base, neighborhood, &cancel).unwrap();
            for state in std::iter::once(base.clone()).chain(moves.iter().take(4).map(|mv| {
                let mut state = base.clone();
                apply(&mut state, &graph, mv);
                state
            })) {
                let mut reference_rng = Mt19937GenRand64::new(0x8899_aabb_ccdd_eeff);
                let mut reference_evals = 0;
                let reference = evaluate(
                    &state,
                    &graph,
                    0.05,
                    neighborhood,
                    &spec,
                    Some(&mut reference_rng),
                    &cancel,
                    &mut reference_evals,
                )
                .unwrap();
                let mut planned_evals = 0;
                let planned = evaluate_with_plan(
                    &state,
                    &graph,
                    0.05,
                    neighborhood,
                    &spec,
                    &plan,
                    &cancel,
                    &mut planned_evals,
                )
                .unwrap();
                assert_eq!(planned.to_bits(), reference.to_bits());
                assert_eq!(planned_evals, reference_evals);
                let mut consumed_plan_rng = plan_rng.clone();
                for _ in 0..624 {
                    assert_eq!(consumed_plan_rng.next_u64(), reference_rng.next_u64());
                }
            }
        }
    }

    #[test]
    fn move_sequence_preserves_canonical_vec_order() {
        let graph = Graph::from_edges(6, vec![[0, 1], [1, 2], [2, 3]]).unwrap();
        let state =
            PartitionState::new(&graph, vec![true, false, true, false, false, true]).unwrap();
        let cancel = CancellationToken::default();
        let sequence = MoveSequence::new(&state, Neighborhood::Swap, &cancel).unwrap();
        assert_eq!(
            sequence.iter().collect::<Vec<_>>(),
            vec![
                Move::Swap(0, 1),
                Move::Swap(0, 3),
                Move::Swap(0, 4),
                Move::Swap(2, 1),
                Move::Swap(2, 3),
                Move::Swap(2, 4),
                Move::Swap(5, 1),
                Move::Swap(5, 3),
                Move::Swap(5, 4),
            ]
        );
        for i in 0..sequence.len() {
            assert_eq!(sequence.get(i), sequence.iter().nth(i).unwrap());
        }
    }

    #[test]
    fn integer_pair_decoder_matches_canonical_nested_loops() {
        for n in 2..40 {
            let expected = (0..n - 1)
                .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
                .collect::<Vec<_>>();
            let actual = (0..expected.len())
                .map(|i| nth_pair(n, i))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }

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
}
