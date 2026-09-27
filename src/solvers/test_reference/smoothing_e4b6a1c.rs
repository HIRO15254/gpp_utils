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
    mut rng: Option<&mut Mt19937GenRand64>,
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
    let first = moves_cancellable(state, neighborhood, cancel)?;
    if first.is_empty() {
        *evaluations += 1;
        return Ok(real);
    }
    let k = match spec {
        SmoothingSpec::RandomKAverage { k } => Some(*k),
        _ => None,
    };
    let first_take = k.map_or(first.len(), |x| x.min(first.len()));
    let mut indices: Vec<usize> = (0..first.len()).collect();
    if k.is_some() {
        let r = rng
            .as_deref_mut()
            .ok_or_else(|| Error::msg("random smoothing requires RNG"))?;
        for i in 0..first_take {
            if i & 1023 == 0 {
                cancel.check()?;
            }
            let j = r.gen_range(i..indices.len());
            indices.swap(i, j);
        }
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for (q, &i) in indices[..first_take].iter().enumerate() {
        if q & 1023 == 0 {
            cancel.check()?
        }
        total += move_score(state, graph, first[i], alpha);
        count += 1;
    }
    if let Some(k) = k
        && k > count
    {
        let needed = k - count;
        let r = rng.unwrap();
        let distance_two =
            (max_random_k(graph.node_count(), neighborhood) as usize).saturating_sub(first.len());
        // Floyd's algorithm samples exact canonical ordinals without replacement.
        let mut ordinals = std::collections::BTreeSet::new();
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
            let mut s = state.clone();
            apply_distance_two(&mut s, graph, neighborhood, ordinal);
            total += s.score(alpha);
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
fn apply_distance_two(
    state: &mut PartitionState,
    graph: &Graph,
    neighborhood: Neighborhood,
    ordinal: usize,
) {
    match neighborhood {
        Neighborhood::Flip => {
            let (a, b) = nth_pair(graph.node_count(), ordinal);
            state.apply_flip(graph, a);
            state.apply_flip(graph, b);
        }
        Neighborhood::Swap => {
            let side_a: Vec<_> = state
                .partition()
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| x.then_some(i))
                .collect();
            let side_b: Vec<_> = state
                .partition()
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| (!x).then_some(i))
                .collect();
            let combinations = side_a.len() * (side_a.len() - 1) / 2;
            let (ai, aj) = nth_pair(side_a.len(), ordinal / combinations);
            let (bi, bj) = nth_pair(side_b.len(), ordinal % combinations);
            state.apply_swap(graph, side_a[ai], side_b[bi]);
            state.apply_swap(graph, side_a[aj], side_b[bj]);
        }
    }
}

fn nth_pair(n: usize, mut ordinal: usize) -> (usize, usize) {
    for a in 0..n - 1 {
        let row = n - a - 1;
        if ordinal < row {
            return (a, a + 1 + ordinal);
        }
        ordinal -= row;
    }
    unreachable!("validated distance-two ordinal")
}
