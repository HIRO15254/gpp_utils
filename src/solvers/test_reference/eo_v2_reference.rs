// Naive executable specification of EO selection rule v2 (algorithm `v2`).
//
// Written independently of the production EO code in `../eo.rs`: every step
// evaluates the fitness through `VertexFitness::values`, sorts all vertices into
// the canonical order `(lambda, side, vertex)`, rebuilds the cumulative
// power-law weights and applies the block-averaged selection rule literally,
// with linear scans instead of indexes. Only shared infrastructure (graph,
// partition state and RNG derivation) is reused. Test-only.

use crate::error::{Error, Result};
use crate::experiment::config::{Condition, Neighborhood, SolverSpec};
use crate::fitness::{FitnessRegistry, VertexFitness};
use crate::graph_partition::{Graph, Move, PartitionState};
use crate::optimization::rng_for;
use rand::{Rng, seq::SliceRandom};
use rand_mt::Mt19937GenRand64;

/// One block of the canonical order that has members on the conditioned side.
pub struct Eligible {
    start: usize,
    end: usize,
    size: usize,
    opposite: usize,
    weight: f64,
    share: f64,
}

/// Canonical order `(lambda, side, vertex)` and its maximal blocks of equal lambda.
pub fn canonical(lambda: &[f64], side: &[bool]) -> (Vec<usize>, Vec<(usize, usize)>) {
    let n = lambda.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        lambda[a]
            .partial_cmp(&lambda[b])
            .expect("finite fitness values")
            .then(side[a].cmp(&side[b]))
            .then(a.cmp(&b))
    });
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && lambda[order[end]] == lambda[order[start]] {
            end += 1;
        }
        blocks.push((start, end));
        start = end;
    }
    (order, blocks)
}

/// Cumulative power-law weights of ranks `1..=n`, normalized by their total.
pub fn cumulative(n: usize, tau: f64) -> Vec<f64> {
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

/// First vertex for the uniform value `u`: the first position whose cumulative
/// weight exceeds `u`, then the member given by the residual inside its block.
pub fn first_for(order: &[usize], blocks: &[(usize, usize)], cum: &[f64], u: f64) -> usize {
    let n = order.len();
    let p = cum.iter().position(|&c| c > u).unwrap_or(n - 1);
    let &(s, e) = blocks
        .iter()
        .find(|&&(s, e)| s <= p && p < e)
        .expect("blocks cover every position");
    let lo = if s == 0 { 0.0 } else { cum[s - 1] };
    let hi = cum[e - 1];
    let m = e - s;
    let per = (hi - lo) / m as f64;
    let off = if per > 0.0 {
        (((u - lo).max(0.0) / per) as usize).min(m - 1)
    } else {
        0
    };
    order[s + off]
}

/// Blocks with side-`o` members, in canonical order, with their shares.
pub fn eligible_blocks(
    order: &[usize],
    blocks: &[(usize, usize)],
    cum: &[f64],
    side: &[bool],
    o: bool,
) -> Vec<Eligible> {
    let mut eligible = Vec::new();
    for &(start, end) in blocks {
        let size = end - start;
        let opposite = order[start..end].iter().filter(|&&v| side[v] == o).count();
        if opposite > 0 {
            let before = if start == 0 { 0.0 } else { cum[start - 1] };
            let weight = cum[end - 1] - before;
            let share = weight * opposite as f64 / size as f64;
            eligible.push(Eligible {
                start,
                end,
                size,
                opposite,
                weight,
                share,
            });
        }
    }
    eligible
}

/// Second swap vertex for the uniform value `u2`; `eligible` must be non-empty.
pub fn second_from(eligible: &[Eligible], order: &[usize], side: &[bool], o: bool, u2: f64) -> usize {
    let mut total = 0.0;
    for block in eligible {
        total += block.share;
    }
    let (chosen, j) = if total > 0.0 {
        let target = u2 * total;
        let mut acc = 0.0;
        let mut chosen = None;
        for (i, block) in eligible.iter().enumerate() {
            let next = acc + block.share;
            if target < next {
                chosen = Some((i, acc));
                break;
            }
            acc = next;
        }
        let (i, acc) = chosen.unwrap_or_else(|| {
            // Nothing selected: take the last eligible block with the
            // running sum accumulated before it.
            let last = eligible.len() - 1;
            let mut before = 0.0;
            for block in &eligible[..last] {
                before += block.share;
            }
            (last, before)
        });
        let block = &eligible[i];
        let per = block.weight / block.size as f64;
        let j = if per > 0.0 {
            (((target - acc).max(0.0) / per) as usize).min(block.opposite - 1)
        } else {
            0
        };
        (i, j)
    } else {
        let block = &eligible[0];
        (
            0,
            ((u2 * block.opposite as f64) as usize).min(block.opposite - 1),
        )
    };
    let block = &eligible[chosen];
    order[block.start..block.end]
        .iter()
        .copied()
        .filter(|&v| side[v] == o)
        .nth(j)
        .expect("j is below the opposite-side count")
}

pub struct ReferenceEo<'a> {
    graph: &'a Graph,
    neighborhood: Neighborhood,
    tau: f64,
    fitness: Box<dyn VertexFitness>,
    pub state: PartitionState,
    pub select_rng: Mt19937GenRand64,
    pub tie_rng: Mt19937GenRand64,
    pub smooth_rng: Mt19937GenRand64,
}

impl<'a> ReferenceEo<'a> {
    /// Same initial partition and RNG derivation labels as the engine.
    pub fn new(
        graph: &'a Graph,
        condition: &Condition,
        seed: u64,
        registry: &FitnessRegistry,
    ) -> Result<Self> {
        let SolverSpec::Eo { tau, fitness } = &condition.solver else {
            return Err(Error::msg("the EO reference needs an EO condition"));
        };
        let n = graph.node_count();
        let hash = graph.content_hash();
        let neighborhood_label = match condition.neighborhood {
            Neighborhood::Flip => b"flip".as_slice(),
            Neighborhood::Swap => b"swap".as_slice(),
        };
        let seed_bytes = seed.to_le_bytes();
        let mut init = rng_for(&[hash.as_bytes(), neighborhood_label, &seed_bytes, b"initial"]);
        let mut partition: Vec<bool> = (0..n).map(|_| init.r#gen()).collect();
        if condition.neighborhood == Neighborhood::Swap {
            partition.fill(false);
            for x in partition.iter_mut().take(n / 2) {
                *x = true;
            }
            partition.shuffle(&mut init);
        }
        let solver = serde_json::to_vec(&condition.solver)?;
        let alpha = condition.alpha.to_bits().to_le_bytes();
        let version = registry
            .versions()
            .get(&fitness.kind)
            .cloned()
            .unwrap_or_default();
        let stream = |label: &[u8]| {
            rng_for(&[
                hash.as_bytes(),
                neighborhood_label,
                &seed_bytes,
                &alpha,
                &solver,
                version.as_bytes(),
                b"algorithm-v1",
                label,
            ])
        };
        Ok(Self {
            graph,
            neighborhood: condition.neighborhood,
            tau: *tau,
            fitness: registry.create(fitness)?,
            state: PartitionState::new(graph, partition)?,
            select_rng: stream(b"select"),
            tie_rng: stream(b"tie"),
            smooth_rng: stream(b"smooth"),
        })
    }

    /// Select one move from the current state, apply it and return it.
    pub fn step(&mut self) -> Result<Move> {
        let n = self.graph.node_count();
        let lambda = self.fitness.values(self.graph, &self.state)?;
        if lambda.len() != n || lambda.iter().any(|x| !x.is_finite()) {
            return Err(Error::msg("fitness returned invalid values"));
        }
        if n == 0 {
            return Err(Error::msg("empty EO conditional rank distribution"));
        }
        let side = self.state.partition().to_vec();
        let (order, blocks) = canonical(&lambda, &side);
        let cum = cumulative(n, self.tau);
        let first = first_for(&order, &blocks, &cum, self.select_rng.r#gen());
        let mv = match self.neighborhood {
            Neighborhood::Flip => Move::Flip(first),
            Neighborhood::Swap => {
                let o = !side[first];
                let eligible = eligible_blocks(&order, &blocks, &cum, &side, o);
                if eligible.is_empty() {
                    return Err(Error::msg("empty EO conditional rank distribution"));
                }
                let u2: f64 = self.select_rng.r#gen();
                Move::Swap(first, second_from(&eligible, &order, &side, o, u2))
            }
        };
        match mv {
            Move::Flip(v) => self.state.apply_flip(self.graph, v),
            Move::Swap(a, b) => self.state.apply_swap(self.graph, a, b),
        }
        Ok(mv)
    }
}
