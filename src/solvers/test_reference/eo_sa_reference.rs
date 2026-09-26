// Naive executable specification of EO-SA (`eo_sa`): moves proposed by EO
// selection rule v2 and judged by the SA Metropolis rule at a fixed temperature.
//
// Written from `docs/algorithms.md` (section "EO-SA") independently of the
// production engine in `../engine.rs` and the EO index in `../eo.rs`, neither
// of which it calls. The proposal reuses the naive EO v2 selection functions
// of `eo_v2_reference.rs`, which the including module must have in scope as
// `super::eo_v2`: every step evaluates the fitness through
// `VertexFitness::values`, sorts all vertices into the canonical order and
// draws from the `select` stream (once for Flip, twice for Swap) whatever the
// outcome. The candidate is scored by recomputing the objective of a copied
// partition from the edge list (`Graph::score`), the Metropolis rule is the
// literal short-circuit expression with its own `accept` stream, an accepted
// move rebuilds the partition state from scratch, and the diagnostic counters
// follow the formulas of the specification. Only shared infrastructure (graph,
// partition state, fitness registry and RNG derivation) is reused. Test-only.

use super::eo_v2::{canonical, cumulative, eligible_blocks, first_for, second_from};
use crate::error::{Error, Result};
use crate::experiment::config::{Condition, Neighborhood, SolverSpec};
use crate::fitness::{FitnessRegistry, VertexFitness};
use crate::graph_partition::{Graph, Move, PartitionState};
use crate::optimization::rng_for;
use rand::{Rng, seq::SliceRandom};
use rand_mt::Mt19937GenRand64;

/// What one judged step did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outcome {
    /// Proposed move, whether or not it was accepted.
    pub mv: Move,
    /// Real score of the candidate state, recomputed from scratch.
    pub next: f64,
    /// `next - current`, with `current` the score held before the step.
    pub delta: f64,
    /// Whether one value was drawn from the `accept` stream.
    pub drew: bool,
    pub accepted: bool,
}

fn neighborhood_label(neighborhood: Neighborhood) -> &'static [u8] {
    match neighborhood {
        Neighborhood::Flip => b"flip",
        Neighborhood::Swap => b"swap",
    }
}

/// Initial partition shared with EO: `n` booleans from the seed's `initial`
/// stream; for Swap, `n / 2` vertices in group A shuffled by the same stream.
pub fn initial_partition(graph: &Graph, neighborhood: Neighborhood, seed: u64) -> Vec<bool> {
    let n = graph.node_count();
    let hash = graph.content_hash();
    let mut init = rng_for(&[
        hash.as_bytes(),
        neighborhood_label(neighborhood),
        &seed.to_le_bytes(),
        b"initial",
    ]);
    let mut partition: Vec<bool> = (0..n).map(|_| init.r#gen()).collect();
    if neighborhood == Neighborhood::Swap {
        partition.fill(false);
        for x in partition.iter_mut().take(n / 2) {
            *x = true;
        }
        partition.shuffle(&mut init);
    }
    partition
}

/// Search stream `label` (`select`, `tie`, `smooth` or `accept`) of an
/// `eo_sa` condition: graph content, neighborhood, seed, alpha bits, solver
/// JSON, fitness definition version and `algorithm-v1`, then the label.
pub fn stream(
    graph: &Graph,
    condition: &Condition,
    seed: u64,
    registry: &FitnessRegistry,
    label: &[u8],
) -> Result<Mt19937GenRand64> {
    let SolverSpec::EoSa { fitness, .. } = &condition.solver else {
        return Err(Error::msg("the EO-SA reference needs an eo_sa condition"));
    };
    let version = registry
        .versions()
        .get(&fitness.kind)
        .cloned()
        .ok_or_else(|| Error::msg(format!("unknown fitness: {}", fitness.kind)))?;
    let hash = graph.content_hash();
    let solver = serde_json::to_vec(&condition.solver)?;
    Ok(rng_for(&[
        hash.as_bytes(),
        neighborhood_label(condition.neighborhood),
        &seed.to_le_bytes(),
        &condition.alpha.to_bits().to_le_bytes(),
        &solver,
        version.as_bytes(),
        b"algorithm-v1",
        label,
    ]))
}

pub struct ReferenceEoSa<'a> {
    graph: &'a Graph,
    neighborhood: Neighborhood,
    alpha: f64,
    tau: f64,
    temperature: f64,
    fitness: Box<dyn VertexFitness>,
    /// Count fitness values like the incremental built-in index (`n` once,
    /// then the re-ranked vertices of each accepted move) instead of `n` per
    /// step as for a caller-supplied definition.
    indexed: bool,
    pub state: PartitionState,
    /// Real score of `state`, held by the search between steps.
    pub current: f64,
    pub applied_moves: u64,
    pub objective_evaluations: u64,
    pub fitness_values: u64,
    pub select_rng: Mt19937GenRand64,
    pub tie_rng: Mt19937GenRand64,
    pub smooth_rng: Mt19937GenRand64,
    pub accept_rng: Mt19937GenRand64,
}

impl<'a> ReferenceEoSa<'a> {
    /// Initial state, streams and counters of an `eo_sa` condition.
    pub fn new(
        graph: &'a Graph,
        condition: &Condition,
        seed: u64,
        registry: &FitnessRegistry,
        indexed: bool,
    ) -> Result<Self> {
        let SolverSpec::EoSa {
            tau,
            temperature,
            fitness,
        } = &condition.solver
        else {
            return Err(Error::msg("the EO-SA reference needs an eo_sa condition"));
        };
        let partition = initial_partition(graph, condition.neighborhood, seed);
        let current = graph.score(&partition, condition.alpha);
        let stream = |label: &[u8]| stream(graph, condition, seed, registry, label);
        Ok(Self {
            graph,
            neighborhood: condition.neighborhood,
            alpha: condition.alpha,
            tau: *tau,
            temperature: *temperature,
            fitness: registry.create(fitness)?,
            indexed,
            state: PartitionState::new(graph, partition)?,
            current,
            applied_moves: 0,
            objective_evaluations: 1,
            fitness_values: if indexed {
                graph.node_count() as u64
            } else {
                0
            },
            select_rng: stream(b"select")?,
            tie_rng: stream(b"tie")?,
            smooth_rng: stream(b"smooth")?,
            accept_rng: stream(b"accept")?,
        })
    }

    /// One complete step: the proposal, then its judgement.
    pub fn step(&mut self) -> Result<Outcome> {
        let mv = self.propose()?;
        self.judge(mv)
    }

    /// Rule step 1: select a move from the current state with EO selection
    /// rule v2. Alone, this is a cancelled step (rule step 2): it consumes the
    /// selection draws and, for a caller-supplied fitness, counts the n values
    /// evaluated for the selection; it changes neither the state nor any other
    /// counter.
    pub fn propose(&mut self) -> Result<Move> {
        let n = self.graph.node_count();
        let lambda = self.fitness.values(self.graph, &self.state)?;
        if !self.indexed {
            // A caller-supplied definition is evaluated for every vertex on
            // every selection, including the selection of a cancelled step.
            self.fitness_values += n as u64;
        }
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
        Ok(match self.neighborhood {
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
        })
    }

    /// Rule steps 3 to 5 for a proposed move: score, Metropolis judgement and,
    /// if accepted, the move with its re-ranking.
    pub fn judge(&mut self, mv: Move) -> Result<Outcome> {
        let mut partition = self.state.partition().to_vec();
        match mv {
            Move::Flip(v) => partition[v] = !partition[v],
            Move::Swap(a, b) => {
                if partition[a] == partition[b] {
                    return Err(Error::msg("swap endpoints must be in different groups"));
                }
                partition[a] = !partition[a];
                partition[b] = !partition[b];
            }
        }
        let next = self.graph.score(&partition, self.alpha);
        self.objective_evaluations += 1;
        if !next.is_finite() {
            return Err(Error::msg("non-finite objective value"));
        }
        let delta = next - self.current;
        let t = self.temperature;
        let mut drew = false;
        let accepted = delta < 0.0
            || (t > 0.0 && {
                drew = true;
                self.accept_rng.r#gen::<f64>() < (-delta / t).exp()
            });
        if accepted {
            self.state = PartitionState::new(self.graph, partition)?;
            self.current = next;
            self.applied_moves += 1;
            if self.indexed {
                self.fitness_values += match mv {
                    Move::Flip(v) => 1 + self.graph.degree(v) as u64,
                    Move::Swap(a, b) => 2 + (self.graph.degree(a) + self.graph.degree(b)) as u64,
                };
            }
        }
        Ok(Outcome {
            mv,
            next,
            delta,
            drew,
            accepted,
        })
    }
}
