use super::eo::Eo;
use crate::error::{Error, Result};
use crate::experiment::config::{Condition, Neighborhood, SmoothingSpec, SolverSpec};
use crate::fitness::FitnessRegistry;
use crate::graph_partition::{Graph, Move, PartitionState};
use crate::optimization::{CancellationToken, rng_for};
use crate::smoothing;
use rand::{Rng, seq::SliceRandom};
use rand_mt::Mt19937GenRand64;
const SA_EXP_CACHE_CAPACITY: usize = 256;
const _: () = assert!(SA_EXP_CACHE_CAPACITY == 256);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepStatus {
    Continue,
    LocalOptimum,
    NoSampledImprovement,
}
pub struct Engine<'a> {
    graph: &'a Graph,
    condition: &'a Condition,
    pub state: PartitionState,
    pub search_evaluation: f64,
    select_rng: Mt19937GenRand64,
    tie_rng: Mt19937GenRand64,
    smooth_rng: Mt19937GenRand64,
    eo: Option<Eo>,
    /// Exact Metropolis thresholds keyed by the observed delta bits. The
    /// temperature is fixed for an Engine, so it is intentionally not in the key.
    sa_exp_cache: [Option<(u64, f64)>; SA_EXP_CACHE_CAPACITY],
    pub objective_evaluations: u64,
    pub fitness_values: u64,
    pub applied_moves: u64,
}
impl<'a> Engine<'a> {
    pub fn new(
        graph: &'a Graph,
        condition: &'a Condition,
        seed: u64,
        registry: &FitnessRegistry,
        cancel: &CancellationToken,
    ) -> Result<Self> {
        let hash = graph.content_hash();
        let nlabel = match condition.neighborhood {
            Neighborhood::Flip => b"flip".as_slice(),
            Neighborhood::Swap => b"swap".as_slice(),
        };
        let seedb = seed.to_le_bytes();
        let mut init = rng_for(&[hash.as_bytes(), nlabel, &seedb, b"initial"]);
        let mut p: Vec<bool> = (0..graph.node_count()).map(|_| init.r#gen()).collect();
        if matches!(condition.neighborhood, Neighborhood::Swap) {
            p.fill(false);
            for x in p.iter_mut().take(graph.node_count() / 2) {
                *x = true
            }
            p.shuffle(&mut init)
        }
        let state = PartitionState::new(graph, p)?;
        let solver = serde_json::to_vec(&condition.solver)?;
        let alpha = condition.alpha.to_bits().to_le_bytes();
        let fitness_version = match &condition.solver {
            SolverSpec::Eo { fitness, .. } => registry
                .versions()
                .get(&fitness.kind)
                .cloned()
                .unwrap_or_default(),
            _ => String::new(),
        };
        let mut e = Self {
            graph,
            condition,
            state,
            search_evaluation: 0.0,
            select_rng: rng_for(&[
                hash.as_bytes(),
                nlabel,
                &seedb,
                &alpha,
                &solver,
                fitness_version.as_bytes(),
                b"algorithm-v1",
                b"select",
            ]),
            tie_rng: rng_for(&[
                hash.as_bytes(),
                nlabel,
                &seedb,
                &alpha,
                &solver,
                fitness_version.as_bytes(),
                b"algorithm-v1",
                b"tie",
            ]),
            smooth_rng: rng_for(&[
                hash.as_bytes(),
                nlabel,
                &seedb,
                &alpha,
                &solver,
                fitness_version.as_bytes(),
                b"algorithm-v1",
                b"smooth",
            ]),
            eo: None,
            sa_exp_cache: [None; SA_EXP_CACHE_CAPACITY],
            objective_evaluations: 0,
            fitness_values: 0,
            applied_moves: 0,
        };
        if let SolverSpec::Eo { fitness, tau } = &condition.solver {
            e.eo = Some(Eo::new(
                registry.create_engine_fitness(fitness)?,
                graph,
                &e.state,
                condition.neighborhood,
                *tau,
                &mut e.fitness_values,
            )?);
            e.search_evaluation = e.state.score(condition.alpha);
            e.objective_evaluations += 1
        } else {
            let spec = e.smoothing_spec().clone();
            e.search_evaluation = smoothing::evaluate(
                &e.state,
                graph,
                condition.alpha,
                condition.neighborhood,
                &spec,
                Some(&mut e.smooth_rng),
                cancel,
                &mut e.objective_evaluations,
            )?
        }
        Ok(e)
    }
    fn smoothing_spec(&self) -> &SmoothingSpec {
        match &self.condition.solver {
            SolverSpec::Hc { smoothing } | SolverSpec::Sa { smoothing, .. } => smoothing,
            SolverSpec::Eo { .. } => unreachable!(),
        }
    }
    pub fn step(&mut self, cancel: &CancellationToken) -> Result<StepStatus> {
        match &self.condition.solver {
            SolverSpec::Hc { .. } => self.hc(cancel),
            SolverSpec::Sa { temperature, .. } => self.sa(*temperature, cancel),
            SolverSpec::Eo { .. } => self.eo(cancel),
        }
    }
    fn hc(&mut self, cancel: &CancellationToken) -> Result<StepStatus> {
        let list = smoothing::MoveSequence::new(&self.state, self.condition.neighborhood, cancel)?;
        let spec = self.smoothing_spec().clone();
        if matches!(spec, SmoothingSpec::RandomKAverage { .. }) {
            self.search_evaluation = smoothing::evaluate(
                &self.state,
                self.graph,
                self.condition.alpha,
                self.condition.neighborhood,
                &spec,
                Some(&mut self.smooth_rng),
                cancel,
                &mut self.objective_evaluations,
            )?;
        }
        let real_move_scores = matches!(
            spec,
            SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 }
        );
        let mut best = self.search_evaluation;
        let mut choice = None;
        let mut ties = 0u64;
        for (i, mv) in list.iter().enumerate() {
            if i & 1023 == 0 {
                cancel.check()?
            }
            let x = if real_move_scores {
                let x = smoothing::move_score(&self.state, self.graph, mv, self.condition.alpha);
                self.objective_evaluations += 1;
                x
            } else {
                smoothing::apply(&mut self.state, self.graph, mv);
                let evaluated = smoothing::evaluate(
                    &self.state,
                    self.graph,
                    self.condition.alpha,
                    self.condition.neighborhood,
                    &spec,
                    Some(&mut self.smooth_rng),
                    cancel,
                    &mut self.objective_evaluations,
                );
                smoothing::apply(&mut self.state, self.graph, mv);
                evaluated?
            };
            if !x.is_finite() {
                return Err(Error::msg("non-finite search evaluation"));
            }
            if x < best {
                best = x;
                choice = Some(mv);
                ties = 1
            } else if choice.is_some() && x == best {
                ties += 1;
                if self.tie_rng.gen_range(0..ties) == 0 {
                    choice = Some(mv)
                }
            }
        }
        if let Some(mv) = choice {
            smoothing::apply(&mut self.state, self.graph, mv);
            self.search_evaluation = best;
            self.applied_moves += 1;
            Ok(StepStatus::Continue)
        } else if matches!(spec, SmoothingSpec::RandomKAverage { .. }) {
            Ok(StepStatus::NoSampledImprovement)
        } else {
            Ok(StepStatus::LocalOptimum)
        }
    }
    fn random_move(&mut self, cancel: &CancellationToken) -> Result<Move> {
        Ok(match self.condition.neighborhood {
            Neighborhood::Flip => Move::Flip(self.select_rng.gen_range(0..self.graph.node_count())),
            Neighborhood::Swap => {
                let mut draws = 0usize;
                let a = loop {
                    if draws & 1023 == 0 {
                        cancel.check()?;
                    }
                    draws += 1;
                    let v = self.select_rng.gen_range(0..self.graph.node_count());
                    if self.state.partition()[v] {
                        break v;
                    }
                };
                let b = loop {
                    if draws & 1023 == 0 {
                        cancel.check()?;
                    }
                    draws += 1;
                    let v = self.select_rng.gen_range(0..self.graph.node_count());
                    if !self.state.partition()[v] {
                        break v;
                    }
                };
                Move::Swap(a, b)
            }
        })
    }
    fn sa(&mut self, t: f64, cancel: &CancellationToken) -> Result<StepStatus> {
        let mv = self.random_move(cancel)?;
        let spec = self.smoothing_spec().clone();
        if matches!(
            spec,
            SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 }
        ) {
            let next = smoothing::move_score(&self.state, self.graph, mv, self.condition.alpha);
            self.objective_evaluations += 1;
            if !next.is_finite() {
                return Err(Error::msg("non-finite search evaluation"));
            }
            let delta = next - self.search_evaluation;
            let accept = self.sa_accept(delta, t);
            if accept {
                smoothing::apply(&mut self.state, self.graph, mv);
                self.search_evaluation = next;
                self.applied_moves += 1;
            }
            return Ok(StepStatus::Continue);
        }
        smoothing::apply(&mut self.state, self.graph, mv);
        let next = match smoothing::evaluate(
            &self.state,
            self.graph,
            self.condition.alpha,
            self.condition.neighborhood,
            &spec,
            Some(&mut self.smooth_rng),
            cancel,
            &mut self.objective_evaluations,
        ) {
            Ok(value) => value,
            Err(error) => {
                smoothing::apply(&mut self.state, self.graph, mv);
                return Err(error);
            }
        };
        if !next.is_finite() {
            smoothing::apply(&mut self.state, self.graph, mv);
            return Err(Error::msg("non-finite search evaluation"));
        }
        let delta = next - self.search_evaluation;
        let accept = self.sa_accept(delta, t);
        if accept {
            self.search_evaluation = next;
            self.applied_moves += 1
        } else {
            smoothing::apply(&mut self.state, self.graph, mv);
        }
        Ok(StepStatus::Continue)
    }

    /// Preserve the original short-circuit and RNG order exactly: improving
    /// moves draw nothing; non-improving positive-temperature moves draw before
    /// looking up or computing the threshold.
    #[inline]
    fn sa_accept(&mut self, delta: f64, temperature: f64) -> bool {
        if delta < 0.0 {
            return true;
        }
        if temperature <= 0.0 {
            return false;
        }
        let draw = self.select_rng.r#gen::<f64>();
        let key = delta.to_bits();
        // Fold high IEEE-754 bits into the low half, multiply to diffuse the
        // common zero mantissa suffix of integer deltas, then use the high byte.
        // The table is fixed at 256 slots by the compile-time assertion above.
        let mixed = (key ^ (key >> 32)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let slot = (mixed >> 56) as usize;
        let threshold = if let Some((cached_key, cached)) = self.sa_exp_cache[slot]
            && cached_key == key
        {
            cached
        } else {
            let computed = (-delta / temperature).exp();
            self.sa_exp_cache[slot] = Some((key, computed));
            computed
        };
        draw < threshold
    }
    /// One EO move (algorithm v2, see [`super::eo`]); always accepted.
    fn eo(&mut self, cancel: &CancellationToken) -> Result<StepStatus> {
        let eo = self
            .eo
            .as_mut()
            .expect("EO conditions initialize the EO selector");
        let mv = eo.select(
            self.graph,
            &self.state,
            self.condition.neighborhood,
            &mut self.select_rng,
            &mut self.fitness_values,
        )?;
        cancel.check()?;
        smoothing::apply(&mut self.state, self.graph, mv);
        eo.applied(self.graph, &self.state, mv, &mut self.fitness_values);
        self.applied_moves += 1;
        self.objective_evaluations += 1;
        self.search_evaluation = self.state.score(self.condition.alpha);
        Ok(StepStatus::Continue)
    }
}

#[cfg(test)]
#[path = "exact_tests.rs"]
mod exact_tests;
