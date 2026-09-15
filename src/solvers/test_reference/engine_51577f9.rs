use crate::error::{Error, Result};
use crate::experiment::config::{Condition, Neighborhood, SmoothingSpec, SolverSpec};
use crate::fitness::{FitnessRegistry, VertexFitness};
use crate::graph_partition::{Graph, Move, PartitionState};
use crate::optimization::{CancellationToken, rng_for};
use crate::smoothing;
use rand::{Rng, seq::SliceRandom};
use rand_mt::Mt19937GenRand64;

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
    fitness: Option<Box<dyn VertexFitness>>,
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
            fitness: None,
            objective_evaluations: 0,
            fitness_values: 0,
            applied_moves: 0,
        };
        if let SolverSpec::Eo { fitness, .. } = &condition.solver {
            e.fitness = Some(registry.create(fitness)?);
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
            SolverSpec::Eo { tau, .. } => self.eo(*tau, cancel),
        }
    }
    fn hc(&mut self, cancel: &CancellationToken) -> Result<StepStatus> {
        let list = smoothing::moves_cancellable(&self.state, self.condition.neighborhood, cancel)?;
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
        let mut best = self.search_evaluation;
        let mut choice = None;
        let mut ties = 0u64;
        for (i, mv) in list.into_iter().enumerate() {
            if i & 1023 == 0 {
                cancel.check()?
            }
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
            let x = evaluated?;
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
            let accept =
                delta < 0.0 || (t > 0.0 && self.select_rng.r#gen::<f64>() < (-delta / t).exp());
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
        let accept =
            delta < 0.0 || (t > 0.0 && self.select_rng.r#gen::<f64>() < (-delta / t).exp());
        if accept {
            self.search_evaluation = next;
            self.applied_moves += 1
        } else {
            smoothing::apply(&mut self.state, self.graph, mv);
        }
        Ok(StepStatus::Continue)
    }
    fn eo(&mut self, tau: f64, cancel: &CancellationToken) -> Result<StepStatus> {
        let fit = self
            .fitness
            .as_ref()
            .unwrap()
            .values(self.graph, &self.state)?;
        self.fitness_values += fit.len() as u64;
        if fit.len() != self.graph.node_count() || fit.iter().any(|x| !x.is_finite()) {
            return Err(Error::msg("fitness returned invalid values"));
        }
        let mut ranked: Vec<usize> = (0..fit.len()).collect();
        ranked.shuffle(&mut self.tie_rng);
        ranked.sort_by(|&a, &b| fit[a].total_cmp(&fit[b]));
        let first = weighted_rank(&ranked, tau, &mut self.select_rng, None, &self.state)?;
        let mv = match self.condition.neighborhood {
            Neighborhood::Flip => Move::Flip(first),
            Neighborhood::Swap => Move::Swap(
                first,
                weighted_rank(
                    &ranked,
                    tau,
                    &mut self.select_rng,
                    Some(!self.state.partition()[first]),
                    &self.state,
                )?,
            ),
        };
        cancel.check()?;
        smoothing::apply(&mut self.state, self.graph, mv);
        self.applied_moves += 1;
        self.objective_evaluations += 1;
        self.search_evaluation = self.state.score(self.condition.alpha);
        Ok(StepStatus::Continue)
    }
}
fn weighted_rank(
    ranked: &[usize],
    tau: f64,
    rng: &mut Mt19937GenRand64,
    side: Option<bool>,
    state: &PartitionState,
) -> Result<usize> {
    let eligible: Vec<(usize, usize)> = ranked
        .iter()
        .enumerate()
        .filter(|(_, v)| side.is_none_or(|s| state.partition()[**v] == s))
        .map(|(i, &v)| (v, i + 1))
        .collect();
    if eligible.is_empty() {
        return Err(Error::msg("empty EO conditional rank distribution"));
    }
    let min_rank = eligible.iter().map(|x| x.1).min().unwrap();
    let weight = |rank: usize| {
        if rank == min_rank {
            1.0
        } else {
            (-tau * (rank as f64 / min_rank as f64).ln()).exp()
        }
    };
    let total: f64 = eligible.iter().map(|x| weight(x.1)).sum();
    let mut u = rng.r#gen::<f64>() * total;
    for &(v, rank) in &eligible {
        u -= weight(rank);
        if u <= 0.0 {
            return Ok(v);
        }
    }
    Ok(eligible.last().unwrap().0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huge_tau_selects_best_eligible_global_rank_without_nan() {
        let graph = Graph::from_edges(4, vec![]).unwrap();
        let state = PartitionState::new(&graph, vec![true, true, false, false]).unwrap();
        let ranked = vec![0, 1, 2, 3];
        for seed in 0..20 {
            let mut rng = Mt19937GenRand64::new(seed);
            assert_eq!(
                weighted_rank(&ranked, 1.0e308, &mut rng, Some(false), &state).unwrap(),
                2
            );
        }
    }
}
