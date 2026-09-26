use super::eo::Eo;
use crate::error::{Error, Result};
use crate::experiment::config::{Condition, Neighborhood, SmoothingSpec, SolverSpec};
use crate::fitness::FitnessRegistry;
use crate::graph_partition::{BestImprovement, Graph, Move, NonFinite, PartitionState};
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
    /// Metropolis acceptance draws of `eo_sa`; `None` for the other solvers.
    accept_rng: Option<Mt19937GenRand64>,
    eo: Option<Eo>,
    /// Scratch of the real-objective HC scan, reused across steps.
    best_improvement: BestImprovement,
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
        let fitness_version = match condition.solver.fitness() {
            Some(fitness) => registry
                .versions()
                .get(&fitness.kind)
                .cloned()
                .unwrap_or_default(),
            None => String::new(),
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
            accept_rng: matches!(condition.solver, SolverSpec::EoSa { .. }).then(|| {
                rng_for(&[
                    hash.as_bytes(),
                    nlabel,
                    &seedb,
                    &alpha,
                    &solver,
                    fitness_version.as_bytes(),
                    b"algorithm-v1",
                    b"accept",
                ])
            }),
            eo: None,
            best_improvement: BestImprovement::new(
                condition.neighborhood,
                condition.alpha,
                NonFinite::Reject,
            ),
            objective_evaluations: 0,
            fitness_values: 0,
            applied_moves: 0,
        };
        if let SolverSpec::Eo { fitness, tau } | SolverSpec::EoSa { fitness, tau, .. } =
            &condition.solver
        {
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
            SolverSpec::Eo { .. } | SolverSpec::EoSa { .. } => unreachable!(),
        }
    }
    pub fn step(&mut self, cancel: &CancellationToken) -> Result<StepStatus> {
        match &self.condition.solver {
            SolverSpec::Hc { .. } => self.hc(cancel),
            SolverSpec::Sa { temperature, .. } => self.sa(*temperature, cancel),
            SolverSpec::Eo { .. } => self.eo(cancel),
            SolverSpec::EoSa { temperature, .. } => self.eo_sa(*temperature, cancel),
        }
    }
    fn hc(&mut self, cancel: &CancellationToken) -> Result<StepStatus> {
        let spec = self.smoothing_spec().clone();
        if matches!(
            spec,
            SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 }
        ) {
            // Real move scores: the same candidates, rule, tie draws,
            // evaluation count and non-finite check as scoring every move with
            // `smoothing::move_score`.
            let found = self.best_improvement.scan(
                self.graph,
                &self.state,
                self.search_evaluation,
                &mut self.tie_rng,
                cancel,
                &mut self.objective_evaluations,
            )?;
            return Ok(match found {
                Some((mv, best)) => {
                    smoothing::apply(&mut self.state, self.graph, mv);
                    self.search_evaluation = best;
                    self.applied_moves += 1;
                    StepStatus::Continue
                }
                None => StepStatus::LocalOptimum,
            });
        }
        let list = smoothing::moves_cancellable(&self.state, self.condition.neighborhood, cancel)?;
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
    /// One EO-SA step: an EO proposal (selection rule v2 on the current
    /// ranking) judged by the SA Metropolis rule on the real objective. The
    /// acceptance draw comes from its own stream, so the selection stream
    /// advances exactly as in EO; a rejected move leaves the state and the
    /// ranking unchanged.
    fn eo_sa(&mut self, t: f64, cancel: &CancellationToken) -> Result<StepStatus> {
        let eo = self
            .eo
            .as_mut()
            .expect("EO-SA conditions initialize the EO selector");
        let mv = eo.select(
            self.graph,
            &self.state,
            self.condition.neighborhood,
            &mut self.select_rng,
            &mut self.fitness_values,
        )?;
        cancel.check()?;
        let next = smoothing::move_score(&self.state, self.graph, mv, self.condition.alpha);
        self.objective_evaluations += 1;
        if !next.is_finite() {
            return Err(Error::msg("non-finite search evaluation"));
        }
        let accept_rng = self
            .accept_rng
            .as_mut()
            .expect("EO-SA conditions initialize the acceptance stream");
        let delta = next - self.search_evaluation;
        let accept = delta < 0.0 || (t > 0.0 && accept_rng.r#gen::<f64>() < (-delta / t).exp());
        if accept {
            smoothing::apply(&mut self.state, self.graph, mv);
            eo.applied(self.graph, &self.state, mv, &mut self.fitness_values);
            self.search_evaluation = next;
            self.applied_moves += 1;
        }
        Ok(StepStatus::Continue)
    }
}

#[cfg(test)]
#[path = "exact_tests.rs"]
mod exact_tests;
