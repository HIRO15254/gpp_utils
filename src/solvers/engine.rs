use super::eo::Eo;
use super::metropolis::MetropolisFactor;
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

/// How [`Engine::advance`] stopped.
#[derive(Debug)]
pub enum Advance {
    /// The step count reached the requested value; every step continued.
    Reached,
    /// The last counted step returned this status other than `Continue`.
    Stopped(StepStatus),
    /// Cancellation was observed before a step.
    Cancelled,
    /// A step failed and was not counted.
    Failed(Error),
}

/// What [`Engine::step`] runs, derived once from the condition.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// HC on the real objective (smoothing `none` or `weighted_average` with
    /// `k = 0`).
    HcReal,
    HcSmoothed,
    /// SA on the real objective at the temperature.
    SaReal(f64),
    SaSmoothed(f64),
    Eo,
    EoSa(f64),
}

/// Whether `spec` leaves the objective unchanged.
fn is_real(spec: &SmoothingSpec) -> bool {
    matches!(
        spec,
        SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 }
    )
}

pub struct Engine<'a> {
    graph: &'a Graph,
    condition: &'a Condition,
    kind: Kind,
    pub state: PartitionState,
    pub search_evaluation: f64,
    select_rng: Mt19937GenRand64,
    tie_rng: Mt19937GenRand64,
    smooth_rng: Mt19937GenRand64,
    /// Metropolis acceptance draws of `eo_sa`; `None` for the other solvers.
    accept_rng: Option<Mt19937GenRand64>,
    eo: Option<Eo>,
    /// `(-delta / t).exp()` of the real-objective SA and of EO-SA at the
    /// job's temperature; `None` for the other solvers.
    metropolis: Option<MetropolisFactor>,
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
        let kind = match &condition.solver {
            SolverSpec::Hc { smoothing } if is_real(smoothing) => Kind::HcReal,
            SolverSpec::Hc { .. } => Kind::HcSmoothed,
            SolverSpec::Sa {
                temperature,
                smoothing,
            } if is_real(smoothing) => Kind::SaReal(*temperature),
            SolverSpec::Sa { temperature, .. } => Kind::SaSmoothed(*temperature),
            SolverSpec::Eo { .. } => Kind::Eo,
            SolverSpec::EoSa { temperature, .. } => Kind::EoSa(*temperature),
        };
        let mut e = Self {
            graph,
            condition,
            kind,
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
            metropolis: match kind {
                Kind::SaReal(t) | Kind::EoSa(t) => Some(MetropolisFactor::new(t)),
                _ => None,
            },
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
            let spec = e.smoothing_spec();
            e.search_evaluation = smoothing::evaluate(
                &e.state,
                graph,
                condition.alpha,
                condition.neighborhood,
                spec,
                Some(&mut e.smooth_rng),
                cancel,
                &mut e.objective_evaluations,
            )?
        }
        Ok(e)
    }
    /// The smoothing of HC and SA, borrowed from the condition rather than
    /// the engine.
    fn smoothing_spec(&self) -> &'a SmoothingSpec {
        match &self.condition.solver {
            SolverSpec::Hc { smoothing } | SolverSpec::Sa { smoothing, .. } => smoothing,
            SolverSpec::Eo { .. } | SolverSpec::EoSa { .. } => unreachable!(),
        }
    }
    pub fn step(&mut self, cancel: &CancellationToken) -> Result<StepStatus> {
        match self.kind {
            Kind::HcReal => self.hc_real(cancel),
            Kind::HcSmoothed => self.hc_smoothed(cancel),
            Kind::SaReal(t) => self.sa_real(t, cancel),
            Kind::SaSmoothed(t) => self.sa_smoothed(t, cancel),
            Kind::Eo => self.eo(cancel),
            Kind::EoSa(t) => self.eo_sa(t, cancel),
        }
    }
    /// Repeats `if cancel.is_cancelled() { stop } ; step ; *completed += 1 ;
    /// after(self, *completed)` while `*completed < until`, stopping after a
    /// step that does not continue or fails (a failed step is not counted).
    ///
    /// The same sequence as calling [`Self::step`] in that loop; the
    /// real-objective SA runs a loop specialized to its neighborhood.
    pub fn advance<A>(
        &mut self,
        until: u64,
        completed: &mut u64,
        cancel: &CancellationToken,
        mut after: A,
    ) -> Advance
    where
        A: FnMut(&Self, u64),
    {
        match (self.kind, self.condition.neighborhood) {
            (Kind::SaReal(t), Neighborhood::Flip) => {
                self.advance_with(until, completed, cancel, &mut after, |e, _| {
                    e.sa_real_flip(t)
                })
            }
            (Kind::SaReal(t), Neighborhood::Swap) => {
                self.advance_with(until, completed, cancel, &mut after, |e, c| {
                    e.sa_real_swap(t, c)
                })
            }
            _ => self.advance_with(until, completed, cancel, &mut after, Self::step),
        }
    }
    #[inline(always)]
    fn advance_with<A, S>(
        &mut self,
        until: u64,
        completed: &mut u64,
        cancel: &CancellationToken,
        after: &mut A,
        mut step: S,
    ) -> Advance
    where
        A: FnMut(&Self, u64),
        S: FnMut(&mut Self, &CancellationToken) -> Result<StepStatus>,
    {
        while *completed < until {
            if cancel.is_cancelled() {
                return Advance::Cancelled;
            }
            let status = match step(self, cancel) {
                Ok(status) => status,
                Err(error) => return Advance::Failed(error),
            };
            *completed += 1;
            after(self, *completed);
            if status != StepStatus::Continue {
                return Advance::Stopped(status);
            }
        }
        Advance::Reached
    }
    /// HC on real move scores: the same candidates, rule, tie draws,
    /// evaluation count and non-finite check as scoring every move with
    /// `smoothing::move_score`.
    fn hc_real(&mut self, cancel: &CancellationToken) -> Result<StepStatus> {
        let found = self.best_improvement.scan(
            self.graph,
            &self.state,
            self.search_evaluation,
            &mut self.tie_rng,
            cancel,
            &mut self.objective_evaluations,
        )?;
        Ok(match found {
            Some((mv, best)) => {
                smoothing::apply(&mut self.state, self.graph, mv);
                self.search_evaluation = best;
                self.applied_moves += 1;
                StepStatus::Continue
            }
            None => StepStatus::LocalOptimum,
        })
    }
    fn hc_smoothed(&mut self, cancel: &CancellationToken) -> Result<StepStatus> {
        let spec = self.smoothing_spec();
        let list = smoothing::moves_cancellable(&self.state, self.condition.neighborhood, cancel)?;
        if matches!(spec, SmoothingSpec::RandomKAverage { .. }) {
            self.search_evaluation = smoothing::evaluate(
                &self.state,
                self.graph,
                self.condition.alpha,
                self.condition.neighborhood,
                spec,
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
                spec,
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
                let (a, b) = self.random_swap(cancel)?;
                Move::Swap(a, b)
            }
        })
    }
    /// The endpoints of a uniform Swap: uniform vertex draws until one lands
    /// in group A, then until one lands in group B. Cancellation is checked
    /// before the first draw of the step and then before every 1024th draw.
    #[inline(always)]
    fn random_swap(&mut self, cancel: &CancellationToken) -> Result<(usize, usize)> {
        let n = self.graph.node_count();
        let partition = self.state.partition();
        let mut draws = 0usize;
        let a = loop {
            if draws & 1023 == 0 {
                cancel.check()?;
            }
            draws += 1;
            let v = self.select_rng.gen_range(0..n);
            if partition[v] {
                break v;
            }
        };
        let b = loop {
            if draws & 1023 == 0 {
                cancel.check()?;
            }
            draws += 1;
            let v = self.select_rng.gen_range(0..n);
            if !partition[v] {
                break v;
            }
        };
        Ok((a, b))
    }
    /// SA on the real objective: [`Self::sa_smoothed`] with the proposal
    /// scored by `smoothing::move_score` and applied only when accepted.
    fn sa_real(&mut self, t: f64, cancel: &CancellationToken) -> Result<StepStatus> {
        match self.condition.neighborhood {
            Neighborhood::Flip => self.sa_real_flip(t),
            Neighborhood::Swap => self.sa_real_swap(t, cancel),
        }
    }
    #[inline(always)]
    fn sa_real_flip(&mut self, t: f64) -> Result<StepStatus> {
        let v = self.select_rng.gen_range(0..self.graph.node_count());
        let next = self.state.flip_score(self.graph, v, self.condition.alpha);
        if self.sa_accepts(next, t)? {
            self.state.apply_flip(self.graph, v);
            self.search_evaluation = next;
            self.applied_moves += 1;
        }
        Ok(StepStatus::Continue)
    }
    #[inline(always)]
    fn sa_real_swap(&mut self, t: f64, cancel: &CancellationToken) -> Result<StepStatus> {
        let (a, b) = self.random_swap(cancel)?;
        // `a` is in group A and `b` in group B.
        let next = self
            .state
            .swap_score_across(self.graph, a, b, self.condition.alpha);
        if self.sa_accepts(next, t)? {
            self.state.apply_swap(self.graph, a, b);
            self.search_evaluation = next;
            self.applied_moves += 1;
        }
        Ok(StepStatus::Continue)
    }
    /// Counts the evaluation of the real score `next`, rejects a non-finite
    /// one and applies the Metropolis rule
    /// `delta < 0.0 || (t > 0.0 && u < (-delta / t).exp())`, where `u` is drawn
    /// from the select stream only when the right operand is evaluated.
    #[inline(always)]
    fn sa_accepts(&mut self, next: f64, t: f64) -> Result<bool> {
        self.objective_evaluations += 1;
        if !next.is_finite() {
            return Err(Error::msg("non-finite search evaluation"));
        }
        let delta = next - self.search_evaluation;
        Ok(self.sa_accept(delta, t))
    }
    /// `delta < 0.0 || (t > 0.0 && u < (-delta / t).exp())` for the job's
    /// temperature `t`, with `u` drawn from the select stream only when the
    /// right operand is evaluated (improvements and `t == 0` draw nothing).
    #[inline(always)]
    fn sa_accept(&mut self, delta: f64, t: f64) -> bool {
        delta < 0.0
            || (t > 0.0 && {
                let u = self.select_rng.r#gen::<f64>();
                // The bits of `(-delta / t).exp()`.
                let factor = self
                    .metropolis
                    .as_mut()
                    .expect("real-objective SA initializes the Metropolis memo")
                    .get(delta);
                u < factor
            })
    }
    fn sa_smoothed(&mut self, t: f64, cancel: &CancellationToken) -> Result<StepStatus> {
        let mv = self.random_move(cancel)?;
        let spec = self.smoothing_spec();
        smoothing::apply(&mut self.state, self.graph, mv);
        let next = match smoothing::evaluate(
            &self.state,
            self.graph,
            self.condition.alpha,
            self.condition.neighborhood,
            spec,
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
        let metropolis = self
            .metropolis
            .as_mut()
            .expect("EO-SA conditions initialize the Metropolis memo");
        let delta = next - self.search_evaluation;
        // `metropolis.get(delta)` has the bits of `(-delta / t).exp()`.
        let accept = delta < 0.0 || (t > 0.0 && accept_rng.r#gen::<f64>() < metropolis.get(delta));
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
