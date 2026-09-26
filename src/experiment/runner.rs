#![allow(clippy::too_many_arguments, clippy::collapsible_if)]

use crate::error::{Error, Result};
use crate::experiment::config::{BasinMode, Condition, Neighborhood, SmoothingSpec, SolverSpec};
use crate::experiment::measurement::checkpoints;
use crate::experiment::result::{
    BasinResult, BasinTermination, Diagnostics, MeasurementRecord, RunResult, RunTermination,
    SolutionId,
};
use crate::fitness::FitnessRegistry;
use crate::graph_partition::{BestImprovement, Graph, Move, NonFinite, PartitionState};
use crate::optimization::{CancellationToken, rng_for};
use crate::smoothing;
use crate::solvers::{Advance, Engine, StepStatus};
use rand_mt::Mt19937GenRand64;
use std::{
    collections::BTreeMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Run one validated scientific condition without filesystem I/O.
///
/// Checks scalar parameters again for direct library callers. `graph` must be
/// the immutable topology intended by the condition; a supplied graph is not
/// regenerated from its generation specification. Cancellation returns the last
/// committed state with [`RunTermination::Cancelled`] when initialization has
/// reached a saveable state; earlier failures return an error. Measurement uses
/// independent RNG streams and never feeds back into search decisions.
pub fn run_one(
    graph: &Graph,
    condition: &Condition,
    seed: u64,
    cancel: &CancellationToken,
    registry: &FitnessRegistry,
) -> Result<RunResult> {
    if graph.node_count() < 2 {
        return Err(Error::msg("node_count must be at least 2"));
    }
    let n = graph.node_count() as f64;
    if !(n * (n - 1.0) / 2.0 + condition.alpha * n * n).is_finite() {
        return Err(Error::msg("maximum possible score is not finite"));
    }
    if condition.measurement.max_basin_steps == 0 {
        return Err(Error::msg("max_basin_steps must be positive"));
    }
    let mut measurement_steps = std::collections::BTreeSet::new();
    for &step in &condition.measurement.steps {
        if step > condition.budget.max_steps || !measurement_steps.insert(step) {
            return Err(Error::msg(
                "measurement steps must be unique and within budget",
            ));
        }
    }
    if condition.measurement.schedule == crate::experiment::config::Schedule::Logarithmic
        && !condition.measurement.steps.is_empty()
    {
        return Err(Error::msg(
            "steps are only allowed for explicit measurement",
        ));
    }
    if condition.graph.node_count != graph.node_count() {
        return Err(Error::msg("condition graph size does not match graph"));
    }
    if !condition.alpha.is_finite() || condition.alpha < 0.0 {
        return Err(Error::msg("alpha must be finite and non-negative"));
    }
    if condition.budget.max_steps == 0 {
        return Err(Error::msg("max_steps must be positive"));
    }
    match &condition.solver {
        SolverSpec::Sa {
            temperature,
            smoothing,
        } => {
            if !temperature.is_finite() || *temperature < 0.0 {
                return Err(Error::msg("temperature must be finite and non-negative"));
            }
            smoothing::validate(smoothing, graph.node_count(), condition.neighborhood)?;
        }
        SolverSpec::Hc { smoothing } => {
            smoothing::validate(smoothing, graph.node_count(), condition.neighborhood)?;
        }
        SolverSpec::Eo { tau, fitness } => {
            if !tau.is_finite() || *tau < 0.0 {
                return Err(Error::msg("tau must be finite and non-negative"));
            }
            registry.validate(fitness)?;
        }
        SolverSpec::EoSa {
            tau,
            temperature,
            fitness,
        } => {
            if !tau.is_finite() || *tau < 0.0 {
                return Err(Error::msg("tau must be finite and non-negative"));
            }
            if !temperature.is_finite() || *temperature < 0.0 {
                return Err(Error::msg("temperature must be finite and non-negative"));
            }
            registry.validate(fitness)?;
        }
    }
    if matches!(condition.neighborhood, Neighborhood::Swap) && !graph.node_count().is_multiple_of(2)
    {
        return Err(Error::msg("swap requires an even node count"));
    }
    let started = Instant::now();
    let mut engine = Engine::new(graph, condition, seed, registry, cancel)?;
    let search_is_real = matches!(
        condition.solver,
        SolverSpec::Sa {
            smoothing: SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 },
            ..
        } | SolverSpec::EoSa { .. }
    );
    let initial = engine.state.partition().to_vec();
    let mut best = initial.clone();
    let mut best_score = engine.state.score(condition.alpha);
    let mut best_step = 0;
    let wanted = checkpoints(&condition.measurement, condition.budget.max_steps);
    let mut next = 0usize;
    let mut completed = 0u64;
    let mut termination = RunTermination::StepLimit;
    let mut raw = Vec::new();
    let mut measurement_evals = 0u64;
    let mut measurement_ms = 0.0;
    let mut best_basin_cache: Option<(Vec<bool>, BasinResult)> = None;
    if let Err(error) = record_if_due(
        graph,
        condition,
        seed,
        &engine,
        &best,
        completed,
        &wanted,
        &mut next,
        &mut raw,
        cancel,
        &mut measurement_evals,
        &mut measurement_ms,
        &mut best_basin_cache,
    ) {
        if cancel.is_cancelled() {
            termination = RunTermination::Cancelled;
            raw.push(RawRecord::plain(0, initial.clone(), initial.clone()));
        } else {
            return Err(error);
        }
    }
    while completed < condition.budget.max_steps && termination != RunTermination::Cancelled {
        // Per step: stop if cancelled, step, count it and track the best
        // solution. `record_if_due` records nothing before the next
        // checkpoint `wanted[next]` (which exists: `max_steps` is one and
        // `next` is the first checkpoint after `completed`), so it is called
        // only after that checkpoint's step or a stopping step.
        let until = wanted[next];
        debug_assert!(until > completed, "checkpoints increase past {completed}");
        let advance = engine.advance(until, &mut completed, cancel, |engine, completed| {
            let real = if search_is_real {
                engine.search_evaluation
            } else {
                engine.state.score(condition.alpha)
            };
            if real < best_score {
                best_score = real;
                best.copy_from_slice(engine.state.partition());
                best_step = completed
            }
        });
        let status = match advance {
            Advance::Reached => StepStatus::Continue,
            Advance::Stopped(status) => status,
            Advance::Cancelled => {
                termination = RunTermination::Cancelled;
                break;
            }
            Advance::Failed(_error) if cancel.is_cancelled() => {
                termination = RunTermination::Cancelled;
                break;
            }
            Advance::Failed(e) => return Err(e),
        };
        if let Err(error) = record_if_due(
            graph,
            condition,
            seed,
            &engine,
            &best,
            completed,
            &wanted,
            &mut next,
            &mut raw,
            cancel,
            &mut measurement_evals,
            &mut measurement_ms,
            &mut best_basin_cache,
        ) {
            if cancel.is_cancelled() {
                termination = RunTermination::Cancelled;
                raw.push(RawRecord::plain(
                    completed,
                    engine.state.partition().to_vec(),
                    best.clone(),
                ));
                break;
            }
            return Err(error);
        }
        match status {
            StepStatus::Continue => {}
            StepStatus::LocalOptimum => {
                termination = RunTermination::LocalOptimum;
                break;
            }
            StepStatus::NoSampledImprovement => {
                termination = RunTermination::NoSampledImprovement;
                break;
            }
        }
    }
    if raw.last().map(|x: &RawRecord| x.step) != Some(completed) && !cancel.is_cancelled() {
        if let Err(error) = record(
            graph,
            condition,
            seed,
            &engine,
            &best,
            completed,
            &mut raw,
            cancel,
            &mut measurement_evals,
            &mut measurement_ms,
            &mut best_basin_cache,
        ) {
            if !cancel.is_cancelled() {
                return Err(error);
            }
            termination = RunTermination::Cancelled;
        }
    }
    if cancel.is_cancelled() {
        termination = RunTermination::Cancelled;
        if raw.last().map(|x| x.step) != Some(completed) {
            raw.push(RawRecord::plain(
                completed,
                engine.state.partition().to_vec(),
                best.clone(),
            ))
        }
    }
    let mut partitions = Vec::<Vec<bool>>::new();
    let mut index = BTreeMap::<Vec<bool>, SolutionId>::new();
    let mut intern = |p: Vec<bool>| {
        if let Some(&id) = index.get(&p) {
            id
        } else {
            let id = SolutionId(partitions.len());
            partitions.push(p.clone());
            index.insert(p, id);
            id
        }
    };
    let records = raw
        .into_iter()
        .map(|r| {
            let current_solution = intern(r.current);
            let best_solution = intern(r.best);
            MeasurementRecord {
                step: r.step,
                current_solution,
                best_solution,
                current_smoothed: r.current_smoothed,
                search_evaluation: r.search_evaluation,
                basin_real: r.basin_real,
                basin_smoothed: r.basin_smoothed,
                basin_best: r.basin_best,
            }
        })
        .collect();
    let final_solution = intern(engine.state.partition().to_vec());
    let best_solution = intern(best);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let diagnostics = condition.measurement.diagnostics.then_some(Diagnostics {
        applied_moves: engine.applied_moves,
        objective_evaluations_search: engine.objective_evaluations,
        objective_evaluations_measurement: measurement_evals,
        fitness_values_computed_search: condition
            .solver
            .fitness()
            .is_some()
            .then_some(engine.fitness_values),
        search_ms: (elapsed_ms - measurement_ms).max(0.0),
        measurement_ms,
    });
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let result = RunResult {
        schema_version: 1,
        attempt_id: format!("{:016x}-{nonce:x}", seed),
        termination,
        completed_steps: completed,
        elapsed_ms,
        partitions,
        final_solution,
        best_solution,
        best_step,
        records,
        diagnostics,
    };
    result.validate(graph, condition)?;
    Ok(result)
}

struct RawRecord {
    step: u64,
    current: Vec<bool>,
    best: Vec<bool>,
    current_smoothed: Option<f64>,
    search_evaluation: Option<f64>,
    basin_real: Option<BasinResult>,
    basin_smoothed: Option<BasinResult>,
    basin_best: Option<BasinResult>,
}
impl RawRecord {
    fn plain(step: u64, current: Vec<bool>, best: Vec<bool>) -> Self {
        Self {
            step,
            current,
            best,
            current_smoothed: None,
            search_evaluation: None,
            basin_real: None,
            basin_smoothed: None,
            basin_best: None,
        }
    }
}
fn record_if_due(
    graph: &Graph,
    c: &Condition,
    seed: u64,
    e: &Engine<'_>,
    best: &[bool],
    step: u64,
    wanted: &[u64],
    next: &mut usize,
    out: &mut Vec<RawRecord>,
    cancel: &CancellationToken,
    evals: &mut u64,
    ms: &mut f64,
    best_basin_cache: &mut Option<(Vec<bool>, BasinResult)>,
) -> Result<()> {
    if wanted.get(*next) == Some(&step) {
        record(
            graph,
            c,
            seed,
            e,
            best,
            step,
            out,
            cancel,
            evals,
            ms,
            best_basin_cache,
        )?;
        *next += 1
    }
    Ok(())
}
fn record(
    graph: &Graph,
    c: &Condition,
    seed: u64,
    e: &Engine<'_>,
    best: &[bool],
    step: u64,
    out: &mut Vec<RawRecord>,
    cancel: &CancellationToken,
    evals: &mut u64,
    ms: &mut f64,
    best_basin_cache: &mut Option<(Vec<bool>, BasinResult)>,
) -> Result<()> {
    let start = Instant::now();
    let smoothing_spec = match &c.solver {
        SolverSpec::Hc { smoothing } | SolverSpec::Sa { smoothing, .. } => Some(smoothing),
        SolverSpec::Eo { .. } | SolverSpec::EoSa { .. } => None,
    };
    let mut current_smoothed = None;
    let mut search_evaluation = None;
    if let Some(spec) = smoothing_spec {
        if !matches!(
            spec,
            SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 }
        ) {
            let hash = graph.content_hash();
            let sb = seed.to_le_bytes();
            let tb = step.to_le_bytes();
            let mut rng = rng_for(&[hash.as_bytes(), &sb, &tb, b"measurement-smoothing"]);
            current_smoothed = Some(smoothing::evaluate(
                &e.state,
                graph,
                c.alpha,
                c.neighborhood,
                spec,
                Some(&mut rng),
                cancel,
                evals,
            )?);
            if matches!(spec, SmoothingSpec::RandomKAverage { .. }) {
                search_evaluation = Some(e.search_evaluation)
            }
        }
    }
    let (basin_real, basin_smoothed) = measure_basins(
        graph,
        c,
        seed,
        step,
        &e.state,
        smoothing_spec,
        cancel,
        evals,
    )?;
    let basin_best = measure_best_basin(graph, c, seed, best, cancel, evals, best_basin_cache)?;
    *ms += start.elapsed().as_secs_f64() * 1000.0;
    out.push(RawRecord {
        step,
        current: e.state.partition().to_vec(),
        best: best.to_vec(),
        current_smoothed,
        search_evaluation,
        basin_real,
        basin_smoothed,
        basin_best,
    });
    Ok(())
}
fn measure_best_basin(
    graph: &Graph,
    c: &Condition,
    seed: u64,
    best: &[bool],
    cancel: &CancellationToken,
    evals: &mut u64,
    cache: &mut Option<(Vec<bool>, BasinResult)>,
) -> Result<Option<BasinResult>> {
    if !c.measurement.best_basin {
        return Ok(None);
    }
    if let Some((partition, result)) = cache.as_ref() {
        if partition == best {
            return Ok(Some(result.clone()));
        }
    }
    let hash = graph.content_hash();
    let neighborhood = match c.neighborhood {
        Neighborhood::Flip => b"flip".as_slice(),
        Neighborhood::Swap => b"swap".as_slice(),
    };
    let alpha = c.alpha.to_bits().to_le_bytes();
    let search_seed = seed.to_le_bytes();
    let partition = best.iter().map(|&x| u8::from(x)).collect::<Vec<_>>();
    let mut ties = rng_for(&[
        hash.as_bytes(),
        neighborhood,
        &alpha,
        &search_seed,
        &partition,
        b"basin-best-real-v1",
    ]);
    let state = PartitionState::new(graph, best.to_vec())?;
    let result = basin(graph, c, &state, None, None, &mut ties, cancel, evals)?.0;
    *cache = Some((best.to_vec(), result.clone()));
    Ok(Some(result))
}
fn measure_basins(
    graph: &Graph,
    c: &Condition,
    seed: u64,
    step: u64,
    state: &PartitionState,
    spec: Option<&SmoothingSpec>,
    cancel: &CancellationToken,
    evals: &mut u64,
) -> Result<(Option<BasinResult>, Option<BasinResult>)> {
    if matches!(c.measurement.basin, BasinMode::None) {
        return Ok((None, None));
    }
    let hash = graph.content_hash();
    let sb = seed.to_le_bytes();
    let tb = step.to_le_bytes();
    let mut real_ties = rng_for(&[hash.as_bytes(), &sb, &tb, b"basin-real-ties"]);
    let (mut real, real_end) = basin(graph, c, state, None, None, &mut real_ties, cancel, evals)?;
    let nontrivial = spec.is_some_and(|x| {
        !matches!(
            x,
            SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 }
        )
    });
    let smooth = if matches!(c.measurement.basin, BasinMode::Both) && nontrivial {
        let mut rng = rng_for(&[hash.as_bytes(), &sb, &tb, b"measurement-smoothing"]);
        let mut ties = rng_for(&[hash.as_bytes(), &sb, &tb, b"basin-smoothed-ties"]);
        Some(
            basin(
                graph,
                c,
                state,
                spec,
                Some(&mut rng),
                &mut ties,
                cancel,
                evals,
            )?
            .0,
        )
    } else {
        None
    };
    if nontrivial {
        let mut rng = rng_for(&[hash.as_bytes(), &sb, &tb, b"measurement-smoothing"]);
        real.smoothed = Some(smoothing::evaluate(
            &real_end,
            graph,
            c.alpha,
            c.neighborhood,
            spec.expect("nontrivial smoothing has a specification"),
            Some(&mut rng),
            cancel,
            evals,
        )?);
    }
    Ok((Some(real), smooth))
}
fn basin(
    graph: &Graph,
    c: &Condition,
    start: &PartitionState,
    spec: Option<&SmoothingSpec>,
    rng: Option<&mut Mt19937GenRand64>,
    tie_rng: &mut Mt19937GenRand64,
    cancel: &CancellationToken,
    evals: &mut u64,
) -> Result<(BasinResult, PartitionState)> {
    let mut state = start.clone();
    let mut steps = 0;
    let fixed_rng = rng.as_ref().map(|source| (**source).clone());
    let mut current = if let Some(s) = spec {
        let mut evaluation_rng = fixed_rng.clone();
        smoothing::evaluate(
            &state,
            graph,
            c.alpha,
            c.neighborhood,
            s,
            evaluation_rng.as_mut(),
            cancel,
            evals,
        )?
    } else {
        *evals += 1;
        state.score(c.alpha)
    };
    // Scratch buffers shared by every real-objective scan of this descent.
    let mut real_scan = BestImprovement::new(c.neighborhood, c.alpha, NonFinite::Compare);
    let termination = loop {
        if steps >= c.measurement.max_basin_steps {
            break BasinTermination::StepLimit;
        }
        steps += 1;
        let found = match spec {
            // The same candidates, rule, tie draws and evaluation count as
            // scoring every move with `smoothing::move_score`.
            None => real_scan.scan(graph, &state, current, tie_rng, cancel, evals)?,
            Some(s) => smoothed_scan(
                graph,
                c,
                &mut state,
                s,
                fixed_rng.as_ref(),
                current,
                tie_rng,
                cancel,
                evals,
            )?,
        };
        match found {
            Some((mv, best)) => {
                smoothing::apply(&mut state, graph, mv);
                current = best
            }
            None => break BasinTermination::LocalOptimum,
        }
    };
    let real = state.score(c.alpha);
    let smoothed = spec.map(|_| current);
    Ok((
        BasinResult {
            real,
            smoothed,
            termination,
            steps: c.measurement.diagnostics.then_some(steps),
        },
        state,
    ))
}

/// One best-improvement scan of [`basin`] on the smoothed objective `spec`:
/// every move in canonical order, each candidate state evaluated with a fresh
/// copy of `fixed_rng`. Returns the chosen move and its smoothed value.
///
/// Candidates are evaluated in place and undone
/// ([`evaluate_smoothed_candidate_with_undo`]), so `state` is unchanged when
/// the scan returns, including by an error.
fn smoothed_scan(
    graph: &Graph,
    c: &Condition,
    state: &mut PartitionState,
    spec: &SmoothingSpec,
    fixed_rng: Option<&Mt19937GenRand64>,
    current: f64,
    tie_rng: &mut Mt19937GenRand64,
    cancel: &CancellationToken,
    evals: &mut u64,
) -> Result<Option<(Move, f64)>> {
    let mut choice = None;
    let mut best = current;
    let mut ties = 0u64;
    for (i, mv) in smoothing::moves_cancellable(state, c.neighborhood, cancel)?
        .into_iter()
        .enumerate()
    {
        if i & 1023 == 0 {
            cancel.check()?
        }
        let x = evaluate_smoothed_candidate_with_undo(
            state,
            graph,
            c.alpha,
            c.neighborhood,
            spec,
            fixed_rng,
            mv,
            cancel,
            evals,
        )?;
        if x < best {
            best = x;
            choice = Some(mv);
            ties = 1;
        } else if choice.is_some() && x == best {
            ties += 1;
            if rand::Rng::gen_range(tie_rng, 0..ties) == 0 {
                choice = Some(mv);
            }
        }
    }
    Ok(choice.map(|mv| (mv, best)))
}

/// The smoothed value of `state` after `mv`, evaluated with a fresh copy of
/// `fixed_rng`: the same value, draws and evaluation count as evaluating a
/// modified clone, without copying the state.
#[allow(clippy::too_many_arguments)]
fn evaluate_smoothed_candidate_with_undo(
    state: &mut PartitionState,
    graph: &Graph,
    alpha: f64,
    neighborhood: Neighborhood,
    spec: &SmoothingSpec,
    fixed_rng: Option<&Mt19937GenRand64>,
    mv: Move,
    cancel: &CancellationToken,
    evals: &mut u64,
) -> Result<f64> {
    smoothing::apply(state, graph, mv);
    let mut evaluation_rng = fixed_rng.cloned();
    let evaluated = smoothing::evaluate(
        state,
        graph,
        alpha,
        neighborhood,
        spec,
        evaluation_rng.as_mut(),
        cancel,
        evals,
    );
    // Every move is its own inverse. Undo before propagating an evaluation
    // error so cancellation leaves the last committed basin state intact.
    smoothing::apply(state, graph, mv);
    evaluated
}

#[cfg(test)]
mod best_basin_tests {
    use super::*;
    use crate::experiment::config::{Budget, GraphKind, GraphSpec, Measurement, Schedule};

    #[test]
    fn candidate_evaluation_error_restores_last_committed_state() {
        let graph = Graph::from_edges(
            6,
            vec![[0, 1], [0, 4], [1, 2], [1, 5], [2, 3], [3, 4], [4, 5]],
        )
        .unwrap();
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            let partition = match neighborhood {
                Neighborhood::Flip => vec![true, false, true, false, false, true],
                Neighborhood::Swap => vec![true, true, true, false, false, false],
            };
            let mut state = PartitionState::new(&graph, partition).unwrap();
            let before = state.clone();
            let spec = SmoothingSpec::WeightedAverage { k: 1 };
            let mv = smoothing::moves(&state, neighborhood)[0];
            let cancelled = CancellationToken::new();
            cancelled.cancel();
            let mut evaluations = 0;
            assert!(
                evaluate_smoothed_candidate_with_undo(
                    &mut state,
                    &graph,
                    0.05,
                    neighborhood,
                    &spec,
                    None,
                    mv,
                    &cancelled,
                    &mut evaluations,
                )
                .is_err()
            );
            assert_eq!(state.partition(), before.partition());
            assert_eq!(state.cut_edges(), before.cut_edges());
            assert_eq!(state.size_a(), before.size_a());
            assert_eq!(state.cuts_at(), before.cuts_at());
            assert_eq!(state.score(0.05).to_bits(), before.score(0.05).to_bits());
            assert_eq!(evaluations, 0);
        }
    }

    #[test]
    fn cancelled_best_basin_does_not_populate_cache() {
        let graph = Graph::from_edges(2, vec![[0, 1]]).unwrap();
        let condition = Condition {
            graph: GraphSpec {
                kind: GraphKind::Random,
                node_count: 2,
                expected_degree: 1.0,
                seed: 0,
            },
            neighborhood: Neighborhood::Flip,
            alpha: 0.0,
            solver: SolverSpec::Hc {
                smoothing: SmoothingSpec::None,
            },
            budget: Budget { max_steps: 1 },
            measurement: Measurement {
                schedule: Schedule::Explicit,
                steps: vec![],
                basin: BasinMode::None,
                max_basin_steps: 4,
                diagnostics: true,
                best_basin: true,
            },
        };
        let cancel = CancellationToken::new();
        cancel.cancel();
        let mut evaluations = 0;
        let mut cache = None;
        assert!(
            measure_best_basin(
                &graph,
                &condition,
                5,
                &[true, false],
                &cancel,
                &mut evaluations,
                &mut cache,
            )
            .is_err()
        );
        assert!(cache.is_none());
    }

    #[test]
    fn real_basin_matches_exhaustive_optimum_on_complete_four_vertex_graph() {
        let graph =
            Graph::from_edges(4, vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]).unwrap();
        let condition = Condition {
            graph: GraphSpec {
                kind: GraphKind::Random,
                node_count: 4,
                expected_degree: 3.0,
                seed: 0,
            },
            neighborhood: Neighborhood::Flip,
            alpha: 1.0,
            solver: SolverSpec::Hc {
                smoothing: SmoothingSpec::None,
            },
            budget: Budget { max_steps: 1 },
            measurement: Measurement {
                schedule: Schedule::Explicit,
                steps: vec![],
                basin: BasinMode::None,
                max_basin_steps: 8,
                diagnostics: true,
                best_basin: true,
            },
        };
        let start = PartitionState::new(&graph, vec![false; 4]).unwrap();
        let mut ties = rng_for(&[b"exhaustive-four"]);
        let mut evaluations = 0;
        let result = basin(
            &graph,
            &condition,
            &start,
            None,
            None,
            &mut ties,
            &CancellationToken::new(),
            &mut evaluations,
        )
        .unwrap()
        .0;
        let exhaustive = (0u8..16)
            .map(|bits| {
                let partition = (0..4).map(|i| bits & (1 << i) != 0).collect::<Vec<_>>();
                graph.score(&partition, condition.alpha)
            })
            .min_by(f64::total_cmp)
            .unwrap();
        assert_eq!(result.real, exhaustive);
        assert_eq!(result.real, 4.0);
        assert_eq!(result.termination, BasinTermination::LocalOptimum);
    }
}

#[cfg(test)]
#[path = "performance_probe.rs"]
mod performance_probe;
