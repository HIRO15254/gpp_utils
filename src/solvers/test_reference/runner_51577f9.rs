use super::{Engine, StepStatus};
use crate::error::{Error, Result};
use crate::experiment::config::{BasinMode, Condition, Neighborhood, SmoothingSpec, SolverSpec};
use crate::experiment::measurement::checkpoints;
use crate::experiment::result::{
    BasinResult, BasinTermination, Diagnostics, MeasurementRecord, RunResult, RunTermination,
    SolutionId,
};
use crate::fitness::FitnessRegistry;
use crate::graph_partition::{Graph, PartitionState};
use crate::optimization::{CancellationToken, rng_for};
use super::super::smoothing_e4b6a1c as smoothing;
use rand_mt::Mt19937GenRand64;
use std::{
    collections::BTreeMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

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
            if !tau.is_finite() || *tau <= 0.0 {
                return Err(Error::msg("tau must be finite and positive"));
            }
            registry.validate(fitness)?;
        }
        SolverSpec::EoSa { .. } => unreachable!("eo_sa postdates this frozen reference"),
    }
    if matches!(condition.neighborhood, Neighborhood::Swap) && !graph.node_count().is_multiple_of(2) {
        return Err(Error::msg("swap requires an even node count"));
    }
    let started = Instant::now();
    let mut engine = Engine::new(graph, condition, seed, registry, cancel)?;
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
    ) {
        if cancel.is_cancelled() {
            termination = RunTermination::Cancelled;
            raw.push(RawRecord::plain(0, initial.clone(), initial.clone()));
        } else {
            return Err(error);
        }
    }
    while completed < condition.budget.max_steps && termination != RunTermination::Cancelled {
        if cancel.is_cancelled() {
            termination = RunTermination::Cancelled;
            break;
        }
        let status = match engine.step(cancel) {
            Ok(x) => x,
            Err(_error) if cancel.is_cancelled() => {
                termination = RunTermination::Cancelled;
                break;
            }
            Err(e) => return Err(e),
        };
        completed += 1;
        let real = engine.state.score(condition.alpha);
        if real < best_score {
            best_score = real;
            best = engine.state.partition().to_vec();
            best_step = completed
        }
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
        .map(|r| MeasurementRecord {
            step: r.step,
            current_solution: intern(r.current),
            best_solution: intern(r.best),
            current_smoothed: r.current_smoothed,
            search_evaluation: r.search_evaluation,
            basin_real: r.basin_real,
            basin_smoothed: r.basin_smoothed,
            basin_best: None,
        })
        .collect();
    let final_solution = intern(engine.state.partition().to_vec());
    let best_solution = intern(best);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let diagnostics = condition.measurement.diagnostics.then_some(Diagnostics {
        applied_moves: engine.applied_moves,
        objective_evaluations_search: engine.objective_evaluations,
        objective_evaluations_measurement: measurement_evals,
        fitness_values_computed_search: matches!(condition.solver, SolverSpec::Eo { .. })
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
) -> Result<()> {
    if wanted.get(*next) == Some(&step) {
        record(graph, c, seed, e, best, step, out, cancel, evals, ms)?;
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
) -> Result<()> {
    let start = Instant::now();
    let smoothing_spec = match &c.solver {
        SolverSpec::Hc { smoothing } | SolverSpec::Sa { smoothing, .. } => Some(smoothing),
        SolverSpec::Eo { .. } => None,
        SolverSpec::EoSa { .. } => unreachable!("eo_sa postdates this frozen reference"),
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
    *ms += start.elapsed().as_secs_f64() * 1000.0;
    out.push(RawRecord {
        step,
        current: e.state.partition().to_vec(),
        best: best.to_vec(),
        current_smoothed,
        search_evaluation,
        basin_real,
        basin_smoothed,
    });
    Ok(())
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
    let termination = loop {
        if steps >= c.measurement.max_basin_steps {
            break BasinTermination::StepLimit;
        }
        steps += 1;
        let mut choice = None;
        let mut best = current;
        let mut ties = 0u64;
        for (i, mv) in smoothing::moves_cancellable(&state, c.neighborhood, cancel)?
            .into_iter()
            .enumerate()
        {
            if i & 1023 == 0 {
                cancel.check()?
            }
            let mut candidate = state.clone();
            smoothing::apply(&mut candidate, graph, mv);
            let x = if let Some(s) = spec {
                let mut evaluation_rng = fixed_rng.clone();
                smoothing::evaluate(
                    &candidate,
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
                candidate.score(c.alpha)
            };
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
        match choice {
            Some(mv) => {
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
