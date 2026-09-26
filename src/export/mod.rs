//! Derived TSV views. Source JSON remains the only experiment data store.
mod columns;
use columns::{ColumnSpec, RUN_COLUMNS, RunField, TRACE_COLUMNS, TraceField};

use crate::{
    error::Result,
    experiment::{
        config::{Condition, SmoothingSpec, SolverSpec},
        plan::{ExperimentPlan, Job},
        result::{BasinView, MeasurementView, RunView, ScoreBreakdown},
    },
    storage::{self, JobInspection, atomic},
};
use anyhow::bail;
use serde::Serialize;
use serde_json::json;
use std::{
    collections::HashMap,
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Serialize)]
pub struct ExportSummary {
    pub jobs: usize,
    pub trace_rows: usize,
    pub incomplete_included: bool,
}
struct RunCtx<'a> {
    plan: &'a ExperimentPlan,
    job: &'a Job,
    inspection: &'a JobInspection,
    view: Option<RunView<'a>>,
    initial: Option<ScoreBreakdown>,
    final_breakdown: Option<ScoreBreakdown>,
    best_breakdown: Option<ScoreBreakdown>,
}
struct TraceCtx<'a> {
    run: &'a RunCtx<'a>,
    measurement: MeasurementView<'a, 'a>,
    current: ScoreBreakdown,
    best: ScoreBreakdown,
}
fn enum_text<T: Serialize>(v: &T) -> String {
    serde_json::to_string(v)
        .expect("serializable enum")
        .trim_matches('"')
        .into()
}
fn option<T: ToString>(v: Option<T>) -> String {
    v.map(|v| v.to_string()).unwrap_or_default()
}
fn stored_float(v: f64) -> String {
    serde_json::to_string(&v).expect("validated finite stored float")
}
fn option_stored_float(v: Option<f64>) -> String {
    v.map(stored_float).unwrap_or_default()
}
fn identity_smoothing(c: &Condition) -> bool {
    matches!(
        smooth(c),
        Some(SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 })
    )
}
fn smooth(c: &Condition) -> Option<&SmoothingSpec> {
    match &c.solver {
        SolverSpec::Hc { smoothing } | SolverSpec::Sa { smoothing, .. } => Some(smoothing),
        SolverSpec::Eo { .. } | SolverSpec::EoSa { .. } => None,
    }
}
fn solver(c: &Condition) -> &SolverSpec {
    &c.solver
}
fn basin_string(b: Option<BasinView<'_>>, value: fn(BasinView<'_>) -> Option<String>) -> String {
    b.and_then(value).unwrap_or_default()
}
fn run_value(c: &RunCtx<'_>, f: RunField) -> String {
    let condition = &c.job.condition;
    let view = c.view.as_ref();
    match f {
        RunField::Batch => c.plan.batch_id.clone(),
        RunField::Condition => c.job.condition_id.clone(),
        RunField::Seed => c.job.seed.to_string(),
        RunField::Status => c.inspection.status.clone(),
        RunField::Latest => c
            .inspection
            .latest_attempt_status
            .clone()
            .unwrap_or_default(),
        RunField::Termination => {
            if c.inspection.status == "completed" {
                view.map(|v| enum_text(&v.result().termination))
                    .unwrap_or_default()
            } else {
                String::new()
            }
        }
        RunField::GraphId => c.job.graph_id.clone(),
        RunField::GraphKind => enum_text(&condition.graph.kind),
        RunField::Nodes => condition.graph.node_count.to_string(),
        RunField::ExpectedDegree => condition.graph.expected_degree.to_string(),
        RunField::GraphSeed => condition.graph.seed.to_string(),
        RunField::Edges => view
            .map(|v| v.graph().edges().len().to_string())
            .unwrap_or_default(),
        RunField::ActualDegree => view
            .map(|v| {
                (2.0 * v.graph().edges().len() as f64 / v.graph().node_count() as f64).to_string()
            })
            .unwrap_or_default(),
        RunField::Alpha => condition.alpha.to_string(),
        RunField::Neighborhood => enum_text(&condition.neighborhood),
        RunField::Solver => match solver(condition) {
            SolverSpec::Hc { .. } => "hc",
            SolverSpec::Sa { .. } => "sa",
            SolverSpec::Eo { .. } => "eo",
            SolverSpec::EoSa { .. } => "eo_sa",
        }
        .into(),
        RunField::Temperature => match solver(condition) {
            SolverSpec::Sa { temperature, .. } | SolverSpec::EoSa { temperature, .. } => {
                temperature.to_string()
            }
            _ => String::new(),
        },
        RunField::Tau => match solver(condition) {
            SolverSpec::Eo { tau, .. } | SolverSpec::EoSa { tau, .. } => tau.to_string(),
            _ => String::new(),
        },
        RunField::Smoothing => match smooth(condition) {
            Some(SmoothingSpec::None) => "none".into(),
            Some(SmoothingSpec::AllAverage) => "all_average".into(),
            Some(SmoothingSpec::RandomKAverage { .. }) => "random_k_average".into(),
            Some(SmoothingSpec::WeightedAverage { .. }) => "weighted_average".into(),
            None => String::new(),
        },
        RunField::K => match smooth(condition) {
            Some(SmoothingSpec::RandomKAverage { k } | SmoothingSpec::WeightedAverage { k }) => {
                k.to_string()
            }
            _ => String::new(),
        },
        RunField::Fitness => solver(condition)
            .fitness()
            .map(|fitness| fitness.kind.clone())
            .unwrap_or_default(),
        RunField::FitnessVersion => {
            let name = run_value(c, RunField::Fitness);
            c.plan
                .experiment
                .versions
                .get(&format!("fitness:{name}"))
                .cloned()
                .unwrap_or_default()
        }
        RunField::FitnessParams => solver(condition)
            .fitness()
            .map(|fitness| fitness.params.to_string())
            .unwrap_or_default(),
        RunField::MaxSteps => condition.budget.max_steps.to_string(),
        RunField::Completed => view
            .map(|v| v.result().completed_steps.to_string())
            .unwrap_or_default(),
        RunField::BestStep => view
            .map(|v| v.result().best_step.to_string())
            .unwrap_or_default(),
        RunField::InitialReal
        | RunField::FinalReal
        | RunField::BestReal
        | RunField::FinalCuts
        | RunField::FinalA
        | RunField::FinalB
        | RunField::FinalPenalty
        | RunField::BestCuts
        | RunField::BestA
        | RunField::BestB
        | RunField::BestPenalty => match f {
            RunField::InitialReal => c.initial,
            RunField::FinalReal
            | RunField::FinalCuts
            | RunField::FinalA
            | RunField::FinalB
            | RunField::FinalPenalty => c.final_breakdown,
            RunField::BestReal
            | RunField::BestCuts
            | RunField::BestA
            | RunField::BestB
            | RunField::BestPenalty => c.best_breakdown,
            _ => unreachable!("non-breakdown field"),
        }
        .map(|b| match f {
            RunField::InitialReal | RunField::FinalReal | RunField::BestReal => b.real.to_string(),
            RunField::FinalCuts | RunField::BestCuts => b.cut_edges.to_string(),
            RunField::FinalA | RunField::BestA => b.size_a.to_string(),
            RunField::FinalB | RunField::BestB => b.size_b.to_string(),
            _ => b.balance_penalty.to_string(),
        })
        .unwrap_or_default(),
        RunField::Elapsed => view
            .map(|v| stored_float(v.result().elapsed_ms))
            .unwrap_or_default(),
        RunField::FinalBasinReal => basin_string(
            view.map(RunView::final_measurement)
                .and_then(|m| m.real_basin()),
            |b| Some(stored_float(b.real())),
        ),
        RunField::FinalBasinRealStatus => basin_string(
            view.map(RunView::final_measurement)
                .and_then(|m| m.real_basin()),
            |b| Some(enum_text(&b.termination())),
        ),
        RunField::FinalBasinSmooth => basin_string(
            view.map(RunView::final_measurement)
                .and_then(|m| m.smoothed_basin()),
            |b| Some(stored_float(b.real())),
        ),
        RunField::FinalBasinSmoothStatus => basin_string(
            view.map(RunView::final_measurement)
                .and_then(|m| m.smoothed_basin()),
            |b| Some(enum_text(&b.termination())),
        ),
        RunField::FinalBasinBest => basin_string(
            view.map(RunView::final_measurement)
                .and_then(|m| m.best_basin()),
            |b| Some(stored_float(b.real())),
        ),
        RunField::FinalBasinBestStatus => basin_string(
            view.map(RunView::final_measurement)
                .and_then(|m| m.best_basin()),
            |b| Some(enum_text(&b.termination())),
        ),
        RunField::Applied | RunField::Accepted => {
            option(view.and_then(|v| v.result().diagnostics.as_ref().map(|d| d.applied_moves)))
        }
        RunField::Rejected => {
            if matches!(
                solver(condition),
                SolverSpec::Sa { .. } | SolverSpec::EoSa { .. }
            ) {
                view.and_then(|v| {
                    v.result().diagnostics.as_ref().map(|d| {
                        v.result()
                            .completed_steps
                            .saturating_sub(d.applied_moves)
                            .to_string()
                    })
                })
                .unwrap_or_default()
            } else {
                String::new()
            }
        }
        RunField::SearchEvals => option(view.and_then(|v| {
            v.result()
                .diagnostics
                .as_ref()
                .map(|d| d.objective_evaluations_search)
        })),
        RunField::MeasurementEvals => option(view.and_then(|v| {
            v.result()
                .diagnostics
                .as_ref()
                .map(|d| d.objective_evaluations_measurement)
        })),
        RunField::FitnessEvals => option(view.and_then(|v| {
            v.result()
                .diagnostics
                .as_ref()
                .and_then(|d| d.fitness_values_computed_search)
        })),
        RunField::SearchMs => option_stored_float(
            view.and_then(|v| v.result().diagnostics.as_ref().map(|d| d.search_ms)),
        ),
        RunField::MeasurementMs => option_stored_float(
            view.and_then(|v| v.result().diagnostics.as_ref().map(|d| d.measurement_ms)),
        ),
    }
}
fn trace_value(c: &TraceCtx<'_>, f: TraceField) -> String {
    let m = &c.measurement;
    match f {
        TraceField::Condition => run_value(c.run, RunField::Condition),
        TraceField::Seed => run_value(c.run, RunField::Seed),
        TraceField::Status => run_value(c.run, RunField::Status),
        TraceField::Step => m.step().to_string(),
        TraceField::CurrentReal => c.current.real.to_string(),
        TraceField::BestReal => c.best.real.to_string(),
        TraceField::SearchEvaluation => {
            if identity_smoothing(&c.run.job.condition) {
                option(m.search_evaluation())
            } else {
                option_stored_float(m.search_evaluation())
            }
        }
        TraceField::CurrentSmoothed => {
            if identity_smoothing(&c.run.job.condition) {
                option(m.smoothed_value())
            } else {
                option_stored_float(m.smoothed_value())
            }
        }
        TraceField::BasinReal => basin_string(m.real_basin(), |b| Some(stored_float(b.real()))),
        TraceField::BasinRealSmoothed => {
            basin_string(m.real_basin(), |b| b.smoothed().map(stored_float))
        }
        TraceField::BasinRealStatus => {
            basin_string(m.real_basin(), |b| Some(enum_text(&b.termination())))
        }
        TraceField::BasinRealSteps => {
            basin_string(m.real_basin(), |b| b.steps().map(|x| x.to_string()))
        }
        TraceField::BasinSmooth => {
            basin_string(m.smoothed_basin(), |b| Some(stored_float(b.real())))
        }
        TraceField::BasinSmoothValue => {
            basin_string(m.smoothed_basin(), |b| b.smoothed().map(stored_float))
        }
        TraceField::BasinSmoothStatus => {
            basin_string(m.smoothed_basin(), |b| Some(enum_text(&b.termination())))
        }
        TraceField::BasinSmoothSteps => {
            basin_string(m.smoothed_basin(), |b| b.steps().map(|x| x.to_string()))
        }
        TraceField::BasinBest => basin_string(m.best_basin(), |b| Some(stored_float(b.real()))),
        TraceField::BasinBestStatus => {
            basin_string(m.best_basin(), |b| Some(enum_text(&b.termination())))
        }
        TraceField::BasinBestSteps => {
            basin_string(m.best_basin(), |b| b.steps().map(|x| x.to_string()))
        }
    }
}
fn write_cell(out: &mut impl Write, s: &str) -> std::io::Result<()> {
    if s.contains(['\t', '\n', '\r', '"']) {
        out.write_all(b"\"")?;
        for part in s.split_inclusive('"') {
            if let Some(prefix) = part.strip_suffix('"') {
                out.write_all(prefix.as_bytes())?;
                out.write_all(b"\"\"")?;
            } else {
                out.write_all(part.as_bytes())?;
            }
        }
        out.write_all(b"\"")
    } else {
        out.write_all(s.as_bytes())
    }
}
fn write_row(out: &mut impl Write, cells: impl Iterator<Item = String>) -> std::io::Result<()> {
    for (index, cell) in cells.enumerate() {
        if index != 0 {
            out.write_all(b"\t")?;
        }
        write_cell(out, &cell)?;
    }
    out.write_all(b"\n")
}
fn describe<F>(cols: &[ColumnSpec<F>]) -> Vec<serde_json::Value> {
    cols.iter()
        .map(|c| json!({"name":c.name,"type":c.kind,"unit":c.unit,"meaning":c.meaning}))
        .collect()
}
fn write_header<F>(out: &mut impl Write, columns: &[ColumnSpec<F>]) -> std::io::Result<()> {
    write_row(out, columns.iter().map(|c| c.name.into()))
}
pub fn export_tsv(
    plan: &ExperimentPlan,
    root: &Path,
    out: &Path,
    include_incomplete: bool,
    overwrite: bool,
) -> Result<ExportSummary> {
    for file in ["runs.tsv", "traces.tsv", "metadata.json"] {
        if out.join(file).exists() && !overwrite {
            bail!("output already exists: {}", out.join(file).display())
        }
    }
    let mut sorted = plan.clone();
    sorted
        .jobs
        .sort_by(|a, b| (&a.condition_id, a.seed).cmp(&(&b.condition_id, b.seed)));
    std::fs::create_dir_all(out)?;
    let mut runs_tmp = tempfile::NamedTempFile::new_in(out)?;
    let mut traces_tmp = tempfile::NamedTempFile::new_in(out)?;
    let metadata_tmp = tempfile::NamedTempFile::new_in(out)?;
    let mut trace_rows = 0;
    let active = atomic::writer_active(root)?;
    let mut graphs = HashMap::<String, Arc<crate::graph_partition::Graph>>::new();
    {
        let mut runs = BufWriter::new(runs_tmp.as_file_mut());
        let mut traces = BufWriter::new(traces_tmp.as_file_mut());
        write_header(&mut runs, RUN_COLUMNS)?;
        write_header(&mut traces, TRACE_COLUMNS)?;
        for job in &sorted.jobs {
            let inspection =
                storage::inspect_job(job, root, include_incomplete, active, &mut graphs)?;
            let view = inspection
                .result
                .as_ref()
                .zip(inspection.graph.as_deref())
                .map(|(r, g)| RunView::from_validated(g, &job.condition, r));
            let initial = view.as_ref().map(|view| {
                view.measurement(&view.result().records[0])
                    .current_breakdown()
            });
            let final_breakdown = view
                .as_ref()
                .map(|view| view.breakdown(view.result().final_solution));
            let best_breakdown = view
                .as_ref()
                .map(|view| view.breakdown(view.result().best_solution));
            let context = RunCtx {
                plan,
                job,
                inspection: &inspection,
                view,
                initial,
                final_breakdown,
                best_breakdown,
            };
            write_row(
                &mut runs,
                RUN_COLUMNS.iter().map(|col| run_value(&context, col.field)),
            )?;
            if let Some(view) = context.view.as_ref() {
                for measurement in view.records() {
                    let trace = TraceCtx {
                        run: &context,
                        current: measurement.current_breakdown(),
                        best: measurement.best_breakdown(),
                        measurement,
                    };
                    write_row(
                        &mut traces,
                        TRACE_COLUMNS
                            .iter()
                            .map(|col| trace_value(&trace, col.field)),
                    )?;
                    trace_rows += 1
                }
            }
        }
        runs.flush()?;
        traces.flush()?;
    }
    let metadata = json!({"schema_version":1,"created_unix_ms":SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),"batch_id":plan.batch_id,"include_incomplete":include_incomplete,"selection":if plan.jobs.len()==1{Some(json!({"condition_id":plan.jobs[0].condition_id,"seed":plan.jobs[0].seed}))}else{None},"columns":{"runs":describe(RUN_COLUMNS),"traces":describe(TRACE_COLUMNS)}});
    {
        let mut metadata_writer = BufWriter::new(metadata_tmp.as_file());
        serde_json::to_writer_pretty(&mut metadata_writer, &metadata)?;
        metadata_writer.write_all(b"\n")?;
        metadata_writer.flush()?;
    }
    // No destination is touched until every source result has been read,
    // validated, and all three complete temporary outputs have been written.
    atomic::persist_temp(runs_tmp, &out.join("runs.tsv"), overwrite)?;
    atomic::persist_temp(traces_tmp, &out.join("traces.tsv"), overwrite)?;
    atomic::persist_temp(metadata_tmp, &out.join("metadata.json"), overwrite)?;
    Ok(ExportSummary {
        jobs: sorted.jobs.len(),
        trace_rows,
        incomplete_included: include_incomplete,
    })
}
