//! Derived TSV views. Source JSON remains the only experiment data store.
use crate::{
    error::Result,
    experiment::{
        config::{Condition, SmoothingSpec, SolverSpec},
        plan::ExperimentPlan,
    },
    storage::{self, atomic},
};
use anyhow::bail;
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};
#[derive(Serialize)]
pub struct ExportSummary {
    pub jobs: usize,
    pub trace_rows: usize,
    pub incomplete_included: bool,
}
const RUNS: &str = "batch_id condition_id seed status latest_attempt_status termination graph_id graph_kind node_count expected_degree graph_seed edge_count actual_average_degree alpha neighborhood solver temperature tau smoothing k fitness fitness_version fitness_params_json max_steps completed_steps best_step initial_real final_real best_real final_cut_edges final_size_a final_size_b final_balance_penalty best_cut_edges best_size_a best_size_b best_balance_penalty elapsed_ms final_basin_real_from_real final_basin_real_status final_basin_real_from_smoothed final_basin_smoothed_status final_basin_real_from_best final_basin_best_status applied_moves accepted_moves rejected_moves objective_evaluations_search objective_evaluations_measurement fitness_values_computed_search search_ms measurement_ms";
const TRACES: &str = "condition_id seed status step current_real best_real search_evaluation current_smoothed basin_real_from_real basin_smoothed_from_real basin_real_status basin_real_steps basin_real_from_smoothed basin_smoothed_from_smoothed basin_smoothed_status basin_smoothed_steps basin_real_from_best basin_best_status basin_best_steps";
fn enum_text<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v)
        .unwrap_or(Value::Null)
        .as_str()
        .unwrap_or("")
        .to_owned()
}
fn cell(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}
fn field(v: &Value, key: &str) -> String {
    cell(&v[key])
}
fn quote(s: &str) -> String {
    if s.contains(['\t', '\n', '\r', '"']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.into()
    }
}
fn row(buf: &mut String, cells: Vec<String>) {
    buf.push_str(
        &cells
            .iter()
            .map(|s| quote(s))
            .collect::<Vec<_>>()
            .join("\t"),
    );
    buf.push('\n');
}
fn smoothing(condition: &Condition) -> Option<&SmoothingSpec> {
    match &condition.solver {
        SolverSpec::Hc { smoothing } | SolverSpec::Sa { smoothing, .. } => Some(smoothing),
        SolverSpec::Eo { .. } => None,
    }
}
fn identity(condition: &Condition) -> bool {
    matches!(
        smoothing(condition),
        Some(SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 })
    )
}
fn real(graph: &crate::graph_partition::Graph, c: &Condition, p: &[bool]) -> String {
    graph.score(p, c.alpha).to_string()
}
fn breakdown(graph: &crate::graph_partition::Graph, c: &Condition, p: &[bool]) -> Vec<String> {
    let a = p.iter().filter(|&&b| b).count();
    let b = p.len() - a;
    let cuts = graph.edges().iter().filter(|e| p[e[0]] != p[e[1]]).count();
    vec![
        cuts.to_string(),
        a.to_string(),
        b.to_string(),
        (c.alpha * (a as f64 - b as f64).powi(2)).to_string(),
    ]
}
fn basin(record: &Value, condition: &Condition, smoothed: bool) -> Value {
    if smoothed && identity(condition) {
        record["basin_real"].clone()
    } else if smoothed {
        record["basin_smoothed"].clone()
    } else {
        record["basin_real"].clone()
    }
}
fn basin_smooth(basin: &Value, condition: &Condition) -> String {
    if smoothing(condition).is_none() {
        String::new()
    } else if identity(condition) {
        field(basin, "real")
    } else {
        field(basin, "smoothed")
    }
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
            bail!("output already exists: {}", out.join(file).display());
        }
    }
    let mut jobs: Vec<_> = plan.jobs.iter().collect();
    jobs.sort_by(|a, b| (&a.condition_id, a.seed).cmp(&(&b.condition_id, b.seed)));
    let mut sorted = plan.clone();
    sorted.jobs = jobs.into_iter().cloned().collect();
    let inspections = storage::inspect(&sorted, root, include_incomplete)?;
    let mut runs = RUNS.split_whitespace().collect::<Vec<_>>().join("\t") + "\n";
    let mut traces = TRACES.split_whitespace().collect::<Vec<_>>().join("\t") + "\n";
    let mut trace_rows = 0;
    for (job, inspection) in sorted.jobs.iter().zip(&inspections) {
        let c = &job.condition;
        let v = inspection
            .result
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?
            .unwrap_or(Value::Null);
        let result = inspection.result.as_ref();
        let graph = inspection.graph.as_ref();
        let mut cells = vec![
            plan.batch_id.clone(),
            job.condition_id.clone(),
            job.seed.to_string(),
            inspection.status.clone(),
            inspection.latest_attempt_status.clone().unwrap_or_default(),
            if inspection.status == "completed" {
                field(&v, "termination")
            } else {
                String::new()
            },
            job.graph_id.clone(),
            enum_text(&c.graph.kind),
            c.graph.node_count.to_string(),
            c.graph.expected_degree.to_string(),
            c.graph.seed.to_string(),
            graph
                .map(|g| g.edges().len().to_string())
                .unwrap_or_default(),
            graph
                .map(|g| (2.0 * g.edges().len() as f64 / g.node_count() as f64).to_string())
                .unwrap_or_default(),
            c.alpha.to_string(),
            enum_text(&c.neighborhood),
        ];
        let (kind, temp, tau, fitness, params) = match &c.solver {
            SolverSpec::Hc { .. } => (
                "hc",
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            ),
            SolverSpec::Sa { temperature, .. } => (
                "sa",
                temperature.to_string(),
                String::new(),
                String::new(),
                String::new(),
            ),
            SolverSpec::Eo { tau, fitness } => (
                "eo",
                String::new(),
                tau.to_string(),
                fitness.kind.clone(),
                fitness.params.to_string(),
            ),
        };
        let sm = smoothing(c)
            .map(serde_json::to_value)
            .transpose()?
            .unwrap_or(Value::Null);
        cells.extend([
            kind.into(),
            temp,
            tau,
            field(&sm, "kind"),
            field(&sm, "k"),
            fitness.clone(),
            plan.experiment
                .versions
                .get(&format!("fitness:{fitness}"))
                .cloned()
                .unwrap_or_default(),
            params,
            c.budget.max_steps.to_string(),
            field(&v, "completed_steps"),
            field(&v, "best_step"),
        ]);
        if let (Some(r), Some(g)) = (result, graph) {
            let initial = &r.partitions[r.records[0].current_solution.0];
            let final_p = &r.partitions[r.final_solution.0];
            let best = &r.partitions[r.best_solution.0];
            cells.extend([real(g, c, initial), real(g, c, final_p), real(g, c, best)]);
            cells.extend(breakdown(g, c, final_p));
            cells.extend(breakdown(g, c, best));
        } else {
            cells.extend(vec![String::new(); 11]);
        }
        cells.push(field(&v, "elapsed_ms"));
        let last = v["records"]
            .as_array()
            .and_then(|r| r.last())
            .cloned()
            .unwrap_or(Value::Null);
        let br = basin(&last, c, false);
        let bs = basin(&last, c, true);
        let bb = last["basin_best"].clone();
        cells.extend([
            field(&br, "real"),
            field(&br, "termination"),
            field(&bs, "real"),
            field(&bs, "termination"),
            field(&bb, "real"),
            field(&bb, "termination"),
        ]);
        let d = &v["diagnostics"];
        let applied = field(d, "applied_moves");
        let rejected = if kind == "sa" {
            v["completed_steps"]
                .as_u64()
                .zip(d["applied_moves"].as_u64())
                .map(|(s, a)| s.saturating_sub(a).to_string())
                .unwrap_or_default()
        } else {
            String::new()
        };
        cells.extend([
            applied.clone(),
            applied,
            rejected,
            field(d, "objective_evaluations_search"),
            field(d, "objective_evaluations_measurement"),
            field(d, "fitness_values_computed_search"),
            field(d, "search_ms"),
            field(d, "measurement_ms"),
        ]);
        debug_assert_eq!(cells.len(), RUNS.split_whitespace().count());
        row(&mut runs, cells);
        if let (Some(r), Some(g)) = (result, graph) {
            for (record, rv) in r
                .records
                .iter()
                .zip(v["records"].as_array().expect("records array"))
            {
                let current = real(g, c, &r.partitions[record.current_solution.0]);
                let best = real(g, c, &r.partitions[record.best_solution.0]);
                let smoothed = if identity(c) {
                    current.clone()
                } else {
                    field(rv, "current_smoothed")
                };
                let search = if smoothing(c).is_none() {
                    String::new()
                } else if matches!(smoothing(c), Some(SmoothingSpec::RandomKAverage { .. })) {
                    field(rv, "search_evaluation")
                } else {
                    smoothed.clone()
                };
                let br = basin(rv, c, false);
                let bs = basin(rv, c, true);
                let bb = rv["basin_best"].clone();
                row(
                    &mut traces,
                    vec![
                        job.condition_id.clone(),
                        job.seed.to_string(),
                        inspection.status.clone(),
                        record.step.to_string(),
                        current,
                        best,
                        search,
                        smoothed,
                        field(&br, "real"),
                        basin_smooth(&br, c),
                        field(&br, "termination"),
                        field(&br, "steps"),
                        field(&bs, "real"),
                        basin_smooth(&bs, c),
                        field(&bs, "termination"),
                        field(&bs, "steps"),
                        field(&bb, "real"),
                        field(&bb, "termination"),
                        field(&bb, "steps"),
                    ],
                );
                trace_rows += 1;
            }
        }
    }
    let describe = |names: &str| {
        names.split_whitespace().map(|name|json!({"name":name,"type":column_type(name),"unit":if name.ends_with("_ms"){Some("ms")}else{None},"meaning":column_meaning(name)})).collect::<Vec<_>>()
    };
    let metadata = json!({"schema_version":1,"created_unix_ms":SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),"batch_id":plan.batch_id,"include_incomplete":include_incomplete,"selection":if plan.jobs.len()==1{Some(json!({"condition_id":plan.jobs[0].condition_id,"seed":plan.jobs[0].seed}))}else{None},"columns":{"runs":describe(RUNS),"traces":describe(TRACES)}});
    atomic::write_bytes(&out.join("runs.tsv"), runs.as_bytes(), overwrite)?;
    atomic::write_bytes(&out.join("traces.tsv"), traces.as_bytes(), overwrite)?;
    atomic::write_json(&out.join("metadata.json"), &metadata, overwrite)?;
    Ok(ExportSummary {
        jobs: inspections.len(),
        trace_rows,
        incomplete_included: include_incomplete,
    })
}
fn column_type(name: &str) -> &'static str {
    if name.ends_with("_ms")
        || name.contains("real") && !name.contains("status") && !name.contains("steps")
        || matches!(
            name,
            "temperature" | "tau" | "alpha" | "expected_degree" | "actual_average_degree"
        )
        || name.contains("smoothed") && !name.contains("status") && !name.contains("steps")
        || name.ends_with("penalty")
        || name == "search_evaluation"
    {
        "number"
    } else if matches!(
        name,
        "seed"
            | "graph_seed"
            | "node_count"
            | "edge_count"
            | "k"
            | "step"
            | "best_step"
            | "completed_steps"
            | "max_steps"
    ) || name.contains("moves")
        || name.contains("evaluations")
        || name.contains("computed")
        || name.contains("size_")
        || name.ends_with("cut_edges")
        || name.ends_with("steps")
    {
        "integer"
    } else {
        "string"
    }
}
fn column_meaning(name: &str) -> String {
    match name {"current_real"=>"Real objective of current partition".into(),"best_real"=>"Real objective of incumbent partition across every visited step".into(),"basin_real_from_best"|"final_basin_real_from_best"=>"Real objective reached by a real-objective basin descent from the incumbent partition".into(),"basin_best_status"|"final_basin_best_status"=>"Termination status of real-objective basin descent from the incumbent partition".into(),"basin_best_steps"=>"Scanned basin steps from the incumbent partition; present when diagnostics are enabled".into(),"search_evaluation"=>"Evaluation retained by the search; blank for EO or unavailable measurement".into(),"latest_attempt_status"=>"Latest unfinished attempt, separately from a reusable completed result".into(),"elapsed_ms"=>"Run initialization, search and measurement time; excludes graph generation and disk write".into(),"fitness"=>"Registered vertex fitness name".into(),"k"=>"Effective smoothing sample count".into(),_=>name.replace('_'," ")}
}
