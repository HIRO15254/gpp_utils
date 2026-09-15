//! Immutable experiment inputs and atomic result storage.
pub mod atomic;
use crate::{
    error::Result,
    experiment::{
        config::Condition,
        plan::{ExperimentPlan, Job, StoredExperiment, compile_stored},
        result::{RunResult, RunTermination},
        runner::run_one,
    },
    fitness::FitnessRegistry,
    graph_partition::Graph,
    optimization::CancellationToken,
};
use anyhow::{Context, bail};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, OnceLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone)]
pub struct RuntimeOptions {
    pub root: PathBuf,
    pub threads: usize,
    pub overwrite: bool,
    pub recover_corrupt: bool,
    pub rounds: bool,
    pub round_deadline: Option<Duration>,
}
impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            root: "data/v1".into(),
            threads: std::thread::available_parallelism().map_or(1, usize::from),
            overwrite: false,
            recover_corrupt: false,
            rounds: false,
            round_deadline: None,
        }
    }
}
#[derive(Debug, Default, Clone, Serialize)]
pub struct BatchSummary {
    pub batch_id: String,
    pub completed: usize,
    pub reused: usize,
    pub failed: usize,
    pub cancelled: usize,
    pub not_started: usize,
    pub completed_rounds: usize,
    pub deadline_reached: bool,
}
#[derive(Debug, Clone, Serialize)]
pub struct JobEvent {
    pub condition_id: String,
    pub seed: u64,
    pub status: String,
    pub message: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    pub code: String,
    pub message: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Incomplete {
    pub schema_version: u32,
    pub attempt_id: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub error: Option<Failure>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub partial_result: Option<serde_json::Value>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredGraph {
    schema_version: u32,
    node_count: usize,
    edges: Vec<[usize; 2]>,
    content_hash: String,
}
#[derive(Clone, Serialize)]
pub struct JobInspection {
    pub condition_id: String,
    pub seed: u64,
    pub status: String,
    pub latest_attempt_status: Option<String>,
    pub error: Option<Failure>,
    #[serde(skip)]
    pub result: Option<RunResult>,
    #[serde(skip)]
    pub graph: Option<Arc<Graph>>,
}

pub fn result_path(root: &Path, job: &Job) -> PathBuf {
    root.join("runs")
        .join(&job.condition_id)
        .join(format!("seed_{}.json", job.seed))
}
pub fn incomplete_path(root: &Path, job: &Job) -> PathBuf {
    root.join("runs")
        .join(&job.condition_id)
        .join(format!("seed_{}.incomplete.json", job.seed))
}
pub fn load_experiment(root: &Path, batch: &str) -> Result<StoredExperiment> {
    validate_id(batch)?;
    atomic::read_json(&root.join("batches").join(batch).join("experiment.json"))
}
pub fn load_plan(root: &Path, batch: &str) -> Result<ExperimentPlan> {
    let plan = compile_stored(load_experiment(root, batch)?)?;
    if plan.batch_id != batch {
        bail!("batch ID does not match experiment conditions");
    }
    Ok(plan)
}
pub fn validate_id(id: &str) -> Result<()> {
    if id.len() != 64 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("expected a full 64-character hexadecimal ID");
    }
    Ok(())
}
pub fn read_graph(root: &Path, job: &Job) -> Result<Graph> {
    let stored: StoredGraph =
        atomic::read_json(&root.join("graphs").join(format!("{}.json", job.graph_id)))?;
    if stored.node_count != job.condition.graph.node_count {
        bail!("graph node count does not match conditions");
    }
    if stored.edges.iter().any(|e| e[0] >= e[1]) || stored.edges.windows(2).any(|x| x[0] >= x[1]) {
        bail!("graph edges are not canonical");
    }
    let graph = Graph::from_edges(stored.node_count, stored.edges)?;
    if graph.content_hash() != stored.content_hash {
        bail!("graph content hash mismatch");
    }
    Ok(graph)
}
fn prepare_graph(root: &Path, job: &Job, cancel: &CancellationToken) -> Result<Graph> {
    let path = root.join("graphs").join(format!("{}.json", job.graph_id));
    if path.exists() {
        return read_graph(root, job);
    }
    let graph = Graph::generate(&job.condition.graph, cancel)?;
    atomic::write_json(
        &path,
        &StoredGraph {
            schema_version: 1,
            node_count: graph.node_count,
            edges: graph.edges.clone(),
            content_hash: graph.content_hash(),
        },
        false,
    )?;
    Ok(graph)
}
pub fn read_result(path: &Path, graph: &Graph, condition: &Condition) -> Result<RunResult> {
    let result: RunResult = atomic::read_json(path)?;
    result.validate(graph, condition)?;
    if result.termination == RunTermination::Cancelled {
        bail!("incomplete result in completed result file");
    }
    Ok(result)
}
static ATTEMPT: AtomicU64 = AtomicU64::new(0);
fn attempt_id() -> String {
    format!(
        "{:x}-{:x}-{:x}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        std::process::id(),
        ATTEMPT.fetch_add(1, Ordering::Relaxed)
    )
}
fn partial(result: &RunResult) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(result)?;
    let map = value.as_object_mut().context("result object required")?;
    map.remove("schema_version");
    map.remove("attempt_id");
    map.remove("termination");
    Ok(value)
}
fn restore_partial(
    marker: &Incomplete,
    graph: &Graph,
    condition: &Condition,
) -> Result<Option<RunResult>> {
    let Some(mut value) = marker.partial_result.clone() else {
        return Ok(None);
    };
    let map = value
        .as_object_mut()
        .context("partial result must be an object")?;
    map.insert("schema_version".into(), 1.into());
    map.insert("attempt_id".into(), marker.attempt_id.clone().into());
    map.insert("termination".into(), "cancelled".into());
    let result: RunResult = serde_json::from_value(value)?;
    result.validate(graph, condition)?;
    Ok(Some(result))
}
fn read_marker(path: &Path) -> Result<Option<Incomplete>> {
    if !path.exists() {
        return Ok(None);
    }
    let m: Incomplete = atomic::read_json(path)?;
    if !matches!(m.status.as_str(), "running" | "failed" | "cancelled") || m.attempt_id.is_empty() {
        bail!("invalid incomplete marker");
    }
    Ok(Some(m))
}
pub fn inspect(
    plan: &ExperimentPlan,
    root: &Path,
    include_incomplete: bool,
) -> Result<Vec<JobInspection>> {
    let active = atomic::writer_active(root)?;
    let mut graphs: HashMap<String, Arc<Graph>> = HashMap::new();
    let mut rows = Vec::with_capacity(plan.jobs.len());
    for job in &plan.jobs {
        let marker = read_marker(&incomplete_path(root, job))?;
        let path = result_path(root, job);
        let mut row = JobInspection {
            condition_id: job.condition_id.clone(),
            seed: job.seed,
            status: "not_started".into(),
            latest_attempt_status: None,
            error: None,
            result: None,
            graph: None,
        };
        if path.exists()
            || (include_incomplete && marker.as_ref().is_some_and(|m| m.partial_result.is_some()))
        {
            let graph = if let Some(g) = graphs.get(&job.graph_id) {
                g.clone()
            } else {
                let g = Arc::new(read_graph(root, job)?);
                graphs.insert(job.graph_id.clone(), g.clone());
                g
            };
            if path.exists() {
                row.result = Some(read_result(&path, &graph, &job.condition)?);
                row.status = "completed".into();
            } else if let Some(m) = &marker {
                row.result = restore_partial(m, &graph, &job.condition)?;
            }
            row.graph = Some(graph);
        }
        if let Some(m) = marker {
            let stale = row
                .result
                .as_ref()
                .is_some_and(|r| row.status == "completed" && r.attempt_id == m.attempt_id);
            if !stale {
                let status = if m.status == "running" && !active {
                    "interrupted".to_owned()
                } else {
                    m.status
                };
                if row.status != "completed" {
                    row.status = status.clone();
                }
                row.latest_attempt_status = Some(status);
                row.error = m.error;
            }
        }
        rows.push(row);
    }
    Ok(rows)
}

pub fn run_batch(
    plan: &ExperimentPlan,
    options: &RuntimeOptions,
    cancel: &CancellationToken,
    registry: &FitnessRegistry,
    event_sink: &(dyn Fn(JobEvent) + Sync),
) -> Result<BatchSummary> {
    let started = Instant::now();
    if options.threads == 0 {
        bail!("threads must be >= 1");
    }
    if options.round_deadline.is_some() && !options.rounds {
        bail!("round deadline requires rounds execution");
    }
    let compiled = compile_stored(plan.experiment.clone())?;
    if compiled.batch_id != plan.batch_id || compiled.jobs != plan.jobs {
        bail!("plan does not match stored experiment");
    }
    validate_registry(plan, registry)?;
    let _lock = atomic::WriterLock::acquire(&options.root)?;
    let path = options
        .root
        .join("batches")
        .join(&plan.batch_id)
        .join("experiment.json");
    if path.exists() {
        let old = load_plan(&options.root, &plan.batch_id)?;
        if old.batch_id != plan.batch_id {
            bail!("stored experiment mismatch");
        }
    } else {
        atomic::write_json(&path, &plan.experiment, false)?;
    }
    type GraphCell = Arc<OnceLock<std::result::Result<Arc<Graph>, String>>>;
    let graphs: HashMap<String, GraphCell> = plan
        .jobs
        .iter()
        .map(|j| (j.graph_id.clone(), Arc::new(OnceLock::new())))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.threads)
        .build()?;
    let execute = |job: &Job| {
        if cancel.is_cancelled() {
            return "not_started".into();
        }
        let result = run_job(job, options, cancel, registry, &graphs[&job.graph_id]);
        let (status, message) = match result {
            Ok(s) => (s, None),
            Err(e) => ("failed".into(), Some(format!("{e:#}"))),
        };
        event_sink(JobEvent {
            condition_id: job.condition_id.clone(),
            seed: job.seed,
            status: status.clone(),
            message,
        });
        status
    };
    let mut completed_rounds = 0;
    let mut deadline_reached = false;
    let statuses: Vec<String> = if options.rounds {
        let mut statuses = vec!["not_started".to_owned(); plan.jobs.len()];
        let mut rounds: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (index, job) in plan.jobs.iter().enumerate() {
            rounds.entry(job.seed).or_default().push(index);
        }
        for indices in rounds.values() {
            if cancel.is_cancelled() {
                break;
            }
            if options
                .round_deadline
                .is_some_and(|deadline| started.elapsed() >= deadline)
            {
                deadline_reached = true;
                break;
            }
            let round: Vec<String> = pool.install(|| {
                indices
                    .par_iter()
                    .map(|&index| execute(&plan.jobs[index]))
                    .collect()
            });
            for (&index, status) in indices.iter().zip(round.iter()) {
                statuses[index] = status.clone();
            }
            if round
                .iter()
                .all(|status| matches!(status.as_str(), "completed" | "reused"))
            {
                completed_rounds += 1;
            } else {
                break;
            }
        }
        statuses
    } else {
        pool.install(|| plan.jobs.par_iter().map(execute).collect())
    };
    let mut summary = BatchSummary {
        batch_id: plan.batch_id.clone(),
        completed_rounds,
        deadline_reached,
        ..Default::default()
    };
    for status in statuses {
        match status.as_str() {
            "completed" => summary.completed += 1,
            "reused" => summary.reused += 1,
            "cancelled" => summary.cancelled += 1,
            "not_started" => summary.not_started += 1,
            _ => summary.failed += 1,
        }
    }
    Ok(summary)
}
fn run_job(
    job: &Job,
    options: &RuntimeOptions,
    cancel: &CancellationToken,
    registry: &FitnessRegistry,
    graph_cell: &OnceLock<std::result::Result<Arc<Graph>, String>>,
) -> Result<String> {
    let path = result_path(&options.root, job);
    let marker_path = incomplete_path(&options.root, job);
    let mut marker = Incomplete {
        schema_version: 1,
        attempt_id: attempt_id(),
        status: "running".into(),
        error: None,
        partial_result: None,
    };
    if !path.exists() {
        atomic::write_json(&marker_path, &marker, true)?;
    }
    let graph = graph_cell.get_or_init(|| {
        prepare_graph(&options.root, job, cancel)
            .map(Arc::new)
            .map_err(|e| format!("{e:#}"))
    });
    let computation = (|| -> Result<String> {
        let graph = graph.as_ref().map_err(|e| anyhow::anyhow!("{e}"))?;
        if path.exists() {
            match read_result(&path, graph, &job.condition) {
                Ok(existing) if !options.overwrite => {
                    if let Some(old) = read_marker(&marker_path)?
                        && old.attempt_id == existing.attempt_id
                    {
                        std::fs::remove_file(&marker_path)?;
                    }
                    return Ok("reused".into());
                }
                Ok(_) => {}
                Err(e) => {
                    if format!("{e:#}").contains("unsupported schema")
                        || !(options.recover_corrupt || options.overwrite)
                    {
                        return Err(e);
                    }
                    let backup = path.with_extension(format!("corrupt-{}.json", marker.attempt_id));
                    std::fs::copy(&path, backup)?;
                }
            }
        }
        atomic::write_json(&marker_path, &marker, true)?;
        let mut result = run_one(graph, &job.condition, job.seed, cancel, registry)?;
        result.attempt_id = marker.attempt_id.clone();
        if result.termination == RunTermination::Cancelled {
            marker.status = "cancelled".into();
            marker.partial_result = Some(partial(&result)?);
            atomic::write_json(&marker_path, &marker, true)?;
            return Ok("cancelled".into());
        }
        result.validate(graph, &job.condition)?;
        atomic::write_json(&path, &result, true)?;
        // A durable complete result wins even if marker cleanup is interrupted.
        // The next writer removes a matching stale marker.
        let _ = std::fs::remove_file(&marker_path);
        Ok("completed".into())
    })();
    match computation {
        Ok(s) => Ok(s),
        Err(e) => {
            marker.status = if cancel.is_cancelled() {
                "cancelled"
            } else {
                "failed"
            }
            .into();
            marker.error = Some(Failure {
                code: if cancel.is_cancelled() {
                    "cancelled"
                } else {
                    "execution"
                }
                .into(),
                message: format!("{e:#}"),
            });
            atomic::write_json(&marker_path, &marker, true)
                .with_context(|| format!("original failure: {e:#}"))?;
            if cancel.is_cancelled() {
                Ok("cancelled".into())
            } else {
                Err(e)
            }
        }
    }
}

/// Verify custom fitness availability and semantic versions before any write or reuse.
pub fn validate_registry(plan: &ExperimentPlan, registry: &FitnessRegistry) -> Result<()> {
    let versions = registry.versions();
    for job in &plan.jobs {
        if let crate::experiment::config::SolverSpec::Eo { fitness, .. } = &job.condition.solver {
            registry.validate(fitness)?;
            let expected = plan
                .experiment
                .versions
                .get(&format!("fitness:{}", fitness.kind));
            if expected != versions.get(&fitness.kind) {
                bail!("fitness version mismatch for {}", fitness.kind);
            }
        }
    }
    Ok(())
}
pub fn load_plan_with_registry(
    root: &Path,
    batch: &str,
    registry: &FitnessRegistry,
) -> Result<ExperimentPlan> {
    let plan = load_plan(root, batch)?;
    validate_registry(&plan, registry)?;
    Ok(plan)
}
