//! Immutable experiment inputs and atomic result storage.
pub mod atomic;
use crate::{
    error::{Result, UnsupportedSchema},
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
            node_count: graph.node_count(),
            edges: graph.edges().to_vec(),
            content_hash: graph.content_hash(),
        },
        false,
    )?;
    Ok(graph)
}
pub fn read_result(path: &Path, graph: &Graph, condition: &Condition) -> Result<RunResult> {
    let result: RunResult = atomic::read_run_result(path)?;
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
enum MarkerState {
    Missing,
    Valid(Incomplete),
    Invalid(anyhow::Error),
    Unsupported(anyhow::Error),
}
fn read_marker(path: &Path) -> Result<MarkerState> {
    let m: Incomplete = match atomic::read_json(path) {
        Ok(m) => m,
        Err(e) if e.is::<UnsupportedSchema>() => return Ok(MarkerState::Unsupported(e)),
        Err(e)
            if e.root_cause()
                .downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
        {
            return Ok(MarkerState::Missing);
        }
        Err(e) if e.root_cause().is::<std::io::Error>() => return Err(e),
        Err(e) => return Ok(MarkerState::Invalid(e)),
    };
    if !matches!(m.status.as_str(), "running" | "failed" | "cancelled") || m.attempt_id.is_empty() {
        return Ok(MarkerState::Invalid(anyhow::anyhow!(
            "invalid incomplete marker in {}",
            path.display()
        )));
    }
    Ok(MarkerState::Valid(m))
}
fn file_exists(path: &Path) -> Result<bool> {
    match std::fs::metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}
fn prepare_marker_for_write(
    state: MarkerState,
    path: &Path,
    options: &RuntimeOptions,
    attempt: &str,
) -> Result<()> {
    match state {
        MarkerState::Missing | MarkerState::Valid(_) => Ok(()),
        MarkerState::Unsupported(e) => Err(e),
        MarkerState::Invalid(e) if options.recover_corrupt || options.overwrite => {
            let backup = path.with_extension(format!("corrupt-{attempt}.json"));
            std::fs::copy(path, &backup)
                .with_context(|| format!("backing up invalid marker: {e:#}"))?;
            Ok(())
        }
        MarkerState::Invalid(e) => Err(e),
    }
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
        rows.push(inspect_job(
            job,
            root,
            include_incomplete,
            active,
            &mut graphs,
        )?);
    }
    Ok(rows)
}

/// Inspect one job while reusing graph instances. Results are validated before
/// return, allowing streaming consumers to drop each result after writing it.
pub(crate) fn inspect_job(
    job: &Job,
    root: &Path,
    include_incomplete: bool,
    active: bool,
    graphs: &mut HashMap<String, Arc<Graph>>,
) -> Result<JobInspection> {
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
    if file_exists(&path)?
        || (include_incomplete
            && matches!(&marker, MarkerState::Valid(m) if m.partial_result.is_some()))
    {
        let graph = if let Some(g) = graphs.get(&job.graph_id) {
            g.clone()
        } else {
            let g = Arc::new(read_graph(root, job)?);
            graphs.insert(job.graph_id.clone(), g.clone());
            g
        };
        if file_exists(&path)? {
            row.result = Some(read_result(&path, &graph, &job.condition)?);
            row.status = "completed".into();
        } else if let MarkerState::Valid(m) = &marker {
            row.result = restore_partial(m, &graph, &job.condition)?;
        }
        row.graph = Some(graph);
    }
    if let MarkerState::Valid(m) = marker {
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
    } else if let MarkerState::Invalid(e) | MarkerState::Unsupported(e) = marker {
        row.latest_attempt_status = Some("invalid_marker".into());
        if row.status != "completed" {
            row.status = "invalid_marker".into();
        }
        row.error = Some(Failure {
            code: "marker_issue".into(),
            message: format!("{e:#}"),
        });
    }
    Ok(row)
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
    let mut graphs: HashMap<String, GraphCell> = HashMap::new();
    for job in &plan.jobs {
        graphs
            .entry(job.graph_id.clone())
            .or_insert_with(|| Arc::new(OnceLock::new()));
    }
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
    let existing_path = file_exists(&path)?;
    let mut marker_state = Some(read_marker(&marker_path)?);
    let mut marker_written = false;
    if !existing_path {
        prepare_marker_for_write(
            marker_state.take().expect("marker classified"),
            &marker_path,
            options,
            &marker.attempt_id,
        )?;
        atomic::write_json_compact(&marker_path, &marker, true)?;
        marker_written = true;
    }
    let graph = graph_cell.get_or_init(|| {
        prepare_graph(&options.root, job, cancel)
            .map(Arc::new)
            .map_err(|e| format!("{e:#}"))
    });
    let computation = (|| -> Result<String> {
        let graph = graph.as_ref().map_err(|e| anyhow::anyhow!("{e}"))?;
        if existing_path {
            match read_result(&path, graph, &job.condition) {
                Ok(existing) if !options.overwrite => {
                    if let Some(MarkerState::Valid(old)) = &marker_state
                        && old.attempt_id == existing.attempt_id
                    {
                        std::fs::remove_file(&marker_path)?;
                    }
                    return Ok("reused".into());
                }
                Ok(_) => {}
                Err(e) => {
                    if e.is::<UnsupportedSchema>()
                        || e.root_cause().is::<std::io::Error>()
                        || !(options.recover_corrupt || options.overwrite)
                    {
                        return Err(e);
                    }
                    let backup = path.with_extension(format!("corrupt-{}.json", marker.attempt_id));
                    std::fs::copy(&path, backup)?;
                }
            }
        }
        if !marker_written {
            prepare_marker_for_write(
                marker_state.take().expect("marker classified"),
                &marker_path,
                options,
                &marker.attempt_id,
            )?;
            atomic::write_json_compact(&marker_path, &marker, true)?;
            marker_written = true;
        }
        let mut result = run_one(graph, &job.condition, job.seed, cancel, registry)?;
        result.attempt_id = marker.attempt_id.clone();
        if result.termination == RunTermination::Cancelled {
            marker.status = "cancelled".into();
            marker.partial_result = Some(partial(&result)?);
            atomic::write_json_compact(&marker_path, &marker, true)?;
            return Ok("cancelled".into());
        }
        // `run_one` validates before returning. Replacing `attempt_id` preserves
        // every validated invariant and the generated ID is always non-empty.
        debug_assert!(!result.attempt_id.is_empty());
        atomic::write_json_compact(&path, &result, true)?;
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
            if marker_written {
                atomic::write_json_compact(&marker_path, &marker, true)
                    .with_context(|| format!("original failure: {e:#}"))?;
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::config::{
        BasinMode, Budget, GraphKind, GraphSpec, Measurement, Neighborhood, Schedule,
        SmoothingSpec, SolverSpec,
    };

    fn condition(node_count: usize, neighborhood: Neighborhood) -> Condition {
        Condition {
            graph: GraphSpec {
                kind: GraphKind::Random,
                node_count,
                expected_degree: 3.0_f64.min((node_count - 1) as f64),
                seed: 17,
            },
            neighborhood,
            alpha: 0.05,
            solver: SolverSpec::Sa {
                temperature: 1.0,
                smoothing: SmoothingSpec::None,
            },
            budget: Budget { max_steps: 300 },
            measurement: Measurement {
                schedule: Schedule::Logarithmic,
                steps: vec![],
                basin: BasinMode::Real,
                max_basin_steps: 50,
                diagnostics: true,
                best_basin: true,
            },
        }
    }

    fn without_marker_fields(mut value: serde_json::Value) -> serde_json::Value {
        let map = value.as_object_mut().unwrap();
        for key in ["schema_version", "attempt_id", "termination"] {
            map.remove(key);
        }
        value
    }

    #[test]
    fn compact_marker_partial_result_round_trips_through_restore_partial() {
        let temp = tempfile::tempdir().unwrap();
        for (node_count, neighborhood) in [
            (2, Neighborhood::Flip),
            (8, Neighborhood::Swap),
            (13, Neighborhood::Flip),
            (123, Neighborhood::Flip),
            (124, Neighborhood::Swap),
        ] {
            let condition = condition(node_count, neighborhood);
            let graph = Graph::generate(&condition.graph, &CancellationToken::new()).unwrap();
            let result = run_one(
                &graph,
                &condition,
                5,
                &CancellationToken::new(),
                &FitnessRegistry::default(),
            )
            .unwrap();
            assert!(result.partitions.len() > 1);
            let marker = Incomplete {
                schema_version: 1,
                attempt_id: "attempt-partial".into(),
                status: "cancelled".into(),
                error: None,
                partial_result: Some(partial(&result).unwrap()),
            };
            let path = temp
                .path()
                .join(format!("seed_{node_count}.incomplete.json"));
            atomic::write_json_compact(&path, &marker, true).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(bytes.iter().filter(|&&b| b == b'\n').count(), 1);
            assert_eq!(bytes.last(), Some(&b'\n'));
            assert_eq!(
                &bytes[..bytes.len() - 1],
                serde_json::to_vec(&marker).unwrap()
            );

            let MarkerState::Valid(read) = read_marker(&path).unwrap() else {
                panic!("marker must stay valid");
            };
            let stored = read.partial_result.as_ref().unwrap();
            assert_eq!(stored["partitions"]["length"], node_count);
            let hex = stored["partitions"]["hex"].as_array().unwrap();
            assert_eq!(hex.len(), result.partitions.len());
            assert!(
                hex.iter()
                    .all(|s| s.as_str().unwrap().len() == 2 * node_count.div_ceil(8))
            );

            let restored = restore_partial(&read, &graph, &condition)
                .unwrap()
                .expect("partial result present");
            assert_eq!(restored.termination, RunTermination::Cancelled);
            assert_eq!(restored.attempt_id, "attempt-partial");
            assert_eq!(restored.partitions, result.partitions);
            assert_eq!(
                (
                    restored.completed_steps,
                    restored.final_solution,
                    restored.best_solution,
                    restored.best_step
                ),
                (
                    result.completed_steps,
                    result.final_solution,
                    result.best_solution,
                    result.best_step
                )
            );
            // Floats go through the same JSON text in both values, so this
            // comparison is independent of serde_json's parsing precision.
            let reparsed: serde_json::Value =
                serde_json::from_slice(&serde_json::to_vec(&result).unwrap()).unwrap();
            assert_eq!(
                without_marker_fields(serde_json::to_value(&restored).unwrap()),
                without_marker_fields(reparsed)
            );
            assert_eq!(partial(&restored).unwrap(), *stored);
        }
    }

    #[test]
    fn restore_partial_rejects_malformed_packed_partitions() {
        let condition = condition(13, Neighborhood::Flip);
        let graph = Graph::generate(&condition.graph, &CancellationToken::new()).unwrap();
        let result = run_one(
            &graph,
            &condition,
            9,
            &CancellationToken::new(),
            &FitnessRegistry::default(),
        )
        .unwrap();
        let base = partial(&result).unwrap();
        type Edit = fn(&mut serde_json::Value);
        let edits: [(Edit, &str); 7] = [
            (
                |p| p["hex"][0] = "0A00".into(),
                "uppercase hexadecimal digit",
            ),
            (
                |p| p["hex"][0] = "0a1f00".into(),
                "expected 4 hexadecimal digits for length 13, found 6",
            ),
            (|p| p["hex"][0] = "ffff".into(), "non-zero padding bits"),
            // Every string stays a valid 14-vertex encoding, so only the
            // graph-aware validation can reject the pool.
            (|p| p["length"] = 14.into(), "invalid partition length"),
            (
                |p| p["bits"] = serde_json::json!([]),
                "unknown field `bits`",
            ),
            (
                |p| {
                    p.as_object_mut().unwrap().remove("length");
                },
                "missing field `length`",
            ),
            (|p| *p = serde_json::json!([vec![true; 13]]), "invalid type"),
        ];
        for (edit, message) in edits {
            let mut partial_result = base.clone();
            edit(&mut partial_result["partitions"]);
            let marker = Incomplete {
                schema_version: 1,
                attempt_id: "attempt".into(),
                status: "cancelled".into(),
                error: None,
                partial_result: Some(partial_result),
            };
            let error = restore_partial(&marker, &graph, &condition).unwrap_err();
            assert!(
                format!("{error:#}").contains(message),
                "{message}: {error:#}"
            );
        }
    }
}
