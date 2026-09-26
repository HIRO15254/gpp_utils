//! Full-plan setup, export, and small-plan batch benchmark.
//!
//! Every storage root is a fresh temporary directory. Export fixtures use the
//! canonical `runs/<condition_id>/seed_<seed>.json` layout under `--fixtures`
//! (the leading `runs/` may be omitted).

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use gpp_utils::{
    RunResult, RuntimeOptions,
    experiment::plan::{Job, compile_experiment, load_spec},
    export::export_tsv,
    fitness::FitnessRegistry,
    graph_partition::Graph,
    optimization::CancellationToken,
    run_batch,
    storage::{self, atomic},
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    hint::black_box,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    Setup,
    Export,
    Run,
}

#[derive(Parser)]
struct Args {
    /// Source TOML/JSON specification. Setup and export may use the full plan.
    #[arg(long)]
    spec: PathBuf,
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Directory containing canonical per-run JSON fixture paths.
    #[arg(long)]
    fixtures: Option<PathBuf>,
}

#[derive(Serialize)]
struct StoredGraph<'a> {
    schema_version: u32,
    node_count: usize,
    edges: &'a [[usize; 2]],
    content_hash: String,
}

fn emit(args: &Args, phase: &str, rep: usize, elapsed_ns: u128, signature: &str) {
    println!(
        "{}",
        json!({
            "case": args.spec.file_stem().unwrap_or_default().to_string_lossy(),
            "phase": phase,
            "repeat": rep,
            "iterations": 1,
            "elapsed_ns": elapsed_ns,
            "signature": signature,
        })
    );
}

fn fixture_path(root: &Path, job: &Job) -> Option<PathBuf> {
    let relative = PathBuf::from(&job.condition_id).join(format!("seed_{}.json", job.seed));
    [
        root.join("runs").join(&relative),
        root.join(&relative),
        root.join(format!("{}-seed_{}.json", job.condition_id, job.seed)),
    ]
    .into_iter()
    .find(|path| path.is_file())
}

fn write_experiment(root: &Path, plan: &gpp_utils::ExperimentPlan) -> Result<()> {
    atomic::write_json(
        &root
            .join("batches")
            .join(&plan.batch_id)
            .join("experiment.json"),
        &plan.experiment,
        false,
    )
}

fn write_graph(root: &Path, job: &Job, graph: &Graph) -> Result<()> {
    atomic::write_json(
        &root.join("graphs").join(format!("{}.json", job.graph_id)),
        &StoredGraph {
            schema_version: 1,
            node_count: graph.node_count(),
            edges: graph.edges(),
            content_hash: graph.content_hash(),
        },
        false,
    )
}

fn prepare_export_root(
    plan: &gpp_utils::ExperimentPlan,
    fixtures: &Path,
    root: &Path,
    cancel: &CancellationToken,
) -> Result<usize> {
    ensure!(fixtures.is_dir(), "--fixtures must name a directory");
    write_experiment(root, plan)?;
    let mut written_graphs = BTreeSet::new();
    let mut count = 0;
    for job in &plan.jobs {
        let Some(source) = fixture_path(fixtures, job) else {
            continue;
        };
        let graph = Graph::generate(&job.condition.graph, cancel)?;
        let result: RunResult = atomic::read_json(&source)
            .with_context(|| format!("reading fixture {}", source.display()))?;
        result.validate(&graph, &job.condition)?;
        ensure!(
            result.termination != gpp_utils::experiment::result::RunTermination::Cancelled,
            "completed fixture is cancelled: {}",
            source.display()
        );
        if written_graphs.insert(job.graph_id.clone()) {
            write_graph(root, job, &graph)?;
        }
        atomic::write_json_compact(&storage::result_path(root, job), &result, false)?;
        count += 1;
    }
    ensure!(count > 0, "no fixtures matched jobs in the compiled plan");
    Ok(count)
}

fn export_signature(out: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(std::fs::read(out.join("runs.tsv"))?);
    hash.update(std::fs::read(out.join("traces.tsv"))?);
    let mut metadata: Value = atomic::read_json(&out.join("metadata.json"))?;
    metadata
        .as_object_mut()
        .context("metadata must be an object")?
        .remove("created_unix_ms");
    hash.update(serde_json::to_vec(&metadata)?);
    Ok(format!("{:x}", hash.finalize()))
}

fn normalize_result(value: &mut Value) {
    if let Value::Object(map) = value {
        map.remove("attempt_id");
        map.remove("elapsed_ms");
        if let Some(Value::Object(diagnostics)) = map.get_mut("diagnostics") {
            diagnostics.remove("search_ms");
            diagnostics.remove("measurement_ms");
        }
    }
}

fn batch_signature(
    plan: &gpp_utils::ExperimentPlan,
    root: &Path,
    completed: usize,
) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update((completed as u64).to_le_bytes());
    for job in &plan.jobs {
        let mut value: Value = atomic::read_json(&storage::result_path(root, job))?;
        normalize_result(&mut value);
        hash.update(job.condition_id.as_bytes());
        hash.update(job.seed.to_le_bytes());
        hash.update(serde_json::to_vec(&value)?);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.repeats > 0, "--repeats must be positive");
    ensure!(args.threads > 0, "--threads must be positive");
    let plan = compile_experiment(load_spec(&args.spec)?)?;
    let cancel = CancellationToken::new();
    let registry = FitnessRegistry::default();
    let mut expected = None;

    for iteration in 0..args.warmup + args.repeats {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("data");
        let (phase, elapsed_ns, signature) = match args.mode {
            Mode::Setup => {
                let options = RuntimeOptions {
                    root,
                    threads: args.threads,
                    rounds: true,
                    round_deadline: Some(Duration::ZERO),
                    ..Default::default()
                };
                let started = Instant::now();
                let summary = run_batch(&plan, &options, &cancel, &registry, &|_| {})?;
                let elapsed = started.elapsed().as_nanos();
                ensure!(
                    summary.not_started == plan.jobs.len()
                        && summary.completed == 0
                        && summary.reused == 0
                        && summary.failed == 0
                        && summary.cancelled == 0
                        && summary.completed_rounds == 0
                        && summary.deadline_reached,
                    "zero-deadline setup executed a job: {:?}",
                    summary
                );
                (
                    "setup",
                    elapsed,
                    format!("{}:{}", plan.batch_id, summary.not_started),
                )
            }
            Mode::Export => {
                let fixtures = args
                    .fixtures
                    .as_deref()
                    .context("export mode requires --fixtures DIR")?;
                let fixture_count = prepare_export_root(&plan, fixtures, &root, &cancel)?;
                let out = temp.path().join("export");
                let started = Instant::now();
                let summary = export_tsv(&plan, &root, &out, false, false)?;
                let elapsed = started.elapsed().as_nanos();
                ensure!(summary.jobs == plan.jobs.len(), "export omitted plan jobs");
                let signature = export_signature(&out)?;
                black_box((fixture_count, summary.trace_rows));
                ("export", elapsed, signature)
            }
            Mode::Run => {
                let options = RuntimeOptions {
                    root: root.clone(),
                    threads: args.threads,
                    ..Default::default()
                };
                let started = Instant::now();
                let summary = run_batch(&plan, &options, &cancel, &registry, &|_| {})?;
                let elapsed = started.elapsed().as_nanos();
                ensure!(
                    summary.completed == plan.jobs.len()
                        && summary.reused == 0
                        && summary.failed == 0
                        && summary.cancelled == 0
                        && summary.not_started == 0,
                    "batch run did not complete every job: {:?}",
                    summary
                );
                let signature = batch_signature(&plan, &root, summary.completed)?;
                ("run", elapsed, signature)
            }
        };
        if let Some(previous) = &expected {
            ensure!(previous == &signature, "unstable {phase} signature");
        }
        expected = Some(signature.clone());
        if iteration >= args.warmup {
            emit(
                &args,
                phase,
                iteration - args.warmup,
                elapsed_ns,
                &signature,
            );
        }
    }
    Ok(())
}
