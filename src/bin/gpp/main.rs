use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use gpp_utils::{ExperimentPlan, StoredExperiment};
use gpp_utils::{
    experiment::{
        plan::{
            compile_experiment_with_versions, compile_stored_with_registry, load_spec,
            minimal_sample_toml,
        },
        result::RunView,
    },
    fitness::FitnessRegistry,
    optimization::CancellationToken,
    storage::{self, RuntimeOptions},
};
use serde::Serialize;

#[derive(Parser)]
#[command(
    name = "gpp",
    about = "Reproducible graph partition experiments",
    after_help = "SA and EO-SA (eo_sa) use a fixed temperature."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Init {
        path: PathBuf,
    },
    Validate {
        path: PathBuf,
        #[arg(long)]
        json: bool,
    },
    Plan {
        path: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    Run {
        #[arg(value_name = "CONFIG", required_unless_present = "experiment")]
        path: Option<PathBuf>,
        #[arg(long, conflicts_with = "path")]
        experiment: Option<PathBuf>,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[arg(long)]
        json: bool,
    },
    Resume {
        #[arg(long)]
        batch: String,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[arg(long)]
        json: bool,
    },
    Inspect {
        #[arg(long)]
        batch: String,
        #[arg(long, default_value = "data/v1")]
        root: PathBuf,
        #[arg(long)]
        condition: Option<String>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        json: bool,
    },
    Export {
        #[arg(long)]
        batch: String,
        #[arg(long, default_value = "data/v1")]
        root: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        condition: Option<String>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        include_incomplete: bool,
        #[arg(long)]
        overwrite: bool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(clap::Args)]
struct RuntimeArgs {
    #[arg(long, default_value = "data/v1")]
    root: PathBuf,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long)]
    overwrite: bool,
    /// Finish all conditions for each search seed before starting the next seed.
    #[arg(long)]
    rounds: bool,
    /// Soft time limit checked before each round; the active round finishes.
    #[arg(long, value_name = "SECONDS", requires = "rounds")]
    deadline_seconds: Option<u64>,
}

fn main() {
    let cli = Cli::parse();
    std::process::exit(match run(cli) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            if error.downcast_ref::<InputError>().is_some() {
                2
            } else {
                1
            }
        }
    });
}

#[derive(Debug)]
struct InputError;
impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("invalid input")
    }
}
impl std::error::Error for InputError {}

fn run(cli: Cli) -> anyhow::Result<i32> {
    match cli.command {
        Command::Init { path } => init(&path),
        Command::Validate { path, json } => {
            let plan = compile_path(&path)?;
            output(
                &serde_json::json!({"valid":true,"batch_id":plan.batch_id,"jobs":plan.jobs.len()}),
                json,
            );
            Ok(0)
        }
        Command::Plan { path, out, json } => {
            let plan = compile_path(&path)?;
            if let Some(out) = out {
                storage::atomic::write_json(&out, &plan.experiment, false)?;
            }
            output(&plan, json);
            Ok(0)
        }
        Command::Run {
            path,
            experiment,
            runtime,
            json,
        } => {
            let plan = match (path, experiment) {
                (Some(path), None) => compile_path(&path)?,
                (None, Some(path)) => compile_stored_with_registry(
                    read_stored(&path)?,
                    &FitnessRegistry::default_registry(),
                )
                .map_err(|error| error.context(InputError))?,
                _ => bail!("provide either CONFIG or --experiment"),
            };
            run_plan(plan, runtime, json, false)
        }
        Command::Resume {
            batch,
            runtime,
            json,
        } => {
            let plan = storage::load_plan(&runtime.root, &batch)?;
            run_plan(plan, runtime, json, true)
        }
        Command::Inspect {
            batch,
            root,
            condition,
            seed,
            json,
        } => inspect(&batch, &root, condition.as_deref(), seed, json),
        Command::Export {
            batch,
            root,
            out,
            condition,
            seed,
            include_incomplete,
            overwrite,
            json,
        } => export(
            &batch,
            &root,
            &out,
            condition.as_deref(),
            seed,
            include_incomplete,
            overwrite,
            json,
        ),
    }
}

fn init(path: &Path) -> anyhow::Result<i32> {
    if path.exists() {
        bail!("refusing to overwrite {}", path.display());
    }
    let data = if path.extension().and_then(|x| x.to_str()) == Some("json") {
        serde_json::to_string_pretty(&toml::from_str::<gpp_utils::ExperimentSpec>(
            minimal_sample_toml(),
        )?)?
    } else {
        minimal_sample_toml().to_owned()
    };
    std::fs::write(path, data).with_context(|| format!("write {}", path.display()))?;
    println!("created {}", path.display());
    Ok(0)
}

fn compile_path(path: &Path) -> anyhow::Result<ExperimentPlan> {
    let registry = FitnessRegistry::default_registry();
    (|| -> anyhow::Result<ExperimentPlan> {
        let plan = compile_experiment_with_versions(load_spec(path)?, &registry.versions())?;
        validate_fitness(&plan, &registry)?;
        Ok(plan)
    })()
    .map_err(|error| error.context(InputError))
}

fn read_stored(path: &Path) -> anyhow::Result<StoredExperiment> {
    (|| -> anyhow::Result<StoredExperiment> {
        Ok(serde_json::from_str(
            &std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
        )?)
    })()
    .map_err(|error| error.context(InputError))
}

fn validate_fitness(plan: &ExperimentPlan, registry: &FitnessRegistry) -> anyhow::Result<()> {
    for job in &plan.jobs {
        if let Some(fitness) = job.condition.solver.fitness() {
            registry.validate(fitness)?;
        }
    }
    Ok(())
}

fn cancellation() -> anyhow::Result<CancellationToken> {
    let token = CancellationToken::new();
    let count = Arc::new(AtomicUsize::new(0));
    let signal = token.clone();
    ctrlc::set_handler(move || {
        if count.fetch_add(1, Ordering::SeqCst) == 0 {
            signal.cancel();
            eprintln!("cancelling; press Ctrl+C again to exit immediately");
        } else {
            std::process::exit(130);
        }
    })?;
    Ok(token)
}

fn run_plan(
    plan: ExperimentPlan,
    runtime: RuntimeArgs,
    json: bool,
    recover_corrupt: bool,
) -> anyhow::Result<i32> {
    let registry = FitnessRegistry::default_registry();
    validate_fitness(&plan, &registry)?;
    let options = RuntimeOptions {
        root: runtime.root,
        threads: runtime
            .threads
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from)),
        overwrite: runtime.overwrite,
        recover_corrupt,
        rounds: runtime.rounds,
        round_deadline: runtime.deadline_seconds.map(std::time::Duration::from_secs),
    };
    if options.threads == 0 {
        return Err(anyhow::Error::new(InputError));
    }
    let token = cancellation()?;
    let summary = storage::run_batch(&plan, &options, &token, &registry, &|event| {
        if !json {
            eprintln!(
                "{} seed={} {}",
                event.condition_id, event.seed, event.status
            );
        }
    })?;
    output(&summary, json);
    Ok(if token.is_cancelled() || summary.cancelled > 0 {
        130
    } else if summary.failed > 0 {
        1
    } else if summary.not_started > 0 && !summary.deadline_reached {
        130
    } else {
        0
    })
}

#[derive(Serialize)]
struct InspectRow<'a> {
    condition_id: &'a str,
    seed: u64,
    status: &'a str,
    condition: &'a gpp_utils::experiment::config::Condition,
    final_score: Option<f64>,
    best_score: Option<f64>,
    latest_attempt_status: Option<&'a str>,
    error: Option<&'a gpp_utils::storage::Failure>,
    termination: Option<gpp_utils::experiment::result::RunTermination>,
    completed_steps: Option<u64>,
    budget_max_steps: u64,
}
#[derive(Serialize)]
struct InspectOutput<'a> {
    jobs: Vec<InspectRow<'a>>,
    counts: std::collections::BTreeMap<String, usize>,
}
fn inspect(
    batch: &str,
    root: &Path,
    condition: Option<&str>,
    seed: Option<u64>,
    json: bool,
) -> anyhow::Result<i32> {
    let plan = storage::load_plan(root, batch)?;
    let rows = storage::inspect(&plan, root, true)?;
    let display: Vec<_> = plan
        .jobs
        .iter()
        .zip(rows.iter())
        .filter(|(job, _)| {
            condition.is_none_or(|id| job.condition_id == id) && seed.is_none_or(|s| job.seed == s)
        })
        .map(|(job, row)| -> anyhow::Result<_> {
            let scores = row
                .result
                .as_ref()
                .zip(row.graph.as_deref())
                .map(|(result, graph)| {
                    RunView::new(graph, &job.condition, result)
                        .map(|view| (view.final_score(), view.best_score()))
                })
                .transpose()?;
            Ok(InspectRow {
                condition_id: &job.condition_id,
                seed: job.seed,
                status: &row.status,
                condition: &job.condition,
                final_score: scores.map(|x| x.0),
                best_score: scores.map(|x| x.1),
                latest_attempt_status: row.latest_attempt_status.as_deref(),
                error: row.error.as_ref(),
                termination: row.result.as_ref().map(|result| result.termination),
                completed_steps: row.result.as_ref().map(|result| result.completed_steps),
                budget_max_steps: job.condition.budget.max_steps,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut counts = std::collections::BTreeMap::new();
    for row in &display {
        *counts.entry(row.status.to_owned()).or_insert(0) += 1;
    }
    output(
        &InspectOutput {
            jobs: display,
            counts,
        },
        json,
    );
    Ok(0)
}

#[allow(clippy::too_many_arguments)]
fn export(
    batch: &str,
    root: &Path,
    out: &Path,
    condition: Option<&str>,
    seed: Option<u64>,
    include_incomplete: bool,
    overwrite: bool,
    json: bool,
) -> anyhow::Result<i32> {
    let mut plan = storage::load_plan(root, batch)?;
    plan.jobs.retain(|job| {
        condition.is_none_or(|id| job.condition_id == id) && seed.is_none_or(|s| job.seed == s)
    });
    if plan.jobs.is_empty() {
        bail!("no jobs match the requested filter");
    }
    let summary = gpp_utils::export::export_tsv(&plan, root, out, include_incomplete, overwrite)?;
    output(&summary, json);
    Ok(0)
}

fn output<T: Serialize>(value: &T, json: bool) {
    if json {
        println!(
            "{}",
            serde_json::to_string(value).expect("serializable output")
        );
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(value).expect("serializable output")
        );
    }
}
