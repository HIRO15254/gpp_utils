//! Same-input performance comparisons for recovery-v1 conditions.
//! See scripts/performance_cases.py and docs/performance-worklog.md.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use gpp_utils::{
    RunResult, RuntimeOptions,
    experiment::{
        config::{Neighborhood, SmoothingSpec},
        plan::{compile_experiment, load_spec},
        result::RunView,
    },
    export::export_tsv,
    fitness::FitnessRegistry,
    graph_partition::{Graph, PartitionState},
    optimization::{CancellationToken, rng_for},
    run_batch, run_one, smoothing,
    storage::{self, atomic},
};
use rand::{Rng, seq::SliceRandom};
use rand_mt::Mt19937GenRand64;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{hint::black_box, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    spec: PathBuf,
    #[arg(long, default_value = "run")]
    mode: String,
    #[arg(long)]
    phase: Option<String>,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 1000)]
    iterations: usize,
    /// Save a measured run, or read that fixed run in io mode.
    #[arg(long)]
    fixture: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    threads: usize,
}

fn signature(result: &RunResult) -> Result<String> {
    let mut copy = result.clone();
    copy.attempt_id.clear();
    copy.elapsed_ms = 0.0;
    if let Some(d) = &mut copy.diagnostics {
        d.search_ms = 0.0;
        d.measurement_ms = 0.0;
    }
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(&copy)?)))
}

fn emit(args: &Args, phase: &str, rep: usize, iterations: usize, ns: u128, sig: &str) {
    println!(
        "{}",
        json!({"case": args.spec.file_stem().unwrap().to_string_lossy(),
        "phase": phase, "repeat": rep, "iterations": iterations,
        "elapsed_ns": ns, "signature": sig})
    );
}

fn measure(
    args: &Args,
    phase: &str,
    iterations: usize,
    mut operation: impl FnMut() -> Result<u64>,
) -> Result<()> {
    if args.phase.as_deref().is_some_and(|filter| filter != phase) {
        return Ok(());
    }
    let mut expected = None;
    for rep in 0..args.warmup + args.repeats {
        let started = Instant::now();
        let mut check = 0u64;
        for _ in 0..iterations {
            check = check.rotate_left(7) ^ black_box(operation()?);
        }
        let elapsed = started.elapsed().as_nanos();
        if let Some(previous) = expected {
            ensure!(previous == check, "unstable {phase} signature");
        }
        expected = Some(check);
        if rep >= args.warmup {
            emit(
                args,
                phase,
                rep - args.warmup,
                iterations,
                elapsed,
                &format!("{check:016x}"),
            );
        }
    }
    Ok(())
}

fn initial(graph: &Graph, neighborhood: Neighborhood, seed: u64) -> Result<PartitionState> {
    let hash = graph.content_hash();
    let label = match neighborhood {
        Neighborhood::Flip => b"flip".as_slice(),
        Neighborhood::Swap => b"swap".as_slice(),
    };
    let mut rng = rng_for(&[hash.as_bytes(), label, &seed.to_le_bytes(), b"initial"]);
    let mut partition: Vec<bool> = (0..graph.node_count()).map(|_| rng.r#gen()).collect();
    if neighborhood == Neighborhood::Swap {
        partition.fill(false);
        partition[..graph.node_count() / 2].fill(true);
        partition.shuffle(&mut rng);
    }
    PartitionState::new(graph, partition)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.repeats > 0 && args.iterations > 0,
        "positive repetition counts required"
    );
    let plan = compile_experiment(load_spec(&args.spec)?)?;
    ensure!(
        plan.jobs.len() == 1,
        "expected one unchanged recovery condition"
    );
    let job = &plan.jobs[0];
    let c = &job.condition;
    let token = CancellationToken::new();
    let graph = Graph::generate(&c.graph, &token)?;
    let registry = FitnessRegistry::default();
    eprintln!(
        "{} {} seed={} steps={}",
        args.spec.display(),
        job.condition_id,
        job.seed,
        c.budget.max_steps
    );
    match args.mode.as_str() {
        "run" => {
            let mut expected = None;
            for rep in 0..args.warmup + args.repeats {
                let start = Instant::now();
                let result = run_one(&graph, c, job.seed, &token, &registry)?;
                let elapsed = start.elapsed().as_nanos();
                let sig = signature(&result)?;
                if let Some(previous) = &expected {
                    ensure!(previous == &sig, "unstable result signature");
                }
                expected = Some(sig.clone());
                if rep >= args.warmup {
                    emit(&args, "run", rep - args.warmup, 1, elapsed, &sig);
                    if let Some(path) = &args.fixture {
                        atomic::write_json_compact(path, &result, true)?;
                    }
                }
            }
        }
        "kernel" => {
            let state = initial(&graph, c.neighborhood, job.seed)?;
            let count = args.iterations;
            measure(&args, "checkpoints", count, || {
                let points = gpp_utils::experiment::measurement::checkpoints(
                    black_box(&c.measurement),
                    c.budget.max_steps,
                );
                Ok(black_box(points).len() as u64)
            })?;
            measure(&args, "hash", count, || {
                let hash = black_box(&graph).content_hash();
                Ok(u64::from_le_bytes(hash.as_bytes()[..8].try_into()?))
            })?;
            measure(&args, "move_list", count, || {
                let moves = smoothing::moves(black_box(&state), c.neighborhood);
                Ok(black_box(moves).len() as u64)
            })?;
            measure(&args, "move_scores", (count / 100).max(1), || {
                let mut check = 0u64;
                for mv in smoothing::moves(&state, c.neighborhood) {
                    check = check.rotate_left(7)
                        ^ smoothing::move_score(black_box(&state), &graph, mv, c.alpha).to_bits();
                }
                Ok(check)
            })?;
            let first_count = match c.neighborhood {
                Neighborhood::Flip => graph.node_count(),
                Neighborhood::Swap => graph.node_count() * graph.node_count() / 4,
            };
            for (name, spec) in [
                ("smooth_all", SmoothingSpec::AllAverage),
                ("smooth_k1", SmoothingSpec::RandomKAverage { k: 1 }),
                ("smooth_k32", SmoothingSpec::RandomKAverage { k: 32 }),
                (
                    "smooth_distance2",
                    SmoothingSpec::RandomKAverage { k: first_count + 4 },
                ),
            ] {
                measure(&args, name, (count / 10).max(1), || {
                    let mut rng = Mt19937GenRand64::new(job.seed);
                    let mut evaluations = 0;
                    let score = smoothing::evaluate(
                        black_box(&state),
                        &graph,
                        c.alpha,
                        c.neighborhood,
                        &spec,
                        Some(&mut rng),
                        &token,
                        &mut evaluations,
                    )?;
                    Ok(score.to_bits() ^ evaluations.rotate_left(3) ^ rng.next_u64())
                })?;
            }
        }
        "io" => {
            let path = args.fixture.as_ref().context("io mode needs --fixture")?;
            let result = storage::read_result(path, &graph, c)?;
            let count = (args.iterations / 10).max(1);
            measure(&args, "validate", count, || {
                result.validate(black_box(&graph), c)?;
                Ok(result.records.len() as u64)
            })?;
            measure(&args, "read_json", count, || {
                let result: RunResult = atomic::read_json(black_box(path))?;
                Ok(result.records.len() as u64)
            })?;
            measure(&args, "read_result", count, || {
                let result = storage::read_result(black_box(path), &graph, c)?;
                Ok(result.records.len() as u64)
            })?;
            measure(&args, "run_view", count, || {
                let view = RunView::new(&graph, c, &result)?;
                let mut check = 0u64;
                for m in view.records() {
                    check = check.rotate_left(7) ^ m.current_breakdown().real.to_bits();
                    check = check.rotate_left(7) ^ m.best_breakdown().real.to_bits();
                    check ^= m.smoothed_value().unwrap_or_default().to_bits();
                    check ^= m.search_evaluation().unwrap_or_default().to_bits();
                }
                Ok(check)
            })?;
            let temp = tempfile::tempdir()?;
            let root = temp.path().join("source");
            atomic::write_json(
                &root
                    .join("batches")
                    .join(&plan.batch_id)
                    .join("experiment.json"),
                &plan.experiment,
                false,
            )?;
            atomic::write_json(
                &root.join("graphs").join(format!("{}.json", job.graph_id)),
                &json!({"schema_version": 1, "node_count": graph.node_count(),
                    "edges": graph.edges(), "content_hash": graph.content_hash()}),
                false,
            )?;
            atomic::write_json_compact(&storage::result_path(&root, job), &result, false)?;
            measure(&args, "reuse", (count / 10).max(1), || {
                let options = RuntimeOptions {
                    root: root.clone(),
                    threads: args.threads,
                    rounds: true,
                    ..Default::default()
                };
                let summary = run_batch(&plan, &options, &token, &registry, &|_| {})?;
                ensure!(
                    summary.reused == 1 && summary.completed == 0,
                    "reuse changed"
                );
                Ok(summary.reused as u64)
            })?;
            let out = temp.path().join("export");
            measure(&args, "export", (count / 10).max(1), || {
                let summary = export_tsv(&plan, &root, &out, false, true)?;
                Ok(summary.trace_rows as u64)
            })?;
            // Content signatures are outside the timed export section.
            if out.exists() {
                let mut hash = Sha256::new();
                for name in ["runs.tsv", "traces.tsv"] {
                    hash.update(std::fs::read(out.join(name))?);
                }
                let mut metadata: serde_json::Value =
                    atomic::read_json(&out.join("metadata.json"))?;
                metadata.as_object_mut().unwrap().remove("created_unix_ms");
                hash.update(serde_json::to_vec(&metadata)?);
                emit(
                    &args,
                    "export_content",
                    0,
                    1,
                    0,
                    &format!("{:x}", hash.finalize()),
                );
            }
        }
        _ => anyhow::bail!("mode must be run, kernel, or io"),
    }
    Ok(())
}
