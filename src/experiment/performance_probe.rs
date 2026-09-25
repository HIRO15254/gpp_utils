//! Ignored, manual-only basin kernel timing derived from a compiled experiment.
//!
//! `GPP_PERF_SPEC` names an existing JSON/TOML experiment subset. The probe
//! compiles it normally, preserves each job's graph, alpha, neighborhood, seed,
//! and initial-partition RNG order, then invokes the private basin kernel with
//! synthetic smoothing only. It never changes a condition ID, step budget, or
//! production data. Run counts come from `REPEATS` and `ITERATIONS` (both
//! default to 5); one additional repetition warms the code and is not emitted.
//! Timing includes the small cost of constructing and serializing the exact
//! signature so every measured repetition is independently comparable.

use super::basin;
use crate::{
    experiment::{
        config::{Neighborhood, SmoothingSpec},
        plan::{compile_experiment, load_spec},
    },
    graph_partition::{Graph, PartitionState},
    optimization::{CancellationToken, rng_for},
};
use anyhow::{Context, Result, ensure};
use rand::{Rng, seq::SliceRandom};
use serde_json::{Value, json};
use std::{env, hint::black_box, path::PathBuf, time::Instant};

fn positive_env(name: &str, default: usize) -> Result<usize> {
    let value = match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .with_context(|| format!("{name} must be a positive integer"))?,
        Err(env::VarError::NotPresent) => default,
        Err(error) => return Err(error.into()),
    };
    ensure!(value > 0, "{name} must be positive");
    Ok(value)
}

fn initial(graph: &Graph, neighborhood: Neighborhood, seed: u64) -> Result<PartitionState> {
    let hash = graph.content_hash();
    let label = match neighborhood {
        Neighborhood::Flip => b"flip".as_slice(),
        Neighborhood::Swap => b"swap".as_slice(),
    };
    let seed_bytes = seed.to_le_bytes();
    let mut rng = rng_for(&[hash.as_bytes(), label, &seed_bytes, b"initial"]);
    let mut partition = (0..graph.node_count())
        .map(|_| rng.r#gen())
        .collect::<Vec<bool>>();
    if matches!(neighborhood, Neighborhood::Swap) {
        partition.fill(false);
        partition[..graph.node_count() / 2].fill(true);
        partition.shuffle(&mut rng);
    }
    PartitionState::new(graph, partition)
}

fn signature(
    result: &crate::experiment::result::BasinResult,
    end: &PartitionState,
    evaluations: u64,
) -> Value {
    json!({
        "real_bits": result.real.to_bits(),
        "smoothed_bits": result.smoothed.map(f64::to_bits),
        "partition": end.partition().iter().map(|&x| u8::from(x)).collect::<Vec<_>>(),
        "termination": result.termination,
        "steps": result.steps,
        "evaluations": evaluations,
    })
}

#[test]
#[ignore = "manual performance probe"]
fn performance_basin_probe() -> Result<()> {
    let path = PathBuf::from(env::var("GPP_PERF_SPEC").context("set GPP_PERF_SPEC")?);
    let repeats = positive_env("REPEATS", 5)?;
    let iterations = positive_env("ITERATIONS", 5)?;
    let plan = compile_experiment(load_spec(&path)?)?;
    let cancel = CancellationToken::new();

    for job in &plan.jobs {
        let graph = Graph::generate(&job.condition.graph, &cancel)?;
        let initial = initial(&graph, job.condition.neighborhood, job.seed)?;
        let mut condition = job.condition.clone();
        condition.measurement.max_basin_steps = condition.measurement.max_basin_steps.min(10_000);
        // Kernel-only instrumentation: expose the already-counted basin steps
        // in the signature without changing search, IDs, or persisted settings.
        condition.measurement.diagnostics = true;
        for (phase, spec) in [
            ("basin_random_k1", SmoothingSpec::RandomKAverage { k: 1 }),
            ("basin_random_k32", SmoothingSpec::RandomKAverage { k: 32 }),
            ("basin_all", SmoothingSpec::AllAverage),
        ] {
            crate::smoothing::validate(&spec, graph.node_count(), condition.neighborhood)?;
            let mut expected = None;
            for repetition in 0..=repeats {
                let started = Instant::now();
                let mut signatures = Vec::with_capacity(iterations);
                for iteration in 0..iterations {
                    let hash = graph.content_hash();
                    let seed_bytes = job.seed.to_le_bytes();
                    let iteration_bytes = u64::try_from(iteration)?.to_le_bytes();
                    let mut smoothing_rng = rng_for(&[
                        hash.as_bytes(),
                        &seed_bytes,
                        phase.as_bytes(),
                        &iteration_bytes,
                        b"performance-basin-smoothing",
                    ]);
                    let mut tie_rng = rng_for(&[
                        hash.as_bytes(),
                        &seed_bytes,
                        phase.as_bytes(),
                        &iteration_bytes,
                        b"performance-basin-ties",
                    ]);
                    let mut evaluations = 0;
                    let (result, end) = basin(
                        &graph,
                        &condition,
                        black_box(&initial),
                        Some(&spec),
                        Some(&mut smoothing_rng),
                        &mut tie_rng,
                        &cancel,
                        &mut evaluations,
                    )?;
                    signatures.push(signature(&result, &end, evaluations));
                    let _ = black_box(signatures.last());
                }
                let batch_signature = Value::Array(signatures);
                let signature_text = serde_json::to_string(&batch_signature)?;
                let elapsed = started.elapsed().as_nanos();
                if let Some(previous) = &expected {
                    ensure!(previous == &signature_text, "unstable {phase} signature");
                }
                expected = Some(signature_text.clone());
                if repetition > 0 {
                    println!(
                        "{}",
                        json!({
                            "case": path.file_stem().and_then(|x| x.to_str()),
                            "condition_id": job.condition_id,
                            "seed": job.seed,
                            "phase": phase,
                            "repeat": repetition - 1,
                            "iterations": iterations,
                            "elapsed_ns": elapsed,
                            "signature": signature_text,
                        })
                    );
                }
            }
        }
    }
    Ok(())
}
