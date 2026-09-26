//! Verify that JSON benchmark subsets retain the source TOML condition IDs.
//!
//! This performs specification parsing and plan compilation only. It neither
//! generates graphs nor evaluates, saves, or resumes experiment results.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use gpp_utils::experiment::{
    config::Condition,
    plan::{compile_experiment, load_spec},
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    /// Directory containing manifest.json and its one-job JSON case files.
    #[arg(long)]
    cases: PathBuf,
    /// Directory containing the source TOML files named by the manifest.
    #[arg(long)]
    sources: PathBuf,
}

#[derive(Deserialize)]
struct ManifestCase {
    name: String,
    source: PathBuf,
    spec: PathBuf,
}

fn digest(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn same_value_bits(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Null, Value::Null)
        | (Value::Bool(_), Value::Bool(_))
        | (Value::String(_), Value::String(_)) => left == right,
        (Value::Number(left), Value::Number(right)) => {
            let left_text = left.to_string();
            let right_text = right.to_string();
            let left_float = left_text.contains(['.', 'e', 'E']);
            let right_float = right_text.contains(['.', 'e', 'E']);
            match (left_float, right_float) {
                (true, true) => {
                    left_text.parse::<f64>().ok().map(f64::to_bits)
                        == right_text.parse::<f64>().ok().map(f64::to_bits)
                }
                (false, false) => left == right,
                _ => false,
            }
        }
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| same_value_bits(left, right))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, value)| {
                    right
                        .get(key)
                        .is_some_and(|other| same_value_bits(value, other))
                })
        }
        _ => false,
    }
}

fn same_condition_bits(left: &Condition, right: &Condition) -> Result<bool> {
    Ok(same_value_bits(
        &serde_json::to_value(left)?,
        &serde_json::to_value(right)?,
    ))
}

fn select_source_subset(
    mut source: gpp_utils::experiment::config::ExperimentSpec,
    case: &Condition,
    seed: u64,
) -> Result<gpp_utils::experiment::config::ExperimentSpec> {
    // Retain values from the parsed TOML; do not recreate any scientific number.
    source.graphs.retain(|sweep| sweep.kind == case.graph.kind);
    for sweep in &mut source.graphs {
        sweep.node_counts.retain(|&n| n == case.graph.node_count);
        sweep
            .expected_degrees
            .retain(|degree| degree.to_bits() == case.graph.expected_degree.to_bits());
        sweep
            .seeds
            .retain(|&graph_seed| graph_seed == case.graph.seed);
    }
    source.graphs.retain(|sweep| {
        !sweep.node_counts.is_empty()
            && !sweep.expected_degrees.is_empty()
            && !sweep.seeds.is_empty()
    });
    source.run_seeds.retain(|&run_seed| run_seed == seed);
    source
        .neighborhoods
        .retain(|neighborhood| *neighborhood == case.neighborhood);
    ensure!(
        source.graphs.len() == 1,
        "source subset must contain one matching graph sweep"
    );
    let graph = &source.graphs[0];
    ensure!(
        graph.kind == case.graph.kind
            && graph.node_counts.len() == 1
            && graph.expected_degrees.len() == 1
            && graph.seeds.len() == 1,
        "source graph sweep did not reduce to the case graph"
    );
    ensure!(
        source.run_seeds.len() == 1,
        "source subset must contain the case run seed"
    );
    ensure!(
        source.neighborhoods.len() == 1,
        "source subset must contain the case neighborhood"
    );
    Ok(source)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let manifest_path = args.cases.join("manifest.json");
    let manifest: Vec<ManifestCase> = serde_json::from_slice(
        &fs::read(&manifest_path).with_context(|| format!("read {}", manifest_path.display()))?,
    )
    .context("parse case manifest")?;
    ensure!(!manifest.is_empty(), "case manifest must not be empty");
    let mut source_sha256 = BTreeMap::new();

    for entry in &manifest {
        let case_path = args.cases.join(&entry.spec);
        let case_plan = compile_experiment(
            load_spec(&case_path).with_context(|| format!("load case {}", entry.name))?,
        )
        .with_context(|| format!("compile case {}", entry.name))?;
        ensure!(
            case_plan.jobs.len() == 1,
            "{} must compile to exactly one job",
            entry.name
        );
        let case_job = &case_plan.jobs[0];
        let source_path = args.sources.join(&entry.source);
        source_sha256
            .entry(entry.source.to_string_lossy().into_owned())
            .or_insert(digest(&source_path)?);
        let source =
            load_spec(&source_path).with_context(|| format!("load source for {}", entry.name))?;
        let subset = select_source_subset(source, &case_job.condition, case_job.seed)
            .with_context(|| format!("select source subset for {}", entry.name))?;
        let source_plan = compile_experiment(subset)
            .with_context(|| format!("compile source subset for {}", entry.name))?;
        let source_job = source_plan
            .jobs
            .iter()
            .find(|candidate| {
                candidate.condition_id == case_job.condition_id
                    && candidate.graph_id == case_job.graph_id
                    && candidate.seed == case_job.seed
            })
            .with_context(|| {
                format!(
                    "{} condition/graph/seed is absent from source subset",
                    entry.name
                )
            })?;
        ensure!(
            same_condition_bits(&case_job.condition, &source_job.condition)?,
            "{} condition differs from source subset at f64-bit precision",
            entry.name
        );
    }
    println!(
        "{}",
        serde_json::json!({
            "verified_cases": manifest.len(),
            "source_sha256": source_sha256,
        })
    );
    Ok(())
}
