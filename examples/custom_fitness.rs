//! Register the same custom fitness definition for planning, running and resume.
use anyhow::{Result, ensure};
use gpp_utils::{
    experiment::{config::ExperimentSpec, plan::compile_experiment_with_versions},
    fitness::{FitnessFactory, FitnessRegistry, VertexFitness},
    graph_partition::{Graph, PartitionState},
    optimization::CancellationToken,
    storage::{RuntimeOptions, load_plan_with_registry, run_batch},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    reverse: bool,
}

struct DegreeFactory;
struct DegreeFitness(Params);

impl FitnessFactory for DegreeFactory {
    fn version(&self) -> &str {
        "degree-fraction-v1"
    }

    fn validate(&self, params: &Value) -> Result<()> {
        let _: Params = serde_json::from_value(params.clone())?;
        Ok(())
    }

    fn create(&self, params: &Value) -> Result<Box<dyn VertexFitness>> {
        Ok(Box::new(DegreeFitness(serde_json::from_value(
            params.clone(),
        )?)))
    }
}

impl VertexFitness for DegreeFitness {
    fn values(&self, graph: &Graph, _: &PartitionState) -> Result<Vec<f64>> {
        // Demonstrates the extension API, not a recommended partition heuristic.
        Ok((0..graph.node_count())
            .map(|vertex| {
                let value = graph.degree(vertex) as f64 / graph.node_count() as f64;
                if self.0.reverse { 1.0 - value } else { value }
            })
            .collect())
    }
}

fn main() -> Result<()> {
    let mut registry = FitnessRegistry::default();
    ensure!(
        registry
            .register("example_degree", Arc::new(DegreeFactory))
            .is_none(),
        "duplicate fitness name"
    );
    let spec: ExperimentSpec = serde_json::from_value(json!({
        "schema_version": 1,
        "run_seeds": [7],
        "neighborhoods": ["flip"],
        "budget": {"max_steps": 8},
        "measurement": {"basin": "none"},
        "graphs": [{"kind": "random", "node_counts": [6], "expected_degrees": [2.0], "seeds": [42]}],
        "solvers": [{"kind": "eo", "taus": [1.5], "fitnesses": [{"kind": "example_degree", "params": {"reverse": false}}]}]
    }))?;
    let plan = compile_experiment_with_versions(spec, &registry.versions())?;
    // Version discovery is separate from factory validation; validate before I/O.
    gpp_utils::storage::validate_registry(&plan, &registry)?;
    let directory = tempfile::tempdir()?;
    let options = RuntimeOptions {
        root: directory.path().into(),
        threads: 1,
        ..Default::default()
    };
    let cancel = CancellationToken::new();
    let first = run_batch(&plan, &options, &cancel, &registry, &|_| {})?;
    ensure!(
        first.completed == 1 && first.failed == 0,
        "example run failed"
    );
    // A default-only registry must reject this saved experiment.
    ensure!(
        load_plan_with_registry(
            directory.path(),
            &plan.batch_id,
            &FitnessRegistry::default()
        )
        .is_err()
    );
    let restored = load_plan_with_registry(directory.path(), &plan.batch_id, &registry)?;
    let resumed = run_batch(&restored, &options, &cancel, &registry, &|_| {})?;
    ensure!(
        resumed.reused == 1 && resumed.failed == 0,
        "example resume failed"
    );
    println!(
        "custom fitness: completed={} reused={}",
        first.completed, resumed.reused
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_fitness_persisted_lifecycle() {
        main().unwrap();
    }

    #[test]
    fn factory_rejects_unknown_or_missing_parameters() {
        assert!(DegreeFactory.validate(&json!({"reverse": false})).is_ok());
        assert!(DegreeFactory.validate(&json!({})).is_err());
        assert!(
            DegreeFactory
                .validate(&json!({"reverse": false, "typo": 1}))
                .is_err()
        );
        assert!(DegreeFactory.validate(&json!({"reverse": 1})).is_err());
    }
}
