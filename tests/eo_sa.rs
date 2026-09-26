use gpp_utils::{
    ExperimentPlan, ExperimentSpec, compile_experiment,
    experiment::{
        config::{
            BasinMode, Budget, Condition, FitnessSpec, GraphKind, GraphSpec, Measurement,
            Neighborhood, Schedule, SolverSpec, SolverSweep,
        },
        plan::{compile_experiment_with_versions, compile_stored_with_registry},
    },
    export::export_tsv,
    fitness::{FitnessFactory, FitnessRegistry, VertexFitness},
    graph_partition::{Graph, PartitionState},
    optimization::CancellationToken,
    run_one,
    storage::{self, RuntimeOptions, run_batch},
};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

const EXAMPLE_TOML: &str = include_str!("../examples/configs/eo_sa.toml");

fn compile_toml(text: &str) -> ExperimentPlan {
    compile_experiment(toml::from_str::<ExperimentSpec>(text).unwrap()).unwrap()
}

/// A minimal spec with a single top-level `eo_sa` solver, parameterized on the
/// `taus`/`temperatures` array literals and an extra line appended to the
/// solver table (e.g. `fitnesses = []` or a nested `[[solvers.fitnesses]]`).
fn eo_sa_toml(taus: &str, temperatures: &str, extra: &str) -> String {
    format!(
        r#"
schema_version = 1
run_seeds = [0]
neighborhoods = ["flip"]

[budget]
max_steps = 10

[[graphs]]
kind = "random"
node_counts = [6]
expected_degrees = [2.0]
seeds = [1]

[[solvers]]
kind = "eo_sa"
taus = {taus}
temperatures = {temperatures}
{extra}
"#
    )
}

fn eo_sa_plan(
    taus: &str,
    temperatures: &str,
    extra: &str,
) -> gpp_utils::error::Result<ExperimentPlan> {
    let spec: ExperimentSpec = toml::from_str(&eo_sa_toml(taus, temperatures, extra)).unwrap();
    compile_experiment(spec)
}

fn small_graph() -> Graph {
    Graph::from_edges(
        6,
        vec![
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [0, 5],
            [0, 3],
            [1, 4],
        ],
    )
    .unwrap()
}

fn eo_sa_condition(
    tau: f64,
    temperature: f64,
    neighborhood: Neighborhood,
    max_steps: u64,
    diagnostics: bool,
) -> Condition {
    Condition {
        graph: GraphSpec {
            kind: GraphKind::Random,
            node_count: 6,
            expected_degree: 2.0,
            seed: 9,
        },
        neighborhood,
        alpha: 0.2,
        solver: SolverSpec::EoSa {
            tau,
            temperature,
            fitness: FitnessSpec::default(),
        },
        budget: Budget { max_steps },
        measurement: Measurement {
            schedule: Schedule::Explicit,
            steps: vec![1, max_steps / 2],
            basin: BasinMode::None,
            max_basin_steps: 20,
            diagnostics,
            best_basin: false,
        },
    }
}

// (a) The example config compiles, documents its job count, and TOML/JSON agree.
#[test]
fn example_config_has_documented_job_count_and_toml_json_agree() {
    assert!(
        EXAMPLE_TOML.contains("20 jobs"),
        "top comment should document the job count"
    );
    let spec: ExperimentSpec = toml::from_str(EXAMPLE_TOML).unwrap();
    let plan = compile_experiment(spec.clone()).unwrap();
    assert_eq!(plan.jobs.len(), 20);

    let json = serde_json::to_string(&spec).unwrap();
    let json_plan = compile_experiment(serde_json::from_str(&json).unwrap()).unwrap();
    assert_eq!(plan.batch_id, json_plan.batch_id);
    assert_eq!(plan.jobs, json_plan.jobs);
}

// (b) Expansion is taus x temperatures x fitnesses x seeds; omitted fitnesses
// default to `default`.
#[test]
fn expansion_multiplies_axes_and_defaults_fitness_when_omitted() {
    let plan = eo_sa_plan("[0.0, 1.0, 2.0]", "[0.1, 1.0]", "").unwrap();
    // 1 graph x 1 neighborhood x (3 taus x 2 temperatures x 1 default fitness) x 1 seed.
    assert_eq!(plan.jobs.len(), 6);
    assert!(plan.jobs.iter().all(|job| matches!(
        &job.condition.solver,
        SolverSpec::EoSa { fitness, .. } if fitness.kind == "default"
    )));
}

#[test]
fn expansion_count_covers_graphs_neighborhoods_solver_axes_and_seeds() {
    let text = r#"
schema_version = 1
run_seeds = [0, 1, 2]
neighborhoods = ["flip", "swap"]

[budget]
max_steps = 10

[[graphs]]
kind = "random"
node_counts = [6, 8]
expected_degrees = [2.0]
seeds = [1]

[[solvers]]
kind = "eo_sa"
taus = [0.0, 1.0]
temperatures = [0.1, 1.0, 2.0]
[[solvers.fitnesses]]
kind = "default"
[[solvers.fitnesses]]
kind = "multiplicative"
params = { alpha = 0.5 }
"#;
    let plan = compile_toml(text);
    // graphs: 2 node_counts x 1 degree x 1 seed = 2.
    // neighborhoods: 2.
    // solver axis: 2 taus x 3 temperatures x 2 fitnesses = 12.
    // seeds: 3.
    assert_eq!(plan.jobs.len(), 2 * 2 * 12 * 3);
    assert_eq!(plan.jobs.len(), 144);
}

#[test]
fn negative_zero_tau_and_temperature_normalize_to_positive_zero_and_same_batch_id() {
    let zero = eo_sa_plan("[0.0, 1.5]", "[0.0, 2.0]", "").unwrap();
    let negative = eo_sa_plan("[-0.0, 1.5]", "[-0.0, 2.0]", "").unwrap();
    assert_eq!(zero.batch_id, negative.batch_id);
    assert!(zero.jobs.iter().any(|job| matches!(
        &job.condition.solver,
        SolverSpec::EoSa { tau, temperature, .. }
            if *tau == 0.0 && tau.is_sign_positive()
                && *temperature == 0.0 && temperature.is_sign_positive()
    )));
}

// (c) Validation errors.
#[test]
fn eo_sa_rejects_empty_or_invalid_taus_and_temperatures() {
    let cases: [(&str, &str, &str); 8] = [
        ("[]", "[0.5]", "eo_sa.taus must be non-empty"),
        ("[-0.5]", "[0.5]", "tau must be finite and non-negative"),
        ("[inf]", "[0.5]", "tau must be finite and non-negative"),
        ("[nan]", "[0.5]", "tau must be finite and non-negative"),
        ("[1.0]", "[]", "eo_sa.temperatures must be non-empty"),
        (
            "[1.0]",
            "[-0.5]",
            "temperature must be finite and non-negative",
        ),
        (
            "[1.0]",
            "[inf]",
            "temperature must be finite and non-negative",
        ),
        (
            "[1.0]",
            "[nan]",
            "temperature must be finite and non-negative",
        ),
    ];
    for (taus, temperatures, expected) in cases {
        let error = eo_sa_plan(taus, temperatures, "").unwrap_err().to_string();
        assert!(
            error.contains(expected),
            "taus={taus} temperatures={temperatures}: {error}"
        );
    }
}

#[test]
fn eo_sa_rejects_empty_fitnesses_array() {
    let error = eo_sa_plan("[1.0]", "[0.5]", "fitnesses = []")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("eo_sa.fitnesses must be non-empty when specified"),
        "{error}"
    );
}

#[test]
fn eo_sa_rejects_unknown_fitness_kind() {
    let error = eo_sa_plan(
        "[1.0]",
        "[0.5]",
        "[[solvers.fitnesses]]\nkind = \"totally_unknown\"",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("unknown fitness"), "{error}");
}

#[test]
fn smoothing_key_on_eo_sa_solver_is_a_parse_error() {
    let text = eo_sa_toml("[1.0]", "[0.5]", "smoothing = []");
    assert!(toml::from_str::<ExperimentSpec>(&text).is_err());
}

#[test]
fn swap_with_odd_node_count_is_rejected() {
    let text = r#"
schema_version = 1
run_seeds = [0]
neighborhoods = ["swap"]

[budget]
max_steps = 10

[[graphs]]
kind = "random"
node_counts = [7]
expected_degrees = [2.0]
seeds = [1]

[[solvers]]
kind = "eo_sa"
taus = [1.0]
temperatures = [0.5]
"#;
    let spec: ExperimentSpec = toml::from_str(text).unwrap();
    let error = compile_experiment(spec).unwrap_err().to_string();
    assert!(error.contains("even node_count"), "{error}");
}

// (d) Condition IDs.
#[test]
fn eo_sa_condition_id_differs_from_eo_and_sa_with_matching_parameters() {
    let text = r#"
schema_version = 1
run_seeds = [0]
neighborhoods = ["flip"]

[budget]
max_steps = 10

[[graphs]]
kind = "random"
node_counts = [6]
expected_degrees = [2.0]
seeds = [1]

[[solvers]]
kind = "sa"
temperatures = [0.5]

[[solvers]]
kind = "eo"
taus = [1.0]

[[solvers]]
kind = "eo_sa"
taus = [1.0]
temperatures = [0.5]
"#;
    let plan = compile_toml(text);
    assert_eq!(plan.jobs.len(), 3);
    let ids: BTreeSet<_> = plan.jobs.iter().map(|j| j.condition_id.clone()).collect();
    assert_eq!(
        ids.len(),
        3,
        "sa, eo and eo_sa must not collide on shared parameters"
    );
}

#[test]
fn adding_eo_sa_solver_does_not_change_existing_sa_hc_eo_condition_ids() {
    let original: ExperimentSpec =
        toml::from_str(gpp_utils::experiment::plan::sample_toml()).unwrap();
    let mut extended = original.clone();
    extended.solvers.push(SolverSweep::EoSa {
        taus: vec![0.7],
        temperatures: vec![0.4],
        fitnesses: None,
    });
    let a = compile_experiment(original).unwrap();
    let b = compile_experiment(extended).unwrap();
    let non_eo_sa_ids = |plan: &ExperimentPlan| {
        plan.jobs
            .iter()
            .filter(|job| !matches!(&job.condition.solver, SolverSpec::EoSa { .. }))
            .map(|job| job.condition_id.clone())
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(non_eo_sa_ids(&a), non_eo_sa_ids(&b));
    assert!(b.jobs.len() > a.jobs.len());
}

#[test]
fn versions_pin_fitness_default_when_only_eo_sa_uses_it() {
    let plan = eo_sa_plan("[1.0]", "[0.5]", "").unwrap();
    assert!(
        !plan
            .jobs
            .iter()
            .any(|job| matches!(&job.condition.solver, SolverSpec::Eo { .. }))
    );
    assert_eq!(
        plan.experiment.versions.get("fitness:default"),
        Some(&"good_edge_fraction-v1".to_owned())
    );
}

// (e) Baseline invariance.
#[test]
fn baseline_v1_batch_id_and_job_count_are_unchanged() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/experiments_sa_eo/baseline_v1.toml"
    ))
    .unwrap();
    let plan = compile_experiment(toml::from_str(&text).unwrap()).unwrap();
    assert_eq!(
        plan.batch_id,
        "651e26afb279e0bf302a05cbf30dc52aa57f4854193ea03a72f5549b0d19cc83"
    );
    assert_eq!(plan.jobs.len(), 163_584);
}

// (f) `run_one` end-to-end.
#[test]
fn run_one_end_to_end_for_eo_sa_flip_and_swap() {
    let graph = small_graph();
    let registry = FitnessRegistry::default();
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        let c = eo_sa_condition(1.2, 0.6, neighborhood, 60, true);
        let result = run_one(&graph, &c, 7, &CancellationToken::new(), &registry).unwrap();
        // `run_one` already validates before returning; this documents the contract.
        result.validate(&graph, &c).unwrap();
        assert_eq!(result.completed_steps, c.budget.max_steps);
        let diagnostics = result.diagnostics.as_ref().unwrap();
        assert!(diagnostics.fitness_values_computed_search.is_some());
        assert!(diagnostics.applied_moves <= result.completed_steps);
    }
}

#[test]
fn zero_temperature_never_worsens_the_real_score_and_rejects_some_moves() {
    let graph = small_graph();
    let c = eo_sa_condition(1.0, 0.0, Neighborhood::Flip, 80, true);
    let result = run_one(
        &graph,
        &c,
        42,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
    )
    .unwrap();
    let initial = graph.score(
        &result.partitions[result.records[0].current_solution.0],
        c.alpha,
    );
    let final_score = graph.score(&result.partitions[result.final_solution.0], c.alpha);
    assert!(final_score <= initial);
    let diagnostics = result.diagnostics.as_ref().unwrap();
    assert!(
        diagnostics.applied_moves < result.completed_steps,
        "expected at least one rejected step at temperature 0"
    );
}

#[test]
fn huge_temperature_accepts_every_proposed_move() {
    let graph = small_graph();
    let c = eo_sa_condition(1.0, 1e300, Neighborhood::Swap, 40, true);
    let result = run_one(
        &graph,
        &c,
        42,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
    )
    .unwrap();
    let diagnostics = result.diagnostics.as_ref().unwrap();
    assert_eq!(diagnostics.applied_moves, result.completed_steps);
}

// (g) Batch run + resume + export.
fn eo_sa_batch_spec() -> ExperimentSpec {
    serde_json::from_value(json!({
        "schema_version": 1, "run_seeds": [1, 2], "neighborhoods": ["flip"],
        "problem": {"alpha": 0.2}, "budget": {"max_steps": 40},
        "measurement": {"schedule":"explicit", "steps":[1, 20], "basin":"none", "max_basin_steps":10, "diagnostics":true},
        "graphs": [{"kind":"random", "node_counts":[6], "expected_degrees":[2.0], "seeds":[8]}],
        "solvers": [{"kind":"eo_sa", "taus":[1.0], "temperatures":[0.5]}]
    }))
    .unwrap()
}

#[test]
fn batch_run_resume_and_export_produce_expected_eo_sa_columns() {
    let plan = compile_experiment(eo_sa_batch_spec()).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let options = RuntimeOptions {
        root: temp.path().join("data"),
        threads: 1,
        overwrite: false,
        recover_corrupt: false,
        rounds: false,
        round_deadline: None,
    };
    let first = run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    assert_eq!(first.completed, 2);
    // A second run acts as resume: both jobs already have a completed result.
    let second = run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    assert_eq!(second.reused, 2);
    assert_eq!(second.completed, 0);

    let out = temp.path().join("export");
    export_tsv(&plan, &options.root, &out, false, false).unwrap();
    let runs = std::fs::read_to_string(out.join("runs.tsv")).unwrap();
    let mut lines = runs.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let index = |name: &str| header.iter().position(|&h| h == name).unwrap();
    let solver_i = index("solver");
    let temperature_i = index("temperature");
    let tau_i = index("tau");
    let smoothing_i = index("smoothing");
    let k_i = index("k");
    let fitness_i = index("fitness");
    let applied_i = index("applied_moves");
    let rejected_i = index("rejected_moves");
    let completed_i = index("completed_steps");

    let mut rows = 0;
    for line in lines {
        let cells: Vec<&str> = line.split('\t').collect();
        assert_eq!(cells[solver_i], "eo_sa");
        assert_eq!(cells[temperature_i].parse::<f64>().unwrap(), 0.5);
        assert_eq!(cells[tau_i].parse::<f64>().unwrap(), 1.0);
        assert_eq!(cells[smoothing_i], "");
        assert_eq!(cells[k_i], "");
        assert_eq!(cells[fitness_i], "default");
        let applied: u64 = cells[applied_i].parse().unwrap();
        let completed: u64 = cells[completed_i].parse().unwrap();
        let rejected: u64 = cells[rejected_i].parse().unwrap();
        assert_eq!(rejected, completed - applied);
        rows += 1;
    }
    assert_eq!(rows, 2);
}

// (h) Registry: a custom fitness used by an eo_sa condition.
struct CustomHFactory;
struct CustomHFitness;
impl FitnessFactory for CustomHFactory {
    fn version(&self) -> &str {
        "custom-h-v1"
    }
    fn validate(&self, params: &serde_json::Value) -> gpp_utils::error::Result<()> {
        if params == &json!({}) {
            Ok(())
        } else {
            Err(gpp_utils::error::Error::msg("unexpected params"))
        }
    }
    fn create(&self, _: &serde_json::Value) -> gpp_utils::error::Result<Box<dyn VertexFitness>> {
        Ok(Box::new(CustomHFitness))
    }
}
impl VertexFitness for CustomHFitness {
    fn values(&self, graph: &Graph, _: &PartitionState) -> gpp_utils::error::Result<Vec<f64>> {
        Ok((0..graph.node_count()).map(|v| v as f64).collect())
    }
}

#[test]
fn eo_sa_custom_fitness_compiles_validates_and_runs_through_the_registry() {
    let spec: ExperimentSpec = serde_json::from_value(json!({
        "schema_version": 1, "run_seeds": [3], "neighborhoods": ["flip"],
        "budget": {"max_steps": 10},
        "measurement": {"basin": "none"},
        "graphs": [{"kind": "random", "node_counts": [6], "expected_degrees": [2.0], "seeds": [5]}],
        "solvers": [{
            "kind": "eo_sa", "taus": [1.0], "temperatures": [0.5],
            "fitnesses": [{"kind": "custom_h", "params": {}}]
        }]
    }))
    .unwrap();

    let versions = BTreeMap::from([("custom_h".to_owned(), "custom-h-v1".to_owned())]);
    let plan = compile_experiment_with_versions(spec, &versions).unwrap();
    assert_eq!(
        plan.experiment.versions.get("fitness:custom_h"),
        Some(&"custom-h-v1".to_owned())
    );

    // Rejected without the factory, by both registry-aware entry points.
    let default_registry = FitnessRegistry::default();
    assert!(storage::validate_registry(&plan, &default_registry).is_err());
    assert!(compile_stored_with_registry(plan.experiment.clone(), &default_registry).is_err());

    // Accepted and runnable once the factory is registered.
    let mut registry = FitnessRegistry::default();
    registry.register("custom_h", Arc::new(CustomHFactory));
    assert!(storage::validate_registry(&plan, &registry).is_ok());
    let restored = compile_stored_with_registry(plan.experiment.clone(), &registry).unwrap();
    assert_eq!(restored.batch_id, plan.batch_id);

    let temp = tempfile::tempdir().unwrap();
    let options = RuntimeOptions {
        root: temp.path().into(),
        threads: 1,
        ..RuntimeOptions::default()
    };
    let summary = run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &registry,
        &|_| {},
    )
    .unwrap();
    assert_eq!(summary.completed, 1);
    assert_eq!(summary.failed, 0);
}
