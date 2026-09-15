use gpp_utils::{
    compile_experiment,
    experiment::plan::compile_experiment_with_versions,
    export::export_tsv,
    storage::{RuntimeOptions, inspect, read_result, result_path, run_batch},
};
use gpp_utils::{
    experiment::{
        config::{
            BasinMode, Budget, Condition, FitnessSpec, GraphKind, GraphSpec, Measurement,
            Neighborhood, Schedule, SmoothingSpec, SolverSpec,
        },
        result::{RunTermination, RunView},
    },
    fitness::{FitnessFactory, FitnessRegistry, VertexFitness},
    graph_partition::{Graph, PartitionState},
    optimization::CancellationToken,
    run_one,
};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn condition(solver: SolverSpec, neighborhood: Neighborhood, diagnostics: bool) -> Condition {
    Condition {
        graph: GraphSpec {
            kind: GraphKind::Random,
            node_count: 6,
            expected_degree: 2.0,
            seed: 9,
        },
        neighborhood,
        alpha: 0.2,
        solver,
        budget: Budget { max_steps: 12 },
        measurement: Measurement {
            schedule: Schedule::Explicit,
            steps: vec![1, 3, 7],
            basin: BasinMode::None,
            max_basin_steps: 20,
            diagnostics,
            best_basin: false,
        },
    }
}

fn graph() -> Graph {
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

fn assert_result_contract(graph: &Graph, condition: &Condition, result: &gpp_utils::RunResult) {
    result.validate(graph, condition).unwrap();
    assert_eq!(result.records.first().unwrap().step, 0);
    assert_eq!(result.records.last().unwrap().step, result.completed_steps);
    assert_eq!(
        result.records.last().unwrap().current_solution,
        result.final_solution
    );
    assert_eq!(
        result.records.last().unwrap().best_solution,
        result.best_solution
    );
    let view = RunView::new(graph, condition, result).unwrap();
    assert!(
        view.best_score()
            <= graph.score(
                result.partitions[result.records[0].current_solution.0].as_slice(),
                condition.alpha
            )
    );
    assert!(result.partitions.len() <= result.records.len() * 2);
}

#[test]
fn every_solver_supports_flip_and_swap_and_preserves_contract() {
    let graph = graph();
    let registry = FitnessRegistry::default();
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        for solver in [
            SolverSpec::Hc {
                smoothing: SmoothingSpec::None,
            },
            SolverSpec::Sa {
                temperature: 0.7,
                smoothing: SmoothingSpec::None,
            },
            SolverSpec::Eo {
                tau: 1.3,
                fitness: FitnessSpec::default(),
            },
        ] {
            let c = condition(solver, neighborhood, true);
            let result = run_one(&graph, &c, 123, &CancellationToken::new(), &registry).unwrap();
            assert_result_contract(&graph, &c, &result);
            assert!(result.diagnostics.as_ref().unwrap().applied_moves <= result.completed_steps);
        }
    }
}

#[test]
fn diagnostics_and_measurement_intervals_do_not_change_trajectory() {
    let graph = graph();
    let registry = FitnessRegistry::default();
    let base = condition(
        SolverSpec::Sa {
            temperature: 1.1,
            smoothing: SmoothingSpec::None,
        },
        Neighborhood::Flip,
        false,
    );
    let mut changed = base.clone();
    changed.measurement.steps = vec![2, 4, 6, 8, 10];
    changed.measurement.diagnostics = true;
    let a = run_one(&graph, &base, 777, &CancellationToken::new(), &registry).unwrap();
    let b = run_one(&graph, &changed, 777, &CancellationToken::new(), &registry).unwrap();
    assert_eq!(
        a.partitions[a.final_solution.0],
        b.partitions[b.final_solution.0]
    );
    assert_eq!(
        a.partitions[a.best_solution.0],
        b.partitions[b.best_solution.0]
    );
    assert_eq!(a.best_step, b.best_step);
    assert!(a.diagnostics.is_none());
    assert!(b.diagnostics.is_some());
}

#[test]
fn random_smoothing_trajectory_ignores_basin_measurement_frequency_and_has_budget_prefix() {
    let graph = graph();
    let registry = FitnessRegistry::default();
    let mut short = condition(
        SolverSpec::Sa {
            temperature: 0.8,
            smoothing: SmoothingSpec::RandomKAverage { k: 5 },
        },
        Neighborhood::Flip,
        false,
    );
    short.budget.max_steps = 5;
    short.measurement.steps = vec![1];
    short.measurement.basin = BasinMode::Both;
    let mut long = short.clone();
    long.budget.max_steps = 12;
    long.measurement.steps = (1..12).collect();
    let a = run_one(&graph, &short, 91, &CancellationToken::new(), &registry).unwrap();
    let b = run_one(&graph, &long, 91, &CancellationToken::new(), &registry).unwrap();
    let at_five = b.records.iter().find(|r| r.step == 5).unwrap();
    assert_eq!(
        a.partitions[a.final_solution.0],
        b.partitions[at_five.current_solution.0]
    );
    assert_eq!(
        a.partitions[a.best_solution.0],
        b.partitions[at_five.best_solution.0]
    );
    assert!(a.records.iter().all(|r| r.current_smoothed.is_some()
        && r.search_evaluation.is_some()
        && r.basin_real.is_some()
        && r.basin_smoothed.is_some()));
}

#[test]
fn incumbent_keeps_an_improvement_between_measurements_after_current_worsens() {
    let graph = graph();
    let registry = FitnessRegistry::default();
    let mut c = condition(
        SolverSpec::Sa {
            temperature: 10.0,
            smoothing: SmoothingSpec::None,
        },
        Neighborhood::Flip,
        false,
    );
    c.budget.max_steps = 30;
    c.measurement.steps = vec![1];
    let result = run_one(&graph, &c, 1, &CancellationToken::new(), &registry).unwrap();
    let measured_steps: Vec<_> = result.records.iter().map(|r| r.step).collect();
    let view = RunView::new(&graph, &c, &result).unwrap();
    assert!(result.best_step > 1 && result.best_step < result.completed_steps);
    assert!(!measured_steps.contains(&result.best_step));
    assert!(view.best_score() < view.final_score());
    let mut dense_condition = c.clone();
    dense_condition.measurement.steps = (1..30).collect();
    let dense = run_one(
        &graph,
        &dense_condition,
        1,
        &CancellationToken::new(),
        &registry,
    )
    .unwrap();
    let dense_best = dense
        .records
        .iter()
        .min_by(|a, b| {
            graph
                .score(&dense.partitions[a.current_solution.0], c.alpha)
                .total_cmp(&graph.score(&dense.partitions[b.current_solution.0], c.alpha))
        })
        .unwrap();
    assert_eq!(
        result.partitions[result.best_solution.0],
        dense.partitions[dense_best.current_solution.0]
    );
    assert_ne!(result.best_solution, result.final_solution);
    assert_eq!(
        result.records.last().unwrap().best_solution,
        result.best_solution
    );
}

struct CancellingFitness(CancellationToken);
impl VertexFitness for CancellingFitness {
    fn values(&self, graph: &Graph, _: &PartitionState) -> gpp_utils::error::Result<Vec<f64>> {
        self.0.cancel();
        Ok(vec![0.0; graph.node_count])
    }
}
struct CancellingFactory(CancellationToken);
impl FitnessFactory for CancellingFactory {
    fn version(&self) -> &str {
        "cancel-v1"
    }
    fn validate(&self, _: &serde_json::Value) -> gpp_utils::error::Result<()> {
        Ok(())
    }
    fn create(&self, _: &serde_json::Value) -> gpp_utils::error::Result<Box<dyn VertexFitness>> {
        Ok(Box::new(CancellingFitness(self.0.clone())))
    }
}

#[test]
fn cancellation_during_search_returns_valid_partial_result() {
    let graph = graph();
    let c = condition(
        SolverSpec::Eo {
            tau: 1.0,
            fitness: FitnessSpec {
                kind: "cancel".into(),
                params: json!({}),
            },
        },
        Neighborhood::Flip,
        false,
    );
    let cancel = CancellationToken::new();
    let mut registry = FitnessRegistry::default();
    registry.register("cancel", Arc::new(CancellingFactory(cancel.clone())));
    let result = run_one(&graph, &c, 5, &cancel, &registry).unwrap();
    assert_eq!(result.termination, RunTermination::Cancelled);
    assert_eq!(result.completed_steps, 0);
    assert_result_contract(&graph, &c, &result);
}

struct CountingFactory(Arc<AtomicUsize>);
struct CountingFitness(Arc<AtomicUsize>);
impl FitnessFactory for CountingFactory {
    fn version(&self) -> &str {
        "counting-v1"
    }
    fn validate(&self, params: &serde_json::Value) -> gpp_utils::error::Result<()> {
        if params == &json!({"bias": 1}) {
            Ok(())
        } else {
            Err(gpp_utils::error::Error::msg("bad params"))
        }
    }
    fn create(&self, _: &serde_json::Value) -> gpp_utils::error::Result<Box<dyn VertexFitness>> {
        Ok(Box::new(CountingFitness(self.0.clone())))
    }
}
impl VertexFitness for CountingFitness {
    fn values(&self, graph: &Graph, _: &PartitionState) -> gpp_utils::error::Result<Vec<f64>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok((0..graph.node_count).map(|x| x as f64).collect())
    }
}

#[test]
fn eo_uses_registered_custom_fitness_and_validates_params() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut registry = FitnessRegistry::default();
    registry.register("counting", Arc::new(CountingFactory(calls.clone())));
    let fitness = FitnessSpec {
        kind: "counting".into(),
        params: json!({"bias": 1}),
    };
    let c = condition(
        SolverSpec::Eo { tau: 1.2, fitness },
        Neighborhood::Flip,
        false,
    );
    run_one(&graph(), &c, 10, &CancellationToken::new(), &registry).unwrap();
    assert!(calls.load(Ordering::SeqCst) > 0);
    assert!(
        registry
            .validate(&FitnessSpec {
                kind: "counting".into(),
                params: json!({})
            })
            .is_err()
    );
}

fn one_job_spec() -> gpp_utils::ExperimentSpec {
    serde_json::from_value(json!({
        "schema_version": 1, "run_seeds": [4], "neighborhoods": ["flip"],
        "problem": {"alpha": 0.2}, "budget": {"max_steps": 3},
        "measurement": {"schedule":"explicit", "steps":[1], "basin":"none", "max_basin_steps":10, "diagnostics":false},
        "graphs": [{"kind":"random", "node_counts":[6], "expected_degrees":[2.0], "seeds":[8]}],
        "solvers": [{"kind":"sa", "temperatures":[0.5], "smoothing":[{"kind":"none"}]}]
    })).unwrap()
}

struct FailingDefaultFactory;
impl FitnessFactory for FailingDefaultFactory {
    fn version(&self) -> &str {
        "good_edge_fraction-v1"
    }
    fn validate(&self, _: &serde_json::Value) -> gpp_utils::error::Result<()> {
        Ok(())
    }
    fn create(&self, _: &serde_json::Value) -> gpp_utils::error::Result<Box<dyn VertexFitness>> {
        Err(gpp_utils::error::Error::msg(
            "injected fitness construction failure",
        ))
    }
}

#[test]
fn failed_overwrite_preserves_completed_result_and_reports_latest_failure() {
    let mut spec = one_job_spec();
    spec.solvers = vec![gpp_utils::experiment::config::SolverSweep::Eo {
        taus: vec![1.2],
        fitnesses: None,
    }];
    let plan = compile_experiment(spec).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let mut options = RuntimeOptions {
        root: temp.path().into(),
        threads: 1,
        overwrite: false,
        recover_corrupt: false,
        rounds: false,
        round_deadline: None,
    };
    run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    let path = result_path(temp.path(), &plan.jobs[0]);
    let original = std::fs::read(&path).unwrap();
    let mut broken = FitnessRegistry::default();
    broken.register("default", Arc::new(FailingDefaultFactory));
    options.overwrite = true;
    let failed = run_batch(&plan, &options, &CancellationToken::new(), &broken, &|_| {}).unwrap();
    assert_eq!(failed.failed, 1);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let rows = inspect(&plan, temp.path(), true).unwrap();
    assert_eq!(rows[0].status, "completed");
    assert_eq!(rows[0].latest_attempt_status.as_deref(), Some("failed"));
    assert!(rows[0].error.as_ref().unwrap().message.contains("injected"));
}

#[test]
fn cancelled_partial_result_is_exported_only_when_requested() {
    let mut spec = one_job_spec();
    spec.solvers = vec![gpp_utils::experiment::config::SolverSweep::Eo {
        taus: vec![1.0],
        fitnesses: Some(vec![FitnessSpec {
            kind: "cancel".into(),
            params: json!({}),
        }]),
    }];
    let plan = compile_experiment_with_versions(
        spec,
        &std::collections::BTreeMap::from([("cancel".into(), "cancel-v1".into())]),
    )
    .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let mut registry = FitnessRegistry::default();
    registry.register("cancel", Arc::new(CancellingFactory(cancel.clone())));
    let options = RuntimeOptions {
        root: temp.path().join("data"),
        threads: 1,
        overwrite: false,
        recover_corrupt: false,
        rounds: false,
        round_deadline: None,
    };
    let summary = run_batch(&plan, &options, &cancel, &registry, &|_| {}).unwrap();
    assert_eq!(summary.cancelled, 1);
    let without = temp.path().join("without");
    export_tsv(&plan, &options.root, &without, false, false).unwrap();
    assert_eq!(
        std::fs::read_to_string(without.join("traces.tsv"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    let with = temp.path().join("with");
    let exported = export_tsv(&plan, &options.root, &with, true, false).unwrap();
    assert!(exported.trace_rows >= 1);
    let traces = std::fs::read_to_string(with.join("traces.tsv")).unwrap();
    assert!(
        traces
            .lines()
            .skip(1)
            .any(|line| line.contains("cancelled"))
    );
}

#[test]
fn batch_results_are_independent_of_worker_count() {
    let mut spec = one_job_spec();
    spec.run_seeds = vec![1, 2, 3, 4];
    let plan = compile_experiment(spec).unwrap();
    let one = tempfile::tempdir().unwrap();
    let many = tempfile::tempdir().unwrap();
    for (root, threads) in [(one.path(), 1), (many.path(), 4)] {
        let options = RuntimeOptions {
            root: root.into(),
            threads,
            overwrite: false,
            recover_corrupt: false,
            rounds: false,
            round_deadline: None,
        };
        let summary = run_batch(
            &plan,
            &options,
            &CancellationToken::new(),
            &FitnessRegistry::default(),
            &|_| {},
        )
        .unwrap();
        assert_eq!(summary.completed, 4);
    }
    for job in &plan.jobs {
        let a: serde_json::Value =
            serde_json::from_slice(&std::fs::read(result_path(one.path(), job)).unwrap()).unwrap();
        let b: serde_json::Value =
            serde_json::from_slice(&std::fs::read(result_path(many.path(), job)).unwrap()).unwrap();
        for key in [
            "partitions",
            "final_solution",
            "best_solution",
            "best_step",
            "records",
            "completed_steps",
            "termination",
        ] {
            assert_eq!(a[key], b[key], "worker-count difference in {key}");
        }
    }
}

#[test]
fn storage_reuses_valid_result_and_rejects_corruption_without_overwriting_it() {
    let temp = tempfile::tempdir().unwrap();
    let plan = compile_experiment(one_job_spec()).unwrap();
    let options = RuntimeOptions {
        root: temp.path().into(),
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
    assert_eq!(first.completed, 1);
    let path = result_path(temp.path(), &plan.jobs[0]);
    let original = std::fs::read(&path).unwrap();
    let second = run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    assert_eq!(second.reused, 1);
    assert_eq!(std::fs::read(&path).unwrap(), original);

    let mut malformed: serde_json::Value = serde_json::from_slice(&original).unwrap();
    malformed["schema_version"] = json!(99);
    std::fs::write(&path, serde_json::to_vec(&malformed).unwrap()).unwrap();
    let corrupt_bytes = std::fs::read(&path).unwrap();
    let summary = run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    assert_eq!(summary.failed, 1);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        corrupt_bytes,
        "failed non-recovery run must preserve corrupt evidence"
    );
}

#[test]
fn a_second_writer_is_rejected_until_the_lock_is_released() {
    let temp = tempfile::tempdir().unwrap();
    let first = gpp_utils::storage::atomic::WriterLock::acquire(temp.path()).unwrap();
    assert!(gpp_utils::storage::atomic::writer_active(temp.path()).unwrap());
    assert!(gpp_utils::storage::atomic::WriterLock::acquire(temp.path()).is_err());
    drop(first);
    assert!(!gpp_utils::storage::atomic::writer_active(temp.path()).unwrap());
    gpp_utils::storage::atomic::WriterLock::acquire(temp.path()).unwrap();
}

#[test]
fn completed_json_omits_derived_scores_and_redundant_condition_fields() {
    let graph = graph();
    let c = condition(
        SolverSpec::Hc {
            smoothing: SmoothingSpec::None,
        },
        Neighborhood::Flip,
        false,
    );
    let result = run_one(
        &graph,
        &c,
        3,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
    )
    .unwrap();
    let value = serde_json::to_value(&result).unwrap();
    for forbidden in [
        "status",
        "seed",
        "condition_id",
        "initial_partition",
        "final_real",
        "best_real",
        "final_cut_edges",
    ] {
        assert!(
            value.get(forbidden).is_none(),
            "unexpected redundant field {forbidden}"
        );
    }
    assert!(
        value["records"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r.get("current_real").is_none() && r.get("best_real").is_none())
    );

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("bad.json");
    let mut bad = value;
    bad["partitions"][0] = json!([true]);
    std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
    assert!(read_result(&path, &graph, &c).is_err());

    let mut wrong_final = result.clone();
    wrong_final.final_solution = wrong_final.records[0].current_solution;
    if wrong_final.final_solution != wrong_final.records.last().unwrap().current_solution {
        assert!(wrong_final.validate(&graph, &c).is_err());
    }
    let mut missing = result.clone();
    if missing.records.len() > 2 {
        missing.records.remove(1);
        assert!(missing.validate(&graph, &c).is_err());
    }
    let mut extra_partition = result.clone();
    extra_partition
        .partitions
        .push(vec![false; graph.node_count]);
    assert!(extra_partition.validate(&graph, &c).is_err());
}
