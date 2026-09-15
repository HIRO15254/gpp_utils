use gpp_utils::{
    compile_experiment,
    experiment::config::{
        BasinMode, Budget, Condition, GraphKind, GraphSpec, Measurement, Neighborhood, Schedule,
        SmoothingSpec, SolverSpec,
    },
    export::export_tsv,
    fitness::FitnessRegistry,
    graph_partition::Graph,
    optimization::CancellationToken,
    run_one,
    storage::{RuntimeOptions, read_result, result_path, run_batch},
};
use serde_json::json;

fn graph() -> Graph {
    Graph::from_edges(
        6,
        vec![[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5], [0, 3]],
    )
    .unwrap()
}

fn condition(neighborhood: Neighborhood, smoothing: SmoothingSpec) -> Condition {
    Condition {
        graph: GraphSpec {
            kind: GraphKind::Random,
            node_count: 6,
            expected_degree: 2.0,
            seed: 3,
        },
        neighborhood,
        alpha: 0.2,
        solver: SolverSpec::Sa {
            temperature: 0.9,
            smoothing,
        },
        budget: Budget { max_steps: 8 },
        measurement: Measurement {
            schedule: Schedule::Explicit,
            steps: (1..8).collect(),
            basin: BasinMode::None,
            max_basin_steps: 20,
            diagnostics: true,
            best_basin: true,
        },
    }
}

#[test]
fn documented_json_example_has_valid_references_and_matching_table_scores() {
    let doc = include_str!("../docs/output-format.md");
    let json = doc
        .split("```json")
        .nth(1)
        .unwrap()
        .split("```")
        .next()
        .unwrap();
    let result: gpp_utils::RunResult = serde_json::from_str(json).unwrap();
    let graph = Graph::from_edges(4, vec![[0, 1], [1, 2], [2, 3], [0, 3]]).unwrap();
    let mut c = condition(Neighborhood::Swap, SmoothingSpec::None);
    c.graph.node_count = 4;
    c.alpha = 0.05;
    c.solver = SolverSpec::Eo {
        tau: 1.5,
        fitness: Default::default(),
    };
    c.budget.max_steps = 3;
    c.measurement.steps = vec![1];
    c.measurement.basin = BasinMode::Real;
    c.measurement.diagnostics = false;
    result.validate(&graph, &c).unwrap();
    for (record, expected) in result.records.iter().zip([4.0_f64, 2.0, 2.0]) {
        let current = graph.score(&result.partitions[record.current_solution.0], c.alpha);
        let best = graph.score(&result.partitions[record.best_solution.0], c.alpha);
        assert_eq!(current.to_bits(), expected.to_bits());
        assert_eq!(best.to_bits(), expected.to_bits());
        assert_eq!(
            record.basin_real.as_ref().unwrap().real.to_bits(),
            2.0_f64.to_bits()
        );
        assert_eq!(
            record.basin_best.as_ref().unwrap().real.to_bits(),
            2.0_f64.to_bits()
        );
        let table_row = format!(
            "| {} | {current} | {best} | 2 | local_optimum | 2 | local_optimum |",
            record.step
        );
        assert!(doc.contains(&table_row));
    }
}

#[test]
fn best_basin_is_real_and_independent_of_current_basin_mode() {
    let graph = graph();
    for (neighborhood, smoothing) in [
        (Neighborhood::Flip, SmoothingSpec::RandomKAverage { k: 3 }),
        (Neighborhood::Swap, SmoothingSpec::AllAverage),
    ] {
        let c = condition(neighborhood, smoothing);
        let result = run_one(
            &graph,
            &c,
            44,
            &CancellationToken::new(),
            &FitnessRegistry::default(),
        )
        .unwrap();
        result.validate(&graph, &c).unwrap();
        assert!(result.records.iter().all(|r| {
            r.basin_real.is_none()
                && r.basin_smoothed.is_none()
                && r.basin_best.as_ref().is_some_and(|b| b.smoothed.is_none())
        }));
    }
}

#[test]
fn repeated_incumbents_reuse_identical_best_basin_and_disabled_rejects_it() {
    let graph = graph();
    let c = condition(Neighborhood::Flip, SmoothingSpec::None);
    let result = run_one(
        &graph,
        &c,
        91,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
    )
    .unwrap();
    for pair in result.records.windows(2) {
        if pair[0].best_solution == pair[1].best_solution {
            let a = pair[0].basin_best.as_ref().unwrap();
            let b = pair[1].basin_best.as_ref().unwrap();
            assert_eq!(
                (a.real, a.termination, a.steps),
                (b.real, b.termination, b.steps)
            );
        }
    }
    let mut disabled = c.clone();
    disabled.measurement.best_basin = false;
    assert!(result.validate(&graph, &disabled).is_err());
}

#[test]
fn enabling_best_basin_does_not_change_search_trajectory() {
    let graph = graph();
    let enabled = condition(Neighborhood::Flip, SmoothingSpec::RandomKAverage { k: 3 });
    let mut disabled = enabled.clone();
    disabled.measurement.best_basin = false;
    let registry = FitnessRegistry::default();
    let a = run_one(&graph, &disabled, 7, &CancellationToken::new(), &registry).unwrap();
    let b = run_one(&graph, &enabled, 7, &CancellationToken::new(), &registry).unwrap();
    assert_eq!(a.best_step, b.best_step);
    assert_eq!(a.records.len(), b.records.len());
    for (ra, rb) in a.records.iter().zip(&b.records) {
        assert_eq!(ra.step, rb.step);
        assert_eq!(
            a.partitions[ra.current_solution.0],
            b.partitions[rb.current_solution.0]
        );
        assert_eq!(
            a.partitions[ra.best_solution.0],
            b.partitions[rb.best_solution.0]
        );
        assert_eq!(ra.current_smoothed, rb.current_smoothed);
        assert_eq!(ra.search_evaluation, rb.search_evaluation);
    }
    assert!(
        b.diagnostics
            .as_ref()
            .unwrap()
            .objective_evaluations_measurement
            > a.diagnostics
                .as_ref()
                .unwrap()
                .objective_evaluations_measurement
    );
}

#[test]
fn unchanged_incumbent_is_measured_once_and_frequency_does_not_change_it() {
    let graph = Graph::from_edges(6, vec![]).unwrap();
    let mut dense = condition(Neighborhood::Flip, SmoothingSpec::None);
    dense.alpha = 0.0;
    dense.solver = SolverSpec::Sa {
        temperature: 0.0,
        smoothing: SmoothingSpec::None,
    };
    let mut sparse = dense.clone();
    sparse.measurement.steps = vec![4];
    let registry = FitnessRegistry::default();
    let a = run_one(&graph, &dense, 12, &CancellationToken::new(), &registry).unwrap();
    let b = run_one(&graph, &sparse, 12, &CancellationToken::new(), &registry).unwrap();
    assert_eq!(
        a.diagnostics
            .as_ref()
            .unwrap()
            .objective_evaluations_measurement,
        7
    );
    assert_eq!(
        b.diagnostics
            .as_ref()
            .unwrap()
            .objective_evaluations_measurement,
        7
    );
    assert!(a.records.iter().all(|r| r.best_solution == a.best_solution));
    let expected = a.records[0].basin_best.as_ref().unwrap().real.to_bits();
    assert!(
        a.records
            .iter()
            .all(|r| r.basin_best.as_ref().unwrap().real.to_bits() == expected)
    );
    assert_eq!(
        a.partitions[a.final_solution.0],
        b.partitions[b.final_solution.0]
    );
    assert_eq!(
        a.records
            .last()
            .unwrap()
            .basin_best
            .as_ref()
            .unwrap()
            .real
            .to_bits(),
        b.records
            .last()
            .unwrap()
            .basin_best
            .as_ref()
            .unwrap()
            .real
            .to_bits()
    );
}

#[test]
fn malformed_best_basins_are_rejected() {
    let graph = graph();
    let c = condition(Neighborhood::Flip, SmoothingSpec::None);
    let result = run_one(
        &graph,
        &c,
        1,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
    )
    .unwrap();
    let mut missing = result.clone();
    missing.records[0].basin_best = None;
    assert!(missing.validate(&graph, &c).is_err());
    let mut smoothed = result.clone();
    smoothed.records[0].basin_best.as_mut().unwrap().smoothed = Some(0.0);
    assert!(smoothed.validate(&graph, &c).is_err());
    let mut steps = result.clone();
    steps.records[0].basin_best.as_mut().unwrap().steps = Some(c.measurement.max_basin_steps + 1);
    assert!(steps.validate(&graph, &c).is_err());
    let mut worse = result.clone();
    let incumbent = graph.score(&worse.partitions[worse.records[0].best_solution.0], c.alpha);
    worse.records[0].basin_best.as_mut().unwrap().real = incumbent + 1.0;
    assert!(worse.validate(&graph, &c).is_err());
}

#[test]
fn best_basin_survives_storage_and_is_exported_with_typed_metadata() {
    let spec = serde_json::from_value(json!({
        "schema_version": 1, "run_seeds": [4], "neighborhoods": ["flip"],
        "problem": {"alpha": 0.2}, "budget": {"max_steps": 3},
        "measurement": {"schedule":"explicit", "steps":[1], "basin":"none", "best_basin":true, "max_basin_steps":10, "diagnostics":true},
        "graphs": [{"kind":"random", "node_counts":[6], "expected_degrees":[2.0], "seeds":[8]}],
        "solvers": [{"kind":"sa", "temperatures":[0.5], "smoothing":[{"kind":"none"}]}]
    }))
    .unwrap();
    let plan = compile_experiment(spec).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let options = RuntimeOptions {
        root: temp.path().join("data"),
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
    let job = &plan.jobs[0];
    let stored = read_result(
        &result_path(&options.root, job),
        &Graph::generate(&job.condition.graph, &CancellationToken::new()).unwrap(),
        &job.condition,
    )
    .unwrap();
    assert!(stored.records.iter().all(|r| r.basin_best.is_some()));
    let out = temp.path().join("export");
    export_tsv(&plan, &options.root, &out, false, false).unwrap();
    let runs = std::fs::read_to_string(out.join("runs.tsv")).unwrap();
    let traces = std::fs::read_to_string(out.join("traces.tsv")).unwrap();
    assert!(runs.lines().next().unwrap().contains("max_steps"));
    assert!(
        runs.lines()
            .next()
            .unwrap()
            .contains("final_basin_real_from_best")
    );
    assert!(traces.lines().next().unwrap().contains("basin_best_steps"));
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("metadata.json")).unwrap()).unwrap();
    let runs_columns = metadata["columns"]["runs"].as_array().unwrap();
    assert!(
        runs_columns
            .iter()
            .any(|c| c["name"] == "max_steps" && c["type"] == "integer")
    );
    assert!(
        runs_columns
            .iter()
            .any(|c| c["name"] == "final_basin_real_from_best" && c["type"] == "number")
    );
}
