use gpp_utils::{
    experiment::{
        config::{
            BasinMode, Budget, Condition, GraphKind, GraphSpec, Measurement, Neighborhood,
            Schedule, SmoothingSpec, SolverSpec,
        },
        result::{MeasurementRecord, RunResult, RunTermination, SolutionId},
    },
    graph_partition::Graph,
};

fn condition() -> Condition {
    Condition {
        graph: GraphSpec {
            kind: GraphKind::Random,
            node_count: 2,
            expected_degree: 1.0,
            seed: 1,
        },
        neighborhood: Neighborhood::Flip,
        alpha: 0.5,
        solver: SolverSpec::Hc {
            smoothing: SmoothingSpec::WeightedAverage { k: 0 },
        },
        budget: Budget { max_steps: 1 },
        measurement: Measurement {
            schedule: Schedule::Explicit,
            steps: vec![],
            basin: BasinMode::None,
            max_basin_steps: 1,
            diagnostics: false,
            best_basin: false,
        },
    }
}

#[test]
fn typed_view_derives_identity_smoothing_and_checked_partitions() {
    let graph = Graph::from_edges(2, vec![[0, 1]]).unwrap();
    let result = RunResult {
        schema_version: 1,
        attempt_id: "view".into(),
        termination: RunTermination::Cancelled,
        completed_steps: 0,
        elapsed_ms: 0.0,
        partitions: vec![vec![true, false]],
        final_solution: SolutionId(0),
        best_solution: SolutionId(0),
        best_step: 0,
        records: vec![MeasurementRecord {
            step: 0,
            current_solution: SolutionId(0),
            best_solution: SolutionId(0),
            current_smoothed: None,
            search_evaluation: None,
            basin_real: None,
            basin_smoothed: None,
            basin_best: None,
        }],
        diagnostics: None,
    };
    let condition = condition();
    let view = gpp_utils::experiment::result::RunView::new(&graph, &condition, &result).unwrap();
    let score = view.breakdown(SolutionId(0));
    assert_eq!(score.real, 1.0);
    assert_eq!(score.cut_edges, 1);
    assert_eq!(score.balance_penalty, 0.0);
    assert!(view.try_partition(SolutionId(1)).is_none());
    let measurement = view.final_measurement();
    assert_eq!(measurement.smoothed_value(), Some(1.0));
    assert_eq!(measurement.search_evaluation(), Some(1.0));
}
