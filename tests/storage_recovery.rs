use gpp_utils::{
    experiment::plan::compile_experiment,
    fitness::FitnessRegistry,
    optimization::CancellationToken,
    storage::{RuntimeOptions, incomplete_path, inspect, result_path, run_batch},
};
use serde_json::json;

fn setup() -> (
    tempfile::TempDir,
    gpp_utils::experiment::plan::ExperimentPlan,
    RuntimeOptions,
) {
    let spec = serde_json::from_value(json!({
        "schema_version": 1, "run_seeds": [4], "neighborhoods": ["flip"],
        "problem": {"alpha": 0.2}, "budget": {"max_steps": 3},
        "measurement": {"schedule":"explicit", "steps":[1], "basin":"none", "max_basin_steps":10, "diagnostics":false},
        "graphs": [{"kind":"random", "node_counts":[6], "expected_degrees":[2.0], "seeds":[8]}],
        "solvers": [{"kind":"sa", "temperatures":[0.5], "smoothing":[{"kind":"none"}]}]
    })).unwrap();
    let plan = compile_experiment(spec).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let options = RuntimeOptions {
        root: temp.path().into(),
        threads: 1,
        overwrite: false,
        recover_corrupt: false,
        rounds: false,
        round_deadline: None,
    };
    (temp, plan, options)
}
fn run(
    plan: &gpp_utils::experiment::plan::ExperimentPlan,
    options: &RuntimeOptions,
) -> gpp_utils::storage::BatchSummary {
    run_batch(
        plan,
        options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap()
}

#[test]
fn valid_result_survives_bad_marker_and_is_reused() {
    for marker in [
        b"{broken".to_vec(),
        serde_json::to_vec(&json!({"schema_version":99})).unwrap(),
    ] {
        let (temp, plan, options) = setup();
        assert_eq!(run(&plan, &options).completed, 1);
        let result = result_path(temp.path(), &plan.jobs[0]);
        let old = std::fs::read(&result).unwrap();
        let incomplete = incomplete_path(temp.path(), &plan.jobs[0]);
        std::fs::write(&incomplete, &marker).unwrap();
        let rows = inspect(&plan, temp.path(), true).unwrap();
        assert_eq!(rows[0].status, "completed");
        assert_eq!(rows[0].error.as_ref().unwrap().code, "marker_issue");
        assert_eq!(run(&plan, &options).reused, 1);
        assert_eq!(std::fs::read(result).unwrap(), old);
        assert_eq!(std::fs::read(incomplete).unwrap(), marker);
    }
}

#[test]
fn invalid_marker_without_result_requires_recovery_and_is_backed_up() {
    let (temp, plan, mut options) = setup();
    let marker = incomplete_path(temp.path(), &plan.jobs[0]);
    std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
    std::fs::write(&marker, b"{broken").unwrap();
    assert_eq!(run(&plan, &options).failed, 1);
    assert_eq!(std::fs::read(&marker).unwrap(), b"{broken");
    options.recover_corrupt = true;
    assert_eq!(run(&plan, &options).completed, 1);
    let backup = std::fs::read_dir(marker.parent().unwrap())
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("corrupt-")
                .then_some(path)
        })
        .collect::<Vec<_>>();
    assert_eq!(backup.len(), 1);
    assert_eq!(std::fs::read(&backup[0]).unwrap(), b"{broken");
}

#[test]
fn future_marker_and_result_are_preserved_on_overwrite() {
    let (temp, plan, mut options) = setup();
    assert_eq!(run(&plan, &options).completed, 1);
    let result = result_path(temp.path(), &plan.jobs[0]);
    let marker = incomplete_path(temp.path(), &plan.jobs[0]);
    let future_marker =
        serde_json::to_vec(&json!({"schema_version":99,"attempt_id":"future","status":"running"}))
            .unwrap();
    std::fs::write(&marker, &future_marker).unwrap();
    options.overwrite = true;
    assert_eq!(run(&plan, &options).failed, 1);
    assert_eq!(std::fs::read(&marker).unwrap(), future_marker);
    let mut future_result: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&result).unwrap()).unwrap();
    future_result["schema_version"] = json!(99);
    let future_result = serde_json::to_vec(&future_result).unwrap();
    std::fs::write(&result, &future_result).unwrap();
    std::fs::remove_file(&marker).unwrap();
    options.recover_corrupt = true;
    assert_eq!(run(&plan, &options).failed, 1);
    assert_eq!(std::fs::read(result).unwrap(), future_result);
}

#[test]
fn future_marker_without_result_blocks_recovery() {
    let (temp, plan, mut options) = setup();
    let marker = incomplete_path(temp.path(), &plan.jobs[0]);
    std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
    let bytes =
        serde_json::to_vec(&json!({"schema_version":99,"attempt_id":"future","status":"running"}))
            .unwrap();
    std::fs::write(&marker, &bytes).unwrap();
    options.recover_corrupt = true;
    assert_eq!(run(&plan, &options).failed, 1);
    assert_eq!(std::fs::read(&marker).unwrap(), bytes);
    assert!(!result_path(temp.path(), &plan.jobs[0]).exists());
}

#[test]
fn matching_stale_marker_is_removed_on_reuse() {
    let (temp, plan, options) = setup();
    assert_eq!(run(&plan, &options).completed, 1);
    let result = result_path(temp.path(), &plan.jobs[0]);
    let result: serde_json::Value =
        serde_json::from_slice(&std::fs::read(result).unwrap()).unwrap();
    let marker = incomplete_path(temp.path(), &plan.jobs[0]);
    std::fs::write(
        &marker,
        serde_json::to_vec(&json!({
            "schema_version":1, "attempt_id":result["attempt_id"], "status":"running"
        }))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(run(&plan, &options).reused, 1);
    assert!(!marker.exists());
}

#[test]
fn corrupt_completed_result_requires_recovery_and_keeps_backup() {
    let (temp, plan, mut options) = setup();
    assert_eq!(run(&plan, &options).completed, 1);
    let result = result_path(temp.path(), &plan.jobs[0]);
    std::fs::write(&result, b"{broken").unwrap();
    assert_eq!(run(&plan, &options).failed, 1);
    assert_eq!(std::fs::read(&result).unwrap(), b"{broken");
    options.recover_corrupt = true;
    assert_eq!(run(&plan, &options).completed, 1);
    let backups = std::fs::read_dir(result.parent().unwrap())
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("corrupt-")
                .then_some(path)
        })
        .collect::<Vec<_>>();
    assert_eq!(backups.len(), 1);
    assert_eq!(std::fs::read(&backups[0]).unwrap(), b"{broken");
}

#[test]
fn marker_io_error_is_not_mistaken_for_recoverable_corruption() {
    let (temp, plan, mut options) = setup();
    assert_eq!(run(&plan, &options).completed, 1);
    let result = result_path(temp.path(), &plan.jobs[0]);
    let bytes = std::fs::read(&result).unwrap();
    let marker = incomplete_path(temp.path(), &plan.jobs[0]);
    // A directory at a file path produces an I/O error on Windows and Linux.
    std::fs::create_dir(&marker).unwrap();
    options.overwrite = true;
    options.recover_corrupt = true;
    assert_eq!(run(&plan, &options).failed, 1);
    assert!(inspect(&plan, temp.path(), true).is_err());
    assert!(marker.is_dir());
    assert_eq!(std::fs::read(&result).unwrap(), bytes);
    assert!(
        std::fs::read_dir(result.parent().unwrap())
            .unwrap()
            .all(|entry| {
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains("corrupt-")
            })
    );
}

#[test]
fn graph_prepare_failure_persists_failed_marker_without_changing_graph() {
    let (temp, plan, options) = setup();
    let graph = temp
        .path()
        .join("graphs")
        .join(format!("{}.json", plan.jobs[0].graph_id));
    std::fs::create_dir_all(graph.parent().unwrap()).unwrap();
    let future = serde_json::to_vec(&json!({"schema_version":99})).unwrap();
    std::fs::write(&graph, &future).unwrap();
    assert_eq!(run(&plan, &options).failed, 1);
    assert_eq!(std::fs::read(&graph).unwrap(), future);
    let marker = incomplete_path(temp.path(), &plan.jobs[0]);
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
    assert_eq!(data["status"], "failed");
    assert!(
        data["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unsupported schema")
    );
}
