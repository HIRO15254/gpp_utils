use gpp_utils::{
    compile_experiment,
    experiment::result::RunResult,
    fitness::FitnessRegistry,
    optimization::CancellationToken,
    storage::{RuntimeOptions, result_path, run_batch},
};
use serde_json::json;
use std::{sync::Mutex, time::Duration};

fn plan() -> gpp_utils::ExperimentPlan {
    let spec = serde_json::from_value(json!({
        "schema_version": 1,
        "run_seeds": [20, 10],
        "neighborhoods": ["flip"],
        "problem": {"alpha": 0.2},
        "budget": {"max_steps": 3},
        "measurement": {
            "schedule":"explicit", "steps":[1], "basin":"none",
            "max_basin_steps":10, "diagnostics":false
        },
        "graphs": [{
            "kind":"random", "node_counts":[6],
            "expected_degrees":[2.0], "seeds":[8]
        }],
        "solvers": [{
            "kind":"sa", "temperatures":[0.5, 1.0],
            "smoothing":[{"kind":"none"}]
        }]
    }))
    .unwrap();
    compile_experiment(spec).unwrap()
}

fn options(root: &std::path::Path) -> RuntimeOptions {
    RuntimeOptions {
        root: root.into(),
        threads: 2,
        rounds: true,
        ..RuntimeOptions::default()
    }
}

#[test]
fn round_events_are_grouped_in_numeric_seed_order() {
    let temp = tempfile::tempdir().unwrap();
    let events = Mutex::new(Vec::new());
    let summary = run_batch(
        &plan(),
        &options(temp.path()),
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|event| events.lock().unwrap().push(event.seed),
    )
    .unwrap();
    assert_eq!(summary.completed_rounds, 2);
    let events = events.into_inner().unwrap();
    assert_eq!(events.len(), 4);
    assert!(events[..2].iter().all(|&seed| seed == 10));
    assert!(events[2..].iter().all(|&seed| seed == 20));
}

#[test]
fn zero_deadline_pauses_without_starting_jobs() {
    let temp = tempfile::tempdir().unwrap();
    let mut options = options(temp.path());
    options.round_deadline = Some(Duration::ZERO);
    let plan = plan();
    let summary = run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    assert!(summary.deadline_reached);
    assert_eq!(summary.not_started, plan.jobs.len());
    assert_eq!(summary.completed_rounds, 0);
    assert!(
        plan.jobs
            .iter()
            .all(|job| !result_path(temp.path(), job).exists())
    );
    assert!(!temp.path().join("runs").exists());
}

#[test]
fn resume_reuses_complete_rounds_and_keeps_result_identity() {
    let temp = tempfile::tempdir().unwrap();
    let plan = plan();
    let options = options(temp.path());
    run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    let ids: Vec<_> = plan
        .jobs
        .iter()
        .map(|job| {
            let result: RunResult =
                serde_json::from_slice(&std::fs::read(result_path(temp.path(), job)).unwrap())
                    .unwrap();
            result.attempt_id
        })
        .collect();
    let resumed = run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    assert_eq!(resumed.reused, plan.jobs.len());
    assert_eq!(resumed.completed_rounds, 2);
    let resumed_ids: Vec<_> = plan
        .jobs
        .iter()
        .map(|job| {
            let result: RunResult =
                serde_json::from_slice(&std::fs::read(result_path(temp.path(), job)).unwrap())
                    .unwrap();
            result.attempt_id
        })
        .collect();
    assert_eq!(resumed_ids, ids);
}

#[test]
fn a_failed_round_blocks_later_seeds() {
    let temp = tempfile::tempdir().unwrap();
    let plan = plan();
    let options = options(temp.path());
    run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    let first = plan.jobs.iter().find(|job| job.seed == 10).unwrap();
    std::fs::write(result_path(temp.path(), first), b"not json").unwrap();
    for job in plan.jobs.iter().filter(|job| job.seed == 20) {
        std::fs::remove_file(result_path(temp.path(), job)).unwrap();
    }
    let summary = run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    assert_eq!(summary.failed, 1);
    assert_eq!(summary.not_started, 2);
    assert!(
        plan.jobs
            .iter()
            .filter(|job| job.seed == 20)
            .all(|job| !result_path(temp.path(), job).exists())
    );
}

#[test]
fn deadline_without_rounds_is_rejected_before_writing() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("untouched");
    let options = RuntimeOptions {
        root: root.clone(),
        round_deadline: Some(Duration::from_secs(1)),
        ..RuntimeOptions::default()
    };
    assert!(
        run_batch(
            &plan(),
            &options,
            &CancellationToken::new(),
            &FitnessRegistry::default(),
            &|_| {},
        )
        .is_err()
    );
    assert!(!root.exists());
}

#[test]
fn round_scheduling_does_not_change_scientific_results() {
    let normal = tempfile::tempdir().unwrap();
    let rounds = tempfile::tempdir().unwrap();
    let plan = plan();
    let normal_options = RuntimeOptions {
        root: normal.path().into(),
        threads: 2,
        ..RuntimeOptions::default()
    };
    run_batch(
        &plan,
        &normal_options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    run_batch(
        &plan,
        &options(rounds.path()),
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    for job in &plan.jobs {
        let mut a: serde_json::Value =
            serde_json::from_slice(&std::fs::read(result_path(normal.path(), job)).unwrap())
                .unwrap();
        let mut b: serde_json::Value =
            serde_json::from_slice(&std::fs::read(result_path(rounds.path(), job)).unwrap())
                .unwrap();
        a.as_object_mut().unwrap().remove("attempt_id");
        b.as_object_mut().unwrap().remove("attempt_id");
        a.as_object_mut().unwrap().remove("elapsed_ms");
        b.as_object_mut().unwrap().remove("elapsed_ms");
        assert_eq!(a, b);
    }
}

#[test]
fn cancellation_takes_precedence_over_a_zero_deadline() {
    let temp = tempfile::tempdir().unwrap();
    let mut options = options(temp.path());
    options.round_deadline = Some(Duration::ZERO);
    let cancel = CancellationToken::new();
    cancel.cancel();
    let plan = plan();
    let summary = run_batch(
        &plan,
        &options,
        &cancel,
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    assert!(!summary.deadline_reached);
    assert_eq!(summary.not_started, plan.jobs.len());
    assert!(cancel.is_cancelled());
}

#[test]
fn deadline_during_a_round_finishes_that_round_then_resume_fills_the_next() {
    let temp = tempfile::tempdir().unwrap();
    let plan = plan();
    let mut options = options(temp.path());
    options.round_deadline = Some(Duration::from_secs(2));
    let delayed = std::sync::atomic::AtomicBool::new(false);
    let summary = run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|event| {
            assert_eq!(event.seed, 10);
            if !delayed.swap(true, std::sync::atomic::Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(2100));
            }
        },
    )
    .unwrap();
    assert_eq!(summary.completed, 2);
    assert_eq!(summary.completed_rounds, 1);
    assert_eq!(summary.not_started, 2);
    assert!(summary.deadline_reached);
    options.round_deadline = None;
    let resumed = run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    assert_eq!(resumed.reused, 2);
    assert_eq!(resumed.completed, 2);
    assert_eq!(resumed.completed_rounds, 2);
}
