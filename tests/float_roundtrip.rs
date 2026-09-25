//! Stored JSON must read every `f64` back bit-exactly. Otherwise condition IDs
//! recomputed from `experiment.json` drift from the submitted settings and a
//! resumed batch no longer finds its results (serde_json `float_roundtrip`).
use gpp_utils::{
    ExperimentSpec, compile_experiment,
    fitness::FitnessRegistry,
    optimization::CancellationToken,
    storage::{RuntimeOptions, load_plan, run_batch},
};

/// `10^(k/1000)` for k = 17, 21, 50: parsed one ULP off without the feature.
const AWKWARD: [f64; 3] = [1.0399201658290593, 1.0495424286523223, 1.1220184543019633];

#[test]
fn json_floats_round_trip_bit_exactly() {
    let powers = (-2000..=2000)
        .step_by(7)
        .map(|k| 10f64.powf(f64::from(k) / 1000.0));
    for x in AWKWARD.into_iter().chain(powers) {
        let text = serde_json::to_string(&x).unwrap();
        let back: f64 = serde_json::from_str(&text).unwrap();
        assert_eq!(back.to_bits(), x.to_bits(), "{text}");
    }
}

#[test]
fn stored_experiment_reproduces_condition_ids() {
    let temperatures = AWKWARD
        .iter()
        .map(|t| format!("{t:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    let spec: ExperimentSpec = toml::from_str(&format!(
        r#"
schema_version = 1
run_seeds = [0]
neighborhoods = ["flip"]
[budget]
max_steps = 10
[measurement]
basin = "none"
[[graphs]]
kind = "random"
node_counts = [8]
expected_degrees = [3.0]
seeds = [0]
[[solvers]]
kind = "sa"
temperatures = [{temperatures}]
"#
    ))
    .unwrap();
    let plan = compile_experiment(spec).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let options = RuntimeOptions {
        root: dir.path().into(),
        threads: 1,
        ..Default::default()
    };
    let summary = run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    assert_eq!(summary.completed, plan.jobs.len());
    let restored = load_plan(dir.path(), &plan.batch_id).unwrap();
    assert_eq!(restored.jobs, plan.jobs);
}
