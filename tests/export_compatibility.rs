// The frozen source resolves these crate paths as it did in the library.
pub use gpp_utils::{error, experiment, graph_partition, storage};
#[path = "reference/export_2aae96a.rs"]
mod original;

use gpp_utils::{
    compile_experiment,
    export::export_tsv,
    fitness::FitnessRegistry,
    optimization::CancellationToken,
    storage::{RuntimeOptions, incomplete_path, result_path, run_batch},
};
use serde_json::{Value, json};

fn plan(diagnostics: bool, basins: bool) -> gpp_utils::ExperimentPlan {
    let smoothing = json!([
        {"kind":"none"}, {"kind":"all_average"},
        {"kind":"random_k_average","ks":[1]},
        {"kind":"weighted_average","ks":[0,1]}
    ]);
    compile_experiment(serde_json::from_value(json!({
        "schema_version":1, "run_seeds":[7], "neighborhoods":["flip","swap"],
        "budget":{"max_steps":3},
        "measurement":{"schedule":"explicit","steps":[1],"basin":if basins {"both"} else {"none"},"best_basin":basins,"diagnostics":diagnostics},
        "graphs":[{"kind":"random","node_counts":[4,6],"expected_degrees":[0.0,2.0],"seeds":[42]}],
        "solvers":[{"kind":"hc","smoothing":smoothing},
                   {"kind":"sa","temperatures":[0.0,1.0],"smoothing":smoothing},
                   {"kind":"eo","taus":[1.5]}]
    })).unwrap()).unwrap()
}

fn compare(plan: &gpp_utils::ExperimentPlan, root: &std::path::Path, include: bool) {
    let old = tempfile::tempdir().unwrap();
    let new = tempfile::tempdir().unwrap();
    original::export_tsv(plan, root, old.path(), include, false).unwrap();
    export_tsv(plan, root, new.path(), include, false).unwrap();
    for name in ["runs.tsv", "traces.tsv"] {
        let a = std::fs::read_to_string(old.path().join(name)).unwrap();
        let b = std::fs::read_to_string(new.path().join(name)).unwrap();
        assert_eq!(a.lines().count(), b.lines().count(), "{name} row count");
        for (line, (a, b)) in a.lines().zip(b.lines()).enumerate() {
            assert_eq!(a, b, "{name} row {line}");
        }
        assert_eq!(a.as_bytes(), b.as_bytes(), "{name} newline/encoding");
    }
    let old: Value =
        serde_json::from_slice(&std::fs::read(old.path().join("metadata.json")).unwrap()).unwrap();
    let new: Value =
        serde_json::from_slice(&std::fs::read(new.path().join("metadata.json")).unwrap()).unwrap();
    for table in ["runs", "traces"] {
        let a = old["columns"][table].as_array().unwrap();
        let b = new["columns"][table].as_array().unwrap();
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            for key in ["name", "type", "unit"] {
                assert_eq!(a[key], b[key], "{table} column {} {key}", a["name"]);
            }
            assert!(!b["meaning"].as_str().unwrap().is_empty());
        }
    }
}

#[test]
fn typed_export_preserves_complete_and_partial_output_bytes() {
    for (diagnostics, basins) in [(true, true), (false, false)] {
        let plan = plan(diagnostics, basins);
        let root = tempfile::tempdir().unwrap();
        let options = RuntimeOptions {
            root: root.path().into(),
            threads: 2,
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
        assert_eq!(summary.completed, 128);
        compare(&plan, root.path(), false);
        // Transform results into valid partial checkpoints with an unmeasured
        // endpoint. The source results remain identical for both exporters.
        for job in &plan.jobs {
            let path = result_path(root.path(), job);
            let mut value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            let attempt = value["attempt_id"].clone();
            let object = value.as_object_mut().unwrap();
            for key in ["schema_version", "attempt_id", "termination"] {
                object.remove(key);
            }
            let endpoint = value["records"]
                .as_array_mut()
                .unwrap()
                .last_mut()
                .unwrap()
                .as_object_mut()
                .unwrap();
            for key in [
                "current_smoothed",
                "search_evaluation",
                "basin_real",
                "basin_smoothed",
                "basin_best",
            ] {
                endpoint.remove(key);
            }
            let marker = json!({"schema_version":1,"attempt_id":attempt,"status":"cancelled","partial_result":value});
            std::fs::write(
                incomplete_path(root.path(), job),
                serde_json::to_vec(&marker).unwrap(),
            )
            .unwrap();
            std::fs::remove_file(path).unwrap();
        }
        compare(&plan, root.path(), true);
        compare(&plan, root.path(), false);
    }
}

#[test]
fn export_validates_all_sources_before_replacing_any_destination() {
    let plan = compile_experiment(
        serde_json::from_value(json!({
            "schema_version":1,
            "run_seeds":[1,2],
            "neighborhoods":["flip"],
            "budget":{"max_steps":1},
            "measurement":{"schedule":"explicit","steps":[1],"basin":"none","diagnostics":false},
            "graphs":[{"kind":"random","node_counts":[4],"expected_degrees":[1.0],"seeds":[3]}],
            "solvers":[{"kind":"sa","temperatures":[1.0],"smoothing":[{"kind":"none"}]}]
        }))
        .unwrap(),
    )
    .unwrap();
    let root = tempfile::tempdir().unwrap();
    let options = RuntimeOptions {
        root: root.path().into(),
        threads: 1,
        ..Default::default()
    };
    run_batch(
        &plan,
        &options,
        &CancellationToken::new(),
        &FitnessRegistry::default(),
        &|_| {},
    )
    .unwrap();
    let mut jobs = plan.jobs.iter().collect::<Vec<_>>();
    jobs.sort_by_key(|job| (&job.condition_id, job.seed));
    std::fs::write(result_path(root.path(), jobs[1]), b"{broken").unwrap();

    let out = tempfile::tempdir().unwrap();
    for name in ["runs.tsv", "traces.tsv", "metadata.json"] {
        std::fs::write(out.path().join(name), format!("old-{name}")).unwrap();
    }
    assert!(export_tsv(&plan, root.path(), out.path(), false, true).is_err());
    for name in ["runs.tsv", "traces.tsv", "metadata.json"] {
        assert_eq!(
            std::fs::read_to_string(out.path().join(name)).unwrap(),
            format!("old-{name}")
        );
    }
}
