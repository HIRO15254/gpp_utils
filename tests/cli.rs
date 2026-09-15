use std::process::Command;

fn gpp() -> Command {
    Command::new(env!("CARGO_BIN_EXE_gpp"))
}

#[test]
fn help_and_init_work() {
    assert!(gpp().arg("--help").status().unwrap().success());
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("experiment.toml");
    assert!(
        gpp()
            .args(["init", config.to_str().unwrap()])
            .status()
            .unwrap()
            .success()
    );
    assert!(config.exists());
    assert!(
        !gpp()
            .args(["init", config.to_str().unwrap()])
            .status()
            .unwrap()
            .success()
    );
}

#[test]
fn validate_and_run_minimal_experiment() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("minimal.toml");
    std::fs::copy("examples/configs/minimal.toml", &config).unwrap();
    let validate = gpp()
        .args(["validate", config.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(validate.status.success());
    assert!(
        serde_json::from_slice::<serde_json::Value>(&validate.stdout).unwrap()["valid"]
            .as_bool()
            .unwrap()
    );
    let root = temp.path().join("data");
    let run = gpp()
        .args([
            "run",
            config.to_str().unwrap(),
            "--root",
            root.to_str().unwrap(),
            "--threads",
            "1",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(run.status.success());
    let summary: serde_json::Value = serde_json::from_slice(&run.stdout).unwrap();
    let batch = summary["batch_id"].as_str().unwrap();
    let inspect = gpp()
        .args([
            "inspect",
            "--batch",
            batch,
            "--root",
            root.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(inspect.status.success());
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&inspect.stdout).unwrap()["jobs"][0]["status"],
        "completed"
    );
}

#[test]
fn grouped_budget_rounds_pause_resume_inspect_and_export() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("data");
    let root = root.to_str().unwrap();
    let bad = gpp()
        .args([
            "run",
            "examples/configs/rounds.toml",
            "--root",
            root,
            "--deadline-seconds",
            "0",
        ])
        .output()
        .unwrap();
    assert_eq!(bad.status.code(), Some(2));
    assert!(!std::path::Path::new(root).exists());
    let paused = gpp()
        .args([
            "run",
            "examples/configs/rounds.toml",
            "--root",
            root,
            "--rounds",
            "--deadline-seconds",
            "0",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        paused.status.success(),
        "{}",
        String::from_utf8_lossy(&paused.stderr)
    );
    let paused: serde_json::Value = serde_json::from_slice(&paused.stdout).unwrap();
    assert_eq!(paused["deadline_reached"], true);
    assert_eq!(paused["not_started"], 24);
    let batch = paused["batch_id"].as_str().unwrap();
    for expected in ["completed", "reused"] {
        let run = gpp()
            .args([
                "resume",
                "--batch",
                batch,
                "--root",
                root,
                "--rounds",
                "--threads",
                "2",
                "--json",
            ])
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let result: serde_json::Value = serde_json::from_slice(&run.stdout).unwrap();
        assert_eq!(result[expected], 24);
        assert_eq!(result["completed_rounds"], 3);
    }
    let inspect = gpp()
        .args(["inspect", "--batch", batch, "--root", root, "--json"])
        .output()
        .unwrap();
    assert!(inspect.status.success());
    let inspect: serde_json::Value = serde_json::from_slice(&inspect.stdout).unwrap();
    let jobs = inspect["jobs"].as_array().unwrap();
    assert_eq!(jobs.len(), 24);
    assert!(
        jobs.iter()
            .all(|job| job["status"] == "completed" && job["budget_max_steps"].as_u64().is_some())
    );
    let out = temp.path().join("export");
    let export = gpp()
        .args([
            "export",
            "--batch",
            batch,
            "--root",
            root,
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        export.status.success(),
        "{}",
        String::from_utf8_lossy(&export.stderr)
    );
    let runs = std::fs::read_to_string(out.join("runs.tsv")).unwrap();
    assert_eq!(runs.lines().count(), 25);
    assert!(
        runs.lines()
            .next()
            .unwrap()
            .split('\t')
            .any(|column| column == "max_steps")
    );
    assert!(
        std::fs::read_to_string(out.join("traces.tsv"))
            .unwrap()
            .contains("basin_real_from_best")
    );
    assert!(out.join("metadata.json").exists());
}
