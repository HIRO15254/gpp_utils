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
