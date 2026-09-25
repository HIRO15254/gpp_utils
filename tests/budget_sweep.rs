use gpp_utils::{ExperimentSpec, compile_experiment};

fn compile_toml(text: &str) -> gpp_utils::ExperimentPlan {
    compile_experiment(toml::from_str::<ExperimentSpec>(text).unwrap()).unwrap()
}

fn base() -> &'static str {
    r#"
schema_version = 1
run_seeds = [0]

[budget]
max_steps = 10

[[graphs]]
kind = "random"
node_counts = [4]
expected_degrees = [2.0]
seeds = [7]
"#
}

#[test]
fn scalar_budget_keeps_the_existing_sample_identity() {
    let spec: ExperimentSpec = toml::from_str(gpp_utils::experiment::plan::sample_toml()).unwrap();
    let plan = compile_experiment(spec).unwrap();
    // Pinned for `versions.algorithm = "v2"`; the v1 identity was
    // e9bb99e4b662c118c11994b98e5d6a192843c112e39dae07521850beb0fb089a.
    assert_eq!(plan.experiment.versions["algorithm"], "v2");
    assert_eq!(
        plan.batch_id,
        "be59ca5ce69255699c2e8c99251c1f0338be5769938371d04f3e0702a3091449"
    );
    assert!(
        !serde_json::to_string(&plan.experiment.spec)
            .unwrap()
            .contains("best_basin")
    );
}

#[test]
fn toml_and_json_budget_sweeps_expand_the_same_plan() {
    let toml = format!(
        "{}\n\n[budget]\nmax_steps = [30, 10]\n\n[[solvers]]\nkind = \"hc\"",
        base().replace("[budget]\nmax_steps = 10\n", "").replace(
            "run_seeds = [0]",
            "run_seeds = [0]\nneighborhoods = [\"flip\"]"
        )
    );
    let spec: ExperimentSpec = toml::from_str(&toml).unwrap();
    let json = serde_json::to_string(&spec).unwrap();
    let a = compile_experiment(spec).unwrap();
    let b = compile_experiment(serde_json::from_str(&json).unwrap()).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.jobs.len(), 2);
    assert_eq!(
        a.jobs
            .iter()
            .map(|job| job.condition.budget.max_steps)
            .collect::<Vec<_>>(),
        vec![10, 30]
    );
}

#[test]
fn conditions_can_target_distinct_axes_and_budgets() {
    let plan = compile_toml(&format!(
        "{}\n[[conditions]]\nneighborhoods = [\"flip\"]\n\n[conditions.budget]\nmax_steps = [5, 10]\n\n[[conditions.solvers]]\nkind = \"hc\"\n\n[[conditions]]\nneighborhoods = [\"swap\"]\n\n[[conditions.solvers]]\nkind = \"sa\"\ntemperatures = [0.5]",
        base()
    ));
    assert_eq!(plan.jobs.len(), 3);
    let actual = plan
        .jobs
        .iter()
        .map(|job| {
            (
                format!("{:?}", job.condition.neighborhood),
                job.condition.budget.max_steps,
            )
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        actual,
        std::collections::BTreeSet::from([
            ("Flip".to_owned(), 5),
            ("Flip".to_owned(), 10),
            ("Swap".to_owned(), 10),
        ])
    );
}

#[test]
fn invalid_budget_and_condition_combinations_are_rejected() {
    for budget in ["[]", "[10, 10]", "[0, 10]", "0"] {
        let text = format!(
            "neighborhoods = [\"flip\"]\n{}\n[[solvers]]\nkind = \"hc\"",
            base().replace("max_steps = 10", &format!("max_steps = {budget}"))
        );
        let spec: ExperimentSpec = toml::from_str(&text).unwrap();
        assert!(
            compile_experiment(spec)
                .unwrap_err()
                .to_string()
                .contains("max_steps")
        );
    }

    let mixed_axes = format!(
        "neighborhoods = [\"flip\"]\n{}\n[[solvers]]\nkind = \"hc\"\n\n[[conditions]]\nneighborhoods = [\"flip\"]\n\n[[conditions.solvers]]\nkind = \"hc\"",
        base()
    );
    let spec = toml::from_str::<ExperimentSpec>(&mixed_axes).unwrap();
    assert!(
        compile_experiment(spec)
            .unwrap_err()
            .to_string()
            .contains("when conditions are specified")
    );

    let duplicate = format!(
        "{}\n[[conditions]]\nneighborhoods = [\"flip\"]\n\n[[conditions.solvers]]\nkind = \"hc\"\n\n[[conditions]]\nneighborhoods = [\"flip\"]\n\n[[conditions.solvers]]\nkind = \"hc\"",
        base()
    );
    let spec = toml::from_str::<ExperimentSpec>(&duplicate).unwrap();
    assert!(
        compile_experiment(spec)
            .unwrap_err()
            .to_string()
            .contains("duplicate effective condition")
    );
}

#[test]
fn one_budget_in_an_array_normalizes_to_the_existing_scalar() {
    let scalar = format!(
        "neighborhoods = [\"flip\"]\n{}\n[[solvers]]\nkind = \"hc\"",
        base()
    );
    assert_eq!(
        compile_toml(&scalar),
        compile_toml(&scalar.replace("max_steps = 10", "max_steps = [10]"))
    );
}

#[test]
fn explicit_measurements_must_fit_every_effective_budget() {
    let text = format!(
        "{}\n[measurement]\nschedule = \"explicit\"\nsteps = [8]\n\n[[conditions]]\nneighborhoods = [\"flip\"]\n\n[conditions.budget]\nmax_steps = [5, 10]\n\n[[conditions.solvers]]\nkind = \"hc\"",
        base()
    );
    let spec: ExperimentSpec = toml::from_str(&text).unwrap();
    assert!(compile_experiment(spec).is_err());
}

#[test]
fn condition_order_does_not_change_identity() {
    let first = format!(
        "{}\n[[conditions]]\nneighborhoods = [\"flip\"]\n\n[[conditions.solvers]]\nkind = \"hc\"\n\n[[conditions]]\nneighborhoods = [\"swap\"]\n\n[[conditions.solvers]]\nkind = \"sa\"\ntemperatures = [0.5]",
        base()
    );
    let second = format!(
        "{}\n[[conditions]]\nneighborhoods = [\"swap\"]\n\n[[conditions.solvers]]\nkind = \"sa\"\ntemperatures = [0.5]\n\n[[conditions]]\nneighborhoods = [\"flip\"]\n\n[[conditions.solvers]]\nkind = \"hc\"",
        base()
    );
    let a = compile_toml(&first);
    let b = compile_toml(&second);
    assert_eq!(a.batch_id, b.batch_id);
    assert_eq!(a.jobs, b.jobs);
}
