//! Validation, canonical identifiers, and sweep expansion.

use anyhow::{Context, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use super::config::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub condition_id: String,
    pub graph_id: String,
    pub condition: Condition,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredExperiment {
    pub schema_version: u32,
    pub versions: BTreeMap<String, String>,
    pub spec: ExperimentSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExperimentPlan {
    pub batch_id: String,
    pub experiment: StoredExperiment,
    pub jobs: Vec<Job>,
}

pub fn hash<T: Serialize>(value: &T) -> crate::error::Result<String> {
    let value = serde_json::to_value(value)?;
    let mut bytes = Vec::new();
    encode_value(&value, &mut bytes)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn length(bytes: &mut Vec<u8>, length: usize) -> crate::error::Result<()> {
    bytes.extend_from_slice(&u64::try_from(length)?.to_le_bytes());
    Ok(())
}

fn encode_value(value: &Value, bytes: &mut Vec<u8>) -> crate::error::Result<()> {
    match value {
        Value::Null => bytes.push(b'n'),
        Value::Bool(value) => {
            bytes.push(b'b');
            bytes.push(u8::from(*value));
        }
        Value::String(value) => {
            bytes.push(b's');
            length(bytes, value.len())?;
            bytes.extend_from_slice(value.as_bytes());
        }
        Value::Number(number) => {
            let text = number.to_string();
            if text.contains(['.', 'e', 'E']) {
                let mut value: f64 = text.parse()?;
                if value == 0.0 {
                    value = 0.0;
                }
                bytes.push(b'f');
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            } else if let Some(value) = number.as_i64() {
                bytes.push(b'i');
                bytes.extend_from_slice(&value.to_le_bytes());
            } else if let Some(value) = number.as_u64() {
                bytes.push(b'u');
                bytes.extend_from_slice(&value.to_le_bytes());
            } else {
                bail!("unsupported JSON number in canonical hash");
            }
        }
        Value::Array(values) => {
            bytes.push(b'a');
            length(bytes, values.len())?;
            for value in values {
                encode_value(value, bytes)?;
            }
        }
        Value::Object(values) => {
            bytes.push(b'o');
            length(bytes, values.len())?;
            for (key, value) in values {
                encode_value(&Value::String(key.clone()), bytes)?;
                encode_value(value, bytes)?;
            }
        }
    }
    Ok(())
}

pub fn load_spec(path: &Path) -> crate::error::Result<ExperimentSpec> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    match path.extension().and_then(|x| x.to_str()) {
        Some("json") => Ok(serde_json::from_str(&text).context("parse JSON experiment")?),
        Some("toml") => Ok(toml::from_str(&text).context("parse TOML experiment")?),
        _ => bail!("experiment file must have .toml or .json extension"),
    }
}

pub fn sample_toml() -> &'static str {
    include_str!("../../examples/configs/comparison.toml")
}
pub fn minimal_sample_toml() -> &'static str {
    include_str!("../../examples/configs/minimal.toml")
}

pub fn compile_experiment(spec: ExperimentSpec) -> crate::error::Result<ExperimentPlan> {
    compile_experiment_with_versions(
        spec,
        &BTreeMap::from([("default".to_owned(), "good_edge_fraction-v1".to_owned())]),
    )
}

pub fn compile_experiment_with_versions(
    spec: ExperimentSpec,
    fitness_versions: &BTreeMap<String, String>,
) -> crate::error::Result<ExperimentPlan> {
    let mut versions = pinned_versions();
    for kind in requested_fitnesses(&spec) {
        let version = fitness_versions
            .get(&kind)
            .ok_or_else(|| anyhow!("unknown fitness: {kind}"))?;
        versions.insert(format!("fitness:{kind}"), version.clone());
    }
    compile_with_stored(StoredExperiment {
        schema_version: 1,
        versions,
        spec,
    })
}

pub fn compile_stored(stored: StoredExperiment) -> crate::error::Result<ExperimentPlan> {
    compile_with_stored(stored)
}

pub fn compile_stored_with_registry(
    stored: StoredExperiment,
    registry: &crate::fitness::FitnessRegistry,
) -> crate::error::Result<ExperimentPlan> {
    let plan = compile_with_stored(stored)?;
    for job in &plan.jobs {
        if let SolverSpec::Eo { fitness, .. } = &job.condition.solver {
            registry.validate(fitness)?;
            let expected = registry
                .versions()
                .get(&fitness.kind)
                .cloned()
                .ok_or_else(|| anyhow!("unknown fitness: {}", fitness.kind))?;
            let stored_version = plan
                .experiment
                .versions
                .get(&format!("fitness:{}", fitness.kind))
                .expect("validated fitness version");
            if stored_version != &expected {
                bail!("unsupported fitness version for {}", fitness.kind);
            }
        }
    }
    Ok(plan)
}

fn pinned_versions() -> BTreeMap<String, String> {
    [
        ("algorithm", "v1"),
        ("rng", "sha256-mt19937-64-v1"),
        ("generation", "v1"),
        ("measurement", "v1"),
        ("normalization", "v1"),
    ]
    .into_iter()
    .map(|(a, b)| (a.into(), b.into()))
    .collect()
}

fn compile_with_stored(mut stored: StoredExperiment) -> crate::error::Result<ExperimentPlan> {
    if stored.schema_version != 1 || stored.spec.schema_version != 1 {
        bail!("unsupported schema_version (only 1 is supported)");
    }
    normalize_spec(&mut stored.spec)?;
    validate_versions(&stored)?;
    validate_top_level(&stored.spec)?;
    let mut jobs = Vec::new();
    let mut graphs = BTreeSet::new();
    let mut conditions = BTreeSet::new();
    let mut graph_ids = BTreeMap::new();
    let condition_groups = expand_condition_groups(&stored.spec)?;
    for graph in expand_graphs(&stored.spec)? {
        validate_graph(&graph, stored.spec.problem.alpha)?;
        let graph_id = hash(&(&graph, stored.versions.get("generation")))?;
        if !graphs.insert(hash(&graph)?) {
            bail!("duplicate graph specification");
        }
        graph_ids.insert(graph.clone_key(), graph_id);
    }
    for graph in expand_graphs(&stored.spec)? {
        let graph_id = graph_ids
            .remove(&graph.clone_key())
            .expect("expanded graph exists");
        for group in &condition_groups {
            for neighborhood in &group.neighborhoods {
                if matches!(neighborhood, Neighborhood::Swap) && graph.node_count % 2 != 0 {
                    bail!(
                        "swap requires an even node_count (got {})",
                        graph.node_count
                    );
                }
                for solver in expand_solvers(&group.solvers, graph.node_count, neighborhood)? {
                    for budget in &group.budgets {
                        let condition = Condition {
                            graph: graph.clone(),
                            neighborhood: *neighborhood,
                            alpha: stored.spec.problem.alpha,
                            solver: solver.clone(),
                            budget: *budget,
                            measurement: stored.spec.measurement.clone(),
                        };
                        let condition_id = hash(&(
                            &condition,
                            condition_versions(&stored.versions, &condition.solver)?,
                        ))?;
                        if !conditions.insert(condition_id.clone()) {
                            bail!("duplicate effective condition");
                        }
                        for &seed in &stored.spec.run_seeds {
                            jobs.push(Job {
                                condition_id: condition_id.clone(),
                                graph_id: graph_id.clone(),
                                condition: condition.clone(),
                                seed,
                            });
                        }
                    }
                }
            }
        }
    }
    jobs.sort_by(|a, b| (&a.condition_id, a.seed).cmp(&(&b.condition_id, b.seed)));
    let batch_id = hash(&(
        &stored.versions,
        jobs.iter()
            .map(|j| (&j.condition_id, j.seed))
            .collect::<Vec<_>>(),
    ))?;
    Ok(ExperimentPlan {
        batch_id,
        experiment: stored,
        jobs,
    })
}

fn condition_versions(
    versions: &BTreeMap<String, String>,
    solver: &SolverSpec,
) -> crate::error::Result<BTreeMap<String, String>> {
    let mut relevant = pinned_versions();
    if let SolverSpec::Eo { fitness, .. } = solver {
        let key = format!("fitness:{}", fitness.kind);
        relevant.insert(
            key.clone(),
            versions
                .get(&key)
                .ok_or_else(|| anyhow!("missing fitness version for {}", fitness.kind))?
                .clone(),
        );
    }
    Ok(relevant)
}

fn requested_fitnesses(spec: &ExperimentSpec) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for solvers in std::iter::once(&spec.solvers)
        .chain(spec.conditions.iter().map(|condition| &condition.solvers))
    {
        for solver in solvers {
            if let SolverSweep::Eo { fitnesses, .. } = solver {
                if let Some(fitnesses) = fitnesses {
                    names.extend(fitnesses.iter().map(|f| f.kind.clone()));
                } else {
                    names.insert("default".into());
                }
            }
        }
    }
    names
}

fn validate_versions(stored: &StoredExperiment) -> crate::error::Result<()> {
    for (key, expected) in pinned_versions() {
        match stored.versions.get(&key) {
            Some(found) if found == &expected => {}
            Some(_) => bail!("unsupported {key} version"),
            None => bail!("missing required version: {key}"),
        }
    }
    let requested = requested_fitnesses(&stored.spec);
    for kind in &requested {
        if !stored.versions.contains_key(&format!("fitness:{kind}")) {
            bail!("missing fitness version for {kind}");
        }
    }
    if requested.contains("default")
        && stored.versions.get("fitness:default") != Some(&"good_edge_fraction-v1".to_owned())
    {
        bail!("unsupported fitness:default version");
    }
    for key in stored.versions.keys() {
        if let Some(kind) = key.strip_prefix("fitness:") {
            if !requested.contains(kind) {
                bail!("unused fitness version: {kind}");
            }
        } else if !pinned_versions().contains_key(key) {
            bail!("unknown version key: {key}");
        }
    }
    Ok(())
}

trait GraphKey {
    fn clone_key(&self) -> String;
}
impl GraphKey for GraphSpec {
    fn clone_key(&self) -> String {
        hash(self).expect("graph is serializable")
    }
}

fn normalize_spec(spec: &mut ExperimentSpec) -> crate::error::Result<()> {
    if spec.problem.alpha == 0.0 {
        spec.problem.alpha = 0.0;
    }
    for graph in &mut spec.graphs {
        for d in &mut graph.expected_degrees {
            if *d == 0.0 {
                *d = 0.0;
            }
        }
    }
    normalize_solvers(&mut spec.solvers);
    for condition in &mut spec.conditions {
        normalize_budget_sweep(condition.budget.as_mut(), "conditions.budget")?;
        normalize_solvers(&mut condition.solvers);
    }
    normalize_budget_sweep(Some(&mut spec.budget), "budget")?;
    spec.conditions
        .sort_by_key(|condition| hash(condition).expect("condition is serializable"));
    Ok(())
}

fn normalize_solvers(solvers: &mut [SolverSweep]) {
    for solver in solvers {
        match solver {
            SolverSweep::Sa { temperatures, .. } => {
                for t in temperatures {
                    if *t == 0.0 {
                        *t = 0.0;
                    }
                }
            }
            SolverSweep::Eo { taus, fitnesses } => {
                for t in taus {
                    if *t == 0.0 {
                        *t = 0.0;
                    }
                }
                if fitnesses.is_none() {
                    *fitnesses = Some(vec![FitnessSpec::default()]);
                }
                if let Some(fitnesses) = fitnesses {
                    for fitness in fitnesses {
                        if fitness.kind == "default" && fitness.params.is_null() {
                            fitness.params = FitnessSpec::default().params;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn normalize_budget_sweep(
    budget: Option<&mut BudgetSweep>,
    label: &str,
) -> crate::error::Result<()> {
    let Some(budget) = budget else {
        return Ok(());
    };
    match &mut budget.max_steps {
        StepCounts::One(steps) => {
            if *steps == 0 {
                bail!("{label}.max_steps must be at least 1");
            }
        }
        StepCounts::Many(steps) => {
            if steps.is_empty() || steps.contains(&0) {
                bail!("{label}.max_steps must be non-empty and positive");
            }
            unique(steps, &format!("{label}.max_steps"))?;
            steps.sort_unstable();
            if steps.len() == 1 {
                budget.max_steps = StepCounts::One(steps[0]);
            }
        }
    }
    Ok(())
}

fn validate_top_level(spec: &ExperimentSpec) -> crate::error::Result<()> {
    if spec.run_seeds.is_empty() || spec.graphs.is_empty() {
        bail!("run_seeds and graphs must be non-empty");
    }
    if spec.conditions.is_empty() {
        if spec.neighborhoods.is_empty() || spec.solvers.is_empty() {
            bail!("neighborhoods and solvers must be non-empty without conditions");
        }
    } else {
        if !spec.neighborhoods.is_empty() || !spec.solvers.is_empty() {
            bail!("neighborhoods and solvers must be empty when conditions are specified");
        }
        for condition in &spec.conditions {
            if condition.neighborhoods.is_empty() || condition.solvers.is_empty() {
                bail!("condition neighborhoods and solvers must be non-empty");
            }
            unique(&condition.neighborhoods, "conditions.neighborhoods")?;
        }
    }
    unique(&spec.run_seeds, "run_seeds")?;
    unique(&spec.neighborhoods, "neighborhoods")?;
    if !spec.problem.alpha.is_finite() || spec.problem.alpha < 0.0 {
        bail!("problem.alpha must be finite and non-negative");
    }
    if spec.measurement.max_basin_steps == 0 {
        bail!("measurement.max_basin_steps must be at least 1");
    }
    match spec.measurement.schedule {
        Schedule::Logarithmic if !spec.measurement.steps.is_empty() => {
            bail!("measurement.steps is only valid for explicit schedule")
        }
        _ => {}
    }
    unique(&spec.measurement.steps, "measurement.steps")?;
    let groups = expand_condition_groups(spec)?;
    if spec.measurement.steps.iter().any(|&step| {
        groups
            .iter()
            .flat_map(|group| group.budgets.iter())
            .any(|budget| step > budget.max_steps)
    }) {
        bail!("measurement.steps must not exceed every effective budget.max_steps");
    }
    for graph in &spec.graphs {
        if graph.node_counts.is_empty()
            || graph.expected_degrees.is_empty()
            || graph.seeds.is_empty()
        {
            bail!("graph sweep arrays must be non-empty");
        }
        unique(&graph.node_counts, "graphs.node_counts")?;
        unique(&graph.seeds, "graphs.seeds")?;
    }
    Ok(())
}

#[derive(Clone)]
struct ConditionGroup {
    neighborhoods: Vec<Neighborhood>,
    solvers: Vec<SolverSweep>,
    budgets: Vec<Budget>,
}

fn expand_budget(budget: &BudgetSweep) -> Vec<Budget> {
    match &budget.max_steps {
        StepCounts::One(max_steps) => vec![Budget {
            max_steps: *max_steps,
        }],
        StepCounts::Many(max_steps) => max_steps
            .iter()
            .map(|max_steps| Budget {
                max_steps: *max_steps,
            })
            .collect(),
    }
}

fn expand_condition_groups(spec: &ExperimentSpec) -> crate::error::Result<Vec<ConditionGroup>> {
    if spec.conditions.is_empty() {
        return Ok(vec![ConditionGroup {
            neighborhoods: spec.neighborhoods.clone(),
            solvers: spec.solvers.clone(),
            budgets: expand_budget(&spec.budget),
        }]);
    }
    Ok(spec
        .conditions
        .iter()
        .map(|condition| ConditionGroup {
            neighborhoods: condition.neighborhoods.clone(),
            solvers: condition.solvers.clone(),
            budgets: expand_budget(condition.budget.as_ref().unwrap_or(&spec.budget)),
        })
        .collect())
}

fn unique<T: Ord + std::fmt::Debug>(items: &[T], label: &str) -> crate::error::Result<()> {
    let mut set = BTreeSet::new();
    for item in items {
        if !set.insert(item) {
            bail!("duplicate value in {label}: {item:?}");
        }
    }
    Ok(())
}

fn validate_graph(g: &GraphSpec, alpha: f64) -> crate::error::Result<()> {
    if g.node_count < 2 {
        bail!("node_count must be at least 2");
    }
    if !g.expected_degree.is_finite()
        || g.expected_degree < 0.0
        || g.expected_degree > (g.node_count - 1) as f64
    {
        bail!("expected_degree must be finite and between 0 and n-1");
    }
    let n = g.node_count as f64;
    let maximum_score = n * (n - 1.0) / 2.0 + alpha * n * n;
    if !maximum_score.is_finite() {
        bail!("maximum possible score is not finite");
    }
    Ok(())
}

fn expand_graphs(spec: &ExperimentSpec) -> crate::error::Result<Vec<GraphSpec>> {
    let mut result = Vec::new();
    for sweep in &spec.graphs {
        for &n in &sweep.node_counts {
            for &degree in &sweep.expected_degrees {
                for &seed in &sweep.seeds {
                    result.push(GraphSpec {
                        kind: sweep.kind,
                        node_count: n,
                        expected_degree: degree,
                        seed,
                    });
                }
            }
        }
    }
    Ok(result)
}

fn expand_solvers(
    sweeps: &[SolverSweep],
    n: usize,
    neighborhood: &Neighborhood,
) -> crate::error::Result<Vec<SolverSpec>> {
    let mut result = Vec::new();
    for sweep in sweeps {
        match sweep {
            SolverSweep::Hc { smoothing } => {
                validate_smoothing_sweeps(smoothing)?;
                for s in expand_smoothing(smoothing, n, neighborhood)? {
                    result.push(SolverSpec::Hc { smoothing: s });
                }
            }
            SolverSweep::Sa {
                temperatures,
                smoothing,
            } => {
                if temperatures.is_empty() {
                    bail!("sa.temperatures must be non-empty");
                }
                validate_smoothing_sweeps(smoothing)?;
                for &t in temperatures {
                    if !t.is_finite() || t < 0.0 {
                        bail!("temperature must be finite and non-negative");
                    }
                    for s in expand_smoothing(smoothing, n, neighborhood)? {
                        result.push(SolverSpec::Sa {
                            temperature: t,
                            smoothing: s,
                        });
                    }
                }
            }
            SolverSweep::Eo { taus, fitnesses } => {
                if taus.is_empty() {
                    bail!("eo.taus must be non-empty");
                }
                let default_fitnesses = [FitnessSpec::default()];
                let fitnesses = fitnesses.as_deref().unwrap_or(&default_fitnesses);
                if fitnesses.is_empty() {
                    bail!("eo.fitnesses must be non-empty when specified");
                }
                for &tau in taus {
                    if !tau.is_finite() || tau <= 0.0 {
                        bail!("tau must be finite and positive");
                    }
                    for fitness in fitnesses {
                        result.push(SolverSpec::Eo {
                            tau,
                            fitness: fitness.clone(),
                        });
                    }
                }
            }
        }
    }
    Ok(result)
}

fn validate_smoothing_sweeps(sweeps: &[SmoothingSweep]) -> crate::error::Result<()> {
    if sweeps.is_empty() {
        bail!("smoothing must be non-empty");
    }
    for sweep in sweeps {
        match sweep {
            SmoothingSweep::RandomKAverage { ks } => {
                if ks.is_empty() || ks.contains(&0) {
                    bail!("random_k_average.ks must be non-empty and positive");
                }
            }
            SmoothingSweep::WeightedAverage { ks } if ks.is_empty() => {
                bail!("weighted_average.ks must be non-empty")
            }
            _ => {}
        }
    }
    Ok(())
}

fn expand_smoothing(
    sweeps: &[SmoothingSweep],
    n: usize,
    neighborhood: &Neighborhood,
) -> crate::error::Result<Vec<SmoothingSpec>> {
    let m = match neighborhood {
        Neighborhood::Flip => n,
        Neighborhood::Swap => {
            n.checked_mul(n)
                .ok_or_else(|| anyhow!("neighborhood count overflow"))?
                / 4
        }
    };
    let d2 = match neighborhood {
        Neighborhood::Flip => {
            n.checked_mul(
                n.checked_sub(1)
                    .ok_or_else(|| anyhow!("node count underflow"))?,
            )
            .ok_or_else(|| anyhow!("neighborhood count overflow"))?
                / 2
        }
        Neighborhood::Swap => {
            let half = n / 2;
            let c = half
                .checked_mul(
                    half.checked_sub(1)
                        .ok_or_else(|| anyhow!("node count underflow"))?,
                )
                .ok_or_else(|| anyhow!("neighborhood count overflow"))?
                / 2;
            c.checked_mul(c)
                .ok_or_else(|| anyhow!("neighborhood count overflow"))?
        }
    };
    let capacity = m
        .checked_add(d2)
        .ok_or_else(|| anyhow!("neighborhood count overflow"))?;
    let mut result = Vec::new();
    for s in sweeps {
        match s {
            SmoothingSweep::None => result.push(SmoothingSpec::None),
            SmoothingSweep::AllAverage => result.push(SmoothingSpec::AllAverage),
            SmoothingSweep::RandomKAverage { ks } => {
                for &k in ks {
                    result.push(SmoothingSpec::RandomKAverage { k: k.min(capacity) })
                }
            }
            SmoothingSweep::WeightedAverage { ks } => {
                for &k in ks {
                    result.push(SmoothingSpec::WeightedAverage { k: k.min(m) })
                }
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_sample_has_54_jobs() {
        assert_eq!(
            compile_experiment(toml::from_str(sample_toml()).unwrap())
                .unwrap()
                .jobs
                .len(),
            54
        );
    }

    #[test]
    fn display_name_and_array_order_do_not_change_ids() {
        let original: ExperimentSpec = toml::from_str(sample_toml()).unwrap();
        let mut reordered = original.clone();
        reordered.name = Some("a different title".into());
        reordered.run_seeds.reverse();
        reordered.neighborhoods.reverse();
        reordered.solvers.reverse();
        let a = compile_experiment(original).unwrap();
        let b = compile_experiment(reordered).unwrap();
        assert_eq!(a.batch_id, b.batch_id);
        assert_eq!(a.jobs, b.jobs);
    }

    #[test]
    fn malformed_eo_and_k_average_are_rejected() {
        let malformed = sample_toml().replace("taus = [1.2, 1.5]", "taus = []");
        assert!(compile_experiment(toml::from_str(&malformed).unwrap()).is_err());
        assert!(
            toml::from_str::<ExperimentSpec>(
                &sample_toml().replace("random_k_average", "k_average")
            )
            .is_err()
        );
    }

    #[test]
    fn toml_and_json_inputs_produce_the_same_plan() {
        let spec: ExperimentSpec = toml::from_str(sample_toml()).unwrap();
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(
            compile_experiment(spec).unwrap(),
            compile_experiment(serde_json::from_str(&json).unwrap()).unwrap()
        );
    }

    #[test]
    fn unrelated_fitness_does_not_change_hc_or_sa_condition_ids() {
        let original: ExperimentSpec = toml::from_str(sample_toml()).unwrap();
        let mut extended = original.clone();
        extended.solvers.push(SolverSweep::Eo {
            taus: vec![2.0],
            fitnesses: Some(vec![FitnessSpec {
                kind: "custom".into(),
                params: Value::Object(Default::default()),
            }]),
        });
        let versions = BTreeMap::from([
            ("default".into(), "good_edge_fraction-v1".into()),
            ("custom".into(), "custom-v1".into()),
        ]);
        let a = compile_experiment_with_versions(original, &versions).unwrap();
        let b = compile_experiment_with_versions(extended, &versions).unwrap();
        let ids = |plan: &ExperimentPlan| {
            plan.jobs
                .iter()
                .filter(|job| !matches!(&job.condition.solver, SolverSpec::Eo { .. }))
                .map(|job| job.condition_id.clone())
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(ids(&a), ids(&b));
    }

    #[test]
    fn stored_default_fitness_version_is_pinned() {
        let mut stored = compile_experiment(toml::from_str(sample_toml()).unwrap())
            .unwrap()
            .experiment;
        stored
            .versions
            .insert("fitness:default".into(), "wrong".into());
        assert!(compile_stored(stored).is_err());
    }

    #[test]
    fn explicit_empty_steps_are_valid_but_steps_past_budget_are_not() {
        let mut spec: ExperimentSpec = toml::from_str(minimal_sample_toml()).unwrap();
        spec.measurement.schedule = Schedule::Explicit;
        assert!(compile_experiment(spec.clone()).is_ok());
        spec.measurement.steps.push(101);
        assert!(compile_experiment(spec).is_err());
    }

    #[test]
    fn canonical_hash_normalizes_negative_zero_but_tags_float_and_integer() {
        assert_eq!(
            hash(&serde_json::json!(-0.0)).unwrap(),
            hash(&serde_json::json!(0.0)).unwrap()
        );
        assert_ne!(
            hash(&serde_json::json!(1)).unwrap(),
            hash(&serde_json::json!(1.0)).unwrap()
        );
    }
}
