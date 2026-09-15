//! Serializable input and expanded experiment configuration.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum GraphKind {
    Random,
    Geometric,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GraphSpec {
    pub kind: GraphKind,
    pub node_count: usize,
    pub expected_degree: f64,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Neighborhood {
    Flip,
    Swap,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SmoothingSpec {
    None,
    AllAverage,
    RandomKAverage { k: usize },
    WeightedAverage { k: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FitnessSpec {
    #[serde(default = "default_fitness_kind")]
    pub kind: String,
    #[serde(default = "default_fitness_params")]
    pub params: Value,
}
fn default_fitness_kind() -> String {
    "default".into()
}
fn default_fitness_params() -> Value {
    Value::Object(Default::default())
}
impl Default for FitnessSpec {
    fn default() -> Self {
        Self {
            kind: default_fitness_kind(),
            params: default_fitness_params(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SolverSpec {
    Hc {
        smoothing: SmoothingSpec,
    },
    Sa {
        temperature: f64,
        smoothing: SmoothingSpec,
    },
    Eo {
        tau: f64,
        fitness: FitnessSpec,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub max_steps: u64,
}

/// One or more step limits used to expand an experiment budget sweep.
///
/// The untagged representation intentionally keeps a single value identical to
/// the historical `max_steps = 100` format.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum StepCounts {
    One(u64),
    Many(Vec<u64>),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BudgetSweep {
    pub max_steps: StepCounts,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConditionSweep {
    pub neighborhoods: Vec<Neighborhood>,
    pub solvers: Vec<SolverSweep>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<BudgetSweep>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Schedule {
    #[default]
    Logarithmic,
    Explicit,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum BasinMode {
    None,
    Real,
    #[default]
    Both,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    #[serde(default)]
    pub schedule: Schedule,
    #[serde(default)]
    pub steps: Vec<u64>,
    #[serde(default)]
    pub basin: BasinMode,
    #[serde(default = "default_max_basin_steps")]
    pub max_basin_steps: u64,
    #[serde(default)]
    pub diagnostics: bool,
    /// Measure a real-objective basin starting at the incumbent partition.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub best_basin: bool,
}
fn default_max_basin_steps() -> u64 {
    10_000
}
impl Default for Measurement {
    fn default() -> Self {
        Self {
            schedule: Schedule::Logarithmic,
            steps: vec![],
            basin: BasinMode::Both,
            max_basin_steps: 10_000,
            diagnostics: false,
            best_basin: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Condition {
    pub graph: GraphSpec,
    pub neighborhood: Neighborhood,
    pub alpha: f64,
    pub solver: SolverSpec,
    pub budget: Budget,
    pub measurement: Measurement,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProblemSweep {
    #[serde(default = "default_alpha")]
    pub alpha: f64,
}
fn default_alpha() -> f64 {
    0.05
}
impl Default for ProblemSweep {
    fn default() -> Self {
        Self {
            alpha: default_alpha(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GraphSweep {
    pub kind: GraphKind,
    pub node_counts: Vec<usize>,
    pub expected_degrees: Vec<f64>,
    pub seeds: Vec<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SmoothingSweep {
    None,
    AllAverage,
    RandomKAverage { ks: Vec<usize> },
    WeightedAverage { ks: Vec<usize> },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SolverSweep {
    Hc {
        #[serde(default = "default_smoothing")]
        smoothing: Vec<SmoothingSweep>,
    },
    Sa {
        temperatures: Vec<f64>,
        #[serde(default = "default_smoothing")]
        smoothing: Vec<SmoothingSweep>,
    },
    Eo {
        taus: Vec<f64>,
        #[serde(default)]
        fitnesses: Option<Vec<FitnessSpec>>,
    },
}
fn default_smoothing() -> Vec<SmoothingSweep> {
    vec![SmoothingSweep::None]
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExperimentSpec {
    pub schema_version: u32,
    #[serde(default)]
    pub name: Option<String>,
    pub run_seeds: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub neighborhoods: Vec<Neighborhood>,
    #[serde(default)]
    pub problem: ProblemSweep,
    pub budget: BudgetSweep,
    #[serde(default)]
    pub measurement: Measurement,
    pub graphs: Vec<GraphSweep>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub solvers: Vec<SolverSweep>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<ConditionSweep>,
}
