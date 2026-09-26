use crate::error::Result;
use crate::experiment::config::{BasinMode, Condition, Neighborhood, SmoothingSpec, SolverSpec};
use crate::graph_partition::Graph;
use anyhow::ensure;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SolutionId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunTermination {
    StepLimit,
    LocalOptimum,
    NoSampledImprovement,
    Cancelled,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BasinTermination {
    LocalOptimum,
    StepLimit,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BasinResult {
    pub real: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smoothed: Option<f64>,
    pub termination: BasinTermination,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementRecord {
    pub step: u64,
    pub current_solution: SolutionId,
    pub best_solution: SolutionId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_smoothed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_evaluation: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub basin_real: Option<BasinResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub basin_smoothed: Option<BasinResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basin_best: Option<BasinResult>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Diagnostics {
    pub applied_moves: u64,
    pub objective_evaluations_search: u64,
    pub objective_evaluations_measurement: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fitness_values_computed_search: Option<u64>,
    pub search_ms: f64,
    pub measurement_ms: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunResult {
    pub schema_version: u32,
    pub attempt_id: String,
    pub termination: RunTermination,
    pub completed_steps: u64,
    pub elapsed_ms: f64,
    /// Deduplicated partition pool in first-reference order, indexed by
    /// [`SolutionId`]; `true` places a vertex in group A.
    ///
    /// JSON stores the pool compactly as `{"length": n, "hex": [...]}`, where
    /// `length` is the vertex count shared by every partition (`0` for an
    /// empty pool). Each string packs vertex `v` into bit `v % 8` of byte
    /// `v / 8` (LSB first, `true` = 1) and writes every byte as two lowercase
    /// hexadecimal digits, so it has exactly `2 * ceil(n / 8)` digits and zero
    /// padding bits. Serialization fails if the partitions differ in length;
    /// deserialization rejects missing or unknown keys, non-string items, a
    /// wrong digit count, non-hexadecimal or uppercase digits, and non-zero
    /// padding. [`Self::validate`] checks `n` against the graph.
    #[serde(with = "crate::experiment::partition_codec")]
    pub partitions: Vec<Vec<bool>>,
    pub final_solution: SolutionId,
    pub best_solution: SolutionId,
    pub best_step: u64,
    pub records: Vec<MeasurementRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Diagnostics>,
}

impl RunResult {
    pub fn validate(&self, graph: &Graph, condition: &Condition) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "unsupported result schema version"
        );
        ensure!(!self.attempt_id.is_empty(), "empty attempt_id");
        ensure!(
            condition.alpha.is_finite() && condition.alpha >= 0.0,
            "invalid alpha"
        );
        ensure!(
            self.elapsed_ms.is_finite() && self.elapsed_ms >= 0.0,
            "invalid elapsed_ms"
        );
        ensure!(
            self.completed_steps <= condition.budget.max_steps,
            "completed_steps exceeds budget"
        );
        ensure!(
            self.best_step <= self.completed_steps,
            "best_step exceeds completed_steps"
        );
        let cancelled = self.termination == RunTermination::Cancelled;
        match self.termination {
            RunTermination::StepLimit => ensure!(
                self.completed_steps == condition.budget.max_steps,
                "step_limit before budget"
            ),
            RunTermination::LocalOptimum => ensure!(
                matches!(condition.solver, SolverSpec::Hc { .. }),
                "local optimum is an HC termination"
            ),
            RunTermination::NoSampledImprovement => ensure!(
                matches!(
                    condition.solver,
                    SolverSpec::Hc {
                        smoothing: SmoothingSpec::RandomKAverage { .. }
                    }
                ),
                "sampled termination requires stochastic HC"
            ),
            RunTermination::Cancelled => {}
        }
        ensure!(!self.partitions.is_empty(), "empty partition pool");
        let mut unique = BTreeSet::new();
        for p in &self.partitions {
            ensure!(p.len() == graph.node_count(), "invalid partition length");
            ensure!(unique.insert(p), "duplicate partition in pool");
            if condition.neighborhood == Neighborhood::Swap {
                ensure!(
                    graph.node_count().is_multiple_of(2)
                        && p.iter().filter(|&&v| v).count() == graph.node_count() / 2,
                    "unbalanced swap partition"
                );
            }
        }
        let valid = |id: SolutionId| id.0 < self.partitions.len();
        ensure!(
            valid(self.final_solution) && valid(self.best_solution),
            "invalid solution reference"
        );
        ensure!(
            self.records.first().map(|r| r.step) == Some(0)
                && self.records.last().map(|r| r.step) == Some(self.completed_steps),
            "records must include initial and final steps"
        );
        let first = &self.records[0];
        let last = self.records.last().unwrap();
        ensure!(
            first.current_solution == first.best_solution,
            "initial incumbent must equal current solution"
        );
        ensure!(
            last.current_solution == self.final_solution
                && last.best_solution == self.best_solution,
            "final references differ from final record"
        );
        let scores: Vec<f64> = self
            .partitions
            .iter()
            .map(|p| graph.score(p, condition.alpha))
            .collect();
        ensure!(
            scores.iter().all(|x| x.is_finite()),
            "non-finite real score"
        );
        let smooth = match &condition.solver {
            SolverSpec::Hc { smoothing } | SolverSpec::Sa { smoothing, .. } => Some(smoothing),
            SolverSpec::Eo { .. } | SolverSpec::EoSa { .. } => None,
        };
        let nontrivial = smooth.is_some_and(|s| {
            !matches!(
                s,
                SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 }
            )
        });
        let random = matches!(smooth, Some(SmoothingSpec::RandomKAverage { .. }));
        let expected = crate::experiment::measurement::checkpoints(
            &condition.measurement,
            condition.budget.max_steps,
        );
        let mut expected: Vec<u64> = expected
            .into_iter()
            .filter(|&step| step <= self.completed_steps)
            .collect();
        if expected.last() != Some(&self.completed_steps) {
            expected.push(self.completed_steps);
        }
        ensure!(
            self.records.iter().map(|r| r.step).eq(expected),
            "unexpected or missing measurement steps"
        );
        let mut previous: Option<&MeasurementRecord> = None;
        let mut best_basins: std::collections::BTreeMap<SolutionId, &BasinResult> =
            std::collections::BTreeMap::new();
        let mut encountered = BTreeSet::new();
        for r in &self.records {
            ensure!(
                valid(r.current_solution) && valid(r.best_solution),
                "invalid measurement solution reference"
            );
            for id in [r.current_solution, r.best_solution] {
                if !encountered.contains(&id.0) {
                    ensure!(
                        id.0 == encountered.len(),
                        "partition pool is not in first occurrence order"
                    );
                    encountered.insert(id.0);
                }
            }
            ensure!(
                scores[r.best_solution.0] <= scores[r.current_solution.0],
                "incumbent is worse than current"
            );
            if let Some(p) = previous {
                ensure!(r.step > p.step, "measurement steps not strictly increasing");
                ensure!(
                    scores[r.best_solution.0] <= scores[p.best_solution.0],
                    "incumbent score increased"
                );
                if scores[r.best_solution.0] == scores[p.best_solution.0] {
                    ensure!(
                        r.best_solution == p.best_solution,
                        "equal-score incumbent was replaced"
                    );
                }
            }
            previous = Some(r);
            for x in [r.current_smoothed, r.search_evaluation]
                .into_iter()
                .flatten()
            {
                ensure!(x.is_finite(), "non-finite measurement");
            }
            let partial_endpoint = cancelled && r.step == self.completed_steps;
            if !nontrivial {
                ensure!(
                    r.current_smoothed.is_none() && r.search_evaluation.is_none(),
                    "redundant/nonapplicable smoothing evaluation"
                );
            } else if !partial_endpoint {
                ensure!(
                    r.current_smoothed.is_some(),
                    "missing smoothing measurement"
                );
            }
            if !random {
                ensure!(r.search_evaluation.is_none(), "redundant search evaluation");
            } else if !partial_endpoint {
                ensure!(
                    r.search_evaluation.is_some(),
                    "missing stochastic search evaluation"
                );
            }
            let wants_real = condition.measurement.basin != BasinMode::None;
            let wants_smooth = condition.measurement.basin == BasinMode::Both && nontrivial;
            let wants_best = condition.measurement.best_basin;
            if !wants_real {
                ensure!(r.basin_real.is_none(), "unexpected real basin");
            } else if !partial_endpoint {
                ensure!(r.basin_real.is_some(), "missing real basin");
            }
            if !wants_smooth {
                ensure!(r.basin_smoothed.is_none(), "unexpected smoothed basin");
            } else if !partial_endpoint {
                ensure!(r.basin_smoothed.is_some(), "missing smoothed basin");
            }
            if !wants_best {
                ensure!(r.basin_best.is_none(), "unexpected best basin");
            } else if !partial_endpoint {
                ensure!(r.basin_best.is_some(), "missing best basin");
            }
            for basin in [r.basin_real.as_ref(), r.basin_smoothed.as_ref()]
                .into_iter()
                .flatten()
            {
                ensure!(
                    basin.real.is_finite() && basin.smoothed.is_none_or(|x| x.is_finite()),
                    "non-finite basin"
                );
                ensure!(
                    basin.smoothed.is_some() == nontrivial,
                    "missing or redundant basin smoothing value"
                );
                ensure!(
                    basin.steps.is_some() == condition.measurement.diagnostics,
                    "basin diagnostics mismatch"
                );
                if let Some(steps) = basin.steps {
                    ensure!(
                        steps > 0 && steps <= condition.measurement.max_basin_steps,
                        "invalid basin scan count"
                    );
                }
            }
            if let Some(basin) = r.basin_best.as_ref() {
                ensure!(basin.real.is_finite(), "non-finite best basin");
                ensure!(
                    basin.real <= scores[r.best_solution.0],
                    "best basin is worse than its incumbent"
                );
                ensure!(
                    basin.smoothed.is_none(),
                    "best basin must use real objective"
                );
                ensure!(
                    basin.steps.is_some() == condition.measurement.diagnostics,
                    "best basin diagnostics mismatch"
                );
                if let Some(steps) = basin.steps {
                    ensure!(
                        steps > 0 && steps <= condition.measurement.max_basin_steps,
                        "invalid best basin scan count"
                    );
                }
                if let Some(previous_basin) = best_basins.get(&r.best_solution) {
                    ensure!(
                        previous_basin.real.to_bits() == basin.real.to_bits()
                            && previous_basin.smoothed.map(f64::to_bits)
                                == basin.smoothed.map(f64::to_bits)
                            && previous_basin.termination == basin.termination
                            && previous_basin.steps == basin.steps,
                        "inconsistent repeated best basin"
                    );
                } else {
                    best_basins.insert(r.best_solution, basin);
                }
            }
        }
        ensure!(
            encountered.len() == self.partitions.len(),
            "unreferenced partition in pool"
        );
        if self.best_step == 0 {
            ensure!(
                self.best_solution == first.current_solution,
                "best_step zero refers to a different partition"
            );
        }
        if self.best_solution == first.current_solution {
            ensure!(
                self.best_step == 0,
                "initial incumbent has nonzero best_step"
            );
        }
        for r in &self.records {
            if r.step < self.best_step {
                ensure!(
                    scores[r.best_solution.0] > scores[self.best_solution.0],
                    "best_step is later than an observed incumbent"
                );
            } else {
                ensure!(
                    r.best_solution == self.best_solution,
                    "best_step disagrees with observed incumbent"
                );
            }
        }
        ensure!(
            self.diagnostics.is_some() == condition.measurement.diagnostics,
            "diagnostics setting mismatch"
        );
        if let Some(d) = &self.diagnostics {
            ensure!(
                d.applied_moves <= self.completed_steps,
                "moves exceed completed steps"
            );
            ensure!(
                d.search_ms.is_finite()
                    && d.search_ms >= 0.0
                    && d.measurement_ms.is_finite()
                    && d.measurement_ms >= 0.0,
                "invalid diagnostic time"
            );
            ensure!(
                d.search_ms + d.measurement_ms <= self.elapsed_ms + 1e-6,
                "time components exceed elapsed time"
            );
            ensure!(
                d.fitness_values_computed_search.is_some() == condition.solver.fitness().is_some(),
                "fitness diagnostic applicability mismatch"
            );
        }
        Ok(())
    }
}

pub struct RunView<'a> {
    graph: &'a Graph,
    condition: &'a Condition,
    result: &'a RunResult,
}

/// A real-objective score together with the terms used to derive it for TSV
/// views. `balance_penalty` intentionally keeps the historical `powi(2)`
/// evaluation used by the exporter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreBreakdown {
    pub real: f64,
    pub cut_edges: usize,
    pub size_a: usize,
    pub size_b: usize,
    pub balance_penalty: f64,
}

/// A typed view of a measured basin result. The selected result may be the
/// real basin used as the smoothed-basin fallback for identity smoothing.
#[derive(Clone, Copy, Debug)]
pub struct BasinView<'a> {
    result: &'a BasinResult,
    smoothed: Option<f64>,
}

impl<'a> BasinView<'a> {
    pub fn real(self) -> f64 {
        self.result.real
    }
    pub fn smoothed(self) -> Option<f64> {
        self.smoothed
    }
    pub fn termination(self) -> BasinTermination {
        self.result.termination
    }
    pub fn steps(self) -> Option<u64> {
        self.result.steps
    }
    pub fn result(self) -> &'a BasinResult {
        self.result
    }
}

/// Typed derived values for one measurement record.
pub struct MeasurementView<'view, 'result> {
    view: &'view RunView<'result>,
    record: &'result MeasurementRecord,
}

impl<'a> RunView<'a> {
    pub fn new(graph: &'a Graph, condition: &'a Condition, result: &'a RunResult) -> Result<Self> {
        result.validate(graph, condition)?;
        Ok(Self {
            graph,
            condition,
            result,
        })
    }
    pub fn graph(&self) -> &'a Graph {
        self.graph
    }
    pub fn condition(&self) -> &'a Condition {
        self.condition
    }
    pub fn result(&self) -> &'a RunResult {
        self.result
    }
    /// Returns `None` when `id` is not in this result's partition pool.
    pub fn try_partition(&self, id: SolutionId) -> Option<&[bool]> {
        self.result.partitions.get(id.0).map(Vec::as_slice)
    }
    /// Returns a partition from this validated result.
    ///
    /// Panics when called with an arbitrary `SolutionId` outside the pool; use
    /// [`Self::try_partition`] for unchecked external IDs.
    pub fn partition(&self, id: SolutionId) -> &[bool] {
        &self.result.partitions[id.0]
    }
    pub fn score(&self, id: SolutionId) -> f64 {
        self.graph.score(self.partition(id), self.condition.alpha)
    }
    pub fn final_score(&self) -> f64 {
        self.score(self.result.final_solution)
    }
    pub fn best_score(&self) -> f64 {
        self.score(self.result.best_solution)
    }
    pub fn breakdown(&self, id: SolutionId) -> ScoreBreakdown {
        let partition = self.partition(id);
        let size_a = partition.iter().filter(|&&value| value).count();
        let size_b = partition.len() - size_a;
        let cut_edges = self
            .graph
            .edges()
            .iter()
            .filter(|&&[a, b]| partition[a] != partition[b])
            .count();
        ScoreBreakdown {
            real: self.score(id),
            cut_edges,
            size_a,
            size_b,
            balance_penalty: self.condition.alpha * (size_a as f64 - size_b as f64).powi(2),
        }
    }
    pub fn measurement<'view>(
        &'view self,
        record: &'a MeasurementRecord,
    ) -> MeasurementView<'view, 'a> {
        MeasurementView { view: self, record }
    }
    pub fn records(&self) -> impl Iterator<Item = MeasurementView<'_, 'a>> {
        self.result
            .records
            .iter()
            .map(|record| self.measurement(record))
    }
    pub fn final_measurement(&self) -> MeasurementView<'_, 'a> {
        self.measurement(
            self.result
                .records
                .last()
                .expect("validated result has records"),
        )
    }
}

impl<'view, 'result> MeasurementView<'view, 'result> {
    pub fn record(&self) -> &'result MeasurementRecord {
        self.record
    }
    pub fn step(&self) -> u64 {
        self.record.step
    }
    pub fn current_breakdown(&self) -> ScoreBreakdown {
        self.view.breakdown(self.record.current_solution)
    }
    pub fn best_breakdown(&self) -> ScoreBreakdown {
        self.view.breakdown(self.record.best_solution)
    }
    /// The displayed smoothing value, including the identity-smoothing fallback.
    pub fn smoothed_value(&self) -> Option<f64> {
        if identity_smoothing(self.view.condition) {
            Some(self.current_breakdown().real)
        } else {
            self.record.current_smoothed
        }
    }
    /// The value used by the search at this measurement. EO has no smoothing
    /// measurement, random smoothing retains its separately sampled value.
    pub fn search_evaluation(&self) -> Option<f64> {
        match smoothing(self.view.condition) {
            None => None,
            Some(SmoothingSpec::RandomKAverage { .. }) => self.record.search_evaluation,
            Some(_) => self.smoothed_value(),
        }
    }
    pub fn real_basin(&self) -> Option<BasinView<'result>> {
        self.record.basin_real.as_ref().map(|result| BasinView {
            result,
            smoothed: basin_smoothed(self.view.condition, result),
        })
    }
    pub fn smoothed_basin(&self) -> Option<BasinView<'result>> {
        let result = if identity_smoothing(self.view.condition) {
            self.record.basin_real.as_ref()
        } else {
            self.record.basin_smoothed.as_ref()
        }?;
        Some(BasinView {
            result,
            smoothed: basin_smoothed(self.view.condition, result),
        })
    }
    pub fn best_basin(&self) -> Option<BasinView<'result>> {
        self.record.basin_best.as_ref().map(|result| BasinView {
            result,
            smoothed: None,
        })
    }
}

fn smoothing(condition: &Condition) -> Option<&SmoothingSpec> {
    match &condition.solver {
        SolverSpec::Hc { smoothing } | SolverSpec::Sa { smoothing, .. } => Some(smoothing),
        SolverSpec::Eo { .. } | SolverSpec::EoSa { .. } => None,
    }
}
fn identity_smoothing(condition: &Condition) -> bool {
    matches!(
        smoothing(condition),
        Some(SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 })
    )
}
fn basin_smoothed(condition: &Condition, basin: &BasinResult) -> Option<f64> {
    match smoothing(condition) {
        None => None,
        Some(_) if identity_smoothing(condition) => Some(basin.real),
        Some(_) => basin.smoothed,
    }
}
