//! Reproducible graph partition experiments with HC, SA and EO.
pub mod error;
pub mod experiment;
pub mod export;
pub mod fitness;
pub mod graph_partition;
pub mod optimization;
pub mod smoothing;
pub mod solvers;
pub mod storage;
pub use experiment::config::ExperimentSpec;
pub use experiment::plan::{ExperimentPlan, StoredExperiment, compile_experiment};
pub use experiment::result::{MeasurementRecord, RunResult, SolutionId};
pub use experiment::runner::run_one;
pub use storage::{BatchSummary, RuntimeOptions, run_batch};
