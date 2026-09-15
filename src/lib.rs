//! Reproducible graph partition experiments with HC, SA and EO.
//!
//! Compile input with [`compile_experiment`], then use [`run_one`] or
//! [`run_batch`]. [`graph_partition::Graph`] owns immutable topology;
//! [`experiment::result::RunView`] resolves saved solutions and derived values.
//! Custom EO fitness uses [`fitness::FitnessFactory`] and
//! [`fitness::FitnessRegistry`] throughout compile, execution and resume.
//!
//! Public Rust APIs follow pre-1.0 minor-version compatibility: breaking changes
//! require a minor release. Data schema and scientific algorithm versions are
//! separate contracts; see `docs/extending.md` before changing either.
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
