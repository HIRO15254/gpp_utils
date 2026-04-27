// New framework (under development)
pub mod optimization;
pub mod smoothing;
pub mod solvers;
pub mod experiment;
pub mod file_utils;
pub mod graph_partition;

// New experiment-driven workflow
pub mod graph_spec;
pub mod run_config;
pub mod run_executor;

// Legacy modules - will be refactored to new architecture in Phase 4-6
// pub mod continuous_relaxation;
// pub mod extremal_optimization;
// pub mod quantum_annealing;
// pub mod simulated_annealing;
// pub mod smoothed_sa;
