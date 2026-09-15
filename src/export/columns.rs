//! Ordered export schema: name, type, unit, meaning and typed value selector.
//! The table-specific selector types prevent mixing run and trace columns.

#[derive(Clone, Copy)]
pub(super) enum RunField {
    Batch,
    Condition,
    Seed,
    Status,
    Latest,
    Termination,
    GraphId,
    GraphKind,
    Nodes,
    ExpectedDegree,
    GraphSeed,
    Edges,
    ActualDegree,
    Alpha,
    Neighborhood,
    Solver,
    Temperature,
    Tau,
    Smoothing,
    K,
    Fitness,
    FitnessVersion,
    FitnessParams,
    MaxSteps,
    Completed,
    BestStep,
    InitialReal,
    FinalReal,
    BestReal,
    FinalCuts,
    FinalA,
    FinalB,
    FinalPenalty,
    BestCuts,
    BestA,
    BestB,
    BestPenalty,
    Elapsed,
    FinalBasinReal,
    FinalBasinRealStatus,
    FinalBasinSmooth,
    FinalBasinSmoothStatus,
    FinalBasinBest,
    FinalBasinBestStatus,
    Applied,
    Accepted,
    Rejected,
    SearchEvals,
    MeasurementEvals,
    FitnessEvals,
    SearchMs,
    MeasurementMs,
}
#[derive(Clone, Copy)]
pub(super) enum TraceField {
    Condition,
    Seed,
    Status,
    Step,
    CurrentReal,
    BestReal,
    SearchEvaluation,
    CurrentSmoothed,
    BasinReal,
    BasinRealSmoothed,
    BasinRealStatus,
    BasinRealSteps,
    BasinSmooth,
    BasinSmoothValue,
    BasinSmoothStatus,
    BasinSmoothSteps,
    BasinBest,
    BasinBestStatus,
    BasinBestSteps,
}
pub(super) struct ColumnSpec<F> {
    pub(super) name: &'static str,
    pub(super) kind: &'static str,
    pub(super) unit: Option<&'static str>,
    pub(super) meaning: &'static str,
    pub(super) field: F,
}
macro_rules! c {
    ($name:literal,$kind:literal,$unit:expr,$meaning:literal,$selector:path) => {
        ColumnSpec {
            name: $name,
            kind: $kind,
            unit: $unit,
            meaning: $meaning,
            field: $selector,
        }
    };
}
macro_rules! run_c {
    ($name:literal,$kind:literal,$unit:expr,$meaning:literal,$field:ident) => {
        c!($name, $kind, $unit, $meaning, RunField::$field)
    };
}
macro_rules! trace_c {
    ($name:literal,$kind:literal,$unit:expr,$meaning:literal,$field:ident) => {
        c!($name, $kind, $unit, $meaning, TraceField::$field)
    };
}
pub(super) const RUN_COLUMNS: &[ColumnSpec<RunField>] = &[
    run_c!("batch_id", "string", None, "Batch identifier", Batch),
    run_c!(
        "condition_id",
        "string",
        None,
        "Expanded condition identifier",
        Condition
    ),
    run_c!("seed", "integer", None, "Run seed", Seed),
    run_c!("status", "string", None, "Stored job status", Status),
    run_c!(
        "latest_attempt_status",
        "string",
        None,
        "Latest unfinished attempt, separately from a reusable completed result",
        Latest
    ),
    run_c!(
        "termination",
        "string",
        None,
        "Completed run termination",
        Termination
    ),
    run_c!(
        "graph_id",
        "string",
        None,
        "Expanded graph identifier",
        GraphId
    ),
    run_c!(
        "graph_kind",
        "string",
        None,
        "Graph generator kind",
        GraphKind
    ),
    run_c!(
        "node_count",
        "integer",
        None,
        "Requested graph node count",
        Nodes
    ),
    run_c!(
        "expected_degree",
        "number",
        None,
        "Requested average degree",
        ExpectedDegree
    ),
    run_c!(
        "graph_seed",
        "integer",
        None,
        "Graph generator seed",
        GraphSeed
    ),
    run_c!(
        "edge_count",
        "integer",
        None,
        "Generated graph edge count",
        Edges
    ),
    run_c!(
        "actual_average_degree",
        "number",
        None,
        "Generated graph average degree",
        ActualDegree
    ),
    run_c!(
        "alpha",
        "number",
        None,
        "Balance penalty coefficient",
        Alpha
    ),
    run_c!(
        "neighborhood",
        "string",
        None,
        "Search neighborhood",
        Neighborhood
    ),
    run_c!("solver", "string", None, "Solver kind", Solver),
    run_c!("temperature", "number", None, "SA temperature", Temperature),
    run_c!("tau", "number", None, "EO rank exponent", Tau),
    run_c!("smoothing", "string", None, "Smoothing kind", Smoothing),
    run_c!("k", "integer", None, "Effective smoothing sample count", K),
    run_c!(
        "fitness",
        "string",
        None,
        "Registered vertex fitness name",
        Fitness
    ),
    run_c!(
        "fitness_version",
        "string",
        None,
        "Registered vertex fitness version",
        FitnessVersion
    ),
    run_c!(
        "fitness_params_json",
        "string",
        None,
        "Serialized vertex fitness parameters",
        FitnessParams
    ),
    run_c!(
        "max_steps",
        "integer",
        None,
        "Configured step budget",
        MaxSteps
    ),
    run_c!(
        "completed_steps",
        "integer",
        None,
        "Completed search steps",
        Completed
    ),
    run_c!(
        "best_step",
        "integer",
        None,
        "Step where incumbent was first reached",
        BestStep
    ),
    run_c!(
        "initial_real",
        "number",
        None,
        "Real objective of initial partition",
        InitialReal
    ),
    run_c!(
        "final_real",
        "number",
        None,
        "Real objective of final partition",
        FinalReal
    ),
    run_c!(
        "best_real",
        "number",
        None,
        "Real objective of incumbent partition",
        BestReal
    ),
    run_c!(
        "final_cut_edges",
        "integer",
        None,
        "Cut edges in final partition",
        FinalCuts
    ),
    run_c!(
        "final_size_a",
        "integer",
        None,
        "True vertices in final partition",
        FinalA
    ),
    run_c!(
        "final_size_b",
        "integer",
        None,
        "False vertices in final partition",
        FinalB
    ),
    run_c!(
        "final_balance_penalty",
        "number",
        None,
        "Balance term in final objective",
        FinalPenalty
    ),
    run_c!(
        "best_cut_edges",
        "integer",
        None,
        "Cut edges in incumbent partition",
        BestCuts
    ),
    run_c!(
        "best_size_a",
        "integer",
        None,
        "True vertices in incumbent partition",
        BestA
    ),
    run_c!(
        "best_size_b",
        "integer",
        None,
        "False vertices in incumbent partition",
        BestB
    ),
    run_c!(
        "best_balance_penalty",
        "number",
        None,
        "Balance term in incumbent objective",
        BestPenalty
    ),
    run_c!(
        "elapsed_ms",
        "number",
        Some("ms"),
        "Run initialization, search and measurement time; excludes graph generation and disk write",
        Elapsed
    ),
    run_c!(
        "final_basin_real_from_real",
        "number",
        None,
        "Final real basin objective",
        FinalBasinReal
    ),
    run_c!(
        "final_basin_real_status",
        "string",
        None,
        "Final real basin termination",
        FinalBasinRealStatus
    ),
    run_c!(
        "final_basin_real_from_smoothed",
        "number",
        None,
        "Final smoothed basin real objective",
        FinalBasinSmooth
    ),
    run_c!(
        "final_basin_smoothed_status",
        "string",
        None,
        "Final smoothed basin termination",
        FinalBasinSmoothStatus
    ),
    run_c!(
        "final_basin_real_from_best",
        "number",
        None,
        "Final incumbent basin real objective",
        FinalBasinBest
    ),
    run_c!(
        "final_basin_best_status",
        "string",
        None,
        "Final incumbent basin termination",
        FinalBasinBestStatus
    ),
    run_c!(
        "applied_moves",
        "integer",
        None,
        "Applied search moves",
        Applied
    ),
    run_c!(
        "accepted_moves",
        "integer",
        None,
        "Applied moves (accepted moves for SA)",
        Accepted
    ),
    run_c!(
        "rejected_moves",
        "integer",
        None,
        "Rejected SA moves",
        Rejected
    ),
    run_c!(
        "objective_evaluations_search",
        "integer",
        None,
        "Objective evaluations during search",
        SearchEvals
    ),
    run_c!(
        "objective_evaluations_measurement",
        "integer",
        None,
        "Objective evaluations during measurement",
        MeasurementEvals
    ),
    run_c!(
        "fitness_values_computed_search",
        "integer",
        None,
        "EO fitness values computed during search",
        FitnessEvals
    ),
    run_c!("search_ms", "number", Some("ms"), "Search time", SearchMs),
    run_c!(
        "measurement_ms",
        "number",
        Some("ms"),
        "Measurement time",
        MeasurementMs
    ),
];
pub(super) const TRACE_COLUMNS: &[ColumnSpec<TraceField>] = &[
    trace_c!(
        "condition_id",
        "string",
        None,
        "Expanded condition identifier",
        Condition
    ),
    trace_c!("seed", "integer", None, "Run seed", Seed),
    trace_c!("status", "string", None, "Stored job status", Status),
    trace_c!("step", "integer", None, "Measurement step", Step),
    trace_c!(
        "current_real",
        "number",
        None,
        "Real objective of current partition",
        CurrentReal
    ),
    trace_c!(
        "best_real",
        "number",
        None,
        "Real objective of incumbent partition",
        BestReal
    ),
    trace_c!(
        "search_evaluation",
        "number",
        None,
        "Evaluation retained by search",
        SearchEvaluation
    ),
    trace_c!(
        "current_smoothed",
        "number",
        None,
        "Current smoothing evaluation",
        CurrentSmoothed
    ),
    trace_c!(
        "basin_real_from_real",
        "number",
        None,
        "Real basin real objective",
        BasinReal
    ),
    trace_c!(
        "basin_smoothed_from_real",
        "number",
        None,
        "Real basin smoothing objective",
        BasinRealSmoothed
    ),
    trace_c!(
        "basin_real_status",
        "string",
        None,
        "Real basin termination",
        BasinRealStatus
    ),
    trace_c!(
        "basin_real_steps",
        "integer",
        None,
        "Real basin scanned steps",
        BasinRealSteps
    ),
    trace_c!(
        "basin_real_from_smoothed",
        "number",
        None,
        "Smoothed basin real objective",
        BasinSmooth
    ),
    trace_c!(
        "basin_smoothed_from_smoothed",
        "number",
        None,
        "Smoothed basin smoothing objective",
        BasinSmoothValue
    ),
    trace_c!(
        "basin_smoothed_status",
        "string",
        None,
        "Smoothed basin termination",
        BasinSmoothStatus
    ),
    trace_c!(
        "basin_smoothed_steps",
        "integer",
        None,
        "Smoothed basin scanned steps",
        BasinSmoothSteps
    ),
    trace_c!(
        "basin_real_from_best",
        "number",
        None,
        "Incumbent basin real objective",
        BasinBest
    ),
    trace_c!(
        "basin_best_status",
        "string",
        None,
        "Incumbent basin termination",
        BasinBestStatus
    ),
    trace_c!(
        "basin_best_steps",
        "integer",
        None,
        "Incumbent basin scanned steps",
        BasinBestSteps
    ),
];
