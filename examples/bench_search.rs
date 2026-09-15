use gpp_utils::{
    experiment::{config::*, runner::run_one},
    fitness::FitnessRegistry,
    graph_partition::Graph,
    optimization::CancellationToken,
};
fn main() -> anyhow::Result<()> {
    let g = GraphSpec {
        kind: GraphKind::Random,
        node_count: 500,
        expected_degree: 5.0,
        seed: 42,
    };
    let token = CancellationToken::new();
    let graph = Graph::generate(&g, &token)?;
    let c = Condition {
        graph: g,
        neighborhood: Neighborhood::Flip,
        alpha: 0.05,
        solver: SolverSpec::Sa {
            temperature: 1.0,
            smoothing: SmoothingSpec::None,
        },
        budget: Budget {
            max_steps: 1_000_000,
        },
        measurement: Measurement {
            schedule: Schedule::Explicit,
            steps: vec![],
            basin: BasinMode::None,
            ..Default::default()
        },
    };
    for i in 0..4 {
        let r = run_one(&graph, &c, 42, &token, &FitnessRegistry::default())?;
        println!("{} {}", i, r.elapsed_ms);
    }
    Ok(())
}
