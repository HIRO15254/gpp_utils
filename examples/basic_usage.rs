use gpp_utils::experiment::{plan::minimal_sample_toml, result::RunView};
use gpp_utils::fitness::FitnessRegistry;
use gpp_utils::graph_partition::Graph;
use gpp_utils::optimization::CancellationToken;
use gpp_utils::{compile_experiment, run_one};

fn main() -> anyhow::Result<()> {
    let plan = compile_experiment(toml::from_str(minimal_sample_toml())?)?;
    let job = &plan.jobs[0];
    let cancel = CancellationToken::new();
    let graph = Graph::generate(&job.condition.graph, &cancel)?;
    let result = run_one(
        &graph,
        &job.condition,
        job.seed,
        &cancel,
        &FitnessRegistry::default(),
    )?;
    let view = RunView::new(&graph, &job.condition, &result)?;
    println!(
        "steps={} final={} incumbent={}",
        result.completed_steps,
        view.final_score(),
        view.best_score()
    );
    Ok(())
}
