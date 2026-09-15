//! Reproducible, comparison-oriented search benchmark.
//!
//! Run with `cargo run --release --example bench_exact -- --help`.  The output is
//! TSV so a baseline executable and a changed executable can be diffed directly.
//! The deterministic signature intentionally omits wall-clock and attempt fields.

use gpp_utils::{
    experiment::{config::*, runner::run_one},
    fitness::FitnessRegistry,
    graph_partition::Graph,
    optimization::CancellationToken,
    storage::atomic,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{env, hint::black_box, path::Path, time::Instant};

const GRAPH_CASES: &[(usize, f64, u64)] = &[
    (500, 5.0, 42),
    (500, 5.0, 77),
    (500, 20.0, 42),
    (500, 20.0, 77),
    (2_000, 5.0, 42),
    (2_000, 5.0, 77),
    (2_000, 20.0, 42),
    (2_000, 20.0, 77),
];
const SEARCH_SEEDS: &[u64] = &[42, 77];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Hc,
    Sa,
    Eo,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phases {
    Compute,
    All,
    Basin,
}
impl Phases {
    fn computes(self) -> bool {
        matches!(self, Self::Compute | Self::All)
    }
    fn basins(self) -> bool {
        matches!(self, Self::All | Self::Basin)
    }
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Hc => "hc",
            Self::Sa => "sa",
            Self::Eo => "eo",
        }
    }
}

struct Options {
    modes: Vec<Mode>,
    neighborhoods: Vec<Neighborhood>,
    case_filter: Option<String>,
    repeats: usize,
    warmup: usize,
    sa_steps: u64,
    eo_steps: u64,
    hc_steps: u64,
    basin_steps: u64,
    phases: Phases,
}

fn usage() -> &'static str {
    "bench_exact [--mode hc|sa|eo|all] [--neighborhood flip|swap|all] \
--case n500-d5-g42[-s42] [--repeats N] [--warmup N] [--steps N] \
[--hc-steps N] [--sa-steps N] [--eo-steps N] [--basin-steps N] [--phases compute|all|basin]\n\
Defaults: mode=all, neighborhood=all, repeats=5, warmup=1, \
hc-steps=20, sa-steps=100000, eo-steps=250, basin-steps=3, phases=compute.\n\
`--basin` is an alias for `--phases all`; `--basin-only` is an alias for\n\
`--phases basin`. Each row is TSV. `compute` is run_one with basin=none;\n\
`basin_total` includes search and measurement, while `basin_measurement`\n\
comes from run diagnostics. `validate`, `serialize`, and `atomic_save` time post-run work."
}

fn take_value(args: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
}

fn parse_modes(value: &str) -> anyhow::Result<Vec<Mode>> {
    match value {
        "hc" => Ok(vec![Mode::Hc]),
        "sa" => Ok(vec![Mode::Sa]),
        "eo" => Ok(vec![Mode::Eo]),
        "all" => Ok(vec![Mode::Sa, Mode::Eo]),
        _ => anyhow::bail!("--mode must be hc, sa, eo, or all"),
    }
}
fn parse_neighborhoods(value: &str) -> anyhow::Result<Vec<Neighborhood>> {
    match value {
        "flip" => Ok(vec![Neighborhood::Flip]),
        "swap" => Ok(vec![Neighborhood::Swap]),
        "all" => Ok(vec![Neighborhood::Flip, Neighborhood::Swap]),
        _ => anyhow::bail!("--neighborhood must be flip, swap, or all"),
    }
}
fn parse_positive(value: String, flag: &str) -> anyhow::Result<u64> {
    let n = value.parse()?;
    if n == 0 {
        anyhow::bail!("{flag} must be positive");
    }
    Ok(n)
}
fn options() -> anyhow::Result<Options> {
    let mut out = Options {
        modes: vec![Mode::Sa, Mode::Eo],
        neighborhoods: vec![Neighborhood::Flip, Neighborhood::Swap],
        case_filter: None,
        repeats: 5,
        warmup: 1,
        sa_steps: 100_000,
        eo_steps: 250,
        hc_steps: 20,
        basin_steps: 3,
        phases: Phases::Compute,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            "--mode" => out.modes = parse_modes(&take_value(&mut args, "--mode")?)?,
            "--neighborhood" => {
                out.neighborhoods = parse_neighborhoods(&take_value(&mut args, "--neighborhood")?)?
            }
            "--case" => out.case_filter = Some(take_value(&mut args, "--case")?),
            "--repeats" => {
                out.repeats = usize::try_from(parse_positive(
                    take_value(&mut args, "--repeats")?,
                    "--repeats",
                )?)?
            }
            "--warmup" => out.warmup = take_value(&mut args, "--warmup")?.parse()?,
            "--steps" => {
                let n = parse_positive(take_value(&mut args, "--steps")?, "--steps")?;
                out.hc_steps = n;
                out.sa_steps = n;
                out.eo_steps = n;
            }
            "--sa-steps" => {
                out.sa_steps = parse_positive(take_value(&mut args, "--sa-steps")?, "--sa-steps")?
            }
            "--hc-steps" => {
                out.hc_steps = parse_positive(take_value(&mut args, "--hc-steps")?, "--hc-steps")?
            }
            "--eo-steps" => {
                out.eo_steps = parse_positive(take_value(&mut args, "--eo-steps")?, "--eo-steps")?
            }
            "--basin-steps" => {
                out.basin_steps =
                    parse_positive(take_value(&mut args, "--basin-steps")?, "--basin-steps")?
            }
            "--basin" => out.phases = Phases::All,
            "--basin-only" => out.phases = Phases::Basin,
            "--phases" => {
                out.phases = match take_value(&mut args, "--phases")?.as_str() {
                    "compute" => Phases::Compute,
                    "all" => Phases::All,
                    "basin" => Phases::Basin,
                    _ => anyhow::bail!("--phases must be compute, all, or basin"),
                }
            }
            _ => anyhow::bail!("unknown argument {arg}\n{}", usage()),
        }
    }
    Ok(out)
}

fn degree_name(degree: f64) -> String {
    if degree.fract() == 0.0 {
        format!("{degree:.0}")
    } else {
        degree.to_string()
    }
}
fn neighborhood_name(n: Neighborhood) -> &'static str {
    match n {
        Neighborhood::Flip => "flip",
        Neighborhood::Swap => "swap",
    }
}
fn case_name(nodes: usize, degree: f64, graph_seed: u64, search_seed: u64) -> String {
    format!(
        "n{nodes}-d{}-g{graph_seed}-s{search_seed}",
        degree_name(degree)
    )
}
fn matches_case(filter: Option<&str>, name: &str) -> bool {
    filter.is_none_or(|needle| name.contains(needle))
}

fn condition(
    graph: GraphSpec,
    mode: Mode,
    neighborhood: Neighborhood,
    steps: u64,
    basin: BasinMode,
    basin_steps: u64,
    diagnostics: bool,
) -> Condition {
    Condition {
        graph,
        neighborhood,
        alpha: 0.05,
        solver: match mode {
            Mode::Hc => SolverSpec::Hc {
                smoothing: SmoothingSpec::None,
            },
            Mode::Sa => SolverSpec::Sa {
                temperature: 1.0,
                smoothing: SmoothingSpec::None,
            },
            Mode::Eo => SolverSpec::Eo {
                tau: 1.5,
                fitness: FitnessSpec::default(),
            },
        },
        budget: Budget { max_steps: steps },
        measurement: Measurement {
            schedule: Schedule::Explicit,
            steps: Vec::new(),
            basin,
            max_basin_steps: basin_steps,
            diagnostics,
            best_basin: false,
        },
    }
}

fn normalize(value: &mut Value) {
    match value {
        Value::Array(values) => values.iter_mut().for_each(normalize),
        Value::Object(map) => {
            map.remove("attempt_id");
            map.remove("elapsed_ms");
            // Diagnostics are absent in this driver, but keep the signature useful
            // if that is changed later.
            if let Some(Value::Object(diagnostics)) = map.get_mut("diagnostics") {
                diagnostics.remove("search_ms");
                diagnostics.remove("measurement_ms");
            }
            map.values_mut().for_each(normalize);
        }
        _ => {}
    }
}
fn signature(result: &gpp_utils::RunResult) -> anyhow::Result<String> {
    let mut value = serde_json::to_value(result)?;
    normalize(&mut value);
    let bytes = serde_json::to_vec(&value)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[allow(clippy::too_many_arguments)]
fn row(
    phase: &str,
    mode: &str,
    neighborhood: &str,
    case: &str,
    nodes: usize,
    degree: f64,
    graph_seed: u64,
    search_seed: u64,
    rep: usize,
    steps: u64,
    elapsed_ns: u128,
    signature: &str,
) {
    println!(
        "{phase}\t{mode}\t{neighborhood}\t{case}\t{nodes}\t{}\t{graph_seed}\t{search_seed}\t{rep}\t{steps}\t{elapsed_ns}\t{signature}",
        degree_name(degree)
    );
}

fn timed_run(
    graph: &Graph,
    condition: &Condition,
    seed: u64,
    token: &CancellationToken,
    registry: &FitnessRegistry,
) -> anyhow::Result<(gpp_utils::RunResult, u128)> {
    let started = Instant::now();
    let result = run_one(graph, condition, seed, token, registry)?;
    let elapsed = started.elapsed().as_nanos();
    Ok((black_box(result), elapsed))
}

fn main() -> anyhow::Result<()> {
    let options = options()?;
    let token = CancellationToken::new();
    let registry = FitnessRegistry::default();
    let temp = tempfile::tempdir()?;
    println!(
        "phase\tmode\tneighborhood\tcase\tnodes\texpected_degree\tgraph_seed\tsearch_seed\trep\tsteps\telapsed_ns\tsignature"
    );

    for &(nodes, degree, graph_seed) in GRAPH_CASES {
        if !SEARCH_SEEDS.iter().copied().any(|search_seed| {
            matches_case(
                options.case_filter.as_deref(),
                &case_name(nodes, degree, graph_seed, search_seed),
            )
        }) {
            continue;
        }
        let graph_spec = GraphSpec {
            kind: GraphKind::Random,
            node_count: nodes,
            expected_degree: degree,
            seed: graph_seed,
        };
        let setup_started = Instant::now();
        let graph = Graph::generate(&graph_spec, &token)?;
        let setup_ns = setup_started.elapsed().as_nanos();
        row(
            "graph_setup",
            "-",
            "-",
            &format!("n{nodes}-d{}-g{graph_seed}", degree_name(degree)),
            nodes,
            degree,
            graph_seed,
            0,
            0,
            0,
            setup_ns,
            "-",
        );

        for &search_seed in SEARCH_SEEDS {
            let case = case_name(nodes, degree, graph_seed, search_seed);
            if !matches_case(options.case_filter.as_deref(), &case) {
                continue;
            }
            for &mode in &options.modes {
                let steps = match mode {
                    Mode::Hc => options.hc_steps,
                    Mode::Sa => options.sa_steps,
                    Mode::Eo => options.eo_steps,
                };
                for &neighborhood in &options.neighborhoods {
                    let compute = condition(
                        graph_spec.clone(),
                        mode,
                        neighborhood,
                        steps,
                        BasinMode::None,
                        options.basin_steps,
                        false,
                    );
                    let basin = condition(
                        graph_spec.clone(),
                        mode,
                        neighborhood,
                        steps,
                        BasinMode::Real,
                        options.basin_steps,
                        true,
                    );
                    for _ in 0..options.warmup {
                        if options.phases.computes() {
                            black_box(timed_run(&graph, &compute, search_seed, &token, &registry)?);
                        }
                        if options.phases.basins() {
                            black_box(timed_run(&graph, &basin, search_seed, &token, &registry)?);
                        }
                    }
                    for rep in 0..options.repeats {
                        if options.phases.computes() {
                            let (result, elapsed) =
                                timed_run(&graph, &compute, search_seed, &token, &registry)?;
                            let sig = signature(&result)?;
                            row(
                                "compute",
                                mode.name(),
                                neighborhood_name(neighborhood),
                                &case,
                                nodes,
                                degree,
                                graph_seed,
                                search_seed,
                                rep,
                                steps,
                                elapsed,
                                &sig,
                            );

                            let started = Instant::now();
                            result.validate(&graph, &compute)?;
                            row(
                                "validate",
                                mode.name(),
                                neighborhood_name(neighborhood),
                                &case,
                                nodes,
                                degree,
                                graph_seed,
                                search_seed,
                                rep,
                                steps,
                                started.elapsed().as_nanos(),
                                &sig,
                            );
                            let started = Instant::now();
                            let bytes = serde_json::to_vec(&result)?;
                            black_box(&bytes);
                            row(
                                "serialize",
                                mode.name(),
                                neighborhood_name(neighborhood),
                                &case,
                                nodes,
                                degree,
                                graph_seed,
                                search_seed,
                                rep,
                                steps,
                                started.elapsed().as_nanos(),
                                &sig,
                            );
                            let path = temp.path().join(format!(
                                "{}-{}-{}-{rep}.json",
                                mode.name(),
                                neighborhood_name(neighborhood),
                                case
                            ));
                            let started = Instant::now();
                            atomic::write_bytes(Path::new(&path), &bytes, false)?;
                            row(
                                "atomic_save",
                                mode.name(),
                                neighborhood_name(neighborhood),
                                &case,
                                nodes,
                                degree,
                                graph_seed,
                                search_seed,
                                rep,
                                steps,
                                started.elapsed().as_nanos(),
                                &sig,
                            );
                        }

                        if options.phases.basins() {
                            let (result, elapsed) =
                                timed_run(&graph, &basin, search_seed, &token, &registry)?;
                            let sig = signature(&result)?;
                            row(
                                "basin_total",
                                mode.name(),
                                neighborhood_name(neighborhood),
                                &case,
                                nodes,
                                degree,
                                graph_seed,
                                search_seed,
                                rep,
                                steps,
                                elapsed,
                                &sig,
                            );
                            let measurement_ns = result
                                .diagnostics
                                .as_ref()
                                .expect("basin benchmark enables diagnostics")
                                .measurement_ms
                                * 1_000_000.0;
                            row(
                                "basin_measurement",
                                mode.name(),
                                neighborhood_name(neighborhood),
                                &case,
                                nodes,
                                degree,
                                graph_seed,
                                search_seed,
                                rep,
                                steps,
                                measurement_ns as u128,
                                &sig,
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
