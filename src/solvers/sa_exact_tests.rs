//! Exact tests of the real-objective SA path; child module of `exact_tests`,
//! so they also run in the release exact regression of `scripts/check.py`.
//!
//! Production SA scores a proposal with the Metropolis memo
//! (`super::super::metropolis`), the cross-side swap score and a step loop
//! specialized by `Engine::advance`. The frozen `51577f9` engine evaluates
//! `(-delta / t).exp()` directly on every uphill proposal and scores moves
//! through the frozen smoothing module, so comparing the two after every step
//! checks the memo on many hits, misses and evictions.

use super::*;

fn random_graph(n: usize, p: f64, seed: u64) -> Graph {
    let mut rng = Mt19937GenRand64::new(seed);
    let mut edges = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            if rng.r#gen::<f64>() < p {
                edges.push([a, b]);
            }
        }
    }
    Graph::from_edges(n, edges).unwrap()
}

/// Zero, the ends and middle of the baseline range `10^-1.5 ..= 10^2.5`, and a
/// temperature at which every factor rounds to 1.
const TEMPERATURES: [f64; 8] = [
    0.0,
    0.03162277660168379,
    0.31622776601683794,
    1.0,
    3.1622776601683795,
    10.0,
    316.22776601683796,
    1.0e300,
];

fn sa(temperature: f64, smoothing: SmoothingSpec) -> SolverSpec {
    SolverSpec::Sa {
        temperature,
        smoothing,
    }
}

/// Every step of the real-objective SA equals the frozen engine: partition,
/// evaluation bits, counters and the complete state of all three streams.
#[test]
fn sa_real_matches_frozen_engine_on_random_graphs_and_temperatures() {
    let registry = FitnessRegistry::default_registry();
    let reference_registry = frozen_registry();
    let cancel = CancellationToken::new();
    let graphs = [
        random_graph(60, 0.1, 1),
        random_graph(124, 10.0 / 123.0, 2),
        eo_graph(),
        isolated_graph(),
        complete_graph(),
    ];
    let (mut accepted, mut uphill_draws) = (0u64, 0u64);
    for g in &graphs {
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            for temperature in TEMPERATURES {
                for (alpha, smoothing, seed) in [
                    (0.05, SmoothingSpec::None, 3),
                    (0.125, SmoothingSpec::WeightedAverage { k: 0 }, u64::MAX),
                    (-0.0, SmoothingSpec::None, 11),
                ] {
                    let c = condition_for(g, neighborhood, sa(temperature, smoothing), alpha);
                    let mut actual = Engine::new(g, &c, seed, &registry, &cancel).unwrap();
                    let mut expected =
                        reference::Engine::new(g, &c, seed, &reference_registry, &cancel).unwrap();
                    for step in 0..2000 {
                        let context = format!(
                            "n={} {neighborhood:?} T={temperature:e} alpha={alpha:e} step {step}",
                            g.node_count()
                        );
                        let before = actual.search_evaluation;
                        let draws = actual.select_rng.clone();
                        assert_eq!(actual.step(&cancel).unwrap(), StepStatus::Continue);
                        assert_eq!(
                            expected.step(&cancel).unwrap(),
                            reference::StepStatus::Continue
                        );
                        assert_eq!(
                            actual.state.partition(),
                            expected.state.partition(),
                            "{context}"
                        );
                        assert_eq!(
                            actual.search_evaluation.to_bits(),
                            expected.search_evaluation.to_bits(),
                            "{context}"
                        );
                        assert_eq!(
                            actual.objective_evaluations, expected.objective_evaluations,
                            "{context}"
                        );
                        assert_eq!(actual.applied_moves, expected.applied_moves, "{context}");
                        let [select, tie, smooth] = expected.rngs();
                        assert!(actual.select_rng == *select, "{context}");
                        assert!(actual.tie_rng == *tie, "{context}");
                        assert!(actual.smooth_rng == *smooth, "{context}");
                        accepted +=
                            u64::from(actual.search_evaluation.to_bits() != before.to_bits());
                        // A Flip proposal draws once; an extra draw is the
                        // acceptance draw of an uphill or tied proposal.
                        if neighborhood == Neighborhood::Flip {
                            let mut one = draws;
                            let _: usize = one.gen_range(0..g.node_count());
                            uphill_draws += u64::from(one != actual.select_rng);
                        }
                    }
                    assert_eq!(rng_probe(&actual, 624), expected.rng_probe(624));
                }
            }
        }
    }
    assert!(
        accepted > 10_000 && uphill_draws > 100_000,
        "{accepted} {uphill_draws}"
    );
}

/// Complete runner output (records, basins, diagnostics, best solution) of
/// real-objective SA equals the frozen runner across the temperature range,
/// with sparse logarithmic and every-step recording.
#[test]
fn sa_runner_matches_frozen_runner_across_temperatures() {
    let registry = FitnessRegistry::default_registry();
    let reference_registry = frozen_registry();
    let cancel = CancellationToken::new();
    let g = random_graph(40, 0.15, 7);
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        for temperature in TEMPERATURES {
            for all_steps in [false, true] {
                let mut c =
                    condition_for(&g, neighborhood, sa(temperature, SmoothingSpec::None), 0.05);
                c.measurement.max_basin_steps = 30;
                if all_steps {
                    c.budget.max_steps = 120;
                    c.measurement.steps = (0..=120).collect();
                } else {
                    c.budget.max_steps = 3000;
                    c.measurement.schedule = Schedule::Logarithmic;
                    c.measurement.steps = vec![];
                    c.measurement.basin = BasinMode::Real;
                }
                let actual =
                    crate::experiment::runner::run_one(&g, &c, 5150, &cancel, &registry).unwrap();
                let expected =
                    reference::runner::run_one(&g, &c, 5150, &cancel, &reference_registry).unwrap();
                assert_eq!(
                    exact_json(actual),
                    exact_json(expected),
                    "{neighborhood:?} T={temperature:e} all_steps={all_steps}"
                );
            }
        }
    }
}

/// Condition of every step kind: real and smoothed SA and HC (stopping at a
/// local optimum or a sampled one), EO and EO-SA.
fn step_kinds() -> Vec<SolverSpec> {
    vec![
        sa(1.0, SmoothingSpec::None),
        sa(0.0, SmoothingSpec::WeightedAverage { k: 0 }),
        sa(1.0, SmoothingSpec::AllAverage),
        SolverSpec::Hc {
            smoothing: SmoothingSpec::None,
        },
        SolverSpec::Hc {
            smoothing: SmoothingSpec::RandomKAverage { k: 2 },
        },
        SolverSpec::Hc {
            smoothing: SmoothingSpec::AllAverage,
        },
        SolverSpec::Eo {
            tau: 1.5,
            fitness: FitnessSpec::default(),
        },
        SolverSpec::EoSa {
            tau: 1.5,
            temperature: 0.5,
            fitness: FitnessSpec::default(),
        },
    ]
}

/// `advance` performs the per-step sequence of the runner's former loop
/// (`step`, count, observe; stop after a non-continuing status) for every
/// step kind and any segmentation of the steps.
#[test]
fn advance_runs_the_steps_of_the_runner_loop() {
    let registry = FitnessRegistry::default_registry();
    let cancel = CancellationToken::new();
    let g = random_graph(24, 0.2, 9);
    let mut stops = 0;
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        for solver in step_kinds() {
            let c = condition_for(&g, neighborhood, solver.clone(), 0.05);
            let mut advanced = Engine::new(&g, &c, 21, &registry, &cancel).unwrap();
            let mut stepped = Engine::new(&g, &c, 21, &registry, &cancel).unwrap();
            let (mut done_a, mut done_s) = (0u64, 0u64);
            let (mut trace_a, mut trace_s) = (Vec::new(), Vec::new());
            for until in [1, 2, 3, 4, 9, 10, 40, 41, 300] {
                let end = advanced.advance(until, &mut done_a, &cancel, |engine, completed| {
                    trace_a.push((
                        completed,
                        engine.search_evaluation.to_bits(),
                        engine.state.partition().to_vec(),
                    ))
                });
                let mut status = StepStatus::Continue;
                while done_s < until {
                    status = stepped.step(&cancel).unwrap();
                    done_s += 1;
                    trace_s.push((
                        done_s,
                        stepped.search_evaluation.to_bits(),
                        stepped.state.partition().to_vec(),
                    ));
                    if status != StepStatus::Continue {
                        break;
                    }
                }
                let context = format!("{neighborhood:?} {solver:?} until {until}");
                assert_eq!(done_a, done_s, "{context}");
                assert_eq!(trace_a, trace_s, "{context}");
                assert!(advanced.select_rng == stepped.select_rng, "{context}");
                assert!(advanced.tie_rng == stepped.tie_rng, "{context}");
                assert!(advanced.smooth_rng == stepped.smooth_rng, "{context}");
                assert!(advanced.accept_rng == stepped.accept_rng, "{context}");
                assert_eq!(
                    advanced.objective_evaluations,
                    stepped.objective_evaluations
                );
                assert_eq!(advanced.fitness_values, stepped.fitness_values);
                assert_eq!(advanced.applied_moves, stepped.applied_moves);
                match end {
                    Advance::Reached => assert_eq!(status, StepStatus::Continue, "{context}"),
                    Advance::Stopped(x) => {
                        assert_eq!(x, status, "{context}");
                        assert_ne!(x, StepStatus::Continue, "{context}");
                        stops += 1;
                        break;
                    }
                    other => panic!("{context}: {other:?}"),
                }
            }
        }
    }
    // Real and sampled HC stop in both neighborhoods; the smoothed ones may too.
    assert!(stops >= 4, "{stops}");
}

/// A cancelled token stops `advance` before any step, draw or count; a failing
/// step (non-finite score) is not counted or observed.
#[test]
fn advance_stops_before_a_cancelled_step_and_does_not_count_a_failed_one() {
    let registry = FitnessRegistry::default_registry();
    let live = CancellationToken::new();
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let g = random_graph(24, 0.2, 9);
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        for solver in step_kinds() {
            let c = condition_for(&g, neighborhood, solver, 0.05);
            let mut engine = Engine::new(&g, &c, 4, &registry, &live).unwrap();
            let fresh = Engine::new(&g, &c, 4, &registry, &live).unwrap();
            let mut done = 0;
            let end = engine.advance(10, &mut done, &cancelled, |_, _| panic!("no step"));
            assert!(matches!(end, Advance::Cancelled), "{end:?}");
            assert_eq!(done, 0);
            assert_eq!(engine.state.partition(), fresh.state.partition());
            assert!(engine.select_rng == fresh.select_rng);
            assert_eq!(engine.objective_evaluations, fresh.objective_evaluations);
        }
        // An infinite balance weight makes every proposal score non-finite.
        let c = condition_for(
            &g,
            neighborhood,
            sa(1.0, SmoothingSpec::None),
            f64::INFINITY,
        );
        let mut engine = Engine::new(&g, &c, 4, &registry, &live).unwrap();
        let mut reference = Engine::new(&g, &c, 4, &registry, &live).unwrap();
        let mut done = 3;
        let end = engine.advance(10, &mut done, &live, |_, _| panic!("no counted step"));
        let error = reference.step(&live).unwrap_err();
        match end {
            Advance::Failed(e) => assert_eq!(e.to_string(), error.to_string()),
            other => panic!("{other:?}"),
        }
        assert_eq!(done, 3);
        assert_eq!(
            engine.objective_evaluations,
            reference.objective_evaluations
        );
        assert!(engine.select_rng == reference.select_rng);
    }
}
