//! Independent-review tests for the SA step and partition-state speedup. A
//! child of `exact_tests`, so the frozen `51577f9` engine and runner are in
//! scope as `reference` and these tests run in the release exact regression.

use super::*;
use crate::solvers::metropolis::MetropolisFactor;
use std::collections::BTreeSet;

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

fn hc(smoothing: SmoothingSpec) -> SolverSpec {
    SolverSpec::Hc { smoothing }
}

/// HC (every smoothing) through the new runner loop against the frozen runner,
/// with the stopping step placed on a checkpoint, between checkpoints, exactly
/// at `max_steps`, one step after `max_steps`, and under a logarithmic schedule.
#[test]
fn hc_runner_stop_positions_match_frozen_runner() {
    let registry = FitnessRegistry::default_registry();
    let reference_registry = frozen_registry();
    let cancel = CancellationToken::new();
    let graphs = [
        random_graph(24, 0.2, 9),
        eo_graph(),
        graph(),
        complete_graph(),
        isolated_graph(),
    ];
    let mut compared = 0usize;
    let mut stops = BTreeSet::new();
    for g in &graphs {
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            if neighborhood == Neighborhood::Swap && g.node_count() % 2 == 1 {
                continue;
            }
            for smoothing in [
                SmoothingSpec::None,
                SmoothingSpec::WeightedAverage { k: 0 },
                SmoothingSpec::WeightedAverage { k: 2 },
                SmoothingSpec::AllAverage,
                SmoothingSpec::RandomKAverage { k: 1 },
                SmoothingSpec::RandomKAverage { k: 5 },
            ] {
                for seed in [1u64, 2] {
                    let mut c = condition_for(g, neighborhood, hc(smoothing.clone()), 0.05);
                    c.budget.max_steps = 100_000;
                    c.measurement.schedule = Schedule::Logarithmic;
                    c.measurement.steps = vec![];
                    c.measurement.max_basin_steps = 6;
                    let probe = crate::experiment::runner::run_one(g, &c, seed, &cancel, &registry)
                        .unwrap();
                    let k = probe.completed_steps;
                    assert!(k < c.budget.max_steps, "HC should stop: {c:?}");
                    stops.insert(format!("{:?}", probe.termination));
                    for variant in 0..6 {
                        let mut v = c.clone();
                        v.measurement.schedule = Schedule::Explicit;
                        match variant {
                            // The stopping step is a checkpoint.
                            0 => {
                                v.budget.max_steps = k + 7;
                                v.measurement.steps = BTreeSet::from([1, k]).into_iter().collect();
                            }
                            // Every step except the stopping one is a checkpoint.
                            1 => {
                                v.budget.max_steps = k + 7;
                                v.measurement.steps = (1..k).chain([k + 1, k + 7]).collect();
                            }
                            // The budget ends exactly at the stopping step.
                            2 => {
                                v.budget.max_steps = k;
                                v.measurement.steps = vec![];
                            }
                            // The budget ends one step before the stopping step.
                            3 => {
                                if k < 2 {
                                    continue;
                                }
                                v.budget.max_steps = k - 1;
                                v.measurement.steps = vec![];
                            }
                            // Sparse explicit checkpoints on both sides.
                            4 => {
                                v.budget.max_steps = 3 * k + 1;
                                v.measurement.steps = BTreeSet::from([k / 2, k + 1, 2 * k])
                                    .into_iter()
                                    .filter(|&s| s > 0)
                                    .collect();
                            }
                            _ => {
                                v.budget.max_steps = k + 1000;
                                v.measurement.schedule = Schedule::Logarithmic;
                                v.measurement.steps = vec![];
                                v.measurement.diagnostics = false;
                                v.measurement.basin = BasinMode::Real;
                            }
                        }
                        let actual =
                            crate::experiment::runner::run_one(g, &v, seed, &cancel, &registry)
                                .unwrap();
                        let expected =
                            reference::runner::run_one(g, &v, seed, &cancel, &reference_registry)
                                .unwrap();
                        assert_eq!(
                            exact_json(actual),
                            exact_json(expected),
                            "n={} {neighborhood:?} {smoothing:?} seed {seed} variant {variant} k {k}",
                            g.node_count()
                        );
                        compared += 1;
                    }
                }
            }
        }
    }
    assert!(compared > 250, "{compared}");
    assert!(
        stops.contains("LocalOptimum") && stops.contains("NoSampledImprovement"),
        "{stops:?}"
    );
}

/// Real and smoothed SA through the new runner against the frozen runner on
/// several graphs, seeds, alphas and baseline-range temperatures (sparse
/// logarithmic records with real basins and diagnostics).
#[test]
fn sa_runner_matches_frozen_runner_on_more_graphs_seeds_and_alphas() {
    let registry = FitnessRegistry::default_registry();
    let reference_registry = frozen_registry();
    let cancel = CancellationToken::new();
    let graphs = [
        random_graph(64, 5.0 / 63.0, 11),
        random_graph(50, 0.3, 12),
        eo_graph(),
    ];
    let temperatures = [
        0.0,
        1e-300,
        0.03,
        0.1778279410038923,
        1.0,
        5.623413251903491,
        316.0,
        1e300,
    ];
    for g in &graphs {
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            if neighborhood == Neighborhood::Swap && g.node_count() % 2 == 1 {
                continue;
            }
            for &temperature in &temperatures {
                for (alpha, smoothing) in [
                    (0.0, SmoothingSpec::None),
                    (0.05, SmoothingSpec::None),
                    (0.3, SmoothingSpec::WeightedAverage { k: 0 }),
                    (0.05, SmoothingSpec::RandomKAverage { k: 3 }),
                ] {
                    let smoothed = !matches!(
                        smoothing,
                        SmoothingSpec::None | SmoothingSpec::WeightedAverage { k: 0 }
                    );
                    let mut c = condition_for(
                        g,
                        neighborhood,
                        SolverSpec::Sa {
                            temperature,
                            smoothing,
                        },
                        alpha,
                    );
                    c.budget.max_steps = if smoothed { 100 } else { 2000 };
                    c.measurement.schedule = Schedule::Logarithmic;
                    c.measurement.steps = vec![];
                    c.measurement.basin = BasinMode::Real;
                    c.measurement.max_basin_steps = 50;
                    for seed in [77u64, 5] {
                        let actual =
                            crate::experiment::runner::run_one(g, &c, seed, &cancel, &registry)
                                .unwrap();
                        let expected =
                            reference::runner::run_one(g, &c, seed, &cancel, &reference_registry)
                                .unwrap();
                        assert_eq!(
                            exact_json(actual),
                            exact_json(expected),
                            "n={} {neighborhood:?} T={temperature:e} alpha={alpha} seed {seed}",
                            g.node_count()
                        );
                    }
                }
            }
        }
    }
}

/// Cancelling from inside the observer after step `k` stops `advance` before
/// step `k + 1` with no further draw, exactly like the runner's former loop
/// (which checked the token before every step).
#[test]
fn advance_cancelled_by_the_observer_stops_before_the_next_step() {
    let registry = FitnessRegistry::default_registry();
    let live = CancellationToken::new();
    let g = random_graph(24, 0.2, 9);
    let solvers = [
        SolverSpec::Sa {
            temperature: 0.3,
            smoothing: SmoothingSpec::None,
        },
        SolverSpec::Sa {
            temperature: 1e300,
            smoothing: SmoothingSpec::WeightedAverage { k: 0 },
        },
        SolverSpec::Sa {
            temperature: 0.0,
            smoothing: SmoothingSpec::None,
        },
        SolverSpec::Sa {
            temperature: 1.0,
            smoothing: SmoothingSpec::RandomKAverage { k: 2 },
        },
        hc(SmoothingSpec::None),
        hc(SmoothingSpec::AllAverage),
        SolverSpec::Eo {
            tau: 1.5,
            fitness: FitnessSpec::default(),
        },
        SolverSpec::EoSa {
            tau: 1.5,
            temperature: 0.5,
            fitness: FitnessSpec::default(),
        },
    ];
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        for solver in &solvers {
            let c = condition_for(&g, neighborhood, solver.clone(), 0.05);
            for k in [1u64, 2, 7, 33] {
                let token = CancellationToken::new();
                let mut advanced = Engine::new(&g, &c, 21, &registry, &live).unwrap();
                let mut stepped = Engine::new(&g, &c, 21, &registry, &live).unwrap();
                let mut done = 0u64;
                let end = advanced.advance(100, &mut done, &token, |_, completed| {
                    if completed == k {
                        token.cancel();
                    }
                });
                let mut done_s = 0u64;
                let mut last = StepStatus::Continue;
                while done_s < 100 {
                    last = stepped.step(&live).unwrap();
                    done_s += 1;
                    if done_s == k || last != StepStatus::Continue {
                        break;
                    }
                }
                let context = format!("{neighborhood:?} {solver:?} k {k}");
                match end {
                    Advance::Cancelled => assert_eq!(done, k, "{context}"),
                    Advance::Stopped(status) => {
                        assert!(done <= k, "{context}");
                        assert_eq!(status, last, "{context}");
                    }
                    other => panic!("{context}: {other:?}"),
                }
                assert_eq!(done, done_s, "{context}");
                assert_eq!(
                    advanced.state.partition(),
                    stepped.state.partition(),
                    "{context}"
                );
                assert_eq!(
                    advanced.search_evaluation.to_bits(),
                    stepped.search_evaluation.to_bits(),
                    "{context}"
                );
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
            }
        }
    }
}

/// Many keys sharing one slot, interleaved with +0.0 / -0.0 and with keys of
/// other slots, for temperatures of the baseline grid and extreme ones.
#[test]
fn memo_stays_exact_under_heavy_collisions() {
    let slot = |x: f64| crate::solvers::metropolis::slot(x.to_bits());
    let temperatures: Vec<f64> = (-150..=250)
        .step_by(5)
        .map(|v| 10f64.powf(v as f64 / 100.0))
        .chain([1e-300, 5e-324, 1e300, f64::MAX, f64::MIN_POSITIVE])
        .collect();
    // Deltas as SA produces them: differences of `cut + alpha * d^2` scores.
    let mut deltas = Vec::new();
    for alpha in [0.0, 0.05, 0.001, 0.3] {
        for cut in [0.0f64, 1.0, 17.0, 623.0, 2500.0] {
            for d in [0i64, 2, 4, 6, 10, 124] {
                let score = cut + alpha * d as f64 * d as f64;
                for dc in -4i64..=4 {
                    for dd in [-2i64, 0, 2] {
                        let d2 = d + dd;
                        let next = (cut + dc as f64) + alpha * d2 as f64 * d2 as f64;
                        deltas.push(next - score);
                    }
                }
            }
        }
    }
    deltas.extend([0.0, -0.0, f64::INFINITY, f64::NAN, 5e-324, -5e-324]);
    // Up to 6 keys sharing the slot of 1.0 and up to 6 sharing the slot of
    // +0.0 (whose initial entries hold +0.0), among nearby multiples of 1/64
    // (exactly representable) of both signs.
    let mut colliding = Vec::new();
    for target in [slot(1.0), slot(0.0)] {
        let keys: Vec<f64> = (1..4_000_000)
            .flat_map(|i| [i as f64 / 64.0, -(i as f64) / 64.0])
            .filter(|&x| slot(x) == target)
            .take(6)
            .collect();
        assert!(keys.len() >= 3, "{keys:?}");
        colliding.extend(keys);
    }
    let mut rng = Mt19937GenRand64::new(99);
    for &t in &temperatures {
        let mut memo = MetropolisFactor::new(t);
        for i in 0..4000 {
            let delta = match i % 3 {
                0 => colliding[rng.gen_range(0..colliding.len())],
                1 => deltas[rng.gen_range(0..deltas.len())],
                _ => {
                    if rng.gen_range(0..2) == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
            };
            let expected = (-std::hint::black_box(delta) / std::hint::black_box(t)).exp();
            assert_eq!(
                memo.get(delta).to_bits(),
                expected.to_bits(),
                "t {t:e} delta {delta:e}"
            );
        }
    }
}

/// `has_edge` agrees with the edge list for every pair of every graph size
/// 0..=130, and for every pair at 2047..=2050 vertices (matrix on both sides
/// of the cap); graphs are shareable and hash consistently across threads.
#[test]
fn has_edge_exhaustive_and_hash_across_threads() {
    let check = |g: &Graph| {
        let n = g.node_count();
        // Rows rebuilt from the edge list only (not from the adjacency lists).
        let mut rows = vec![Vec::new(); n];
        for &[a, b] in g.edges() {
            rows[a].push(b);
            rows[b].push(a);
        }
        let mut row = vec![false; n];
        for (a, neighbors) in rows.iter().enumerate() {
            row.fill(false);
            for &b in neighbors {
                row[b] = true;
            }
            for (b, &expected) in row.iter().enumerate() {
                assert_eq!(g.has_edge(a, b), expected, "n={n} {a}-{b}");
            }
            for b in [n, n + 1, n + 64, usize::MAX] {
                assert!(!g.has_edge(a, b));
            }
        }
    };
    for n in 0..=130usize {
        for (p, seed) in [(0.0, 1u64), (0.1, 2), (0.6, 3), (1.0, 4)] {
            check(&random_graph(n, p, seed + n as u64));
        }
    }
    for n in 2047..=2050usize {
        let mut edges = Vec::new();
        let mut rng = Mt19937GenRand64::new(n as u64);
        for a in 0..n {
            for _ in 0..3 {
                let b = rng.gen_range(0..n);
                if a != b {
                    edges.push([a.min(b), a.max(b)]);
                }
            }
        }
        edges.extend([[0, n - 1], [n - 2, n - 1], [63, 64], [2046, n - 1]]);
        edges.sort();
        edges.dedup();
        edges.retain(|e| e[0] != e[1]);
        check(&Graph::from_edges(n, edges).unwrap());
    }

    fn shareable<T: Send + Sync + std::panic::UnwindSafe + std::panic::RefUnwindSafe>() {}
    shareable::<Graph>();
    let g = random_graph(300, 0.05, 8);
    let copy_before = g.clone();
    let recomputed = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"gpp-graph-v1\0");
        h.update((g.node_count() as u64).to_le_bytes());
        for &[a, b] in g.edges() {
            h.update((a as u64).to_le_bytes());
            h.update((b as u64).to_le_bytes());
        }
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                for _ in 0..50 {
                    assert_eq!(g.content_hash(), recomputed);
                }
            });
        }
    });
    assert_eq!(copy_before.content_hash(), recomputed);
    assert_eq!(g.clone().content_hash(), recomputed);
}
