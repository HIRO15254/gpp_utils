//! Exact tests of the optimized smoothing (`crate::smoothing`); child module of
//! `exact_tests`, so they also run in the release exact regression of
//! `scripts/check.py`.
//!
//! `smoothing::evaluate` is compared directly with the frozen e4b6a1c copy
//! (`super::smoothing_e4b6a1c`, see `test_reference/README.md`): the returned
//! bits or error message, the evaluation count and the complete RNG state
//! afterwards. The cases cover Flip and Swap, every specification kind with
//! `k` across the distance-one and distance-two ranges up to the maximum,
//! tie-heavy and random graphs, many states, several alphas and seeds, a
//! missing RNG and a cancelled token. The engine and runner are compared with
//! the frozen engine and runner, which call the frozen copy.

use super::smoothing_e4b6a1c as frozen;
use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Includes signed zeros, a tiny and a huge value, infinity and NaN.
const ALPHAS: [f64; 10] = [
    0.05,
    0.0,
    -0.0,
    0.125,
    1.0 / 3.0,
    7.5,
    1.0e300,
    f64::MIN_POSITIVE,
    f64::INFINITY,
    f64::NAN,
];
const SEEDS: [u64; 3] = [0, 7, u64::MAX];

/// Evaluates `state` with production and frozen smoothing from identical RNG
/// copies and asserts identical outcomes. Returns whether a value was returned.
fn assert_same(
    graph: &Graph,
    state: &PartitionState,
    alpha: f64,
    neighborhood: Neighborhood,
    spec: &SmoothingSpec,
    seed: Option<u64>,
    cancel: &CancellationToken,
) -> bool {
    let mut actual_rng = seed.map(Mt19937GenRand64::new);
    let mut expected_rng = actual_rng.clone();
    let (mut actual_count, mut expected_count) = (11, 11);
    let actual = smoothing::evaluate(
        state,
        graph,
        alpha,
        neighborhood,
        spec,
        actual_rng.as_mut(),
        cancel,
        &mut actual_count,
    );
    let expected = frozen::evaluate(
        state,
        graph,
        alpha,
        neighborhood,
        spec,
        expected_rng.as_mut(),
        cancel,
        &mut expected_count,
    );
    let context = || {
        format!(
            "{neighborhood:?} {spec:?} alpha {alpha:e} seed {seed:?} partition {:?}",
            state.partition()
        )
    };
    match (&actual, &expected) {
        (Ok(a), Ok(e)) => assert_eq!(a.to_bits(), e.to_bits(), "value: {}", context()),
        (Err(a), Err(e)) => assert_eq!(a.to_string(), e.to_string(), "error: {}", context()),
        _ => panic!("{actual:?} != {expected:?}: {}", context()),
    }
    assert_eq!(actual_count, expected_count, "evaluations: {}", context());
    assert!(actual_rng == expected_rng, "RNG state: {}", context());
    actual.is_ok()
}

fn distance_one_count(state: &PartitionState, neighborhood: Neighborhood) -> usize {
    match neighborhood {
        Neighborhood::Flip => state.partition().len(),
        Neighborhood::Swap => state.size_a() * state.size_b(),
    }
}

/// Every specification kind. `random_k_average` covers `k = 0`, small `k`,
/// `k` around `m` and, if `distance_two`, the distance-two range up to one
/// past the maximum; otherwise only `k <= m` and one past the maximum.
fn specs(n: usize, neighborhood: Neighborhood, m: usize, distance_two: bool) -> Vec<SmoothingSpec> {
    let max = usize::try_from(smoothing::max_random_k(n, neighborhood)).unwrap();
    let mut ks = vec![0, 1, 2, 3, m.saturating_sub(1), m, max + 1];
    if distance_two {
        let middle = m + max.saturating_sub(m) / 2;
        ks.extend([m + 1, m + 2, middle, max.saturating_sub(1), max]);
    }
    ks.sort_unstable();
    ks.dedup();
    let mut specs = vec![SmoothingSpec::None, SmoothingSpec::AllAverage];
    specs.extend(ks.into_iter().map(|k| SmoothingSpec::RandomKAverage { k }));
    let mut weights = vec![0, 1, 2, m.saturating_sub(1), m, m + 1, usize::MAX];
    weights.sort_unstable();
    weights.dedup();
    specs.extend(
        weights
            .into_iter()
            .map(|k| SmoothingSpec::WeightedAverage { k }),
    );
    specs
}

/// Compares all specifications for `state`, with every alpha for the first
/// seed and the first alpha for the other seeds.
fn assert_state(
    graph: &Graph,
    state: &PartitionState,
    neighborhood: Neighborhood,
    distance_two: bool,
) {
    let cancel = CancellationToken::new();
    let m = distance_one_count(state, neighborhood);
    for spec in specs(graph.node_count(), neighborhood, m, distance_two) {
        for alpha in ALPHAS {
            assert_same(
                graph,
                state,
                alpha,
                neighborhood,
                &spec,
                Some(SEEDS[0]),
                &cancel,
            );
        }
        if matches!(spec, SmoothingSpec::RandomKAverage { .. }) {
            for seed in &SEEDS[1..] {
                assert_same(
                    graph,
                    state,
                    ALPHAS[0],
                    neighborhood,
                    &spec,
                    Some(*seed),
                    &cancel,
                );
            }
        }
    }
}

fn complete(n: usize) -> Graph {
    Graph::from_edges(
        n,
        (0..n)
            .flat_map(|a| (a + 1..n).map(move |b| [a, b]))
            .collect(),
    )
    .unwrap()
}

fn star(n: usize) -> Graph {
    Graph::from_edges(n, (1..n).map(|v| [0, v]).collect()).unwrap()
}

fn generated(kind: GraphKind, node_count: usize, expected_degree: f64, seed: u64) -> Graph {
    let spec = GraphSpec {
        kind,
        node_count,
        expected_degree,
        seed,
    };
    Graph::generate(&spec, &CancellationToken::new()).unwrap()
}

/// Tie-heavy and degenerate graphs with at most 12 vertices.
fn small_graphs() -> Vec<Graph> {
    let mut graphs = vec![
        Graph::from_edges(0, vec![]).unwrap(),
        Graph::from_edges(1, vec![]).unwrap(),
        Graph::from_edges(2, vec![]).unwrap(),
        Graph::from_edges(2, vec![[0, 1]]).unwrap(),
        Graph::from_edges(3, vec![[0, 1], [1, 2]]).unwrap(),
        Graph::from_edges(4, vec![[0, 1], [1, 2], [2, 3], [0, 3]]).unwrap(),
        complete(5),
        star(6),
        Graph::from_edges(7, vec![]).unwrap(),
        graph(),
        complete(9),
        star(11),
    ];
    graphs.push(generated(GraphKind::Random, 12, 3.0, 5));
    graphs
}

fn random_partition(n: usize, rng: &mut Mt19937GenRand64) -> Vec<bool> {
    let p = [0.5, 0.2, 0.8][rng.gen_range(0..3)];
    (0..n).map(|_| rng.gen_bool(p)).collect()
}

fn partition_with(n: usize, size_a: usize, rng: &mut Mt19937GenRand64) -> Vec<bool> {
    let mut partition = vec![false; n];
    partition[..size_a].fill(true);
    partition.shuffle(rng);
    partition
}

/// Flip: every partition up to 6 vertices, otherwise both uniform partitions
/// and random ones. Swap: the engine's balanced sizes (side A holds `n / 2`)
/// with every `k`, and other sizes with `k` up to the distance-one count.
fn assert_graph(graph: &Graph, random_states: usize) {
    let n = graph.node_count();
    let mut rng = Mt19937GenRand64::new(n as u64 + 1000);
    let mut flip_states: Vec<Vec<bool>> = if n <= 6 {
        (0u32..1 << n)
            .map(|bits| (0..n).map(|v| bits & (1 << v) != 0).collect())
            .collect()
    } else {
        vec![vec![false; n], vec![true; n]]
    };
    if n > 6 {
        flip_states.extend((0..random_states).map(|_| random_partition(n, &mut rng)));
    }
    for partition in flip_states {
        let state = PartitionState::new(graph, partition).unwrap();
        assert_state(graph, &state, Neighborhood::Flip, true);
    }
    for size_a in 0..=n {
        let balanced = size_a == n / 2;
        let count = if balanced { random_states } else { 1 };
        for _ in 0..count {
            let state = PartitionState::new(graph, partition_with(n, size_a, &mut rng)).unwrap();
            assert_state(graph, &state, Neighborhood::Swap, balanced);
        }
    }
}

#[test]
fn evaluate_matches_frozen_copy_on_small_and_tie_heavy_graphs() {
    for graph in small_graphs() {
        assert_graph(&graph, 3);
    }
}

/// Moves `count` distinct side-A and as many distinct side-B vertices (fewer
/// if a side is smaller) to the other side.
fn swap_pairs(partition: &mut [bool], count: usize, rng: &mut Mt19937GenRand64) {
    let mut sides = [true, false].map(|side| {
        (0..partition.len())
            .filter(|&v| partition[v] == side)
            .collect::<Vec<_>>()
    });
    let count = count.min(sides[0].len()).min(sides[1].len());
    for vertices in &mut sides {
        vertices.shuffle(rng);
        for &v in &vertices[..count] {
            partition[v] = !partition[v];
        }
    }
}

/// Successive Swap evaluations on one thread reuse the buffers of earlier
/// calls: walks of one or two swaps with reverts, 16 and 17 changed vertices,
/// jumps to unrelated partitions and graphs of other sizes, with side lists
/// and, from 128 vertices, rank selection for one sample.
#[test]
fn successive_swap_evaluations_match_frozen_copy() {
    let cancel = CancellationToken::new();
    let graphs = [
        generated(GraphKind::Random, 40, 5.0, 8),
        generated(GraphKind::Geometric, 70, 8.0, 9),
        complete(10),
        generated(GraphKind::Random, 161, 6.0, 10),
    ];
    let mut rng = Mt19937GenRand64::new(17);
    for _ in 0..4 {
        for graph in &graphs {
            let n = graph.node_count();
            let mut partition = partition_with(n, n / 2, &mut rng);
            let mut previous = partition.clone();
            for _ in 0..50 {
                let current = partition.clone();
                match rng.gen_range(0..12) {
                    0..=5 => swap_pairs(&mut partition, 1, &mut rng),
                    6 | 7 => swap_pairs(&mut partition, 2, &mut rng),
                    8 => partition = previous.clone(),
                    9 => swap_pairs(&mut partition, 8, &mut rng),
                    10 => {
                        swap_pairs(&mut partition, 8, &mut rng);
                        let v = rng.gen_range(0..n);
                        partition[v] = !partition[v];
                    }
                    _ => partition = partition_with(n, n / 2, &mut rng),
                }
                previous = current;
                let state = PartitionState::new(graph, partition.clone()).unwrap();
                let m = distance_one_count(&state, Neighborhood::Swap);
                let mut specs = vec![
                    SmoothingSpec::RandomKAverage { k: 1 },
                    SmoothingSpec::RandomKAverage { k: 5 },
                    SmoothingSpec::AllAverage,
                ];
                if state.size_a() == n / 2 {
                    specs.push(SmoothingSpec::RandomKAverage { k: m + 3 });
                }
                for spec in &specs {
                    let seed = rng.r#gen();
                    assert_same(
                        graph,
                        &state,
                        0.05,
                        Neighborhood::Swap,
                        spec,
                        Some(seed),
                        &cancel,
                    );
                }
            }
        }
    }
}

#[test]
fn evaluate_matches_frozen_copy_on_random_graphs() {
    let graphs = [
        eo_graph(),
        generated(GraphKind::Random, 17, 4.0, 1),
        generated(GraphKind::Geometric, 24, 6.0, 2),
        generated(GraphKind::Random, 31, 30.0, 3),
    ];
    for graph in graphs {
        assert_graph(&graph, 2);
    }
}

/// Larger identity permutations and side lists, with the `k` values of the
/// experiments, `k` around `m` and a distance-two sample. Swap selects the
/// vertices of up to `n / 128` samples by rank, so the sizes from 131 on also
/// take that path, including sizes that are odd or not multiples of 8.
#[test]
fn evaluate_matches_frozen_copy_on_large_graphs() {
    let cancel = CancellationToken::new();
    for (neighborhood, graph) in [
        (
            Neighborhood::Flip,
            generated(GraphKind::Random, 250, 10.0, 0),
        ),
        (
            Neighborhood::Flip,
            generated(GraphKind::Geometric, 125, 20.0, 4),
        ),
        (
            Neighborhood::Swap,
            generated(GraphKind::Random, 124, 10.0, 0),
        ),
        (
            Neighborhood::Swap,
            generated(GraphKind::Geometric, 65, 5.0, 6),
        ),
        (
            Neighborhood::Swap,
            generated(GraphKind::Random, 131, 10.0, 7),
        ),
        (
            Neighborhood::Swap,
            generated(GraphKind::Geometric, 262, 8.0, 8),
        ),
        (
            Neighborhood::Swap,
            generated(GraphKind::Random, 500, 10.0, 0),
        ),
    ] {
        let n = graph.node_count();
        let mut rng = Mt19937GenRand64::new(n as u64);
        for _ in 0..3 {
            let partition = match neighborhood {
                Neighborhood::Flip => random_partition(n, &mut rng),
                Neighborhood::Swap => partition_with(n, n / 2, &mut rng),
            };
            let state = PartitionState::new(&graph, partition).unwrap();
            let m = distance_one_count(&state, neighborhood);
            let mut specs = vec![
                SmoothingSpec::AllAverage,
                SmoothingSpec::WeightedAverage { k: 1 },
                SmoothingSpec::WeightedAverage { k: m / 2 },
            ];
            specs.extend(
                [1, 2, 3, 4, 32, m - 1, m, m + 1, m + 300]
                    .map(|k| SmoothingSpec::RandomKAverage { k }),
            );
            for spec in &specs {
                for seed in SEEDS {
                    assert_same(
                        &graph,
                        &state,
                        0.05,
                        neighborhood,
                        spec,
                        Some(seed),
                        &cancel,
                    );
                }
            }
        }
    }
}

#[test]
fn missing_rng_and_cancelled_token_match_frozen_copy() {
    let live = CancellationToken::new();
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    for graph in small_graphs() {
        let n = graph.node_count();
        let mut rng = Mt19937GenRand64::new(99);
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            let mut sizes = vec![0, n / 2, n];
            sizes.dedup();
            for size_a in sizes {
                let state =
                    PartitionState::new(&graph, partition_with(n, size_a, &mut rng)).unwrap();
                let m = distance_one_count(&state, neighborhood);
                let balanced = neighborhood == Neighborhood::Flip || size_a == n / 2;
                for spec in specs(n, neighborhood, m, balanced) {
                    for (seed, cancel) in [(None, &live), (None, &cancelled), (Some(3), &cancelled)]
                    {
                        assert_same(&graph, &state, 0.05, neighborhood, &spec, seed, cancel);
                    }
                }
            }
        }
    }
    // The missing RNG and the cancellation are reported, not ignored.
    let graph = graph();
    let state = PartitionState::new(
        &graph,
        vec![true, false, true, false, true, false, true, false],
    )
    .unwrap();
    let spec = SmoothingSpec::RandomKAverage { k: 3 };
    let error = |neighborhood, rng: Option<&mut Mt19937GenRand64>, cancel| {
        smoothing::evaluate(
            &state,
            &graph,
            0.05,
            neighborhood,
            &spec,
            rng,
            cancel,
            &mut 0,
        )
        .unwrap_err()
        .to_string()
    };
    assert_eq!(
        error(Neighborhood::Flip, None, &live),
        "random smoothing requires RNG"
    );
    assert_eq!(
        error(Neighborhood::Flip, None, &cancelled),
        "random smoothing requires RNG"
    );
    assert_eq!(
        error(Neighborhood::Swap, None, &cancelled),
        "operation cancelled"
    );
    let mut rng = Mt19937GenRand64::new(1);
    assert_eq!(
        error(Neighborhood::Flip, Some(&mut rng), &cancelled),
        "operation cancelled"
    );
}

/// Swap states whose side sizes differ from the engine's balanced sizes have
/// distance-two ordinals outside the pair ranges; both implementations panic
/// on the same inputs, after the same draws. A later evaluation on the same
/// thread is unaffected.
#[test]
fn irregular_swap_distance_two_panics_like_frozen_copy() {
    let mut panics = 0;
    let mut values = 0;
    for (n, size_a) in [(7, 4), (10, 3), (10, 1), (9, 8), (12, 2), (8, 6), (11, 7)] {
        let graph = generated(GraphKind::Random, n, 3.0, n as u64);
        let mut rng = Mt19937GenRand64::new(size_a as u64);
        let state = PartitionState::new(&graph, partition_with(n, size_a, &mut rng)).unwrap();
        let m = distance_one_count(&state, Neighborhood::Swap);
        let max = smoothing::max_random_k(n, Neighborhood::Swap) as usize;
        for k in [m + 1, m + 2, (m + max) / 2, max] {
            if k <= m || k > max {
                continue;
            }
            let spec = SmoothingSpec::RandomKAverage { k };
            for seed in SEEDS {
                let cancel = CancellationToken::new();
                let run = |evaluate: &dyn Fn(&mut Mt19937GenRand64, &mut u64) -> Result<f64>| {
                    let mut rng = Mt19937GenRand64::new(seed);
                    let mut count = 0;
                    let outcome = catch_unwind(AssertUnwindSafe(|| evaluate(&mut rng, &mut count)));
                    (outcome.map(|x| x.unwrap().to_bits()).ok(), count, rng)
                };
                let actual = run(&|rng, count| {
                    smoothing::evaluate(
                        &state,
                        &graph,
                        0.05,
                        Neighborhood::Swap,
                        &spec,
                        Some(rng),
                        &cancel,
                        count,
                    )
                });
                let expected = run(&|rng, count| {
                    frozen::evaluate(
                        &state,
                        &graph,
                        0.05,
                        Neighborhood::Swap,
                        &spec,
                        Some(rng),
                        &cancel,
                        count,
                    )
                });
                assert_eq!(
                    actual.0, expected.0,
                    "n {n} size_a {size_a} k {k} seed {seed}"
                );
                assert_eq!(actual.1, expected.1);
                assert!(actual.2 == expected.2);
                if actual.0.is_some() {
                    values += 1;
                } else {
                    panics += 1;
                }
                let balanced =
                    PartitionState::new(&graph, partition_with(n, n / 2, &mut rng)).unwrap();
                let spec = SmoothingSpec::RandomKAverage {
                    k: distance_one_count(&balanced, Neighborhood::Swap) + 1,
                };
                assert_same(
                    &graph,
                    &balanced,
                    0.05,
                    Neighborhood::Swap,
                    &spec,
                    Some(seed),
                    &cancel,
                );
            }
        }
    }
    assert!(panics > 0 && values > 0, "{panics} panics, {values} values");
}

/// The functions that other modules use unchanged still match the frozen copy.
#[test]
fn unchanged_helpers_match_frozen_copy() {
    let cancel = CancellationToken::new();
    for graph in small_graphs().into_iter().chain([eo_graph()]) {
        let n = graph.node_count();
        let mut rng = Mt19937GenRand64::new(5);
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            assert_eq!(
                smoothing::max_random_k(n, neighborhood),
                frozen::max_random_k(n, neighborhood)
            );
            for k in 0..=frozen::max_random_k(n, neighborhood) as usize + 1 {
                for spec in [
                    SmoothingSpec::RandomKAverage { k },
                    SmoothingSpec::WeightedAverage { k },
                ] {
                    assert_eq!(
                        smoothing::validate(&spec, n, neighborhood).map_err(|e| e.to_string()),
                        frozen::validate(&spec, n, neighborhood).map_err(|e| e.to_string())
                    );
                }
            }
            for _ in 0..4 {
                let state = PartitionState::new(&graph, random_partition(n, &mut rng)).unwrap();
                let moves = smoothing::moves(&state, neighborhood);
                assert_eq!(moves, frozen::moves(&state, neighborhood));
                assert_eq!(
                    moves,
                    smoothing::moves_cancellable(&state, neighborhood, &cancel).unwrap()
                );
                for &mv in &moves {
                    assert_eq!(
                        smoothing::move_score(&state, &graph, mv, 0.05).to_bits(),
                        frozen::move_score(&state, &graph, mv, 0.05).to_bits()
                    );
                    let mut actual = state.clone();
                    let mut expected = state.clone();
                    smoothing::apply(&mut actual, &graph, mv);
                    frozen::apply(&mut expected, &graph, mv);
                    assert_eq!(actual.partition(), expected.partition());
                    assert_eq!(actual.score(0.05).to_bits(), expected.score(0.05).to_bits());
                }
            }
        }
    }
}

/// Engine steps with every smoothing kind, `k` in both ranges and the
/// maximum, after every step: partitions, evaluation bits, counters and the
/// next 624 outputs of each RNG stream.
#[test]
fn smoothed_engine_matches_frozen_engine_for_every_k() {
    for g in [graph(), complete_graph(), isolated_graph(), eo_graph()] {
        let n = g.node_count();
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            let m = match neighborhood {
                Neighborhood::Flip => n,
                Neighborhood::Swap => n / 2 * (n - n / 2),
            };
            let max = smoothing::max_random_k(n, neighborhood) as usize;
            let mut ks = vec![1, 2, m - 1, m, m + 1, m + 7];
            if n <= 8 {
                ks.push(max);
            }
            ks.sort_unstable();
            ks.dedup();
            let mut specs: Vec<_> = ks
                .into_iter()
                .map(|k| SmoothingSpec::RandomKAverage { k })
                .collect();
            specs.extend([
                SmoothingSpec::AllAverage,
                SmoothingSpec::WeightedAverage { k: 1 },
                SmoothingSpec::WeightedAverage { k: m },
            ]);
            for smoothing in specs {
                let sa = condition_for(
                    &g,
                    neighborhood,
                    SolverSpec::Sa {
                        temperature: 1.0,
                        smoothing: smoothing.clone(),
                    },
                    0.125,
                );
                assert_engine_exact_on(&g, &sa, 4321, 40);
                let hc = condition_for(&g, neighborhood, SolverSpec::Hc { smoothing }, 0.05);
                assert_engine_exact_on(&g, &hc, 1234, 6);
            }
        }
    }
}

/// Complete runner results with smoothed measurements and smoothed basins.
#[test]
fn smoothed_runner_output_matches_frozen_runner() {
    let g = graph();
    let registry = FitnessRegistry::default_registry();
    let reference_registry = frozen_registry();
    let cancel = CancellationToken::new();
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        let m = match neighborhood {
            Neighborhood::Flip => 8,
            Neighborhood::Swap => 16,
        };
        let max = smoothing::max_random_k(8, neighborhood) as usize;
        let mut specs: Vec<_> = [1, m, m + 3, max]
            .into_iter()
            .map(|k| SmoothingSpec::RandomKAverage { k })
            .collect();
        specs.extend([
            SmoothingSpec::AllAverage,
            SmoothingSpec::WeightedAverage { k: 3 },
        ]);
        for smoothing in specs {
            for solver in [
                SolverSpec::Sa {
                    temperature: 1.0,
                    smoothing: smoothing.clone(),
                },
                SolverSpec::Hc {
                    smoothing: smoothing.clone(),
                },
            ] {
                let c = condition(neighborhood, solver, 0.05);
                let actual =
                    crate::experiment::runner::run_one(&g, &c, 7788, &cancel, &registry).unwrap();
                let expected =
                    reference::runner::run_one(&g, &c, 7788, &cancel, &reference_registry).unwrap();
                assert_eq!(exact_json(actual), exact_json(expected), "{c:?}");
            }
        }
    }
}
