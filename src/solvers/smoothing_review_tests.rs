//! Independent-review adversarial differential tests of the optimized
//! `crate::smoothing::evaluate` against the frozen e4b6a1c copy.
//!
//! Every comparison runs both implementations under `catch_unwind` from
//! identical RNG copies and evaluation counters and compares: returned bits,
//! error message or panic message; the evaluation counter; the complete RNG
//! state. After every production call the thread-local scratch must be free
//! (not borrowed) and its identity permutation intact.

use super::smoothing_e4b6a1c as frozen;
use super::*;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[derive(Debug, PartialEq)]
enum Outcome {
    Value(u64),
    Error(String),
    Panic(String),
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".into()
    }
}

type EvalFn = fn(
    &PartitionState,
    &Graph,
    f64,
    Neighborhood,
    &SmoothingSpec,
    Option<&mut Mt19937GenRand64>,
    &CancellationToken,
    &mut u64,
) -> Result<f64>;

#[allow(clippy::too_many_arguments)]
fn run(
    f: EvalFn,
    graph: &Graph,
    state: &PartitionState,
    alpha: f64,
    neighborhood: Neighborhood,
    spec: &SmoothingSpec,
    rng: &mut Option<Mt19937GenRand64>,
    cancel: &CancellationToken,
    count: &mut u64,
) -> Outcome {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        f(
            state,
            graph,
            alpha,
            neighborhood,
            spec,
            rng.as_mut(),
            cancel,
            count,
        )
    }));
    match outcome {
        Ok(Ok(x)) => Outcome::Value(x.to_bits()),
        Ok(Err(e)) => Outcome::Error(e.to_string()),
        Err(p) => Outcome::Panic(panic_message(p)),
    }
}

fn assert_scratch_ok(context: &dyn Fn() -> String) {
    let (_, intact, free) = crate::smoothing::scratch_status();
    assert!(free, "scratch still borrowed: {}", context());
    assert!(intact, "identity permutation not restored: {}", context());
}

/// Compares production and frozen smoothing; returns the outcome.
#[allow(clippy::too_many_arguments)]
fn compare(
    graph: &Graph,
    state: &PartitionState,
    alpha: f64,
    neighborhood: Neighborhood,
    spec: &SmoothingSpec,
    seed: Option<u64>,
    cancel: &CancellationToken,
    initial_count: u64,
) -> Outcome {
    let mut actual_rng = seed.map(Mt19937GenRand64::new);
    let mut expected_rng = actual_rng.clone();
    let (mut actual_count, mut expected_count) = (initial_count, initial_count);
    let actual = run(
        crate::smoothing::evaluate,
        graph,
        state,
        alpha,
        neighborhood,
        spec,
        &mut actual_rng,
        cancel,
        &mut actual_count,
    );
    let context = || {
        format!(
            "n {} {neighborhood:?} {spec:?} alpha {alpha:e} seed {seed:?} size_a {} cancelled {}",
            graph.node_count(),
            state.size_a(),
            cancel.is_cancelled(),
        )
    };
    assert_scratch_ok(&context);
    let expected = run(
        frozen::evaluate,
        graph,
        state,
        alpha,
        neighborhood,
        spec,
        &mut expected_rng,
        cancel,
        &mut expected_count,
    );
    assert_eq!(actual, expected, "outcome: {}", context());
    assert_eq!(actual_count, expected_count, "evaluations: {}", context());
    assert!(actual_rng == expected_rng, "RNG state: {}", context());
    actual
}

fn random_graph(n: usize, rng: &mut Mt19937GenRand64) -> Graph {
    match rng.gen_range(0..8) {
        0 => Graph::from_edges(n, vec![]).unwrap(),
        1 => Graph::from_edges(
            n,
            (0..n)
                .flat_map(|a| (a + 1..n).map(move |b| [a, b]))
                .collect(),
        )
        .unwrap(),
        2 if n > 1 => Graph::from_edges(n, (1..n).map(|v| [0, v]).collect()).unwrap(),
        3 if n >= 2 => {
            let spec = GraphSpec {
                kind: GraphKind::Geometric,
                node_count: n,
                expected_degree: rng.gen_range(0.0..(n - 1) as f64),
                seed: rng.r#gen(),
            };
            Graph::generate(&spec, &CancellationToken::new()).unwrap()
        }
        _ => {
            let p = [0.02, 0.1, 0.3, 0.6, 0.95][rng.gen_range(0..5)];
            let mut edges = Vec::new();
            for a in 0..n {
                for b in a + 1..n {
                    if rng.gen_bool(p) {
                        // Random endpoint order exercises normalization.
                        edges.push(if rng.gen_bool(0.5) { [a, b] } else { [b, a] });
                    }
                }
            }
            Graph::from_edges(n, edges).unwrap()
        }
    }
}

fn random_partition(n: usize, rng: &mut Mt19937GenRand64) -> Vec<bool> {
    let with_size = |size_a: usize, rng: &mut Mt19937GenRand64| {
        let mut p = vec![false; n];
        p[..size_a.min(n)].fill(true);
        p.shuffle(rng);
        p
    };
    match rng.gen_range(0..10) {
        0..=3 => with_size(n / 2, rng),
        4 => with_size(n.div_ceil(2), rng),
        5 => with_size(if rng.gen_bool(0.5) { 0 } else { n }, rng),
        6 => with_size(
            if rng.gen_bool(0.5) {
                1
            } else {
                n.saturating_sub(1)
            },
            rng,
        ),
        7 => with_size(
            if rng.gen_bool(0.5) {
                2
            } else {
                n.saturating_sub(2)
            },
            rng,
        ),
        _ => {
            let p = [0.5, 0.1, 0.9, 0.01, 0.99][rng.gen_range(0..5)];
            (0..n).map(|_| rng.gen_bool(p)).collect()
        }
    }
}

fn random_alpha(rng: &mut Mt19937GenRand64) -> f64 {
    match rng.gen_range(0..16) {
        0 => 0.0,
        1 => -0.0,
        2 => f64::INFINITY,
        3 => f64::NEG_INFINITY,
        4 => f64::NAN,
        5 => 1.0e300,
        6 => f64::MIN_POSITIVE,
        7 => -0.05,
        8 => 1.0 / 3.0,
        9 => 5e-324,
        10 => f64::from_bits(0x7ff8_0000_0000_0123), // NaN with payload
        11 => -f64::NAN,
        _ => rng.gen_range(0.0..2.0),
    }
}

fn distance_one_count(state: &PartitionState, neighborhood: Neighborhood) -> usize {
    match neighborhood {
        Neighborhood::Flip => state.partition().len(),
        Neighborhood::Swap => state.size_a() * state.size_b(),
    }
}

/// A random specification; distance-two `k` keeps the number of distance-two
/// samples below `max_needed`.
fn random_spec(
    n: usize,
    neighborhood: Neighborhood,
    m: usize,
    max_needed: usize,
    rng: &mut Mt19937GenRand64,
) -> SmoothingSpec {
    let max = crate::smoothing::max_random_k(n, neighborhood).min(usize::MAX as u128) as usize;
    match rng.gen_range(0..12) {
        0 => SmoothingSpec::None,
        1 => SmoothingSpec::AllAverage,
        2 | 3 => {
            let k = match rng.gen_range(0..7) {
                0 => 0,
                1 => 1,
                2 => m,
                3 => m + 1,
                4 => usize::MAX,
                _ => rng.gen_range(0..=m + 2),
            };
            SmoothingSpec::WeightedAverage { k }
        }
        _ => {
            let k = match rng.gen_range(0..12) {
                0 => 0,
                1 => 1,
                2 => 2,
                3 => m,
                4 => m.saturating_sub(1),
                5 => m + 1,
                6 => max + 1,
                7 if max.saturating_sub(m) <= max_needed => max,
                7 | 8 => m + rng.gen_range(1..=max_needed.max(1)),
                9 => m + rng.gen_range(1..=8),
                _ => rng.gen_range(1..=m.max(1)),
            };
            SmoothingSpec::RandomKAverage { k }
        }
    }
}

fn random_seed(rng: &mut Mt19937GenRand64) -> Option<u64> {
    match rng.gen_range(0..20) {
        0 => None,
        1 => Some(0),
        2 => Some(u64::MAX),
        _ => Some(rng.r#gen()),
    }
}

/// Iteration budget: `release` in optimized builds (the release exact
/// regression of `scripts/check.py`), a sixteenth in debug builds; the
/// environment variable `REVIEW_ITERS` overrides both.
fn budget(release: usize) -> usize {
    std::env::var("REVIEW_ITERS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(if cfg!(debug_assertions) {
            (release / 16).max(1)
        } else {
            release
        })
}

/// Stride that thins deterministic sweeps in debug builds.
fn debug_stride(debug: usize) -> usize {
    if cfg!(debug_assertions) { debug } else { 1 }
}

#[derive(Default, Debug)]
struct Tally {
    values: usize,
    errors: usize,
    panics: usize,
    distance_two: usize,
}

impl Tally {
    fn add(&mut self, outcome: &Outcome) {
        match outcome {
            Outcome::Value(_) => self.values += 1,
            Outcome::Error(_) => self.errors += 1,
            Outcome::Panic(_) => self.panics += 1,
        }
    }
}

/// Random graphs, states, specifications, alphas and seeds on one thread, so
/// every call reuses the scratch of earlier calls of other sizes.
#[test]
fn review_random_differential() {
    let mut rng = Mt19937GenRand64::new(
        std::env::var("REVIEW_SEED")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(20260926),
    );
    let live = CancellationToken::new();
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let mut tally = Tally::default();
    for _ in 0..budget(800) {
        let n = match rng.gen_range(0..20) {
            0..=6 => rng.gen_range(0..=12),
            7..=14 => rng.gen_range(13..=64),
            15..=18 => rng.gen_range(65..=300),
            _ => rng.gen_range(1000..=1400),
        };
        let graph = random_graph(n, &mut rng);
        for _ in 0..rng.gen_range(1..=4) {
            let state = PartitionState::new(&graph, random_partition(n, &mut rng)).unwrap();
            for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
                if n >= 1000 && neighborhood == Neighborhood::Swap && rng.gen_bool(0.7) {
                    continue;
                }
                let m = distance_one_count(&state, neighborhood);
                for _ in 0..rng.gen_range(1..=4) {
                    let max_needed = if n > 300 { 300 } else { 3000 };
                    let spec = random_spec(n, neighborhood, m, max_needed, &mut rng);
                    if matches!(spec, SmoothingSpec::RandomKAverage { k } if k > m && m > 0) {
                        tally.distance_two += 1;
                    }
                    let alpha = random_alpha(&mut rng);
                    let seed = random_seed(&mut rng);
                    let cancel = if rng.gen_ratio(1, 12) {
                        &cancelled
                    } else {
                        &live
                    };
                    let outcome = compare(
                        &graph,
                        &state,
                        alpha,
                        neighborhood,
                        &spec,
                        seed,
                        cancel,
                        rng.gen_range(0..1000),
                    );
                    tally.add(&outcome);
                }
            }
        }
    }
    eprintln!("review_random_differential: {tally:?}");
    assert!(tally.values > 0 && tally.errors > 0 && tally.panics > 0);
}

/// Unbalanced Swap states in the distance-two range: panics must match
/// including the panic message, and the scratch stays usable.
#[test]
fn review_irregular_swap_distance_two_messages() {
    let mut rng = Mt19937GenRand64::new(77);
    let mut messages = std::collections::BTreeMap::<String, usize>::new();
    for n in (2..=24).step_by(debug_stride(5)) {
        let graph = random_graph(n, &mut rng);
        for size_a in 1..n {
            let mut p = vec![false; n];
            p[..size_a].fill(true);
            p.shuffle(&mut rng);
            let state = PartitionState::new(&graph, p).unwrap();
            let m = size_a * (n - size_a);
            let max = crate::smoothing::max_random_k(n, Neighborhood::Swap) as usize;
            let mut ks: Vec<usize> = vec![m + 1, m + 2, max, max.saturating_sub(1)];
            for _ in 0..4 {
                if max > m {
                    ks.push(rng.gen_range(m + 1..=max));
                }
            }
            for k in ks {
                if k <= m || k > max {
                    continue;
                }
                for _ in 0..3 {
                    let outcome = compare(
                        &graph,
                        &state,
                        0.05,
                        Neighborhood::Swap,
                        &SmoothingSpec::RandomKAverage { k },
                        Some(rng.r#gen()),
                        &CancellationToken::new(),
                        5,
                    );
                    let key = match outcome {
                        Outcome::Value(_) => "value".to_string(),
                        Outcome::Error(e) => format!("error: {e}"),
                        Outcome::Panic(p) => format!("panic: {p}"),
                    };
                    *messages.entry(key).or_default() += 1;
                }
            }
        }
    }
    eprintln!("irregular swap distance-two outcomes: {messages:#?}");
}

/// Real asynchronous cancellation from another thread at random times.
#[test]
fn review_async_cancellation_then_reuse() {
    let spec = GraphSpec {
        kind: GraphKind::Random,
        node_count: 400,
        expected_degree: 8.0,
        seed: 11,
    };
    let graph = Graph::generate(&spec, &CancellationToken::new()).unwrap();
    let n = graph.node_count();
    let mut rng = Mt19937GenRand64::new(5);
    let mut p = vec![false; n];
    p[..n / 2].fill(true);
    p.shuffle(&mut rng);
    let state = PartitionState::new(&graph, p).unwrap();
    let m = n / 2 * (n - n / 2);
    let mut cancelled_calls = 0;
    let rounds = budget(300);
    for round in 0..rounds {
        let cancel = CancellationToken::new();
        let delay = std::time::Duration::from_micros(rng.gen_range(0..3000));
        let remote = cancel.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(delay);
            remote.cancel();
        });
        let k = [m, m + 3000, 20000, m / 2][round % 4];
        let mut r = Mt19937GenRand64::new(round as u64);
        loop {
            let outcome = crate::smoothing::evaluate(
                &state,
                &graph,
                0.05,
                Neighborhood::Swap,
                &SmoothingSpec::RandomKAverage { k },
                Some(&mut r),
                &cancel,
                &mut 0,
            );
            assert_scratch_ok(&|| format!("round {round}"));
            if outcome.is_err() {
                cancelled_calls += 1;
                break;
            }
        }
        handle.join().unwrap();
        compare(
            &graph,
            &state,
            0.05,
            Neighborhood::Swap,
            &SmoothingSpec::RandomKAverage { k: m + 17 },
            Some(round as u64),
            &CancellationToken::new(),
            0,
        );
    }
    assert_eq!(cancelled_calls, rounds);
}

/// Explicit size sequences on one thread: large, then small, 0, 1 and large
/// again, with full shuffles that permute the whole identity.
#[test]
fn review_size_sequences_on_one_thread() {
    let mut rng = Mt19937GenRand64::new(99);
    let sizes = [
        500, 3, 0, 1, 2, 64, 500, 7, 1, 300, 0, 128, 129, 1000, 2, 500,
    ];
    for (i, &n) in sizes.iter().enumerate().step_by(debug_stride(3)) {
        let graph = random_graph(n, &mut rng);
        for neighborhood in [Neighborhood::Swap, Neighborhood::Flip] {
            let mut p = vec![false; n];
            p[..n / 2].fill(true);
            p.shuffle(&mut rng);
            let state = PartitionState::new(&graph, p).unwrap();
            let m = distance_one_count(&state, neighborhood);
            for k in [1, m / 3, m, m + 1, m + 5, 0] {
                compare(
                    &graph,
                    &state,
                    0.05,
                    neighborhood,
                    &SmoothingSpec::RandomKAverage { k },
                    Some(rng.r#gen()),
                    &CancellationToken::new(),
                    i as u64,
                );
            }
            compare(
                &graph,
                &state,
                0.05,
                neighborhood,
                &SmoothingSpec::AllAverage,
                None,
                &CancellationToken::new(),
                0,
            );
        }
        let (len, intact, free) = crate::smoothing::scratch_status();
        assert!(intact && free, "after n {n}: len {len}");
    }
}

/// Many threads at once (rayon), each with its own scratch.
#[test]
fn review_parallel_threads() {
    use rayon::prelude::*;
    let cases: Vec<u64> = (0..budget(400) as u64).collect();
    cases.par_iter().for_each(|&case| {
        let mut rng = Mt19937GenRand64::new(case);
        let n = rng.gen_range(2..=150);
        let graph = random_graph(n, &mut rng);
        for _ in 0..6 {
            let state = PartitionState::new(&graph, random_partition(n, &mut rng)).unwrap();
            for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
                let m = distance_one_count(&state, neighborhood);
                let spec = random_spec(n, neighborhood, m, 2000, &mut rng);
                compare(
                    &graph,
                    &state,
                    0.05,
                    neighborhood,
                    &spec,
                    Some(rng.r#gen()),
                    &CancellationToken::new(),
                    0,
                );
            }
        }
    });
}

/// Rank selection (`k * 128 <= n`) with extreme and irregular side sizes,
/// sizes around multiples of 8 and 64, every allowed `k` of that path and
/// the first `k` of the listed path.
#[test]
fn review_rank_selection_path_with_irregular_sides() {
    let mut rng = Mt19937GenRand64::new(128);
    let mut cases = 0;
    for n in (128..=700)
        .step_by(37 * debug_stride(4))
        .chain([128, 255, 256, 257, 383, 384, 385, 511, 512, 513, 1024])
    {
        let graph = random_graph(n, &mut rng);
        for size_a in [
            1,
            2,
            7,
            8,
            9,
            63,
            64,
            65,
            n / 2,
            n - 65,
            n - 64,
            n - 8,
            n - 1,
        ] {
            let mut p = vec![false; n];
            p[..size_a].fill(true);
            match rng.gen_range(0..3) {
                0 => p.shuffle(&mut rng),
                1 => p.reverse(),
                _ => {}
            }
            let state = PartitionState::new(&graph, p).unwrap();
            for k in (1..=n / 128 + 1).chain([n / 128 + 2]) {
                for _ in 0..3 {
                    compare(
                        &graph,
                        &state,
                        0.05,
                        Neighborhood::Swap,
                        &SmoothingSpec::RandomKAverage { k },
                        Some(rng.r#gen()),
                        &CancellationToken::new(),
                        0,
                    );
                    cases += 1;
                }
            }
        }
    }
    eprintln!("rank selection cases: {cases}");
}

/// Production engine against the frozen engine (which calls the frozen
/// smoothing) on random graphs of 10 to 130 vertices, SA and HC with random
/// smoothing specifications, every step compared by `assert_engine_exact_on`.
#[test]
fn review_random_engines_match_frozen_engine() {
    let mut rng = Mt19937GenRand64::new(31337);
    for case in 0..budget(800).div_ceil(40) {
        let n = 2 * rng.gen_range(5..=65);
        let graph = random_graph(n, &mut rng);
        let neighborhood = if rng.gen_bool(0.5) {
            Neighborhood::Flip
        } else {
            Neighborhood::Swap
        };
        let m = match neighborhood {
            Neighborhood::Flip => n,
            Neighborhood::Swap => n / 2 * (n - n / 2),
        };
        let smoothing = loop {
            let spec = random_spec(n, neighborhood, m, 400, &mut rng);
            let valid = match spec {
                SmoothingSpec::RandomKAverage { k } => {
                    k >= 1 && (k as u128) <= crate::smoothing::max_random_k(n, neighborhood)
                }
                _ => true,
            };
            if valid {
                break spec;
            }
        };
        let alpha = [0.05, 0.0, 0.125, 1.0 / 3.0][rng.gen_range(0..4)];
        let seed = rng.r#gen();
        if rng.gen_bool(0.6) {
            let temperature = [0.0, 0.3, 1.0, 5.0][rng.gen_range(0..4)];
            let c = condition_for(
                &graph,
                neighborhood,
                SolverSpec::Sa {
                    temperature,
                    smoothing,
                },
                alpha,
            );
            assert_engine_exact_on(&graph, &c, seed, 60);
        } else {
            // HC evaluates every candidate; keep it short on large neighborhoods.
            let steps = if m * m > 4_000_000 { 2 } else { 6 };
            let c = condition_for(&graph, neighborhood, SolverSpec::Hc { smoothing }, alpha);
            assert_engine_exact_on(&graph, &c, seed, steps);
        }
        let (_, intact, free) = crate::smoothing::scratch_status();
        assert!(intact && free, "case {case}");
    }
}
