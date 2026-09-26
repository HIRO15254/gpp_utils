//! Differential tests of tracked descents.
//!
//! A descent tracks its partition once ([`BestImprovement::track`]) and then
//! alternates [`BestImprovement::scan_tracked`] and [`BestImprovement::apply`].
//! These tests compare every scan of whole descents with a full scan written
//! out below (every move in canonical order, scored with the
//! [`PartitionState`] methods, the rule inline), on graphs with isolated
//! vertices, stars, cliques, hubs above the level cap, the 64-bit word
//! boundaries and the experiment generators; they also run the HC engine and
//! the runner against the frozen 51577f9 engine and runner on larger graphs.
use super::*;
use crate::experiment::config::{
    BasinMode, Budget, Condition, GraphKind, GraphSpec, Measurement, Schedule, SmoothingSpec,
    SolverSpec,
};
use crate::fitness::FitnessRegistry;
use crate::smoothing;
use rand::seq::SliceRandom;

/// Iteration budget: `release` in optimized builds and about a tenth in debug
/// builds (where every tracked scan also rebuilds and compares the tracked
/// data); the environment variable `DESCENT_ITERS` overrides both.
fn budget(release: usize) -> usize {
    std::env::var("DESCENT_ITERS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(if cfg!(debug_assertions) {
            (release / 10).max(1)
        } else {
            release
        })
}

/// The full scan: every move in canonical order (Flip `0..n`; Swap group A
/// ascending, then group B ascending), one evaluation each, the score of the
/// [`PartitionState`] method, `Reject` failing after counting the first
/// non-finite score, and the rule of the module documentation.
#[allow(clippy::too_many_arguments)]
fn full_scan(
    graph: &Graph,
    state: &PartitionState,
    neighborhood: Neighborhood,
    alpha: f64,
    non_finite: NonFinite,
    start: f64,
    tie_rng: &mut Mt19937GenRand64,
    evaluations: &mut u64,
) -> std::result::Result<Option<(Move, f64)>, ()> {
    let partition = state.partition();
    let n = partition.len();
    let moves: Vec<Move> = match neighborhood {
        Neighborhood::Flip => (0..n).map(Move::Flip).collect(),
        Neighborhood::Swap => {
            let side_a: Vec<usize> = (0..n).filter(|&v| partition[v]).collect();
            let side_b: Vec<usize> = (0..n).filter(|&v| !partition[v]).collect();
            side_a
                .iter()
                .flat_map(|&a| side_b.iter().map(move |&b| Move::Swap(a, b)))
                .collect()
        }
    };
    let mut best = start;
    let mut choice = None;
    let mut ties = 0u64;
    for mv in moves {
        *evaluations += 1;
        let x = match mv {
            Move::Flip(v) => state.flip_score(graph, v, alpha),
            Move::Swap(a, b) => state.swap_score(graph, a, b, alpha),
        };
        if non_finite == NonFinite::Reject && !x.is_finite() {
            return Err(());
        }
        if x < best {
            best = x;
            choice = Some(mv);
            ties = 1;
        } else if choice.is_some() && x == best {
            ties += 1;
            if tie_rng.gen_range(0..ties) == 0 {
                choice = Some(mv);
            }
        }
    }
    Ok(choice.map(|mv| (mv, best)))
}

fn graph_from(n: usize, edges: impl IntoIterator<Item = [usize; 2]>) -> Graph {
    let edges: std::collections::BTreeSet<[usize; 2]> = edges
        .into_iter()
        .filter(|&[a, b]| a != b)
        .map(|[a, b]| [a.min(b), a.max(b)])
        .collect();
    Graph::from_edges(n, edges.into_iter().collect()).unwrap()
}

fn random_graph(n: usize, degree: f64, rng: &mut Mt19937GenRand64) -> Graph {
    let p = if n > 1 { degree / (n - 1) as f64 } else { 0.0 };
    let mut edges = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            if rng.r#gen::<f64>() < p {
                edges.push([a, b]);
            }
        }
    }
    graph_from(n, edges)
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

/// Named graphs: isolated vertices, stars, cliques, paths, a clique joined to
/// a star with an isolated tail, hubs of degree above [`LEVEL_CAP`], random
/// graphs on both sides of the 64-bit word boundaries and the experiment
/// generators. `large` adds graphs of up to 600 vertices.
fn graphs(rng: &mut Mt19937GenRand64, large: bool) -> Vec<(String, Graph)> {
    let mut out: Vec<(String, Graph)> = Vec::new();
    for n in [1, 2, 63, 64, 65, 129] {
        out.push((format!("isolated{n}"), graph_from(n, [])));
    }
    for n in [2, 3, 12, 33] {
        let edges = (0..n).flat_map(|a| (a + 1..n).map(move |b| [a, b]));
        out.push((format!("clique{n}"), graph_from(n, edges)));
    }
    for (n, center) in [(10, 0), (65, 64), (130, 70)] {
        let edges = (0..n).map(move |v| [center, v]);
        out.push((format!("star{n}@{center}"), graph_from(n, edges)));
    }
    out.push((
        "path127".into(),
        graph_from(127, (0..126).map(|v| [v, v + 1])),
    ));
    // A 12-clique, a star from vertex 12 over 13..40 and an isolated tail.
    let mixed = (0..12)
        .flat_map(|a| (a + 1..12).map(move |b| [a, b]))
        .chain((13..40).map(|v| [12, v]));
    out.push(("mixed48".into(), graph_from(48, mixed)));
    // Hubs of degree 100 and 80 over a sparse random graph (clamped levels at
    // the default cap).
    let mut hub = random_graph(150, 3.0, rng).edges().to_vec();
    hub.extend((1..101).map(|v| [0, v]));
    hub.extend((50..130).map(|v| [149, v]));
    out.push(("hubs150".into(), graph_from(150, hub)));
    for n in [63, 64, 65, 127, 128, 129] {
        for degree in [3.0, 12.0] {
            out.push((format!("random{n}d{degree}"), random_graph(n, degree, rng)));
        }
    }
    out.push((
        "geometric124d10".into(),
        generated(GraphKind::Geometric, 124, 10.0, 0),
    ));
    out.push((
        "random124d5".into(),
        generated(GraphKind::Random, 124, 5.0, 0),
    ));
    if large {
        out.push((
            "random250d20".into(),
            generated(GraphKind::Random, 250, 20.0, 0),
        ));
        out.push((
            "geometric250d5".into(),
            generated(GraphKind::Geometric, 250, 5.0, 0),
        ));
        out.push((
            "random500d10".into(),
            generated(GraphKind::Random, 500, 10.0, 0),
        ));
        out.push((
            "geometric500d20".into(),
            generated(GraphKind::Geometric, 500, 20.0, 0),
        ));
        out.push(("random600d6".into(), random_graph(600, 6.0, rng)));
    }
    out
}

/// All in one group, one vertex in group A, balanced and random partitions.
fn partitions(n: usize, rng: &mut Mt19937GenRand64) -> Vec<Vec<bool>> {
    let mut out = vec![vec![true; n], vec![false; n]];
    if n > 0 {
        let mut one = vec![false; n];
        one[rng.gen_range(0..n)] = true;
        out.push(one);
    }
    let mut balanced: Vec<bool> = (0..n).map(|v| v < n / 2).collect();
    balanced.shuffle(rng);
    out.push(balanced);
    for bias in [0.5, 0.2] {
        out.push((0..n).map(|_| rng.r#gen::<f64>() < bias).collect());
    }
    out
}

/// Alphas: ordinary ones, ones whose two Flip penalties differ by an integer
/// (so Flips of the two groups tie in f64: 0.125, 0.25, 0.5), a third, large
/// and negative ones, and ones with non-finite penalties.
const ALPHAS: [f64; 13] = [
    0.0,
    0.05,
    0.125,
    0.25,
    0.5,
    1.0 / 3.0,
    1.0,
    2.5,
    1e17,
    -0.05,
    1e306,
    f64::INFINITY,
    f64::NAN,
];

/// What the compared descents went through.
#[derive(Default, Debug)]
struct Tally {
    descents: u64,
    scans: u64,
    moves: u64,
    candidates: u64,
    scored: u64,
    draws: u64,
    errors: u64,
    clamped: u64,
    longest: u64,
}

/// One case: a descent from `partition` with its first scan starting at the
/// state score plus `offset`.
struct Case<'a> {
    name: &'a str,
    graph: &'a Graph,
    partition: Vec<bool>,
    offset: f64,
    level_cap: i64,
    seed: u64,
    max_steps: u64,
}

/// Run one tracked descent and the full-scan descent side by side and compare
/// every scan: outcome bits, evaluation counts, the tie RNG state and errors;
/// after every move the partitions, the scores (also against the edge-list
/// score) and the tracked data against data rebuilt from the state.
fn compare_descent(scan: &mut BestImprovement, case: &Case<'_>, tally: &mut Tally) {
    let graph = case.graph;
    let (neighborhood, alpha, non_finite) = (scan.neighborhood, scan.alpha, scan.non_finite);
    let mut state = PartitionState::new(graph, case.partition.clone()).unwrap();
    let mut reference = state.clone();
    scan.level_cap = case.level_cap;
    scan.track(graph, &state);
    tally.clamped += u64::from(scan.tracker.cap < scan.tracker.dmax);
    let mut tie_rng = Mt19937GenRand64::new(case.seed);
    // Start at varied offsets of the 312-word MT19937-64 blocks.
    for _ in 0..case.seed % 400 {
        let _: u64 = tie_rng.r#gen();
    }
    let mut reference_rng = tie_rng.clone();
    let (mut evaluations, mut reference_evaluations) = (1, 1);
    let mut current = state.score(alpha) + case.offset;
    let cancel = CancellationToken::new();
    tally.descents += 1;
    for step in 0..case.max_steps {
        let context = format!(
            "{} {neighborhood:?} {non_finite:?} alpha={alpha:e} offset={} cap={} seed={} step={step}",
            case.name, case.offset, case.level_cap, case.seed,
        );
        let before = tie_rng.clone();
        let fast = scan.scan_tracked(
            graph,
            &state,
            current,
            &mut tie_rng,
            &cancel,
            &mut evaluations,
        );
        let slow = full_scan(
            graph,
            &reference,
            neighborhood,
            alpha,
            non_finite,
            current,
            &mut reference_rng,
            &mut reference_evaluations,
        );
        assert_eq!(evaluations, reference_evaluations, "evaluations {context}");
        assert!(tie_rng == reference_rng, "tie RNG {context}");
        tally.scans += 1;
        tally.scored += scan.scored;
        tally.draws += u64::from(tie_rng != before);
        let chosen = match (fast, slow) {
            (Ok(fast), Ok(slow)) => {
                assert_eq!(
                    fast.map(|(mv, x)| (mv, x.to_bits())),
                    slow.map(|(mv, x)| (mv, x.to_bits())),
                    "outcome {context}"
                );
                fast
            }
            (Err(error), Err(())) => {
                assert_eq!(
                    error.to_string(),
                    "non-finite search evaluation",
                    "{context}"
                );
                tally.errors += 1;
                None
            }
            (fast, slow) => panic!("{context}: {fast:?} vs {slow:?}"),
        };
        let Some((mv, best)) = chosen else {
            tally.longest = tally.longest.max(step);
            break;
        };
        scan.apply(graph, &mut state, mv);
        smoothing::apply(&mut reference, graph, mv);
        tally.moves += 1;
        assert_eq!(state.partition(), reference.partition(), "{context}");
        assert_eq!(state.score(alpha).to_bits(), best.to_bits(), "{context}");
        assert_eq!(
            best.to_bits(),
            graph.score(state.partition(), alpha).to_bits(),
            "{context}"
        );
        let mut rebuilt = Tracker::default();
        rebuilt.track(graph, &state, case.level_cap, scan.tracker.counts);
        assert!(scan.tracker == rebuilt, "tracked data {context}");
        current = best;
    }
    tally.candidates += reference_evaluations - 1;
}

#[test]
fn tracked_descents_match_the_full_scan() {
    let mut rng = Mt19937GenRand64::new(20260926);
    let graphs = graphs(&mut rng, !cfg!(debug_assertions));
    let mut tallies = [Tally::default(), Tally::default()];
    for (index, neighborhood) in [Neighborhood::Flip, Neighborhood::Swap]
        .into_iter()
        .enumerate()
    {
        // The share of the (mode, alpha, graph, partition) cases to run: the
        // full Swap scan is quadratic, and debug builds are slower.
        let share = match (neighborhood, cfg!(debug_assertions)) {
            (Neighborhood::Flip, false) => 1.0,
            (Neighborhood::Swap, false) => 0.5,
            (Neighborhood::Flip, true) => 0.25,
            (Neighborhood::Swap, true) => 0.1,
        };
        for non_finite in [NonFinite::Compare, NonFinite::Reject] {
            for &alpha in &ALPHAS {
                // One value reused across every graph, partition and cap.
                let mut scan = BestImprovement::new(neighborhood, alpha, non_finite);
                for (name, graph) in &graphs {
                    let n = graph.node_count();
                    let share = if neighborhood == Neighborhood::Swap && n > 200 {
                        share / 8.0
                    } else {
                        share
                    };
                    for partition in partitions(n, &mut rng) {
                        if rng.r#gen::<f64>() >= share {
                            continue;
                        }
                        let offset =
                            [0.0, 0.0, 0.0, 1.0, -0.5, 0.5, f64::INFINITY][rng.gen_range(0..7)];
                        let level_cap = [LEVEL_CAP, LEVEL_CAP, 0, 1, 2, 5][rng.gen_range(0..6)];
                        let case = Case {
                            name,
                            graph,
                            partition,
                            offset,
                            level_cap,
                            seed: rng.r#gen(),
                            max_steps: 1000,
                        };
                        compare_descent(&mut scan, &case, &mut tallies[index]);
                    }
                }
            }
        }
    }
    let moves = if cfg!(debug_assertions) { 500 } else { 5000 };
    for tally in &tallies {
        eprintln!("{tally:?}");
        assert!(tally.descents > 100 && tally.moves > moves, "{tally:?}");
        assert!(tally.draws * 4 > tally.scans, "{tally:?}");
        assert!(tally.errors > 0 && tally.clamped > 0, "{tally:?}");
        assert!(tally.longest >= 20, "{tally:?}");
    }
}

/// Whole descents on graphs of the experiment generators from random
/// partitions with the baseline alpha and the default level cap: the path of
/// the baseline runs.
#[test]
fn tracked_descents_on_experiment_graphs_match_the_full_scan() {
    let mut rng = Mt19937GenRand64::new(7);
    let mut specs = vec![
        (GraphKind::Random, 124, 5.0),
        (GraphKind::Geometric, 124, 20.0),
    ];
    if !cfg!(debug_assertions) {
        specs.extend([
            (GraphKind::Random, 250, 10.0),
            (GraphKind::Geometric, 250, 20.0),
            (GraphKind::Random, 500, 5.0),
            (GraphKind::Geometric, 500, 10.0),
            (GraphKind::Random, 500, 20.0),
        ]);
    }
    let mut tallies = [Tally::default(), Tally::default()];
    for (kind, n, degree) in specs {
        let graph = generated(kind, n, degree, 0);
        for (index, neighborhood) in [Neighborhood::Flip, Neighborhood::Swap]
            .into_iter()
            .enumerate()
        {
            for non_finite in [NonFinite::Compare, NonFinite::Reject] {
                let mut scan = BestImprovement::new(neighborhood, 0.05, non_finite);
                for _ in 0..budget(3) {
                    let mut partition: Vec<bool> = (0..n).map(|v| v < n / 2).collect();
                    partition.shuffle(&mut rng);
                    if neighborhood == Neighborhood::Flip {
                        partition = (0..n).map(|_| rng.r#gen()).collect();
                    }
                    let case = Case {
                        name: "experiment",
                        graph: &graph,
                        partition,
                        offset: 0.0,
                        level_cap: LEVEL_CAP,
                        seed: rng.r#gen(),
                        max_steps: 10_000,
                    };
                    compare_descent(&mut scan, &case, &mut tallies[index]);
                }
            }
        }
    }
    for tally in &tallies {
        eprintln!("{tally:?}");
        assert!(
            tally.longest >= 20 && tally.draws * 2 > tally.scans,
            "{tally:?}"
        );
        // The bitsets skip most candidates.
        assert!(tally.scored * 10 < tally.candidates, "{tally:?}");
    }
}

/// [`RowChecks`] makes every check of the full Swap scan, which checks before
/// row `r` when the rows since its last check reach [`CHECK_INTERVAL`]
/// candidates, before the first visited row at or after `r` or, after the
/// last visited row, before counting; and it checks only then.
#[test]
fn row_checks_cover_every_check_of_the_full_scan() {
    let mut rng = Mt19937GenRand64::new(3);
    let mut covered_late = 0u64;
    for _ in 0..budget(3000) {
        let row_len = [1, 2, 3, 100, 250, 341, 342, 1023, 1024, 1025, 3000][rng.gen_range(0..11)];
        let rows = rng.gen_range(1..700);
        let density = [0.0, 0.01, 0.2, 1.0][rng.gen_range(0..4)];
        let visited: Vec<usize> = (0..rows).filter(|_| rng.r#gen::<f64>() < density).collect();
        // The full scan.
        let mut full_checks = Vec::new();
        let mut unchecked = 0;
        for row in 0..rows {
            unchecked += row_len;
            if unchecked >= CHECK_INTERVAL {
                unchecked = 0;
                full_checks.push(row);
            }
        }
        // Checks of a scan that visits `visited`; `rows` stands for the end.
        let mut checks = RowChecks::new(row_len);
        let mut made = Vec::new();
        for &row in &visited {
            if checks.due(row) {
                made.push(row);
            }
        }
        if checks.due_at_end(rows) {
            made.push(rows);
        }
        // Each full-scan check at `r` is made at the first visited row `>= r`
        // or at the end, and each made check covers at least one.
        for &r in &full_checks {
            let at = visited.iter().copied().find(|&v| v >= r).unwrap_or(rows);
            assert!(
                made.contains(&at),
                "{row_len} {rows} {r} {visited:?} {made:?}"
            );
            covered_late += u64::from(at != r);
        }
        let mut previous = None;
        for &m in &made {
            assert!(
                full_checks
                    .iter()
                    .any(|&r| r <= m && previous.is_none_or(|p| r > p)),
                "{row_len} {rows} {m} {full_checks:?}"
            );
            previous = Some(m);
        }
    }
    assert!(covered_late > 0);
}

/// A tracked scan cancelled before it starts fails before counting, drawing or
/// changing the tracked data. Cancelling from another thread during descents
/// on graphs with more than [`CHECK_INTERVAL`] vertices leaves every scan
/// either failed without counting or complete and equal to the same scan with
/// a live token, and the tracked data unchanged, so the descent continues
/// exactly.
#[test]
fn cancelled_tracked_scans_fail_before_counting_and_keep_the_tracker() {
    let mut rng = Mt19937GenRand64::new(11);
    // A circulant graph with random chords: degree about 8.
    let n = if cfg!(debug_assertions) { 3000 } else { 30_000 };
    let edges = (0..n)
        .flat_map(|v| [1, 2, 7].map(|k| [v, (v + k) % n]))
        .chain((0..n).map(|_| [rng.gen_range(0..n), rng.gen_range(0..n)]))
        .collect::<Vec<_>>();
    let graph = graph_from(n, edges);
    let (mut cancelled, mut completed) = (0, 0);
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        let mut partition: Vec<bool> = (0..n).map(|v| v < n / 2).collect();
        partition.shuffle(&mut rng);
        let mut state = PartitionState::new(&graph, partition).unwrap();
        let mut scan = BestImprovement::new(neighborhood, 0.05, NonFinite::Reject);
        scan.track(&graph, &state);
        let dead = CancellationToken::new();
        dead.cancel();
        let mut tie_rng = Mt19937GenRand64::new(5);
        let untouched = tie_rng.clone();
        let mut evaluations = 9;
        let tracked = scan.tracker.clone();
        let result = scan.scan_tracked(
            &graph,
            &state,
            f64::INFINITY,
            &mut tie_rng,
            &dead,
            &mut evaluations,
        );
        assert_eq!(result.unwrap_err().to_string(), "operation cancelled");
        assert_eq!(evaluations, 9);
        assert!(tie_rng == untouched);
        assert!(scan.tracker == tracked);
        let live = CancellationToken::new();
        let mut current = state.score(0.05);
        for round in 0..budget(60) {
            let racing = CancellationToken::new();
            let trigger = racing.clone();
            // Released together with the scan, the other thread cancels after
            // a busy wait of 0 to 60 microseconds.
            let delay = std::time::Duration::from_micros(round as u64 % 13 * 5);
            let start = std::sync::Arc::new(std::sync::Barrier::new(2));
            let go = start.clone();
            let canceller = std::thread::spawn(move || {
                go.wait();
                let begin = std::time::Instant::now();
                while begin.elapsed() < delay {
                    std::hint::spin_loop();
                }
                trigger.cancel();
            });
            start.wait();
            let before = evaluations;
            let (mut raced_rng, mut raced_evaluations) = (tie_rng.clone(), evaluations);
            let raced = scan.scan_tracked(
                &graph,
                &state,
                current,
                &mut raced_rng,
                &racing,
                &mut raced_evaluations,
            );
            canceller.join().unwrap();
            let tracked = scan.tracker.clone();
            let expected = scan
                .scan_tracked(
                    &graph,
                    &state,
                    current,
                    &mut tie_rng,
                    &live,
                    &mut evaluations,
                )
                .unwrap();
            assert!(scan.tracker == tracked);
            match raced {
                Ok(found) => {
                    assert_eq!(
                        found.map(|(mv, x)| (mv, x.to_bits())),
                        expected.map(|(mv, x)| (mv, x.to_bits())),
                        "round {round}"
                    );
                    assert_eq!(raced_evaluations, evaluations);
                    assert!(raced_rng == tie_rng);
                    completed += 1;
                }
                Err(error) => {
                    assert_eq!(error.to_string(), "operation cancelled");
                    assert_eq!(raced_evaluations, before, "round {round}");
                    cancelled += 1;
                }
            }
            let Some((mv, best)) = expected else { break };
            scan.apply(&graph, &mut state, mv);
            current = best;
        }
        let mut rebuilt = Tracker::default();
        rebuilt.track(&graph, &state, LEVEL_CAP, scan.tracker.counts);
        assert!(scan.tracker == rebuilt);
    }
    eprintln!("{completed} completed, {cancelled} cancelled");
    assert!(completed + cancelled > 0);
}

// Frozen copies of the 51577f9 engine and runner and of the e4b6a1c smoothing
// module they call, as in `crate::solvers::engine::exact_tests`.
#[allow(clippy::too_many_arguments, dead_code)]
mod smoothing_e4b6a1c {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/solvers/test_reference/smoothing_e4b6a1c.rs"
    ));
}

#[allow(dead_code)]
mod reference {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/solvers/test_reference/engine_51577f9.rs"
    ));

    #[allow(clippy::too_many_arguments, clippy::collapsible_if)]
    pub mod runner {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/solvers/test_reference/runner_51577f9.rs"
        ));
    }
}

fn condition(
    graph: &Graph,
    neighborhood: Neighborhood,
    solver: SolverSpec,
    alpha: f64,
) -> Condition {
    Condition {
        graph: GraphSpec {
            kind: GraphKind::Random,
            node_count: graph.node_count(),
            expected_degree: 2.0 * graph.edges().len() as f64 / graph.node_count() as f64,
            seed: 0,
        },
        neighborhood,
        alpha,
        solver,
        budget: Budget { max_steps: 100_000 },
        measurement: Measurement {
            schedule: Schedule::Logarithmic,
            steps: vec![],
            basin: BasinMode::Real,
            max_basin_steps: 10_000,
            diagnostics: true,
            best_basin: true,
        },
    }
}

/// The non-smoothed HC engine against the frozen engine, step by step until
/// a local optimum or the non-finite error, on graphs larger than those of
/// `exact_tests`.
#[test]
fn hc_engine_matches_the_frozen_engine_on_larger_graphs() {
    let registry = FitnessRegistry::default_registry();
    let cancel = CancellationToken::new();
    let mut cases = vec![
        (
            generated(GraphKind::Random, 124, 10.0, 0),
            Neighborhood::Flip,
        ),
        (
            generated(GraphKind::Geometric, 128, 5.0, 1),
            Neighborhood::Swap,
        ),
    ];
    if !cfg!(debug_assertions) {
        cases.extend([
            (
                generated(GraphKind::Geometric, 250, 20.0, 0),
                Neighborhood::Flip,
            ),
            (
                generated(GraphKind::Random, 500, 5.0, 0),
                Neighborhood::Flip,
            ),
            (
                generated(GraphKind::Random, 124, 20.0, 2),
                Neighborhood::Swap,
            ),
        ]);
    }
    let (mut optima, mut errors) = (0, 0);
    for (graph, neighborhood) in &cases {
        for smoothing in [SmoothingSpec::None, SmoothingSpec::WeightedAverage { k: 0 }] {
            for alpha in [0.05, 0.125, 0.0, 1e306] {
                for seed in 0..budget(2) as u64 {
                    let solver = SolverSpec::Hc {
                        smoothing: smoothing.clone(),
                    };
                    let c = condition(graph, *neighborhood, solver, alpha);
                    let mut actual =
                        crate::solvers::Engine::new(graph, &c, seed, &registry, &cancel).unwrap();
                    let mut expected =
                        reference::Engine::new(graph, &c, seed, &registry, &cancel).unwrap();
                    for step in 0.. {
                        let context =
                            format!("{neighborhood:?} {smoothing:?} {alpha:e} {seed} {step}");
                        let a = actual.step(&cancel);
                        let b = expected.step(&cancel);
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
                        match (a, b) {
                            (Ok(x), Ok(y)) => {
                                assert_eq!(format!("{x:?}"), format!("{y:?}"), "{context}");
                                if format!("{x:?}") != "Continue" {
                                    optima += 1;
                                    break;
                                }
                            }
                            (Err(x), Err(y)) => {
                                assert_eq!(x.to_string(), y.to_string(), "{context}");
                                errors += 1;
                                break;
                            }
                            (x, y) => panic!("{context}: {x:?} vs {y:?}"),
                        }
                    }
                }
            }
        }
    }
    assert!(optima > 0 && errors > 0);
}

/// The result of `run_one` with timings and attempt id cleared and every f64
/// replaced by its bits.
fn exact_json(mut result: crate::experiment::result::RunResult) -> serde_json::Value {
    result.attempt_id.clear();
    result.elapsed_ms = 0.0;
    if let Some(diagnostics) = &mut result.diagnostics {
        diagnostics.search_ms = 0.0;
        diagnostics.measurement_ms = 0.0;
    }
    let mut value = serde_json::to_value(result).unwrap();
    fn encode_numbers(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Array(values) => values.iter_mut().for_each(encode_numbers),
            serde_json::Value::Object(values) => values.values_mut().for_each(encode_numbers),
            serde_json::Value::Number(number) if number.is_f64() => {
                *value = serde_json::Value::String(format!(
                    "f64:{:016x}",
                    number.as_f64().unwrap().to_bits()
                ));
            }
            _ => {}
        }
    }
    encode_numbers(&mut value);
    value
}

/// The basin of the incumbent `best` as the runner measures it, recomputed
/// with [`full_scan`] from the documented tie-stream derivation.
fn naive_best_basin(
    graph: &Graph,
    c: &Condition,
    seed: u64,
    best: &[bool],
    evaluations: &mut u64,
) -> crate::experiment::result::BasinResult {
    use crate::experiment::result::{BasinResult, BasinTermination};
    let neighborhood = match c.neighborhood {
        Neighborhood::Flip => b"flip".as_slice(),
        Neighborhood::Swap => b"swap".as_slice(),
    };
    let partition: Vec<u8> = best.iter().map(|&x| u8::from(x)).collect();
    let mut tie_rng = crate::optimization::rng_for(&[
        graph.content_hash().as_bytes(),
        neighborhood,
        &c.alpha.to_bits().to_le_bytes(),
        &seed.to_le_bytes(),
        &partition,
        b"basin-best-real-v1",
    ]);
    let mut state = PartitionState::new(graph, best.to_vec()).unwrap();
    *evaluations += 1;
    let mut current = state.score(c.alpha);
    let mut steps = 0;
    let termination = loop {
        if steps >= c.measurement.max_basin_steps {
            break BasinTermination::StepLimit;
        }
        steps += 1;
        let found = full_scan(
            graph,
            &state,
            c.neighborhood,
            c.alpha,
            NonFinite::Compare,
            current,
            &mut tie_rng,
            evaluations,
        )
        .unwrap();
        match found {
            Some((mv, best)) => {
                smoothing::apply(&mut state, graph, mv);
                current = best;
            }
            None => break BasinTermination::LocalOptimum,
        }
    };
    BasinResult {
        real: state.score(c.alpha),
        smoothed: None,
        termination,
        steps: c.measurement.diagnostics.then_some(steps),
    }
}

/// Whole runs with real basins at the logarithmic checkpoints (HC to its
/// optimum, SA with the baseline alpha) against the frozen runner, on graphs
/// larger than those of `exact_tests`; with `best_basin` the basins of the
/// incumbents against [`naive_best_basin`], with the other output unchanged
/// except for the measurement evaluations those basins add.
#[test]
fn runner_basins_match_the_frozen_runner_on_larger_graphs() {
    let registry = FitnessRegistry::default_registry();
    let cancel = CancellationToken::new();
    let mut cases = vec![
        (
            generated(GraphKind::Random, 64, 10.0, 0),
            Neighborhood::Swap,
            300,
        ),
        (
            generated(GraphKind::Geometric, 124, 10.0, 0),
            Neighborhood::Flip,
            2000,
        ),
    ];
    if !cfg!(debug_assertions) {
        cases.extend([
            (
                generated(GraphKind::Random, 250, 5.0, 0),
                Neighborhood::Flip,
                20_000,
            ),
            (
                generated(GraphKind::Random, 124, 10.0, 1),
                Neighborhood::Swap,
                2000,
            ),
            (
                generated(GraphKind::Geometric, 128, 20.0, 0),
                Neighborhood::Swap,
                1000,
            ),
        ]);
    }
    let mut best_basins = 0;
    for (graph, neighborhood, sa_steps) in &cases {
        for (solver, max_steps) in [
            (
                SolverSpec::Hc {
                    smoothing: SmoothingSpec::None,
                },
                100_000,
            ),
            (
                SolverSpec::Sa {
                    temperature: 1.0,
                    smoothing: SmoothingSpec::None,
                },
                *sa_steps,
            ),
            (
                SolverSpec::Sa {
                    temperature: 316.22776601683796,
                    smoothing: SmoothingSpec::WeightedAverage { k: 0 },
                },
                *sa_steps,
            ),
        ] {
            let mut c = condition(graph, *neighborhood, solver, 0.05);
            c.budget.max_steps = max_steps;
            for seed in 0..budget(2) as u64 {
                let context = format!("{neighborhood:?} {:?} seed {seed}", c.solver);
                // The frozen runner predates `best_basin`.
                c.measurement.best_basin = false;
                let plain = crate::experiment::runner::run_one(graph, &c, seed, &cancel, &registry)
                    .unwrap();
                let expected =
                    reference::runner::run_one(graph, &c, seed, &cancel, &registry).unwrap();
                assert!(plain.records.iter().all(|r| r.basin_real.is_some()));
                let plain = exact_json(plain);
                assert_eq!(plain, exact_json(expected), "{context}");
                c.measurement.best_basin = true;
                let mut with_best =
                    crate::experiment::runner::run_one(graph, &c, seed, &cancel, &registry)
                        .unwrap();
                let mut extra = 0;
                let mut cached: Option<Vec<bool>> = None;
                for record in &mut with_best.records {
                    let best = with_best.partitions[record.best_solution.0].clone();
                    let mut evaluations = 0;
                    let naive = naive_best_basin(graph, &c, seed, &best, &mut evaluations);
                    if cached.as_ref() != Some(&best) {
                        extra += evaluations;
                        cached = Some(best);
                    }
                    let actual = record.basin_best.take().expect("best basin");
                    assert_eq!(actual.real.to_bits(), naive.real.to_bits(), "{context}");
                    assert_eq!(actual.smoothed, None, "{context}");
                    assert_eq!(actual.termination, naive.termination, "{context}");
                    assert_eq!(actual.steps, naive.steps, "{context}");
                    best_basins += 1;
                }
                let diagnostics = with_best.diagnostics.as_mut().unwrap();
                diagnostics.objective_evaluations_measurement -= extra;
                assert_eq!(exact_json(with_best), plain, "{context}");
            }
        }
    }
    assert!(best_basins > 20);
}
