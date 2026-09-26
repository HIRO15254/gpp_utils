//! Independent review oracle for tracked descents.
//!
//! Every scan of a tracked descent (`track`, `scan_tracked`, `apply`) is
//! compared with a full scan written from the module documentation alone: the
//! cut data are recomputed from the edge list for every scan (never read from
//! `PartitionState` or the tracker), candidates are visited in canonical order
//! and, on small graphs, every candidate score is `Graph::score` of a copied
//! partition. After every move the tracked data are checked against their
//! documented definition (gains, group bits, cumulative clamped levels and B
//! gain counts), not against `Tracker::track`.
use super::*;
use crate::experiment::config::{GraphKind, GraphSpec};
use rand::seq::SliceRandom;

fn budget(release: usize, debug: usize) -> usize {
    std::env::var("ORACLE_ITERS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(if cfg!(debug_assertions) {
            debug
        } else {
            release
        })
}

/// Cut data of `p` from the edge list.
struct Fresh {
    cut: i64,
    cuts_at: Vec<i64>,
    size_a: usize,
}

fn fresh(graph: &Graph, p: &[bool]) -> Fresh {
    let mut cuts_at = vec![0i64; p.len()];
    let mut cut = 0i64;
    for &[a, b] in graph.edges() {
        if p[a] != p[b] {
            cut += 1;
            cuts_at[a] += 1;
            cuts_at[b] += 1;
        }
    }
    Fresh {
        cut,
        cuts_at,
        size_a: p.iter().filter(|&&x| x).count(),
    }
}

fn penalty(alpha: f64, n: usize, a: usize) -> f64 {
    let d = a as i64 - (n - a) as i64;
    alpha * d as f64 * d as f64
}

/// The documented full scan. `exhaustive` scores each candidate with
/// `Graph::score` of the moved partition; otherwise with the integer delta of
/// the edge-list cut data (the same value, cheaper).
#[allow(clippy::too_many_arguments)]
fn oracle(
    graph: &Graph,
    p: &[bool],
    neighborhood: Neighborhood,
    alpha: f64,
    non_finite: NonFinite,
    start: f64,
    tie_rng: &mut Mt19937GenRand64,
    evaluations: &mut u64,
    exhaustive: bool,
) -> std::result::Result<Option<(Move, f64)>, ()> {
    let n = p.len();
    let f = fresh(graph, p);
    let gain = |v: usize| graph.neighbors(v).len() as i64 - 2 * f.cuts_at[v];
    let mut moved = p.to_vec();
    let mut score = |mv: Move| -> f64 {
        if exhaustive {
            match mv {
                Move::Flip(v) => {
                    moved[v] = !moved[v];
                    let x = graph.score(&moved, alpha);
                    moved[v] = !moved[v];
                    x
                }
                Move::Swap(a, b) => {
                    moved.swap(a, b);
                    let x = graph.score(&moved, alpha);
                    moved.swap(a, b);
                    x
                }
            }
        } else {
            match mv {
                Move::Flip(v) => {
                    let a = if p[v] { f.size_a - 1 } else { f.size_a + 1 };
                    (f.cut + gain(v)) as f64 + penalty(alpha, n, a)
                }
                Move::Swap(a, b) => {
                    let adjacent = graph.neighbors(a).binary_search(&b).is_ok() as i64;
                    (f.cut + gain(a) + gain(b) + 2 * adjacent) as f64 + penalty(alpha, n, f.size_a)
                }
            }
        }
    };
    let moves: Vec<Move> = match neighborhood {
        Neighborhood::Flip => (0..n).map(Move::Flip).collect(),
        Neighborhood::Swap => {
            let mut out = Vec::new();
            for a in (0..n).filter(|&v| p[v]) {
                for b in (0..n).filter(|&v| !p[v]) {
                    out.push(Move::Swap(a, b));
                }
            }
            out
        }
    };
    let (mut best, mut choice, mut ties) = (start, None, 0u64);
    for mv in moves {
        *evaluations += 1;
        let x = score(mv);
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

/// The tracked data against their documented definition for partition `p`.
fn assert_definition(t: &Tracker, graph: &Graph, p: &[bool], level_cap: i64, context: &str) {
    let n = p.len();
    let f = fresh(graph, p);
    let dmax = (0..n).map(|v| graph.degree(v) as i64).max().unwrap_or(0);
    let cap = dmax.min(level_cap);
    assert!(t.active, "{context}");
    assert_eq!(
        (t.n, t.words, t.dmax, t.cap),
        (n, n.div_ceil(64), dmax, cap),
        "{context}"
    );
    let words = n.div_ceil(64);
    let levels = (2 * cap + 3) as usize;
    assert_eq!(t.levels.len(), levels * words, "{context}");
    for (v, &side) in p.iter().enumerate() {
        let g = graph.degree(v) as i64 - 2 * f.cuts_at[v];
        assert_eq!(t.gain[v], g, "gain {v} {context}");
        assert_eq!(
            t.in_a[v / 64] >> (v % 64) & 1 == 1,
            side,
            "side {v} {context}"
        );
        let level = (g.clamp(-cap - 1, cap + 1) + cap + 1) as usize;
        for i in 0..levels {
            let bit = t.levels[i * words + v / 64] >> (v % 64) & 1 == 1;
            assert_eq!(bit, level <= i, "level {i} of {v} {context}");
        }
    }
    // No stray bits past `n`.
    if !n.is_multiple_of(64) {
        let high = !((1u64 << (n % 64)) - 1);
        assert_eq!(t.in_a[words - 1] & high, 0, "{context}");
        for i in 0..levels {
            assert_eq!(t.levels[i * words + words - 1] & high, 0, "{context}");
        }
    }
    if t.counts {
        let mut counts = vec![0u32; 2 * dmax as usize + 1];
        for v in (0..n).filter(|&v| !p[v]) {
            counts[(graph.degree(v) as i64 - 2 * f.cuts_at[v] + dmax) as usize] += 1;
        }
        assert_eq!(t.count_b, counts, "{context}");
    } else {
        assert!(t.count_b.is_empty(), "{context}");
    }
}

#[derive(Default, Debug)]
struct Stats {
    descents: u64,
    scans: u64,
    moves: u64,
    draws: u64,
    errors: u64,
    clamped_descents: u64,
    longest: u64,
}

#[allow(clippy::too_many_arguments)]
fn descent(
    graph: &Graph,
    name: &str,
    partition: &[bool],
    neighborhood: Neighborhood,
    alpha: f64,
    non_finite: NonFinite,
    level_cap: i64,
    start: f64,
    seed: u64,
    max_steps: u64,
    stats: &mut Stats,
) {
    let exhaustive = graph.node_count() <= 40;
    let mut state = PartitionState::new(graph, partition.to_vec()).unwrap();
    let mut p = partition.to_vec();
    let mut scan = BestImprovement::new(neighborhood, alpha, non_finite);
    scan.level_cap = level_cap;
    scan.track(graph, &state);
    stats.clamped_descents += u64::from(scan.tracker.cap < scan.tracker.dmax);
    let mut rng = Mt19937GenRand64::new(seed);
    for _ in 0..seed % 700 {
        let _: u64 = rng.r#gen();
    }
    let mut oracle_rng = rng.clone();
    let (mut evaluations, mut oracle_evaluations) = (seed % 5, seed % 5);
    let mut current = start;
    let cancel = CancellationToken::new();
    stats.descents += 1;
    for step in 0..max_steps {
        let context = format!(
            "{name} {neighborhood:?} {non_finite:?} alpha={alpha:e} start={start:e} cap={level_cap} seed={seed} step={step}"
        );
        let before = rng.clone();
        let fast = scan.scan_tracked(graph, &state, current, &mut rng, &cancel, &mut evaluations);
        let slow = oracle(
            graph,
            &p,
            neighborhood,
            alpha,
            non_finite,
            current,
            &mut oracle_rng,
            &mut oracle_evaluations,
            exhaustive,
        );
        assert_eq!(evaluations, oracle_evaluations, "evaluations {context}");
        assert!(rng == oracle_rng, "tie RNG {context}");
        stats.scans += 1;
        stats.draws += u64::from(rng != before);
        let found = match (fast, slow) {
            (Ok(a), Ok(b)) => {
                assert_eq!(
                    a.map(|(m, x)| (m, x.to_bits())),
                    b.map(|(m, x)| (m, x.to_bits())),
                    "{context}"
                );
                a
            }
            (Err(e), Err(())) => {
                assert_eq!(e.to_string(), "non-finite search evaluation", "{context}");
                stats.errors += 1;
                None
            }
            (a, b) => panic!("{context}: {a:?} vs {b:?}"),
        };
        let Some((mv, best)) = found else {
            stats.longest = stats.longest.max(step);
            break;
        };
        scan.apply(graph, &mut state, mv);
        match mv {
            Move::Flip(v) => p[v] = !p[v],
            Move::Swap(a, b) => {
                assert!(p[a] && !p[b], "{context}");
                p.swap(a, b)
            }
        }
        stats.moves += 1;
        assert_eq!(state.partition(), &p[..], "{context}");
        // The full definition check is O(n * levels): every step on small
        // graphs, every 16th step and at the end on larger ones.
        if graph.node_count() <= 130 || step % 16 == 0 {
            assert_definition(&scan.tracker, graph, &p, level_cap, &context);
        }
        current = best;
    }
    assert_definition(&scan.tracker, graph, &p, level_cap, name);
}

fn graph_from(n: usize, edges: impl IntoIterator<Item = [usize; 2]>) -> Graph {
    let set: std::collections::BTreeSet<[usize; 2]> = edges
        .into_iter()
        .filter(|&[a, b]| a != b)
        .map(|[a, b]| [a.min(b), a.max(b)])
        .collect();
    Graph::from_edges(n, set.into_iter().collect()).unwrap()
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

fn graphs(rng: &mut Mt19937GenRand64, large: bool) -> Vec<(String, Graph)> {
    let mut out: Vec<(String, Graph)> = vec![("empty0".into(), graph_from(0, []))];
    for n in [1, 2, 3, 64, 65] {
        out.push((format!("isolated{n}"), graph_from(n, [])));
    }
    for n in [2, 3, 5, 20] {
        out.push((
            format!("clique{n}"),
            graph_from(n, (0..n).flat_map(|a| (a + 1..n).map(move |b| [a, b]))),
        ));
    }
    // Complete bipartite: every gain is extreme.
    for m in [3, 17] {
        let edges = (0..m).flat_map(move |a| (m..2 * m).map(move |b| [a, b]));
        out.push((format!("k{m},{m}"), graph_from(2 * m, edges)));
    }
    // Cycles: every gain in {-2, 0, 2}, many ties.
    for n in [4, 63, 130] {
        out.push((
            format!("cycle{n}"),
            graph_from(n, (0..n).map(|v| [v, (v + 1) % n])),
        ));
    }
    // Two hubs above the default cap sharing leaves, word-straddling ids.
    let mut hubs: Vec<[usize; 2]> = (1..140).map(|v| [63, v]).collect();
    hubs.extend((60..200).map(|v| [128, v]));
    hubs.extend(random_graph(200, 2.0, rng).edges().iter().copied());
    out.push(("twohubs200".into(), graph_from(200, hubs)));
    // A star whose centre has degree 70 (just above the cap) plus a clique.
    let mut star = (0..71).map(|v| [70, v]).collect::<Vec<_>>();
    star.extend((71..80).flat_map(|a| (a + 1..80).map(move |b| [a, b])));
    out.push(("star71clique9".into(), graph_from(80, star)));
    for n in [63, 64, 65, 127, 128, 129, 191, 192, 193] {
        for degree in [1.5, 6.0, 30.0] {
            if degree < n as f64 {
                out.push((format!("random{n}d{degree}"), random_graph(n, degree, rng)));
            }
        }
    }
    for (kind, n, d) in [
        (GraphKind::Geometric, 124, 20.0),
        (GraphKind::Random, 124, 10.0),
    ] {
        let spec = GraphSpec {
            kind,
            node_count: n,
            expected_degree: d,
            seed: 3,
        };
        out.push((
            format!("{kind:?}{n}d{d}"),
            Graph::generate(&spec, &CancellationToken::new()).unwrap(),
        ));
    }
    if large {
        for (kind, n, d) in [
            (GraphKind::Geometric, 500, 20.0),
            (GraphKind::Random, 500, 5.0),
            (GraphKind::Geometric, 250, 10.0),
        ] {
            let spec = GraphSpec {
                kind,
                node_count: n,
                expected_degree: d,
                seed: 1,
            };
            out.push((
                format!("{kind:?}{n}d{d}"),
                Graph::generate(&spec, &CancellationToken::new()).unwrap(),
            ));
        }
    }
    out
}

fn partitions(n: usize, rng: &mut Mt19937GenRand64) -> Vec<Vec<bool>> {
    let mut out = vec![vec![true; n], vec![false; n]];
    if n > 0 {
        let mut one_a = vec![false; n];
        one_a[rng.gen_range(0..n)] = true;
        out.push(one_a);
        let mut one_b = vec![true; n];
        one_b[rng.gen_range(0..n)] = false;
        out.push(one_b);
    }
    let mut balanced: Vec<bool> = (0..n).map(|v| v < n / 2).collect();
    balanced.shuffle(rng);
    out.push(balanced);
    // Whole 64-bit words on one side.
    out.push((0..n).map(|v| (v / 64) % 2 == 0).collect());
    out.push((0..n).map(|_| rng.r#gen::<f64>() < 0.5).collect());
    out.push((0..n).map(|_| rng.r#gen::<f64>() < 0.1).collect());
    out
}

const ALPHAS: [f64; 17] = [
    0.0,
    -0.0,
    0.05,
    0.125,
    0.5,
    1.0 / 3.0,
    2.5,
    1e-300,
    // Penalties near 2^53: score spacing 1 or 2 makes rounding ties.
    2.0e13,
    1e17,
    -0.05,
    -1e17,
    1e306,
    f64::MAX,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
];

fn starts(state_score: f64, rng: &mut Mt19937GenRand64) -> f64 {
    match rng.gen_range(0..10) {
        0 => f64::INFINITY,
        1 => f64::NEG_INFINITY,
        2 => f64::NAN,
        3 => state_score + 1.0,
        4 => state_score - 0.5,
        5 => state_score.next_up(),
        6 => state_score.next_down(),
        _ => state_score,
    }
}

#[test]
fn oracle_review_tracked_descents() {
    let mut rng = Mt19937GenRand64::new(0x0dd_5eed);
    let large = !cfg!(debug_assertions);
    let graphs = graphs(&mut rng, large);
    let share = budget(1000, 60) as f64 / 1000.0;
    let mut stats = [Stats::default(), Stats::default()];
    for (index, neighborhood) in [Neighborhood::Flip, Neighborhood::Swap]
        .into_iter()
        .enumerate()
    {
        for (name, graph) in &graphs {
            let n = graph.node_count();
            for partition in partitions(n, &mut rng) {
                for &alpha in &ALPHAS {
                    for non_finite in [NonFinite::Compare, NonFinite::Reject] {
                        let mut share = share;
                        if neighborhood == Neighborhood::Swap {
                            share *= if n > 200 { 0.05 } else { 0.4 };
                        }
                        if rng.r#gen::<f64>() >= share {
                            continue;
                        }
                        let level_cap = [64, 64, 0, 1, 2, 3, 7, 1000][rng.gen_range(0..8)];
                        let state_score = graph.score(&partition, alpha);
                        let start = starts(state_score, &mut rng);
                        descent(
                            graph,
                            name,
                            &partition,
                            neighborhood,
                            alpha,
                            non_finite,
                            level_cap,
                            start,
                            rng.r#gen(),
                            400,
                            &mut stats[index],
                        );
                    }
                }
            }
        }
    }
    for s in &stats {
        eprintln!("{s:?}");
        assert!(s.descents > 50 && s.moves > 200 && s.errors > 0, "{s:?}");
        assert!(s.draws * 8 > s.scans && s.clamped_descents > 0, "{s:?}");
    }
}

/// `Bound::limit` against a linear search with large cuts (where `(cut + s)
/// as f64` has plateaus), penalties and bests next to representable scores.
#[test]
fn oracle_review_bound_limit_with_plateaus() {
    let mut rng = Mt19937GenRand64::new(99);
    for _ in 0..budget(20_000, 2_000) {
        let radius: i64 = [0, 1, 2, 3, 64, 500, 1500][rng.gen_range(0..7)];
        let cut: i64 = match rng.gen_range(0..4) {
            0 => rng.gen_range(0..100),
            1 => (1i64 << 53) - rng.gen_range(0..3000),
            2 => (1i64 << 54) + rng.gen_range(0..3000),
            _ => rng.gen_range(0..(1i64 << 60)),
        };
        let penalty = [
            0.0,
            -0.0,
            0.05,
            -7.5,
            1e17,
            -1e17,
            2.0f64.powi(53),
            1e300,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ][rng.gen_range(0..12)];
        let s0 = rng.gen_range(-radius - 2..=radius + 2);
        let base = (cut + s0) as f64 + penalty;
        let best = match rng.gen_range(0..6) {
            0 => base,
            1 => base.next_up(),
            2 => base.next_down(),
            3 => base + rng.r#gen::<f64>() * 10.0 - 5.0,
            4 => [
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                f64::MAX,
                f64::MIN,
            ][rng.gen_range(0..5)],
            _ => (cut as f64) * rng.r#gen::<f64>(),
        };
        for choice in [None, Some(Move::Flip(1))] {
            let rule = Rule {
                best,
                choice,
                ties: 1,
            };
            let bound = Bound {
                cut,
                penalty,
                radius,
            };
            let expected = (-radius..=radius)
                .rev()
                .find(|&s| rule.may_change((cut + s) as f64 + penalty))
                .unwrap_or(-radius - 1);
            assert_eq!(
                bound.limit(&rule),
                expected,
                "{cut} {penalty:e} {best:e} {radius} {choice:?}"
            );
        }
    }
}
