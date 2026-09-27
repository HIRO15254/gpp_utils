//! Independent-review tests of the exact best-improvement scan.
use super::*;
use crate::smoothing;
use rand::seq::SliceRandom;

/// Fully independent oracle: canonical moves enumerated here, every candidate
/// scored by recomputing `Graph::score` on a copied partition.
#[allow(clippy::too_many_arguments)]
fn oracle(
    graph: &Graph,
    partition: &[bool],
    neighborhood: Neighborhood,
    alpha: f64,
    non_finite: NonFinite,
    start: f64,
    tie_rng: &mut Mt19937GenRand64,
    evaluations: &mut u64,
) -> std::result::Result<Option<(Move, f64)>, ()> {
    let n = partition.len();
    let moves: Vec<Move> = match neighborhood {
        Neighborhood::Flip => (0..n).map(Move::Flip).collect(),
        Neighborhood::Swap => {
            let a: Vec<usize> = (0..n).filter(|&v| partition[v]).collect();
            let b: Vec<usize> = (0..n).filter(|&v| !partition[v]).collect();
            let mut out = Vec::new();
            for &x in &a {
                for &y in &b {
                    out.push(Move::Swap(x, y));
                }
            }
            out
        }
    };
    let state = PartitionState::new(graph, partition.to_vec()).unwrap();
    let mut best = start;
    let mut choice = None;
    let mut ties = 0u64;
    for mv in moves {
        *evaluations += 1;
        let mut p = partition.to_vec();
        match mv {
            Move::Flip(v) => p[v] = !p[v],
            Move::Swap(a, b) => {
                p[a] = !p[a];
                p[b] = !p[b];
            }
        }
        let x = graph.score(&p, alpha);
        // Sanity: the incremental score of the old code has the same bits.
        let old = smoothing::move_score(&state, graph, mv, alpha);
        assert_eq!(
            x.to_bits(),
            old.to_bits(),
            "oracle vs move_score {mv:?} alpha {alpha:e}"
        );
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

fn graph_from(n: usize, edges: Vec<[usize; 2]>) -> Graph {
    Graph::from_edges(n, edges).unwrap()
}

fn random_graph(n: usize, p: f64, rng: &mut Mt19937GenRand64) -> Graph {
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

fn families(rng: &mut Mt19937GenRand64) -> Vec<Graph> {
    let mut out = vec![
        graph_from(0, vec![]),
        graph_from(1, vec![]),
        graph_from(2, vec![]),
        graph_from(2, vec![[0, 1]]),
        graph_from(3, vec![[0, 1], [1, 2]]),
    ];
    for n in [4, 6, 9, 12, 17, 24] {
        out.push(graph_from(n, vec![]));
        out.push(graph_from(
            n,
            (0..n)
                .flat_map(|a| (a + 1..n).map(move |b| [a, b]))
                .collect(),
        ));
        out.push(graph_from(n, (1..n).map(|b| [0, b]).collect()));
        out.push(graph_from(
            n,
            (0..n)
                .map(|a| [a, (a + 1) % n])
                .filter(|e| e[0] != e[1])
                .map(|[a, b]| if a < b { [a, b] } else { [b, a] })
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
        ));
        let m = n / 2;
        out.push(graph_from(
            n,
            (0..m)
                .flat_map(|a| (m..2 * m).map(move |b| [a, b]))
                .collect(),
        ));
    }
    // Torus 5x5 and 4x4 (4-regular: many equal gains).
    for w in [4usize, 5] {
        let mut edges = std::collections::BTreeSet::new();
        for r in 0..w {
            for c in 0..w {
                let v = r * w + c;
                for u in [r * w + (c + 1) % w, ((r + 1) % w) * w + c] {
                    edges.insert(if v < u { [v, u] } else { [u, v] });
                }
            }
        }
        out.push(graph_from(w * w, edges.into_iter().collect()));
    }
    for (n, p) in [
        (5, 0.5),
        (8, 0.3),
        (10, 0.9),
        (13, 0.2),
        (20, 0.1),
        (26, 0.4),
        (33, 0.15),
        (48, 0.08),
        (64, 0.05),
    ] {
        out.push(random_graph(n, p, rng));
    }
    out
}

fn partitions(n: usize, rng: &mut Mt19937GenRand64) -> Vec<Vec<bool>> {
    let mut out = vec![vec![true; n], vec![false; n]];
    if n > 0 {
        for v in [0, n / 2, n - 1] {
            let mut one = vec![false; n];
            one[v] = true;
            out.push(one.clone());
            out.push(one.iter().map(|x| !x).collect());
        }
    }
    for bias in [0.1, 0.5, 0.9] {
        out.push((0..n).map(|_| rng.r#gen::<f64>() < bias).collect());
    }
    let mut balanced: Vec<bool> = (0..n).map(|v| v < n / 2).collect();
    balanced.shuffle(rng);
    out.push(balanced);
    out
}

/// Alphas the adversarial test runs (bit patterns): signed zeros, a subnormal,
/// ordinary values, 2^52, huge, negative and non-finite ones.
const SAMPLED_ALPHAS: [u64; 12] = [
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x3FA9_9999_9999_999A,
    0x3FF0_0000_0000_0000,
    0x4330_0000_0000_0000,
    0x7E37_E43C_8800_759C,
    0x7FEF_FFFF_FFFF_FFFF,
    0xBFF0_0000_0000_0000,
    0x7FF0_0000_0000_0000,
    0xFFF0_0000_0000_0000,
    0x7FF8_0000_0000_0000,
];

const ALPHAS: [f64; 30] = [
    0.0,
    -0.0,
    0.05,
    0.5,
    1.0,
    2.0,
    3.0,
    1.0 / 3.0,
    5e-324,
    1e-310,
    f64::MIN_POSITIVE,
    1e15,
    4503599627370496.0,
    9007199254740992.0,
    1e16,
    1e17,
    1e18,
    1e300,
    1e306,
    1e307,
    1e308,
    f64::MAX,
    -1.0,
    -0.05,
    -1e300,
    -1e308,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    0.125,
];

fn starts(
    graph: &Graph,
    state: &PartitionState,
    neighborhood: Neighborhood,
    alpha: f64,
    rng: &mut Mt19937GenRand64,
) -> Vec<f64> {
    let score = state.score(alpha);
    let mut out = vec![
        score,
        score.next_up(),
        score.next_down(),
        score - 1.0,
        score + 1.0,
        score - 0.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        0.0,
        -0.0,
        f64::MAX,
        -f64::MAX,
    ];
    let scores: Vec<f64> = smoothing::moves(state, neighborhood)
        .into_iter()
        .map(|mv| smoothing::move_score(state, graph, mv, alpha))
        .collect();
    if let Some(min) = scores
        .iter()
        .copied()
        .filter(|x| !x.is_nan())
        .reduce(f64::min)
    {
        out.extend([min, min.next_up(), min.next_down(), min + 1.0]);
    }
    if !scores.is_empty() {
        let pick = scores[rng.gen_range(0..scores.len())];
        out.extend([pick, pick.next_up(), pick.next_down()]);
    }
    out
}

#[test]
fn review_scan_matches_independent_oracle_on_adversarial_inputs() {
    let mut rng = Mt19937GenRand64::new(0xfeed);
    let graphs = families(&mut rng);
    let mut total = 0u64;
    let mut errors = 0u64;
    let mut draws = 0u64;
    let mut improved = 0u64;
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        for alpha in ALPHAS
            .iter()
            .copied()
            .filter(|a| SAMPLED_ALPHAS.contains(&a.to_bits()))
        {
            for non_finite in [NonFinite::Compare, NonFinite::Reject] {
                // One instance reused across every graph and partition.
                let mut scan = BestImprovement::new(neighborhood, alpha, non_finite);
                for graph in &graphs {
                    for partition in partitions(graph.node_count(), &mut rng) {
                        let state = PartitionState::new(graph, partition.clone()).unwrap();
                        let all_starts = starts(graph, &state, neighborhood, alpha, &mut rng);
                        let sampled = std::iter::once(all_starts[0])
                            .chain(all_starts[1..].choose_multiple(&mut rng, 5).copied());
                        for start in sampled {
                            let mut tie_rng = Mt19937GenRand64::new(rng.r#gen());
                            for _ in 0..rng.gen_range(0..400) {
                                let _: u64 = tie_rng.r#gen();
                            }
                            let mut oracle_rng = tie_rng.clone();
                            let untouched = tie_rng.clone();
                            let base = rng.gen_range(0..1000u64);
                            let (mut fast_evals, mut oracle_evals) = (base, base);
                            let fast = scan.scan(
                                graph,
                                &state,
                                start,
                                &mut tie_rng,
                                &CancellationToken::new(),
                                &mut fast_evals,
                            );
                            let expected = oracle(
                                graph,
                                &partition,
                                neighborhood,
                                alpha,
                                non_finite,
                                start,
                                &mut oracle_rng,
                                &mut oracle_evals,
                            );
                            let ctx = format!(
                                "{neighborhood:?} {non_finite:?} alpha={alpha:e} start={start:e} n={} partition={partition:?}",
                                graph.node_count()
                            );
                            assert_eq!(fast_evals, oracle_evals, "evals {ctx}");
                            assert!(tie_rng == oracle_rng, "rng {ctx}");
                            draws += u64::from(tie_rng != untouched);
                            total += 1;
                            match (fast, expected) {
                                (Ok(a), Ok(b)) => {
                                    assert_eq!(
                                        a.map(|(m, x)| (m, x.to_bits())),
                                        b.map(|(m, x)| (m, x.to_bits())),
                                        "outcome {ctx}"
                                    );
                                    improved += u64::from(a.is_some());
                                }
                                (Err(e), Err(())) => {
                                    assert_eq!(
                                        e.to_string(),
                                        "non-finite search evaluation",
                                        "{ctx}"
                                    );
                                    errors += 1;
                                }
                                (a, b) => panic!("{ctx}: {a:?} vs {b:?}"),
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("review scans={total} errors={errors} draws={draws} improved={improved}");
    assert!(errors > 0 && draws > 0 && improved > 0);
}

/// Mixed finite / non-finite penalties between the two Flip groups: the
/// error must come at the first vertex of the non-finite group, after the rule
/// has processed (and drawn for) the earlier vertices of the finite group.
#[test]
fn review_reject_flip_with_one_non_finite_group() {
    // n = 20, |A| = 3: flipping B gives d = -12 (144 * alpha), flipping A
    // gives d = -16 (256 * alpha). alpha = 1e306 -> A penalty is +inf.
    let n = 20;
    let graph = graph_from(n, vec![]);
    for first_a in [0usize, 5, 19] {
        let mut partition = vec![false; n];
        partition[first_a] = true;
        partition[(first_a + 7) % n] = true;
        partition[(first_a + 11) % n] = true;
        let state = PartitionState::new(&graph, partition.clone()).unwrap();
        let alpha = 1e306;
        assert!(
            state
                .flip_score(&graph, (0..n).find(|&v| !partition[v]).unwrap(), alpha)
                .is_finite()
        );
        for start in [f64::INFINITY, state.score(alpha), 1.44e308] {
            let mut scan = BestImprovement::new(Neighborhood::Flip, alpha, NonFinite::Reject);
            let mut rng = Mt19937GenRand64::new(9);
            let mut oracle_rng = rng.clone();
            let (mut e1, mut e2) = (0, 0);
            let fast = scan.scan(
                &graph,
                &state,
                start,
                &mut rng,
                &CancellationToken::new(),
                &mut e1,
            );
            let slow = oracle(
                &graph,
                &partition,
                Neighborhood::Flip,
                alpha,
                NonFinite::Reject,
                start,
                &mut oracle_rng,
                &mut e2,
            );
            let first = partition.iter().position(|&x| x).unwrap();
            assert!(fast.is_err() && slow.is_err());
            assert_eq!(e1, first as u64 + 1);
            assert_eq!(e1, e2);
            assert!(rng == oracle_rng);
        }
    }
}

/// Whole descents with a mid-size random graph; also verifies scan reuse
/// across descents on different graphs.
#[test]
fn review_long_descents_match_oracle() {
    let mut rng = Mt19937GenRand64::new(4242);
    let mut graphs = Vec::new();
    for (n, p) in [(40, 0.1), (60, 0.05), (50, 0.5), (36, 0.0)] {
        graphs.push(random_graph(n, p, &mut rng));
    }
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        for alpha in [0.0, 0.05, 0.5, 1e17, 5e-324] {
            let mut scan = BestImprovement::new(neighborhood, alpha, NonFinite::Compare);
            for graph in &graphs {
                for partition in partitions(graph.node_count(), &mut rng) {
                    let mut p = partition.clone();
                    let mut tie_rng = Mt19937GenRand64::new(rng.r#gen());
                    let mut oracle_rng = tie_rng.clone();
                    let mut current = graph.score(&p, alpha);
                    let (mut e1, mut e2) = (0, 0);
                    for _ in 0..500 {
                        let state = PartitionState::new(graph, p.clone()).unwrap();
                        let fast = scan
                            .scan(
                                graph,
                                &state,
                                current,
                                &mut tie_rng,
                                &CancellationToken::new(),
                                &mut e1,
                            )
                            .unwrap();
                        let slow = oracle(
                            graph,
                            &p,
                            neighborhood,
                            alpha,
                            NonFinite::Compare,
                            current,
                            &mut oracle_rng,
                            &mut e2,
                        )
                        .unwrap();
                        assert_eq!(
                            fast.map(|(m, x)| (m, x.to_bits())),
                            slow.map(|(m, x)| (m, x.to_bits()))
                        );
                        assert_eq!(e1, e2);
                        assert!(tie_rng == oracle_rng);
                        let Some((mv, best)) = fast else { break };
                        match mv {
                            Move::Flip(v) => p[v] = !p[v],
                            Move::Swap(a, b) => {
                                p[a] = !p[a];
                                p[b] = !p[b];
                            }
                        }
                        current = best;
                    }
                }
            }
        }
    }
}

/// Cancellation in the middle of a large scan: the error is returned, the
/// evaluation counter is untouched and nothing panics.
#[test]
fn review_cancel_mid_scan_leaves_counter() {
    let mut rng = Mt19937GenRand64::new(1);
    let graph = random_graph(3000, 0.001, &mut rng);
    let partition: Vec<bool> = (0..3000).map(|v| v % 2 == 0).collect();
    let state = PartitionState::new(&graph, partition).unwrap();
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        let cancel = CancellationToken::new();
        let mut scan = BestImprovement::new(neighborhood, 0.05, NonFinite::Compare);
        let mut evals = 5;
        let mut tie = Mt19937GenRand64::new(3);
        // Not cancelled: completes.
        scan.scan(&graph, &state, f64::INFINITY, &mut tie, &cancel, &mut evals)
            .unwrap();
        cancel.cancel();
        let before = evals;
        assert!(
            scan.scan(&graph, &state, f64::INFINITY, &mut tie, &cancel, &mut evals)
                .is_err()
        );
        assert_eq!(before, evals);
    }
}

/// n > CHECK_INTERVAL exercises the periodic cancellation checks of both
/// loops; compare whole descents with the move_score-based full scan.
#[test]
fn review_large_n_descents_match_full_scan() {
    let mut rng = Mt19937GenRand64::new(77);
    for (n, p) in [(2100usize, 0.002), (1030, 0.0), (1600, 0.01)] {
        let graph = random_graph(n, p, &mut rng);
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            for alpha in [0.05, 0.0] {
                let mut p: Vec<bool> = (0..n).map(|v| v < n / 2).collect();
                p.shuffle(&mut rng);
                if neighborhood == Neighborhood::Flip {
                    p[..n / 5].fill(true);
                }
                let mut state = PartitionState::new(&graph, p).unwrap();
                let mut scan = BestImprovement::new(neighborhood, alpha, NonFinite::Reject);
                let mut tie = Mt19937GenRand64::new(rng.r#gen());
                let mut naive_tie = tie.clone();
                let (mut e1, mut e2) = (0u64, 0u64);
                let mut current = state.score(alpha);
                for _ in 0..12 {
                    let fast = scan
                        .scan(
                            &graph,
                            &state,
                            current,
                            &mut tie,
                            &CancellationToken::new(),
                            &mut e1,
                        )
                        .unwrap();
                    let mut best = current;
                    let mut choice = None;
                    let mut ties = 0u64;
                    for mv in smoothing::moves(&state, neighborhood) {
                        e2 += 1;
                        let x = smoothing::move_score(&state, &graph, mv, alpha);
                        if x < best {
                            best = x;
                            choice = Some(mv);
                            ties = 1;
                        } else if choice.is_some() && x == best {
                            ties += 1;
                            if naive_tie.gen_range(0..ties) == 0 {
                                choice = Some(mv);
                            }
                        }
                    }
                    let slow = choice.map(|mv| (mv, best));
                    assert_eq!(
                        fast.map(|(m, x)| (m, x.to_bits())),
                        slow.map(|(m, x)| (m, x.to_bits()))
                    );
                    assert_eq!(e1, e2);
                    assert!(tie == naive_tie);
                    let Some((mv, best)) = fast else { break };
                    smoothing::apply(&mut state, &graph, mv);
                    current = best;
                }
            }
        }
    }
}
