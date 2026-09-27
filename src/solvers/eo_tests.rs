use super::*;
use crate::experiment::config::FitnessSpec;
use crate::fitness::FitnessRegistry;
use rand::seq::SliceRandom;

const GRID: usize = 1 << 16;

fn spec(kind: &str, params: serde_json::Value) -> FitnessSpec {
    FitnessSpec {
        kind: kind.into(),
        params,
    }
}

fn builtin_specs() -> Vec<FitnessSpec> {
    let mut specs = vec![FitnessSpec::default()];
    for alpha in [0.0, 0.5, 1.0] {
        specs.push(spec(
            "multiplicative",
            serde_json::json!({ "alpha": alpha }),
        ));
    }
    for beta in [0.0, 0.5, 3.0] {
        specs.push(spec("additive", serde_json::json!({ "beta": beta })));
    }
    specs
}

fn builtin_kind(spec: &FitnessSpec) -> BuiltinFitness {
    match FitnessRegistry::default()
        .create_engine_fitness(spec)
        .unwrap()
    {
        EngineFitness::Builtin(kind) => kind,
        EngineFitness::Custom(_) => unreachable!("default registry entries are built-in"),
    }
}

/// Degrees 0 to 6, a triangle, a path and six isolated vertices.
fn tie_graph() -> Graph {
    let mut edges: Vec<[usize; 2]> = (1..=6).map(|v| [0, v]).collect();
    edges.extend([
        [7, 8],
        [8, 9],
        [7, 9],
        [10, 11],
        [11, 12],
        [12, 13],
        [1, 7],
        [2, 10],
        [3, 4],
        [5, 6],
    ]);
    Graph::from_edges(20, edges).unwrap()
}

fn random_state(graph: &Graph, seed: u64) -> PartitionState {
    let mut rng = Mt19937GenRand64::new(seed);
    let partition = (0..graph.node_count()).map(|_| rng.r#gen()).collect();
    PartitionState::new(graph, partition).unwrap()
}

/// Canonical order and blocks computed directly from `values`.
fn naive_blocks(values: &[f64], side: &[bool]) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap()
            .then(side[a].cmp(&side[b]))
            .then(a.cmp(&b))
    });
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        blocks.push((start, end));
        start = end;
    }
    (order, blocks)
}

/// Index-path and sorted-path selectors for the same built-in definition.
fn both_paths(
    spec: &FitnessSpec,
    graph: &Graph,
    state: &PartitionState,
    neighborhood: Neighborhood,
    tau: f64,
) -> [Eo; 2] {
    let registry = FitnessRegistry::default();
    let mut count = 0;
    let indexed = Eo::new(
        registry.create_engine_fitness(spec).unwrap(),
        graph,
        state,
        neighborhood,
        tau,
        &mut count,
    )
    .unwrap();
    assert_eq!(count, graph.node_count() as u64);
    let mut sorted = Eo::new(
        EngineFitness::Custom(registry.create(spec).unwrap()),
        graph,
        state,
        neighborhood,
        tau,
        &mut count,
    )
    .unwrap();
    rank_sorted(&mut sorted, graph, state);
    assert!(indexed.is_indexed() && !sorted.is_indexed());
    [indexed, sorted]
}

fn rank_sorted(eo: &mut Eo, graph: &Graph, state: &PartitionState) {
    if let Ranker::Sorted(sorted) = &mut eo.ranker {
        sorted.rank(graph, state, &mut 0).unwrap();
    }
}

/// Normalized weight `r^-tau / Z` of each rank position, computed per rank.
fn rank_weights(n: usize, tau: f64) -> Vec<f64> {
    let raw: Vec<f64> = (1..=n).map(|r| (r as f64).powf(-tau)).collect();
    let z: f64 = raw.iter().sum();
    raw.iter().map(|w| w / z).collect()
}

/// Probability of each vertex under the first-vertex rule: `W_block / c_block`.
fn first_closed_form(weights: &[f64], order: &[usize], blocks: &[(usize, usize)]) -> Vec<f64> {
    let mut p = vec![0.0; order.len()];
    for &(s, e) in blocks {
        let weight: f64 = weights[s..e].iter().sum();
        for &v in &order[s..e] {
            p[v] = weight / (e - s) as f64;
        }
    }
    p
}

/// Probability of each vertex under the second-vertex rule conditioned on `o`.
fn second_closed_form(
    weights: &[f64],
    order: &[usize],
    blocks: &[(usize, usize)],
    side: &[bool],
    o: bool,
) -> Vec<f64> {
    let mut p = vec![0.0; order.len()];
    let mut total = 0.0;
    for &(s, e) in blocks {
        let weight: f64 = weights[s..e].iter().sum();
        for &v in &order[s..e] {
            if side[v] == o {
                p[v] = weight / (e - s) as f64;
                total += p[v];
            }
        }
    }
    p.iter().map(|x| x / total).collect()
}

fn grid(k: usize) -> f64 {
    (k as f64 + 0.5) / GRID as f64
}

fn assert_measure(counts: &[usize], expected: &[f64], context: &str) {
    for (v, (&count, &p)) in counts.iter().zip(expected).enumerate() {
        let measured = count as f64 / GRID as f64;
        assert!(
            (measured - p).abs() <= 2.0 / GRID as f64,
            "{context}: vertex {v} measured {measured}, expected {p}"
        );
    }
}

#[test]
fn power_law_weights_are_finite_normalized_and_monotone() {
    for n in [1, 2, 7, 500] {
        for tau in [0.0, 1.0e-300, 0.5, 1.5, 3.0, 1.0e308, f64::MAX] {
            let cum = power_law_cdf(n, tau);
            assert_eq!(cum.len(), n);
            assert!(cum.iter().all(|c| c.is_finite()), "n={n} tau={tau}");
            assert!(cum.windows(2).all(|w| w[0] <= w[1]), "n={n} tau={tau}");
            assert_eq!(cum[n - 1], 1.0);
        }
        for (k, &c) in power_law_cdf(n, 0.0).iter().enumerate() {
            assert_eq!(c, (k + 1) as f64 / n as f64, "tau = 0 is uniform");
        }
        assert!(power_law_cdf(n, 1.0e308).iter().all(|&c| c == 1.0));
    }
    assert!(power_law_cdf(0, 1.5).is_empty());
}

#[test]
fn first_vertex_measure_matches_block_average_on_both_paths() {
    let graph = tie_graph();
    for seed in [1, 2, 3] {
        let state = random_state(&graph, seed);
        for spec in builtin_specs() {
            let values = FitnessRegistry::default()
                .create(&spec)
                .unwrap()
                .values(&graph, &state)
                .unwrap();
            let (order, blocks) = naive_blocks(&values, state.partition());
            assert!(blocks.len() < graph.node_count(), "the fixture has ties");
            for tau in [0.0, 0.5, 1.5, 3.0] {
                let [indexed, sorted] = both_paths(&spec, &graph, &state, Neighborhood::Flip, tau);
                let mut counts = vec![0; graph.node_count()];
                for k in 0..GRID {
                    let u = grid(k);
                    let v = indexed.first(&state, u);
                    assert_eq!(v, sorted.first(&state, u), "{spec:?} tau={tau} u={u}");
                    counts[v] += 1;
                }
                let weights = rank_weights(graph.node_count(), tau);
                let expected = first_closed_form(&weights, &order, &blocks);
                let context = format!("{spec:?} seed={seed} tau={tau}");
                assert_measure(&counts, &expected, &context);
                if tau == 0.0 {
                    let uniform = vec![1.0 / graph.node_count() as f64; graph.node_count()];
                    assert_measure(&counts, &uniform, &context);
                }
            }
        }
    }
}

#[test]
fn second_vertex_measure_matches_conditional_closed_form_on_both_paths() {
    let graph = tie_graph();
    // Blocks without conditioned members followed by eligible ones: the index
    // path adds their +0.0 shares and overwrites their entries.
    let mut skipped = 0;
    for seed in [4, 5] {
        let state = random_state(&graph, seed);
        let side = state.partition();
        for spec in builtin_specs() {
            let values = FitnessRegistry::default()
                .create(&spec)
                .unwrap()
                .values(&graph, &state)
                .unwrap();
            let (order, blocks) = naive_blocks(&values, side);
            for tau in [0.0, 0.5, 1.5, 3.0] {
                let [indexed, sorted] = both_paths(&spec, &graph, &state, Neighborhood::Swap, tau);
                for o in [false, true] {
                    let (indexed_blocks, total) = indexed.conditional_blocks(&state, o).unwrap();
                    let (sorted_blocks, sorted_total) =
                        sorted.conditional_blocks(&state, o).unwrap();
                    assert_eq!(total.to_bits(), sorted_total.to_bits());
                    assert!(total > 0.0);
                    // Bits of every share and of T, with the contract's operation order.
                    let cum = &indexed.cum;
                    let mut literal = 0.0;
                    let mut shares = Vec::new();
                    let mut pending = 0;
                    for &(s, e) in &blocks {
                        let c_o = order[s..e].iter().filter(|&&v| side[v] == o).count();
                        if c_o > 0 {
                            let lo = if s == 0 { 0.0 } else { cum[s - 1] };
                            let w = cum[e - 1] - lo;
                            let x = w * c_o as f64 / (e - s) as f64;
                            literal += x;
                            shares.push((w.to_bits(), x.to_bits()));
                            skipped += pending;
                            pending = 0;
                        } else {
                            pending += 1;
                        }
                    }
                    assert_eq!(total.to_bits(), literal.to_bits());
                    for eligible in [&indexed_blocks, &sorted_blocks] {
                        let actual: Vec<_> = eligible
                            .iter()
                            .map(|b| (b.weight.to_bits(), b.share.to_bits()))
                            .collect();
                        assert_eq!(actual, shares);
                    }
                    let mut counts = vec![0; graph.node_count()];
                    for k in 0..GRID {
                        let u2 = grid(k);
                        let v = indexed.second(&state, &indexed_blocks, o, total, u2);
                        assert_eq!(v, sorted.second(&state, &sorted_blocks, o, total, u2));
                        assert_eq!(side[v], o);
                        counts[v] += 1;
                    }
                    let weights = rank_weights(graph.node_count(), tau);
                    let expected = second_closed_form(&weights, &order, &blocks, side, o);
                    let context = format!("{spec:?} seed={seed} tau={tau} o={o}");
                    assert_measure(&counts, &expected, &context);
                    if tau == 0.0 {
                        let members = side.iter().filter(|&&s| s == o).count();
                        let uniform: Vec<f64> = side
                            .iter()
                            .map(|&s| if s == o { 1.0 / members as f64 } else { 0.0 })
                            .collect();
                        assert_measure(&counts, &uniform, &context);
                    }
                }
            }
        }
    }
    assert!(
        skipped > 0,
        "the fixture skips blocks between eligible ones"
    );
}

#[test]
fn exact_boundary_draws_select_the_upper_rank_and_block() {
    // tau = 0 and n = 4 give the dyadic weights 1/4, 1/2, 3/4, 1. The cut edge
    // puts 1 (B) and 0 (A) in block 0, isolated 2 (B) and 3 (A) in block 1.
    let graph = Graph::from_edges(4, vec![[0, 1]]).unwrap();
    let state = PartitionState::new(&graph, vec![true, false, false, true]).unwrap();
    for eo in both_paths(
        &FitnessSpec::default(),
        &graph,
        &state,
        Neighborhood::Swap,
        0.0,
    ) {
        assert_eq!(eo.cum, [0.25, 0.5, 0.75, 1.0]);
        // `u == cum[k]` selects rank k + 2, the start of the next block for k = 1.
        for (u, v) in [(0.0, 1), (0.25, 0), (0.5, 2), (0.75, 3), (0.999, 3)] {
            assert_eq!(eo.first(&state, u), v, "u={u}");
        }
        // Both blocks share 1/4 per side; `target == next` moves to the next block.
        for (o, members) in [(false, [1, 2]), (true, [0, 3])] {
            let (eligible, total) = eo.conditional_blocks(&state, o).unwrap();
            assert_eq!(total, 0.5);
            for (u2, v) in [(0.0, members[0]), (0.5, members[1]), (0.999, members[1])] {
                assert_eq!(
                    eo.second(&state, &eligible, o, total, u2),
                    v,
                    "o={o} u2={u2}"
                );
            }
        }
    }
}

/// Vertex 0 (group A) is the only fully cut vertex. The lowest group-B block
/// is `{1, 6}` (lambda0 = 2/3), above it `{2, 3, 7, 8}` with isolated A vertices.
fn fallback_fixture() -> (Graph, PartitionState) {
    let graph =
        Graph::from_edges(12, vec![[0, 1], [0, 6], [1, 2], [1, 3], [6, 7], [6, 8]]).unwrap();
    let a = [0, 4, 5, 9, 10, 11];
    let partition = (0..12).map(|v| a.contains(&v)).collect();
    (
        graph.clone(),
        PartitionState::new(&graph, partition).unwrap(),
    )
}

#[test]
fn huge_tau_falls_back_to_lowest_opposite_block_without_nan() {
    let (graph, state) = fallback_fixture();
    for [mut indexed, sorted] in [
        both_paths(
            &FitnessSpec::default(),
            &graph,
            &state,
            Neighborhood::Swap,
            1.0e308,
        ),
        both_paths(
            &FitnessSpec::default(),
            &graph,
            &state,
            Neighborhood::Swap,
            f64::MAX,
        ),
    ] {
        for k in 0..GRID {
            assert_eq!(indexed.first(&state, grid(k)), 0);
            assert_eq!(sorted.first(&state, grid(k)), 0);
        }
        // Every group-B block has zero weight, so T underflows to exactly zero.
        let (indexed_blocks, total) = indexed.conditional_blocks(&state, false).unwrap();
        assert_eq!(total.to_bits(), 0.0f64.to_bits());
        let (sorted_blocks, sorted_total) = sorted.conditional_blocks(&state, false).unwrap();
        assert_eq!(sorted_total, 0.0);
        let mut counts = [0usize; 12];
        for k in 0..GRID {
            let v = indexed.second(&state, &indexed_blocks, false, total, grid(k));
            assert_eq!(
                v,
                sorted.second(&state, &sorted_blocks, false, total, grid(k))
            );
            counts[v] += 1;
        }
        assert_eq!(counts[1], GRID / 2);
        assert_eq!(counts[6], GRID / 2);
        // Through the engine entry point: two draws per swap, never NaN or panic.
        for seed in 0..20 {
            let mut rng = Mt19937GenRand64::new(seed);
            let mut twin = rng.clone();
            let mv = indexed
                .select(&graph, &state, Neighborhood::Swap, &mut rng, &mut 0)
                .unwrap();
            assert!(matches!(mv, Move::Swap(0, 1) | Move::Swap(0, 6)), "{mv:?}");
            let u2 = {
                let _: f64 = twin.r#gen();
                twin.r#gen::<f64>()
            };
            assert_eq!(mv, Move::Swap(0, if u2 < 0.5 { 1 } else { 6 }));
            assert!(rng == twin, "swap consumes exactly two draws");
        }
    }
    // One tied block holding both sides: rank 1 carries all weight, which the
    // block shares equally, so T = 2/4 and each B member gets half of it.
    let graph = Graph::from_edges(4, vec![]).unwrap();
    let state = PartitionState::new(&graph, vec![true, true, false, false]).unwrap();
    for eo in both_paths(
        &FitnessSpec::default(),
        &graph,
        &state,
        Neighborhood::Swap,
        1.0e308,
    ) {
        for (u, v) in [(0.1, 2), (0.3, 3), (0.6, 0), (0.9, 1)] {
            assert_eq!(eo.first(&state, u), v);
        }
        let (eligible, total) = eo.conditional_blocks(&state, false).unwrap();
        assert_eq!(total, 0.5);
        assert_eq!(eo.second(&state, &eligible, false, total, 0.25), 2);
        assert_eq!(eo.second(&state, &eligible, false, total, 0.75), 3);
    }
}

#[test]
fn empty_and_one_sided_distributions_are_errors() {
    let registry = FitnessRegistry::default();
    let empty = Graph::from_edges(0, vec![]).unwrap();
    let empty_state = PartitionState::new(&empty, vec![]).unwrap();
    let single = Graph::from_edges(1, vec![]).unwrap();
    let single_state = PartitionState::new(&single, vec![false]).unwrap();
    for custom in [false, true] {
        let fitness = || {
            if custom {
                EngineFitness::Custom(registry.create(&FitnessSpec::default()).unwrap())
            } else {
                registry
                    .create_engine_fitness(&FitnessSpec::default())
                    .unwrap()
            }
        };
        let mut rng = Mt19937GenRand64::new(3);
        let untouched = rng.clone();
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            let mut eo =
                Eo::new(fitness(), &empty, &empty_state, neighborhood, 1.5, &mut 0).unwrap();
            let error = eo
                .select(&empty, &empty_state, neighborhood, &mut rng, &mut 0)
                .unwrap_err();
            assert_eq!(error.to_string(), EMPTY_DISTRIBUTION);
            assert!(rng == untouched, "no draw before the emptiness check");
        }
        let mut eo = Eo::new(
            fitness(),
            &single,
            &single_state,
            Neighborhood::Flip,
            0.0,
            &mut 0,
        )
        .unwrap();
        let mv = eo
            .select(&single, &single_state, Neighborhood::Flip, &mut rng, &mut 0)
            .unwrap();
        assert_eq!(mv, Move::Flip(0));
        let mut eo = Eo::new(
            fitness(),
            &single,
            &single_state,
            Neighborhood::Swap,
            0.0,
            &mut 0,
        )
        .unwrap();
        let error = eo
            .select(&single, &single_state, Neighborhood::Swap, &mut rng, &mut 0)
            .unwrap_err();
        assert_eq!(error.to_string(), EMPTY_DISTRIBUTION);
    }
}

#[test]
fn custom_values_are_validated_and_counted() {
    struct Fixed(Vec<f64>);
    impl VertexFitness for Fixed {
        fn values(&self, _: &Graph, _: &PartitionState) -> Result<Vec<f64>> {
            Ok(self.0.clone())
        }
    }
    let graph = Graph::from_edges(3, vec![[0, 1]]).unwrap();
    let state = PartitionState::new(&graph, vec![true, false, true]).unwrap();
    for (values, valid) in [
        (vec![0.0, -0.0, 1.0], true),
        (vec![0.0, 1.0], false),
        (vec![0.0, f64::NAN, 1.0], false),
        (vec![0.0, f64::INFINITY, 1.0], false),
    ] {
        let len = values.len() as u64;
        let mut built = 0;
        let mut eo = Eo::new(
            EngineFitness::Custom(Box::new(Fixed(values))),
            &graph,
            &state,
            Neighborhood::Flip,
            1.5,
            &mut built,
        )
        .unwrap();
        assert_eq!(built, 0);
        let mut counted = 0;
        let result = eo.select(
            &graph,
            &state,
            Neighborhood::Flip,
            &mut Mt19937GenRand64::new(1),
            &mut counted,
        );
        assert_eq!(counted, len);
        match result {
            Ok(_) => assert!(valid),
            Err(error) => {
                assert!(!valid);
                assert_eq!(error.to_string(), "fitness returned invalid values");
            }
        }
    }
    // -0.0 == 0.0: vertices 0 (side A) and 1 (side B) form one block, B first.
    let mut eo = Eo::new(
        EngineFitness::Custom(Box::new(Fixed(vec![0.0, -0.0, 1.0]))),
        &graph,
        &state,
        Neighborhood::Flip,
        0.0,
        &mut 0,
    )
    .unwrap();
    rank_sorted(&mut eo, &graph, &state);
    let Ranker::Sorted(sorted) = &eo.ranker else {
        unreachable!()
    };
    let order: Vec<_> = (0..3).map(|p| sorted.vertex(p)).collect();
    assert_eq!(order, [1, 0, 2]);
    assert_eq!(sorted.block_around(0), (0, 2));
    assert_eq!(sorted.block_around(1), (0, 2));
    assert_eq!(sorted.block_around(2), (2, 3));
}

#[test]
fn canonical_keys_order_like_partial_cmp_then_side_then_vertex() {
    let values = [
        -f64::MAX,
        -1.5,
        -f64::MIN_POSITIVE,
        -5.0e-324,
        -0.0,
        0.0,
        5.0e-324,
        f64::MIN_POSITIVE,
        0.1,
        1.0,
        f64::MAX,
    ];
    let mut keys = Vec::new();
    for (i, &a) in values.iter().enumerate() {
        for side in [false, true] {
            for vertex in [0, 1, 7, usize::MAX >> 1] {
                keys.push((a, side, vertex, i));
            }
        }
    }
    for &(a, side_a, vertex_a, _) in &keys {
        let key_a = canonical_key(a, side_a, vertex_a);
        assert_eq!(key_side(key_a), side_a);
        assert_eq!(key_vertex(key_a), vertex_a);
        for &(b, side_b, vertex_b, _) in &keys {
            let key_b = canonical_key(b, side_b, vertex_b);
            let expected = a
                .partial_cmp(&b)
                .unwrap()
                .then(side_a.cmp(&side_b))
                .then(vertex_a.cmp(&vertex_b));
            assert_eq!(key_a.cmp(&key_b), expected, "{a:?} {b:?}");
            assert_eq!(key_value(key_a) == key_value(key_b), a == b);
        }
    }
}

/// Members of `cell` in ascending order, read bit by bit from its bitset (an
/// empty cell may not hold a slot; `Rankings::assert_pool_consistent` checks
/// that no other bits are set).
fn cell_members(ranking: Ranking<'_>, cell: usize) -> Vec<usize> {
    if ranking.counts(cell / 2)[cell % 2] == 0 {
        return Vec::new();
    }
    let words = ranking.members(cell);
    let mut members = Vec::new();
    for (w, &word) in words.iter().enumerate().filter(|&(_, &word)| word != 0) {
        members.extend((0..64).filter(|&b| word >> b & 1 == 1).map(|b| 64 * w + b));
    }
    members
}

/// Canonical order and blocks represented by one ranking.
fn ranking_order(ranking: Ranking<'_>) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut order = Vec::new();
    let mut blocks = Vec::new();
    for bucket in 0..GROUP * ranking.groups.len() {
        let start = order.len();
        order.extend(cell_members(ranking, 2 * bucket));
        order.extend(cell_members(ranking, 2 * bucket + 1));
        if order.len() > start {
            blocks.push((start, order.len()));
        }
    }
    (order, blocks)
}

/// Check every maintained ranking against values computed directly.
fn assert_rankings_match_values(
    index: &BuiltinIndex,
    spec: &FitnessSpec,
    graph: &Graph,
    state: &PartitionState,
) {
    let kind = builtin_kind(spec);
    let side = state.partition();
    let current = FitnessRegistry::default()
        .create(spec)
        .unwrap()
        .values(graph, state)
        .unwrap();
    assert_eq!(
        ranking_order(index.ranking(state)),
        naive_blocks(&current, side),
        "current state of {spec:?}"
    );
    for (relation, &ranking) in index.ranking_of_state.iter().enumerate() {
        if index.rankings.count() == 1 && relation != 2 && kind.depends_on_majority() {
            continue; // Swap keeps only the size state of its initial partition (balanced for valid runs).
        }
        // Majority by the size relation: A larger, B larger, equal.
        let values: Vec<f64> = (0..graph.node_count())
            .map(|v| {
                let majority = match relation {
                    0 => side[v],
                    1 => !side[v],
                    _ => false,
                };
                kind.lambda(lambda0(graph.degree(v), state.cuts_at()[v]), majority)
            })
            .collect();
        assert_eq!(
            ranking_order(index.rankings.get(ranking)),
            naive_blocks(&values, side),
            "size relation {relation} of {spec:?}"
        );
    }
    for r in 0..index.rankings.count() {
        assert_ranking_structures(index.rankings.get(r), graph.node_count());
    }
}

/// Check the bucket-level structures and the member selection of `ranking`
/// against naive scans of its cell bitsets: `select` and `member` against the
/// ascending member lists, `locate` on every position against a prefix scan of
/// the bucket sizes, the non-empty iteration against a full scan, and the
/// group totals and flags against the bucket sizes.
fn assert_ranking_structures(ranking: Ranking<'_>, n: usize) {
    let buckets = GROUP * ranking.groups.len();
    let mut start = 0;
    let mut nonempty = Vec::new();
    for bucket in 0..buckets {
        let low = cell_members(ranking, 2 * bucket);
        let high = cell_members(ranking, 2 * bucket + 1);
        for (side, members) in [&low, &high].into_iter().enumerate() {
            let cell = 2 * bucket + side;
            assert_eq!(members.len(), ranking.counts(bucket)[side] as usize);
            for (k, &v) in members.iter().enumerate() {
                assert_eq!(ranking.select(cell, k), v, "cell {cell} rank {k}");
            }
        }
        let block: Vec<usize> = low.iter().chain(&high).copied().collect();
        assert_eq!(ranking.size(bucket), block.len());
        for (offset, &v) in block.iter().enumerate() {
            assert_eq!(ranking.member(bucket, offset), v);
            let position = start + offset;
            assert_eq!(
                ranking.locate(position),
                (bucket, start),
                "position {position}"
            );
        }
        if !block.is_empty() {
            nonempty.push(bucket);
        }
        start += block.len();
    }
    assert_eq!(start, n);
    let mut visited = Vec::new();
    ranking.for_each_nonempty(|bucket, counts| {
        assert_eq!(counts, ranking.counts(bucket), "bucket {bucket}");
        visited.push(bucket);
    });
    assert_eq!(visited, nonempty, "non-empty buckets");
    for (g, group) in ranking.groups.iter().enumerate() {
        let sizes: Vec<usize> = (GROUP * g..GROUP * (g + 1))
            .map(|bucket| ranking.size(bucket))
            .collect();
        assert_eq!(
            group.size as usize,
            sizes.iter().sum::<usize>(),
            "group {g}"
        );
        for (b, &size) in sizes.iter().enumerate() {
            assert_eq!(
                group.nonempty >> b & 1 == 1,
                size > 0,
                "group {g} bucket {b}"
            );
        }
    }
}

#[test]
fn index_matches_rebuild_and_direct_values_after_random_moves() {
    with_layout(false, || assert_index_matches_after_random_moves(1500));
    with_layout(true, || assert_index_matches_after_random_moves(400));
}

fn assert_index_matches_after_random_moves(moves: usize) {
    let graph = tie_graph();
    let n = graph.node_count();
    for spec in builtin_specs() {
        let kind = builtin_kind(&spec);
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            let mut rng = Mt19937GenRand64::new(0xe0);
            let mut partition: Vec<bool> = (0..n).map(|_| rng.r#gen()).collect();
            if neighborhood == Neighborhood::Swap {
                partition = (0..n).map(|v| v < n / 2).collect();
                partition.shuffle(&mut rng);
            }
            let mut state = PartitionState::new(&graph, partition).unwrap();
            let mut count = 0;
            let mut eo = Eo::new(
                EngineFitness::Builtin(kind),
                &graph,
                &state,
                neighborhood,
                1.0,
                &mut count,
            )
            .unwrap();
            assert_eq!(
                eo.is_pooled(),
                FORCE_POOLED.with(std::cell::Cell::get).unwrap()
            );
            let expected_rankings = match neighborhood {
                Neighborhood::Flip if kind.depends_on_majority() => 3,
                _ => 1,
            };
            let mut visited = [false; 3];
            for _ in 0..moves {
                let mv = match neighborhood {
                    Neighborhood::Flip => Move::Flip(rng.gen_range(0..n)),
                    Neighborhood::Swap => {
                        let pick = |rng: &mut Mt19937GenRand64, side: bool| loop {
                            let v = rng.gen_range(0..n);
                            if state.partition()[v] == side {
                                break v;
                            }
                        };
                        let a = pick(&mut rng, true);
                        Move::Swap(a, pick(&mut rng, false))
                    }
                };
                crate::smoothing::apply(&mut state, &graph, mv);
                let before = count;
                eo.applied(&graph, &state, mv, &mut count);
                assert_eq!(
                    count - before,
                    match mv {
                        Move::Flip(v) => 1 + graph.degree(v) as u64,
                        Move::Swap(a, b) => 2 + (graph.degree(a) + graph.degree(b)) as u64,
                    }
                );
                visited[size_state(state.size_a(), state.size_b())] = true;
                eo.assert_index_consistent(&graph, &state, neighborhood);
                let Ranker::Index(index) = &eo.ranker else {
                    unreachable!()
                };
                assert_eq!(index.rankings.count(), expected_rankings);
                assert_rankings_match_values(index, &spec, &graph, &state);
            }
            if neighborhood == Neighborhood::Flip {
                assert_eq!(visited, [true; 3], "{spec:?} crossed every size state");
            }
        }
    }
}

/// `moves` random relocations of `n` vertices among the keys of
/// `slot_value`, ranked in the `R` size states `relations` (in the bitset
/// layout of the thread, see [`with_layout`]); see
/// [`ranking_structures_match_naive_scans_after_random_relocations`].
fn assert_random_relocations<const R: usize>(
    kind: BuiltinFitness,
    slot_value: &[f64],
    relations: [usize; R],
    n: usize,
    moves: usize,
    rng: &mut Mt19937GenRand64,
) {
    let slots = slot_value.len();
    let keys = 2 * slots as u32;
    let majority = relations.map(majority_flags);
    let naive = |key_of: &[u32], flags: [bool; 2]| {
        let values: Vec<f64> = key_of
            .iter()
            .map(|&key| {
                let key = key as usize;
                kind.lambda(slot_value[key % slots], flags[key / slots])
            })
            .collect();
        let side: Vec<bool> = key_of.iter().map(|&key| key >= slots as u32).collect();
        naive_blocks(&values, &side)
    };
    // A few popular keys keep most buckets empty and a few crowded.
    let popular: Vec<u32> = (0..6).map(|_| rng.gen_range(0..keys)).collect();
    let draw = |rng: &mut Mt19937GenRand64| {
        if rng.gen_bool(0.7) {
            popular[rng.gen_range(0..popular.len())]
        } else {
            rng.gen_range(0..keys)
        }
    };
    let build = |key_of: &[u32]| {
        let (mut rankings, cells) = Rankings::new(kind, slot_value, &majority, n).unwrap();
        for (v, &key) in key_of.iter().enumerate() {
            for &cell in &cells[key as usize * R..(key as usize + 1) * R] {
                rankings.insert(v, cell);
            }
        }
        (rankings, cells)
    };
    let mut key_of: Vec<u32> = (0..n).map(|_| draw(rng)).collect();
    let (mut rankings, cells) = build(&key_of);
    let row = |key: u32| -> [u32; R] { std::array::from_fn(|r| cells[key as usize * R + r]) };
    assert_eq!(rankings.count(), R);
    for step in 0..=moves {
        if step > 0 {
            let v = rng.gen_range(0..n);
            let key = draw(rng);
            rankings.relocate_one(v, row(key_of[v]), row(key));
            key_of[v] = key;
        }
        let context = format!(
            "{kind:?} n={n} relations {relations:?} pooled {} step {step}",
            rankings.pool.pooled
        );
        for (r, &flags) in majority.iter().enumerate() {
            let ranking = rankings.get(r);
            assert_eq!(ranking_order(ranking), naive(&key_of, flags), "{context}");
            assert_ranking_structures(ranking, n);
        }
        rankings.assert_pool_consistent();
        assert!(rankings == build(&key_of).0, "{context}");
    }
}

/// Random relocations on synthetic rankings with up to 300 buckets (five
/// groups, the last one partly padding), most of them empty: the membership
/// and blocks equal the canonical order computed from the values, the
/// structures pass the naive scans of [`assert_ranking_structures`], and the
/// rankings equal a rebuild after every move. `alpha = 0` and `beta = 0` map
/// many keys to one cell, so some moves keep their cell in some rankings.
#[test]
fn ranking_structures_match_naive_scans_after_random_relocations() {
    let kinds = [
        BuiltinFitness::Default,
        BuiltinFitness::Multiplicative { alpha: 0.5 },
        BuiltinFitness::Multiplicative { alpha: 0.0 },
        BuiltinFitness::Additive { beta: 3.0 },
        BuiltinFitness::Additive { beta: 0.0 },
    ];
    let slot_value: Vec<f64> = (0..150).map(|i| i as f64 / 149.0).collect();
    let mut rng = Mt19937GenRand64::new(0x5e1ec7);
    for n in [1, 63, 64, 65, 200] {
        for kind in kinds {
            for relation in 0..SIZE_STATES {
                assert_random_relocations(kind, &slot_value, [relation], n, 100, &mut rng);
            }
            assert_random_relocations(kind, &slot_value, [0, 1, 2], n, 100, &mut rng);
        }
    }
}

/// The same with pooled bitsets: cells take a slot when they gain their first
/// member and free it with their last one, including in one move (a vertex
/// alone in its cell moving to an empty cell).
#[test]
fn pooled_ranking_structures_match_naive_scans_after_random_relocations() {
    let kinds = [
        BuiltinFitness::Default,
        BuiltinFitness::Multiplicative { alpha: 0.5 },
        BuiltinFitness::Additive { beta: 0.0 },
    ];
    let slot_value: Vec<f64> = (0..150).map(|i| i as f64 / 149.0).collect();
    let mut rng = Mt19937GenRand64::new(0x5107);
    with_layout(true, || {
        for n in [1, 2, 65, 200] {
            for kind in kinds {
                assert_random_relocations(kind, &slot_value, [2], n, 100, &mut rng);
                assert_random_relocations(kind, &slot_value, [0, 1, 2], n, 100, &mut rng);
            }
        }
    });
}

/// Dense bitsets for the baseline graphs; pooled ones when dense bitsets would
/// outgrow both 32 MiB and the pooled bound (degrees around 1000 make hundreds
/// of thousands of buckets). On a graph with many degrees the pooled bitsets
/// stay within `R * n` bitsets while moves empty and fill cells.
#[test]
fn bitset_layout_bounds_memory() {
    // (vertices, buckets of all rankings, rankings): baseline-sized indexes and
    // random graphs with n = 2000, d = 200 (28 MiB of dense bitsets).
    for (n, buckets, count) in [
        (124, 1152, 3),
        (500, 1664, 3),
        (500, 384, 1),
        (2000, 56_832, 3),
    ] {
        assert!(!pooled_layout(buckets, n, count), "{n} {buckets} {count}");
    }
    // Random graphs with n = 4000, d = 1000 (595 MiB and 149 MiB dense) and
    // n = 10000, d = 100 (55 MiB dense, above the pooled bound of 36 MiB).
    for (n, buckets, count) in [(4000, 613_888, 3), (4000, 153_472, 1), (10_000, 22_720, 3)] {
        assert!(pooled_layout(buckets, n, count), "{n} {buckets} {count}");
    }
    with_layout(true, assert_pooled_bitsets_stay_bounded);
}

fn assert_pooled_bitsets_stay_bounded() {
    // Degrees 0 to 150 on 400 vertices.
    let n = 400;
    let mut edges = std::collections::BTreeSet::new();
    for v in 0..n {
        for k in 1..=v % 151 / 2 {
            let u = (v + 7 * k) % n;
            edges.insert([v.min(u), v.max(u)]);
        }
    }
    let graph = Graph::from_edges(n, edges.into_iter().collect()).unwrap();
    let mut state = random_state(&graph, 11);
    let spec = spec("multiplicative", serde_json::json!({ "alpha": 0.5 }));
    let [mut eo, _] = both_paths(&spec, &graph, &state, Neighborhood::Flip, 1.0);
    let Ranker::Index(index) = &eo.ranker else {
        unreachable!()
    };
    let pool = &index.rankings.pool;
    let bitset = 8 * pool.words;
    let dense = 2 * index.rankings.base[3] * bitset;
    assert!(
        pool.pooled && dense > 20 * 3 * n * bitset,
        "{dense} bytes dense"
    );
    let mut rng = Mt19937GenRand64::new(5);
    for step in 0..60 {
        let mv = eo
            .select(&graph, &state, Neighborhood::Flip, &mut rng, &mut 0)
            .unwrap();
        crate::smoothing::apply(&mut state, &graph, mv);
        eo.applied(&graph, &state, mv, &mut 0);
        if step % 20 == 19 {
            eo.assert_index_consistent(&graph, &state, Neighborhood::Flip);
        }
    }
    let Ranker::Index(index) = &eo.ranker else {
        unreachable!()
    };
    let pool = &index.rankings.pool;
    assert!(
        pool.bits.len() * 8 <= 3 * n * bitset,
        "{} bytes",
        pool.bits.len() * 8
    );
}

/// Independent review: synthetic rankings whose distinct value counts are
/// exactly 1 to 4 whole groups (no padding) or one bucket off, with `n`
/// around word boundaries, after every relocation.
#[test]
fn review_rankings_with_exact_group_multiples() {
    let moves = 30;
    let mut rng = Mt19937GenRand64::new(0x64);
    for slots in [1usize, 2, 63, 64, 65, 127, 128, 129, 192, 255, 256] {
        let slot_value: Vec<f64> = (0..slots)
            .map(|i| {
                if slots == 1 {
                    0.5
                } else {
                    i as f64 / (slots - 1) as f64
                }
            })
            .collect();
        for n in [1usize, 2, 63, 64, 65, 128, 129] {
            // Default: exactly `slots` buckets.
            assert_random_relocations(
                BuiltinFitness::Default,
                &slot_value,
                [2],
                n,
                moves,
                &mut rng,
            );
            // Minority values all round to 1.0: `slots + 1` buckets.
            assert_random_relocations(
                BuiltinFitness::Additive { beta: 1.0e-17 },
                &slot_value,
                [0, 1, 2],
                n,
                moves,
                &mut rng,
            );
            assert_random_relocations(
                BuiltinFitness::Multiplicative { alpha: 0.5 },
                &slot_value,
                [0, 1, 2],
                n,
                moves,
                &mut rng,
            );
            assert_random_relocations(
                BuiltinFitness::Multiplicative { alpha: 5.0e-324 },
                &slot_value,
                [1],
                n,
                moves,
                &mut rng,
            );
        }
    }
}

#[test]
fn select_in_word_finds_every_set_bit() {
    let mut rng = Mt19937GenRand64::new(0xb175);
    let mut words = vec![1, 1 << 63, u64::MAX, 0x8000_0000_0000_0001, 0x5555 << 30];
    words.extend((0..300).map(|_| rng.r#gen::<u64>() & rng.r#gen::<u64>()));
    words.extend((0..300).map(|_| rng.r#gen::<u64>() | rng.r#gen::<u64>()));
    for word in words {
        let bits: Vec<usize> = (0..64).filter(|&b| word >> b & 1 == 1).collect();
        for (k, &b) in bits.iter().enumerate() {
            assert_eq!(select_in_word(word, k), b, "{word:#x} k={k}");
        }
    }
}

// Naive executable specification of EO algorithm v2, independent of `eo.rs`.
// Only its pure selection functions are used here; the stepping engine is
// exercised by `exact_tests.rs`.
#[allow(dead_code)]
mod eo_v2_reference {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/solvers/test_reference/eo_v2_reference.rs"
    ));
}

/// Draw value `k * 2^-53`: every value `gen::<f64>()` can return has this form.
fn u_at(k: u64) -> f64 {
    k as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Smallest `k < 2^53` with `f(k) >= target` for a non-decreasing `f`, or `2^53`.
fn threshold(f: &dyn Fn(u64) -> usize, target: usize) -> u64 {
    let (mut lo, mut hi) = (0u64, 1u64 << 53);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if f(mid) >= target {
            hi = mid
        } else {
            lo = mid + 1
        }
    }
    lo
}

/// Every threshold on the whole draw grid equals the reference arithmetic, so a
/// change that only moves a boundary by one ulp is detected. Sampled draws and
/// trajectories cannot see such changes.
#[test]
fn selection_thresholds_match_the_reference_on_the_whole_draw_grid() {
    let registry = FitnessRegistry::default();
    let generated = Graph::generate(
        &crate::experiment::config::GraphSpec {
            kind: crate::experiment::config::GraphKind::Random,
            node_count: 60,
            expected_degree: 5.0,
            seed: 3,
        },
        &crate::optimization::CancellationToken::new(),
    )
    .unwrap();
    for graph in [tie_graph(), generated] {
        let n = graph.node_count();
        for spec in builtin_specs() {
            for tau in [0.0, 0.7, 1.5, 3.0] {
                for seed in 0..2 {
                    let state = random_state(&graph, seed);
                    let context = format!("{spec:?} tau {tau} seed {seed} n {n}");
                    let values = registry
                        .create(&spec)
                        .unwrap()
                        .values(&graph, &state)
                        .unwrap();
                    let side = state.partition();
                    let (order, blocks) = eo_v2_reference::canonical(&values, side);
                    let cum = eo_v2_reference::cumulative(n, tau);
                    let mut position = vec![0; n];
                    for (p, &v) in order.iter().enumerate() {
                        position[v] = p;
                    }
                    let eo = Eo::new(
                        EngineFitness::Builtin(builtin_kind(&spec)),
                        &graph,
                        &state,
                        Neighborhood::Flip,
                        tau,
                        &mut 0,
                    )
                    .unwrap();
                    let production = |k: u64| position[eo.first(&state, u_at(k))];
                    let reference = |k: u64| {
                        position[eo_v2_reference::first_for(&order, &blocks, &cum, u_at(k))]
                    };
                    for target in 1..n {
                        assert_eq!(
                            threshold(&production, target),
                            threshold(&reference, target),
                            "first vertex, position {target}: {context}"
                        );
                    }
                    for o in [false, true] {
                        let members: Vec<usize> =
                            order.iter().copied().filter(|&v| side[v] == o).collect();
                        if members.is_empty() {
                            continue;
                        }
                        let mut rank = vec![usize::MAX; n];
                        for (r, &v) in members.iter().enumerate() {
                            rank[v] = r;
                        }
                        let (production_blocks, total) = eo.conditional_blocks(&state, o).unwrap();
                        let eligible =
                            eo_v2_reference::eligible_blocks(&order, &blocks, &cum, side, o);
                        let production =
                            |k: u64| rank[eo.second(&state, &production_blocks, o, total, u_at(k))];
                        let reference = |k: u64| {
                            rank[eo_v2_reference::second_from(&eligible, &order, side, o, u_at(k))]
                        };
                        for target in 1..members.len() {
                            assert_eq!(
                                threshold(&production, target),
                                threshold(&reference, target),
                                "second vertex on side {o}, rank {target}: {context}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Swaps move every touched vertex at most once, on each path of
/// `BuiltinIndex::refresh_swap`: two flips when the closed-neighborhood
/// signatures of `a` and `b` are disjoint, otherwise the marked pass, for
/// adjacent endpoints, common neighbors, and disjoint neighborhoods whose
/// signatures collide (`n > 512` folds `v` and `v + 512` onto one bit). The
/// rows moved are exactly those of the vertices whose cut count or side
/// changed (common neighbors keep theirs), the index equals a rebuild after
/// every swap, and the rows match the state (a debug assertion of
/// `refresh_move`).
#[test]
fn swaps_rerank_every_touched_vertex_once_on_every_path() {
    let n = 700;
    let mut rng = Mt19937GenRand64::new(0x5a9);
    let mut edges = std::collections::BTreeSet::new();
    while edges.len() < 2 * n {
        let (a, b) = (rng.gen_range(0..n), rng.gen_range(0..n));
        if a != b {
            edges.insert([a.min(b), a.max(b)]);
        }
    }
    let graph = Graph::from_edges(n, edges.into_iter().collect()).unwrap();
    let closed = |v: usize| {
        let mut set: Vec<usize> = graph.neighbors(v).to_vec();
        set.push(v);
        set
    };
    let fold = |set: &[usize]| {
        let mut bits = [0u64; 8];
        for &u in set {
            bits[u % 512 / 64] |= 1 << (u % 64);
        }
        bits
    };
    for spec in [
        FitnessSpec::default(),
        spec("additive", serde_json::json!({ "beta": 3.0 })),
    ] {
        for pooled in [false, true] {
            with_layout(pooled, || {
                let mut partition: Vec<bool> = (0..n).map(|v| v < n / 2).collect();
                partition.shuffle(&mut rng);
                let mut state = PartitionState::new(&graph, partition).unwrap();
                let mut eo = Eo::new(
                    EngineFitness::Builtin(builtin_kind(&spec)),
                    &graph,
                    &state,
                    Neighborhood::Swap,
                    1.0,
                    &mut 0,
                )
                .unwrap();
                // [disjoint, adjacent, common neighbor, colliding signatures]
                let mut paths = [0; 4];
                for step in 0..400 {
                    let side = state.partition().to_vec();
                    let a = loop {
                        let a = rng.gen_range(0..n);
                        if side[a] {
                            break a;
                        }
                    };
                    let opposite = |v: &usize| !side[*v];
                    let neighbors = graph.neighbors(a);
                    let second: Vec<usize> = neighbors
                        .iter()
                        .flat_map(|&u| graph.neighbors(u).iter().copied())
                        .collect();
                    let partner: Vec<usize> = [a.checked_add(512), a.checked_sub(512)]
                        .into_iter()
                        .flatten()
                        .filter(|&v| v < n)
                        .collect();
                    let pick = |list: &[usize], rng: &mut Mt19937GenRand64| {
                        let list: Vec<usize> = list.iter().copied().filter(opposite).collect();
                        list.choose(rng).copied()
                    };
                    let b = match step % 4 {
                        1 => pick(neighbors, &mut rng),
                        2 => pick(&second, &mut rng),
                        3 => pick(&partner, &mut rng),
                        _ => None,
                    }
                    .unwrap_or_else(|| {
                        loop {
                            let b = rng.gen_range(0..n);
                            if !side[b] {
                                break b;
                            }
                        }
                    });
                    let (closed_a, closed_b) = (closed(a), closed(b));
                    let collide = fold(&closed_a)
                        .iter()
                        .zip(&fold(&closed_b))
                        .any(|(x, y)| x & y != 0);
                    let adjacent = closed_a.contains(&b);
                    let common = closed_a.iter().any(|u| *u != a && closed_b.contains(u));
                    let path = match (collide, adjacent, common) {
                        (false, ..) => 0,
                        (true, true, _) => 1,
                        (true, false, true) => 2,
                        (true, false, false) => 3,
                    };
                    paths[path] += 1;
                    let mv = Move::Swap(a, b);
                    let mut touched: Vec<usize> =
                        closed_a.iter().chain(&closed_b).copied().collect();
                    touched.sort_unstable();
                    touched.dedup();
                    let vertex_state = |state: &PartitionState, u: usize| {
                        (state.cuts_at()[u], state.partition()[u])
                    };
                    let before: Vec<_> = touched.iter().map(|&u| vertex_state(&state, u)).collect();
                    crate::smoothing::apply(&mut state, &graph, mv);
                    let changed = touched
                        .iter()
                        .zip(&before)
                        .filter(|&(&u, &old)| vertex_state(&state, u) != old)
                        .count() as u64;
                    let mut count = 0;
                    let moved = MOVED_ROWS.with(std::cell::Cell::get);
                    eo.applied(&graph, &state, mv, &mut count);
                    let moved = MOVED_ROWS.with(std::cell::Cell::get) - moved;
                    assert_eq!(moved, changed, "rows moved by swapping {a} and {b}");
                    assert_eq!(count, 2 + (graph.degree(a) + graph.degree(b)) as u64);
                    eo.assert_index_consistent(&graph, &state, Neighborhood::Swap);
                }
                assert!(
                    paths.iter().all(|&p| p > 0),
                    "{spec:?} pooled {pooled}: {paths:?}"
                );
            });
        }
    }
}

/// Independent review (round 2): one relocation pass in which every relocated
/// vertex leaves a cell that stays non-empty for a cell that was empty, in
/// every ranking, so the pass takes `R` new pool slots per vertex. The pool
/// before the pass holds `R * pairs` non-empty cells in as many slots as the
/// build grew to, so the free stack is below, at and above what passes of
/// 1 to `pairs` vertices take; the reservation of `Rankings::relocator` must
/// cover each pass, and it may not grow the pool beyond `min(R * n, cells)`.
#[test]
fn review_pool_reserves_a_new_slot_per_relocated_vertex_and_ranking() {
    fn check<const R: usize>(kind: BuiltinFitness, relations: [usize; R]) {
        // Distinct values: every key of side `false` has its own cell in
        // every ranking.
        let slot_value: Vec<f64> = (0..300).map(|i| i as f64 / 299.0).collect();
        let majority = relations.map(majority_flags);
        for pairs in [1usize, 5, 6, 8, 11, 16, 21, 22, 32, 43, 48, 64] {
            let n = 2 * pairs;
            for moved in 0..=pairs {
                let (mut rankings, cells) = Rankings::new(kind, &slot_value, &majority, n).unwrap();
                let row = |key: usize| -> [u32; R] { std::array::from_fn(|r| cells[key * R + r]) };
                // Vertices 2i and 2i + 1 share key i; vertex 2i of the first
                // `moved` pairs moves to the unused key 150 + i.
                let mut key_of: Vec<usize> = (0..n).map(|v| v / 2).collect();
                let build = |key_of: &[usize]| {
                    let (mut rankings, _) = Rankings::new(kind, &slot_value, &majority, n).unwrap();
                    for (v, &key) in key_of.iter().enumerate() {
                        for cell in row(key) {
                            rankings.insert(v, cell);
                        }
                    }
                    rankings
                };
                for (v, &key) in key_of.iter().enumerate() {
                    for cell in row(key) {
                        rankings.insert(v, cell);
                    }
                }
                let before = rankings.pool.free.len() - 2 - rankings.pool.top;
                assert_eq!(before, R * pairs);
                {
                    let mut relocator = rankings.relocator(moved);
                    for i in 0..moved {
                        relocator.relocate::<R, true>(2 * i, row(i), row(150 + i));
                    }
                }
                for i in 0..moved {
                    key_of[2 * i] = 150 + i;
                }
                let context = format!("{kind:?} pairs {pairs} moved {moved}");
                rankings.assert_pool_consistent();
                assert_eq!(
                    rankings.pool.free.len() - 2 - rankings.pool.top,
                    R * (pairs + moved),
                    "{context}"
                );
                assert!(rankings == build(&key_of), "{context}");
                // The largest possible reservation stays within the bound.
                drop(rankings.relocator(n));
                rankings.assert_pool_consistent();
            }
        }
    }
    with_layout(true, || {
        check(BuiltinFitness::Default, [2]);
        check(BuiltinFitness::Multiplicative { alpha: 0.5 }, [0, 1, 2]);
    });
}

/// Independent review (round 2): the layout an index chooses by itself (no
/// [`with_layout`]): dense for a baseline graph, pooled once majority-dependent
/// rankings of a random graph with n = 2000 and d = 400 would need about
/// 89 MiB of dense bitsets, where the pooled bitsets stay within `R * n` and
/// the index follows a few EO moves exactly.
#[test]
fn review_indexes_choose_their_layout_without_override() {
    use crate::experiment::config::{GraphKind, GraphSpec};
    use crate::optimization::CancellationToken;
    let generate = |node_count, expected_degree| {
        let spec = GraphSpec {
            kind: GraphKind::Random,
            node_count,
            expected_degree,
            seed: 0,
        };
        Graph::generate(&spec, &CancellationToken::new()).unwrap()
    };
    let registry = FitnessRegistry::default();
    let build = |spec: &FitnessSpec, graph: &Graph, state: &PartitionState| {
        Eo::new(
            registry.create_engine_fitness(spec).unwrap(),
            graph,
            state,
            Neighborhood::Flip,
            1.5,
            &mut 0,
        )
        .unwrap()
    };
    let baseline = generate(500, 20.0);
    let state = random_state(&baseline, 2);
    for spec in builtin_specs() {
        assert!(!build(&spec, &baseline, &state).is_pooled(), "{spec:?}");
    }
    let graph = generate(2000, 400.0);
    let mut state = random_state(&graph, 3);
    assert!(!build(&FitnessSpec::default(), &graph, &state).is_pooled());
    let multiplicative = spec("multiplicative", serde_json::json!({ "alpha": 0.5 }));
    let mut eo = build(&multiplicative, &graph, &state);
    assert!(eo.is_pooled());
    let mut rng = Mt19937GenRand64::new(9);
    for _ in 0..3 {
        let mv = eo
            .select(&graph, &state, Neighborhood::Flip, &mut rng, &mut 0)
            .unwrap();
        crate::smoothing::apply(&mut state, &graph, mv);
        eo.applied(&graph, &state, mv, &mut 0);
    }
    eo.assert_index_consistent(&graph, &state, Neighborhood::Flip);
    let Ranker::Index(index) = &eo.ranker else {
        unreachable!()
    };
    let pool = &index.rankings.pool;
    assert!(pool.bits.len() <= 3 * 2000 * pool.words);
}
/// Independent review (round 3): swaps whose endpoints share every neighbor,
/// with and without the edge between them, isolated endpoints whose
/// signatures collide (`v` and `v + 512`), an isolated endpoint colliding
/// with a neighbor of the other one, a pendant edge, and endpoints of very
/// different degrees with one common neighbor, in both directions and both
/// layouts: each swap moves exactly the rows whose cut count or side changed,
/// counts `2 + deg(a) + deg(b)` fitness values and leaves an index equal to a
/// rebuild.
#[test]
fn review_swaps_with_shared_neighborhoods_and_isolated_endpoints() {
    let n = 1030;
    let mut edges = vec![
        // 0 and 1 share their neighbors 2 and 3 and are not adjacent.
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        // 4 and 5 are adjacent and share their neighbors 6 and 7.
        [4, 5],
        [4, 6],
        [4, 7],
        [5, 6],
        [5, 7],
        // 521 (bit 9) has a neighbor; 9 is isolated.
        [521, 10],
        // 12 is adjacent to 523 (bit 11); 11 is isolated.
        [12, 523],
        // A pendant edge.
        [13, 14],
        // 15 has five neighbors, 21 one, 16 is common.
        [15, 16],
        [15, 17],
        [15, 18],
        [15, 19],
        [15, 20],
        [21, 16],
    ];
    // A sparse background, so that the fitness values vary.
    let mut rng = Mt19937GenRand64::new(0x3e3);
    let mut background = std::collections::BTreeSet::new();
    while background.len() < 600 {
        let (a, b) = (rng.gen_range(100..n), rng.gen_range(100..n));
        if a != b && ![520, 521, 523].contains(&a) && ![520, 521, 523].contains(&b) {
            background.insert([a.min(b), a.max(b)]);
        }
    }
    edges.extend(background);
    let graph = Graph::from_edges(n, edges).unwrap();
    // [a, b] with `a` on side `true` and `b` on side `false`.
    let pairs = [
        [0, 1],
        [4, 5],
        [8, 520],
        [9, 521],
        [11, 12],
        [13, 14],
        [15, 21],
    ];
    let mut partition = vec![false; n];
    for [a, _] in pairs {
        partition[a] = true;
    }
    let mut assigned = pairs.len();
    for (v, side) in partition.iter_mut().enumerate() {
        if assigned < n / 2 && v >= 100 && !pairs.iter().any(|p| p.contains(&v)) {
            *side = true;
            assigned += 1;
        }
    }
    assert_eq!(assigned, n / 2);
    for spec in [
        FitnessSpec::default(),
        spec("additive", serde_json::json!({ "beta": 3.0 })),
    ] {
        for pooled in [false, true] {
            with_layout(pooled, || {
                let mut state = PartitionState::new(&graph, partition.clone()).unwrap();
                let mut eo = Eo::new(
                    EngineFitness::Builtin(builtin_kind(&spec)),
                    &graph,
                    &state,
                    Neighborhood::Swap,
                    1.0,
                    &mut 0,
                )
                .unwrap();
                assert_eq!(eo.is_pooled(), pooled);
                // Each pair forth and back, in both argument orders.
                for round in 0..4 {
                    for [a, b] in pairs {
                        let (a, b) = if round % 2 == 0 { (a, b) } else { (b, a) };
                        let (a, b) = if round < 2 { (a, b) } else { (b, a) };
                        assert_ne!(state.partition()[a], state.partition()[b]);
                        let mut touched: Vec<usize> = [a, b]
                            .iter()
                            .flat_map(|&v| {
                                std::iter::once(v).chain(graph.neighbors(v).iter().copied())
                            })
                            .collect();
                        touched.sort_unstable();
                        touched.dedup();
                        let key = |state: &PartitionState, u: usize| {
                            (state.cuts_at()[u], state.partition()[u])
                        };
                        let before: Vec<_> = touched.iter().map(|&u| key(&state, u)).collect();
                        let mv = Move::Swap(a, b);
                        crate::smoothing::apply(&mut state, &graph, mv);
                        let changed = touched
                            .iter()
                            .zip(&before)
                            .filter(|&(&u, &old)| key(&state, u) != old)
                            .count() as u64;
                        let mut count = 0;
                        let moved = MOVED_ROWS.with(std::cell::Cell::get);
                        eo.applied(&graph, &state, mv, &mut count);
                        let moved = MOVED_ROWS.with(std::cell::Cell::get) - moved;
                        let context = format!("{spec:?} pooled {pooled} swap {a} {b}");
                        assert_eq!(moved, changed, "{context}");
                        assert_eq!(
                            count,
                            2 + (graph.degree(a) + graph.degree(b)) as u64,
                            "{context}"
                        );
                        eo.assert_index_consistent(&graph, &state, Neighborhood::Swap);
                    }
                }
                // Four swaps of each pair restore the partition.
                assert_eq!(state.partition(), &partition[..]);
            });
        }
    }
}
