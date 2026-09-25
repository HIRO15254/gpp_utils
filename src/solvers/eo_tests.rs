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
                let [mut indexed, mut sorted] =
                    both_paths(&spec, &graph, &state, Neighborhood::Swap, tau);
                for o in [false, true] {
                    let total = indexed.conditional_blocks(&state, o).unwrap();
                    assert_eq!(
                        total.to_bits(),
                        sorted.conditional_blocks(&state, o).unwrap().to_bits()
                    );
                    assert!(total > 0.0);
                    // Bits of every share and of T, with the contract's operation order.
                    let cum = &indexed.cum;
                    let mut literal = 0.0;
                    let mut shares = Vec::new();
                    for &(s, e) in &blocks {
                        let c_o = order[s..e].iter().filter(|&&v| side[v] == o).count();
                        if c_o > 0 {
                            let lo = if s == 0 { 0.0 } else { cum[s - 1] };
                            let w = cum[e - 1] - lo;
                            let x = w * c_o as f64 / (e - s) as f64;
                            literal += x;
                            shares.push((w.to_bits(), x.to_bits()));
                        }
                    }
                    assert_eq!(total.to_bits(), literal.to_bits());
                    for eo in [&indexed, &sorted] {
                        let actual: Vec<_> = eo
                            .eligible
                            .iter()
                            .map(|b| (b.weight.to_bits(), b.share.to_bits()))
                            .collect();
                        assert_eq!(actual, shares);
                    }
                    let mut counts = vec![0; graph.node_count()];
                    for k in 0..GRID {
                        let u2 = grid(k);
                        let v = indexed.second(&state, o, total, u2);
                        assert_eq!(v, sorted.second(&state, o, total, u2));
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
}

#[test]
fn exact_boundary_draws_select_the_upper_rank_and_block() {
    // tau = 0 and n = 4 give the dyadic weights 1/4, 1/2, 3/4, 1. The cut edge
    // puts 1 (B) and 0 (A) in block 0, isolated 2 (B) and 3 (A) in block 1.
    let graph = Graph::from_edges(4, vec![[0, 1]]).unwrap();
    let state = PartitionState::new(&graph, vec![true, false, false, true]).unwrap();
    for mut eo in both_paths(
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
            let total = eo.conditional_blocks(&state, o).unwrap();
            assert_eq!(total, 0.5);
            for (u2, v) in [(0.0, members[0]), (0.5, members[1]), (0.999, members[1])] {
                assert_eq!(eo.second(&state, o, total, u2), v, "o={o} u2={u2}");
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
    for [mut indexed, mut sorted] in [
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
        let total = indexed.conditional_blocks(&state, false).unwrap();
        assert_eq!(total.to_bits(), 0.0f64.to_bits());
        assert_eq!(sorted.conditional_blocks(&state, false).unwrap(), 0.0);
        let mut counts = [0usize; 12];
        for k in 0..GRID {
            let v = indexed.second(&state, false, total, grid(k));
            assert_eq!(v, sorted.second(&state, false, total, grid(k)));
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
    for mut eo in both_paths(
        &FitnessSpec::default(),
        &graph,
        &state,
        Neighborhood::Swap,
        1.0e308,
    ) {
        for (u, v) in [(0.1, 2), (0.3, 3), (0.6, 0), (0.9, 1)] {
            assert_eq!(eo.first(&state, u), v);
        }
        let total = eo.conditional_blocks(&state, false).unwrap();
        assert_eq!(total, 0.5);
        assert_eq!(eo.second(&state, false, total, 0.25), 2);
        assert_eq!(eo.second(&state, false, total, 0.75), 3);
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

/// Canonical order and blocks represented by one ranking.
fn ranking_order(ranking: &Ranking) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut order = Vec::new();
    let mut blocks = Vec::new();
    for bucket in 0..ranking.counts.len() {
        let start = order.len();
        for side in 0..2 {
            order.extend(
                ranking.members[2 * bucket + side]
                    .iter()
                    .map(|&v| v as usize),
            );
        }
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
        if index.rankings.len() == 1 && relation != 2 && kind.depends_on_majority() {
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
            ranking_order(&index.rankings[ranking]),
            naive_blocks(&values, side),
            "size relation {relation} of {spec:?}"
        );
    }
    for ranking in &index.rankings {
        let mut start = 0;
        for (bucket, count) in ranking.counts.iter().enumerate() {
            for (side, &side_count) in count.iter().enumerate() {
                let members = &ranking.members[2 * bucket + side];
                assert_eq!(members.len(), side_count as usize);
                assert!(members.windows(2).all(|w| w[0] < w[1]));
            }
            let size = ranking.size(bucket);
            for position in start..start + size {
                assert_eq!(ranking.locate(position), (bucket, start));
            }
            start += size;
        }
        assert_eq!(start, graph.node_count());
    }
}

#[test]
fn index_matches_rebuild_and_direct_values_after_random_moves() {
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
            let expected_rankings = match neighborhood {
                Neighborhood::Flip if kind.depends_on_majority() => 3,
                _ => 1,
            };
            let mut visited = [false; 3];
            for _ in 0..1500 {
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
                assert_eq!(index.rankings.len(), expected_rankings);
                assert_rankings_match_values(index, &spec, &graph, &state);
            }
            if neighborhood == Neighborhood::Flip {
                assert_eq!(visited, [true; 3], "{spec:?} crossed every size state");
            }
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
                    let mut eo = Eo::new(
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
                        let total = eo.conditional_blocks(&state, o).unwrap();
                        let eligible =
                            eo_v2_reference::eligible_blocks(&order, &blocks, &cum, side, o);
                        let production = |k: u64| rank[eo.second(&state, o, total, u_at(k))];
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
