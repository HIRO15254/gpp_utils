use gpp_utils::{
    experiment::config::{Neighborhood, SmoothingSpec},
    graph_partition::{Graph, PartitionState},
    optimization::CancellationToken,
    smoothing,
};

fn exhaustive_distance_one_average(
    graph: &Graph,
    partition: &[bool],
    alpha: f64,
    neighborhood: Neighborhood,
) -> f64 {
    let state = PartitionState::new(graph, partition.to_vec()).unwrap();
    let moves = smoothing::moves(&state, neighborhood);
    moves
        .iter()
        .map(|&mv| {
            let mut candidate = state.clone();
            smoothing::apply(&mut candidate, graph, mv);
            candidate.score(alpha)
        })
        .sum::<f64>()
        / moves.len() as f64
}

#[test]
fn random_full_k_on_six_vertices_equals_all_distance_one_and_two_states() {
    let graph = Graph::from_edges(6, vec![[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5]]).unwrap();
    let partition = vec![true, false, true, true, false, false];
    let state = PartitionState::new(&graph, partition.clone()).unwrap();
    let mut scores = Vec::new();
    for first in 0..6 {
        let mut p = partition.clone();
        p[first] = !p[first];
        scores.push(graph.score(&p, 0.1));
    }
    for first in 0..6 {
        for second in first + 1..6 {
            let mut p = partition.clone();
            p[first] = !p[first];
            p[second] = !p[second];
            scores.push(graph.score(&p, 0.1));
        }
    }
    let expected = scores.iter().sum::<f64>() / scores.len() as f64;
    let mut rng = gpp_utils::optimization::rng_for(&[b"full-k-reference"]);
    let mut evaluations = 0;
    let actual = smoothing::evaluate(
        &state,
        &graph,
        0.1,
        Neighborhood::Flip,
        &SmoothingSpec::RandomKAverage { k: 21 },
        Some(&mut rng),
        &CancellationToken::new(),
        &mut evaluations,
    )
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(evaluations, 21);
}

#[test]
fn all_and_weighted_smoothing_match_small_exhaustive_reference() {
    let graph = Graph::from_edges(4, vec![[0, 1], [1, 2], [2, 3], [0, 3], [0, 2]]).unwrap();
    let cancel = CancellationToken::new();
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        let partition = match neighborhood {
            Neighborhood::Flip => vec![true, false, false, true],
            Neighborhood::Swap => vec![true, true, false, false],
        };
        let state = PartitionState::new(&graph, partition.clone()).unwrap();
        let reference = exhaustive_distance_one_average(&graph, &partition, 0.3, neighborhood);
        let mut evaluations = 0;
        let actual = smoothing::evaluate(
            &state,
            &graph,
            0.3,
            neighborhood,
            &SmoothingSpec::AllAverage,
            None,
            &cancel,
            &mut evaluations,
        )
        .unwrap();
        assert_eq!(actual, reference);
        assert_eq!(
            evaluations,
            smoothing::moves(&state, neighborhood).len() as u64
        );

        let k = 2;
        let weighted = smoothing::evaluate(
            &state,
            &graph,
            0.3,
            neighborhood,
            &SmoothingSpec::WeightedAverage { k },
            None,
            &cancel,
            &mut 0,
        )
        .unwrap();
        let m = smoothing::moves(&state, neighborhood).len();
        let w = k.min(m) as f64 / m as f64;
        assert_eq!(weighted, w * reference + (1.0 - w) * state.score(0.3));
    }
}

#[test]
fn weighted_zero_is_exactly_real_and_random_k_rejects_impossible_k() {
    let graph = Graph::from_edges(4, vec![[0, 1], [1, 2]]).unwrap();
    let state = PartitionState::new(&graph, vec![true, true, false, false]).unwrap();
    let cancel = CancellationToken::new();
    let mut evaluations = 0;
    let value = smoothing::evaluate(
        &state,
        &graph,
        0.2,
        Neighborhood::Swap,
        &SmoothingSpec::WeightedAverage { k: 0 },
        None,
        &cancel,
        &mut evaluations,
    )
    .unwrap();
    assert_eq!(value, state.score(0.2));
    assert_eq!(evaluations, 1);
    assert!(
        smoothing::validate(
            &SmoothingSpec::RandomKAverage { k: 9 },
            4,
            Neighborhood::Swap
        )
        .is_err()
    );
}
