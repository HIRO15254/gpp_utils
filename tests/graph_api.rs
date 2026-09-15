use gpp_utils::graph_partition::{Graph, PartitionState};

#[test]
fn rebuilding_topology_keeps_adjacency_and_delta_evaluation_consistent() {
    let original = Graph::from_edges(4, vec![]).unwrap();
    let mut edges = original.edges().to_vec();
    edges.extend([[1, 0], [3, 2]]);
    let rebuilt = Graph::from_edges(original.node_count(), edges).unwrap();
    assert!(original.edges().is_empty());
    assert_eq!(original.degree(0), 0);
    assert_eq!(rebuilt.edges(), &[[0, 1], [2, 3]]);
    assert_eq!(rebuilt.neighbors(0), &[1]);
    assert_ne!(original.content_hash(), rebuilt.content_hash());
    for mask in 0..16 {
        let partition: Vec<_> = (0..4).map(|vertex| mask & (1 << vertex) != 0).collect();
        for vertex in 0..4 {
            let mut state = PartitionState::new(&rebuilt, partition.clone()).unwrap();
            let mut candidate = partition.clone();
            candidate[vertex] = !candidate[vertex];
            let expected = rebuilt.score(&candidate, 0.05);
            assert_eq!(
                state.flip_score(&rebuilt, vertex, 0.05).to_bits(),
                expected.to_bits()
            );
            state.apply_flip(&rebuilt, vertex);
            assert_eq!(state.score(0.05).to_bits(), expected.to_bits());
        }
    }
}
