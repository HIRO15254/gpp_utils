use super::*;
use crate::experiment::config::{
    BasinMode, Budget, FitnessSpec, GraphKind, GraphSpec, Measurement, Schedule,
};
use rand::Rng;
use std::sync::Arc;

// Frozen copy of the smoothing module (`crate::smoothing`, non-test code) at
// e4b6a1c. The frozen engine and runner import it instead of the live module,
// so they stay an independent oracle when production smoothing is optimized.
#[allow(clippy::too_many_arguments)]
mod smoothing_e4b6a1c {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/solvers/test_reference/smoothing_e4b6a1c.rs"
    ));
}

// Executable oracle for SA, HC, smoothing and the runner: production engine at
// 51577f9, with Graph getter adapters. EO changed intentionally in algorithm v2
// and is compared with `eo_v2` below instead.
mod reference {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/solvers/test_reference/engine_51577f9.rs"
    ));

    impl<'a> Engine<'a> {
        pub(super) fn rng_probe(&self, count: usize) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
            use rand::Rng;
            let mut select = self.select_rng.clone();
            let mut tie = self.tie_rng.clone();
            let mut smooth = self.smooth_rng.clone();
            (
                (0..count).map(|_| select.r#gen()).collect(),
                (0..count).map(|_| tie.r#gen()).collect(),
                (0..count).map(|_| smooth.r#gen()).collect(),
            )
        }

        /// The select, tie and smoothing streams, for exact state comparison.
        pub(super) fn rngs(&self) -> [&Mt19937GenRand64; 3] {
            [&self.select_rng, &self.tie_rng, &self.smooth_rng]
        }
    }

    #[allow(clippy::too_many_arguments, clippy::collapsible_if)]
    pub mod runner {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/solvers/test_reference/runner_51577f9.rs"
        ));
    }
}

// Naive executable specification of EO algorithm v2, independent of `../eo.rs`.
mod eo_v2 {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/solvers/test_reference/eo_v2_reference.rs"
    ));
}

fn rng_probe(engine: &Engine<'_>, count: usize) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let mut select = engine.select_rng.clone();
    let mut tie = engine.tie_rng.clone();
    let mut smooth = engine.smooth_rng.clone();
    (
        (0..count).map(|_| select.r#gen()).collect(),
        (0..count).map(|_| tie.r#gen()).collect(),
        (0..count).map(|_| smooth.r#gen()).collect(),
    )
}

fn graph() -> Graph {
    // Includes ties, triangles, and an isolated vertex (7).
    Graph::from_edges(
        8,
        vec![
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 4],
            [3, 5],
            [4, 6],
            [5, 6],
        ],
    )
    .unwrap()
}

fn condition(neighborhood: Neighborhood, solver: SolverSpec, alpha: f64) -> Condition {
    condition_for(&graph(), neighborhood, solver, alpha)
}

fn condition_for(
    graph: &Graph,
    neighborhood: Neighborhood,
    solver: SolverSpec,
    alpha: f64,
) -> Condition {
    Condition {
        graph: GraphSpec {
            kind: GraphKind::Random,
            node_count: graph.node_count(),
            expected_degree: if graph.node_count() > 1 {
                2.0 * graph.edges().len() as f64 / graph.node_count() as f64
            } else {
                0.0
            },
            seed: 41,
        },
        neighborhood,
        alpha,
        solver,
        budget: Budget { max_steps: 40 },
        measurement: Measurement {
            schedule: Schedule::Explicit,
            steps: vec![0, 1, 2, 3, 7, 16, 40],
            basin: BasinMode::Both,
            max_basin_steps: 40,
            diagnostics: true,
            best_basin: false,
        },
    }
}

struct FrozenDefaultFactory;
struct FrozenDefaultFitness;

impl crate::fitness::FitnessFactory for FrozenDefaultFactory {
    fn version(&self) -> &str {
        "good_edge_fraction-v1"
    }

    fn validate(&self, params: &serde_json::Value) -> crate::error::Result<()> {
        if params.as_object().is_some_and(|x| x.is_empty()) || params.is_null() {
            Ok(())
        } else {
            Err(crate::error::Error::msg(
                "default fitness params must be empty",
            ))
        }
    }

    fn create(
        &self,
        _params: &serde_json::Value,
    ) -> crate::error::Result<Box<dyn crate::fitness::VertexFitness>> {
        Ok(Box::new(FrozenDefaultFitness))
    }
}

impl crate::fitness::VertexFitness for FrozenDefaultFitness {
    fn values(&self, graph: &Graph, state: &PartitionState) -> crate::error::Result<Vec<f64>> {
        Ok((0..graph.node_count())
            .map(|vertex| {
                let degree = graph.degree(vertex);
                if degree == 0 {
                    1.0
                } else {
                    let good = graph
                        .neighbors(vertex)
                        .iter()
                        .filter(|&&neighbor| {
                            state.partition()[neighbor] == state.partition()[vertex]
                        })
                        .count();
                    good as f64 / degree as f64
                }
            })
            .collect())
    }
}

fn frozen_registry() -> FitnessRegistry {
    let mut registry = FitnessRegistry::default_registry();
    registry.register("default", Arc::new(FrozenDefaultFactory));
    registry
}

fn assert_engine_exact(c: &Condition, seed: u64, steps: usize) {
    assert_engine_exact_on(&graph(), c, seed, steps)
}

fn assert_engine_exact_on(g: &Graph, c: &Condition, seed: u64, steps: usize) {
    let registry = FitnessRegistry::default_registry();
    let reference_registry = frozen_registry();
    let cancel = CancellationToken::new();
    let mut actual = Engine::new(g, c, seed, &registry, &cancel).unwrap();
    let mut expected = reference::Engine::new(g, c, seed, &reference_registry, &cancel).unwrap();

    for step in 0..=steps {
        assert_eq!(
            actual.state.partition(),
            expected.state.partition(),
            "partition at {step}"
        );
        assert_eq!(
            actual.state.score(c.alpha).to_bits(),
            expected.state.score(c.alpha).to_bits(),
            "real score at {step}"
        );
        assert_eq!(
            actual.state.score(c.alpha).to_bits(),
            g.score(actual.state.partition(), c.alpha).to_bits(),
            "incremental state differs from independent graph score at {step}"
        );
        assert_eq!(
            actual.search_evaluation.to_bits(),
            expected.search_evaluation.to_bits(),
            "search score at {step}"
        );
        assert_eq!(actual.objective_evaluations, expected.objective_evaluations);
        assert_eq!(actual.fitness_values, expected.fitness_values);
        assert_eq!(actual.applied_moves, expected.applied_moves);
        // 624 outputs cover two full 312-word MT19937-64 state blocks. Comparing future output also
        // detects a consumed draw even when the selected move happened to tie.
        assert_eq!(
            rng_probe(&actual, 624),
            expected.rng_probe(624),
            "RNG at {step}"
        );
        if step == steps {
            break;
        }
        let a = actual.step(&cancel).unwrap();
        let b = expected.step(&cancel).unwrap();
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "status at {step}");
        if a != StepStatus::Continue {
            break;
        }
    }
}

fn isolated_graph() -> Graph {
    Graph::from_edges(8, vec![]).unwrap()
}

fn complete_graph() -> Graph {
    Graph::from_edges(
        8,
        (0..8)
            .flat_map(|a| (a + 1..8).map(move |b| [a, b]))
            .collect(),
    )
    .unwrap()
}

/// Heterogeneous degrees (0 to 10), many ties and three isolated vertices.
fn eo_graph() -> Graph {
    let mut edges: Vec<[usize; 2]> = (1..=10).map(|v| [0, v]).collect();
    for a in 11..=15 {
        for b in a + 1..=15 {
            edges.push([a, b]);
        }
    }
    edges.extend((16..21).map(|v| [v, v + 1]));
    edges.extend([
        [1, 11],
        [2, 16],
        [3, 21],
        [4, 5],
        [6, 7],
        [8, 9],
        [12, 17],
        [13, 22],
        [22, 23],
        [23, 24],
        [22, 24],
        [25, 26],
    ]);
    Graph::from_edges(30, edges).unwrap()
}

fn builtin_fitness_specs() -> Vec<FitnessSpec> {
    let mut specs = vec![FitnessSpec::default()];
    for alpha in [0.0, 0.5, 1.0] {
        specs.push(FitnessSpec {
            kind: "multiplicative".into(),
            params: serde_json::json!({ "alpha": alpha }),
        });
    }
    for beta in [0.0, 0.5, 3.0] {
        specs.push(FitnessSpec {
            kind: "additive".into(),
            params: serde_json::json!({ "beta": beta }),
        });
    }
    specs
}

const EO_TAUS: [f64; 6] = [0.0, 1.0e-300, 0.5, 1.5, 3.0, 1.0e308];

fn eo_condition(
    g: &Graph,
    neighborhood: Neighborhood,
    tau: f64,
    fitness: FitnessSpec,
) -> Condition {
    condition_for(g, neighborhood, SolverSpec::Eo { tau, fitness }, 0.05)
}

fn size_state_of(state: &PartitionState) -> usize {
    match state.size_a().cmp(&state.size_b()) {
        std::cmp::Ordering::Greater => 0,
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Equal => 2,
    }
}

fn probe(mut rng: Mt19937GenRand64, count: usize) -> Vec<u64> {
    (0..count).map(|_| rng.r#gen()).collect()
}

/// Compare the engine with the naive v2 reference after every step: partition,
/// select RNG state, evaluation bits and counters. `indexed` selects the
/// expected fitness path and diagnostics formula. Returns the size states in
/// which selections were made.
fn assert_eo_matches_reference(
    g: &Graph,
    c: &Condition,
    seed: u64,
    steps: usize,
    registry: &FitnessRegistry,
    indexed: bool,
) -> [bool; 3] {
    let cancel = CancellationToken::new();
    let mut actual = Engine::new(g, c, seed, registry, &cancel).unwrap();
    let mut expected = eo_v2::ReferenceEo::new(g, c, seed, registry).unwrap();
    assert_eq!(actual.eo.as_ref().unwrap().is_indexed(), indexed);
    let n = g.node_count() as u64;
    let mut fitness_values = if indexed { n } else { 0 };
    let mut visited = [false; 3];
    for step in 0..=steps {
        let context = format!("{:?} seed {seed} step {step}", c.solver);
        assert_eq!(
            actual.state.partition(),
            expected.state.partition(),
            "partition: {context}"
        );
        assert!(
            actual.select_rng == expected.select_rng,
            "select RNG: {context}"
        );
        assert_eq!(
            actual.search_evaluation.to_bits(),
            g.score(expected.state.partition(), c.alpha).to_bits(),
            "search evaluation: {context}"
        );
        assert_eq!(actual.objective_evaluations, step as u64 + 1, "{context}");
        assert_eq!(actual.applied_moves, step as u64, "{context}");
        assert_eq!(actual.fitness_values, fitness_values, "{context}");
        actual
            .eo
            .as_ref()
            .unwrap()
            .assert_index_consistent(g, &actual.state, c.neighborhood);
        if step == steps {
            break;
        }
        visited[size_state_of(&actual.state)] = true;
        assert_eq!(actual.step(&cancel).unwrap(), StepStatus::Continue);
        fitness_values += match expected.step().unwrap() {
            _ if !indexed => n,
            Move::Flip(v) => 1 + g.degree(v) as u64,
            Move::Swap(a, b) => 2 + (g.degree(a) + g.degree(b)) as u64,
        };
    }
    // 624 outputs cover two MT19937-64 state blocks; EO never draws from the
    // tie or smoothing streams.
    assert_eq!(
        rng_probe(&actual, 624),
        (
            probe(expected.select_rng.clone(), 624),
            probe(expected.tie_rng.clone(), 624),
            probe(expected.smooth_rng.clone(), 624),
        )
    );
    visited
}

#[test]
fn eo_v2_builtin_index_matches_naive_reference_on_every_step() {
    let registry = FitnessRegistry::default_registry();
    let graphs = [graph(), eo_graph(), isolated_graph(), complete_graph()];
    for spec in builtin_fitness_specs() {
        for g in &graphs {
            // Majority-dependent flips keep three size-state rankings; count the
            // runs whose selections used all of them.
            let mut flip_runs = 0;
            let mut switching_runs = 0;
            for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
                for tau in EO_TAUS {
                    for seed in [0, 0x5eed, u64::MAX - 17] {
                        let c = eo_condition(g, neighborhood, tau, spec.clone());
                        let visited =
                            assert_eo_matches_reference(g, &c, seed, 400, &registry, true);
                        if neighborhood == Neighborhood::Flip {
                            flip_runs += 1;
                            switching_runs += usize::from(visited == [true; 3]);
                        } else {
                            assert_eq!(visited, [false, false, true], "swap stays balanced");
                        }
                    }
                }
            }
            assert!(
                2 * switching_runs >= flip_runs,
                "{spec:?}: only {switching_runs} of {flip_runs} flip runs switch between \
                 all size states"
            );
        }
    }
}

/// A caller-supplied definition that wraps a built-in one, forcing the sorted path.
struct SortedPathFactory(String);

impl crate::fitness::FitnessFactory for SortedPathFactory {
    fn version(&self) -> &str {
        crate::fitness::BUILTIN_FITNESSES
            .iter()
            .find(|(name, _)| *name == self.0)
            .unwrap()
            .1
    }

    fn validate(&self, params: &serde_json::Value) -> crate::error::Result<()> {
        FitnessRegistry::default().validate(&FitnessSpec {
            kind: self.0.clone(),
            params: params.clone(),
        })
    }

    fn create(
        &self,
        params: &serde_json::Value,
    ) -> crate::error::Result<Box<dyn crate::fitness::VertexFitness>> {
        FitnessRegistry::default().create(&FitnessSpec {
            kind: self.0.clone(),
            params: params.clone(),
        })
    }
}

/// Many ties, negative values and both signed zeros (`-0.0 == 0.0`).
struct TieHeavyFactory;
struct TieHeavyFitness;

impl crate::fitness::FitnessFactory for TieHeavyFactory {
    fn version(&self) -> &str {
        "tie-heavy-v1"
    }

    fn validate(&self, _: &serde_json::Value) -> crate::error::Result<()> {
        Ok(())
    }

    fn create(
        &self,
        _: &serde_json::Value,
    ) -> crate::error::Result<Box<dyn crate::fitness::VertexFitness>> {
        Ok(Box::new(TieHeavyFitness))
    }
}

impl crate::fitness::VertexFitness for TieHeavyFitness {
    fn values(&self, graph: &Graph, state: &PartitionState) -> crate::error::Result<Vec<f64>> {
        let side = state.partition();
        Ok((0..graph.node_count())
            .map(|v| {
                let same = graph
                    .neighbors(v)
                    .iter()
                    .filter(|&&u| side[u] == side[v])
                    .count();
                match (graph.degree(v) + same) % 4 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => -1.5,
                    _ => same as f64,
                }
            })
            .collect())
    }
}

#[test]
fn eo_v2_custom_fitness_uses_sorted_path_matching_reference() {
    let mut registry = FitnessRegistry::default_registry();
    for (name, _) in crate::fitness::BUILTIN_FITNESSES {
        registry.register(name, Arc::new(SortedPathFactory(name.into())));
    }
    registry.register("tie_heavy", Arc::new(TieHeavyFactory));
    let builtin = FitnessRegistry::default_registry();
    let mut specs = builtin_fitness_specs();
    specs.push(FitnessSpec {
        kind: "tie_heavy".into(),
        params: serde_json::json!({}),
    });
    for spec in specs {
        let wraps_builtin = spec.kind != "tie_heavy";
        for g in [eo_graph(), graph()] {
            for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
                for tau in EO_TAUS {
                    for seed in [7, u64::MAX] {
                        let c = eo_condition(&g, neighborhood, tau, spec.clone());
                        assert_eo_matches_reference(&g, &c, seed, 300, &registry, false);
                        if wraps_builtin {
                            // Same values and RNG streams: identical to the index path.
                            let cancel = CancellationToken::new();
                            let mut sorted = Engine::new(&g, &c, seed, &registry, &cancel).unwrap();
                            let mut indexed = Engine::new(&g, &c, seed, &builtin, &cancel).unwrap();
                            for _ in 0..300 {
                                sorted.step(&cancel).unwrap();
                                indexed.step(&cancel).unwrap();
                            }
                            assert_eq!(sorted.state.partition(), indexed.state.partition());
                            assert!(sorted.select_rng == indexed.select_rng);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn eo_v2_runner_reports_fitness_values_of_the_selected_path() {
    let g = eo_graph();
    let cancel = CancellationToken::new();
    let mut custom = FitnessRegistry::default_registry();
    custom.register("tie_heavy", Arc::new(TieHeavyFactory));
    let additive = FitnessSpec {
        kind: "additive".into(),
        params: serde_json::json!({ "beta": 3.0 }),
    };
    let tie_heavy = FitnessSpec {
        kind: "tie_heavy".into(),
        params: serde_json::json!({}),
    };
    for (neighborhood, spec, indexed) in [
        (Neighborhood::Flip, additive.clone(), true),
        (Neighborhood::Swap, additive, true),
        (Neighborhood::Flip, FitnessSpec::default(), true),
        (Neighborhood::Swap, tie_heavy, false),
    ] {
        let mut c = eo_condition(&g, neighborhood, 1.5, spec);
        c.budget.max_steps = 250;
        c.measurement.steps = vec![0, 17, 250];
        c.measurement.basin = BasinMode::None;
        c.measurement.diagnostics = true;
        let result = crate::experiment::runner::run_one(&g, &c, 99, &cancel, &custom).unwrap();
        let mut reference = eo_v2::ReferenceEo::new(&g, &c, 99, &custom).unwrap();
        let n = g.node_count() as u64;
        let mut count = if indexed { n } else { 0 };
        for _ in 0..250 {
            count += match reference.step().unwrap() {
                _ if !indexed => n,
                Move::Flip(v) => 1 + g.degree(v) as u64,
                Move::Swap(a, b) => 2 + (g.degree(a) + g.degree(b)) as u64,
            };
        }
        assert_eq!(
            result.partitions[result.final_solution.0],
            reference.state.partition()
        );
        let diagnostics = result.diagnostics.unwrap();
        assert_eq!(diagnostics.fitness_values_computed_search, Some(count));
        assert_eq!(diagnostics.applied_moves, 250);
        assert_eq!(diagnostics.objective_evaluations_search, 251);
    }
}

#[test]
fn sa_all_steps_match_frozen_engine() {
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        for temperature in [0.0, 1.0, 1.0e300] {
            for smoothing in [SmoothingSpec::None, SmoothingSpec::WeightedAverage { k: 0 }] {
                let c = condition(
                    neighborhood,
                    SolverSpec::Sa {
                        temperature,
                        smoothing,
                    },
                    -0.0,
                );
                assert_engine_exact(&c, 918_273, 40);
            }
        }
    }
}

#[test]
fn hc_and_smoothed_paths_match_frozen_engine() {
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        for smoothing in [
            SmoothingSpec::None,
            SmoothingSpec::AllAverage,
            SmoothingSpec::RandomKAverage { k: 3 },
            SmoothingSpec::WeightedAverage { k: 2 },
        ] {
            let hc = condition(
                neighborhood,
                SolverSpec::Hc {
                    smoothing: smoothing.clone(),
                },
                0.125,
            );
            assert_engine_exact(&hc, 1234, 12);
            let sa = condition(
                neighborhood,
                SolverSpec::Sa {
                    temperature: 1.0,
                    smoothing,
                },
                0.125,
            );
            assert_engine_exact(&sa, 4321, 12);
        }
    }
}

#[test]
fn optimized_default_fitness_matches_original_neighbor_scan_bits() {
    let g = graph();
    let registry = FitnessRegistry::default_registry();
    let fitness = registry.create(&FitnessSpec::default()).unwrap();
    for bits in 0u16..256 {
        let partition = (0..8).map(|v| bits & (1 << v) != 0).collect();
        let state = PartitionState::new(&g, partition).unwrap();
        let actual = fitness.values(&g, &state).unwrap();
        let expected: Vec<_> = (0..g.node_count())
            .map(|v| {
                let degree = g.degree(v);
                if degree == 0 {
                    1.0
                } else {
                    let good = g
                        .neighbors(v)
                        .iter()
                        .filter(|&&u| state.partition()[u] == state.partition()[v])
                        .count();
                    good as f64 / degree as f64
                }
            })
            .collect();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}

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

#[test]
fn complete_runner_output_matches_frozen_runner_with_measurements_and_basins() {
    let g = graph();
    let registry = FitnessRegistry::default_registry();
    let reference_registry = frozen_registry();
    let cancel = CancellationToken::new();
    let cases = [
        condition(
            Neighborhood::Flip,
            SolverSpec::Sa {
                temperature: 1.0,
                smoothing: SmoothingSpec::None,
            },
            0.05,
        ),
        condition(
            Neighborhood::Swap,
            SolverSpec::Sa {
                temperature: 1.0e300,
                smoothing: SmoothingSpec::WeightedAverage { k: 0 },
            },
            f64::MIN_POSITIVE,
        ),
        condition(
            Neighborhood::Flip,
            SolverSpec::Sa {
                temperature: 0.0,
                smoothing: SmoothingSpec::None,
            },
            1.0e300,
        ),
        condition(
            Neighborhood::Flip,
            SolverSpec::Hc {
                smoothing: SmoothingSpec::AllAverage,
            },
            0.05,
        ),
    ];
    for base in cases {
        for all_steps in [false, true] {
            for (diagnostics, basin) in [
                (false, BasinMode::None),
                (true, BasinMode::None),
                (false, BasinMode::Both),
                (true, BasinMode::Both),
            ] {
                let mut c = base.clone();
                c.measurement.diagnostics = diagnostics;
                c.measurement.basin = basin;
                if all_steps {
                    c.measurement.steps = (0..=c.budget.max_steps).collect();
                }
                let actual =
                    crate::experiment::runner::run_one(&g, &c, 7788, &cancel, &registry).unwrap();
                let expected =
                    reference::runner::run_one(&g, &c, 7788, &cancel, &reference_registry).unwrap();
                if all_steps && actual.completed_steps == c.budget.max_steps {
                    assert_eq!(actual.records.len(), c.budget.max_steps as usize + 1);
                }
                assert_eq!(exact_json(actual), exact_json(expected));
            }
        }
    }
}

/// Cancellation after an EO selection consumes exactly the selection draws
/// (1 for Flip, 2 for Swap) and changes neither the state nor the index, so
/// the interrupted engine continues like one that only skipped those draws.
#[test]
fn eo_cancellation_consumes_selection_draws_and_changes_nothing_else() {
    let g = graph();
    let registry = FitnessRegistry::default_registry();
    let live = CancellationToken::new();
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    for (neighborhood, draws) in [(Neighborhood::Flip, 1), (Neighborhood::Swap, 2)] {
        for fitness in [
            FitnessSpec::default(),
            FitnessSpec {
                kind: "additive".into(),
                params: serde_json::json!({ "beta": 3.0 }),
            },
        ] {
            let c = condition(neighborhood, SolverSpec::Eo { tau: 1.5, fitness }, 0.05);
            let mut interrupted = Engine::new(&g, &c, 5, &registry, &live).unwrap();
            let mut expected = Engine::new(&g, &c, 5, &registry, &live).unwrap();
            for _ in 0..7 {
                interrupted.step(&live).unwrap();
                expected.step(&live).unwrap();
            }
            let partition = interrupted.state.partition().to_vec();
            let (moves, values) = (interrupted.applied_moves, interrupted.fitness_values);
            assert!(interrupted.step(&cancelled).is_err(), "{c:?}");
            assert_eq!(interrupted.state.partition(), partition.as_slice());
            assert_eq!(interrupted.applied_moves, moves);
            assert_eq!(interrupted.fitness_values, values);
            for _ in 0..draws {
                let _: f64 = expected.select_rng.r#gen();
            }
            assert!(interrupted.select_rng == expected.select_rng, "{c:?}");
            for step in 0..50 {
                interrupted.step(&live).unwrap();
                expected.step(&live).unwrap();
                assert_eq!(
                    interrupted.state.partition(),
                    expected.state.partition(),
                    "step {step} after cancellation: {c:?}"
                );
            }
        }
    }
}

/// The Metropolis rule of real-objective SA through the memo of
/// `(-delta / t).exp()`: improvements and temperature zero draw nothing, every
/// other judgement draws exactly once, and the threshold has the bits of the
/// expression on misses, hits, signed zeros, keys sharing a memo slot and
/// underflow. The memo is fixed to the job's temperature, so each temperature
/// has its own engine.
#[test]
fn sa_metropolis_memo_preserves_rng_shortcuts_bits_and_collisions() {
    let g = graph();
    let cancel = CancellationToken::new();
    let registry = FitnessRegistry::default_registry();
    let conditions = [1.0, 0.0, f64::MIN_POSITIVE].map(|temperature| {
        condition(
            Neighborhood::Flip,
            SolverSpec::Sa {
                temperature,
                smoothing: SmoothingSpec::None,
            },
            0.05,
        )
    });
    let [warm, cold, tiny] = &conditions;
    let mut engine = Engine::new(&g, warm, 123, &registry, &cancel).unwrap();
    let untouched = engine.select_rng.clone();
    assert!(engine.sa_accept(-1.0, 1.0));
    assert!(engine.select_rng == untouched, "improvement must not draw");
    let mut frozen = Engine::new(&g, cold, 123, &registry, &cancel).unwrap();
    let untouched = frozen.select_rng.clone();
    assert!(!frozen.sa_accept(1.0, 0.0));
    assert!(
        frozen.select_rng == untouched,
        "temperature zero must not draw"
    );

    let check = |engine: &mut Engine<'_>, delta: f64, temperature: f64| {
        let mut expected_rng = engine.select_rng.clone();
        let draw: f64 = expected_rng.r#gen();
        let threshold = (-delta / temperature).exp();
        assert_eq!(engine.sa_accept(delta, temperature), draw < threshold);
        assert!(engine.select_rng == expected_rng, "Metropolis draw count");
        threshold
    };

    let mut minute = Engine::new(&g, tiny, 123, &registry, &cancel).unwrap();
    let threshold = check(&mut minute, f64::MIN_POSITIVE, f64::MIN_POSITIVE);
    assert_eq!(threshold.to_bits(), (-1.0f64).exp().to_bits());
    let underflow = check(&mut minute, f64::MAX, f64::MIN_POSITIVE);
    assert_eq!(underflow.to_bits(), 0.0f64.to_bits());

    for zero in [0.0, -0.0, 0.0] {
        check(&mut engine, zero, 1.0);
    }
    // Two integer deltas in one memo slot: replacing either key only turns
    // the next lookup into an exact miss.
    let slot = |x: f64| crate::solvers::metropolis::slot(x.to_bits());
    let first = 15.0f64;
    let second = (16..100_000u32)
        .map(f64::from)
        .find(|&x| slot(x) == slot(first))
        .expect("an integer delta sharing the slot of 15");
    for delta in [first, first, second, first, second, second] {
        check(&mut engine, delta, 1.0);
    }
}

// EO-SA (`eo_sa`): EO proposals judged by the Metropolis rule. Compared with an
// independent naive reference built on the `eo_v2` selection oracle.
#[path = "eo_sa_exact_tests.rs"]
mod eo_sa;

// Smoothing: production `crate::smoothing` compared with the frozen e4b6a1c
// copy directly and through the frozen engine and runner.
#[path = "smoothing_exact_tests.rs"]
mod smoothing_exact;

// Real-objective SA (Metropolis memo, cross-side swap score, specialized step
// loop) against the frozen engine and runner, and `Engine::advance` against
// the loop of `Engine::step` calls it replaces in the runner.
#[path = "sa_exact_tests.rs"]
mod sa_exact;

// Independent review: non-smoothed HC (none / weighted_average k = 0) against the
// frozen engine on graphs up to n = 100 and on alphas that bypass run_one
// validation, including the non-finite search-evaluation error.
#[test]
fn hc_real_paths_match_frozen_engine_including_errors() {
    let registry = FitnessRegistry::default_registry();
    let reference_registry = frozen_registry();
    let cancel = CancellationToken::new();
    let mut rng = Mt19937GenRand64::new(99);
    let mut graphs = vec![graph(), isolated_graph(), complete_graph(), eo_graph()];
    for (n, p) in [
        (2usize, 1.0),
        (3, 0.5),
        (20, 0.2),
        (40, 0.1),
        (64, 0.08),
        (100, 0.05),
    ] {
        let mut edges = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                if rng.r#gen::<f64>() < p {
                    edges.push([a, b]);
                }
            }
        }
        graphs.push(Graph::from_edges(n, edges).unwrap());
    }
    let (mut errors, mut optima, mut steps_total) = (0, 0, 0);
    for g in &graphs {
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            if neighborhood == Neighborhood::Swap && g.node_count() % 2 == 1 {
                continue;
            }
            for smoothing in [SmoothingSpec::None, SmoothingSpec::WeightedAverage { k: 0 }] {
                for alpha in [
                    0.0,
                    -0.0,
                    0.05,
                    0.125,
                    1.0,
                    1e17,
                    1e300,
                    1e306,
                    1e308,
                    f64::MAX,
                    -1.0,
                    -0.05,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NAN,
                    5e-324,
                ] {
                    for seed in [0u64, 1, 2] {
                        let c = condition_for(
                            g,
                            neighborhood,
                            SolverSpec::Hc {
                                smoothing: smoothing.clone(),
                            },
                            alpha,
                        );
                        let mut actual = Engine::new(g, &c, seed, &registry, &cancel).unwrap();
                        let mut expected =
                            reference::Engine::new(g, &c, seed, &reference_registry, &cancel)
                                .unwrap();
                        for step in 0..100_000 {
                            let ctx = format!(
                                "{neighborhood:?} {smoothing:?} a={alpha:e} seed={seed} n={} step={step}",
                                g.node_count()
                            );
                            let a = actual.step(&cancel);
                            let b = expected.step(&cancel);
                            assert_eq!(
                                actual.state.partition(),
                                expected.state.partition(),
                                "{ctx}"
                            );
                            assert_eq!(
                                actual.search_evaluation.to_bits(),
                                expected.search_evaluation.to_bits(),
                                "{ctx}"
                            );
                            assert_eq!(
                                actual.objective_evaluations, expected.objective_evaluations,
                                "{ctx}"
                            );
                            assert_eq!(actual.applied_moves, expected.applied_moves, "{ctx}");
                            assert_eq!(rng_probe(&actual, 3), expected.rng_probe(3), "{ctx}");
                            steps_total += 1;
                            match (a, b) {
                                (Ok(x), Ok(y)) => {
                                    assert_eq!(format!("{x:?}"), format!("{y:?}"), "{ctx}");
                                    if x != StepStatus::Continue {
                                        optima += 1;
                                        break;
                                    }
                                }
                                (Err(x), Err(y)) => {
                                    assert_eq!(x.to_string(), y.to_string(), "{ctx}");
                                    errors += 1;
                                    break;
                                }
                                (x, y) => panic!("{ctx}: {x:?} vs {y:?}"),
                            }
                        }
                        assert_eq!(rng_probe(&actual, 624), expected.rng_probe(624));
                    }
                }
            }
        }
    }
    assert!(errors > 0 && optima > 0 && steps_total > errors + optima);
}

// Independent review of the smoothing speedup: random differential tests against
// the frozen copy (n up to 3000, under catch_unwind), asynchronous mid-call
// cancellation followed by reuse, size sequences and parallel threads, checking
// that the reusable scratch permutation is always restored.
#[path = "smoothing_review_tests.rs"]
mod smoothing_review;

// Independent review of the SA step and partition-state speedup: HC and SA
// runners against the frozen runner at every stop position, cancellation from
// the observer of `Engine::advance`, memo collisions (including the slot of
// +0.0) and exhaustive adjacency tests around the matrix cap.
#[path = "sa_review_tests.rs"]
mod sa_review;
