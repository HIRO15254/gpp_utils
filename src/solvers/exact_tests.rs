use super::*;
use crate::experiment::config::{
    BasinMode, Budget, FitnessSpec, GraphKind, GraphSpec, Measurement, Schedule,
};
use rand::Rng;
use std::sync::Arc;

// Executable oracle: byte-for-byte production engine at commit 51577f9.
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
    }

    #[allow(clippy::too_many_arguments, clippy::collapsible_if)]
    pub mod runner {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/solvers/test_reference/runner_51577f9.rs"
        ));
    }
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
            node_count: graph.node_count,
            expected_degree: if graph.node_count > 1 {
                2.0 * graph.edges.len() as f64 / graph.node_count as f64
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
        Ok((0..graph.node_count)
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

#[test]
fn eo_ties_cross_mt_state_boundaries_exactly() {
    let isolated = Graph::from_edges(8, vec![]).unwrap();
    let complete = Graph::from_edges(
        8,
        (0..8)
            .flat_map(|a| (a + 1..8).map(move |b| [a, b]))
            .collect(),
    )
    .unwrap();
    for g in [&isolated, &complete] {
        for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
            for seed in [0, u64::MAX - 17] {
                let c = condition_for(
                    g,
                    neighborhood,
                    SolverSpec::Eo {
                        tau: 1.5,
                        fitness: FitnessSpec::default(),
                    },
                    0.05,
                );
                assert_engine_exact_on(g, &c, seed, 340);
            }
        }
    }
}

#[test]
fn eo_all_steps_match_frozen_engine() {
    for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
        for tau in [1.0e-300, 0.5, 1.5, 1.0e308] {
            let c = condition(
                neighborhood,
                SolverSpec::Eo {
                    tau,
                    fitness: FitnessSpec::default(),
                },
                0.05,
            );
            assert_engine_exact(&c, 0x0515_77f9, 40);
        }
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
        let expected: Vec<_> = (0..g.node_count)
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
            Neighborhood::Swap,
            SolverSpec::Eo {
                tau: 1.5,
                fitness: FitnessSpec::default(),
            },
            0.05,
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
