//! Exact tests of the EO-SA solver (`eo_sa`); child module of `exact_tests`,
//! so they also run in the release exact regression of `scripts/check.py`.
//!
//! The production engine is compared after every step with
//! `test_reference/eo_sa_reference.rs`, a naive executable specification
//! written from `docs/algorithms.md` that reuses only the naive EO v2 selection
//! oracle (`super::eo_v2`) and shared infrastructure, never `../engine.rs` or
//! `../eo.rs`.

use super::*;
use crate::experiment::result::RunTermination;

// Resolves the EO v2 selection oracle as `super::eo_v2`, which the glob import
// above brings into this module.
mod eo_sa_reference {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/solvers/test_reference/eo_sa_reference.rs"
    ));
}

use eo_sa_reference::{Outcome, ReferenceEoSa, stream};

/// Zero, small, moderate and huge temperatures. `1e300` rounds every
/// acceptance probability `exp(-delta / T)` to 1.0.
const TEMPERATURES: [f64; 5] = [0.0, 0.05, 0.5, 2.0, 1.0e300];
const SEEDS: [u64; 3] = [0, 0x5eed, u64::MAX - 17];
const STEPS: usize = 150;

fn eo_sa_condition(
    g: &Graph,
    neighborhood: Neighborhood,
    tau: f64,
    temperature: f64,
    fitness: FitnessSpec,
) -> Condition {
    condition_for(
        g,
        neighborhood,
        SolverSpec::EoSa {
            tau,
            temperature,
            fitness,
        },
        0.05,
    )
}

fn temperature_of(c: &Condition) -> f64 {
    match c.solver {
        SolverSpec::EoSa { temperature, .. } => temperature,
        _ => unreachable!("an eo_sa condition"),
    }
}

fn tie_heavy() -> FitnessSpec {
    FitnessSpec {
        kind: "tie_heavy".into(),
        params: serde_json::json!({}),
    }
}

fn additive(beta: f64) -> FitnessSpec {
    FitnessSpec {
        kind: "additive".into(),
        params: serde_json::json!({ "beta": beta }),
    }
}

/// Failure context, formatted only when an assertion fails.
struct Context<'a> {
    condition: &'a Condition,
    seed: u64,
    step: usize,
}

impl std::fmt::Display for Context<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} {:?} on {} vertices, seed {} step {}",
            self.condition.neighborhood,
            self.condition.solver,
            self.condition.graph.node_count,
            self.seed,
            self.step
        )
    }
}

/// Compare the engine with the reference: partition, the complete select and
/// accept RNG states, evaluation bits, counters and the EO ranking index.
fn assert_same(
    actual: &Engine<'_>,
    expected: &ReferenceEoSa<'_>,
    g: &Graph,
    context: &Context<'_>,
) {
    let alpha = context.condition.alpha;
    assert_eq!(
        actual.state.partition(),
        expected.state.partition(),
        "partition: {context}"
    );
    assert!(
        actual.select_rng == expected.select_rng,
        "select RNG: {context}"
    );
    assert!(
        actual.accept_rng.as_ref() == Some(&expected.accept_rng),
        "accept RNG: {context}"
    );
    let real = g.score(actual.state.partition(), alpha).to_bits();
    assert_eq!(
        actual.search_evaluation.to_bits(),
        expected.current.to_bits(),
        "search evaluation: {context}"
    );
    assert_eq!(
        actual.search_evaluation.to_bits(),
        real,
        "search evaluation vs recomputation: {context}"
    );
    assert_eq!(
        actual.state.score(alpha).to_bits(),
        real,
        "incremental state vs recomputation: {context}"
    );
    assert_eq!(
        actual.applied_moves, expected.applied_moves,
        "applied moves: {context}"
    );
    assert_eq!(
        actual.objective_evaluations, expected.objective_evaluations,
        "objective evaluations: {context}"
    );
    assert_eq!(
        actual.fitness_values, expected.fitness_values,
        "fitness values: {context}"
    );
    actual
        .eo
        .as_ref()
        .expect("eo_sa ranks with EO")
        .assert_index_consistent(g, &actual.state, context.condition.neighborhood);
}

/// Branches of the Metropolis rule taken by the reference.
#[derive(Clone, Copy, Debug, Default)]
struct Tally {
    /// `delta < 0`: accepted without an accept draw.
    improvements: u64,
    /// `delta == 0` at a positive temperature: drawn and always accepted.
    ties: u64,
    /// `delta > 0` at a positive temperature: drawn, then accepted or rejected.
    uphill_accepted: u64,
    uphill_rejected: u64,
    /// `delta >= 0` at `T = 0`: rejected without an accept draw.
    frozen: u64,
    /// Size states (A larger, B larger, equal) in which moves were selected.
    visited: [bool; 3],
}

impl Tally {
    fn record(&mut self, outcome: Outcome, temperature: f64) {
        let Outcome {
            delta,
            drew,
            accepted,
            ..
        } = outcome;
        assert_eq!(drew, delta >= 0.0 && temperature > 0.0, "{outcome:?}");
        if delta < 0.0 {
            assert!(accepted, "{outcome:?}");
            self.improvements += 1;
        } else if !drew {
            assert!(!accepted, "{outcome:?}");
            self.frozen += 1;
        } else if delta == 0.0 {
            assert!(accepted, "a positive temperature accepts ties: {outcome:?}");
            self.ties += 1;
        } else if accepted {
            self.uphill_accepted += 1;
        } else {
            self.uphill_rejected += 1;
        }
    }

    fn add(&mut self, other: &Tally) {
        self.improvements += other.improvements;
        self.ties += other.ties;
        self.uphill_accepted += other.uphill_accepted;
        self.uphill_rejected += other.uphill_rejected;
        self.frozen += other.frozen;
        for (seen, other) in self.visited.iter_mut().zip(other.visited) {
            *seen |= other;
        }
    }
}

/// Run the engine and the naive reference side by side and compare them after
/// every step. `indexed` selects the expected fitness path and diagnostics
/// formula. Returns the branches of the Metropolis rule that were taken.
fn assert_eo_sa_matches_reference(
    g: &Graph,
    c: &Condition,
    seed: u64,
    steps: usize,
    registry: &FitnessRegistry,
    indexed: bool,
) -> Tally {
    let temperature = temperature_of(c);
    let cancel = CancellationToken::new();
    let mut actual = Engine::new(g, c, seed, registry, &cancel).unwrap();
    let mut expected = ReferenceEoSa::new(g, c, seed, registry, indexed).unwrap();
    assert_eq!(actual.eo.as_ref().unwrap().is_indexed(), indexed, "{c:?}");
    let mut tally = Tally::default();
    for step in 0..=steps {
        let context = Context {
            condition: c,
            seed,
            step,
        };
        assert_same(&actual, &expected, g, &context);
        assert_eq!(expected.objective_evaluations, step as u64 + 1);
        if step == steps {
            break;
        }
        tally.visited[size_state_of(&actual.state)] = true;
        assert_eq!(
            actual.step(&cancel).unwrap(),
            StepStatus::Continue,
            "{context}"
        );
        tally.record(expected.step().unwrap(), temperature);
    }
    // 624 outputs cover two MT19937-64 state blocks. eo_sa never draws from
    // the tie or smoothing streams, which the reference holds undrawn.
    assert_eq!(
        rng_probe(&actual, 624),
        (
            probe(expected.select_rng.clone(), 624),
            probe(expected.tie_rng.clone(), 624),
            probe(expected.smooth_rng.clone(), 624),
        ),
        "{c:?} seed {seed}"
    );
    assert_eq!(
        probe(actual.accept_rng.clone().unwrap(), 624),
        probe(expected.accept_rng.clone(), 624),
        "{c:?} seed {seed}"
    );
    tally
}

/// Every branch of the rule must be exercised: `tallies[k]` belongs to
/// `TEMPERATURES[k]`.
fn assert_all_branches_taken(what: &str, tallies: &[Tally; TEMPERATURES.len()]) {
    for (t, &temperature) in tallies.iter().zip(&TEMPERATURES) {
        assert!(t.improvements > 0, "{what}, T = {temperature}: {t:?}");
        if temperature > 0.0 {
            assert!(
                t.ties > 0 && t.uphill_accepted > 0,
                "{what}, T = {temperature}: {t:?}"
            );
        }
        if temperature > 0.0 && temperature < 1.0e300 {
            assert!(t.uphill_rejected > 0, "{what}, T = {temperature}: {t:?}");
        }
    }
}

/// Branches that each temperature allows: no accept draw at `T = 0`, every
/// draw accepted at `T = 1e300`.
fn assert_branches_allowed(what: &str, tallies: &[Tally; TEMPERATURES.len()]) {
    for (t, &temperature) in tallies.iter().zip(&TEMPERATURES) {
        let drawn = t.ties + t.uphill_accepted + t.uphill_rejected;
        if temperature == 0.0 {
            assert!(t.frozen > 0 && drawn == 0, "{what}, T = 0: {t:?}");
        } else {
            assert!(
                t.frozen == 0 && drawn > 0,
                "{what}, T = {temperature}: {t:?}"
            );
        }
        if temperature == 1.0e300 {
            assert_eq!(t.uphill_rejected, 0, "{what}, T = 1e300: {t:?}");
        }
    }
}

/// (a) Built-in fitness definitions through the incremental index.
#[test]
fn eo_sa_builtin_index_matches_naive_reference_on_every_step() {
    let registry = FitnessRegistry::default_registry();
    let graphs = [
        ("graph", graph()),
        ("eo_graph", eo_graph()),
        ("isolated_graph", isolated_graph()),
        ("complete_graph", complete_graph()),
    ];
    let mut tallies = [[Tally::default(); TEMPERATURES.len()]; 4];
    // Majority-dependent flips keep three size-state rankings; count the runs
    // whose selections used all of them. Low temperatures on the complete
    // graph drive flips to an extreme imbalance, so only the total is bounded.
    let (mut all_flip_runs, mut all_switching_runs) = (0, 0);
    for spec in builtin_fitness_specs() {
        for (gi, (name, g)) in graphs.iter().enumerate() {
            let (mut flip_runs, mut switching_runs) = (0, 0);
            for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
                for (ti, &tau) in EO_TAUS.iter().enumerate() {
                    for (ki, &temperature) in TEMPERATURES.iter().enumerate() {
                        let seed = SEEDS[(ti + ki) % SEEDS.len()];
                        let c = eo_sa_condition(g, neighborhood, tau, temperature, spec.clone());
                        let tally =
                            assert_eo_sa_matches_reference(g, &c, seed, STEPS, &registry, true);
                        tallies[gi][ki].add(&tally);
                        if neighborhood == Neighborhood::Flip {
                            flip_runs += 1;
                            switching_runs += usize::from(tally.visited == [true; 3]);
                        } else {
                            assert_eq!(tally.visited, [false, false, true], "swap stays balanced");
                        }
                    }
                }
            }
            assert!(
                switching_runs > 0,
                "{spec:?} on {name}: no flip run switches between all size states"
            );
            all_flip_runs += flip_runs;
            all_switching_runs += switching_runs;
        }
    }
    assert!(
        2 * all_switching_runs >= all_flip_runs,
        "only {all_switching_runs} of {all_flip_runs} flip runs switch between all size states"
    );
    for ((name, _), tallies) in graphs.iter().zip(&tallies) {
        assert_branches_allowed(name, tallies);
    }
    // The graphs with nontrivial cuts take every branch at every temperature.
    for ((name, _), tallies) in graphs.iter().zip(&tallies).take(2) {
        assert_all_branches_taken(name, tallies);
    }
}

/// (b) Caller-supplied definitions through the sorted path.
#[test]
fn eo_sa_custom_fitness_uses_sorted_path_matching_reference() {
    let mut registry = FitnessRegistry::default_registry();
    for (name, _) in crate::fitness::BUILTIN_FITNESSES {
        registry.register(name, Arc::new(SortedPathFactory(name.into())));
    }
    registry.register("tie_heavy", Arc::new(TieHeavyFactory));
    let builtin = FitnessRegistry::default_registry();
    let mut specs = builtin_fitness_specs();
    specs.push(tie_heavy());
    let mut tallies = [Tally::default(); TEMPERATURES.len()];
    for spec in specs {
        let wraps_builtin = spec.kind != "tie_heavy";
        for g in [eo_graph(), graph()] {
            for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
                for (ti, &tau) in EO_TAUS.iter().enumerate() {
                    for (ki, &temperature) in TEMPERATURES.iter().enumerate() {
                        let seed = [7, u64::MAX][(ti + ki) % 2];
                        let c = eo_sa_condition(&g, neighborhood, tau, temperature, spec.clone());
                        let tally =
                            assert_eo_sa_matches_reference(&g, &c, seed, STEPS, &registry, false);
                        tallies[ki].add(&tally);
                        if wraps_builtin {
                            // Same values and RNG streams as the built-in index:
                            // the same run, counting n fitness values per step.
                            let cancel = CancellationToken::new();
                            let mut sorted = Engine::new(&g, &c, seed, &registry, &cancel).unwrap();
                            let mut indexed = Engine::new(&g, &c, seed, &builtin, &cancel).unwrap();
                            for _ in 0..STEPS {
                                sorted.step(&cancel).unwrap();
                                indexed.step(&cancel).unwrap();
                            }
                            assert_eq!(sorted.state.partition(), indexed.state.partition());
                            assert!(sorted.select_rng == indexed.select_rng, "{c:?}");
                            assert!(sorted.accept_rng == indexed.accept_rng, "{c:?}");
                            assert_eq!(sorted.applied_moves, indexed.applied_moves);
                            assert_eq!(
                                sorted.search_evaluation.to_bits(),
                                indexed.search_evaluation.to_bits()
                            );
                        }
                    }
                }
            }
        }
    }
    assert_branches_allowed("custom fitness", &tallies);
    assert_all_branches_taken("custom fitness", &tallies);
}

/// (c) With `T = 1e300` every move is accepted, so given the same selection
/// stream eo_sa follows EO exactly, drawing once from the accept stream for
/// every move with `delta >= 0`.
#[test]
fn eo_sa_with_huge_temperature_follows_eo_given_the_same_selection_stream() {
    let mut registry = FitnessRegistry::default_registry();
    registry.register("tie_heavy", Arc::new(TieHeavyFactory));
    let specs = [
        FitnessSpec::default(),
        FitnessSpec {
            kind: "multiplicative".into(),
            params: serde_json::json!({ "alpha": 0.5 }),
        },
        additive(0.0),
        additive(3.0),
        tie_heavy(),
    ];
    let cancel = CancellationToken::new();
    let (mut drawn, mut improvements) = (0, 0);
    for spec in specs {
        for g in [graph(), eo_graph(), isolated_graph(), complete_graph()] {
            for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
                for (ti, &tau) in EO_TAUS.iter().enumerate() {
                    let seed = SEEDS[ti % SEEDS.len()];
                    let eo_c = eo_condition(&g, neighborhood, tau, spec.clone());
                    let c = eo_sa_condition(&g, neighborhood, tau, 1.0e300, spec.clone());
                    let mut eo = Engine::new(&g, &eo_c, seed, &registry, &cancel).unwrap();
                    let mut eo_sa = Engine::new(&g, &c, seed, &registry, &cancel).unwrap();
                    assert_eq!(eo_sa.state.partition(), eo.state.partition(), "{c:?}");
                    assert_eq!(eo_sa.fitness_values, eo.fitness_values, "{c:?}");
                    assert!(
                        eo_sa.select_rng != eo.select_rng,
                        "the solver JSON separates the streams: {c:?}"
                    );
                    eo_sa.select_rng = eo.select_rng.clone();
                    let mut accept = stream(&g, &c, seed, &registry, b"accept").unwrap();
                    for step in 0..STEPS {
                        let context = Context {
                            condition: &c,
                            seed,
                            step,
                        };
                        let before = eo_sa.search_evaluation;
                        assert_eq!(eo.step(&cancel).unwrap(), StepStatus::Continue);
                        assert_eq!(eo_sa.step(&cancel).unwrap(), StepStatus::Continue);
                        assert_eq!(eo_sa.state.partition(), eo.state.partition(), "{context}");
                        assert!(eo_sa.select_rng == eo.select_rng, "{context}");
                        assert_eq!(eo_sa.fitness_values, eo.fitness_values, "{context}");
                        assert_eq!(
                            eo_sa.search_evaluation.to_bits(),
                            eo.search_evaluation.to_bits(),
                            "{context}"
                        );
                        assert_eq!(eo_sa.applied_moves, step as u64 + 1, "{context}");
                        assert_eq!(eo.applied_moves, step as u64 + 1, "{context}");
                        assert_eq!(
                            eo_sa.objective_evaluations, eo.objective_evaluations,
                            "{context}"
                        );
                        if eo_sa.search_evaluation - before >= 0.0 {
                            let _: f64 = accept.r#gen();
                            drawn += 1;
                        } else {
                            improvements += 1;
                        }
                        assert!(eo_sa.accept_rng.as_ref() == Some(&accept), "{context}");
                    }
                }
            }
        }
    }
    assert!(drawn > 0 && improvements > 0, "{drawn} {improvements}");
}

/// (d) `T = 0` (and `-0.0`, which is not positive either) accepts only strict
/// improvements and never draws from the accept stream.
#[test]
fn eo_sa_at_zero_temperature_accepts_only_strict_improvements_without_accept_draws() {
    let registry = FitnessRegistry::default_registry();
    let cancel = CancellationToken::new();
    let specs = [
        FitnessSpec::default(),
        FitnessSpec {
            kind: "multiplicative".into(),
            params: serde_json::json!({ "alpha": 0.0 }),
        },
        additive(3.0),
    ];
    let (mut accepted, mut rejected) = (0, 0);
    for temperature in [0.0, -0.0] {
        for spec in &specs {
            for g in [graph(), eo_graph(), isolated_graph(), complete_graph()] {
                for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
                    for (ti, &tau) in EO_TAUS.iter().enumerate() {
                        let seed = SEEDS[ti % SEEDS.len()];
                        let c = eo_sa_condition(&g, neighborhood, tau, temperature, spec.clone());
                        let fresh = |label: &[u8]| stream(&g, &c, seed, &registry, label).unwrap();
                        let mut engine = Engine::new(&g, &c, seed, &registry, &cancel).unwrap();
                        for step in 0..STEPS {
                            let context = Context {
                                condition: &c,
                                seed,
                                step,
                            };
                            let before = engine.search_evaluation;
                            let partition = engine.state.partition().to_vec();
                            let moves = engine.applied_moves;
                            engine.step(&cancel).unwrap();
                            if engine.applied_moves == moves {
                                assert_eq!(
                                    engine.search_evaluation.to_bits(),
                                    before.to_bits(),
                                    "{context}"
                                );
                                assert_eq!(engine.state.partition(), partition, "{context}");
                                rejected += 1;
                            } else {
                                assert_eq!(engine.applied_moves, moves + 1, "{context}");
                                assert!(engine.search_evaluation < before, "{context}");
                                accepted += 1;
                            }
                        }
                        assert!(
                            engine.accept_rng.as_ref() == Some(&fresh(b"accept")),
                            "the accept stream is never drawn: {c:?}"
                        );
                        assert!(engine.tie_rng == fresh(b"tie"), "{c:?}");
                        assert!(engine.smooth_rng == fresh(b"smooth"), "{c:?}");
                    }
                }
            }
        }
    }
    assert!(accepted > 0 && rejected > 0, "{accepted} {rejected}");
    // `-0.0` derives its own streams from the solver JSON ("-0.0"), which the
    // reference reproduces, and never draws either.
    for spec in &specs {
        for g in [graph(), eo_graph()] {
            for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
                let c = eo_sa_condition(&g, neighborhood, 1.5, -0.0, spec.clone());
                let t = assert_eo_sa_matches_reference(&g, &c, 3, STEPS, &registry, true);
                let drawn = t.ties + t.uphill_accepted + t.uphill_rejected;
                assert!(t.improvements > 0 && t.frozen > 0 && drawn == 0, "{t:?}");
            }
        }
    }
}

/// (e) A cancelled step consumes exactly the selection draws (1 for Flip, 2
/// for Swap) and, for a caller-supplied fitness, the n values evaluated for
/// the selection; it changes neither the state, the index, the other counters
/// nor the accept stream. The engine then continues like one that only
/// skipped those draws, and like the reference after a proposal without
/// judgement.
#[test]
fn eo_sa_cancellation_consumes_selection_draws_and_changes_nothing_else() {
    let mut registry = FitnessRegistry::default_registry();
    registry.register("tie_heavy", Arc::new(TieHeavyFactory));
    let live = CancellationToken::new();
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    for g in [graph(), eo_graph()] {
        for (neighborhood, draws) in [(Neighborhood::Flip, 1), (Neighborhood::Swap, 2)] {
            for (fitness, indexed) in [
                (FitnessSpec::default(), true),
                (additive(3.0), true),
                (tie_heavy(), false),
            ] {
                for temperature in [0.0, 0.5, 1.0e300] {
                    let c = eo_sa_condition(&g, neighborhood, 1.5, temperature, fitness.clone());
                    let mut interrupted = Engine::new(&g, &c, 5, &registry, &live).unwrap();
                    let mut expected = Engine::new(&g, &c, 5, &registry, &live).unwrap();
                    let mut reference = ReferenceEoSa::new(&g, &c, 5, &registry, indexed).unwrap();
                    for _ in 0..7 {
                        interrupted.step(&live).unwrap();
                        expected.step(&live).unwrap();
                        reference.step().unwrap();
                    }
                    let partition = interrupted.state.partition().to_vec();
                    let evaluation = interrupted.search_evaluation.to_bits();
                    let (moves, evaluations, values) = (
                        interrupted.applied_moves,
                        interrupted.objective_evaluations,
                        interrupted.fitness_values,
                    );
                    let (accept, tie, smooth) = (
                        interrupted.accept_rng.clone(),
                        interrupted.tie_rng.clone(),
                        interrupted.smooth_rng.clone(),
                    );
                    assert!(interrupted.step(&cancelled).is_err(), "{c:?}");
                    assert_eq!(interrupted.state.partition(), partition, "{c:?}");
                    assert_eq!(interrupted.search_evaluation.to_bits(), evaluation);
                    assert_eq!(interrupted.applied_moves, moves, "{c:?}");
                    assert_eq!(interrupted.objective_evaluations, evaluations, "{c:?}");
                    // EO-SA rules 1-2 (as in EO): a caller-supplied fitness (sorted
                    // path) counts the n values it evaluated for the selection,
                    // which precedes the cancellation check; the built-in index
                    // evaluates nothing to select.
                    let selection_values = if indexed { 0 } else { g.node_count() as u64 };
                    assert_eq!(
                        interrupted.fitness_values,
                        values + selection_values,
                        "fitness values of a cancelled step: {c:?}"
                    );
                    assert!(interrupted.accept_rng == accept, "{c:?}");
                    assert!(interrupted.tie_rng == tie && interrupted.smooth_rng == smooth);
                    interrupted.eo.as_ref().unwrap().assert_index_consistent(
                        &g,
                        &interrupted.state,
                        neighborhood,
                    );
                    for _ in 0..draws {
                        let _: f64 = expected.select_rng.r#gen();
                    }
                    assert!(interrupted.select_rng == expected.select_rng, "{c:?}");
                    // The reference's cancelled step: the proposal without judgement.
                    reference.propose().unwrap();
                    for step in 0..50 {
                        let context = Context {
                            condition: &c,
                            seed: 5,
                            step,
                        };
                        assert_same(&interrupted, &reference, &g, &context);
                        assert_eq!(
                            interrupted.state.partition(),
                            expected.state.partition(),
                            "after cancellation: {context}"
                        );
                        assert!(interrupted.select_rng == expected.select_rng, "{context}");
                        assert!(interrupted.accept_rng == expected.accept_rng, "{context}");
                        assert_eq!(
                            (
                                interrupted.applied_moves,
                                interrupted.objective_evaluations,
                                interrupted.fitness_values,
                            ),
                            (
                                expected.applied_moves,
                                expected.objective_evaluations,
                                expected.fitness_values + selection_values,
                            ),
                            "{context}"
                        );
                        interrupted.step(&live).unwrap();
                        expected.step(&live).unwrap();
                        reference.step().unwrap();
                    }
                }
            }
        }
    }
}

/// (f) The runner reports the reference trajectory: final and incumbent
/// solutions (the first minimum of the real scores), measurement records and
/// diagnostics, with basins in real space only.
#[test]
fn eo_sa_runner_reports_reference_trajectory_and_counters() {
    let g = eo_graph();
    let cancel = CancellationToken::new();
    let mut registry = FitnessRegistry::default_registry();
    registry.register("tie_heavy", Arc::new(TieHeavyFactory));
    let steps = 250usize;
    let (mut later_ties, mut worse_final) = (false, false);
    for (neighborhood, spec, indexed, tau, temperature) in [
        (Neighborhood::Flip, FitnessSpec::default(), true, 1.5, 0.5),
        (Neighborhood::Swap, additive(3.0), true, 1.0, 2.0),
        (Neighborhood::Flip, additive(0.5), true, 0.0, 0.05),
        (Neighborhood::Swap, FitnessSpec::default(), true, 1.5, 0.0),
        (Neighborhood::Swap, tie_heavy(), false, 1.5, 0.5),
        (Neighborhood::Flip, tie_heavy(), false, 3.0, 1.0e300),
    ] {
        let mut c = eo_sa_condition(&g, neighborhood, tau, temperature, spec);
        c.budget.max_steps = steps as u64;
        c.measurement.steps = vec![0, 1, 17, 100, 250];
        c.measurement.basin = BasinMode::Both;
        c.measurement.diagnostics = true;
        let result = crate::experiment::runner::run_one(&g, &c, 99, &cancel, &registry).unwrap();
        result.validate(&g, &c).unwrap();

        let mut engine = Engine::new(&g, &c, 99, &registry, &cancel).unwrap();
        let mut reference = ReferenceEoSa::new(&g, &c, 99, &registry, indexed).unwrap();
        let mut trajectory = vec![reference.state.partition().to_vec()];
        for _ in 0..steps {
            engine.step(&cancel).unwrap();
            reference.step().unwrap();
            trajectory.push(reference.state.partition().to_vec());
        }
        let scores: Vec<f64> = trajectory.iter().map(|p| g.score(p, c.alpha)).collect();
        // Index of the first minimum of the real scores up to `step`.
        let first_best = |step: usize| {
            (0..=step).fold(0, |best, s| if scores[s] < scores[best] { s } else { best })
        };
        let best = first_best(steps);
        assert_eq!(result.termination, RunTermination::StepLimit, "{c:?}");
        assert_eq!(result.completed_steps, steps as u64, "{c:?}");
        assert_eq!(result.best_step, best as u64, "{c:?}");
        assert_eq!(result.partitions[result.best_solution.0], trajectory[best]);
        assert_eq!(
            result.partitions[result.final_solution.0],
            trajectory[steps]
        );
        later_ties |= scores[best + 1..].contains(&scores[best]);
        worse_final |= scores[steps] > scores[best];

        let steps_recorded: Vec<u64> = result.records.iter().map(|r| r.step).collect();
        assert_eq!(steps_recorded, c.measurement.steps, "{c:?}");
        for record in &result.records {
            let step = record.step as usize;
            assert_eq!(
                result.partitions[record.current_solution.0],
                trajectory[step]
            );
            assert_eq!(
                result.partitions[record.best_solution.0],
                trajectory[first_best(step)]
            );
            assert!(record.current_smoothed.is_none() && record.search_evaluation.is_none());
            assert!(record.basin_real.is_some() && record.basin_smoothed.is_none());
        }

        let diagnostics = result.diagnostics.unwrap();
        assert_eq!(
            diagnostics.fitness_values_computed_search,
            Some(engine.fitness_values)
        );
        assert_eq!(engine.fitness_values, reference.fitness_values, "{c:?}");
        assert_eq!(diagnostics.applied_moves, reference.applied_moves, "{c:?}");
        assert_eq!(diagnostics.objective_evaluations_search, steps as u64 + 1);
    }
    // Ties with the incumbent keep the first one; later worse states do not
    // replace it.
    assert!(later_ties && worse_final);
}
