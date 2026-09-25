# Exact regression reference

`engine_51577f9.rs` and `runner_51577f9.rs` are copies of the corresponding
production sources from Git commit `51577f9`. The runner has one test-only
import rewrite so that it calls the frozen engine. Its original module-level
Clippy allowances are attached to the adapter module in `../exact_tests.rs`,
because an inner attribute cannot be retained inside `include!`. The fixtures
are compiled only by the solver's unit-test module and form an executable
behavioral oracle.

The additive `MeasurementRecord.basin_best` field is initialized to `None` in
the frozen runner's result adapter. This compatibility-only field addition
does not change the reference algorithm or the pre-existing output fields.

Keep this fixture frozen. Changes to the production engine must be reconciled
by changing the adapters/assertions in `../exact_tests.rs`, not this file.

The 0.2 Rust API encapsulates `Graph` fields. The only topology-access adapter
inside these fixtures replaces `graph.node_count` with `graph.node_count()`.
It performs the same field read; no loop, arithmetic, RNG operation or
algorithm branch changes. Future semantic changes must not update this oracle.

## EO since algorithm v2

EO changed intentionally in algorithm `v2`: ties are no longer shuffled with
the tie RNG and stably sorted every step. Vertices are ranked in the canonical
order `(fitness, side, vertex)` and every block of equal fitness shares its
averaged power-law weight; Swap selects its second vertex from the exact
conditional distribution over the opposite side. The frozen `51577f9` fixtures
therefore remain the oracle for SA, HC, smoothing, basins and the runner only.
Their EO code is still compiled but is no longer compared with production.

`eo_v2_reference.rs` is the EO oracle instead. It is an independent, naive
executable specification: every step it evaluates the fitness through
`VertexFitness::values`, sorts all vertices into the canonical order, rebuilds
the cumulative weights and applies the selection rule literally with linear
scans. It shares only the graph, partition state and RNG derivation with
production and never calls the production EO code in `../eo.rs`.
`../exact_tests.rs` compares both the incremental built-in index and the
sorted custom-fitness path with it after every step: partitions, the complete
select RNG state, the untouched tie and smoothing streams, evaluation bits and
diagnostic counters, for all built-in fitness definitions, Flip and Swap and
`tau` from 0 to `1e308`. Change this reference only together with a new
`algorithm` version.
