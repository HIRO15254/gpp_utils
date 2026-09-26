# Exact regression reference

`engine_51577f9.rs` and `runner_51577f9.rs` are copies of the corresponding
production sources from Git commit `51577f9`. The runner has one test-only
import rewrite so that it calls the frozen engine, and both fixtures call the
frozen smoothing module described in "Smoothing" below. Its original module-level
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

The `eo_sa` solver kind was added to `SolverSpec` after `51577f9`. Rust
requires exhaustive matches, so the frozen engine and runner carry one
compile-only arm per exhaustive `match` on `SolverSpec`:
`SolverSpec::EoSa { .. } => unreachable!("eo_sa postdates this frozen reference")`.
The arms add no branch for the pre-existing variants, and no test passes an
`eo_sa` condition to this oracle.

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

## EO-SA

`eo_sa_reference.rs` is the oracle for `eo_sa` (EO proposals judged by the
Metropolis rule). It is a naive executable specification written from the
EO-SA section of `docs/algorithms.md` and never calls `../engine.rs` or
`../eo.rs`. It proposes each move with the naive selection functions of
`eo_v2_reference.rs`, which the including module must have in scope as
`super::eo_v2`, scores the candidate by recomputing the objective of a copied
partition from the edge list, applies the literal short-circuit acceptance
expression with its own `accept` stream, rebuilds the partition state from
scratch after an accepted move and keeps the diagnostic counters by the
formulas of the specification. `propose` alone models a cancelled step.

`../eo_sa_exact_tests.rs` (a child of `../exact_tests.rs`) compares the
incremental built-in index and the sorted custom-fitness path with it after
every step: partitions, the complete select and accept RNG states, evaluation
bits against full recomputation, applied moves, objective evaluations and
fitness values, and the index consistency; at the end the untouched tie and
smoothing streams. The runs cover all built-in fitness definitions, four
graphs, Flip and Swap, `tau` from 0 to `1e308` and temperatures 0, 0.05, 0.5,
2 and `1e300`, and assert that every branch of the rule (improvement, tie,
accepted and rejected uphill draws, rejection without a draw at `T = 0`) is
taken. The same file checks the limits (`T = 1e300` follows EO given the same
select stream; `T = 0` and `-0.0` never draw and accept only strict
improvements), cancellation and the runner's incumbent and diagnostics.

## Smoothing

`smoothing_e4b6a1c.rs` is a copy of the non-test code of
`src/smoothing/mod.rs` at Git commit `e4b6a1c`, the smoothing that the frozen
engine and runner called until then. Since `51577f9` that module had changed
only by the `Graph` getter adapter described above. Its module-level
`#![allow(clippy::too_many_arguments)]` is attached to the adapter module
`smoothing_e4b6a1c` in `../exact_tests.rs`, because an inner attribute cannot
be retained inside `include!`; the copy is otherwise verbatim.

Production smoothing was then optimized without changing results: it no longer
builds the move list or copies candidate states, replays the partial shuffle
on a reused identity permutation with the same draws, and computes each score
from the same integer counts with the same floating-point expression. So that
the frozen fixtures do not share that code with production, each has one more
test-only import rewrite, with no other change:

- `engine_51577f9.rs`: `use crate::smoothing;` became
  `use super::smoothing_e4b6a1c as smoothing;`
- `runner_51577f9.rs`: `use crate::smoothing;` became
  `use super::super::smoothing_e4b6a1c as smoothing;`

The rewrite selects the code that the fixtures executed before, so their
behavior is unchanged. `../smoothing_exact_tests.rs` (a child of
`../exact_tests.rs`) compares production `smoothing::evaluate` directly with
this copy: the returned bits or error message, the evaluation count and the
complete RNG state afterwards, for Flip and Swap, every specification kind
with `k` in the distance-one and distance-two ranges up to the maximum,
tie-heavy, degenerate and random graphs up to 500 vertices, many states
(balanced and unbalanced sides, successive states of a search), alphas
including both zeros, `1e300`, the smallest normal value, infinity and NaN, a
missing RNG, a cancelled token and the panics of irregular Swap distance-two
inputs. It also compares the functions that other modules call unchanged and
runs the production engine and runner against the frozen ones for every
smoothing kind. Keep this copy frozen; a change of smoothing results needs a
new algorithm version, not an update of this oracle.
