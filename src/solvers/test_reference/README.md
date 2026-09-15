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
