# Export compatibility oracle

`export_2aae96a.rs` is the exporter at commit `2aae96a`, with only read-only
Graph getter adapters. It is compiled by `tests/export_compatibility.rs` and
does not call the replacement RunView/column rendering implementation.

Compare TSV bytes (including float formatting, order and blanks) against this
oracle when changing export code. Metadata names, types and units must match;
human-readable meanings may be improved without changing stored science.
Do not update the oracle to make a regression pass.
