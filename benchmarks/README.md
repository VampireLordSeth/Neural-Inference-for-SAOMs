# benchmarks

The RSiena comparison (GETTING_STARTED §5) lives here. Nothing is in place yet.

Planned files:

- `rsiena_compare.R` — loads `s501`/`s502`, builds the Siena data object,
  writes the target statistics for `density`, `recip`, `transTrip`, `cycle3`
  and a covariate effect to `rsiena_targets.json`
- `rsiena_targets.json` — committed output of the above
- `test_rsiena_parity.py` — pytest that loads the same networks into `saomsim`
  and compares `statistics()` to the JSON; runs without R
