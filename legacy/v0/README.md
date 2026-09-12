# legacy/v0 — the original saomsim (previous session)

The files GETTING_STARTED.md was written against, recovered after the rebuild
in `saomsim/` was already done and validated against RSiena. Kept for the
record; not importable as a package and not run by pytest.

Differences from the current package, for anyone reading both:

- `cycle3` statistic here is the actor sum (3x per cycle). RSiena's target
  counts each cycle once; the current package follows RSiena.
- "Conditional" simulation here runs a fixed number of ministeps. RSiena's
  conditional mode stops when the Hamming distance from the start network
  reaches the observed distance; the current package implements that rule.
- `theta` is `(p,)` only; the current package accepts per-chain `(B, K)`.
- `estimate_mom` is a Robbins-Monro single-panel estimator; the current
  package ports it as `estimate_rm` next to the multi-panel Gauss-Newton
  `estimate`.
- 61 tests (60 fast + 1 slow), all subsumed by the current suite.
