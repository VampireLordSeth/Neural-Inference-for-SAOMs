# benchmarks — RSiena parity

The M0 gate. Two comparisons against RSiena 1.6.6 (R 4.5.2, Windows), both
committed as data so the pytest side runs without R.

| script (R) | output | test (Python) | checks |
|---|---|---|---|
| `rsiena_compare.R` | `s501.csv`, `s502.csv`, `s50_covariates.csv`, `rsiena_targets.json` | `test_rsiena_parity.py` | target statistics of s502, exact equality |
| `rsiena_simulate.R` | `rsiena_sims.csv`, `rsiena_sims.json` | `test_rsiena_dynamics.py` | distribution of simulated statistics from s501 at fixed θ |

Regenerate with `Rscript benchmarks/rsiena_compare.R` and
`Rscript benchmarks/rsiena_simulate.R` from the project root (needs `RSiena`,
`jsonlite`). On this machine `Rscript` is at
`C:\Program Files\R\R-4.5.2\bin\Rscript.exe`.

## Result

**Target statistics** (s501 → s502, n = 50, 2026-09-11):

| statistic | RSiena | saomsim (first run) | resolution |
|---|---|---|---|
| Rate (tie changes) | 115 | 115 | — |
| density | 116 | 116 | — |
| recip | 70 | 70 | actor-summed: each mutual dyad counts twice in **both** |
| transTrip | 88 | 88 | `Σ_{j,h} x_ij x_ih x_hj`, both two-path terms in the change statistic |
| cycle3 | 28 | **84** | RSiena's target counts each 3-cycle once; actor sum counts it three times. **Fixed:** `statistics()` now divides by 3. Change statistic (`Σ_h x_jh x_hi`) unchanged — verified by the dynamics benchmark below. |
| altX(alc) | −4.08 | −4.08 | requires the **centred** covariate (`coCovar` centres by default); raw values do not match |
| egoX(alc) | 2.92 | 2.92 | same |
| sameX(smk) | 83 | 83 | centring-invariant |

Confirmed independently on a 6-node toy network with two 3-cycles and one
mutual dyad: RSiena reports `cycle3 = 2`, `recip = 2`.

**Simulated distributions** (start s501; θ = rate 5.0, density −2.3, recip 2.2,
transTrip 0.45, cycle3 −0.25, altX 0.10, egoX 0.05, sameX 0.40; RSiena
`cond = FALSE, simOnly = TRUE, n3 = 2000` vs saomsim B = 4000):

| statistic | R mean | saomsim mean | z | R sd | saomsim sd | ratio |
|---|---|---|---|---|---|---|
| Rate | 115.97 | 115.92 | 0.15 | 10.40 | 10.36 | 0.996 |
| density | 158.74 | 159.05 | −0.87 | 13.51 | 13.20 | 0.978 |
| recip | 94.69 | 95.12 | −1.34 | 11.77 | 11.42 | 0.970 |
| transTrip | 99.44 | 99.68 | −0.29 | 30.14 | 28.88 | 0.958 |
| cycle3 | 28.80 | 28.92 | −0.45 | 9.41 | 9.13 | 0.970 |
| altX(alc) | 20.99 | 20.81 | 0.51 | 13.22 | 12.93 | 0.978 |
| egoX(alc) | 12.05 | 12.40 | −1.03 | 12.64 | 12.49 | 0.988 |
| sameX(smk) | 109.81 | 110.10 | −0.96 | 10.89 | 10.83 | 0.994 |

Every mean agrees within Monte Carlo error and every spread within 5 %. This is
the evidence that the *ministep process* — Poisson(n·rate) step count, uniform
actor choice, multinomial logit over the n options including no-change, and
the change statistics as used inside the objective — matches RSiena's
unconditional simulation, not just that the summary statistics are defined the
same way.

## What is not covered

- Conditional simulation (`cond = TRUE`, run until the observed number of
  changes) — saomsim has no equivalent and does not need one for amortized
  inference.
- Endowment/creation effects, rate covariates, structural zeros, missing data,
  composition change, behaviour co-evolution. None are implemented in saomsim.
- Effects beyond the seven implemented. Adding one means: loop reference,
  parametrized reference test, *and* a row in `rsiena_compare.R`.
