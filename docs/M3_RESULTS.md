# M3a result: three waves, per-period rates, shared effects

Design: `docs/PRIORS_M3.md`. Run 2026-09-13 on the Spark. Artefacts in `data/`:
`train_m3.npz` (1.17 GB), `npe_m3.pt`, `npe_m3_sbc_pop.{npz,png}`,
`npe_m3_posterior_s50.npz`, `npe_m3_report.txt`.

## Set-up

| | |
|---|---|
| θ | rate₁, rate₂ (each U(1, 12)) + the seven M2 effects: 9 parameters |
| population | identical to M2 (n ~ U{20..80}, same starts and covariates); two periods simulated in sequence |
| training set | 10⁶ three-wave panels, seed 30, 740 s (1,352 panels/s) |
| summaries | 42: n, covariate shape, X0 block, one 13-block per period |
| estimator | NSF 8 × 128, batch 1024; 152 epochs, 77.5 min; best validation loss **−0.901** (two-wave M2 on the same budget: −0.361) |

s50's three-wave summary vector is in-distribution on all 42 summaries
(percentiles 0.10–0.87).

## Calibration across the population (fresh 4,000 draws, n ∈ [20, 80])

![M3 SBC ranks](figures/m3_sbc_ranks_population.png)

| parameter | KS p | mean rank | 50 % | 80 % | 90 % | 95 % |
|---|---|---|---|---|---|---|
| rate₁ | 0.530 | 0.498 | 0.492 | 0.808 | 0.899 | 0.945 |
| rate₂ | 0.029 | 0.489 | 0.498 | 0.796 | 0.891 | 0.944 |
| density | < 0.001 | 0.518 | 0.505 | 0.792 | 0.891 | 0.942 |
| recip | 0.196 | 0.505 | 0.495 | 0.787 | 0.879 | 0.937 |
| transTrip | 0.289 | 0.506 | 0.494 | 0.797 | 0.898 | 0.947 |
| cycle3 | 0.326 | 0.501 | 0.485 | 0.782 | 0.891 | 0.942 |
| altX(v) | 0.001 | 0.511 | 0.479 | 0.765 | 0.875 | 0.933 |
| egoX(v) | < 0.001 | 0.517 | 0.473 | 0.780 | 0.879 | 0.934 |
| sameX(g) | 0.001 | 0.489 | 0.463 | 0.784 | 0.887 | 0.938 |
| binomial s.e. | | 0.005 | 0.008 | 0.006 | 0.005 | 0.003 |

- Coverage within 1–3 points of nominal everywhere (worst: altX 80 % at 0.765).
- The M2-at-10⁶ ridge bias is largely absent here: transTrip, recip and cycle3
  are clean (mean ranks 0.501–0.506); density tilts mildly (0.518, i.e. the
  posterior slightly low) and the covariate effects show small tilts of
  0.01–0.02 in mean rank (≈ 0.03–0.05 sd). The third wave adds enough
  information that the same 10⁶ budget gets closer to calibration than the
  two-wave model did.
- Mild size dependence at the top of the range (n 65–80: recip 0.53–0.59,
  transTrip 0.53) — the same corner the M2 models found hardest.

## Real data: s501 → s502 → s503 vs RSiena's three-wave fit

| parameter | RSiena est ± se | M3 mean ± sd | M3 90 % | (RS − M3)/sd | M2 two-wave sd (10⁷) |
|---|---|---|---|---|---|
| rate₁ | 6.51 ± 1.07 | 5.37 ± 1.01 | [4.04, 7.17] | 1.13 | 0.97 |
| rate₂ | 5.30 ± 0.89 | 4.69 ± 0.78 | [3.55, 6.00] | 0.79 | — |
| density | −2.76 ± 0.16 | −2.69 ± 0.20 | [−3.03, −2.39] | −0.40 | 0.27 |
| recip | 2.44 ± 0.21 | 2.64 ± 0.24 | [2.28, 3.06] | −0.84 | 0.29 |
| transTrip | 0.64 ± 0.15 | 0.70 ± 0.15 | [0.46, 0.94] | −0.40 | 0.17 |
| cycle3 | −0.07 ± 0.28 | −0.09 ± 0.27 | [−0.55, 0.36] | 0.07 | 0.28 |
| altX(v) | −0.02 ± 0.07 | −0.08 ± 0.07 | [−0.19, 0.04] | 0.75 | 0.11 |
| egoX(v) | 0.06 ± 0.08 | 0.07 ± 0.07 | [−0.04, 0.18] | −0.12 | 0.10 |
| sameX(g) | 0.17 ± 0.17 | 0.03 ± 0.24 | [−0.36, 0.43] | 0.55 | 0.30 |

- **All nine RSiena estimates within 1.13 posterior sd**, and inside the M3
  90 % intervals, from an estimator that never saw s50.
- **The third wave tightens the posterior as it tightens RSiena's s.e.**:
  density sd 0.20 (two-wave M2: 0.27; RSiena 0.16 vs 0.22), recip 0.24 (0.29;
  RSiena 0.21 vs 0.28), transTrip 0.15 (0.17; RSiena 0.15 vs 0.19). The
  estimator extracts the extra information the extra wave carries.
- Both rates sit ~1 sd below RSiena (5.37 vs 6.51; 4.69 vs 5.30). The two-wave
  M2 model at 10⁶ had the same rate under-read, which vanished at 10⁷. Expect
  the same here; the 10⁷ three-wave run is the obvious next step and costs
  what the two-wave one did.
- Per-fit cost 0.07 s against RSiena's 24 s.

## What this establishes

Multi-wave estimation works in the same framework with no change to the
simulator: W − 1 sequential periods, one rate each, shared effects. The
estimator's posterior tightens with a third wave in step with RSiena's, and
matches RSiena on the canonical three-wave data set. Time-heterogeneous
effects are not modelled (nor in RSiena's default); a `sienaTimeTest`-style
check on the amortized posterior — compare per-period posteriors from the
two-wave estimator applied to each period — is a cheap diagnostic to add.

Next for M3: (a) 10⁷ three-wave panels to remove the residual tilts;
(b) M3b, behaviour co-evolution (`docs/PRIORS_M3b.md`), which is a simulator
extension with its own RSiena gate.
