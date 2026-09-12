# M1 result: amortized posterior for s501 → s502

First trained estimator (GETTING_STARTED §7 items 5–6; paper plan M1).
Run 2026-09-11 on the DGX Spark. Artefacts in `data/` (not in git; regenerable):
`npe_s50.pt`, `npe_s50_posterior.npz`, `npe_s50_sbc.{npz,png}`,
`npe_s50_training.npz`, `npe_s50_report.txt`. Scripts: `benchmarks/generate_s50.py`,
`benchmarks/npe_s50.py`, `benchmarks/rsiena_estimate.R`.

## Set-up

| | |
|---|---|
| data | s501 → s502 (n = 50), covariates alc (centred), smk |
| model | rate + density, recip, transTrip, cycle3, altX(alc), egoX(alc), sameX(smk) — the set validated against RSiena 1.6.6 |
| prior | box in `docs/PRIORS.md`; fixed empirical start X0 = s501 |
| training set | 10⁶ panels, seed 1, torch CUDA float32, 150 s (6,684 panels/s) |
| summaries | 13: MoM targets (changes + 7 statistics) + outdeg_sd, indeg_sd, isolates, mutual_dyads, tie_fraction; log1p on counts, asinh on signed sums |
| estimator | `sbi` 0.27 NPE, neural spline flow, 6 transforms × 100 hidden, batch 1024, lr 5e-4, early stopping (patience 20) |
| training | 999,000 draws, 158 epochs, 74.9 min; validation loss 1.494 → −1.316 (best, epoch 138) |
| held out | 1,000 draws for SBC |

## Posterior for the observed data, vs RSiena and Robbins–Monro

RSiena: `siena07`, unconditional (`cond = FALSE`), converged in one run
(max convergence ratio 0.096), 11 s. RM: `saomsim.estimate_rm`, conditional on
the observed distance (so no rate), 102 s. NPE: 20,000 posterior samples in
0.06 s.

| parameter | RSiena est ± se | RM est ± se | NPE mean ± sd | NPE 90% | (RS − NPE)/sd | sd/se |
|---|---|---|---|---|---|---|
| rate | 6.07 ± 1.03 | — | 5.88 ± 0.93 | [4.52, 7.52] | 0.20 | 0.90 |
| density | −2.66 ± 0.22 | −2.67 ± 0.22 | −2.78 ± 0.23 | [−3.18, −2.41] | 0.48 | 1.06 |
| recip | 2.11 ± 0.28 | 2.13 ± 0.27 | 2.06 ± 0.30 | [1.58, 2.57] | 0.17 | 1.08 |
| transTrip | 0.56 ± 0.19 | 0.56 ± 0.14 | 0.64 ± 0.15 | [0.41, 0.90] | −0.56 | 0.79 |
| cycle3 | 0.05 ± 0.35 | 0.05 ± 0.31 | 0.07 ± 0.26 | [−0.39, 0.46] | −0.11 | 0.75 |
| altX(alc) | −0.07 ± 0.09 | −0.07 ± 0.10 | −0.07 ± 0.09 | [−0.23, 0.08] | −0.02 | 1.03 |
| egoX(alc) | 0.03 ± 0.10 | 0.03 ± 0.10 | 0.07 ± 0.10 | [−0.09, 0.23] | −0.40 | 1.04 |
| sameX(smk) | 0.24 ± 0.23 | 0.24 ± 0.23 | 0.28 ± 0.24 | [−0.11, 0.68] | −0.14 | 1.06 |

- Every RSiena estimate lies inside the NPE 90 % interval; the largest
  discrepancy is 0.56 posterior sd (transTrip).
- Posterior sd matches RSiena's standard error to within ~10 % for six of
  eight parameters. NPE is tighter for transTrip (0.79×) and cycle3 (0.75×);
  see the SBC note on cycle3 below before reading that as a gain.
- No 90 % interval touches the prior box.
- **This is the M1 acceptance criterion** ("agrees with RSiena on real data
  where RSiena converges") met, at a per-fit cost of 0.06 s against 11 s
  (RSiena) and 102 s (RM), after a one-off 78 min of simulation + training.

Posterior correlations worth showing (|r| > 0.3): density × sameX −0.70,
transTrip × cycle3 −0.65, density × recip −0.52, egoX × sameX +0.43,
density × transTrip −0.35, recip × cycle3 −0.32, altX × sameX +0.31.
This is the identification geometry the paper's §4.1 promises to show: the
posterior reports it directly; a point estimate with marginal s.e. hides it.
The rate is *not* strongly correlated with the objective parameters here —
one period from a fixed sparse start identifies it well enough through the
change count.

## Simulation-based calibration (1,000 held-out draws × 1,000 samples)

![SBC rank histograms](figures/m1_sbc_ranks.png)

| parameter | KS p | C2ST(ranks) |
|---|---|---|
| rate | 0.168 | 0.589 |
| density | 0.406 | 0.594 |
| recip | 0.323 | 0.585 |
| transTrip | 0.500 | 0.559 |
| cycle3 | **0.011** | 0.582 |
| altX(alc) | 0.856 | 0.566 |
| egoX(alc) | 0.551 | 0.570 |
| sameX(smk) | 0.323 | 0.570 |

C2ST of the data-averaged posterior against the prior: 0.499 (ideal 0.5).

Reading: seven of eight parameters are consistent with uniform ranks. The
rank histograms (`npe_s50_sbc.png`) sit inside the uniformity band except
**cycle3**, which has excess mass at low ranks — the true value tends to sit
below most posterior samples, i.e. the posterior for cycle3 is slightly
biased upward and/or slightly too narrow. This is the same parameter whose
posterior sd is 0.75× RSiena's s.e., so that tightness should not be
claimed as a gain. Magnitude is modest (KS p = 0.011 with 1,000 draws; the
C2ST values are all in the 0.56–0.59 range, a mild but uniform detectable
departure). Candidate causes, in the order to try:

1. Summary information loss: cycle3's target statistic is heavy-tailed and
   strongly negatively correlated with transTrip in the posterior; the
   13 summaries may not separate them well. A learned embedding on the
   stored networks is the principled fix (and the M2 direction anyway).
2. Flow capacity / training: validation loss was still drifting at epoch
   138; a larger flow or lower learning rate is a cheap test using
   `--limit 100000` first.
3. Prior edge: cycle3's range is (−1.5, 0.5) and RSiena puts the s50 value
   near 0.05; nothing indicates edge effects for the observed data, but the
   SBC draws span the whole box.

## Cost accounting

| step | wall clock |
|---|---|
| simulate 10⁶ panels (GPU) | 150 s |
| train NSF (GPU) | 75 min |
| posterior for one dataset | 0.06 s |
| SBC, 1,000 × 1,000 | 12 s |
| RSiena, one dataset | 11 s |
| RM (saomsim), one dataset | 102 s |

Break-even against RSiena is ~400 datasets; against the amortized
estimator's own SBC-checked posterior there is no classical equivalent.
The 75 min is dominated by 158 epochs over 10⁶ rows at batch 1024; the
smoke run suggests 10⁵ rows trains in a few minutes — use `--limit` to tune,
then train once at full size.

## What to do next

- Tune on `--limit 100000` (batch 4096, more transforms) and check whether
  cycle3's calibration recovers; if not, it points to the embedding.
- Coverage table (nominal vs empirical) from the same held-out set for §4.
- Posterior predictive check on statistics not in the embedding
  (e.g. triad census, geodesic distribution) — §4 "goodness of fit in graph
  space".
- Then M2: condition on X0 so one estimator covers many networks.
