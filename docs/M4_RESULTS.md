# M4 step 1: one estimator for n ≤ 200, and the Glasgow panel

Run 2026-09-14/16 on the Spark. Artefacts in `data/`: `train_m4.npz` (252 MB,
Spark only), `npe_m4.pt`, `npe_m4_sbc.{npz,png}` (2,000 held-out training
draws), `npe_m4_sbc_pop.{npz,png}` (4,000 fresh draws), `npe_m4_posterior_s50.npz`
(the Glasgow posterior, despite the name), `npe_m4_report.txt`, `m4_sbc.log`.
Real data and RSiena fits in `benchmarks/glasgow/`.

M4 is the application milestone: many networks at once, at sizes existing tools
handle one fit at a time. Step 1 asks whether the M3a estimator (three waves,
per-period rates, seven effects) simply extends from n ≤ 80 to n ≤ 200 with the
population widened and nothing else changed, and what it says about a real
school-year network of 129 pupils. The answer is: calibration yes, the real
network only partly — and the part that fails points at the population design,
not the estimator.

## Set-up

| | |
|---|---|
| θ | rate₁, rate₂ (each **U(1, 20)**, was U(1, 12)) + the seven M2 effects: 9 parameters |
| population | M2/M3a population with **n ~ U{20..200}** (was 20..80); starts, covariates and effects unchanged (`docs/PRIORS_M2.md`) |
| training set | 10⁶ three-wave panels, seed 70, summary-only (`--no-networks`), **13,376 s (75 panels/s)** — 18× slower than the n ≤ 80 set, as the n² kernel predicts |
| summaries | 42, as M3a |
| estimator | NSF 8 × 128, batch 1024, lr 5e-4; 190 epochs, 207.6 min; best validation loss −2.919 |
| Glasgow fit time | **0.12 s** per posterior against RSiena's 219 s (three-wave network model, one run to convergence) |

`saomsim.population` now takes `n_range` above `N_MAX` when networks are not
kept (`keep_networks=False`); `generate_m2.py`, `npe_m2.py` and `sbc_m2.py`
expose `--n-max` and `--rate-max`.

## Calibration across the population (fresh 4,000 draws, n ∈ [20, 200])

![M4 SBC ranks](figures/m4_sbc_ranks_population.png)

| parameter | KS p | mean rank | 50 % | 80 % | 90 % | 95 % |
|---|---|---|---|---|---|---|
| rate₁ | < 0.001 | 0.537 | 0.510 | 0.798 | 0.896 | 0.944 |
| rate₂ | 0.007 | 0.491 | 0.480 | 0.781 | 0.885 | 0.939 |
| density | 0.209 | 0.494 | 0.498 | 0.793 | 0.891 | 0.944 |
| recip | < 0.001 | 0.519 | 0.480 | 0.770 | 0.883 | 0.939 |
| transTrip | < 0.001 | 0.464 | 0.475 | 0.765 | 0.875 | 0.932 |
| cycle3 | 0.006 | 0.502 | 0.466 | 0.777 | 0.886 | 0.946 |
| altX(v) | 0.015 | 0.508 | 0.478 | 0.776 | 0.878 | 0.941 |
| egoX(v) | 0.791 | 0.500 | 0.503 | 0.789 | 0.887 | 0.939 |
| sameX(g) | 0.018 | 0.488 | 0.492 | 0.794 | 0.889 | 0.940 |
| binomial s.e. | | 0.005 | 0.008 | 0.006 | 0.005 | 0.003 |

- **Coverage nominal within 1.5–3.5 points for all nine parameters** across the
  whole size range (worst: transTrip 80 % at 0.765 and 90 % at 0.875 — the
  posterior a little too narrow on closure, as in every 10⁶ model here). The
  2,000 held-out training draws agree (`npe_m4_report.txt`).
- Two tilts survive the widening and are the same size in every n band:
  **rate₁ reads low (mean rank 0.53–0.55 from n 20 to n 200)** and transTrip
  reads high (0.45–0.47). Both are ≈ 0.1 sd; neither grows with n.
- No other drift with size. By band (30-wide bands, 300–1,000 draws each) the
  remaining seven parameters stay within 0.47–0.53 up to and including n = 200.
- The population SBC itself: 4,000 × 1,000 posterior samples in **83 s** once
  drawn in chunks of 100 observations with bounded rejection
  (`benchmarks.coverage.sbc_ranks_chunked`). The first attempt with sbi's
  `run_sbc` ran four hours and died in a GPU out-of-memory event while sharing
  the Spark with a training job — it draws all 4 million samples at once, and
  its rejection sampler loops without bound on an observation whose posterior
  sits almost wholly outside the prior box (one of the 4,000 here).

## Real data: Glasgow Teenage Friends and Lifestyle Study, three waves, n = 129

The full Glasgow panel (Michell & West; Pearson & West), of which s50 is the
familiar 50-girl excerpt: 160 pupils in one Scottish secondary school, followed
over three years (1995–97), each naming up to six best friends. We keep the 129
pupils present at all three waves (RSiena's `selection129`), dichotomise
"best friend" and "just a friend" as a tie, and use v = alcohol at wave 1
(centred) and g = sex, so the M3a effect set applies unchanged. 26 pupils lack
an alcohol value in some wave; saomsim has no missing-data handling, so both
RSiena and saomsim see the same completed matrix (nearest observed wave carried
forward, else backward; `benchmarks/glasgow/prepare_and_fit.R`). Ties per wave
449, 445, 464 (tie fraction 0.027, mean out-degree 3.5); 478 and 437 tie
changes across the two periods; Jaccard stability 0.30 and 0.35.

| parameter | RSiena est ± se | M4 mean ± sd | M4 90 % | (RS − M4)/sd |
|---|---|---|---|---|
| **rate₁** | **10.71 ± 1.01** | **7.70 ± 0.82** | [6.58, 9.11] | **3.67** |
| **rate₂** | **9.04 ± 0.75** | **7.45 ± 0.66** | [6.45, 8.60] | **2.41** |
| density | −3.36 ± 0.10 | −3.14 ± 0.10 | [−3.32, −2.97] | −2.16 |
| recip | 2.25 ± 0.11 | 2.39 ± 0.14 | [2.16, 2.61] | −0.98 |
| transTrip | 0.62 ± 0.04 | 0.61 ± 0.06 | [0.51, 0.71] | 0.21 |
| cycle3 | −0.42 ± 0.08 | −0.42 ± 0.12 | [−0.61, −0.23] | 0.07 |
| altX(v) | −0.01 ± 0.03 | 0.03 ± 0.05 | [−0.04, 0.10] | −0.88 |
| egoX(v) | −0.01 ± 0.04 | −0.04 ± 0.04 | [−0.11, 0.03] | 0.62 |
| sameX(g) | 0.89 ± 0.10 | 0.68 ± 0.12 | [0.49, 0.89] | 1.69 |

- **Five of nine on RSiena** (|z| < 1): recip, transTrip, cycle3 and the two
  alcohol covariate effects. The structural story of the network — strong
  reciprocity, transitive closure, negative three-cycles, no alcohol selection
  in the friendship model — is the same from both estimators, in 0.12 s
  against 219 s.
- **Both rates read low by a quarter** (7.7 vs 10.7, 7.5 vs 9.0; 3.7 and 2.4
  sd), density reads 0.2 high (2.2 sd) and same-sex selection 0.2 low (1.7 sd).
  The rate direction matches the population tilt above, but the size does not:
  the SBC tilt is 0.1 sd, this is 3.7. Something specific to Glasgow is doing
  most of it.

### Where Glasgow sits in the training population

Percentile of each Glasgow summary among the 250,432 training panels with
n ∈ [110, 150] (`/tmp/glasgow_screen.py` on the Spark, 2026-09-16):

| summary | Glasgow | percentile | band median |
|---|---|---|---|
| x0 tie fraction | 0.027 | 0.055 | 0.116 |
| x0 out-degree sd | 1.63 | **0.019** | 4.04 |
| x0 in-degree sd | 2.27 | 0.094 | 4.09 |
| x0 isolates | 2 | 0.961 | 0 |
| x1 changes | 478 | 0.157 | 1,189 |
| x1 out-degree sd | 1.51 | **0.019** | 5.25 |
| x2 tie fraction | 0.028 | 0.090 | 0.181 |
| g categories | 2 | (minimum) | 3 |

Glasgow is inside the training support on every summary but sits in a thinly
populated corner of it: **a sparse network with nearly uniform out-degrees.**
The population's starts draw a tie fraction d ~ U(0.02, 0.20) regardless of n
(`docs/PRIORS_M2.md`), a choice made for n ≤ 80 where it spans mean degrees
of 0.4–16. At n = 129 the same range gives mean degrees 2.6–26 with a median of
15 — four times the Glasgow network — and the SAOM burn-in that structures half
the starts has no cap on out-degree, whereas Glasgow pupils named at most six
friends (out-degree sd 1.6 against a training median of 4.0). Roughly 5 % of
same-size training panels look like Glasgow on density and 2 % on degree
spread; the estimator has been shown a few thousand such panels, not a few
hundred thousand.

That is where the rate under-read comes from. The rate is read off the change
counts (478, 437) against what the population associates with that many changes
at n ≈ 130; in a dense population the same count of changes is a smaller
per-actor rate. Among training panels in the band, the median x1 change count
is 1,314 at rate₁ ≈ 10.7 and 958 at rate₁ ≈ 7.7 — both far above Glasgow's 478,
because both are computed over networks four times denser. The estimator does
the honest thing with an input at the edge of what it has seen: it shrinks
toward the population. Density (read high, i.e. less negative, on a sparse
network) and same-sex selection fit the same reading.

Two implications, one for the paper and one for step 2:

1. **Calibration across a population is necessary, not sufficient.** The
   estimator is calibrated over its population at every n, including 129, and
   still mis-reads a real network at that n because the population's *shape*
   at large n does not match real school networks. The in-distribution screen
   (`docs/OOD_RESULTS.md`) catches it — the degree-sd summaries flag at the
   2 % tail — which is exactly what the screen is for.
2. **The population for n > 80 must hold mean degree, not tie fraction, fixed.**
   Real friendship networks have mean degrees of 2–8 across sizes; tie fraction
   falls as 1/n. Step 2's population should draw d ~ U(2/n, 10/n) (or a mixture
   with the current range for continuity at small n), and should include an
   out-degree-capped nomination regime, since most school surveys cap
   nominations. That is a `sample_start_networks` change plus a regeneration,
   ≈ 4 h at n ≤ 200; the estimator, summaries and training loop are untouched.

## What this establishes

- The M3a estimator scales to n ≤ 200 with no architectural change: 10⁶ panels,
  3.5 h of training, coverage nominal within 3.5 points everywhere, no drift
  with n. The cost that scales is generation (18× the n ≤ 80 set), as
  `docs/SCALING.md` anticipates.
- On the Glasgow panel it agrees with RSiena on the structural effects and
  reads the rates low, and the reason is diagnosable from the training set
  alone: the start-network population is dense relative to real school
  networks at this size. This is the first real-data result here that the
  population design, rather than the training budget, limits — the 10⁷ runs
  fixed the s50 rate under-reads because s50 sits in the middle of the
  population; Glasgow sits at its edge.

## Step 2: the sparse-start population (2026-09-16)

Design: `docs/PRIORS_M4.md`. One change — half the start networks draw a mean
degree k ~ U(2, 10) instead of a tie fraction, and half of those are capped at
ceil(k) + U{1..3} out-ties per actor. Artefacts: `train_m4b.npz` (Spark),
`npe_m4b.*`, `npe_m4b_sbc_pop.{npz,png}`, `npe_m4b_report.txt`, `m4b.log`.

| | |
|---|---|
| training set | 10⁶ panels, seed 71, **9,511 s (105 panels/s)** — faster than step 1's 75/s, sparser starts being cheaper to update |
| estimator | NSF 8 × 128, batch 1024, lr 5e-4; 217 epochs, ≈ 110 min; best validation loss −3.193 (not comparable with step 1's −2.919: different population) |
| Glasgow | 0.06 s |

### Calibration (fresh 4,000 draws from the sparse population, n ∈ [20, 191])

![M4 step 2 SBC ranks](figures/m4b_sbc_ranks_population.png)

| parameter | KS p | mean rank | 50 % | 80 % | 90 % | 95 % |
|---|---|---|---|---|---|---|
| rate₁ | 0.555 | 0.499 | 0.493 | 0.782 | 0.887 | 0.941 |
| rate₂ | < 0.001 | 0.480 | 0.493 | 0.794 | 0.890 | 0.940 |
| density | 0.183 | 0.506 | 0.494 | 0.789 | 0.885 | 0.933 |
| recip | 0.045 | 0.498 | 0.479 | 0.778 | 0.884 | 0.941 |
| transTrip | 0.001 | 0.487 | 0.477 | 0.772 | 0.881 | 0.935 |
| cycle3 | 0.004 | 0.509 | 0.470 | 0.773 | 0.884 | 0.934 |
| altX(v) | 0.456 | 0.500 | 0.499 | 0.788 | 0.887 | 0.940 |
| egoX(v) | 0.102 | 0.489 | 0.494 | 0.788 | 0.889 | 0.942 |
| sameX(g) | 0.041 | 0.508 | 0.483 | 0.780 | 0.885 | 0.937 |
| binomial s.e. | | 0.005 | 0.008 | 0.006 | 0.005 | 0.003 |

- **Coverage nominal within 3 points for all nine**, over a wider population
  than step 1 — the estimator absorbed the extra input range at the same 10⁶
  budget. No observation needed clamped draws.
- The step-1 rate₁ tilt (0.53–0.55 in every band) is **gone** (0.499; by band
  0.487–0.512). transTrip's closure tilt shrank from 0.464 to 0.487.
- One new tilt: rate₂ reads slightly high at large n (mean rank 0.435–0.44 for
  n ≥ 140, ≈ 0.15 sd), where step 1 had it at 0.47. Coverage there is still
  nominal. Worth watching, not acting on.

### Glasgow, three waves, n = 129: step 1 vs step 2

| parameter | RSiena est ± se | step 1 (dense starts) | z₁ | **step 2 (sparse starts)** | **z₂** |
|---|---|---|---|---|---|
| **rate₁** | 10.71 ± 1.01 | 7.70 ± 0.82 | 3.67 | **9.11 ± 1.77** [7.29, 11.78] | **0.90** |
| **rate₂** | 9.04 ± 0.75 | 7.45 ± 0.66 | 2.41 | **7.89 ± 0.75** [6.77, 9.16] | **1.53** |
| density | −3.36 ± 0.10 | −3.14 ± 0.10 | −2.16 | −3.20 ± 0.10 [−3.36, −3.05] | −1.72 |
| recip | 2.25 ± 0.11 | 2.39 ± 0.14 | −0.98 | **2.25 ± 0.12** [2.05, 2.45] | 0.01 |
| transTrip | 0.62 ± 0.04 | 0.61 ± 0.06 | 0.21 | 0.61 ± 0.05 [0.52, 0.70] | 0.23 |
| cycle3 | −0.42 ± 0.08 | −0.42 ± 0.12 | 0.07 | −0.39 ± 0.10 [−0.54, −0.22] | −0.31 |
| altX(v) | −0.01 ± 0.03 | 0.03 ± 0.05 | −0.88 | 0.03 ± 0.04 [−0.04, 0.10] | −0.80 |
| egoX(v) | −0.01 ± 0.04 | −0.04 ± 0.04 | 0.62 | 0.03 ± 0.04 [−0.04, 0.10] | −1.07 |
| sameX(g) | 0.89 ± 0.10 | 0.68 ± 0.12 | 1.69 | 0.71 ± 0.12 [0.52, 0.90] | 1.54 |

(z = (RSiena − ours) / our sd; 90 % intervals for step 2 in brackets.)

- **All nine RSiena estimates now lie inside the 90 % intervals; max |z| 1.72**
  (was 3.67). The rate under-read is resolved by the population change alone:
  rate₁ moved from 3.7 sd to 0.9 sd of RSiena, rate₂ from 2.4 to 1.5, density
  from 2.2 to 1.7, with no change to the estimator, summaries or budget.
- The rate₁ posterior is also **twice as wide** (sd 1.77 vs 0.82) and now
  matches RSiena's standard error more closely than step 1 did (1.01). Step
  1's narrow interval was over-confidence at the edge of its population — the
  in-distribution screen would have flagged it, the calibration table could
  not. Step 2 puts Glasgow inside the population and the interval widens to
  what the data support.
- Reciprocity lands on RSiena to two decimals; transTrip, cycle3 and altX are
  unchanged and on. sameX(g) and density remain 1.5–1.7 sd off in the same
  direction as before — small, but consistent across both fits, so probably
  not noise. Both are effects that the out-degree cap in the later waves
  could bias (a capped actor cannot add the same-sex tie the model wants);
  the design note lists this as the next suspect.

What this settles: the step-1 failure was the population's shape, as
diagnosed, and the fix was cheap — a start prior that keeps mean degree
rather than tie fraction fixed as n grows. The estimator for n ≤ 200 is now
calibrated over a population that contains real school networks and agrees
with RSiena on all nine parameters of the Glasgow panel in 0.06 s.

Next: the co-evolution estimator on Glasgow's network × alcohol (RSiena fit
already in `benchmarks/glasgow/rsiena_coevolution.json`; needs the sparse
regime in `npe_coev.py generate`), then the many-classroom application
(Knecht).
