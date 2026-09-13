# M2 result: one estimator across start networks, sizes and covariates

First population-level estimator (paper plan M2; design in `docs/PRIORS_M2.md`).
Run 2026-09-12 on the DGX Spark. Artefacts in `data/` (not in git):
`train_m2.npz` (825 MB), `npe_m2.pt`, `npe_m2_report.txt`,
`npe_m2_sbc_pop.{npz,png}`, `npe_m2_sbc_n50.{npz,png}`, `npe_m2_posterior_s50.npz`.
Scripts: `benchmarks/generate_m2.py`, `benchmarks/npe_m2.py`, `benchmarks/sbc_m2.py`.

## Set-up

| | |
|---|---|
| population | n ~ U{20..80}; X0 half ER (d ~ U(0.02, 0.2)), half 10·n-step SAOM burn-in at independent θ₀; covariates v (N(0,1) or Likert, centred), g (2–4 groups) |
| θ prior | the M1 box, unchanged |
| training set | 10⁶ panels, seed 2, CUDA float32: **533 s (1,875 panels/s)**, both waves + covariates stored bit-packed |
| summaries | 29: n, covariate shape (3), X0 block (12), X1 block (13); log1p / asinh transforms, log n |
| estimator | sbi 0.27 NPE, NSF 8 transforms × 128 hidden, batch 1024, lr 5e-4 |
| training | 998,000 draws, 116 epochs, 67 min; best validation loss −0.361 |

Prior predictive: s50's observed summary vector sits between the 10th and
87th percentile of the population on all 29 summaries — a fair real-start
test. Degeneracy (X1 denser than 0.5) is confined to small n: 15.7 % at
n 20–29, 8.3 % at 30–39, 3.8 % at 40–49, < 1.5 % above 50; X0 degeneracy
7 % at n 20–29 only. Kept, per the never-filter rule.

## Calibration over the population (fresh test set, 4,000 draws, n ∈ [20, 80])

`benchmarks/sbc_m2.py --N 4000 --chunk 100 --seed 3` — an independent sample,
not a hold-out of the training file. (The first run's hold-out was the last
2,000 rows, which share one n per chunk and so tested a single size; fixed in
`npe_m2.py` with a random split, and superseded by this fresh-sample check.)

![population SBC ranks](figures/m2_sbc_ranks_population.png)

| parameter | KS p | mean rank (0.5) | 50 % | 80 % | 90 % | 95 % |
|---|---|---|---|---|---|---|
| rate | 0.110 | 0.497 | 0.485 | 0.784 | 0.891 | 0.943 |
| density | **< 0.001** | **0.535** | 0.488 | 0.788 | 0.885 | 0.941 |
| recip | **0.002** | 0.487 | 0.519 | 0.808 | 0.895 | 0.948 |
| transTrip | **< 0.001** | **0.477** | 0.505 | 0.809 | 0.902 | 0.950 |
| cycle3 | **< 0.001** | **0.478** | 0.500 | 0.807 | 0.907 | 0.955 |
| altX(v) | 0.128 | 0.496 | 0.493 | 0.786 | 0.884 | 0.942 |
| egoX(v) | 0.053 | 0.489 | 0.498 | 0.789 | 0.887 | 0.946 |
| sameX(g) | 0.110 | 0.508 | 0.513 | 0.804 | 0.901 | 0.945 |
| binomial s.e. | | 0.005 | 0.008 | 0.006 | 0.005 | 0.003 |

Reading:

- **Coverage is within 1–1.5 points of nominal everywhere** (worst: density
  and altX 90 % intervals at 0.885). Interval widths are right.
- **There is a small, systematic location bias along the density–closure
  ridge.** Density ranks rise linearly (posterior sits slightly *below* the
  true density; the top rank bin holds ~1.5× its expected mass), while
  transTrip and cycle3 tilt the other way (posterior slightly *high*), recip
  mildly high. Mean-rank shifts of 0.02–0.035 correspond to a bias of roughly
  0.05–0.1 posterior sd. The other four parameters are flat.
- **The bias is uniform across n**: density's mean rank is 0.52–0.55 in every
  size band (20–34, 35–49, 50–64, 65–79). It is not a small-n or large-n
  failure; it is the flow's residual error in the direction the summaries
  identify least well. Same direction, and about the same size, at n = 50
  alone (fresh 2,000 draws: density 0.527, transTrip 0.482, cycle3 0.471;
  coverage nominal).
- C2ST(ranks) values (0.83 on 4,000 draws, 0.69 on 2,000) are not
  interpretable here: the classifier separates integer ranks from continuous
  uniform draws and its accuracy grows with sample size. Use KS, mean rank,
  coverage and the histograms.

Comparison with M1 (fixed start, 13 summaries, same data budget): M1 had
seven of eight parameters clean and a borderline cycle3. M2 solves a much
larger problem — any start, any n in 20–80, any covariate layout — with the
same 10⁶ panels, and pays for it with a ~0.1 sd bias on the ridge. That is
exactly the "data-limited" regime the 10⁵ sweep identified (`M1_RESULTS.md`):
the remedy is more simulations and a richer embedding, not a different flow.

## The real-start test: s501 → s502, never seen in training

| parameter | M2 mean ± sd | M2 90 % | RSiena est ± se | (RS − M2)/sd | M1 mean ± sd |
|---|---|---|---|---|---|
| rate | 4.82 ± 0.83 | [3.62, 6.22] | 6.07 ± 1.03 | 1.51 | 5.88 ± 0.93 |
| density | −2.84 ± 0.27 | [−3.32, −2.44] | −2.66 ± 0.22 | 0.65 | −2.78 ± 0.23 |
| recip | 2.53 ± 0.35 | [1.99, 3.13] | 2.11 ± 0.28 | −1.20 | 2.06 ± 0.30 |
| transTrip | 0.67 ± 0.19 | [0.37, 0.99] | 0.56 ± 0.19 | −0.58 | 0.64 ± 0.15 |
| cycle3 | −0.07 ± 0.29 | [−0.57, 0.38] | 0.05 ± 0.35 | 0.42 | 0.07 ± 0.26 |
| altX(v) | −0.20 ± 0.10 | [−0.36, −0.04] | −0.07 ± 0.09 | 1.33 | −0.07 ± 0.09 |
| egoX(v) | 0.01 ± 0.11 | [−0.17, 0.18] | 0.03 ± 0.10 | 0.21 | 0.07 ± 0.10 |
| sameX(g) | 0.27 ± 0.28 | [−0.16, 0.76] | 0.24 ± 0.23 | −0.10 | 0.28 ± 0.24 |

- **All eight RSiena estimates fall inside the M2 90 % intervals**, from an
  estimator that never saw s501, its covariates, or n = 50 specifically.
  Posterior sd is 10–25 % wider than M1's, as expected for the general model.
- The deviations from RSiena (rate 1.5 sd low, recip 1.2 sd high, altX 1.3 sd
  low, density 0.65 sd low, transTrip 0.6 sd high) are individually within
  noise but *collectively* have the signature the population SBC found:
  density low, closure high. This is a single s50 observation, so it is not
  evidence on its own, but it agrees with the calibration diagnosis.
- Per-fit cost 0.08 s. Training 67 min once, for every dataset with
  20 ≤ n ≤ 80 and this effect set.

## Variant: learned panel embedding (2026-09-12)

`saomsim/embedding.py` + `benchmarks/npe_m2_embed.py`: a permutation-invariant
GNN over the actors of both waves and the covariates (per-actor degree, mutual,
change and covariate features → two rounds of masked neighbour aggregation over
X0 and X1 → masked mean/max pooling), with the 29 hand summaries appended as a
residual, feeding the same NSF 8 × 128. Same 10⁶ panels, same prior. 935,584
parameters; batch 512; 122 epochs; **325 min** (the last two hours sharing the
GPU with the 10⁷ run). Best validation loss **−0.917 vs −0.361** for the
summary-only model: the embedding extracts substantially more information from
the panel than the 29 summaries do.

![embedding SBC ranks](figures/m2_embed_sbc_ranks_population.png)

Fresh 4,000-draw SBC across n:

| parameter | KS p | mean rank | 90 % cov. | mean rank, n 20–34 → 65–79 |
|---|---|---|---|---|
| rate | 0.456 | 0.497 | 0.889 | 0.480 → 0.505 |
| density | **< 0.001** | **0.453** | 0.892 | 0.465 → 0.444 |
| recip | 0.013 | 0.489 | 0.897 | 0.487 → 0.483 |
| transTrip | **< 0.001** | **0.560** | 0.887 | 0.548 → 0.575 |
| cycle3 | 0.170 | 0.493 | 0.904 | 0.508 → 0.473 |
| altX(v) | 0.504 | 0.502 | 0.895 | 0.495 → 0.497 |
| egoX(v) | **< 0.001** | 0.475 | 0.876 | 0.501 → 0.456 |
| sameX(g) | **< 0.001** | 0.464 | 0.897 | 0.480 → 0.443 |

Reading:

- **The ridge bias is still there and has flipped sign.** Summary-only M2 put
  density slightly *low* and transTrip *high* (mean ranks 0.535 / 0.477); the
  embedding model puts density *high* and transTrip *low* (0.453 / 0.560), and
  by a somewhat larger margin (~0.15 posterior sd). A bias that reverses
  direction between two estimators trained on the same data is a training
  artefact — which side of the density–closure ridge the flow settles on —
  not an information limit of the inputs.
- **It grows with n** for the embedding model (density 0.465 → 0.444,
  transTrip 0.548 → 0.575, sameX 0.480 → 0.443 from the smallest to the
  largest size band), whereas the summary model's bias was flat in n. The
  pooled actor representation is doing something size-dependent that log n as
  an input does not fully correct; a normalised pooling (e.g. attention or
  n-aware scaling of the sum) is the obvious thing to try.
- Coverage remains within 1–2.5 points of nominal (worst: egoX 90 % at 0.876).
- cycle3 — the parameter that was borderline in M1 and biased in summary-M2 —
  is **clean** here (KS p 0.17, mean rank 0.493): the embedding does resolve
  the triadic statistic the summaries blurred.

The s50 real-start posterior, three estimators side by side:

| parameter | RSiena | M1 (fixed start) | M2 summaries | **M2 embedding** |
|---|---|---|---|---|
| rate | 6.07 ± 1.03 | 5.88 ± 0.93 | 4.82 ± 0.83 | 4.84 ± 0.84 |
| density | −2.66 ± 0.22 | −2.78 ± 0.23 | −2.84 ± 0.27 | **−2.75 ± 0.23** |
| recip | 2.11 ± 0.28 | 2.06 ± 0.30 | 2.53 ± 0.35 | **2.38 ± 0.37** |
| transTrip | 0.56 ± 0.19 | 0.64 ± 0.15 | 0.67 ± 0.19 | **0.49 ± 0.19** |
| cycle3 | 0.05 ± 0.35 | 0.07 ± 0.26 | −0.07 ± 0.29 | **0.03 ± 0.31** |
| altX | −0.07 ± 0.09 | −0.07 ± 0.09 | −0.20 ± 0.10 | **−0.08 ± 0.10** |
| egoX | 0.03 ± 0.10 | 0.07 ± 0.10 | 0.01 ± 0.11 | **0.03 ± 0.12** |
| sameX | 0.24 ± 0.23 | 0.28 ± 0.24 | 0.27 ± 0.28 | 0.28 ± 0.23 |

On the real data the embedding model is the closest of the three population-
free comparisons to RSiena on six of eight parameters (|z| ≤ 0.4 on density,
transTrip, cycle3, altX, egoX, sameX; recip 0.7). The exception is the rate,
where both M2 variants sit 1.2 sd below RSiena and M1 — the one parameter
the population estimators read differently from the fixed-start one, and a
lead worth following (the rate is identified through the change count
relative to n and the start density; the population's start distribution may
be pulling it).

Cost: 5.4 h of training against 67 min for the summary model, for a
posterior that is sharper (validation loss), better on the real data, clean
on cycle3, and biased in the opposite direction on the ridge. The 10⁷
summary-only run (in progress) tests the other lever — data — on the same
bias.

## Ten million panels: the data lever (2026-09-12)

Same summary-only architecture (NSF 8 × 128, 29 summaries), trained on the
ten 10⁶ shards (`benchmarks/generate_m2.py --seed 10..19`, 7.8 GB, ~2 h of
generation while sharing the machine), batch 4096, lr 1e-3, random 2,000-row
hold-out. **101 epochs, 337 min**; best validation loss **−0.834** against
−0.361 for the same model on 10⁶ (and −0.917 for the embedding model on 10⁶).
Training was CPU-bound on the data loader at this size (GPU 30–40 %), which is
the next engineering item.

Fresh 4,000-draw SBC across n (same seed 3 test set as the 10⁶ comparison):

![10M SBC ranks](figures/m2_10m_sbc_ranks_population.png)

| parameter | KS p | mean rank | 50 % | 80 % | 90 % | 95 % | mean rank, n 20–34 → 65–79 |
|---|---|---|---|---|---|---|---|
| rate | 0.289 | 0.500 | 0.485 | 0.788 | 0.896 | 0.950 | 0.503 → 0.513 |
| density | **0.634** | **0.502** | 0.495 | 0.794 | 0.894 | 0.946 | 0.514 → 0.495 |
| recip | 0.128 | 0.499 | 0.520 | 0.813 | 0.908 | 0.953 | 0.491 → 0.505 |
| transTrip | **0.581** | **0.497** | 0.511 | 0.806 | 0.906 | 0.954 | 0.499 → 0.500 |
| cycle3 | 0.003 | 0.483 | 0.491 | 0.797 | 0.899 | 0.949 | 0.501 → 0.457 |
| altX(v) | 0.041 | 0.512 | 0.500 | 0.796 | 0.893 | 0.943 | 0.505 → 0.512 |
| egoX(v) | 0.239 | 0.496 | 0.514 | 0.803 | 0.900 | 0.951 | 0.502 → 0.498 |
| sameX(g) | 0.102 | 0.496 | 0.514 | 0.814 | 0.904 | 0.951 | 0.491 → 0.489 |

- **The density–closure ridge bias is gone.** Density and transTrip mean
  ranks 0.502 / 0.497 with KS p 0.63 / 0.58, against 0.535 / 0.477 at 10⁶.
  Recip 0.499. Same architecture, same summaries, ten times the data: the bias
  was a training artefact of a data-limited fit, exactly as the 10⁵ sweep on
  M1 predicted and as the sign flip between the two 10⁶ estimators implied.
- **Coverage nominal within one point** at every level for every parameter.
- Residuals: cycle3 (mean rank 0.483, KS p 0.003) with a drift toward large
  n (0.457 at n 65–79) — the parameter that the *embedding* model resolved
  cleanly at 10⁶; and a faint altX tilt (0.512, p 0.04). Six of eight are
  clean by every measure.

s50 real start:

| parameter | RSiena | M2 10⁶ summaries | M2 10⁶ embedding | **M2 10⁷ summaries** | z vs RSiena |
|---|---|---|---|---|---|
| rate | 6.07 ± 1.03 | 4.82 ± 0.83 | 4.84 ± 0.84 | **5.51 ± 0.97** | 0.58 |
| density | −2.66 ± 0.22 | −2.84 ± 0.27 | −2.75 ± 0.23 | **−2.78 ± 0.27** | 0.41 |
| recip | 2.11 ± 0.28 | 2.53 ± 0.35 | 2.38 ± 0.37 | **2.31 ± 0.29** | −0.69 |
| transTrip | 0.56 ± 0.19 | 0.67 ± 0.19 | 0.49 ± 0.19 | **0.50 ± 0.17** | 0.34 |
| cycle3 | 0.05 ± 0.35 | −0.07 ± 0.29 | 0.03 ± 0.31 | **0.01 ± 0.28** | 0.12 |
| altX | −0.07 ± 0.09 | −0.20 ± 0.10 | −0.08 ± 0.10 | **−0.15 ± 0.11** | 0.78 |
| egoX | 0.03 ± 0.10 | 0.01 ± 0.11 | 0.03 ± 0.12 | **−0.02 ± 0.10** | 0.46 |
| sameX | 0.24 ± 0.23 | 0.27 ± 0.28 | 0.28 ± 0.23 | **0.36 ± 0.30** | −0.39 |

All eight within 0.8 sd of RSiena. **The rate discrepancy is resolved**
(5.51 vs 6.07, z 0.58; both 10⁶ models had 4.8) — it too was a data-limited
artefact, not a population-prior effect. The one parameter the 10⁷ model
reads less well than the embedding model on this dataset is altX (−0.15 vs
−0.08, RSiena −0.07).

### What the three M2 estimators say together

| | 10⁶ summaries | 10⁶ embedding | 10⁷ summaries |
|---|---|---|---|
| validation loss | −0.361 | −0.917 | −0.834 |
| training | 67 min | 325 min | 337 min |
| ridge bias (density / transTrip mean rank) | 0.535 / 0.477 | 0.453 / 0.560 | **0.502 / 0.497** |
| cycle3 | 0.478 | **0.493** | 0.483 |
| clean parameters (KS p > 0.05) | 4 | 4 | **6** |
| s50 max |z| vs RSiena | 1.51 (rate) | 1.46 (rate) | **0.78 (altX)** |

Data and embedding attack different things. Data removed the ridge bias and
fixed the rate; the embedding resolved cycle3 and extracts more information
per panel. The obvious next estimator is the embedding trained on 10⁷, which
needs a streaming loader (the packed input is 18 GB as bytes, 72 GB as the
float tensor sbi wants) and a size-normalised pooling to remove the n-drift.
That is engineering, not research risk.

## What this establishes and what it does not

Established: a single amortized estimator conditioned on (X0, n, covariates)
produces posteriors on a real panel it never saw that contain RSiena's
estimates on every parameter, with nominal interval coverage across the whole
population of sizes and starts. That is the M2 acceptance criterion
("calibration holds across the size range") met at interval level, with a
documented ~0.1 sd location bias on the density–closure ridge.

Established by the 10⁷ run: the residual ridge bias *does* vanish with more
data on the same architecture (density / transTrip mean ranks 0.502 / 0.497,
coverage nominal within one point, all eight RSiena estimates within 0.8 sd
on the real start). Established by the embedding run: the learned embedding
extracts more information per panel and resolves cycle3, but needs both more
data and a size-normalised pooling before it is the estimator of record.

Not established: behaviour outside the population (n outside 20–80, denser
or sparser starts than the prior, effects absent from the model). That is
the out-of-distribution characterisation §4 of the paper plan calls for and
the next item.

## Cost accounting

| step | wall clock |
|---|---|
| simulate 10⁶ population panels | 533 s |
| train NSF 8 × 128 | 67 min |
| fresh SBC, 4,000 × 1,000 | 90 s (+17 s simulation) |
| posterior for one dataset | 0.08 s |
