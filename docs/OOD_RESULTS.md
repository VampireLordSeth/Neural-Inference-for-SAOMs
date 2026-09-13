# Out-of-distribution behaviour of the M2 estimator (paper plan §4)

Estimator: the 10⁷-panel summary-only M2 model (`data/npe_m2_10m.pt`,
`docs/M2_RESULTS.md`). Script: `benchmarks/ood_m2.py`. 1,000 panels per set,
1,000 posterior samples each; run 2026-09-13, 3.5 min on the Spark.

The question §4 asks: when a real dataset falls outside the training
population, **how** does the posterior fail, and **can the user tell**?

## The sets

| set | what leaves the population | true θ in the box? |
|---|---|---|
| id | nothing (control) | yes |
| n_small | n = 15 (population 20–80) | yes |
| n_large | n = 110 | yes |
| dense_x0 | Erdős–Rényi start at tie fraction 0.35 (population ER 0.02–0.20; burnt-in starts reach ~0.5) | yes |
| covariates | v with sd 3 (population ≈ 1); g with 6 categories (population 2–4) | yes |
| theta_out | rate 14, density −4.5, transTrip 1.8 (box: ≤ 12, ≥ −4, ≤ 1.5) | **no** |
| hidden_hom | homophily β = 1.5 on an attribute the analyst does not observe | not in the fitted model |

## Detection heuristic

For the applied user: transform the observed 29-summary vector as in
training, z-score each against 10⁶ training panels, take the maximum |z|
(`maxz`). A robust variant maps each summary to its empirical percentile in
the training set and probit-transforms it (`robust`), which is what catches
a summary that is simply *outside the training range* (here: n). Threshold at
the control's 95th percentile (max|z| 3.93; robust 4.89).

| set | flagged (max\|z\|) | flagged (robust) | median max\|z\| |
|---|---|---|---|
| id | 0.050 | 0.000 | 1.72 |
| n_small | 0.254 | 0.044 | 3.26 |
| n_large | 0.167 | **1.000** | 2.20 |
| dense_x0 | 0.018 | 0.016 | 2.01 |
| covariates | **1.000** | **1.000** | 9.93 |
| theta_out | 0.309 | 0.504 | 3.48 |
| hidden_hom | 0.038 | 0.000 | 1.65 |
| **s50 (real)** | no | no | 1.41 |

## How the posterior fails, set by set

**n_small (15 actors): the estimator knows it does not know.** Coverage
stays nominal at every level (all within 1 point); mean ranks 0.48–0.53. The
posterior simply widens — sd × 3.8 for rate, × 2.3–2.5 for recip, transTrip,
cycle3, × 2 for altX and sameX. Calibrated and honest, five actors below the
training range. Detection is partial (25 %) because a 15-actor panel's
summaries mostly stay inside the training range. *Not dangerous.*

**n_large (110 actors): mild bias, still usable, always flagged.** Coverage
within 1–2.5 points (density 90 % → 0.883, transTrip → 0.874); mean ranks
show the cycle3 drift seen in-distribution grow (0.436) and density tilt low
(0.459). Posteriors slightly *narrower* than the control (× 0.8). The robust
detector flags every panel because n itself is outside the training range.
*Flagged, mildly degraded.*

**dense_x0 (tie fraction 0.35): modest degradation, mostly undetected.**
Coverage 86–87 % at nominal 90 % for density, recip, transTrip; the density
posterior widens × 2.5. Only 2 % flagged: burnt-in training starts can be
this dense, so the summaries are not unusual — the start is *near* the
population edge, not outside it, and the modest degradation is consistent
with that. *Honest degradation, borderline in-distribution.*

**covariates (sd 3, 6 categories): the dangerous case — and it is caught.**
Coverage collapses: altX 90 % interval covers 42 %, transTrip 55 %, egoX
55 %, cycle3 72 %, density 71 %. Posteriors are *confidently wrong* (altX and
egoX sd shrink to 0.66 × control). This is exactly the silent failure the
paper plan worries about — and **100 % of these panels are flagged**, with a
median max|z| of 9.9 against a threshold of 3.9. The covariate-shape
summaries (v_sd, g_ncat, g_entropy) make it loud. *Detected; the user must
rescale covariates to the training convention (unit sd, ≤ 4 categories) or
retrain.*

**theta_out (true θ beyond the box): the face warning works.** 89 % of the
rate posterior mass sits within 5 % of the upper face (posterior mean 11.7,
truth 14); 33 % for transTrip (1.21 vs 1.8), 27 % for density (−2.89 vs
−4.5 — note the density posterior does not reach its face: the estimator
attributes the sparse outcome partly to other parameters). The rule from
`docs/PRIORS.md` — mass piling against a face means the data want a value
outside the box — is confirmed as a usable signal. Detection 31–50 %.
*Visible; widen the prior and retrain.*

**hidden_hom (unobserved homophily): the genuinely silent failure.** The
density posterior is biased **+2.0 sd** (unobserved clustering is read as a
denser process); cycle3 +0.5 sd, recip +0.4 sd; the covariate effects are
unaffected. Posterior widths are normal, nothing piles at a face, and only 4 %
are flagged — the summaries stay in-range because the extra homophily
produces networks that look like *some* in-population network. **The
training-population percentile cannot detect model misspecification**; only
a posterior predictive check on graph-space statistics outside the embedding
can (`benchmarks/ppc_s50.py` does this for M1; the excess clustering would
show up in transitivity / mutual-dyad checks). *Undetected by percentile
screening; this is what the PPC step is for.*

**s50** passes as in-distribution (max|z| 1.41, below the control median).

## What to tell the user of the estimator

1. Run the percentile screen. If it flags, do not report the posterior: check
   n against 20–80, rescale covariates to unit sd and ≤ 4 categories, or
   retrain with a population that covers the data.
2. If any parameter's posterior piles against a prior face, widen the box and
   retrain.
3. Always run a posterior predictive check on statistics outside the
   embedding. It is the only defence against misspecification, and it is the
   same defence RSiena users have (`sienaGOF`).
4. Small networks (down to 15) give wide but honest posteriors; large ones
   (~110) give slightly over-confident ones and are always flagged.

## For the paper

This is the "documented operating envelope" of §4 and the risk register's
"amortized posterior fails silently out of distribution" row. Of six ways to
leave the population, four are detected by a one-line percentile screen
(covariate scale, network size, and half of out-of-box θ) or by the face
warning; one (dense start) degrades mildly and honestly; one (misspecification)
is silent to screening and requires posterior predictive checking — the same
limitation every likelihood-based method has, and one the amortized posterior
makes cheap to check because simulating from it costs nothing.
