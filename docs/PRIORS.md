# Prior specification for the s50 amortized estimator

Written before any training data is generated (GETTING_STARTED §7, item 4).
Everything the estimator can later be trusted for is decided here. Code:
`saomsim/prior.py` (generic), `benchmarks/s50.py` (this case),
`benchmarks/prior_predictive.py` (the check below).

## 1. What is being amortized over

**Decision: fixed empirical start.** The start network `X0` is the observed
first wave (`s501`, n = 50) and is *conditioned on*, exactly as in the SAOM
itself, where the first observation is never modelled. The covariates
(`alc` centred, `smk`) are likewise fixed. The estimator therefore learns

    p(theta | X1, X0 = s501, covariates)

and is specific to this dataset's first wave, actor set and covariates. It is
*not* a general SAOM estimator; a new dataset means a new training set (at
122k panels/s on the GB10 that is seconds, not hours, so this is the right
trade for a first estimator). Amortization is over `theta` and the process
randomness only.

Alternatives considered and set aside for now: random `X0` from a density
prior (broader estimator, but the posterior would then also be over a start
distribution no real analysis has); a stationary draw at `theta` (changes the
model — SAOM panels are not stationary draws).

## 2. Parameters and the model

`theta = (rate, density, recip, transTrip, cycle3, altX(alc), egoX(alc), sameX(smk))`,
eight dimensions, the model validated against RSiena 1.6.6 in `benchmarks/`.
The rate is estimated (unconditional simulation, Poisson(n · rate)
ministeps); the tie-change count is the corresponding statistic. This is the
`cond = FALSE` set-up whose simulator matched RSiena's in
`test_rsiena_dynamics.py`.

## 3. The prior box

Independent uniforms. A box is honest about what we know (typical ranges from
the SAOM literature and the s50 data), easy to reason about, and makes
simulation-based calibration straightforward. The box is the estimator's
**domain of validity**: posterior mass piling against a face means the data
want a value outside it and the estimator must not be trusted there.

| parameter | prior | reasoning |
|---|---|---|
| rate | U(1, 12) | Expected ministeps per actor. Observed 115 changes on 50 actors need ≥ 2.3 net changes per actor; back-and-forth toggles push the rate above the Hamming/n ratio. RSiena fits on s50 sit around 5–7. Below 1 the observed change count is essentially unreachable; above ~12 the process is close to stationarity and the rate stops being identified from one period (the change statistic saturates). |
| density | U(−4, 0) | Sparse friendship networks: RSiena estimates typically −1.5 to −3.5. Positive density with any positive closure effect drives the network toward complete; s502 has tie fraction 0.047. |
| recip | U(−1, 4) | Almost always strongly positive in friendship data (1.5–3). Negative values are allowed so the estimator can *say* "no reciprocity" rather than being forced to find it. |
| transTrip | U(−0.5, 1.5) | Typical estimates 0.2–0.8. The upper end produces heavily clustered networks (see §4) but not degenerate ones; it is kept so the estimator covers strong closure. |
| cycle3 | U(−1.5, 0.5) | Typically negative (local hierarchy, −0.2 to −0.8). The positive part covers "no hierarchy". |
| altX(alc), egoX(alc) | U(−1, 1) | Covariate centred, range about ±2, so an effect of 1 shifts an actor's log-odds by up to 2 across the covariate range — already large. Published values are usually |β| < 0.5. |
| sameX(smk) | U(−1, 2) | Homophily on a three-category covariate; typical 0.3–1. Negative allowed for heterophily. |

## 4. Prior predictive check (2026-09-11)

`python benchmarks/prior_predictive.py --N 50000 --backend torch` on the
Spark; 50,000 panels from s501, nothing filtered.

| summary | q01 | q05 | median | q95 | q99 | observed s502 | F(obs) |
|---|---|---|---|---|---|---|---|
| changes | 46 | 64 | 181 | 488 | 563 | 115 | 0.200 |
| density | 20 | 35 | 215 | 585 | 669 | 116 | 0.257 |
| recip | 0 | 0 | 94 | 372 | 488 | 70 | 0.397 |
| transTrip | 0 | 0 | 205 | 3998 | 5863 | 88 | 0.357 |
| cycle3 | 0 | 0 | 39 | 951 | 1600 | 28 | 0.441 |
| altX(alc) | −406 | −250 | 5 | 258 | 435 | −4.08 | 0.456 |
| egoX(alc) | −188 | −122 | 2 | 111 | 176 | 2.92 | 0.505 |
| sameX(smk) | 8 | 18 | 145 | 414 | 489 | 83 | 0.288 |
| outdeg_sd | 0.54 | 0.80 | 2.26 | 4.43 | 5.38 | 1.28 | 0.159 |
| indeg_sd | 0.64 | 0.91 | 2.83 | 8.88 | 12.07 | 1.54 | 0.189 |
| isolates | 0 | 0 | 0 | 15 | 23 | 2 | 0.653 |
| mutual_dyads | 0 | 0 | 47 | 186 | 244 | 35 | 0.397 |
| tie_fraction | 0.01 | 0.01 | 0.09 | 0.24 | 0.27 | 0.047 | 0.257 |

Readings:

- **Coverage.** Every observed statistic falls between the 16th and 65th
  percentile of its prior predictive. The prior is not fighting the data.
- **Degeneracy: none.** 0 empty, 0 complete, 0 panels denser than 0.5 in
  50,000; the densest is 0.32. With density ≤ 0 and this rate range the box
  does not reach the explosive regime, so the "never filter" rule costs
  nothing here.
- **Heavy tails in count statistics.** 25 % of draws have a transTrip
  statistic above 1000 (observed: 88), coming from the corner transTrip > 1,
  rate > 8, density > −1.5: dense, clustered but non-degenerate networks.
  The statistic is cubic in ties, so this is expected. Consequence for the
  estimator, not the prior: **standardise count summaries with `log1p`** (or
  a quantile transform) before they reach the flow; z-scoring alone will let
  that corner dominate the embedding.
- **The prior is centred denser than the data** (median tie fraction 0.09 vs
  0.047). Acceptable: the data sit at the 26th percentile, well inside, and
  narrowing density to keep sparse networks would cut off the hypotheses the
  estimator should be able to reject.

## 5. What this commits us to

- Training data are drawn **only** from this box, from `s501`, with these
  covariates. Report the box with every result.
- SBC (item 6) is run over this same prior. Calibration outside the box is
  undefined and should not be claimed.
- Changing any range means regenerating training data *and* re-running SBC;
  edit `benchmarks/s50.py` and this file together.
- Summary vector for the first estimator: the 13 columns above
  (`saomsim.prior.summaries`), log1p-transformed counts. Raw networks are
  stored optionally (`keep_networks=True`) for a later learned embedding.
