# M5 prior: a two-wave population for a multi-group school study

Written 2026-09-21, before generating the training set. Target: the 19 schools
of the Dutch Social Behavior study (Chris Baerveldt; Snijders & Baerveldt
2003), the reference multi-group SAOM analysis — two waves, n = 31–91,
emotional-support ties, delinquency at both waves, sex. Data and per-school
RSiena fits in `benchmarks/baerveldt/` (`prepare_and_fit.R`; source
`CB_data.zip` from the Siena site). This is the first real dataset with many
networks, and the first two-wave one since M2; the network-only and
co-evolution estimators of M4 are three-wave and cannot read it.

## What the schools look like

From `benchmarks/baerveldt/schools.csv` (wave 1 unless stated):

| | range over the 19 schools | our sparse regime (M4) |
|---|---|---|
| n | 31–91 | 20–200 |
| mean out-degree | 0.84–2.38 (wave 2: 1.00–3.42) | 2–10 on the sparse half |
| isolates (no out-ties) | 22–43 % | — |
| max out-degree | 3–11 (survey allowed 12) | cap ceil(k)+U{1..3} on a quarter |
| reciprocity (mutual / ties) | 0.38–0.67 | — |
| transitivity (closed / two-paths) | 0.33–0.52 | — |
| Jaccard(w1, w2) | 0.23–0.38 | — |
| RSiena rate | 2.9–9.7 | U(1, 20) |
| delinquency | 5 categories (0–4), both waves | 3–5 categories |

Two things are outside every population we have trained: the mean degree,
which sits *below* the floor of the M4 sparse regime for 18 of 19 schools,
and the second wave being the last. Everything else (sizes, rates, effect
magnitudes, a binary and a continuous covariate, a 5-category behaviour) is
inside.

## The population

Two estimators, both two-wave (one period, one network rate), generated with
`--waves 2 --n-min 30 --n-max 100 --start survey --rate-max 12`:

| component | value |
|---|---|
| n | U{30..100} |
| start regime | `survey` (`population.sample_start_networks`, `population_coev._start_networks`): every start draws a mean degree k ~ U(0.5, 6), ER seed at d = k/(n−1); 50 % burnt in for 10·n ministeps at θ₀ ~ prior (structural effects only for the co-evolution population); 50 % capped at ceil(k) + U{1..3} out-ties after the burn-in |
| waves | 2 |
| network effects and box | M2/M4: rate U(1, 12); density U(−4, 0); recip U(−1, 4); transTrip U(−0.5, 1.5); cycle3 U(−1.5, 0.5); network-only adds altX(v), egoX(v) U(−1, 1), sameX(g) U(−1, 2) |
| covariates (network-only) | M2: v 50 % N(0,1) / 50 % Likert K ∈ {3,4,5}, centred; g K ∈ {2,3,4} groups. For the schools v = ln(offences+1) centred, g = sex |
| co-evolution | M3b: rate_beh U(0.3, 6); egoZ, altZ U(−1, 1); simZ U(−1, 4); linear U(−1.5, 1.5); quad U(−1.5, 0.5); avAlt U(−1, 4); behaviour on 3–5 categories, one period. Delinquency 0–4 is passed as 1–5 |
| summaries | the M2 two-wave set (network-only) and the M3b set at two waves (co-evolution), as the code defines them for `waves=2` |
| budget | 10⁶ first (a look), then 10⁷ — the rule from M4: a new population is not judged at 10⁶ |

## Why these choices

- **Mean degree 0.5–6, all starts.** The schools span 0.8–3.4; the lower end
  has to be covered generously because a start with mean degree 1 at n = 31
  has 30 ties and a third of the actors isolated, and the burn-in half will
  add or remove ties according to θ₀. Nothing above 6 is needed for this
  design; the M4 mixture kept a dense half for small dense groups, which no
  school here is. One regime, not a mixture, so the whole budget goes where
  the data are.
- **Cap on half.** The survey allowed 12 nominations and no pupil used more
  than 11, so the cap rarely binds at these densities; keeping it on half the
  starts keeps the out-degree spread of capped surveys in the population
  (Glasgow's lesson, `docs/PRIORS_M4.md`).
- **n 30–100.** The schools are 31–91; the margins cost little at two waves
  (the whole 10⁶ set is minutes on the Spark, `docs/SCALING.md`).
- **Rates to 12.** The largest RSiena rate is 9.7 (school 14); M2's box.
- **Two separate estimators** rather than one with covariates *and* a
  behaviour: the RSiena references are fitted in exactly these two layouts,
  and the 2003 paper's model is the network-only one (delinquency as a
  covariate with ego, alter and similarity effects — we have ego and alter).
- **Two waves, not "wave-generic".** The estimator's summary vector is
  per-period; a wave-generic design is a different piece of work. The Knecht
  classroom (`benchmarks/knecht/`, four waves) is read on three-wave windows
  with the M4 estimators (`docs/M4_RESULTS.md`).

## What the schools test

1. Nineteen posteriors from one estimator, each against its RSiena fit, with
   the screen: the first time the in-distribution claim is made on many real
   networks at once.
2. A population stage: the 19 posteriors combined into a random-effects model
   (μ, τ per parameter) against `siena08` on the RSiena fits
   (`benchmarks/baerveldt/siena08_network2w.csv`) and against Snijders &
   Baerveldt (2003).
3. Co-evolution with two waves: whether influence is identified at all from
   one period (RSiena diverges on two of the schools), and whether the
   selection/influence reads keep the Glasgow offset.

## What would make it wrong

- Starts whose reciprocity and transitivity are far below the schools'
  (0.4–0.7 and 0.3–0.5): ER seeds have neither, and only the burnt-in half
  gains them. The screen on x0 summaries will say; if the schools sit in the
  tails there, the fix is a burn-in on every start, or a larger share.
- Delinquency's distribution: 0–4 with most mass on 1–2 (skewed); the
  behaviour population draws its starting distribution from the generic M3b
  scheme. Screened by zbar and the behaviour summaries.

## Cost

Two waves at n ≤ 100: about 15 min per 10⁶ on the Spark for the network-only
set, ~30 min for co-evolution; 10⁷ at ten shards each, 3–5 h; training 2–4 h
per estimator at 10⁶ settings, ~20 h at 10⁷. Two days of Spark for all four
estimators, queued, never overlapping with SBC.
