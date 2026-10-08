# A critical pass over the whole project

*2026-10-07. Deliberately adversarial: the question asked was not "is this defensible"
but "what is wrong with it". Findings are ordered by how much they change what we can
claim. Each was checked against the artefacts rather than argued from memory.*

## The core mathematics is sound

Taken first because it is the thing most worth being wrong about, and it is not.

The claim is that with a flat box prior the per-period two-wave posteriors multiply to
the joint posterior. Writing q_w for the two-wave posterior of period w,

    q_w  =  L_w · π_λ(λ_w) · π_β(β) / Z_w
    Π_w q_w  ∝  [Π_w L_w] · [Π_w π_λ(λ_w)] · π_β(β)^P
    joint    ∝  [Π_w L_w] · [Π_w π_λ(λ_w)] · π_β(β)

so the product over-counts the prior by π_β(β)^(P−1) = π_β(β)^(W−2), constant on a box.
Correct, and the exponent in the write-ups matches.

The step that deserves more scrutiny than it got is that `q_w` conditions on a *summary*
`s_w`, not on the wave pair, and `s_1` and `s_2` **share components exactly** — both
contain n, the covariate shape, and the statistics of x₁. Shared inputs would normally
break a product. They do not here, and the reason is worth stating because it is the same
reason M7 failed: in the training population the start network is drawn independently of
θ, so the flow learns a conditional in which the start block is *ancillary* — carrying no
information about θ — and `q_w ∝ L_w · π` holds. The product is then exact. Make the start
informative, as M7 did, and the same algebra breaks. The two facts are one fact.

## 1. "A third sharper than RSiena on influence and curvature" is mostly the prior

**This is in the abstract.** It does not survive the obvious control: comparing the
posterior not with RSiena but with the prior it started from.

Co-evolution model, averaged over the 19 Baerveldt classes:

| parameter | prior sd | our sd | **post/prior** | RSiena se | our/RSiena |
|---|---|---|---|---|---|
| density | 1.44 | 0.26 | **0.18** | 0.28 | 0.94 |
| recip | 1.44 | 0.39 | **0.27** | 0.40 | 0.97 |
| transTrip | 1.01 | 0.26 | **0.26** | 0.25 | 1.02 |
| cycle3 | 1.15 | 0.46 | **0.40** | 0.46 | 1.00 |
| quad | 0.58 | 0.31 | **0.53** | 0.47 | **0.65** |
| simZ | 1.44 | 1.05 | **0.73** | 1.25 | **0.84** |
| avAlt | 1.44 | 1.17 | **0.81** | 1.83 | **0.64** |

The parameters we claim to be sharper on are exactly the parameters the data barely
constrain. On the structural block the posterior is 18–40 % of the prior's width — the
data are doing the work — and we match RSiena to within a few per cent, which is the
meaningful result. On influence the posterior is **81 %** of the prior's width: the data
reduce the uncertainty by a fifth, and the rest is the box.

The decisive comparison: the prior U(−1, 4) has sd **1.44**, and RSiena's standard error
is **1.83**. *A posterior that ignored the data entirely would already be "sharper than
RSiena" on influence.* The claim conflates "the data are informative" with "we imposed a
bounded prior and RSiena's asymptotic interval is not bounded".

It is not simply truncation at the box face — splitting the 17 classes by whether the
posterior piles against the face gives sd ratios of 0.60 and 0.66 against 0.64 overall,
so the effect survives in classes clear of the boundary. The issue is the width of the
prior, not contact with its edge.

The network model is **clean**: posterior 20–39 % of prior width on every parameter, and
our/RSiena 0.96–1.14. The problem is confined to the behaviour block, which is exactly
where §4.3 already reports a ridge the data do not resolve. The two findings are the same
finding seen from different sides, and the paper currently presents one as a strength and
the other as a caveat.

**What to do.** Stop claiming sharpness on influence and curvature. The honest statement
is: on the structural mechanisms, where the data dominate, two unrelated methods agree on
both the estimate and its precision; on influence and curvature the data are weak enough
that the bounded prior, not the likelihood, sets the interval width, and neither method's
interval should be read as a measurement of how well those parameters are determined.

## 2. The Knecht "periods disagree" flags are not significant

`period_spread` flags an effect when the largest standardised gap between period pairs
exceeds 2. On the four-wave Knecht classroom that is 3 period pairs × 7 effects = **21
comparisons**, where chance alone gives 0.96 flags at a 2σ threshold. We observed two
(transTrip 2.06, egoX 2.35), and P(≥2 by chance) = **0.25**.

`docs/MULTIWAVE.md` calls these "a statement about this classroom and not about the
method". They are not a statement about anything. The diagnostic itself is worth keeping
— a joint estimator genuinely cannot produce it — but its threshold is uncorrected for
multiplicity and should either be raised or reported with the comparison count beside it.

## 3. The evidence that "splitting a panel into periods costs nothing" is confounded

The only direct comparison is the product of `npe_m5c` (two waves, 10⁷ panels) against
`npe_m4b` (three waves, **10⁶** panels) on s50. The product won on precision. The docs
note the budget difference and conclude "splitting costs nothing measurable here" — but a
tenfold training-budget gap is precisely the sort of thing that could mask a real cost in
either direction.

**Correction, same day.** This section went on to propose comparing the co-evolution
product against `npe_coev_m4_10m` at matched budget, and called it "far better" than what
the paper has. On checking the two model cards rather than remembering them, that is wrong.
The populations differ in n range ([30, 100] vs [20, 200]), rate prior (U(1, 12) vs
U(1, 20)), effect box (**sparse** vs **default**) and start regime (**survey** vs
**homophilous**) — four confounds, two of which this audit did not notice when it proposed
the test. Trading one confound of known direction for four of unknown direction is not an
improvement.

No pair of existing artefacts gives a clean test: every three-wave estimator here was
trained on an m4-family population and every 10⁷ two-wave estimator on an m5c-family one.
The clean test needs a run, which is pre-registered in `docs/PRODUCT_VS_NATIVE.md` and
generating now — a three-wave network estimator on the M6 population, differing from
`npe_m6` in wave count and nothing else, compared against `npe_m6`'s product on one shared
set of panels (`sbc_multiwave.py --native`).

Note what the §1 finding and this one have in common: both are cases where a comparison was
made against the wrong baseline, and in both the error flattered the method.

## 4. We return 10,000 draws regardless of effective sample size

`combine()` resamples `draws` points from the importance proposal with replacement
whatever the ESS. On the co-evolution model ESS is routinely 900–3,500, and on the
four-wave Knecht panel it was **867** — so a saved file advertising 8,000 "samples"
carries fewer than 900 independent ones. The ESS is printed and stored, but a user who
computes a 95 % interval from the array and treats n = 8,000 will understate its Monte
Carlo error by roughly threefold.

The samples are not wrong, and self-normalised importance resampling is the standard
construction. But the output should carry the effective size next to the nominal one, and
`saom-fit` should probably refuse to write a file whose ESS is below some floor without
saying so loudly.

**Fixed, and it was worse than written above.** In the Metropolis branch the ESS being
reported was the *importance* ESS — the number that had just failed the `min_ess` test and
caused the fallback. It is a statement about the proposal that was abandoned and says
nothing about the chain that replaced it, so every "metropolis, ESS 867" line in this
project's output was mislabelled. `chain_ess()` now computes the real thing (Geyer's
initial-positive-sequence estimator, checked against the analytic AR(1) value
N(1−ρ)/(1+ρ) to within 4 %), `combine()` reports it as `ess` and keeps the importance
figure as `ess_importance`, and `saom-fit` prints *N draws carrying M effective*, adds an
`mcse` column so the unreal decimals are visible, and warns on write when M is low.

## 5. A single unused import was skipping a whole test module

Found while fixing §4. `tests/test_multiwave.py` began its sampler section with

    torch = pytest.importorskip("torch")

`torch` is then never used anywhere in the file. Because `importorskip` raises at *module*
scope, that line skipped **all 22 tests in the file** on any machine without torch — not
just the sampler tests below it, but the ten period-view tests above it, which are the ones
that check the implementation of the factorisation this project rests on. Nothing in the
file needs torch: the sampler imports it lazily and these tests exercise the numpy half.
The suite reported one tidy `SKIPPED [1]` line and looked healthy.

Removed, and the 22 tests pass. The lesson is narrower than it looks: a skip is not a pass,
and a suite that prints its skips as a count rather than a list hides how many. The other
`importorskip` calls were checked and are correct — `test_embedding.py` genuinely needs
torch, `test_gof.py` genuinely needs networkx for the triad-lookup cross-check.

## 6. Smaller points, recorded rather than acted on

**The population stage treats per-class posteriors as Gaussian**, summarising each by a
mean and an sd. Eight of nineteen classes have influence posteriors against the prior
face, so those are bounded and skewed, and a normal–normal stage over their moments is an
approximation whose error is unquantified. It also discards the within-class correlation
between parameters — notably the influence–curvature ridge, which is the strongest
correlation in the model.

**"152 of 152 inside the 90 % interval" is not a calibration statement.** Two estimators
applied to the same data are strongly correlated, so the expected hit rate is not 90 %;
under independence with comparable precisions it would be about 75 %, and 100 % reflects
that correlation rather than interval accuracy. The paper frames it as agreement, which is
right, but the number invites the other reading and should say which it is.

**The Mahalanobis goodness-of-fit statistic is computed in-sample**, with each simulated
distance using a covariance estimated from a set including that point. This biases the
simulated distances slightly small and the p-value slightly low. At B = 2,000 the effect
is negligible; `sienaGOF` does the same thing.

**2–3.5 % of SBC panels fall back to Metropolis** with ESS below 500 (7 to 14 of 400).
Their ranks come from short adaptive chains and may be unreliable, which adds noise to
every joint SBC figure reported, in an unknown direction.

## What came through unscathed

Worth stating, since an audit that only finds fault is not an audit.

- The factorisation and its prior-exponent correction are right, including the
  summary-sharing subtlety that looked like a problem and is not.
- The simulator's parity with RSiena: exact on target statistics, within 5 % on the
  spread of simulated statistics, with brute-force references for every change statistic.
- The structural-parameter agreement with RSiena across 19 classes, which the control in
  §1 above strengthens rather than weakens: those posteriors are 20–40 % of prior width,
  so the agreement is between two data-driven estimates.
- The 10⁶ → 10⁷ calibration finding, independently reproduced on several populations.
- The goodness-of-fit result and the `outTrunc` follow-up, where the negative results were
  found by testing rather than assumed.
