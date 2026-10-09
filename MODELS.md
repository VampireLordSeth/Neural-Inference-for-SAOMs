# The trained estimators

An amortized estimator is an instrument, and like any instrument it has a range outside
which its readings mean nothing. This file says, for each one, what it was trained on,
what it returns, and how well calibrated it was found to be. **Read the range before the
estimate.** Outside it the posterior does not become noisy — it becomes confident and
wrong, shrinking toward the training population rather than toward your data.

The estimator files are not in the git repository (they total 36 MB and are build
artefacts, not source). See **Getting the files** at the bottom.

Common to all of them: a masked autoregressive neural spline flow, eight transforms of
128 hidden units, batch 1,024, learning rate 5e-4, early stopping after twenty epochs
without improvement on a held-out tenth. Priors are independent uniforms on a box, which
is what makes the per-period product of §`docs/MULTIWAVE.md` exact. Nothing simulated was
discarded in training: degenerate panels stay in, so the flow learns that region rather
than being shielded from it.

---

## `npe_m7.pt` — two waves only, n 20–150 *(do not use for multiwave)*

`npe_m6.pt` plus **evolved starts**: half the training panels have their start network
simulated forward one period at the panel's own effects. As a *two-wave* estimator it is
the best calibrated in this project — 0 of 8 rank distributions rejected on 4,000 fresh
draws (smallest p 0.102), mean ranks 0.493–0.508, 90 % coverage 0.893–0.904 — and it fixes
the per-period tilt that every other estimator here shows on an evolved start.

**It is nevertheless the wrong estimator for reading a panel of more than two waves**, and
that is a result rather than an oversight. Making the start informative about β is exactly
what the per-period product forbids: the start then enters q_w twice, once through the
previous period's likelihood and once through the flow reading β out of it, and the
product over-concentrates. Its joint 90 % coverage is 0.854 against `npe_m6`'s 0.885, with
all nine parameters below nominal. `docs/PRIORS_M7.md` has the mechanism and the evidence.

Use it for a genuine two-wave panel. Use `npe_m6.pt` for anything longer.

## `npe_m6.pt` — network model, two waves, n 20–150 *(the widest)*

Same population as `npe_m5c.pt` in every respect except the size range, so the two are
directly comparable and any difference is attributable to **n** alone. Use this one when
your panel falls outside [30, 100]; use `npe_m5c.pt` when it does not, because its
calibration on the narrower range is better.

| | |
|---|---|
| **returns** | 8 parameters, as `npe_m5c.pt` |
| **actors** | n ∈ **[20, 150]** |
| **priors, starts, covariates** | identical to `npe_m5c.pt` below |
| **training** | 10⁷ panels in ten shards, 5 h 34 m to generate (480–547 panels/s), **10 h 56 m** to train |
| **calibration** | 4,000 fresh draws: **3 of 8 rank distributions rejected** (sameX 0.001, recip 0.003, density 0.034), mean ranks 0.487–0.515, but coverage nominal — 90 % 0.889–0.901, 95 % 0.943–0.951 against binomial se 0.005 and 0.003, with no drift across n bands |
| **read this** | at the same 10⁷ budget, `npe_m5c.pt` on [30, 100] rejects **0 of 8**. A wider size range is a harder learning problem and the same budget buys less calibration when spread over it. The tilts are small (≈0.015 in mean rank) and coverage does not suffer, so the estimator is usable — but on [30, 100] prefer `npe_m5c.pt` |
| **against RSiena** | Knecht klas12b, four waves, n = 25: largest \|z\| **0.62** over ten parameters (against 1.04 when `npe_m5c` had to extrapolate). Glasgow, three waves, n = 129: largest \|z\| 1.62, on the two rates, which read low |

## `npe_m5c.pt` — network model, two waves *(the default on [30, 100])*

The one to use when your panel is in its range. It reads **any number of waves** through
`saom-fit`, because the per-period posteriors multiply (`docs/MULTIWAVE.md`).

| | |
|---|---|
| **returns** | 8 parameters: one rate per period, then density, reciprocity, transitive triplets, three-cycles, altX, egoX, sameX |
| **actors** | n ∈ **[30, 100]** |
| **rate prior** | U(1, 12) per period |
| **effects prior** | density U(−5, 0), recip U(−1, 4), transTrip U(−0.5, 3), cycle3 U(−3.5, 0.5), altX/egoX U(−1, 1), sameX U(−1, 2) |
| **start networks** | `survey`: mean degree k ~ U(0.5, 6), half burnt in for 10n ministeps, half capped at ⌈k⌉ + U{1..3} out-ties |
| **covariates** | one numeric (half standard normal, half Likert 3–5 points, centred), one categorical with 2–4 groups |
| **training** | 10⁷ panels in ten shards, 2 h 21 m to generate, 10 h 54 m to train |
| **calibration** | SBC on 4,000 fresh draws: **0 of 8 rank distributions rejected** (smallest KS p 0.102), mean ranks 0.494–0.507, 90 % coverage 0.889–0.904 and 95 % 0.938–0.953 against a binomial se of 0.005 |
| **against RSiena** | 19 Baerveldt school classes, **152 of 152** parameters inside the 90 % interval, largest \|z\| 1.47; posterior sds sit on RSiena's own standard errors |
| **known weakness** | on a panel of more than two waves the *later* periods start from an evolved network, which this population does not contain, and rate and density are read slightly low there (rank 0.528). See `docs/MULTIWAVE.md` §"Where the degradation comes from" |

## `npe_coev_m6.pt` — network × behaviour, two waves, n 20–150 *(the widest)*

The co-evolution counterpart of `npe_m6.pt`: the `npe_coev_m5c.pt` population widened in
the size range alone. Use it when your panel falls outside [30, 100]; inside it, prefer
`npe_coev_m5c.pt`, whose calibration is better.

| | |
|---|---|
| **returns** | 12 parameters, as `npe_coev_m5c.pt` |
| **actors** | n ∈ **[20, 150]**; behaviour on 3–5 ordered categories |
| **priors, starts** | identical to `npe_coev_m5c.pt` below |
| **training** | 10⁷ panels, **7 h 14 m** to generate (369–404 panels/s), **19 h 52 m** to train |
| **calibration** | 4,000 fresh draws: **3 of 12 rejected** (cycle3 0.000, avAlt 0.002, transTrip 0.004), mean ranks 0.481–0.516; 90 % coverage 0.884–0.905 and 95 % 0.939–0.952 against binomial se 0.005 and 0.003. No drift across n bands — every band's mean rank is within 0.475–0.527 from n = 20 to n = 150 |
| **read this** | `npe_coev_m5c.pt` on [30, 100] rejects 1 of 12 at the same budget. The same trade as the network model: a wider size range is a harder learning problem and 10⁷ buys less calibration spread over it |
| **against RSiena** | Knecht klas12b, **four waves**, n = 25, 16 parameters with delinquency co-evolving: largest \|z\| **1.23** (1.09 when `npe_coev_m5c` had to extrapolate). Note the product sampler struggles here — 867 effective draws — so the `--cross-check` Metropolis run is doing the work; it agrees to 0.091 sd with max split-Rhat 1.037 |

## `npe_coev_m5c.pt` — network × behaviour, two waves

| | |
|---|---|
| **returns** | 12 parameters: network rate and behaviour rate per period, then density, recip, transTrip, cycle3, egoZ, altZ, simZ (selection), linear, quad, avAlt (influence) |
| **actors** | n ∈ **[30, 100]**; behaviour on 3–5 ordered categories |
| **priors** | as above, plus behaviour rate U(0.3, 6), simZ U(−1, 4), linear U(−1.5, 1.5), quad U(−1.5, 0.5), avAlt U(−1, 4) |
| **training** | 10⁷ panels, 3 h 22 m to generate, 20 h 40 m to train |
| **calibration** | 90 % coverage 0.890–0.906, 95 % 0.938–0.952; KS rejects 1 of 12 (quad, p 0.004, mean rank 0.514) |
| **against RSiena** | 17 Baerveldt classes where RSiena converges, **204 of 204** within 1.645 combined sd, largest 1.52. Matches RSiena's precision on the structural block (sd ratio 0.94–1.02). Our intervals are narrower on influence (0.64) and curvature (0.65), but **read `docs/AUDIT.md` §1 before quoting that**: those posteriors are 81 % and 53 % of their *prior's* width, so the bounded prior sets most of the interval and a data-free posterior would already beat RSiena's 1.83 standard error on influence |
| **known weakness** | the product sampler is inefficient here (~1,500–3,500 effective draws) because of the influence–curvature ridge. Use `--cross-check` in `benchmarks/multiwave.py`; importance sampling and Metropolis agree to 0.08–0.14 sd, so it is efficiency and not bias |

## `npe_m4b.pt` — network model, three waves, larger networks

Superseded by `npe_m5c.pt` for two- and three-wave panels, but it covers a wider **n**.

| | |
|---|---|
| **returns** | 9 parameters: two rates, then the same seven effects |
| **actors** | n ∈ **[20, 200]** |
| **rate prior** | U(1, 20) per period; effects on the wider "default" box |
| **training** | **10⁶** panels — a tenth of the budget above, and it shows (`docs/M4_RESULTS.md`) |
| **against RSiena** | Glasgow, 129 pupils: all nine parameters agree, 0.06 s against RSiena's 219 s |

## `npe_m8.pt` — network model, three waves, the M6 population

Built to answer one question rather than to be used: it is `npe_m6.pt`'s population with
`--waves 3` and nothing else changed, so that comparing it against `npe_m6`'s product
isolates the wave count (`docs/PRODUCT_VS_NATIVE.md`). It is also, incidentally, the
best-calibrated three-wave network estimator here.

| | |
|---|---|
| **returns** | 9 parameters: two rates, then density, recip, transTrip, cycle3, altX(v), egoX(v), sameX(g) |
| **actors** | n ∈ **[20, 150]**; exactly **three** waves |
| **priors** | identical to `npe_m6.pt` — rate U(1, 12) per period, sparse box, survey starts |
| **training** | 10⁷ panels, 8 h 30 m to generate, 18 h 12 m to train, 202 epochs, best validation loss −0.743 |
| **calibration** | KS rejects **2 of 9** (cycle3 p 0.017, altX 0.047); 90 % coverage 0.896–0.906; mean ranks near 0.5 in every n band from 20 to 150 |
| **against RSiena** | s50, three waves, n = 50, a real start never seen in training: all nine parameters within **\|z\| ≤ 0.33**, in 0.07 s |
| **what it showed** | reading the same three-wave panels as a product of two-wave posteriors costs **0.0156 of 90 % coverage** (paired se 0.0046) and yields posteriors 2 % narrower — mild over-confidence. Not the sampler: flat across ESS quartiles. See `docs/PRODUCT_VS_NATIVE.md` for why the evolved-start shift is the better explanation than the training budget |
| **use it** | only for exactly three waves. For any other length use `npe_m6.pt` and the product |

## `npe_coev_m4_10m.pt` — network × behaviour, three waves, larger networks

| | |
|---|---|
| **returns** | 14 parameters: two network rates, two behaviour rates, ten effects |
| **actors** | n ∈ [20, 200] |
| **training** | 10⁷ panels |
| **calibration** | 90 % coverage 0.889–0.906, 95 % 0.941–0.956; KS rejects 6 of 14 on mean-rank tilts of 0.02–0.03 in no consistent direction — at the resolution of a 4,000-draw test |

## `npe_m5c_mom.pt` — the ablation, not for use

Identical to `npe_m5c.pt` but conditioned on 19 summaries instead of 29, dropping the
descriptors no estimator targets. Kept because the comparison is a result
(`docs/M5_RESULTS.md`): the moment conditions are sufficient — posteriors are 2 %
*narrower*, not wider — but training took 37 % longer and ended with two KS flags instead
of none. Use `npe_m5c.pt`.

## Others

`npe_m5.pt`, `npe_m5b.pt`, `npe_coev_m5.pt`, `npe_coev_m5b.pt`, `npe_coev_m4.pt`,
`npe_coev_m4b.pt` are earlier steps kept for the record. They are trained at 10⁶ or on
prior boxes later found too narrow, and `docs/M4_RESULTS.md` and `docs/M5_RESULTS.md`
explain what each one established. Do not use them for new data.

---

## Choosing one

| your data | estimator |
|---|---|
| 30–100 actors, network only, any number of waves | `npe_m5c.pt` |
| 20–150 actors, network only, any number of waves | `npe_m6.pt` (outside [30, 100]; inside it, prefer `npe_m5c.pt`) |
| 30–100 actors, network and a 1–5 behaviour, any number of waves | `npe_coev_m5c.pt` |
| 20–150 actors, network and a 1–5 behaviour, any number of waves | `npe_coev_m6.pt` (outside [30, 100]; inside it, prefer `npe_coev_m5c.pt`) |
| 20–200 actors, network only, exactly three waves | `npe_m4b.pt` via `benchmarks/fit.py` |
| 20–200 actors, co-evolution, exactly three waves | `npe_coev_m4_10m.pt` via `benchmarks/fit.py` |
| fewer than 20 or more than 200 actors | none of these; train one (`benchmarks/generate_m2.py`, `benchmarks/npe_m2.py`) |
| effects other than the sets above | none of these; the effect set is fixed at training time |

That last row is the sharpest limitation of the approach and is worth stating plainly: an
amortized estimator knows only the effects it was trained on. Adding one means a new
population and a new training run, currently 11–21 hours on one GPU. This suits studies
that apply one specification to many networks, not exploratory work on one network.

## Getting the files

The estimators are not in git; they are published as release assets and archived on
Zenodo. `benchmarks/fetch_data.py` fetches the *datasets*, `benchmarks/fetch_models.py`
fetches these:

    pip install "saomsim[fit]"
    python benchmarks/fetch_models.py            # the three you probably want
    python benchmarks/fetch_models.py --all      # all twelve
    python benchmarks/fetch_models.py --list     # what is published, with checksums

Downloads are verified against SHA256. That check is on by default because a truncated or
substituted flow returns posteriors perfectly happily and gives no sign of trouble.

They can also be regenerated end to end, which is the ~13 GPU-hour path:

    python benchmarks/generate_m2.py --N 1000000 --seed 30 --waves 2 \
        --n-min 30 --n-max 100 --rate-max 12 --start survey --box sparse \
        --out data/shards/train_s30.npz          # x10 seeds, ~2.5 h on one GPU
    python benchmarks/npe_m2.py --data "data/shards/train_s*.npz" --out data/npe_m5c
