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

## `npe_m5c.pt` — network model, two waves *(the default)*

The one to use unless you need something it does not cover. It reads **any number of
waves** through `saom-fit`, because the per-period posteriors multiply
(`docs/MULTIWAVE.md`).

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

## `npe_coev_m5c.pt` — network × behaviour, two waves

| | |
|---|---|
| **returns** | 12 parameters: network rate and behaviour rate per period, then density, recip, transTrip, cycle3, egoZ, altZ, simZ (selection), linear, quad, avAlt (influence) |
| **actors** | n ∈ **[30, 100]**; behaviour on 3–5 ordered categories |
| **priors** | as above, plus behaviour rate U(0.3, 6), simZ U(−1, 4), linear U(−1.5, 1.5), quad U(−1.5, 0.5), avAlt U(−1, 4) |
| **training** | 10⁷ panels, 3 h 22 m to generate, 20 h 40 m to train |
| **calibration** | 90 % coverage 0.890–0.906, 95 % 0.938–0.952; KS rejects 1 of 12 (quad, p 0.004, mean rank 0.514) |
| **against RSiena** | 17 Baerveldt classes where RSiena converges, **204 of 204** within 1.645 combined sd, largest 1.52. Sharper than the method of moments on the parameters the studies are about: sd ratio **0.64 on influence**, **0.66 on curvature**, 0.95–1.07 on the structural block |
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
| 30–100 actors, network and a 1–5 behaviour, any number of waves | `npe_coev_m5c.pt` |
| 20–200 actors, network only, exactly three waves | `npe_m4b.pt` via `benchmarks/fit.py` |
| 20–200 actors, co-evolution, exactly three waves | `npe_coev_m4_10m.pt` via `benchmarks/fit.py` |
| fewer than 20 or more than 200 actors | none of these; train one (`benchmarks/generate_m2.py`, `benchmarks/npe_m2.py`) |
| effects other than the sets above | none of these; the effect set is fixed at training time |

That last row is the sharpest limitation of the approach and is worth stating plainly: an
amortized estimator knows only the effects it was trained on. Adding one means a new
population and a new training run, currently 11–21 hours on one GPU. This suits studies
that apply one specification to many networks, not exploratory work on one network.

## Getting the files

The estimators are not in git. `benchmarks/fetch_data.py` fetches the *datasets*; the
trained estimators are build artefacts of the runs documented in `docs/M4_RESULTS.md` and
`docs/M5_RESULTS.md` and are published separately. Until that publication exists, they can
be regenerated end to end:

    python benchmarks/generate_m2.py --N 1000000 --seed 30 --waves 2 \
        --n-min 30 --n-max 100 --rate-max 12 --start survey --box sparse \
        --out data/shards/train_s30.npz          # x10 seeds, ~2.5 h on one GPU
    python benchmarks/npe_m2.py --data "data/shards/train_s*.npz" --out data/npe_m5c
