# Does splitting a panel into periods cost anything?

*Written 2026-10-07, before the run. This is the experiment `docs/AUDIT.md` §3 asked for,
after finding that the experiment §3 actually proposed does not work.*

## The claim, and why its current evidence is weak

`docs/MULTIWAVE.md` establishes that one two-wave estimator reads a panel of any length:
the SAOM likelihood factorises over periods, the priors are flat, so the per-period
posteriors multiply to the joint one. That is a claim about **validity**, and joint SBC
tests it directly — the product is calibrated, which is the result that matters.

Riding alongside it is a claim about **efficiency**: that you lose nothing by doing this
rather than training an estimator on the full panel length. The only evidence is a
comparison of `npe_m5c`'s product (two waves, 10⁷ panels) against `npe_m4b` (three waves,
**10⁶** panels) on s50, which the product won. A tenfold training-budget gap in the
product's favour makes that no evidence at all.

## Why the audit's proposed fix is not a fix

§3 proposed comparing the co-evolution product against `npe_coev_m4_10m`, a three-wave
co-evolution estimator at 10⁷, and called it "far better" because it matches budget. It is
not. The two populations differ in:

| | `npe_coev_m5c` (2 waves) | `npe_coev_m4_10m` (3 waves) |
|---|---|---|
| actors | n ∈ [30, 100] | n ∈ [20, 200] |
| rate prior | U(1, 12) | U(1, 20) |
| effect box | **sparse** (widened density, transTrip, cycle3) | **default** |
| start regime | **survey** (every start sparse, k ~ U(0.5, 6)) | **homophilous** |

So it trades one confound for four, two of which the audit did not notice. A matched
budget bought with a different prior box and a different start regime is worse than the
unmatched comparison it replaces, because the budget gap at least had a known direction.

**No pair of existing artefacts gives a clean test.** Every three-wave estimator in the
project was trained on an m4-family population; every 10⁷ two-wave estimator was trained on
an m5c-family one. The comparison requires a run.

## The design

Train a three-wave network estimator on the **M6 population** — the one `npe_m6` was
trained on — changing the wave count and nothing else:

    --waves 3 --n-min 20 --n-max 150 --rate-max 12 --start survey --box sparse

against `npe_m6`'s `--waves 2` with every other flag identical. Call it **M8**. Then on a
single set of fresh three-wave panels drawn from that population, evaluate both:

- **A**: `npe_m6` applied per period, the two posteriors multiplied (the product sampler).
- **B**: M8 applied directly to all three waves.

Same panels, same truths, same seeds. Compare KS uniformity, 90 % and 95 % coverage, and
posterior sd parameter by parameter.

## The budget asymmetry, stated before the result

At 10⁷ panels each, B sees **2 × 10⁷ period transitions** and A sees 10⁷. Three-wave panels
also cost about 1.5× as much wall-clock to generate. Matching on panels is the choice that
means something operationally — panels are what you generate and store — but it hands B an
advantage of exactly the kind under test.

That asymmetry sets what the result can establish, so it is worth fixing the reading now:

- **A ties or beats B.** Decisive. The product costs nothing even against a native
  estimator with twice its transitions, and the efficiency claim is earned.
- **A loses to B.** *Inconclusive*, and this is the point of writing it down first. It
  cannot distinguish "the product construction is lossy" from "B simply saw twice the
  transitions". Resolving it would need a second B trained at 5 × 10⁶ panels — matched on
  transitions instead — which brackets the answer. That run is not being done up front
  because it is only needed in one branch.
- **A loses badly** (coverage off by more than ~0.02, or sd ratios above 1.1): the
  transition count is unlikely to explain a gap that size, and the honest reading is that
  the efficiency claim should be dropped even without the bracketing run.

A tie is the outcome that would let the paper keep its claim, and the design is deliberately
tilted against it.

## What this cannot tell us

B is a three-wave estimator. It says nothing about four or more waves, where the product's
advantage is that it needs no new training at all and a native estimator would need one run
per panel length. The efficiency question only gets harder for the native estimator as W
grows, so a tie at W = 3 is the hardest case for the product, not the easiest.

It is also the network model only. The co-evolution product is the one with the known
sampling difficulty (the influence–curvature ridge, 900–3,500 effective draws), so a clean
result here does not transfer to it.

## Cost

M6's two-wave generation ran at ~500 panels/s, 2,000 s per 10⁶ shard, 5 h 34 m for ten.
Three waves should cost roughly 1.5× — call it 8 h — and training took 10 h 56 m for M6,
so about 19 h end to end.

---

# Outcome

*Pending.*
