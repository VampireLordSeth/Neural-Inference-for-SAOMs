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

# Outcome (2026-10-09): the product loses, by about a point and a half of coverage

M8 generated in 8 h 30 m and trained in 18 h 12 m, converging on patience after 202 epochs
at a best validation loss of −0.743. As a native three-wave estimator it is good: its own
SBC on 2,000 held-out draws rejects **2 of 9** (cycle3 p 0.017, altX 0.047), 90 % coverage
runs 0.896–0.906, mean ranks hold near 0.5 in every *n* band from 20 to 150, and on the
three-wave s50 panel every parameter lands within **|z| ≤ 0.33** of RSiena in 0.07 s.

Both estimators on the same 400 three-wave panels:

| | product of `npe_m6` | native `npe_m8` |
|---|---|---|
| mean 90 % coverage | **0.8792** | **0.8947** |
| mean 95 % coverage | 0.9344 | 0.9461 |
| KS rejections | **3 of 9** (rate_1, egoX, sameX) | **0 of 9** |
| mean posterior sd, product ÷ native | **0.980** | — |

**Paired difference in 90 % coverage: −0.0156, se 0.0046.** Paired over panels, so this is
not the difference of two noisy marginals; it excludes zero comfortably.

The product is also **2 % sharper** while covering less, which is the signature of mild
over-confidence rather than of extra noise. Against nominal: the native estimator is
essentially calibrated (0.895 against 0.90), and the product sits about two points under.

## It is not the sampler

The obvious alternative explanation is the importance sampler, since `docs/AUDIT.md` records
that a few per cent of SBC panels fall back to Metropolis on short chains and adds unknown
noise. Now that per-panel ESS is saved, that can be checked rather than assumed:

| | panels | product | native | paired difference |
|---|---|---|---|---|
| fell back to Metropolis (ESS < 500) | 11 (2.8 %) | 0.8384 | 0.7980 | +0.0404 (se 0.0271) |
| importance sampling (ESS ≥ 500) | 389 | 0.8803 | 0.8975 | **−0.0171 (se 0.0046)** |
| ESS quartile 1 (low) → 4 (high) | 100 each | — | — | −0.016, −0.010, −0.019, −0.018 |

Dropping the fallback panels makes the gap slightly **larger**, and there is no trend across
ESS quartiles. The deficit is flat in sampler quality, so the sampler is not producing it.
The fallback panels on their own favour the product, but at n = 11 and se 0.027 that is
noise.

## Reading it against the criteria set before the run

By the rule written down in advance, this is the **inconclusive** branch, and the discipline
is worth keeping. The gap is −0.0156, below the −0.02 that was nominated as "loses badly",
and every sd ratio is ≤ 1.013 against a threshold of 1.1. So the result cannot, by its own
pre-registered terms, separate "the product construction is lossy" from "the native saw
twice the period transitions at matched panels".

The pre-registered remedy stands: a second native trained at 5 × 10⁶ panels, matched on
transitions rather than panels, brackets the answer. That run has not been done.

## But there is a third explanation that fits better than either, and it is not about budget

`docs/MULTIWAVE.md` already established, by a per-period diagnostic and before this run was
conceived, that period *w* > 1 of a panel conditions on an **evolved** start network, which a
two-wave training population does not contain, and that `npe_m6` reads rate and density low
there — 4 of 8 rejected in period 2 against 7 of 8 uniform in period 1.

A native three-wave estimator is immune to this **by construction**: its training panels
have period 2 starting from an evolved network, so the thing that is out of distribution for
the product is in distribution for M8. That mechanism predicts exactly what was observed,
was documented before the comparison, and has nothing whatever to do with training budget.

It is also the mechanism `docs/PRIORS_M7.md` tried to remove by evolving half the training
starts, which fixed the per-period tilt and **broke the joint product** by making the start
informative about β. So this deficit is not a defect awaiting a fix; the M7 result says it
is the price of a valid factorisation.

That makes the transition-count confound less interesting than it looked when the
pre-registration was written, but it does not remove it, and the honest position is that
the evolved-start account is the better-supported of three live explanations rather than
the demonstrated one.

## The regression check, which came out weaker than intended

The product arm was supposed to reproduce `npe_m6`'s published joint coverage of 0.885,
confirming that the `chain_ess` refactor had not perturbed the sampler. It returned
**0.8792**, a difference of 0.0058, comfortably inside sampling variation for a different
draw of 400 panels.

That is consistent, but it is not the clean check it was meant to be: the original run's
seed and flags could not be recovered from the logs, so the panels are probably not the same
ones, and "consistent with" is all that can be claimed. A bit-identical reproduction would
have needed the original command, and the lesson is that the command should have been
recorded in `docs/MULTIWAVE.md` beside the number.

## What the paper should now say

Not "splitting a panel into periods costs nothing measurable" — that claim came from a
comparison against a 10⁶ estimator and does not survive a fair one. The supportable
statement is:

> Reading a three-wave panel as a product of two-wave posteriors costs about **1.6
> percentage points of 90 % coverage** against an estimator trained natively on three
> waves, with posteriors about 2 % narrower — mild over-confidence rather than lost
> precision. The deficit is not an artefact of the product sampler. It is consistent with
> the documented evolved-start distribution shift, which a native estimator does not have,
> and which `docs/PRIORS_M7.md` shows cannot be removed from the training population
> without breaking the factorisation itself.

And the practical recommendation is unchanged, because it never rested on the product being
free: if you have a three-wave panel and a three-wave estimator, use the native one. The
product exists for the panel lengths nobody has trained an estimator for, where the
alternative is not a 1.6-point penalty but no estimate at all.
