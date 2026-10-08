# M7: a population containing evolved starts

*Written 2026-10-05, before the run.*

## The defect this is for

`docs/MULTIWAVE.md` establishes that one two-wave estimator reads a panel of any length,
by multiplying the per-period posteriors, and measures the one place it costs something.
Period 1 of a panel starts from a network drawn the way the training population draws
starts. Period *w* > 1 starts from a network that has **already evolved**, and no
population we have built contains such a thing. The per-period diagnostic
(`sbc_multiwave.py --per-period`) localises the effect precisely:

| | period 1 (population start) | period 2 (evolved start) |
|---|---|---|
| `npe_m5c`, n ∈ [30, 100] | 7 of 8 uniform | 2 of 8 rejected, rate and density mean rank 0.528 |
| `npe_m6`, n ∈ [20, 150] | 7 of 8 uniform | **4 of 8 rejected**, density 0.551, transTrip 0.466 |

A mean rank above 0.5 means the truth lies above the posterior median too often, so rate
and density are read **low** on an evolved start. The effect is small but systematic, it
reproduces across two populations, and it is *worse* on the wider one — a wider range of
n gives period 2 a wider range of evolved networks to begin from, none of them in the
training population either way.

The factorisation is not what costs. The training population is. M7 changes the
population and nothing else.

## The change

A fraction of training panels get a **pre-period**: the start network is simulated forward
one period before the panel begins, and the panel is then generated from the result.
Everything else — the sparse effect box, the survey start regime, rate U(1, 12), the
covariate layout, two waves, n ∈ [20, 150] — is held at the M6 setting, so that M7 − M6
isolates the evolved start exactly as M6 − M5c isolated the size range.

Three decisions worth stating, because each could reasonably have gone the other way.

**The pre-period runs at the panel's own effects β, not at an independent draw.** This is
the opposite of the choice made for the *burn-in* that structures start networks
(`docs/PRIORS_M2.md`), and deliberately so. The burn-in uses an independent θ₀ precisely
so the population does not assert that the observed first wave is stationary under the θ
being inferred — we do not know how wave 1 came about. But period 2's start is not wave 1.
It is a network this very model produced from wave 1 at these very effects, and the
estimator will be asked to read it on that understanding. Using an independent θ₀ here
would reproduce the mismatch rather than remove it.

**The pre-period's rate is an independent draw from the rate prior.** In the model the
per-period rates are independent, so x₁ arises from x₀ at rate₁ while the period being
inferred runs at rate₂. Reusing the panel's own rate would couple them and teach the
estimator a dependence the model does not have.

**Half the panels get a pre-period, not all of them.** The estimator still has to read
period 1 of every panel, whose start *is* a population draw. A population made entirely of
evolved starts would fix period 2 by breaking period 1. Half mirrors the existing
convention, under which half the starts are burnt in and half are not, and it leaves the
diagnostic able to see both cases.

## What would count as success, and what as failure

Stated now so the result is not read backwards afterwards.

- **Success**: the per-period test on M7's own population shows period 2 no better or
  worse than period 1 — both around 7 of 8 uniform, with rate and density mean ranks back
  near 0.50 from 0.528 and 0.551. The joint product SBC should then stop under-covering.
- **Partial success, and the likelier outcome**: period 2 improves but does not match
  period 1, because one pre-period is not the same as the arbitrary number of prior
  periods a long panel's later waves have seen.
- **Failure worth reporting**: period 1 degrades. Half the population now has starts drawn
  a different way, and if that costs period 1 more than it buys period 2 then the
  trade is not worth making and the honest conclusion is that the defect is cheaper to
  document than to fix.
- **Null result**: nothing moves, which would mean the evolved-start explanation is wrong
  and the tilt has another cause. The per-period diagnostic was what suggested the
  explanation; it is also what can refute it.

## Cost

M6's co-evolution run took 7 h 14 m to generate and 19 h 52 m to train. M7 adds one extra
simulated period to half the panels, so generation should cost roughly 50 % more than the
network M6's 5 h 34 m — call it 8 h — and training the same ~11 h. Network model only:
if it works, the co-evolution counterpart is a second run.

---

# Outcome (2026-10-07): it fixed what it aimed at and broke something better

Generated in 7 h 15 m at 368–389 panels/s, trained in 17 h 58 m — against M6's 10 h 56 m
on the same population size, the flow kept finding improvements for seven hours longer.

## Against the criteria above

**The per-period criterion was met, almost exactly as written.**

| | period 1 | period 2 | rate rank | density rank |
|---|---|---|---|---|
| M6, no pre-period | 7 of 8 uniform | **4 of 8 rejected** | 0.529 | 0.551 |
| M7, half evolved | 7 of 8 uniform | **7 of 8 uniform** | **0.508** | **0.510** |

The directional tilts are gone: rate 0.529 → 0.508, density 0.551 → 0.510, transTrip
0.466 → 0.478, sameX 0.522 → 0.504, and period 2's coverage moves from 0.869–0.911 to
0.901–0.913. Period 1 did not degrade, which was the failure mode anticipated above: it
still passes 7 of 8, with a different parameter flagged (egoX 0.013 against altX 0.040)
at a similar magnitude.

M7's own two-wave SBC also rejects **0 of 8** where M6 rejects 3 of 8 on the identical
size range. As a *two-wave* estimator it is the best-calibrated one in this project.

## And the joint product got worse, which matters more

| | per-period, period 2 | joint product, mean 90 % coverage |
|---|---|---|
| `npe_m5c` [30, 100] | 2 of 8 rejected | **0.898** |
| `npe_m6` [20, 150] | 4 of 8 rejected | 0.885 |
| `npe_m7` [20, 150] + evolved | **1 of 8 rejected** | **0.854** |

All nine of M7's parameters cover below nominal, the shared effects worst — `altX` at
0.812, six standard errors low. Effective sample sizes are comparable across the three
runs (median ≈ 5,000; 7, 12 and 14 panels below 500), so this is not the sampler. The
product is systematically **over-confident**.

## Why, and why the criterion was the wrong one

The product rests on the SAOM factorisation, in which each wave is a **fixed conditioning
state** carrying no information about θ beyond the transition out of it. All of x₁'s
information about β belongs to L₁.

M7 deliberately broke that. Evolving half the starts at the panel's own β makes the start
*informative about β*, and the flow learns to read β out of it:

    q_1  ∝  L_1 · π(θ)
    q_2  ∝  L_2 · p(x_1 | θ) · π(θ)

and p(x₁ | θ) is essentially L₁ over again. The product **double-counts** the information
in x₁, which is exactly what over-concentration looks like.

**The per-period test cannot detect this**, and that is the methodological lesson. It asks
whether q₂ is the correct posterior for a panel whose x₁ really was evolved at θ — and it
is, which is why M7 passes it. The per-period diagnostic validates the *marginal*; only
the joint validates the *factorisation*. Naming it as the success criterion above was a
mistake, made before the distinction was visible.

## Testing the mechanism rather than asserting it

Double-counting makes a prediction that could have killed it: the over-concentration
should grow with the number of periods, because x_1 is double-counted once in a
three-wave panel and twice in a four-wave one. A fixed offset unrelated to panel length
predicts no such growth. Joint product SBC on 300 four-wave panels, against the
three-wave runs:

| | 3 waves | 4 waves | change (bootstrap 95 %) |
|---|---|---|---|
| `npe_m6` | 0.885 | 0.875 | −0.010 [−0.027, +0.007] |
| `npe_m7` | 0.854 | 0.832 | **−0.022** [−0.041, −0.003] |

M7 degrades about twice as fast, and its own degradation excludes zero. But the
*difference* between the two is −0.012 [−0.037, +0.014], so P(M7 degrades faster) ≈ 0.82.
**The prediction survives a test that could have refuted it without being established by
it**; 300 panels was sized for a quick check, and anything load-bearing would want more.

M6 degrades with panel length too, only more slowly. That fits: its per-period error comes
from evolved starts being out of distribution, which compounds across periods as well,
just less sharply than double-counting does.

## What this revises

The burn-in at an **independent** θ₀, which `docs/PRIORS_M2.md` justified on the grounds
that we should not assert wave 1 is stationary under the θ being inferred, turns out to
have a second and stronger justification: **it is what the product construction requires.**
Period 2's miscalibration in M6 is the price of a valid factorisation, not a defect to be
fixed. The two goals are in direct tension, and the earlier write-ups — which called the
evolved start "a population problem, not a flaw in the factorisation" and named this run
as the obvious remedy — had it backwards.

It also revises the M6 reading. M6's three-of-eight rejections were attributed to the
wider size range being a harder learning problem. M7 covers the same range and rejects
none, so most of that was the evolved-start mismatch rather than n.

## Recommendation

**Keep `npe_m6` for multiwave work; do not ship `npe_m7` for it.** M7 is the better
two-wave estimator and is kept for that, and as the evidence for this finding. If the
per-period tilt is ever worth removing, it needs an approach that leaves the start
uninformative about β — conditioning the flow on a wave index, or training on period-2
views whose starts are evolved at an *independent* θ, neither of which has been tried.
