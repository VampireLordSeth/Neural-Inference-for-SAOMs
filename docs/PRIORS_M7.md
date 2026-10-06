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
