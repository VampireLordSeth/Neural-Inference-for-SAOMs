# M3a: multiple waves

Paper plan M3 has two parts: multiple waves, and behaviour co-evolution. This
is the first. Design written 2026-09-13 before generating data.

## Model

W observed waves, W − 1 periods. RSiena's default layout and ours: **one rate
parameter per period, evaluation effects shared across periods**. θ =
(rate₁, …, rate_{W−1}, density, recip, transTrip, cycle3, altX(v), egoX(v),
sameX(g)). For W = 3 that is nine parameters. Each period is simulated from
the observed previous wave (the SAOM conditions on each wave in turn), so the
simulator is `simulate_period` applied W − 1 times — the code path validated
against RSiena in `benchmarks/`, applied sequentially.

Time-heterogeneity of effects (RSiena's `sienaTimeTest`) is not modelled;
the estimator assumes shared effects, as RSiena's default does.

## Population

Identical to M2 (`docs/PRIORS_M2.md`): n ~ U{20..80}, the same start-network
and covariate priors, the same effect ranges. Each rate has the M2 rate prior
U(1, 12) independently. Nothing else changes, so the two-wave M2 estimator and
the three-wave M3 estimator differ only in what they condition on.

## What the estimator sees

The M2 summary layout with one 13-column block per period:
n, covariate shape (3), X0 block (12), then for each later wave the change
count from the previous wave plus that wave's 12 statistics. W = 3 gives 42
summaries. Later waves are stored bit-packed (`Xw`) for the embedding variant.

## Checks

1. Prior predictive: s50's three-wave summary vector within the population.
2. Fresh-sample SBC and coverage across n (`benchmarks/sbc_m2.py --waves 3`).
3. Real data: s501 → s502 → s503 against RSiena's three-wave fit
   (`benchmarks/rsiena_estimate_3w.R`, converged first run, 24 s): rate₁ 6.51
   ± 1.07, rate₂ 5.30 ± 0.89, density −2.76 ± 0.16, recip 2.44 ± 0.21,
   transTrip 0.64 ± 0.15, cycle3 −0.07 ± 0.28, altX −0.02 ± 0.07, egoX 0.06
   ± 0.08, sameX 0.17 ± 0.17. Note the standard errors: ~30 % tighter than the
   two-wave fit, which is what a third wave should buy, and what the M3
   posterior should show relative to M2's.

## What comes after

M3b, behaviour co-evolution, needs the simulator to carry a dependent actor
variable with its own rate and objective (linear/quadratic shape, average
similarity or average alter for influence; the network's `sameX`/`simX` on
that variable for selection), and an RSiena gate for the behaviour statistics
before any estimator is trained. It is a separate design document.
