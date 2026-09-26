# Goodness of fit on network statistics

*2026-09-26.*

The posterior predictive checks in this repository so far have looked at behaviour
statistics (`benchmarks/ppc_coev.py`, which adjudicated the influence/curvature ridge on
Glasgow) or at scalar summaries of one period of s50 (`benchmarks/ppc_s50.py`). Neither
is what a SAOM reader means by goodness of fit. That is `sienaGOF` (Lospinoso & Snijders
2019): simulate from the fit, and compare the observed network with the simulated ones on
**auxiliary statistics the estimator did not target** — the out- and in-degree
distributions, the triad census, and the distribution of geodesic distances — judging
each vector as a whole by the Mahalanobis distance of the observation from the simulated
cloud.

`saomsim/gof.py` and `benchmarks/gof_network.py` do this. Two implementation notes worth
recording:

- **The triad census lookup is built from `networkx`, not derived by hand.** A dyad has
  four states and a triple has three dyads, so 64 combinations map onto the 16
  Holland–Leinhardt types. Rather than write that mapping out and hope, `triad_lookup()`
  constructs each of the 64 labelled three-node digraphs once and asks
  `networkx.triadic_census` to classify it. The labels are then right by construction,
  and the counting afterwards is pure numpy: 200 networks at n = 129 in 1.3 s.
  `tests/test_gof.py` re-checks the whole census against `networkx` on random graphs.
- **The covariance is inverted by pseudo-inverse.** The entries of a census or a degree
  distribution sum to a constant, so the simulated covariance is singular by
  construction. `sienaGOF` uses `MASS::ginv` for the same reason.

The p-value is the fraction of simulated networks at least as far from the simulated mean
as the observation, so it assumes no distribution. Per-entry standardised deviations say
*where* a poor fit fails, which is the part that suggests what to add to the model.

## s50, three waves, network model

2,000 simulations per period from the observed start of that period, under the joint
posterior of `benchmarks/multiwave.py` — so the effects are shared across periods, which
is the relevant null when the question is whether one parameter vector reproduces the
whole panel.

| statistic | period 1 | period 2 |
|---|---|---|
| outdegree | 0.429 | 0.312 |
| indegree | 0.666 | 0.653 |
| triad census | 0.092 | 0.106 |
| geodesic | 0.479 | 0.066 |

**Nothing is rejected.** The canonical effect set, fitted amortized, reproduces degree
distributions, triad counts and path lengths it never saw.

### The comparison that makes this interpretable

Running the identical check at RSiena's method-of-moments point, and — the control that
matters — at *our own posterior mean treated as a point*:

| point | p1 triads | p2 triads | p2 geodesics |
|---|---|---|---|
| amortized posterior (integrates θ) | 0.092 | 0.106 | 0.066 |
| amortized mean, as a point | **0.023** | **0.041** | **0.019** |
| RSiena MoM point | 0.051 | **0.047** | **0.006** |

**Our own point estimate is rejected in the same places as RSiena's, and slightly
harder.** So this is not a claim that the amortized estimate sits in a better place. The
three differ only in what is held fixed: conditioning on any single θ discards parameter
uncertainty, and the resulting predictive distribution is too narrow, so the observation
falls outside it. Integrating over the posterior widens the predictive distribution by
exactly the amount the data leave undetermined, and the observation falls inside.

This is worth stating carefully in both directions.

- It is a **methodological argument for having a posterior**: standard `sienaGOF`
  practice conditions on a point estimate and therefore over-rejects, and the amortized
  estimator makes the properly integrated check available at no extra cost. One cannot do
  this with a method-of-moments fit without bootstrapping it, which costs another
  simulation campaign per replicate.
- It is **not** a licence to conclude the model fits. The *direction* of misfit is the
  same under all three points — 021D at +2.7 to +4.0 sd in period 1, and d = 4, 5
  under-represented with unreachable pairs over-represented in period 2 — so the
  specification really does under-reproduce out-star concentration and over-connect the
  network, which is the familiar signature of a model lacking degree-related effects
  (`outActSqrt`, `inPopSqrt`). What the posterior check says is that at n = 50, with
  these parameters this weakly determined, the evidence against the specification is not
  significant. A point-based check overstates that evidence.

## Glasgow, three waves, network model

1,500 simulations per period. Note n = 129 is outside `npe_m5c`'s training range and the
tool says so; for this purpose it does not matter much, because the finding below holds
identically at RSiena's point, which is not extrapolating anything.

| statistic | posterior, p1 | posterior, p2 | RSiena point, p1 | RSiena point, p2 |
|---|---|---|---|---|
| outdegree | **0.001** | **0.001** | **0.001** | **0.001** |
| indegree | 0.789 | 0.223 | 0.669 | 0.103 |
| triad census | **0.004** | **0.001** | **0.001** | **0.001** |
| geodesic | **0.017** | **0.001** | **0.001** | **0.001** |

**Everything except in-degree is rejected, under every point, including the full
posterior.** Integrating over parameter uncertainty is not enough here: at n = 129 the
misfit is far larger than what the parameters leave undetermined. Where s50 said "the
evidence against this specification is weak", Glasgow says "this specification is wrong",
and the difference is statistical power.

### What is wrong, precisely

The deviations point in a consistent direction on both panels and under all three points:
the observed networks have **too many actors at moderate out-degree** (Glasgow: out 5 at
+4.2 sd, out 4 at +3.4; s50: out 3 at +2.3) and are **less connected at long range**
(d = 4 and d = 5 under-represented, unreachable pairs over-represented).

Both datasets are nomination-limited surveys, and that is the whole story:

| | observed cap | actors at the cap | simulated max out-degree | simulations breaching the cap | simulated actors over the cap |
|---|---|---|---|---|---|
| Glasgow | 6 | 13–16 % | mean 11.1 | **100 %** | 13.3 % (observed 0 %) |
| s50 | 5 | 4–8 % | mean 6.7 | 79 % | 5.7 % (observed 0 %) |

The questionnaires allowed six friends (Glasgow) and five (s50); nobody could name more,
and 13–16 % of Glasgow pupils named exactly the maximum. **The SAOM as specified has no
mechanism that enforces a nomination cap during simulation**, so simulating forward
produces networks with out-degrees the survey could not have recorded, and with the
longer reach that those extra ties buy. The degree, triad and geodesic statistics all
register the same thing.

Three consequences, in order of how much they matter.

1. **This is a property of the model class, not of the estimator.** RSiena's simulator is
   the same process, so its own `sienaGOF` on these data would say the same. Nothing here
   distinguishes amortized inference from method of moments; the remedy is to model the
   cap or to add degree effects (`outActSqrt`, `outTrunc`) that mimic it, and that is a
   specification change both methods would share.
2. **It explains the n-dependence of the s50 result.** The breach is twice as severe at
   Glasgow as at s50 (13.3 % against 5.7 % of actors), and Glasgow has 2.6× the actors to
   detect it with. A specification failure that a 50-actor panel cannot resolve is plain
   at 129.
3. **It exposes an inconsistency in our own training population**, which is worth
   recording because we introduced it deliberately and did not think it through.
   `docs/PRIORS_M4.md` and `PRIORS_M5.md` cap out-degree on a fraction of *start*
   networks, "as nomination surveys cap them" — but the simulator then evolves those
   starts without a cap, so the later waves of a training panel are uncapped even when
   its first wave is not. Real capped surveys are capped at every wave. Whether that
   mismatch biases the estimator is a testable question and we have not tested it; the
   agreement with RSiena throughout this repository bounds how large the effect can be,
   since RSiena is fitted to the same capped data with the same uncapped simulator.

## What this closes and what it does not

## What this closes and what it does not

It closes the gap flagged in `paper/arxiv/README.md`: the paper's goodness-of-fit
evidence was behaviour-only, and a reviewer would have asked for the network side.

It does not cover the co-evolution model's network and behaviour statistics jointly,
which `sienaGOF` can do and we have not. It also leaves two questions open that it
raised itself: whether fitting a specification with degree effects removes the Glasgow
misfit (it should, and that is the obvious next experiment), and whether the
capped-start / uncapped-simulation mismatch in the training population costs anything
measurable.

Reproduce with:

    python benchmarks/gof_network.py --posterior data/npe_m5c.pt --real s50 --waves 3 \
        --B 2000 --rsiena benchmarks/rsiena_estimate_3w.json --point-control
