# M3b: network–behaviour co-evolution (selection versus influence) — design

Written 2026-09-13, before any code. The substantive payoff of the SAOM family
(Steglich, Snijders & Pearson 2010): a dependent actor variable z (e.g.
alcohol use) that changes *because of* the network (influence) while the
network changes *because of* z (selection). RSiena's canonical example is the
s50 data with `s50a` (alcohol, three waves) — the same data we already use, so
the real-data test and the RSiena benchmark are both available.

## 1. The model to implement

Two coupled continuous-time processes on the same actor set.

**Network process** — as now: rate λ^X per actor per period; ministep =
uniform actor, multinomial logit over the n tie options; the objective adds
selection effects that read z:

| effect | actor statistic s_i(x, z) | change for toggling x_ij |
|---|---|---|
| egoX(z) | Σ_j x_ij z̃_i | ± z̃_i |
| altX(z) | Σ_j x_ij z̃_j | ± z̃_j |
| simX(z) | Σ_j x_ij (sim_ij − ŝim) | ± (sim_ij − ŝim) |

with z̃ = z − z̄ (grand mean over actors and waves), sim_ij = 1 − |z_i − z_j| /
range(z), ŝim = the mean of sim_ij over all pairs and waves. egoX/altX exist
(`saomsim.effects`); simX and the centring constants are new.

**Behaviour process** — new: rate λ^Z per actor per period; ministep = uniform
actor; three options {z_i − 1, z_i, z_i + 1} clipped to the scale
{z_min..z_max}, multinomial logit on f^Z_i(z') = Σ_k β_k Δ s^Z_ik:

| effect | actor statistic s^Z_i(x, z) | meaning |
|---|---|---|
| linear | z̃_i | overall trend |
| quad | z̃_i² | attraction to / repulsion from the mean |
| avAlt | z̃_i · (Σ_j x_ij z̃_j) / x_i+ (0 if x_i+ = 0) | influence: move toward alters' average |
| avSim | (Σ_j x_ij (sim_ij − ŝim)) / x_i+ | influence via similarity (alternative to avAlt) |
| indeg / outdeg | z̃_i · x_+i, z̃_i · x_i+ | position effects on behaviour (optional) |

Change statistics for a behaviour move are exact differences of s^Z_i between
z_i ± 1 and z_i with the network fixed.

**Joint simulation** (RSiena's unconditional scheme): events arrive at total
rate n(λ^X + λ^Z); the number of events in a period is Poisson of that;
each event is a network ministep with probability λ^X / (λ^X + λ^Z), else a
behaviour ministep. Batched: per step draw the event type per chain, run the
network kernel on the network-type chains and the behaviour kernel on the
others (both masked, as the stopping rules already are).

Parameters for one period: (λ^X, λ^Z, network effects incl. selection,
behaviour effects). With the M2 network set + simX(z) + {linear, quad, avAlt}
that is 2 + 8 + 3 = 13 parameters per period-shared model; W waves add
W − 2 more rates of each kind.

## 2. The gate — before any estimator

M0's lesson: the one convention mismatch (cycle3) was found by the RSiena
gate, not by reasoning. Behaviour statistics have more conventions to get
wrong: which wave's network the behaviour target statistic uses, how z is
centred, how ŝim is computed, what avAlt does at out-degree zero, whether the
target for behaviour effects is evaluated at the end-of-period behaviour with
the *start*-of-period network. So:

1. `benchmarks/rsiena_coevolution.R`: s501/s502/s503 + alcohol as a
   `sienaDependent(type = "behavior")`, effects above, `getTargets` → JSON;
   plus `siena07(simOnly = TRUE, cond = FALSE)` simulated statistics for both
   the network and the behaviour at fixed θ, as for M0.
2. `saomsim` reference implementations (loops) for every new statistic and
   change statistic; vectorised kernels checked against them at several
   densities and behaviour distributions; then the RSiena parity tests for
   targets and for the joint simulated dynamics.

No estimator until both pass.

## 3. Prior and population (to be fixed after the gate)

- Behaviour scale: integer 1..K with K ∈ {3, 4, 5} (s50 alcohol: 1..5),
  start distribution drawn per panel (roughly unimodal, mean anywhere on the
  scale) — as with covariates, the estimator conditions on z(t₀), so this
  prior only needs coverage.
- λ^Z ~ U(0.3, 6): behaviour changes less often than ties; s50 alcohol
  changes for roughly a third of actors per period.
- linear U(−1, 1), quad U(−1, 0.5) (negative quad = pull toward the mean, the
  usual sign), avAlt U(−1, 3) (positive = influence), simX U(−1, 3)
  (positive = selection on similarity). Ranges to be checked against
  RSiena's s50 co-evolution estimates before committing.
- Network side unchanged from M2.

## 4. What the paper gets from it

Selection and influence estimated jointly, with a posterior that shows their
correlation directly — the identification question Steglich et al. address by
design (both processes in one model) becomes visible as posterior geometry.
The M4 application (many school networks with a behaviour) is the natural
follow-on and the reason this milestone exists.

## 5. Cost

Simulator extension: behaviour kernel + three effects + simX, reference
implementations, RSiena R script, parity tests — roughly M0's size again.
Population and estimator: reuse of the M2/M3a machinery with a larger θ and
extra summary blocks (behaviour distribution per wave; selection/influence
target statistics). The 10⁷-panel runs cost the same order as before.
