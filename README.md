# saomsim

Batched NumPy simulator for stochastic actor-oriented models (SAOMs, the model
family behind RSiena), with reference implementations, a method-of-moments
estimator, and a test suite that checks every vectorized kernel against a
loop-based reference and the ministep dynamics against an exact Markov chain.

Milestone M0 of the amortized neural inference project. See
[GETTING_STARTED.md](GETTING_STARTED.md) for setup and the working order.

## What it does

```python
import numpy as np
from saomsim import Model, random_network, simulate_period, statistics, estimate

rng = np.random.default_rng(0)
grp = np.repeat([0.0, 1.0], 15)
model = Model(["density", "recip", "transTrip", ("sameX", "grp")], {"grp": grp})

X0 = random_network(B=1000, n=30, density=0.08, rng=rng)        # (B, n, n) int8
X1 = simulate_period(X0, theta=[-2.2, 1.2, 0.35, 0.6], rate=3.0, model=model, rng=rng)
S = statistics(X1, model)                                        # (B, K) target statistics
res = estimate(X0[:200], X1[:200], model, rng)                   # method of moments
print(res.table())
```

`B` chains run in lockstep on `(B, n, n)` arrays. Parameters may be shared
(`theta` of shape `(K,)`) or per chain (`(B, K)`), which is what the training
data generator needs.

## Layout

| file | role |
|---|---|
| `backend.py` | the only module that indexes into the `(B, n, n)` layout: row products, softmax, inverse-CDF sampling, toggle. Torch port replaces this file. |
| `effects.py` | `Effect`, `Model`; vectorized change statistics and target statistics for `density`, `recip`, `transTrip`, `cycle3`, `sameX`, `altX`, `egoX` |
| `reference.py` | the same statistics as explicit loops on a single network. Slow. Sacred. |
| `simulate.py` | `simulate_period`, `simulate_panel`, `random_network`, `rate_statistic`, `statistics` |
| `estimate.py` | simulated method of moments: CRN finite-difference Jacobian, scaled Gauss-Newton with trust region and backtracking, sandwich standard errors |

## Model

One period of the basic SAOM (constant rate; evaluation function only):

- number of ministeps per chain ~ Poisson(n · rate)
- each ministep: one actor `i` chosen uniformly; `i` picks among the `n`
  options {toggle `x_ij` : j ≠ i} ∪ {no change} with probability
  ∝ exp(Σ_k θ_k Δ_ijk), where Δ_ijk = s_ik(x with x_ij toggled) − s_ik(x)
  and the no-change option has Δ = 0
- the chosen tie is toggled

Actor statistics follow the RSiena manual and target statistics follow RSiena's
conventions, verified against RSiena 1.6.6 (see `benchmarks/`): `recip` is
actor-summed (a mutual dyad contributes 2), `cycle3` counts each 3-cycle once
(actor sum ÷ 3). Covariates are used as given; RSiena centres them by default,
so pass centred values for `altX`/`egoX` if you want to match its numbers.

## Conventions

- **Reference implementations are sacred.** When a kernel and its reference
  disagree, the kernel is wrong.
- **Nothing is filtered.** Empty and complete networks are returned like any
  other outcome. Dropping them would silently change the effective prior.
- **One RNG, passed explicitly.** `simulate_period` consumes a fixed number of
  draws per ministep regardless of which chains are still active, so a run is
  fully determined by `(X0, theta, rate, seed)`.
- **Test before feature.** A new effect needs its loop reference and a
  parametrized `*_match_reference` test in the same commit.

## Verification

```
pytest -q -m "not slow"   # 98 tests, ~8 s  (includes benchmarks/)
pytest -q                 # + 1 recovery test, ~17 s
python examples/quickstart.py
```

The tests that carry the weight:

- `test_change_statistics_match_reference` / `test_statistics_match_reference`
  — every effect, three densities, random actors, against `reference.py`
- `test_simulation_matches_exact_stationary_distribution` — the simulator's
  long-run law on n = 3 (64 states) against the stationary distribution of the
  exact ministep transition matrix. This is the test that validates the
  *dynamics* (actor choice, softmax over the neighbourhood, no-change option,
  toggle) rather than the statistics.
- `test_estimate_recovers_parameters` (slow) — truth within 3 s.e. on 150
  panels.
- `benchmarks/test_rsiena_parity.py` — target statistics on s501/s502 equal
  RSiena 1.6.6's to the last digit, all seven effects plus the rate.
- `benchmarks/test_rsiena_dynamics.py` — distribution of simulated statistics
  from s501 at fixed θ matches RSiena's own simulator (means within Monte Carlo
  error, spreads within 5 %). This is what licenses the phrase "agrees with
  RSiena" for the model class implemented here.

## A result worth noticing

With 200 simulated panels at n = 30 the method-of-moments standard errors in
the quickstart are ~0.02–0.04. Divide by √200 the other way and a *single*
panel carries a standard error of roughly 0.3–0.5 on each parameter — for
`transTrip` that is larger than typical published effect sizes. This is not a
bug. A small one-period panel simply carries little information about triadic
parameters, and any estimator, neural or classical, will report wide
uncertainty on it. If the amortized posterior looks wide on single panels,
check its calibration before assuming it is under-trained.

## Throughput (CPU, this laptop)

`density + recip + transTrip`, rate 3, float64:

| n | B | panels/s |
|---|---|---|
| 30 | 4000 | ~2,800 |
| 100 | 400 | ~210 |

Per-ministep cost is dominated by Python/NumPy overhead at n = 30 and by the
batched matvecs at n = 100. Wider batches help more than anything else.
