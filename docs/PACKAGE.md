# saomsim — package reference

Batched NumPy simulator for stochastic actor-oriented models (SAOMs, the model
family behind RSiena), with reference implementations, a method-of-moments
estimator, and a test suite that checks every vectorized kernel against a
loop-based reference and the ministep dynamics against an exact Markov chain.

Milestone M0 of the amortized neural inference project. See
[GETTING_STARTED.md](GETTING_STARTED.md) for setup and the working order, and
[docs/PRIORS.md](docs/PRIORS.md) for the prior the first estimator is trained
under (fixed empirical start `X0 = s501`; box prior; prior predictive check).

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
res = estimate(X0[:200], X1[:200], model, rng)                   # method of moments, 200 panels
res1 = estimate_rm(X0[0], X1[0], model, rng)                     # Robbins-Monro, one panel
print(res.table(), res1.table())
```

`B` chains run in lockstep on `(B, n, n)` arrays. Parameters may be shared
(`theta` of shape `(K,)`) or per chain (`(B, K)`), which is what the training
data generator needs.

### Backends

```python
simulate_period(X0, theta, 3.0, model, rng)                      # numpy (default)
simulate_period(X0, theta, 3.0, model, rng, backend="torch")     # torch, CUDA if available
from saomsim.backend_torch import TorchBackend
bk = TorchBackend("cuda", dtype=torch.float32)                   # fastest
simulate_period(X0, theta, 3.0, model, rng, backend=bk)
```

The Poisson ministep count always comes from the numpy `Generator`, so a seed
fixes the ministep schedule on every backend; the in-loop draws use a torch
`Generator` seeded from it. Inputs and outputs are numpy on every backend.
The whole test suite runs against each backend (`backend` fixture in
`conftest.py`; `SAOMSIM_TORCH_DEVICE`, `SAOMSIM_TORCH_DTYPE` select device
and dtype).

### Stopping rules

```python
simulate_period(X0, theta, rate=3.0, ...)        # Poisson(n * rate) ministeps (RSiena cond=FALSE)
simulate_period(X0, theta, n_steps=200, ...)     # exactly 200 ministeps
simulate_period(X0, theta, distance=115, ...)    # until Hamming(X, X0) == 115 (RSiena cond=TRUE)
X1, info = simulate_period(..., return_info=True)  # info.n_steps, info.n_changes, info.reached
```

## Layout

| file | role |
|---|---|
| `backend.py` | NumPy kernels and the backend interface (`NumpyBackend`, `get_backend`): row products, softmax, inverse-CDF sampling, toggle, Hamming distance. The only place that indexes into the `(B, n, n)` layout. |
| `backend_torch.py` | `TorchBackend`: the same interface on torch tensors, any device, float64 or float32 |
| `effects.py` | `Effect`, `Model`; vectorized change statistics and target statistics for `density`, `recip`, `transTrip`, `cycle3`, `sameX`, `altX`, `egoX` |
| `reference.py` | the same statistics as explicit loops on a single network. Slow. Sacred. |
| `simulate.py` | `simulate_period`, `simulate_panel`, `random_network`, `rate_statistic`, `statistics` |
| `estimate.py` | `estimate`: multi-panel method of moments (CRN Jacobian, scaled Gauss-Newton, trust region, sandwich s.e.). `estimate_rm`: RSiena-style Robbins-Monro from a single panel, conditional on the observed distance |
| `prior.py` | `BoxPrior`, `summaries`, `generate_training_set`, `TrainingSet` (npz round trip), `prior_predictive_report`. The s50 prior itself lives in `benchmarks/s50.py` and is justified in `docs/PRIORS.md` |

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
pytest -q -m "not slow"   # 105 tests numpy-only; 160 with torch installed (~10 s)
pytest -q                 # + 1 recovery test
python examples/quickstart.py
python benchmarks/throughput.py
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
- `test_estimate_rm_recovers_from_one_panel` (slow) — Robbins-Monro from one
  n = 30 panel converges by RSiena's rule and lands within 3 s.e.
- `benchmarks/test_rsiena_parity.py` — target statistics on s501/s502 equal
  RSiena 1.6.6's to the last digit, all seven effects plus the rate.
- `benchmarks/test_rsiena_dynamics.py` — distribution of simulated statistics
  from s501 at fixed θ matches RSiena's own simulator (means within Monte Carlo
  error, spreads within 5 %). This is what licenses the phrase "agrees with
  RSiena" for the model class implemented here. Runs on every backend.
- `test_backends_agree_in_distribution` — torch vs numpy at the same θ.
- `test_conditional_simulation_stops_at_observed_distance` — the `distance`
  rule yields exactly the observed Hamming distance, per chain.

## A result worth noticing

With 200 simulated panels at n = 30 the method-of-moments standard errors in
the quickstart are ~0.02–0.04. From a *single* panel (`estimate_rm`, quickstart
§4) they are 0.2–0.6 — for `transTrip` that is larger than typical published
effect sizes. This is not a bug. A small one-period panel simply carries little information about triadic
parameters, and any estimator, neural or classical, will report wide
uncertainty on it. If the amortized posterior looks wide on single panels,
check its calibration before assuming it is under-trained.

## Throughput

`density + recip + transTrip`, rate 3, panels/s, best of 2 after warm-up
(`benchmarks/throughput.py`). Laptop = Windows x86 (numpy). Spark = DGX Spark,
GB10, aarch64.

| n | B | laptop numpy | Spark numpy | Spark CUDA f64 | Spark CUDA f32 |
|---|---|---|---|---|---|
| 30 | 1000 | | 9,202 | 32,543 | 32,916 |
| 30 | 4000 | ~2,800 | 5,240 | 57,383 | **122,671** |
| 30 | 16000 | | 4,912 | 55,638 | 108,747 |
| 100 | 400 | ~210 | 414 | 2,182 | 4,535 |
| 100 | 2000 | | 367 | 2,954 | **5,096** |
| 200 | 500 | | 108 | 494 | 717 |

CUDA float32 is 23x Spark numpy at n = 30 and 14x at n = 100. At n = 30 the
GPU is launch-bound (~20 small kernels per ministep), so the gain comes from
batch width up to B ≈ 4000 and flattens after; CUDA graphs or `torch.compile`
would be the next lever. At n = 200 the batched matvecs dominate and the gap
narrows to 7x. Torch on CPU is slower than numpy here and is not a target.

float32 passes the full suite (reference comparisons, exact chain, RSiena
dynamics); use it for training-data generation, float64 for validation runs.
