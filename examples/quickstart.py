"""saomsim quickstart.

Four things:
  1. more reciprocity parameter  -> more mutual dyads
  2. more homophily parameter    -> larger share of within-group ties
  3. method-of-moments recovery of a known parameter vector from a batch of
     simulated panels (Gauss-Newton), with standard errors
  4. the classical baseline: Robbins-Monro estimation from a *single* panel,
     conditional on the observed distance, as RSiena does it
  5. throughput, which sets the training-set budget
"""

import time

import numpy as np

from saomsim import Model, estimate, estimate_rm, random_network, simulate_period, statistics

rng = np.random.default_rng(20240911)
n, B = 30, 1000

# ---------------------------------------------------------------- 1. reciprocity
print("== reciprocity sweep  (n=30, density=-2.0, rate=4, B=1000) ==")
m = Model(["density", "recip"])
recip_only = Model(["recip"])
X0 = random_network(B, n, 0.08, rng)
print(f"{'beta_recip':>10} {'mutual dyads':>13} {'ties':>8}")
for beta in [-1.0, 0.0, 1.0, 2.0, 3.0]:
    X1 = simulate_period(X0, [-2.0, beta], 4.0, m, rng)
    mutual = statistics(X1, recip_only).mean() / 2  # recip stat counts each dyad twice
    print(f"{beta:>10.1f} {mutual:>13.2f} {X1.sum(axis=(1, 2)).mean():>8.1f}")

# ------------------------------------------------------------------ 2. homophily
print("\n== homophily sweep  (two equal groups; chance level = 14/29 = 0.483) ==")
grp = np.repeat([0.0, 1.0], n // 2)
m = Model(["density", ("sameX", "grp")], {"grp": grp})
same_only = Model([("sameX", "grp")], {"grp": grp})
print(f"{'beta_sameX':>10} {'within-group share':>19}")
for beta in [0.0, 0.5, 1.0, 1.5, 2.0]:
    X1 = simulate_period(X0, [-2.0, beta], 4.0, m, rng)
    share = statistics(X1, same_only).sum() / X1.sum()
    print(f"{beta:>10.1f} {share:>19.3f}")

# ------------------------------------------------------------------- 3. recovery
print("\n== parameter recovery  (200 panels, n=30) ==")
m = Model(["density", "recip", "transTrip", ("sameX", "grp")], {"grp": grp})
truth = np.array([3.0, -2.2, 1.2, 0.35, 0.6])  # rate, then effects in model order
P = 200
X0 = random_network(P, n, 0.08, rng)
t0 = time.perf_counter()
X1 = simulate_period(X0, truth[1:], truth[0], m, rng)
changes = np.abs(X1 - X0).sum(axis=(1, 2)).mean()
print(
    f"simulated {P} panels in {time.perf_counter() - t0:.2f}s; "
    f"mean density {X1.mean() * n / (n - 1):.3f}, mean changes {changes:.1f}"
)
t0 = time.perf_counter()
res = estimate(X0, X1, m, rng, n_sim=10, max_iter=25, fd_step=0.15)
print(f"estimated in {time.perf_counter() - t0:.1f}s\n")
print(res.table(truth))

# --------------------------------------------------- 4. single-panel baseline
print("\n== single-panel Robbins-Monro baseline  (n=30, conditional on distance) ==")
m = Model(["density", "recip", "transTrip"])
truth = np.array([-2.0, 1.6, 0.2])
x0 = random_network(1, n, 0.09, rng)
x1 = simulate_period(x0, truth, model=m, rng=rng, n_steps=180)[0]
print(f"observed distance x0 -> x1: {int((x0[0] != x1).sum())}")
t0 = time.perf_counter()
res = estimate_rm(x0[0], x1, m, rng)
if not res.converged:  # RSiena practice: rerun from the previous answer
    res = estimate_rm(x0[0], x1, m, rng, theta0=res.theta)
print(f"estimated in {time.perf_counter() - t0:.1f}s\n")
print(res.table(truth))
print(
    "\nCompare the s.e. column with section 3: one panel carries roughly sqrt(200)"
    "\ntimes less information than 200 of them. See README 'A result worth noticing'."
)

# ------------------------------------------------------------------- throughput
print("\n== throughput (density+recip+transTrip, rate=3; see benchmarks/throughput.py) ==")
m = Model(["density", "recip", "transTrip"])
backends = ["numpy"]
try:
    import torch  # noqa: F401

    backends.append("torch")
except ImportError:
    pass
for bk in backends:
    for n_, B_ in [(30, 4000), (100, 400)]:
        X0 = random_network(B_, n_, 0.1, rng)
        simulate_period(X0, [-2.0, 1.0, 0.2], 3.0, m, rng, backend=bk)  # warm-up
        t0 = time.perf_counter()
        simulate_period(X0, [-2.0, 1.0, 0.2], 3.0, m, rng, backend=bk)
        dt = time.perf_counter() - t0
        print(f"{bk:<6} n={n_:>3}  B={B_:>5}  {B_ / dt:>8,.0f} panels/s")
print(
    "\nRead panels/s as the training-set budget: 10^5-10^6 simulated panels divided by"
    "\nthis number is the wall clock for generating the neural estimator's data."
)
