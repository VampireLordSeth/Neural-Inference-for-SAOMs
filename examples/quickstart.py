"""saomsim quickstart.

Three things:
  1. more reciprocity parameter  -> more mutual dyads
  2. more homophily parameter    -> larger share of within-group ties
  3. method-of-moments recovery of a known parameter vector from a batch of
     simulated panels, with standard errors
"""

import time

import numpy as np

from saomsim import Model, estimate, random_network, simulate_period, statistics

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

# ------------------------------------------------------------------- throughput
print("\n== throughput (CPU, density+recip+transTrip, rate=3) ==")
m = Model(["density", "recip", "transTrip"])
for n_, B_ in [(30, 4000), (100, 400)]:
    X0 = random_network(B_, n_, 0.1, rng)
    t0 = time.perf_counter()
    simulate_period(X0, [-2.0, 1.0, 0.2], 3.0, m, rng)
    dt = time.perf_counter() - t0
    print(f"n={n_:>3}  B={B_:>5}  {B_ / dt:>8,.0f} panels/s")
