"""
saomsim quickstart.

Four demonstrations:
  1. simulate a batch of network panels from known parameters
  2. show that the effects do what their names claim
  3. recover known parameters by method of moments (the baseline to beat)
  4. measure simulation throughput, which sets the training-set budget

Run:  python examples/quickstart.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saomsim import (
    estimate_mom,
    make_effects,
    observed_distance,
    random_network,
    simulate_period,
    statistics,
)


def rule(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


# ---------------------------------------------------------------- 1
rule("1. Simulating a batch of panels")

rng = np.random.default_rng(20260911)
n, B = 40, 256
effects = make_effects(["density", "recip", "transTrip", ("sameX", "v")])
theta = np.array([-2.0, 1.5, 0.25, 0.8])
v = rng.integers(0, 2, size=n).astype(float)
cov = {"v": v}

X1 = random_network(n, 0.08, rng, B=B)
res = simulate_period(X1, theta, effects, cov, n_steps=150, rng=rng)

print(f"{B} chains, n = {n}, 150 micro-steps each")
print(f"effects       : {[e.name for e in effects]}")
print(f"theta         : {theta}")
print(f"mean wave-1 ties : {X1.sum(axis=(1,2)).mean():.1f}")
print(f"mean wave-2 ties : {res.X.sum(axis=(1,2)).mean():.1f}")
print(f"mean tie changes : {res.n_changes.mean():.1f} of 150 steps")
print(f"mean Hamming distance wave1->wave2 : "
      f"{observed_distance(X1, res.X).mean():.1f}")

S = statistics(res.X, effects, cov)
print("\nmean network statistics at wave 2:")
for k, e in enumerate(effects):
    print(f"  {e.name:<12} {S[:, k].mean():10.2f}  (sd {S[:, k].std():6.2f})")


# ---------------------------------------------------------------- 2
rule("2. Effects behave as advertised")

n2 = 24
X0 = random_network(n2, 0.15, rng, B=200)

eff_r = make_effects(["density", "recip"])
for r in (0.0, 1.5, 3.0):
    out = simulate_period(X0, [-1.5, r], eff_r, n_steps=400, rng=rng).X
    mutual = (out * np.swapaxes(out, 1, 2)).sum(axis=(1, 2)).mean() / 2
    print(f"  recip theta = {r:>4.1f}  ->  mean mutual dyads {mutual:7.2f}")

print()
v2 = np.repeat([0.0, 1.0], n2 // 2)
same = (v2[:, None] == v2[None, :]).astype(float)
eff_h = make_effects(["density", ("sameX", "v")])
for h in (0.0, 1.0, 2.5):
    out = simulate_period(X0, [-1.5, h], eff_h, {"v": v2}, n_steps=400, rng=rng).X
    within = ((out * same).sum(axis=(1, 2)) /
              np.maximum(out.sum(axis=(1, 2)), 1)).mean()
    print(f"  sameX theta = {h:>4.1f}  ->  share of ties within group {within:6.3f}")


# ---------------------------------------------------------------- 3
rule("3. Method-of-moments recovery (the baseline)")

n3 = 35
theta_true = np.array([-2.0, 1.6, 0.20])
eff3 = make_effects(["density", "recip", "transTrip"])

Xa = random_network(n3, 0.09, rng, B=1)
Xb = simulate_period(Xa, theta_true, eff3, n_steps=180, rng=rng).X[0]

print(f"true theta    : {theta_true}")
print(f"observed distance wave1->wave2 : {observed_distance(Xa, Xb[None])[0]}")
print("estimating (conditional on observed distance)...\n")

t0 = time.time()
fit = estimate_mom(Xa[0], Xb, eff3, rng=rng,
                   n_sim_phase1=200, n_iter_per_subphase=50,
                   n_sim_phase3=800)
elapsed = time.time() - t0

print(fit.summary())
print(f"\nwall clock: {elapsed:.1f}s  ({fit.n_iterations} Robbins-Monro iterations)")
bias = fit.theta - theta_true
print("bias vs truth : " + ", ".join(f"{b:+.3f}" for b in bias))
print("in s.e. units : " + ", ".join(
    f"{b / s:+.2f}" for b, s in zip(bias, np.maximum(fit.se, 1e-9))))


# ---------------------------------------------------------------- 4
rule("4. Throughput (sets the training-set budget)")

eff4 = make_effects(["density", "recip", "transTrip"])
for n4, B4, steps in [(30, 512, 100), (60, 256, 200), (100, 128, 300)]:
    Xs = random_network(n4, 0.06, rng, B=B4)
    t0 = time.time()
    simulate_period(Xs, [-2.0, 1.2, 0.2], eff4, n_steps=steps, rng=rng)
    dt = time.time() - t0
    print(f"  n={n4:>4}  B={B4:>4}  steps={steps:>4}  "
          f"{dt:6.2f}s   {B4 / dt:8.1f} panels/s")

print("""
Read the last column as the training-set budget. A neural posterior estimator
needs on the order of 10^5 to 10^6 simulated panels; divide that by the rate
above to get the CPU wall clock, then note that the per-step cost is three
batched matrix-vector products, which is exactly the shape that moves onto a
GPU well. That is the argument for the torch port, and it should be made with
measured numbers rather than asserted.
""")
