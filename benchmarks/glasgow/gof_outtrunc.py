"""Does adding a truncated out-degree effect repair the Glasgow goodness of fit?

    python benchmarks/glasgow/gof_outtrunc.py [--B 2000] [--caps 5 4]

`docs/GOF_RESULTS.md` found that the canonical effect set cannot reproduce Glasgow's
out-degree distribution, triad census or geodesic distribution, and that truncating the
simulated networks does not repair it: the fault is in the dynamics, not the observation.
This runs the same check at RSiena's estimate *with* `outTrunc(c)` in the model, against
the same check at its canonical estimate, so the two are directly comparable — same data,
same statistics, same simulator, one effect different.

Both are point estimates, so both under-state their predictive spread in the same way
(`docs/GOF_RESULTS.md`); the comparison between them is what this is for, not the absolute
p-values.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from saomsim import Model, simulate_period  # noqa: E402
from saomsim.gof import TRIAD_TYPES, auxiliary, mahalanobis_test  # noqa: E402

GLAS = ROOT / "benchmarks" / "glasgow"
BASE = ["density", "recip", "transTrip", "cycle3", ("altX", "v"), ("egoX", "v"), ("sameX", "g")]
KEYS = ["net:density", "net:recip", "net:transTrip", "net:cycle3",
        "net:altX", "net:egoX", "net:sameX"]
LABELS = {
    "outdegree": [f"out {k}" for k in range(8)] + ["out 8+"],
    "indegree": [f"in {k}" for k in range(8)] + ["in 8+"],
    "triad census": TRIAD_TYPES,
    "geodesic": ["d=1", "d=2", "d=3", "d=4", "d=5", "d>5 or inf"],
}


def load_panel():
    X = [np.loadtxt(GLAS / f"glasgow_net{w}.csv", delimiter=",", dtype=np.int8) for w in (1, 2, 3)]
    cov = np.genfromtxt(GLAS / "glasgow_covariates.csv", delimiter=",", names=True)
    return X, cov["alc1_centred"], cov["sex"] - cov["sex"].min()


def run(tag, path, cap, X, v, g, B, rng):
    """Simulate both periods at one RSiena point and test the auxiliary statistics."""
    est = json.loads(Path(path).read_text(encoding="utf-8"))["estimate"]
    effects = list(BASE) if cap is None else [*BASE, ("outTrunc", None, cap)]
    keys = list(KEYS) if cap is None else [*KEYS, "net:outTrunc"]
    theta = np.array([est[k] for k in keys])
    rates = [est["net:rate_1"], est["net:rate_2"]]

    covs = {"v": np.repeat(v[None], B, 0), "g": np.repeat(g[None], B, 0)}
    model = Model(effects, covs)
    print(f"\n### {tag}")
    print("    " + "  ".join(f"{e if isinstance(e, str) else e[0]}={t:+.2f}"
                             for e, t in zip(effects, theta, strict=True)))
    out = {}
    for w in range(2):
        X0 = np.repeat(X[w][None], B, 0).astype(np.int8)
        Xs = simulate_period(X0, np.tile(theta, (B, 1)), rates[w], model, rng)
        sim, obs = auxiliary(Xs), auxiliary(X[w + 1][None].astype(np.int8))
        od = Xs.sum(2)
        print(f"  period {w + 1}   simulated max out-degree {od.max(1).mean():4.1f}, "
              f"{100 * (od > 6).mean():4.1f} % of actors above the survey cap of 6")
        for key in ("outdegree", "indegree", "triad census", "geodesic"):
            p, _, contrib = mahalanobis_test(sim[key], obs[key][0])
            worst = np.argsort(-np.abs(contrib))[:2]
            bits = ", ".join(f"{LABELS[key][j]} {contrib[j]:+.1f}" for j in worst)
            print(f"    {key:<14}p {p:6.3f}   {bits}" + ("   <-- rejected" if p < 0.05 else ""))
            out[(w, key)] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--caps", type=int, nargs="*", default=[5, 4])
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    X, v, g = load_panel()
    rng = np.random.default_rng(a.seed)
    print(f"Glasgow, n = {len(X[0])}, {a.B} simulations per period, "
          f"observed max out-degree {max(x.sum(1).max() for x in X)}")

    results = {"canonical": run("canonical effect set", GLAS / "rsiena_network3w.json",
                                None, X, v, g, a.B, rng)}
    for c in a.caps:
        f = GLAS / f"rsiena_network3w_outtrunc{c}.json"
        if not f.exists():
            print(f"\n(skipping outTrunc({c}): {f.name} not found)")
            continue
        results[f"outTrunc({c})"] = run(f"+ outTrunc({c})", f, c, X, v, g, a.B, rng)

    print("\n" + "=" * 62)
    print(f"{'statistic':<16}" + "".join(f"{k:>16}" for k in results))
    for w in range(2):
        for key in ("outdegree", "indegree", "triad census", "geodesic"):
            row = "".join(f"{results[k][(w, key)]:>16.3f}" for k in results)
            print(f"p{w + 1} {key:<13}" + row)
    tot = {k: sum(p < 0.05 for p in r.values()) for k, r in results.items()}
    print(f"{'rejected of 8':<16}" + "".join(f"{tot[k]:>16d}" for k in results))


if __name__ == "__main__":
    main()
