"""Per-class comparison tables for the paper's appendices.

    python benchmarks/paper_tables.py [--model net|coev] [--markdown]

Emits, for every Baerveldt class, how far RSiena's estimate sits from our posterior. Two
standardisations are reported because they answer different questions and the results
documents have not always been explicit about which one is in use:

* **posterior sd** -- (our mean - RSiena's point) / our posterior sd. This asks whether
  RSiena's answer lies inside our interval, and is the number to quote alongside "all 152
  inside the 90 % interval".
* **combined sd** -- the same numerator over sqrt(our var + RSiena's se^2). This asks
  whether the two *estimates* agree given both uncertainties, and is the fairer test when
  the reference is itself imprecise: a class where RSiena's standard error is 6.5 cannot
  be said to disagree with anything.

The two differ by about a third here (largest 1.47 against 1.08), so quoting one while
computing the other would misstate the agreement in either direction.

Regenerating rather than transcribing these counts is the point of the script, and it
earned that on the first run: ``docs/M5_RESULTS.md`` had recorded 203 of 204 co-evolution
pairs inside the 90 % interval, a figure not reproducible from any of the three saved
posterior sets, which give 192, 196 and 200. The document has been corrected to 200. The
four pairs outside are class 9's ``simZ``, ``quad`` and ``avAlt`` -- the three whose
RSiena standard errors are 3.3, 2.1 and 6.5, so essentially unidentified -- and class
18's ``simZ``; all four sit inside once RSiena's own error is counted.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "net": (
        "data/baerveldt_c/school*_posterior.npz",
        "benchmarks/baerveldt/rsiena_network2w_{}.json",
    ),
    "coev": (
        "data/baerveldt_c/coev_school*_posterior.npz",
        "benchmarks/baerveldt/rsiena_coevolution_{}.json",
    ),
}


def match_key(est, name):
    """Our parameter name -> the key RSiena's export uses, or None.

    RSiena prefixes a co-evolution fit's keys with the *name of the variable*, so the
    network rate is ``net:rate_1`` but the behaviour rate is ``delB:rate_1`` -- the prefix
    is whatever the study called the behaviour, not the word "behaviour". Matching on the
    prefix that is not the network one is what makes this general.
    """
    base = name.split("(")[0]
    if base.startswith("rate"):
        rates = [k for k in est if ":rate" in k or k.startswith("rate")]
        if base in ("rate", "rate_net") or base.startswith("rate_net"):
            return next((k for k in rates if k.startswith("net:")), rates[0] if rates else None)
        # the behaviour rate: the one whose prefix is not "net"
        return next((k for k in rates if not k.startswith("net:")), None)
    sel = {"egoZ": "egoX", "altZ": "altX", "simZ": "simX"}.get(base, base)
    return next((k for k in est if k.split(":")[-1].split("(")[0] == sel), None)


def rows_for(model):
    pattern, ref_tmpl = SPECS[model]
    out = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        g = re.search(r"school(\d+)", Path(f).name).group(1)
        ref = ROOT / ref_tmpl.format(g)
        if not ref.exists():
            continue
        z = np.load(f, allow_pickle=True)
        names, S = [str(s) for s in z["names"]], z["samples"]
        d = json.loads(ref.read_text(encoding="utf-8"))
        est, se = d["estimate"], d.get("se", {})
        zp, zc, inside, within = [], [], 0, 0
        for k, nm in enumerate(names):
            key = match_key(est, nm)
            if key is None:
                continue
            num = abs(S[:, k].mean() - est[key])
            ps = S[:, k].std(ddof=1)
            zp.append(num / ps if ps > 0 else np.nan)
            e = se.get(key, 0.0) or 0.0
            zc.append(num / np.hypot(ps, e))
            lo, hi = np.percentile(S[:, k], [5, 95])
            inside += lo <= est[key] <= hi
            within += zc[-1] < 1.645
        n = len(np.loadtxt(ROOT / f"benchmarks/baerveldt/school{g}_net1.csv", delimiter=","))
        out.append((g, n, len(zp), inside, within, np.nanmax(zp), np.nanmax(zc), np.nanmean(zc)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="net", choices=list(SPECS))
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    rows = rows_for(a.model)
    if not rows:
        raise SystemExit(f"no per-class posteriors found for {a.model!r}")

    hdr = ["class", "n", "pairs", "in 90%", "comb<1.645", "max |z| post", "max |z| comb",
           "mean |z| comb"]
    if a.markdown:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for g, n, k, ins, wi, mp, mc, mn in rows:
            print(f"| {g} | {n} | {k} | {ins}/{k} | {wi}/{k} | {mp:.2f} | {mc:.2f} | {mn:.2f} |")
    else:
        print(f"{hdr[0]:>6}{hdr[1]:>5}{hdr[2]:>7}{hdr[3]:>9}{hdr[4]:>12}"
              f"{hdr[5]:>14}{hdr[6]:>14}{hdr[7]:>15}")
        for g, n, k, ins, wi, mp, mc, mn in rows:
            print(f"{g:>6}{n:>5}{k:>7}{ins:>6}/{k:<2}{wi:>9}/{k:<2}"
                  f"{mp:>14.2f}{mc:>14.2f}{mn:>15.2f}")

    tot = sum(r[2] for r in rows)
    ins, wi = sum(r[3] for r in rows), sum(r[4] for r in rows)
    print(
        f"\n{len(rows)} classes, {tot} parameter comparisons."
        f"\n  {ins} of {tot} with RSiena's point inside our 90 % posterior interval"
        f"\n  {wi} of {tot} within 1.645 combined sd"
        f"\n  largest |z| anywhere: {max(r[5] for r in rows):.2f} posterior sd, "
        f"{max(r[6] for r in rows):.2f} combined sd"
    )


if __name__ == "__main__":
    main()
