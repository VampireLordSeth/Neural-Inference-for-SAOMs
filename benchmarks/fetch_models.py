"""Fetch the trained estimators.

    python benchmarks/fetch_models.py                  # the ones you probably want
    python benchmarks/fetch_models.py --all            # every published estimator
    python benchmarks/fetch_models.py npe_m6.pt        # one by name
    python benchmarks/fetch_models.py --list

The estimators are build artefacts of the runs documented in ``docs/M*_RESULTS.md``: ~39 MB
of trained flows that are not in git. They are published as assets on a GitHub release and
archived on Zenodo, and either source works. Downloads are checked against the SHA256 sums
recorded here, because a truncated or substituted flow will happily return posteriors and
give no sign that anything is wrong.

``MODELS.md`` says what each one covers and, more importantly, the range of ``n`` outside
which its readings should not be trusted.
"""

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "VampireLordSeth/Neural-Inference-for-SAOMs"
TAG = "estimators-v0.2.0"
GITHUB = f"https://github.com/{REPO}/releases/download/{TAG}/"
# Filled in once the Zenodo deposition exists; until then GitHub is the only source.
ZENODO = None

# name -> (sha256, one-line description). Sums generated when the release was staged; a
# mismatch means the file is not the one these results were produced with.
MODELS = {
    "npe_m6.pt": (
        "370806dac62ef315",
        "network, 2 waves, n 20-150 -- widest; use outside [30, 100]",
    ),
    "npe_m5c.pt": (
        "5a2a34267d5bdd0d",
        "network, 2 waves, n 30-100 -- best calibrated; the default",
    ),
    "npe_coev_m5c.pt": (
        "b7d3887186f4a1d0",
        "network x behaviour, 2 waves, n 30-100",
    ),
    "npe_m4b.pt": ("e3db9efedbd7e037", "network, 3 waves, n 20-200 (10^6 budget)"),
    "npe_coev_m4_10m.pt": ("802c88b460579153", "network x behaviour, 3 waves, n 20-200"),
    "npe_m5c_mom.pt": ("511f72399a3824ed", "the summary-set ablation; a result, not for use"),
    "npe_m5.pt": ("703fa783e43efbc6", "superseded; kept for the record"),
    "npe_m5b.pt": ("9235b3fd825f4b47", "superseded; kept for the record"),
    "npe_coev_m5.pt": ("93135efd3ed67745", "superseded; kept for the record"),
    "npe_coev_m5b.pt": ("daf24b0402a4e180", "superseded; kept for the record"),
    "npe_coev_m4.pt": ("b7ce904f04da1a81", "superseded; kept for the record"),
    "npe_coev_m4b.pt": ("2b69d8ef129f1d8c", "superseded; kept for the record"),
}
DEFAULT = ["npe_m6.pt", "npe_m5c.pt", "npe_coev_m5c.pt"]


def fetch(name: str, dest: Path, check: bool = True) -> Path:
    out = dest / name
    if out.exists() and (not check or _sha16(out) == MODELS[name][0]):
        print(f"  {name}: already present")
        return out
    last = None
    for base in (GITHUB, ZENODO):
        if base is None:
            continue
        try:
            print(f"  {name}: {base}")
            with urllib.request.urlopen(base + name, timeout=300) as r:
                out.write_bytes(r.read())
            break
        except (urllib.error.URLError, OSError) as e:
            last = e
    else:
        raise SystemExit(
            f"could not download {name} ({last}).\n"
            f"Download it by hand from https://github.com/{REPO}/releases/tag/{TAG} "
            f"into {dest}, or regenerate it (MODELS.md)."
        )
    if check:
        got = _sha16(out)
        if got != MODELS[name][0]:
            out.unlink(missing_ok=True)
            raise SystemExit(
                f"{name}: checksum {got} does not match the expected {MODELS[name][0]}. "
                "The download was truncated or the file is not the published one; it has "
                "been deleted rather than left in place."
            )
    print(f"  -> {out} ({out.stat().st_size / 1e6:.1f} MB, sha256 ok)")
    return out


def _sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", help="estimator file names; default is the useful ones")
    ap.add_argument("--all", action="store_true", help="every published estimator")
    ap.add_argument("--list", action="store_true", help="show what is published and stop")
    ap.add_argument("--out", default=str(ROOT / "data"), help="download directory")
    ap.add_argument("--no-check", action="store_true", help="skip the checksum (not advised)")
    a = ap.parse_args()

    if a.list:
        print(f"{'estimator':<22}{'sha256[:16]':<20}what it is")
        for n, (h, d) in MODELS.items():
            print(f"{n:<22}{h:<20}{d}")
        print(f"\nDefault set: {', '.join(DEFAULT)}\nSee MODELS.md for ranges and calibration.")
        return

    names = list(MODELS) if a.all else (a.which or DEFAULT)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        sys.exit(f"unknown estimator(s) {unknown}; try --list")
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        fetch(n, dest, check=not a.no_check)
    print(f"\n{len(names)} estimator(s) in {dest}. MODELS.md says what each one covers.")


if __name__ == "__main__":
    main()
