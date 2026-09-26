"""Fetch the benchmark datasets from their original sources.

    python benchmarks/fetch_data.py            # all four
    python benchmarks/fetch_data.py s50 glasgow

None of these datasets is redistributed in this repository: they belong to the studies
that collected them and are obtained here from the sources those studies publish. The
prepare scripts (`benchmarks/*/prepare_and_fit.R`) turn the downloads into the CSVs the
benchmarks read, and each writes its own RSiena reference fits.

| key | what | source | prepare with |
|---|---|---|---|
| s50 | 50 pupils, 3 waves, alcohol + smoking | the `RSiena` R package's own example
  data | (extracted directly, no R script) |
| glasgow | Teenage Friends and Lifestyle Study, 129 pupils, 3 waves | Siena datasets
  page (Glasgow_data.zip) | `glasgow/prepare_and_fit.R <dir>` |
| baerveldt | Dutch Social Behavior study, 19 schools, 2 waves | Siena datasets page
  (CB_data.zip) | `baerveldt/prepare_and_fit.R <dir>` |
| knecht | one classroom, 26 pupils, 4 waves | Siena datasets page (klas12b.zip) |
  `knecht/prepare_and_fit.R <dir>` |

Cite the original studies when you use them (`benchmarks/README.md`). The Siena datasets
page states no licence; treat the data as the authors' and follow their citation requests.
The Knecht classroom derives from a dissertation archived at DANS, which has its own
access conditions — check them before using it for anything beyond reproducing our fits.
"""

import argparse
import io
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
SIENA = "https://www.stats.ox.ac.uk/~snijders/siena/"
SOURCES = {
    "glasgow": SIENA + "Glasgow_data.zip",
    "baerveldt": SIENA + "CB_data.zip",
    "knecht": SIENA + "klas12b.zip",
}


def fetch_zip(url: str, dest: Path) -> None:
    """Download and unpack. As of 2026-09 the Siena site's TLS certificate has expired,
    which Python rejects and curl (with its own bundle) accepts, so fall back to curl and
    then to a manual instruction rather than turning verification off."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  {url}")
    blob = None
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            blob = r.read()
    except urllib.error.URLError as e:
        print(f"  urllib failed ({e.reason}); trying curl")
        try:
            blob = subprocess.run(
                ["curl", "-sSL", "--fail", url], check=True, capture_output=True
            ).stdout
        except (FileNotFoundError, subprocess.CalledProcessError) as e2:
            raise SystemExit(
                f"could not download {url} ({e2}).\n"
                f"Download it by hand into {dest} and unzip it, then re-run the prepare "
                "script with that directory."
            ) from e2
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(dest)
    print(f"  -> {dest} ({len(list(dest.iterdir()))} files)")


def fetch_s50(dest: Path) -> None:
    """s50 ships inside the RSiena package; export it rather than downloading."""
    dest.mkdir(parents=True, exist_ok=True)
    script = (
        'suppressPackageStartupMessages(library(RSiena));'
        f'd <- "{dest.as_posix()}";'
        'for (w in 1:3) write.table(get(paste0("s50", w)), file.path(d, paste0("s50", w, ".csv")),'
        ' sep=",", row.names=FALSE, col.names=FALSE);'
        'write.table(s50a, file.path(d, "s50a.csv"), sep=",", row.names=FALSE, col.names=FALSE);'
        'write.csv(data.frame(alc_centred = s50a[,1] - mean(s50a[,1]), smk = s50s[,1]),'
        ' file.path(d, "s50_covariates.csv"), row.names=FALSE);'
        'cat("s50 exported\n")'
    )
    try:
        subprocess.run(["Rscript", "-e", script], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise SystemExit(
            "s50 comes from the RSiena R package. Install R and RSiena, or put Rscript on "
            f"PATH, then re-run. ({e})"
        ) from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=["s50", "glasgow", "baerveldt", "knecht"])
    ap.add_argument("--out", default=str(HERE / "_data"), help="download directory")
    a = ap.parse_args()
    out = Path(a.out)
    for key in a.which:
        print(f"[{key}]")
        if key == "s50":
            fetch_s50(HERE)
        elif key in SOURCES:
            fetch_zip(SOURCES[key], out / key)
        else:
            sys.exit(f"unknown dataset {key!r}; choose from s50, glasgow, baerveldt, knecht")
    if any(k != "s50" for k in a.which):
        print(
            "\nnow run the prepare scripts, e.g.\n"
            f"  Rscript benchmarks/baerveldt/prepare_and_fit.R {out / 'baerveldt'}"
        )


if __name__ == "__main__":
    main()
