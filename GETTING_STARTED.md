# Getting started — saomsim in VS Code

Setup and first tasks for the SAOM simulator (milestone M0 of the amortized
neural inference project). The package was rebuilt from this document on
2026-09-11; the project root is this folder (`Network Inference/`), which
plays the role of `saom-amortized/` below.

---

## 1. Lay out the project

```
saom-amortized/
├── saomsim/            # the package (already written)
│   ├── __init__.py
│   ├── backend.py
│   ├── effects.py
│   ├── simulate.py
│   ├── estimate.py
│   └── reference.py
├── tests/
│   └── test_saomsim.py
├── examples/
│   └── quickstart.py
├── benchmarks/         # RSiena comparison lives here (README only so far)
├── notebooks/          # scratch only, nothing load-bearing
├── .vscode/            # settings.json, launch.json
├── pytest.ini
├── pyproject.toml
└── README.md
```

All of this exists. The repo is initialised as its own git repository — note
that `C:\Users\IG-88` is *also* a git repository, so a `git add -A` from the
wrong directory would try to stage the whole home directory. Always run git
from inside this folder.

The test suite is the asset; the first commit is the known-good point to
bisect back to.

## 2. Environment

```bash
python3 -m venv .venv              # Windows: py -3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

The package itself needs only NumPy. Everything else is tooling.

Torch comes later and is deliberately **not** in the base environment — see §6.
Keeping the simulator torch-free means the validation suite can't quietly start
depending on a GPU.

### `pyproject.toml`

```toml
[project]
name = "saomsim"
version = "0.1.0"
description = "Batched simulator for stochastic actor-oriented models"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
dev = ["pytest>=7", "pytest-xdist", "ruff"]
torch = ["torch>=2.2"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]
```

`pip install -e ".[dev]"` makes imports work from anywhere; there is no
`sys.path` juggling in the tests or examples.

## 3. VS Code

### Extensions

- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)
- **Ruff** (charliermarsh.ruff)
- **Jupyter** — only if you want inline plots while exploring
- **R** (REditorSupport.r) — needed for §5, the RSiena comparison

### `.vscode/settings.json`

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",   // Windows: .venv/Scripts/python.exe
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests", "-m", "not slow"],
  "python.testing.unittestEnabled": false,
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": { "source.organizeImports": "explicit" }
  },
  "files.exclude": { "**/__pycache__": true, "**/.pytest_cache": true }
}
```

Note the `-m "not slow"` in the test args. The Test Explorer should stay fast
enough that you actually run it. The slow recovery test goes in the launch
config instead.

### `.vscode/launch.json`

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Quickstart",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/examples/quickstart.py",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Tests (fast)",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["tests", "-q", "-m", "not slow"],
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Tests (including slow)",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["tests", "-q"],
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Current file",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

`justMyCode: false` matters — when a change statistic is wrong you want to step
into NumPy's einsum call, not bounce off it.

## 4. Verify

```bash
pytest -q -m "not slow"     # numpy-only env: 109 passed, 57 skipped (torch cases)
pytest -q                   # with torch: 165 passed
python examples/quickstart.py
```

The quickstart should print rising mutual dyads as the reciprocity parameter
increases, rising within-group tie share as homophily increases (chance level
0.483 at beta = 0), and a recovery table where each parameter sits within a
couple of standard errors of the truth (five parameters, so one of them near
|z| = 2 is normal; |z| > 3 is not).

Besides the reference comparisons, the suite contains
`test_simulation_matches_exact_stationary_distribution`, which checks the
simulator's long-run law on n = 3 against the exact 64-state ministep Markov
chain. It is the strongest internal check of the dynamics we have.

If any of that fails before you've changed anything, stop and work out why.
Everything downstream assumes this baseline.

## 5. First real task — close the M0 gate

**Status: closed 2026-09-11.** Both comparisons below pass against RSiena
1.6.6; see `benchmarks/README.md` for the term-by-term resolution. The one
convention difference found was `cycle3`: RSiena's target counts each 3-cycle
once, the actor sum counts it three times. `statistics()` now follows RSiena.
Everything else — including `recip` double-counting — agreed on the first run.

The original brief, kept for the record:

The simulator is internally consistent but has never been compared to RSiena.
Until it has, no claim of agreement can go in the paper.

```bash
# R side
R -e 'install.packages("RSiena", repos="https://cloud.r-project.org")'
```

Write `benchmarks/rsiena_compare.R` and `benchmarks/test_rsiena_parity.py`:

1. Pick a small fixed network pair — the `s501`/`s502` waves shipped with
   RSiena are the obvious choice — and export them to CSV.
2. In R, build the Siena data object and pull the **target statistics** for
   `density`, `recip`, `transTrip`, `cycle3`, and a covariate effect. Write
   them to JSON.
3. In Python, load the same CSVs, call `statistics()`, and compare.
4. Make the comparison a pytest test that reads the committed JSON, so it runs
   in CI without an R dependency. Commit both the JSON and the R script that
   generated it.

**Check `recip` and `cycle3` counting first.** The package sums
`s_i = Σ_j x_ij x_ji` over all actors, counting each mutual dyad twice (and each
3-cycle three times). RSiena's actor statistics are defined the same way, so
these may well agree — but confirm it against the manual before moving on. The
more likely discrepancies are in `transTrip` (which of the two two-path terms
RSiena includes) and in covariate centring (RSiena centres actor covariates by
default; `saomsim` uses them as given).

**Statistics agreeing is necessary, not sufficient.** Matching target
statistics validates the effect definitions, not the ministep dynamics. So
`benchmarks/rsiena_simulate.R` also runs `siena07(simOnly = TRUE, cond = FALSE)`
at fixed parameters from s501 and `test_rsiena_dynamics.py` compares the
distribution of simulated statistics with `simulate_period` from the same
`X0`. Means agree within Monte Carlo error, spreads within 5 %.

Resolve conventions term by term, and write down what you found. That
documentation is what lets a reviewer trust the benchmark section.

## 6. Second task — the torch port

**Status: done 2026-09-11.** `saomsim/backend_torch.py` implements the backend
interface; `simulate_period(..., backend="torch")` or a `TorchBackend(device,
dtype)` instance. The full suite runs against numpy and torch via the
`backend` fixture in `conftest.py`; 160 fast tests pass on the GB10 in both
float64 and float32, including the RSiena dynamics benchmark. Measured
throughput is in the package README; headline: 122k panels/s at n = 30
(CUDA f32) vs 5.2k numpy on the same machine.

One gotcha found: torch 2.13 routes a `(B, n, K) @ (B, K, 1)` bmm to a Triton
JIT kernel, which needs `python3-dev` on the host. The objective uses a
multiply-reduce instead. If Triton errors ever appear elsewhere,
`sudo apt install python3-dev` on the Spark is the fix.

The original brief, kept for the record:

Only after §5 passes.

The target machine is the DGX Spark (`ssh spark`, 192.168.1.49, GB10,
aarch64). It has torch 2.13+cu130 with CUDA in `~/spark-env`. Do **not** develop
in that venv: create a torch-free `.venv` in the project there too, so the
validation suite stays GPU-independent, and add torch to a second env:

```bash
pip install -e ".[torch]"
```

Everything that touches array layout is in `backend.py`. The port is:

| NumPy | torch |
|---|---|
| `X[rows, actor, :]` | `X[torch.arange(B), actor, :]` |
| `np.einsum("bh,bhj->bj", xi, X)` | `torch.bmm(xi.unsqueeze(1), X).squeeze(1)` |
| `np.einsum("bjh,bh->bj", X, xi)` | `torch.bmm(X, xi.unsqueeze(2)).squeeze(2)` |
| `softmax` (hand-rolled) | `torch.softmax` |
| `rng.random`, `rng.poisson` | `torch.rand`, `torch.poisson` |

Add a `device`/`backend` argument rather than replacing NumPy. Then run the
**same test suite** against both backends by parametrising a fixture. If the
torch path passes the brute-force change-statistic tests, the port is correct;
if it doesn't, you'll know before any training run burns hours.

Measure throughput before claiming a speedup. The CPU baseline from the
quickstart on the Windows laptop is roughly 2,800 panels/s at n=30 (B=4000) and
210/s at n=100 (B=400); `batched_row_products` already uses `matmul`, so the
`bmm` mapping in the table is one-to-one. On the Spark,
expect the win to come from batch width rather than per-chain latency — memory
bandwidth is modest, so treat it as a throughput box.

## 7. Working order

1. ~~Close the RSiena gate (§5).~~ Done 2026-09-11.
2. ~~Torch port, validated against the same tests (§6).~~ Done 2026-09-11.
3. Multi-wave support — `simulate_panel` runs consecutive periods and is
   tested for sequencing and per-period parameters; multi-wave *estimation*
   is not.
   (Also done 2026-09-11, from `legacy/v0`: `estimate_rm`, the RSiena-style
   single-panel Robbins-Monro baseline, conditional on observed distance; and
   the `n_steps=` / `distance=` stopping rules in `simulate_period`.)
4. Prior specification: write down the ranges and the reasoning *before*
   generating training data. This is a design decision with consequences for
   where the estimator can be trusted, not an implementation detail.
5. Generate the training set; attach a conditional normalizing flow via `sbi`.
6. Simulation-based calibration.

## 8. Conventions worth keeping

- **The reference implementations in `reference.py` are sacred.** They are slow
  and obviously correct. Never optimize them. When a vectorized kernel and its
  reference disagree, the kernel is wrong.
- **Never filter degenerate simulations.** Dropping explosive or empty networks
  silently changes the effective prior and biases the learned posterior. If you
  find yourself adding a `if stats.sum() > 0` guard, stop.
- **One RNG, passed explicitly.** No module-level global state. Reproducibility
  from a seed is what makes a failed run diagnosable.
- **Add the test before the feature.** Every effect added needs a brute-force
  reference and a parametrized test in the same commit.

## 9. If something breaks

| Symptom | Look at |
|---|---|
| `test_change_statistics_match_reference` fails | The einsum index order in `backend.batched_row_products`. `shared_out` and `two_path` are easy to transpose. |
| Recovery test fails but change stats pass | The derivative matrix in `estimate.py` — likely near-singular. Check `fd_step`, the `ridge` (applied in standardized units), `max_step`, and whether backtracking is exhausting `max_backtrack`. Run with `verbose=True`. |
| Simulation produces all-empty networks | Density parameter too negative, or the no-change option isn't being zeroed. |
| Non-reproducible runs | Something is drawing from a second RNG, or `categorical_sample` is consuming a variable number of draws. |
| Wildly large standard errors | Probably correct. A single small panel carries little information about triadic parameters. See §"A result worth noticing" in the package README. |
| `estimate` returns `converged=False` | Check `res.tratios`; if one statistic is stuck, its parameter is weakly identified from these panels. More panels or a larger `n_sim`; not a smaller `tol`. |
