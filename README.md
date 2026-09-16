# Neural Inference for SAOMs

**Teaching a neural network to read social-network change — so that fitting a
model to a new network takes a fraction of a second instead of a fresh
simulation run every time.**

This repository holds the code, experiments and write-ups for a research
project on *amortized neural inference for stochastic actor-oriented models*.
Everything here is open: the simulator, the validation against the standard
software, the trained-estimator experiments, and the documents that record
each design decision before it was made.

---

## The problem in plain terms

Social scientists often observe the same group of people at several points in
time and ask *why* the ties between them changed. Did people befriend those
who were already friends of their friends? Did they pick friends who behave
like them, or start behaving like the friends they picked? A **stochastic
actor-oriented model (SAOM)** answers these questions by treating network
change as a series of small decisions by the people involved, and estimating
how strongly each mechanism — reciprocity, transitive closure, homophily,
influence — shaped those decisions.

SAOMs are the standard tool for this (the R package **RSiena** implements
them), but fitting one is expensive. There is no formula for the likelihood, so
RSiena estimates parameters by *simulating* the network forward thousands of
times and nudging the parameters until the simulations resemble the data. That
loop runs afresh for **every** network you analyse, takes seconds to minutes,
and can fail to converge. In practice this limits most studies to a handful of
networks. Comparing mechanisms across hundreds of classrooms, teams or online
communities — the kind of question the field would like to ask — is out of
reach.

## The idea

Instead of searching for parameters one network at a time, **learn the inverse
map once.** We use the simulator as an unlimited source of training examples:
draw plausible parameters, simulate a network panel, and record the pair. A
neural network (a *normalizing flow*) is trained on millions of such pairs to
turn an observed panel into a full probability distribution over the
parameters that could have produced it — a Bayesian posterior.

After training, analysing a new network is a single forward pass: about a
tenth of a second, no simulation loop, and you get uncertainty for free. The
up-front cost is paid once and shared across every network you analyse
afterwards — hence *amortized*. This approach is well established in physics
and biology (it is called simulation-based inference) and has recently been
applied to cross-sectional network models; to our knowledge this is its first
application to longitudinal actor-oriented models.

## What has been built and shown

The work follows a milestone plan in which nothing downstream depends on an
unvalidated step.

| milestone | what it is | status |
|---|---|---|
| **M0 — Simulator** | A fast, batched SAOM simulator (NumPy and GPU/PyTorch backends). Checked against RSiena on the standard `s50` teenage-friendship data: every statistic matches to the last digit, and simulated networks from both simulators have the same distribution. | done · `benchmarks/README.md` |
| **M1 — Proof of concept** | An estimator for one dataset (fixed start network). Trained on 10⁶ simulated panels; its posterior agrees with RSiena's estimates on all eight parameters and passes calibration checks. | done · `docs/M1_RESULTS.md` |
| **M2 — Generalization** | One estimator for *any* start network with 20–80 actors and any covariate layout. With 10⁷ training panels it is calibrated across the whole range and, on a real dataset it never saw, lands within 0.8 standard deviations of RSiena on every parameter. | done · `docs/M2_RESULTS.md` |
| **Out-of-distribution envelope** | What happens when a dataset falls outside the training population, and a one-line screen that catches most such cases. | done · `docs/OOD_RESULTS.md` |
| **M3a — Multiple waves** | Three observation waves, one rate per period. Matches RSiena's three-wave fit; the posterior tightens with the extra wave just as RSiena's standard errors do. | done · `docs/M3_RESULTS.md` |
| **M3b — Selection vs influence** | Networks and a behaviour (e.g. alcohol use) evolving together. Simulator validated against RSiena; a 14-parameter estimator is calibrated across sizes and behaviour scales and agrees with RSiena on the classic alcohol-and-friendship data, recovering both selection and influence. | done · `docs/M3_RESULTS.md` |
| **M4 — Application** | Many networks at once, at a scale existing tools cannot reach. Step 1: the estimator extends to networks of up to 200 actors, calibrated at every size; on the full Glasgow school panel (129 pupils) the first version matched RSiena on the structural effects but read the change rates low, traced to a training population too dense for a real school network; with starts that keep mean degree fixed as n grows (step 2) all nine parameters agree with RSiena, in 0.06 s against RSiena's 219 s. | in progress · `docs/M4_RESULTS.md` |

Headline numbers so far:

- **Speed.** Simulating one network period at n = 30 runs at ~120,000 panels
  per second on a single DGX Spark GPU; a million training panels takes a few
  minutes. Analysing a new dataset with a trained estimator takes ~0.1 s,
  against 11–30 s for RSiena's iterative fit.
- **Agreement.** On the `s50` benchmark data the amortized posterior contains
  RSiena's estimate for every parameter, in the two-wave, three-wave and
  general-population settings.
- **Honesty.** Calibration is tested by simulation-based calibration and
  coverage on thousands of held-out simulations; where the estimator is not
  perfect (a small bias on one parameter, a mild size dependence) the
  documents say so and show the data.
- **Failure modes.** Off-scale inputs are flagged automatically; a badly
  misspecified model is *not* — and the write-up explains why that is true of
  every method and what check catches it (posterior predictive checks, which
  the amortized posterior makes nearly free).

## How to read this repository

- **`docs/`** — the story, in order. `PRIORS*.md` files record each design
  decision *before* the corresponding experiment (what the training
  distribution is, why those ranges, what would count as failure). `*_RESULTS.md`
  files report what happened, including negative results.
- **`saomsim/`** — the Python package: the simulator (`simulate.py`,
  `effects.py`, `behaviour.py`), the array backends (`backend.py`,
  `backend_torch.py`), the slow-but-obviously-correct reference
  implementations every kernel is tested against (`reference.py`), two
  classical estimators for comparison (`estimate.py`), the training-set
  machinery (`prior.py`, `population*.py`) and a learned graph embedding
  (`embedding.py`).
- **`benchmarks/`** — the RSiena comparisons (R scripts and their exported
  results, plus Python tests that reproduce them without R), the training
  and evaluation scripts for each milestone, and the `s50` example data.
- **`tests/`** — the test suite (~150 tests, run against both backends).
- **`GETTING_STARTED.md`** — setup and the working log.
- **`legacy/v0/`** — the original prototype, kept for the record.

## Try it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                  # NumPy only; add ".[torch]" for the GPU backend
pytest -q -m "not slow"
python examples/quickstart.py
```

`examples/quickstart.py` simulates networks, shows that reciprocity and
homophily parameters do what their names say, and recovers known parameters
with the classical method-of-moments estimator. The neural estimators are
trained with the scripts in `benchmarks/` (see the docstring at the top of
each); trained models and training sets are not stored in git.

Minimal use of the simulator:

```python
import numpy as np
from saomsim import Model, random_network, simulate_period, statistics

rng = np.random.default_rng(0)
model = Model(["density", "recip", "transTrip"])
X0 = random_network(B=1000, n=30, density=0.08, rng=rng)                 # 1000 start networks
X1 = simulate_period(X0, theta=[-2.2, 1.2, 0.35], rate=3.0, model=model, rng=rng)
S = statistics(X1, model)                                                  # (1000, 3) statistics
```

Add `backend="torch"` to run on a GPU.

## Conventions worth knowing

- **Reference implementations are sacred.** Every vectorized kernel is
  checked against a loop-based version written straight from the formulas.
  When they disagree, the fast one is wrong.
- **Nothing is filtered.** Simulated networks that end up empty or complete
  stay in the training set; dropping them would silently change what the
  estimator learns.
- **One random-number generator, passed explicitly.** A seed fixes the whole
  run, on either backend.
- **RSiena is the arbiter of conventions.** Where a statistic can be defined
  more than one way (it happened twice: `cycle3` counting and the behaviour
  similarity constant), the definition RSiena uses is the one implemented,
  and the check that found it is a permanent test.

## Data

The `s50` data (`benchmarks/s50*.csv`) are the example data distributed with
RSiena (GPL-3), an excerpt of the Teenage Friends and Lifestyle Study. They
are redistributed unchanged so the parity tests run without R.

## Status and contact

Active research, September 2026. The plan, results and open questions are all
in `docs/`. Issues and questions are welcome on the repository.
