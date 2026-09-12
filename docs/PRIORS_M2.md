# M2 prior: one estimator across start networks, sizes and covariates

Written before generating M2 training data (paper plan M2; GETTING_STARTED §7
after item 6). M1 (`docs/PRIORS.md`, `docs/M1_RESULTS.md`) conditioned on one
empirical start. M2 learns

    p(theta | X1, X0, covariates, n)

for a *population* of starts, so a single trained network serves any dataset
whose (n, X0, covariates) fall inside the population. Decision taken
2026-09-12: **option (b), a generative prior over start networks**, rather than
a fixed set of empirical starts.

## 1. Why conditioning on X0 makes the X0 prior cheap to get "wrong"

The estimator conditions on X0. In-distribution, the learned q(theta | X1, X0)
targets the true posterior *given* X0, whatever distribution X0 was drawn from
during training — the X0 prior decides **where** the estimator is trained,
not **what** it estimates. So the X0 population only has to be broad enough
to cover the real starts we care about; it does not have to be realistic in
its frequencies. Coverage is checked, not assumed (§5).

## 2. The population

| component | prior | reasoning |
|---|---|---|
| n | integer U{20, …, 80} | Covers classroom / small-group panels (s50 at 50 sits mid-range). Below 20 the statistics are too coarse; above 80 the cost per panel grows as n² and larger groups are rarer in the applied literature. One n per simulation chunk, so batches stay dense tensors. |
| covariate `v` (continuous) | 50 %: N(0, 1); 50 %: uniform on {1..K} with K ∈ {3, 4, 5}, centred | Standardised continuous covariates and Likert-type items (like s50's alcohol) are the two shapes seen in practice. Centred because RSiena centres. |
| covariate `g` (categorical) | K_g ∈ {2, 3, 4} categories, actors assigned uniformly | For `sameX`. Group sizes vary naturally. |
| X0 | 50 %: Erdős–Rényi with tie fraction d ~ U(0.02, 0.20); 50 %: 10·n SAOM ministeps from such an ER seed at θ₀ ~ the θ prior | ER alone has no reciprocity or clustering; empirical starts do. Burning in under an *independent* θ₀ produces structured starts (mutual dyads, triangles, degree heterogeneity) without asserting that X0 is stationary under the θ being inferred — that assumption is not true of real panels and would leak information into the posterior. |
| θ = (rate, density, recip, transTrip, cycle3, altX(v), egoX(v), sameX(g)) | the M1 box (`docs/PRIORS.md` §3), unchanged | Same effects as M1 so the s50 data are a valid held-out test with a real start. Same ranges so M1 and M2 posteriors are comparable. |

The three covariate effects refer to `v`, `v`, `g` respectively. Applying
the estimator to s50 maps alc → `v` (centred), smk → `g`.

## 3. What is observed by the estimator

Everything the analyst has: n, both waves, both covariates. Baseline
embedding (this milestone): hand summaries, then a flow. Learned embedding
on the raw (X0, X1, v, g) is the increment after it; networks are stored for
that purpose (padded to 80 × 80, bit-packed, with n).

Baseline summary vector (`m2_summaries`), all counts `log1p`, signed sums
`asinh`:

- n, and the covariate shape: sd(v), number of categories of g, entropy of g
- X0: the 7 model statistics of X0, outdeg_sd, indeg_sd, isolates, mutual
  dyads, tie fraction (12)
- X1: the same 12 plus the change count between waves (13)

Twenty-nine numbers. Sizes and densities enter both through n and through the
X0 block, which is what §2.3 of the paper plan calls "size and density
conditioning".

## 4. What is kept from M1's rules

- Nothing filtered. Degenerate X1 (and degenerate X0 from a burn-in that
  emptied or filled the network) stay in.
- One numpy Generator drives everything; a seed fixes the set.
- The θ prior box is the domain of validity; posterior mass against a face
  is a warning, not a result.

## 5. Checks before training (recorded in `docs/M2_RESULTS.md` when run)

1. Prior predictive over the population: quantiles of every summary, and the
   fraction of degenerate X0 / X1 by n.
2. **Is s50 in-distribution?** The observed (s501, s502, alc, smk) summary
   vector's percentile within the population prior predictive, per summary.
   Values in (0.01, 0.99) everywhere → s50 is a fair real-data test of M2.
3. After training: SBC and coverage over the population; posterior for s50
   compared with RSiena and with the M1 estimator (which had the advantage of
   being trained on s501 alone).

## 6. Budget

10⁶ panels at n ~ U{20..80} costs roughly the same as 10⁶ at n = 50
(≈ 3 min of GPU), plus the X0 burn-in (10·n ministeps, about a third of a
period) — call it 5 min. The stored networks are ~1.6 GB per 10⁶ (two waves,
packed). Training the summary-based flow: ~1–1.5 h as for M1. Moving to
10⁷ is a matter of minutes of simulation and a longer training run, and is
the plan once the learned embedding exists.
