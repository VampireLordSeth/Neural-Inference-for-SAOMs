# M4 prior, step 2: start networks that stay sparse as n grows

Written 2026-09-16, before generating the step-2 training set. Motivation in
`docs/M4_RESULTS.md`: the step-1 estimator (M2 population widened to n ≤ 200)
is calibrated at every size but reads the Glasgow rates 25 % low, and the
Glasgow start sits at the 5th percentile of the training population on tie
fraction and the 2nd on out-degree spread. The M2 start prior draws a tie
fraction d ~ U(0.02, 0.20) whatever n is; real friendship networks keep a mean
degree of roughly 2–8 whatever n is, so their tie fraction falls as 1/n.

## What changes

One thing: the start-network regime (`sample_start_networks(..., start="sparse")`).

| component | step 1 (`start="m2"`) | step 2 (`start="sparse"`) |
|---|---|---|
| tie fraction of the ER seed | d ~ U(0.02, 0.20) | 50 %: as before. 50 %: mean degree k ~ U(2, 10), d = k / (n − 1) |
| burn-in | 50 % of starts: 10·n SAOM ministeps at θ₀ ~ prior | unchanged, applied to both regimes |
| nomination cap | none | half of the mean-degree starts: after burn-in, every actor keeps at most ceil(k) + U{1, 2, 3} out-ties (a random subset of theirs) |

Everything else — n ~ U{20..200}, rates U(1, 20), the seven effects and their
box, covariates, summaries, the estimator — is step 1's.

## Why these choices

- **A mixture, not a replacement.** Keeping half the starts on the old range
  preserves coverage of dense small groups (at n = 20, d = 0.20 is mean degree
  4; the old range is the sparse one there) and keeps s50 — mean degree 2.3 at
  n = 50 — inside both halves. The new half puts a quarter of the training set
  where real school networks of 100–200 pupils actually are: mean degree 2–10,
  tie fraction 0.01–0.10.
- **Mean degree uniform on (2, 10).** Glasgow 3.5, s50 2.3, the Knecht
  classrooms 3–5, typical adolescent friendship surveys 3–8. The upper end
  overlaps the old range at every n so there is no gap in tie fraction.
- **A cap, on half the sparse starts.** Most school surveys ask for "up to k
  friends", and it shows: Glasgow's out-degree sd is 1.6 against 4.0 for a
  training start of the same size. The cap is set just above the mean degree
  (ceil(k) + 1…3) so it binds on a minority of actors, as a real cap does, and
  it is applied after the burn-in so the burnt-in structure (mutuality,
  closure) survives in the ties that are kept. Only the start is capped; the
  dynamics are the plain SAOM, as in RSiena's default. If a later wave is
  observed under a cap the estimator will see a lower out-degree sd than the
  model produces — a mismatch we accept for now and can measure on Glasgow's
  waves 2–3 summaries.
- **The population is wider, not different.** Every step-1 start is still
  possible; the estimator learns a broader conditional. The cost is the same
  generation budget (10⁶ panels ≈ 3.7 h at n ≤ 200) spread over more of the
  input space, which the step-1 calibration margins (within 3.5 points) leave
  room for.

## What would make it wrong

If the Glasgow rates still read low with the new population, the density
explanation is insufficient and the next suspects are the summary set (no
statistic separates "few changes because sparse" from "few changes because
slow") and the cap on later waves. Both are testable on the same trained
model: the in-distribution screen on Glasgow's x1/x2 blocks, and a synthetic
Glasgow-sized panel simulated at RSiena's estimates.

## Cost

`saomsim.population`: one regime flag, a cap function, tests. Generation
≈ 3.7 h, training ≈ 3.5 h, SBC minutes. Seed 71; `data/train_m4b.npz`,
`npe_m4b.*`.
