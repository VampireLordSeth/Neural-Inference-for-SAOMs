# Literature check: simulation-based / neural inference for SAOMs

Task zero in the paper plan (§7): confirm or revise the novelty claim before
the introduction is written. Web search, 2026-09-11. Not a substitute for a
Scholar / Semantic Scholar pass and a read of the SIENA bibliography page —
do those before submission — but enough to commit to the framing.

## Finding

**No published or preprint work was found that applies neural posterior
estimation, normalizing flows, ABC, or any amortized simulation-based
inference method to stochastic actor-oriented models.** Seven targeted
queries (SAOM / actor-oriented / RSiena × NPE / SBI / normalizing flow /
ABC / likelihood-free / neural / deep learning / amortized), plus a pass over
TERGMs, returned nothing on the longitudinal actor-oriented case. The
novelty claim in the draft stands, provisionally.

## Nearest neighbours (cite these)

**Cross-sectional ERGMs — the established case.**

- Fan, Y., & White, S. R. (2026). Neural posterior estimation on exponential
  random graph models: evaluating bias and implementation challenges.
  *Statistics and Computing*. arXiv:2504.09349 (submitted 12 Apr 2025).
  First systematic NPE for ERGMs; evaluates bias against exchange-algorithm
  Bayesian fits; notes ERGM-specific difficulties. Reports that training on
  5×10⁵ simulations replaces ~4×10⁹ simulations of conventional inference.
  *This completes the draft's placeholder reference.*
- Fan, Y., & White, S. R. (2025). A scalable exponential random graph model:
  amortised hierarchical sequential neural posterior estimation with
  applications in neuroscience. arXiv:2506.04558. Multiple-network ERGM,
  hierarchical NPE, Cam-CAN fMRI. Cross-sectional; no temporal component.
  *Relevant to §5 (many networks) as the ERGM analogue of what we propose.*

Neither mentions longitudinal or dynamic networks, SAOMs, TERGMs, or RSiena.

**Likelihood-based and Bayesian SAOM inference — per-network, not amortized.**

- Koskinen, J. H., & Snijders, T. A. B. (2007). Bayesian inference for
  dynamic social network data. *Journal of Statistical Planning and
  Inference*, 137, 3930–3938. MCMC with data augmentation over the
  unobserved ministep sequence.
- Snijders, T. A. B., Koskinen, J., & Schweinberger, M. (2010). Maximum
  likelihood estimation for social network dynamics. *Annals of Applied
  Statistics*, 4(2), 567–588. MCMC approximation of the MLE; the likelihood
  route RSiena offers alongside MoM.
- Amati, V., Schönenberger, F., & Snijders, T. A. B. (2015). Estimation of
  stochastic actor-oriented models for the evolution of networks by
  generalized method of moments. *Journal de la Société Française de
  Statistique*, 156(3), 140–165.
- Koskinen, J., & Snijders, T. A. B. (2023). Multilevel longitudinal
  analysis of social networks. *JRSS Series A*, 186(3), 376–400. Random
  coefficient SAOM across many networks, Bayesian. **This is the field's
  existing answer to multi-network inference** and the natural comparator
  for the §5 framing: hierarchical MCMC over all networks jointly, versus
  amortized per-network posteriors that can then be modelled hierarchically
  at negligible cost.
- Lospinoso, J., & Snijders, T. A. B. (2019). Goodness of fit for stochastic
  actor-oriented models. *Methodological Innovations*, 12(3). The GOF
  protocol §4 should mirror in graph space.

**SAOM extensions using simulation-heavy inference (not neural).**

- Accounting for edge uncertainty in stochastic actor-oriented models for
  dynamic network analysis (2025, bioRxiv / PMC). EM with particle
  filtering for measurement error in ties. Same per-network cost structure.

**TERGMs.** No neural or amortized inference found for temporal ERGMs
either (tergm package: MCMC-MLE / MPLE). So the longitudinal gap is not
SAOM-specific; it is the whole discrete-observation network-dynamics class.

**General SBI references the draft already has** (Cranmer et al. 2020;
Papamakarios & Murray 2016; Greenberg et al. 2019; Talts et al. 2018) are
correct. Add:

- Boelts, J., et al. (2025). sbi reloaded: a toolkit for simulation-based
  inference workflows. arXiv:2411.17337. (The library we use, v0.27.)
- Simulation-Based Inference: A Practical Guide (2025). arXiv:2508.12939.

## What this means for the draft

1. §1.3 can state the novelty claim, citing Fan & White (2026, 2025) as the
   cross-sectional precedent and Koskinen & Snijders (2007, 2023) and
   Snijders et al. (2010) as the existing Bayesian/likelihood routes for
   SAOMs — all per-network, none amortized.
2. §5 gains a sharper comparator: the multilevel SAOM (Koskinen & Snijders
   2023) is how many-network SAOM questions are answered today; its cost is
   the argument for amortization.
3. Risk register row "prior work already exists": severity can drop from
   High to Low pending the Scholar pass.

## Sources consulted

- https://arxiv.org/abs/2504.09349 · https://link.springer.com/article/10.1007/s11222-026-10896-8
- https://arxiv.org/abs/2506.04558
- https://projecteuclid.org/euclid.aoas/1280842131 · https://arxiv.org/pdf/1011.1753
- https://academic.oup.com/jrsssa/article/186/3/376/6998523
- http://www.numdam.org/item/JSFS_2015__156_3_140_0/
- https://dx.doi.org/10.1177/2059799119884282
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12959939/
- https://arxiv.org/pdf/2411.17337 · https://arxiv.org/pdf/2508.12939
- https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-060116-054035
