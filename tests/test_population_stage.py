"""The population stage recovers known (mu, tau) from per-group posterior summaries."""

import numpy as np
import pytest

from benchmarks.population_stage import gibbs


@pytest.mark.parametrize("G,mu0,tau0,s0", [(19, 2.3, 0.30, 0.25), (64, -2.2, 0.35, 0.15)])
def test_gibbs_recovers_mu_and_tau(G, mu0, tau0, s0):
    rng = np.random.default_rng(11)
    mus_hat, taus_hat, cover_mu = [], [], 0
    for r in range(40):
        theta_g = rng.normal(mu0, tau0, G)
        s = s0 * rng.uniform(0.7, 1.4, G)
        m = rng.normal(theta_g, s)
        mus, taus = gibbs(m, s, draws=3000, burn=400, rng=np.random.default_rng(r))
        q = np.quantile(mus, [0.05, 0.95])
        cover_mu += q[0] <= mu0 <= q[1]
        mus_hat.append(mus.mean())
        taus_hat.append(taus.mean())
    assert abs(np.mean(mus_hat) - mu0) < 0.1 * max(abs(mu0), 1)
    assert abs(np.mean(taus_hat) - tau0) < 0.1
    assert cover_mu >= 32  # 90% nominal, 40 replications


def test_gibbs_shrinks_tau_when_groups_agree():
    """No between-group variation: tau is pulled well below the within-group sd."""
    rng = np.random.default_rng(3)
    G, s0 = 19, 0.25
    s = s0 * rng.uniform(0.7, 1.4, G)
    m = rng.normal(0.5, s)  # all groups share theta = 0.5
    _, taus = gibbs(m, s, draws=4000, burn=500, rng=np.random.default_rng(0))
    assert taus.mean() < s0  # not resolvable from zero at this resolution
    assert np.quantile(taus, 0.05) < s0
