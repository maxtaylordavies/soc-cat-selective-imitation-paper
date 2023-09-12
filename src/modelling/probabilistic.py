from collections import Counter

import numpy as np
import jax.numpy as jnp
from scipy.special import gamma
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive


def clipped_gaussian(mus, stds, num=1):
    cov = np.diag(stds**2)
    samples = np.random.default_rng().multivariate_normal(mus, cov, num)
    return np.clip(samples, 0, 1)


def boltzmann1d(r, beta):
    p = np.exp(r / beta)
    return p / np.sum(p)


def boltzmann2d(r, beta):
    p = np.exp(r / beta)
    return p / np.sum(p, axis=1).reshape((-1, 1))


def crp(z, c, log=True):
    counts = Counter(z)
    coeff = ((c ** len(counts)) * gamma(c)) / gamma(c + len(z))

    if log:
        return coeff + np.sum([gamma(counts[k]) for k in counts])

    return coeff * np.prod([gamma(counts[k]) for k in counts])


def dirichlet_multinomial(counts, T, L, g, log=True):
    coeff = (gamma(L * g) * gamma(T + 1)) / gamma(T + (L * g))
    prods = np.prod(gamma(counts + g) / (gamma(g) * gamma(counts + 1)), axis=1)

    tmp = coeff * prods
    return np.sum(np.log(tmp)) if log else np.prod(tmp)


def multivariate_gaussian_model(data):
    dim = data.shape[-1]
    mu = numpyro.sample("mu", dist.Normal(0.5 * jnp.ones(dim), jnp.ones(dim)))
    sigma = numpyro.sample(
        "sigma", dist.InverseGamma(0.5 * jnp.ones(dim), 0.5 * jnp.ones(dim))
    )
    cov = jnp.diag(sigma)
    numpyro.sample("obs", dist.MultivariateNormal(mu, cov), obs=data)


def posterior_predictive(rng_key, model, data, num_warmpup=500, num_samples=1000):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmpup, num_samples=num_samples)
    mcmc.run(rng_key, data)
    posterior_samples = mcmc.get_samples()
    predictive = Predictive(model, posterior_samples, num_samples=1, batch_ndims=0)
    return predictive(rng_key, data)
