import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from src.utils.utils import value_similarity, log, norm_unit_sum
from src.utils.probabilistic import posterior_predictive


def multivariate_gaussian(data):
    dim = data.shape[-1]
    mu = numpyro.sample("mu", dist.Normal(0.5 * jnp.ones(dim), jnp.ones(dim)))
    sigma = numpyro.sample(
        "sigma", dist.InverseGamma(0.5 * jnp.ones(dim), 0.5 * jnp.ones(dim))
    )
    cov = jnp.diag(sigma)
    numpyro.sample("obs", dist.MultivariateNormal(mu, cov), obs=data)


def explicit_value_funcs(
    known_agents, cat_phis, vself, type="fit_distribution", rng_key=None
):
    weights = jnp.zeros(len(cat_phis))

    grouped = {}
    for a in known_agents:
        if a["phi"] not in grouped:
            grouped[a["phi"]] = []
        grouped[a["phi"]].append(a["v"])

    for k in grouped:
        grouped[k] = jnp.array(grouped[k])

    if type == "empirical_mean":
        means = {k: jnp.mean(v) for k, v in grouped.items()}
        for i, phi in enumerate(cat_phis):
            weights = weights.at[i].set(value_similarity(vself, means[phi]))
    elif type == "fit_distribution":
        # for each phi, infer the posterior predictive distribution over value functions
        # for agents with that phi, and use it to compute the expected similarity
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        for i, phi in enumerate(cat_phis):
            log(f"inferring posterior predictive distribution for phi={phi}")
            samples = posterior_predictive(rng_key, multivariate_gaussian, grouped[phi])
            log(f"computing expected similarity for phi={phi}")
            expected_sim = jnp.mean(
                jnp.array([value_similarity(vself, v) for v in samples["obs"]])
            )
            weights = weights.at[i].set(expected_sim)
    else:
        raise ValueError(f"Unknown type: {type}")

    return norm_unit_sum(weights)
