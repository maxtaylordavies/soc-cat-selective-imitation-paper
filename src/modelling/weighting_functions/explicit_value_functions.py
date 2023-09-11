import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from src.utils.utils import value_similarity, log, norm_unit_sum
from src.utils.probabilistic import posterior_predictive
from src.modelling.world import traj_reward_1d


def multivariate_gaussian(data):
    dim = data.shape[-1]
    mu = numpyro.sample("mu", dist.Normal(0.5 * jnp.ones(dim), jnp.ones(dim)))
    sigma = numpyro.sample(
        "sigma", dist.InverseGamma(0.5 * jnp.ones(dim), 0.5 * jnp.ones(dim))
    )
    cov = jnp.diag(sigma)
    numpyro.sample("obs", dist.MultivariateNormal(mu, cov), obs=data)


def value_funcs_known(rng_key, known_agents, cat_phis, vself):
    weights = jnp.zeros(len(cat_phis))

    grouped = {}
    for a in known_agents:
        if a["phi"] not in grouped:
            grouped[a["phi"]] = []
        grouped[a["phi"]].append(a["v"])

    # for each phi, infer the posterior predictive distribution over value functions
    # for agents with that phi, and use it to compute the expected similarity
    for i, phi in enumerate(cat_phis):
        samples = posterior_predictive(
            rng_key, multivariate_gaussian, jnp.array(grouped[phi])
        )
        expected_sim = jnp.mean(
            jnp.array([value_similarity(vself, v) for v in samples["obs"]])
        )
        weights = weights.at[i].set(expected_sim)

    return norm_unit_sum(weights)


def value_func_likelihood(v, trajectories, c, alpha):
    r = jnp.sum(jnp.array([traj_reward_1d(traj, v, c) for traj in trajectories]))
    return jnp.exp(alpha * r)


# specify numpyro model for value function distribution based on set of trajectories
def value_func_model(trajectories, c, alpha, dim):
    mu = numpyro.sample("mu", dist.Normal(0.5 * jnp.ones(dim), jnp.ones(dim)))
    sigma = numpyro.sample(
        "sigma", dist.InverseGamma(0.5 * jnp.ones(dim), 0.5 * jnp.ones(dim))
    )
    cov = jnp.diag(sigma)
    v = numpyro.sample("v", dist.MultivariateNormal(mu, cov))
    numpyro.factor("likelihood", value_func_likelihood(v, trajectories, c, alpha))


# def value_funcs_inferred(rng_key, known_agents, cat_phis, vself):
#     weights = jnp.zeros(len(cat_phis))

#     return norm_unit_sum(weights)
