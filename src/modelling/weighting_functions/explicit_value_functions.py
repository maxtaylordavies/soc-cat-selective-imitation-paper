from jax import random
import jax.numpy as jnp
from tqdm import tqdm

from src.utils.utils import value_similarity, log, norm_unit_sum
from src.modelling.probabilistic import multivariate_gaussian_model, mcmc_posterior_predictive

# from src.modelling.item_gridworld import traj_reward_1d, item_values_1d


def value_funcs_known(rng_key, agents, phis, vself):
    weights = jnp.zeros(len(phis))

    grouped = {}
    for a in agents:
        if a["phi"] not in grouped:
            grouped[a["phi"]] = []
        grouped[a["phi"]].append(a["v"])

    # for each phi, infer the posterior predictive distribution over value functions
    # for agents with that phi, and use it to compute the expected similarity
    for i, phi in enumerate(phis):
        samples = mcmc_posterior_predictive(
            rng_key, multivariate_gaussian_model, jnp.array(grouped[phi])
        )
        expected_sim = jnp.mean(
            jnp.array([value_similarity(vself, v) for v in samples["obs"]])
        )
        weights = weights.at[i].set(expected_sim)

    return norm_unit_sum(weights)


# def value_funcs_inferred_individual(rng_key, agents, phis, vself, c=0.1, alpha=10, bins=50):
#     # infer value function for each agent
#     agents_ = []
#     for a in tqdm(agents):
#         domain, probs = brute_force_birl_1d(a["trajs"], c, alpha, bins=bins)
#         v = domain[jnp.argmax(probs)]
#         agents_.append({"phi": a["phi"], "v": v})

#     # then use value_funcs_known with the inferred value functions
#     return value_funcs_known(rng_key, agents_, phis, vself)


# def value_funcs_inferred_group(rng_key, agents, phis, vself, c=0.1, alpha=10, bins=50):
#     weights = jnp.zeros(len(phis))

#     grouped = {}
#     for a in agents:
#         if a["phi"] not in grouped:
#             grouped[a["phi"]] = []
#         grouped[a["phi"]].extend(a["trajs"])

#     for i, phi in enumerate(phis):
#         domain, probs = brute_force_birl_1d(grouped[phi], c, alpha, bins=bins)
#         samples = random.choice(rng_key, domain, p=probs, shape=(1000,))
#         expected_sim = jnp.mean(jnp.array([value_similarity(vself, v) for v in samples]))
#         weights = weights.at[i].set(expected_sim)

#     return norm_unit_sum(weights)


# def value_func_likelihood_1d(v, trajectories, c, alpha):
#     v_ = item_values_1d(v[0])
#     r = jnp.sum(jnp.array([traj_reward_1d(traj, v_, c) for traj in trajectories]))
#     return jnp.exp(r / alpha)


# def brute_force_birl_1d(trajectories, c, alpha, bins=50, prior=None):
#     domain = jnp.linspace(0, 1, bins).reshape((-1, 1))
#     probs = jnp.zeros(bins)
#     for i, v in enumerate(domain):
#         p = value_func_likelihood_1d(v, trajectories, c, alpha)
#         if prior is not None:
#             p *= prior(v)
#         probs = probs.at[i].set(p)
#     return domain, probs / probs.sum()
