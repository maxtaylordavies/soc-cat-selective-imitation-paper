import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from src.modelling.item_gridworld import item_values
from src.utils import value_similarity


def individual_agent_model(data):
    """
    Dirichlet process mixture model via the stick-breaking construction
    """
    choices, noise = data
    choices_per_agent = jnp.sum(choices)

    v = numpyro.sample(
        "v", dist.MultivariateNormal(jnp.zeros(2) + 0.5, 1.0 * jnp.eye(2))
    )
    v_ = jnp.stack(item_values(v[..., 0], v[..., 1], as_dict=False), axis=-1)
    p = jnp.exp(v_ / noise)
    p /= jnp.sum(p, axis=-1, keepdims=True)

    # choices contains the empirical choice proportions for each agent
    # we can compute their log likelihood under a multinomial distribution
    choices_log_prob = dist.Multinomial(
        probs=p, total_count=choices_per_agent
    ).log_prob(choices)

    numpyro.factor(
        "obs",
        choices_log_prob,
    )


def individual_inference(
    rng_key,
    obs_history,
    new_obs,
    beta,
    v_self,
    v_domain,
    phi_self,
    ingroup_strength,
    num_options=4,
    model=individual_agent_model,
    **kwargs,
):
    if new_obs[0]["choices"] is None:
        return jnp.ones(len(new_obs)) / len(new_obs)

    weights = jnp.zeros(len(new_obs))
    for m, a in enumerate(new_obs):
        tmp = jnp.array(a["choices"])
        choice_counts = jnp.bincount(tmp, minlength=num_options)

        # infer value function from observed choice counts
        kernel = numpyro.infer.NUTS(individual_agent_model)
        mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=1000)
        mcmc.run(rng_key, (choice_counts, beta))

        # compute expected similarity over posterior samples
        samples = mcmc.get_samples()
        sims = jnp.array([value_similarity(v_self, v)[1] for v in samples["v"]])
        weights = weights.at[m].set(sims.mean())

    if jnp.sum(weights) == 0:
        return jnp.ones(len(new_obs)) / len(new_obs)
    return weights / jnp.sum(weights)
