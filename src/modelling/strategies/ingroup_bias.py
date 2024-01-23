import jax.numpy as jnp


def ingroup_bias(
    rng_key,
    obs_history,
    new_obs,
    beta,
    v_self,
    v_domain,
    phi_self,
    ingroup_strength,
    **kwargs,
) -> jnp.ndarray:
    if phi_self is None:
        return jnp.ones(len(new_obs)) / len(new_obs)

    target_phis = jnp.array(sorted([a["phi"] for a in new_obs]))
    weights = jnp.zeros_like(target_phis)
    weights = weights.at[target_phis == phi_self].set(1)
    return weights * ingroup_strength
