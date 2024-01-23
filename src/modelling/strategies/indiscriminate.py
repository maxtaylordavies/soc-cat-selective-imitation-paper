import jax.numpy as jnp


def indiscriminate(
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
    target_phis = jnp.array([a["phi"] for a in new_obs])
    w = jnp.ones_like(target_phis)
    return w / jnp.sum(w)
