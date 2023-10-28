import jax.numpy as jnp


def indiscriminate(num_phis: int) -> jnp.ndarray:
    return jnp.ones(num_phis) / num_phis


def ingroup_bias(num_phis: int, phi_self: int, strength: float) -> jnp.ndarray:
    weights = jnp.zeros(num_phis)
    weights = weights.at[phi_self].set(1.0)
    return weights * strength
