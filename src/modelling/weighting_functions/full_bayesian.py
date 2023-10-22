from jax import random
import jax.numpy as jnp
import numpyro
from numpyro.contrib.funsor import config_enumerate
import numpyro.distributions as dist
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..item_gridworld import item_values
from ..probabilistic import run_svi
from src.utils.utils import value_similarity, log, norm_unit_sum

sns.set_theme()


@config_enumerate
def gmm_2d(data):
    choices, phis, K, beta = data

    # global variables
    weights = numpyro.sample("weights", dist.Dirichlet(jnp.ones(K) / K))
    with numpyro.plate("components", K):
        # sample mixture component means from a normal distribution
        mu = numpyro.sample(
            "mu", dist.MultivariateNormal(jnp.zeros(2) + 0.5, 0.1 * jnp.eye(2))
        )

        # sample covariance matrices. each component's covariance matrix is
        # diagonal, where the diagonal entries are distributed according to
        # an inverse gamma distribution.
        sigma1 = numpyro.sample("sigma1", dist.InverseGamma(0.5))
        sigma2 = numpyro.sample("sigma2", dist.InverseGamma(0.5))

        # sigma1 has shape (K,) and sigma2 has shape (K,), where sigma1[K] and sigma2[K]
        # are the diagonal entries of the covariance matrix for component K. now we use these
        # to construct the covariance matrix for each component. First we create a Kx2 matrix sigma,
        # where sigma[K, :] is the diagonal entries of the covariance matrix for component K.
        # Then we create a Kx2x2 matrix cov, where cov[K, :, :] is the covariance matrix for component K.
        sigmas = jnp.stack([sigma1, sigma2], axis=1)
        covs = jnp.stack([jnp.diag(s) for s in sigmas], axis=0)

        # each agent also expresses a discrete scalar 'surface feature', that may help identify their
        # latent group membership. we assume that this feature is distributed according to a categorical
        # distribution with K categories - each group has a different set of weights that parameterises the
        # categorical distribution.
        phi_weights = numpyro.sample("phi_weights", dist.Dirichlet(jnp.ones(K) / K))

    with numpyro.plate("agents", len(choices)):
        # sample group assignments
        z = numpyro.sample("z", dist.Categorical(weights))

        # sample a surface feature for each agent according to the sampled group assignments
        numpyro.sample("phi", dist.Categorical(phi_weights[z]), obs=phis)

        # sample a value function for each agent according to the sampled group assignments,
        # and convert each value function into a probability distribution over the four items
        v = numpyro.sample("v", dist.MultivariateNormal(mu[z], covs[z]))
        v_ = jnp.stack(item_values(v[..., 0], v[..., 1], as_dict=False), axis=-1)
        p = jnp.exp(v_ / beta)
        p /= jnp.sum(p, axis=-1, keepdims=True)

        # choices contains the empirical choice proportions for each agent
        # we can compute their log likelihood under a multinomial distribution
        log_likelihood = dist.Multinomial(probs=p, total_count=100).log_prob(choices)

        numpyro.factor(
            "obs",
            log_likelihood,
        )


def infer_conditional_v_distributions(
    rng_key,
    data,
    v_domain,
    plot_convergence=False,
    plot_dir="results/tmp",
    init_iter=50,
    run_iter=200,
):
    _, phis, K, _ = data

    # helper func to generate random parameter initialisations
    init_vals_func = lambda seed: {
        "weights": jnp.ones(K) / K,
        "mu": random.multivariate_normal(
            random.PRNGKey(seed), jnp.zeros(2) + 0.5, 0.1 * jnp.eye(2), shape=(K,)
        ),
        "sigma1": 0.5 * jnp.ones(K),
        "sigma2": 0.5 * jnp.ones(K),
        "phi_weights": jnp.ones((K, K)) / K,
    }

    # run SVI
    map_estimates = run_svi(
        rng_key,
        gmm_2d,
        data,
        init_vals_func,
        ["weights", "mu", "sigma1", "sigma2", "components", "phi_weights"],
        plot_convergence=plot_convergence,
        plot_dir=plot_dir,
        init_iter=init_iter,
        run_iter=run_iter,
    )

    # extract MAP estimates of parameters we're interested in
    weights = map_estimates["weights_auto_loc"]
    phi_weights = map_estimates["phi_weights_auto_loc"]
    mus = map_estimates["mu_auto_loc"]
    sigmas = jnp.stack(
        [map_estimates["sigma1_auto_loc"], map_estimates["sigma2_auto_loc"]], axis=-1
    )

    # compute probabilities over v_domain for each latent group k
    group_probs = jnp.zeros((K, len(v_domain)))
    for k in range(K):
        group_probs = group_probs.at[k].set(
            dist.MultivariateNormal(mus[k], jnp.diag(sigmas[k])).log_prob(v_domain)
        )
    group_probs = jnp.exp(group_probs)
    group_probs /= jnp.sum(group_probs, axis=1, keepdims=True)

    # then, use this to compute probabilities over v_domain for each surface feature phi
    phi_to_k = phi_weights.T[jnp.unique(phis)] * weights
    phi_to_k /= jnp.sum(phi_to_k, axis=-1, keepdims=True)

    phi_probs = jnp.zeros((len(phi_to_k), len(v_domain)))
    for phi in range(len(phi_to_k)):
        weighted = phi_to_k[phi].reshape((-1, 1)) * group_probs
        phi_probs = phi_probs.at[phi].set(jnp.sum(weighted, axis=0))
    phi_probs /= jnp.sum(phi_probs, axis=1, keepdims=True)

    return phi_probs


def full_bayesian(rng_key, agents, phis, K, beta, v_self, v_domain, plot_dir):
    weights = jnp.zeros(len(phis))

    choice_counts = jnp.zeros((len(agents), 4))
    _phis = jnp.zeros(len(agents), dtype=int)
    for m, a in enumerate(agents):
        tmp = jnp.array(a["choices"])
        choice_counts = choice_counts.at[m].set(jnp.bincount(tmp, minlength=4))
        _phis = _phis.at[m].set(a["phi"])

    posterior = infer_conditional_v_distributions(
        rng_key=rng_key,
        data=(choice_counts, _phis, K, beta),
        v_domain=v_domain,
        plot_convergence=True,
        plot_dir=plot_dir,
        init_iter=100,
    )

    visualise_conditional_posterior(rng_key, v_domain, posterior, f"{plot_dir}/posterior.png")

    sims = jnp.array([value_similarity(v_self, v) for v in v_domain])
    for i, phi in enumerate(phis):
        expected_sim = jnp.sum(sims * posterior[i])
        weights = weights.at[i].set(expected_sim)

    return norm_unit_sum(weights)


def visualise_group_parameters(rng_key, mus, sigmas, fpath):
    K, data = len(mus), []
    for k in range(K):
        cov = jnp.diag(sigmas[k])
        points = random.multivariate_normal(rng_key, mus[k], cov, shape=(1000,))
        data.extend([{"v1": float(p[0]), "v2": float(p[1]), "k": k} for p in points])
    data = pd.DataFrame(data)

    fig, ax = plt.subplots()
    g = sns.kdeplot(
        data=data,
        x="v1",
        y="v2",
        hue="k",
        fill=True,
        palette=sns.color_palette("viridis", n_colors=K),
        levels=5,
        legend=False,
        ax=ax,
    )
    g.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
    fig.savefig(fpath)


def visualise_conditional_posterior(rng_key, v_domain, posterior, fpath, num_samples=1000):
    data = []
    num_phis = posterior.shape[0]
    for phi in range(num_phis):
        sample_idxs = random.choice(
            rng_key, len(v_domain), shape=(num_samples,), p=posterior[phi]
        )
        samples = v_domain[sample_idxs]
        data.extend(
            [{"v1": float(v[0]), "v2": float(v[1]), "phi": int(phi)} for v in samples]
        )

    fig, ax = plt.subplots()
    g = sns.kdeplot(
        data=pd.DataFrame(data),
        x="v1",
        y="v2",
        hue="phi",
        fill=True,
        palette=sns.color_palette("viridis", n_colors=num_phis),
        levels=5,
        legend=False,
        ax=ax,
    )
    g.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
    fig.savefig(fpath)
