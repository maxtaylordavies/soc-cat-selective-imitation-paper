from jax import random
import jax.numpy as jnp
import numpy as np
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

max_num_clusters = 12


def stickbreak(betas):
    batch_ndims = len(betas.shape) - 1
    cumprod = jnp.exp(jnp.log1p(-betas).cumsum(-1))
    one_betas = jnp.pad(betas, [[0, 0]] * batch_ndims + [[0, 1]], constant_values=1)
    c_one = jnp.pad(cumprod, [[0, 0]] * batch_ndims + [[1, 0]], constant_values=1)
    return one_betas * c_one


@config_enumerate
def gmm(data):
    choices, phis, K, beta = data
    choices_per_agent = jnp.sum(choices[0])

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
        log_likelihood = dist.Multinomial(probs=p, total_count=choices_per_agent).log_prob(
            choices
        )

        numpyro.factor(
            "obs",
            log_likelihood,
        )


@config_enumerate
def dpmm(data):
    """
    Dirichlet process mixture model via the stick-breaking construction
    """
    choices, phis, noise = data
    choices_per_agent = jnp.sum(choices[0])

    alpha = numpyro.sample("alpha", dist.Gamma(1, 1))
    with numpyro.plate("beta_plate", max_num_clusters - 1):
        # sample beta parameters
        beta = numpyro.sample("beta", dist.Beta(1, alpha))

    with numpyro.plate("theta_plate", max_num_clusters):
        # sample mixture component means from a normal distribution
        mu = numpyro.sample(
            "mu", dist.MultivariateNormal(jnp.zeros(2) + 0.5, 1.0 * jnp.eye(2))
        )

        # sample covariance matrices. each component's covariance matrix is
        # diagonal, where the diagonal entries are distributed according to
        # an inverse gamma distribution.
        sigma1 = numpyro.sample("sigma1", dist.InverseGamma(0.5))
        sigma2 = numpyro.sample("sigma2", dist.InverseGamma(0.5))
        sigmas = jnp.stack([sigma1, sigma2], axis=1)
        covs = jnp.stack([jnp.diag(s) for s in sigmas], axis=0)

        # each agent also expresses a discrete scalar 'surface feature', that may help identify their
        # latent group membership. we assume that this feature is distributed according to a categorical
        # distribution with K categories - each group has a different set of weights that parameterises the
        # categorical distribution.
        phi_weights = numpyro.sample(
            "phi_weights", dist.Dirichlet(jnp.ones(max_num_clusters) / max_num_clusters)
        )

    with numpyro.plate("agents", len(choices)):
        # compute mixture weights from beta parameters
        mix_weights = stickbreak(beta)

        # sample group assignments
        z = numpyro.sample("z", dist.Categorical(mix_weights))

        # sample a surface feature for each agent according to the sampled group assignments
        numpyro.sample("phi", dist.Categorical(phi_weights[z]), obs=phis)

        # sample a value function for each agent according to the sampled group assignments,
        # and convert each value function into a probability distribution over the four items
        v = numpyro.sample("v", dist.MultivariateNormal(mu[z], covs[z]))
        v_ = jnp.stack(item_values(v[..., 0], v[..., 1], as_dict=False), axis=-1)
        p = jnp.exp(v_ / noise)
        p /= jnp.sum(p, axis=-1, keepdims=True)

        # choices contains the empirical choice proportions for each agent
        # we can compute their log likelihood under a multinomial distribution
        choices_log_prob = dist.Multinomial(probs=p, total_count=choices_per_agent).log_prob(
            choices
        )

        numpyro.factor(
            "obs",
            choices_log_prob,
        )


def infer_conditional_v_distributions(
    rng_key,
    data,
    v_domain,
    intermediate_plots=False,
    plot_dir="results/tmp",
    init_iter=200,
    run_iter=1000,
):
    _, phis, _ = data

    # helper func to generate random parameter initialisations
    init_vals_func = lambda seed: {
        "mu": random.multivariate_normal(
            random.PRNGKey(seed),
            jnp.zeros(2) + 0.5,
            1.0 * jnp.eye(2),
            shape=(max_num_clusters,),
        ),
        "sigma1": 0.5 * jnp.ones(max_num_clusters),
        "sigma2": 0.5 * jnp.ones(max_num_clusters),
        "phi_weights": jnp.ones((max_num_clusters, max_num_clusters)) / max_num_clusters,
    }

    # run SVI
    map_estimates = run_svi(
        rng_key,
        dpmm,
        data,
        init_vals_func,
        [
            "mu",
            "sigma1",
            "sigma2",
            "beta_plate",
            "theta_plate",
            "alpha",
            "beta",
            "phi_weights",
        ],
        plot_convergence=intermediate_plots,
        plot_dir=plot_dir,
        init_iter=init_iter,
        run_iter=run_iter,
    )

    # extract MAP estimates of parameters we're interested in
    betas = map_estimates["beta_auto_loc"]
    phi_weights = map_estimates["phi_weights_auto_loc"]
    mus = map_estimates["mu_auto_loc"]
    sigmas = jnp.stack(
        [map_estimates["sigma1_auto_loc"], map_estimates["sigma2_auto_loc"]], axis=-1
    )

    # compute mixture weights from beta parameters
    mixture_weights = stickbreak(betas)

    if intermediate_plots:
        fig, ax = plt.subplots()
        sns.barplot(x=np.arange(max_num_clusters), y=np.array(mixture_weights), ax=ax)
        fig.savefig(f"{plot_dir}/mixture_weights.png")

        visualise_group_parameters(
            rng_key, mus, sigmas, f"{plot_dir}/inferred_parameters.png", num_samples=200
        )

    # compute probabilities over v_domain for each latent group k
    group_probs = jnp.zeros((max_num_clusters, len(v_domain)))
    for k in range(max_num_clusters):
        group_probs = group_probs.at[k].set(
            dist.MultivariateNormal(mus[k], jnp.diag(sigmas[k])).log_prob(v_domain)
        )
    group_probs = jnp.exp(group_probs)
    group_probs /= jnp.sum(group_probs, axis=1, keepdims=True)

    # then, use this to compute probabilities over v_domain for each surface feature phi
    phi_to_k = phi_weights.T[jnp.unique(phis)] * mixture_weights
    phi_to_k /= jnp.sum(phi_to_k, axis=-1, keepdims=True)

    print("phi_to_k:")
    print(phi_to_k)

    phi_probs = jnp.zeros((len(phi_to_k), len(v_domain)))
    for phi in range(len(phi_to_k)):
        weighted = phi_to_k[phi].reshape((-1, 1)) * group_probs
        phi_probs = phi_probs.at[phi].set(jnp.sum(weighted, axis=0))
    phi_probs /= jnp.sum(phi_probs, axis=1, keepdims=True)

    return phi_probs


def full_bayesian(rng_key, agents, beta, v_self, v_domain, plot_dir):
    phis = jnp.unique(jnp.array([a["phi"] for a in agents]))
    weights = jnp.zeros(len(phis))

    choice_counts = jnp.zeros((len(agents), 4))
    _phis = jnp.zeros(len(agents), dtype=int)
    for m, a in enumerate(agents):
        tmp = jnp.array(a["choices"])
        choice_counts = choice_counts.at[m].set(jnp.bincount(tmp, minlength=4))
        _phis = _phis.at[m].set(a["phi"])

    posterior = infer_conditional_v_distributions(
        rng_key=rng_key,
        data=(choice_counts, _phis, beta),
        v_domain=v_domain,
        intermediate_plots=False,
        plot_dir=plot_dir,
    )

    visualise_conditional_posterior(rng_key, v_domain, posterior, f"{plot_dir}/posterior.png")

    sims = jnp.array([value_similarity(v_self, v) for v in v_domain])
    for i, phi in enumerate(phis):
        expected_sim = jnp.sum(sims * posterior[i])
        weights = weights.at[i].set(expected_sim)

    return norm_unit_sum(weights)


def visualise_group_parameters(rng_key, mus, sigmas, fpath, num_samples=1000):
    K, data = len(mus), []
    for k in range(K):
        cov = jnp.diag(sigmas[k])
        points = random.multivariate_normal(rng_key, mus[k], cov, shape=(num_samples,))
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
        levels=4,
        legend=False,
        ax=ax,
    )
    g.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
    fig.savefig(fpath)


def visualise_conditional_posterior(rng_key, v_domain, posterior, fpath, num_samples=250):
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
        levels=4,
        legend=False,
        ax=ax,
    )
    g.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
    fig.savefig(fpath)
