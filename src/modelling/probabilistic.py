from collections import Counter, defaultdict

from jax import pure_callback
import jax.numpy as jnp
import numpy as np
from scipy.special import gamma
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import (
    MCMC,
    NUTS,
    Predictive,
    SVI,
    TraceEnum_ELBO,
    autoguide,
    init_to_value,
)
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm


def clipped_gaussian(mus, stds, num=1):
    cov = np.diag(stds**2)
    samples = np.random.default_rng().multivariate_normal(mus, cov, num)
    return np.clip(samples, 0, 1)


def boltzmann1d(x, beta):
    p = jnp.exp(x / beta)
    return p / jnp.sum(p)


def boltzmann2d(x, beta):
    p = jnp.exp(x / beta)
    return p / jnp.sum(p, axis=1).reshape((-1, 1))


def crp(z, c, log=True):
    counts = Counter(z)
    coeff = ((c ** len(counts)) * gamma(c)) / gamma(c + len(z))

    if log:
        return coeff + jnp.sum([gamma(counts[k]) for k in counts])

    return coeff * jnp.prod([gamma(counts[k]) for k in counts])


def dirichlet_multinomial(counts, T, L, g, log=True):
    coeff = (gamma(L * g) * gamma(T + 1)) / gamma(T + (L * g))
    prods = jnp.prod(gamma(counts + g) / (gamma(g) * gamma(counts + 1)), axis=1)

    tmp = coeff * prods
    return jnp.sum(jnp.log(tmp)) if log else jnp.prod(tmp)


def multivariate_gaussian_model(data):
    dim = data.shape[-1]
    mu = numpyro.sample("mu", dist.Normal(0.5 * jnp.ones(dim), jnp.ones(dim)))
    sigma = numpyro.sample(
        "sigma", dist.InverseGamma(0.5 * jnp.ones(dim), 0.5 * jnp.ones(dim))
    )
    cov = jnp.diag(sigma)
    numpyro.sample("obs", dist.MultivariateNormal(mu, cov), obs=data)


def mcmc_posterior_predictive(rng_key, model, data, num_warmpup=500, num_samples=1000):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmpup, num_samples=num_samples)
    mcmc.run(rng_key, data)
    posterior_samples = mcmc.get_samples()
    predictive = Predictive(model, posterior_samples, num_samples=1, batch_ndims=0)
    return predictive(rng_key, data)


# helper function to collect gradient norms during training
def hook_optax(optimizer):
    gradient_norms = defaultdict(list)

    def append_grad(grad):
        for name, g in grad.items():
            gradient_norms[name].append(float(jnp.linalg.norm(g)))
        return grad

    def update_fn(grads, state, params=None):
        grads = pure_callback(append_grad, grads, grads)
        return optimizer.update(grads, state, params=params)

    return optax.GradientTransformation(optimizer.init, update_fn), gradient_norms


# do stochastic variational inference on a model
def run_svi(
    rng_key,
    model,
    data,
    init_vals_func,
    visible_site_names,
    plot_convergence=False,
    plot_dir="results/tmp",
    init_iter=50,
    run_iter=200,
):
    elbo = TraceEnum_ELBO()

    def initialize(seed, return_guide=False):
        init_vals = init_vals_func(seed)
        _model = handlers.block(
            handlers.seed(model, rng_key),
            hide_fn=lambda site: site["name"] not in visible_site_names,
        )
        guide = autoguide.AutoDelta(_model, init_loc_fn=init_to_value(values=init_vals))
        handlers.seed(guide, rng_key)(data)  # warm up the guide
        loss = elbo.loss(rng_key, {}, model, guide, data)
        return (loss, guide) if return_guide else loss

    # choose the best among init_iter random initializations.
    _, seed = min(
        (initialize(s), s) for s in tqdm(range(init_iter), desc="initializing guide")
    )
    loss, guide = initialize(seed, return_guide=True)  # initialize the guide

    # train the model
    optim, gradient_norms = hook_optax(optax.adam(learning_rate=0.1, b1=0.8, b2=0.99))
    svi = SVI(model, guide, optim, loss=elbo)
    svi_result = svi.run(rng_key, run_iter, data)

    # optionally plot convergence and gradient norms
    if plot_convergence:
        plt.figure(figsize=(10, 3), dpi=100).set_facecolor("white")
        plt.plot(svi_result.losses)
        plt.xlabel("iters")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.title("Convergence of SVI")
        plt.savefig(f"{plot_dir}/svi_convergence.png")

        plt.figure(figsize=(10, 4), dpi=100).set_facecolor("white")
        for name, grad_norms in gradient_norms.items():
            plt.plot(grad_norms, label=name)
        plt.xlabel("iters")
        plt.ylabel("gradient norm")
        plt.yscale("log")
        plt.legend(loc="best")
        plt.title("Gradient norms during SVI")
        plt.savefig(f"{plot_dir}/svi_gradient_norms.png")

    return svi_result.params
