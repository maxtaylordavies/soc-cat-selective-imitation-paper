from numpyro.infer import MCMC, NUTS, Predictive


def posterior_predictive(rng_key, model, data, num_warmpup=500, num_samples=1000):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmpup, num_samples=num_samples)
    mcmc.run(rng_key, data)
    posterior_samples = mcmc.get_samples()
    predictive = Predictive(model, posterior_samples, num_samples=1, batch_ndims=0)
    return predictive(rng_key, data)
