import numpy as np
import matplotlib.pyplot as plt
from jax.random import PRNGKey, split
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro import distributions as dist
import arviz as az
import corner
import os
import pickle

# This code is heavily based on a similar notebook in `exoplanet` 
# docs: https://docs.exoplanet.codes/en/latest/tutorials/intro-to-pymc3/.
# And kelp docs: https://kelp.readthedocs.io/en/latest/kelp/optimization.html#jax-numpyro


# Sadly, I could not install jupyter notebook for native environment (which contains my jax installation)
# So, we have to do with a python code!

# Set the number of cores on your machine for parallelism:
cpu_cores = 2
numpyro.set_host_device_count(cpu_cores)

# Visualising the data
x, y, ye = np.loadtxt(os.getcwd() + '/Data/line.dat', usecols=(0,1,2), unpack=True)

fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
axs.errorbar(x, y, yerr=ye, fmt='.')
axs.set_xlabel('x', fontsize=14)
axs.set_ylabel('y', fontsize=14)
plt.setp(axs.get_xticklabels(), fontsize=12)
plt.setp(axs.get_yticklabels(), fontsize=12)
plt.grid()
plt.show()

# And modelling it
## First defining a numpyro model
def model():
    # Priors on the model parameters
    m = numpyro.sample('m', dist.Uniform(low=0., high=1.))
    c = numpyro.sample('c', dist.Uniform(low=1., high=5.))
    log_sigw = numpyro.sample('logs', dist.Uniform(low=-5, high=5))
    # Let's also track the value of line for each iteration
    numpyro.deterministic('y', m*x + c)
    # And the likelihood function,
    numpyro.sample('obs', dist.Normal(loc=m*x + c, scale=jnp.sqrt( jnp.power(ye, 2) + jnp.power(jnp.exp(log_sigw), 2) ) ), obs=y)

## -------   And sampling
# Random numbers in jax are generated like this:
rng_seed = 42
rng_keys = split(PRNGKey(rng_seed), cpu_cores)

# Define a sampler, using here the No U-Turn Sampler (NUTS)
# with a dense mass matrix:
sampler = NUTS(model, dense_mass=True)

# Monte Carlo sampling for a number of steps and parallel chains:
mcmc = MCMC(sampler, num_warmup=3_000, num_samples=3_000, num_chains=cpu_cores)

# Run the MCMC
mcmc.run(rng_keys)
# -------- Sampling Done!!


# Using arviz to extract results
# arviz converts a numpyro MCMC object to an `InferenceData` object based on xarray:
result = az.from_numpyro(mcmc)
pickle.dump(result, open(os.getcwd() + '/Line/Analysis/res_numpyro.pkl','wb'))

# Trace plots
_ = az.plot_trace(result, var_names=['m', 'c', 'logs'])
plt.tight_layout()
plt.show()

# Result summary
summary = az.summary(result, var_names=['m', 'c', 'logs'])
print(summary)

# Corner plot
truth = dict(zip(['m', 'c', 'logs'], np.array([0.27, 2.7, -4]),))
_ = corner.corner(result, var_names=['m', 'c', 'logs'], truths=truth,);
plt.show()

# Defining posteriors for each variable
m1 = np.array(result.posterior['m']).flatten()
c1 = np.array(result.posterior['c']).flatten()
ls1 = np.array(result.posterior['logs']).flatten()
y1 = np.array(result.posterior['y']).reshape(6000, 20)

# Plotting the results
fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
axs.errorbar(x, y, yerr=np.sqrt(ye**2 + np.exp(np.median(ls1))**2), fmt='.')
axs.plot(x, np.median(m1)*x + np.median(c1), 'k-', lw=2.)
for i in range(50):
    axs.plot(x, y1[np.random.choice(np.arange(6000), replace=False),:], 'k-', alpha=0.1)
axs.set_xlabel('x', fontsize=14)
axs.set_ylabel('y', fontsize=14)
plt.setp(axs.get_xticklabels(), fontsize=12)
plt.setp(axs.get_yticklabels(), fontsize=12)
plt.grid()
plt.show()