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
x1, y1, ye1 = np.loadtxt(os.getcwd() + '/Data/line.dat', usecols=(0,1,2), unpack=True)
x2, y2, ye2 = np.copy(x1), y1 + 1.2, np.copy(ye1)
x3, y3, ye3 = np.copy(x1), y1 + 2.2, np.copy(ye1)
xs, ys, yes = [x1, x2, x3], [y1, y2, y3], [ye1, ye2, ye3]

fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
for i in range(len(xs)):
    axs.errorbar(xs[i], ys[i], yerr=yes[i], fmt='.')
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
    for i in range(len(xs)):
        cs = numpyro.sample('c' + str(i+1), dist.Uniform(low=1., high=10.))
        log_sigws = numpyro.sample('logs' + str(i+1), dist.Uniform(low=-5, high=5))
        # Let's also track the value of line for each iteration
        numpyro.deterministic('y' + str(i+1), m*xs[i] + cs)
        # And the likelihood function,
        numpyro.sample('obs' + str(i+1), dist.Normal(loc=m*xs[i] + cs, scale=jnp.sqrt( jnp.power(yes[i], 2) + jnp.power(jnp.exp(log_sigws), 2) ) ), obs=ys[i])

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
pickle.dump(result, open(os.getcwd() + '/Line/Analysis/res_numpyro_3data.pkl','wb'))

all_var_names = ['m', 'c1', 'c2', 'c3', 'logs1', 'logs2', 'logs3']
# Trace plots
_ = az.plot_trace(result, var_names=all_var_names)
plt.tight_layout()
plt.show()

# Result summary
summary = az.summary(result, var_names=all_var_names)
print(summary)

# Corner plot
truth = dict(zip(all_var_names, np.array([0.27, 2.7, 2.7+1.2, 2.7+2.2, -4, -4, -4]),))
_ = corner.corner(result, var_names=all_var_names, truths=truth,);
plt.show()

# Defining posteriors for each variable
ms = np.array(result.posterior['m']).flatten()

# Plotting the results
fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
for j in range(len(xs)):
    ls1 =  np.array(result.posterior['logs' + str(j+1)]).flatten()
    cs1 = np.array(result.posterior['c' + str(j+1)]).flatten()
    ys1 = np.array(result.posterior['y' + str(j+1)]).reshape(6000, 20)
    axs.errorbar(xs[j], ys[j], yerr=np.sqrt(yes[j]**2 + np.exp(np.median(ls1))**2), fmt='.')#, c='k')
    axs.plot(xs[j], np.median(ms)*xs[j] + np.median(cs1), 'k-', lw=2.)
    for i in range(50):
        axs.plot(xs[j], ys1[np.random.choice(np.arange(6000), replace=False),:], 'k-', alpha=0.1)
axs.set_xlabel('x', fontsize=14)
axs.set_ylabel('y', fontsize=14)
plt.setp(axs.get_xticklabels(), fontsize=12)
plt.setp(axs.get_yticklabels(), fontsize=12)
plt.grid()
plt.show()