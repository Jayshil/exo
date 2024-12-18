import numpy as np
import matplotlib.pyplot as plt
from jax.random import PRNGKey, split
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro import distributions as dist
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits import TransitOrbit
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
pout = os.getcwd() + '/Transit/Analysis/NumPyro_Synth'
cpu_cores = 2
numpyro.set_host_device_count(cpu_cores)

# Visualising the data
tim, fl, fle = np.loadtxt(os.getcwd() + '/Data/kelt-11-synthetic.dat', usecols=(0,1,2), unpack=True)
tim = tim - tim[0]

fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
axs.errorbar(tim, fl, yerr=fle, fmt='.')
axs.set_xlabel('x', fontsize=14)
axs.set_ylabel('y', fontsize=14)
plt.setp(axs.get_xticklabels(), fontsize=12)
plt.setp(axs.get_yticklabels(), fontsize=12)
plt.grid()
plt.show()

per, per_err = 4.7360990, np.sqrt((0.0000290**2) + (0.0000270**2))
tc1, tc1_err = 2458553.81381, 0.00033
cycle = round((tim[0] - tc1)/per)
t00 = tc1 + (cycle * per)
ar, ar_err = 4.98, 0.05
bb, bb_err = 0.404, np.sqrt((0.013**2) + (0.018**2))
rprs, rprs_err = 0.0475, 0.0006

# And modelling it
## First defining a numpyro model
def model():
    # Priors on the model parameters
    tc = numpyro.sample('tc', dist.Uniform(low=0.2, high=0.5))
    dur = numpyro.sample('duration', dist.Uniform(low=0.1, high=0.5))
    bb = numpyro.sample('bb', dist.Uniform(low=0., high=1.))
    rprs = numpyro.sample('rprs', dist.Uniform(low=0., high=1.))
    u1 = numpyro.sample('u1', dist.Uniform(low=0., high=1.))
    u2 = numpyro.sample('u2', dist.Uniform(low=0., high=1.))
    sig_w = numpyro.sample('sig_w', dist.LogUniform(low=0.1, high=1e4))
    # Let's also track the value of line for each iteration
    # The orbit and light curve
    orbit = TransitOrbit(period=per, duration=dur, time_transit=tc, impact_param=bb, radius=rprs)
    y_pred = limb_dark_light_curve(orbit, [u1, u2])(tim) + 1.
    numpyro.deterministic('y', y_pred)
    # And the likelihood function,
    numpyro.sample('obs', dist.Normal(loc=y_pred, scale=jnp.sqrt( jnp.power(fle, 2) + jnp.power(sig_w*1e-6, 2) ) ), obs=fl)

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
pickle.dump(result, open(pout + '/res_numpyro.pkl','wb'))

# Trace plots
_ = az.plot_trace(result, var_names=['tc', 'duration', 'bb', 'rprs', 'u1', 'u2', 'sig_w'])
plt.tight_layout()
#plt.show()
plt.savefig(pout + '/trace.png', dpi=500)

# Result summary
summary = az.summary(result, var_names=['tc', 'duration', 'bb', 'rprs', 'u1', 'u2', 'sig_w'])
print(summary)

# Corner plot
#truth = dict(zip(['tc', 'duration', 'bb', 'rprs', 'u1', 'u2', 'sig_w'], np.array([0.27, 2.7, -4]),))
_ = corner.corner(result, var_names=['tc', 'duration', 'bb', 'rprs', 'u1', 'u2', 'sig_w']);#, truths=truth,);
#plt.show()
plt.savefig(pout + '/corner.png', dpi=500)

# Defining posteriors for each variable
y1 = np.array(result.posterior['y']).reshape(6000, len(tim))

# Plotting the results
fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
axs.errorbar(tim, fl, yerr=fle, fmt='.')
axs.plot(tim, np.nanmedian(y1, axis=0), 'k-', lw=2., zorder=10)
for i in range(50):
    axs.plot(tim, y1[np.random.choice(np.arange(6000), replace=False),:], 'k-', alpha=0.1, zorder=10)
axs.set_xlabel('Time (BJD)', fontsize=14)
axs.set_ylabel('Relative Flux', fontsize=14)
plt.setp(axs.get_xticklabels(), fontsize=12)
plt.setp(axs.get_yticklabels(), fontsize=12)
plt.grid()
#plt.show()
plt.savefig(pout + '/model.png', dpi=500)