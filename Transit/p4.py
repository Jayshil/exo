import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update(
    "jax_enable_x64", True
)  # For 64-bit precision since JAX defaults to 32-bit
from jax.random import PRNGKey, split
import celerite2.jax
from celerite2.jax import terms as jax_terms
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro import distributions as dist
from numpyro.infer.autoguide import AutoLaplaceApproximation
import numpyro_ext.optim
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
pout = os.getcwd() + '/Transit/Analysis/NumPyro_Real'
cpu_cores = 2
numpyro.set_host_device_count(cpu_cores)

# Visualising the data
tim, fl, fle, roll = np.loadtxt(os.getcwd() + '/Data/kelt-11-cheops.dat', usecols=(0,1,2,3), unpack=True)
roll = np.radians(roll)
#tim = tim - tim[0]

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
    tc = numpyro.sample('tc', dist.Uniform(low=0.4, high=0.6))
    dur = numpyro.sample('duration', dist.Uniform(low=0.1, high=0.5))
    bb = numpyro.sample('bb', dist.Uniform(low=0., high=1.))
    rprs = numpyro.sample('rprs', dist.Uniform(low=0., high=1.))
    u1 = numpyro.sample('u1', dist.Uniform(low=0., high=1.))
    u2 = numpyro.sample('u2', dist.Uniform(low=0., high=1.))

    ## Instrumental
    mflx = numpyro.sample('mflx', dist.Normal(loc=0., scale=0.1))
    sig_w = numpyro.sample('sig_w', dist.LogUniform(low=0.1, high=1e4))

    ## Linear
    a1 = numpyro.sample('a1', dist.Uniform(low=-1., high=1.))
    a2 = numpyro.sample('a2', dist.Uniform(low=-1., high=1.))
    a3 = numpyro.sample('a3', dist.Uniform(low=-1., high=1.))
    b1 = numpyro.sample('b1', dist.Uniform(low=-1., high=1.))
    b2 = numpyro.sample('b2', dist.Uniform(low=-1., high=1.))
    b3 = numpyro.sample('b3', dist.Uniform(low=-1., high=1.))

    # GP parameters
    log_w0 = numpyro.sample('logW0', dist.Uniform(low=-2.3, high=8.))
    log_s0 = numpyro.sample('logS0', dist.Uniform(low=-30., high=0.))

    # Transit light curve
    orbit = TransitOrbit(period=per, duration=dur, time_transit=tc, impact_param=bb, radius=rprs)
    transit_model = limb_dark_light_curve(orbit, [u1, u2])(tim) + 1.

    # Roll angle model
    roll_model = (a1 * jnp.sin(roll)) + (a2 * jnp.sin(2*roll)) + (a3 * jnp.sin(3*roll)) +\
        (b1 * jnp.cos(roll)) + (b2 * jnp.cos(2*roll)) + (b3 * jnp.cos(3*roll))
    
    # Total deterministic model
    tot_model = (transit_model / (1 + mflx)) + roll_model

    # Residuals
    resid = fl - tot_model

    # GP model for stellar variability
    kernel = jax_terms.SHOTerm(S0=jnp.exp(log_s0), w0=jnp.exp(log_w0), Q=1/np.sqrt(2))
    gp = celerite2.jax.GaussianProcess(kernel, mean=0.)
    gp.compute(tim, diag=fle**2 + (sig_w*1e-6)**2, check_sorted=False)

    # Let's also track the value of line for each iteration
    
    numpyro.deterministic('transit', transit_model)
    numpyro.deterministic('total', tot_model)
    numpyro.deterministic('linear', roll_model)
    numpyro.deterministic('gp', gp.predict(resid))
    # And the likelihood function,
    numpyro.sample("obs", gp.numpyro_dist(), obs=resid)


all_var_names = ['tc', 'duration', 'bb', 'rprs', 'u1', 'u2', 'mflx', 'sig_w',\
                 'a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'logW0', 'logS0']

"""run_optim = numpyro_ext.optim.optimize(model, init_strategy=numpyro.infer.init_to_median())
opt_params = run_optim(jax.random.PRNGKey(42))#, tim, fle, fl)

for k, v in opt_params.items():
    if k in all_var_names:
        print(f"{k}: {v}")"""

"""optimizer = numpyro.optim.Minimize()
guide = AutoLaplaceApproximation(model)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
init_state = svi.init(PRNGKey(42))#, tim, fle, fl)
optimal_state, loss = svi.update(init_state)#, tim, fle, fl)
params = svi.get_params(optimal_state)  # get guide's parameters
quantiles = guide.quantiles(params, 0.5)
print(quantiles)"""

## -------   And sampling
# Random numbers in jax are generated like this:
rng_seed = 42
rng_keys = split(PRNGKey(rng_seed), cpu_cores)

# Define a sampler, using here the No U-Turn Sampler (NUTS)
# with a dense mass matrix:
sampler = NUTS(model, dense_mass=True,\
               regularize_mass_matrix=False,\
               init_strategy=numpyro.infer.init_to_median())

# Monte Carlo sampling for a number of steps and parallel chains:
mcmc = MCMC(sampler, num_warmup=3_000, num_samples=3_000, num_chains=cpu_cores)

# Run the MCMC
mcmc.run(rng_keys)#, tim, fle, fl)
# -------- Sampling Done!!


# Using arviz to extract results
# arviz converts a numpyro MCMC object to an `InferenceData` object based on xarray:
result = az.from_numpyro(mcmc)
pickle.dump(result, open(pout + '/res_numpyro.pkl','wb'))

# Trace plots
_ = az.plot_trace(result, var_names=all_var_names)
plt.tight_layout()
#plt.show()
plt.savefig(pout + '/trace.png', dpi=500)

# Result summary
summary = az.summary(result, var_names=all_var_names)
print(summary)

# Corner plot
#truth = dict(zip(['tc', 'duration', 'bb', 'rprs', 'u1', 'u2', 'sig_w'], np.array([0.27, 2.7, -4]),))
_ = corner.corner(result, var_names=all_var_names);#, truths=truth,);
#plt.show()
plt.savefig(pout + '/corner.png', dpi=500)

# Defining posteriors for each variable
total_model = np.array(result.posterior['total']).reshape(6000, len(tim))
gp_model = np.array(result.posterior['gp']).reshape(6000, len(tim))

# Plotting the results
fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
axs.errorbar(tim, fl, yerr=fle, fmt='.')
axs.plot(tim, np.nanmedian(total_model, axis=0) + np.nanmedian(gp_model, axis=0), 'k-', lw=2., zorder=10)
for i in range(50):
    axs.plot(tim, total_model[np.random.choice(np.arange(6000), replace=False),:]+gp_model[np.random.choice(np.arange(6000), replace=False),:], 'r-', alpha=0.1, zorder=10)
axs.set_xlabel('Time (BJD)', fontsize=14)
axs.set_ylabel('Relative Flux', fontsize=14)
plt.setp(axs.get_xticklabels(), fontsize=12)
plt.setp(axs.get_yticklabels(), fontsize=12)
plt.grid()
#plt.show()
plt.savefig(pout + '/model.png', dpi=500)