import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
jax.config.update(
    "jax_enable_x64", True
)  # For 64-bit precision since JAX defaults to 32-bit
from jax.random import PRNGKey, split
import astropy.constants as con
import celerite2.jax
from celerite2.jax import terms as jax_terms
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro import distributions as dist
from numpyro.infer.autoguide import AutoLaplaceApproximation
import numpyro_ext.optim
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.units import unit_registry as ureg
from jaxoplanet.orbits import keplerian
import arviz as az
import corner
import os
import pickle
import time

cpu_cores = 2
numpyro.set_host_device_count(cpu_cores)

# Some constants that we need
G = con.G.value
rho_sun = (con.M_sun/(con.R_sun**3)).value

instruments = ['synth1', 'synth2']
tim, fl, fle = {}, {}, {}
for i in range(len(instruments)):
    tim1, fl1, fle1 = np.loadtxt(os.getcwd() + '/Data/kelt-11-' + instruments[i] + '.dat', usecols=(0,1,2), unpack=True)
    tim[instruments[i]], fl[instruments[i]], fle[instruments[i]] = jnp.asarray(tim1), jnp.asarray(fl1), jnp.asarray(fle1)

# Set the number of cores on your machine for parallelism:
pout = os.getcwd() + '/Transit/Analysis/Synth_2data_vmap'

"""def evaluate_model(times, density, t0, per, bb, rprs, u1, u2):
    star = keplerian.Central(density=density[0]/rho_sun)
    body = keplerian.Body(time_transit=t0[0], period=per[0], impact_param=bb[0], eccentricity=0., omega_peri=jnp.pi/2, radius=rprs, radial_velocity_semiamplitude=20.*ureg.m/ureg.s)

    orbit = keplerian.System(central=star).add_body(body=body)

    model = limb_dark_light_curve(orbit, [u1, u2])(times) + 1.

    return model"""

def evaluate_model(times, density, t0, per, bb, rprs, u1, u2):

    star = keplerian.Central(density=density/rho_sun)
    body = keplerian.Body(time_transit=t0, period=per, impact_param=bb, eccentricity=0., omega_peri=jnp.pi/2, radius=rprs, radial_velocity_semiamplitude=20.*ureg.m/ureg.s)

    orbit = keplerian.System(central=star).add_body(body=body)

    model = limb_dark_light_curve(orbit, [u1, u2])(times) + 1.

    return model


evaluate_model_vmap = jax.vmap(evaluate_model, in_axes=(0, None, None, None, None, 0, 0, 0), out_axes=0)

"""Y = evaluate_model_vmap(jnp.asarray([ tim[instruments[0]], tim[instruments[1]] ]),\
                        jnp.asarray([104.]),\
                        jnp.asarray([2458553.81381]),\
                        jnp.asarray([4.7360990]),\
                        jnp.asarray([0.404]),\
                        jnp.asarray([0.0475, 0.0575]),\
                        jnp.asarray([0.1, 0.15]),\
                        jnp.asarray([0.3, 0.25]))"""

obs = jnp.array([ fl[instruments[0]], fl[instruments[1]] ])#.reshape(2, len(fl1), 1)

t1 = time.time()
## First defining a numpyro model
def model():
    # Priors on the model parameters
    density = numpyro.sample('density', dist.Normal(loc=101., scale=17.) )
    t0 = numpyro.sample('t0', dist.Normal(loc=2458553.81381, scale=0.00033) )
    per = numpyro.sample('per', dist.Normal(loc=4.7360990, scale=0.00003) )
    bb = numpyro.sample('bb', dist.Uniform(low=0., high=1.) )
    rprs = numpyro.sample('rprs', dist.Uniform(low=jnp.array([0., 0.]), high=jnp.array([1., 1.])) )
    u1 = numpyro.sample('u1', dist.Uniform(low=jnp.array([0., 0.]), high=jnp.array([1., 1.])) )
    u2 = numpyro.sample('u2', dist.Uniform(low=jnp.array([0., 0.]), high=jnp.array([1., 1.])) )
    sig1 = numpyro.sample('sig_w1', dist.LogUniform(low=0.1, high=1e4))
    sig2 = numpyro.sample('sig_w2', dist.LogUniform(low=0.1, high=1e4))
    # Let's also track the value of line for each iteration
    
    # The orbit and light curve
    y_pred = evaluate_model_vmap(jnp.asarray([ tim[instruments[0]], tim[instruments[1]] ]),\
                                 density,\
                                 t0,\
                                 per,\
                                 bb,\
                                 rprs,\
                                 u1,\
                                 u2)
    
    numpyro.deterministic('y', y_pred)
    # And the likelihood function,
    fle_all = jnp.asarray([ jnp.sqrt(fle[instruments[0]]**2 + (sig1*1e-6)**2), jnp.sqrt(fle[instruments[1]]**2 + (sig2*1e-6)**2) ])
    numpyro.sample('obs', dist.Normal(loc=y_pred[:,:,0], scale=fle_all), obs=obs)


all_var_names = ['density', 't0', 'per', 'bb', 'rprs', 'u1', 'u2', 'sig_w1', 'sig_w2']

## -------   And sampling

# Uses adam optimiser and a Laplace approximation calculated from the hessian of the log posterior as a guide
optimizer = numpyro.optim.Adam(step_size=1e-4)
guide = AutoLaplaceApproximation(model, init_loc_fn = numpyro.infer.init_to_median())
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
svi_result = svi.run(PRNGKey(5), 100000)#, Y)
params = svi_result.params
p_fit = guide.median(params)

for k, v in p_fit.items():
    if k in all_var_names:
        print(f"{k}: {v}")


# Random numbers in jax are generated like this:
rng_seed = 10
rng_keys = split(PRNGKey(rng_seed), cpu_cores)

# Define a sampler, using here the No U-Turn Sampler (NUTS)
# with a dense mass matrix:
sampler = NUTS(model, dense_mass=True, regularize_mass_matrix=False, init_strategy=numpyro.infer.init_to_value(values=p_fit))
# init_strategy=numpyro.infer.init_to_median())

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

t2 = time.time()

print('>>>> --- It takes {:.2f} min'.format((t2 - t1) / 60))