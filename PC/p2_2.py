import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update(
    "jax_enable_x64", True
)  # For 64-bit precision since JAX defaults to 32-bit
from jax.random import PRNGKey, split
import jax.numpy as jnp
import celerite2.jax
from celerite2.jax import terms as jax_terms
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro import distributions as dist
import arviz as az
import corner
import os
import pickle
import utils
from kelp.jax import thermal_phase_curve, reflected_phase_curve_inhomogeneous, reflected_phase_curve
from kelp import Filter
import astropy.units as u
import time

# This code is heavily based on a similar notebook in `exoplanet` 
# docs: https://docs.exoplanet.codes/en/latest/tutorials/intro-to-pymc3/.
# And kelp docs: https://kelp.readthedocs.io/en/latest/kelp/optimization.html#jax-numpyro


# Sadly, I could not install jupyter notebook for native environment (which contains my jax installation)
# So, we have to do with a python code!

# Planet parameters
per, tc = 2.7240314376, 2459024.6067578471
ar, rprs = 4.6561340944, 0.0716620112

# Set the number of cores on your machine for parallelism:
cpu_cores, nos_burn, nos_samp = 2, 10000, 10000
numpyro.set_host_device_count(cpu_cores)

files = ['WASP-189_CHEOPS.dat', 'WASP-189_CHEOPS2.dat']
dataset = {}

for i in range(len(files)):
    # Visualising the data
    tim, fl, fle, roll = np.loadtxt(os.getcwd() + '/Data/' + files[i], usecols=(0,1,2,3), unpack=True)
    roll = np.radians(roll)

    ## Converting times to phases:
    phases_unsorted = ((tim - tc) % per) / per             ## Un-sorted phases
    idx_phase_sort = np.argsort(phases_unsorted)          ## This would sort any array acc to phase
    phases_sorted = phases_unsorted[idx_phase_sort]       ## Sorted phase array
    times_sorted_acc_phs = tim[idx_phase_sort]            ## Time array sorted acc to phase
    idx_that_sort_arr_acc_times = np.argsort(times_sorted_acc_phs)   ## This array would sort array acc to time

    # Fixed models
    transit_model, eclipse_model = utils.transit_model(tim)

    # Fixed thermal phase curve ----------------------------------
    # Filter used
    filt = Filter.from_name('CHEOPS')
    filt_wavelength, filt_trans = filt.wavelength.to(u.m).value, filt.transmittance
    ## Phases
    xi = 2 * np.pi * (phases_sorted - 0.5)
    ## Parameters for creating meshgrid
    phi = np.linspace(-2 * np.pi, 2 * np.pi, 75)
    theta = np.linspace(0, np.pi, 5)
    theta2d, phi2d = np.meshgrid(theta, phi)

    ## Computing f from eps (Cowan & Agol 2011)
    thermal_pc, _ = thermal_phase_curve(xi=xi, hotspot_offset=np.radians(0.),\
                                        omega_drag=4.5, alpha=0.6, C_11=0.15, T_s=8000.,\
                                        a_rs=ar, rp_a=rprs/ar, A_B=0., theta2d=theta2d, phi2d=phi2d,\
                                        filt_wavelength=filt_wavelength, filt_transmittance=filt_trans,\
                                        f=2**-0.5)
    thm_pc_sorted_acc_time = thermal_pc[idx_that_sort_arr_acc_times]#/1e6

    # Saving data
    dataset['ins' +  str(i+1)] = {}
    dataset['ins' +  str(i+1)]['tim'], dataset['ins' +  str(i+1)]['fl'], dataset['ins' +  str(i+1)]['fle'], dataset['ins' +  str(i+1)]['roll'] = tim, fl, fle, roll
    dataset['ins' +  str(i+1)]['transit'], dataset['ins' +  str(i+1)]['eclipse'], dataset['ins' +  str(i+1)]['thermal'] = transit_model, eclipse_model, thm_pc_sorted_acc_time
    dataset['ins' +  str(i+1)]['sorted_phs'], dataset['ins' +  str(i+1)]['idx'] = phases_sorted, idx_that_sort_arr_acc_times

t1 = time.time()
# And modelling it
## First defining a numpyro model
def model():
    # Priors on the model parameters
    ## Planetary parameters
    omega = numpyro.sample('omega', dist.Uniform(low=0, high=1))
    g = numpyro.sample('g', dist.TwoSidedTruncatedDistribution(dist.Normal(loc=0, scale=0.4), low=0, high=1))
    ## Linear model
    a1 = numpyro.sample('a1', dist.Uniform(low=-1., high=1.))
    a2 = numpyro.sample('a2', dist.Uniform(low=-1., high=1.))
    a3 = numpyro.sample('a3', dist.Uniform(low=-1., high=1.))
    a4 = numpyro.sample('a4', dist.Uniform(low=-1., high=1.))
    a5 = numpyro.sample('a5', dist.Uniform(low=-1., high=1.))
    b1 = numpyro.sample('b1', dist.Uniform(low=-1., high=1.))
    b2 = numpyro.sample('b2', dist.Uniform(low=-1., high=1.))
    b3 = numpyro.sample('b3', dist.Uniform(low=-1., high=1.))
    b4 = numpyro.sample('b4', dist.Uniform(low=-1., high=1.))
    b5 = numpyro.sample('b5', dist.Uniform(low=-1., high=1.))
    ## GP model
    log_s0 = numpyro.sample('log_s0', dist.Uniform(-25., -14.))
    rho1 = numpyro.sample('rho1', dist.Normal(1.2, 0.2))
    log_q0 = numpyro.sample('log_q0', dist.Uniform(-4, 12))

    for ins in range(2):
        ## Instrumental parameters
        sig_w = numpyro.sample('sig_w' + str(ins+1), dist.LogUniform(low=0.1, high=1e4))
        mflx = numpyro.sample('mflux' + str(ins+1), dist.TwoSidedTruncatedDistribution(dist.Normal(loc=0, scale=0.1), low=-0.3, high=0.3))

        ins_data = dataset['ins' + str(ins+1)]
        roll1 = ins_data['roll']
        # Building phase-curve model

        # Reflective phase curve (homogeneous) -----------------
        refl_fl_ppm, Ag, QQ = reflected_phase_curve(phases=ins_data['sorted_phs'], omega=omega, g=g, a_rp=ar/rprs)
        refl_pc_sorted_acc_time = refl_fl_ppm[ins_data['idx']]/1e6

        # Physical light curve
        phy_lc = ins_data['transit'] + ((refl_pc_sorted_acc_time + ins_data['thermal'])*ins_data['eclipse'])

        # Roll angle model
        roll_model = (a1 * jnp.sin(roll1)) + (a2 * jnp.sin(2*roll1)) + (a3 * jnp.sin(3*roll1)) + (a4 * jnp.sin(4*roll1)) + (a5 * jnp.sin(5*roll1)) +\
            (b1 * jnp.cos(roll1)) + (b2 * jnp.cos(2*roll1)) + (b3 * jnp.cos(3*roll1)) + (b4 * jnp.cos(4*roll1)) + (b5 * jnp.cos(5*roll1))
        
        tot_model = (phy_lc / (1 + mflx)) + roll_model

        # Residuals
        resid = ins_data['fl'] - tot_model

        # GP model for stellar variability
        kernel = jax_terms.SHOTerm(S0=jnp.exp(log_s0), rho=rho1, Q=jnp.exp(log_q0))
        gp = celerite2.jax.GaussianProcess(kernel, mean=0.)
        gp.compute(ins_data['tim'], diag=ins_data['fle']**2 + (sig_w*1e-6)**2, check_sorted=False)

        # Let's also track the value of line for each iteration
        numpyro.deterministic('phy' + str(ins+1), phy_lc)
        numpyro.deterministic('tot' + str(ins+1), tot_model)
        numpyro.deterministic('lin' + str(ins+1), roll_model)
        numpyro.deterministic('gp' + str(ins+1), gp.predict(resid))
        # And the likelihood function,
        #numpyro.sample('obs', dist.Normal(loc=tot_model, scale=jnp.sqrt(fle**2 + (sig1**2))), obs=fl)
        numpyro.sample('obs' + str(ins+1), gp.numpyro_dist(), obs=resid)
    numpyro.deterministic('Ag', Ag)
    numpyro.deterministic('q', QQ)

## -------   And sampling
# Random numbers in jax are generated like this:
rng_seed = 42
rng_keys = split(PRNGKey(rng_seed), cpu_cores)

# Define a sampler, using here the No U-Turn Sampler (NUTS)
# with a dense mass matrix:
sampler = NUTS(model, dense_mass=True)

# Monte Carlo sampling for a number of steps and parallel chains:
mcmc = MCMC(sampler, num_warmup=nos_burn, num_samples=nos_samp, num_chains=cpu_cores)

# Run the MCMC
mcmc.run(rng_keys)
# -------- Sampling Done!!
t2 = time.time()

print('>>>>>> ----- Total time taken: {:.4f} min'.format((t2-t1)/60))


# Using arviz to extract results
# arviz converts a numpyro MCMC object to an `InferenceData` object based on xarray:
result = az.from_numpyro(mcmc)
pickle.dump(result, open(os.getcwd() + '/PC/Analysis/NumPyro2/res_numpyro.pkl','wb'))

all_var_names = ['Ag', 'q', 'omega', 'g', 'sig_w1', 'sig_w2', 'mflux1', 'mflux2', 'a1', 'a2', 'a3', 'a4', 'a5',\
                 'b1', 'b2', 'b3', 'b4', 'b5', 'log_s0', 'rho1', 'log_q0']

# Trace plots
_ = az.plot_trace(result, var_names=all_var_names)
plt.tight_layout()
plt.savefig(os.getcwd() + '/PC/Analysis/NumPyro2/trace.png', dpi=500)
plt.close()
#plt.show()

# Result summary
summary = az.summary(result, var_names=all_var_names)
print(summary)

# Corner plot
_ = corner.corner(result, var_names=all_var_names);#, truths=truth,);
plt.savefig(os.getcwd() + '/PC/Analysis/NumPyro2/corner.png', dpi=500)
plt.close()
#plt.show()

# Defining posteriors for each variable
## Extra uncertainties and mflux
for ins in range(2):
    tim = dataset['ins' + str(ins+1)]['tim']
    sig1 = np.array(result.posterior['sig_w' + str(ins+1)]).flatten()
    mfl1 = np.array(result.posterior['mflux' + str(ins+1)]).flatten()
    ## GP model, Linear model and Total model
    gp_model = np.array(result.posterior['gp' + str(ins+1)]).reshape(int(cpu_cores * nos_samp), len(tim))
    lin_model = np.array(result.posterior['lin' + str(ins+1)]).reshape(int(cpu_cores * nos_samp), len(tim))
    tot_model = np.array(result.posterior['tot' + str(ins+1)]).reshape(int(cpu_cores * nos_samp), len(tim))
    phys_model = np.array(result.posterior['phy' + str(ins+1)]).reshape(int(cpu_cores * nos_samp), len(tim))

    # Detrending the data
    fl_det = (dataset['ins' + str(ins+1)]['fl'] - np.nanmedian(gp_model, axis=0) - np.nanmedian(lin_model, axis=0)) * (1 + np.nanmedian(mfl1))

    # Plotting the results
    fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
    axs.errorbar(dataset['ins' + str(ins+1)]['tim'], fl_det, yerr=np.sqrt(dataset['ins' + str(ins+1)]['fle']**2 + (np.median(sig1) * 1e-6)**2), fmt='.')
    axs.plot(dataset['ins' + str(ins+1)]['tim'], np.nanmedian(phys_model, axis=0), 'k-', lw=2., zorder=10)
    for i in range(50):
        axs.plot(dataset['ins' + str(ins+1)]['tim'], phys_model[np.random.randint(0,int(cpu_cores * nos_samp)),:], 'b-', alpha=0.1)
    axs.set_xlabel('Time (BJD)', fontsize=14)
    axs.set_ylabel('Normalised Flux', fontsize=14)
    plt.setp(axs.get_xticklabels(), fontsize=12)
    plt.setp(axs.get_yticklabels(), fontsize=12)
    plt.grid()
    plt.savefig(os.getcwd() + '/PC/Analysis/NumPyro2/model_' + str(ins+1) + '.png', dpi=500)
    #plt.show()