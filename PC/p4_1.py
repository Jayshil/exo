import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import celerite2
from celerite2 import terms
import os
import pickle
import utils
from kelp.jax import thermal_phase_curve, reflected_phase_curve_inhomogeneous, reflected_phase_curve
import corner
from exotoolbox.utils import get_quantiles
from kelp import Filter
import astropy.units as u
import time
from nautilus import Prior, Sampler
import multiprocessing
multiprocessing.set_start_method('fork')

#>>>>>> ----- Total time taken: 23.6903 min

pout = os.getcwd() + '/PC/Analysis/Nautilus2'

# Planet parameters
per, tc = 2.7240314376, 2459024.6067578471
ar, rprs = 4.6561340944, 0.0716620112

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
def log_like(param):
    global tim, fl, fle, roll
    omega, g, mflx1, mflx2, sig_w1, sig_w2, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, log_s0, rho1, log_q0 =\
        param['omega'], param['g'], param['mflx1'], param['mflx2'], param['sig_w1'], param['sig_w2'], param['a1'], param['a2'], param['a3'], param['a4'], param['a5'],\
        param['b1'], param['b2'], param['b3'], param['b4'], param['b5'], param['log_s0'], param['rho1'], param['log_q0']
    mflx_all, sig_w_all = [mflx1, mflx2], [sig_w1, sig_w2]

    log_likelihood = 0.

    for ins in range(2):
        ins_data = dataset['ins' + str(ins+1)]
        roll1 = ins_data['roll']
        # Building phase-curve model

        # Reflective phase curve (homogeneous) -----------------
        refl_fl_ppm, _, _ = reflected_phase_curve(phases=ins_data['sorted_phs'], omega=omega, g=g, a_rp=ar/rprs)
        refl_pc_sorted_acc_time = refl_fl_ppm[ins_data['idx']]/1e6

        # Physical light curve
        phy_lc = ins_data['transit'] + ((refl_pc_sorted_acc_time + ins_data['thermal'])*ins_data['eclipse'])

        # Roll angle model
        roll_model = (a1 * np.sin(roll1)) + (a2 * np.sin(2*roll1)) + (a3 * np.sin(3*roll1)) + (a4 * np.sin(4*roll1)) + (a5 * np.sin(5*roll1)) +\
            (b1 * np.cos(roll1)) + (b2 * np.cos(2*roll1)) + (b3 * np.cos(3*roll1)) + (b4 * np.cos(4*roll1)) + (b5 * np.cos(5*roll1))
        
        tot_model = (phy_lc / (1 + mflx_all[ins])) + roll_model

        # Residuals
        resid = ins_data['fl'] - tot_model

        # GP model for stellar variability
        kernel = terms.SHOTerm(S0=np.exp(log_s0), rho=rho1, Q=np.exp(log_q0))
        gp = celerite2.GaussianProcess(kernel, mean=0.)
        gp.compute(ins_data['tim'], diag=ins_data['fle']**2 + (sig_w_all[ins]*1e-6)**2, check_sorted=False)
        log_likelihood = log_likelihood + gp.log_likelihood(resid)

    return log_likelihood

# Defining prior cube
def uniform(t, a, b):
    return (b-a)*t + a
def stand(a, loc, scale):
    return (a-loc)/scale

par = ['omega', 'g', 'mflx1', 'mflx2', 'sig_w1', 'sig_w2', 'a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4', 'b5', 'log_s0', 'rho1', 'log_q0']
dist = ['uniform', 'truncatednormal', 'truncatednormal', 'truncatednormal', 'loguniform', 'loguniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'normal', 'uniform']
hyper = [[0., 1.], [0., 0.4, 0., 1.], [0., 0.1, -0.3, 0.3], [0., 0.1, -0.3, 0.3], [0.1, 1e4], [0.1, 1e4], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-25., -14.], [1.2, 0.2], [-4., 12.]]

prior = Prior()
#prior.add_parameter('omega', dist=(0., 1.))
#prior.add_parameter('g', dist=stats.truncnorm(-5, +5))
#prior.add_parameter('c', dist=norm(loc=0, scale=2.0))

for i in range(len(par)):
    if dist[i] == 'uniform':
        prior.add_parameter(par[i], dist=(hyper[i][0], hyper[i][1]))
    elif dist[i] == 'normal':
        prior.add_parameter(par[i], dist=stats.norm(loc=hyper[i][0], scale=hyper[i][1]))
    elif dist[i] == 'truncatednormal':
        prior.add_parameter(par[i], dist=stats.truncnorm(a=stand(hyper[i][2], hyper[i][0], hyper[i][1]), b=stand(hyper[i][3], hyper[i][0], hyper[i][1]), loc=hyper[i][0], scale=hyper[i][1]))
    elif dist[i] == 'loguniform':
        prior.add_parameter(par[i], stats.loguniform(a=hyper[i][0], b=hyper[i][1]))
    else:
        raise Exception('Please use proper distribution!')

# Saving prior file
f11 = open(pout + '/priors.dat', 'w')
for i in range(len(par)):
    f11.write(par[i] + '\t' + dist[i] + '\t' + str(hyper[i]) + '\n')
f11.close()

# Saving the data files
fdata = open(pout + '/lc.dat', 'w')
fdata.write('#Time\tFlux\tFlux_err\tRoll\tU2\n')
for i in range(len(tim)):
    fdata.write(str(tim[i]) + '\t' + str(fl[i]) + '\t' + str(fle[i]) + '\t' + str(roll[i]) + '\n')
fdata.close()

# And the sampling
sampler = Sampler(prior, log_like, n_live=1000, pool=8)#, pass_dict=False, pool=8)
sampler.run(verbose=True)

posterior_samples, log_w, log_l = sampler.posterior(equal_weight=True)

t2 = time.time()

print('>>>>>> ----- Total time taken: {:.4f} min'.format((t2-t1)/60))

f22 = open(pout + '/posteriors.dat', 'w')
post_samps = {}
post_samps['samples'] = {}
for i in range(len(par)):
    post_samps['samples'][par[i]] = posterior_samples[:, i]
    qua = get_quantiles(posterior_samples[:, i])
    f22.write(par[i] + '\t' + str(qua[0]) + '\t' + str(qua[1]-qua[0]) + '\t' + str(qua[0]-qua[2]) + '\n')
f22.close()

# logZ
post_samps['lnZ'] = sampler.evidence()

# Dumping a pickle
pickle.dump(post_samps,open(pout + '/posteriors.pkl','wb'))

#points, log_w, log_l = sampler.posterior()
#ndim = points.shape[1]
#fig, axes = plt.subplots(ndim, ndim, figsize=(3.5, 3.5))
corner.corner(posterior_samples, show_titles=True)#, fig=fig)
plt.savefig(pout + '/corner.pdf')