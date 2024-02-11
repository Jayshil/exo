import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import celerite2
from celerite2 import terms
import os
import pickle
import utils
from kelp.jax import thermal_phase_curve, reflected_phase_curve_inhomogeneous, reflected_phase_curve
import dynesty
from dynesty.utils import resample_equal
from dynesty import plotting as dyplot
from exotoolbox.utils import get_quantiles
from kelp import Filter
import astropy.units as u
import time
from multiprocessing import Pool
import contextlib
import multiprocessing
multiprocessing.set_start_method('fork')

pout = os.getcwd() + '/PC/Analysis/Dyn2'

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
def log_like(x):
    global tim, fl, fle, roll
    omega, g, mflx1, mflx2, sig_w1, sig_w2, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, log_s0, rho1, log_q0 = x
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
#omega, g, mflx1, mflx2, sig_w1, sig_w2, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, log_s0, rho1, log_q0
par = ['omega', 'g', 'mflx1', 'mflx2', 'sig_w1', 'sig_w2', 'a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4', 'b5', 'log_s0', 'rho1', 'log_q0']
dist = ['uniform', 'truncatednormal', 'truncatednormal', 'truncatednormal', 'loguniform', 'loguniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'normal', 'uniform']
hyper = [[0., 1.], [0., 0.4, 0., 1.], [0., 0.1, -0.3, 0.3], [0., 0.1, -0.3, 0.3], [0.1, 1e4], [0.1, 1e4], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-25., -14.], [1.2, 0.2], [-4., 12.]]

def prior_transform(ux):
    x = np.array(ux)
    for i in range(len(par)):
        if dist[i] == 'uniform':
            x[i] = uniform(ux[i], hyper[i][0], hyper[i][1])
        elif dist[i] == 'normal':
            x[i] = stats.norm.ppf(ux[i], loc=hyper[i][0], scale=hyper[i][1])
        elif dist[i] == 'truncatednormal':
            x[i] = stats.truncnorm.ppf(ux[i], a=stand(hyper[i][2], hyper[i][0], hyper[i][1]), b=stand(hyper[i][3], hyper[i][0], hyper[i][1]), loc=hyper[i][0], scale=hyper[i][1])
        elif dist[i] == 'loguniform':
            x[i] = stats.loguniform.ppf(ux[i], a=hyper[i][0], b=hyper[i][1])
        elif dist[i] == 'fixed':
            x[i] = hyper[i]
        else:
            raise Exception('Please use proper distribution!')
    return x

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
nthreads=8
with contextlib.closing(Pool(processes=nthreads-1)) as executor:
    dsampler = dynesty.DynamicNestedSampler(loglikelihood=log_like, prior_transform=prior_transform,\
        ndim=len(par), nlive=500, bound='single', sample='rwalk', pool=executor, queue_size=nthreads)
    dsampler.run_nested()
dres = dsampler.results

t2 = time.time()

print('>>>>>> ----- Total time taken: {:.4f} min'.format((t2-t1)/60))

weights = np.exp(dres['logwt'] - dres['logz'][-1])
posterior_samples = resample_equal(dres.samples, weights)

f22 = open(pout + '/posteriors.dat', 'w')
post_samps = {}
post_samps['samples'] = {}
for i in range(len(par)):
    post_samps['samples'][par[i]] = posterior_samples[:, i]
    qua = get_quantiles(posterior_samples[:, i])
    f22.write(par[i] + '\t' + str(qua[0]) + '\t' + str(qua[1]-qua[0]) + '\t' + str(qua[0]-qua[2]) + '\n')
f22.close()

# logZ
post_samps['lnZ'] = dres.logz
post_samps['lnZ_err'] = dres.logzerr

# Dumping a pickle
pickle.dump(post_samps,open(pout + '/posteriors.pkl','wb'))

fig, axes = dyplot.traceplot(dres, labels=par)
fig.tight_layout()
plt.savefig(pout + '/trace_plot.pdf')

fig, axes = dyplot.cornerplot(dres, show_titles=True, labels=par)
plt.savefig(pout + '/corner.pdf')