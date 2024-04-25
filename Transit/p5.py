import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
import juliet
import os
import multiprocessing
multiprocessing.set_start_method('fork')

# Analysing KELT-11 CHEOPS data with juliet and with the same priors

pout = os.getcwd() + '/Transit/Analysis/Juliet_Real'
instrument = 'KELT11'

tim, fl, fle = {}, {}, {}
tim[instrument], fl[instrument], fle[instrument], roll = np.loadtxt(os.getcwd() + '/Data/kelt-11-cheops.dat', usecols=(0,1,2,3), unpack=True)
roll = np.radians(roll)

lins = np.vstack([np.sin(roll), np.cos(roll), np.sin(2*roll), np.cos(2*roll), np.sin(3*roll), np.cos(3*roll)])
lin_pars = {}
lin_pars[instrument] = np.transpose(lins)

# Priors
par_P = ['P_p1', 't0_p1', 'p_p1_' + instrument, 'b_p1', 'ecc_p1', 'omega_p1', 'q1_' + instrument, 'q2_' + instrument, 'a_p1']
dist_P = ['fixed', 'uniform', 'uniform', 'uniform', 'fixed', 'fixed', 'uniform', 'uniform', 'uniform']
hyper_P = [4.7360990, [0.4, 0.6], [0., 1.], [0., 1.], 0., 90., [0., 1.], [0., 1.], [4.5, 5.5]]

par_ins = ['mdilution_' + instrument, 'mflux_' + instrument, 'sigma_w_' + instrument]
dist_ins = ['fixed', 'normal', 'loguniform']
hyper_ins = [1.0, [0., 0.1], [0.1, 1e4]]

par_lin, dist_lin, hyper_lin = [], [], []
for i in range(lins.shape[0]):
    par_lin.append('theta' + str(i) + '_' + instrument)
    dist_lin.append('uniform')
    hyper_lin.append([-1., 1.])

par_gp = ['GP_S0_' + instrument, 'GP_omega0_' + instrument, 'GP_Q_' + instrument]
dist_gp = ['uniform', 'uniform', 'fixed']
hyper_gp = [[np.exp(-30.), np.exp(0.)], [np.exp(-2.3), np.exp(8)], 1/np.sqrt(2)]

par_tot = par_P + par_ins + par_lin + par_gp
dist_tot = dist_P + dist_ins + dist_lin + dist_gp
hyper_tot = hyper_P + hyper_ins + hyper_lin + hyper_gp

priors = juliet.utils.generate_priors(params=par_tot, dists=dist_tot, hyperps=hyper_tot)

dataset = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, GP_regressors_lc=tim, linear_regressors_lc=lin_pars, out_folder=pout)
results = dataset.fit(sampler='dynesty', nthreads=8)

# juliet best fit model
model = results.lc.evaluate(instrument)

# Making a plot
fig = plt.figure(figsize=(16,9))
gs = gd.GridSpec(2,1, height_ratios=[2,1])

# Top panel
ax1 = plt.subplot(gs[0])
ax1.errorbar(tim[instrument], fl[instrument], yerr=fle[instrument], fmt='.', alpha=0.3)
ax1.plot(tim[instrument], model, c='k', zorder=100)
ax1.set_ylabel('Relative Flux')
ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
ax1.xaxis.set_major_formatter(plt.NullFormatter())

# Bottom panel
ax2 = plt.subplot(gs[1])
ax2.errorbar(tim[instrument], (fl[instrument]-model)*1e6, yerr=fle[instrument]*1e6, fmt='.', alpha=0.3)
ax2.axhline(y=0.0, c='black', ls='--', zorder=10)
ax2.set_ylabel('Residuals (ppm)')
ax2.set_xlabel('Time (BJD)')
ax2.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
plt.savefig(pout + '/Figure_1.png', dpi=500)