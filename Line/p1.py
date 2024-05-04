import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import dynesty
from dynesty.utils import resample_equal
from dynesty import plotting as dyplot
from exotoolbox.utils import get_quantiles
import pickle
import os
from multiprocessing import Pool
import contextlib
import multiprocessing
multiprocessing.set_start_method('fork')

# Output folder
pout = os.getcwd() + '/Line/Analysis/Dyn1'

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

def log_like(vars):
    global x, y, ye
    m, c = vars

    # Model of a line
    model = m*x + c

    # Gaussian log-likelihood function
    log2pi = np.log(2*np.pi)
    resid = (y-model)**2
    taus = 1. / ye**2
    return -0.5 * (len(resid) * log2pi + np.sum(-np.log(taus.astype(float)) + taus * (resid**2)))

# Defining prior cube
def uniform(t, a, b):
    return (b-a)*t + a
def stand(a, loc, scale):
    return (a-loc)/scale

par = ['m', 'c']
dist = ['uniform', 'uniform']
hyper = [[0., 1.], [1., 5.]]

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

# And the sampling
nthreads=8
with contextlib.closing(Pool(processes=nthreads-1)) as executor:
    dsampler = dynesty.DynamicNestedSampler(loglikelihood=log_like, prior_transform=prior_transform,\
        ndim=len(par), nlive=500, bound='single', sample='rwalk', pool=executor, queue_size=nthreads)
    dsampler.run_nested()
dres = dsampler.results


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