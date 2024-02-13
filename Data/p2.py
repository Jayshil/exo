import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import mad_std
import batman
import os

# This file is to generate a synthetic transit data for KELT-11 b
# We will use noise properties from a real transit of KELT-11 b observed with CHEOPS

tim_real, fl_real, fle_real = np.loadtxt(os.getcwd() + '/Data/kelt-11-cheops.dat', usecols=(0,1,2), unpack=True)
tim_real = tim_real + 2458918
scatter = np.nanmedian(np.abs(np.diff(fl_real, axis=0)), axis=0)

# And the planetary parameters
per, per_err = 4.7360990, np.sqrt((0.0000290**2) + (0.0000270**2))
tc, tc_err = 2458553.81381, 0.00033
ar, ar_err = 4.98, 0.05
bb, bb_err = 0.404, np.sqrt((0.013**2) + (0.018**2))
rprs, rprs_err = 0.0475, 0.0006


# And batman model
params = batman.TransitParams()
params.t0 = tc
params.per = per
params.rp = rprs
params.a = ar
params.inc = np.rad2deg(np.arccos(bb/ar))
params.ecc = 0.
params.w = 90.
params.u = [0.1, 0.3]
params.limb_dark = "quadratic"

m = batman.TransitModel(params, tim_real)    #initializes model
flux_deter = m.light_curve(params)

fl_synth = np.array([np.random.normal(flux_deter[i], scatter) for i in range(len(tim_real))])
fle_synth = np.random.normal(np.nanmedian(fle_real), mad_std(fle_real), len(fle_real))

f1 = open(os.getcwd() + '/Data/kelt-11-synthetic.dat', 'w')
f1.write('#time\t\tflux\t\tflux_err\n')
for i in range(len(tim_real)):
    f1.write(str(tim_real[i]) + '\t' + str(fl_synth[i]) + '\t' + str(fle_synth[i]) + '\n')
f1.close()

plt.errorbar(tim_real, fl_real, yerr=fle_real, fmt='.', color='k')
plt.errorbar(tim_real, fl_synth, yerr=fle_synth, fmt='.', color='b')
plt.show()