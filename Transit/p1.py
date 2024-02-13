import numpy as np
import matplotlib.pyplot as plt
from exotoolbox.utils import reverse_ld_coeffs, tdur
import exoplanet as xo
import batman
import pymc3 as pm

rst, rst_err = 2.69, 0.04
mst, mst_err = 1.44, 0.07
per, t0 = 2.7240314376, 2459024.6067578471
bb, inc = 0.4547446640, np.degrees(np.arccos(0.4547446640/4.6561340944))
rprs = 0.0716620112
ar = 4.6561340944
u1, u2 = reverse_ld_coeffs('quadratic', 0.2647731490, 0.3212826253)
duration = tdur(per=per, ar=ar, rprs=rprs, bb=bb)

def transit_model(tim):
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rprs
    params.a = ar
    params.inc = inc
    params.ecc = 0.
    params.w = 90.
    params.u = [u1, u2]
    params.limb_dark = "quadratic"
    transit_model = batman.TransitModel(params=params, t=tim).light_curve(params)
    return transit_model


tims = np.linspace(t0-(per/10), t0+(per/10), 1000)
tmodel_bat = transit_model(tims)

# The light curve calculation requires an orbit
orbit = xo.orbits.SimpleTransitOrbit(period=per, duration=duration, t0=t0, b=bb, ror=rprs)
light_curve = (xo.LimbDarkLightCurve(u1, u2).get_light_curve(orbit=orbit, r=rprs, t=tims).eval())
lc1 = pm.math.sum(light_curve, axis=-1) + 1.

plt.plot(tims, tmodel_bat, 'k-')
plt.plot(tims, lc1.eval(), 'b-')
plt.axvline(t0-(duration/2))
plt.axvline(t0+(duration/2))
plt.show()

plt.plot(tims, tmodel_bat - lc1.eval(), 'k-')
plt.show()