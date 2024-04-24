import numpy as np
import matplotlib.pyplot as plt
from exotoolbox.utils import reverse_ld_coeffs, tdur
import batman
import jaxoplanet
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits import TransitOrbit

rst, rst_err = 2.69, 0.04
mst, mst_err = 1.44, 0.07
per, t0 = 2.7240314376, 0.0
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


tims = np.linspace(-(per/10), (per/10), 1000000)
tmodel_bat = transit_model(tims)

# The light curve calculation requires an orbit
orbit = TransitOrbit(period=per, duration=duration, time_transit=t0, impact_param=bb, radius=rprs)


# Compute a limb-darkened light curve for this orbit
u = [u1, u2]  # Quadratic limb-darkening coefficients
lc = limb_dark_light_curve(orbit, u)(tims) + 1.

plt.plot(tims, tmodel_bat, 'k-')
plt.plot(tims, lc, 'b-')
plt.axvline(t0-(duration/2))
plt.axvline(t0+(duration/2))
plt.show()

plt.plot(tims, tmodel_bat - lc, 'k-')
plt.show()