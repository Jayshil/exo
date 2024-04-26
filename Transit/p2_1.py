import numpy as np
import batman
import jax.numpy as jnp
from luas.exoplanet import transit_light_curve
import matplotlib.pyplot as plt
from jaxoplanet.orbits import KeplerianOrbit
from jaxoplanet.light_curves import QuadLightCurve

rst, rst_err = 2.69, 0.04
mst, mst_err = 1.44, 0.07
per, t0 = 2.7240314376, 0.0
bb, inc = 0.4547446640, np.degrees(np.arccos(0.4547446640/4.6561340944))
rprs = 0.0716620112
ar = 4.6561340944

"""def transit_light_curve(par, t):
    light_curve = QuadLightCurve.init(u1=par["u1"], u2=par["u2"])
    orbit = KeplerianOrbit.init(
        time_transit=par["T0"],
        period=par["P"],
        semimajor=par["a"],
        impact_param=par["b"],
        radius=par["rho"],
    )    
    flux = 1+light_curve.light_curve(orbit, t)[0]
    return flux"""


# Let's use the literature values used in Gibson et al. (2017) to start us off
mfp = {
    "T0":t0,     # Central transit time
    "P":per,     # Period (days)
    "a":ar,      # Semi-major axis to stellar ratio aka a/R*
    "rho":rprs,   # Radius ratio rho aka Rp/R*
    "b":bb,     # Impact parameter
    "u1":0.5,    # First quadratic limb darkening coefficient
    "u2":0.1,    # Second quadratic limb darkening coefficient
    "Foot":1.,
    "Tgrad":0.,
    "eccentricity":0.
}

# Batman model
def transit_model(tim):
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rprs
    params.a = ar
    params.inc = inc
    params.ecc = 0.
    params.w = 90.
    params.u = [0.5, 0.1]
    params.limb_dark = "quadratic"
    transit_model = batman.TransitModel(params=params, t=tim).light_curve(params)
    return transit_model

# Generate 100 evenly spaced time points (in units of days)
x_t = np.linspace(-(per/10), (per/10), 1000000)
tmodel_bat = transit_model(x_t)
x_t = jnp.asarray(x_t)
lc1 = transit_light_curve(mfp, x_t)

plt.plot(x_t, lc1, "k-")
plt.plot(x_t, tmodel_bat, 'b-')
plt.xlabel("Time (days)")
plt.ylabel("Normalised Flux")
plt.show()

plt.plot(x_t, lc1-tmodel_bat, 'k-')
plt.show()