import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from exotoolbox.utils import reverse_ld_coeffs, tdur
import batman
import radvel
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.units import unit_registry as ureg
from jaxoplanet.orbits import keplerian
import astropy.constants as con

G = con.G.value

import jax
jax.config.update(
    "jax_enable_x64", True
)  # For 64-bit precision since JAX defaults to 32-bit

# The ultimate-model tester
# We will use the planetary parameters from KELT-11 and compare the batman and radvel model with jaxoplanet model

# Planetary parameters
rst, rst_err = 2.69, 0.04
mst, mst_err = 1.44, 0.07
per, t0 = 2.7240314376, 0.0
bb, inc = 0.4547446640, np.degrees(np.arccos(0.4547446640/4.6561340944))
rprs = 0.0716620112
ar = 4.6561340944
u1, u2 = reverse_ld_coeffs('quadratic', 0.2647731490, 0.3212826253)
K = 29.2 # m/s

stellar_density = ((3 * mst * con.M_sun) / (4 * np.pi * ( (rst * con.R_sun)**3 ))).value

def batman_transit_model(tim, density=None):
    if density is not None:
        ar1 = ((density * G * ((per * 24. * 3600.)**2)) / (3. * np.pi))**(1. / 3.)
    else:
        ar1 = ar
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rprs
    params.a = ar1
    params.inc = inc
    params.ecc = 0.
    params.w = 90.
    params.u = [u1, u2]
    params.limb_dark = "quadratic"
    transit_model = batman.TransitModel(params=params, t=tim).light_curve(params)
    return transit_model

def radvel_rv_model(tim):
    anybasis_params = radvel.Parameters(1,basis='per tc e w k', planet_letters={1: 'b'})    # initialize Parameters object

    anybasis_params['per1'] = radvel.Parameter(value=per)      # period of 1st planet
    anybasis_params['tc1'] = radvel.Parameter(value=t0)     # time of inferior conjunction of 1st planet
    anybasis_params['e1'] = radvel.Parameter(value=0.)          # eccentricity of 1st planet
    anybasis_params['w1'] = radvel.Parameter(value=np.pi/2.)      # argument of periastron of the star's orbit for 1st planet
    anybasis_params['k1'] = radvel.Parameter(value=K)          # velocity semi-amplitude for 1st p

    rv_radvel = radvel.model.RVModel(anybasis_params).__call__(tim)
    return rv_radvel

# Keplerian jaxoplanet orbit
def jaxoplanet_model(tim, density=None):
    rho_sun = (con.M_sun/(con.R_sun**3)).value

    if density is None:
        density = (3 * jnp.pi * (ar**3) ) / (G * ((per * 24. * 3600.)**2) )
    else:
        density = density
    star = keplerian.Central(density=density/rho_sun)
    body = keplerian.Body(time_transit=t0, period=per, inclination=np.radians(inc), eccentricity=0., omega_peri=np.pi/2, radius=rprs, radial_velocity_semiamplitude=K*ureg.m/ureg.s)
    orbit_kep = keplerian.System(central=star).add_body(body=body)

    lc = limb_dark_light_curve(orbit_kep, [u1,u2])(tim)[:,0] + 1.
    rv = orbit_kep.radial_velocity(tim)[0].to(ureg.m/ureg.s)

    return lc, rv

den = stellar_density

tims = np.linspace(-1*per, 1*per, 1000)
lc_bat, rv_rad = batman_transit_model(tim=tims, density=den), radvel_rv_model(tim=tims)
lc_jax, rv_jax = jaxoplanet_model(tim=tims, density=den)

plt.plot(tims, lc_bat, 'k-')
plt.plot(tims, lc_jax, 'b--')
plt.title('LC BAT vs JAX')
plt.show()

plt.plot(tims, lc_bat - lc_jax, 'k-')
plt.title('Diff LC')
plt.show()

plt.plot(tims, rv_rad, 'k-')
plt.plot(tims, rv_jax, 'b--')
plt.title('RV BAT vs JAX')
plt.show()

plt.plot(tims, rv_rad - rv_jax.magnitude, 'k-')
plt.title('Diff RV')
plt.show()