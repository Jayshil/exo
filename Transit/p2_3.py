import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from exotoolbox.utils import reverse_ld_coeffs
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


# Keplerian jaxoplanet orbit
def jaxoplanet_model(density=None):
    rho_sun = (con.M_sun/(con.R_sun**3)).value

    if density is None:
        density = (3 * jnp.pi * (ar**3) ) / (G * ((per * 24. * 3600.)**2) )
    else:
        density = density
    star = keplerian.Central(density=density/rho_sun)
    body = keplerian.Body(time_transit=t0, period=per, inclination=np.radians(inc), eccentricity=0., omega_peri=np.pi/2, radius=rprs, radial_velocity_semiamplitude=K*ureg.m/ureg.s)
    #orbit_kep = keplerian.System(central=star).add_body(body=body)

    return star, body

tims = np.linspace(-1*per, 1*per, 1000)

star, planet = jaxoplanet_model()
print(planet)

lc1 = limb_dark_light_curve(keplerian.System(central=star, bodies=[planet]), [u1,u2])(tims)[:,0] + 1.

#planet.radius._magnitude = 0.1
planet.period._magnitude = 2.75

print(planet)

lc2 = limb_dark_light_curve(keplerian.System(central=star, bodies=[planet]), [u1,u2])(tims)[:,0] + 1.

plt.plot(tims, lc1, 'k-')
plt.plot(tims, lc2, 'b-')
plt.show()

plt.plot(tims, lc1-lc2, 'k-')
plt.show()