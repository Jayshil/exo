import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
jax.config.update(
    "jax_enable_x64", True
)  # For 64-bit precision since JAX defaults to 32-bit
from jax.random import PRNGKey, split
import astropy.constants as con
import celerite2.jax
from celerite2.jax import terms as jax_terms
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro import distributions as dist
from numpyro.infer.autoguide import AutoLaplaceApproximation
import numpyro_ext.optim
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.units import unit_registry as ureg
from jaxoplanet.orbits import keplerian
import arviz as az
import corner
import os
import pickle
import time
from utils import transit_model

cpu_cores = 2
numpyro.set_host_device_count(cpu_cores)

# Some constants that we need
G = con.G.value
rho_sun = (con.M_sun/(con.R_sun**3)).value

# And the planetary parameters
per, per_err = 4.7360990, np.sqrt((0.0000290**2) + (0.0000270**2))
tc, tc_err = 2458553.81381, 0.00033
ar, ar_err = 4.98, 0.05
bb, bb_err = 0.404, np.sqrt((0.013**2) + (0.018**2))
rprs, rprs_err = 0.0475, 0.0006

tim1 = np.linspace(tc-2*per, tc+2*per, 100000)

# First planet, three instruments
fl_p11 = transit_model(tim=tim1, tc=tc, per=4.74, rprs=0.0475, ar=4.98, bb=0.404, u1=0.1, u2=0.3)
fl_p12 = transit_model(tim=tim1, tc=tc, per=4.74, rprs=0.0485, ar=4.98, bb=0.404, u1=0.15, u2=0.35)
fl_p13 = transit_model(tim=tim1, tc=tc, per=4.74, rprs=0.0495, ar=4.98, bb=0.404, u1=0.20, u2=0.40)

# Second planet, three instruments
fl_p21 = transit_model(tim=tim1, tc=tc, per=5.74, rprs=0.0575, ar=5.98, bb=0.504, u1=0.1, u2=0.3)
fl_p22 = transit_model(tim=tim1, tc=tc, per=5.74, rprs=0.0585, ar=5.98, bb=0.504, u1=0.15, u2=0.35)
fl_p23 = transit_model(tim=tim1, tc=tc, per=5.74, rprs=0.0595, ar=5.98, bb=0.504, u1=0.20, u2=0.40)


def evaluate_model(times, density, t0, per, bb, rprs, u1, u2):

    star = keplerian.Central(density=density/rho_sun)
    body = keplerian.Body(time_transit=t0, period=per, impact_param=bb, eccentricity=0., omega_peri=jnp.pi/2, radius=rprs, radial_velocity_semiamplitude=20.*ureg.m/ureg.s)

    orbit = keplerian.System(central=star).add_body(body=body)

    model = limb_dark_light_curve(orbit, [u1, u2])(times) + 1.

    return model

den1 = density = (3 * jnp.pi * (4.98**3) ) / (G * ((4.74 * 24. * 3600.)**2) )
den2 = density = (3 * jnp.pi * (5.98**3) ) / (G * ((5.74 * 24. * 3600.)**2) )

rprs_all = jnp.array([ [0.0475, 0.0485, 0.0495], [0.0575, 0.0585, 0.0595] ])
per_all = jnp.array([4.74, 5.74])
tc_all = jnp.array([tc, tc])
den_all = jnp.array([den1, den2])
bb_all = jnp.array([0.404, 0.504])
u1_all = jnp.array([[0.1, 0.15, 0.20], [0.1, 0.15, 0.20] ])
u2_all = jnp.array([[0.3, 0.35, 0.40], [0.3, 0.35, 0.40] ])

print(rprs_all[0,1])

evaluate_model_vmap = jax.vmap(evaluate_model, in_axes=(None, None, None, None, None, 0, 0, 0) )
evaluate_model_vmap1 = jax.vmap(evaluate_model_vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, 0) )

Y = evaluate_model_vmap1(tim1,
                         den_all,
                         tc_all,
                         per_all,
                         bb_all,
                         rprs_all,
                         u1_all,
                         u2_all)

print(Y.shape)

plt.plot(tim1, fl_p11, 'k-')
plt.plot(tim1, Y[0,0,:,0], 'b--')
plt.title('Planet 1, Ins 1')
plt.show()

plt.plot(tim1, fl_p12, 'k-')
plt.plot(tim1, Y[0,1,:,0], 'b--')
plt.title('Planet 1, Ins 2')
plt.show()

plt.plot(tim1, fl_p13, 'k-')
plt.plot(tim1, Y[0,2,:,0], 'b--')
plt.title('Planet 1, Ins 3')
plt.show()

plt.plot(tim1, fl_p21, 'k-')
plt.plot(tim1, Y[1,0,:,0], 'b--')
plt.title('Planet 2, Ins 1')
plt.show()

plt.plot(tim1, fl_p22, 'k-')
plt.plot(tim1, Y[1,1,:,0], 'b--')
plt.title('Planet 2, Ins 2')
plt.show()

plt.plot(tim1, fl_p23, 'k-')
plt.plot(tim1, Y[1,2,:,0], 'b--')
plt.title('Planet 2, Ins 3')
plt.show()




plt.plot(tim1, fl_p11-Y[0,0,:,0], 'k-')
plt.title('Planet 1, Ins 1')
plt.show()

plt.plot(tim1, fl_p12-Y[0,1,:,0], 'k-')
plt.title('Planet 1, Ins 2')
plt.show()

plt.plot(tim1, fl_p13-Y[0,2,:,0], 'k-')
plt.title('Planet 1, Ins 3')
plt.show()

plt.plot(tim1, fl_p21-Y[1,0,:,0], 'k-')
plt.title('Planet 2, Ins 1')
plt.show()

plt.plot(tim1, fl_p22-Y[1,1,:,0], 'k-')
plt.title('Planet 2, Ins 2')
plt.show()

plt.plot(tim1, fl_p23-Y[1,2,:,0], 'k-')
plt.title('Planet 2, Ins 3')
plt.show()