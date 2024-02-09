import numpy as np
import batman
from kelp import Planet
from exotoolbox.utils import reverse_ld_coeffs

def transit_model(tim):
    q1, q2 = 0.2647731490, 0.3212826253
    u1, u2 = reverse_ld_coeffs('quadratic', q1, q2)
    planet = Planet(
                    per=2.7240314376,
                    t0=2459024.6067578471,
                    inc=np.degrees(np.arccos(0.4547446640/4.6561340944)),
                    rp=0.0716620112,
                    ecc=0.,
                    w=90.,
                    a=4.6561340944,
                    u=[u1, u2],
                    fp=1e-6,
                    t_secondary=2459024.6067578471 + (2.7240314376/2),
                    T_s=8000,
                    rp_a=0.0716620112/4.6561340944,
                    name='WASP-189'
                )
    transit_model = batman.TransitModel(params=planet, t=tim).light_curve(planet)
    eclipse_model = batman.TransitModel(params=planet, t=tim, transittype='secondary').light_curve(planet)
    eclipse_model = (eclipse_model - 1.)/1e-6
    return transit_model, eclipse_model