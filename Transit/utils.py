import numpy as np
import batman

def transit_model(tim, tc, per, rprs, ar, bb, u1, u2):
    # And batman model
    params = batman.TransitParams()
    params.t0 = tc
    params.per = per
    params.rp = rprs
    params.a = ar
    params.inc = np.rad2deg(np.arccos(bb/ar))
    params.ecc = 0.
    params.w = 90.
    params.u = [u1, u2]
    params.limb_dark = "quadratic"

    m = batman.TransitModel(params, tim)    #initializes model
    flux_deter = m.light_curve(params)

    return flux_deter