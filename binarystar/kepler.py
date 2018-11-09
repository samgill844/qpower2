import numba, numba.cuda
import math 
from math import fmod
import numpy as np
from qpower2.binarystar.utils import *

@numba.njit('float64(float64,float64)', nogil=True)
def getEccentricAnomaly(M, e):
    # calculates the eccentric anomaly (see Seager Exoplanets book:  Murray & Correia eqn. 5 -- see section 3)
    if (e == 0.0) : return M

    m = np.fmod(M, (2*math.pi))
    #m = M % 2*math.pi
    flip = 0
    if (m > math.pi) : m = 2*math.pi - m; flip = 1

    alpha = (3*math.pi + 1.6*(math.pi-math.fabs(m))/(1+e) )/(math.pi - 6/math.pi)
    d = 3*(1 - e) + alpha*e
    r = 3*alpha*d * (d-1+e)*m + m*m*m
    q = 2*alpha*d*(1-e) - m*m
    w = math.pow((math.fabs(r) + math.sqrt(q*q*q + r*r)),(2/3))
    E = (2*r*w/(w*w + w*q + q*q) + m) / d
    f_0 = E - e*math.sin(E) - m
    f_1 = 1 - e*math.cos(E)
    f_2 = e*math.sin(E)
    f_3 = 1-f_1
    d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1)
    d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3*d_3*d_3)*f_3/6)
    E = E -f_0/(f_1 + 0.5*d_4*f_2 + d_4*d_4*f_3/6 - d_4*d_4*d_4*f_2/24)

    if (flip==1) : E =  2*math.pi - E
    return E

@numba.njit('float64(float64,float64,float64,float64,float64)', nogil=True)
def getTrueAnomaly(time, e, w, period,t_zero):
    nu = math.pi/2. - w                                              # true anomaly corresponding to time of primary transit center
    n = 2.*math.pi/period;	                                         # mean motion
    E = 2.*math.atan(math.sqrt((1. - e)/(1. + e))*math.tan(nu/2.))   # corresponding eccentric anomaly
    M = E - e*math.sin(E)                                            # Mean anomaly
    tp = t_zero - period*M/2./math.pi                                # time of periastron

    if (e < 1.0e-5) : return ((time - tp)/period - math.floor( (((time - tp)/period))))*2.*math.pi
    else:
        M = n*(time - tp)
        E = getEccentricAnomaly(M, e)
        return 2.*math.atan(math.sqrt((1.+e)/(1.-e))*math.tan(E/2.)); 

@numba.njit('float64(float64,float64,float64,float64,float64)', nogil=True)
def get_z(e, incl, nu, w, radius_1) : return (1-e*e) * math.sqrt( 1.0 - math.sin(incl)*math.sin(incl)  *  math.sin(nu + w)*math.sin(nu + w)) / (1 + e*math.sin(nu)) /radius_1;


@numba.njit('float64(float64,float64,float64)', nogil=True)
def getProjectedPosition(nu, w, incl) : return math.sin(nu + w)*math.sin(incl)





####################
# Mass function
##################
@numba.njit
def mass_function_1(e, P, K1):
    G = 6.67408e-11
    return ((1-e**2)**1.5)*P*86400.1*((K1*10**3)**3)/(2*math.pi*G*1.989e30) 


@numba.njit#('float64(float64, float64[:])')
def d_mass_function(M2, z0):
    #M1, i, e, P, K1 = z0
    #return ((M2*math.sin(i))**3 / ((M1 + M2)**2)) - mass_function_1(e, P, K1)
    return ((M2*math.sin(z0[1]))**3 / ((z0[0] + M2)**2)) - mass_function_1(z0[2], z0[3], z0[4])


####################
# GPU functions
##################
if numba.cuda.is_available():
    @numba.cuda.jit('float64(float64,float64)', device=True, inline=True)
    def d_getEccentricAnomaly(M, e):
        # calculates the eccentric anomaly (see Seager Exoplanets book:  Murray & Correia eqn. 5 -- see section 3)
        if (e == 0.0) : return M

        #m = np.fmod(M, 2.2)# (2*math.pi))
        m = M % 2*math.pi
        flip = 0
        if (m > math.pi) : m = 2*math.pi - m; flip = 1

        alpha = (3*math.pi + 1.6*(math.pi-math.fabs(m))/(1+e) )/(math.pi - 6/math.pi)
        d = 3*(1 - e) + alpha*e
        r = 3*alpha*d * (d-1+e)*m + m*m*m
        q = 2*alpha*d*(1-e) - m*m
        w = math.pow((math.fabs(r) + math.sqrt(q*q*q + r*r)),(2/3))
        E = (2*r*w/(w*w + w*q + q*q) + m) / d
        f_0 = E - e*math.sin(E) - m
        f_1 = 1 - e*math.cos(E)
        f_2 = e*math.sin(E)
        f_3 = 1-f_1
        d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1)
        d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3*d_3*d_3)*f_3/6)
        E = E -f_0/(f_1 + 0.5*d_4*f_2 + d_4*d_4*f_3/6 - d_4*d_4*d_4*f_2/24)

        if (flip==1) : E =  2*math.pi - E
        return E

    @numba.cuda.jit('float64(float64,float64,float64,float64,float64)', device=True, inline=True)
    def d_getTrueAnomaly(time, e, w, period,t_zero):
        nu = math.pi/2. - w                                              # true anomaly corresponding to time of primary transit center
        n = 2.*math.pi/period;	                                         # mean motion
        E = 2.*math.atan(math.sqrt((1. - e)/(1. + e))*math.tan(nu/2.))   # corresponding eccentric anomaly
        M = E - e*math.sin(E)                                            # Mean anomaly
        tp = t_zero - period*M/2./math.pi                                # time of periastron

        if (e < 1.0e-5) : return ((time - tp)/period - math.floor( (((time - tp)/period))))*2.*math.pi
        else:
            M = n*(time - tp)
            E = d_getEccentricAnomaly(M, e)
            return 2.*math.atan(math.sqrt((1.+e)/(1.-e))*math.tan(E/2.)); 

    @numba.cuda.jit('float64(float64,float64,float64)', device=True, inline=True)
    def d_getProjectedPosition(nu, w, incl) : return math.sin(nu + w)*math.sin(incl)

    @numba.cuda.jit('float64(float64,float64,float64,float64,float64)', device=True, inline=True)
    def d_get_z(e, incl, nu, w, radius_1) : return (1-e*e) * math.sqrt( 1.0 - math.sin(incl)*math.sin(incl)  *  math.sin(nu + w)*math.sin(nu + w)) / (1 + e*math.sin(nu)) /radius_1;
