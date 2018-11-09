import numba, numba.cuda
import math 

from qpower2.binarystar.utils import *

@numba.njit('float64(float64,float64,float64,float64,float64,float64)', nogil=True)
def q1(z, p, c, a, g, I_0):
	zt = clip(abs(z), 0,1-p)
	s = 1-zt*zt
	c0 = (1-c+c*math.pow(s,g))
	c2 = 0.5*a*c*math.pow(s,(g-2))*((a-1)*zt*zt-1)
	return 1-I_0*math.pi*p*p*(c0 + 0.25*p*p*c2 - 0.125*a*c*p*p*math.pow(s,(g-1)))



@numba.njit('float64(float64,float64,float64,float64,float64,float64,float64)', nogil=True)
def q2(z, p, c, a, g, I_0, eps):
	zt = clip(abs(z), 1-p,1+p)
	d = clip((zt*zt - p*p + 1)/(2*zt),0,1)
	ra = 0.5*(zt-p+d)
	rb = 0.5*(1+d)
	sa = clip(1-ra*ra,eps,1)
	sb = clip(1-rb*rb,eps,1)
	q = clip((zt-d)/p,-1,1)
	w2 = p*p-(d-zt)*(d-zt)
	w = math.sqrt(clip(w2,eps,1))
	c0 = 1 - c + c*math.pow(sa,g)
	c1 = -a*c*ra*math.pow(sa,(g-1))
	c2 = 0.5*a*c*math.pow(sa,(g-2))*((a-1)*ra*ra-1)
	a0 = c0 + c1*(zt-ra) + c2*(zt-ra)*(zt-ra)
	a1 = c1+2*c2*(zt-ra)
	aq = math.acos(q)
	J1 =  (a0*(d-zt)-(2./3.)*a1*w2 + 0.25*c2*(d-zt)*(2.0*(d-zt)*(d-zt)-p*p))*w + (a0*p*p + 0.25*c2*math.pow(p,4))*aq 
	J2 = a*c*math.pow(sa,(g-1))*math.pow(p,4)*(0.125*aq + (1./12.)*q*(q*q-2.5)*math.sqrt(clip(1-q*q,0.0,1.0)) )
	d0 = 1 - c + c*math.pow(sb,g)
	d1 = -a*c*rb*math.pow(sb,(g-1))
	K1 = (d0-rb*d1)*math.acos(d) + ((rb*d+(2./3.)*(1-d*d))*d1 - d*d0)*math.sqrt(clip(1-d*d,0.0,1.0))
	K2 = (1/3)*c*a*math.pow(sb,(g+0.5))*(1-d)
	if J1 > 1 : J1 = 0
	return 1 - I_0*(J1 - J2 + K1 - K2)

@numba.njit('float64(float64,float64,float64,float64,float64,float64)', nogil=True)
def Flux_drop_analytical_power_2(z, k, c, a, f, eps):
    '''
    Calculate the analytical flux drop por the power-2 law.

    Parameters
    z : double
        Projected seperation of centers in units of stellar radii.
    k : double
        Ratio of the radii.
    c : double
        The first power-2 coefficient.
    a : double
        The second power-2 coefficient.
    f : double
        The flux from which to drop light from.
    eps : double
        Factor (1e-9)
    '''
    I_0 = (a+2)/(math.pi*(a-c*a+2))
    g = 0.5*a

    if (z < 1-k) : return q1(z, k, c, a, g, I_0)
    elif (abs(z-1) < k) : return q2(z, k, c, a, g, I_0, eps)
    else: return 1.0

#@numba.njit('float64(float64,float64)')
def ctoh1(c1, c2) : return 1 - c1*(1 - 2**(-c2))    

#@numba.njit('float64(float64,float64)')
def ctoh2(c1, c2) : return c1*2**(-c2)    


#@numba.njit('float64(float64,float64)')
def htoc1(h1, h2) : return 1 - h1 + h2

#@numba.njit('float64(float64,float64)')
def htoc2(c1, h2) : return np.log2(c1 / h2)


####################
# GPU functions
##################
if numba.cuda.is_available():
    @numba.cuda.jit('float64(float64,float64,float64,float64,float64,float64)', device=True, inline=True)
    def d_q1(z, p, c, a, g, I_0):
        zt = d_clip(abs(z), 0,1-p)
        s = 1-zt*zt
        c0 = (1-c+c*math.pow(s,g))
        c2 = 0.5*a*c*math.pow(s,(g-2))*((a-1)*zt*zt-1)
        return 1-I_0*math.pi*p*p*(c0 + 0.25*p*p*c2 - 0.125*a*c*p*p*math.pow(s,(g-1)))

    @numba.cuda.jit('float64(float64,float64,float64,float64,float64,float64,float64)', device=True, inline=True)
    def d_q2(z, p, c, a, g, I_0, eps):
        zt = d_clip(abs(z), 1-p,1+p)
        d = d_clip((zt*zt - p*p + 1)/(2*zt),0,1)
        ra = 0.5*(zt-p+d)
        rb = 0.5*(1+d)
        sa = d_clip(1-ra*ra,eps,1)
        sb = d_clip(1-rb*rb,eps,1)
        q = d_clip((zt-d)/p,-1,1)
        w2 = p*p-(d-zt)*(d-zt)
        w = math.sqrt(d_clip(w2,eps,1))
        c0 = 1 - c + c*math.pow(sa,g)
        c1 = -a*c*ra*math.pow(sa,(g-1))
        c2 = 0.5*a*c*math.pow(sa,(g-2))*((a-1)*ra*ra-1)
        a0 = c0 + c1*(zt-ra) + c2*(zt-ra)*(zt-ra)
        a1 = c1+2*c2*(zt-ra)
        aq = math.acos(q)
        J1 =  (a0*(d-zt)-(2./3.)*a1*w2 + 0.25*c2*(d-zt)*(2.0*(d-zt)*(d-zt)-p*p))*w + (a0*p*p + 0.25*c2*math.pow(p,4))*aq 
        J2 = a*c*math.pow(sa,(g-1))*math.pow(p,4)*(0.125*aq + (1./12.)*q*(q*q-2.5)*math.sqrt(d_clip(1-q*q,0.0,1.0)) )
        d0 = 1 - c + c*math.pow(sb,g)
        d1 = -a*c*rb*math.pow(sb,(g-1))
        K1 = (d0-rb*d1)*math.acos(d) + ((rb*d+(2./3.)*(1-d*d))*d1 - d*d0)*math.sqrt(d_clip(1-d*d,0.0,1.0))
        K2 = (1/3)*c*a*math.pow(sb,(g+0.5))*(1-d)
        if J1 > 1 : J1 = 0
        return 1 - I_0*(J1 - J2 + K1 - K2)

    @numba.cuda.jit('float64(float64,float64,float64,float64,float64,float64)', device=True, inline=True)
    def d_Flux_drop_analytical_power_2(z, k, c, a, f, eps):
        '''
        Calculate the analytical flux drop por the power-2 law.

        Parameters
        z : double
            Projected seperation of centers in units of stellar radii.
        k : double
            Ratio of the radii.
        c : double
            The first power-2 coefficient.
        a : double
            The second power-2 coefficient.
        f : double
            The flux from which to drop light from.
        eps : double
            Factor (1e-9)
        '''
        I_0 = (a+2)/(math.pi*(a-c*a+2))
        g = 0.5*a

        if (z < 1-k) : return d_q1(z, k, c, a, g, I_0)
        elif (abs(z-1) < k) : return d_q2(z, k, c, a, g, I_0, eps)
        else: return 1.0