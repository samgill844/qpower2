import numpy as np 
import numba, numba.cuda 
import math


###################################################
# Temperature colour relations of Boyajian et al. 2013 (http://iopscience.iop.org/article/10.1088/0004-637X/771/1/40/pdf)
###################################################

@numba.jit#('float64(float64, float64)', nopython=True)
def BV_to_Teff(BV, z0) : 
    if (BV >= -0.02) & (BV <= 1.73) : return -z0 + 9552 -17443*BV + 44350*BV**2 -68940*BV**3 + 57338*BV**4 -24072*BV**5 + 4009*BV**6
    else: return np.nan 


    
@numba.njit#('float64(float64, float64)', nopython=True)
def VJ_to_Teff(VJ, z0) :
    if (VJ >= -0.12) and (VJ <= 4.24) : return -z0 + 9052 -3972*VJ + 1039*VJ**2 -101*VJ**3
    else: return np.nan 

@numba.njit#('float64(float64, float64)', nopython=True)
def VH_to_Teff(VH, z0) : 
    if (VH >= -0.13) and (VH <= 4.77) : return -z0 + 8958 -3023*VH + 632*VH**2 -52.9*VH**3
    else: return np.nan 

@numba.njit#('float64(float64, float64)', nopython=True)
def VK_to_Teff(VK, z0) : 
    if (VK >= -0.15) and (VK <= 5.04) : return -z0 + 8984 -2914*VK + 588*VK**2 -47.4*VK**3
    else: return np.nan 



@numba.njit#('float64(float64, float64)', nopython=True)
def gr_to_Teff(gr, z0):
    if (gr >= -0.23) & (gr <= 1.40): return -z0 + 7526 -5570*gr + 3750*gr**2 - 1332.9*gr**3 #     return np.polyval([-1332.9, 3750, -5570, 7526],gr)
    else : return np.nan

@numba.jit#('float64(float64, float64)', nopython=True)
def gi_to_Teff(gi, z0): 
    if (gi >= -0.43) & (gi <= 2.78) : return -z0 + 7279 - 3356*gi + 112*gi*2 - 153.9*gi**3 #return np.polyval([-153.9, 1112, -3356, 7279],gi)
    else : return np.nan

@numba.jit#('float64(float64, float64)', nopython=True)
def gJ_to_Teff(gJ, z0):
    if (gJ >= -0.02) & (gJ <= 5.06) : return -z0 + 8759 - 2993*gJ + 623*gJ**2 - 51.5*gJ**3 # return np.polyval([-51.5, 623, -2933, 8759],gJ)
    else: return np.nan

@numba.njit#('float64(float64, float64)', nopython=True)
def gH_to_Teff(gH, z0):
    if (gH >= -0.12) & (gH <= 5.59) : return  -z0 + 8744 - 2396*gH + 432*gH**2 - 32.3*gH**3 #   np.polyval([-32.3, 432, -2396, 8744],gH)
    else : return np.nan

@numba.njit#('float64(float64, float64)', nopython=True)
def gK_to_Teff(gK, z0):
    if (gK >= -0.01) & (gK <= 5.86) : return -z0 + 8618 - 2178*gK + 365*gK**2 - 25.8*gK**3 #    np.polyval([-25.8, 365, -2178, 8618],gK)
    else : return np.nan




@numba.njit#('float64(float64,float64)', nopython=True)
def gr_to_B(g, r):
    # Lupton (2005) transformation 
    # See https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php
    # sigma_B = 0.0107
    # sigma_V = 0.0054
    return g + 0.3130*(g - r) + 0.2271


@numba.njit#('float64(float64,float64)', nopython=True)
def gr_to_V(g, r):
    # Lupton (2005) transformation 
    # See https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php
    # sigma_B = 0.0107
    # sigma_V = 0.0054
    return  g - 0.5784*(g - r) - 0.0038





###################################################
# mass-colour relations of Moya et al for solar-type stars (https://arxiv.org/pdf/1806.06574.pdf)
###################################################
@numba.njit
def MassToTeff(M, z0): 
    # Good between 0.630 - 31.622 M_sol
    M = math.log(M)
    #if (M < -0.2) or (M > 1.5) : return np.nan
    return -z0 +  10**(3.73 + 0.567*M + 0.284*M**2 - 0.182*M**3) 

###################################################
# Temperaature-mass relation from Southworth 2009
###################################################
@numba.njit
def TfromM(M) : return 3217 - 2427*M + 7509*M**2 - 2771*M**3

@numba.njit
def RfromM(M) : return 0.00676 + 1.01824*M 

###################################################
# Fortran conversions
###################################################
@numba.njit
def sign(a,b) : 
    if b >= 0.0 : return abs(a)
    return -abs(a)



###################################################
# Brent minimisation
###################################################

@numba.njit
def brent(func,x1,x2, z0):
    # pars
    tol = 1e-5
    itmax = 100
    eps = 1e-5

    a = x1
    b = x2
    c = 0.
    d = 0.
    e = 0.
    fa = func(a,z0)
    fb = func(b,z0)

    fc = fb

    for iter in range(itmax):
        if (fb*fc > 0.0):
            c = a
            fc = fa
            d = b-a
            e=d   

        if (abs(fc) < abs(fb)):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol1 = 2.0*eps*abs(b)+0.5*tol
        xm = (c-b)/2.0
        if (abs(xm) <  tol1 or fb == 0.0) : return b

        if (abs(e) > tol1 and abs(fa) >  abs(fb)):
            s = fb/fa
            if (a == c):
                p = 2.0*xm*s
                q = 1.0-s
            else:
                q = fa/fc
                r = fb/fc
                p = s*(2.0*xm*q*(q-r)-(b-a)*(r-1.0))
                q = (q-1.0)*(r-1.0)*(s-1.0)
            
            if (p > 0.0) : q = - q
            p = abs(p)
            if (2.0*p < min(3.0*xm*q-abs(tol1*q),abs(e*q))):
                e = d
                d = p/q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d   

        a = b
        fa = fb      
         
        if( abs(d) > tol1) : b = b + d
        else : b = b + sign(tol1, xm)

        fb = func(b,z0)
    return 1






###################################################
# Extiction relations
###################################################
@numba.jit#('float64(float64)', nopython=True)
def Ar_to_EBV(A_r):
  """
  Using A(r)/E(B-V) from Fig. 3 of Fiorucci & Munari, 2003A&A...401..781F 
  Returns value =  A_r/2.770 
  """
  return A_r/2.770

@numba.jit#('float64(float64)', nopython=True)
def EBV_to_Ar(EBV):
  """
  Using A(r)/E(B-V) from Fig. 3 of Fiorucci & Munari, 2003A&A...401..781F 
  Returns value =  A_r/2.770 
  """
  return EBV*2.770

'''

B,  V,  g,  r,  i,  J,  H,  K  = 12.142 , 11.541 , 11.785 , 11.438 , 11.317 , 10.530 , 10.249 , 10.184
Be, Ve, ge, re, ie, Je, He, Ke = 0.039  , 0.010  , 0.013,   0.033  , 0.013  , 0.023  , 0.022,   0.019
EBV_map, EBV_mape = 0.010, 0.034
EBV_map = 0.1
EBV_mape *= 5
print('B - V : ', BV_to_Teff(B-V,0))
print('V - J : ', VJ_to_Teff(V-J,0))
print('V - H : ', VH_to_Teff(V-H,0))
print('V - K : ', VK_to_Teff(V-K,0))
print('g - r : ', gr_to_Teff(g-r,0))
print('g - i : ', gi_to_Teff(g-i,0))
print('g - J : ', gJ_to_Teff(g-J,0))
print('g - H : ', gH_to_Teff(g-H,0))
print('g - K : ', gK_to_Teff(g-K,0))
'''






# free params = g0, teff
@numba.njit #('float64(float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64)')
def get_model_go_teff(theta, _B,  _V,  _g,  _r,  _i,  _J,  _H,  _K, EBV_map, Be, Ve, ge, re, ie, Je, He, Ke, EBV_mape):

    g_0, teff, EBV, sig_ext = theta

    if (g_0 < 10) or (g_0 > 12) : return -np.inf
    if (sig_ext < 0) or (sig_ext > 10) : return -np.inf
    if (teff < 3500) or (teff > 8000)  :return -np.inf
    if (EBV < 0.) or (EBV > 1) : return -np.inf


    A_r = EBV_to_Ar(EBV) 

    r_0 = g_0 - brent(gr_to_Teff,-0.23,1.4, teff)

    B_0 = gr_to_B(g_0, r_0)
    V_0 = gr_to_V(g_0, r_0)
    V = V_0 + 3.1*EBV
    B = V + (B_0-V_0) + EBV

    g = g_0 + 1.39*A_r
    r = r_0 + A_r
    i_0 = g_0 - brent(gi_to_Teff,-0.43,2.78, teff) 
    i = i_0 + 0.76*A_r

    J_0 = g_0 - brent(gJ_to_Teff,-0.02,5.06, teff)  
    J_J = J_0 + 0.30*A_r # Johnson J
    H_J = g_0 - brent(gH_to_Teff,-0.12,5.59, teff)  + 0.21*A_r  # Johnson H
    K_0 = g_0 - brent(gK_to_Teff,-0.01, 5.86, teff) # Unreddened Johnson K
    K_J = K_0 + 0.15*A_r 
    # Transformation from Johnson to Bessel & Brett system from
    # 1988PASP..100.1134B
    VK_BB = 0.01 + 0.993 * (V - K_J)
    JH_BB = -0.004 + 1.01*(J_J - H_J)
    JK_BB = 0.01 + 0.99*(J_J - K_J)

    K_BB = V - VK_BB 
    Ks = K_BB -0.039 + 0.001*JK_BB 
    JKs = 0.983*JK_BB - 0.018
    JH  = 0.990*JH_BB - 0.049
    # HKs = 0.971*HK_BB + 0.034
    J = JKs + Ks
    H = J - JH 

    # Now do loglikes
    wt = 1.0 / (Be**2 + sig_ext**2)
    loglike = -0.5*((_B - B)**2 * wt - math.log(wt))

    wt = 1.0 / (Ve**2 + sig_ext**2)
    loglike -= 0.5*((_V - V)**2 * wt - math.log(wt))

    wr = 1.0 / (ge**2 + sig_ext**2)
    loglike -= 0.5*((_g - g)**2 * wt - math.log(wt))

    wt = 1.0 / (re**2 + sig_ext**2)
    loglike -= 0.5*((_r - r)**2 * wt - math.log(wt))

    wt = 1.0 / (ie**2 + sig_ext**2)
    loglike -= 0.5*((_i - i)**2 * wt - math.log(wt))

    wt = 1.0 / (Je**2 + sig_ext**2)
    loglike -= 0.5*((_J - J)**2 * wt - math.log(wt))

    wt = 1.0 / (He**2 + sig_ext**2)
    loglike -= 0.5*((_H - H)**2 * wt - math.log(wt))

    wt = 1.0 / (Ke**2 + sig_ext**2)
    loglike -= 0.5*((_K - Ks)**2 * wt - math.log10(wt))

    wt = 1.0 / (EBV_mape**2 + sig_ext**2)
    loglike -= 0.5*((EBV - EBV_map)**2 * wt - math.log10(wt))

    return loglike


'''
args = (B,  V,  g,  r,  i,  J,  H,  K, EBV_map, Be, Ve, ge, re, ie, Je, He, Ke, EBV_mape)

import emcee

theta = np.array([11.48, 5999, 0.014,0.1], dtype = np.float64)

ndim = len(theta)

nwalkers = 4*ndim

p0 = np.array([np.random.normal(theta, 1e-5) for i in range(nwalkers)], dtype=np.float64)

sampler = emcee.EnsembleSampler(nwalkers, ndim, get_model_go_teff, args = args)

sampler.run_mcmc(p0, 100000)

samples = sampler.chain[:, 80000:, :].reshape((-1, ndim))

import corner
fig = corner.corner(samples, labels=[r'$g_0$', r'$T_{\rm eff} \, \rm [K]$', r'$E(B-V)$', r'$\sigma_{\rm ext}$'])

import matplotlib.pyplot as plt 
plt.show()
'''






if numba.cuda.is_available():
    @numba.cuda.jit('void(float64[:,:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:])')
    def d_get_model_go_teff_batch(theta, _B,  _V,  _g,  _r,  _i,  _J,  _H,  _K, EBV_map, Be, Ve, ge, re, ie, Je, He, Ke, EBV_mape, loglike):
        # Get model index
        i = numba.cuda.grid(1)

        # Now get loglike
        loglike[i] =  get_model_go_teff(theta[i,:], _B,  _V,  _g,  _r,  _i,  _J,  _H,  _K, EBV_map, Be, Ve, ge, re, ie, Je, He, Ke, EBV_mape)


    def get_model_go_teff_batch(theta, _B,  _V,  _g,  _r,  _i,  _J,  _H,  _K, EBV_map, Be, Ve, ge, re, ie, Je, He, Ke, EBV_mape):
        # Allocate to device
        d_theta = numba.cuda.to_device(theta)
        d_loglike = numba.cuda.to_device(np.zeros(len(theta)))

        d_get_model_go_teff_batch[int(np.ceil(len(theta)/256.0)), 256]( d_theta, _B,  _V,  _g,  _r,  _i,  _J,  _H,  _K, EBV_map, Be, Ve, ge, re, ie, Je, He, Ke, EBV_mape, d_loglike)

        return d_loglike.copy_to_host()