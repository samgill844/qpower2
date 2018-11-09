from qpower2.photometry.phot import brent, MassToTeff, TfromM
from qpower2.binarystar.kepler import d_mass_function
import numpy as np 
import numba, numba.cuda


T = np.random.normal(6000,200,10240)
M = np.random.normal(0,0.01,10240)
i = np.pi/2 * np.ones(len(T))
e = np.zeros(len(T))
P = 3.55 * np.ones(len(T))
K1 = 21.9 * np.ones(len(T))

@numba.jit
def get_T2(teff, FeH, incl, e, P, K1):
    # Allocation
    M1 = np.empty_like(teff)
    M2 = np.empty_like(teff)
    T2 = np.empty_like(teff)

    # First get the mass a the mass-temperature relation
    for i in range(len(teff)) : M1[i] = brent(MassToTeff, 0.9, 1.3, teff[i])

    # Now get M2
    for i in range(len(teff)):
        M2[i] = brent(d_mass_function, 0.05, 0.5, [M1[i], incl[i], e[i], P[i], K1[i]])
        T2[i] = TfromM(M2[i]) 

    return T2



@numba.cuda.jit('void(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:], float64[:], float64[:,:])')
def d_get_T2(teff, FeH, incl, e, P, K1, T2, args):
    i = numba.cuda.grid(1)

    # First get the mass a the mass-temperature relation
    M1 = brent(MassToTeff, 0.9, 1.3, teff[i])

    # Now get M2
    args[i][0] = M1
    args[i][1] = incl[i]
    args[i][2] =  e[i]
    args[i][3] = P[i]
    args[i][4] = K1[i]
    
    M2 = brent(d_mass_function, 0.05, 0.5, args[i])
    T2[i] = TfromM(M2) 


def get_T2_gpu(teff, FeH, incl, e, P, K1, return_device_array=False):
    N = len(teff)
    T2 = numba.cuda.to_device(teff.astype(np.float64))
    teff = numba.cuda.to_device(teff.astype(np.float64))
    FeH = numba.cuda.to_device(FeH.astype(np.float64))
    incl = numba.cuda.to_device(incl.astype(np.float64))
    e = numba.cuda.to_device(e.astype(np.float64))
    P = numba.cuda.to_device(P.astype(np.float64))
    K1 = numba.cuda.to_device(K1.astype(np.float64))

    # Have to do this to comply with the brent function input (z0) 
    # which requires a numpy 1d array of length 5 (for d_mass_function)
    args = numba.cuda.to_device(np.zeros((N, 5)).astype(np.float64))

    # Lanuch 
    d_get_T2[int(np.ceil(N/512)), 512](teff, FeH, incl, e, P, K1, T2, args)

    if return_device_array : return T2
    return T2.copy_to_host()








