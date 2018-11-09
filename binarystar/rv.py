import numba, numba.cuda
import math 
import numpy as np

from qpower2.binarystar.utils import *
from qpower2.binarystar.kepler import *

N_BATCH = 10240

__all__ = ['rv', 'rv_gpu', 'rv_gpu_batch']

@numba.njit(nogil=True, parallel=True)
def rv(time, t_zero=0.0, period=1.0, K1=10., K2=10.0, fs=0.0, fc=0.0, V0=0.0, dV0=0.0):
    # Unpack args
    w = math.atan2(fs, fc)
    e = fs*fs + fc*fc

    # Allocate the RV arrays
    RV = np.empty((2,len(time)))
    for i in numba.prange(len(time)):
        nu = getTrueAnomaly(time[i], e, w, period, t_zero)
        RV[0][i] = K1*math.cos(nu + w)              + V0  + dV0*(time[i] - t_zero)
        RV[1][i] = K1*math.cos(nu + w + math.pi)    + V0  + dV0*(time[i] - t_zero)
    return RV


####################
# GPU functions
##################
if numba.cuda.is_available():
    @numba.cuda.jit('void(float64[:],float64,float64,float64,float64,float64,float64,float64,float64,float64[:,:])')
    def __d_rv(time, t_zero, period, K1, K2, fs, fc, V0, dV0, RV):
        # Unpack args
        w = math.atan2(fs, fc)
        e = fs*fs + fc*fc

        # Get index
        i = numba.cuda.grid(1)

        nu = d_getTrueAnomaly(time[i], e, w, period, t_zero)
        RV[0][i] = K1*math.cos(nu + w)              + V0  + dV0*(time[i] - t_zero)
        RV[1][i] = K1*math.cos(nu + w + math.pi)    + V0  + dV0*(time[i] - t_zero)



    def rv_gpu(time, t_zero=0.0, period=1.0, K1=10.0, K2=10.0, fs=0.0, fc=0.0, V0=0.0, dV0=0.0):
        # Allocate the RV array
        RV = numba.cuda.to_device(np.empty((2,len(time)), dtype = np.float64))
        
        # Copy over the time array
        d_time = numba.cuda.to_device(time.astype(np.float64))

        # Call the kernel
        __d_rv[int(np.ceil(len(time)/512.0)), 512](d_time,t_zero, period, K1, K2, fs, fc, V0, dV0, RV )

        return RV.copy_to_host()

    
    @numba.cuda.jit('void(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],b1,float64[:,:],float64[:,:])')
    def d_rv_batch(time, t_zero, period, K1, K2, fs, fc, V0, dV0, RV, loglike_switch, loglike_RV, loglike_err):

        # Get model
        j = numba.cuda.grid(1)

        # Unpack args
        w = math.atan2(fs[j], fc[j])
        e = fs[j]*fs[j] + fc[j]*fc[j]

        for i in range(len(time)):
            nu = d_getTrueAnomaly(time[i], e, w, period[j], t_zero[j])
            if loglike_switch:
                RV[2*j]   -= 0.5* ((K1[j]*math.cos(nu + w)              + V0[j]  + dV0[j]*(time[i] - t_zero[j])) - loglike_RV[0,i])**2 / (loglike_err[0,i]**2)
                RV[2*j+1] -= 0.5* ((K2[j]*math.cos(nu + w + math.pi)    + V0[j]  + dV0[j]*(time[i] - t_zero[j])) - loglike_RV[0,i])**2 / (loglike_err[0,i]**2)
            else:
                RV[2*j*len(time) + i]             = K1[j]*math.cos(nu + w)              + V0[j]  + dV0[j]*(time[i] - t_zero[j])
                RV[2*j*len(time) + len(time) + i] = K2[j]*math.cos(nu + w + math.pi)    + V0[j]  + dV0[j]*(time[i] - t_zero[j])

    def rv_gpu_batch(time, t_zero=0.0*np.ones(N_BATCH), period=1.0*np.ones(N_BATCH), K1=10.0*np.ones(N_BATCH), K2=10.0*np.ones(N_BATCH), fs=0.0*np.ones(N_BATCH), fc=0.0*np.ones(N_BATCH), V0=0.0*np.ones(N_BATCH), dV0=0.0*np.ones(N_BATCH), loglike_switch=False, loglike_RV = np.zeros((2,2)), loglike_err=np.zeros((2,2))):
        # Copy over the time array
        d_time = numba.cuda.to_device(time.astype(np.float64))

        # Allocate the RV array depending if we want RV or loglikes
        if loglike_switch : RV = numba.cuda.to_device(np.empty((N_BATCH*2), dtype = np.float64) )
        else              : RV = numba.cuda.to_device(np.empty((N_BATCH*2*len(time)), dtype = np.float64))

        # Copy other parameters over
        d_period = numba.cuda.to_device(period.astype(np.float64))
        d_t_zero = numba.cuda.to_device(t_zero.astype(np.float64))
        d_K1 = numba.cuda.to_device(K1.astype(np.float64))
        d_K2 = numba.cuda.to_device(K2.astype(np.float64))
        d_fs = numba.cuda.to_device(fs.astype(np.float64))
        d_fc = numba.cuda.to_device(fc.astype(np.float64))
        d_V0 = numba.cuda.to_device(V0.astype(np.float64))
        d_dV0 = numba.cuda.to_device(dV0.astype(np.float64))

        # And loglike params
        d_loglike_RV = numba.cuda.to_device(loglike_RV.astype(np.float64))
        d_loglike_err = numba.cuda.to_device(loglike_err.astype(np.float64))

        # Make the call
        d_rv_batch[int(np.ceil(N_BATCH/256.0)), 256](d_time, d_t_zero, d_period, d_K1, d_K2, d_fs, d_fc, d_V0, d_dV0, RV, loglike_switch, d_loglike_RV, d_loglike_err)

        # Return appropriate shape
        if loglike_switch : return RV.copy_to_host().reshape((N_BATCH, 2))
        else              : return RV.copy_to_host().reshape((N_BATCH, 2, len(time)))
           


