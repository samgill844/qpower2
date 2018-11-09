import numba, numba.cuda
import math 
import numpy as np

from qpower2.binarystar.utils import *
from qpower2.binarystar.kepler import *
from qpower2.binarystar.qpower2 import *
from qpower2.binarystar.uniform import *

N_BATCH = 10240

__all__ = ['lc', 'lc_gpu', 'lc_gpu_batch']


#@numba.njit('float64[:](float64[:],float64,float64,float64,float64,float64,float64,float64,float64[:],float64)', nogil=True,parallel=True)
@numba.njit(cache=False, nogil=True,parallel=False)
def lc(time, radius_1=0.2, k=0.2, fs=0.0, fc=0.0, incl=90., period=1.0, t_zero=0.0, ldc_1=np.array([0.8,0.8]), SBR = 0.0, light_3 = 0.0, loglike_switch=False, loglike_flux = np.array([0.8]), loglike_err = np.array([0.8]) ):
    # Unpack args
    w = math.atan2(fs, fc)
    e = fs*fs + fc*fc
    incl = math.pi*incl/180

    # Allocate the LC array
    LC = np.ones(len(time))

    for i in numba.prange(len(time)):
        # Get true anomaly
        nu = getTrueAnomaly(time[i], e, w, period, t_zero)

        # Get projected seperation
        z = get_z(e, incl, nu, w, radius_1)

        # At this point, we might check if the distance between
        if (z > (1.0+ k)) and not loglike_switch : continue

        # So it's eclipsing? Lets see if its primary or secondary by
        # its projected motion!
        f = getProjectedPosition(nu, w, incl)

        if (f > 0): 
            # Calculate the flux drop for a priamry eclipse
            LC[i] =  Flux_drop_analytical_power_2(z, k, ldc_1[0], ldc_1[1], LC[i], 1e-5)    # Primary eclipse

            # Dont forget about the third light from the companion if SBR > 0
            if (SBR > 0) : LC[i] = LC[i]/(1. + k*k*SBR) + (1.-1.0/(1 + k*k*SBR))

        elif (SBR>0) : LC[i] =  Flux_drop_analytical_uniform(z, k, SBR, LC[i]) # Secondary eclipse

        # Now account for third light
        if (light_3 > 0.0) : LC[i] = LC[i]/(1. + light_3) + (1.-1.0/(1. + light_3))

        # Now check if the user wants the log-likliehood returned
        if loglike_switch : LC[i] = -0.5*(LC[i] - loglike_flux[i])**2 / (loglike_err[i]**2)
    return LC






####################
# GPU functions
##################
if numba.cuda.is_available():
    @numba.cuda.jit('void(float64[:], float64, float64, float64, float64, float64, float64, float64, float64[:], float64, float64,b1, float64[:], float64[:], float64[:], float64[:])')
    def d_lc(time, radius_1, k, fs, fc, incl, period, t_zero, ldc_1, SBR, light_3, loglike_switch, loglike_flux, loglike_err, loglike_result, LC):
        # Unpack args
        w = math.atan2(fs, fc)
        e = fs*fs + fc*fc
        incl = math.pi*incl/180

        # Get index
        i = numba.cuda.grid(1)

        # Get true anomaly
        nu = d_getTrueAnomaly(time[i], e, w, period, t_zero)

        # Get projected seperation
        z = d_get_z(e, incl, nu, w, radius_1)
        # At this point, we might check if the distance between
        #if (z > (1.0+ k)) and not loglike_switch : continue

        # So it's eclipsing? Lets see if its primary or secondary by
        # its projected motion!
        f = d_getProjectedPosition(nu, w, incl)

        if (f > 0): 
            # Calculate the flux drop for a priamry eclipse
            LC[i] =  d_Flux_drop_analytical_power_2(z, k, ldc_1[0], ldc_1[1], LC[i], 1e-5)    # Primary eclipse

            # Dont forget about the third light from the companion if SBR > 0
            if (SBR > 0) : LC[i] = LC[i]/(1. + k*k*SBR) + (1.-1.0/(1 + k*k*SBR))

        elif (SBR>0) : LC[i] =  d_Flux_drop_analytical_uniform(z, k, SBR, LC[i]) # Secondary eclipse

        # Now account for third light
        if (light_3 > 0.0) : LC[i] = LC[i]/(1. + light_3) + (1.-1.0/(1. + light_3))

        # Now check if the user wants the log-likliehood returned
        if loglike_switch :
            LC[i] = -0.5*(LC[i] - loglike_flux[i])**2 / (loglike_err[i]**2)
            numba.cuda.atomic.add(loglike_result, 0, LC[i])
            numba.cuda.syncthreads()
            

    def lc_gpu(time, radius_1 = 0.2, k = 0.2, fs = 0.0, fc = 0.0, incl = 90., period = 1.0, t_zero=0.0, ldc_1=np.array([0.8,0.8]), SBR = 0.0, light_3 = 0.0, loglike_switch=False, loglike_flux = np.array([1.0]), loglike_err = np.array([1.0]) ):
        # First put time as gpu array
        d_time = numba.cuda.to_device(time.astype(np.float64))

        # Now put ldc_1 on gpu array
        d_ldc_1 = numba.cuda.to_device(ldc_1.astype(np.float64))

        # Now create LC array
        d_LC = numba.cuda.to_device(np.ones(len(time)).astype(np.float64))

        # Now create the loglike arrays
        d_loglike_flux = numba.cuda.to_device(loglike_flux)
        d_loglike_err = numba.cuda.to_device(loglike_err)
        d_loglike_result = numba.cuda.to_device(np.array([0], dtype=np.float64))

        # Call the kernel
        d_lc[int(np.ceil(len(time)/256.0)), 256](d_time, radius_1, k, fs, fc, incl, period, t_zero, d_ldc_1, SBR, light_3, loglike_switch, d_loglike_flux, d_loglike_err,d_loglike_result, d_LC)
        if loglike_switch : return d_loglike_result.copy_to_host()
        else : return d_LC.copy_to_host()



    @numba.cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], b1, float64[:], float64[:], float64[:], float64[:], float64[:])')
    def d_lc_batch(time, radius_1, k, fs, fc, incl, period, t_zero, ldc_1, SBR, light_3, loglike_switch, loglike_flux, loglike_err, loglike_jitter, loglike_zp, LC):
        # Get model index
        j = numba.cuda.grid(1)

        # Unpack args
        w = math.atan2(fs[j], fc[j])
        e = fs[j]*fs[j] + fc[j]*fc[j]
        incl[j] = math.pi*incl[j]/180

        
        for i in range(len(time)):
            # Get true anomaly
            nu = d_getTrueAnomaly(time[i], e, w, period[j], t_zero[j])
        
            # Get projected seperation
            z = d_get_z(e, incl[j], nu, w, radius_1[j])
            # At this point, we might check if the distance between
            #if (z > (1.0+ k)) and not loglike_switch : continue
            
            # So it's eclipsing? Lets see if its primary or secondary by
            # its projected motion!
            f = d_getProjectedPosition(nu, w, incl[j])
            l=1.
            if (f > 0): 
                # Calculate the flux drop for a priamry eclipse
                l =  d_Flux_drop_analytical_power_2(z, k[j], ldc_1[j*2], ldc_1[j*2+1], 1.0, 1e-5)    # Primary eclipse
        
                # Dont forget about the third light from the companion if SBR > 0
                if (SBR[j] > 0) : l = l/(1. + k[j]*k[j]*SBR[j]) + (1.-1.0/(1 + k[j]*k[j]*SBR[j]))
                    
            elif (SBR[j]>0) : l =  d_Flux_drop_analytical_uniform(z, k[j], SBR[j], l) # Secondary eclipse
        
            # Now account for third light
            if (light_3[j] > 0.0) : l = l/(1. + light_3[j]) + (1.-1.0/(1. + light_3[j]))
        
            # Now check if the user wants the log-likliehood returned
            if loglike_switch: 
                wt = 1.0 / (loglike_jitter[j]**2 + loglike_err[i]**2)
                LC[j] -= 0.5*(( ( loglike_zp[j]-2.5*math.log10(l)) - loglike_flux[i])**2*wt - math.log(wt) )
            else              : LC[j*len(time) + i] = l
            
        
        

    def lc_gpu_batch(time, radius_1 = 0.2*np.ones(N_BATCH), k = 0.2*np.ones(N_BATCH), fs = 0.0*np.ones(N_BATCH), fc = 0.0*np.ones(N_BATCH), incl = 90.*np.ones(N_BATCH), period = 1.0*np.ones(N_BATCH), t_zero=0.0*np.ones(N_BATCH), ldc_1=0.8*np.ones(2*N_BATCH), SBR = 0.0*np.ones(N_BATCH), light_3 = 0.0*np.ones(N_BATCH), loglike_switch=False, loglike_flux = np.array([1.0]), loglike_err = np.array([1.0]), loglike_jitter = np.ones(N_BATCH), loglike_zp = np.zeros(N_BATCH)):
        # First put time as gpu array
        d_time = numba.cuda.to_device(time.astype(np.float64))

        # Now put all the other parameters on the gpu
        d_radius_1 = numba.cuda.to_device(radius_1.astype(np.float64))
        d_k = numba.cuda.to_device(k.astype(np.float64))
        d_fs = numba.cuda.to_device(fs.astype(np.float64))
        d_fc = numba.cuda.to_device(fc.astype(np.float64))
        d_incl = numba.cuda.to_device(incl.astype(np.float64))
        d_period = numba.cuda.to_device(period.astype(np.float64))
        d_t_zero = numba.cuda.to_device(t_zero.astype(np.float64))
        d_ldc_1 = numba.cuda.to_device(ldc_1.astype(np.float64))
        d_SBR = numba.cuda.to_device(SBR.astype(np.float64))
        d_light_3 = numba.cuda.to_device(light_3.astype(np.float64))
        d_loglike_jitter = numba.cuda.to_device(loglike_jitter.astype(np.float64))
        d_loglike_zp = numba.cuda.to_device(loglike_zp.astype(np.float64))

        # Now create LC array depending if we want the lightcurves back or just the loglikes
        if loglike_switch : d_LC = numba.cuda.to_device(np.zeros(len(radius_1)).astype(np.float64))
        else :              d_LC = numba.cuda.to_device(np.ones(len(time)*len(radius_1)).astype(np.float64))

        # Now create the loglike arrays
        d_loglike_flux = numba.cuda.to_device(loglike_flux)
        d_loglike_err = numba.cuda.to_device(loglike_err)

        # Call the kernel
        d_lc_batch[int(np.ceil(len(radius_1)/256.0)), 256](d_time, d_radius_1, d_k, d_fs, d_fc, d_incl, d_period, d_t_zero, d_ldc_1, d_SBR, d_light_3, loglike_switch, d_loglike_flux, d_loglike_err, d_loglike_jitter, d_loglike_zp, d_LC)

        # Make the retrun
        if loglike_switch : return d_LC.copy_to_host()
        else : return d_LC.copy_to_host().reshape((len(radius_1), len(time)))







    







