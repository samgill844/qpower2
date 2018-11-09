import numba, numba.cuda, math
import numpy as np

#fwhm = (r_1)*sqrt(1-b**2)/np.pi

@numba.njit
def logg2(period, e, K , radius_1, radius_2, b ):
	incl = math.acos(b*radius_1)
	period = period*86400.1 # in seconds
	K = K*1000. # m/s
	g = ((2*math.pi)/period) * (math.sqrt(1-e**2)*K)/(radius_2**2 *(math.sin(incl))) # m/s/s
	return math.log10(100*g)*1.0



@numba.cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])')
def d_logg2(period, e, K, radius_1, radius_2, b, logg2_arr):
    i = numba.cuda.grid(1)
    logg2_arr[i] =  logg2(period[i], e[i], K[i], radius_1[i], radius_2[i], b[i])

def logg2_gpu(period, e, K, radius_1, radius_2, b, return_device_array=False):
    N = len(period)

    # Allocation
    logg2_arr = numba.cuda.to_device(period.astype(np.float64))
    period = numba.cuda.to_device(period.astype(np.float64))
    e = numba.cuda.to_device(e.astype(np.float64))
    K = numba.cuda.to_device(K.astype(np.float64))
    radius_1 = numba.cuda.to_device(radius_1.astype(np.float64))
    radius_2 = numba.cuda.to_device(radius_2.astype(np.float64))
    b = numba.cuda.to_device(b.astype(np.float64))

    # Launch
    d_logg2[int(np.ceil(N/512)),512](period, e, K, radius_1, radius_2, b, logg2_arr)

    if return_device_array : return logg2_arr
    return logg2_arr.copy_to_host()




@numba.njit 
def fwhm(radius_1=0.2, b=0.1) : return (radius_1)*math.sqrt(1-b**2)/math.pi