import numba, numba.cuda
import math 
import pyculib.fft
import numpy as np

@numba.njit('float64(float64,float64,float64)', nogil=True)
def clip(a, b, c):
    if (a < b)   : return b
    elif (a > c) : return c
    else         : return a


@numba.njit('float64(float64,float64,float64)', nogil=True)
def area(z, r1, r2):
    arg1 = clip((z*z + r1*r1 - r2*r2)/(2.*z*r1),-1,1)
    arg2 = clip((z*z + r2*r2 - r1*r1)/(2.*z*r2),-1,1)
    arg3 = clip(max((-z + r1 + r2)*(z + r1 - r2)*(z - r1 + r2)*(z + r1 + r2), 0.),-1,1)

    if   (r1 <= r2 - z) : return math.pi*r1*r1							                              # planet completely overlaps stellar circle
    elif (r1 >= r2 + z) : return math.pi*r2*r2						                                  # stellar circle completely overlaps planet
    else                : return r1*r1*math.acos(arg1) + r2*r2*math.acos(arg2) - 0.5*math.sqrt(arg3)  # partial overlap



####################
# GPU functions
##################
if numba.cuda.is_available():

    # FFT plan
    from pyculib.fft.binding import Plan, CUFFT_C2C
    fftplan17 = Plan.one(CUFFT_C2C, 2**17)
    fftplan18 = Plan.one(CUFFT_C2C, 2**18)


    @numba.cuda.jit('float64(float64,float64,float64)', device=True, inline=True)
    def d_clip(a, b, c):
        if (a < b)   : return b
        elif (a > c) : return c
        else         : return a


    @numba.cuda.jit('float64(float64,float64,float64)', device=True, inline=True)
    def d_area(z, r1, r2):
        arg1 = d_clip((z*z + r1*r1 - r2*r2)/(2.*z*r1),-1,1)
        arg2 = d_clip((z*z + r2*r2 - r1*r1)/(2.*z*r2),-1,1)
        arg3 = d_clip(max((-z + r1 + r2)*(z + r1 - r2)*(z - r1 + r2)*(z + r1 + r2), 0.),-1,1)

        if   (r1 <= r2 - z) : return math.pi*r1*r1							                              # planet completely overlaps stellar circle
        elif (r1 >= r2 + z) : return math.pi*r2*r2						                                  # stellar circle completely overlaps planet
        else                : return r1*r1*math.acos(arg1) + r2*r2*math.acos(arg2) - 0.5*math.sqrt(arg3)  # partial overlap


    @numba.cuda.jit('void(complex64[:], complex64[:])')
    def cuda_mult(data1, data2):
        i = numba.cuda.grid(1)
        data1[i] *= data2[i]

    def fft_convolve(data,  kernel):

        # First pad the kernel
        kernel = np.pad(kernel, (0,len(data) - len(kernel)), 'constant', constant_values=(0,0)).astype(np.float64)

        # Now create the gpu arrays
        gpu_data   = numba.cuda.to_device(  data.astype(np.complex64))  
        gpu_kernel = numba.cuda.to_device(kernel.astype(np.complex64))	

        # now FFT the arrays
        pyculib.fft.fft_inplace(gpu_data)    # implied host->device
        pyculib.fft.fft_inplace(gpu_kernel)  # implied host->device

        # Now multiply out1 and out2, storing all on out2
        cuda_mult[int(np.ceil(len(data)/512.0)), 512](gpu_data, gpu_kernel)  # all on device

        # Now IFFT back 
        pyculib.fft.ifft_inplace(gpu_data)

        return np.real(gpu_data.copy_to_host())

    def fft_convolve18(data,  kernel):
        # First pad the kernel
        kernel = np.pad(kernel, (0,2**18 - len(kernel)), 'constant', constant_values=(0,0)).astype(np.complex64)

        # Now create the gpu arrays
        gpu_data   = numba.cuda.to_device(  data.astype(np.complex64))  
        gpu_kernel = numba.cuda.to_device(  kernel                   )	

        # now FFT the arrays
        fftplan18.forward(gpu_data  , gpu_data  )
        fftplan18.forward(gpu_kernel, gpu_kernel)

        # Now multiply out1 and out2, storing all on out2
        cuda_mult[int(np.ceil(len(data)/512.0)), 512](gpu_data, gpu_kernel)  # all on device

        # Now IFFT back 
        fftplan18.inverse(gpu_data  , gpu_data  )

        np.real(gpu_data.copy_to_host())

    def fft_convolve18t(gpu_data,  gpu_kernel):
        # now FFT the arrays
        fftplan18.forward(gpu_data  , gpu_data  )
        fftplan18.forward(gpu_kernel, gpu_kernel)

        # Now multiply out1 and out2, storing all on out2
        cuda_mult[int(np.ceil(2**18/256.0)), 256](gpu_data, gpu_kernel)  # all on device

        # Now IFFT back 
        fftplan18.inverse(gpu_data  , gpu_data  )

        return gpu_data







@numba.jit('void(float64[:],float64[:],float64[:],float64[:],float64[:],)')
def interpolate(x,y, temp,xref,yref):
    # Here x and xref are the shape length
    # temp is an array of 100 elements
    # This is to temperely hold interpolated values
    n_start = 0
    for i in range(100):
        for j in range(n_start, len(xref)):
            if x[i]==xref[j] : temp[i] = xref[j]
            if (x[i] > xref[j]) and (x[i] < xref[j+1]) : temp[i] = yref[j] + ((x[i] - xref[j]) / (xref[j+1] - xref[j])) * (yref[j+1] - yref[j]) 

