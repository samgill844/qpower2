import numba, numba.cuda
import math 
import pyculib.fft
import numpy as np




if numba.cuda.is_available():
    # CUDA arithmetic
    ###################################################
    @numba.cuda.jit('void(complex64[:], complex64[:])')
    def cuda_add(data1, data2):
        i = numba.cuda.grid(1)
        data1[i] += data2[i]

    @numba.cuda.jit('void(complex64[:], complex64[:])')
    def cuda_sub(data1, data2):
        i = numba.cuda.grid(1)
        data1[i] -= data2[i]

    @numba.cuda.jit('void(complex64[:], complex64[:])')
    def cuda_mult(data1, data2):
        i = numba.cuda.grid(1)
        data1[i] *= data2[i]

    @numba.cuda.jit('void(complex64[:], complex64[:])')
    def cuda_divide(data1, data2):
        i = numba.cuda.grid(1)
        data1[i] /= data2[i]    