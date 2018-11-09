import numpy as np 
from astropy.io import fits 
import numba, numba.cuda




def preliminary(table):
    # Extract TMLV arrays
    T,M,L  = table.T
    # extract grid points
    TT,MM,LL = np.sort(np.unique(T)).astype(np.float64),  np.sort(np.unique(M)).astype(np.float64), np.sort(np.unique(L)).astype(np.float64)

    # Now get step size
    Tstep = TT[1]-TT[0]
    Mstep= MM[1]-MM[0]
    Lstep = LL[1]-LL[0]

    return TT, MM, LL, Tstep, Mstep, Lstep


def uniform_space(wavelength):
    lnwave_ = np.linspace(np.log(wavelength[0]),np.log(wavelength[-1]),len(wavelength), dtype=np.float64)
    velocity_step = (np.exp(lnwave_[1]-lnwave_[0])-1)*299792.458
    return lnwave_, velocity_step



def rec_array_to_array(x):
    y = np.zeros((len(x),3))
    for i in range(len(x)):
        for j in range(3):
            y[i,j] = float(x[i][j])
    return y.astype(np.float64)


def load_grid(path_to_grid, gpu_accelleration=False, N_spectra_gpu = 500, PHEONIX_DOWNSAMPLED=False):
    h = fits.open(path_to_grid)
    table = rec_array_to_array(h[1].data)
    #spectra = np.copy(h[0].data).astype(np.float64)
    spectra = h[0].data
    if PHEONIX_DOWNSAMPLED : wavelength = np.linspace(350, 900, 1024).astype(np.float64)
    else : wavelength = np.linspace(350,900,2**18).astype(np.float64)
    lnwave, velocity_step = uniform_space(wavelength)
    TT,MM,LL,Tstep,Mstep,Lstep = preliminary(table)
    xxx = dict()
    xxx['grid'] = spectra
    xxx['wavelength'] = wavelength
    xxx['lnwave'] = lnwave
    xxx['logwave'] = np.log(wavelength)
    xxx['explnwave'] = np.exp(lnwave)
    xxx['velocity_step'] = velocity_step
    xxx['TT'] = TT
    xxx['MM'] = MM
    xxx['LL'] = LL
    xxx['Tstep'] = Tstep
    xxx['Mstep'] = Mstep
    xxx['Lstep'] = Lstep

    if not numba.cuda.is_available() and gpu_accelleration : print('GPU acceleration not available')
    if gpu_accelleration and numba.cuda.is_available():
        xxx['d_grid'] = numba.cuda.to_device(  spectra.astype(np.float64))  
        xxx['d_wavelength'] = numba.cuda.to_device(  wavelength.astype(np.float64))  
        xxx['d_lnwave'] = numba.cuda.to_device(  lnwave.astype(np.float64)) 
        xxx['d_logwave'] = numba.cuda.to_device(  np.log(wavelength).astype(np.float64))  
        xxx['d_explnwave'] = numba.cuda.to_device(  np.exp(lnwave).astype(np.float64))   
        xxx['d_TT'] = numba.cuda.to_device(  TT.astype(np.float64))
        xxx['d_MM'] = numba.cuda.to_device(  MM.astype(np.float64))
        xxx['d_LL'] = numba.cuda.to_device(  LL.astype(np.float64)) 
        xxx['d_spectra'] = numba.cuda.to_device(  np.empty((N_spectra_gpu, len(wavelength))).astype(np.float64) )  
        xxx['N_spectra_gpu'] = N_spectra_gpu
        xxx['d_spectra_indexes'] = numba.cuda.to_device(  np.empty((N_spectra_gpu, 3, 2)).astype(np.int32)) 
        xxx['d_spectra_indexes2'] = numba.cuda.to_device(  np.empty(8).astype(np.int32)) 
        

    return xxx


