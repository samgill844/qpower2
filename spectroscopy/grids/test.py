from qpower2.spectroscopy.load_grid import load_grid
from qpower2.spectroscopy.interpolate_grid import GPU_interpolate_spectra
import numpy as np
T = np.random.normal(6000,200,10240);M = np.random.normal(0,0.01,10240);L = np.random.normal(4.44, 0.01, 10240)


grid1 = load_grid('PHEONIX_downsampled.fits', PHEONIX_DOWNSAMPLED=True, gpu_accelleration=True, N_spectra_gpu=10240) 

GPU_interpolate_spectra(grid1, T, M, L)

