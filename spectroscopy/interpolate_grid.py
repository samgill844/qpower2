import numpy as np 
import numba 
import numba.cuda 
import math


@numba.cuda.jit('void(int32[:,:,:],int32[:], float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64,float64,float64,float64[:,:],float64[:,:])')
def _GPU_interpolate_spectra(indexes, indexes2, Teff,MH,Logg, TT,MM,LL,  Tstep,Mstep,Lstep, spectra, grid):
    idx = numba.cuda.grid(1)
    if idx < len(spectra):
        for i in range(len(TT)):
            if (TT[i] < Teff[idx]) and (TT[i+1] > Teff[idx]) : indexes[idx,0,0], indexes[idx,0,1] = i, i+1   ;break
            elif (Teff[idx] == TT[i])                        : indexes[idx,0,0], indexes[idx,0,1] = i, i     ;break
            elif (Teff[idx] == TT[i+1])                      : indexes[idx,0,0], indexes[idx,0,1] = i+1, i+1 ;break

        for i in range(len(MM)):
            if (MM[i] < MH[idx]) and (MM[i+1] > MH[idx]) : indexes[idx,1,0], indexes[idx,1,1] = i, i+1   ;break
            elif (MH[idx] == MM[i])                      : indexes[idx,1,0], indexes[idx,1,1] = i, i     ;break
            elif (MH[idx] == MM[i+1])                    : indexes[idx,1,0], indexes[idx,1,1] = i+1, i+1 ;break

        for i in range(len(LL)):
            if (LL[i] < Logg[idx]) and (LL[i+1] > Logg[idx]) : indexes[idx,2,0], indexes[idx,2,1] = i, i+1   ;break
            elif (Logg[idx] == LL[i])                        : indexes[idx,2,0], indexes[idx,2,1] = i, i     ;break
            elif (Logg[idx] == LL[i+1])                      : indexes[idx,2,0], indexes[idx,2,1] = i+1, i+1 ;break


        
        # Calculate xd, yd, zd
        if indexes[idx,0,0]==indexes[idx,0,1]: xd = 1.0
        else:                                  xd = (Teff[idx] - TT[indexes[idx,0,0]])/(TT[indexes[idx,0,1]]-TT[indexes[idx,0,0]])
        if indexes[idx,1,0]==indexes[idx,1,1]: yd = 1.0
        else:                                  yd = (MH[idx] - MM[indexes[idx,1,0]])/(MM[indexes[idx,1,1]]-MM[indexes[idx,1,0]])
        if indexes[idx,2,0]==indexes[idx,2,1]: zd = 1.0
        else:                                  zd = (Logg[idx] - LL[indexes[idx,2,0]])/(TT[indexes[idx,2,1]]-TT[indexes[idx,2,0]])
        
        # Now convert this to model index
        count = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    indexes2[count] = indexes[idx,0,i]*len(MM)*len(LL) + indexes[idx,1,j]*len(LL) + indexes[idx,3,k]
                    count += 1
                    
        # Now we have the models it is time for tri-linear interpolation
        
        for i in range(10):
            spectra[idx,i] = grid[indexes2[0],i] *(1.-xd)*(1.-yd)*(1.-zd) + \
                grid[indexes2[4],i]*xd*(1.-yd)*(1-zd) + \
                grid[indexes2[2],i]*(1.-xd)*yd*(1-zd) + \
                grid[indexes2[1],i]*(1.-xd)*(1-yd)*zd + \
                grid[indexes2[5],i]*xd*(1-yd)*zd + \
                grid[indexes2[3],i]*(1.-xd)*yd*zd + \
                grid[indexes2[6],i]*xd*yd*(1.-zd) + \
                grid[indexes2[7],i]*xd*yd*zd
    



def GPU_interpolate_spectra(data, Teff, MH, Logg):
    d_Teff = numba.cuda.to_device(Teff.astype(np.float64))
    d_MH = numba.cuda.to_device(MH.astype(np.float64))
    d_Logg = numba.cuda.to_device(Logg.astype(np.float64))
    
    # Interplate the spectra
    _GPU_interpolate_spectra[int(np.ceil(data['N_spectra_gpu']/512.0)), 512](data['d_spectra_indexes'], data['d_spectra_indexes2'], d_Teff,d_MH,d_Logg, data['d_TT'], data['d_MM'], data['d_LL'],  data['Tstep'], data['Mstep'], data['Lstep'], data['d_spectra'], data['d_grid'])
