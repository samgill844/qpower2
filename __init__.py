try:
    import numba.cuda
    print('CPU acceleration              [\033[92m   OK' +'\033[0;37;0m   ]')
except:
    print('CPU acceleration              [\033[31m  FAIL' +'\033[0;37;0m  ]')
    pass

if numba.cuda.is_available(): print('GPU acceleration              [\033[92m   OK' +'\033[0;37;0m   ]')
else: print('GPU acceleration              [\033[31m  FAIL' +'\033[0;37;0m  ]')


# Import lighcurve modules
import qpower2.binarystar

# Import emceegpu
import qpower2.emcee_gpu

# import photometric relations
import qpower2.photometry

# Import spectral analysis
from .spectroscopy import * 

# Import filters
import qpower2.Filters

# Import utils
import qpower2.Utilities
