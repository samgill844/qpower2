import numba, numba.cuda
import math 


@numba.njit('float64(float64,float64,float64,float64)', nogil=True)
def Flux_drop_analytical_uniform( z, k, SBR, f):
		if(z >= 1. + k) : return f		                  # no overlap
		if(z >= 1.) and ( z <= k - 1.) : return 0.0;      # total eclipse of the star
		elif (z <= 1. - k) : return f - SBR*k*k	          # planet is fully in transit		
		else:						                      # planet is crossing the limb
			kap1 = math.acos(min((1. - k*k + z*z)/2./z, 1.))
			kap0 = math.acos(min((k*k + z*z - 1.)/2./k/z, 1.))
			return f - SBR*  (k*k*kap0 + kap1 - 0.5*math.sqrt(max(4.*z*z - math.pow(1. + z*z - k*k, 2.), 0.)))/math.pi







####################
# GPU functions
##################
if numba.cuda.is_available():
    @numba.cuda.jit('float64(float64,float64,float64,float64)', device=True, inline=True)
    def d_Flux_drop_analytical_uniform( z, k, SBR, f):
        if(z >= 1. + k) : return f		                  # no overlap
        if(z >= 1.) and ( z <= k - 1.) : return 0.0;      # total eclipse of the star
        elif (z <= 1. - k) : return f - SBR*k*k	          # planet is fully in transit		
        else:						                      # planet is crossing the limb
            kap1 = math.acos(min((1. - k*k + z*z)/2./z, 1.))
            kap0 = math.acos(min((k*k + z*z - 1.)/2./k/z, 1.))
            return f - SBR*  (k*k*kap0 + kap1 - 0.5*math.sqrt(max(4.*z*z - math.pow(1. + z*z - k*k, 2.), 0.)))/math.pi
