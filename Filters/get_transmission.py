import numpy as np, os, sys
import qpower2
this_dir, this_filename = os.path.split(qpower2.__file__)



def get_transmission(band='Kp', custom=None):

	#################
	# Kepler/K2
	#################
	if band=='Kp':
		file_name = this_dir+'/Filters/Kepler/K_response.txt'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1])
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)
		
	#################
	# SDSS
	#################
	if band=='SDSS_g':
		file_name = this_dir+'/Filters/SDSS/g_SDSS.res'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1])
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)

	if band=='SDSS_i':
		file_name = this_dir+'/Filters/SDSS/i_SDSS.res'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1])
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)

	if band=='SDSS_r':
		file_name = this_dir+'/Filters/SDSS/r_SDSS.res'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1])
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)
		
	if band=='SDSS_u':
		file_name = this_dir+'/Filters/SDSS/u_SDSS.res'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1])
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)
		
	if band=='SDSS_z':
		file_name = this_dir+'/Filters/SDSS/z_SDSS.res'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1])
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)
		


	#################
	# Johnsons Cousins
	#################
	if band=='Johnson_B':
		file_name = this_dir+'/Filters/Johnson-Cousins/nBessel_B-1.txt'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1])
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)
		
	if band=='Johnson_I':
		file_name = this_dir+'/Filters/Johnson-Cousins/nBessel_I-1.txt'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1])
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)
		
	if band=='Johnson_R':
		file_name = this_dir+'/Filters/Johnson-Cousins/nBessel_R-1.txt'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1])
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)
		
	if band=='Johnson_U':
		file_name = this_dir+'/Filters/Johnson-Cousins/nBessel_U-1.txt'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1])
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)
		
	if band=='Johnson_V':
		file_name = this_dir+'/Filters/Johnson-Cousins/nBessel_V-1.txt'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1])
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)
		


	if band=='J':
		file_name = this_dir+'/Filters/2MASS/J_2mass.res'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1]/10)
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)
		
	if band=='H':
		file_name = this_dir+'/Filters/2MASS/H_2mass.res'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1]/10)
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)
		
	if band=='K':
		file_name = this_dir+'/Filters/2MASS/Ks_2mass.res'
		#return create_spectrum_structure(waveobs=np.loadtxt(file_name).T[0],flux=np.loadtxt(file_name).T[1]/10)
		data = np.loadtxt(file_name)
		return np.array([ data.T[0], data.T[1] ], dtype = np.float64)
		
		

