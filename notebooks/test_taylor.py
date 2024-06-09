import lym1d_wrapper
import numpy as np
import os

# First, get relative directory of public data for Taylor emulator
path = os.path.abspath(os.path.join(os.path.basename(__file__),"..","public_data"))

# A very simple testing script to check that your Taylor emulator works correctly
ops ={}
ops.update({"base_directory":path})

# Additional options (like priors, and nuisance corrections)
ops.update({'H0prior' : {'mean':67.3,'sigma':1.0}, 'use_thermal_prior':True,'gammaPriorMean':1.3,
 'splice_kind':2})
ops.update({'has_cor' :{'noise': True, 'DLA': False,'reso': True, 'SN': True, 'AGN': True, 'zreio':False,'SiIII':True,'SiII':True,'norm':True,'UV':True,'splice':True}})
# Creating the wrapper object !
wrap = lym1d_wrapper.lym1d_wrapper("taylor_splicing",**ops)

print(wrap) #Should be a lym1d_wrapper.wrapper.lym1d_wrapper object

# Execute with parameters that the wrapper knows (bestfit) for which we already know the answer of the likelihood
pars = {'omega_cdm': 0.1019436, 'H0': 67.79706, 'ln10^{10}A_s': 3.251898, 'n_s': 0.9538248, 'M_tot': 0.007674836, 'T0': 7473.512, 'gamma': 1.026419, 'AmpTauEff': 0.4833614, 'SlopeTauEffInf': 3.840216, 'fSiIII': 0.005987892, 'fSiII': 0.0004267999, 'T0SlopeInf': -4.240167, 'T0SlopeBreak': -0.01271859, 'gammaSlopeInf': 0.9863761, 'SplicingCorr': 0.07823827, 'SplicingOffset': 0.1064135, 'Lya_AGN': 0.363703, 'Lya_SN': 1.107606, 'Lya_UVFluct': 0.05972924, 'noise1': 0.001810679, 'noise2': -0.009112814, 'noise3': -0.03098166, 'noise4': -0.04527296, 'noise5': -0.0629536, 'noise6': -0.02185368, 'noise7': -0.01260106, 'noise8': 0.0156841, 'noise9': 0.01728685, 'noise10': 0.02316432, 'noise11': 0.03911431, 'noise12': 0.007262989, 'noise13': 0.001428109}

# Additionally, ignore other parameters that we do not care about
pars.update({'gammaSlopeBreak':0,'SlopeTauEffBreak':0,'Lya_DLA':0,'ResoAmpl':0,'ResoSlope':0})
for i in range(13):
  pars.update({'normalization{}'.format(i+1) :1})
  pars.update({'tauError{}'.format(i+1) :0})

# Now, start a classy run , passing the cosmo arguments and the required precision+setting parameters
import classy
cosmo = classy.Class()
cosmo.set(wrap.need_cosmo_arguments)
print("need_cosmo_arugments = {} ".format(wrap.need_cosmo_arguments))

cosmopars = {'k_pivot': 0.05, 'N_ur': 0.00641, 'N_ncdm': 1, 'T_ncdm': 0.71611, 'deg_ncdm': 3, 'output': 'mPk', 'P_k_max_1/Mpc': 10.0, 'z_max_pk': 6, 'omega_b': 0.022253, 'omega_cdm': 0.1019436, 'H0': 67.79706, 'n_s': 0.9538248, 'tau_reio': 0.0561, 'A_s': 2.5839336465371342e-09, 'm_ncdm': 0.0025582786666666665}

cosmo.set(cosmopars)
cosmo.compute()

# After cosmo is computed, let's use the wrapper to find the chi2
chi2 = wrap.chi2(cosmo,pars)
print("{:.3f}".format(0.5*chi2))
assert(np.abs(chi2-459.2714013432472)<0.01)

