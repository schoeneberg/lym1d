from cobaya.likelihood import Likelihood
from .wrapper import lym1d_wrapper
import os
import numpy as np

class cobaya_wrapper(Likelihood):

  speed = 30
  # DEFAULT parameters

  base_directory = "/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/"
  #models_path = "nyx_files/models_Nyx_Oct2023.hdf5"
  runmode = "nyx_auvb"
  #has_cor = 'None'
  #emupath ="nyx_files/lym1d_full_emulator_Oct2023_LP.npz"
  #data_path = "data_files/Chabanier19/"
  arguments = {'has_cor':None}

  @classmethod
  def getDefaultParameterBasis(cls):
    pars = {}
    N_redshift_DEFAULT = 13
    for i in range(N_redshift_DEFAULT):
      pars['normalization{}'.format(i+1)]=1
    for i in range(N_redshift_DEFAULT):
      pars['noise{}'.format(i+1)]={'prior':{'min':-1,'max':1},'ref':{'min':-1e-2,'max':1e-2}}

    pars.update({
         'A_lya':{'prior':{'min':7,'max':9},'ref':{'min':7.9,'max':8}},
         'n_lya':{'prior':{'min':-2.47142933875679,'max':-2.2485707267925283},'ref':{'min':-2.4,'max':-2.3}},
         'AmpTauEff':{'prior':{'min':0,'max':1.5},'ref':{'min':0.4,'max':0.44}},#0.42
         'SlopeTauEffInf':{'prior':{'min':0,'max':7},'ref':{'min':3.5,'max':4.0}},#3.8
         'SlopeTauEffBreak':0,
         'T0':{'prior':{'min':0,'max':25000},'ref':{'min':6000,'max':10000}},#8000
         'T0SlopeInf':{'prior':{'min':-5,'max':2},'ref':{'min':-2,'max':-1.5}},#8000
         'T0SlopeBreak':{'prior':{'min':-10,'max':10},'ref':{'min':0,'max':0.1}},#8000
         'gamma':{'prior':{'min':0.3,'max':2.0},'ref':{'min':1.1,'max':1.5}},#1.3
         'gammaSlopeInf':{'prior':{'min':-5,'max':2},'ref':{'min':0,'max':0.1}},#0
         'gammaSlopeBreak':0,
         'A_UVB':{'prior':{'min':0,'max':2},'ref':{'min':0.9,'max':1.1}},
         'Lya_DLA':{'prior':{'min':0,'max':1}},
         'Lya_SN':{'prior':{'min':0,'max':1}},
         'Lya_AGN':{'prior':{'min':0,'max':1}},
         'Lya_UVFluct':0,
         'fSiIII':{'prior':{'min':-1e-2,'max':1e-2}},
         'fSiII':{'prior':{'min':-1e-2,'max':1e-2}},
         'ResoAmpl':0,
         'ResoSlope':0
         })
    return pars

  def initialize(self):

    print("[lym1d_cobaya_wrapper] Beginning likelihood initialization")
    self.wrapper = lym1d_wrapper(
        runmode=self.runmode,
        base_directory = self.base_directory,
        **self.arguments
        )
    nuisance = self.wrapper.nuisance_parameters
    print("[lym1d_cobaya_wrapper] nuisance parameters: ",nuisance)
    #print("SELF PARAMS = ",self.params)
    #quit()
    self.sampled_params = list(self.params.keys())
    self.sampled_params.extend(nuisance)
    print("[lym1d_cobaya_wrapper] Likelihood initialized")

  def get_can_support_params(self):
    return self.sampled_params

  def get_requirements(self):
    return {'Hubble':{'z':0},'Omega_m':None,'z_reio':None,'Pk_interpolator':{'z':[0,10],'k_max':10,'nonlinear':False}}

  def logp(self, **params):
    #print("params = ",params)

    cosmo = self.provider.requirement_providers['Hubble'].classy

    try:
      chi2 = self.wrapper.chi2(cosmo,params)
    except ValueError as ve:
      print("[lym1d_cobaya_wrapper] : Encountered value error : ",ve)
      raise #return -np.inf
    except KeyError as ke:
      print("[lym1d_cobaya_wrapper] : Encountered missing key : ",ke)
      raise
    if chi2==None:
      return -np.inf
    else:
      return -0.5*chi2

# VERY important line to correclty setup cobaya object (since cobaya inspects the CLASS, not the instance (!), therefore the class needs to be modified)
# Also, since the code for setting up the default parameter basis is not 100% trivial, we need to do it in a function
cobaya_wrapper.params = cobaya_wrapper.getDefaultParameterBasis()
