from cobaya.likelihood import Likelihood
from .wrapper import lym1d_wrapper
import os
import numpy as np
from scipy.interpolate import CubicSpline

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
  # These are just DEFAULT option, feel free to overwrite with your own!!
  params = {'T0':{'prior':{'min':0,'max':25000},'ref':{'min':6000,'max':10000}},#8000
            'T0SlopeInf':{'prior':{'min':-5,'max':2},'ref':{'min':-2,'max':-1.5}},#8000
            'T0SlopeBreak':{'prior':{'min':-10,'max':10},'ref':{'min':0,'max':0.1}},#8000
            'gamma':{'prior':{'min':0.3,'max':2.0},'ref':{'min':1.1,'max':1.5}},#1.3
            'gammaSlopeInf':{'prior':{'min':-5,'max':2},'ref':{'min':0,'max':0.1}},#0}
            'gammaSlopeBreak':0,
            'AmpTauEff':{'prior':{'min':0,'max':1.5},'ref':{'min':0.4,'max':0.44}},#0.42
            'SlopeTauEffInf':{'prior':{'min':0,'max':7},'ref':{'min':3.5,'max':4.0}},#3.8
            'SlopeTauEffBreak':0,
            }
  to_check = True

  def initialize(self):

    print("[lym1d_cobaya_wrapper] Beginning likelihood initialization")
    self.wrapper = lym1d_wrapper(
        runmode=self.runmode,
        base_directory = self.base_directory,
        **(self.arguments.copy())
        )
    nuisance = self.wrapper.nuisance_parameters
    print("[lym1d_cobaya_wrapper] nuisance parameters: ",nuisance)
    #print("SELF PARAMS = ",self.params)
    #quit()
    self.sampled_params = list(self.params.keys())
    self.sampled_params.extend(nuisance)
    self.sampled_params = np.array(self.sampled_params,dtype=str)
    print("[lym1d_cobaya_wrapper] Likelihood initialized")

  def get_can_support_params(self):
    return self.sampled_params

  def get_requirements(self):
    self.zs = np.linspace(0,10,num=100)
    # Just some garbage to satisfy cobaya's policy of not computing anything unless specifically tasked to do so
    replaced = self.wrapper.replace_with_nuisance
    dic = {'Hubble':{'z':self.zs},'Omega_m':None}

    if self.wrapper.needs_cosmo_pk:
      dic.update({'Pk_interpolator':{'z':[0,10],'k_max':10,'nonlinear':False}})
    if not replaced.get('sigma8',True):
      dic.update({'sigma8_z':{'z':[0]}})
    #if not replaced.get('mnu',True):
    dic.update({'Omega_nu_massive':{'z':[0]}})
    if not replaced.get('zreio',True):
      dic.update({'z_reio':None})
    return dic

  def logp(self, **params):

    ##print("comso = ",cosmo.get_current_derived_parameters(['A_s'])['A_s'], cosmo.n_s(), cosmo.Omega_m(), cosmo.h(), cosmo.sigma8())

    replaced = self.wrapper.replace_with_nuisance

    class Container(object):
      pass
    FakeCosmo = Container()

    Omega_m = self.provider.get_param('Omega_m')
    FakeCosmo.Omega_m = lambda : Omega_m

    h = self.provider.get_param('h')
    FakeCosmo.h = lambda : h

    Hubble = CubicSpline(self.zs,self.provider.get_Hubble(self.zs))
    FakeCosmo.Hubble = lambda : Hubble

    #if not replaced.get('mnu',True):
    Omega_nu = self.provider.get_Omega_nu_massive(0)[0]
    FakeCosmo.Omega_nu = Omega_nu
    if not replaced.get('sigma8',True):
      sigma8 = self.provider.get_sigma8_z(0)[0]
      FakeCosmo.sigma8 = lambda : sigma8
    if not replaced.get('ns',True):
      n_s = self.provider.get_param('n_s')
      FakeCosmo.n_s = lambda : n_s
    if not replaced.get('zreio',True):
      pars = {'z_reio':self.provider.get_param('z_reio')}
      FakeCosmo.get_current_derived_parameters = lambda xs:{x:pars[x] for x in xs}

    if self.wrapper.needs_cosmo_pk:
      Pk_interp = self.provider.get_Pk_interpolator(nonlinear=False)
      FakeCosmo.get_pk_all = lambda ks, z, nonlinear=False, cdmbar=False: Pk_interp.P(z,ks)

    # For one-time checks (like checking all the required nuisance parameters are being sampled)
    if self.to_check:
      sampled = np.zeros_like(self.sampled_params,dtype=bool)
      for ip, p in enumerate(self.sampled_params):
        if p in params:
          sampled[ip] = True
      if not np.all(sampled):
        raise ValueError("[lym1d_cobaya_wrapper] : Missing parameters from list of sampled cobaya parameters: '{}'".format("' , '".join(self.sampled_params[~sampled])))
      self.to_check = False

    try:
      chi2 = self.wrapper.chi2(FakeCosmo,params)
    except ValueError as ve:
      print("[lym1d_cobaya_wrapper] : Encountered value error : ",ve)
      import traceback
      print(traceback.format_exc())
      raise #return -np.inf
    except KeyError as ke:
      print("[lym1d_cobaya_wrapper] : Encountered missing key : ",ke)
      raise

    if chi2==None or np.isnan(chi2):
      return -np.inf
    else:
      return -0.5*chi2

# VERY important line to correclty setup cobaya object (since cobaya inspects the CLASS, not the instance (!), therefore the class needs to be modified)
# Also, since the code for setting up the default parameter basis is not 100% trivial, we need to do it in a function
#cobaya_wrapper.params = cobaya_wrapper.getDefaultParameterBasis()
