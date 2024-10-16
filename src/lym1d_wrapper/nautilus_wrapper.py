from .wrapper import lym1d_wrapper
import os
import numpy as np
from scipy.interpolate import CubicSpline

import classy

class nautilus_wrapper:
  
  runmode = "nyx"

  base_directory = "/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/"

  arguments = {'has_cor':None}
  
  verbose = 1

  params = {
               # name : [fiducial, min, max]
               'h':[0.7,0.6075,0.741],
               'omega_m':[0.14,0.119,0.16],
               'A_lya_nuisance':[9.0,5.6,12.4],
               'n_lya_nuisance':[-2.5,-2.36,-2.24],
               'T0':[10000.0,3800.0,25000.],
               'T0SlopeInf':[1.0,-8.6,8.7],
               'T0SlopeBreak':[1.0, -15., 12.],
               'gamma':[1.5,1.,2.],
               'gammaSlopeInf':[0.1,-1.21, 1.21],
               'gammaSlopeBreak':[0.0,-1.,1.],
               'lambdaP':[75.,60.,100.],
               'AmpTauEff':[-np.log(0.7),0.2,1.5],
               'SlopeTauEffInf':[3.5,1.5,7.1],
               'SlopeTauEffBreak':[0.0,-0.5,0.5],
               'lambdaPSlopeInf':[0.0,-1.6,2.],
               'lambdaPSlopeBreak':[0.0,-3.5,2.1],
  }
  
  _names = None

  k_max = 10.
  
  default_cosmo = {'omega_b':0.02233,"N_ncdm": 0}

  cosmo = classy.Class()

  def __init__(self, **kwargs):

    print("[lym1d_nautilus_wrapper] Beginning likelihood initialization")
    runmode = kwargs.pop("runmode", self.runmode)
    base_directory = kwargs.pop("base_directory", self.base_directory)
    arguments = kwargs.pop("arguments", self.arguments)
    self.default_cosmo.update(kwargs.pop("default_cosmo",{}))
    self.verbose = kwargs.pop("verbose", self.verbose)

    self.wrapper = lym1d_wrapper(
        runmode=runmode,
        base_directory = base_directory,
        **arguments
        )

    nuisance = self.wrapper.nuisance_parameters
    print("[lym1d_nautilus_wrapper] nuisance parameters: ",nuisance)
    print("[lym1d_cobaya_wrapper] Likelihood initialized")

  def add_params(self,params):
    self.params.update(params)

  @property
  def cosmo_pars(self):
    dic = self.default_cosmo.copy()
    if self.wrapper.needs_cosmo_pk:
      dic.update({'output':'mPk','P_k_max_1/Mpc':self.k_max})
    return dic

  @property
  def names(self):
    if not self._names:
      self._names = list(self.params.keys())
    return self._names

  def logp(self, pars):

    cosmo_pars = {}
    nuisance_pars = {}
    for par in pars:
      if par not in self.names:
        cosmo_pars.update(par=pars[par])
      else:
        nuisance_pars.update(par=pars[par])

    self.cosmo.set(self.cosmo_pars)
    self.cosmo.set(cosmo_pars)
    self.cosmo.compute()
    chi2=self.wrapper.chi2(self.cosmo, pars)
    return -0.5 * chi2 if chi2 is not None else -np.inf

  def generate_sampler(self, rng=None, n_networks=8, pool=8, n_live=8000, resume=True, filepath="chains/out.hdf5", **opts):
    try:
      from nautilus import Sampler, Prior 
    except ImportError as ie:
      self.log("Error importing nautilus. Is it correclty installed?\nOriginal message: "+str(ie))
      raise

    prior_flat=Prior()
    for n in self.names:
      prior_flat.add_parameter(n,dist=(self.params[n][1],self.params[n][2]))

    return Sampler(prior_flat, self.logp, n_live=n_live, n_networks=n_networks, pool=pool, filepath=filepath, resume=resume, **opts)

  def log(self, msg, level=1):
    if level <= self.verbose:
      print("[lym1d_nautilus_wrapper] "+"\n[lym1d_nautilus_wrapper] ".join(msg.split("\n")))
#nautilus_wrapper.add_params({'fSiIII':[0.0,-0.2,0.2]})

