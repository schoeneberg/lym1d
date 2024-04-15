import lym1d
from .util import OptionDict
import numpy as np
import os

class lym1d_wrapper:

  def __init__(self, runmode, base_directory="/home/nilsor/codes/montepython_lyadesi_private/montepython/data/Lya_DESI",  **kwargs):
    print("KWARGS = ",kwargs)

    self.prefix = "[lym1d_wrapper] "

    self.runmode = runmode.lower()

    self.Anmode = kwargs.pop("Anmode","default")
    self.verbose = kwargs.pop("verbose",1)
    self.use_thermal_prior = kwargs.pop("use_thermal_prior",False)
    self.nuisance_replacements = kwargs.pop("nuisance_replacements",[])
    self.free_thermal = kwargs.pop("free_thermal",[])
    self.use_nuisance = ['inv_wdm_mass','fSiIII','fSiII','ResoAmpl','ResoSlope','Lya_DLA','Lya_AGN','Lya_SN','Lya_UVFluct']

    self.zlist_thermo = kwargs.pop("zlist_thermo",[2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6])
    self.has_cor = kwargs.pop("has_cor",{})
    self.zmin = kwargs.pop("zmin",min(self.zlist_thermo))
    self.zmax = kwargs.pop("zmax",max(self.zlist_thermo))
    self.H0prior = kwargs.pop("H0prior",False)
    self.emuname = kwargs.pop("emuname","")
    self.data_filename = kwargs.pop("data_filename","pk_1d_DR12_13bins.out")
    self.inversecov_filename = kwargs.pop("inversecov_filename","pk_1d_DR12_13bins_invCov.out")

    self.nz_thermo = len(self.zlist_thermo)

    self.need_cosmo_arguments = {}

    # And now we can finally do the rest of the specific initalization of this likelihood
    self.replace_with_nuisance = OptionDict({"zreio":False,"mnu":False})
    self.free_thermal_for = OptionDict({"Fbar":False,"gamma":False,"T0":False})
    if "taylor" in self.runmode:
      self.replace_with_nuisance.addkeys({"sigma8":False,"ns":False})
    elif "nyx" in self.runmode:
      if self.Anmode == 'default':
        self.replace_with_nuisance.addkeys({"A_lya":False,"n_lya":False})
      elif self.Anmode == 'skm':
        self.replace_with_nuisance.addkeys({"A_lya_skm":False,"n_lya_skm":False})
      elif self.Anmode == 'sigma':
        self.replace_with_nuisance.addkeys({"sigma8":False,"ns":False})
      else:
        raise ValueError(self.prefix+"Unrecognized Anmode '{}'.".format(self.Anmode))
      if "auv" in self.runmode:
        self.free_thermal_for.addkeys({"AUVB":False})
      else:
        self.free_thermal_for.addkeys({"lambdaP":False})
    elif "lace" in self.runmode:
        self.free_thermal_for.addkeys({"kF":False})
        self.replace_with_nuisance.addkeys({"Delta2_p":False,"n_p":False})
    else:
      raise ValueError(self.prefix+"Missing the emulator type in 'runmode' argument")

    # Check and print options
    self.log("Initializing lym1d_wrapper")
    self.log("Runmode : {}".format(self.runmode))
    if "nyx" in self.runmode and self.Anmode!="default":
      self.log("A_and_n_lya_mode = {}".format(self.Anmode))
    self.log("Prior on T0/gamma = {}, prior on H0 = {}".format(self.use_thermal_prior,self.H0prior))
    if not hasattr(self,"shortening_factor"):
      self.shortening_factor = 0.
    if not hasattr(self,"use_H"):
      self.use_H = True
    if not hasattr(self,"use_omm"):
      self.use_omm = True

    #Catch simplified input notation
    if isinstance(self.nuisance_replacements,list):
      self.nuisance_replacements = {key:True for key in self.nuisance_replacements}
    if isinstance(self.free_thermal,list):
      self.free_thermal = {key:True for key in self.free_thermal}
    self.replace_with_nuisance.update(self.nuisance_replacements)
    self.free_thermal_for.update(self.free_thermal)

    for key in self.replace_with_nuisance.iterate():
      self.log("Parameter "+key+" set to nuisance mode",level=2)
      self.use_nuisance+=[key+"_nuisance"]
    for key in self.free_thermal_for.iterate():
      self.log("Parameter "+key+" set to free thermal mode",level=2)
      self.use_nuisance+=[key+"__{}".format(i+1) for i in range(self.nz_thermo)]

    if "amplgrad" in self.runmode.lower():
      self.use_nuisance+=['invAmpl','invGrad']

    if "nyx" in self.runmode.lower():
      if "auv" in self.runmode.lower():
        if not self.free_thermal_for['AUVB']:
          self.use_nuisance+=["A_UVB"]
      else:
        if not self.free_thermal_for['lambdaP']:
          self.use_nuisance+=["lambdaP","lambdaPSlopeInf","lambdaPSlopeBreak"]
    if "lace" in self.runmode.lower():
      if not self.free_thermal_for['kF']:
        self.use_nuisance+=['kF','kFSlopeInf','kFSlopeBreak']
    if not self.free_thermal_for['T0']:
      self.use_nuisance+=['T0','T0SlopeInf','T0SlopeBreak']
    if not self.free_thermal_for['gamma']:
      self.use_nuisance+=['gamma','gammaSlopeInf','gammaSlopeBreak']
    if not self.free_thermal_for['Fbar']:
      self.use_nuisance+=['AmpTauEff','SlopeTauEffInf','SlopeTauEffBreak']

    if "splic" in self.runmode:
      self.log("Running in splicing mode")
      self.use_nuisance+=["SplicingCorr","SplicingOffset"]

    # The normal runmode
    if ("taylor" in self.runmode and not (self.replace_with_nuisance['sigma8'] and self.replace_with_nuisance['ns']))\
        or ("nyx" in self.runmode and
          (self.Anmode == 'default' and not (self.replace_with_nuisance['A_lya'] and self.replace_with_nuisance['n_lya'])
          or self.Anmode == 'skm' and not (self.replace_with_nuisance['A_lya_skm'] and self.replace_with_nuisance['n_lya_skm'])
          or self.Anmode == 'sigma' and not (self.replace_with_nuisance['sigma8'] and self.replace_with_nuisance['ns'])
          )) or ("lace" in self.runmode):
      print("ASDF")
      self.need_cosmo_arguments = {'output': 'mPk','z_max_pk':6,'P_k_max_1/Mpc':10.}

    self.log("Finished initializing lym1d_wrapper")

    arguments = {
      'runmode':self.runmode,
      'An_mode':self.Anmode,
      'has_cor':self.has_cor,
      'zmin':self.zmin, 'zmax':self.zmax, 'zs' : self.zlist_thermo,
      'emupath':"Lya_emu{}{}{}{}{}.npz".format(self.emuname,"_lambda_P" if not ( "auv" in self.runmode) else "",("_{}".format(self.Anmode)) if (self.Anmode is not 'default') else "","_{}".format('noH') if not self.use_H else "","_{}".format('noOm') if not self.use_omm else ""),
      'data_filename':self.data_filename,
      'inversecov_filename':self.inversecov_filename,
      'shortening_factor':(self.shortening_factor if hasattr(self,"shortening_factor") else 0.),
      'convex_hull_mode':(self.convex_hull_mode if hasattr(self,"convex_hull_mode") else False),
      'use_H':(self.use_H if hasattr(self,"use_H") else True),
      'lace_type':(self.lace_type if hasattr(self,'lace_type') else 'gadget'),
      'splice_kind':(self.splice_kind if hasattr(self,"splice_kind") else 1),
      'verbose':3
    }

    import lym1d
    self.lyalkl = lym1d.lym1d(base_directory, **arguments)

  @property
  def nuisance_parameters(self):
    nuisance_parameters = ['normalization%d'%(ih+1) for ih in range(self.nz_thermo)]
    nuisance_parameters+= ['noise%d'%(ih+1) for ih in range(self.nz_thermo)]
    nuisance_parameters+= ['tauError%d'%(ih+1) for ih in range(self.nz_thermo)]

    for key in ['fSiIII','fSiII','ResoAmpl','ResoSlope','Lya_DLA','Lya_AGN','Lya_SN','Lya_UVFluct','A_UVB','AmpTauEff','SlopeTauEffInf', 'SlopeTauEffBreak','T0SlopeInf','T0SlopeBreak','gammaSlopeInf', 'gammaSlopeBreak', 'T0', 'gamma','lambdaPSlopeInf','lambdaPSlopeBreak', 'lambdaP','kF','kFSlopeInf','kFSlopeBreak']:
      if key in self.use_nuisance:
        nuisance_parameters.append(key)

    for key in self.replace_with_nuisance.iterate():
      nuisance_parameters.append(key+"_nuisance")

    return nuisance_parameters

  def chi2(self, cosmo, parameters):

    # 1) Read current value of nuisance parameters
    self.normalization = [None]*self.nz_thermo
    self.noise = [None]*self.nz_thermo
    self.tauError = [None]*self.nz_thermo
    for ih in range(self.nz_thermo):
      self.normalization[ih] = parameters['normalization%d'%(ih+1)]
      self.noise[ih] = parameters['noise%d'%(ih+1)]
      self.tauError[ih] = parameters['tauError%d'%(ih+1)]

    nuisance_parameters=[]
    for key in ['fSiIII','fSiII','ResoAmpl','ResoSlope','Lya_DLA','Lya_AGN','Lya_SN','Lya_UVFluct','A_UVB','AmpTauEff','SlopeTauEffInf', 'SlopeTauEffBreak','T0SlopeInf','T0SlopeBreak','gammaSlopeInf', 'gammaSlopeBreak', 'T0', 'gamma','lambdaPSlopeInf','lambdaPSlopeBreak', 'lambdaP','kF','kFSlopeInf','kFSlopeBreak']:
      if key in self.use_nuisance:
        nuisance_parameters.append(key)

    for key in self.replace_with_nuisance.iterate():
      nuisance_parameters.append(key+"_nuisance")

    for nuisance in nuisance_parameters:
      setattr(self,nuisance,parameters[nuisance])

    cosmopar = {}

    cosmopar['Omega_m'] = cosmo.Omega_m()
    cosmopar['omega_m'] = cosmo.Omega_m()*cosmo.h()**2
    cosmopar['H0'] = cosmo.h()*100.
    c_kms = 299792.458 # This factor of c [km/s] is required to convert cosmo.Hubble from CLASS units [1/Mpc] to SI units [km/s/Mpc].
    cosmopar['H(z)'] = lambda z: c_kms * cosmo.Hubble(z)
    cosmopar['Hubble'] = lambda z: cosmo.Hubble(z)
    cosmopar['Omega_nu'] = cosmo.Omega_nu

    if "nyx" in self.runmode:
      if self.Anmode == 'default':
        if not ( self.replace_with_nuisance['A_lya'] and self.replace_with_nuisance['n_lya']):
          Alya, nlya, alphalya = self.postprocessing_A_and_n_lya(cosmo)
          cosmopar['A_lya'] = Alya
          cosmopar['n_lya'] = nlya
        if self.replace_with_nuisance['A_lya']:
          cosmopar['A_lya'] = self.A_lya_nuisance
        if self.replace_with_nuisance['n_lya']:
          cosmopar['n_lya'] = self.n_lya_nuisance
      elif self.Anmode == 'skm':
        if not ( self.replace_with_nuisance['A_lya_skm'] and self.replace_with_nuisance['n_lya_skm']):
          Alya, nlya, alphalya = self.postprocessing_A_and_n_lya(cosmo,units='skm',k_p=0.009)
          cosmopar['A_lya_skm'] = Alya
          cosmopar['n_lya_skm'] = nlya
        if self.replace_with_nuisance['A_lya_skm']:
          cosmopar['A_lya_skm'] = self.A_lya_skm_nuisance
        if self.replace_with_nuisance['n_lya_skm']:
          cosmopar['n_lya_skm'] = self.n_lya_skm_nuisance

    if "taylor" in self.runmode or ("nyx" in self.runmode and self.Anmode=='sigma'):
      if self.replace_with_nuisance['sigma8']:
        cosmopar['sigma8'] = self.sigma8_nuisance
      else:
        cosmopar['sigma8'] = cosmo.sigma8()
      if self.replace_with_nuisance['ns']:
        cosmopar['n_s'] = self.ns_nuisance
      else:
        cosmopar['n_s'] = cosmo.n_s()
    if self.replace_with_nuisance['zreio']:
      cosmopar['z_reio'] = self.zreio_nuisance
    else:
      cosmopar['z_reio'] = cosmo.get_current_derived_parameters(["z_reio"])["z_reio"]

    if "lace" in self.runmode.lower():
      from lace.cosmo.fit_linP import fit_linP_Mpc_zs 
      if not (self.replace_with_nuisance['Delta2_p'] and self.replace_with_nuisance['n_p']):
        zs = self.zlist_thermo
        fp = [cosmo.scale_independent_growth_factor_f(z) for z in zs]
        k_Mpc = np.geomspace(0.001,9.,num=1000)
        P_Mpc = cosmo.get_pk_all(k_Mpc,z=zs)
        kp_Mpc = 0.7
        return_array = fit_linP_Mpc_zs(k_Mpc, P_Mpc, fp, kp_Mpc, zs)
      if self.replace_with_nuisance['Delta2_p']:
        cosmopar['Delta2_p'] = lambda z: self.Delta2_p_nuisance
      else:
        cosmopar['Delta2_p'] = CubicSpline(self.zlist_thermo,[return_array[iz]['Delta2_p'] for iz,z in enumerate(self.zlist_thermo)])
      if self.replace_with_nuisance['n_p']:
        cosmopar['n_p'] = lambda z:self.n_p_nuisance
      else:
        cosmopar['n_p'] = CubicSpline(self.zlist_thermo,[return_array[iz]['n_p'] for iz,z in enumerate(self.zlist_thermo)])

    therm = {}
    for key in self.free_thermal_for.iterate():
      vals = [parameters[key+'__%d'%(ih+1)]['current'] * parameters[key+'__%d'%(ih+1)]['scale'] for ih in range(self.nz_thermo)]
      therm[key] = CubicSpline(self.zlist_thermo,vals)

    for key in self.free_thermal_for.inverse_iterate():
      if key=="T0":
        therm['T0'] = {'amp':parameters['T0'],'slope':parameters['T0SlopeInf'],'break':parameters['T0SlopeBreak']}
      elif key=="gamma":
        therm['gamma'] = {'amp':parameters['gamma'],'slope':parameters['gammaSlopeInf'],'break':parameters['gammaSlopeBreak']}
      elif key=="Fbar":
        therm['tau_eff'] = {'amp':parameters['AmpTauEff'],'slope':parameters['SlopeTauEffInf'],'break':parameters['SlopeTauEffBreak']}
      elif key=="lambdaP":
        therm['lambdaP'] = {'amp':parameters['lambdaP'],'slope':parameters['lambdaPSlopeInf'],'break':parameters['lambdaPSlopeBreak']}
      elif key=="AUVB":
        therm['UV'] = lambda z: parameters['A_UVB']
      elif key=="kF":
        therm['kF'] = {'amp':parameters['kF'],'slope':parameters['kFSlopeInf'],'break':parameters['kFSlopeBreak']}
      else:
        raise ValueError("Unknown key '"+key+"' in free_thermal_for")

    nuisance = {}
    #nuisance['noise'] = [parameters['normalization%d'%(ih+1)] for ih in range(self.nz_thermo)]
    #nuisance['normalization'] = [parameters['noise%d'%(ih+1)] for ih in range(self.nz_thermo)]
    #nuisance['tauError'] = [parameters['tauError%d'%(ih+1)] for ih in range(self.nz_thermo)]
    nuisance['noise'] = self.noise
    nuisance['normalization'] = self.normalization
    nuisance['tauError'] = self.tauError

    nuisance['DLA'] = parameters['Lya_DLA']
    nuisance['SN'] = parameters['Lya_SN']
    nuisance['AGN'] = parameters['Lya_AGN']
    nuisance['reso_ampl'] = parameters['ResoAmpl']
    nuisance['reso_slope'] = parameters['ResoSlope']
    nuisance['fSiIII'] = parameters['fSiIII']
    nuisance['fSiII'] = parameters['fSiII']

    # Only used in the Taylor emulator case
    nuisance['UVFluct'] = parameters['Lya_UVFluct']
    # Above
    if "taylor" in self.runmode and not self.free_thermal_for['Fbar']:
      nuisance['AmpTauEff'] = self.AmpTauEff
      nuisance['SlopeTauEff'] = self.SlopeTauEffInf
      if self.SlopeTauEffBreak!=0.0:
        raise Exception("Invalid parameter SlopeTauEffBreak!=0 even though taylor mdoe")
    if "amplgrad" in self.runmode:
      nuisance['invAmpl'] = parameters['invAmpl']['current'] * parameters['invAmpl']['scale']
      nuisance['invGrad'] = parameters['invGrad']['current'] * parameters['invGrad']['scale']
    if "splic" in self.runmode:
      nuisance['splicing_corr'] = parameters['SplicingCorr']['current'] * parameters['SplicingCorr']['scale']
      nuisance['splicing_offset'] = parameters['SplicingOffset']['current'] * parameters['SplicingOffset']['scale']

    chi_squared = self.lyalkl.chi2_plus_prior(cosmopar,therm,nuisance)
    if chi_squared==None:
      return None

    #5.6) Add T0 and Gamma fitting
    if(self.use_thermal_prior):
      #Gaussian constraints on T0 slope inf,sup and gamma slope, gamma amp
      chi_squared +=  pow((parameters['T0SlopeInf']-2.0)/2.0,2.0)
      chi_squared +=  pow((parameters['T0SlopeInf']+parameters['T0SlopeBreak']+2.0)/3.0,2.0)
      chi_squared +=  pow((parameters['gammaSlopeInf']-0.1)/1.0,2.0)
      chi_squared +=  pow((parameters['gamma']-self.gammaPriorMean)/0.3,2.0)

    if(self.H0prior):
      chi_squared += pow((cosmopar['H0']-self.H0prior['mean'])/self.H0prior['sigma'],2.0)

    return chi_squared





  def postprocessing_A_and_n_lya(self, cosmo, z_p = 3.0, k_p = 1.0, units = "Mpc", normalize = True, cdmbar = False):
    ks = np.geomspace(1e-5,5,num=10000)
    pks = cosmo.get_pk_all(ks, z=z_p, nonlinear = False, cdmbar = cdmbar)
    if units == "Mpc" or units == "MPC" or units == "mpc":
      unit = 1.
    elif units == "skm" or units == "SKM" or units == "kms" or units == "KMS":
      unit = cosmo.Hubble(z_p)/cosmo.Hubble(0)*cosmo.h()*100./(1.+z_p)
    elif "h" in units or "H" in units:
      unit = cosmo.h()
    else:
      raise ValueError(self.prefix+"Your input of units='{}' could not be interpreted".format(units))
    x,y = np.log(ks),np.log(pks)
    k_p_Mpc = k_p*unit
    x0 = np.log(k_p_Mpc)
    scale = 0.1
    w = np.exp(-0.5*(x-x0)*(x-x0)/scale/scale) #Unit = 1
    dw = (x-x0)/scale/scale*np.exp(-0.5*(x-x0)*(x-x0)/scale/scale) #Unit = 1/scale
    ddw = (-1./scale/scale + ((x-x0)/scale/scale)**2)*np.exp(-0.5*(x-x0)*(x-x0)/scale/scale) #Unit = 1/scale^2
    s = np.trapz(w,x)
    r = np.trapz(y*w,x)/s
    dr = np.trapz(y*dw,x)/s
    ddr = np.trapz(y*ddw,x)/s
    A_lya_Mpc = np.exp(r)
    n_lya = dr
    alpha_lya = ddr
    # Unit conversion
    if not normalize:
      A_lya = A_lya_Mpc/unit**3
    else:
      A_lya = A_lya_Mpc*k_p_Mpc**3
    return A_lya, n_lya, alpha_lya

  def log(self, msg, level=1):
    if level <= self.verbose:
      print(self.prefix+("\n"+self.prefix).join(msg.split("\n")))
