import lym1d
from .util import OptionDict
import numpy as np
import os

class lym1d_wrapper:

  nuisance_mapping = {'DLA':'Lya_DLA','AGN':'Lya_AGN','SN':'Lya_SN',
    'UVFluct':'Lya_UVFluct', # Only used in old taylor case
    'fSiIII':'fSiIII','fSiII':'fSiII',
    'reso_ampl':'ResoAmpl','reso_slope':'ResoSlope',
    'splicing_corr':'SplicingCorr','splicing_offset':'SplicingOffset'}

  # Initialization routine
  def __init__(self, runmode, base_directory="",  **kwargs):

    # Set some default parameters
    self.prefix = "[lym1d_wrapper] "
    self.need_cosmo_arguments = {}

    # Interpret the runmode
    self.runmode = runmode.lower()

    # Consume the keyword argument inputs (apart from base_directory)
    self.consume_input(kwargs)

    # Initialize the thermal powerlaw parameters
    self.initialize_thermal_powerlaws()

    # Initialize the nuisance replacements
    self.initialize_parameter_nuisance_replacements()

    # Propagate info to cosmo argument requirements
    if self.needs_cosmo_pk:
      self.need_cosmo_arguments = {'output': 'mPk','z_max_pk':6,'P_k_max_1/Mpc':10.}

    self.log("Finished initializing lym1d_wrapper")

    # Now, finally, compose the lym1d arguments for the final call
    arguments = self.compose_lym1d_arguments()
    # Pass also the arguments from the kwargs that weren't consumed yet
    arguments.update(kwargs)

    # Now import lym1d and run!
    import lym1d
    self.lyalkl = lym1d.lym1d(base_directory, **arguments)

    # Update which nuisance parameters are required based on likelihood requirements
    self.update_base_nuisances()

  # Get the chi2 (including all priors)
  def chi2(self, cosmo, parameters):

    # 1a) Simple cosmological parameters
    cosmopar = {}

    cosmopar['Omega_m'] = cosmo.Omega_m()
    cosmopar['omega_m'] = cosmo.Omega_m()*cosmo.h()**2
    cosmopar['H0'] = cosmo.h()*100.
    c_kms = 299792.458 # This factor of c [km/s] is required to convert cosmo.Hubble from CLASS units [1/Mpc] to SI units [km/s/Mpc].
    cosmopar['H(z)'] = lambda z: c_kms * cosmo.Hubble(z)
    cosmopar['Hubble'] = lambda z: cosmo.Hubble(z)
    cosmopar['Omega_nu'] = cosmo.Omega_nu

    # 1b) Cosmological parameters from power spectrum
    self.optionally_get_cosmo_or_nuisance(cosmo, cosmopar, parameters)

    # 2) thermal parameters
    therm = self.get_thermo_powerlaw_or_free(parameters)

    # 3) nuisance parameters
    nuisance = self.get_nuisances(parameters)

    # 4) chi square
    chi_squared = self.lyalkl.chi2_plus_prior(cosmopar,therm,nuisance)
    if chi_squared==None:
      return None

    # 5) Additional thermal priors (mostly backward compatibility)
    if(self.use_thermal_prior):
      #Gaussian constraints on T0 slope inf,sup and gamma slope, gamma amp
      chi_squared +=  pow((parameters['T0SlopeInf']-2.0)/2.0,2.0)
      chi_squared +=  pow((parameters['T0SlopeInf']+parameters['T0SlopeBreak']+2.0)/3.0,2.0)
      chi_squared +=  pow((parameters['gammaSlopeInf']-0.1)/1.0,2.0)
      chi_squared +=  pow((parameters['gamma']-self.gammaPriorMean)/0.3,2.0)

    # 6) Additional H0 prior (mostly backward compatibility)
    if(self.H0prior):
      chi_squared += pow((cosmopar['H0']-self.H0prior['mean'])/self.H0prior['sigma'],2.0)

    return chi_squared






  def consume_input(self, kwargs):

    if not (("taylor" in self.runmode) or ("nyx" in self.runmode) or ("lace" in self.runmode)):
      raise ValueError(self.prefix+"Missing the emulator type in 'runmode' argument")

    self.Anmode = kwargs.pop("Anmode","default")
    if self.Anmode not in ['default','skm','sigma']:
      raise ValueError("Unknown Anmode = {}".format(self.Anmode))
    self.verbose = kwargs.pop("verbose",1)
    self.use_thermal_prior = kwargs.pop("use_thermal_prior",False)
    self.nuisance_replacements = kwargs.pop("nuisance_replacements",[])
    self.free_thermal = kwargs.pop("free_thermal",[])

    self.base_nuisance = []

    self.zlist_thermo = kwargs.pop("zlist_thermo",[2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6])
    self.zmin = kwargs.pop("zmin",min(self.zlist_thermo))
    self.zmax = kwargs.pop("zmax",max(self.zlist_thermo))
    self.H0prior = kwargs.pop("H0prior",False)
    self.emuname = kwargs.pop("emuname","")
    for arg in ['emupath']:
      if arg in kwargs:
        setattr(self,arg,kwargs.pop(arg))
    self.FbarMode = kwargs.pop("FbarMode","tau_eff")
    if self.FbarMode not in ['tau_eff','Fbar']:
      raise ValueError("Unknown FbarMode = {}".format(self.FbarMode))

    self.use_H = kwargs.pop("use_H",True)
    self.use_omm = kwargs.pop("use_omm",True)

    self.gammaPriorMean = kwargs.pop("gammaPriorMean",1.3)

    self.nz_thermo = len(self.zlist_thermo)

    # Check and print options
    self.log("Initializing lym1d_wrapper")
    self.log("Runmode : {}".format(self.runmode))
    if "nyx" in self.runmode and self.Anmode!="default":
      self.log("A_and_n_lya_mode = {}".format(self.Anmode))
    self.log("Prior on T0/gamma = {}, prior on H0 = {}".format(self.use_thermal_prior,self.H0prior))

  def initialize_thermal_powerlaws(self):

    self.powerlaw_keys = {'T0': {'amp':'T0','slope':'T0SlopeInf','break':'T0SlopeBreak'},
                       'gamma': {'amp':'gamma','slope':'gammaSlopeInf','break':'gammaSlopeBreak'},
                     'tau_eff': {'amp':'AmpTauEff','slope':'SlopeTauEffInf','break':'SlopeTauEffBreak'},
                        'Fbar': {'amp':'Fbar','slope':'FbarSlopeInf','break':'FbarSlopeBreak'},
                     'lambdaP': {'amp':'lambdaP','slope':'lambdaPSlopeInf','break':'lambdaPSlopeBreak'},
                          'kF': {'amp':'kF','slope':'kFSlopeInf','break':'kFSlopeBreak'},
                          'UV': {'amp':'A_UVB'}}

    self.thermal_is_activated = {'T0':True,
                                 'gamma':True,
                                 'tau_eff':(self.FbarMode=='tau_eff'),
                                 'Fbar':(self.FbarMode=='Fbar'),
                                 'kF':("lace" in self.runmode),
                                 'lambdaP':("nyx" in self.runmode and not "auv" in self.runmode),
                                 'UV':("nyx" in self.runmode and "auv" in self.runmode)}

    assert(set(self.powerlaw_keys.keys())==set(self.thermal_is_activated.keys()))

    # Check which powerlaw/free parameters will be wanted
    self.free_thermal_for = OptionDict({})
    for key in self.powerlaw_keys:
      if self.thermal_is_activated[key]:
         self.free_thermal_for.addkeys({key:False})

    if isinstance(self.free_thermal,list):
      self.free_thermal = {key:True for key in self.free_thermal}
    self.free_thermal_for.update(self.free_thermal)


    for key in self.powerlaw_keys:
      if self.thermal_is_activated[key]:
        # Free thermal parameter
        if self.free_thermal_for[key]:
          self.log("Parameter "+key+" set to free thermal mode",level=2)
          self.base_nuisance+=[key+"__{}".format(i+1) for i in range(self.nz_thermo)]
        # Powerlaw modeling
        else:
          self.base_nuisance+=list(self.powerlaw_keys[key].values())

    # Done !


  def initialize_parameter_nuisance_replacements(self):

    self.replace_is_activated = {'zreio':True,
                                 'mnu':("taylor" in self.runmode),
                                 'sigma8':("taylor" in self.runmode or ("nyx" in self.runmode and self.Anmode=="sigma")),
                                 'ns':("taylor" in self.runmode or ("nyx" in self.runmode and self.Anmode=="sigma")),
                                 'A_lya':("nyx" in self.runmode and self.Anmode=='default'),
                                 'n_lya':("nyx" in self.runmode and self.Anmode=='default'),
                                 'A_lya_skm':("nyx" in self.runmode and self.Anmode=='skm'),
                                 'n_lya_skm':("nyx" in self.runmode and self.Anmode=='skm'),
                                 'Delta2_p':("lace" in self.runmode),
                                 'n_p':("lace" in self.runmode)
                                 }

    # Check which replace parameters will be wanted
    self.replace_with_nuisance = OptionDict({})
    for key in self.replace_is_activated:
      if self.replace_is_activated[key]:
         self.replace_with_nuisance.addkeys({key:False})

    #Catch simplified input notation
    if isinstance(self.nuisance_replacements,list):
      self.nuisance_replacements = {key:True for key in self.nuisance_replacements}

    self.replace_with_nuisance.update(self.nuisance_replacements)

    cosmo_pk_params = ["sigma8","ns","A_lya","n_lya","A_lya_skm","n_lya_skm","Delta2_p","n_p"]
    self.needs_cosmo_pk = any([(isactive and not self.replace_with_nuisance[k]) for (k,isactive) in self.replace_is_activated.items() if k in cosmo_pk_params])

    for key in self.replace_with_nuisance.iterate():
      self.log("Parameter "+key+" set to nuisance mode",level=2)
      self.base_nuisance+=[key+"_nuisance"]


  def compose_lym1d_arguments(self):

    arguments = {
      'runmode':self.runmode,
      'An_mode':self.Anmode,
      'zmin':self.zmin, 'zmax':self.zmax, 'zs' : self.zlist_thermo,
      'emupath':("Lya_emu{}{}{}{}{}.npz".format(self.emuname,"_lambda_P" if not ( "auv" in self.runmode) else "",("_{}".format(self.Anmode)) if (self.Anmode!='default') else "","_{}".format('noH') if not self.use_H else "","_{}".format('noOm') if not self.use_omm else "") if "nyx" in self.runmode else ""),
      'use_H':self.use_H,
      'verbose':self.verbose
    }
    for attr in ['emupath']:
      if hasattr(self,attr):
        arguments.update({attr:getattr(self,attr)})
    return arguments

  def get_thermo_powerlaw_or_free(self, parameters):

    therm = {}
    for key in self.free_thermal_for.iterate():
      vals = [parameters[key+'__%d'%(ih+1)] for ih in range(self.nz_thermo)]
      therm[key] = CubicSpline(self.zlist_thermo,vals)
    # Put the thermal powerlaw parameters into the thermal dictionary
    for key in self.free_thermal_for.inverse_iterate():
      therm[key] = {x:parameters[y] for (x,y) in self.powerlaw_keys[key].items()}

    return therm

  # TODO :: refactor
  def optionally_get_cosmo_or_nuisance(self, cosmo, cosmopar, parameters):

    compute_A_and_n_lya = False
    if self.replace_is_activated['A_lya'] and not self.replace_with_nuisance['A_lya']:
      compute_A_and_n_lya = True
    if self.replace_is_activated['n_lya'] and not self.replace_with_nuisance['n_lya']:
      compute_A_and_n_lya = True
    if compute_A_and_n_lya:
      Alya, nlya = self.postprocessing_A_and_n_lya(cosmo)
    if self.replace_is_activated['A_lya'] and self.replace_with_nuisance['A_lya']:
      Alya = parameters['A_lya_nuisance']
    if self.replace_is_activated['n_lya'] and self.replace_with_nuisance['n_lya']:
      nlya = parameters['n_lya_nuisance']
    if self.replace_is_activated['A_lya']:
      cosmopar['A_lya'] = Alya
    if self.replace_is_activated['n_lya']:
      cosmopar['n_lya'] = nlya

    compute_A_and_n_lya_skm = False
    if self.replace_is_activated['A_lya_skm'] and not self.replace_with_nuisance['A_lya_skm']:
      compute_A_and_n_lya_skm = True
    if self.replace_is_activated['n_lya_skm'] and not self.replace_with_nuisance['n_lya_skm']:
      compute_A_and_n_lya_skm = True
    if compute_A_and_n_lya_skm:
      Alya_skm, nlya_skm = self.postprocessing_A_and_n_lya(cosmo,units='skm',k_p=0.009)
    if self.replace_is_activated['A_lya_skm'] and self.replace_with_nuisance['A_lya_skm']:
      Alya_skm = parameters['A_lya_skm_nuisance']
    if self.replace_is_activated['n_lya_skm'] and self.replace_with_nuisance['n_lya_skm']:
      nlya_skm = parameters['n_lya_skm_nuisance']
    if self.replace_is_activated['A_lya_skm']:
      cosmopar['A_lya_skm'] = Alya_skm
    if self.replace_is_activated['n_lya_skm']:
      cosmopar['n_lya_skm'] = nlya_skm

    if self.replace_is_activated['sigma8'] and not self.replace_with_nuisance['sigma8']:
      cosmopar['sigma8'] = cosmo.sigma8()
    if self.replace_is_activated['sigma8'] and self.replace_with_nuisance['sigma8']:
      cosmopar['sigma8'] = parameters['sigma8_nuisance']

    if self.replace_is_activated['ns'] and not self.replace_with_nuisance['ns']:
      cosmopar['n_s'] = cosmo.n_s()
    if self.replace_is_activated['ns'] and self.replace_with_nuisance['ns']:
      cosmopar['n_s'] = parameters['ns_nuisance']

    if self.replace_is_activated['zreio'] and not self.replace_with_nuisance['zreio']:
      cosmopar['zreio'] = cosmo.get_current_derived_parameters(["z_reio"])["z_reio"]
    if self.replace_is_activated['zreio'] and self.replace_with_nuisance['zreio']:
      cosmopar['zreio'] = parameters['zreio_nuisance']

    compute_lace = False
    if self.replace_is_activated['Delta2_p'] and not self.replace_with_nuisance['Delta2_p']:
      compute_lace = True
    if self.replace_is_activated['n_p'] and not self.replace_with_nuisance['n_p']:
      compute_lace = True
    if compute_lace:
      Delta2_p, n_p = self.pk_lace(cosmo)
    if self.replace_is_activated['Delta2_p'] and self.replace_with_nuisance['Delta2_p']:
      Delta_2_p = parameters['Delta2_p_nuisance']
    if self.replace_is_activated['n_p'] and self.replace_with_nuisance['n_p']:
      n_p = parameters['n_p_nuisance']
    if self.replace_is_activated['Delta2_p']:
      cosmopar['Delta2_p'] = Delta_2_p
    if self.replace_is_activated['n_p']:
      cosmopar['n_p'] = n_p

  def get_nuisances(self, parameters):

    lkl_nuisances = self.lyalkl.nuisance_parameters.copy()

    nuisance = {}
    if 'normalization' in lkl_nuisances:
      nuisance['normalization'] = [parameters['normalization%d'%(ih+1)] for ih in range(self.nz_thermo)]
      lkl_nuisances.remove('normalization')
    if 'noise' in lkl_nuisances:
      nuisance['noise']         = [parameters['noise%d'%(ih+1)]         for ih in range(self.nz_thermo)]
      lkl_nuisances.remove('noise')

    for key in self.nuisance_mapping:
      if key in lkl_nuisances:
        try:
          nuisance[key] = parameters[self.nuisance_mapping[key]]
        except KeyError as ke:
          self.log("Missing parameter not supplied to the wrapper : {}".format(self.nuisance_mapping[key]),level=0)
          raise

    # Above
    if "taylor" in self.runmode and ((self.FbarMode=='tau_eff' and not self.free_thermal_for['tau_eff']) or (self.FbarMode=='Fbar' and not self.free_thermal_for['Fbar'])):
      nuisance['AmpTauEff'] = parameters['AmpTauEff']
      nuisance['SlopeTauEff'] = parameters['SlopeTauEffInf']
      if parameters['SlopeTauEffBreak']!=0.0:
        raise ValueError("Invalid parameter SlopeTauEffBreak!=0 even though taylor mode")

    return nuisance

  def update_base_nuisances(self):

    lkl_nuisances = self.lyalkl.nuisance_parameters.copy()

    if 'normalization' in lkl_nuisances:
      self.base_nuisance.extend(['normalization%d'%(ih+1) for ih in range(self.nz_thermo)])
      lkl_nuisances.remove('normalization')

    if 'noise' in lkl_nuisances:
      self.base_nuisance.extend(['noise%d'%(ih+1) for ih in range(self.nz_thermo)])
      lkl_nuisances.remove('noise')

    for par in lkl_nuisances:
      self.base_nuisance.append(self.nuisance_mapping[par])

  @property
  def nuisance_parameters(self):

    nuisance_parameters = []

    for key in self.base_nuisance:
      nuisance_parameters.append(key)

    for key in self.replace_with_nuisance.iterate():
      nuisance_parameters.append(key+"_nuisance")

    return nuisance_parameters

#  @property
#  def small_nuisances(self):
#    return ['fSiIII','fSiII','ResoAmpl','ResoSlope','Lya_DLA','Lya_AGN','Lya_SN','Lya_UVFluct','A_UVB','AmpTauEff','SlopeTauEffInf', 'SlopeTauEffBreak','T0SlopeInf','T0SlopeBreak','gammaSlopeInf', 'gammaSlopeBreak', 'T0', 'gamma','lambdaPSlopeInf','lambdaPSlopeBreak', 'lambdaP','kF','kFSlopeInf','kFSlopeBreak']




  def pk_lace(self, cosmo):

    from lace.cosmo.fit_linP import fit_linP_Mpc_zs

    zs = self.zlist_thermo
    fp = [cosmo.scale_independent_growth_factor_f(z) for z in zs]
    k_Mpc = np.geomspace(0.001,9.,num=1000)
    P_Mpc = cosmo.get_pk_all(k_Mpc,z=zs)
    kp_Mpc = 0.7
    return_array = fit_linP_Mpc_zs(k_Mpc, P_Mpc, fp, kp_Mpc, zs)

    return CubicSpline(self.zlist_thermo,[return_array[iz]['Delta2_p'] for iz,z in enumerate(self.zlist_thermo)]), CubicSpline(self.zlist_thermo,[return_array[iz]['n_p'] for iz,z in enumerate(self.zlist_thermo)])

  @staticmethod
  def postprocessing_A_and_n_lya(cosmo, z_p = 3.0, k_p = 1.0, units = "Mpc", normalize = True, cdmbar = False):
    ks = np.geomspace(1e-5,5,num=10000)
    pks = cosmo.get_pk_all(ks, z=z_p, nonlinear = False, cdmbar = cdmbar)
    if units == "Mpc" or units == "MPC" or units == "mpc":
      unit = 1.
    elif units == "skm" or units == "SKM" or units == "kms" or units == "KMS":
      unit = cosmo.Hubble(z_p)/cosmo.Hubble(0)*cosmo.h()*100./(1.+z_p)
    elif "h" in units or "H" in units:
      unit = cosmo.h()
    else:
      raise ValueError("Your input of units='{}' could not be interpreted".format(units))
    x,y = np.log(ks),np.log(pks)
    k_p_Mpc = k_p*unit
    x0 = np.log(k_p_Mpc)
    scale = 0.1
    w = np.exp(-0.5*(x-x0)*(x-x0)/scale/scale) #Unit = 1
    dw = (x-x0)/scale/scale*np.exp(-0.5*(x-x0)*(x-x0)/scale/scale) #Unit = 1/scale
    ## ddw = (-1./scale/scale + ((x-x0)/scale/scale)**2)*np.exp(-0.5*(x-x0)*(x-x0)/scale/scale) #Unit = 1/scale^2
    s = np.trapz(w,x)
    r = np.trapz(y*w,x)/s
    dr = np.trapz(y*dw,x)/s
    ## ddr = np.trapz(y*ddw,x)/s
    A_lya_Mpc = np.exp(r)
    n_lya = dr
    ## alpha_lya = ddr
    # Unit conversion
    if not normalize:
      A_lya = A_lya_Mpc/unit**3
    else:
      A_lya = A_lya_Mpc*k_p_Mpc**3
    return A_lya, n_lya

  def log(self, msg, level=1):
    if level <= self.verbose:
      print(self.prefix+("\n"+self.prefix).join(msg.split("\n")))
