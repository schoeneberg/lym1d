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

  def convert_parameters(self, cosmo, parameters):

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

    return cosmopar, therm, nuisance

  # Get the chi2 (including all priors)
  def chi2(self, cosmo, parameters):

    cosmopar, therm, nuisance = self.convert_parameters(cosmo, parameters)

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

    self.Anmode = kwargs.pop("An_mode","default")
    if self.Anmode not in ['default','sigma','star','star_alpha','post','post_alpha']:
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

    self.has_cosmo  = {'zreio':True,
                       'mnu':("taylor" in self.runmode),
                       'sigma8':("taylor" in self.runmode or ("nyx" in self.runmode and self.Anmode=="sigma")),
                       'ns':("taylor" in self.runmode or ("nyx" in self.runmode and self.Anmode=="sigma")),
                       'A_lya':("nyx" in self.runmode and self.Anmode=='default'),
                       'n_lya':("nyx" in self.runmode and self.Anmode=='default'),
                       'Delta_lya_from_lym1d':("nyx" in self.runmode and 'post' in self.Anmode),
                       'n_lya_from_lym1d':("nyx" in self.runmode and 'post' in self.Anmode),
                       'alpha_lya_from_lym1d':("nyx" in self.runmode and self.Anmode=='post_alpha'),
                       'Delta_star':("nyx" in self.runmode and 'star' in self.Anmode),
                       'n_star':("nyx" in self.runmode and 'star' in self.Anmode),
                       'alpha_star':("nyx" in self.runmode and self.Anmode=='star_alpha'),
                       'Delta2_p':("lace" in self.runmode),
                       'n_p':("lace" in self.runmode),
                       'alpha_p':("lace" in self.runmode)
                       }

    # Check which replace parameters will be wanted
    self.replace_with_nuisance = OptionDict({})
    for key in self.has_cosmo:
      if self.has_cosmo[key]:
         self.replace_with_nuisance.addkeys({key:False})

    #Catch simplified input notation
    if isinstance(self.nuisance_replacements,list):
      self.nuisance_replacements = {key:True for key in self.nuisance_replacements}

    self.replace_with_nuisance.update(self.nuisance_replacements)

    # TODO :: CHECK THAT ALL REPLACE_WITH_NUISANCE do actually have HAS_COSMO =TRUE
    cosmo_pk_params = [k for k in self.has_cosmo.keys() if not (k=='zreio' or k=='mnu')]
    self.needs_cosmo_pk = any([(isactive and not self.replace_with_nuisance[k]) for (k,isactive) in self.has_cosmo.items() if k in cosmo_pk_params])

    for key in self.replace_with_nuisance.iterate():
      self.log("Parameter "+key+" set to nuisance mode",level=2)
      self.base_nuisance+=[key+"_nuisance"]


  def compose_lym1d_arguments(self):

    emupath = ""
    if "nyx" in self.runmode:
      emupath = "Lya_emu{}{}{}{}{}.npz".format(self.emuname,"_lambda_P" if not ( "auv" in self.runmode) else "",("_{}".format(self.Anmode)) if (self.Anmode!='default') else "","_{}".format('noH') if not self.use_H else "","_{}".format('noOm') if not self.use_omm else "")
    if "lace" in self.runmode:
      emupath = self.emuname

    arguments = {
      'runmode':self.runmode,
      'An_mode':self.Anmode,
      'zmin':self.zmin, 'zmax':self.zmax, 'zs' : self.zlist_thermo,
      'emupath':emupath,
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

  def optionally_get_cosmo_or_nuisance(self, cosmo, cosmopar, parameters):

    grouped_params_options = {'default':[['A_lya','n_lya'],{'units':'Mpc','k_p':1,'normalize':False}], 'star':[['Delta_star','n_star','alpha_star'],{'units':'skm','k_p':0.009,'normalize':True}], 'post':[['Delta_lya_from_lym1d','n_lya_from_lym1d','alpha_lya_from_lym1d'],{'units':'Mpc','k_p':1,'normalize':True}]}

    # For each group, check if we need to do some cosmology evaluation, or if we have everything replaced by nuisances
    for group in grouped_params_options:
      params, options = grouped_params_options[group]
      # Check if everything replaced by nuisance
      needs_cosmo_eval = False
      for param in params:
        if self.has_cosmo[param] and not self.replace_with_nuisance[param]:
          needs_cosmo_eval = True
          self.log("Cosmological parameter not replaced (hence needs cosmo evaluations) :: ", param)
      # If not everything replaced by nuisances, get true cosmological parameter values from the cosmo object
      if needs_cosmo_eval:
        cosmo_parameters = self.postprocessing_A_and_n_lya(cosmo, **options)
      # Now iterate through parameters and assign
      for iparam, param in enumerate(params):
        if not self.has_cosmo[param]:
          continue
        # If it can be replaced, replace, otherwise, take from true cosmo object values
        if self.replace_with_nuisance[param]:
          cosmopar[param] = parameters[param+'_nuisance']
        else:
          cosmopar[param] = cosmo_parameters[iparam]

    if self.has_cosmo['sigma8'] and not self.replace_with_nuisance['sigma8']:
      cosmopar['sigma8'] = cosmo.sigma8()
    if self.has_cosmo['sigma8'] and self.replace_with_nuisance['sigma8']:
      cosmopar['sigma8'] = parameters['sigma8_nuisance']

    if self.has_cosmo['ns'] and not self.replace_with_nuisance['ns']:
      cosmopar['n_s'] = cosmo.n_s()
    if self.has_cosmo['ns'] and self.replace_with_nuisance['ns']:
      cosmopar['n_s'] = parameters['ns_nuisance']

    if self.has_cosmo['zreio'] and not self.replace_with_nuisance['zreio']:
      cosmopar['zreio'] = cosmo.get_current_derived_parameters(["z_reio"])["z_reio"]
    if self.has_cosmo['zreio'] and self.replace_with_nuisance['zreio']:
      cosmopar['zreio'] = parameters['zreio_nuisance']

    compute_lace = False
    if self.has_cosmo['Delta2_p'] and not self.replace_with_nuisance['Delta2_p']:
      compute_lace = True
    if self.has_cosmo['n_p'] and not self.replace_with_nuisance['n_p']:
      compute_lace = True
    if self.has_cosmo['alpha_p'] and not self.replace_with_nuisance['alpha_p']:
      compute_lace = True
    if compute_lace:
      lace_dict = self.pk_lace(cosmo)
      Delta2_p = lace_dict['Delta2_p']
      n_p = lace_dict['n_p']
      alpha_p = lace_dict['alpha_p']
    if self.has_cosmo['Delta2_p'] and self.replace_with_nuisance['Delta2_p']:
      Delta2_p = parameters['Delta2_p_nuisance']
    if self.has_cosmo['n_p'] and self.replace_with_nuisance['n_p']:
      n_p = parameters['n_p_nuisance']
    if self.has_cosmo['alpha_p'] and self.replace_with_nuisance['alpha_p']:
      alpha_p = parameters['alpha_p_nuisance']
    if self.has_cosmo['Delta2_p']:
      cosmopar['Delta2_p'] = Delta2_p
    if self.has_cosmo['n_p']:
      cosmopar['n_p'] = n_p
    if self.has_cosmo['alpha_p']:
      cosmopar['alpha_p'] = alpha_p

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
    from scipy.interpolate import CubicSpline
    zs = self.zlist_thermo
    fp = [cosmo.scale_independent_growth_factor_f(z) for z in zs]
    k_Mpc = np.geomspace(0.001,9.,num=1000)
    P_Mpc = cosmo.get_pk_all(k_Mpc,z=zs)
    kp_Mpc = 0.7
    return_array = fit_linP_Mpc_zs(k_Mpc, P_Mpc, fp, kp_Mpc, zs)

    pk_lace = {'Delta2_p':CubicSpline(self.zlist_thermo,[return_array[iz]['Delta2_p'] for iz,z in enumerate(self.zlist_thermo)]),'n_p': CubicSpline(self.zlist_thermo,[return_array[iz]['n_p'] for iz,z in enumerate(self.zlist_thermo)]),'alpha_p':CubicSpline(self.zlist_thermo,[return_array[iz]['alpha_p'] for iz,z in enumerate(self.zlist_thermo)])}
    return pk_lace

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
    ddw = (-1./scale/scale + ((x-x0)/scale/scale)**2)*np.exp(-0.5*(x-x0)*(x-x0)/scale/scale) #Unit = 1/scale^2
    s = np.trapz(w,x)
    r = np.trapz(y*w,x)/s
    dr = np.trapz(y*dw,x)/s
    ddr = np.trapz(y*ddw,x)/s
    A_lya_Mpc = np.exp(r)
    n_lya = dr
    alpha_lya = ddr
    # Unit conversion (either A_lya or Delta_lya)
    if not normalize:
      A_lya = A_lya_Mpc/unit**3
    else:
      A_lya = A_lya_Mpc*k_p_Mpc**3/(2*np.pi**2)
    return A_lya, n_lya, alpha_lya

  def log(self, msg, level=1):
    if level <= self.verbose:
      print(self.prefix+("\n"+self.prefix).join(msg.split("\n")))
