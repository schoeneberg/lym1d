"""
General Lyman-Alpha likelihood file, which does not depend on MontePython and can be integrated into other codes (like Cobaya)

The main object is the lya_2021_likelihood class, which implements all the necessary logic of the Lya likelihood.
Properties of lya_2021_likelihood:
  data_directory (str): Where the data files (such as emulators, observables) are stored, give an absolute path to be sure
  verbose (int): verbosity -- 0 should disable all output, 1 should print only the relevant info, while 2 will also print debug info
  runmode (str): A string describing properties of the run (
  An_mode (str): A string describing the A_lya/n_lya basis (normal ('default'), in s/km ('skm'), or sigma8+n_s ('sigma'))

@author Nils Schoeneberg (@schoeneberg)

"""

import numpy as np
import os
import numbers
from copy import deepcopy
from functools import partial

from .util import OptionDict
from .flux import FluxPrior
from .blinding import get_blindings

from .emulator import EmulatorOutOfBoundsException



from scipy.interpolate import CubicSpline as interp
from scipy.interpolate import interp1d as interp_lin

from scipy.linalg import block_diag

c_kms = 299792.458


# Different interpolation methods that can be used on the final emulator outputs
def interp_inv(x,y):
  return interp(x[::-1],y[::-1])
def interp_log(x,y,newx):
  return np.exp(interp_lin(np.log(x),np.log(y))(np.log(newx)))


name_LaCE = 'LaCE'
name_Nyx = 'Nyx_GP'
name_Taylor = 'taylor'

# The actual likelihood
class lym1d:

  An_parameters = {'default':['A_lya','n_lya','omega_m'],'post':['Delta_lya_from_lym1d','n_lya_from_lym1d','omega_m'], 'post_alpha':['Delta_lya_from_lym1d','n_lya_from_lym1d','alpha_lya_from_lym1d'], 'star':['Delta_star','n_star','omega_m'], 'star_alpha':['Delta_star','n_star','alpha_star']}

  # Initialization of all relevant quantities and computational methods
  def __init__(self, base_directory, **opts):

    # Store data directory for later use
    self.base_directory = base_directory

    # -> Load verbosity
    self.verbose = opts.pop('verbose',1)
    self.bounds_verbose = opts.pop('bounds_verbose',0)

    self.kmin = opts.pop('kmin',0.001)

    self.log("Initializing Emulator Lyman Alpha Likelihood (2021)")

    self.runmode = opts.pop('runmode','normal')
    models_path = opts.pop("models_path",'models.hdf5')
    data_path = opts.pop("data_path",'')
    self.data_directory = os.path.join(base_directory, data_path)
    emupath = opts.pop("emupath",'Lya_emu.npz')

    smartpath = opts.pop("smartpath", True)

    self.correct_nuisance_order = opts.pop('correct_nuisance_order',False)

    self.use_H = opts.pop('use_H',True)
    if "use_omm_or_alpha" in opts:
      self.use_omm_or_alpha = opts.pop('use_omm_or_alpha',True)
    else:
      #this is for backwards compatibility
      self.use_omm_or_alpha = opts.pop('use_omm',True)

    self.has_emu_cov = opts.pop("has_emu_cov",False)

    # -> Load options
    # 1) Number of bins
    self.NzAGN = opts.pop("NzAGN",9)
    self.zmin = opts.pop("zmin",0.)
    self.zmax = opts.pop("zmax",100.)

    # This is passed so it can be cross-checked with the data, and raise an exception otherwise
    self.zlist_to_check_against_data = opts.pop("zs",None)

    # 2) Which corrections are enabled?
    self.has_cor = OptionDict({'noise':True,'DLA':True,'reso':True,'SN':True,'AGN':True,'zreio':True,'SiIII':True,'SiII':True,'norm':True,'splice':False,'UV':False,'IC':False})
    if "has_cor" in opts:
      coropts = opts.pop('has_cor')
      if not coropts or coropts == "None" or coropts==False: #Signal flag for setting all corrections off
        for k,v in self.has_cor.items():
          self.has_cor[k]=False
      elif coropts=='True' or coropts == True:
        for k,v in self.has_cor.items():
          self.has_cor[k]=True
      elif isinstance(coropts,list):
        for k,v in self.has_cor.items():
          self.has_cor[k]=False
        for k in coropts:
          self.has_cor[k]=True
      else:
        self.has_cor.update(coropts) #Otherwise, the flags are set individually

    self.splice_kind = opts.pop('splice_kind',1)
    self.DLA_kind = opts.pop('DLA_kind',1)
    self.silicon_norm_kind = opts.pop('silicon_norm_kind',0)
    self.silicon_damping = opts.pop('silicon_damping',False)
    self.ic_cor_kind = opts.pop('ic_cor_kind',0)

    self.nuisance_parameters = self.get_nuisance_parameters()

    # 3) Fixed quantities (TODO :: update to more precise values?)
    self.dvSiII = 5577.0
    self.dvSiIII = 2271.0

    # 5) Data files
    self.data_filename = opts.pop('data_filename','pk_1d_DR12_13bins.out')
    self.inversecov_filename = opts.pop('inversecov_filename','pk_1d_DR12_13bins_invCov.out')
    self.agn_corr_filename = opts.pop('agn_corr_filename','AGN_corr.dat')

    # -> Load all relevant data (and set self.basis_z)
    self.load_data(data_format = opts.pop('data_format','DR14'), smartpath=smartpath)

    self.use_flux_prior = opts.pop("use_flux_prior",False)
    self.flux_prior_type = opts.pop("flux_prior_type","becker13")

    self.emu_options = opts.pop("emulator_options",{})
    self.An_mode = opts.pop('An_mode','default') # TODO ?? : promote to emulator options??

    # Check options are reasonable
    if self.has_cor['UV'] and not "taylor" in self.runmode:
      raise ValueError("Cannot use UV corrections in non-taylor mode")

    # Check all options are popped before building emulator (!)
    if opts:
      raise ValueError("There are unexpected remaining input options : '{}'".format(opts))

    # Now build emulator!
    self.build_emulator(emupath, models_path)

    # Optionally put flux prior
    if self.use_flux_prior:
      self.fluxprior = FluxPrior(self.basis_z,  priortype=self.flux_prior_type, verbose=self.verbose)

    # Done !


  def build_emulator(self, emupath, models_path):

    need_save = False

    runmode_conversion = {'taylor':name_Taylor, 'lace':name_LaCE, 'nyx':name_Nyx}
    found_runtypes_iter = iter([rt in self.runmode.lower() for rt in runmode_conversion])
    # The first any finds the first occurance, the other and checks that there is not a second one
    if any(found_runtypes_iter) and not any(found_runtypes_iter):
      self.emutype = [runmode_conversion[rt] for rt in runmode_conversion if rt in self.runmode.lower()][0]
    else:
      raise ValueError("Conflicting emulator in runmode, possible options are 'nyx','lace','taylor', but found '{}'!".format(self.runmode.lower()))

    self.log(f" -> Emulator type = {self.emutype}")

    # -> Build emulator
    # Once the emulator is constructed, it's easy to call it many many times, and relatively fast
    if self.emutype==name_Nyx:
      from .emulator_Nyx import Emulator_Nyx
      try:
        print("Loading from emupath = ", os.path.join(self.base_directory,emupath))
        self.emu = Emulator_Nyx.load(os.path.join(self.base_directory,emupath))
        self.log("Loaded Nyx emulator from "+emupath+"\nParameters: "+str(self.emu.parnames))
      except FileNotFoundError as fnfe:
        self.log("(!) No previous NYX-GP emulator found, creating a new one\n(!) [from {}](!)\nOriginal warning message : \n".format(os.path.join(self.base_directory,models_path))+str(fnfe))
        if not self.An_mode in self.An_parameters:
          raise ValueError("An_mode '{}' not recognized".format(self.An_mode))
        self.log("Constructing Nyx emulator")
        self.emu=Emulator_Nyx({'modelset':os.path.join(self.base_directory,models_path),'zmin':2.1,'zmax':5.6,'output_cov':self.has_emu_cov,'use_lP':not ('auv' in self.runmode),'use_H':self.use_H,'use_omm_or_alpha':self.use_omm_or_alpha,'A_lya_n_lya_alpha_lya':self.An_parameters[self.An_mode],'verbose':self.verbose>1})
        self.log("Constructed Nyx emulator")
        need_save=True

    # LaCE (GP or NN) emulator
    elif self.emutype==name_LaCE:
      from .emulator_LaCE import Emulator_LaCE
      try:
        self.log("Loading LaCE emulator for emulator name = {} , models_path = {}".format(emupath,os.path.join(self.base_directory,models_path)))
        self.emu = Emulator_LaCE.load(emupath, os.path.join(self.base_directory,models_path))
        self.log("Loaded LaCE emulator")
      except FileNotFoundError as fnfe:
        self.log("(!) No LaCE emulator found at {}, creating a new one\n(!) [from {}](!)\nOriginal warning message : \n".format(emupath,os.path.join(self.base_directory,models_path))+str(fnfe))
        self.log("Constructing LaCE emulator")
        lace_options = {}
        if 'lace_type' in self.emu_options:
          lace_options['lace_type'] = self.emu_options['lace_type']
          if self.emu_options['lace_type']=='nyx':
             lace_options['NYX_PATH'] = os.path.abspath(self.base_directory)
        self.emu=Emulator_LaCE(lace_options)
        self.log("Constructed LaCE emulator")
        need_save=True

    # Taylor emulator
    else:
      from .emulator_Taylor import Emulator_Taylor
      if self.zmax>4.61:
        raise ValueError(f"Taylor basis currently only defined for z<=4.6, but Lya_DESI.zmax={self.zmax}")
      self.emu=Emulator_Taylor({'path':os.path.join(self.base_directory,emupath)
        ,'zmin':0.0,'zmax':4.6,'fit_opts':{'FitNsRunningExplicit':False,'FitT0Gamma':('amplgrad' not in self.runmode),'useMnuCosm':True,'useZreioCosm':False,'CorrectionIC':self.has_cor['IC']},'verbose':self.verbose, **self.emu_options})

    # Also save emulator after creation
    if need_save:
      self.emu.save(os.path.join(self.base_directory,emupath))
      self.log("Emulator saved at "+str(os.path.join(self.base_directory,emupath)))

    # Now check to pass additional options
    if self.emutype==name_Nyx:
      self.emu.shortening_factor = self.emu_options.pop('shortening_factor',0)
      if self.emu.shortening_factor > 0.:
        self.log("Shortening factor = {}".format(self.emu.shortening_factor))
      self.emu.convex_hull_mode = self.emu_options.pop('convex_hull_mode',False)
      if self.emu.convex_hull_mode:
        self.log("Convex hull mode: {}".format(self.emu.convex_hull_mode))

    # Print some emulator params, if very verbose
    if self.emutype==name_Nyx:
      for i, (z, names, pars) in enumerate(zip(self.emu.redshifts, self.emu.emuparnames, self.emu.emupars)):
        self.log(f"Parameters for emulator index {i:d}, redshift {z:.2f} \n",level=3)
        self.log("   ".join(names)+"\n",level=3)
        self.log("   ".join([str(p) for p in pars])+"\n",level=3)

  def get_flux_pk(self, iz, z, cosmo, therm, nuisance):
    """
      Get the flux for a given redshift and given parameters. This is the UN-corrected flux pk directly from the emulator.

      Args:
        iz (int): Redshift index (in data) of the current redshift
        z (float): Current redshift
        cosmo (dict: (str,float/function)): Dictionary of cosmological quantities, either values or functions of redshift
        therm (dict: (str,float/function)): Dictionary of thermal quantities, either values or functions of redshift
        nuisance (dict: (str,float/function)): Dictionary of nuisance quantities, either values or functions of redshift
    """

    # Convert from input notation to emulator notation
    if self.emutype==name_Nyx:
      params = {'Fbar':therm['Fbar'](z),'T_0':therm['T0'](z),'gamma':therm['gamma'](z)}

      if not self.An_mode in self.An_parameters:
        raise ValueError("An_mode '{}' not recognized".format(self.An_mode))

      for name in self.An_parameters[self.An_mode]:
        params[name] = cosmo[name] - self.blindings[name]

      if self.use_H:
        params['H_0'] = cosmo['H0']
      if not 'auv' in self.runmode:
        params['lambda_P']=therm['lambdaP'](z)
      else:
        params['A_UVB']=therm['UV'](z)

    elif self.emutype==name_Taylor:
      params = {'sigma8':cosmo['sigma8'] - self.blindings['sigma8'],
                'n_s':cosmo['n_s'] - self.blindings['n_s'],
                'T0':therm['T0'](z),'gamma':therm['gamma'](z),
                'Omega_m':cosmo['Omega_m'],'H0':cosmo['H0'],
                'AmpTauEff':nuisance['AmpTauEff'],'SlopeTauEff':nuisance['SlopeTauEff'],
                'Omega_nu':cosmo['Omega_nu']}

      if "amplgrad" in self.runmode:
          params.update({'invAmpl':nuisance['invAmpl'],'invGrad':nuisance['invGrad']})
      if "fbar" in self.runmode:
          params.update({'Fbar':therm['Fbar'](z)})

    elif self.emutype==name_LaCE:
      params = {'Delta2_p':cosmo['Delta2_p'](z) - self.blindings['Delta2_p'],
                'n_p':cosmo['n_p'](z) - self.blindings['n_p'],
                'alpha_p':cosmo['alpha_p'](z) - self.blindings['alpha_p'],
                'mF':therm['Fbar'](z),
                'sigT_Mpc':9.1*np.sqrt(therm['T0'](z)/1e4)*(1+z)/cosmo['H(z)'](z),
                'gamma':therm['gamma'](z), 'kF_Mpc':1000./therm['lambdaP'](z)#therm['kF'](z)
                }

    # Check if parameters are in bounds
    try:
      self.emu.in_bounds(params,z)
    except EmulatorOutOfBoundsException as e:
      self.log("(!) Emulator out of bounds :: "+str(e), level=self.bounds_verbose)
      return None

    ## Honestly, it might be preferable to use H(z) for this, if we are considering non-LCDM models
    # Convert k from s/km to 1/Mpc, and convert P(k) from log(P(k)/Mpc) to P(k)/(km/s)
    k_conversion_factor = np.sqrt(1e4*cosmo['omega_m']*((1.+z)**3-1)+cosmo['H0']**2)/(1.+z)

    # Get P^flux(k) from simulator
    if self.emutype == name_LaCE:
      data_k_in_Mpc = self.data_k[iz] * k_conversion_factor
      sim_flux_pk, sim_cov = self.emu(params,z, k=data_k_in_Mpc)
    else:
      sim_flux_pk, sim_cov = self.emu(params,z)

    if self.has_emu_cov and sim_cov is None:
      raise ValueError("The used emulator that was loaded/built does not support giving a covmat")
    #print(params,z)
    # k , P_F(k) = self.get_karr, sim_flux_pk

    # Post-processing of the results !
    if self.emutype==name_Nyx:
      kemu = self.emu.get_karr(z)/k_conversion_factor
      pkemu = sim_flux_pk*k_conversion_factor
      if self.has_emu_cov:
        covemu = sim_cov * k_conversion_factor**2

      # This is a bit hacky, and we could probably get rid of it if we adopt the same convention as for LaCE
      if kemu[0]>self.data_k[iz][0]:
        newlowk = kemu[0]*0.5
        slope = np.log(pkemu[1]/pkemu[0])/np.log(kemu[1]/kemu[0])
        newlowpk = np.exp(np.log(pkemu[0])+slope*np.log(newlowk/kemu[0]))
        kemu = np.insert(kemu,0,newlowk)
        pkemu = np.insert(pkemu,0,newlowpk)
      self.sim_pk = interp_log(kemu,pkemu,self.data_k[iz])
      if self.has_emu_cov:
        from scipy.interpolate import interp2d
        self.sim_cov = interp2d(kemu, kemu, covemu, kind='cubic')(self.data_k[iz],self.data_k[iz])

    elif self.emutype==name_Taylor:
      # No unit conversion necessary (since Taylor emulator is in s/km) ! -- just change wavenumbers
      self.sim_pk = interp_log(self.emu.get_karr(z),sim_flux_pk,self.data_k[iz])

    elif self.emutype==name_LaCE:
      pkemu = sim_flux_pk*k_conversion_factor
      self.sim_pk = pkemu
      if self.has_emu_cov:
        self.sim_cov = sim_cov*k_conversion_factor**2

    return self.sim_pk



  def get_obs_pk(self,cosmo,therm,nuisance):
    """
      Get the OBSERVED flux for a given redshift and given parameters. This is the CORRECTED flux pk.

      Args:
        cosmo (dict: (str,float/function)): Dictionary of cosmological quantities, either values or functions of redshift
        therm (dict: (str,float/function)): Dictionary of thermal quantities, either values or functions of redshift
        nuisance (dict: (str,float/function)): Dictionary of nuisance quantities, either values or functions of redshift
    """

    self.theory_pk = [np.empty(self.Nkperbin[iz]) for iz in range(self.Nzbin)]
    if self.has_emu_cov:
      self.theory_cov = [np.empty(self.Nkperbin[iz], self.Nkperbin[iz]) for iz in range(self.Nzbin)]

    # 1) Loop over data points...
    for iz,z in enumerate(self.basis_z):

      z = self.basis_z[iz]
      if ( z >= self.zmin-self.epsilon_z and z <= self.zmax+self.epsilon_z):

        # 2) Get the raw flux P(k) from the emulator
        self.sim_pk = self.get_flux_pk(iz, z, cosmo, therm, nuisance)
        if self.sim_pk is None:
          return None

        # 3) Add corrections
        self.apply_corr_pk_at_z(iz,z,cosmo,therm,nuisance)

        # 4) Safe into vector
        self.theory_pk[iz] = self.sim_pk
        if self.has_emu_cov:
          self.theory_cov[iz] = self.sim_cov

    # 5) Return the written vector
    return self.theory_pk

  def get_data_pk(self):
    return self.data_pk



  def chi2(self,cosmo,thermo_in,nuisance_in, add_prior=False):
    """
      Get the chi^2 by comparing data and observation. WITHOUT additional thermal/nuisance priors.

      Args:
        cosmo (dict: (str,float/function)): Dictionary of cosmological quantities, either values or functions of redshift
        therm (dict: (str,float/function)): Dictionary of thermal quantities, either values or functions of redshift
        nuisance (dict: (str,float/function)): Dictionary of nuisance quantities, either values or functions of redshift
    """

    thermo = self.convert_thermal(thermo_in)

    nuisance = self.convert_nuisance(nuisance_in)

    # 1) Get observed P^flux(k) from the emulator/theory
    opk = self.get_obs_pk(cosmo,thermo,nuisance)
    if opk is None:
      return None

    if self.has_emu_cov:
      self.inv_covmat = np.linalg.inv(self.covmat + block_diag(*self.theory_cov))

    # 2) Obtain the effective chi square as dP_i C^(-1)_ij dP_j
    # Where the indices i,j run over BOTH redshift bins AND k bins
    chi_squared = np.dot(np.hstack(self.data_pk)-np.hstack(self.theory_pk),  #dP_i
                  np.dot(self.inv_covmat,                                    #C^(-1)_ij
                         np.hstack(self.data_pk)-np.hstack(self.theory_pk))) #dP_j

    if(chi_squared == None):
      return None

    if add_prior==True:
      self.log("Chi-square before priors: {}".format(chi_squared),level=3)
      chi_squared += self.prior(cosmo,thermo,nuisance)

    return chi_squared

  def chi2_plus_prior(self, cosmo, thermo, nuisance):
    return self.chi2(cosmo, thermo, nuisance, add_prior=True)











  def convert_thermal(self, therm):

    if 'Fbar' in therm and 'tau_eff' in therm:
      raise ValueError("Cannot pass both 'Fbar' and 'tau_eff' in thermal dictionary")

    if 'lambdaP' in therm and 'kF' in therm:
      raise ValueError("Cannot pass both 'lambdaP' and 'kF' in thermal dictionary")

    thermout = therm.copy()
    for par in ['T0','Fbar','tau_eff','gamma','kF','UV','lambdaP']:
      if par not in therm:
        continue
      thermout[par] = self.parinfo_to_function(therm[par], par)
    if 'tau_eff' in thermout:
      taueff = thermout.pop('tau_eff')
      thermout['Fbar'] = np.vectorize(lambda z: np.exp(-taueff(z)))
    if 'kF' in thermout:
      kF = thermout.pop('kF')
      thermout['lambdaP'] = np.vectorize(lambda z: 1000./kF(z))

    return thermout

  def convert_nuisance(self, nuisance):
    nuisanceout = nuisance.copy()
    for par in ['fSiIII','fSiII','a_damp','noise','normalization']:
      if par in nuisanceout:
        nuisanceout[par] = self.parinfo_to_function(nuisanceout[par],par)
    return nuisanceout






  def apply_corr_pk_at_z(self,iz,z,cosmo,therm,nuisance):
      """
        Apply all necessary corrections to the emulated flux P(k) at the given redshift with the given parameters

        Args:
          iz (int): Redshift index (in data) of the current redshift
          z (float): Current redshift
          cosmo (dict: (str,float/function)): Dictionary of cosmological quantities, either values or functions of redshift
          therm (dict: (str,float/function)): Dictionary of thermal quantities, either values or functions of redshift
          nuisance (dict: (str,float/function)): Dictionary of nuisance quantities, either values or functions of redshift
      """

      ks = self.data_k[iz]

      # 3) Add corrections
      #3.1) SPLICING CORRECTION
      if self.has_cor['splice']:
        corSplice = 1.
        if (self.splice_kind==1):
          #### TBC: TODO
          corSplice = 1.01 + nuisance['splicing_corr'] * ks
          #corSplice = 1.+self.SplicingOffset + self.SplicingCorr * k
        elif (self.splice_kind==2):
          z_p = 3.5
          if (z<z_p):
            k_p = 0.00244 + (0.00196-0.00244) *  (z-2.2)/(3.4-2.2)
            y_p = 1.0 + nuisance['splicing_offset']
            slope = np.choose(ks<k_p, [nuisance['splicing_corr'],-21])
          else:
            k_p = 0.00196 + (0.00188-0.00196) *  (z-3.4)/(4.4-3.4)
            y_p = 1.0 + nuisance['splicing_offset'] - 0.02 *  (z-z_p)/(4.4-z_p)
            slope = np.choose(ks<k_p, [nuisance['splicing_corr'] + np.abs(nuisance['splicing_corr']) *  (z-3.4)/(4.4-3.4),-27])
          corSplice = y_p + slope * (ks-k_p)

        self.sim_pk /= corSplice
      #3.2) NOISE CORRECTION
      if not self.correct_nuisance_order and self.has_cor['noise']:
        self.sim_pk += self.data_noise_pk[iz]*nuisance['noise'](z)

      #3.3) SYSTEMATICS CORRECTION
      if self.has_cor['DLA']:
        if self.DLA_kind == 1:
          corDLA = 1. - (1.0/(15000.0*ks-8.9) + 0.018)*0.2*  nuisance['DLA']
        elif self.DLA_kind == 2:
          zratio = (1+z)/(1+2.0)
          correction = (zratio**nuisance['DLA_d'] /
                    (
                      nuisance['DLA_a0']*zratio**nuisance['DLA_a1'] *
                      np.exp(nuisance['DLA_b0']*zratio**nuisance['DLA_b1']*ks) -1
                    ) **2
                    + nuisance['DLA_c0']*zratio**nuisance['DLA_c1'])
          corDLA = 1./(correction)
      else:
        corDLA = 1.

      if self.has_cor['reso']:
        # slope = z-dependence of resolution in units of (1km/s)^2 per delta_z = 1
        corReso =  np.exp( ks*ks * (nuisance['reso_ampl'] +(z-3.0)* nuisance['reso_slope']))
      else:
        corReso = 1

      if self.has_cor['SN']:
        k0=0.001
        k1=0.02
        #Supernovae SN
        tmpLowk=[-0.06,-0.04,-0.02]
        tmpHighk=[-0.01,-0.01,-0.01]
        if z < 2.5:
          d0 = tmpLowk[0]
          d1 = tmpHighk[0]
        elif z < 3.5:
          d0 = tmpLowk[1]
          d1 = tmpHighk[1]
        else:
          d0 = tmpLowk[2]
          d1 = tmpHighk[2]
        delta = d0 + (d1-d0)  *  (ks-k0)/(k1-k0)
        corSN = 1. + delta * nuisance['SN']
      else:
        corSN = 1.

      if self.has_cor['AGN']:
        if z <= np.max(self.AGN_z):
          delta = interp_lin(self.AGN_z,(self.AGN_expansion[:,0]+self.AGN_expansion[:,1]*np.exp(-self.AGN_expansion[:,2]*ks[:,None])))(z)
        else:
          AGN_upper = self.AGN_expansion[0,0]+self.AGN_expansion[0,1]*np.exp(-self.AGN_expansion[0,2]*ks)
          AGN_lower = self.AGN_expansion[1,0]+self.AGN_expansion[1,1]*np.exp(-self.AGN_expansion[1,2]*ks)
          z_upper = self.AGN_z[0]
          z_lower = self.AGN_z[1]
          delta = (AGN_upper-AGN_lower)/(z_upper-z_lower)*(z-z_upper)+AGN_upper
        corAGN = 1./(1.+delta* nuisance['AGN'])
      else:
        corAGN = 1.

      #3.3.5) REIO CORRECTIONS used for sterile paper 2015-2016
      #Correction estimate from McDonald 2005 (z_reio=7 -> z_reio=17)
      if self.has_cor['zreio']:
        zvalze=[2.1,3.2,4.0]
        Corrze = np.zeros((3, len(ks)))
        Corrze[0]= 1.001 - 1.11*ks + 15.7*ks*ks
        Corrze[1]= 1.009 - 2.29*ks + 9.39*ks*ks
        Corrze[2]= 1.029 - 3.74*ks + 4.62*ks*ks
        #Previous z_estim, also in the formulas below
        if(z<zvalze[1]):
          distz = (z-zvalze[0])/(zvalze[1]-zvalze[0]);
          corMcDo =  Corrze[0]*(1-distz) + Corrze[1]*(distz)
        else:
          distz = (z-zvalze[1])/(zvalze[2]-zvalze[1]);
          corMcDo =  Corrze[1]*(1-distz) + Corrze[2]*(distz)

        #Correction based on McDonald simulation (2005 paper) at zreioRef
        zreioRef = 12.0
        corZreio = 1./((corMcDo-1.0)*(cosmo['zreio']-zreioRef) / 10. + 1.0)
      else:
        corZreio = 1.

      #3.3.6) P_NYX_twofluidIC(k, z) = P_NYX_onefluidIC(k, z) /  correction(k, z)
      if self.has_cor['IC']:
        ic_corr_k = 0.003669741766936781
        if self.ic_cor_kind == 0:
          ic_corr_z = np.array([ 0.15261529, -2.30600644, 2.61877894])
          ancorIC = (ic_corr_z[0]*z**2 + ic_corr_z[1]*z + ic_corr_z[2]) * (1 - np.exp(-ks/ic_corr_k))
          corICs = (1-ancorIC/100)
        elif self.ic_cor_kind == 1:
          ic_corr_z = (nuisance['IC_A'] + nuisance['IC_B'] * (z-3) + nuisance['IC_C'] * (z-3)**2)
          corICs = (1- ic_corr_z * (1 - np.exp(-ks/ic_corr_k)))
      else:
          corICs = 1

      self.sim_pk /= corZreio * corDLA * corReso * corAGN * corSN * corICs

      #3.3.7) UV corrections (only for Taylor emulator)
      if not self.correct_nuisance_order and self.has_cor['UV']:
        self.sim_pk += self.emu.get_UV_corr(z,ks, nuisance['UVFluct'])

      #3.4) SI CORRECTION of correlation with Si-III and Si-II
      if self.has_cor['SiIII'] or self.has_cor['SiII']:
        if self.silicon_norm_kind==1:
          Fbar = therm['Fbar'](z)
        else:
          # this is how it was originally implemented in the Taylor likelihood
          # (we keep this option only for legacy)
          Fbar = np.exp(-self.taylor_tau_eff(z))
          if self.has_cor['norm']:
            Fbar *= np.sqrt(nuisance['normalization'](z))
        AmpSiIII = nuisance['fSiIII'](z) / (1.0-Fbar)
        AmpSiII  = nuisance['fSiII'](z) / (1.0-Fbar)

        if self.silicon_damping == True:
          a_damp, alpha_damp = nuisance['a_damp'](z), nuisance['alpha_damp']
          damping = (1+a_damp * ks)**alpha_damp * np.exp(-(a_damp * ks) ** alpha_damp)
        else:
          damping = 1

        if self.has_cor['SiIII']:
          self.sim_pk *= ( 1.0 + AmpSiIII*AmpSiIII *damping**2 + 2.0 * AmpSiIII * np.cos( ks * self.dvSiIII ) * damping )
        if self.has_cor['SiII']:
          self.sim_pk *= ( 1.0 +   AmpSiII*AmpSiII *damping**2 + 2.0 *  AmpSiII * np.cos( ks *  self.dvSiII ) * damping )

      #3.5) NORMALIZATION of flux
      if self.has_cor['norm']:
        self.sim_pk *= nuisance['normalization'](z)

      #3.6) ADDITIVE nuisance parameters
      if self.correct_nuisance_order and self.has_cor['noise']:
        self.sim_pk += self.data_noise_pk[iz]*nuisance['noise'](z)
      if self.correct_nuisance_order and self.has_cor['UV']:
        self.sim_pk += self.emu.get_UV_corr(z,ks, nuisance['UVFluct'])
      #Done with new power spectrum calculation + correction




  def prior(self,cosmo,therm,nuisance):
    """
      Give the default chi^2 of all the priors from the different effects.

      Args:
        cosmo (dict: (str,float/function)): Dictionary of cosmological quantities, either values or functions of redshift
        therm (dict: (str,float/function)): Dictionary of thermal quantities, either values or functions of redshift
        nuisance (dict: (str,float/function)): Dictionary of nuisance quantities, either values or functions of redshift
    """

    chi_squared = 0.

    #5.2) Add noise correction (10% DR9, 2% DR12)
    if self.has_cor['noise']:
      for iz in range(self.Nzbin):
        #noiseLevel=0.1    #DR9
        noiseLevel = 0.02  #DR12
        chi_squared += pow(nuisance['noise'](self.basis_z[iz])/noiseLevel,2.0)

    # Flat prior assumption => 1./sqrt(12) error from rectangular distribution
    #5.4) Add resolution correction (5 km/s)
    if self.has_cor['reso']:
      chi_squared += pow((nuisance['reso_ampl']-0.0)/1.0,2.0)
      chi_squared += pow((nuisance['reso_slope']-0.)*np.sqrt(12),2.0)
    # Astrophysical effects modeling
    if self.has_cor['DLA']:
      if self.DLA_kind == 1:
        chi_squared += pow((nuisance['DLA']-0.)*np.sqrt(12),2.0)
    if self.has_cor['SN']:
      chi_squared += pow((nuisance['SN']-1.)*np.sqrt(12),2.0)
    if self.has_cor['AGN']:
      chi_squared += pow((nuisance['AGN']-1.)*np.sqrt(12),2.0)
    if self.has_cor['UV']:
      chi_squared += pow((nuisance['UVFluct']-0.)*np.sqrt(12),2.0)
    if self.has_cor['zreio']:
      #prior arround zreio= 10 +/- 2 (Gaussian)
      chi_squared += pow((cosmo['zreio']-10.)/2.0,2.0)
    if self.has_cor['splice']:
      if (self.splice_kind==1):
        chi_squared +=  pow((nuisance['splicing_corr']-0.0)/2.5,2.0)
      elif(self.splice_kind==2):
        chi_squared +=  pow((nuisance['splicing_offset']-0.01)/0.05,2.0)
        chi_squared +=  pow((nuisance['splicing_corr']+0.9)/5.0,2.0)

    if self.use_flux_prior:
      chi_squared += self.fluxprior.chi_square(therm['Fbar'])
    return chi_squared




  def load_data(self, data_format = "DR14", smartpath = True):
    """
      Load the required data and covariance matrix from the file, making sure to cut it, and put it into the correct shapes
    """
    fpath = self.check_path(self.data_filename, smartpath=smartpath)
    if not data_format == 'Y1':
      fpath_icov = self.check_path(self.inversecov_filename, smartpath=smartpath)

    # -> Read power spectrum data (such as SDSS DR14 eBOSS P(k), or DESI EDR P(k))
    self.blindings = get_blindings(False) # Assume no blinding by default
    if data_format == "DR14":
      z,k,Pk,sPk,nPk,bPk,tPk = np.loadtxt(fpath,unpack=True)
      covdata = np.loadtxt(fpath_icov)
    elif data_format == "EDR":
      z,k,Pk,sPk = np.loadtxt(fpath,unpack=True)
      nPk,bPk,tPk = np.zeros_like(z),np.zeros_like(z),np.zeros_like(z)
      covdata = np.loadtxt(fpath_icov)
    elif data_format == "QMLE":
      with open(fpath) as f:
        lines = (line for line in f if not line.startswith('#'))
        names = next(lines).split()
        adic = dict(zip(names, np.loadtxt(lines).T))
      z, k, Pk, sPk, nPk = adic['z'],adic['kc'],adic['Pest'],adic['ErrorP'],adic['b']
      bPk,tPk = np.zeros_like(z),np.zeros_like(z)
      covdata = np.loadtxt(fpath_icov)
    elif data_format == "Y1":
      from astropy.io import fits
      hdul = fits.open(fpath)
      with fits.open(fpath) as hdul:
        if 'P1D' in hdul:
          dat = hdul['P1D'].data
        elif 'P1D_BLIND' in hdul:
          dat = hdul['P1D_BLIND'].data
        else:
          raise ValueError("Corrupted Y1 data format, does not contain a table with either 'P1D' or 'P1D_BLIND'.")
        # Attempt to read blinding scheme if exists!
        self.blindings = get_blindings(hdul)
        covdata = hdul['COVARIANCE'].data
      z, k, Pk, sPk, tPk = dat['Z'],dat['K'],dat['PLYA'],dat['E_STAT'],dat['E_SYST']
      nPk = (dat['PNOISE'] if 'PNOISE' in dat.names else np.zeros_like(z))
      bPk = np.zeros_like(z)
    else:
      raise ValueError("Unrecognized data format '{}'".format(data_format))

    # Get z values first: Note that the data z (z, sorted_unique_z, self.data_z) will have repeated entries usually, while the basis_z we construct has no repeated entries
    # When cutting down the data, make sure to use minimal difference between any two unique elements in the data as a little buffer for float comparisons
    sorted_unique_z = np.sort(list(set(z)))
    self.epsilon_z = min(0.01,0.1*np.min(np.abs(np.diff(sorted_unique_z))))
    data_zmask=(z>=self.zmin-self.epsilon_z)&(z<=self.zmax+self.epsilon_z)
    data_z=z[data_zmask]

    # Use the data redshifts to construct an array of (!) unique (!) z values over which we can iterate (data_z contains many copies)
    # Careful: compared to sorted_unique_z, here we only use those elements inside [zmin, zmax]
    self.basis_z = np.sort(list(set(data_z)))
    self.Nzbin = len(self.basis_z)

    # If the user provided a list of z values to be checked against the basis_z (like when they have a nuisance parameter for each z bin), then assert that they are both the same
    if self.zlist_to_check_against_data:
      # Use shorthand notation
      z_check = np.array(self.zlist_to_check_against_data)
      z_check = z_check[(z_check>=self.zmin-self.epsilon_z)&(z_check<=self.zmax+self.epsilon_z)]
      # Construct list of unique z values in data (do not allow for floating point differences)
      if len(self.basis_z)!=len(z_check):
        raise ValueError("Mismatching number of redshifts. The data file includes redshifts {}, while the redshifts provided to the likelihood are {}".format(self.basis_z, z_check))
      for zval in self.basis_z:
        number_of_matches = np.count_nonzero(np.isclose(zval,z_check))
        if number_of_matches==1:
          continue
        if number_of_matches==0:
          raise ValueError("Your redshifts provided to the likelihood do not correspond to the redshifts of the data. This is not yet supported. Problematic z={} of data cannot be found in provided z_list={} ".format(zval,z_check))
        else:
          raise ValueError("The data matches too many redshifts, check your provided z_list : {} (matching '{}' {} times)".format(z_check,zval,number_of_matches))

    # Report to the user how well the cutting of z values worked
    self.log(" -> Original array of z values in the data file : {} \n".format(sorted_unique_z)+
             " -> zmin = {}, zmax = {} (epsilon_z = {})\n".format(self.zmin, self.zmax, self.epsilon_z)+
             " -> Resulting basis of z values: {} ({} values)".format(self.basis_z,self.Nzbin))
    self.original_iz = np.array([np.arange(len(sorted_unique_z))[np.isclose(zval,sorted_unique_z)][0] for zval in self.basis_z])

    # Now we can use the rest of the file content

    # First: z-cut!
    data_k = k[data_zmask]
    data_noise_pk = nPk[data_zmask]
    data_spk = np.sqrt(sPk*sPk+tPk*tPk)[data_zmask] # sigma_stat^2 + sigma_sys^2
    data_pk = Pk[data_zmask]
    data_bpk = bPk[data_zmask]

    # Prepare for k-cut as well
    self.data_mask = data_zmask.copy()

    # Now, change the shape of all ingredients
    self.data_k, self.data_noise_pk, self.data_pk, self.data_spk, self.data_bpk = [], [], [], [], []
    self.Nkperbin = np.empty_like(self.basis_z,dtype=int)
    for iz,zval in enumerate(self.basis_z):
      # First, locate the z value indices within the flat array
      indexes_of_z_in_flattened_array = np.isclose(data_z,zval)

      # Now, get k values at this z
      k_values = data_k[indexes_of_z_in_flattened_array]

      # Perform k-cut
      if data_format == 'Y1':
        dlambda_Lya = 0.8 # in Angstrom :  DESI specification
        lambda_Lya = 1215.67 # in Angstrom : Lyman-Alpha wavelength
        dv = c_kms * dlambda_Lya/(lambda_Lya*(1+zval))
        kmax = 0.5 * np.pi/dv # Half the Nyquist frequency
        #kmax = 0.01955 # Old eBOSS cuts
        k_mask = np.logical_and(k_values > self.kmin, k_values < kmax)
      else:
        k_mask = np.ones_like(k_values, dtype=bool)

      # Now, store final data as ragged array
      self.data_k.append(k_values[k_mask])
      self.data_noise_pk.append(data_noise_pk[indexes_of_z_in_flattened_array][k_mask])
      self.data_pk.append(data_pk[indexes_of_z_in_flattened_array][k_mask])
      self.data_spk.append(data_spk[indexes_of_z_in_flattened_array][k_mask])
      self.data_bpk.append(data_bpk[indexes_of_z_in_flattened_array][k_mask])
      self.Nkperbin[iz] = np.count_nonzero(k_mask)

      # Now, get final mask for flat array (need to work with OVERALL uncut z array here!)
      self.data_mask[np.isclose(z,zval)] = k_mask

    # Let's make extra-sure we did everything correctly, and there were no issues here
    assert(np.array_equal(np.hstack(self.data_k),k[self.data_mask],equal_nan=True))
    assert(np.array_equal(np.hstack(self.data_pk),Pk[self.data_mask],equal_nan=True))
    assert(np.array_equal(np.hstack(self.data_spk),np.sqrt(sPk*sPk+tPk*tPk)[self.data_mask],equal_nan=True))

    # Declare inv covmat array (FLAT!)
    try:
      if data_format == 'Y1':
        self.inv_covmat = np.linalg.inv(covdata[:,self.data_mask][self.data_mask,:])
      else:
        self.inv_covmat = covdata[:,self.data_mask][self.data_mask,:]
    except IndexError as e:
      raise ValueError("something went wrong when reading the covariance matrix, are data file "
                       "and covariance file matching in length?") from e

    flag_nan_pk = np.any([np.any(np.isnan(pkz)) for pkz in self.data_pk])
    flag_nan_cov = np.any([np.any(np.isnan(covz)) for covz in self.inv_covmat])
    if flag_nan_pk or flag_nan_cov:
      raise ValueError("Invalid data file, containing NaN in the {}".format("Pk and covmat" if (flag_nan_pk and flag_non_cov) else ("Pk" if flag_nan_pk else "covmat")))

    #TODO: This could be cleaned up similar as above
    # -> Load AGN correction file
    if self.has_cor['AGN']:
      datafile = open(os.path.join(self.data_directory,self.agn_corr_filename),'r')
      # Declare AGN correction array
      self.AGN_z           = np.ndarray(self.NzAGN,'float')
      self.AGN_expansion   = np.ndarray((self.NzAGN,3),'float')
      for i in range(self.NzAGN):
        line = datafile.readline()
        values = [float(valstring) for valstring in line.split()]
        self.AGN_z[i] = values[0]
        self.AGN_expansion[i] = values[1:]
      datafile.close()

    if self.has_emu_cov:
      self.covmat = np.linalg.inv(self.inv_covmat)

  def check_path(self, path, smartpath = True):
    if os.path.exists(path) and smartpath:
      fpath = self.data_filename
    elif os.path.exists(os.path.join(self.data_directory,path)):
      fpath = os.path.join(self.data_directory,path)
    else:
      raise ValueError("Could not find the data at the supplied location : ",os.path.join(self.data_directory,path))
    return fpath





  # Function to convert information about a parameter (its value, a list of values for each redshift, a functional form, or a dictionary describing a powerlaw) into a functional form
  # parinfo = number/callable/dictionary to be converted to function
  # parname = parameter name if error message is thrown
  def parinfo_to_function(self, parinfo, parname):

    if callable(parinfo):
      output = parinfo
    elif isinstance(parinfo,numbers.Number) and not isinstance(parinfo,bool):
      output = lambda z:parinfo
    elif isinstance(parinfo, (list, np.ndarray)):
      output = interp_lin(self.basis_z, parinfo)
    else:
      if not isinstance(parinfo,dict):
        raise ValueError("Expected parameter '{}' to be a number, a list, callable or dictionary.".format(parname))
      if 'amp' not in parinfo:
        raise ValueError("Excpeted 'amp' parameter in dictionary for '{}'".format(parname))
      amp = parinfo.pop('amp')

      if 'slope_inf' in parinfo and 'slope' in parinfo:
          raise ValueError("Cannot have 'slope' and 'slope_inf' in dictionary for '{}'".format(parname))
      if 'break' in parinfo and 'slope_sup' in parinfo:
        raise ValueError("Cannot have 'break' and 'slope_sup' in dictionary for '{}'".format(parname))
      if 'slope_inf' in parinfo:
        slope = parinfo.pop('slope_inf')
      else:
        slope = parinfo.pop('slope',0)
      if 'slope_sup' in parinfo:
        slope_break = parinfo.pop('slope_sup')-slope
      else:
        slope_break = parinfo.pop('break',0)
      zpiv = parinfo.pop('z_piv',3)
      if len(parinfo)>0:
        raise ValueError("Too many entries in dictionary for '{}'. Unread: '{}'".format(parname,parinfo))

      powerlaw = lambda amp,slope,slope_break,zpiv,z: amp*pow((1+z)/(1+zpiv), slope if z<=zpiv else slope+slope_break)
      output = partial(powerlaw,amp,slope,slope_break,zpiv)

    # Wrap in vectorize for easier pickling and allowing to call at multiple redshifts
    return np.vectorize(output)




  def get_nuisance_parameters(self):
    # A simple function to check which nuisance parameters will need to be passed in the 'nuisance' dictionary
    parameters = []
    if self.has_cor['splice']:
      if (self.splice_kind==1):
        parameters.append('splicing_corr')
      elif (self.splice_kind==2):
        parameters.append('splicing_offset')
        parameters.append('splicing_corr')
    if self.has_cor['noise']:
      parameters.append('noise')
    if self.has_cor['DLA']:
      if self.DLA_kind == 1:
        parameters.append('DLA')
      elif self.DLA_kind == 2:
        parameters.append('DLA_a0')
        parameters.append('DLA_a1')
        parameters.append('DLA_b0')
        parameters.append('DLA_b1')
        parameters.append('DLA_c0')
        parameters.append('DLA_c1')
        parameters.append('DLA_d')
    if self.has_cor['reso']:
      parameters.append('reso_ampl')
      parameters.append('reso_slope')
    if self.has_cor['SN']:
      parameters.append('SN')
    if self.has_cor['AGN']:
      parameters.append('AGN')
    if self.has_cor['UV']:
      parameters.append('UVFluct')
    if self.has_cor['SiIII'] or self.has_cor['SiII']:
      parameters.append('fSiIII')
      parameters.append('fSiII')
      if self.silicon_damping:
        parameters.append('a_damp')
        parameters.append('alpha_damp')
    if self.has_cor['IC'] and self.ic_cor_kind==1:
      parameters.append('IC_A')
      parameters.append('IC_B')
      parameters.append('IC_C')
    if self.has_cor['norm']:
      parameters.append('normalization')
    if 'amplgrad' in self.runmode:
      parameters.append('invAmpl')
      parameters.append('invGrad')
    # This is where WDM nuisance could be added
    return parameters

  def taylor_tau_eff(self,z):
    if not hasattr(self, "_taylor_tau_eff_interp_function"):
      z_therm = [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6]
      tau_therm = [0.184924702397, 0.231425921518, 0.285929695332, 0.349252333224, 0.422242531447, 0.505780851386, 0.600779232412, 0.708180535481, 0.828958114192, 0.9641154105, 1.1146855727, 1.28173109358, 1.46634346696]
      self._taylor_tau_eff_interp_function = interp_lin(z_therm, tau_therm)
    return self._taylor_tau_eff_interp_function(z)


  def log(self, msg, level=1):
    if level <= self.verbose:
      print("[lym1d] "+"\n[lym1d] ".join(msg.split("\n")))
