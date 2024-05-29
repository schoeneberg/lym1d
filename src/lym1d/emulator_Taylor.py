import numpy as np
import os
from .emulator import EmulatorBase

class Emulator_Taylor(EmulatorBase):
  # Initialization routine

  """
    Read all necessary power spectra files, and compute derivatives of Taylor expansion

    Each power spectrum file has the contents:

    z value, k_value, P(k)=Pk, sigma_P(k) = sPk

    It will always be ordered first by the k value (growing),
    and then the groups of Pk by k-value will be ordered by z value (growing)

  """

  def __init__(self, args):

    """
      Emulator initialization. Arguments provided as the args dictionary
      Args:
        'path' (str) is the directory of the files
        'verbose' (int) verbosity of the emulator
        'taylor_opts' (dict: (str, bool)) is a dictionary of options for the input of the Taylor emulator.
          Accepted options are 'DLNorma' for normalized simulation files,
            'DL100k' for the extended simulation files,
            'WDM' for warm dark matter simulation files.
        'fit_opts' (dict: (str,bool)) is a dictionary of options for the precise treatment of cosmological inputs
          Accepted options are 'FitNsRunningExplicit=True' in order to modify 'n_s' as a function of k instead of using alpha_s as an additional input,
            'FitT0Gamma=True' to use T0 and Gamma as an input instead of tauA, tauS
            'useMnuCosm=True' to use the 'Omega_nu' input parameter for neutrinos. If False, attempts to use that given by montepython (and likely fails, depracated)
            'useZreioCosm=True' does nothing currently (depracated),
            'CorrectionIC=True' does nothing currently (depracated),
            'Fbar_free=True' to use the Fbar parameter in each redshift to recompute tauA instead of relying on the powerlaw approximation
        'extend_opts' (list str):
          Can contain up to three entries with the names of additional parameters to include on top of the default ones
            This includes the running 'alpha_s',
              the reionization redshift 'z_reio',
              and the effective neutrino number 'N_eff'
        'new_central_model_file' (str): while the Taylor emulator computes all derivatives as before, the central P1D can be taken from a file different from the bestguess.
    """
    # PARSE INPUT ARGUMENTS
    if not isinstance(args, dict):
      raise TypeError("The provided 'args' argument must be a dict, but got {} instead".format(type(args)))
    self.verbose = args.get('verbose',1)

    if self.verbose > 0:
      print("Initializing BOSS Lyman Alpha Emulator (2019)")

    ### FIXED parameters independent of input
    self.Nredshiftbin = 13
    self.Nkperbin = 35
    self.NzAGN = 9
    self.nuDerivPrecise = False
    # central values for Taylor expansion
    self.sigma8 = 0.83
    self.ns = 0.96
    self.invAmpl0 = -2.6/(1.0+2.6)
    self.invGrad0 = 5.55/(5.55+0.45)
    self.tauA = 0.0025
    self.tauS = 3.7
    self.Omegam = 0.31
    self.H0 = 67.5
    self.nuMass = 0.0
    self.nuMass_WDM = 0.0
    self.Neff = 3.046
    self.nsrun = 0.0
    self.zreio = 12.0
    # variation in derivative for Taylor expansion
    self.dsigma8 = 0.05
    self.dns = 0.05
    self.dinvAmpl = 13./90.
    self.dinvGrad = 5.55/66.
    self.dtauA = 0.0020
    self.dtauS = 0.4
    self.dOmegam = 0.05
    self.dH0 = 5.
    self.dnuMass = 0.4
    self.dnuMass_WDM = 0.2
    self.dNeff = 1.0
    self.dnsrun = 0.04
    self.dzreio = 4.0

    # TODO :: This can be simplified, a lot of options are depracated
    self.data_directory = args.get('path')
    if self.data_directory is None:
      raise ValueError("Need to specify a path")
    elif not isinstance(self.data_directory,str):
      raise TypeError("Need to specify path as string")
    self.taylor_opts = args.get('taylor_opts',{'DLNorma':True,'DL100k':True,'WDM':False})
    self.fit_opts = args.get('fit_opts',{'FitNsRunningExplicit':False,'FitT0Gamma':True,'useMnuCosm':True,'useZreioCosm':False,'CorrectionIC':False})
    if 'Fbar_free' not in self.fit_opts:
      self.fit_opts['Fbar_free'] = False
    self.extend_opts = args.get('extend_opts',[])

    self.WDM = self.taylor_opts['WDM']
    self.varExtend = len(self.extend_opts)>0

    # Check options
    if self.verbose > 0:
      print("General Options : ")
      print("DLNorma = {}, DL100k = {}".format(self.taylor_opts['DLNorma'],self.taylor_opts['DL100k']))
      print("WDM = {}".format(self.WDM))
      print("Fitting options :")
      print("NsrunExpl = {}, T0Gam = {}".format(self.fit_opts['FitNsRunningExplicit'],self.fit_opts['FitT0Gamma']))
      print("zreioCosm = {}, mnuCosm = {}".format(self.fit_opts['useZreioCosm'],self.fit_opts['useMnuCosm']))
      #print("H0prior = {}, GammaPrior = {}, OldAGNSN = {}".format(self.H0prior,self.NewGammaPrior,self.OldAGNandSN))
      print("Var extend = {} ".format(self.extend_opts))

    assert(not (self.taylor_opts['DLNorma'] and not self.taylor_opts['DL100k']))
    assert(not (self.varExtend and not self.taylor_opts['DLNorma']))
    assert(not (self.WDM and not self.taylor_opts['DL100k']))
    assert(not (self.WDM and self.nuDerivPrecise))
    assert(not (self.WDM and self.varExtend))

    # Cosmological and Astrophysical parameter names (as used in the files)
    if self.taylor_opts['DLNorma']:
      if self.varExtend:
        self.param_name = np.array(['sig8','ns','t0','gamma','omegam','h0','tauA','tauS','Neff','nrun','zreio','mnu'])
      else:
        self.param_name = np.array(['sig8','ns','t0','gamma','omegam','h0','tauA','tauS','mnu'])
    else:
      self.param_name = np.array(['sig8','ns','t0','gamma','omegam','h0','mnu'])
    self.param_num = len(self.param_name)

    #########################################################################
    #
    #  Prepare the Taylor expansion central and derivative values
    #  They are stored in arrays "basis" and "step" respectively
    #
    #########################################################################

    # Parameter central values
    self.basis = np.ndarray(self.param_num,'float')
    self.basis[0] = self.sigma8
    self.basis[1] = self.ns
    self.basis[2] = self.invAmpl0 #The taylor expansion is ACTUALLY in invAmpl, not T0
    self.basis[3] = self.invGrad0 #The taylor expansion is ACTUALLY in invGrad, not gamma
    self.basis[4] = self.Omegam
    self.basis[5] = self.H0
    if(self.taylor_opts['DLNorma']):
      self.basis[6] = self.tauA
      self.basis[7] = self.tauS
    if(self.varExtend):
      self.basis[8] = self.Neff
      self.basis[9] = self.nsrun
      self.basis[10] = self.zreio
    if(self.WDM):
      self.basis[self.param_num-1] = self.nuMass_WDM
    else:
      self.basis[self.param_num-1] = self.nuMass

    # Parameter deviations for derivatives
    self.step = np.ndarray(self.param_num,'float')
    self.step[0] = self.dsigma8
    self.step[1] = self.dns
    self.step[2] = self.dinvAmpl #The taylor expansion is ACTUALLY in invAmpl, not T0
    self.step[3] = self.dinvGrad #The taylor expansion is ACTUALLY in invGrad, not gamma
    self.step[4] = self.dOmegam
    self.step[5] = self.dH0
    if(self.taylor_opts['DLNorma']):
      self.step[6] = self.dtauA
      self.step[7] = self.dtauS
    if(self.varExtend):
      self.step[8] = self.dNeff
      self.step[9] = self.dnsrun
      self.step[10] = self.dzreio
    if(self.WDM):
      #In the WDM case, the expansion is of stepsize dnuMass=0.2, not dnuMass=0.4
      self.dnuMass = self.dnuMass_WDM
    self.step[self.param_num-1] = self.dnuMass

    #########################################################################
    #
    #  Prepare additional arrays that will be used eventually
    #
    #########################################################################

    # Number of TOTAL DATA points (redshift bins x k bins)
    self.np = self.Nredshiftbin * self.Nkperbin

    # Declare vectors and matrices for basis point of Taylor expansion
    self.basis_k         = np.ndarray(self.np,'float')            # basis point k values
    self.basis_pk        = np.ndarray(self.np,'float')            # basis point Pk values
    self.basis_z         = np.ndarray(self.Nredshiftbin,'float')  # basis point z values
    self.basis_tau       = np.ndarray(self.Nredshiftbin,'float')  # basis point optical depth value
    self.basis_T0        = np.ndarray(self.Nredshiftbin,'float')  # basis point T0 values
    self.basis_gamma     = np.ndarray(self.Nredshiftbin,'float')  # basis point gamma values


    # Declare Taylor expansion parameters
    self.newval = np.zeros(self.param_num,'float')                # New taylor parameter values at which likelihood is evaluated
    self.delta = np.ndarray(self.param_num,'float')               # Difference in taylor parameter values with respect to basis (i.e. central) values

    # Declare derivatives related to Taylor expansion
    self.derivPk         = np.ndarray((self.np,2*self.param_num),'float') # Derivatives up to order 2
    self.derivCrossPk    = np.ndarray((self.np,self.param_num,self.param_num),'float') # Cross-derivatives
    self.inv_covmat      = np.ndarray((self.np,self.np),'float')  # Inverse Pk covmat in parameter-space

    # Declare actual Taylor spectrum
    self.taylor_pk       = np.ndarray(self.np,'float')            # Pk values at new position (from Taylor)

    # Declare nuisance parameter arrays
    self.normalization   = np.ndarray(self.Nredshiftbin,'float')
    self.tauError        = np.ndarray(self.Nredshiftbin,'float')
    self.noise           = np.ndarray(self.Nredshiftbin,'float')
    # Declare AGN correction array
    self.AGN_z           = np.ndarray(self.NzAGN,'float')
    self.AGN_expansion   = np.ndarray((self.NzAGN,3),'float')

    #########################################################################
    #
    #  Read the best fit file as a power spectrum
    #  It will always be ordered first by the k value (growing),
    #  and then the groups of Pk by k-value will be ordered by z value (growing)
    #
    #########################################################################

    # Read the best fit power spectrum now
    bestguess_filename = "expansion/best_guess_power_spectrum"
    if(self.taylor_opts['DL100k']):
      bestguess_filename = "expansion100k/best_guess_0_99999_power_spectrum"
    if(self.taylor_opts['DLNorma']):
      bestguess_filename = "expansionNorma100k/best_guess_0_99999_power_spectrum_normalised"

    datafile = open(os.path.join(self.data_directory,bestguess_filename),'r')
    for i in range(self.np):
      line = datafile.readline()
      values = [float(valstring) for valstring in line.split()]

      z,k,Pk,sPk = values
      self.basis_k[i] = k
      self.basis_pk[i] = Pk
    datafile.close()

    # Construct an alias to basis_k for external use
    self.karr = self.basis_k

    #########################################################################
    #
    #  Read the best fit thermal history file
    #  It will always be ordered by z value (growing)
    #
    #########################################################################

    # Read the bestdata fit thermal history file now
    bestthermal_filename = "expansion/best_guess_thermo"
    if(self.taylor_opts['DL100k']):
      bestthermal_filename = "expansion100k/best_guess_0_99999_thermo"
    if(self.taylor_opts['DLNorma']):
      bestthermal_filename = "expansionNorma100k/best_guess_0_99999_thermo_normalised"

    datafile = open(os.path.join(self.data_directory,bestthermal_filename),'r')
    for i in range(self.Nredshiftbin):
      line = datafile.readline()
      values = [float(valstring) for valstring in line.split()]
      self.basis_z[i]      = values[0]
      self.basis_tau[i]    = values[1]
      self.basis_T0[i]     = values[2]
      self.basis_gamma[i]  = values[3]
    datafile.close()

    #########################################################################
    #
    #  Read the Taylor expansions of the thermal histories for fitting T0,Gamma
    #  They will always be ordered by z value (growing)
    #
    #########################################################################

    # Read t0 and gamma thermal history files for fitting T0,Gamma
    fitthermal_filename = "expansion/%s_thermo"
    if(self.taylor_opts['DL100k']):
      fitthermal_filename = "expansion100k/%s_0_99999_thermo"
    if(self.taylor_opts['DLNorma']):
      fitthermal_filename = "expansionNorma100k/%s_0_99999_thermo_normalised"

    fitthermalnames = ["t0+","t0-","gamma+","gamma-"]
    fitthermalpositions = [2,2,3,3] #t0,t0 , gamma,gamma  - columns of thermal file
    self.T0GFit = {}
    for j,fitthermalname in enumerate(fitthermalnames):
      datafile = open(os.path.join(self.data_directory,fitthermal_filename % fitthermalname),'r')
      self.T0GFit[fitthermalname] = np.zeros(self.Nredshiftbin)
      for i in range(self.Nredshiftbin):
        line = datafile.readline()
        values = [float(valstring) for valstring in line.split()]
        self.T0GFit[fitthermalname][i] = values[fitthermalpositions[j]]
      datafile.close()

    #########################################################################
    #
    #  Read the AGN corrections file
    #
    #########################################################################

    agn_corr_filename = "AGN_corr.dat"
    datafile = open(os.path.join(self.data_directory,agn_corr_filename),'r')
    for i in range(self.NzAGN):
      line = datafile.readline()
      values = [float(valstring) for valstring in line.split()]
      self.AGN_z[i] = values[0]
      self.AGN_expansion[i] = values[1:]
    datafile.close()

    #########################################################################
    #
    #  Compute derivatives
    #
    #########################################################################

    self._computeDerivatives()

    self.new_central_model_file = args.get('new_central_model_file')
    if self.new_central_model_file is not None:
      # self.basis_pk is changed: done only once derivatives were computed!
      print("Using new central model file", self.new_central_model_file)
      datafile = open(self.new_central_model_file,'r')
      for i in range(self.np):
        line = datafile.readline()
        values = [float(valstring) for valstring in line.split()]
        z,k,Pk,sPk = values
        self.basis_pk[i] = Pk
      datafile.close()

    print("Finished initializing BOSS Lyman Alpha Emulator (2019)")
    pass
    #End of initialization routine




  def __call__(self, args, z, karr=None):
    """
    Evaluate the emulator at given parameters (args) and redshift (z), and possibly wavenumbers (k)

    Args:
      'args' (dict(str,float)) corresponds to the set of parameters at which to emulate
      'z' (float) is the redshift at which to emulate
      'k' (array(float), opt) are the wavenumbers in 1/Mpc at which to evaluate the emulator.
          If None are provided, the output is at the internally defined karr

    Returns:
        array(float): emulated power spectrum (without log) in ! s/km !
    """
    if not isinstance(args,dict):
      raise TypeError("The arguments need to be a dictionary.")
    #########################################################################
    #
    #  Read nuisance parameters and obtain Taylor spectrum
    #
    #########################################################################

    useind=np.abs(self.basis_z-z)<0.01
    if np.any(useind):
        useind=useind.nonzero()[0][0]
    else:
        raise ValueError("Invalid redshift: {} not in {}".format(z,self.basis_z))
    ih = useind

    # 2) Loop over data points...
    for ik in range(self.Nkperbin):
      kx = ih*self.Nkperbin + ik

      k = self.basis_k[kx]
      z = self.basis_z[ih]

      # 2.1) Find the new Taylor parameters from the provided cosmological/astrophysical parameters
      self._assign_parameters(args,k,z,ih)

      # 2.2) Get P(k) from Taylor expansion in the Taylor parameters
      self._getPk(kx)

      #Done with new power spectrum calculation + correction
    if karr is None:
      return self.taylor_pk[ih*self.Nkperbin:(ih+1)*self.Nkperbin], None
    else:
      return CubicSpline(self.basis_k,self.taylor_pk)(karr), None



  @classmethod
  def load(cls, path):
    """ Loading the emulator is currently not implemented (and not rebuilding slow enough to be required) """
    raise NotImplementedError("Load not implemented for Taylor emulator yet")

  def save(self, path):
    """ Saving the emulator is currently not implemented (and not rebuilding slow enough to be required) """
    return

  def in_bounds(self, args, z):
    """ The current emulator assumes always to be in bounds due to the nature of the Taylor approach.
          This could be restricted in the future"""
    # Never raises an exception --> Assume always in bounds
    return

  def get_karr(self, z):
    """ Small special utility function to get only the k values for a given z value. UNITS s/km !!"""
    iz=np.abs(self.basis_z-z)<0.01
    if np.any(iz):
        iz=iz.nonzero()[0][0]
    else:
        raise ValueError("Invalid redshift: {} not in {}".format(z,self.basis_z))
    return self.karr[iz*self.Nkperbin:(iz+1)*self.Nkperbin]

  def get_UV_corr(self, z,k, UV):
    """ Small special utility function to get UV corrections from the Taylor emulator that are only applied in this Taylor emulator case"""
    iz=np.abs(self.basis_z-z)<0.01
    if np.any(iz):
        iz=iz.nonzero()[0][0]
    else:
        raise ValueError("Invalid redshift: {} not in {}".format(z,self.basis_z))
    return self.basis_pk[iz*self.Nkperbin+self.Nkperbin//2] * 0.05 * UV

  def get_taueff(self, z, amptaueff, slopetaueff):
    ''' Returns tau_eff(z) for a given value of (AmpTauEff, SlopeTauEff)'''
    iz=np.abs(self.basis_z-z)<0.01
    return amptaueff * (1+z)**slopetaueff + 0.5 * np.log(self.normalization[iz])

  def get_taueff_frombasis(self,z):
    ''' Returns tau_eff(z) in the basis model'''
    iz=np.abs(self.basis_z-z)<0.01
    return self.basis_tau[iz] + 0.5 * np.log(self.normalization[iz])

  #Compute Derivatives routine

  """
    Computes the derivatives with respect to all Taylor parameters of the grid
    This includes
      The amplitude of the final power spectrum averaged over 8Mpc/h
      sigma 8 -> sig8
      The tilt of the primordial power spectrum at the pivot scale
      n s -> ns
      Astrophysical temperature of the IGM at no overdensity
      T 0 -> t0
      Astrophysical slope of the temperature-density relation of the IGM
      gamma -> gamma
      The fractional total matter content of the universe today
      omega m -> omegam
      The current expansion rate of the universe
      H 0 -> h0

      Amplitude of the effective optical depth (as power law in redshift)
      A tau -> tauA
      Slope of the effective optical depth (as power law in redshift)
      S tau -> tauS

      Effective number of relativistic degrees of freedom in non-photon species (neutrinos and others)
      N eff -> Neff
      Running of the primordial power spectrum tilt
      alpha s -> nrun
      Redshift of reionization
      z reio -> zreio

      Neutrino mass / inverse warm dark matter mass
      m_nu -> mnu   /    1/m_WDM -> inv_wdm_mass

    The derivatives with respect to these quantities are calculated and tabulated
    Later, they can be used to estimate the power spectra at parameter points
    previously not evaluated by N-body simulations
  """
  def _computeDerivatives(self):

    # Select the correct set of spectra (using these tags)
    if self.taylor_opts['DLNorma']:
      pretag = 'Norma100k'
      posttag = '_0_99999_power_spectrum_normalised'
    elif self.taylor_opts['DL100k']:
      pretag = '100k'
      posttag = '_0_99999_power_spectrum'
    else:
      pretag = ''
      posttag = '_power_spectrum'

    # Declare temporary arrays
    # Only the positive variations are being used for cross derivatives
    valP =                np.ndarray((self.np,self.param_num),'float')
    # The neutrino variations have special form, thus get their own arrays
    nuP =                 np.ndarray(self.np,'float')
    nuPP =                np.ndarray(self.np,'float')

    positive_name = "expansion%s/%s+%s"       # Single parameter +
    if(self.WDM):
      WDM_name = "expansion%s/ms_inv_0.4%s"         # WDM mass            (+0.0)
      positive_WDM_name = "expansion%s/ms_inv+%s"   # WDM mass +          (+0.2)
      doublepos_WDM_name = "expansion%s/ms_inv++%s" # WDM mass ++         (+0.4)
    else:
      positive_nu_name = "expansion%s/%s+%s"       # Neutrino mass +      (+0.4)
      doublepos_nu_name = "expansion%s/%s++%s"     # Neutrino mass ++     (+0.8)
    negative_name = "expansion%s/%s-%s"       # Single parameter -
    crosspos_name = "expansion%s/%s+_%s+%s"   # Two parameters, each +
    if(self.WDM):
      crosspos1_WDM_name = "expansion%s/%s+_ms_inv_04%s" # Parameter +, WDM mass +
      crosspos2_WDM_name = "expansion%s/ms_inv_04_%s+%s" # WDM mass +, Parameter +
    else:
      crosspos1_nu_name = "expansion%s/%s+_%s++%s" # Parameter +, neutrino mass ++
      crosspos2_nu_name = "expansion%s/%s++_%s+%s" # neutrino mass ++, Parameter +

    # First, deal with all parameters except for neutrino mass
    # (Neutrino mass has no negative direction (m>0), only a positive direction)
    for ivar in range(self.param_num-1):
      pkThFileP = open(os.path.join(self.data_directory,( positive_name % (pretag,self.param_name[ivar],posttag))),'r')
      pkThFileM = open(os.path.join(self.data_directory,( negative_name % (pretag,self.param_name[ivar],posttag))),'r')
      for i in range(self.np):
        # We are going to assume the k bins and z bins remain the same for all spectra

        # -> In positive + direction
        line = pkThFileP.readline()
        values = [float(valstring) for valstring in line.split()]
        valP[i,ivar] = values[2]     #We only care about the actual power spectrum value
        # -> In negative - direction
        line = pkThFileM.readline()
        values = [float(valstring) for valstring in line.split()]
        valM = values[2]             #We only care about the actual power spectrum value

        # First and second derivative, assuming equal spacing of + and - from the central value
        self.derivPk[i,ivar*2] = (valP[i,ivar]-valM)/2/self.step[ivar]
        self.derivPk[i,ivar*2+1] = (valP[i,ivar]+valM-2.*self.basis_pk[i])/self.step[ivar]/self.step[ivar]

      pkThFileP.close()
      pkThFileM.close()

    # Second, deal with all neutrinos
    #   This is done only if the relevant files are there
    #   (else, only the basic CDM grid is loaded)
    ivar = self.param_num-1
    self.full_grid = True
    if self.WDM:
      file_test = os.path.join(self.data_directory, ( WDM_name % (pretag,posttag)))
    else:
      file_test = os.path.join(self.data_directory, ( doublepos_nu_name % (pretag,self.param_name[ivar],posttag)))
    if not os.path.isfile(file_test):
        self.full_grid = False
        print("Warning, emulator_Taylor: loading the LCDM grid only, no neutrinos/extensions.")

    if self.full_grid:
      if self.WDM:
        pkThFileP   = open(os.path.join(self.data_directory, ( doublepos_WDM_name % (pretag,posttag))),'r')
        pkThFilePP  = open(os.path.join(self.data_directory, ( WDM_name % (pretag,posttag))),'r')
      else:
        pkThFileP   = open(os.path.join(self.data_directory, ( positive_nu_name % (pretag,self.param_name[ivar],posttag))),'r')
        pkThFilePP  = open(os.path.join(self.data_directory, ( doublepos_nu_name % (pretag,self.param_name[ivar],posttag))),'r')

      if self.nuDerivPrecise:
        checkmate = "checks/check_normalised_subsamples/post_process_mnu"
        nuMass01_name = checkmate+"_01eV/mnu-%s"       # Neutrino mass    (+0.1)
        nuMass02_name = checkmate+"_02eV/mnu-+%s"      # Neutrino mass    (+0.2)
        nuMass01_file = open(os.path.join(self.data_directory, ( nuMass01_name % (posttag))),'r')
        nuMass02_file = open(os.path.join(self.data_directory, ( nuMass02_name % (posttag))),'r')

      for i in range(self.np):
        # We are going to assume the k bins and z bins remain the same for all spectra

        # -> In positive + direction
        line = pkThFileP.readline()
        values = [float(valstring) for valstring in line.split()]
        nuP[i] = values[2]     #We only care about the actual power spectrum value
        # -> In negative - direction
        line = pkThFilePP.readline()
        values = [float(valstring) for valstring in line.split()]
        nuPP[i] = values[2]     #We only care about the actual power spectrum value

        # First and second derivative, assuming equal spacing of 0 and ++ from the + value
        # (i.e. the 0eV, 0.4eV, 0.8eV cases)
        # Forward Finite Difference formulas are being used
        # Otherwise, do PolyFit
        if self.nuDerivPrecise:
          # -> At numass 0.1
          line = nuMass01.readline()
          values = [float(valstring) for valstring in line.split()]
          nuMass01_value = values[2]     #We only care about the actual power spectrum value
          # -> At numass 0.2
          line = nuMass02.readline()
          values = [float(valstring) for valstring in line.split()]
          nuMass02_value = values[2]     #We only care about the actual power spectrum value

          nuMassVals = np.array([0.0,0.1,0.2,0.4,0.8])
          dnuMassVals = nuMassVals - self.nuMass
          nuPkVals = np.array([self.basis_pk[i],nuMass01_value,nuMass02_value,nuP[i],nuPP[i]])
          self.nuPoly = nppoly.polyfit(dnuMassVals,nuPkVals)
        else:
          self.derivPk[i,ivar*2] = (4.*nuP[i]-nuPP[i]-3.*self.basis_pk[i])/self.step[ivar]/2.
          self.derivPk[i,ivar*2+1] = (nuPP[i]+self.basis_pk[i]-2.*nuP[i])/self.step[ivar]/self.step[ivar]

      pkThFileP.close()
      pkThFilePP.close()

    # Third, deal with all non-neutrino cross terms
    # For all cosmological parameters except for neutrinos
    for ivar1 in range(self.param_num-1):
      for ivar2 in range(ivar1+1,self.param_num-1):
        # Make sure we didn't just accidentially name them in the wrong order (between + and +)
        try:
          pkThFileCross  = open(os.path.join(self.data_directory,( crosspos_name % (pretag,self.param_name[ivar1],self.param_name[ivar2],posttag))),'r')
        except:
          pkThFileCross  = open(os.path.join(self.data_directory,( crosspos_name % (pretag,self.param_name[ivar2],self.param_name[ivar1],posttag))),'r')
        # Get the cross spectrum out, and immediately calculate the cross derivative
        for i in range(self.np):
          line = pkThFileCross.readline()
          values = [float(valstring) for valstring in line.split()]
          PkCross = values[2]
          self.derivCrossPk[i,ivar1,ivar2]=(PkCross-valP[i,ivar1]-valP[i,ivar2]+self.basis_pk[i])/self.step[ivar1]/self.step[ivar2]
        pkThFileCross.close()

    # Fourth, deal with neutrino-related cross terms
    if self.full_grid:
      ivar2 = self.param_num-1
      for ivar1 in range(self.param_num-1):
        # Make sure we didn't just accidentially name them in the wrong order
        if self.WDM:
          try:
            pkThFileCross  = open(os.path.join(self.data_directory,( crosspos1_WDM_name % (pretag,self.param_name[ivar1],posttag))),'r')
          except:
            pkThFileCross  = open(os.path.join(self.data_directory,( crosspos2_WDM_name % (pretag,self.param_name[ivar1],posttag))),'r')
        else:
          try:
            pkThFileCross  = open(os.path.join(self.data_directory,( crosspos1_nu_name % (pretag,self.param_name[ivar1],self.param_name[ivar2],posttag))),'r')
          except:
            pkThFileCross  = open(os.path.join(self.data_directory,( crosspos2_nu_name % (pretag,self.param_name[ivar2],self.param_name[ivar1],posttag))),'r')

        # Get the cross spectrum out, and immediately calculate the cross derivative (between + and ++)
        for i in range(self.np):
          line = pkThFileCross.readline()
          values = [float(valstring) for valstring in line.split()]
          crossPP = values[2]

          self.derivCrossPk[i,ivar1,ivar2] = (crossPP-valP[i,ivar1]-nuPP[i]+self.basis_pk[i])/self.step[ivar2]/self.step[ivar1]/2.0

    pass
    # Now, we have got all derivatives, both normal and cross derivatives
    #End of Compute Derivatives routine

  # Assign parameter routine
  """
    Compute the new parameters at which the P(k) should be calculated
  """
  # LIST OF PARAMETERS
  # 0 -> sigma 8
  # 1 -> n_s
  # 2 -> inv_Ampl/T0
  # 3 -> invGrad/Gamma
  # 4 -> Omega_m
  # 5 -> h
  # (6) -> tauA
  # (7) -> tauS
  # ((8)) -> Neff
  # ((9)) -> nsrun
  # ((10)) -> zreio
  # Last -> mnu / inv_wdm_mass
  def _assign_parameters(self,args,k,z,ih):

    # Read parameters from CLASS
    if len(self.extend_opts)>0:
      try:
        params = cosmo.get_current_derived_parameters(["alpha_s","Neff","z_reio"])
        alpha_s = params["alpha_s"]
        N_eff = params["Neff"]
        z_reio = params["z_reio"]
      except:
        raise Exception("Could not get the parameters 'alpha_s','Neff' and/or 'z_reio' as derived parameters for the 'Lya_BOSS' emulator, even though you asked for option 'extend_opts'")


    self.newval[0] = args['sigma8']#0.8#0.8203#

    # Get new n_s from running parameter and current k value, or set it from cosmo
    if(self.fit_opts['FitNsRunningExplicit']):
      # We REQUIRE k_pivot to be set to 0.05 here
      k_pivot_CMB = 0.05
      c_kms = 299792.458
      # omm = self.newval[4] # Only for comparison purposes
      # k_invMpc = k*0.675*100.0*np.sqrt(omm*pow(1+z,3)+(1.-omm))/(1+z)
      k_invMpc = k*(c_kms*args['H(z)'](z)/(1+z))
      self.newval[1] = args['n_s'] +  self.alpha_s * 0.5 * np.log(k_invMpc/k_pivot_CMB)
    else:
      self.newval[1] = args['n_s']#0.96#0.9552#
    # 2.1) If desired, fit T0 and Gamma into invAmpl and invGrad
    if(self.fit_opts['FitT0Gamma']):
      try:
        self.T0 = args['T0']
        self.gamma = args['gamma']
      except:
        raise Exception("The parameters 'T0' and/or 'gamma' were not passed to the 'Lya_BOSS' emulator, even though you asked for option 'FitT0Gamma'")
      #self.T0=10298
      #self.gamma=0.8295
      #self.T0 = 10000
      #self.gamma = 0.8

      #invAmpl from T0, broken powerlaw
      #T0Slope = args['T0slope']
      #dT0 = self.T0 * pow((1+z)/4.0,T0Slope) - self.basis_T0[ih]
      dT0 = self.T0 - self.basis_T0[ih]
      invAmpl = self.invAmpl0 + self.dinvAmpl*2.0/(self.T0GFit["t0+"][ih]-self.T0GFit["t0-"][ih])*dT0

      self.newval[2] = invAmpl

      #invGrad from gamma
      #gammaSlope = self.gammaSlopeSup;
      #if(z<=3.0):
      #    gammaSlope=self.gammaSlopeInf;
      #gammaSlope = args['Gammaslope']
      #gammaSlope = self.gammaSlopeInf #Only single unbroken powerlaw
      #dgamma = self.gamma*pow((1+z)/4.0,gammaSlope) - self.basis_gamma[ih];
      dgamma = self.gamma - self.basis_gamma[ih]
      invGrad = self.invGrad0 + self.dinvGrad*2.0/(self.T0GFit["gamma+"][ih]-self.T0GFit["gamma-"][ih])*dgamma

      self.newval[3] = invGrad
    else:
      try:
        ampl = args['invAmpl']
        grad = args['invGrad']
      except:
        raise Exception("The parameters 'invAmpl' and/or 'invGrad' were not passed to the 'Lya_BOSS' emulator. Please enable 'FitT0Gamma', or provide them.")

      invAmpl,invGrad = self._InvAmplGrad(ampl,grad)
      self.newval[2] = invAmpl
      self.newval[3] = invGrad

    self.newval[4] = args['Omega_m']#0.3#0.2690#Omega_m
    self.newval[5] = args['H0']#70#67.0#100*h

    if(self.taylor_opts['DLNorma']):
      if self.fit_opts['Fbar_free']:
        try:
          self.newval[6] = -np.log(args['Fbar'])/pow((1+z),args['SlopeTauEff'])
        except KeyError as er:
          raise KeyError("The parameter 'Fbar' was not passed to the 'Lya_BOSS' emulator, even though you asked for the option 'DLNorma' and 'Fbar_free'.") from er
      else:
        try:
          self.newval[6] = args['AmpTauEff']/((1.+3)**(args['SlopeTauEff']))#0.003#0.002382#tauA
        except KeyError as er:
          raise Exception("The parameter 'AmpTauEff' was not passed to the 'Lya_BOSS' emulator, even though you asked for the option 'DLNorma'.") from er
      try:
        self.newval[7] = args['SlopeTauEff']
      except KeyError as er:
        raise Exception("The parameter 'SlopeTauEff' was not passed to the 'Lya_BOSS' emulator, even though you asked for the option 'DLNorma'.") from er
    if(self.varExtend):
      try:
        if self.extendNeff:
          self.newval[8] = N_eff#3.25#3.046#Neff
        else:
          self.newval[8] = self.basis[8]
        if self.extendAlphas:
          self.newval[9] = alpha_s#0.02#0.01#nsrun
        else:
          self.newval[9] = self.basis[9]
        if self.extendZreio:
          self.newval[10] = z_reio#9.0#12.0#zreio
        else:
          self.newval[10] = self.basis[10]
      except:
        raise Exception("The parameters 'N_eff', 'alpha_s', and/or 'z_reio' could not be obtained as derived parameters for the 'Lya_BOSS' emulator, even though you asked for the option 'varExtend'.")

    if(self.WDM):
      try:
        self.newval[self.param_num-1] = data.mcmc_parameters['inv_wdm_mass']['current']*data.mcmc_parameters['inv_wdm_mass']['scale']
      except:
        raise Exception("The parameter 'inv_wdm_mass' was not passed to the 'Lya_BOSS' emulator, even though you asked for the option 'WDM'.")
    else:
      if self.fit_opts['useMnuCosm']:
        self.newval[self.param_num-1] = args['Omega_nu']*((args['H0']/100.)**2)*93.14
        try:
          self.newval[self.param_num-1] = data.mcmc_parameters['M_tot']['current']*data.mcmc_parameters['M_tot']['scale']
        except:
          pass
      else:
        self.newval[self.param_num-1] = self.mnuNuisance


    #print(["%.10e"%x for x in self.newval])
    self.delta[0] = self.newval[0] - self.basis[0]#self.sigma8
    self.delta[1] = self.newval[1] - self.basis[1]#self.ns
    self.delta[2] = self.newval[2] - self.basis[2]#self.invAmpl0
    self.delta[3] = self.newval[3] - self.basis[3]#self.invGrad0
    self.delta[4] = self.newval[4] - self.basis[4]#self.Omegam
    self.delta[5] = self.newval[5] - self.basis[5]#self.H0
    if(self.taylor_opts['DLNorma']):
      self.delta[6] = self.newval[6] - self.basis[6]#self.tauA
      self.delta[7] = self.newval[7] - self.basis[7]#self.tauS
    if(self.varExtend):
      self.delta[8] = self.newval[8] - self.basis[8]#self.Neff
      self.delta[9] = self.newval[9] - self.basis[9]#self.nsrun
      self.delta[10] = self.newval[10] - self.basis[10]#self.zreio

    if(self.WDM):
      self.delta[self.param_num-1] = self.newval[self.param_num-1] - self.nuMass_WDM
    else:
      self.delta[self.param_num-1] = self.newval[self.param_num-1] - self.nuMass


    # End of Assign Parameter Routine
    #print(["%.10e"%x for x in self.delta])
    #print(self.basis[2],self.basis[3])

  # Estimate P(k) using Taylor expansion ( at bin index i )
  def _getPk(self, i):

    self.taylor_pk[i] = self.basis_pk[i]

    for ivar in range(self.param_num-1):
      self.taylor_pk[i] += self.derivPk[i,2*ivar]*self.delta[ivar] + 0.5*self.derivPk[i,2*ivar+1]*self.delta[ivar]*self.delta[ivar]
      # Precise quadratic formula (!)

    ivar = self.param_num-1
    if self.nuDerivPrecise:
      self.taylor_pk[i] += self.nuPoly(self.delta[ivar])
      # Precise formula for more than 3 data points
      # (minimizing least squared distance of second order polynomial at all data points)
    else:
      self.taylor_pk[i] += self.derivPk[i,2*ivar]*self.delta[ivar] + 0.5*self.derivPk[i,2*ivar+1]*self.delta[ivar]*self.delta[ivar]
      # Precise quadratic formula (!)

    for ivar1 in range(self.param_num):
      for ivar2 in range(ivar1+1,self.param_num):
        self.taylor_pk[i] += self.derivCrossPk[i,ivar1,ivar2]*self.delta[ivar1]*self.delta[ivar2]
        # Precise quadratic formula (!)


    ## Possible additional terms accounting for various fitted corrections
    """
    #Correction of the splicing technique unifying large box/large N simulations
    k=self.basis_k[i]
    if (self.CorrectionSplicing):
      #cor = 1.03511-15.1998*k+1265.68*k*k-34905.6*pow(k,3);
      cor = 1.00788 - 2.60629*k;
      self.taylor_pk[i]/= cor;
    """

    #IC correction (old, not used anymore, new correction in likelihood.py)
    # if self.fit_opts['CorrectionIC']:
    #   IC = [ 1.0143101513  ,  0.0883987252358 , -24.0139126015  ,
    #          1.01555167089 ,  0.0613395935053 , -32.6599908809  ,
    #          1.01703152597 , -0.162215644164  , -29.644398257   ,
    #          1.01574144779 , -0.158196495978  , -37.678794373   ,
    #          1.01450641631 , -0.188175932733  , -43.1394324376  ,
    #          1.01316420029 , -0.306136525333  , -44.6694620443  ,
    #          1.01119692534 , -0.663381149321  , -37.8838959187  ,
    #          1.01235929315 , -1.07184077409   , -32.3858907615  ,
    #          1.01230886936 , -1.77021036115   ,  -8.68343329663 ,
    #          1.01324011064 , -2.23292190126   ,  -2.2205744647  ,
    #          1.01244647355 , -2.4057300289    ,  -9.2059017279  ,
    #          1.01447478246 , -3.00118956784   ,   4.13672888546 ,
    #          1.01701190333 , -3.55315002783   ,  13.4670306749  ]

    #   ih = int(i/self.Nkperbin)
    #   k = self.basis_k[i]
    #   self.taylor_pk[i] *= (IC[ih*3]+IC[ih*3+1]*k+IC[ih*3+2]*k*k)
    # #print(self.taylor_pk[i])
    # pass
    # # End of Estimate P(k) using Taylor expansion

  """ Some calibrated utility functions to convert input parameters (T0, gamma) to gadget parameters (ampl, grad) and vice versa"""
  def _T0Gamma(self,ampl,grad):
    return (-1621.*ampl*ampl+12162.*ampl+3355.),(0.559561*grad + 1.57562)
  def _InvAmplGrad(self,ampl,grad):
    return (-2.6/(ampl+2.6)),(5.55/(5.55-grad))
  def _AmplGrad(self,invampl,invgrad):
    return (-2.6*(1./(invampl)+1)),(5.55*(1.-1./invgrad))
