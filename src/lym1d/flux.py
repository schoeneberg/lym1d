from scipy.stats import multivariate_normal
import numpy as np
from scipy.optimize import curve_fit

# Flux prior class, constructing priors on the mean flux
class FluxPrior:

  def __init__(self, z, priortype="becker13", verbose=1):

    self.verbose = verbose

    # Save variables for later evaluations (such as redshift and type of prior)
    self.pt = priortype.lower()
    self.z = z
    Nz = len(z)

    self.log("Constructing flux prior (type = {})".format(self.pt))

    if self.pt=="becker13":

      # This is the tau_eff(z) from Becker+13
      def taueff_becker(z,tau0,beta,c,z0=3.5):
        return tau0*((1+z)/(1+z0))**beta + c

      # These are the mean and covariance matrix from Becker+13 for their parameters
      mean_becker = np.array([0.751,2.9,-0.132])
      cov_becker = np.array([[0.00049, -0.00241, -0.00043], [-0.00241, 0.01336, 0.00224], [-0.00043, 0.00224, 0.00049]])

      # Generate evolutions given the mean and covariance values
      evols = self.generate_evols(taueff_becker, z, mean_becker, cov_becker)

      # We could do something fancy, but let's do the simplest thing for now
      self.mean_tau = np.mean(evols,axis=0)
      self.sigma_tau = np.std(evols,axis=0)

    elif self.pt=="turner24":
      # Table 3 from Turner+24 (2405.06743) -- final, and sigma_tot
      self.turner_z = np.array([2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3.05, 3.15, 3.25, 3.35, 3.45, 3.55, 3.65, 3.75, 3.85, 3.95,4.05, 4.15])
      self.turner_tau = np.array([0.147, 0.158, 0.179, 0.200, 0.226, 0.235, 0.268, 0.292, 0.316, 0.342, 0.373, 0.410, 0.455, 0.498, 0.527, 0.579, 0.638, 0.694, 0.770, 0.830, 0.854, 0.928])
      self.turner_sigma_tot_tau = np.array([0.012, 0.012, 0.015, 0.016, 0.016, 0.018, 0.019, 0.020, 0.021, 0.022, 0.023, 0.023, 0.022, 0.025, 0.030,0.032, 0.031, 0.032, 0.033, 0.034, 0.036, 0.039])
      # Problem: We need to generate points also in-between and outside these measurements.

      # 1) For extrapolation, we use the powerlaw behavior

      # 1a) Find which redshifts are outside
      z_mask_extra = np.logical_or(z<2.05, z>4.15)
      z_extra = z[z_mask_extra]
      if len(z_extra)>0:

        # 1b) We fit a powerlaw to the data
        taueff_turner = lambda zval, a, b : a*(1+zval)**b
        mean_turner, cov_turner = curve_fit(taueff_turner, self.turner_z, self.turner_tau, p0 = [2.4e-3, 3.6], sigma = self.turner_sigma_tot_tau)
        if mean_turner[0]>2.5e-3 or mean_turner[0]<2.3e-3 or mean_turner[1] < 3.5 or mean_turner[1] > 3.7:
          raise RuntimeError("Something went wrong with the Turner+24 fit!\nParameters: mean = {}, std = {}, cov = {}".format(mean_turner, np.sqrt(np.diag(cov_turner)) , cov_turner))

        # 1c) Then, we generate evolutions with the powerlaw parameters
        evols = self.generate_evols(taueff_turner, z_extra, mean_turner, cov_turner)

        # 1d) We could do something fancy, but let's do the simplest thing for now
        mean_tau_extra = np.mean(evols,axis=0)
        sigma_tau_extra = np.std(evols,axis=0)

      # 2) For interpolation, use the given data and CRUDELY (linearly) interpolate
      z_mask_intra = np.logical_and(z>=2.05, z<=4.15)
      z_intra = z[z_mask_intra]
      if len(z_intra)>0:
        mean_tau_intra = np.interp(z_intra, self.turner_z, self.turner_tau)
        sigma_tau_intra = np.interp(z_intra, self.turner_z, self.turner_sigma_tot_tau)

      # 3) Now, merge the two sets, but keeping things well-ordered
      # (all points should be either inside or outside)
      assert(np.all(np.logical_or(z_mask_extra, z_mask_intra)))
      self.mean_tau = np.empty_like(z)
      self.sigma_tau = np.empty_like(z)
      if len(z_extra)>0:
        self.mean_tau[z_mask_extra] = mean_tau_extra
        self.sigma_tau[z_mask_extra] = sigma_tau_extra
      if len(z_intra)>0:
        self.mean_tau[z_mask_intra] = mean_tau_intra
        self.sigma_tau[z_mask_intra] = sigma_tau_intra
    else:
      raise Exception("Unknown prior type")

    self.log("Flux prior ready !")

  # Generate evolutions in z for a given parametric function at new z values, given a mean and cov for the parameters
  def generate_evols(self, func, newz, mean, cov, seed=42, N_samps = 10000):

      # We construct a multivariate_normal function that represents these parameters
      dist = multivariate_normal(mean,cov,seed=seed)

      # Then we sample a bunch of these parameters
      samps = dist.rvs(size=N_samps)

      # For each parameter, we get the corresponding tau_eff(z)
      evols = np.empty((N_samps,len(newz)),dtype=float)
      for i in range(N_samps):
        evols[i] = func(newz,*samps[i])

      # Now, return the evolutions (to e.g. take mean and cov from)
      return evols

  def chi_square(self, fbar_function):
      flux = fbar_function(self.z)
      if self.pt == "becker13" or self.pt == "turner24":
        # Convert flux to tau_eff(z)
        tau = -np.log(flux)
        # Use mean and covmat to do multivariate Gaussian likelihood at corresponding redshifts
        #chi_squared = np.dot(np.dot(tau-self.mean_tau,self.tau_icov),tau-self.mean_tau)
        chi_squared = np.sum((tau-self.mean_tau)**2/self.sigma_tau**2)
        #assert(chi_squared>=0)
        return chi_squared

  def log(self, msg, level=1):
    if level <= self.verbose:
      print("[lym1d_fluxprior] "+"\n[lym1d_fluxprior] ".join(msg.split("\n")))
