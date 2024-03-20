"""
Very simple utility file with various definitions of useful classes/functions
(currently only OptionDict)
"""
from scipy.stats import multivariate_normal
import numpy as np

# Small utility class to make passing of arguments safer
# You can only pass boolean values
# You can only set specific options to true or false
# You can iterate easily through only true or false options
# You can add keys a posteriori
class OptionDict(dict):
  # Initialize the OptionDict with dictionary default_values
  # The keys of the dictionary will be the allowed options
  # The values of the dictionary provide the default boolean option values
  def __init__(self,default_values):
    self._keys = list(default_values.keys())
    for key in self._keys:
      self[key] = default_values[key]
  # Setting an item only works if the key is in the list of keys
  def __setitem__(self, key, val):
    if key not in self._keys:
      raise KeyError
    if type(val)!=bool:
      raise Exception("option can only be bool")
    dict.__setitem__(self, key, val)
  # Iterate through all true options
  def iterate(self):
    for key in self._keys:
      if self[key]==True:
        yield key
    return
  # Iterate through all false options
  def inverse_iterate(self):
    for key in self._keys:
      if self[key]==False:
        yield key
    return
  # Add additional options (and give their correspoinding default values) with dictionary default_values
  def addkeys(self,default_values):
    self._keys += list(default_values.keys())
    for key in default_values.keys():
      self[key] = default_values[key]

# Flux prior class, constructing priors on the mean flux
class FluxPrior:

  def __init__(self, z, priortype="becker13"):

    # Save variables for later evaluations (such as redshift and type of prior)
    self.pt = priortype
    self.z = z
    Nz = len(z)

    if self.pt=="becker13":

      # This is the tau_eff(z) from Becker+13
      def taueff_becker(z,tau0,beta,c,z0=3.5):
        return tau0*((1+z)/(1+z0))**beta + c

      # These are the mean and covariance matrix from Becker+13 for their parameters
      mean_becker = np.array([0.751,2.9,-0.132])
      cov_becker = np.array([[0.00049, -0.00241, -0.00043], [-0.00241, 0.01336, 0.00224], [-0.00043, 0.00224, 0.00049]])

      # We construct a multivariate_normal function that represents these parameters
      dist = multivariate_normal(mean_becker,cov_becker,seed=42)

      # Then we sample a bunch of these parameters
      N_samps = 10000
      samps = dist.rvs(size=N_samps)

      # For each parameter, we get the corresponding tau_eff(z)
      evols = np.empty((N_samps,Nz),dtype=float)
      for i in range(N_samps):
        evols[i] = taueff_becker(z,*samps[i])

      # From these histories, we estimate at each of the redshifts the corresponding mean and sigma
      # More precisely, since they will be highly correlated, we do a multivariate Gaussian fit
      def mean_icov(points, threshold=1e-12):
        # equivalent to mean, cov = multivariate_normal.fit(points)
        # but that requires specific numpy version
        m = np.mean(points,axis=0)
        c = (points-m).T @ (points-m) /len(points)
        # Now, we rectify the covmat if there are numerically negative eigenvalues
        eigval, eigvec = np.linalg.eigh(c)
        eigval[eigval<threshold*max(eigval)] = threshold*max(eigval)
        # Then, return mean and INVERSE covmat
        return m, eigvec @ np.diag(1.0/eigval) @ np.linalg.inv(eigvec)
      # Now, we are done by fitting the points
      self.mean_tau, self.tau_icov = mean_icov(evols)
    else:
      raise Exception("Unknown prior type")

  def chi_square(self, fbar_function):
      flux = fbar_function(self.z)
      if self.pt == "becker13":
        # Convert flux to tau_eff(z)
        tau = -np.log(flux)
        # Use mean and covmat to do multivariate Gaussian likelihood at corresponding redshifts
        chi_squared = np.dot(np.dot(tau-self.mean_tau,self.tau_icov),tau-self.mean_tau)
        assert(chi_squared>=0)
        return chi_squared
