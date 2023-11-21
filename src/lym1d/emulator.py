from abc import ABC, abstractmethod

# To be thrown when the in_bounds method fails
class EmulatorOutOfBoundsException(Exception):
  pass

class EmulatorBase(ABC):

  # Initialize the emulator, loading all relevant files (e.g. simulation P_F(k))
  # Construct relevant intermediate quantities (e.g. GP kernel)
  # args is a dictionary of the arguments for the emulator
  @abstractmethod
  def __init__(self, args):
    pass

  # Give back P_F(k) for given k, z and given args = emulator params
  # args is a dictionary of the arguments for the emulator
  @abstractmethod
  def __call__(self, args, z, k=None):
    pass

  # Check if given set of parameters are inside of the emulator bounds (for given redshift)
  # args is a dictionary of the arguments for the emulator
  # Throws EmulatorOutOfBoundsException if out of bounds
  @abstractmethod
  def in_bounds(self, args, z):
    pass

  # Load an emulator from a stored file
  # Can be implemented as "pass" if construction is very fast (otherwise, prefer pickling)
  @classmethod
  @abstractmethod
  def load(cls, path):
    pass

  # Save emulator to a file
  # Can be implemented as "pass" if construction is very fast (otherwise, prefer picloing)
  @abstractmethod
  def save(self, path):
    pass
