"""
Very simple utility file with various definitions of useful classes/functions
(currently only OptionDict)
"""

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

