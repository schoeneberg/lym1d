"""
Emulator based on LaCE

@author: Nils Schöneberg (@schoneberg)
"""

import numpy as np
from .emulator import EmulatorBase, EmulatorOutOfBoundsException
import h5py
import copy
from scipy.interpolate import interp2d, CubicSpline


class printPrepender:
  class prepend:
    def __init__(self, prefix):
      self.prefix=prefix
    def write(self, string):
      if string!="\n":
        self.stream.write(self.prefix)
        self.stream.write(string)
        self.stream.write("\n")
    def flush(self):
      self.stream.flush()
  def __init__(self,prefix):
    self.original_stdout = None
    self.new_stdout = self.prepend(prefix)
  def __enter__(self):
    import sys
    self.original_stdout = sys.stdout
    self.new_stdout.stream = self.original_stdout
    sys.stdout = self.new_stdout
  def __exit__(self, type, value, traceback):
    import sys
    sys.stdout = self.original_stdout


class Emulator_LaCE(EmulatorBase):

  parnames = ['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
  lace_type = "gadget"
  karr = np.geomspace(0.05,4,num=100)

  def __init__(self, args):
    if args!=None:
       self.construct(args)

  def construct(self,args):
    raise ValueError("Outdated setup of emulator")
    from lace.emulator.nn_emulator import NNEmulator
    self.lace_type = args.get("lace_type",self.lace_type)
    with printPrepender("[LaCE] "):
      print("Starting to load the LaCE emulator")
      print("Using LaCE-type == {}".format(self.lace_type))
      if self.lace_type == "gadget":
        from lace.archive import gadget_archive
        archive = gadget_archive.GadgetArchive(postproc="Pedersen21")
        training_data = archive.get_training_data(emu_params = self.parnames)
        #print(training_data)
        #self.emulator = NNEmulator(archive=archive, nepochs=1)
        self.emulator = NNEmulator(
          emulator_label='Cabayol23',training_set="Cabayol23",model_path="NNmodels/NNEmulator_LaCEHC.pt",train=False)#, nepochs=10)
      elif self.lace_type == "nyx":
        from lace.archive import nyx_archive
        if 'NYX_PATH' in args:
            import os
            os.environ['NYX_PATH'] = args.get('NYX_PATH','')
        archive = nyx_archive.NyxArchive(verbose=True)
        self.emulator = NNEmulator(emulator_label="Nyx_v0",archive=archive, train=True)
      else:
        raise Exception("No lace type {}".format(self.lace_type))

  def _args_to_list(self, args):
      if len(args)!=len(self.parnames):
          raise ValueError("Invalid number of arguments for this emulator. Expected {}, but got {}".format(self.parnames,args))
      # Go through every required parameter for the emulator, and retrieve it from args
      pars = np.empty(len(args))
      try:
        for ipar, parname in enumerate(self.parnames):
            pars[ipar] = args[parname]
      except KeyError as e:
        raise KeyError('The parameter {} is missing from your input arguments for the emulator.'.format(parname)) from None
      return pars

  def __call__(self, args, z, k=None):
    params = self._args_to_list(args)
    #print(params)
    with printPrepender("[LaCE] "):
      pk1d = self.emulator.emulate_p1d_Mpc(args,self.karr)
    #print("!! "+" , ".join(["%.5e"%x for x in pk1d]))
    cov = None #For now, no uncertainty propagation
    return pk1d, cov


  def in_bounds(self, args, z):
    # -2.340, -2.258
    # 0.294, 0.447
    return True
  @classmethod
  def load(cls, emuname, path):
    # Load from a given path for the nyx_file the given emulator_label = emuname
    from lace.emulator.emulator_manager import set_emulator
    with printPrepender("[LaCE] "):
        ret = cls(None)
        #nyx_emu_params = ['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
        import os

        emulator = set_emulator(
            emulator_label = emuname,
            nyx_file = path
        )
        ret.emulator = emulator
        return ret

  def save(self,path='emulator_LaCE.npz'):
    pass

  def get_karr(self,z):
    return self.karr
