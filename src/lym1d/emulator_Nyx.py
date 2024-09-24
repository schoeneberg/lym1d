"""
Emulator based on simulations run with the Nyx code, using the emulator_george_backend.py backend.

@author: Michael Walther (@Waltheus)
"""

import numpy as np
from .emulator_Nyx_george_backend import create_emulator
#from .emulator_Nyx_sklearn_backend import create_emulator
from .emulator import EmulatorBase, EmulatorOutOfBoundsException
import h5py
import copy
from scipy.interpolate import interp2d, CubicSpline

class Modelset:
    input_filename=None

    model_grid_attrs=None
    model_grid_data=None
    model_grid_redshifts=None

    flat_model_grid_k=None
    flat_model_grid_pk=None
    flat_model_grid_params=None
    flat_model_grid_Delta=None

    parlist=None

    fiducial_attrs=None
    fiducial_data=None
    fiducial_redshifts=None  #this is read at initialization, but not used further

    def __init__(self,hdf5name,use_lP=False,verbose=False,leave_out=[]):
        """
          Reads models from hdf5 file

          Args:
            hdf5name (str): the file path
            use_lP (bool): whether to use the pressure scale lambda_P instead of the UVB rescaling A_UVB
        """
        self.input_filename=hdf5name
        with h5py.File(hdf5name,'r') as f:
            self.extract_model_grid(f,use_lP=use_lP,verbose=verbose,leave_out=leave_out)
            self.extract_fiducial_model(f,use_lP=use_lP)

    def extract_model_grid(self,f, use_lP=False,verbose=False,leave_out=[]):
        """
          Reads the model grid from the file, and cuts it to those redshifts where all models are available.

          Args:
            f (hdf5): file object
            use_lP (bool): whether to use the pressure scale lambda_P instead of the UVB rescaling A_UVB
        """
        modellist=list(f.keys())
        modellist = [x for x in modellist if not (x in leave_out)]
        redshiftstrlist=list((f[m].keys() for m in modellist))
        used_modellist= [m for m,r in zip(modellist,redshiftstrlist) if len(r)>0 and 'cosmo_grid' in m]
        used_redshiftstrlist=[r for m,r in zip(modellist,redshiftstrlist) if len(r)>0 and 'cosmo_grid' in m]
        used_attrs_global=[dict(f[m].attrs.items())  for m in used_modellist]

        redshiftslist=[[float(s.split('_')[1]) for s in r] for r in used_redshiftstrlist]
        all_redshifts=np.unique([z for r in redshiftslist for z in r])

        #get only redshifts that are available for every model
        used_redshifts=[]
        for z in all_redshifts:
            if np.all([z in r for r in redshiftslist]):
                used_redshifts.append(z)

        thermal_grid_str=[[[t for t in f[f'{m}/redshift_{s:.1f}'].keys() if 'thermal' in t] for m in used_modellist] for s in used_redshifts]
        thermal_grid_str=thermal_grid_str[0][0] #might add a check here for assuring all are same length etc


        self.model_grid_data=[]
        self.model_grid_attrs=[]
        used_redshifts_out=[]
        for z in used_redshifts:
            skipredshift=False
            model_data_single_z = []
            model_attrs_single_z = []
            for m,cosmo_attrs in zip(used_modellist,used_attrs_global):
                model_data_single_cosmo = []
                model_attrs_single_cosmo = []
                for t in thermal_grid_str:
                    if t not in f[f'{m}/redshift_{z}'].keys():
                        if verbose:
                          print(f'[emulator_Nyx] model {m}/redshift_{z}/{t} is non-existing, continuing anyway')
                        continue
                    if use_lP:
                        if "lambda_P" not in f[f'{m}/redshift_{z}'].attrs:
                            if verbose:
                              print(f'[emulator_Nyx] model {m}/redshift_{z}/{t} has no lambda_P, cannot build emulator for this redshift, skipping')
                            skipredshift = True
                            break
                        model_attrs_single_cosmo.append(dict(f[f'{m}/redshift_{z}/{t}'].attrs.items(),lambda_P=f[f'{m}/redshift_{z}'].attrs['lambda_P'],**cosmo_attrs))   #the stuff inside dict() is for adding thermal and cosmo pars together, python 3.9 has an operator for this dicta|dictb
                    else:
                        model_attrs_single_cosmo.append(dict(f[f'{m}/redshift_{z}/{t}'].attrs.items(),**cosmo_attrs)) #the stuff inside dict() is for adding thermal and cosmo pars together, python 3.9 has an operator for this dicta|dictb
                    model_data_single_cosmo.append(f[f'{m}/redshift_{z}/{t}']['1d power'][:])
                if skipredshift:
                    break
                model_data_single_z.append(model_data_single_cosmo)
                model_attrs_single_z.append(model_attrs_single_cosmo)
            if skipredshift:
                continue
            self.model_grid_data.append(model_data_single_z)
            self.model_grid_attrs.append(model_attrs_single_z)
            used_redshifts_out.append(z)
        self.model_grid_redshifts=np.array(used_redshifts_out)

    def extract_fiducial_model(self,f,use_lP=False):
        """
        Extracts the fiducial model from the file

        Args:
            f (hdf5): file object
            use_lP (bool): whether to use the pressure scale lambda_P instead of the UVB rescaling A_UVB
        """
        redshiftstrlist=list(f['fiducial'].keys())
        redshiftslist=[float(s.split('_')[1]) for s in redshiftstrlist]

        used_attrs_global=dict(f['fiducial'].attrs.items())
        if use_lP:
            #if for the fiducial lambda_p is missing we do not care
            try:
                self.fiducial_attrs=[dict(f[f'fiducial/{zstr}/rescale_Fbar_fiducial'].attrs.items(),lambda_P=f[f'fiducial/{zstr}'].attrs['lambda_P'],**used_attrs_global) for zstr in redshiftstrlist] #the stuff inside dict() is for adding thermal and cosmo pars together, python 3.9 has an operator for this dicta|dictb
            except:
                self.fiducial_attrs=[dict(f[f'fiducial/{zstr}/rescale_Fbar_fiducial'].attrs.items(),**used_attrs_global) for zstr in redshiftstrlist] #the stuff inside dict() is for adding thermal and cosmo pars together, python 3.9 has an operator for this dicta|dictb
        else:
            self.fiducial_attrs=[dict(f[f'fiducial/{zstr}/rescale_Fbar_fiducial'].attrs.items(),**used_attrs_global) for zstr in redshiftstrlist] #the stuff inside dict() is for adding thermal and cosmo pars together, python 3.9 has an operator for this dicta|dictb
        self.fiducial_data=[f[f'fiducial/{zstr}/rescale_Fbar_fiducial']['1d power'][:] for zstr in redshiftstrlist]
        self.fiducial_redshifts=np.array(redshiftslist)

    def flatten_model_grid(self,usedpars=['A_lya','n_lya','omega_m','H_0','Fbar','T_0','gamma','A_UVB']):
        """
        Reformats data structures to the way expected by the emulator,
          as well as cuts the parameters to those passed in usedpars.

        Args:
            usedpars (list, optional): parameters to keep (the file contains redundancy). Defaults to ['A_lya','n_lya','omega_m','H_0','Fbar','T_0','gamma','A_UVB'].
        """
        self.parlist=usedpars
        self.flat_model_grid_params=[np.array([[d2[par] for par in usedpars] for d1 in d for d2 in d1]) for d in self.model_grid_attrs]
        self.flat_model_grid_k=[np.array([d2['k'] for d1 in d for d2 in d1]) for d in self.model_grid_data]
        model_grid_pk=[np.array([d2['Pk1d'] for d1 in d for d2 in d1]) for d in self.model_grid_data]
        self.flat_model_grid_pk=model_grid_pk
        self.flat_model_grid_Delta=[pk*k/np.pi for k,pk in zip(self.flat_model_grid_k,self.flat_model_grid_pk)]


    def restrict_k(self,kmin=None,kmax=None):
        """
        Cut the model set to only contain a given range in modes (kmin, kmax)

        Args:
            kmin, kmax ([type], optional): range of modes (included in output). Defaults to None.
        """        
        if kmin is None:
            kmin=0
        if kmax is None:
            kmax=np.inf
        if self.flat_model_grid_params is None:
            self.flatten_model_grid()
        new_k=[]
        new_pk=[]
        new_delta=[]
        for k,pk,delta in zip(self.flat_model_grid_k,self.flat_model_grid_pk,self.flat_model_grid_Delta):
            useinds=np.sum((k<kmax+1e-8)&(k>kmin-1e-8),axis=0,dtype=bool) #assuming same limit for all redshifts here, else would be complicated
            new_k.append(k[:,useinds])
            new_pk.append(pk[:,useinds])
            new_delta.append(delta[:,useinds])
        self.flat_model_grid_k, self.flat_model_grid_pk, self.flat_model_grid_Delta = new_k, new_pk, new_delta
        

    def restrict_z(self,zmin=None,zmax=None):
        """
        Cut the modelset to a range in redshift specified by (zmin, zmax)

        Args:
            zmin, zmax ([type], optional): range of redshifts (included in output). Defaults to None.
        """        
        if zmin is None:
            zmin=0
        if zmax is None:
            zmax=np.inf
        if self.flat_model_grid_params is None:
            self.flatten_model_grid()
        useinds=((self.model_grid_redshifts<zmax+1e-8)&(self.model_grid_redshifts>zmin-1e-8)).nonzero()[0]
        self.flat_model_grid_params=[self.flat_model_grid_params[ind] for ind in useinds]
        self.flat_model_grid_k=[self.flat_model_grid_k[ind] for ind in useinds]
        self.flat_model_grid_pk=[self.flat_model_grid_pk[ind] for ind in useinds]
        self.flat_model_grid_Delta=[self.flat_model_grid_Delta[ind] for ind in useinds]

        self.model_grid_redshifts=[self.model_grid_redshifts[ind] for ind in useinds]

    def __getitem__(self,item):
        """
        Get the models for a redshift or multiple

        Args:
            item: index, slice or redshift

        Returns:
            (array(floats),array(floats),array(floats)): arrays in the way expected by the emulator
        """        
        if self.flat_model_grid_params is None:
            self.flatten_model_grid()
        if isinstance(item, (int, slice)):
            ind=item
        elif isinstance(item, (float)):
            ind=np.abs(self.model_grid_redshifts-item)<0.01
            if np.any(ind):
                ind=ind.nonzero()[0][0]
        else:
            raise ValueError("Index for Modelset needs to be int, slice (index of redshift in both cases) or float (redshift value)")
        return (self.flat_model_grid_params[ind],self.flat_model_grid_k[ind],self.flat_model_grid_pk[ind])

    def copy(self):
        return copy.deepcopy(self)

# Dummy model set class
class _NoneModelSet:
    pass





class Emulator_Nyx(EmulatorBase):
    karr=None
    emulatorarr=None
    redshifts=None

    usematern=None
    uselogpower=None
    usepca=None
    output_cov=None
    varywhitenoise=None

    parnames=None

    def __init__(self, args):
        """
          Emulator initialization. Arguments provided as the args dictionary

          Args:
            Attempts to use the 'modelset' argument as either a ModelSet instance or a path to a file containing such
            'uselogpower' (bool, opt) can be used to EMULATE power in logk-logPk space, note that the __call__ still returns in k-Pk space (default : True)
            'kmin' (float, opt) can be used to set a minimum k of emulation (default: 0)
            'kmax' (float, opt) can be used to set a maximum k of emulation (default: 5)
            'zmin' (float, opt) can be used to set a minimum z of emulation (default: 2.2)
            'zmax' (float, opt) can be used to set a maximum z of emulation (default: 5.4)
            'usematern' (bool, opt) can be used to set the matern kernel, otherwise uses squared exp (default : True)
            'usepca' (bool, opt) can be used to decompose the power spectra into a PCA basis and emulating the PCA components, the output still in k-Pk (default: False)
            'output_cov' (bool, opt) can be used to force an output of the covaraince matrix (default: True)
            'npc' (int, opt) gives the number of PCA components to be used if 'usepca' is True, defaults to almost all components (default: 160)
            'varywhitenoise' (bool, opt) forces to the hyperparameter-optimization of the white noise parameters of the kernel (otherwise just a small amount) (default: True)
            'use_lP' uses a pressure smoothing scale lambda_P value per redshift instead of the global A_UVB renormalization of the UV background
        """
        # PARSE INPUT ARGUMENTS
        if not isinstance(args, dict):
          raise TypeError("The provided 'args' argument must be a dict, but got {} instead".format(type(args)))
        modelset_in = args.get('modelset')
        self.use_lP = args.get('use_lP',False)
        self.use_H = args.get('use_H',True)
        self.use_omm = args.get('use_omm',True)
        self.A_lya_n_lya_strs = args.get('A_lya_n_lya',['A_lya','n_lya'])

        if modelset_in is None:
          raise ValueError("The 'args' argument is missing the entry 'modelset'")
        elif isinstance(modelset_in, _NoneModelSet):
          return #for explicitly passing no modelset, should not be used by user
        elif isinstance(modelset_in, Modelset):
          modelset = modelset_in.copy()
        elif isinstance(modelset_in, str):
          modelset = Modelset(modelset_in, use_lP=self.use_lP, leave_out=args.get('leave_out',[]))
          A_str = self.A_lya_n_lya_strs[0]
          n_str = self.A_lya_n_lya_strs[1]
          if self.use_lP:
              usedpars=[A_str, n_str,'Fbar','T_0','gamma','lambda_P']
          else:
              usedpars=[A_str, n_str,'Fbar','T_0','gamma','A_UVB']
          if self.use_H:
              usedpars.append('H_0')
          if self.use_omm:
              usedpars.append('omega_m')
          modelset.flatten_model_grid(usedpars=usedpars)
        else:
          raise ValueError("The 'modelset' argument inside 'args' could not be understood. You provided: "+str(modelset_in))
        kmin = args.get('kmin',0.0)
        kmax = args.get('kmax',5.0)
        zmin = args.get('zmin',2.2)
        zmax = args.get('zmax',5.4)
        if kmin is not None or kmax is not None:
            modelset.restrict_k(kmin,kmax)
        if zmin is not None or zmax is not None:
            modelset.restrict_z(zmin,zmax)
        self.usematern=args.get('usematern',True)
        self.uselogpower=args.get('uselogpower',True)
        self.usepca=args.get('usepca',False)
        self.output_cov=args.get('output_cov',True)
        self.varywhitenoise=args.get('varywhitenoise',True)
        npc = args.get('npc',160)

        # SETUP INTERNAL VARIABLES
        self.redshifts = modelset.model_grid_redshifts
        self.bounds = np.empty(len(self.redshifts),dtype=dict)
        self.pararr = np.empty(len(self.redshifts),dtype=object)
        for iz,z in enumerate(self.redshifts):
          allpars = (modelset.flat_model_grid_params[iz].T)
          self.bounds[iz] = {}
          self.pararr[iz] = allpars
          for ipar,par in enumerate(modelset.parlist):
            pararr = allpars[ipar]
            self.bounds[iz][par] = [np.min(pararr),np.max(pararr)]
        self.parnames = modelset.parlist

        # BUILD EMULATORS (for each redshift)
        emulatorarr=[]
        karr=[]
        emuparnamelist=[]
        emupararrlist=[]
        for z in modelset.model_grid_redshifts:
            params,k,pk=modelset[z]
            smooth_lengths = np.array(5*params.std(axis=0))
            emu, update_emu, emupars, emuparnames = create_emulator(
                params,
                pk if not self.uselogpower else np.log10(pk),
                smooth_lengths,
                noise=(1e-8 if not self.varywhitenoise else None),
                npc=npc,
                optimize=True,
                output_cov=self.output_cov,
                sigma_0=np.sqrt(
                    1
                ),  # this allows training the signal variance (doesn't properly work on small datasets with PCA)
                #sigma_l=np.sqrt(0.1),            #this allows adding a dot-kernel corresponding to linear interpolation
                noPCA=not (self.usepca),
                kerneltype="SE" if not (self.usematern) else "M52",
            )
            karr.append(k[0])
            emuparnames=np.array(emuparnames,dtype='<U50')
            for i,e in enumerate(emuparnames):
                if 'M_' in e:
                    parnameind=int(e.split('M_')[1].split('_')[0])
                    emuparnames[i]=e.replace(f'M_{parnameind}_{parnameind}',f'M_{self.parnames[parnameind]}_{self.parnames[parnameind]}')
            emulatorarr.append(emu)
            emuparnamelist.append(emuparnames)
            emupararrlist.append(emupars)
        self.emuparnames=emuparnamelist
        self.emupars=emupararrlist

        # SAVE FINAL INTERNAL VARIABLES
        self.karr=karr
        self.emulatorarr=emulatorarr

    def save(self,path='emulator_nyx.npz'):
        """
        Saves the emulator to an .npz file (i.e. use numpy to pickle the data and gzip compress)
        (this could probably be done way better in hdf5 or so)

        Args:
          'path' (str, opt) is the path to save at. (default: 'emulator_nyx.npz')
        """        
        np.savez_compressed(path, emu=self.emulatorarr,k=self.karr,z=self.redshifts,bounds=self.bounds,parnames=self.parnames,
                            emuparnames=self.emuparnames, emupars=self.emupars, pararr=self.pararr,
                            meta=dict(usematern=self.usematern,
                                      uselogpower=self.uselogpower,
                                      usepca=self.usepca,
                                      output_cov=self.output_cov,
                                      varywhitenoise=self.varywhitenoise))
    @classmethod
    def load(cls, path):
        """
        Load emulator from file specified through 'path'

        Args:
          'path' (str, opt) is the path to load from.
        """
        data=np.load(path,allow_pickle=True)
        obj=cls({'modelset':_NoneModelSet()})
        obj.emulatorarr=data['emu'].tolist()
        obj.karr=data['k'].tolist()
        obj.redshifts=data['z']
        obj.bounds=data['bounds']
        obj.parnames=data['parnames']
        obj.pararr=data['pararr']
        obj.use_lP='lambda_P' in obj.parnames
        metadict=data['meta'].tolist()
        obj.usematern=metadict['usematern']
        obj.usepca=metadict['usepca']
        obj.output_cov=metadict['output_cov']
        obj.varywhitenoise=metadict['varywhitenoise']
        obj.uselogpower=metadict['uselogpower']
        obj.emuparnames=data['emuparnames']
        obj.emupars=data['emupars']

        return obj

    def __call__(self, args, z, k=None):
        """
        Evaluate the emulator at given parameters (args) and redshift (z), and possibly wavenumbers (k)

        Args:
          'args' (dict(str,float)) corresponds to the set of parameters at which to emulate
          'z' (float) is the redshift at which to emulate
          'k' (array(float), opt) are the wavenumbers in 1/Mpc at which to evaluate the emulator.
              If None are provided, the output is at the internally defined karr

        Returns:
            array(float): emulated power spectrum (without log) in Mpc
        """
        params = self._args_to_list(args)
        useind=np.abs(self.redshifts-z)<0.01
        if np.any(useind):
            useind=useind.nonzero()[0][0]
            emu=self.emulatorarr[useind]
            if self.output_cov:
                data,cov=emu(params)
            else:
                data=emu(params)[0]
                cov=None
            if self.uselogpower:
                data=10**data
                if self.output_cov:
                    cov=np.log(10) ** 2 * (np.diag(data)).dot(cov.dot(np.diag(data))) #this is only approximately right
            if k is None:
              return data, cov
            elif self.output_cov:
              return CubicSpline(self.karr[useind],data)(k), interp2d(self.karr[useind],self.karr[useind], cov, kind='cubic')(k,k)
            else:
              return CubicSpline(self.karr[useind],data)(k), None
        else:
            raise ValueError("Invalid redshift: {} not in {}".format(z,self.redshifts))

    def in_bounds(self, args, z):
        """
        Check that the emulator bounds cover a given set of parameters (args) and redshift (z)

        Args:
          'args' (dict: (str,float)) corresponds to the set of parameters at which to check if emulation is possible
          'z' (float) is the redshift at which to check if emulation is possible

        Throws EmulatorOutOfBoundsException if out of bounds
        """
        iz=np.abs(self.redshifts-z)<0.01
        if np.any(iz):
            iz=iz.nonzero()[0][0]
        else:
            raise ValueError("Invalid redshift: {} not in {}".format(z,self.redshifts))
        if not isinstance(args,dict):
            raise TypeError("The 'args' argument for 'in_bounds' has to be a python dictionary")
        pars = self._args_to_list(args)
        if hasattr(self,"convex_hull_mode") and self.convex_hull_mode == True:
          #if not hasattr(self,"convex_hull"):
          #  self.convex_hull = {}
          #if iz not in self.convex_hull:
          #  from scipy.spatial import Delaunay
          #  print(self.pararr[iz])
          #  self.convex_hull[iz] = Delaunay(self.pararr[iz].T)
          #in_hull = (self.convex_hull[iz].find_simplex(pars)>=0)
          from scipy.optimize import linprog
          def in_hull_func(points, x):
              n_points = len(points)
              n_dim = len(x)
              c = np.zeros(n_points)
              A = np.r_[points.T,np.ones((1,n_points))]
              b = np.r_[x, np.ones(1)]
              lp = linprog(c, A_eq=A, b_eq=b)
              return lp.success
          in_hull = in_hull_func(self.pararr[iz].T,pars)
          if not in_hull:
            raise EmulatorOutOfBoundsException("Out of convex hull bounds for emulator")
          return
        for ipar,parname in enumerate(self.parnames):
            left,right = self.bounds[iz][parname]
            if hasattr(self,"shortening_factor") and self.shortening_factor > 0.:
                newleft = left*(right/left)**self.shortening_factor
                newright = left*(right/left)**(1.-self.shortening_factor)
                if args[parname]<newleft or args[parname]>newright:
                  raise EmulatorOutOfBoundsException("Parameter {}={} is not in bounds [{},{}] -- z={} (original bounds of [{},{}] shortened by factor {})".format(parname,args[parname],newleft,newright,z,left,right,self.shortening_factor))
            if args[parname]<left or args[parname]>right:
                # Instantly fail if out of bounds (provide info on failure through use of OOB exception)
                raise EmulatorOutOfBoundsException("Parameter {}={} is not in bounds [{},{}] -- z={}".format(parname,args[parname],left,right,z))
        # If all checks have passed, simply return
        return

    def _args_to_list(self, args):
        """
        Convert the argument dictionary 'args' to a list in the same ordering as used by the emulator

        Args:
          'args' (dict: (str,float)) corresponds to the set of parameters at which to check if emulation is possible

        Returns:
          'pars' (list: float) a list of the parameters that can be passed directly to the emulator
        """
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

    def get_karr(self, z):
        """
        Get the underlying array of wavenumbers k from the emulator at a given redshift

        Args:
          'z' (float) is the redshift at which to obtain the k array

        Returns:
          'karr' (array: float) is the list of wavenumbers of the emulator at the given redshift
        """
        iz=np.abs(self.redshifts-z)<0.01
        if np.any(iz):
            iz=iz.nonzero()[0][0]
        else:
            raise ValueError("Invalid redshift: {} not in {}".format(z,self.redshifts))
        return self.karr[iz]



"""
Some code for debugging the modelset and emulator implementations
"""
if __name__=='__main__':
    #read in model data
    models=Modelset('models.hdf5')
    #reformat so that it fits emulator input
    models.flatten_model_grid()

    #build emulator
    emu=Emulator_Nyx(models,zmin=2.2,zmax=2.4,output_cov=False)
    emu.save('test_emu.npz')


    #run the emulator at the center of the grid for the first of the redshifts
    testpars=np.mean(models.flat_model_grid_params[0],axis=0)
    test_emulated=emu(testpars,redshift=models.model_grid_redshifts[0])

