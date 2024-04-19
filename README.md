# lym1d
## A 1D Lyman-Alpha power spectrum emulation suite

### How to install?

Simply install by doing `pip install -e` .

### How to run?

The lym1d likelihood can in principle be called directly with three dictionaries, one for cosmo parameters, one for thermo parameters (functions), and one for nuisance parameters. However, for most purposes it's likely to be easier to call it through the wrapper (located at `src/lym1d_wrapper`), which allows interfacing with more common tools such as cobaya or MontePython. 

#### Running with MontePython

In case of running with MontePython (the public_lyadesi repository), the command is simply

    python montepython/MontePython run -p input/Lya_H0_eBOSS_orig_usingdesi.param -o chains/Lya_H0_eBOSS_orig_usingdesi --bestfit chains/Lya_H0_eBOSS_orig_usingdesi/Lya_H0_eBOSS_orig_usingdesi.bestfit --conf lya.conf

Here the file `lya.conf` needs to be created in the folder, and it should contain the usual lines, pointing to your favorite version of classy, such as for example

    root = '/path/to/your/codes'

    path['cosmo']		= root+'/your_favorite_class/class_public'
    path['clik']            = root+'/Planck3/code/plc_3.0/plc-3.1/'

As such, it's paramount that you have pulled and manually installed (not pip-installed) a version of CLASS on your system before running. The tutorial is given on [the class github](https://github.com/lesgourg/class_public).

Additional arguments are of course the usual MontePython arguments. Running with option `-N 1 -f 0` should give a value of `229.647` if everything is installed correctly. The real runs can be started with larger options for `-N` and not including any `-f`. Additionally, in that case the MontePython command can be run with `mpirun -np XX` for multi-system MPI parallelization.

**Note that the default running mode requires your data to be in the folder `data/Lya_DESI` in MontePython** If this is not the case, you can either copy the data there, or change the arguments for the paths. In particular, there is the `base_directory` path.

In the case of NERSC use, this can point to `/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/`, whereas `data_path` would be set to `data_files/Chabanier19/`. Please find also the corresponding input file with `_NERSC` on the corresponding MontePython directory (!).

#### Running with Cobaya

The cobaya wrapper interaction will be published shortly
