# lym1d
## A 1D Lyman-Alpha power spectrum emulation suite

### How to install?

Simply install by doing `pip install -e` .

### How to run?

The lym1d likelihood can in principle be called directly with three dictionaries, one for cosmo parameters, one for thermo parameters (functions), and one for nuisance parameters. However, for most purposes it's likely to be easier to call it through the wrapper (located at `src/lym1d_wrapper`), which allows interfacing with more common tools such as cobaya or MontePython.

#### Specific NERSC installation instructions

    env MPICC=/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/bin/mpicc conda install pip cython numpy scipy mpi4py
    rm /global/homes/<firstletter_user_directory>/<your_user_directory>/.conda/envs/TEST/compiler_compat/ld
    git clone git@github.com:lesgourg/class_public.git
    cd class_public; make -j
    cd python; python3 -m pip install .
    cd ../..
    git clone git@github.com:schoeneberg/lym1d.git
    cd lym1d; python3 -m pip install . ; cd ..

#### Running with Cobaya

In case of running with cobaya, the setup is as follows. First, follow the [NERSC installation steps](#Specific-nersc-installation-instructions), as well as installing cobaya

    pip install cobaya

Finally, the final run is as easy as

    python3 run_Planque_2020.py

while updating the path inside the file to point to YOUR data (the predefined is NERSC). Alternatively

    mpirun -n NP python3 run_Planque_2020.py

#### Running with MontePython

To install, please use the [NERSC installation steps](#Specific-nersc-installation-instructions), and then use

    git clone git@github.com:schoeneberg/montepython_public_lyadesi.git

In case of running with MontePython (the public_lyadesi repository), the command is simply

    python3 montepython/MontePython.py run -p input/Lya_H0_eBOSS_orig_usingdesi.param -o chains/Lya_H0_eBOSS_orig_usingdesi --bestfit chains/Lya_H0_eBOSS_orig_usingdesi/Lya_H0_eBOSS_orig_usingdesi.bestfit --conf lya.conf

Here the file `lya.conf` needs to be created in the folder, and it should contain the usual lines, pointing to your favorite version of classy, such as for example

    root = '/path/to/your/codes'

    path['cosmo']		= root+'/your_favorite_class/class_public'
    path['clik']            = root+'/Planck3/code/plc_3.0/plc-3.1/'

As such, it's paramount that you have pulled and manually installed (not pip-installed) a version of CLASS on your system before running. The tutorial is given on [the class github](https://github.com/lesgourg/class_public).

Additional arguments are of course the usual MontePython arguments. Running with option `-N 1 -f 0` should give a value of `229.647` if everything is installed correctly. The real runs can be started with larger options for `-N` and not including any `-f`. Additionally, in that case the MontePython command can be run with `mpirun -np XX` for multi-system MPI parallelization.

**Note that the default running mode requires your data to be in the folder `data/Lya_DESI` in MontePython** If this is not the case, you can either copy the data there, or change the arguments for the paths. In particular, there is the `base_directory` path.

In the case of NERSC use, this can point to `/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/`, whereas `data_path` would be set to `data_files/Chabanier19/`. Please find also the corresponding input file with `_NERSC` on the corresponding MontePython directory (!).


##### What should my run configuration look like?

There is no general definite way to run MontePython, and it depends a bit on the specifics of your setup. For smaller systems, running 4 MPI chains in parallel is a good idea, with up to 4 cores per chain. On larger systems, there is in principle no upper bound. Between 8 and 16 MPI chains in parallel are decent, and the number of OpenMPI threads should be at least around 4-8 per chain. An example would consist of 8 MPI chains run with 4 cores per chain. Random Access Memory requirements of class are typically not an issue, and much less MontePython, so they can be of order ~100MB per core. *For the specifics of having set most parameters to nuisance mode, it is generally a good idea to take fewer OpenMPI threads and more MPI runs (e.g. 16 chains each with a single core)*
The cobaya wrapper interaction will be published shortly
