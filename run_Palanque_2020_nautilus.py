from lym1d_wrapper.nautilus_wrapper import nautilus_wrapper
import numpy as np

rng=np.random.default_rng(12345)

arguments = {
   
'base_directory':'/home/nilsor/codes/montepython_lyadesi_public/montepython_public_lyadesi/data/Lya_DESI/',

'runmode':'nyx',

'arguments':{'has_cor':False,
             'nuisance_replacements':['A_lya','n_lya'],
             'data_filename':'/home/nilsor/data/PROJECTS/LYA_nersc/Mock/MockChallengeSnapshot/mockchallenge-0.2/mock_challenge_0.2_nonoise_fiducial.fits.gz',
             #'data_filename':'/home/nilsor/data/PROJECTS/LYA_nersc/DATA/p1d_fft_y1_measurement_kms.fits',
             #'data_filename':'/home/nilsor/data/PROJECTS/LYA_nersc/DATA/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits',
             'data_format':'Y1',
             'bounds_verbose':10
             #'models_path':'all_model_outputs_NEW_CORRECTED_with_lP_refine0LHC1_refine1_refine2LHC1_refine3LHC1.hdf5'
             }
}
#basepath='mockchallenge-0.2'
#modelpath=''  #the path to where you store the model file


#base_file='mock_challenge_0.2_fsiiii8.0e-03_fiducial.fits.gz'
#cov_file='mock_challenge_0.2_fsiiii8.0e-03_fiducial.fits.gz'

nw = nautilus_wrapper(**arguments)

sampler = nw.generate_sampler(rng=rng)

sampler.run(verbose=True)
