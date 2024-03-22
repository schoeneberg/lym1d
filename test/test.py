import lym1d

import os
path_nersc = "/global/cfs/cdirs/desi/science/lya/y1-p1d/lace_nyx_files"
if not os.path.exists(path_nersc):
  raise Exception("Please enter a new path if you're not working on NERSC!")

lkl_obj = lym1d.lym1d(
    data_directory=path_nersc,
    path="models_Nyx_Oct2023.hdf5",
    runmode="nyx_auvb",
    has_cor='None',
    emupath="Lya_emu_noLP.npz"
    )
cosmo = {'A_lya':9,'n_lya':-2.3,'H0':70,'omega_m':0.14}
therm = {'Fbar':(lambda z:1.3-z*0.25), 'T0':(lambda z:1e4), 'gamma':(lambda z:1.3),'UV':(lambda z:1)}
nuisance = {'noise':[0.1]*13}
chi2 = lkl_obj.chi2_plus_prior(cosmo,therm,nuisance)
import numpy as np
print(chi2)
print("passes random validation ? ",np.round(chi2,decimals=1)==-22313.1)
assert(np.round(chi2,decimals=1)==-22313.1)


from scipy.interpolate import CubicSpline
zs = np.linspace(2.2,4.6,num=13)

lkl_obj = lym1d.lym1d(**{'runmode': 'nyx_auvb', 'An_mode': 'default', 'has_cor': {'noise': True, 'DLA': True, 'reso': True, 'SN': True, 'AGN': True, 'zreio': False, 'SiIII': True, 'SiII': True, 'norm': True, 'UV': False, 'IC': True}, 'zmin': 2.2, 'zmax': 4.6, 'zs': [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6], 'data_filename': 'pk_1d_DR12_13bins.out', 'inversecov_filename': 'pk_1d_DR12_13bins_invCov.out', 'shortening_factor': 0.0, 'convex_hull_mode': False, 'use_H': True,
'data_directory':"/home/nilsor/codes/montepython_lyadesi_public/montepython_public_lyadesi/data/Lya_DESI/",'path':"all_model_outputs_NEW_CORRECTED_with_lP_refine0LHC1_refine1_refine2LHC1_refine3LHC1.hdf5",})
cosmo = {'omega_m': 0.15142460000000002,'A_lya': 6.148379, 'n_lya': -2.360224,'H0':73.87656}
therm = {
'Fbar': CubicSpline(zs,[0.7767282983606734,0.7379092872280267,0.6964453517916305,0.6527479438815664,0.6072983457647912,0.5606348322781511,0.5133366890826097,0.46600570976119254,0.4192460189030024,0.37364324726013615,0.3297441968235758,0.2880381611324573,0.2489409994993224]),
'T0':CubicSpline(zs,[4502.173655565471,4878.248644557711,5261.548183027292,5651.797930185045,6048.748,6290.940560355421,6530.889738094114,6768.716970520429,7004.532141844283,7238.435116908597,7470.517018188775,7700.861297250011,7929.54464012606]),
'gamma':CubicSpline(zs,[1.7375619003768796,1.6362714094491404,1.5461862378388276,1.4655421433210694,1.392927,1.3271970970037443,1.2674162346483282,1.2128106874034283,1.1627354110184753,1.1166483489930101,1.0740906602741438,1.0346713348023413,0.998055101204461]),
'UV':(lambda z:1.596186)
}
nuisance = {'fSiIII':0.005863926, 'fSiII': 0.0009931204,'AGN':0.3541188,'SN':1.027372,'UVfluct':0.7013166,'noise':[0.00731251, 0.005438384, -0.02381092, -0.0404488, -0.05244976, -0.01967307, 0.001954131, 0.02547697 , 0.01904235, 0.05621201 , 0.03249751 , 0.01537355, -0.01813769, 1.596186],'DLA':0.,'reso_ampl':0,'reso_slope':0,'normalization':[1]*13}


chi2 = lkl_obj.chi2_plus_prior(cosmo,therm,nuisance)
print(chi2)
print("passes bestfit validation ? ",np.round(chi2,decimals=2)==-224.24)#-224.39)
assert(np.round(chi2,decimals=2)==-224.24)#-224.39)
import time; t = time.time()
for i in range(10):
  chi2 = lkl_obj.chi2_plus_prior(cosmo,therm,nuisance)
print(" -> {:.4f} s per call".format((time.time()-t)/10.))
