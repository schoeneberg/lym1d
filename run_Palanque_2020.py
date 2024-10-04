
info = {'likelihood':{'lym1d_wrapper.cobaya_wrapper':{

          'base_directory':'/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/taylor_files/',

          'runmode':'taylor_splicing',

          'arguments':{'has_cor':{'noise': True, 'DLA': False,'reso': True, 'SN': True, 'AGN': True, 'zreio':False,'SiIII':True,'SiII':True,'norm':True,'UV':True,'splice':True},
                       'nuisance_replacements':[],
                       'data_filename':'pk_1d_DR12_13bins.out',
                       'H0prior':{'mean':67.3,'sigma':1.0},
                       'use_thermal_prior':True,
                       'gammaPriorMean':1.3,
                       'splice_kind':2,
                       }
         }},
        'params':{'omega_cdm':{'prior':{'min':0.00,'max':0.5},'ref':{'dist':'norm','loc':1.019436e-01,'scale':0.013}},
                  'h':{'prior':{'min':0.55,'max':0.8},'ref':{'dist':'norm','loc':0.6779706,'scale':0.01}},
                  'omega_b':0.022253,
                  'ln_A_s_1e10':{'prior':{'min':0.0,'max':5.0},'ref':{'dist':'norm','loc':3.251898e+00,'scale':0.015}},
                  'n_s':{'prior':{'min':0.5,'max':1.5},'ref':{'dist':'norm','loc':9.538248e-01,'scale':0.0042}},
                  'tau_reio':0.0561,
                  'm_ncdm':{'prior':{'min':0.0,'max':5.0},'ref':{'dist':'norm','loc':7.674836e-03/3,'scale':1e-2}},
                  'T0':{'prior':{'min':0.0,'max':1e5},'ref':{'dist':'norm','loc':7.473512e+03,'scale':5e3}},
                  'gamma':{'prior':{'min':0.0,'max':5.0},'ref':{'dist':'norm','loc':1.026419e+00,'scale':0.2}},
                  'AmpTauEff':{'prior':{'min':0.0,'max':1.5},'ref':{'dist':'norm','loc':4.833614e-01,'scale':0.03}},
                  'SlopeTauEffInf':{'prior':{'min':0.0,'max':7.0},'ref':{'dist':'norm','loc':3.840216e+00,'scale':0.3}},
                  'SlopeTauEffBreak':0,
                  'fSiIII':{'prior':{'min':-0.2,'max':0.2},'ref':{'dist':'norm','loc':5.987892e-03,'scale':2e-3}},
                  'fSiII':{'prior':{'min':-0.2,'max':0.2},'ref':{'dist':'norm','loc':4.267999e-04,'scale':2e-4}},
                  'T0SlopeInf':{'prior':{'min':-5.0,'max':2.0},'ref':{'dist':'norm','loc':-4.240167e+00,'scale':1.0}},
                  'T0SlopeBreak':{'prior':{'min':-15.0,'max':7.0},'ref':{'dist':'norm','loc':-1.271859e-02,'scale':1e-2}},
                  'gammaSlopeInf':{'prior':{'min':-5.0,'max':2.0},'ref':{'dist':'norm','loc':9.863761e-01,'scale':0.5}},
                  'gammaSlopeBreak':0,
                  'SplicingCorr': {'prior':{'min':-40.0,'max':40.0},'ref':{'dist':'norm','loc':7.823827e-02,'scale':0.5}},
                  'SplicingOffset':{'prior':{'min':-1.0,'max':1.0},'ref':{'dist':'norm','loc':1.064135e-01,'scale':0.05}},
                  'ResoAmpl':0,
                  'ResoSlope':0,
                  'Lya_AGN':{'prior':{'min':0.0,'max':3.0},'ref':{'dist':'norm','loc':3.637030e-01,'scale':0.03}},
                  'Lya_SN': {'prior':{'min':0.0,'max':3.0},'ref':{'dist':'norm','loc':1.107606e+00,'scale':0.03}},
                  'Lya_UVFluct':{'prior':{'min':0.0,'max':3.0},'ref':{'dist':'norm','loc':5.972924e-02,'scale':0.01}},
                  'noise1':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':1.810679e-03,'scale':0.003}},
                  'noise2':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':-9.112814e-03,'scale':0.003}},
                  'noise3':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':-3.098166e-02,'scale':0.003}},
                  'noise4':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':-4.527296e-02,'scale':0.003}},
                  'noise5':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':-6.295360e-02,'scale':0.003}},
                  'noise6':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':-2.185368e-02,'scale':0.003}},
                  'noise7':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':-1.260106e-02,'scale':0.003}},
                  'noise8':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':1.568410e-02,'scale':0.003}},
                  'noise9':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':1.728685e-02,'scale':0.003}},
                  'noise10':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':2.316432e-02,'scale':0.003}},
                  'noise11':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':3.911431e-02,'scale':0.003}},
                  'noise12':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':7.262989e-03,'scale':0.003}},
                  'noise13':{'prior':{'min':-2.0,'max':2.0},'ref':{'dist':'norm','loc':1.428109e-03,'scale':0.003}},
                  'z_reio':{'latex':r'z_\mathrm{reio}'},
                  'Omega_Lambda':{'latex':r'\Omega_\Lambda'},
                  'Omega_m':{'latex':r'\Omega_m'},
                  'YHe':{'latex':r'Y_\mathrm{He}'},
                  'theta_s_100':{'latex':r'100\cdot \theta_s'},
                  'A_s':{'latex':r'A_s'},
                  'sigma8':{'latex':r'\sigma_8'},
                  },

        'sampler':{'mcmc':{
          'covmat':'palanque_covmat_v3.dat',
          'max_tries':10000},
          },
        'theory':{'classy':{'extra_args':{'non_linear':'no',
                                          'N_ur':0.00641,
                                          'N_ncdm':1,
                                          'T_ncdm':0.71611,
                                          'deg_ncdm':3,
                                          #'output':'mPk',
                                          #'P_k_max_1/Mpc':1,
                                          }}},
        'output':'chains/eBOSS_taylor_mnu',
        #'force':True,
        'resume':True,
        #'debug':True
        }

# Technically one could also do has_cor['norm'] = False, but this is for consistency with the MP parameter file
for i in range(13):
  info['params']['normalization{}'.format(i+1)]=1






# For tests only!
#from cobaya.model import get_model
#model = get_model(info)

# For actual runs:
from cobaya.run import run
updated_info, sampler = run(info)

# Now convert to liquidcosmo and plot (you can do anything with it that you want!)
import liquidcosmo as lc
fo = lc.load_from(samp)

fo['Omega_m','h','sigma8','n_s'].plot_getdist(show=True)

# Now to convert to getdist and plot (you can do anything with it that you want!)
#gd_sample = sampler.products(to_getdist=True)["sample"]

#import getdist.plots as gdplt

#gdplot = gdplt.get_subplot_plotter()
#gdplot.triangle_plot(gd_sample, filled=True)

#import matplotlib.pyplot as plt

#plt.show()
