"""
=======================
iminuit_interface
=======================
E. Armengaud - 2024
Part of the lym1d package

- Simple interface to iminuit, handling arbitrary likelihood, parameters, priors.
- Wrapper likelihood to lym1d
"""


import time
import iminuit
import functools
import yaml
import numpy as np


def read_minuitparams(yaml_file):
    """ Reads yaml file describing Minuit parameters
    
    Parameters
    ----------
    yaml_file : :class:`str`
        YAML-format list of parameters, see example lya_fitpars_np20_v1.yaml
        for each entry: 'value' is requested,
                        'fix', 'limit', 'error', 'prior' are optional

    Returns
    -------
    :class:`dict`
        "Model" dictionnary to be used by RunMinuit
    """
    with open(yaml_file) as f:
        yaml_dict = yaml.safe_load(f)
    model_info = dict()
    for param_name in yaml_dict.keys():
        parinfo = yaml_dict[param_name]
        model_info[param_name] = parinfo['value']
        for kwd in ['fix', 'limit', 'error', 'prior']:
            if kwd in parinfo:
                model_info[kwd+'_'+param_name] = parinfo[kwd]
        #- Ambiguity if a param is fixed and has a prior at the same time: we don't want that 
        if ('fix_'+param_name in model_info.keys()) and (model_info['fix_'+param_name]==True):
            if 'prior_'+param_name in model_info.keys():
                raise ValueError(param_name + ": Minuit parameter is fixed, cannot impose a prior")
    return model_info



def args_to_kwargs(func):
    """ Trick, see eg. montelss.minimizer (A. De Mattias)
        Requires self.parameters
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        kwargs.update({key:val for key,val in zip(self.parameters, args)})
        return func(self, **kwargs)
    return wrapper


class _CallableForIMinuit:
    """ Interface function to iminuit """
    
    def __init__(self, function, anydata, func_is_chi2):
        self.anydata = anydata
        self.function = function
        #- If func_is_chi2 is False, assumes function returns ln(likelihood)
        self.is_chi2 = func_is_chi2
        self.parameters = None
        self.priors = None

    def set_parameters(self, params):
        self.parameters = params
        
    def set_priors(self, priors):
        self.priors = priors

    @args_to_kwargs
    def __call__(self, **kwargs):
        """ Function called by iminuit.
            It calls `self.function`, and optionally adds a gaussian prior.
            Returns a chi2 value
        """
        func_value = self.function(self.anydata, **kwargs)
        #- Convert to chi2
        if not self.is_chi2:
            func_value = -2.*func_value
        #- Add priors
        if self.priors is not None:
            for par in self.priors.keys():
                func_value += ((kwargs[par]-self.priors[par][0])/self.priors[par][1])**2
        return func_value


def RunMinuit(anydata, model, function=None, verbose=False, minos=False, func_is_chi2=True):
    """ Wrapper to iMinuit, initializes and calls Minuit to fit model to data
    
    Parameters
    ----------
    anydata: object containing any kind of information to be provided to function
    model: :class:`dict`
        Minuit parameters to be used. 
        For each fit parameter PAR, model must include 'PAR'=initial_value,
            optionally: 'limit_PAR'=(min, max), 'fix_PAR'=bool, 'error_PAR'=float
            optionally: 'prior_PAR'=(central value, sigma) gaussian prior added to the likelihood
    function: callable
        Call should be function(anydata, **params)
    minos: :class:`bool`
        If True, run minos after migrad.
    func_is_chi2: :class:`bool`
        If True, interpret function as chi2, else interpret function as ln(likelihood)
     
    Returns
    -------
    :class:`iminuit.Minuit`
        Minuit object containing fit results
    """
    
    callfunc = _CallableForIMinuit(function, anydata, func_is_chi2)
    parameters = [ x for x in model.keys() if x[:6] not in ['limit_','error_','prior_'] and x[:4]!='fix_' ]
    callfunc.set_parameters(parameters)
    priors = { x[6:]:model[x] for x in model.keys() if x[:6]=='prior_' }
    callfunc.set_priors(priors)
    if verbose:
        print("* Minuit parameters:")
        for par in parameters:
            if not ('fix_'+par in model) or model['fix_'+par]==False:
                print("    *", par, ":", model[par], "(varying)")
        for par in parameters:
            if ('fix_'+par in model) and model['fix_'+par]==True:
                print("    *", par, ":", model[par], "(fixed)")
        print("*  Gaussian priors:",callfunc.priors)
    kwargs = model.copy()
    for key in priors.keys(): kwargs.pop('prior_'+key)
    
    # keep compatibility with old versions of iminuit:
    if iminuit.__version__ > '1.4.2' :
        for par in parameters:
            if ('fix_'+par in model): kwargs.pop('fix_'+par)
            if ('limit_'+par in model): kwargs.pop('limit_'+par)
            if ('error_'+par in model): kwargs.pop('error_'+par)
    else:
        kwargs['pedantic'] = False
        kwargs['print_level'] = 1 if verbose else 0

    if iminuit.__version__ <= '1.4.2' :
        minuit_obj = iminuit.Minuit(callfunc, forced_parameters=parameters, **kwargs)
    else:
        minuit_obj = iminuit.Minuit(callfunc, name=parameters, **kwargs)
        for par in parameters:
            if ('fix_'+par in model) and model['fix_'+par]==True:
                minuit_obj.fixed[par] = True
            if ('limit_'+par in model):
                minuit_obj.limits[par] = tuple(model['limit_'+par])
            if ('error_'+par in model):
                minuit_obj.errors[par] = model['error_'+par]
    minuit_obj.print_level = 1 if verbose else 0

    ti=time.time()
    minuit_obj.migrad()
    tf=time.time()
    if verbose:
        print("* Migrad time:",tf-ti,"sec.")
    if iminuit.__version__ <= '1.4.2':
        fmin = minuit_obj.get_fmin()
    else:
        fmin = minuit_obj.fmin
    if fmin.is_valid == False : print("* Warning: Migrad fit not valid")
    if minos:
        ti=time.time()
        minuit_obj.minos()
        tf=time.time()
        if verbose: print("* Minos time:",tf-ti,"sec.")
    return minuit_obj


def lym1d_raw_chi2_to_iminuit(wrapper_obj, **params):
    return wrapper_obj.raw_chi2(params)


def lym1d_chi2_wrapper(lym1d_obj, **params):
    """ Computes Lya log likelihood: wrapper for iminuit
    Currently only works in the case of a 'Taylor' likelihood.

    Parameters
    ----------
    lym1d_obj : :class:`lym1d`
    params : :class:`dict`
        likelihood parameters {param: value}
        
    Returns
    -------
    chi2 : float
    """
    
    #- lym1d takes as separate input: cosmo, therm, nuisance:
    #- cosmo dict
    cosmopar = {'Omega_m': params['Omega_m'],
                'H0': 100.*params['h'],
                'sigma8': params['sigma8'],
                'n_s': params['n_s'],
                'z_reio': 10.0,
                'Omega_nu': 0,  # params['m_nu']/193eV  TODO support neutrino mass
                }
    
    #- thermal dict
    T0_func = lambda z: params['T0'] * pow((1+z)/4.0,
                                           (params['T0SlopeInf'] if z<=3 else params['T0SlopeSup']))
    gamma_func = lambda z:params['gamma']*pow((1+z)/4.0,
                                              (params['gammaSlopeInf'] if z<=3 else params['gammaSlopeInf']))
    # ! definition of AmpTauEff: here stick to the definition in Taylor grid
    AmpTauEff_rescaled = params['AmpTauEff'] * 4**params['SlopeTauEff']
    taueff_func = lambda z: AmpTauEff_rescaled*pow((1+z)/4,
                                                   (params['SlopeTauEff'] if z<=3 else params['SlopeTauEff']))
    Fbar_func = lambda z: np.exp(-taueff_func(z))
    therm = {'Fbar':Fbar_func,
             'gamma':gamma_func,
             'T0':T0_func,
            }
    
    #- nuisance dict
    nuisance = {'noise':[], 'normalization':[], 'tauError':[]}
    for i in range(1, 14):
        nuisance['noise'].append(params['noise'+str(i)])
        nuisance['normalization'].append(1)  # We shall not use something else
        nuisance['tauError'].append(0.0)  # TODO check if useful?
    for key in ['DLA', 'SN', 'AGN', 'UVFluct']:
        nuisance[key] = params['Lya_'+key]
    for key in ['fSiIII', 'fSiII', 'SlopeTauEff']:
        nuisance[key] = params[key]
    nuisance['reso_ampl'] = params['ResoAmpl']
    nuisance['reso_slope'] = params['ResoSlope']
    nuisance['splicing_corr'] = params['SplicingCorr']
    nuisance['splicing_offset'] = params['SplicingOffset']    
    nuisance['AmpTauEff'] = AmpTauEff_rescaled
    
    return lym1d_obj.chi2(cosmopar, therm, nuisance)
