import numpy as np

c_kms = 299792.458

class Si_nocorr:
  def get_params(self):
    return []
  def eval(self, nuisance, z, Fbar, ks):
    return 1.

class Si_eBOSS:
  dvSiII = 5577.0
  dvSiIII = 2271.0
  
  def get_params(self):
    return ['fSiIII','fSiII']
    
  def eval(self, nuisance, z, Fbar, ks):
    AmpSiIII = nuisance['fSiIII'](z) / (1.0-Fbar)
    AmpSiII  = nuisance['fSiII'](z) / (1.0-Fbar)

    corr = 1.
    corr *= ( 1.0 + AmpSiIII*AmpSiIII + 2.0 * AmpSiIII * np.cos( ks * self.dvSiIII ) )
    corr *= ( 1.0 +   AmpSiII*AmpSiII + 2.0 *  AmpSiII * np.cos( ks *  self.dvSiII ) )
    return corr

class Si_v1:
  lambda_Ha = 1215.67
  lambda_SiIII = 1206.52
  lambda_SiII = 1193.28
  lambda_SiII_2 = 1190.42
  
  silicon_damping = True
  lines = ['SiIII','SiII','SiCross_1','SiCross_2','SiCross_3','SiCross_4']
  has_cor = {line:True for line in ['SiIII','SiII','SiCross_1','SiCross_2','SiCross_3','SiCross_4']}

  def __init__(self):
    self.dv = {}
    self.dv['SiII'] = (1-self.lambda_SiII/self.lambda_Ha)*c_kms
    self.dv['SiIII'] = (1-self.lambda_SiIII/self.lambda_Ha)*c_kms
    #from fig7 of 2505.07974
    #self.dv['SiCross_1'] = (1-lambda_SiII/lambda_SiIII)*c_kms
    #self.dv['SiCross_2'] = (1-lambda_SiII_2/lambda_SiIII)*c_kms
    self.dv['SiCross_1'] = (self.lambda_SiIII/self.lambda_SiII-1)*c_kms  # Convention of Jonas/cup1d
    self.dv['SiCross_2'] = (self.lambda_SiIII/self.lambda_SiII_2-1)*c_kms  # Convention of Jonas/cup1d
    # Other SiII lines we can ignore 1260.42, 1304.37, 1526.72 (higher lambda than Ha)
    self.dv['SiCross_3'] = (self.lambda_SiII/self.lambda_SiII_2-1)*c_kms  # Convention of Jonas/cup1d
    self.dv['SiCross_4'] = (1-self.lambda_SiII_2/self.lambda_Ha)*c_kms  # Convention of Jonas/cup1d

  def get_params(self):
    pars = []
    for line in self.lines:
      if self.has_cor[line]:
        pars.append("f"+line)
        if self.silicon_damping:
          pars.append("a_damp_"+line)
    if self.silicon_damping:
      pars.append("alpha_damp")
    return pars

  def eval(self, nuisance, z, Fbar, ks):
    Amp = {}
    for line in self.lines:
      if self.has_cor[line]:
        Amp[line] = nuisance["f"+line](z)/(1.0-Fbar)

    if self.silicon_damping == True:
      damping = {}
      #a_damp = nuisance['a_damp'](z)
      alpha_damp = nuisance['alpha_damp']
      #damping0 = (1+a_damp * ks)**alpha_damp * np.exp(-(a_damp * ks) ** alpha_damp)
      for line in self.lines:
        a_damp = nuisance['a_damp_'+line](z)
        #damping[line] = (1+a_damp * ks)**alpha_damp * np.exp(-(a_damp * ks) ** alpha_damp)
        damping[line] = 1-1/(1+np.exp(-(a_damp * ks)))
    else:
      damping = {line:1 for line in self.lines}

    corr = 1.
    for line in self.lines:
      corr *= ( 1.0 + Amp[line]**2 *damping[line]**2 + 2.0 * Amp[line] * np.cos( ks * self.dv[line] ) * damping[line] )

    return corr

def get_Si_model(modelname):
  if "boss" in modelname.lower():
    return Si_eBOSS()
  elif "v1" in modelname.lower():
    return Si_v1()
  elif "no" in modelname.lower():
    return Si_nocorr()
  else:
    raise ValueError("unrecognized silicon model: ",modelname)
