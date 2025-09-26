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

#all credit to Jonas/cup1d (https://github.com/igmhub/cup1d/blob/main/cup1d/nuisance/si_mult.py)
class Si_Jonas:
  wav = {"SiIII": 1206.51, "SiIIc": 1260.42, "SiIIb": 1193.28, "SiIIa": 1190.42, "Lya": 1215.67}
  osc_strength = {"SiIII": 1.67,"SiIIc": 1.22,"SiIIb": 0.575,"SiIIa": 0.277}
  def __init__(self):
    vel_diff = lambda lambda1, lambda2: np.abs(np.log(lambda2 / lambda1)) * c_kms
    self.dv =  {"SiIII_Lya": vel_diff(self.wav["SiIII"], self.wav["Lya"]),
                "SiIIc_Lya": vel_diff(self.wav["SiIIc"], self.wav["Lya"]),
                "SiIIb_Lya": vel_diff(self.wav["SiIIb"], self.wav["Lya"]),
                "SiIIa_Lya": vel_diff(self.wav["SiIIa"], self.wav["Lya"]),
                "SiIII_SiIIc": vel_diff(self.wav["SiIII"], self.wav["SiIIc"]),
                "SiIII_SiIIb": vel_diff(self.wav["SiIII"], self.wav["SiIIb"]),
                "SiIII_SiIIa": vel_diff(self.wav["SiIII"], self.wav["SiIIa"]),
                "SiIIc_SiIIb": vel_diff(self.wav["SiIIc"], self.wav["SiIIb"]),
                "SiIIc_SiIIa": vel_diff(self.wav["SiIIc"], self.wav["SiIIa"]),
                "SiIIb_SiIIa": vel_diff(self.wav["SiIIb"], self.wav["SiIIa"])}
    rstrength = lambda lambda1, lambda2, f1, f2: (lambda1 * f1) / (lambda2 * f2)
    self.rat = {
        "SiIIa_SiIII": rstrength(
            self.wav["SiIIa"],
            self.wav["SiIII"],
            self.osc_strength["SiIIa"],
            self.osc_strength["SiIII"],
        ),
        "SiIIb_SiIII": rstrength(
            self.wav["SiIIb"],
            self.wav["SiIII"],
            self.osc_strength["SiIIb"],
            self.osc_strength["SiIII"],
        ),
        "SiIIc_SiIII": rstrength(
            self.wav["SiIIc"],
            self.wav["SiIII"],
            self.osc_strength["SiIIc"],
            self.osc_strength["SiIII"],
        ),
        "SiIIa_SiIIc": rstrength(
            self.wav["SiIIa"],
            self.wav["SiIIc"],
            self.osc_strength["SiIIa"],
            self.osc_strength["SiIIc"],
        ),
        "SiIIb_SiIIc": rstrength(
            self.wav["SiIIb"],
            self.wav["SiIIc"],
            self.osc_strength["SiIIb"],
            self.osc_strength["SiIIc"],
        ),
    }
    self.off = {
        "SiIII_Lya": 1,
        "SiIIa_Lya": 1,
        "SiIIb_Lya": 1,
        "SiIIc_Lya": 0,
        "SiIII_SiIIa": 1,
        "SiIII_SiIIb": 1,
        "SiIII_SiIIc": 0,
        "SiIIc_SiIIb": 0,
        "SiIIc_SiIIa": 0,
        "SiIIb_SiIIa": 0,
    }

    def get_params(self):
      list_coeffs = [
          "f_Lya_SiIII",
          "s_Lya_SiIII",
          "f_Lya_SiII",
          "s_Lya_SiII",
          "f_SiIIa_SiIII",
          "f_SiIIb_SiIII",
          # "f_SiIIa_SiIIb",
      ]
      return list_coeffs
    def eval(self, nuisance, z, Fbar, ks):
      ra3 = self.rat["SiIIa_SiIII"]
      rb3 = self.rat["SiIIb_SiIII"]
      rc3 = self.rat["SiIIc_SiIII"]
      # SiII-SiII only additive
      self.off["SiIIb_SiIIa"] = 0
      self.off["SiIIc_SiIIa"] = 0
      self.off["SiIIc_SiIIb"] = 0

      vals = nuisance ##Using Jonas' notation

      # k-dependent damping of Lya-SiIII
      G_SiIII_Lya = 2 - 2 / (
          1 + np.exp(-vals["s_Lya_SiIII"](z) * ks)
      )
      # k-dependent damping of Lya-SiII
      G_SiII_Lya = 2 - 2 / (
          1 + np.exp(-vals["s_Lya_SiII"](z) * ks)
      )

      # scale amplitude of Cmm
      G_SiII_SiIII = vals.get("f_SiIIb_SiIII",lambda z:1)(z)

      # deviations from optically-thin limit
      f_SiIIa_SiIII = vals.get("f_SiIIa_SiIII",lambda z:1)(z)

      # not relevant anymore modeled here anymore
      if "s_SiIIa_SiIIb" in vals:
          G_SiII_SiII = 2 - 2 / (
              1 + np.exp(-vals["s_SiIIa_SiIIb"](z) * ks)
          )
      else:
          G_SiII_SiII = 1
      if "f_SiIIa_SiIIb" in vals:
          G_SiII_SiII *= vals["f_SiIIa_SiIIb"](z)

      # amplitude of SiIII
      aSiIII = vals["f_Lya_SiIII"](z) / (1 - Fbar)
      # amplitude of SiII, rb3 for convenience
      aSiII = rb3 * vals["f_Lya_SiII"](z) / (1 - Fbar)
      # deviations of ra3 from optically-thin limit
      _ra3 = ra3 * f_SiIIa_SiIII

      # print(G_SiII_SiIII)
      # print(_ra3 / rb3)

      C0 = aSiIII**2 * self.off["SiIII_Lya"] + aSiII**2 * (
          (_ra3 / rb3) ** 2 * self.off["SiIIa_Lya"]
          + self.off["SiIIb_Lya"]
          + (rc3 / rb3) ** 2 * self.off["SiIIc_Lya"]
      )

      CSiIII_Lya = (
          2
          * aSiIII
          * G_SiIII_Lya
          * self.off["SiIII_Lya"]
          * np.cos(self.dv["SiIII_Lya"] * ks)
      )

      CSiII_Lya = (
          2
          * aSiII
          * G_SiII_Lya
          * (
              self.off["SiIIa_Lya"]
              * (_ra3 / rb3)
              * np.cos(self.dv["SiIIa_Lya"] * ks)
              + self.off["SiIIb_Lya"]
              * np.cos(self.dv["SiIIb_Lya"] * ks)
              + self.off["SiIIc_Lya"]
              * (rc3 / rb3)
              * np.cos(self.dv["SiIIc_Lya"] * ks)
          )
      )

      Cam = CSiIII_Lya + CSiII_Lya

      Cmm = (
          2
          * aSiIII
          * aSiII
          * G_SiII_SiIII
          * (
              self.off["SiIII_SiIIc"]
              * (rc3 / rb3)
              * np.cos(self.dv["SiIII_SiIIc"] * ks)
              + self.off["SiIII_SiIIb"]
              * np.cos(self.dv["SiIII_SiIIb"] * ks)
              + self.off["SiIII_SiIIa"]
              * (_ra3 / rb3)
              * np.cos(self.dv["SiIII_SiIIa"] * ks)
          )
      )

      Cm = (
          2
          * aSiII**2
          * G_SiII_SiII
          * (
              self.off["SiIIc_SiIIb"]
              * (rc3 / rb3)
              * np.cos(self.dv["SiIIc_SiIIb"] * ks)
              + self.off["SiIIc_SiIIa"]
              * (rc3 / rb3)
              * (ra3 / rb3)
              * np.cos(self.dv["SiIIc_SiIIa"] * ks)
              + self.off["SiIIb_SiIIa"]
              * (_ra3 / rb3)
              * np.cos(self.dv["SiIIb_SiIIa"] * ks)
          )
      )

      metal_corr = (1 + C0 + Cam + Cmm + Cm)

      return metal_corr

def get_Si_model(modelname):
  if "boss" in modelname.lower():
    return Si_eBOSS()
  elif "v1" in modelname.lower():
    return Si_v1()
  elif "jonas" in modelname.lower():
    return Si_Jonas()
  elif "no" in modelname.lower():
    return Si_nocorr()
  else:
    raise ValueError("unrecognized silicon model: ",modelname)
