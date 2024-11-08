import numpy as np

blinding_widths = {'omega_m':0.01,  'Delta_star':0.1, 'n_star':0.01, 'alpha_star':0.01, 'A_lya':0.1, 'n_lya':0.01, 'Delta_lya_from_lym1d':0.1, 'n_lya_from_lym1d':0.01, 'alpha_lya_from_lym1d':0.01, 'sigma8':0.1, 'n_s':0.01, 'A_lya_skm':0.1, 'n_lya_skm':0.01}

def get_blindings(handle):
  if not handle:
    return {name:0 for name in blinding_widths}

  else:
    # If str, then it is a blinding key
    if isinstance(handle, str):
      blinding_key = handle
      blinding_seed = int.from_bytes(blinding_key.encode('utf-8'), byteorder='big')
    # If int, then it is the seed
    elif isinstance(handle, int):
      blinding_seed = handle
    # Otherwise, assume it's a valid HDU from astropy.io.fits
    else:
      # If it does not contain the 'P1D_BLIND' field, it's not blinded
      if not ('P1D_BLIND' in handle):
        return {name:0 for name in blinding_widths}
      if not ('BLINDING' in handle['P1D_BLIND'].header):
        raise ValueError("Contains 'P1D_BLIND', but the header doesn't contain the 'BLINDING' keyword. Corrupted file")
      blinding_key = handle['P1D_BLIND'].header['BLINDING']
      blinding_seed = int.from_bytes(blinding_key.encode('utf-8'), byteorder='big')
    rng = np.random.default_rng(blinding_seed)
    return {name:rng.normal(0, blinding_widths[name]) for name in blinding_widths}

