# -*- coding: utf-8 -*-
"""

Backend to implement a general Gaussian-Process based emulator with the underlying george kernels, and perform the relevant GP operations.
Allows for a possible PCA decomposition of the training set as well.

Created 2019

@author: Michael Walther (@Waelthus)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import functools as ft

import george
import numpy as np
import scipy.optimize as op

def predict_weights_func(x, npc, gparr, ww, npcmax,output_cov=True):
    """
    Predicts the principal component weights from a gaussian process (Most input values are set when creating the emulator with create_emulator)

    input:
      x:
        new position in parameter space to be evaluated
      npc:
        number of pca components to be used for this evaluation
      gparr:
        list of gaussian processes to use
      ww:
        array of old weights for each PCA component and each grid point
      npcmax:
        maximum number of PCA components possible for this statistics and parameter grid

    """
    if npc > npcmax:
        npc = npcmax
    new_weights = []
    new_var = []
    if len(x.shape) <= 1:
        x = x[np.newaxis, :]
    # loop over PCA components and predict new values for the PCA weights (and their variance) given new parameters x
    for gp, w in zip(gparr[:npc], ww[:npc]):
        if output_cov:
          nw, nwvar = gp.predict(w, x, return_var=True)
          new_var.append(nwvar)
        else:
          nw = gp.predict(w, x, return_cov=False)
        new_weights.append(nw)

    return np.array(new_weights), np.array(new_var)



def predict_weights_deriv_func(x, npc, gparr, ww, npcmax):
    """
    Predicts the principal component weights derivatives from a gaussian process (Most input values are set when creating the emulator with create_emulator)

    input:
      x:
        new position in parameter space to be evaluated
      npc:
        number of pca components to be used for this evaluation
      gparr:
        list of gaussian processes to use
      ww:
        array of old weights for each PCA component and each grid point
      npcmax:
        maximum number of PCA components possible for this statistics and parameter grid

    """
    if npc > npcmax:
        npc = npcmax
    new_weights = []
    if len(x.shape) <= 1:
        x = x[np.newaxis, :]
    # loop over PCA components and predict new values for the PCA weight derivatives given new parameters x
    for gp, w in zip(gparr[:npc], ww[:npc]):
        nw, nwvar = gp.predict_derivative(w, x)
        new_weights.append(nw)

    return np.array(new_weights)


def emulator_func(x, mean, std, pca, predict_new_weights, npcmax, npc=None, output_cov=True):
    """
    This function computes the real function from the PCA component prediction made by the GP. (Most input values are set when creating the emulator with create_emulator)

    input:
        pars:
          parameters in the same units as when calling create_emulator
        npc:
          by default use the number of PCA components from creating the emulator, setting this allows getting an emulator with fewer components
        other keywords:
          are fixed at create_emulator and a partial func is returned
      output:
        recon:
          reconstructed statistics for this parameter combination
    """
    if not npc:
        npc = npcmax
    em = 0
    # get the maximum number of PCA components possible and set npc to this if higher
    npc = min([npc, mean.size])
    # actually get the new PC-weights (and their covariance)
    new_weights, weights_var = predict_new_weights(x, npc)
    # multiply each pca vector with the respective standard deviation (because the PCA was generated for std normalized components)
    res_pca = pca[0:npc, :] * std[np.newaxis, :]
    # sum up weigts*rescaled pca vectors
    em = np.einsum('ij,ik->kj', res_pca[0:npc, :], new_weights)
    # get the covariance matrix by summing up PCA-vector^T * variance in weights * PCA-vector
    if output_cov:
        Cov_em = np.einsum('ij,ik,il->kjl', res_pca[0:npc, :], weights_var, res_pca[0:npc, :])

    # add the mean to the emulated sum of PCA components
    out = em + mean[np.newaxis, :]

    if out.shape[0] == 1:
        out = out[0, ...]
        if output_cov:
            Cov_em = Cov_em[0, ...]
    # return the emulated statistics and cov_matrix
    return out, (Cov_em if output_cov else None)


def PCA_analysis(stat_grid, npc=5):
    """
    This function converts some grid of training data (e.g. statistics) and returns the corresponding PC vectors, weights, as well as the mean and standard deviation of the grid

    input:
        stat_grid:
          grid of training data
      output:
        PC:
          Principle component Analysis eigenvectors (Principle components)
        ww:
          Principle component Analysis eigenvalues
        meangrid:
          The mean of the grid
        stdgrid:
          The standard deviation of the grid
    """
    meangrid = np.mean(stat_grid, axis=0)
    stdgrid = np.std(stat_grid, axis=0)
    # subtract the mean from the statistics and normalize by standard deviation
    normgrid = (stat_grid.T - meangrid[:, np.newaxis]) / stdgrid[:, np.newaxis]
    nbins = normgrid.shape[0]
    nm = normgrid.shape[1]
    npc = min(min(nm, nbins), npc)
    # svd and pca decomposition
    U, D, V = np.linalg.svd(normgrid, full_matrices=False)
    # combine U and D and rescale to get the principal component vectors
    PC = np.dot(U, np.diag(D))[:, :npc].T / np.sqrt(nm)
    # rescale V to get the principal component weights
    ww = V[:npc, :] * np.sqrt(nm)
    # Variance accounted for by each component
    variance = D**2 / np.sum(D**2)
    # Save to file
    np.savez('PCA_results.npz', PCA=PC, weights=ww,
             meangrid=meangrid, stdgrid=stdgrid, variance=variance)
    return PC, ww, meangrid, stdgrid



def create_emulator(par_grid, stat_grid, smooth_lengths, noise=None, npc=5, optimize=True, output_cov=True,sigma_l=None,sigma_0=None,noPCA=False, kerneltype='SE',make_deriv_emulator=False):
    """
      generates the emulator given a grid of parameters and the statistics for it

      input:
        par_grid:
          grid of parameters

          par_grid.shape[0]: number of models in grid
          par_grid.shape[1]: number of parameters used for each point
        stat_grid:
          grid of statistics at the positions given by par_grid

          stat_grid.shape[0]: number of points in statistic to emulate
          stat_grid.shape[1]: number of parameters used for each point

        smooth_length:
          correlation length in each of the model parameters used as initial
          guess. This is refined later in this routine.

        noise:
          additional noise assumed in the statistics points (if None uses a default value of the george package)

        npc:
          number of principal components to use

        optimize:
          should an optimization be performed on the hyperpars (using downhill simplex algorithm to find maximum likelihood values)

        output_cov:
          should the covariance matrix of the results also be returned from the created emulator?

        sigma_l:
          Kernel length scale for a dot-product kernel

        sigma_0:
          Kernel length scale for a constant kernel

        noPCA:
          Disable the PCA compression

        kerneltype:
          Type of the GP kernel

        make_deriv_emulator:
          Compute additional derivatives of the emulated mean prediction w.r.t. the input parameters
    """
    ndim = par_grid.shape[1]
    nbins = stat_grid.shape[1]
    if not noPCA:
      # compute the PCA, mean and std of the statistics
      PC, ww, mean_stat, std_stat = PCA_analysis(stat_grid, npc)
    else:
      mean_stat = np.mean(stat_grid, axis=0)
      std_stat = np.std(stat_grid, axis=0)
      PC = np.diag(np.ones(nbins))
      ww = (stat_grid.T - mean_stat[:, np.newaxis]) / std_stat[:, np.newaxis]
    # get the maximum possible number of PCA components and dimensionality of the problem
    npcmax = len(PC)
    # set noise to the default value if not set
    if not noise:
        noise = george.gp.TINY
    # choose a squared exp. kernel for the gaussian process add a noise term along the diagonal of the kernel (WhiteKernel does this)
    if george.__version__ < '0.3.0':   #note that this does not include some new options from below
        if kerneltype=='SE':
          kernel = george.kernels.ExpSquaredKernel(smooth_lengths ** 2, ndim=ndim)
        elif kerneltype=='M52':
          kernel = george.kernels.Matern52Kernel(smooth_lengths ** 2, ndim=ndim)
        elif kerneltype=='M32':
          kernel = george.kernels.Matern32Kernel(smooth_lengths ** 2, ndim=ndim)
        kernel+= george.kernels.WhiteKernel(noise ** 2, ndim=ndim)
    # create the gaussian process object
        gp_start = george.GP(kernel)
    else:
        if kerneltype=='SE':
          kernel = george.kernels.ExpSquaredKernel(smooth_lengths ** 2, ndim=ndim)
        elif kerneltype=='M52':
          kernel = george.kernels.Matern52Kernel(smooth_lengths ** 2, ndim=ndim)
        elif kerneltype=='M32':
          kernel = george.kernels.Matern32Kernel(smooth_lengths ** 2, ndim=ndim)
        if sigma_0 is not None:
          kernel*=sigma_0**2
        if sigma_l is not None:
          kernel+=sigma_l**2*george.kernels.DotProductKernel(ndim=ndim)
        gp_start = george.GP(kernel, white_noise=np.log(noise**2),
                             fit_white_noise=(True if not noise is None else False), fit_mean=False)
    # compute the gaussian process given the initial hyperpars and the parameter grid
    gp_start.compute(par_grid)

    # generate lists for gaussian processes for each PCA component
    gparr = []
    resultsarr = []
    pararr = []
    likearr = []
    for i, w in enumerate(ww):
        # copy the gp object to have a new one for this principal component
        gp = copy.deepcopy(gp_start)
        gparr.append(gp)
    if optimize:
        # helper functions for the optimization (from GEORGE docs), here the likelihood
        def weightfunc(i): return 1/(i+1)

        def nll(p):
            # Update the kernel parameters and compute the likelihood.
            ll = 0
            for i,(gp, w) in enumerate(zip(gparr, ww)):
                if george.__version__ < '0.3.0':
                    gp.kernel[:] = p
                else:
                    gp.set_parameter_vector(p)
                ll += weightfunc(i)*gp.lnlikelihood(w, quiet=True)            #in principle one might want to change the weight for different components to make fitting the lower indices more important

            # The scipy optimizer doesn't play well with infinities.
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the likelihood
        def grad_nll(p):
            gll = 0
            for i,(gp, w) in enumerate(zip(gparr, ww)):
                # Update the kernel parameters and compute the likelihood.
                if george.__version__ < '0.3.0':
                    gp.kernel[:] = p
                else:
                    gp.set_parameter_vector(p)
                gll += -weightfunc(i) * gp.grad_lnlikelihood(w, quiet=True)
            return gll

        # get the initial parameters in the space the gp uses
        if george.__version__ < '0.3.0':
            p0 = gparr[0].kernel.vector
        else:
            p0 = gparr[0].get_parameter_vector()

        # optimize the hyperparameters for this principal component (scipy.optimize based, options from there), save best hyperpars to pars
        results = op.minimize(nll, p0, jac=grad_nll, method='L-BFGS-B')
        pars = results.x
        like = -results.fun

        # if this didn't work well try a different optimizer and play the whole game again
        if like < -10:
            newresults = op.minimize(nll, p0, jac=grad_nll, method='Nelder-Mead')
            newlike = -newresults.fun
            print('likelihood (second method):\n {}\n\n'.format(like))

            if newlike > like:
                like = newlike
                pars = newresults.x

#      pars,results=gp.optimize(par_grid,w,method='BFGS',options={'maxiter':30000})
#      print('best args (BFGS) are:\n{}'.format(pars))

        for gp in gparr:
            if george.__version__ < '0.3.0':
                gp.kernel[:] = pars
            else:
                gp.set_parameter_vector(pars)
    else:
        # atm this is just a placeholder if optimization is not wanted
        if george.__version__ < '0.3.0':
            p0 = gparr[0].kernel.vector
        else:
            p0 = gparr[0].get_parameter_vector()
        pars = p0
        results = None
        like = gp.lnlikelihood(w, quiet=True)
    pararr = pars
    likearr = like

    resultsarr = results
    # generate a function to predict PCA weights using the GP objects just created
    pararr = np.array(pararr)
    likearr = np.array(likearr)
    predict_weights = ft.partial(predict_weights_func, gparr=gparr, ww=ww, npcmax=npcmax,output_cov=output_cov)
    # using this function generate an emulator of the statistics
    emulator = ft.partial(emulator_func, mean=mean_stat, std=std_stat, pca=PC,
                          predict_new_weights=predict_weights, npcmax=npcmax, output_cov=output_cov)
    if make_deriv_emulator:
          predict_weights_deriv = ft.partial(predict_weights_deriv_func, gparr=gparr, ww=ww, npcmax=npcmax)
          deriv_emulator = ft.partial(emulator_func, mean=0, std=std_stat, pca=PC,
                          predict_new_weights=predict_weights_deriv, npcmax=npcmax, output_cov=output_cov)        #probably this will throw errors somewhere
    # create an array of all kernel parameters and print it
    #print('best args are:\n{}'.format(pararr))
    # print('likelihoods:\n{}'.format(likearr.T))
    parnames=gparr[0].get_parameter_names()
    outentries=[emulator, None, pararr, parnames]
    if make_deriv_emulator:
      outentries.append(deriv_emulator)
    return outentries

