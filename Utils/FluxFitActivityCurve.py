import os

import pandas as pd
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from RMS.Misc import formatScientific
from RMS.Math import lineFunc
from Utils.Flux import calculateZHR

def showerActivity(sol, sol_peak, background_flux, peak_flux, bp, bm):
    """ Shower activity model described with a double exponential. 

    Arguments:
        sol: [float or ndarray] Solar longiude (deg), independant variable.
        sol_peak: [float] Solar longitude of the peak (deg).
        background_flux: [float] The flux of the sporadic background, the baseline flux.
        peak_flux: [float] The flux of the peak.
        bp: [float] Rising parameter.
        bm: [float] Falling parameter.

    Return:
        flux: [float or ndarray]
    """

    # Determine if the given solar longitude is before or after the peak
    angle_diff = (sol%360 - sol_peak + 180)%360 - 180

    if angle_diff <= 0:
        b = bp
        sign = 1

    else:
        b = bm
        sign = -1

        # Handle symmetric activity which is defined as Bm being zero and thus Bp should be used too
        if bm == 0:
            b = bp

    # Compute the flux
    flux = background_flux + peak_flux*10**(sign*b*angle_diff)

    return flux

# Vectorize the shower activity function
showerActivityVect = np.vectorize(showerActivity, \
    excluded=['sol_peak', 'background_flux', 'peak_flux', 'bp', 'bm'])


def showerActivityNoBackground(sol, sol_peak, peak_flux, bp, bm):
    """ A modified shower activity model without the background flux. This is used for fitting additional 
    peaks.

    Arguments:
        sol: [ndarray] Solar longiude (deg), independant variable.
        sol_peak: [float] Solar longitude of the peak (deg).
        peak_flux: [float] The flux of the peak.
        bp: [float] Rising parameter.
        bm: [float] Falling parameter.
    
    Return:
        flux: [float or ndarray]
    """
    return showerActivityVect(sol, sol_peak, 0, peak_flux, bp, bm)


def showerActivityCombined(sol, all_models):
    """ Compute the combined model flux.

    Arguments:
        sol: [float or ndarray] Solar longitude (deg), independant variable.
        all_models: [list] List of model parameters. Each model is a tuple of the model. The base model has
            5 parameters and each additional peak has 4 parameters.

    Return:
        flux: [ndarray]
    """

    # Compute base model component
    flux_modelled = showerActivityVect(sol, *all_models[:5])

    # Compute the remining residuals using the additional peaks
    if len(all_models) > 5:

        additional_models = all_models[5:]

        # Split the remianing models into groups of 4 paramers
        additional_models = [additional_models[i:i+4] for i in range(0, len(additional_models), 4)]

        # Compute the contribution of each additional peak
        for model in additional_models:
            flux_modelled += showerActivityNoBackground(sol, *model)


    return flux_modelled


def showerActivityResiduals(model_fit, sol, flux, fit_background, weights):
    """ Compute the residuals of the model fit for the scipy.optimize.minimize function.
    
    Arguments:
        model_fit: [list] List of model parameters.
        sol: [ndarray] Solar longitude (deg), independant variable.
        flux: [ndarray] Observed flux.
        fit_background: [bool] If True, the background flux is fitted. If False, the background flux is fixed 
            to 0.
        weights: [ndarray] Weights for each data point.
    """

    # Compute the residuals
    if fit_background:
        residuals = flux - showerActivityVect(sol, *model_fit)
    else:
        residuals = flux - showerActivityNoBackground(sol, *model_fit)

    # Compute the weighted sum of absolute residuals
    return np.sum(np.abs(residuals*weights))


def showerActivityResidualsAll(all_models, sol, flux, weights):
    """ Compute the residuals using all peaks. The background is only estimated for the base model.

    Arguments:
        all_models: [list] List of model parameters. Each model is a tuple of the model. The base model has
            5 parameters and each additional peak has 4 parameters.
        sol: [ndarray] Solar longitude (deg), independant variable.
        flux: [ndarray] Observed flux.
        weights: [ndarray] Weights for each data point.

    Return:
        residuals: [ndarray]
    """

    # Compute the combined flux
    flux_modelled = showerActivityCombined(sol, all_models)

    # Compute the residuals
    residuals = flux - flux_modelled

    # Compute the weighted sum of absolute residuals
    return np.sum(np.abs(residuals*weights))



class ShowerActivityParameters(object):
    def __init__(self, sol_peak, background_flux, peak_flux, bp, bm):
        """ Parameters of the shower activity model.

        Arguments:
            sol_peak: [float] Solar longitude of the peak (deg).
            background_flux: [float] The flux of the sporadic background, the baseline flux.
            peak_flux: [float] The flux of the peak.
            bp: [float] Rising parameter.
            bm: [float] Falling parameter.
        """

        self.sol_peak = sol_peak
        self.background_flux = background_flux
        self.peak_flux = peak_flux
        self.bp = bp
        self.bm = bm

        self.background_zhr = None
        self.peak_zhr = None


    def computeZHR(self, population_index):
        """ Compute the background and peak ZHR. """

        # Compute the background ZHR
        self.background_zhr = calculateZHR(self.background_flux, population_index)

        # Compute the peak ZHR
        self.peak_zhr = calculateZHR(self.peak_flux, population_index)


    def fluxParameters(self, no_bg=False):
        """ Return the flux parameters. 
        
        Keyword arguments:
            no_bg: [bool] If True, the background flux is not returned.
        """

        if no_bg:
            return [self.sol_peak, self.peak_flux, self.bp, self.bm]
        else:
            return [self.sol_peak, self.background_flux, self.peak_flux, self.bp, self.bm]

    def zhrParameters(self, no_bg=False):
        """ Return the ZHR parameters. 
        
        Keyword arguments:
            no_bg: [bool] If True, the background ZHR is not returned.
        """

        if no_bg:
            return [self.sol_peak, self.peak_zhr, self.bp, self.bm]
        else:
            return [self.sol_peak, self.background_zhr, self.peak_zhr, self.bp, self.bm]


    def extractParameters(self, no_bg=False, zhr=False):
        """ Extract the parameters. 
        
        Keyword arguments:
            no_bg: [bool] If True, the background flux is not returned.
            zhr: [bool] If True, the ZHR is returned instead of the flux.
        """

        if zhr:
            return self.zhrParameters(no_bg=no_bg)
        else:
            return self.fluxParameters(no_bg=no_bg)




class ShowerActivityModel(object):
    def __init__(self, initial_param_estimation='auto', sol_peak0=None, bg_flux0=None, peak_flux0=None, 
        bp0=None, bm0=None, refine_fits_individually=True, mc_error_estimation=False, mc_runs=100):
        """ Model for shower activity as described by one or more double exponentials.
            Initial fit parameters can either be determined automatically or given manually.
            If more then one peak is given, the base peak should be the widest and only for that peak the 
            background will be estimated.

        Keyword arguments:
            initial_param_estimation: [str] 'auto' or 'manual'. If 'auto', the initial parameters will be
                determined automatically. If 'manual', they should be specified in the arguments below.
            sol_peak0: [float] Solar longitude of the peak (deg).
            bg_flux0: [float] The flux of the sporadic background, the baseline flux.
            peak_flux0: [float] The flux of the peak.
            bp0: [float] Rising parameter.
            bm0: [float] Falling parameter.
            refine_fits_individually: [bool] If True, the fits on each peak are refined individually. 
                If False, the peaks are only fit together. This should be set to False if the peaks are
                hard to fit automatically and some manual refinement is required.
            mc_error_estimation: [bool] If True, the error on the fit parameters is estimated using a
                Monte Carlo method.
            mc_runs: [int] Number of Monte Carlo runs. Only used if mc_error_estimation is True.
                100 runs by default.
        """

        self.initial_param_estimation = initial_param_estimation

        self.refine_fits_individually = refine_fits_individually

        self.mc_error_estimation = mc_error_estimation
        self.mc_runs = mc_runs


        # Check that initial parameters were given if manual estimation is used
        if initial_param_estimation == 'manual':
            if (sol_peak0 is None) or (bg_flux0 is None) or (peak_flux0 is None) or (bp0 is None) \
                or (bm0 is None):

                raise ValueError("All initial parameters must be specified if manual estimation is used.")


        # Assign initial parameters
        self.initial_base_parameters = ShowerActivityParameters(sol_peak0, bg_flux0, peak_flux0, bp0, bm0)

        # Parameters of the base fit
        self.base_fit = None
        self.base_zhr = None

        # Initial parameters of additional peaks
        self.additional_peaks = []

        # List of fit parameters of additional peaks
        self.additional_fits = []
        self.additional_zhr = []


    def addPeak(self, sol_peak, peak_flux, bp, bm):
        """ Add a peak to the model. The background is not estimated for additional peaks.

        Arguments:
            sol_peak: [float] Solar longitude of the peak (deg).
            peak_flux: [float] The flux of the peak.
            bp: [float] Rising parameter.
            bm: [float] Falling parameter.
        """

        self.additional_peaks.append(ShowerActivityParameters(sol_peak, 0.0, peak_flux, bp, bm))


    def estimateInitialParameters(self, sol_data, flux_data):
        """ Estimate the initial parameters for the fit.

        Arguments:
            sol_data: [ndarray] Solar longitudes (deg).
            flux_data: [ndarray] Fluxes.
        """

        # Estimate the background
        bg_flux0 = np.percentile(flux_data, 2.0)

        # Estimate the peak as the flux-weighted sol
        sol_peak0 = np.average(sol_data, weights=flux_data)

        # Estimate the peak flux
        peak_flux0 = np.percentile(flux_data, 95.0) - bg_flux0


        # Estimate initial rise parameter
        sol_data_rise = sol_data[sol_data < sol_peak0]
        flux_data_rise = flux_data[sol_data < sol_peak0]
        popt, _ = scipy.optimize.curve_fit(lineFunc, sol_data_rise, np.log10(flux_data_rise))
        bp0 = popt[0]

        # Estimate fall parameter
        sol_data_fall = sol_data[sol_data >= sol_peak0]
        flux_data_fall = flux_data[sol_data >= sol_peak0]
        popt, _ = scipy.optimize.curve_fit(lineFunc, sol_data_fall, np.log10(flux_data_fall))
        bm0 = np.abs(popt[0])

        print("bg guess:", bg_flux0)
        print("sol peak guess:", sol_peak0)
        print("peak flux guess:", peak_flux0)
        print("bp0:", bp0)
        print("bm0:", bm0)

        self.initial_base_parameters = ShowerActivityParameters(sol_peak0, bg_flux0, peak_flux0, bp0, bm0)


    def unpackAllFitParameters(self, base_fit, additional_fits, zhr=False):
        """ Extract all fit parameters from the base fit and additional fits.

        Arguments:
            base_fit: [ShowerActivityParameters] Parameters of the base fit.
            additional_fits: [list] List of ShowerActivityParameters for additional peaks.

        Keyword arguments:
            zhr: [bool] If True, the ZHR parameters is returned instead of the flux.

        Return:
            all_params: [list] List of all fit parameters as a flat array.
        """

        # Extract all initial parameters
        all_params = base_fit.extractParameters(zhr=zhr)
        all_params += [item for peak_params in additional_fits 
                            for item in peak_params.extractParameters(no_bg=True, zhr=zhr)]

        return all_params


    def splitFitParameters(self, fit_results):
        """ Split the fit parameters into the base fit and the additional fits.

        Arguments:
            fit_results: [list] List of all fit parameters as a flat array.

        Return:
            base_fit: [ShowerActivityParameters] Parameters of the base fit.
            additional_fits: [list] List of ShowerActivityParameters for additional peaks.
        """

        # Extract the base fit
        base_fit = ShowerActivityParameters(*fit_results[:5])

        # Extract the additional fits
        additional_fits = []
        for i in range(len(self.additional_fits)):
            sol_peak, peak_flux, bp, bm = fit_results[(5 + 4*i):(5 + 4*(i + 1))]
            additional_fits.append(ShowerActivityParameters(sol_peak, 0.0, peak_flux, bp, bm))

        return base_fit, additional_fits


    def fitAllPeaks(self, sol_data, flux_data, base_fit, additional_fits, weights):
        """ Fit all peaks on the given data together in one optimization loop.
        
        Arguments:
            sol_data: [ndarray] Solar longitudes (deg).
            flux_data: [ndarray] Fluxes.
            base_fit: [ShowerActivityParameters] Parameters of the base fit.
            additional_fits: [list] List of ShowerActivityParameters for additional peaks.
            weights: [ndarray] Weights for the fit.

        Return:
            fit_results: [list] List of all fit parameters as a flat array.
        """

        # Extract all initial parameters
        init_params = self.unpackAllFitParameters(base_fit, additional_fits)

        # Fit all peaks together
        res = scipy.optimize.minimize(showerActivityResidualsAll, init_params, 
            args=(sol_data, flux_data, weights),
            bounds=((self.sol_min, self.sol_max), (0.0, None), (0.0, None), (0.0, None), (0.0, None)) + \
                ((self.sol_min, self.sol_max), (0.0, None), (0.0, None), (0.0, None))*len(additional_fits))

        # Return the fit parameters
        return res.x


    def monteCarloErrorEstimation(self, sol_data, flux_data, base_fit, additional_fits, weights, 
        population_index, mc_runs=100):
        """ Estimate the errors on the fit parameters by performing a Monte Carlo simulation.

        Arguments:
            sol_data: [ndarray] Solar longitudes (deg).
            flux_data: [ndarray] Fluxes.
            base_fit: [list] List of the base fit parameters.
            additional_fits: [list] List of the additional fit parameters.
            weights: [ndarray] Weights for the fit.
            population_index: [float] Shower population index used to compute the ZHR.

        Keyword arguments:
            mc_runs: [int] Number of Monte Carlo runs. 100 by default.

        Return:
            base_fit_std: [list] List of the standard deviations of the base fit parameters.
            additional_fits_std: [list] List of the standard deviations of the additional fit parameters.
        """

        # Compute the standard deviation of the intial fit
        residuals = flux_data - showerActivityCombined(sol_data, \
            self.unpackAllFitParameters(base_fit, additional_fits))
        stddev_res = np.std(residuals)

        print("Stddev of the added flux noise: {:.5f}".format(stddev_res))

        # Perform the Monte Carlo runs
        mc_results = []
        for i in range(mc_runs):

            print("Monte Carlo run", i + 1, "of", mc_runs)

            # Compute noise to the added to the flux data
            flux_noise = np.random.normal(0.0, stddev_res, len(flux_data))

            # Add noise to the flux data
            flux_data_mc = flux_data + flux_noise

            # Fit the peaks with the added noise
            mc_results.append(self.fitAllPeaks(sol_data, flux_data_mc, base_fit, additional_fits, weights))

        # Calculate the standard deviations of individual fit parameters
        mc_results = np.array(mc_results)
        fit_results_std = np.std(mc_results, axis=0)
        base_fit_std, additional_fits_std = self.splitFitParameters(fit_results_std)

        # Compute the standard deviations of the ZHR
        base_fit_std.computeZHR(population_index)
        for peak in additional_fits_std:
            peak.computeZHR(population_index)

        return base_fit_std, additional_fits_std


    def fit(self, sol_data, flux_data, population_index, weights=None):
        """ Fit the activity model to the given data.
        
        Arguments:
            sol_data: [ndarray] Solar longitudes (deg).
            flux_data: [ndarray] Flux at +6.5M in meteoroids / 1000 km^2 h.
            population_index: [float] Shower population index, used to compute the ZHR.

        Keyword arguments:
            weights: [ndarray] Weights for the data points. If None, all points are weighted equally.
        """

        # Save the fit data
        self.sol_data = sol_data
        self.flux_data = flux_data

        # Compute the sol limits
        self.sol_min = np.min(sol_data)
        self.sol_max = np.max(sol_data)

        # Compute weights if not specified
        if weights is None:

            # All weights are equal, their sum should be 1
            weights = np.ones(len(sol_data))/len(sol_data)


        # Estimate parameters automatically if not manually specified
        if self.initial_param_estimation == 'auto':
            self.estimateInitialParameters(sol_data, flux_data)

        # Individually fit all peaks first before fitting them together
        if self.refine_fits_individually:

            # Fit the base peak
            res = scipy.optimize.minimize(showerActivityResiduals, 
                self.initial_base_parameters.fluxParameters(), args=(sol_data, flux_data, True, weights),
                bounds=((self.sol_min, self.sol_max), (0.0, None), (0.0, None), (0.0, None), (0.0, None)))

            # Extract fit parameters
            self.base_fit = ShowerActivityParameters(*res.x)

            # Compute the flux without the base peak
            flux_data_add = flux_data.copy() - showerActivityVect(sol_data, *self.base_fit.fluxParameters())
            
            # Fit additional peaks
            for peak in self.additional_peaks:

                # Extract the peak parameters
                sol_peak, peak_flux, bp, bm = peak

                # Fit the additional peak (no background)
                res = scipy.optimize.minimize(showerActivityResiduals, [sol_peak, peak_flux, bp, bm],
                    args=(sol_data, flux_data_add, False, weights),
                    bounds=((self.sol_min, self.sol_max), (0.0, None), (0.0, None), (0.0, None)))

                sol_peak, peak_flux, bp, bm = list(res.x)

                # Store parameters of additional peaks in an object
                peak_params = ShowerActivityParameters(sol_peak, 0.0, peak_flux, bp, bm)

                # Store the fitted peak
                self.additional_fits.append(peak_params)

                # Subtract the fitted peak from the flux
                flux_data_add -= showerActivityVect(sol_data, *peak_params.fluxParameters())

        else:
            self.base_fit = self.initial_base_parameters
            self.additional_fits = self.additional_peaks


        # Fit all peaks together
        fit_results = self.fitAllPeaks(sol_data, flux_data, self.base_fit, self.additional_fits, weights)

        # Extract the fit parameters into base and additional peaks
        self.base_fit, self.additional_fits = self.splitFitParameters(fit_results)

        # Compute the ZHR of the fitted peaks
        self.base_fit.computeZHR(population_index)
        for peak in self.additional_fits:
            peak.computeZHR(population_index)

        # Set the standard deviation of all parameters to None, until they get computed
        self.base_fit_std = ShowerActivityParameters(None, None, None, None, None)
        self.additional_fits_std = [ShowerActivityParameters(None, None, None, None, None) \
                                    for i in range(len(self.additional_fits))]
        
        # Compute the standard deviations of the fit parameters using Monte Carlo
        if self.mc_error_estimation:
            self.base_fit_std, self.additional_fits_std = self.monteCarloErrorEstimation(sol_data, flux_data,\
                self.base_fit, self.additional_fits, weights, population_index, mc_runs=self.mc_runs)



    def evaluateFlux(self, sol_data):
        """ Evaluate the model at the given solar longitudes.

        Arguments:
            sol_data: [ndarray] Solar longitudes (deg).

        Returns:
            [ndarray] The model evaluated at the given solar longitudes.
        """

        return showerActivityCombined(sol_data, \
            self.unpackAllFitParameters(self.base_fit, self.additional_fits))


    def evaluateZHR(self, sol_data):
        """ Evaluate the ZHR at the given solar longitudes.

        Arguments:
            sol_data: [ndarray] Solar longitudes (deg).

        Returns:
            [ndarray] The ZHR evaluated at the given solar longitudes.
        """

        #return calculateZHR(self.evaluateFlux(sol_data), population_index)
        return showerActivityCombined(sol_data, \
            self.unpackAllFitParameters(self.base_fit, self.additional_fits, zhr=True))


    def plotFluxAndModel(self, sol_data, flux_data, plot_dir, shower_code, year_str, flux_label, 
        plot_individual_models=False, flux_log_scale=False, plot_data=True, show_plots=True):
        """ Plot the flux and the model.

        Arguments:
            sol_data: [ndarray] Solar longitudes (deg).
            flux_data: [ndarray] Fluxes.
            plot_dir: [str] Directory to save the plot.
            shower_code: [str] Shower code.
            year_str: [str] Year string.
            flux_label: [str] Label for the flux axis.

        Keyword arguments:
            plot_individual_models: [bool] Plot the individual models.
            flux_log_scale: [bool] Plot the flux on a log scale.
            plot_data: [bool] Plot the data together with the model.
            show_plots: [bool] Show the plots on the screen.
        """

        ### Make the plots

        plot_suffix = ""

        if plot_data:

            # Plot the data
            plt.scatter(sol_data, flux_data, label='Data', color='0.2', alpha=0.8, s=5)

            plot_suffix += "_data"


        # Sample the solar longitude
        sol_arr = np.linspace(np.min(sol_data), np.max(sol_data), 1000)

        # Get the full model fit
        flux_arr = self.evaluateFlux(sol_arr)

        # Plot the full model fit
        plt.plot(sol_arr, flux_arr, 
            label='Model fit',
            # label="Sol peak = {:.1f} $\\pm$ {:.1f} deg\n"
            # "Peak flux = {:.1f} $\\pm$ {:.1f}\n"
            # "Bp = {:.3f} $\\pm$ {:.3f}\n"
            # "Bm = {:.3f} $\\pm$ {:.3f}".format(sol_peak, sol_peak_std, peak_flux, peak_flux_std, 
            #     bp, bp_std, bm, bm_std),
            linestyle='dashed', color='k')


        # Plot the base peak and the additional peaks individually
        if plot_individual_models and (len(self.additional_fits) > 0):

            # Plot the base peak
            base_arr = showerActivityVect(sol_arr, *self.base_fit.fluxParameters())
            plt.plot(sol_arr, base_arr, linestyle='dotted', color='0.5', alpha=0.5)

            # Plot the additional peaks
            for peak in self.additional_fits:
                peak_arr = showerActivityVect(sol_arr, *peak.fluxParameters())
                plt.plot(sol_arr, peak_arr, linestyle='dotted', color='k', alpha=0.5)

        ###

        plt.xlabel("Solar longitude (deg)")
        plt.ylabel(flux_label)

        plt.title(shower_code + " " + year_str.replace("_", " "))

        plt.legend()

        plt.tight_layout()

        # Set flux axis to log scale
        if flux_log_scale:

            # Set two y-axis labels per order of magnitude
            plt.gca().yaxis.set_minor_locator(ticker.LogLocator(subs='all'))

            plt.yscale('log')
            plot_suffix += "_log"

            # Get y limits
            ymin, ymax = plt.gca().get_ylim()

            # Make sure the bottom is always in between 0.5 and 0.9
            if np.min([np.min(flux_data), ymin]) < 0.5:
                plt.gca().set_ylim(ymin=0.5)
            elif np.min([np.min(flux_data), ymax]) > 1.0:
                plt.gca().set_ylim(ymin=0.9)


            # Set the appropriate upper limit
            plt.gca().set_ylim(ymax=1.2*np.max([np.max(flux_data), np.max(flux_arr)]))

        else:

            # Set the Y minimum to 0
            plt.gca().set_ylim(ymin=0)
            


        # Make the plot directory if it doesn't exist
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Save the plot
        plt.savefig(os.path.join(plot_dir, shower_code + "_flux_" + year_str + plot_suffix + ".png"), dpi=150)

        if show_plots:
            plt.show()
        else:
            plt.clf()
            plt.close()


    def plotZHRAndModel(self, sol_data, zhr_data, plot_dir, shower_code, year_str, zhr_label,
        plot_individual_models=False, plot_data=True, show_plots=True):
        """ Plot the ZHR and the model.

        Arguments:
            sol_data: [ndarray] Solar longitudes (deg).
            zhr_data: [ndarray] ZHR measurements.
            plot_dir: [str] Directory to save the plot.
            shower_code: [str] Shower code.
            zhr_label: [str] Label for the ZHR axis.
            year_str: [str] Year string.

        Keyword arguments:
            plot_individual_models: [bool] Plot the individual models.
            plot_data: [bool] Plot the data together with the model.
            show_plots: [bool] Show the plots on the screen.
        """

        ### Make the plots

        plot_suffix = ""

        if plot_data:
                
            # Plot the data
            plt.scatter(sol_data, zhr_data, label='Data', color='0.2', alpha=0.8, s=5)

            plot_suffix += "_data"

        
        # Sample the solar longitude
        sol_arr = np.linspace(np.min(sol_data), np.max(sol_data), 1000)

        # Get the full model fit
        zhr_arr = self.evaluateZHR(sol_arr)

        # Plot the full model fit
        plt.plot(sol_arr, zhr_arr, label='Model fit', linestyle='dashed', color='k')


        # Plot the base peak and the additional peaks individually
        if plot_individual_models and (len(self.additional_fits) > 0):
                
                # Plot the base peak
                zhr_arr = showerActivityVect(sol_arr, *self.base_fit.zhrParameters())
                plt.plot(sol_arr, zhr_arr, linestyle='dotted', color='0.5', alpha=0.5)
    
                # Plot the additional peaks
                for peak in self.additional_fits:
                    zhr_arr = showerActivityVect(sol_arr, *peak.zhrParameters())
                    plt.plot(sol_arr, zhr_arr, linestyle='dotted', color='k', alpha=0.5)

        ###

        plt.xlabel("Solar longitude (deg)")
        plt.ylabel(zhr_label)

        plt.title(shower_code)

        plt.legend()

        plt.tight_layout()

        # Make the plot directory if it doesn't exist
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Save the plot
        plt.savefig(os.path.join(plot_dir, shower_code + "_zhr_" + year_str + plot_suffix + ".png"), dpi=150)

        if show_plots:
            plt.show()
        else:
            plt.clf()
            plt.close()


    def fitSummary(self, shower_code, year_str):
        """ Return a string with the summary of the fit, including all separate peaks."""


        out_str = ""

        # Columns:
        # Shower code, year, peak type (base or additional), sol peak, bg flux, peak flux, bg ZHR, peak ZHR, 
        # bp, bm
        # <the second row is the standard deviation>

        # Consolidate model parameters
        model_params = [self.base_fit] + self.additional_fits
        model_std = [self.base_fit_std] + self.additional_fits_std
        background_status = [True] + [False]*len(self.additional_fits)

        for i, (bg_status, fit_params, fit_std) in enumerate(
            zip(background_status, model_params, model_std)
            ):

            # Extract fit parameters for each model
            sol_peak_temp, bg_flux_temp, peak_flux_temp, bp_temp, bm_temp = fit_params.fluxParameters()
            sol_peak_std_temp, bg_flux_std_temp, peak_flux_std_temp, bp_std_temp, bm_std_temp = fit_std.fluxParameters()
            _, bg_zhr_temp, peak_zhr_temp, _, _ = fit_params.zhrParameters()
            _, bg_zhr_std_temp, peak_zhr_std_temp, _, _ = fit_std.zhrParameters()

            # Set background parameters to -1 for additional peaks
            if not bg_status:
                bg_flux_temp = -1
                bg_flux_std_temp = -1
                bg_zhr_temp = -1
                bg_zhr_std_temp = -1

            output_values = [shower_code, year_str]
            std_values = [" "*len(shower_code), " "*len(year_str)]

            # Peak type
            if i == 0:
                peak_type = "Base"
            else:
                peak_type = "{:4d}".format(i)

            # Append to output values
            output_values.append(peak_type)
            std_values.append(" "*len(peak_type))

            # Formatting function which will set values to -1 if they are none
            fmt = lambda x: -1 if x is None else x

            # Append fit parameters
            output_values.append("{:7.3f}".format(sol_peak_temp))
            std_values.append(   "{:7.3f}".format(fmt(sol_peak_std_temp)))
            output_values.append("{:7.2f}".format(bg_flux_temp))
            std_values.append(   "{:7.2f}".format(fmt(bg_flux_std_temp)))
            output_values.append("{:7.2f}".format(peak_flux_temp))
            std_values.append(   "{:7.2f}".format(fmt(peak_flux_std_temp)))
            output_values.append("{:7.2f}".format(bg_zhr_temp))
            std_values.append(   "{:7.2f}".format(fmt(bg_zhr_std_temp)))
            output_values.append("{:7.2f}".format(peak_zhr_temp))
            std_values.append(   "{:7.2f}".format(fmt(peak_zhr_std_temp)))
            output_values.append("{:7.3f}".format(bp_temp))
            std_values.append(   "{:7.3f}".format(fmt(bp_std_temp)))
            output_values.append("{:7.3f}".format(bm_temp))
            std_values.append(   "{:7.3f}".format(fmt(bm_std_temp)))

            # Format the output string
            out_str += " & ".join(output_values) + "\n"
            out_str += " & ".join(std_values) + "\n"

        return out_str
        


        

    def saveModelFluxAndFitParameters(self, dir_path, shower_code, year_str, sol_delta=0.01):
        """ Save the model parametrs to a text file for every peak, together with the values of the combined
            model and each individual peak.

        Arguments:
            dir_path: [str] Directory to save the file.
            shower_code: [str] Shower code.
            year_str: [str] Year string.
            sol_delta: [float] Solar longitude step (deg).
        """    

        def _errorStr(val_format, val):
            """ Format the error if available. 
            
            Arguments:
                val_format: [str] Format string for the standard deviation.
                val: [float] Value.

            Returns:
                [str] Formatted string.
            """
            if val is None:
                return ""
            else:
                return " +/- " + val_format.format(val)

        
        # Make the directory if it doesn't exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        txt_file_path = os.path.join(dir_path, "{:s}_{:s}_model_params.txt".format(shower_code, year_str))

        # Open the file
        with open(txt_file_path, "w") as f:

            f.write("Shower: {:s}-{:s}\n".format(shower_code, year_str))
            f.write("\n")
            f.write("----------------------------------------\n")
            f.write("\n")

            # Write the number of Monte Carlo runs, if error estimation was used
            if self.mc_error_estimation:
                f.write("Monte Carlo runs: {:d}\n".format(self.mc_runs))
                f.write("\n")

            # Consolidate model parameters
            model_params = [self.base_fit] + self.additional_fits
            model_std = [self.base_fit_std] + self.additional_fits_std
            background_status = [True] + [False]*len(self.additional_fits)

            for i, (bg_status, fit_params, fit_std) in enumerate(
                zip(background_status, model_params, model_std)
                ):

                # Extract fit parameters for each model
                sol_peak_temp, bg_flux_temp, peak_flux_temp, bp_temp, bm_temp = fit_params.fluxParameters()
                sol_peak_std_temp, bg_flux_std_temp, peak_flux_std_temp, bp_std_temp, bm_std_temp = fit_std.fluxParameters()
                _, bg_zhr_temp, peak_zhr_temp, _, _ = fit_params.zhrParameters()
                _, bg_zhr_std_temp, peak_zhr_std_temp, _, _ = fit_std.zhrParameters()

                f.write("Peak {:d}\n".format(i + 1))

                # Write the model parameters (include errors if available)
                f.write("Sol peak   = {:8.4f}{:s}\n".format(sol_peak_temp, 
                    _errorStr("{:6.4f}", sol_peak_std_temp)))
                f.write("Peak flux  = {:8.4f}{:s}\n".format(peak_flux_temp, 
                    _errorStr("{:6.4f}", peak_flux_std_temp)))
                if bg_status:
                    f.write("Bg flux    = {:8.4f}{:s}\n".format(bg_flux_temp, 
                        _errorStr("{:6.4f}", bg_flux_std_temp)))
                f.write("Peak ZHR   = {:8.4f}{:s}\n".format(peak_zhr_temp,
                    _errorStr("{:6.4f}", peak_zhr_std_temp)))
                if bg_status:
                    f.write("Bg ZHR     = {:8.4f}{:s}\n".format(bg_zhr_temp,
                        _errorStr("{:6.4f}", bg_zhr_std_temp)))
                f.write("Bp         = {:8.4f}{:s}\n".format(bp_temp, 
                    _errorStr("{:6.4f}", bp_std_temp)))
                f.write("Bm         = {:8.4f}{:s}\n".format(bm_temp, 
                    _errorStr("{:6.4f}", bm_std_temp)))
                f.write("\n")

            f.write("----------------------------------------\n")

            # Generate a solar longitude array
            sol_arr = np.arange(self.sol_min, self.sol_max, sol_delta)

            # Evaluate the combined model
            flux_arr = self.evaluateFlux(sol_arr)

            # If there are additional peaks, evaluate each peak individually, including the base peak
            flux_individual_models = []
            if len(self.additional_fits) > 0:
                flux_individual_models.append(showerActivityVect(sol_arr, *self.base_fit.fluxParameters()))
                for peak in self.additional_fits:
                    flux_individual_models.append(showerActivityVect(sol_arr, *peak.fluxParameters()))

            # Write the header
            f.write("Solar longitude (deg), Combined model flux")
            if len(self.additional_fits) > 0:
                f.write(", Base peak")
                for i in range(len(self.additional_fits)):
                    f.write(", Peak {:d}".format(i + 1))

            f.write("\n")

            # Write the combined and individual model values to the file
            for i, sol in enumerate(sol_arr):

                # Write the solar longitude and the combined model
                f.write("{:7.3f}, {:7.3f}".format(sol, flux_arr[i]))

                # Write the individual model values
                if len(self.additional_fits) > 0:
                    for flux in flux_individual_models:
                        f.write(", {:7.3f}".format(flux[i]))
                f.write("\n")




if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Fit shower activity models to given CSV files produced by the flux code. Note that the shower fits have to be defined in the code itself.")

    arg_parser.add_argument('csv_path', metavar='CSV_PATH', type=str, \
        help="Path to the directory containing the batch flux CSV files.")

    arg_parser.add_argument("--showplots", action="store_true", \
        help="Show the plots on the screen as well as saving them to file.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################



    # Path to CSV files
    csv_path = cml_args.csv_path

    # Plot directory (set the path so it's next to the CSV directory)
    plot_dir = "plots"
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), plot_dir)

    # Make the plot directory if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)


    print("Plots will be saved to: {:s}".format(plot_dir))


    # Show plots on the screen
    show_plots = cml_args.showplots


    # Column names in the CSV files that will be extracted
    sol_column = ' Mean Sol (deg)'
    flux_column = ' Flux@+6.5M (met / 1000 km^2 h)'
    flux_ci_low_column = ' Flux CI low'
    flux_ci_high_column = ' Flux CI high'
    zhr_column = ' ZHR'



    ### Set which showers to process ###


    showers = {}


    ###################


    ### CAPRICORNIDS ###

    # Define the shower model for the Capricornids
    cap_all_years = ShowerActivityModel(initial_param_estimation='manual', sol_peak0=128, bg_flux0=0.5,
        peak_flux0=1.5, bp0=0.05, bm0=0.05, refine_fits_individually=False, mc_error_estimation=True)
    
    showers["CAP"] = [
        ["all_years", cap_all_years]
    ]

    ### ###

    ### ETA AQUARIIDS ###

    # Define the shower model for the Eta Aquariids
    eta_all_years = ShowerActivityModel(initial_param_estimation='manual', sol_peak0=43, bg_flux0=2,
        peak_flux0=15, bp0=0.3, bm0=0.3, refine_fits_individually=False, mc_error_estimation=True)
    eta_all_years.addPeak(47, 15, 0.3, 0.3)

    showers['ETA'] = [
        ["year_2022", eta_all_years]
    ]

    ### ###

    ### GEMINIDS ###
    # Define the shower model for the Geminids
    gem_all_years = ShowerActivityModel(initial_param_estimation='manual', 
        sol_peak0=262.1, bg_flux0=0.2, peak_flux0=19.5, bp0=0.3, bm0=1.0, refine_fits_individually=False,
        mc_error_estimation=True)
    gem_all_years.addPeak(261.7, 5, 0.7, 0.7)

    showers['GEM'] = [
        ["all_years", gem_all_years],
    ]

    ### ###


    ### HYDRIDS ###

    # Define the shower model for the Hydrids
    hyd_all_years = ShowerActivityModel(initial_param_estimation='manual', 
        sol_peak0=255.0, bg_flux0=1.8, peak_flux0=6.0, bp0=0.1, bm0=0.1, refine_fits_individually=False,
        mc_error_estimation=True)

    showers['HYD'] = [
        ["all_years", hyd_all_years],
    ]

    ###


    ### KAPPA CYGNIDS ###

    # Define the shower model for the Capricornids
    kcg_2021 = ShowerActivityModel(initial_param_estimation='auto', refine_fits_individually=False, 
        mc_error_estimation=True)
    
    showers["KCG"] = [
        ["year_2021", kcg_2021]
    ]

    ### ###


    ### LEONIDS ###
    # Define the shower model for the Leonids
    leo_all_years = ShowerActivityModel(initial_param_estimation='auto', mc_error_estimation=True)

    showers['LEO'] = [
        ["all_years", leo_all_years],
    ]

    ### ###


    ### LEONIS MINORIDS ###
    # Define the shower model for the Leonis Minorids
    lmi_all_years = ShowerActivityModel(initial_param_estimation='manual',
        sol_peak0=210.0, bg_flux0=0.0, peak_flux0=1.6, bp0=0.1, bm0=0.1,
        mc_error_estimation=True)

    showers['LMI'] = [
        ["all_years", lmi_all_years],
    ]

    ### ###


    ### LYRIDS ###
    # Define the shower model for the Lyrids
    lyr_all_years = ShowerActivityModel(initial_param_estimation='auto', mc_error_estimation=True)

    showers['LYR'] = [
        ["all_years", lyr_all_years],
    ]

    ### ###

    ### MONOCEROTIDS ###
    # Define the shower model for the Monocerotids
    mon_2022 = ShowerActivityModel(initial_param_estimation='auto', mc_error_estimation=True)
    
    showers['MON'] = [
        ["year_2022", mon_2022],
    ]


    ### ORIONIDS ###

    # Define the shower model for the Orionids
    ori_2022 = ShowerActivityModel(initial_param_estimation='auto', mc_error_estimation=True)
    showers['ORI'] = [
        ["year_2022", ori_2022],
    ]

    ###


    ### PERSEIDS ###

    # Define the shower model for the 2020 Perseids
    per_2020 = ShowerActivityModel(
        initial_param_estimation='manual', sol_peak0=138.0, bg_flux0=0.5, peak_flux0=7.0, bp0=0.03, bm0=0.07,
        refine_fits_individually=False, mc_error_estimation=True)
    per_2020.addPeak(140, 23, 0.3, 0.4)
    # per_2020 = ShowerActivityModel(initial_param_estimation='auto')

    # Define the shower model for the 2021 Perseids
    per_2021 = ShowerActivityModel(
        initial_param_estimation='manual', sol_peak0=138.0, bg_flux0=0.5, peak_flux0=7.0, bp0=0.03, bm0=0.07,
        refine_fits_individually=False, mc_error_estimation=True)
    per_2021.addPeak(140, 23, 0.3, 0.4)
    per_2021.addPeak(141.7, 35.0, 3.0, 3.0) # Outburst

    
    # # Define the shower model for the 2022 Perseids
    # per_2022 = ShowerActivityModel(
    #     initial_param_estimation='manual', sol_peak0=139.0, bg_flux0=0.5, peak_flux0=5.0, bp0=0.04, bm0=0.07,
    #     refine_fits_individually=False)
    # per_2022.addPeak(140, 23, 0.3, 0.4)
    # #per_2022.addPeak(141.7, 5.0, 3.0, 3.0) # Small outburst?

    showers['PER'] = [
        # Define additional peaks
        #                sol,    peak, bp0, bm0        
        #["all_years", [ [139,    5.0, 0.01, 0.02], [141.7, 30.0, 1.0, 1.0] ] ],
        ["year_2020", per_2020 ],
        ["year_2021", per_2021 ],
        #["year_2022", per_2022 ],
    ]

    ###


    ### QUADRANTIDS ###
    # Define the shower model for the Quadrantids
    
    qua_all_years = ShowerActivityModel(initial_param_estimation='auto', mc_error_estimation=True)
    
    qua_2020 = ShowerActivityModel(initial_param_estimation='auto', mc_error_estimation=True)
    
    qua_2021 = ShowerActivityModel(initial_param_estimation='auto', mc_error_estimation=True)
    
    qua_2022 = ShowerActivityModel(initial_param_estimation='manual', 
        sol_peak0=282.8, bg_flux0=0.0, peak_flux0=9.0, bp0=1.0, bm0=1.0, refine_fits_individually=False,
        mc_error_estimation=True)
    qua_2022.addPeak(283.3, 10, 0.8, 0.8)
    
    showers['QUA'] = [
        ["all_years", qua_all_years],
        ["year_2020", qua_2020],
        ["year_2021", qua_2021],
        ["year_2022", qua_2022],
    ]

    ### ###


    ### SOUTHERN DELTA AQUARIIDS ###
    # Define the shower model for the Southern Delta Aquariids
    sda_all_years = ShowerActivityModel(initial_param_estimation='auto', mc_error_estimation=True)
    showers['SDA'] = [
        ["all_years", sda_all_years],
    ]

    ### ###


    ### SEPTEMBER PERSEIDS ###
    # Define the shower model for the September Perseids
    spe_all_years = ShowerActivityModel(initial_param_estimation='manual', mc_error_estimation=True,
        sol_peak0=166.55, bg_flux0=0.7, peak_flux0=1.0, bp0=0.6, bm0=0.5, refine_fits_individually=False,
        )

    showers['SPE'] = [
        ["all_years", spe_all_years],
    ]

    # ### ###

    ### URSIDS ###
    # Define the shower model for the Ursids
    urs_all_years = ShowerActivityModel(initial_param_estimation='manual', 
        sol_peak0=270.3, bg_flux0=0, peak_flux0=1.0, bp0=1.0, bm0=1.0, refine_fits_individually=False,
        mc_error_estimation=True)
    showers['URS'] = [
        ["all_years", urs_all_years],
    ]

    ### ###


    ### AURIGIDS ###
    
    # Define the shower model for the Aurigids
    aur_2021 = ShowerActivityModel(initial_param_estimation='manual',
        sol_peak0=158.0, bg_flux0=1.0, peak_flux0=3.0, bp0=0.1, bm0=0.1, refine_fits_individually=False,
        mc_error_estimation=True)

    showers['AUR'] = [
        ["year_2021", aur_2021],
    ]

    ### ###


    ### 2022 TAU HERCULIDS ###

    # Define the shower model for the 2022 Tau Herculid outburst
    tah_2022 = ShowerActivityModel(initial_param_estimation='auto', mc_error_estimation=True)
    showers['oTAH2022'] = [
        ["year_2022", tah_2022],
    ]

    ### ###


    # List for storing the fit results
    results = []


    # Load files in the CSV directory
    csv_files = [os.path.join(os.path.abspath(csv_path), file_name) 
        for file_name in sorted(os.listdir(csv_path)) if file_name.lower().endswith('.csv')]


    # Go through all shower codes
    for shower_code in showers:

        shower_data = showers[shower_code]

        # Go through all years
        for year_entry in shower_data:

            year_str, shower_model = year_entry

            print("Processing {:s} {:s}...".format(shower_code, year_str))

            # Load the appropraite CSV file
            csv_candidates = [file_path for file_path in csv_files 
                if (shower_code in file_path) and (year_str in file_path)]

            if len(csv_candidates) == 0:
                print("CSV for {:s} {:s} not found!".format(shower_code, year_str))
                continue

            csv_file = csv_candidates[0]

            # Read the CSV file
            with open(csv_file) as f:
                csv_contents = f.readlines()

            # Read metadata
            for line in csv_contents:

                # Read the mass limit
                if "m_lim @ +6.5M" in line:
                    line = line.split('=')
                    m_lim_6_5m = float(line[-1].strip().replace("kg", "").strip())
                    
                # Read the population index
                if " r     " in line:
                    line = line.split('=')
                    population_index = float(line[-1].strip())

            # Load the CSV info pandas
            data = pd.read_csv(csv_file, delimiter=',', skiprows=13, escapechar='#')

            # Prune the last line (only the sol bin edge)
            data = data[:-1]

            # Pune the first and the last point (some points are outliers)
            data = data[1:-1]

            # print(data)
            # print(data.columns)


            # Extract sol, flux, and ZHR data
            sol_data = data[sol_column].to_numpy()
            flux_data = data[flux_column].to_numpy()
            zhr_data = data[zhr_column].to_numpy()


            ### Compute the fit weights ###

            # Extract the flux confidence interval
            flux_ci_low = data[flux_ci_low_column].to_numpy()
            flux_ci_high = data[flux_ci_high_column].to_numpy()

            # Compute the weights (smaller range = higher weight), handle zero values
            flux_ci_diff = np.abs(flux_ci_high - flux_ci_low)
            flux_ci_diff[flux_ci_diff == 0] = 0.5*np.min(flux_ci_diff[flux_ci_diff != 0])
            flux_weights = 1/flux_ci_diff

            # Normalize the weights to sum to 1
            flux_weights = flux_weights/np.sum(flux_weights)

            ### ###



            # Create the flux label
            flux_label = r"Flux at ${:s}$ kg (meteoroids / 1000 $\cdot$ km$^2$ $\cdot$ h)".format(formatScientific(m_lim_6_5m, 0))

            # Fit the model
            shower_model.fit(sol_data, flux_data, population_index, weights=flux_weights)

            # Save the fitted model to a text file
            shower_model.saveModelFluxAndFitParameters(plot_dir, shower_code, year_str)

            # Show the model fit plot
            shower_model.plotFluxAndModel(sol_data, flux_data, plot_dir, shower_code, year_str, flux_label, 
                plot_individual_models=True, flux_log_scale=False, plot_data=True, show_plots=show_plots)

            shower_model.plotFluxAndModel(sol_data, flux_data, plot_dir, shower_code, year_str, flux_label, 
                plot_individual_models=True, flux_log_scale=False, plot_data=False, show_plots=False)
            shower_model.plotFluxAndModel(sol_data, flux_data, plot_dir, shower_code, year_str, flux_label, 
                plot_individual_models=True, flux_log_scale=True, plot_data=True, show_plots=False)
            shower_model.plotFluxAndModel(sol_data, flux_data, plot_dir, shower_code, year_str, flux_label, 
                plot_individual_models=True, flux_log_scale=True, plot_data=False, show_plots=False)

            # Plot the ZHR
            shower_model.plotZHRAndModel(sol_data, zhr_data, plot_dir, shower_code, year_str, "ZHR",
                plot_individual_models=True, plot_data=True, show_plots=False)


            # Save the fit results
            results.append([shower_code, year_str.strip("year_").strip("_years"), shower_model])


    # Print the fit results
    print("Fit results:")
    print()
    print("Shower & Year & Component & $\lambda_{\odot M}$ & $F_{\mathrm{spo}}$ & $F_M$ & $\mathrm{ZHR}_{\mathrm{spo}}$ & $\mathrm{ZHR}_{M}$ & $B_p$ & $B_m$")
    print()
    combined_summary = ""
    for result in results:
        shower_code, year_str, shower_model = result
        combined_summary += shower_model.fitSummary(shower_code, year_str)

    print(combined_summary)



