import os
import copy
from datetime import datetime
import datetime as dt
import threading
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.optimize
import scipy.stats
import scipy.special
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from RMS.Routines.Image import signalToNoise, loadDark, loadFlat
from RMS.Routines import Image
from RMS.Formats.FrameInterface import detectInputTypeFolder, detectInputTypeFile
from RMS import ConfigReader
from RMS.Astrometry.Conversions import trueRaDec2ApparentAltAz
from RMS.Astrometry.ApplyAstrometry import computeFOVSize, xyToRaDecPP, rotationWrtHorizon

try:
    from pyswarms.single.global_best import GlobalBestPSO
except Exception as e:
    print(f'ASTRA cannot be run, pyswarms not installed: {e}')

# Only import platepar in terminal execution (avoids circular import error)
if __name__ == '__main__':
    from Utils.SkyFit2 import Platepar



class ASTRA:

    def __init__(self, img_obj, pick_dict, astra_config, data_path, config, dark, flat, 
                 progress_callback=None, ECSV_save_path=None):
        """
        ASTRA: Astrometric Streak Tracking and Refinement Algorithm.

        Initializes an ASTRA instance with imaging data, picks, configuration, and
        processing options. This constructor also parses configuration blocks
        (ASTRA, PSO, second-pass optimizer, Kalman), sets operational thresholds, and
        prepares instance attributes used throughout the pipeline.

        Args:
            img_obj: Imaging source object providing frame access and timing
                (must support .loadChunk() returning an object with .avepixel
                and a per-frame timestamp dictionary like `frame_dt_dict`).
            frames (np.ndarray): Array of corrected or raw frames,
                shape (N, H, W), dtype numeric.
            picks (array-like): Meteor detection picks, shape (M, 2), as (x, y) pixel coords.
            pick_frame_indices (array-like): Length-M list/array of frame indices
                corresponding to `picks`.
            times (list[datetime.datetime]): Timestamps for frames containing picks.
            astra_config (dict): Nested config for sections 'astra', 'pso', 'kalman'
                and others as used in code.
            saturation_threshold (float | int): Saturation cap for intensity modeling.
            data_path (str | os.PathLike): Root path used for saving outputs (e.g., frames/animations).
            camera_config: Object with camera settings used for gamma/bit-depth corrections
                (must expose `.gamma` and `.bit_depth`).
            dark (np.ndarray | None): Optional dark frame, shape (H, W).
            flat (Any | None): Optional flat-field structure accepted by `Image.applyFlat`.
            progress_callback (callable | None): Optional function taking a single float/int
                to update progress (0–100).

        Sets:
            - Multiple configuration dicts (PSO, cropping, second-pass, Kalman).
            - Processing thresholds (SNR, background, oob penalties).
            - Raw I/O and working arrays (frames, times, picks, masks, logs).
            - Verbosity and save flags.

        Raises:
            KeyError: If required config keys are missing and not defaulted.
            ValueError: If inputs have incompatible shapes or types.
        """
        # -- Constants & Settings --

        # Unpack calculated args
        self.first_pick_global_index = min(pick_dict.keys())
        self.pick_frame_indices = [fr_no - self.first_pick_global_index for fr_no in list(pick_dict.keys())]
        self.picks = np.array([[value["x_centroid"], value["y_centroid"]] for value in pick_dict.values()])
        self.saturation_threshold = int(round(0.98*(2**config.bit_depth - 1)))

        # Unpack passed args
        self.config = config
        self.astra_config = astra_config
        self.data_path = data_path
        self.dark = dark
        self.flat_struct = flat
        self.img_obj = img_obj
        self.ecsv_save_path = ECSV_save_path

        # Initialize callback and set progress to 0
        self.progress_callback = progress_callback
        if self.progress_callback is not None:
            self.progress_callback(0)

        # Unpack astra_config parameters

        # Image processing parameters
        self.BACKGROUND_STD_THRESHOLD = float(self.astra_config.get('astra', {}).get('star_thresh', 3))
        self.snr_threshold = float(self.astra_config.get('astra', {}).get('min SNR', 5))

        # PSO parameters
        w = float(self.astra_config.get('pso', {}).get('w (0-1)', 0.9))
        c1 = float(self.astra_config.get('pso', {}).get('c_1 (0-1)', 0.4))
        c2 = float(self.astra_config.get('pso', {}).get('c_2 (0-1)', 0.3))
        max_iter = int(self.astra_config.get('pso', {}).get('max itter', 100))
        n_particles = int(self.astra_config.get('pso', {}).get('n_particles', 100))
        velocity_coeff = float(self.astra_config.get('pso', {}).get('V_c (0-1)', 0.3))
        ftol = float(self.astra_config.get('pso', {}).get('ftol', 1e-4))
        ftol_iter = int(self.astra_config.get('pso', {}).get('ftol_itter', 25))
        bh_strategy = self.astra_config.get('pso', {}).get('bh_strategy', 'nearest')
        vh_strategy = self.astra_config.get('pso', {}).get('vh_strategy', 'invert')
        explorative_coeff = float(self.astra_config.get('pso', {}).get('expl_c', 3.0))

        # Settings and padding coefficients for cropping
        initial_padding_coeff = float(self.astra_config.get('astra', {}).get('P_crop', 1.5))
        recursive_padding_coeff = float(self.astra_config.get('astra', {}).get('P_crop', 1.5))
        init_sigma_guess = float(self.astra_config.get('astra', {}).get('sigma_init (px)', 2))
        max_sigma_coeff = float(self.astra_config.get('astra', {}).get('sigma_max', 1.2))
        max_length_coeff = float(self.astra_config.get('astra', {}).get('L_max', 1.5))
        if self.astra_config.get('astra', {}).get('pick_offset') == "leading-edge":
            self.pick_offset = 3
        elif self.astra_config.get('astra', {}).get('pick_offset') == "center":
            self.pick_offset = 0
        else:
            self.pick_offset = float(self.astra_config.get('astra', {}).get('pick_offset', 3))


        # Boolean attributes
        self.verbose = str(self.astra_config.get('astra', {}).get('Verbose', 'false')).lower() == 'true'
        self.save_animation = str(self.astra_config.get('astra', {}).get('Save Animation', 'false')
                                        ).lower() == 'true'

        # Repack some astra_config parameters
        self.first_pass_settings = {
            "residuals_method": 'abs',
            "options": {
                "w": w,
                "c1": c1,
                "c2": c2
            },
            "max_iter": max_iter,
            "n_particles": n_particles,
            "Velocity_coeff": velocity_coeff,
            "ftol": ftol,
            "ftol_iter": ftol_iter,
            "bh_strategy": bh_strategy,
            "vh_strategy": vh_strategy,
            "explorative_coeff": explorative_coeff,
            "oob_penalty": 1e6
        }

        # 2) PSO Constants/Settings
        std_parameter_constraint = float(self.astra_config.get('pso', {}).get('P_sigma', 3))
        self.std_parameter_constraint = [
            std_parameter_constraint, # level_sum
            std_parameter_constraint, # height / sigma_y
            std_parameter_constraint, # x0
            std_parameter_constraint, # y0
            std_parameter_constraint  # length / sigma_x
        ]


        self.cropping_settings = {
            'initial_padding_coeff': initial_padding_coeff,
            'recursive_padding_coeff': recursive_padding_coeff,
            'init_sigma_guess': init_sigma_guess,
            'max_sigma_coeff': max_sigma_coeff,
            'max_length_coeff': max_length_coeff
        }

        # 3) Second-Pass Local Optimization Constants/Settings
        self.second_pass_settings = {
            "residuals_method": 'abs_squared',
            "method": 'L-BFGS-B',
            "oob_penalty": 1e6,
        }

        # Instantiate variables to store progress
        self.progressed_frames = {
            'cropping': 0,
            'refining': 0,
            'removing': 0
        }
        self.total_frames = self.pick_frame_indices[-1] - self.pick_frame_indices[0] + 1

    # 1) -- Functional Methods --

    def processImageData(self):
        """
        Correct frames, subtract background, and apply star masking.

        Steps:
            - Apply dark and flat corrections where provided.
            - Apply gamma correction based on camera config.
            - Load and correct a background model (`avepixel_background`).
            - Subtract background from corrected frames (clip negatives at 0).
            - Compute a robust star mask from background mean/std and apply it
            consistently to data and background.

        Args:
            img_obj: Imaging object providing `loadChunk()` that returns an object
                with `.avepixel`.
            frames (np.ndarray): Input frames, shape (N, H, W).

        Returns:
            tuple:
                avepixel_background (np.ma.MaskedArray | np.ndarray): Corrected background, shape (H, W).
                subtracted_frames (np.ma.MaskedArray): BG-subtracted frames with star mask, shape (N, H, W).
                masked_frames (np.ma.MaskedArray): Corrected frames with same mask, shape (N, H, W).

        Raises:
            Exception: On background load/subtraction or mask application failure.
        """

        # 1) -- Background Subtraction --
        # Load background using RMS
        fake_ff_obj = self.img_obj.loadChunk()
        avepixel_background = fake_ff_obj.avepixel.astype(np.float32)
        corrected_avepixel = avepixel_background.copy()

        # # Correct avepixel_background
        if self.dark is not None:
            corrected_avepixel = Image.applyDark(corrected_avepixel, self.dark)

        if self.flat_struct is not None:
            corrected_avepixel = Image.applyFlat(corrected_avepixel, self.flat_struct)

        if self.dark is not None or self.flat_struct is not None:
            corrected_avepixel = Image.gammaCorrectionImage(corrected_avepixel, self.config.gamma, 
                                                        bp=0, wp=(2**self.config.bit_depth - 1),
                                                        out_type=np.float32)

        else:
            corrected_avepixel = Image.gammaCorrectionImage(avepixel_background, self.config.gamma, 
                                                        bp=0, wp=(2**self.config.bit_depth - 1),
                                                        out_type=np.float32)

        # Set backgrounds as a class var
        self.corrected_avepixel = corrected_avepixel
        
        # 2) -- Star Masking --

        # Calculate std of background
        background_std = np.std(self.corrected_avepixel)
        background_mean = np.median(self.corrected_avepixel)

        # Calculate the masking threshold
        threshold = background_mean + self.BACKGROUND_STD_THRESHOLD * background_std

        # Save star mask as a class variable
        self.star_mask = np.ma.MaskedArray(self.corrected_avepixel > threshold)        


    def cropAllMeteorFrames(self):
        """ 
        Recursively crop all meteor frames and fit a moving Gaussian (first pass).

        Algorithm:
            - Pick an initial consecutive seed triplet of frames and estimate direction,
            angle `omega`, and initial length from seed geometry.
            - For each seed, crop a maximal bounding window and run PSO to fit
            moving-Gaussian parameters.
            - Build spline predictors from first-pass params to estimate the next crop’s
            center and expected param magnitudes.
            - Recurse forward and backward across the event using
            `recursiveCroppingAlgorithm`.

        Args:
            pick_frame_indices (Sequence[int]): Frame indices for all picks.
            picks (np.ndarray): Global (x, y) picks, shape (M, 2).

        Returns:
            tuple:
                cropped_frames (list[np.ndarray]): Crops used for fitting.
                first_pass_params (np.ndarray): PSO parameters per frame,
                    columns [level_sum, sigma, x0, y0, length], shape (K, 5).
                crop_vars (np.ndarray): [cx, cy, xmin, xmax, ymin, ymax] per crop, shape (K, 6).

        Raises:
            ValueError: If a seed triplet cannot be established (e.g., <3 points or
                no consecutive triple).
        """

        # 1) -- Unpack variables & Calculate seed picks/frames--
        seed_picks_global, seed_indices = self.selectSeedTriplet(self.picks, self.pick_frame_indices)

        omega = np.arctan2(self.picks[-1][1] - self.picks[0][1], 
                           -self.picks[-1][0] + self.picks[0][0])  % (2*np.pi)

        if self.verbose:
            print(f"Starting recursive cropping with {len(seed_indices)} seed picks at indices" 
                  f"{seed_indices} and omega {omega} radians.")

        # 2) -- Estimate initial parameters from the seed picks --

        # Estimate the length (take largest value to be safe)
        init_length = np.max([np.linalg.norm(seed_picks_global[0] - seed_picks_global[1]),
                                np.linalg.norm(seed_picks_global[1] - seed_picks_global[2])])        

        # Instantiate omega as a instance variable
        self.omega = omega

        # Determine direction of motion
        norm = seed_picks_global[-1] - seed_picks_global[0]
        directions = (-1 if norm[0] < 0 else 1, -1 if norm[1] < 0 else 1)
        self.directions = directions

        # 3) -- Instantiate nessesary instance arrays --

        # (N, w, h) array of cropped frames
        self.cropped_frames = [None] * len(seed_indices)

        # (N, 5) array of first pass parameters (level_sum, height, x0)
        self.first_pass_params = np.zeros((len(seed_indices), 5), dtype=np.float32) 

        # (N, 6) array of crop variables (cx, cy, xmin, xmax, ymin, ymax)
        self.crop_vars = np.zeros((len(seed_indices), 6), dtype=np.float32) 

        # Load in the subtracted seed frames
        seed_subtracted_frames = self.getFrame(seed_indices)

        # # 4) -- Process each seed pick to kick-start recursion --
        for i in range(len(seed_indices)):

            # Crop initial frames
            self.cropped_frames[i], self.crop_vars[i] = self.cropFrameToGaussian(
                        seed_subtracted_frames[i],
                        self.estimateCenter(seed_picks_global[i], omega, init_length, directions=directions),
                        self.cropping_settings["init_sigma_guess"],
                        init_length * self.cropping_settings["max_length_coeff"],
                        omega
                        )
            
            # Run a first-pass Gaussian on the cropped frames
            self.first_pass_params[i], _ = self.gaussianPSO(self.cropped_frames[i], omega, directions, 
                                                                init_length=init_length)

            # Update progress
            self.progressed_frames['cropping'] += 1
            self.updateProgress()
            
        # Verbose output   
        if self.verbose:
            print(f"Finished cropping {len(seed_indices)} frames with est. centroids:" 
                        f"{self.first_pass_params[:, 2:4]}")

        # Instantiate parameter estimation functions
        parameter_estimation_functions = self.updateParameterEstimationFunctions(
                                                            self.crop_vars, 
                                                            self.first_pass_params, 
                                                            forward_pass=True
                                                    ) # Set forward_pass to True since there are only 3 points

        # Estimate next forward center
        forward_next_center_global = self.estimateNextCenter(
            self.estimateCenter(seed_picks_global[-1], omega, init_length, directions=directions),
            self.estimateNextParameters(parameter_estimation_functions, 3, forward_pass=True)['norm'],
            omega,
            directions=directions
        )

        # Begin forwards pass on crop
        self.recursiveCroppingAlgorithm(seed_indices[-1] + 1,
                                      forward_next_center_global,
                                      parameter_estimation_functions,
                                      omega,
                                      directions=directions,
                                      forward_pass=True,
                                      )

        # Estimate prev. backward center
        backward_next_center_global = self.estimateNextCenter(
            self.estimateCenter(seed_picks_global[0], omega, init_length, directions=directions),
            self.estimateNextParameters(parameter_estimation_functions, 
                                        self.first_pass_params.shape[0], forward_pass=False)['norm'],
            omega,
            directions=tuple(-x for x in list(directions)) # Invert directions
        )

        # Begin backwards pass on crop
        self.recursiveCroppingAlgorithm(seed_indices[0] - 1,
                                      backward_next_center_global,
                                      parameter_estimation_functions,
                                      omega,
                                      directions=tuple(-x for x in list(directions)), # Invert directions
                                      forward_pass=False,
                                      )

    def refineAllMeteorCrops(self, first_pass_params, cropped_frames, omega, directions):
        """
        Second-pass local optimization of moving-Gaussian fits (per crop).

        Uses L-BFGS-B with adaptive bounds derived from first-pass variability to refine
        [level_sum, sigma, x0, y0, length]. Also renders fitted images for diagnostics.

        Args:
            first_pass_params (np.ndarray): Initial params, shape (K, 5).
            cropped_frames (Sequence[np.ndarray]): Cropped images, length K.
            omega (float): Track angle in radians.
            directions (tuple[int, int]): Direction multipliers (+1/-1) for x/y axes.

        Returns:
            tuple:
                refined_fit_params (np.ndarray): Refined parameters, shape (K, 5).
                fit_costs (np.ndarray): Objective values per crop, shape (K,).
                fit_imgs (list[np.ndarray]): Reconstructed fitted images, len K.

        Raises:
            ValueError: If bounds are malformed.
            RuntimeError: If optimizer fails catastrophically (non-recoverable).
        """


        # Instantiate arrays to store fit data

        # (N, 5) array of refined fit parameters (level_sum, height, x0, y0, length)
        self.refined_fit_params = np.zeros((first_pass_params.shape[0], 5), dtype=np.float32) 

        # (N,) array of fit costs
        self.fit_costs = np.zeros((first_pass_params.shape[0],), dtype=np.float32) 
        
        self.fit_imgs = []

        # Iterate over all cropped frames and fit the second pass Gaussian
        for i, (p0, obs_frame) in enumerate(zip(first_pass_params, cropped_frames)):

            # Calculate the adaptive bounds based on the first pass parameters
            bounds = self.calculateAdaptiveBounds(first_pass_params, p0, True)

            # Perform the local Gaussian fit 
            fit_params, cost, img = self.localGaussianFit(p0, obs_frame, omega, bounds, directions)

            # Append results to the lists
            self.refined_fit_params[i] = fit_params
            self.fit_costs[i] = cost
            self.fit_imgs.append(img)

            # Update progress
            self.progressed_frames['refining'] += 1
            self.updateProgress()


    def removeLowSNRPicks(self, refined_params, fit_imgs, cropped_frames, crop_vars, 
                          pick_frame_indices, fit_costs):
        """
        Filter out low-SNR or out-of-bounds picks and materialize global outputs.

        For each fitted crop, build a photometric mask from the fitted image,
        measure SNR using a CCD equation, enforce geometric in-bounds constraints,
        and retain only valid frames. Also translates centers into global coordinates
        and moves them to the leading edge of the streak.

        Args:
            refined_params (np.ndarray): Refined Gaussian params, shape (K, 5).
            fit_imgs (Sequence[np.ndarray]): Per-crop fitted images.
            frames (np.ndarray): Full frame stack (N, H, W), used for SNR computation.
            cropped_frames (Sequence[np.ndarray]): Per-crop image crops.
            crop_vars (np.ndarray): Crop bookkeeping (K, 6): [cx, cy, xmin, xmax, ymin, ymax].
            pick_frame_indices (Sequence[int]): Frame index per crop, length K.
            fit_costs (np.ndarray): Objective values per crop, shape (K,).
            times (Sequence[datetime.datetime]): Original timestamps.

        Returns:
            tuple:
                refined_params (np.ndarray): Post-filter params, shape (M, 5).
                fit_imgs (list[np.ndarray]): Post-filter fit images, len M.
                cropped_frames (list[np.ndarray]): Post-filter crops, len M.
                crop_vars (np.ndarray): Post-filter crop vars, shape (M, 6).
                pick_frame_indices (np.ndarray): Post-filter frame indices, shape (M,).
                global_picks (np.ndarray): Edge-aligned global picks, shape (M, 2).
                fit_costs (np.ndarray): Post-filter costs, shape (M,).
                times (np.ndarray): Post-filter timestamps, shape (M,).
                abs_level_sums (np.ndarray): Background-subtracted flux sums, shape (M,).

        Raises:
            Exception: If photometry or SNR computation fails catastrophically.
        """


        # Instantiate boolean array to store rejected frames & array to store SNR values
        self.abs_level_sums = []
        snr_rejection_bool = np.zeros((refined_params.shape[0],), dtype=bool)
        frame_snr_values = []
        self.background_levels = []
        self.photometry_pixels = []
        self.saturated_bool_list = []

        if self.verbose:
            self.temp_count = 0

        # Itterate over each frame
        for i in range(len(fit_imgs)):

            frame_idx = pick_frame_indices[i]
            frame, uncorr_frame, non_sub_frame = self.getFrame(frame_idx, include_raw=True, 
                                                                include_non_subtracted=True)

            # Calculate photom_pixels
            photom_pixels = self.computePhotometryPixels(fit_imgs[i], cropped_frames[i], crop_vars[i])

            # Calculate SNR, and photom values
            snr = self.computeIntensitySum(photom_pixels, 
                    self.translatePicksToGlobal((refined_params[i, 2], refined_params[i, 3]), crop_vars[i]), 
                    frame, uncorr_frame, non_sub_frame)

            # Set index for previous parameters 
            # (util. previous since optim. will reshape curr params to fit even if streak is partially OOB)
            idx = i - 1 if i > 0 else 0

            # Reject SNR below the threshold
            if snr < self.snr_threshold:
                snr_rejection_bool[i] = True

                if self.verbose:
                    print(f"Rejecting frame {i} with SNR {snr} below threshold {self.snr_threshold}.")

            # Check if the streak is outside the image bounds
            elif not self.checkStreakInBounds(
                    self.translatePicksToGlobal((refined_params[i, 2], refined_params[i, 3]), crop_vars[i]),
                    refined_params[idx, 4], refined_params[idx, 1], self.omega, self.directions):
                snr_rejection_bool[i] = True

                if self.verbose:
                    pick_coords = self.translatePicksToGlobal((refined_params[i, 2], refined_params[i, 3]), crop_vars[i])
                    print(f"Rejecting frame {i} with out-of-bounds pick {pick_coords}.")

            # If passes all checks, append SNR and level sum
            else:
                frame_snr_values.append(snr)
            
            # update progress
            self.progressed_frames['removing'] += 1
            self.updateProgress()

        # Print number of rejected frames
        print(f"Rejected {np.sum(snr_rejection_bool)} frames with SNR below {self.snr_threshold}.")
            
        # Reject bad frames by removing indexes of low-SNR frames (numpy arrays)
        refined_params = self.refined_fit_params[~snr_rejection_bool]
        crop_vars = crop_vars[~snr_rejection_bool]
        pick_frame_indices = np.array(pick_frame_indices)[~snr_rejection_bool]
        fit_costs = fit_costs[~snr_rejection_bool]
        self.abs_level_sums = np.array(self.abs_level_sums)[~snr_rejection_bool]
        self.background_levels = np.array(self.background_levels)[~snr_rejection_bool]
        self.saturated_bool_list = np.array(self.saturated_bool_list)[~snr_rejection_bool]

        # Reject bad frames for non-numpy arrays

        # Save copies before indexing to avoid recursion errors
        fit_imgs_copy = fit_imgs.copy()
        cropped_frames_copy = cropped_frames.copy()
        photometry_pixels_copy = self.photometry_pixels.copy()

        # Remove low-SNR frames from fit_imgs and cropped_frames
        fit_imgs = [fit_imgs_copy[i] for i in range(len(fit_imgs_copy)) if not snr_rejection_bool[i]]
        cropped_frames = [cropped_frames_copy[i] for i in range(len(cropped_frames_copy)) 
                                    if not snr_rejection_bool[i]]
        self.photometry_pixels = [photometry_pixels_copy[i] for i in range(len(photometry_pixels_copy)) 
                                    if not snr_rejection_bool[i]]

        # Save all leading edge picks by translating to global and moving to edge
        global_picks = np.array([self.movePickToEdge(
                    self.translatePicksToGlobal((refined_params[i, 2], refined_params[i, 3]), crop_vars[i]),
                    self.omega,
                    refined_params[i][4],
                    directions=self.directions,
                    pick_offset=self.pick_offset) for i in range(len(refined_params))])

        # Save all as instance variables
        self.refined_fit_params = refined_params
        self.fit_imgs = fit_imgs
        self.cropped_frames = cropped_frames
        self.crop_vars = crop_vars
        self.pick_frame_indices = pick_frame_indices
        self.global_picks = global_picks
        self.fit_costs = fit_costs

        # Clip to avoid division errors
        self.snr = np.clip(frame_snr_values, 0.01, None)
        self.abs_level_sums = np.clip(self.abs_level_sums, 1, None)



    def saveAni(self, data_path):
        """
        Save diagnostic frames as JPEGs (thread-safe, headless-safe).

        Renders a 3-panel figure per kept crop (crop, fit, absolute residuals) and
        annotates key parameters/SNR. Uses Matplotlib's Agg backend and writes
        to `<data_path>/ASTRA_Kalman_Results/ASTRA_frames_YYYYmmdd_HHMMSS/`.

        Args:
            data_path (str | os.PathLike): Root directory for output.

        Returns:
            None

        Raises:
            OSError: If directories cannot be created or files written.
        """

        current_backend = matplotlib.get_backend()
        matplotlib.use("Agg")  # headless-safe backend

        # ---- thread safety (one-at-a-time per instance) ----
        if not hasattr(self, "_save_lock"):
            self._save_lock = threading.RLock()

        with self._save_lock:
            picks = self.global_picks

            # Output folder: {data_path}/ASTRA_Kalman_Results/ASTRA_frames_YYYYmmdd_HHMMSS
            root = os.path.join(data_path, "ASTRA_Kalman_Results")
            outdir = os.path.join(root, f"ASTRA_frames_{datetime.now():%Y%m%d_%H%M%S}")
            os.makedirs(outdir, exist_ok=True)

            n_crops = len(self.pick_frame_indices)
            if n_crops == 0:
                return

            # Render & save each frame independently (no shared pyplot state)
            for i in range(n_crops):
                crop       = np.asarray(self.cropped_frames[i])
                crop_vars  = self.crop_vars[i]
                fit        = np.asarray(self.fit_imgs[i])
                fit_params = self.refined_fit_params[i]
                snr_val    = float(self.snr[i])
                frame_num  = int(self.pick_frame_indices[i])
                level_sum  = self.abs_level_sums[i]
                background_std = self.background_levels[i]
                photom_count = len(np.array(self.photometry_pixels[i]).T[0])
                sat_bool = self.saturated_bool_list[i]

                # translate pick to local crop coords for plotting
                try:
                    x, y = self.translatePicksToGlobal(picks[i], crop_vars, global_to_local=True)
                except Exception:
                    # fall back: no overlay if translation fails
                    x, y = None, None

                # Build a fresh Figure per frame (thread-safe)
                fig = Figure(figsize=(12, 6), dpi=100)
                _ = FigureCanvas(fig)
                axs = fig.subplots(1, 3)

                # Title + footer text (no pyplot)
                fig.suptitle(f"Crop {i} — SNR: {snr_val:.2f}", fontsize=14)

                # left: crop image
                ax1 = axs[0]
                ax1.set_title(f"Crop {i}")
                ax1.imshow(crop, cmap="gray", vmin=0, vmax=np.max(crop) if crop.size else 1)
                if x is not None and y is not None:
                    ax1.plot(x, y, "ro", markersize=5)

                # middle: fit image
                ax2 = axs[1]
                ax2.set_title("Gaussian Fit")
                ax2.imshow(fit, cmap="gray", vmin=0, vmax=np.max(fit) if fit.size else 1)
                if x is not None and y is not None:
                    ax2.plot(x, y, "ro", markersize=5)

                # right: residuals
                ax3 = axs[2]
                ax3.set_title("Abs Fit Residuals")
                abs_res = np.abs(crop - fit) if (crop.shape == fit.shape) else np.zeros_like(crop)
                vmin = np.min(abs_res) if abs_res.size else 0
                vmax = np.max(abs_res) if abs_res.size else 1
                im = ax3.imshow(abs_res, cmap="coolwarm", vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax3, shrink=0.6)

                # footer with params
                # (avoid fig.text removal loops; just write once)
                # Add subtitle with additional parameters
                fig.text(
                    0.5, 0.92,
                    f"Level Sum: {level_sum:.0f}  •  Background STD: {background_std:.2f}  •  "
                    f"Photom Count: {photom_count}  •  Saturated: {sat_bool}",
                    ha="center", va="center", fontsize=10
                )

                fig.text(
                    0.5, 0.02,
                    f"Frame: {frame_num}  •  "
                    f"Fit Level Sum: {fit_params[0]:.2f}  •  Sigma: {fit_params[1]:.2f}  •  "
                    f"x0: {fit_params[2]:.2f}  •  y0: {fit_params[3]:.2f}  •  L: {fit_params[4]:.2f}",
                    ha="center", va="center", fontsize=10
                )

                # Make a portable file name
                fname = f"{i:04d}_frame{frame_num}.jpg"
                fpath = os.path.join(outdir, fname)

                # Save as JPEG (Pillow via Matplotlib); set reasonable quality
                fig.savefig(
                    fpath,
                    format="jpg",
                    bbox_inches="tight",
                    dpi=100,
                    pil_kwargs={"quality": 95, "optimize": True}
                )

                # Explicitly clear to free memory in long sequences
                fig.clf()

            # Optional: print where they went
            print(f"Saved {n_crops} JPEGs to: {outdir}")

        # Revert backend so parent class can still use matplotlib
        matplotlib.use(current_backend)

    # 2) -- Calculation/Conversion Methods --

    def checkStreakInBounds(self, global_pick, prev_length, prev_sigma, omega, directions):
        """
        Check whether a streak centered at `global_pick` (with length/sigma
        from the previous frame) lies safely within image bounds.

        The check considers both along-track edges (front/back) and cross-track
        extents (~1 sigma) and enforces a small edge buffer.

        Args:
            global_pick (tuple[float, float]): Global (x, y) centroid.
            prev_length (float): Fitted track length from a previous step.
            prev_sigma (float): Fitted cross-track sigma from a previous step.
            omega (float): Track angle [rad].
            directions (tuple[int, int]): Direction multipliers (+1/-1) for x/y.

        Returns:
            bool: True if fully in-bounds (with buffer), False otherwise.
        """

        # Define return bool
        in_bounds = True

        # Calculate front and back of the streak
        edge_pick_back = self.movePickToEdge(global_pick, omega, prev_length, directions=directions)
        edge_pick_front = self.movePickToEdge(global_pick, omega, prev_length,
                                              directions=tuple(-x for x in list(directions)))

        # Calculate the top and bottom of the streak (only check a single sigma,
        # as long as there is clearence for center pick it is okay)
        top_pick = self.movePickToEdge(global_pick, omega+(np.pi / 2), prev_sigma,
                                       directions=(directions[0], -directions[1]))
        bottom_pick = self.movePickToEdge(global_pick, omega+(np.pi / 2), prev_sigma,
                                          directions=(-directions[0], directions[1]))

        # Package all picks into a list
        bounding_points = [edge_pick_back, edge_pick_front, top_pick, bottom_pick]
        
        # Get correct image dimensions
        height, width = self.corrected_avepixel.shape[0], self.corrected_avepixel.shape[1]

        # Check if any of the bounding points are out of bounds
        # Correctly compare x (point[0]) with width and y (point[1]) with height.
        if np.any([point[0] < 0 or point[0] >= width or point[1] < 0 or
                   point[1] >= height for point in bounding_points]):
            in_bounds = False

        # Check if pick is close enough to edge to be considered OOB
        edge_buffer = 3  # pixels
        # Correctly compare x (global_pick[0]) with width and y (global_pick[1]) with height.
        if (global_pick[0] < edge_buffer or global_pick[0] >= width - edge_buffer or
            global_pick[1] < edge_buffer or global_pick[1] >= height - edge_buffer):
            in_bounds = False

        # Return the in_bounds bool
        return in_bounds


    def translatePicksToGlobal(self, local_pick, frame_crop_vars, global_to_local=False):
        """
        Translate between local-crop and global image coordinates.

        Args:
            local_pick (tuple[float, float]): (x, y) coordinates in the source space;
                interpreted as local or global based on `global_to_local`.
            frame_crop_vars (Sequence[float]): [cx, cy, xmin, xmax, ymin, ymax] for
                the crop.
            global_to_local (bool): If True, treat `local_pick` as global and convert
                to crop-local; otherwise local→global.

        Returns:
            tuple[float, float]: Converted (x, y) coordinate.
        """

        # Unpack crop variables
        _, _, xmin, _, ymin, _ = frame_crop_vars

        if global_to_local:
            # Translate global pick to local coordinates
            x_local = local_pick[0] - xmin
            y_local = local_pick[1] - ymin
            return (x_local, y_local)
        
        else:
            # Translate pick to global coordinates
            x_global = local_pick[0] + xmin
            y_global = local_pick[1] + ymin

            return (x_global, y_global)
    

    def movePickToEdge(self, center_pick, omega, length, directions=(1, 1), pick_offset=None):
        """
        Move a center pick to the leading edge of the streak.

        Computes a half-length offset along `omega` using direction multipliers,
        optionally scaled by `pick_offset` for “center-to-edge” behavior.

        Args:
            center_pick (tuple[float, float]): Center (x0, y0).
            omega (float): Track angle [rad].
            length (float): Fitted length.
            directions (tuple[int, int]): Multipliers (+1/-1) for x and y axes.
            pick_offset (float | None): Optional center-to-edge coefficient; if provided,
                the used length is `(length/3) * pick_offset`.

        Returns:
            tuple[float, float]: Edge (x, y).
        """


        # Unpack picks
        x0, y0 = center_pick

        if pick_offset is not None:
            # Add length adjustment
            length = (length / 3) * pick_offset

        # Calculate the offset
        x_midpoint_offset = (length / 2) * np.abs(np.cos(omega)) * directions[0]
        y_midpoint_offset = (length / 2) * np.abs(np.sin(omega)) * directions[1]

        # Calculate the new pick position
        edge_x = x0 + x_midpoint_offset
        edge_y = y0 + y_midpoint_offset

        # Return the new picks
        return (edge_x, edge_y)
    

    def estimateCenter(self, edge_pick, omega, length, directions=(1, 1)):
        """
        Estimate center (x, y) from an edge pick, angle, and length.

        This is the inverse of `movePickToEdge`—it moves from an edge back to the center
        by flipping the direction multipliers.

        Args:
            edge_pick (tuple[float, float]): Edge (x, y).
            omega (float): Track angle [rad].
            length (float): Fitted length.
            directions (tuple[int, int]): Multipliers (+1/-1) for x and y.

        Returns:
            tuple[float, float]: Center (x, y).
        """

        # Invert directions
        inverted_directions = (-directions[0], -directions[1])

        # Appy inverted direction to movePickToEdge
        center_pick = self.movePickToEdge(edge_pick, omega, length, inverted_directions)

        # Return the center pick
        return center_pick
    

    def calculateAdaptiveVelocityClamp(self, bounds):
        """
        Compute a per-dimension PSO velocity clamp as a fraction of each
        parameter’s feasible domain (lower/upper bounds).

        Args:
            bounds (tuple[np.ndarray, np.ndarray]): (lb, ub), each shape (D,).

        Returns:
            tuple[np.ndarray, np.ndarray]: (vmin, vmax), shape (D,).
        """

        # Unpack bounds
        lb, ub = bounds

        # Calculate the domain of each parameter
        parameter_ranges = ub - lb

        # Set fraction of parameter domain
        adaptive_velocity_clamps = parameter_ranges * self.first_pass_settings['Velocity_coeff']

        # Return the velocity clamp as a tuple
        return (-adaptive_velocity_clamps, adaptive_velocity_clamps)
    

    def calculateAdaptiveBounds(self, first_pass_parameters, p0, scipy_format = False):
        """
        Build adaptive bounds around a point `p0` using first-pass variability.

        Bounds are centered at `p0` with half-widths proportional to the empirical
        std across first-pass params. Optionally returns SciPy “(low, high)” pairs.

        Args:
            first_pass_parameters (np.ndarray): Prior params, shape (K, 5).
            p0 (array-like): Center point for bounds, length 5.
            scipy_format (bool): If True, return tuple[(low, high), ...] for L-BFGS-B.
                Otherwise return (lb, ub) arrays.

        Returns:
            tuple:
                - If `scipy_format` is False: (lb, ub) as np.ndarrays, shape (5,).
                - If True: tuple of 5 (low, high) pairs.

        Raises:
            ValueError: If shapes do not match expected 5-parameter layout.
        """

        # Calculate the std of all values from the first pass
        std_parameters = np.std(first_pass_parameters, axis=0)

        # Calculate the adaptive bounds based on the mean and std of the first pass parameters
        # The std_parameter_constraint is a list of multiples of the std to use for each parameter
        adaptive_bounds = (
        np.array([p0[0] - self.std_parameter_constraint[0] * std_parameters[0], # level_sum
                    p0[1] - self.std_parameter_constraint[1] * std_parameters[1], # STD / height of gaussian
                    p0[2] - self.std_parameter_constraint[2] * std_parameters[2], # X-center of gaussian
                    p0[3] - self.std_parameter_constraint[3] * std_parameters[3], # Y-center of gaussian
                    p0[4] - self.std_parameter_constraint[4] * std_parameters[4] # 3*STD / length of gaussian
                    ]), # Lower bounds
        np.array([p0[0] + self.std_parameter_constraint[0] * std_parameters[0], # level_sum
                    p0[1] + self.std_parameter_constraint[1] * std_parameters[1], # STD / height of gaussian
                    p0[2] + self.std_parameter_constraint[2] * std_parameters[2], # X-center of gaussian
                    p0[3] + self.std_parameter_constraint[3] * std_parameters[3], # Y-center of gaussian
                    p0[4] + self.std_parameter_constraint[4] * std_parameters[4] # 3*STD / length of gaussian
                    ]) # Upper bounds
        )

        # Clip bounds minimum to slightly above zero
        adaptive_bounds = (np.clip(adaptive_bounds[0], 1e-5, None),
                           np.clip(adaptive_bounds[1], 1e-4, None))
        
        # Change the adaptive bounds format to scipy format if requested
        if scipy_format:
            lb, ub = adaptive_bounds
            adaptive_bounds = tuple(zip(lb, ub))

        # Return the adaptive bounds
        return adaptive_bounds
    

    def estimateNextCenter(self, current_global_center, length, omega, directions=(1, 1)):
        """
        Project the next center along the track from a current center.

        Args:
            current_global_center (tuple[float, float]): Current (x, y).
            length (float): Step size to move along the track (e.g., predicted norm).
            omega (float): Track angle [rad].
            directions (tuple[int, int]): Direction multipliers (+1/-1) for x/y.

        Returns:
            tuple[float, float]: Next center (x, y).
        """


        # Unpack current global center
        x0, y0 = current_global_center

        # Calculate the offset
        x_midpoint_offset = length * np.abs(np.cos(omega)) * directions[0]
        y_midpoint_offset = length * np.abs(np.sin(omega)) * directions[1]

        # Calculate the next center position
        next_x = x0 + x_midpoint_offset
        next_y = y0 + y_midpoint_offset

        # Return the new picks
        return (next_x, next_y)

    # 3) -- Helper Methods --

    def updateParameterEstimationFunctions(self, crop_vars, first_pass_params, forward_pass):
        """
        Fit simple polynomial predictors for level_sum, height (sigma), length, and
        inter-frame norm based on first-pass parameters.

        Used to extrapolate the next crop’s expected magnitude and spacing during
        the recursive pass.

        Args:
            crop_vars (np.ndarray): Crop vars, shape (K, 6).
            first_pass_params (np.ndarray): Params [level_sum, sigma, x0, y0, length], shape (K, 5).
            forward_pass (bool): If True, fit end-anchored predictors; otherwise start-anchored.

        Returns:
            dict[str, numpy.polynomial.Polynomial]:
                Keys: {"level_sum", "height", "length", "norm"}; each maps to a
                degree-1 Polynomial used for extrapolation.
        """


        # convert all_params & crop_vars to a numpy array
        all_params = first_pass_params.copy()
        xymin = crop_vars.copy()[:, [2, 4]]

        # Unpack parameters
        level_sums, heights, lengths = all_params[:, 0], all_params[:, 1], all_params[:, 4]

        # Calculate magnitudes of all norms
        global_centroids = all_params[:, 2:4] + xymin

        # Calculate and sort norms between all consequitive frames
        global_centroids = global_centroids[np.argsort(global_centroids[:, 0])]
        deltas = global_centroids[1:] - global_centroids[:-1]
        norms = np.linalg.norm(deltas, axis=1)

        # Ensure level_sums has only values above zero
        level_sums = np.clip(level_sums, 1e-6, None)

        # Translate level_sum into log space
        level_sums = np.log(level_sums)

        # Use moving linear method for level_sum (Use three last point to estimate next)
        paramter_estimation_functions = {
            'level_sum': np.polynomial.Polynomial.fit((np.array([0, 1, 2]) + (len(level_sums) - 3)), 
                        level_sums[-3:], 1) if forward_pass 
                        else np.polynomial.Polynomial.fit(range(3), level_sums[:3], 1),
            'height': np.polynomial.Polynomial.fit(range(len(heights)), heights, 1),
            'length':  np.polynomial.Polynomial.fit((np.array([0, 1, 2]) + (len(lengths) - 3)), 
                        lengths[-3:], 1) if forward_pass 
                        else np.polynomial.Polynomial.fit(range(3), lengths[:3], 1),
            'norm' : np.polynomial.Polynomial.fit(range(len(norms)), norms, 1)
        }

        return paramter_estimation_functions


    def estimateNextParameters(self, paramter_estimation_functions, n_points, forward_pass):
        """
        Extrapolate next parameter magnitudes using fitted polynomials.

        For forward passes, evaluates predictors at `n_points + 1` with fallback to
        previous index if any prediction is non-positive; for backward passes,
        evaluates at -1 with fallback to 0. Restores `level_sum` from log-space.

        Args:
            paramter_estimation_functions (dict[str, Polynomial]): Predictors from
                `updateParameterEstimationFunctions`.
            n_points (int): Current number of param rows.
            forward_pass (bool): Direction flag.

        Returns:
            dict[str, float]: {'level_sum', 'height', 'length', 'norm'} predictions.
        """


        # Estimate next value if forward pass is true
        if forward_pass:
            x = n_points + 1
            prev = n_points
        else:
            x = -1
            prev = 0

        # Estimate next values
        next_level_sum = paramter_estimation_functions['level_sum'](x)
        next_height = paramter_estimation_functions['height'](x)
        next_length = paramter_estimation_functions['length'](x)
        next_norm = paramter_estimation_functions['norm'](x)

        # Translate next_level_sum back into regualr space from log space
        next_level_sum = np.exp(next_level_sum)

        # Check if any values are below or equal to zero, and set them to the function evaluation at prev
        if next_level_sum <= 0:
            next_level_sum = paramter_estimation_functions['level_sum'](prev)
        if next_height <= 0:
            next_height = paramter_estimation_functions['height'](prev)
        if next_length <= 0:
            next_length = paramter_estimation_functions['length'](prev)
        if next_norm <= 0:
            next_norm = paramter_estimation_functions['norm'](prev)

        # Pack next parameters as a dict
        next_params = {
            'level_sum': next_level_sum,
            'height': next_height,
            'length': next_length,
            'norm': next_norm
        }

        # Return next parameters
        return next_params


    def localGaussianFit(self, p0, obs_frame, omega, bounds, directions):
        """
        Locally refine a moving-Gaussian fit on a single crop using L-BFGS-B.

        Objective variants include absolute, squared, or cubed residuals. Adds
        penalties for parameter bound violations and for the streak leaving crop bounds.

        Args:
            p0 (np.ndarray): Initial params [level_sum, sigma, x0, y0, length], shape (5,).
            obs_frame (np.ndarray): Crop image, shape (h, w).
            omega (float): Track angle [rad].
            bounds (Sequence[tuple[float, float]]): Per-param bounds for L-BFGS-B.
            directions (tuple[int, int]): Direction multipliers (+1/-1) for x/y.

        Returns:
            tuple:
                best_pos (np.ndarray): Optimized parameters, shape (5,).
                best_cost (float): Objective value at optimum.
                img (np.ndarray): Reconstructed fitted image, shape (h, w).

        Raises:
            ValueError: On unknown residual method.
        """

        # Define initial parameters
        y, x = np.indices(obs_frame.shape)
        data_tuple = (x, y)
        y_obs = obs_frame.ravel()

        # Instantiate optimizer
        res = scipy.optimize.minimize(
            fun=self.localObjectiveFunction,
            x0=p0,  # Exclude omega from initial guess
            args=(data_tuple, y_obs, 0, omega, bounds, directions),
            method=self.second_pass_settings["method"],
            bounds=bounds
        )

        # If unsuccesful revert to initial guess
        if res.success is False:
            print(f"Warning: Local optimization failed with message: "
                  f"{res.message}. Reverting to initial guess.")
            best_pos = p0
            best_cost = self.localObjectiveFunction(p0, data_tuple, y_obs, 0, omega, bounds, directions)
        else:
            # Get best cost and position
            best_pos = res.x
            best_cost = res.fun
 
        # Check if best_pos is within bounds
        for i, (val, bound) in enumerate(zip(best_pos, bounds)):
            lower, upper = bound
            if val < lower or val > upper: 
                print(f"Warning: Parameter {i} (value: {val}) is outside the bounds [{lower}, {upper}]")

        # Calculate the fitted image
        img = self.movingGaussian(data_tuple, omega, 0, *best_pos).reshape(obs_frame.shape)

        # Return all values
        return best_pos, best_cost, img

        
    def gaussianPSO(self, cropped_frame, omega, directions, estim_next_params=None, init_length=None):
        """
        First-pass moving-Gaussian fit per crop using Particle Swarm Optimization.

        Initial guesses come from the crop’s sum/size or predicted magnitudes from
        splines. Applies an out-of-bounds penalty and enforces parameter constraints.

        Args:
            cropped_frame (np.ndarray): Crop image, shape (h, w).
            omega (float): Track angle [rad].
            directions (tuple[int, int]): Direction multipliers (+1/-1) for x/y.
            estim_next_params (dict | None): Optional predictions {'level_sum','height','length'}.
            init_length (float | None): Optional initial length for first seed crops.

        Returns:
            tuple:
                best_pos (np.ndarray): Best parameters [level_sum, sigma, x0, y0, length], shape (5,).
                best_cost (float): Objective value.

        Raises:
            Exception: If PSO setup or optimization fails.
        """

        # 1) -- Define inital params --
        y, x = np.indices(cropped_frame.shape)
        data_tuple = (x, y)
        y_len, x_len = cropped_frame.shape
        y_obs = cropped_frame.ravel()

        # Determine estimations based on passed spline or init_length
        if estim_next_params is not None:
            est_level_sum = estim_next_params['level_sum']
            est_height = estim_next_params['height']
            est_length = estim_next_params['length']
            if self.verbose:
                print(f"Using provided estimations: {estim_next_params}")
        else:
            try:
                est_level_sum = np.sum(cropped_frame)
                est_length = init_length
                est_height = 2
            except Exception as e:
                raise ValueError("Either estim_next_params "
                "or init_length must be provided for first_pass_gaussian.")
        
        # 2) -- Build up p0, i0, v0 and bounds --
        p0 = [
            est_level_sum, # level_sum
            est_height, # height / sigma_y
            x_len / 2, # x0
            y_len / 2, # y0
            est_length # length (std_x * 6)
        ]

        bounds = (
                np.array([100, # level_sum 
                        p0[1] * 0.2, # STD / height of gaussian
                        x_len * 0.25, # X-center of gaussian
                        y_len * 0.25, # Y-center of gaussian
                        p0[4] * 0.5, # 3*STD / length of gaussian
                        ]), # Lower bounds
                np.array([np.sum(cropped_frame), # level_sum
                        est_height * 1.35, # STD / height of gaussian
                        x_len * 0.75, # X-center of gaussian
                        y_len * 0.75, # Y-center of gaussian
                        p0[4] * 1.5, # 3*STD / length of gaussian
                        ]) # Upper bounds
            )

        # Clip bounds to above zero, and p0 to bounds
        bounds = (np.clip(bounds[0], 0.01, None), np.clip(bounds[1], 0.1, None))
        p0 = np.clip(p0, bounds[0], bounds[1])
        
        # Generate initial particle positions
        i0 = self.generateInitialParticles(bounds, self.first_pass_settings['n_particles'], p0=p0)

        # Generate adaptive velocity clamp
        v0 = self.calculateAdaptiveVelocityClamp(bounds)

        # 3) -- Run PSO --
        try:
            optimizer = GlobalBestPSO(
                n_particles = self.first_pass_settings["n_particles"],
                bh_strategy=self.first_pass_settings["bh_strategy"],
                vh_strategy=self.first_pass_settings["vh_strategy"],
                ftol = self.first_pass_settings["ftol"],
                ftol_iter= self.first_pass_settings["ftol_iter"],
                velocity_clamp = v0,
                dimensions = 5,
                bounds=bounds,
                options=self.first_pass_settings["options"],
                init_pos=i0
            )

            # Solve optimizer
            best_cost, best_pos = optimizer.optimize(
                objective_func = self.psoObjectiveFunction,
                iters = self.first_pass_settings["max_iter"],
                verbose = self.verbose,
                data_tuple = data_tuple,
                y_obs = y_obs,
                a0 = 0,
                bounds = bounds,
                omega = omega,
                directions = directions
            )
        except Exception as e:
            raise Exception(f"Error running PSO: {e}")
        
        # Raise warnings if best_pos is out of bounds
        for i, (val, lower, upper) in enumerate(zip(best_pos, bounds[0], bounds[1])):
            if val < lower or val > upper:
                print(f"Warning: Parameter {i} (value: {val}) is outside the bounds [{lower}, {upper}]")

        return best_pos, best_cost


    def psoObjectiveFunction(self, params, data_tuple, y_obs, a0, omega, bounds, directions):
        """
        PSO objective: residuals between observed crop and moving-Gaussian image.

        For each particle, this computes the fitted intensity image and evaluates
        the requested residual metric. Also applies penalties if params leave bounds
        or if the streak’s front/back leave the crop area.

        Args:
            params (np.ndarray): Swarm positions, shape (P, 5).
            data_tuple (tuple[np.ndarray, np.ndarray]): (x_index, y_index) grids.
            y_obs (np.ndarray): Observed crop flattened, shape (h*w,).
            a0 (float): Background level (kept at 0 in current calls).
            omega (float): Track angle [rad].
            bounds (tuple[np.ndarray, np.ndarray]): (lb, ub) arrays, each shape (5,).
            directions (tuple[int, int]): Direction multipliers for x/y.

        Returns:
            np.ndarray: Residual per particle, shape (P,).

        Raises:
            ValueError: On unknown residual method.
        """

        # 1) -- Instantiate residuals object --
        residuals = np.zeros(params.shape[0])

        # 2) -- Compute each p gaussian --
        for i, p in enumerate(params):

            # Unpack parameters
            level_sum, sigma, x0, y0, length = p
            
            # Calculate intensity of the moving gaussian
            intens = self.movingGaussian(data_tuple, omega, a0, level_sum, sigma, x0, y0, length)

            # Calculate the residuals based on the specified method
            if self.first_pass_settings["residuals_method"] == 'abs_squared':
                residuals[i] = np.sum(np.abs(intens - y_obs)**2)
            elif self.first_pass_settings["residuals_method"] == 'abs':
                residuals[i] = np.sum(np.abs(intens - y_obs))
            elif self.first_pass_settings["residuals_method"] == 'abs_cubed':
                residuals[i] = np.sum(np.abs(intens - y_obs)**3)
            else:
                raise ValueError(f"Unknown residuals method: {self.first_pass_settings['residuals_method']}")

            # Penalty function for OOB particles
            if np.any(p < bounds[0]) or np.any(p > bounds[1]):
                residuals[i] += self.first_pass_settings["oob_penalty"]

            # Penalty for OOB streaks
            front = self.movePickToEdge((p[2:4]), omega, p[4], directions=directions)
            back = self.movePickToEdge((p[2:4]), omega, -p[4], directions=directions)

            if (front[0] < 0 or front[0] >= data_tuple[0].shape[1] or 
                    front[1] < 0 or front[1] >= data_tuple[0].shape[0]):
                residuals[i] += self.first_pass_settings["oob_penalty"]
            if (back[0] < 0 or back[0] >= data_tuple[0].shape[1] or 
                    back[1] < 0 or back[1] >= data_tuple[0].shape[0]):
                residuals[i] += self.first_pass_settings["oob_penalty"]

        return residuals
    

    def localObjectiveFunction(self, params, data_tuple, y_obs, a0, omega, bounds, directions):
        """
        Local optimizer objective: residual for a single parameter vector.

        Evaluates residuals (abs, squared, or cubed) and applies parameter
        bounds penalties and crop out-of-bounds penalties for the streak’s ends.

        Args:
            params (np.ndarray): [level_sum, sigma, x0, y0, length], shape (5,).
            data_tuple (tuple[np.ndarray, np.ndarray]): (x_index, y_index) grids.
            y_obs (np.ndarray): Observed crop flattened, shape (h*w,).
            a0 (float): Background level (0 in current calls).
            omega (float): Track angle [rad].
            bounds (Sequence[tuple[float, float]]): L-BFGS-B bounds.
            directions (tuple[int, int]): Direction multipliers (+1/-1) for x/y.

        Returns:
            float: Objective value.

        Raises:
            ValueError: On unknown residual method.
        """

            
        # Unpack parameters
        level_sum, sigma, x0, y0, length = params
        
        # Calculate intensity of the moving gaussian
        intens = self.movingGaussian(data_tuple, omega, a0, level_sum, sigma, x0, y0, length)

        # Calculate the residuals based on the specified method
        if self.second_pass_settings["residuals_method"] == 'abs_squared':
            residuals = np.sum(np.abs(intens - y_obs)**2)
        elif self.second_pass_settings["residuals_method"] == 'abs':
            residuals = np.sum(np.abs(intens - y_obs))
        elif self.second_pass_settings["residuals_method"] == 'abs_cubed':
            residuals = np.sum(np.abs(intens - y_obs)**3)
        else:
            raise ValueError(f"Unknown residuals method: {self.second_pass_settings['residuals_method']}")

        # Penalty function for OOB particles
        for i in range(len(params)):
            if params[i] < bounds[i][0] or params[i] > bounds[i][1]:
                residuals += self.second_pass_settings["oob_penalty"]

        # Penalty for OOB streaks
        front = self.movePickToEdge((params[2:4]), omega, params[4], directions=directions)
        back = self.movePickToEdge((params[2:4]), omega, -params[4], directions=directions)

        if (front[0] < 0 or front[0] >= data_tuple[0].shape[1] or 
                front[1] < 0 or front[1] >= data_tuple[0].shape[0]):
            residuals += self.second_pass_settings["oob_penalty"]
        if (back[0] < 0 or back[0] >= data_tuple[0].shape[1] or 
                back[1] < 0 or back[1] >= data_tuple[0].shape[0]):
            residuals += self.second_pass_settings["oob_penalty"]

        return residuals


    def cropFrameToGaussian(self, sub_frame, est_global_center, max_sigma, max_length, omega, 
                            cropping_settings=None):
        """
        Crop a frame tightly around a maximal moving-Gaussian envelope.

        Generates a binary mask from a high-flux, long Gaussian envelope centered on
        `est_global_center` with provided maximum sigma/length, zeros out background,
        and returns the minimal rectangular crop bounding the non-zero region.

        Args:
            sub_frame (np.ndarray): Background-subtracted frame, shape (H, W).
            est_global_center (tuple[float, float]): Estimated global center (x, y).
            max_sigma (float): Max sigma for envelope (scaled by config).
            max_length (float): Max length for envelope (scaled by config).
            omega (float): Track angle [rad].
            cropping_settings (dict | None): Optional override dict containing
                'max_sigma_coeff' and 'max_length_coeff'.

        Returns:
            tuple:
                cropped_frame (np.ndarray): Crop image, shape (h, w).
                crop_vars (list[float]): [cx, cy, xmin, xmax, ymin, ymax].

        Raises:
            ValueError: If mask has no non-zero region (degenerate crop).
        """


        # Determine maximum size values
        if cropping_settings is None:
            max_sigma = max_sigma * self.cropping_settings['max_sigma_coeff']
            max_length = max_length * self.cropping_settings['max_length_coeff']
        else:
            max_sigma = max_sigma * cropping_settings['max_sigma_coeff']
            max_length = max_length * cropping_settings['max_length_coeff']

        # Unpack other values
        y, x = np.indices(sub_frame.shape)
        data_tuple = (x, y)

        # Generate maximal gaussian mask for the optimizer to work within
        optim_mask = self.movingGaussian(data_tuple, omega, 0, 1e6, max_sigma, 
                                         est_global_center[0], est_global_center[1], 
                                         max_length, saturation_level=None
                                         )
        optim_mask = optim_mask.reshape(sub_frame.shape)

        # Clip the mask to binary values
        optim_mask[optim_mask > 1e-3] = 1
        optim_mask[optim_mask <= 1e-3] = 0

        # get bounds of non-zero values, and crop sub_frame to these bounds
        non_zero_indices = np.nonzero(optim_mask)
        
        # set the sub frame to zero where the mask is zero
        sub_frame[optim_mask == 0] = 0

        # Crop the sub_frame to the non-zero indices (plus one for indexing start/stop properly)
        xmin = int(np.min(non_zero_indices[1]))
        xmax = int(np.max(non_zero_indices[1]) + 1)
        ymin = int(np.min(non_zero_indices[0]))
        ymax = int(np.max(non_zero_indices[0]) + 1)

        # Crop the sub_frame to the bounds
        cropped_frame = sub_frame[ymin:ymax, xmin:xmax]

        # Store the crop vars

        crop_vars = [est_global_center[0] - xmin, #cx
                     est_global_center[1] - ymin, #cy
                     xmin, xmax, ymin, ymax]
        
        return cropped_frame, crop_vars


    def movingGaussian(self, data_tuple, omega, a0, level_sum, sigma, x0, y0, L, saturation_level=None):
        """ Moving Gaussian function with saturation intensity limiting.

        Based on:
            Peter Veres, Robert Jedicke, Larry Denneau, Richard Wainscoat, Matthew J. Holman and Hsing-Wen Lin
            Publications of the Astronomical Society of the Pacific
            Vol. 124, No. 921 (November 2012), pp. 1197-1207 

            The original equation given in the paper has a typo in the exp term, after sin(omega) there 
            should be a minus, not a plus.


        Arguments:
            data_tuple: [tuple]
                - x: [ndarray] Array of X image coordinates.
                - y: [ndarray] Array of Y image coordiantes.
            a0: [float] Background level.
            level_sum: [float] Total flux of the Gaussian.
            sigma: [float] Standard deviation.
            x0: [float] X coordinate of the centre of the track.
            y0: [float] Y coordinate of the centre of the track.
            L: [float] Length of the track.
            omega: [float] Angle of the track.

        Keyword arguments:
            saturation_level: [float] Level of saturation. None by default.

        """

        x, y = data_tuple

        # Rotate the coordinates
        x_m = (x - x0)*np.cos(omega) - (y - y0)*np.sin(omega)
        y_m = (x - x0)*np.sin(omega) + (y - y0)*np.cos(omega)


        u1 = (x_m + L/2.0)/(sigma*np.sqrt(2))
        u2 = (x_m - L/2.0)/(sigma*np.sqrt(2))

        f1 = scipy.special.erf(u1) - scipy.special.erf(u2)

        # Evaluate the intensity at every pixel
        intens = a0 + level_sum/(2*sigma*np.sqrt(2*np.pi)*L)*np.exp(-y_m**2/(2*sigma**2))*f1


        # Limit intensity values to the given saturation limit
        if saturation_level is not None:
            intens[intens > saturation_level] = saturation_level

        return intens.ravel()


    def generateInitialParticles(self, bounds, n_particles, p0=None):
        """
        Generate PSO initial particles within bounds.

        If `p0` is provided, draws each dimension from a truncated normal centered at
        `p0` with sigma scaled by the distance to bounds and an explorative coefficient;
        otherwise, samples uniformly within [lb, ub].

        Args:
            bounds (tuple[np.ndarray, np.ndarray]): (lb, ub), each shape (D,).
            n_particles (int): Number of particles.
            p0 (np.ndarray | None): Optional center, shape (D,).

        Returns:
            np.ndarray: Particle positions, shape (n_particles, D).
        """


        # Unpack bounds & explorative coefficient
        explorative_coefficient = self.first_pass_settings['explorative_coeff']
        lb, ub = bounds

        # Normally disperse particles around center if p0 is not None
        if p0 is not None:

            D = len(lb)
            pos = np.empty((n_particles, D))

            # 1) Compute a “natural” sigma for each dimension:
            dist_to_lower = p0 - lb
            dist_to_upper = ub - p0
            sigma = np.minimum(dist_to_lower, dist_to_upper) / explorative_coefficient

            # 2) But make sure sigma isn't vanishingly small:
            min_sigma = (ub - lb) / (explorative_coefficient * 10)
            sigma = np.maximum(sigma, min_sigma)

            # 3) Build the standardized bounds a, b for truncnorm
            a = (lb - p0) / sigma
            b = (ub - p0) / sigma

            # 4) Draw each dim from its 1D truncated normal
            for i in range(D):
                pos[:, i] = scipy.stats.truncnorm.rvs(
                    a[i], b[i],
                    loc=p0[i], scale=sigma[i],
                    size=n_particles
                )
            
        # Return a uniformly distributed particles if p0 is None
        else:
            # Otherwise, generates particles uniformly within the bounds
            pos = np.random.uniform(low=lb, high=ub, size=(n_particles, len(lb)))

        # Return the generated particles
        return pos

    def getFrame(self, fr_no, include_raw=False, crop_vars=None, include_non_subtracted=False):
        """
        Loads the frame number from the img_obj and returns the subtracted, and cropped if vars are passed on
        """

        # Adjust to total frame index
        fr_no = [x + self.first_pick_global_index for x in fr_no] if isinstance(fr_no, (list, np.ndarray)) \
            else fr_no + self.first_pick_global_index

        # Operate on list of frames
        if isinstance(fr_no, (list, np.ndarray)):

            frames, raw_frames, non_sub_frames = [], [], []

            first_frame_num = self.img_obj.current_frame

            for fr in fr_no:
                
                # Load frame
                self.img_obj.setFrame(fr)
                frame = self.img_obj.loadFrame().astype(np.float32)

                # Store raw frame if include_raw
                raw_frame = copy.deepcopy(frame) if include_raw else None

                # Correct frame
                if include_non_subtracted:
                    frame, non_sub_frame = self.correctFrame(frame, include_non_subtracted=True)
                    non_sub_frames.append(non_sub_frame)
                else:
                    frame = self.correctFrame(frame)

                # Crop if requested
                if crop_vars is not None:
                    cx, cy, xmin, xmax, ymin, ymax = crop_vars
                    frame = frame[ymin:ymax, xmin:xmax]

                frames.append(frame)
                raw_frames.append(raw_frame)


            # Reset to first frame
            self.img_obj.setFrame(first_frame_num)

            if include_raw is False and include_non_subtracted is False:
                return np.array(frames)
            elif include_raw is True and include_non_subtracted is False:
                return np.array(frames), np.array(raw_frames)
            elif include_raw is False and include_non_subtracted is True:
                return np.array(frames), np.array(non_sub_frames)
            elif include_raw is True and include_non_subtracted is True:
                return np.array(frames), np.array(raw_frames), np.array(non_sub_frames)

        # Operate on a single frame
        else:
        
            frame = None
            raw_frame = None
            non_sub_frame = None

            # Get initial frame num
            first_frame_num = self.img_obj.current_frame

            # Set and load fr_no
            self.img_obj.setFrame(fr_no)
            frame = self.img_obj.loadFrame().astype(np.float32)

            # Reset to the initial frame
            self.img_obj.setFrame(first_frame_num)

            # Store raw frame if include_raw
            raw_frame = copy.deepcopy(frame) if include_raw else None

            # Correct frame
            if include_non_subtracted:
                frame, non_sub_frame = self.correctFrame(frame, include_non_subtracted=True)
            else:
                frame = self.correctFrame(frame)

            # Crop if requested
            if crop_vars is not None:
                cx, cy, xmin, xmax, ymin, ymax = crop_vars
                frame = frame[ymin:ymax, xmin:xmax]

            if include_raw is False and include_non_subtracted is False:
                return np.array(frame)
            elif include_raw is True and include_non_subtracted is False:
                return np.array(frame), np.array(raw_frame)
            elif include_raw is False and include_non_subtracted is True:
                return np.array(frame), np.array(non_sub_frame)
            elif include_raw is True and include_non_subtracted is True:
                return np.array(frame), np.array(raw_frame), np.array(non_sub_frame)


    def correctFrame(self, frame, include_non_subtracted=False):

        # 1. correct using dark, flat, gamma
        corr_frame = frame.copy()
        
        if self.dark is not None:
            corr_frame = Image.applyDark(corr_frame, self.dark)

        if self.flat_struct is not None:
            corr_frame = Image.applyFlat(corr_frame, self.flat_struct)
        
        if self.dark is not None or self.flat_struct is not None:
            corr_frame = Image.gammaCorrectionImage(corr_frame, self.config.gamma,
                                                    bp=0, wp=(2**self.config.bit_depth - 1),
                                                    out_type=np.float32)
        else:
            corr_frame = Image.gammaCorrectionImage(corr_frame, self.config.gamma,
                                                    bp=0, wp=(2**self.config.bit_depth - 1),
                                                    out_type=np.float32)

        # 2. Background subtraction
        unsub_frame = corr_frame.copy()
        sub_frame = corr_frame.astype(np.int32) - self.corrected_avepixel.astype(np.int32)

        # 3. Star masking
        final_frame = np.ma.masked_array(sub_frame, mask=self.star_mask)

        # Return the frame
        if include_non_subtracted:
            return final_frame, unsub_frame
        else:
            return final_frame

    def recursiveCroppingAlgorithm(self, frame_index, est_center_global, paramter_estimation_functions, omega, 
                                   directions, forward_pass=False):
        """
        Recursive forward/backward pass to crop and fit subsequent frames.

        Uses parameter predictors to estimate the next crop window and magnitudes,
        runs PSO on that crop, appends/inserts the result according to pass direction,
        updates predictors, projects the next center, and recurses until indices run
        out of event bounds.

        Args:
            frame_index (int): Index of the frame to process next.
            est_center_global (tuple[float, float]): Estimated global center (x, y).
            paramter_estimation_functions (dict): Predictors from
                `updateParameterEstimationFunctions`.
            omega (float): Track angle [rad].
            directions (tuple[int, int]): Direction multipliers (+1/-1) for x/y.
            forward_pass (bool): Direction flag; True for forward, False for backward.

        Returns:
            None
        """

        # If the frame is outside the event bounds, quit cropping
        if frame_index > max(self.pick_frame_indices) or frame_index < min(self.pick_frame_indices):
            return

        # Estimate next parameters using the parameter estimation functions
        est_next_params = self.estimateNextParameters(paramter_estimation_functions, 
                                                      self.first_pass_params.shape[0], 
                                                      forward_pass=forward_pass
                                                      )

        # Crop the frame around the new center
        cropped_frame, crop_vars = self.cropFrameToGaussian(self.getFrame(frame_index), 
                                    est_center_global, 
                                    est_next_params['height'] * self.cropping_settings['max_sigma_coeff'], 
                                    est_next_params['length'] * self.cropping_settings['max_length_coeff'],
                                    omega
                                    )

        # Run a PSO on the cropped frame
        best_fit, _ = self.gaussianPSO(cropped_frame, omega, directions, estim_next_params=est_next_params)

        # If forward pass add new parameters to the end of array to maintain symmetry with pick_frame_indicies
        if forward_pass:
            
            self.cropped_frames.append(cropped_frame)
            self.crop_vars = np.vstack([self.crop_vars, crop_vars])
            self.first_pass_params = np.vstack([self.first_pass_params, best_fit])

            # If the frame number is not in pick_frame_indices, add it
            if frame_index not in self.pick_frame_indices:
                self.pick_frame_indices.append(frame_index)
                self.pick_frame_indices = sorted(self.pick_frame_indices)

        # If backwards pass add new parameters to the start of the array to maintain symmetry
        else:

            self.cropped_frames.insert(0,cropped_frame)
            self.crop_vars = np.vstack([crop_vars, self.crop_vars])
            self.first_pass_params = np.vstack([best_fit, self.first_pass_params])
            
            # If the frame number is not in pick_frame_indices, add it
            if frame_index not in self.pick_frame_indices:
                self.pick_frame_indices.insert(0, frame_index)
                self.pick_frame_indices = sorted(self.pick_frame_indices)

        # Update estimation functions off new parameters
        parameter_estimation_functions = self.updateParameterEstimationFunctions(
            self.crop_vars, self.first_pass_params, forward_pass=forward_pass)

        # Translate fit center to global coordinates
        global_best_fit_center = self.translatePicksToGlobal(
            (best_fit[2], best_fit[3]),
            crop_vars
        )

        # Estimate the next frame meteor center
        next_center_global = self.estimateNextCenter(
            global_best_fit_center,
            est_next_params['norm'],
            omega,
            directions=directions
        )

        # Set the pass coeff (index step) based on forward pass or not
        pass_coeff = 1 if forward_pass else -1

        # Update progress
        self.progressed_frames['cropping'] += 1
        self.updateProgress()

        # Verbose print
        if self.verbose:
            print(f"Recursive cropping at frame {frame_index} with center {est_center_global} "
                  f"and next center {next_center_global}, Forward pass: {forward_pass}")

        # Recurse with next index
        self.recursiveCroppingAlgorithm(frame_index + pass_coeff,
                                        next_center_global,
                                        parameter_estimation_functions,
                                        omega,
                                        directions=directions,
                                        forward_pass=forward_pass,
                                        )
    
    def updateProgress(self, progress=None):
        """
        Calculates approx. progress based on either Gaussian or Kalman mode. Updates the progress callback.
        The progress is calculated based on the number of frames processed in each step and total num frames.
        The weights for each step are defined in the time_weights_gaus dictionary.
        """

        # Weight the different phases of the program by differeing weights
        time_weights_gaus = {
            'cropping' : 0.8,
            'refining' : 0.1,
            'removing' : 0.1
        }

        # If available set progress to callback
        if progress is not None and self.progress_callback is not None:
            self.progress_callback(int(progress))

        elif self.progress_callback is not None:
            current_percentage = sum(
                self.progressed_frames[step] * time_weights_gaus[step]
                for step in self.progressed_frames.keys()
            ) / self.total_frames * 100
            self.progress_callback(int(current_percentage))

        # Else print callback to console
        if self.progress_callback is None:
            if progress is not None:
                progress_bar = '*' * int(progress) + '-' * (100 - int(progress))
                print(f'Progress: {progress_bar} : {int(progress)}%')
            else:
                current_percentage = sum(
                    self.progressed_frames[step] * time_weights_gaus[step]
                    for step in self.progressed_frames.keys()
                ) / self.total_frames * 100
                progress_bar = '*' * int(current_percentage) + '-' * (100 - int(current_percentage))
                print(f'{self.config.stationID} Progress: {progress_bar} : {int(current_percentage)}%')

    def selectSeedTriplet(self, picks, pick_frame_indices):
        """
        Select a consecutive seed triplet of picks/frames to initialize recursion.

        Sorts picks by frame index, finds indices `i` such that frames
        [k[i], k[i+1], k[i+2]] are consecutive, prefers triplets not touching the
        sequence ends, then those whose middle frame is closest to the sequence center,
        and finally earliest start as a tiebreaker.

        Args:
            picks (array-like): (M, 2) float array of (x, y) picks.
            pick_frame_indices (array-like): (M,) integer frame indices.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                seed_picks: (3, 2) picks for the chosen triplet.
                seed_pick_frame_indices: (3,) frame indices for the triplet.

        Raises:
            ValueError: If fewer than 3 points or no consecutive triple is found,
                or if shapes are inconsistent.
        """

        # Convert & validate
        p = np.asarray(picks)
        f = np.asarray(pick_frame_indices)

        # Auto-swap if args were passed in reverse
        if f.ndim != 1 and p.ndim == 1:
            p, f = f, p

        if f.ndim != 1:
            raise ValueError(f"pick_frame_indices must be 1D; got shape {f.shape}")
        if p.shape[0] != f.shape[0]:
            raise ValueError(f"Length mismatch: picks {p.shape[0]} vs frames {f.shape[0]}")

        # Sort by frame index (keep picks aligned)
        f = f.astype(np.int64, copy=False)
        order = np.argsort(f)
        keys = f[order]
        p_sorted = p[order]

        if keys.size < 3:
            raise ValueError("Need at least 3 points to form a consecutive triple.")

        # Find starts i where [k[i], k[i+1], k[i+2]] are consecutive
        mask = (keys[:-2] + 1 == keys[1:-1]) & (keys[:-2] + 2 == keys[2:])
        starts = np.where(mask)[0]
        if starts.size == 0:
            raise ValueError("No triple of consecutive frames found.")

        # Preference:
        # 1) prefer triples NOT touching ends
        touch_end = (starts == 0) | (starts + 2 == len(keys) - 1)   # True if touches first/last
        pref1 = touch_end.astype(np.int64)                           # 0 is better than 1

        # 2) then prefer middle frame closest to center of [keys[0], keys[-1]]
        center = 0.5 * (keys[0] + keys[-1])
        middle_frames = keys[starts + 1]
        dist = np.abs(middle_frames - center).astype(np.float64)

        # 3) stable tie-breaker: earliest start
        idx = np.lexsort((starts, dist, pref1))[0]
        start = starts[idx]

        seed_pick_frame_indices = keys[start:start+3]
        seed_picks = p_sorted[start:start+3]

        # Return seed_picks and indices (do not need to be stored in memory)
        return seed_picks, seed_pick_frame_indices

    def computeIntensitySum(self, photom_pixels, global_centroid, corr_frame, uncorr_frame, unsub_frame):
        
        # Get photometry pixels as as an array of x_indices, and y_indices
        photom_x_indices, photom_y_indices = np.array(photom_pixels).T


        # Store a copy of the corrected frame without star_mask to avoid reference errors
        corrected_frame = np.asarray(corr_frame.data).astype(np.float32, copy=True)

        # Store a copy of the uncorrected frame without star_mask to avoid reference errors
        uncorrected_frame = np.asarray(uncorr_frame).astype(np.float32, copy=True)

        # Store a copy of the corrected avepixel to avoid reference errors
        corrected_avepixel = np.asarray(self.corrected_avepixel).astype(np.float32, copy=True)

        # Store a copy of the star mask to avoid reference errors
        star_mask = np.asarray(self.star_mask).astype(bool, copy=True)

        # Store a copy of the unsub frame to avoid reference errors
        unsubtracted_frame = np.asarray(unsub_frame).astype(np.float32, copy=True)

        # Define a crop window as twice the size of the colored pixels
        x_color_size = np.max(photom_x_indices) - np.min(photom_x_indices)
        y_color_size = np.max(photom_y_indices) - np.min(photom_y_indices)

        xmin = int(global_centroid[0] - x_color_size)
        xmax = int(global_centroid[0] + x_color_size)
        ymin = int(global_centroid[1] - y_color_size)
        ymax = int(global_centroid[1] + y_color_size)

        # Limit the size to be within the bounds

        H, W = corr_frame.shape

        if xmin < 0: xmin = 0
        if xmax > W: xmax = W
        if ymin < 0: ymin = 0
        if ymax > H: ymax = H

        # Get cropped versions of all arrays
        cropped_corrected_frame = corrected_frame[ymin:ymax, xmin:xmax]
        cropped_uncorrected_frame = uncorrected_frame[ymin:ymax, xmin:xmax]
        cropped_corrected_avepixel = corrected_avepixel[ymin:ymax, xmin:xmax]
        cropped_star_mask = star_mask[ymin:ymax, xmin:xmax]
        cropped_unsubtracted_frame = unsubtracted_frame[ymin:ymax, xmin:xmax]
        photom_x_indices = photom_x_indices - xmin
        photom_y_indices = photom_y_indices - ymin

        # Create combined masks

        # Generate mask for photometry pixels, cropped
        cropped_mask_photom_included = np.ones_like(cropped_corrected_frame, dtype=bool) 
        cropped_mask_photom_included[photom_y_indices, photom_x_indices] = False # Unmask photometry pixels

        # Generate mask excluding photometry pixels, cropped
        cropped_mask_photom_excluded = np.zeros_like(cropped_corrected_frame, dtype=bool) 
        cropped_mask_photom_excluded[photom_y_indices, photom_x_indices] = True # Mask photometry pixels

        # 1) Compute intensity sum

        # Get array of cropped photometric area from the corrected frame
        photom_pixels_nobg = np.ma.masked_array(cropped_corrected_frame, 
                                                                    mask=cropped_mask_photom_included,
                                                                    copy=True)
        
        # Replace photometry pixels that are masked by a star with the median value of the photom. area

        # Check where the photometric pixels intersect with the star mask
        photom_star_masked_indices = np.where((photom_pixels_nobg.mask == 0) & (cropped_star_mask == 1))

        # Add the star mask
        photom_pixels_nobg.mask = (photom_pixels_nobg.mask | cropped_star_mask).copy()   

        # Apply correction only if the streak intersects a star
        if len(photom_star_masked_indices[0]) > 0:
            # Calulate masked median
            masked_stars_streak_median = np.ma.median(photom_pixels_nobg)

            # Unmask those areas
            photom_pixels_nobg.mask[photom_star_masked_indices] = False

            # Replace with median
            photom_pixels_nobg[photom_star_masked_indices] = masked_stars_streak_median

        # Sum the array of corrected photometric pixels with the background subtracted
        intensity_sum = np.abs(np.ma.sum(photom_pixels_nobg))

        # 2) Compute background STD

        # Get corrected cropped frame with photometry pixels & stars masked out
        photom_excluded_sm_cropped_corrected_frame = np.ma.masked_array(cropped_unsubtracted_frame, 
                                                    mask=cropped_mask_photom_excluded | cropped_star_mask,
                                                    copy=True)
        
        # Compute the standard deviation of the background
        background_stddev = np.ma.std(photom_excluded_sm_cropped_corrected_frame)

        # 3) Compute background level
        
        # Compute background level using previous masked frame
        background_lvl = np.ma.median(photom_excluded_sm_cropped_corrected_frame)

        # 4) Compute source pixel count

        source_px_count = np.ma.sum(~photom_pixels_nobg.mask) # Count of unmasked pixels in photometric area

        # 5) Check for saturation

        # Get cropped photometric area from the uncorrected frame
        photom_included_cropped_uncorrected_frame = np.ma.masked_array(cropped_uncorrected_frame, 
                                                                      mask=cropped_mask_photom_included,
                                                                      copy=True)
        
        # If at least 2 pixels are saturated in the photometric area, mark the pick as saturated
        if np.sum(photom_included_cropped_uncorrected_frame > self.saturation_threshold) >= 2:
            saturated_bool = True
        else:
            saturated_bool = False
        
        # Add all to class variable lists
        self.photometry_pixels.append(photom_pixels)
        self.saturated_bool_list.append(saturated_bool)
        self.abs_level_sums.append(intensity_sum)
        self.background_levels.append(background_lvl)

        # Compute SNR using CCD equation
        snr = signalToNoise(intensity_sum, source_px_count, background_lvl, background_stddev)

        # DEBUG - Save diagnostic photometry images
        # try:
        #     # Save the current matplotlib backend and switch to Agg
        #     current_backend = matplotlib.get_backend()
        #     matplotlib.use('Agg')

        #     # Create directory for photometry diagnostics if it doesn't exist
        #     photom_dir = os.path.join(self.data_path, "ASTRA_Photometry_Diagnostics")
        #     os.makedirs(photom_dir, exist_ok=True)

        #     # Get frame number for filename
        #     frame_number = self.pick_frame_indices[len(self.photometry_pixels)-1] + self.first_pick_global_index

        #     # Create a figure with a 3x2 grid - use Figure directly for thread safety
        #     fig = Figure(figsize=(15, 10))
        #     canvas = FigureCanvas(fig)
        #     grid = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        #     # Create all needed axes
        #     ax1 = fig.add_subplot(grid[0, 0])  # Original cropped frame
        #     ax2 = fig.add_subplot(grid[0, 1])  # Background subtracted
        #     ax3 = fig.add_subplot(grid[1, 0])  # Star mask
        #     ax4 = fig.add_subplot(grid[1, 1])  # Photometry mask
        #     ax5 = fig.add_subplot(grid[2, 0])  # Final photometry pixels
        #     ax6 = fig.add_subplot(grid[2, 1])  # Combined result

        #     # Show cropped_corrected_frame
        #     vmin1 = np.percentile(cropped_corrected_frame, 1)
        #     vmax1 = np.percentile(cropped_corrected_frame, 99)
        #     im1 = ax1.imshow(cropped_corrected_frame, cmap='gray', vmin=vmin1, vmax=vmax1)
        #     ax1.set_title("cropped_corrected_frame")
        #     fig.colorbar(im1, ax=ax1, shrink=0.7)

        #     # Show photometry pixels with background subtracted
        #     filled_nobg = photom_pixels_nobg.filled(0)
        #     vmax2 = np.percentile(filled_nobg, 99)
        #     im2 = ax2.imshow(filled_nobg, cmap='gray', vmin=0, vmax=vmax2)
        #     ax2.set_title("photom_pixels_nobg")
        #     fig.colorbar(im2, ax=ax2, shrink=0.7)

        #     # Show star mask
        #     star_mask_viz = np.ones_like(cropped_star_mask)
        #     im3 = ax3.imshow(cropped_star_mask, cmap='Reds', vmin=0, vmax=1)
        #     ax3.set_title("cropped_star_mask")

        #     # Show photometry mask (included and excluded)
        #     im4 = ax4.imshow(cropped_mask_photom_excluded, cmap='Blues', vmin=0, vmax=1)
        #     ax4.set_title("cropped_mask_photom_excluded")

        #     # Visualization for final photometry pixels 
        #     phot_final = np.zeros_like(cropped_corrected_frame)
        #     for p in photom_pixels:
        #         # Convert to local coordinates
        #         px, py = p[0] - xmin, p[1] - ymin
        #         # Check if within crop bounds
        #         if 0 <= px < phot_final.shape[1] and 0 <= py < phot_final.shape[0]:
        #             phot_final[py, px] = 1

        #     im5 = ax5.imshow(phot_final, cmap='viridis', vmin=0, vmax=1)
        #     ax5.set_title("photometry_pixels")

        #     # Combined visualization - frame with photometry overlay
        #     combined = np.zeros((*cropped_corrected_frame.shape, 3))
        #     # Grayscale background
        #     normalized = np.clip(cropped_corrected_frame / (vmax1 + 0.01), 0, 1)
        #     for i in range(3):
        #         combined[:,:,i] = normalized

        #     # Add red overlay for photometry pixels
        #     for p in photom_pixels:
        #         px, py = p[0] - xmin, p[1] - ymin
        #         if 0 <= px < combined.shape[1] and 0 <= py < combined.shape[0]:
        #             combined[py, px, 0] = 1.0  # Red channel
        #             combined[py, px, 1] = 0.3  # Green channel
        #             combined[py, px, 2] = 0.3  # Blue channel

        #     # Add blue overlay for stars
        #     for y in range(cropped_star_mask.shape[0]):
        #         for x in range(cropped_star_mask.shape[1]):
        #             if cropped_star_mask[y, x]:
        #                 combined[y, x, 0] = 0.3  # Red channel
        #                 combined[y, x, 1] = 0.3  # Green channel
        #                 combined[y, x, 2] = 1.0  # Blue channel

        #     im6 = ax6.imshow(combined)
        #     ax6.set_title("combined_visualization")

        #     # Add summary stats as figure title
        #     fig.suptitle(f"Frame {frame_number} - SNR: {snr:.2f}, Sum: {intensity_sum:.0f}, " + 
        #         f"Bg: {background_lvl:.2f}, Pixels: {source_px_count}, " +
        #         f"Saturated: {saturated_bool}", fontsize=12)

        #     # Save the figure in a thread-safe way
        #     if not hasattr(self, "_plot_lock"):
        #         self._plot_lock = threading.RLock()

        #     with self._plot_lock:
        #         fig_path = os.path.join(photom_dir, f"photometry_frame_{frame_number:04d}.jpg")
        #         fig.savefig(fig_path, format='jpg', dpi=100, bbox_inches='tight', 
        #             pil_kwargs={"quality": 90, "optimize": True})

        #     # Explicitly close to free memory
        #     fig.clf()

        #     # Restore the original matplotlib backend
        #     matplotlib.use(current_backend)

        # except Exception as e:
        #     print(f"Warning: Could not save photometry diagnostic plot for frame {frame_number}: {e}")

        # Verbose print
        if self.verbose:
            print("ASTRA SNR update on frame {:2d}: intensity sum = {:8.1f}, source px count = {:5d}, " \
            "background lvl = {:8.2f}, background stddev = {:6.2f}, SNR = {:.2f}".format(
                self.pick_frame_indices[self.temp_count]+self.first_pick_global_index, 
                float(intensity_sum), source_px_count, 
                background_lvl, background_stddev, snr))
            self.temp_count += 1

        # Return SNR - returned rather than state-set
        return snr  

    def computePhotometryPixels(self, fit_img, cropped_frame, crop_vars):
        """
        Derive photometry pixels by thresholding the fitted image and cleaning morphology.

        Creates a binary mask from the fitted image, thresholds by a configured
        percentile over the masked crop, runs a morphological close to fill gaps,
        and returns the non-zero locations in global (x, y) order.

        Args:
            fit_img (np.ndarray): Fitted image for the crop, shape (h, w).
            cropped_frame (np.ndarray): Crop image, shape (h, w).
            crop_vars (Sequence[int | float]): [cx, cy, xmin, xmax, ymin, ymax].

        Returns:
            list[tuple[int, int]]: Global (x, y) coordinates of photometry pixels.
        """

        # Round crop variables to integers for indexing
        _, _, x_min, x_max, y_min, y_max = map(int, crop_vars.copy())

        # Copy over variables to avoid refference errors
        fit_img = fit_img.copy()
        cropped_frame = cropped_frame.copy()

        # Clip fit image to zero and one
        fit_img[fit_img <= 1] = 0
        fit_img[fit_img > 1] = 1

        # Mask cropped frame with fit image to remove the background
        masked_cropped = fit_img * cropped_frame

        masked_cropped[masked_cropped < np.nanpercentile(masked_cropped, 
                                                    float(self.astra_config['astra']['photom_thresh'])*100)
                                                    ] = 0

        # binarize mask_cropped
        masked_cropped[masked_cropped > 0] = 1
        masked_cropped[masked_cropped <= 0] = 0

        # Use morphological operator to close up photometry pixels (complete holes etc)
        kernel = np.ones((3, 3), np.uint8)

        masked_cropped = cv2.morphologyEx(masked_cropped, cv2.MORPH_CLOSE, kernel)

        # Get indices for all non-zero pixels
        nonzero_indices = np.argwhere(masked_cropped > 0)

        # Convert photometry pixels to global coordinates and adjust indices
        nonzero_indices[:, 0] += y_min
        nonzero_indices[:, 1] += x_min
        photometry_pixels = [tuple(idx[::-1]) for idx in nonzero_indices]

        return photometry_pixels

    def refactorPicksToSkyFitFormat(self):

        self.pick_list = {}

        for i, frame_index in enumerate(self.pick_frame_indices):
            self.pick_list[frame_index + self.first_pick_global_index] = {
                "x_centroid" : self.global_picks[i][0],
                "y_centroid" : self.global_picks[i][1],
                "mode" : 1, # Default to mode 1
                "intensity_sum" : self.abs_level_sums[i],
                "photometry_pixels" : self.photometry_pixels[i],
                "background_intensity" : self.background_levels[i],
                "snr" : self.snr[i],
                "saturated" : self.saturated_bool_list[i]
            }

    
#  --- Interfacing Functions --- 

    def process(self):
        """
        Run the full ASTRA pipeline on the provided frames/picks.

        Pipeline:
            1) Preprocess frames (dark/flat correction, gamma, background subtraction,
            star masking) via `processImageData`.
            2) Recursively crop and fit a moving Gaussian track across frames using PSO
            via `cropAllMeteorFrames`.
            3) Locally refine each PSO solution with L-BFGS-B via `refineAllMeteorCrops`.
            4) Remove low-SNR or out-of-bounds fits and convert to global coords
            via `removeLowSNRPicks`.
            5) Optionally save a diagnostic animation via `saveAni`.

        Returns:
            ASTRA: The same instance with the following key fields populated:
                - avepixel_background (np.ndarray): Background model, shape (H, W).
                - subtracted_frames (np.ma.MaskedArray): BG-subtracted & star-masked frames, shape (N, H, W).
                - cropped_frames (list[np.ndarray]): Per-pick crops for fitting.
                - first_pass_params (np.ndarray): PSO fit params per crop, shape (K, 5).
                - refined_fit_params (np.ndarray): Local-refined params per crop, shape (K, 5).
                - crop_vars (np.ndarray): Crop bookkeeping (cx, cy, xmin, xmax, ymin, ymax), shape (K, 6).
                - global_picks (np.ndarray): Edge-aligned global picks, shape (K, 2).
                - pick_frame_indices (np.ndarray): Frame index per kept crop, shape (K,).
                - fit_costs (np.ndarray): Objective values per refined crop, shape (K,).
                - times (np.ndarray): Timestamps per kept crop, shape (K,).
                - snr (list[float]): SNR per kept crop (post filtering).

        Raises:
            Exception: If any individual stage throws; errors are caught/logged per step,
                but uncaught exceptions will propagate.
        """


        # 1. Gets corrected background and star_mask for later corrections 
        self.processImageData()

        # 2. Recursively crop & fit a moving gaussian across whole event
        self.cropAllMeteorFrames()

        # 3. Refine the moving gaussian fit by using a local optimizer
        self.refineAllMeteorCrops(self.first_pass_params, self.cropped_frames, 
                                      self.omega, self.directions)

        # 4. Remove picks with low SNR and out-of-bounds picks, refactors into global coordinates
        self.removeLowSNRPicks(self.refined_fit_params, self.fit_imgs, self.cropped_frames, 
                                    self.crop_vars, self.pick_frame_indices, self.fit_costs)

        # 5. save animation
        if self.save_animation:
            try:
                self.saveAni(self.data_path if self.ecsv_save_path is None else self.ecsv_save_path)
            except Exception as e:
                print(f'Error saving animation: {e}')

        # Set progress to 100
        self.updateProgress(100)

        # Refactor picks to SkyFit format before saving
        self.refactorPicksToSkyFitFormat()

        return True

    def saveECSV(self, platepar):
        """ Save the picks into the GDEF ECSV standard. """

        self.platepar = platepar

        # If no picks, save nothing and send no-picks to FTPDetectionInfo save
        if not hasattr(self, 'pick_list') or \
            not self.pick_list or all(val.get('x_centroid') is None for val in self.pick_list.values()):
            print("No valid picks to save.")
            return False

        isodate_format_file = "%Y-%m-%dT%H_%M_%S"
        isodate_format_entry = "%Y-%m-%dT%H:%M:%S.%f"

        # Reference time
        dt_ref = self.img_obj.beginning_datetime

        # ECSV file name
        ecsv_file_name = dt_ref.strftime(isodate_format_file) + '_ASTRA_' + self.config.stationID + ".ecsv"

        # Compute alt/az pointing
        azim, elev = trueRaDec2ApparentAltAz(self.platepar.RA_d, self.platepar.dec_d, self.platepar.JD,
            self.platepar.lat, self.platepar.lon, refraction=False)

        # Compute FOV size
        fov_horiz, fov_vert = computeFOVSize(self.platepar)

        if self.img_obj.input_type == 'ff':
            ff_name = self.img_obj.current_ff_file
        else:
            ff_name = "FF_{:s}_".format(self.platepar.station_code) \
                + self.img_obj.beginning_datetime.strftime("%Y%m%d_%H%M%S_") \
                + "{:03d}".format(int(self.img_obj.beginning_datetime.microsecond//1000)) \
                + "_0000000.fits"

        # Get the number of stars in the list
        if self.platepar.star_list is not None:
            n_stars = len(self.platepar.star_list)
        else:
            n_stars = 0

        # Write the meta header
        meta_dict = {
            'obs_latitude': self.platepar.lat,
            'obs_longitude': self.platepar.lon,
            'obs_elevation': self.platepar.elev,
            'origin': 'ASTRA',
            'camera_id': self.config.stationID,
            'cx' : self.platepar.X_res,
            'cy' : self.platepar.Y_res,
            'photometric_band' : self.platepar.mag_band_string if \
                hasattr(self.platepar, 'mag_band_string') else 'V',
            'image_file' : ff_name,
            'isodate_start_obs': str(dt_ref.strftime(isodate_format_entry)),
            'astrometry_number_stars' : n_stars,
            'mag_label': 'mag_data',
            'no_frags': 1,
            'obs_az': azim,
            'obs_ev': elev,
            'obs_rot': rotationWrtHorizon(self.platepar),
            'fov_horiz': fov_horiz,
            'fov_vert': fov_vert,
        }

        # Write the header
        out_str = """# %ECSV 0.9
# ---
# datatype:
# - {name: datetime, datatype: string}
# - {name: ra, unit: deg, datatype: float64}
# - {name: dec, unit: deg, datatype: float64}
# - {name: azimuth, datatype: float64}
# - {name: altitude, datatype: float64}
# - {name: x_image, unit: pix, datatype: float64}
# - {name: y_image, unit: pix, datatype: float64}
# - {name: integrated_pixel_value, datatype: int64}
# - {name: background_pixel_value, datatype: int64}
# - {name: saturated_pixels, datatype: bool}
# - {name: mag_data, datatype: float64}
# - {name: err_minus_mag, datatype: float64}
# - {name: err_plus_mag, datatype: float64}
# - {name: snr, datatype: float64}
# delimiter: ','
# meta: !!omap
"""
        # Add the meta information
        for key, value in meta_dict.items():
            value_str = f"'{value}'" if isinstance(value, str) else str(value)
            out_str += f"# - {{{key}: {value_str}}}\n"

        out_str += "# schema: astropy-2.0\n"
        out_str += "datetime,ra,dec,azimuth,altitude,x_image,y_image,integrated_pixel_value,' \
            'background_pixel_value,saturated_pixels,mag_data,err_minus_mag,err_plus_mag,snr\n"

        # Add the data (sort by frame)
        for frame, pick in sorted(self.pick_list.items(), key=lambda x: x[0]):

            if pick.get('x_centroid') is None or pick.get('mode', 0) == 0:
                continue

            snr = pick.get('snr', 0.0)
            if snr is None or snr <= 0:
                snr = 0.0
                mag_err_random = 0.0
            else:
                mag_err_random = 2.5 * np.log10(1 + 1 / snr)

                mag_err_total = np.sqrt(mag_err_random**2 + self.platepar.mag_lev_stddev**2)

                pp_tmp = copy.deepcopy(self.platepar)
                
                time_data = [self.img_obj.currentFrameTime(frame_no=frame)]

                jd_data, ra_data, dec_data, mag_data = xyToRaDecPP(time_data, [pick['x_centroid']],
                [pick['y_centroid']], [pick['intensity_sum']], pp_tmp, measurement=True)

                jd, ra, dec, mag = jd_data[0], ra_data[0], dec_data[0], mag_data[0]

                azim, alt = trueRaDec2ApparentAltAz(ra, dec, jd, pp_tmp.lat, pp_tmp.lon, refraction=False)

                frame_dt = self.img_obj.currentFrameTime(frame_no=frame, dt_obj=True)

                entry = [
                frame_dt.strftime(isodate_format_entry),
                f"{ra:10.6f}", f"{dec:+10.6f}",
                f"{azim:10.6f}", f"{alt:+10.6f}",
                f"{pick['x_centroid']:9.3f}", f"{pick['y_centroid']:9.3f}",
                f"{int(pick.get('intensity_sum', 0)):10d}",
                f"{int(pick.get('background_intensity', 0)):10d}",
                f"{str(pick.get('saturated', False)):5s}",
                f"{mag:+7.2f}", f"{-mag_err_total:+6.2f}", f"{mag_err_total:+6.2f}",
                f"{snr:10.2f}"
                ]
                out_str += ",".join(entry) + "\n"

        ecsv_file_path = os.path.join(self.ecsv_save_path, ecsv_file_name)

        with open(ecsv_file_path, 'w') as f:
            f.write(out_str)

        print("ECSV file saved to:", ecsv_file_path)
        return True

    def getResults(self, skyfit_format=False):
        
        if skyfit_format is True:

            return self.pick_list

        else:
            global_pick_indices = [fr + self.first_pick_global_index for fr in self.pick_frame_indices]
            return (self.global_picks, global_pick_indices, self.snr, self.abs_level_sums, 
                self.photometry_pixels, self.background_levels, self.saturated_bool_list)
    
    def getTotalPicks(self):
        return len(self.pick_frame_indices)
    
    def getMinSnr(self):
        return self.snr_threshold

# Terminal call functions
def checkAstraConfig(astra_config):
    """Checks the config only has valid values"""
    config = astra_config
    pso_ranges_and_types = {
        "w (0-1)": (0.0, 1.0, float),
        "c_1 (0-1)": (0.0, 1.0, float),
        "c_2 (0-1)": (0.0, 1.0, float),
        "max itter": (1, None, int),
        "n_particles": (1, None, int),
        "V_c (0-1)": (0.0, 1.0, float),
        "ftol": (0.0, 100.0, float),
        "ftol_itter": (1, None, int),
        "expl_c": (1.0, None, float),
        "P_sigma": (0.0, None, float)
    }

    astra_ranges_and_types = {
        "star_thresh": (0, None, float), "min SNR": (0, None, float),
        "P_crop": (0, None, float), "sigma_init (px)": (0.1, None, float), "sigma_max": (1, None, float),
        "L_max": (1, None, float), "Verbose": (True, False, bool), "photom_thresh": (0, 1, float),
        "Save Animation": (True, False, bool), "pick_offset": (0, None, float)
    }

    kalman_ranges_and_types = {
        "Monotonicity": (True, False, bool), "sigma_xy (px)": (0, None, float), 
        "sigma_vxy (%)": (0, 100, float), "save results": (True, False, bool)
    }

    errors = {}

    # Check PSO parameters
    for param, (min_val, max_val, param_type) in pso_ranges_and_types.items():
        value_str = config["pso"].get(param, "")
        try:
            if param_type == bool:
                value = value_str.strip().lower() == "true"
                if value not in [True, False]:
                    errors[f"pso.{param}"] = f"{param} must be True or False, got {value_str}"
            else:
                value = param_type(value_str)
                if min_val is not None and value < min_val:
                    errors[f"pso.{param}"] = f"{param} must be >= {min_val}, got {value}"
                elif max_val is not None and value > max_val:
                    errors[f"pso.{param}"] = f"{param} must be <= {max_val}, got {value}"
        except (ValueError, TypeError):
            errors[f"pso.{param}"] = f"{param} must be {param_type.__name__}, got {value_str}"

    # Check ASTRA parameters
    for param, (min_val, max_val, param_type) in astra_ranges_and_types.items():
        value_str = config["astra"].get(param, "")
        try:
            if param_type == bool:
                value = value_str.strip().lower() == "true"
                if value not in [True, False]:
                    errors[f"astra.{param}"] = f"{param} must be True or False, got {value_str}"
            else:
                value = param_type(value_str)
                if min_val is not None and value < min_val:
                    errors[f"astra.{param}"] = f"{param} must be >= {min_val}, got {value}"
                elif max_val is not None and value > max_val:
                    errors[f"astra.{param}"] = f"{param} must be <= {max_val}, got {value}"
        except (ValueError, TypeError):
            errors[f"astra.{param}"] = f"{param} must be {param_type.__name__}, got {value_str}"

    # Check Kalman parameters
    for param, (min_val, max_val, param_type) in kalman_ranges_and_types.items():
        value_str = config["kalman"].get(param, "")
        try:
            if param_type == bool:
                value = value_str.strip().lower() == "true"
                if value not in [True, False]:
                    errors[f"kalman.{param}"] = f"{param} must be True or False, got {value_str}"
            else:
                value = param_type(value_str)
                if min_val is not None and value < min_val:
                    errors[f"kalman.{param}"] = f"{param} must be >= {min_val}, got {value}"
                elif max_val is not None and value > max_val:
                    errors[f"kalman.{param}"] = f"{param} must be <= {max_val}, got {value}"
        except (ValueError, TypeError):
            errors[f"kalman.{param}"] = f"{param} must be {param_type.__name__}, got {value_str}"

    return errors

def loadEvTxt(txt_file_path):
    """
    Loads the Ev*.txt file and adds the relevant info to pick_list
    Args:
        txt_file_path (str): Path to the Ev*.txt file to load
    Returns:
        picks (dict): (N : [8]) dict following same format as self.pick_list 
    """

    picks = [] # (N, x, y) array of picks
    pick_frame_indices = [] # (N,) array of frame indices

    # Temp bool to extract header line
    first_bool = False

    # Wrap in try except for uncaught errors
    try:
        # Opens and parses pick file
        with open(txt_file_path, 'r') as file:
            for i, line in enumerate(file):
                if line.strip() and not line.startswith('#'):

                    # Clean the line
                    line = [part.strip() for part in line.split() if part]

                    # Unpack the header
                    if first_bool == False:
                        first_bool = True

                        # Unpack header info
                        column_names = [part.strip() for part in temp_line.split()[1:] if part]

                        # Extract indices
                        frame_number_idx = column_names.index('fr') if 'fr' in column_names else None
                        x_centroid_idx = column_names.index('cx') if 'cx' in column_names else None
                        y_centroid_idx = column_names.index('cy') if 'cy' in column_names else None

                        # Raise error if not found in txt file
                        if frame_number_idx is None or x_centroid_idx is None or y_centroid_idx is None:
                            raise ValueError("TXT file must contain 'fr', 'cx', and 'cy' columns.")
                    
                    else:
                        # Unpack the pick
                        frame_number = int(line[frame_number_idx])
                        cx, cy = float(line[x_centroid_idx]), float(line[y_centroid_idx])

                        picks.append([cx, cy])
                        pick_frame_indices.append(frame_number)

                # Store a temp line to hit previous for col names
                temp_line = line
        
        # Create ASTRA-type pick_dict
        pick_dict = {}
        for i, fr_no in enumerate(pick_frame_indices):
            pick_dict[fr_no] = {
                "x_centroid" : picks[i][0],
                "y_centroid" : picks[i][1]
            }

        return pick_dict

    # Raise error message
    except Exception as e:
        raise ValueError(f"Error reading TXT file: {str(e)}")
    
def loadECSV(ECSV_file_path, dir_path, img_obj):
    # ASTRA Addition - Justin DT
    """
    Loads the ECSV file and adds the relevant info to pick_list
    Args:
        ECSV_file_path (str): Path to the ECSV file to load
    Returns:
        picks (np.ndarray): (N, 2) array of (x, y) picks
        pick_frame_indices (np.ndarray): (N,) array of frame indices for each pick
        mags (np.ndarray): (N,) array of magnitudes for each pick
    """

    # Instantiate arrays to be populated
    picks = []  # N x args_dict array for addCentroid
    pick_frame_indices = []  # (N,) array of frame indices
    pick_frame_times = []  # (N,) array of frame times

    # Opens and parses the ECSV file
    with open(ECSV_file_path, 'r') as file:

        # Read the file contents
        contents = file.readlines()

        # Temp bool to get the column names
        first_bool = False

        # Process the contents
        for line in contents:

            # Clean the line
            line = [part.strip() for part in line.split(',') if part]

            # Skip header lines
            if line[0].startswith('#'):
                continue

            if first_bool == False:
                # Set first bool to True so header is not unpacked twice
                first_bool = True

                # Unpack column names
                column_names = line

                # Map column names to their indices
                pick_frame_times_idx = column_names.index('datetime') if 'datetime' in column_names else None
                x_ind = column_names.index('x_image') if 'x_image' in column_names else None
                y_ind = column_names.index('y_image') if 'y_image' in column_names else None

                continue

            else:
                # Unpack line

                # Populate arrays
                cx, cy = float(line[x_ind]), float(line[y_ind])
                pick_frame_times.append(datetime.strptime(line[pick_frame_times_idx], '%Y-%m-%dT%H:%M:%S.%f'))

                # Load in pick parameters, use default for other values
                picks.append([cx, cy])

    # Converts times into frame indices, accounting for floating-point errors
    pick_frame_indices = []
    frame_count = img_obj.total_frames
    time_idx = 0
    for i in range(frame_count):
        frame_time = img_obj.currentFrameTime(frame_no=i, dt_obj = True)
        time = pick_frame_times[time_idx]
        if frame_time == time or \
            frame_time == time + dt.timedelta(microseconds=1) or \
            frame_time == time - dt.timedelta(microseconds=1):
            pick_frame_indices.append(i)
            time_idx += 1
        if time_idx >= len(pick_frame_times):
            break

    # Format into ASTRA pick_dict format
    pick_dict = {}
    for i, fr_no in enumerate(pick_frame_indices):
        pick_dict[fr_no] = {
            "x_centroid" : picks[i][0],
            "y_centroid" : picks[i][1]
        }

    return pick_dict



if __name__ == "__main__":

    # first check if two events, if yes then compute both else just one
    # args:
        # file directory
        # astra_config path (optional)
        # run kalman (optional)

    # To determine if there is more than one event, check for children directories that have a config in them
    import argparse
    ArgumentParser = argparse.ArgumentParser    

    parser = ArgumentParser(description="""
Command-line interface for the ASTRA (Astrometric Streak Tracking and Refinement Algorithm) pipeline.

This script serves as the main entry point for processing meteor events using ASTRA. It can operate in two modes: 
single-station or multi-station. The script parses command-line arguments to locate event data, load 
configurations, and run the full analysis pipeline, saving the refined results in the GDEF ECSV format.

Workflow:
1.  Configuration Loading: 
    It loads an ASTRA-specific configuration file (JSON) that controls the algorithm's parameters 
    (e.g., PSO settings, SNR thresholds). If no config is provided, it uses a built-in default. 
    The configuration is validated to prevent runtime errors.

2.  Data Discovery:
    -   Single-Station Mode: The script expects a single directory containing all necessary files: 
        a RMS `config.json`, a dark/bias frame, a flat frame, and an initial picks file (`.ecsv` or `.txt`).
    -   Multi-Station Mode (`--multi_station`): The script scans the provided parent directory for 
        subdirectories. Each subdirectory that contains a `config.json` is treated as a separate 
        station to be processed.

3.  ASTRA Processing:
    -   An `ASTRA` object is instantiated for each station.
    -   The processing is launched, which involves:
        - Image correction and background subtraction.
        - Recursive cropping and first-pass fitting with PSO.
        - Local refinement of fits with L-BFGS-B.
        - Filtering of low-quality detections based on SNR and geometry.
    -   For multi-station events, processing for each station is run in parallel using a 
        `ThreadPoolExecutor` to improve performance.

4.  Output Generation: 
    After processing, the refined picks, photometry, and astrometry are saved to a GDEF-compliant 
    ECSV file in the specified output directory. If `--save_animation` is enabled, diagnostic plots 
    for each frame are also saved.

Arguments:
    file_directory (str):
        Positional argument. The path to the root directory containing the event data.
        For single-station mode, this is the directory with the config, dark/flat, and pick files.
        For multi-station mode, this is the parent directory containing subdirectories for each station.

    ECSV_save_path (str):
        Positional argument. The directory where the output ECSV file(s) will be saved.
        If not specified, it defaults to the `file_directory` for each station.

    --astra_config_path (str, optional):
        Path to a JSON file containing the ASTRA configuration. This file defines parameters for PSO,
        cropping, filtering, and other algorithm-specific settings. If omitted, a default configuration is used.

    --save_animation (bool, optional):
        Flag to enable saving a series of diagnostic JPEG images for each processed frame.
        These images show the original crop, the Gaussian fit, and the residuals. Defaults to False.

    --verbose (bool, optional):
        Flag to enable detailed print statements during the execution of the pipeline, which is useful
        for debugging. Defaults to False.

    --multi_station (bool, optional):
        Flag to indicate that the `file_directory` contains data for multiple stations in subdirectories.
        The script will process each station in parallel. Defaults to False.

    --use_txt_picks (bool, optional):
        Flag to force the script to load initial picks from a `Ev*.txt` file instead of the default `.ecsv` file.
        Useful for processing older data formats. Defaults to False.

Usage Examples:
    # Process a single-station event
    python Astra.py /path/to/single_event_data /path/to/output

    # Process a multi-station event with verbose output and save animations
    python Astra.py /path/to/multi_station_parent /path/to/output --multi_station --verbose --save_animation

    # Process an event using a custom ASTRA configuration
    python Astra.py /path/to/event /path/to/output --astra_config_path /path/to/my_astra_config.json
""", formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("file_directory", type=str, help="Path to the file directory")
    parser.add_argument("ECSV_save_path", type=str, nargs='?', default=None, 
                        help="Path to save ECSV file, defaults to dir with config")
    parser.add_argument("-c", "--astra_config_path", type=str, 
                        help="ASTRA_config dict or Path to the ASTRA config json")
    parser.add_argument("-sa", "--save_animation", default=False, action="store_true", 
                        help="Whether to save animation of frame fitting")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", 
                        help="Whether to print verbose output")
    parser.add_argument("-m", "--multi_station", default=False, action="store_true", 
                        help="Whether to process multi-station data")
    parser.add_argument("-txt", "--use_txt_picks", default=False, action="store_true", 
                        help="Whether to use picks from DetApp TXT file")

    args = parser.parse_args()

    # Check filepath
    if not os.path.exists(args.file_directory):
        raise FileNotFoundError(f"File directory not found: {args.file_directory}")
    else:
        file_path = args.file_directory

    # Check multi_station
    if args.multi_station is True:
        multi_station = True

    # Load and check astra_config
    if args.astra_config_path:

        # Load if a json file
        if args.astra_config_path.endswith(".json"):
            with open(args.astra_config_path, "r") as f:
                astra_config = json.load(f)

        # If astra_config provided, check values
        try:
            errors = checkAstraConfig(astra_config)
        except Exception as e:
            print(f"Error processing astra_config, check Astra.py execution header for proper format: {e}")
            raise
        if errors:
            print("Unexpected astra_config items or values")
            for key, error in errors.items():
                print(f"astra_config error: {key}: {error}")
            raise ValueError("Invalid astra_config values. See errors above.")

    else:
        print("No ASTRA config provided. Using default configuration.")
        astra_config = {'file_path': file_path, 
                        'pso': {'w (0-1)': 0.9, 'c_1 (0-1)': 0.45, 'c_2 (0-1)': 0.25, 'max itter': 125, 
                                'n_particles': 125, 'V_c (0-1)': 0.3, 'ftol': 1e-5, 'ftol_itter': 25, 
                                'expl_c': 3, 'P_sigma': 3}, 
                        'astra': {'star_thresh': 3, 'min SNR': 10, 'P_crop': 1.5, 'sigma_init (px)': 2, 
                                  'sigma_max': 1.2, 'L_max': 1.5, 'Verbose': False, 'photom_thresh': 0.05, 
                                  'Save Animation': False, 'pick_offset': 3}, 
                        'kalman': {'Monotonicity': True, 'sigma_xy (px)': 0.5, 'sigma_vxy (%)': 100, 
                                   'save results': False}}
    
    dir_path = file_path
    if not os.path.isdir(file_path):
        dir_path = os.path.dirname(file_path)

    # if multi-station, parse directory for subfolders including a .config object
    if args.multi_station:

        subfolders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        config_folders = [folder for folder in subfolders if [file for file in os.listdir(folder) \
                                                               if file.endswith('.config')] != []]
        if not config_folders:
            raise FileNotFoundError("No config folders found.")
        print(f"Found config folders: {config_folders}")
    else:
        config_folders = [file_path] if os.path.isdir(file_path) else [os.path.dirname(file_path)]

    # if no ecsv save path specified, use each config folder
    if not args.ECSV_save_path:
        args.ECSV_save_path = config_folders
    # Else make sure directory(ies) exist
    for path in (args.ECSV_save_path if isinstance(args.ECSV_save_path, list) else [args.ECSV_save_path]):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {path}: {e}")

    # Set astra config with verbose and save animation
    astra_config['astra']['Verbose'] = args.verbose
    astra_config['astra']['Save Animation'] = args.save_animation

    configs = []
    img_objs = []
    darks = []
    flats = []
    pick_dicts = []
    platepars = []

    # For each camera (single if multi_station is false) prep data
    for i, config_path in enumerate(config_folders):

        print("Using config path:", config_path)

        # Load config obj
        config = ConfigReader.loadConfigFromDirectory('.', config_path)

        # Load img obj
        if os.path.isfile(file_path):
            img_handle = detectInputTypeFile(file_path, config)

        else:
            img_handle = detectInputTypeFolder(config_path, config)

        # Load dark
        dark_name = [file_name for file_name in os.listdir(config_path) if \
                        "dark" in file_name or "bias" in file_name]
        if dark_name != []:
            dark = Image.loadDark(config_path, dark_name[0], byteswap=img_handle.byteswap)
            if len(dark_name) > 1:
                print(f"Warning: Multiple dark fields found. Using {dark_name[0]}.")
        else:
            dark = None
            print(f"No dark found in {config_path}, using no dark correction.")

        # Load flat
        flat_name = [file_name for file_name in os.listdir(config_path) if "flat" in file_name]
        if flat_name != []:
            flat = Image.loadFlat(config_path, flat_name[0], byteswap=img_handle.byteswap)
            if len(flat_name) > 1:
                print(f"Warning: Multiple flat fields found. Using {flat_name[0]}.")
        else:
            flat = None
            print(f"No flat found in {config_path}, using no flat correction.")

        # Load ECSV picks
        if args.use_txt_picks:

            txt_name = [file_name for file_name in os.listdir(config_path) if file_name.endswith('.txt') \
                            and file_name.startswith('ev')]
            
            if txt_name != []:
                pick_dict = loadEvTxt(os.path.join(config_path, txt_name[0]), img_handle)
            else:
                raise ValueError("No TXT picks found.")
        
        else:
            ecsv_name = [file_name for file_name in os.listdir(config_path) if "ecsv" in file_name]

            if ecsv_name != []:
                pick_dict = loadECSV(os.path.join(config_path, ecsv_name[0]), config_path, img_handle)

                if len(ecsv_name) > 1:
                    print(f"Warning: Multiple ECSV files found. Using {ecsv_name[0]}.")
            else:
                raise ValueError("No ECSV picks found.")
        
        if pick_dict == {}:
            raise ValueError("No picks loaded.")


        # Load platepar
        platepar_name = [file_name for file_name in os.listdir(config_path) if "platepar" in file_name]
        platepar = Platepar()
        if platepar_name != []:
            platepar.read(os.path.join(config_path, platepar_name[0]))
            if len(platepar_name) > 1:
                print(f"Warning: Multiple platepar fields found. Using {platepar_name[0]}.")
        else:
            raise ValueError("No platepar found.")


        # Once no errors raised, add all to lists
        configs.append(config)
        img_objs.append(img_handle)
        flats.append(flat)
        darks.append(dark)
        pick_dicts.append(pick_dict)
        platepars.append(platepar)

        print(f"Loaded data ({i+1} / {len(config_folders)}) from {config_path}")
        

    # Using a threadpool run all astra processes (if only one, do not use threadpool)
    def process_astra(platepar, *args):
        
        astra_obj = ASTRA(*args)

        astra_obj.process()

        astra_obj.saveECSV(platepar)

    if len(configs) == 1 :
        print('Starting ASTRA Process on one camera')
        process_astra(platepars[0], img_objs[0], pick_dicts[0], astra_config, 
                      config_folders[0], configs[0], darks[0], flats[0], None, 
                      args.ECSV_save_path if isinstance(args.ECSV_save_path, str) else args.ECSV_save_path[0])
    else:
        print(f'Starting ASTRA Process on {len(configs)} cameras')
        with ThreadPoolExecutor(max_workers=len(configs)) as executor:
            for i in range(len(configs)):
                executor.submit(process_astra, platepars[i], img_objs[i], pick_dicts[i], astra_config, 
                              config_folders[i], configs[i], darks[i], flats[i], None, 
                      args.ECSV_save_path if isinstance(args.ECSV_save_path, str) else args.ECSV_save_path[i])
    print(f'FINISHED PROCESSING {len(configs)} cameras')
    print(f'SAVED RESULTS TO:')
    for i, config in enumerate(configs):
        save_path = args.ECSV_save_path[i] if isinstance(args.ECSV_save_path, list) else args.ECSV_save_path
        print(f'{config.stationID}  {save_path}')
