import os
import numpy as np
import scipy.optimize
import scipy.stats
import scipy.special
from RMS.Routines.Image import signalToNoise
from pyswarms.single.global_best import GlobalBestPSO
from RMS.Routines import Image
import cv2

class ASTRA:

    def __init__(self, data_dict, progress_callback=None):
        """
        ASTRA : Astrometric Streak Tracking and Refinement Algorithm
        A class for proccessing meteor picks in EMCCD data, using a recursive algorithm to retreive accurate picks automatically.
        ________________________
        Initializes the ASTRA object with the provided data dictionary and progress callback.

        Args:
            data_dict (dict): A dictionary containing the necessary data for processing:
            progress_callback (callable, optional): A callback function to update progress. Defaults to None.
        """

        # -- Constants & Settings --

        # Try loading initial data
        try:
            self.astra_config = data_dict["astra_config"]  # Load ASTRA configuration from data_dict
        except KeyError:
            print("Warning, 'config' key not found in passed data_dict. Aborting process")
            return
        
        # Initialize callback and set progress to 0
        self.progress_callback = progress_callback
        if self.progress_callback is not None:
            self.progress_callback(0)  # Initialize progress to 0

        # 1) Image Data Processing Constants/Settings
        self.BACKGROUND_STD_THRESHOLD = float(self.astra_config['astra']['star_thresh']) # multiple of std dev to mask pixels above
        self.saturation_threshold = float(data_dict["saturation_threshold"]) # Saturation threshold for EMCCD cameras, used to mask saturated pixels

        # 2) PSO Constants/Settings

        # Represents the multiples of their standard deviations that the parameters are allowed to deviate within in the second pass gaussian
        self.std_parameter_constraint = [
            float(self.astra_config['pso']['P_sigma']), # level_sum
            float(self.astra_config['pso']['P_sigma']), # height / sigma_y
            float(self.astra_config['pso']['P_sigma']), # x0
            float(self.astra_config['pso']['P_sigma']), # y0
            float(self.astra_config['pso']['P_sigma'])  # length / sigma_x
        ]

        # The settings passed onto the PSO on the first pass
        self.first_pass_settings = {
            "residuals_method" : 'abs', # method for calculating residuals
            "oob_penalty" : 1e6, # penalty for out-of-bounds particles
            "options": {"w": float(self.astra_config['pso']['w (0-1)']), # cognitive component
                        "c1": float(self.astra_config['pso']['c_1 (0-1)']), # social component
                        "c2": float(self.astra_config['pso']['c_2 (0-1)'])}, # inertia weight
            "max_iter": int(self.astra_config['pso']['max itter']), # maximum iterations
            "n_particles": int(self.astra_config['pso']['n_particles']), # number of particles
            "Velocity_coeff": float(self.astra_config['pso']['V_c (0-1)']), # percentage of parameter range for velocity clamp
            "ftol" : float(self.astra_config['pso']['ftol']), # function tolerance for convergence
            "ftol_iter" : int(self.astra_config['pso']['ftol_itter']), # number of iterations for function tolerance
            "bh_strategy": 'nearest', # best-historical strategy
            "vh_strategy": 'invert', # velocity-historical strategy
            "explorative_coeff" : float(self.astra_config['pso']['expl_c']) #Number of std to enforce when generating initial particles (3 is a good value)
        }

        # Settings and padding coefficients for cropping
        self.cropping_settings = {
            'initial_padding_coeff': float(self.astra_config['astra']['P_crop']), # Multiples of max sigma and length to pad the crop for middle frames
            'recursive_padding_coeff': float(self.astra_config['astra']['P_crop']), # Multiples of max sigma and length to pad the crop for recursive cropping
            'init_sigma_guess' : float(self.astra_config['astra']['sigma_init (px)']), # The initial guess for sigma in the middle frames
            'max_sigma_coeff' : float(self.astra_config['astra']['sigma_max']), # Multiples of the p0 sigma to define as a maximum value
            'max_length_coeff' : float(self.astra_config['astra']['L_max']) # Multiples of the p0 length to define as a maximum value
        }

        # 3) Second-Pass Local Optimization Constants/Settings

        # The settings passed onto the local optimizer on the second pass
        # NOTE depreciated, using local not dual PSO
        self.second_pass_settings = {
            "residuals_method" : 'abs_squared', # method for calculating residuals
            "method" : 'L-BFGS-B', # method for optimization
            "oob_penalty" : 1e6, # penalty for out-of-bounds function evaluations
        }

        # 4) Kalman Settings
        self.kalman_settings = {
            'monotonicity': self.astra_config['kalman']['Monotonicity'].lower() == 'true',
            'use_accel': False,
            'sigma_xy': float(self.astra_config['kalman']['sigma_xy (px)']),
            'sigma_vxy_perc': float(self.astra_config['kalman']['sigma_vxy (%)']),
            'save results' : self.astra_config['kalman']["save results"].lower() == 'true'
        }

        # The SNR threshold for rejecting low SNR frames
        self.SNR_threshold = float(self.astra_config['astra']['min SNR'])

        # -- Data Attributes --
        # Unpack data from the data_dict

        self.data_dict = data_dict # Data dictionary containing all necessary data
        self.img_obj = data_dict["img_obj"] # Image object containing frame data and metadata
        self.frames = data_dict["frames"] # Numpy array of image frames (N, w, h)
        self.times = data_dict["times"] # Numpy array of times corresponding to the picks (N,)

        # Loads all picks & pick_frame_indices
        self.picks = np.array(data_dict["picks"]) # Numpy array of DetApp picks (N, 2) where each row is (x, y) coordinates
        self.pick_frame_indices = data_dict["pick_frame_indices"].tolist() # List of frame indices corresponding to the picks

        self.verbose = self.astra_config['astra']['Verbose'].lower() == 'true'
        self.save_animation = self.astra_config['astra']['Save Animation'].lower() == 'true'
        self.data_path = self.data_dict['data_path']
        self.dark = data_dict['dark']
        self.flat_struct = data_dict['flat']
        self.skyfit_config = data_dict['img_config']


    def process(self):
        """
        Processes all the camera data by performing the following steps:
            1. Recursively crop all frames to extract each ROI in each meteor-present frame
            2. Fit the cropped frames using a PSO to a moving gaussian model
            3. Refine the moving gaussian fit by using a local optimizer
            4. Remove picks with low SNR and out-of-bounds picks
            5. Return the ASTRA object for later processing/saving
        Args:
            None
        Returns:
            self (ASTRA): The ASTRA object with processed data.
        """

        self.mode = 'Gaussian'  # Set the mode to Gaussian for processing

        # Instantiate variables to store progress
        self.progressed_frames = {
            'cropping': 0,
            'refining': 0,
            'removing': 0
        }
        self.total_frames = self.pick_frame_indices[-1] - self.pick_frame_indices[0]

        # 1. Processes all frames by background subtracting, masking starsm and saving the avepixel_background. Also correct all frames with dark/flat and gamma corerction
        try:
            self.avepixel_background, self.subtracted_frames, self.frames = self.processImageData(self.img_obj, self.frames)
        except Exception as e:
            print(f'Error processing image data: {e}')

        # 2. Recursively crop & fit a moving gaussian across whole event
        try:
            self.cropAllMeteorFrames(self.pick_frame_indices, self.picks)
        except Exception as e:
            print(f'Error cropping and fitting meteor frames: {e}')

        # 3. Refine the moving gaussian fit by using a local optimizer
        try:
            self.refineAllMeteorCrops(self.first_pass_params, self.cropped_frames, self.omega, self.directions)
        except Exception as e:
            print(f'Error refining meteor crops: {e}')

        # 4. Remove picks with low SNR and out-of-bounds picks, refactors into global coordinates
        try:
            self.removeLowSNRPicks(self.refined_fit_params, self.fit_imgs, self.frames, self.cropped_frames, self.crop_vars, self.pick_frame_indices, self.fit_costs, self.times)
        except Exception as e:
            print(f'Error removing low SNR picks: {e}')
        
        # 5. save animation
        if self.save_animation:
            try:
                self.saveAni(self.data_path)
            except Exception as e:
                print(f'Error saving animation: {e}')

        # Set progress to 100
        self.updateProgress(100)

        # 6. Return the ASTRA object for later processing/saving
        return self


    # 1) -- Functional Methods --

    def runKalman(self, measurements, times):
        """
        Runs the Kalman filter on the final picks if enabled.
        Args:
            None
        Returns:
            self (ASTRA): The ASTRA object with processed data.
        """

        # Instantiate variables to store progress
        self.mode = 'Kalman'  # Set the mode to Kalman for processing
        self.exec_count = 0
        self.total_exec = 1  # Initialize total execution count to avoid division by zero
        self.updateProgress()

        # Add another multiple of frames to process if the monotonicity is enabled
        if self.kalman_settings['monotonicity']:
            time_coeff = 3
        else:
            time_coeff = 2

        # Set progress to 0
        if self.progress_callback is not None:
            self.progress_callback(0)

        # Try to run kalman filter
        try:
            self.total_exec = len(times) * time_coeff
            smooth_picks, smooth_P = self.applyKalmanFilter(measurements, times)
        except Exception as e:
            print(f'Error applying Kalman filter: {e}')

        # Set progress bar to 100
        self.exec_count = self.total_exec
        self.updateProgress()

        # Return adjusted picks and covariance matrix
        return smooth_picks, smooth_P
    

    def processImageData(self, img_obj, frames):
        """
        Processes the image data by performing background subtraction and star masking.
        Args:
            img_obj (InputTypeImages): An instance of the InputTypeImages class containing the image data.
            frames (numpy.ndarray): A 3D numpy array of shape (N, w, h) where N is the number of frames, 
                                    w is the width, and h is the height of each frame.
        Returns:
            tuple: A tuple containing:
                - avepixel_background (numpy.ndarray): A 2D numpy array of shape (w, h) representing the average pixel values of the background.
                - subtracted_frames (numpy.masked_array): A masked array of shape (N, w, h) containing the subtracted frames with star masking applied.
        """

        corrected_frames = []
        subtracted_frames = []
        masked_frames = []

        # 2) -- Correct all frames
        try:
            for frame in frames:
                
                # Apply dark if available
                if self.dark is not None:
                    corrected_frame = Image.applyDark(frame, self.dark)
                
                # Apply flat if available
                if self.flat_struct is not None:
                    corrected_frame = Image.applyFlat(corrected_frame, self.flat_struct)

                if self.flat_struct is not None or self.dark is not None:
                    # Apply gamma correction
                    corrected_frame = Image.gammaCorrectionImage(corrected_frame, self.skyfit_config.gamma, bp=0, wp=(2**self.skyfit_config.bit_depth - 1))
                else:
                    corrected_frame = Image.gammaCorrectionImage(frame, self.skyfit_config.gamma, bp=0, wp=(2**self.skyfit_config.bit_depth - 1))

                # Append corrected frame
                corrected_frames.append(corrected_frame)
        except Exception as e:
            print(f"Error correcting frames: {e}")

        # 1) -- Background Subtraction --
        try:
            # Load background using RMS
            fake_ff_obj = img_obj.loadChunk()
            avepixel_background = fake_ff_obj.avepixel
            corrected_frames = np.array(corrected_frames)

            # Correct avepixel_background
            if self.dark is not None:
                corrected_avepixel = Image.applyDark(avepixel_background, self.dark)

            if self.flat_struct is not None:
                corrected_avepixel = Image.applyFlat(corrected_avepixel, self.flat_struct)
            
            if self.dark is not None or self.flat_struct is not None:
                corrected_avepixel = Image.gammaCorrectionImage(corrected_avepixel, self.skyfit_config.gamma, bp=0, wp=(2**self.skyfit_config.bit_depth - 1))

            else:
                corrected_avepixel = Image.gammaCorrectionImage(avepixel_background, self.skyfit_config.gamma, bp=0, wp=(2**self.skyfit_config.bit_depth - 1))
            
            corrected_avepixel = np.clip(corrected_avepixel, 0, None)

            # Subtract frames
            subtracted_frames = corrected_frames.copy() - corrected_avepixel.copy()

            # Clip subtracted frames to above zero
            subtracted_frames = np.clip(subtracted_frames, 0, None)

        except Exception as e:
            raise Exception(f"Error loading background or subtracting frames: {e}")
        
        # 2) -- Star Masking --

        # Calculate std of background
        background_std = np.std(corrected_avepixel)
        background_mean = np.mean(corrected_avepixel)

        # Calculate the masking threshold
        threshold = background_mean + self.BACKGROUND_STD_THRESHOLD * background_std

        # Mask values exceeding the threshold
        ave_mask = np.ma.MaskedArray(corrected_avepixel > threshold)

        # Tries to implement mask
        try:
            subtracted_frames = np.ma.masked_array(subtracted_frames, 
                                                   mask=np.repeat(ave_mask[np.newaxis, :, :], subtracted_frames.shape[0], axis=0))
            corrected_avepixel = np.ma.masked_array(corrected_avepixel, mask=ave_mask)
            masked_frames = np.ma.masked_array(corrected_frames, 
                                                    mask=np.repeat(ave_mask[np.newaxis, :, :], corrected_frames.shape[0], axis=0))
        except Exception as e:
            raise Exception(f"Error applying mask to subtracted frames: {e}")
        
        # 3) -- Returns Data --
        return corrected_avepixel, subtracted_frames, masked_frames


    def cropAllMeteorFrames(self, pick_frame_indices, picks):
        """
        A recursive algorithm that crops all meteor frames by continously fitting and re-estimated physical parameters at each step.
        Args:
            detApp_picks (numpy.ndarray): A 2D numpy array of shape (N, 2) where N is the number of picks and each row contains the (x, y) coordinates of a pick.
            pick_frame_indices (list): A list of frame indices corresponding to each pick in detApp_picks.
        Returns:
            tuple: A tuple containing:
                - cropped_frames (numpy.ndarray): A 3D numpy array of shape (N, w, h) where N is the number of cropped frames, w is the width, and h is the height of each cropped frame.
                - first_pass_params (numpy.ndarray): A 2D numpy array of shape (N, 5) where N is the number of frames and each row contains the parameters (level_sum, height, x0, y0, length) from the first pass.
                - crop_vars (numpy.ndarray): A 2D numpy array of shape (N, 6) where N is the number of frames and each row contains the crop variables (cx, cy, xmin, xmax, ymin, ymax).
        """

        # 1) -- Unpack variables & Calculate seed picks/frames--
        seed_picks_global, seed_indices = self.select_seed_triplet(picks, pick_frame_indices)
        omega = np.arctan2(picks[-1][1] - picks[0][1], -picks[-1][0] + picks[0][0])  % (2*np.pi)

        if self.verbose:
            print(f"Starting recursive cropping with {len(seed_indices)} seed picks at indices {seed_indices} and omega {omega} radians.")

        # 2) -- Estimate initial parameters from the seed picks --

        # Estimate the length (take largest value to be safe)
        init_length = np.max([np.linalg.norm(seed_picks_global[0] - seed_picks_global[1]),
                                np.linalg.norm(seed_picks_global[1] - seed_picks_global[2])])
        
        # Determine the omega (angle) based on detapp
        # NOTE: If a method other than detapp is used, this omega will need to be refined as it is crucial in estimation yet does not differ
        

        # Instantiate omega as a instance variable
        self.omega = omega

        # Determine direction of motion
        norm = seed_picks_global[-1] - seed_picks_global[0]
        directions = (-1 if norm[0] < 0 else 1, -1 if norm[1] < 0 else 1)
        self.directions = directions

        # 3) -- Instantiate nessesary instance arrays --
        # NOTE: times will also need to be here for full DetApp independance
        self.cropped_frames = [None] * len(seed_indices) # (N, w, h) array of cropped frames
        self.first_pass_params = np.zeros((len(seed_indices), 5), dtype=np.float32) # (N, 5) array of first pass parameters (level_sum, height, x0)
        self.crop_vars = np.zeros((len(seed_indices), 6), dtype=np.float32) # (N, 6) array of crop variables (cx, cy, xmin, xmax, ymin, ymax)

        # # 4) -- Process each seed pick to kick-start recursion --
        for i in range(len(seed_indices)):

            # Crop initial frames
            self.cropped_frames[i], self.crop_vars[i] = self.cropFrameToGaussian(self.subtracted_frames[seed_indices[i]],
                                                self.estimateCenter(seed_picks_global[i], omega, init_length, directions=directions),
                                                self.cropping_settings["init_sigma_guess"],
                                                init_length * self.cropping_settings["max_length_coeff"],
                                                omega)
            
            # Run a first-pass Gaussian on the cropped frames
            self.first_pass_params[i], _ = self.GaussianPSO(self.cropped_frames[i], omega, directions, init_length=init_length)

            # Update progress
            self.progressed_frames['cropping'] += 1
            self.updateProgress()
            
        # Update progress   
        if self.verbose:
            print(f"Finished cropping {len(seed_indices)} frames with est. centroids: {self.first_pass_params[:, 2:4]}")

        # Instantiate splines
        parameter_estimation_functions = self.updateParameterEstimationFunctions(self.crop_vars, self.first_pass_params, forward_pass=True) # Set forward_pass to True since there are only 3 points

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
            self.estimateNextParameters(parameter_estimation_functions, self.first_pass_params.shape[0], forward_pass=False)['norm'],
            omega,
            directions=tuple(-x for x in list(directions))
        )

        # Begin backwards pass on crop
        self.recursiveCroppingAlgorithm(seed_indices[0] - 1,
                                      backward_next_center_global,
                                      parameter_estimation_functions,
                                      omega,
                                      directions=tuple(-x for x in list(directions)),
                                      forward_pass=False,
                                      )
        
        # Return modified instance variables
        return self.cropped_frames, self.first_pass_params, self.crop_vars


    def refineAllMeteorCrops(self, first_pass_params, cropped_frames, omega, directions):
        """
        Refines the moving Gaussian fit by using a local optimizer on the second pass.
        Args:
            first_pass_params (numpy.ndarray): A 2D numpy array of shape (N, 5) where N is the number of frames and each row contains the parameters (level_sum, height, x0, y0, length) from the first pass.
            cropped_frames (numpy.ndarray): A 3D numpy array of shape (N, w, h) where N is the number of frames, w is the width, and h is the height of each cropped frame.
            omega (float): The angle in radians at which the crop is oriented.
        Returns:
            tuple: A tuple containing:
                - refined_fit_params (numpy.ndarray): A 2D numpy array of shape (N, 5) where each row contains the refined parameters (level_sum, height, x0, y0, length) after the second pass.
                - fit_costs (numpy.ndarray): A 1D numpy array of shape (N,) containing the fit costs for each frame.
                - fit_imgs (list): A list of images showing the fitted Gaussian for each frame.
        """


        # Instantiate arrays to store fit data
        self.refined_fit_params = np.zeros((first_pass_params.shape[0], 5), dtype=np.float32) # (N, 5) array of refined fit parameters (level_sum, height, x0, y0, length)
        self.fit_costs = np.zeros((first_pass_params.shape[0],), dtype=np.float32) # (N,) array of fit costs
        self.fit_imgs = []

        # Iterate over all cropped frames and fit the second pass Gaussian
        for i, (p0, obs_frame) in enumerate(zip(first_pass_params, cropped_frames)):

            # Calculate the adaptive bounds based on the first pass parameters
            bounds = self.calculateAdaptiveBounds(first_pass_params, p0, True)

            # Perform the local Gaussian fit 
            fit_params, cost, img = self.LocalGaussianFit(p0, obs_frame, omega, bounds, directions)

            # Append results to the lists
            self.refined_fit_params[i] = fit_params
            self.fit_costs[i] = cost
            self.fit_imgs.append(img)

            # Update progress
            self.progressed_frames['refining'] += 1
            self.updateProgress()

        return self.refined_fit_params, self.fit_costs, self.fit_imgs


    def removeLowSNRPicks(self, refined_params, fit_imgs, frames, cropped_frames, crop_vars, pick_frame_indices, fit_costs, times):
        """
        Removes picks with low SNR and out-of-bounds picks from the refined parameters, fit images, cropped frames, crop variables, pick frame indices, fit costs, and times.
        Args:
            refined_params (numpy.ndarray): A 2D numpy array of shape (N, 5) where N is the number of frames and each row contains the refined parameters (level_sum, height, x0, y0, length).
            fit_imgs (list): A list of images showing the fitted Gaussian for each frame.
            frames (numpy.ndarray): A 3D numpy  array of shape (N, w, h) where N is the number of frames, w is the width, and h is the height of each frame.
            cropped_frames (list): A list of cropped frames corresponding to the refined parameters.
            crop_vars (numpy.ndarray): A 2D numpy array of shape (N, 6) where N is the number of frames and each row contains the crop variables (cx, cy, xmin, xmax, ymin, ymax).
            pick_frame_indices (numpy.ndarray): A 1D numpy array of shape (N,) containing the frame indices corresponding to each pick.
            fit_costs (numpy.ndarray): A 1D numpy array of shape (N,) containing the fit costs for each pick frame.
            times (list): A list of datetime objects corresponding to each pick frame.
        Returns:
            tuple: A tuple containing:
                - refined_params (numpy.ndarray): A 2D numpy array of shape (M, 5) where M is the number of frames after removing low SNR picks and each row contains the refined parameters (level_sum, height, x0, y0, length).
                - fit_imgs (list): A list of images showing the fitted Gaussian for each frame after removing low SNR picks.
                - cropped_frames (list): A list of cropped frames after removing low SNR picks.
                - crop_vars (numpy.ndarray): A 2D numpy array of shape (M, 6) where M is the number of frames after removing low SNR picks and each row contains the crop variables (cx, cy, xmin, xmax, ymin, ymax).
                - pick_frame_indices (numpy.ndarray): A 1D numpy array of shape (M,) containing the frame indices corresponding to each pick after removing low SNR picks.
                - global_picks (numpy.ndarray): A 2D numpy array of shape (M, 2) where M is the number of frames after removing low SNR picks and each row contains the global pick coordinates (x_global, y_global).
                - fit_costs (numpy.ndarray): A 1D numpy array of shape (M,) containing the fit costs for each frame after removing low SNR picks.
                - times (list): A list of datetime objects corresponding to each frame after removing low SNR picks.
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

            # Calculate photom_pixels
            photom_pixels = self.computePhotometryPixels(fit_imgs[i], cropped_frames[i], crop_vars[i])

            # Calculate SNR, and photom values
            snr = self.computeIntensitySum(photom_pixels, self.translatePicksToGlobal((refined_params[i, 2], refined_params[i, 3]), crop_vars[i]), frames[pick_frame_indices[i]])
            
            # Determine SNR - DEPRECIATED
            # snr = self.calculateSNR(fit_imgs[i], frames[pick_frame_indices[i]], cropped_frames[i], crop_vars[i])

            # Set index for previous parameters (util. previous since optim. will reshape curr params to fit even if streak is partially OOB)
            idx = i - 1 if i > 0 else 0

            # Reject SNR below the threshold
            if snr < self.SNR_threshold:
                snr_rejection_bool[i] = True
            # Check if the streak is outside the streak
            elif not self.checkStreakInBounds(self.translatePicksToGlobal((refined_params[i, 2], refined_params[i, 3]), crop_vars[i]),
                                                refined_params[idx, 4], refined_params[idx, 1], self.omega, self.directions):
                snr_rejection_bool[i] = True
            # If passes all checks, append SNR and level sum
            else:
                frame_snr_values.append(snr)
            
            # update progress
            self.progressed_frames['removing'] += 1
            self.updateProgress()

        # Print number of rejected frames
        print(f"Rejected {np.sum(snr_rejection_bool)} frames with SNR below {self.SNR_threshold}.")
            
        # Reject bad frames by removing indexes of low-SNR frames
        refined_params = self.refined_fit_params[~snr_rejection_bool]
        crop_vars = crop_vars[~snr_rejection_bool]
        pick_frame_indices = np.array(pick_frame_indices)[~snr_rejection_bool]
        fit_costs = fit_costs[~snr_rejection_bool]
        self.abs_level_sums = np.array(self.abs_level_sums)[~snr_rejection_bool]
        self.background_levels = np.array(self.background_levels)[~snr_rejection_bool]
        self.saturated_bool_list = np.array(self.saturated_bool_list)[~snr_rejection_bool]

        # Save copies before indexing to avoid recursion errors
        fit_imgs_copy = fit_imgs.copy()
        cropped_frames_copy = cropped_frames.copy()
        times_copy = times.copy()
        photometry_pixels_copy = self.photometry_pixels.copy()

        # Remove low-SNR frames from fit_imgs and cropped_frames
        fit_imgs = [fit_imgs_copy[i] for i in range(len(fit_imgs_copy)) if not snr_rejection_bool[i]]
        cropped_frames = [cropped_frames_copy[i] for i in range(len(cropped_frames_copy)) if not snr_rejection_bool[i]]
        times = [times_copy[i] for i in range(len(times_copy)) if not snr_rejection_bool[i]]
        self.photometry_pixels = [photometry_pixels_copy[i] for i in range(len(photometry_pixels_copy)) if not snr_rejection_bool[i]]

        # Translate to global coordinates as new variable
        global_picks = np.array([self.movePickToEdge(self.translatePicksToGlobal((refined_params[i, 2], refined_params[i, 3]), crop_vars[i]),
                                                     self.omega,
                                                     refined_params[i][4],
                                                     directions = self.directions) for i in range(len(refined_params))])

        # Save all as instance variables
        self.refined_fit_params, self.fit_imgs, self.cropped_frames, self.crop_vars, self.pick_frame_indices, self.global_picks, self.fit_costs, self.times, self.snr = (
            refined_params, fit_imgs, cropped_frames, crop_vars, pick_frame_indices, global_picks, fit_costs, times, frame_snr_values)

        # Return all
        return self.refined_fit_params, self.fit_imgs, self.cropped_frames, self.crop_vars, self.pick_frame_indices, self.global_picks, self.fit_costs, self.times, self.abs_level_sums

    def saveAni(self, data_path):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import os
        import datetime

        picks = self.global_picks
        method = 'GAUSSIAN'

        fig_dir = os.path.join(data_path, "ASTRA_figures")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        # Make dest path
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = os.path.join(fig_dir, f"ASTRA_masking_{now_str}.mp4")

        fig, axs = plt.subplots(1, 3, figsize=(12, 6))
        fig.suptitle("Gaussian Fits", fontsize=16)

        n_crops = len(self.cropped_frames)

        def update(i: int) -> None:
            crop = self.cropped_frames[i]
            crop_vars = self.crop_vars[i]
            fit = self.fit_imgs[i]
            fit_params = self.refined_fit_params[i]
            snr = self.snr[i]
            x, y = self.translatePicksToGlobal(picks[i], crop_vars, global_to_local=True)
            frame_num = self.pick_frame_indices[i]
            time = self.times[i]

            # Add title indicating rejection status and index number
            fig.suptitle(f"Crop {i}:, SNR: {snr:.2f}", fontsize=14)

            # Display fit parameters at the bottom of the figure
            # Clear previous figtext annotations to avoid overlap
            for txt in fig.texts:
                txt.remove()
            plt.figtext(0.5, 0.01, f"Frame number: {frame_num}, Time: {time}, Level Sum: {fit_params[0]:.2f}, Sigma: {fit_params[1]:.2f}, "
                                   f"x0: {fit_params[2]:.2f}, y0: {fit_params[3]:.2f}, L: {fit_params[4]:.2f}",
                        ha="center", fontsize=12)

            # left: crop image
            ax1 = axs[0]
            ax1.clear()
            ax1.set_title(f"Crop {i}")
            ax1.imshow(crop, vmin=0, vmax=np.max(crop), cmap='gray')
            ax1.plot(x, y, "ro", markersize=5)

            # middle: overlay fit point
            ax2 = axs[1]
            ax2.clear()
            ax2.set_title("Gaussian Fit")
            ax2.imshow(fit, vmin=0, vmax=np.max(fit), cmap='gray')
            ax2.plot(x, y, "ro", markersize=5)

            # right: fit residuals
            abs_res = np.abs(crop - fit)
            ax3 = axs[2]
            ax3.clear()
            ax3.set_title("Fit Residuals")
            ax3.imshow(abs_res, vmin=np.min(abs_res), vmax=np.max(abs_res), cmap='coolwarm')

        def animate(i: int) -> None:
            update(i % n_crops)

        ani = animation.FuncAnimation(fig, animate, frames=n_crops, interval=1000, repeat=True)

        ani.save(dest_path, writer='ffmpeg', fps=1)

    # 2) -- Calculation/Conversion Methods --

    def checkStreakInBounds(self, global_pick, prev_length, prev_sigma, omega, directions):
        """
        Checks if the streak defined by the global pick, previous length, previous sigma, omega, and directions is within the bounds of the frames.
        Args:
            global_pick (tuple): A tuple containing the (x, y) coordinates of the global pick.
            prev_length (float): The previous length of the streak.
            prev_sigma (float): The previous sigma of the streak.
            omega (float): The angle in radians at which the streak is oriented.
            directions (tuple): A tuple containing the direction of the streak as (direction_x, direction_y).
        Returns:
            bool: True if the streak is within bounds, False otherwise.
        """

        # Define return bool
        in_bounds = True

        # Calculate front and back of the streak
        edge_pick_back = self.movePickToEdge(global_pick, omega, prev_length, directions=directions)
        edge_pick_front = self.movePickToEdge(global_pick, omega, prev_length, directions=tuple(-x for x in list(directions)))

        # Calculate the top and bottom of the streak (only check a single sigma, as long as there is clearence for center pick it is okay)
        top_pick = self.movePickToEdge(global_pick, omega+(np.pi / 2), prev_sigma, directions=(directions[0], -directions[1]))
        bottom_pick = self.movePickToEdge(global_pick, omega+(np.pi / 2), prev_sigma, directions=(-directions[0], directions[1]))

        # Package all picks into a list
        bounding_points = [edge_pick_back, edge_pick_front, top_pick, bottom_pick]

        # Check if any of the bounding points are out of bounds
        if np.any([point[0] < 0 or point[0] >= self.frames.shape[1] or point[1] < 0 or point[1] >= self.frames.shape[2] for point in bounding_points]):
            in_bounds = False

        # Check if pick is close enough to edge to be considered OOB
        edge_buffer = 3  # pixels
        if (global_pick[0] < edge_buffer or global_pick[0] >= self.frames.shape[1] - edge_buffer or
            global_pick[1] < edge_buffer or global_pick[1] >= self.frames.shape[2] - edge_buffer):
            in_bounds = False

        # Return the in_bounds bool
        return in_bounds


    def translatePicksToGlobal(self, local_pick, frame_crop_vars, global_to_local=False):
        """
        Translates local pick coordinates to global coordinates based on the frame crop variables.
        
        Args:
            local_pick (tuple): A tuple containing the local pick coordinates (x, y).
            frame_crop_vars (list): A list of tuples containing the crop variables for each frame.
        
        Returns:
            tuple: A tuple containing the global pick coordinates (x_global, y_global).
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
    

    def movePickToEdge(self, center_pick, omega, length, directions=(1, 1)):
        """
        Moves the center pick to the leading edge of the crop based on the fitted parameters.
        Args:
            center_pick (tuple): A tuple containing the x and y coordinates of the center pick.
            omega (float): The angle in radians at which the crop is oriented.
            length (float): The length of the crop in pixels.
            directions (tuple): A tuple containing the direction coefficients for the x and y axes.
        Returns:
            tuple: A tuple containing the new x and y coordinates of the pick at the leading edge.
        """

        # Unpack picks
        x0, y0 = center_pick

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
        Estimates the center of the crop based on the edge pick, angle, and length.
        Args:
            edge_pick (tuple): A tuple containing the x and y coordinates of the edge pick.
            angle (float): The angle in radians at which the crop is oriented.
            length (list): A list containing the lengths of the crop in the x and y directions.
            directions (tuple): A tuple containing the direction coefficients for the x and y axes.
        Returns:
            tuple: A tuple containing the estimated x and y coordinates of the center of the crop.
        """

        # Invert directions
        inverted_directions = (-directions[0], -directions[1])

        # Appy inverted direction to movePickToEdge
        center_pick = self.movePickToEdge(edge_pick, omega, length, inverted_directions)

        # Return the center pick
        return center_pick
    

    def calculateAdaptiveVelocityClamp(self, bounds):
        """
        Calculates the velocity clamp to be based on a fraction of the parameter domain.
        Args:
            bounds (list): A list of tuples containing the lower and upper bounds for each parameter.
        Returns:
            tuple: A tuple containing the lower and upper bounds for the velocity clamp.
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
        Calculates the adaptive bounds for the parameters based on the first pass results.

        Args:
            first_pass_parameters (np.ndarray): The parameters from the first pass.
            scipy_format (optional, bool): Whether to return the bounds in a format compatible with SciPy. Defaults to PySwarms format.
        Returns:
            tuple: Tuple object compatiable with PySwarms or SciPy, containing the lower and upper bounds for each parameter.
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
        Estimates the next center of the crop based on the current global center, length, and angle.
        Args:
            current_global_center (tuple): A tuple containing the x and y coordinates of the current global center.
            length (float): The length of the fit (or norm to next center)
            omega (float): The angle in radians at which the crop is oriented.
            directions (tuple): A tuple containing the direction coefficients for the x and y axes.
        Returns:
            tuple: A tuple containing the estimated x and y coordinates of the next center of the crop.
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


    def calculateSNR(self, fit_img, frame, cropped_frame, crop_vars):
        # DEPRECATED
        """
        Calculates the Signal-to-Noise Ratio (SNR) for a given fit image and frame.
        Args:
            fit_img (numpy.ndarray): The fit image containing the fitted gaussian.
            frame (numpy.ndarray): The original frame from which the fit image was derived.
            cropped_frame (numpy.ndarray): The cropped frame containing the region of interest.
            crop_vars (tuple): A tuple containing the crop variables (x_min, x_max, y_min, y_max).
        Returns:
            float: The calculated SNR value.
        """
        
        # Round crop variables to integers for indexing
        _, _, x_min, x_max, y_min, y_max = map(int, crop_vars.copy())

        # Copy fit image, cropped_frame, and avepixel_background to avoid modifying the original data
        fit_img = fit_img.copy()
        cropped_frame = cropped_frame.copy()
        avebk = self.avepixel_background.copy()

        # Clip fit image to zero and one
        fit_img[fit_img <= 1] = 0
        fit_img[fit_img > 1] = 1

        # Mask cropped frame with fit image to remove the background
        masked_cropped = fit_img * cropped_frame

        # Invert fit_img: 1 becomes 0, 0 becomes 1
        inverted_fit_img = 1 - fit_img.copy()

        # Mask the avebk with the inverted fit image to remove the streak
        masked_avebk = inverted_fit_img * avebk[y_min:y_max, x_min:x_max]

        # Calculate the median and standard deviation of the masked average background
        median_ave = np.median(masked_avebk[masked_avebk > 0])
        std_ave = np.std(masked_avebk[masked_avebk > 0])

        # Crop the masked frame to the percentile threshold defined by P_thresh
        masked_cropped[masked_cropped < np.percentile(masked_cropped, float(self.astra_config['astra']['photom_thresh']))] = 0

        # Count the non-zero values in the masked cropped frame, and calculate the level sum
        nonzero_count = np.count_nonzero(masked_cropped)
        level_sum = np.ma.sum(masked_cropped)

        # Find all indices of non-zero values for the photometry pixels
        nonzero_indices = np.argwhere(masked_cropped > 0)
        
        # Convert photometry pixels to global coordinates and adjust indices
        nonzero_indices[:, 0] += y_min
        nonzero_indices[:, 1] += x_min
        photometry_pixels = [tuple(idx[::-1]) for idx in nonzero_indices]

        # Check if any of the photometry pixels are saturated
        saturated_bool = np.any(frame[photometry_pixels] >= self.saturation_threshold)

        # Append data to instance variables
        self.photometry_pixels.append(photometry_pixels)
        self.saturated_bool_list.append(saturated_bool)
        self.abs_level_sums.append(level_sum)
        self.background_levels.append(median_ave)

        # Print debug information if verbose is enabled
        if self.verbose:
            print(f"Level sum: {level_sum}, STD ave: {std_ave}, Nonzero count: {nonzero_count}, Background std: {median_ave}, Saturated: {saturated_bool}")

        # Finally calculate SNR
        return signalToNoise(
            level_sum,
            nonzero_count,
            median_ave,
            std_ave
    )


    # 3) -- Helper Methods --

    def updateParameterEstimationFunctions(self, crop_vars, first_pass_params, forward_pass):
        """
        Updates the parameter estimation functions based on the current crop variables and first pass parameters.
        Args:
            crop_vars (numpy.ndarray): The crop variables for the current frame.
            first_pass_params (numpy.ndarray): The parameters from the first pass.
            forward_pass (bool): Whether the update is for a forward pass or not.
        Returns:
            dict: A dictionary containing the updated parameter estimation functions.
        """ 

        # convert all_params & crop_vars to a numpy array
        all_params = first_pass_params.copy()
        xymin = crop_vars.copy()[:, [2, 4]]

        # Unpack parameters
        level_sums, heights, lengths = all_params[:, 0], all_params[:, 1], all_params[:, 4]

        # Calculate magnitudes of all norms
        global_centroids = all_params[:, 2:4] + xymin

        # NOTE : sorting is likely not needed, but it is here to ensure the splines are calculated in the correct order
        global_centroids = global_centroids[np.argsort(global_centroids[:, 0])]
        deltas = global_centroids[1:] - global_centroids[:-1]
        norms = np.linalg.norm(deltas, axis=1)

        # Ensure level_sums has only values above zero
        level_sums = np.clip(level_sums, 1e-6, None)

        # Translate level_sum into log space
        level_sums = np.log(level_sums)

        # Use moving linear method for level_sum (Use three last point to estimate next)
        paramter_estimation_functions = {
            'level_sum': np.polynomial.Polynomial.fit((np.array([0, 1, 2]) + (len(level_sums) - 3)), level_sums[-3:], 1) if forward_pass else np.polynomial.Polynomial.fit(range(3), level_sums[:3], 1),
            'height': np.polynomial.Polynomial.fit(range(len(heights)), heights, 1),
            'length':  np.polynomial.Polynomial.fit((np.array([0, 1, 2]) + (len(lengths) - 3)), lengths[-3:], 1) if forward_pass else np.polynomial.Polynomial.fit(range(3), lengths[:3], 1),
            'norm' : np.polynomial.Polynomial.fit(range(len(norms)), norms, 1)
        }

        return paramter_estimation_functions


    def estimateNextParameters(self, paramter_estimation_functions, n_points, forward_pass):
        """
        Estimates the next parameters based on the current parameter estimation functions.
        Args:
            paramter_estimation_functions (dict): The current parameter estimation functions.
            n_points (int): The number of points in the current frame.
            forward_pass (bool): Whether the estimation is for a forward pass or not.
        Returns:
            dict: A dictionary containing the estimated next parameters.
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


    def LocalGaussianFit(self, p0, obs_frame, omega, bounds, directions):

        # Define initial parameters
        y, x = np.indices(obs_frame.shape)
        data_tuple = (x, y)
        y_obs = obs_frame.ravel()

        # Instantiate optimizer
        res = scipy.optimize.minimize(
            fun=self.LocalObjectiveFunction,
            x0=p0,  # Exclude omega from initial guess
            args=(data_tuple, y_obs, 0, omega, bounds, directions),
            method=self.second_pass_settings["method"],
            bounds=bounds
        )

        # If unsuccesful revert to initial guess
        if res.success is False:
            print(f"Warning: Local optimization failed with message: {res.message}. Reverting to initial guess.")
            best_pos = p0
            best_cost = self.LocalObjectiveFunction(p0, data_tuple, y_obs, 0, omega, bounds, directions)
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

        
    def GaussianPSO(self, cropped_frame, omega, directions, estim_next_params=None, init_length=None):
        """
        Performs a Particle Swarm Optimization (PSO) to fit a moving Gaussian to the cropped frame.
        Args:
            cropped_frame (numpy.ndarray): The cropped frame to fit the Gaussian to.
            omega (float): The angle in radians at which the Gaussian is oriented.
            estim_next_params (dict, optional): A dictionary containing the estimated next parameters for the Gaussian.
                                                If None, initial parameters will be estimated from the cropped frame.
            init_length (float, optional): The initial length of the Gaussian. If None, it will be estimated from the cropped frame.
        Returns:
            tuple: A tuple containing:
                - best_pos (numpy.ndarray): The best position found by the PSO.
                - best_cost (float): The best cost found by the PSO.
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
                raise ValueError("Either spline_estim or init_length must be provided for first_pass_gaussian.")
        
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
                objective_func = self.PSOObjectiveFunction,
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
        # NOTE: Change to actual warnings.warning later
        for i, (val, lower, upper) in enumerate(zip(best_pos, bounds[0], bounds[1])):
            if val < lower or val > upper:
                print(f"Warning: Parameter {i} (value: {val}) is outside the bounds [{lower}, {upper}]")

        return best_pos, best_cost


    def PSOObjectiveFunction(self, params, data_tuple, y_obs, a0, omega, bounds, directions):
        """ 
        Objective function for the PSO optimization, calculating the residuals based on the moving Gaussian fit.
        Args:
            params (numpy.ndarray): The parameter positions of each particle.
            data_tuple (tuple): A tuple containing the x and y indices of the data.
            y_obs (numpy.ndarray): The observed data to fit against.
            a0 (float): The initial amplitude of the Gaussian.
            omega (float): The angle in radians at which the Gaussian is oriented.
            bounds (tuple): A tuple containing the lower and upper bounds for each parameter.
        Returns:
            numpy.ndarray: The residuals for each particle.
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

            if front[0] < 0 or front[0] >= data_tuple[0].shape[1] or front[1] < 0 or front[1] >= data_tuple[0].shape[0]:
                residuals[i] += self.first_pass_settings["oob_penalty"]
            if back[0] < 0 or back[0] >= data_tuple[0].shape[1] or back[1] < 0 or back[1] >= data_tuple[0].shape[0]:
                residuals[i] += self.first_pass_settings["oob_penalty"]

        return residuals
    

    def LocalObjectiveFunction(self, params, data_tuple, y_obs, a0, omega, bounds, directions):
        """
        Objective function for the local optimization, calculating the residuals based on the moving Gaussian fit.
        Args:
            params (numpy.ndarray): The parameters for the moving Gaussian fit.
            data_tuple (tuple): A tuple containing the x and y indices of the data.
            y_obs (numpy.ndarray): The observed data to fit against.
            a0 (float): The initial amplitude of the Gaussian.
            omega (float): The angle in radians at which the Gaussian is oriented.
            bounds (tuple): A tuple containing the lower and upper bounds for each parameter.
            directions (tuple): A tuple containing the direction coefficients for the x and y axes.
        Returns:
            float: The residuals for the moving Gaussian fit."""
            
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

        if front[0] < 0 or front[0] >= data_tuple[0].shape[1] or front[1] < 0 or front[1] >= data_tuple[0].shape[0]:
            residuals += self.second_pass_settings["oob_penalty"]
        if back[0] < 0 or back[0] >= data_tuple[0].shape[1] or back[1] < 0 or back[1] >= data_tuple[0].shape[0]:
            residuals += self.second_pass_settings["oob_penalty"]

        return residuals


    def cropFrameToGaussian(self, sub_frame, est_global_center, max_sigma, max_length, omega, cropping_settings=None):
        """
        Crops a frame to a gaussian based on the largest possible values within the optimizer.
        Args:
            sub_frame (numpy.ndarray): The subtracted frame to be cropped.
            est_global_center (tuple): A tuple containing the estimated global center coordinates (x, y).
            max_sigma (float): The maximum sigma value for the gaussian.
            max_length (float): The maximum length of the gaussian.
        Returns:
            tuple: A tuple containing:
                - cropped_frame (numpy.ndarray): The cropped frame.
                - crop_vars (list): A list containing the crop variables (cx, cy, xmin, xmax, ymin, ymax).
        """

        # Copy subframe to avoid reference errors
        sub_frame = sub_frame.copy()

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
        optim_mask = self.movingGaussian(data_tuple, omega, 0, 1e6, max_sigma, est_global_center[0], est_global_center[1], max_length, saturation_level=None)
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

            The original equation given in the paper has a typo in the exp term, after sin(omega) there shoudl be 
            a minus, not a plus.


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

        saturation_level = self.saturation_threshold
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
        Generate initial particles for the optimization process.
        Args:
            bounds (tuple): A tuple containing the lower and upper bounds for each parameter.
            n_particles (int): The number of particles to generate.
            p0 (numpy.ndarray, optional): A 1D array of initial particle positions, for normalized distribution.
        Returns:
            numpy.ndarray: A 2D array of shape (n_particles, len(bounds)) containing the initial particle positions.
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


    def recursiveCroppingAlgorithm(self, frame_index, est_center_global, paramter_estimation_functions, omega, directions, forward_pass = False):
        """
        Recursive cropping algorithm to refine the meteor crops based on the estimated parameters.
        Args:
            frame_index (int): The index of the current frame to process.
            est_center_global (tuple): The estimated global center coordinates (x, y) for the current frame.
            paramter_estimation_functions (dict): The current parameter estimation functions.
            omega (float): The angle in radians at which the crop is oriented.
            directions (tuple): A tuple containing the direction coefficients for the x and y axes.
            forward_pass (bool): Whether the recursion is for a forward pass or not.
            use_DetApp (bool): Whether to use DetApp for frame selection.
        Returns:
            None: The function modifies the instance variables directly."""

        if frame_index > max(self.pick_frame_indices) or frame_index < min(self.pick_frame_indices):
            return

        # Estimate next parameters using the parameter estimation functions
        est_next_params = self.estimateNextParameters(paramter_estimation_functions, self.first_pass_params.shape[0], forward_pass=forward_pass)

        # Crop the frame around the new center
        cropped_frame, crop_vars = self.cropFrameToGaussian(self.subtracted_frames[frame_index], est_center_global, 
                                                                    est_next_params['height'] * self.cropping_settings['max_sigma_coeff'], 
                                                                    est_next_params['length'] * self.cropping_settings['max_length_coeff'],
                                                                    omega)
        
        # Run a PSO on the cropped frame
        best_fit, _ = self.GaussianPSO(cropped_frame, omega, directions, estim_next_params=est_next_params)

        # Update instance variables with gaussian parameters
        # NOTE: if there is to be no DetApp reliance pick_frame_indices also needs to be updated to include the frame_index
        if forward_pass:
            self.cropped_frames.append(cropped_frame)
            self.crop_vars = np.vstack([self.crop_vars, crop_vars])
            self.first_pass_params = np.vstack([self.first_pass_params, best_fit])
            if frame_index not in self.pick_frame_indices:
                self.pick_frame_indices.append(frame_index)
                self.pick_frame_indices = sorted(self.pick_frame_indices)


        else:
            self.cropped_frames.insert(0,cropped_frame)
            self.crop_vars = np.vstack([crop_vars, self.crop_vars])
            self.first_pass_params = np.vstack([best_fit, self.first_pass_params])
            if frame_index not in self.pick_frame_indices:
                self.pick_frame_indices.insert(0, frame_index)
                self.pick_frame_indices = sorted(self.pick_frame_indices)

        # Update estimation functions off new parameters
        parameter_estimation_functions = self.updateParameterEstimationFunctions(self.crop_vars, self.first_pass_params, forward_pass=forward_pass)

        # Determine the next center
        global_best_fit_center = self.translatePicksToGlobal(
            (best_fit[2], best_fit[3]),
            crop_vars
        )

        next_center_global = self.estimateNextCenter(
            global_best_fit_center,
            est_next_params['norm'],
            omega,
            directions=directions
        )

        # Quit condition
        pass_coeff = 1 if forward_pass else -1

        # Update progress
        self.progressed_frames['cropping'] += 1
        self.updateProgress()
        if self.verbose:
            print(f"Recursive cropping at frame {frame_index} with center {est_center_global} and next center {next_center_global}, Forward pass: {forward_pass}")
            print(f' Frame index: {frame_index}, Est. Center: {est_center_global}, Next Center: {next_center_global}, Pass Coeff: {pass_coeff}')


        # Recurse
        self.recursiveCroppingAlgorithm(frame_index + pass_coeff,
                                        next_center_global,
                                        parameter_estimation_functions,
                                        omega,
                                        directions=directions,
                                        forward_pass=forward_pass,
                                        )
    

    def applyKalmanFilter(self, global_edge_picks, times):
        
        # Prepare the measurements and noise covariance matrices
        measurements = np.array(global_edge_picks)
        times = np.array(times)

        # Normalize times
        t0 = times[0]  # Use the first time as the reference
        normalized_times = np.array([
            (t - t0).total_seconds()
            for t in times
        ])
        self.exec_count += 1
        self.updateProgress()

        R = self.computeR(measurements, normalized_times)
        Q_base = self.computeQBase(measurements, normalized_times)
        self.exec_count += 2
        self.updateProgress()
    
        # Call the Kalman filter function
        x_smooth, p_smooth = self.kalmanFilterCA(
            measurements,
            normalized_times,
            Q_base=Q_base,
            R=R,
            monotonicity=self.kalman_settings['monotonicity'],
            use_accel=self.kalman_settings['use_accel']
        )

        if self.kalman_settings['save results']:
            self.saveKalmanUncertaintiesToCSV(self.data_path, normalized_times, measurements, x_smooth, p_smooth)

        self.smoothed_picks = x_smooth[:, :2]  # Extract smoothed positions (x, y)
        self.smoothed_covars = p_smooth[:, :2, :2]  # Extract smoothed position covariances

        return self.smoothed_picks, self.smoothed_covars


    def computeQBase(self, measurements, times):
        # TRANSlATE ALL INTO PIXELS / SECOND
        # px / frame * frame/seconds

        # Calculate average velocity (px/s)
        avrg_velocity = np.mean(np.diff(np.linalg.norm(measurements, axis=1)) / np.diff(times))  # px/s

        # Instantiate process noise covariance
        sigma_vxy = self.kalman_settings['sigma_vxy_perc']/100 * avrg_velocity
        sigma_xy = self.kalman_settings['sigma_xy'] 

        Q_base = np.array([
            [sigma_xy**2, 0 ,0 ,0 ,0 ,0],
            [0, sigma_xy**2, 0 ,0 ,0 ,0],
            [0, 0, sigma_vxy**2, 0 ,0 ,0],
            [0, 0, 0, sigma_vxy**2, 0 ,0],
            [0, 0, 0, 0, 1, 0], #Default accel to 1 since it will be ignoreed in Use_Accel=False
            [0, 0, 0, 0, 0, 1]
        ])

        # save as instance var for later saving
        self.Q_base = Q_base

        return Q_base
    

    def computeR(self, measurements, times):

        # fit x and y vs time to a quadratic
        x_fit = np.polyfit(times, measurements[:, 0], 2)
        y_fit = np.polyfit(times, measurements[:, 1], 2)

        # Calculate all residuals between fit and measurements
        x_residuals = measurements[:, 0] - np.polyval(x_fit, times)
        y_residuals = measurements[:, 1] - np.polyval(y_fit, times)

        # Calculate the STD of residuals
        x_std = np.std(x_residuals)
        y_std = np.std(y_residuals)

        # Form R as a 2x2 array of diagonal variances
        R = np.array([[x_std**2, 0], 
                      [0, y_std**2]])
        
        # Save as instance var for later saving
        self.R = R

        return R

    def kalmanFilterCA(self, measurements, times, Q_base, R, monotonicity, use_accel, epsilon=1e-6):
        """
        Kalman filter with Rauch–Tung–Striebel (RTS) smoothing using a constant-acceleration model.
        Handles irregular time intervals between measurements by recomputing dynamics at each step.

        Arguments:
        measurements: [ndarray] Nx2 array of (x, y) position observations.
        times: [ndarray] N-length array of observation timestamps (monotonically increasing).
        Q_base: [ndarray] 6x6 baseline process noise covariance matrix (assumed per unit time).
        R: [ndarray] 2x2 measurement noise covariance matrix (for x and y).

        Keyword arguments:
        monotonicity: [bool] If True, enforces monotonic motion along dominant axis (optional constraint).
        epsilon: [float] Threshold to prevent false violations due to floating-point error. Used for enforcing monotonicity.
        use_accel: [bool] If True, enables acceleration estimation instead of constant velocity.

        Return:
        x_smooth, P_smooth: [tuple of ndarrays]
        - x_smooth: Nx6 array of smoothed state vectors [x, y, vx, vy, ax, ay].
        - P_smooth: Nx6x6 array of smoothed state covariance matrices.
        """
        N = len(measurements)  # Number of time steps
        H = np.array([
            [1, 0, 0, 0, 0, 0],  # x position
            [0, 1, 0, 0, 0, 0]   # y position
        ])
        I = np.eye(6)  # Identity matrix for 6D state space
        x_pred = np.zeros((N, 6))  # Predicted state (prior)
        P_pred = np.zeros((N, 6, 6))  # Predicted covariance
        x_forward = np.zeros((N, 6))  # Filtered state (posterior)
        P_forward = np.zeros((N, 6, 6))  # Filtered covariance

        # --- INITIAL STATE ESTIMATION ---
        if N >= 3:
            dt_est = (times[2] - times[0]) / 2
            vx0 = (measurements[2, 0] - measurements[0, 0]) / (2 * dt_est)
            vy0 = (measurements[2, 1] - measurements[0, 1]) / (2 * dt_est)
            ax0 = (measurements[2, 0] - 2 * measurements[1, 0] + measurements[0, 0]) / (dt_est ** 2)
            ay0 = (measurements[2, 1] - 2 * measurements[1, 1] + measurements[0, 1]) / (dt_est ** 2)
        else:
            dt_est = times[1] - times[0]
            vx0 = (measurements[1, 0] - measurements[0, 0]) / dt_est
            vy0 = (measurements[1, 1] - measurements[0, 1]) / dt_est
            ax0 = 0.0
            ay0 = 0.0

        x0 = np.array([measurements[0, 0], measurements[0, 1], vx0, vy0, ax0, ay0])
        P0 = np.zeros((6, 6))
        P0[0:2, 0:2] = R
        P0[2:4, 2:4] = (2 * R) / (dt_est ** 2)
        P0[4:6, 4:6] = (6 * R) / (dt_est ** 4)

        x_est = x0.copy()
        P_est = P0.copy()
        x_forward[0] = x0
        P_forward[0] = P0
        x_pred[0] = x0
        P_pred[0] = P0

        # -------- FORWARD KALMAN FILTER PASS --------
        for k in range(1, N):
            dt = times[k] - times[k - 1]
            if use_accel:
                A = np.array([
                    [1, 0, dt, 0, 0.5 * dt ** 2, 0],
                    [0, 1, 0, dt, 0, 0.5 * dt ** 2],
                    [0, 0, 1, 0, dt, 0],
                    [0, 0, 0, 1, 0, dt],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]
                ])
            else:
                A = np.array([
                    [1, 0, dt, 0, 0, 0],
                    [0, 1, 0, dt, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]
                ])

            Q = Q_base * dt
            x_pred[k] = A @ x_est
            P_pred[k] = A @ P_est @ A.T + Q
            z = measurements[k]
            y = z - H @ x_pred[k]
            S = H @ P_pred[k] @ H.T + R
            K = P_pred[k] @ H.T @ np.linalg.solve(S, np.eye(2))
            x_est = x_pred[k] + K @ y
            P_est = (I - K @ H) @ P_pred[k]
            x_forward[k] = x_est
            P_forward[k] = P_est

        # -------- BACKWARD PASS (RTS SMOOTHING) --------
        x_smooth = np.zeros_like(x_forward)
        P_smooth = np.zeros_like(P_forward)
        x_smooth[-1] = x_forward[-1]
        P_smooth[-1] = P_forward[-1]

        for k in range(N - 2, -1, -1):
            dt = times[k + 1] - times[k]
            A = np.array([
                [1, 0, dt, 0, 0.5 * dt ** 2, 0],
                [0, 1, 0, dt, 0, 0.5 * dt ** 2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            G = P_forward[k] @ A.T @ np.linalg.inv(P_pred[k + 1])
            x_smooth[k] = x_forward[k] + G @ (x_smooth[k + 1] - x_pred[k + 1])
            P_smooth[k] = P_forward[k] + G @ (P_smooth[k + 1] - P_pred[k + 1]) @ G.T

        # -------- OPTIONAL MONOTONICITY ENFORCEMENT --------
        if monotonicity:
            dx_total = x_smooth[-1, 0] - x_smooth[0, 0]
            dy_total = x_smooth[-1, 1] - x_smooth[0, 1]
            if abs(dx_total) >= abs(dy_total):
                dominant = 0
                velocity_idx = 2
            else:
                dominant = 1
                velocity_idx = 3
            direction = np.sign(x_smooth[-1, dominant] - x_smooth[0, dominant])
            for k in range(1, N):
                if direction > 0 and x_smooth[k, dominant] + epsilon < x_smooth[k - 1, dominant]:
                    x_smooth[k, dominant] = x_smooth[k - 1, dominant]
                    x_smooth[k, velocity_idx] = 0.0
                elif direction < 0 and x_smooth[k, dominant] - epsilon > x_smooth[k - 1, dominant]:
                    x_smooth[k, dominant] = x_smooth[k - 1, dominant]
                    x_smooth[k, velocity_idx] = 0.0

        return x_smooth, P_smooth


    def updateProgress(self, progress=None):
        """
        Calculates approx. progress based on either Gaussian or Kalman mode. Updates the progress callback.
        The progress is calculated based on the number of frames processed in each step and the total number of frames.
        The weights for each step are defined in the time_weights_gaus dictionary.
        """
        time_weights_gaus = {
            'cropping' : 0.7,
            'refining' : 0.2,
            'removing' : 0.1
        }

        if progress is not None and self.progress_callback is not None:
            self.progress_callback(int(progress))

        if self.mode == 'Gaussian':
            if self.progress_callback is not None:
                current_percentage = sum(
                    self.progressed_frames[step] * time_weights_gaus[step]
                    for step in self.progressed_frames.keys()
                ) / self.total_frames * 100
                self.progress_callback(int(current_percentage))
        if self.mode == 'Kalman' is not None:
            if self.progress_callback is not None:
                current_percentage = (self.exec_count / self.total_exec) * 100
                self.progress_callback(int(current_percentage))

    def select_seed_triplet(self, picks, pick_frame_indices):
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
        # Use lexsort with last key as primary: sort by (pref1, dist, starts)
        # lexsort sorts by last key first, so stack in reverse priority order:
        idx = np.lexsort((starts, dist, pref1))[0]
        start = starts[idx]

        seed_pick_frame_indices = keys[start:start+3]
        seed_picks = p_sorted[start:start+3]
        return seed_picks, seed_pick_frame_indices

    def computeIntensitySum(self, photom_pixels, global_centroid, raw_frame):
        """ Compute the background subtracted sum of intensity of colored pixels. The background is estimated
            as the median of near pixels that are not colored.
        """

        x_arr, y_arr = np.array(photom_pixels).T
        raw_frame = raw_frame.copy()
        avepixel = self.avepixel_background.copy()

        # Take a window twice the size of the colored pixels
        x_color_size = np.max(x_arr) - np.min(x_arr)
        y_color_size = np.max(y_arr) - np.min(y_arr)

        x_min = int(global_centroid[0] - x_color_size)
        x_max = int(global_centroid[0] + x_color_size)
        y_min = int(global_centroid[1] - y_color_size)
        y_max = int(global_centroid[1] + y_color_size)

        # Limit the size to be within the bounds
        if x_min < 0: x_min = 0
        if x_max > raw_frame.shape[1]: x_max = raw_frame.shape[1]
        if y_min < 0: y_min = 0
        if y_max > raw_frame.shape[0]: y_max = raw_frame.shape[0]

        # Take only the colored part
        mask_img = np.ones_like(raw_frame)
        mask_img[y_arr, x_arr] = 0
        masked_img = np.ma.masked_array(raw_frame, mask_img)
        crop_img = masked_img[y_min:y_max, x_min:x_max]

        # Mask out the colored in pixels
        mask_img_bg = np.zeros_like(raw_frame)
        mask_img_bg[y_arr, x_arr] = 1

        # Take the image where the colored part is masked out and crop the surroundings
        masked_img_bg = np.ma.masked_array(raw_frame, mask_img_bg)
        crop_bg = masked_img_bg[y_min:y_max, x_min:x_max]
        
        # Mask out the colored in pixels
        avepixel_masked = np.ma.masked_array(avepixel, mask_img_bg)
        avepixel_crop_no_color = avepixel_masked[y_min:y_max, x_min:x_max]
        avepixel_crop_color = np.ma.masked_array(avepixel, mask_img)[y_min:y_max, x_min:x_max]

        # Compute background level
        background_lvl = np.ma.median(avepixel_crop_no_color)

        # Subtract the avepixel crop from the data crop, clip the negative values to 0 and sum up the intensity
        crop_img_nobg = crop_img.astype(float) - avepixel_crop_color.astype(float)
        crop_img_nobg = np.clip(crop_img_nobg, 0, None)
        intensity_sum = np.ma.sum(crop_img_nobg)

        # Check if the result is masked
        if np.ma.is_masked(intensity_sum):
            # If the result is masked (i.e. error reading pixels), set the intensity sum to 1
            print("Warning: intensity sum is masked, setting to 1") #Fallback to regular method, ENSURE NEVER 0
            intensity_sum = 1
        else:
            intensity_sum = intensity_sum.astype(int)

        ### Measure the SNR of the pick ###

        # Compute the standard deviation of the background
        background_stddev = np.ma.std(crop_bg)

        # Count the number of pixels in the photometric area
        source_px_count = np.ma.sum(~crop_img.mask)

        # Compute the signal to noise ratio using the CCD equation
        snr = signalToNoise(intensity_sum, source_px_count, background_lvl, background_stddev)

        ### Determine if there is any saturation in the measured photometric area

        # Compute the saturation threshold
        saturation_threshold = int(0.98*(2**self.skyfit_config.bit_depth))

        # If at least 2 pixels are saturated in the photometric area, mark the pick as saturated
        if np.sum(crop_img > saturation_threshold) >= 2:
            saturated_bool = True
        else:
            saturated_bool = False

        # Append values to class arrays
        self.photometry_pixels.append(photom_pixels)
        self.saturated_bool_list.append(saturated_bool)
        self.abs_level_sums.append(intensity_sum)
        self.background_levels.append(background_lvl)

        if self.verbose:
            print("SNR update on frame {:2d}: intensity sum = {:8d}, source px count = {:5d}, background lvl = {:8.2f}, background stddev = {:6.2f}, SNR = {:.2f}".format(
                self.pick_frame_indices[self.temp_count], intensity_sum, source_px_count, background_lvl, background_stddev, snr))
            self.temp_count += 1

        # return SNR
        return snr


    def computePhotometryPixels(self, fit_img, cropped_frame, crop_vars):
        """
        Compute all pixels to be included in photometry, by thresholding pixel intensity over a certain percentile the fit image.
        args:
            fit_img: The fit image to use for photometry.
            cropped_frame: The cropped frame to use for photometry.
            crop_vars: The crop variables to use for photometry.
        returns:
            A list of tuples representing the coordinates of the photometry pixels.
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

        masked_cropped[masked_cropped < np.percentile(masked_cropped, float(self.astra_config['astra']['photom_thresh']))] = 0

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

    def saveKalmanUncertaintiesToCSV(self, data_path, times, measurements, x_smooth, p_smooth):
        import os
        import datetime
        import csv

        fig_dir = os.path.join(data_path, "ASTRA_Kalman_Results")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        # Make dest path
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        data = {"time (sec since start)" : [],
                "original x" : [],
                "original y" : [],
                "smoothed x" : [],
                "smoothed y" : [],
                "smoothed vx" : [],
                "smoothed vy" : [],
                "smoothed ax" : [],
                "smoothed ay" : [],
                "kalman uncertainty/STD (x)" : [],
                "kalman uncertainty/STD (y)" : [],
                "kalman uncertainty/STD (vx)" : [],
                "kalman uncertainty/STD (vy)" : [],
                "kalman uncertainty/STD (ax)" : [],
                "kalman uncertainty/STD (ay)" : []
                }
         
        #  populate data dict
        for i in range(len(times)):
            data['time (sec since start)'].append(times[i])
            data['original x'].append(measurements[i][0])
            data['original y'].append(measurements[i][1])
            data['smoothed x'].append(x_smooth[i][0])
            data['smoothed y'].append(x_smooth[i][1])
            data['smoothed vx'].append(x_smooth[i][2])
            data['smoothed vy'].append(x_smooth[i][3])
            data['smoothed ax'].append(x_smooth[i][4])
            data['smoothed ay'].append(x_smooth[i][5])
            data['kalman uncertainty/STD (x)'].append(np.sqrt(p_smooth[i][0][0]))
            data['kalman uncertainty/STD (y)'].append(np.sqrt(p_smooth[i][1][1]))
            data['kalman uncertainty/STD (vx)'].append(np.sqrt(p_smooth[i][2][2]))
            data['kalman uncertainty/STD (vy)'].append(np.sqrt(p_smooth[i][3][3]))
            data['kalman uncertainty/STD (ax)'].append(np.sqrt(p_smooth[i][4][4]))
            data['kalman uncertainty/STD (ay)'].append(np.sqrt(p_smooth[i][5][5]))

        # Add header row for Q_base and R std values
        std_q_base = np.sqrt(np.diag(self.Q_base))
        std_r = np.sqrt(np.diag(self.R))
        header_q_base = ["Q_base STD (x)", "Q_base STD (y)", "Q_base STD (vx)", "Q_base STD (vy)", "Q_base STD (ax)", "Q_base STD (ay)"]
        header_r = ["R STD (x)", "R STD (y)"]

        # Write header row to CSV before the main data
        csv_path = os.path.join(fig_dir, f"kalman_results_{now_str}.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write Q_base std header and values
            writer.writerow(header_q_base)
            writer.writerow([f"{v:.6f}" for v in std_q_base])
            # Write R std header and values
            writer.writerow(header_r)
            writer.writerow([f"{v:.6f}" for v in std_r])
            # Write main data header
            writer.writerow(list(data.keys()))
            # Write main data rows
            for i in range(len(times)):
                writer.writerow([data[key][i] for key in data.keys()])

        print(f"Kalman results saved to {csv_path}")
