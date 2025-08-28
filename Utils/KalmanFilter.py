import os
import csv
import datetime
from datetime import datetime as dt
import shutil

import numpy as np
import pandas as pd

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.Conversions import trueRaDec2ApparentAltAz



class KalmanFilter():

    def __init__(self, sigma_xy, perc_sigma_vxy, measurements=None, times=None, platepar=None,
                 epsilon=1e-6, monotonicity=True, use_accel=False,
                 data_path=None, ecsv_save_path=None,
                 save_stats_results=False, use_all_files=False):

        # Check validity of passed on args

        # Check sigma_xy
        try:
            sigma_xy = float(sigma_xy)
        except (ValueError, TypeError):
            raise ValueError("Invalid sigma_xy, must be castable to float")
        if sigma_xy <= 0:
            raise ValueError("Invalid sigma_xy, must be positive")

        # Check perc_sigma_vxy
        try:
            perc_sigma_vxy = float(perc_sigma_vxy)
        except (ValueError, TypeError):
            raise ValueError("Invalid perc_sigma_vxy, must be castable to float")
        if perc_sigma_vxy <= 0:
            raise ValueError("Invalid perc_sigma_vxy, must be a positive percentage")
        
        # Check epsilon
        try:
            epsilon = float(epsilon)
        except (ValueError, TypeError):
            raise ValueError("Invalid epsilon, must be castable to float")
        if epsilon <= 0:
            raise ValueError("Invalid epsilon, must be positive")

        # Cast bool-like args
        self.monotonicity = str(monotonicity).lower() == 'true'
        self.use_accel = str(use_accel).lower() == 'true'
        self.use_all_files = str(use_all_files).lower() == 'true'

        # Check save_path
        if ecsv_save_path is not None:
            if not os.path.isdir(ecsv_save_path):
                raise ValueError("ecsv_save_path must be a valid directory")
            self.ecsv_save_path = ecsv_save_path

        # Use measurements and times directly
        if measurements is not None or times is not None:
            USE_FILE = False
        
        # Else use data, check if valid file or directory with valid file
        elif data_path is not None:
            if os.path.isfile(data_path):
                # Single file provided
                if not data_path.endswith('.ecsv'):
                    raise ValueError("File must have .ecsv extension")
                else:
                    # Add to a list
                    data_path = [data_path]
            elif os.path.isdir(data_path):
                # Directory provided, find all .ecsv files
                ecsv_files = []
                for root, dirs, files in os.walk(data_path):
                    for file in files:
                        if file.endswith('.ecsv'):
                            ecsv_files.append(os.path.join(root, file))
                
                if not ecsv_files:
                    raise ValueError("No .ecsv files found in directory")
                
                if use_all_files:
                    data_path = ecsv_files
                elif len(ecsv_files) == 1:
                    data_path = ecsv_files[0]
                else:
                    raise ValueError("Found multiple .ecsv files in directory. Either give path to single file or set use_all_files=True.")
            else:
                raise ValueError("data_path must be a valid file or directory")
            
            USE_FILE = True

        # load measurements and times from files
        if USE_FILE:
            measurements = []
            times = []
            if len(data_path) > 1:
                for path in data_path:
                    t, m = self.loadECSV(path)
                    times.append(t)
                    measurements.append(m)
            else:
                t, m = self.loadECSV(data_path)
                times.append(t)
                measurements.append(m)

        # Set class attributes
        self.measurements = measurements if len(np.array(measurements).shape) != 2 else [measurements]
        self.times = times if len(np.array(times).shape) != 1 else [times]
        self.sigma_xy = sigma_xy
        self.perc_sigma_vxy = perc_sigma_vxy
        self.monotonicity = monotonicity
        self.use_accel = use_accel
        self.epsilon = epsilon
        self.save_stats_results = save_stats_results
        self.data_path = data_path  
        self.platepar = platepar


        # Determine stats save path
        if self.save_stats_results:
            if data_path is not None:
                self.stats_save_path = [os.path.dirname(path) if os.path.isfile(path) else path for path in data_path]
            else:
                raise ValueError("data_path must be provided if saving stats results, even if measurements and times were passed on.")

    def run(self):

        x_smooth = []
        p_smooth = []
        Q_base = []
        R = []
        norm_times = []

        # Run kalman filters
        if self.save_stats_results is True:
            for m, t, s in zip(self.measurements, self.times, self.stats_save_path):
                x, p, Q, r, norm_t = self.runKalmanStatic(t, m, self.sigma_xy, self.perc_sigma_vxy, use_accel=self.use_accel, 
                                    monotonicity=self.monotonicity, epsilon=self.epsilon, 
                                    save_results=True, stats_save_path=s)
                x_smooth.append(x)
                p_smooth.append(p)
                Q_base.append(Q)
                R.append(r)
                norm_times.append(norm_t)
        else:
            for m, t in zip(self.measurements, self.times):
                x, p, Q, r, norm_t = self.runKalmanStatic(t, m, self.sigma_xy, self.perc_sigma_vxy, use_accel=self.use_accel, 
                                    monotonicity=self.monotonicity, epsilon=self.epsilon, 
                                    save_results=False)
                x_smooth.append(x)
                p_smooth.append(p)
                Q_base.append(Q)
                R.append(r)
                norm_times.append(norm_t)
        
        if np.array(self.measurements).shape[0] == 1:
            return (x_smooth[0], p_smooth[0], Q_base[0], R[0], norm_times[0])
        else:
            return (x_smooth, p_smooth, Q_base, R, norm_times)

    def saveECSV(self, measurements, times, save_path, orig_path):
        # Copy the original ecsv to the save location
        shutil.copy(orig_path, os.path.join(save_path, os.path.basename(orig_path) + '_kalman.ecsv'))

        # Recompute ra dec, alt az
        jd, ra, dec, _ = xyToRaDecPP(times, measurements[0, :], measurements[1, :], None, self.platepar)
        az, alt = trueRaDec2ApparentAltAz(ra, dec ,jd, self.platepar.lat, self.platepar.lon, self.platepar)

        new_data = list(zip(ra, dec, az, alt, measurements[0, :], measurements[1, :]))
        file_path = os.path.join(save_path, os.path.basename(orig_path) + '_kalman.ecsv')

        # Read file, keeping comments (astropy-ecsv header) untouched
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Find where the CSV table starts (first non-# line with "datetime,ra,...")
        for i, line in enumerate(lines):
            if line.startswith("datetime,"):
                header_index = i
                break
        header = lines[:header_index+1]   # everything before and including header
        data_lines = lines[header_index+1:]

        # Load table part into pandas
        from io import StringIO
        df = pd.read_csv(StringIO("".join([lines[header_index], *data_lines])))

        if len(df) != len(new_data):
            raise ValueError(f"new_data has {len(new_data)} rows, but file has {len(df)}")

        # Unpack and assign replacements
        df.loc[:, "ra"]       = [row[0] for row in new_data]
        df.loc[:, "dec"]      = [row[1] for row in new_data]
        df.loc[:, "azimuth"]  = [row[2] for row in new_data]
        df.loc[:, "altitude"] = [row[3] for row in new_data]
        df.loc[:, "x_image"]  = [row[4] for row in new_data]
        df.loc[:, "y_image"]  = [row[5] for row in new_data]

        # Write back out: preserve header, overwrite data
        with open(file_path, "w") as f:
            f.writelines(header)
            df.to_csv(f, index=False)

    def computeQBase(self, measurements, times, sigma_xy, perc_sigma_vxy):
        """
        Build a process-noise covariance base matrix for the Kalman model.

        Estimates average velocity from linear fits of x(t) and y(t), derives a
        velocity std (as a percentage of average speed), and forms a 6×6 Q_base
        with position variances `sigma_xy^2`, velocity variances `sigma_vxy^2`,
        and identity accel variance placeholders (ignored when `use_accel=False`).

        Args:
            measurements (np.ndarray): (K, 2) positions (x, y).
            times (np.ndarray): (K,) times in seconds (normalized).
            perc_sigma_vxy (float): Percent of average velocity as velocity std.
            sigma_xy (float): Position std [px].

        Returns:
            np.ndarray: Q_base, shape (6, 6).
        """

        # Calculate average velocity (px/s)
        x_meas = measurements[:, 0]
        y_meas = measurements[:, 1]
        
        # Fit a line through time vs coordinates to estimate average velocity
        x_fit = np.polyfit(times, x_meas, 1)
        y_fit = np.polyfit(times, y_meas, 1)

        # Take the geometric mean of the velocities as the average
        avrg_velocity = np.hypot(x_fit[0], y_fit[0])  # Geometric mean of x and y velocities

        # Instantiate process noise covariance
        sigma_vxy = perc_sigma_vxy/100 * avrg_velocity

        Q_base = np.array([
            [sigma_xy**2, 0 ,0 ,0 ,0 ,0],
            [0, sigma_xy**2, 0 ,0 ,0 ,0],
            [0, 0, sigma_vxy**2, 0 ,0 ,0],
            [0, 0, 0, sigma_vxy**2, 0 ,0],
            [0, 0, 0, 0, 1, 0], #Default accel to 1 since it will be ignoreed in Use_Accel=False
            [0, 0, 0, 0, 0, 1]
        ])

        return Q_base

    def computeR(self, measurements, times):
        """
        Estimate measurement noise covariance R from line-fit residuals.

        Fits x(t) and y(t) independently with a line, computes residual std devs,
        and returns a 2×2 diagonal covariance matrix.

        Args:
            measurements (np.ndarray): (K, 2) positions (x, y).
            times (np.ndarray): (K,) times in seconds (normalized).

        Returns:
            np.ndarray: R, shape (2, 2).
        """

        # fit x and y vs time to a line
        x_fit = np.polyfit(times, measurements[:, 0], 1)
        y_fit = np.polyfit(times, measurements[:, 1], 1)

        # Calculate all residuals between fit and measurements
        x_residuals = measurements[:, 0] - np.polyval(x_fit, times)
        y_residuals = measurements[:, 1] - np.polyval(y_fit, times)

        # Calculate the STD of residuals
        x_std = np.std(x_residuals)
        y_std = np.std(y_residuals)

        # Form R as a 2x2 array of diagonal variances
        R = np.array([[x_std**2, 0], 
                      [0, y_std**2]])
        
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
        epsilon: [float] Threshold to prevent false violations due to floating-point error. 
        Used for enforcing monotonicity.
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

    def saveKalmanUncertaintiesToCSV(self, data_path, times, measurements, x_smooth, p_smooth, Q_base, R):
        """
        Save Kalman smoothing results, uncertainties, and noise model parameters to CSV.

        Creates a timestamped CSV file under `<data_path>/self_Kalman_Results/` containing:
            - Per-frame times (seconds since start).
            - Original measurements (x, y).
            - Smoothed state estimates (x, y, vx, vy, ax, ay).
            - Kalman uncertainty (per-state standard deviations from `p_smooth`).
            - Header rows with square-root diagonal values of `Q_base` and `R`.

        Args:
            data_path (str | os.PathLike): Root directory for output CSV.
            times (Sequence[float]): Relative times (s since start) for each state.
            measurements (np.ndarray): Original measurements, shape (K, 2).
            x_smooth (np.ndarray): Smoothed state estimates, shape (K, 6):
                [x, y, vx, vy, ax, ay].
            p_smooth (np.ndarray): Smoothed state covariances, shape (K, 6, 6).
            Q_base (np.ndarray): Process noise base matrix, shape (6, 6).
            R (np.ndarray): Measurement noise covariance, shape (2, 2).

        Returns:
            None. Writes CSV to disk.

        Raises:
            OSError: If directories or CSV file cannot be created.
            ValueError: If array shapes are inconsistent.
        """

        fig_dir = os.path.join(data_path, "self_Kalman_Results")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        # Make dest path
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

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
            data['kalman uncertainty/STD (x)'].append(np.sqrt(np.abs(p_smooth[i][0][0])))
            data['kalman uncertainty/STD (y)'].append(np.sqrt(np.abs(p_smooth[i][1][1])))
            data['kalman uncertainty/STD (vx)'].append(np.sqrt(np.abs(p_smooth[i][2][2])))
            data['kalman uncertainty/STD (vy)'].append(np.sqrt(np.abs(p_smooth[i][3][3])))
            data['kalman uncertainty/STD (ax)'].append(np.sqrt(np.abs(p_smooth[i][4][4])))
            data['kalman uncertainty/STD (ay)'].append(np.sqrt(np.abs(p_smooth[i][5][5])))

        # Add header row for Q_base and R std values
        std_q_base = np.sqrt(np.diag(Q_base))
        std_r = np.sqrt(np.diag(R))
        header_q_base = ["Q_base STD (x)", "Q_base STD (y)", 
                         "Q_base STD (vx)", "Q_base STD (vy)", 
                         "Q_base STD (ax)", "Q_base STD (ay)"]
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

    def runKalmanStatic(self, times, measurements, sigmaxy, perc_sigma_xy, 
                        save_results=False, save_path=None, use_accel=False, monotonicity=True, epsilon=1e-6):
        """
        Run the Kalman smoother as a static pipeline for given measurements.

        Normalizes times to seconds since first frame, constructs process noise `Q_base`
        and measurement noise `R`, runs the constant-velocity Kalman filter/smoother,
        and optionally saves results to CSV.

        Args:
            times (Sequence[datetime.datetime]): Absolute timestamps, length K.
            measurements (np.ndarray): Original measurements, shape (K, 2).
            sigmaxy (float): Position standard deviation [px].
            perc_sigma_xy (float): Velocity std as a percent of average velocity.
            save_results (bool): If True, save CSV of results.
            save_path (str | os.PathLike | None): Directory for saving CSV if
                `save_results=True`. If None, no file is written.
            use_accel (bool): If True, use acceleration states (constant-accel model);
                if False, constant-velocity model.
            monotonicity (bool): If True, enforce monotonic motion along dominant axis.
            epsilon (float): Numerical tolerance for monotonicity enforcement.

        Returns:
            tuple:
                x_smooth (np.ndarray): Smoothed state estimates, shape (K, 6).
                p_smooth (np.ndarray): Smoothed covariance estimates, shape (K, 6, 6).
                Q_base (np.ndarray): Process noise covariance base, shape (6, 6).
                R (np.ndarray): Measurement noise covariance, shape (2, 2).
                norm_times (list[float]): Relative times [s].

        Raises:
            RuntimeError: If Kalman filter computation fails.
            ValueError: If inputs have inconsistent lengths.
        """

        # Set measurements to ndarray
        measurements = np.array(measurements)

        # Normalize times to start
        norm_times = [(t - times[0]).total_seconds() for t in times]

        # Compute Kalman matricies
        Q_base = self.computeQBase(measurements, norm_times, sigmaxy, perc_sigma_xy)
        R = self.computeR(measurements, norm_times)

        x_smooth, p_smooth = self.kalmanFilterCA(
            measurements, norm_times, Q_base, R, 
            use_accel=use_accel, monotonicity=monotonicity, epsilon=epsilon
        )

        if save_results and save_path is not None:
            self.saveKalmanUncertaintiesToCSV(save_path, times, measurements, x_smooth, p_smooth, Q_base, R)
        
        return x_smooth, p_smooth, Q_base, R, norm_times
    
    def loadECSV(self, ECSV_file_path=None):
        """
        Loads the ECSV file and adds the relevant info to pick_list
        Args:
            self (PlateTool : optional): PlateTool object if used within SkyFit2
            ECSV_file_path (str : optional): Path to the ECSV file to load
        Returns:
            if self: # Return as a skyfit pick_list dict
                picks (dict): (N : [8]) dict following same format as self.pick_list 
            else: # Return times and measurements
                frame_times (np.ndarray) : N, array of measurement times
                measurements (np.ndarray) : N x 2 array of x,y measurements
        """

        # Instantiate arrays to be populated
        picks = []  # N x args_dict array for addCentroid
        pick_frame_times = []  # (N,) array of frame times

        # wrap in try except to catch unknown errors
        try:
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
                        pick_frame_times_idx = column_names.index('datetime') if 'datetime' in column_names \
                                                                                        else None
                        x_ind = column_names.index('x_image') if 'x_image' in column_names \
                                                                                        else None
                        y_ind = column_names.index('y_image') if 'y_image' in column_names \
                                                                                        else None
                        background_ind = column_names.index('background_pixel_value') \
                                                        if 'background_pixel_value' in column_names else None
                        sat_ind = column_names.index('saturated_pixels') if 'saturated_pixels' in column_names \
                                                                                        else None
                        snr_ind = column_names.index('snr') if 'snr' in column_names else None

                        # Ensure essential values are in the ECSV
                        if x_ind is None or y_ind is None or pick_frame_times_idx is None:
                            raise ValueError("ECSV file must contain 'x_image', 'y_image', "
                                           "and 'datetime' columns.")

                        continue

                    else:
                        # Unpack line

                        # Populate arrays
                        cx, cy = float(line[x_ind]), float(line[y_ind])
                        background = float(line[background_ind]) if background_ind is not None else 0
                        saturated = bool(line[sat_ind]) if sat_ind is not None else False
                        snr = float(line[snr_ind]) if snr_ind is not None else 1
                        pick_frame_times.append(datetime.datetime.strptime(line[pick_frame_times_idx], 
                                                                           '%Y-%m-%dT%H:%M:%S.%f'))

                        # Load in pick parameters, use default for other values
                        picks.append({
                            'x_centroid': cx,
                            'y_centroid': cy,
                            'mode': 1,
                            'intensity_sum': 1,
                            'photometry_pixels': None,
                            'background_intensity': background,
                            'snr': snr,
                            'saturated': saturated,
                        })



            return np.array(pick_frame_times), np.array([[pick['x_centroid'], pick['y_centroid']] for pick in picks])
            
        # Raise error box
        except Exception as e:
            raise Exception(f"Unknown Error reading ECSV file, check correct file loaded.: {str(e)}")
            
    



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='''
Kalman Filter and Smoother for Trajectory Data.

This script applies a Kalman filter with a Rauch-Tung-Striebel (RTS) smoother
to trajectory data provided in ECSV files. It is designed to smooth noisy 
position measurements (x, y) and estimate the kinematic state (position, velocity, 
and optionally acceleration) of an object over time.

The filter can operate in two primary modes:
1. Constant-Velocity (CV) Model (default): Assumes the object moves at a constant velocity 
   between measurements.
2. Constant-Acceleration (CA) Model (`--use_accel`): Assumes the object moves with 
   constant acceleration.

Key Features:
- Handles irregularly spaced time intervals between measurements.
- Automatically estimates measurement noise (`R` matrix) from the data.
- Constructs the process noise (`Q` matrix) based on user-provided parameters for 
  position and velocity uncertainty.
- Optionally enforces monotonic progression along the dominant axis of motion to prevent 
  the smoothed path from moving backward.
- Can output a new ECSV file with the smoothed trajectory data.
- Can save detailed statistical results, including state estimates, uncertainties, 
  and noise parameters, to a separate CSV file for analysis.
''',
        formatter_class=argparse.RawTextHelpFormatter
    )

    
    # Mandatory arguments
    parser.add_argument('data_path', type=str,
                       help='Path to .ecsv file or directory containing .ecsv files')
    parser.add_argument('sigma_xy', type=float,
                       help='Position standard deviation [px]')
    parser.add_argument('perc_sigma_vxy', type=float,
                       help='Velocity standard deviation as percentage of average velocity')
    
    # Optional arguments with defaults
    parser.add_argument('--epsilon', '--e', type=float, default=1e-6,
                       help='Numerical tolerance for monotonicity enforcement (default: 1e-6)')
    parser.add_argument('--monotonicity', '--m', type=bool, default=True,
                       help='Enforce monotonic motion along dominant axis (default: True)')
    parser.add_argument('--use_accel', '--a', type=bool, default=False,
                       help='Use constant-acceleration model instead of constant-velocity (default: False)')
    parser.add_argument('--save_path', '--p', type=str, default=None,
                       help='Directory to save output ECSV files (default: None)')
    parser.add_argument('--save_stats_results', '--r', type=bool, default=False,
                       help='Save statistical results to CSV (default: False)')
    parser.add_argument('--use_all_files', '--u', type=bool, default=False,
                       help='Process all .ecsv files in directory (default: False)')
    
    args = parser.parse_args()
    
    # Extract values
    sigma_xy = args.sigma_xy
    perc_sigma_vxy = args.perc_sigma_vxy
    data_path = args.data_path
    epsilon = args.epsilon
    monotonicity = args.monotonicity
    use_accel = args.use_accel
    save_path = args.save_path
    save_stats_results = args.save_stats_results
    use_all_files = args.use_all_files

    kalman = KalmanFilter(
        sigma_xy=sigma_xy, perc_sigma_vxy=perc_sigma_vxy,
        epsilon=epsilon, monotonicity=monotonicity, use_accel=use_accel,
        data_path=data_path, ecsv_save_path=save_path,
        save_stats_results=save_stats_results, use_all_files=use_all_files
    )