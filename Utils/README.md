# RMS Utils Directory

This directory contains various utility scripts for the Global Meteor Network (GMN) / RMS (Raspberry Pi Meteor Station) project. These scripts handle everything from camera control to deep data analysis, calibration, and visualization.

Most of these scripts can be run directly from the command line using `python -m Utils.<ScriptName>`. Many of them accept `-h` or `--help` to show detailed usage instructions and arguments.

---

## 📷 Camera Control & Configuration

Scripts for interacting directly with the IP cameras, manipulating their settings, or migrating configuration files.

- **`CameraControl.py`**: Controls CMS-compatible IP cameras (e.g., IMX291). Allows getting and setting camera parameters, exposure, and gain blocks.
- **`CamManager.py`**: A device manager module for open IPC devices.
- **`MigrateConfig.py`**: Upgrades single or multi-station config files to the latest `configTemplate` format while preserving existing attributes.
- **`SetCameraAddress.py`**: Utility to assist with network configuration and IP assignments for cameras.
- **`setAllCameraParams.py`**: Script to bulk-set camera parameters.
- **`ShowLiveStream.py`**: Shows the live stream output from the camera directly on the screen.
- **`AuditConfig.py`**: Helper functions used for extracting and auditing options when migrating configs.

## ✨ Calibration & Astrometry

Tools for sky fitting, making flat fields, calibrating night data, and astrometric calculations.

- **`SkyFit2.py`**: A comprehensive GUI application for sky fitting, meteor detection, and manual astrometric reduction.
- **`Astra.py`**: Astrometric Streak Tracking and Refinement Algorithm.
- **`MakeFlat.py`**: Generates a flat field image from frames accumulated throughout the night to estimate the background and correct vignetting.
- **`CalibrationReport.py`**: Generates a calibration report with the quality of astrometric and photometric calibration given the folder of a night.
- **`RetroactiveFixup.py`**: Recalibrates all data older than a specified date and optionally applies the most recent config to older directories.
- **`TrackStack.py`**: Generates a stack image with aligned stars so the sky appears static. Useful for making mosaics of meteor shower meteors.
- **`KalmanFilter.py`**: Applies a Kalman filter to astrometric measurements.

## 🔄 Data Conversion & Extraction

Utilities for transforming data formats (like `.bin` files and FF files) into readable textures, CSVs, or formats used by other software.

- **`FFtoFrames.py`**: Converts FF (Fits-like Format) files into individual reconstructed video frames (PNG).
- **`BatchFFtoImage.py`**: Batch converts FF files to standard image files (e.g., JPG, PNG).
- **`FieldSumToTxt.py`**: Converts FS (Field Sum) binary files into CSV files.
- **`RMS2UFO.py`**: Converts `FTPdetectinfo` meteor lists into the input CSV format required by UFOOrbit.
- **`FRbinMosaic.py`**: Extracts fireball detection subframes from FR `.bin` files to make a mosaic.
- **`Vidchop.py`**: Takes a standard video file and slices it into individual PNG frames.
- **`Grouping3DRunner.py`**: Script for running the 3D fireball extractor on FF files.

## 📊 Visualization & Plotting

Scripts for visualizing RMS system operations visually, rendering detection thresholds, and plotting time-series or image stacks.

- **`StackFFs.py`**: Stacks all max-pixel FF files in a given folder into a single compiled image.
- **`StackImgs.py`**: Stacks all standard images in a specified folder into one image.
- **`GenerateMP4s.py` / `GenerateTimelapse.py`**: Generates high-quality MP4 movies or timelapses from FF files.
- **`LiveViewer.py`**: Monitors a directory for FF files and displays them as a live slideshow as new ones are created.
- **`PlotFieldsums.py`**: Generates a plot showing all intensities from field-sum files over time.
- **`PlotMeteorPSFProfile.py`**: Plots the meteor PSF (Point Spread Function) profile from the detection and FF file.
- **`PlotTimeIntervals.py`**: Plots the intervals between timestamps from FF files and scores the system's timing performance.
- **`ShowThresholdLevels.py`**: Plots color-coded images representing the required signal threshold to detect specific features or fireballs in the images.
- **`GenerateThumbnails.py`**: Generates thumbnail images of all FF files in a specific directory.
- **`DrawConstellations.py`**: Draws constellation lines over an astrometrically calibrated image.
- **`FOVKML.py`**: Generates a KML file outlining the field of view of the camera for Google Earth.
- **`FOVSkyMap.py`**: Plots the position of the Moon over time.
- **`FRbinStack.py`**: Stacks fireball (FR) detections into a single max-value image.
- **`FRbinViewer.py`**: Cross-platform display tool to view fireball detections from FR bin files.
- **`PointsViewer.py`**: 3D scatter plot visualizer for viewing coordinate points.
- **`CheckNight.py`**: Reviews night exposures, building a max-pixel checking image for rapid visual inspection.

## 🌠 Meteor & Shower Analysis (Flux)

Detailed atmospheric and orbital analysis tools, particularly for computing meteor mass, luminosity, and shower flux.

- **`Flux.py`**: Computes single-station meteor shower flux and mass indexes.
- **`FluxAuto.py`**: Automatically runs flux calculations and produces graphs based on available data from multiple stations.
- **`FluxBatch.py`**: Batch runs the flux scripts using a defined batch configuration file.
- **`FluxFitActivityCurve.py`**: Provides functions for fitting the flux activity curve of a meteor shower based on the standard double-exponential profile.
- **`ShowerAssociation.py`**: Calculates single-station shower associations using astrometric data.
- **`SaturationCorrection.py`**: Corrects meteor magnitudes and levels in detection files, accounting for sensor saturation.
- **`SaturationSimulation.py`**: Simulates pixel saturation given a moving Gaussian meteor profile.
- **`RecomputeCollectionAreas.py`**: Recomputes the atmospheric collection area (`.json` files) across directories.

## 🗃️ File & Log Management

- **`LogArchiver.py`**: Handles archiving of logs to the main server.

---

### General Usage
To use a script, you can generally invoke it as a Python module to ensure that the main `RMS` dependency paths are resolved properly:
```bash
python -m Utils.MakeFlat <DIR_PATH>
```
Try passing `-h` or `--help` to see individual script arguments.
