from __future__ import print_function, division, absolute_import

import os
import logging
import numpy as np
import copy

from RMS.Routines import Image
from RMS.Routines import MaskImage


# Get the logger from the main module
log = logging.getLogger("logger")


class ImageCalibrationLoader:
    """
    A class to handle loading and caching of image calibration files.

    This class provides methods to load mask, dark, and flat field images
    with efficient caching to reduce I/O operations.
    """

    def __init__(self):
        self.file_cache = {}
        self.binned_cache = {}

    def findFile(self, file_name, dir_path, config_path):
        """
        Helper function to find the first valid file in given directories.
        """

        # List paths to check in order of priority
        paths_to_check = [
            os.path.join(dir_path, file_name),
            os.path.join(config_path, file_name)
        ]

        # Initialize result_path
        result_path = None

        # Find the first valid path
        for path in paths_to_check:
            if os.path.exists(path):
                result_path = path
                break

        # Log
        if result_path:
            log.info("Found {file_name} at: {result_path}"
                     .format(file_name=file_name, result_path=result_path))
        else:
            log.info("No {file_name} file has been found in.".format(file_name=file_name))

        return result_path

    def loadFile(self, file_path, loader_func, *args, **kwargs):
        """
        Load a file and cache its contents if it has changed since last load.

        Arguments:
            file_path (str): Path to the file to be loaded.
            loader_func (callable): Function to use for loading the file.
            *args: Variable length argument list to pass to the loader_func.

        Keyword Arguments:
            is_mask (bool, optional): Whether the file being loaded is a mask.
                                      Defaults to False.

        Returns:
            tuple: A tuple containing (file_content, was_reloaded), where
            file_content is the loaded file data and was_reloaded is a boolean
            indicating if the file was reloaded or loaded from cache.
        """

        # Determine if the file is intended to be a mask
        is_mask = kwargs.get('is_mask', False)

        if file_path and os.path.exists(file_path):

            # Get the last modification time of the file
            current_mtime = os.path.getmtime(file_path)

            # Check if the file is not cached or has been modified
            if file_path not in self.file_cache or current_mtime != self.file_cache[file_path][0]:

                # Load the file content using the provided loader function
                content = loader_func(*args)

                # If mask, determine if the image content is entirely white
                if is_mask:
                    is_all_white = np.all(content.img == 255) if content is not None else None

                    # Cache the time, content, and the all-white status
                    self.file_cache[file_path] = (current_mtime, content, is_all_white)
                else:
                    # Cache the time, content, and the non-all-white status
                    self.file_cache[file_path] = (current_mtime, content, None)

                # Return the content and True, indicating the file was reloaded
                return content, True

            # If the file was already cached and hasn't been modified,
            # return the cached content and False
            return self.file_cache[file_path][1], False

        # If the file path is invalid or the file does not exist,
        # return None and False
        return None, False

    def loadImageCalibration(self, dir_path, config, dtype=None, byteswap=False):
        """
        Load the mask, dark and flat with file-specific caching and optimized
        mask handling.

        Arguments:
        dir_path: [str] Path to the directory with calibration.
        config: [ConfigStruct]

        Keyword arguments:
        dtype: [object] Numpy array dtype for the image. None by default, in
               which case it will be determined from the input image.
        byteswap: [bool] If the dark and flat should be byteswapped. False by
                  default, and should be True for UWO PNGs.

        Returns:
            tuple: A tuple containing (original, binned), where each is a
                   tuple of (mask, dark, flat_struct).
        """

        # Initialize variables
        mask, dark, flat_struct = None, None, None
        mask_reloaded, dark_reloaded, flat_reloaded = False, False, False
        mask_path, dark_path, flat_path = None, None, None

        # --- Load mask ---

        # Try loading the mask from CaptureFiles directory first,
        # then from the config directory
        mask_path = self.findFile(config.mask_file, dir_path, config.config_file_path)
        mask, mask_reloaded = self.loadFile(mask_path, MaskImage.loadMask, mask_path, is_mask=True)

        if mask_reloaded or (mask_path in self.file_cache and self.file_cache[mask_path][2] is None):

            # Check if all white only if the mask was reloaded
            if mask is not None and np.all(mask.img == 255):
                log.info('Loaded mask is all white, setting it to None: {0}'.format(mask_path))
                mask = None
                # Update cache to reflect all-white status
                self.file_cache[mask_path] = (self.file_cache[mask_path][0], None, True)

            else:
                log.info('Loaded mask: {0}'.format(mask_path))

        # Check cached all-white status
        elif mask_path in self.file_cache and self.file_cache[mask_path][2]:
            log.info('Cached mask is all white, setting it to None: {0}'.format(mask_path))
            mask = None

        else:
            log.info('Using cached mask: {0}'.format(mask_path))

        # --- Load dark frame ---
        if config.use_dark:

            # Try loading the dark file from CaptureFiles directory first,
            # then from the config directory
            dark_path = self.findFile(config.dark_file, dir_path, config.config_file_path)

            # Load the dark file if a valid path was found
            if dark_path:
                dark, dark_reloaded = self.loadFile(dark_path,
                                                    Image.loadDark,
                                                    os.path.split(dark_path)[0],
                                                    os.path.split(dark_path)[1],
                                                    dtype=dtype,
                                                    byteswap=byteswap)
                if dark is not None:
                    log.info('{0} dark: {1}'.format("Reloaded" if dark_reloaded else "Using cached",
                                                    dark_path))

        # --- Load flat field image ---
        if config.use_flat:

            # Try loading the flat file from CaptureFiles directory first,
            # then from the config directory
            flat_path = self.findFile(config.flat_file, dir_path, config.config_file_path)

            # Load the flat file if a valid path was found
            if flat_path:
                flat_struct, flat_reloaded = self.loadFile(flat_path,
                                                           Image.loadFlat,
                                                           os.path.split(flat_path)[0],
                                                           os.path.split(flat_path)[1],
                                                           dtype=dtype,
                                                           byteswap=byteswap)
                if flat_struct is not None:
                    log.info('{0} flat: {1}'.format("Reloaded" if flat_reloaded else "Using cached",
                                                    flat_path))

        # Create binned versions if any file was reloaded or don't exist
        bin_key = (config.detection_binning_factor, 'avg')
        if mask_reloaded or dark_reloaded or flat_reloaded or bin_key not in self.binned_cache:
            binned_mask, binned_dark, binned_flat = self.binImageCalibration(config, mask, dark, flat_struct)
            self.binned_cache[bin_key] = (binned_mask, binned_dark, binned_flat)

        original = (mask, dark, flat_struct)
        binned = self.binned_cache[bin_key]

        return original, binned

    def binImageCalibration(self, config, mask, dark, flat_struct):
        """Bin the calibration images."""
        binning_factor = config.detection_binning_factor
        binning_method = 'avg'

        # Bin the mask
        binned_mask = None
        if mask is not None:
            binned_mask = copy.deepcopy(mask)
            binned_mask.img = Image.binImage(binned_mask.img, binning_factor, binning_method)

        # Bin the dark
        binned_dark = None
        if dark is not None:
            binned_dark = np.copy(dark)
            binned_dark = Image.binImage(dark, binning_factor, binning_method)

        # Bin the flat
        binned_flat = None
        if flat_struct is not None:
            binned_flat = copy.deepcopy(flat_struct)
            binned_flat.binFlat(binning_factor, binning_method)

        return binned_mask, binned_dark, binned_flat

    def cleanup(self):
        """
        Clean up the instance by clearing caches and removing attributes.
        """
        self.file_cache.clear()
        self.binned_cache.clear()

        # Explicitly remove attributes
        self.file_cache = None
        self.binned_cache = None

        log.info("imageCalibrationLoader instance has been cleaned up.")


# Create an instance. This will be shared across modules within the same
# process, but each separate Python process will have its own instance.
imageCalibrationLoader = ImageCalibrationLoader()


# Function to get the singleton instance
def getImageCalibrationLoader():
    """
    Get the singleton instance of ImageCalibrationLoader.

    This function provides access to a single, shared instance of the
    ImageCalibrationLoader class, ensuring that the same cache is used
    across all parts of the application.

    Returns:
        ImageCalibrationLoader: The singleton instance of
                                ImageCalibrationLoader.
    """
    return imageCalibrationLoader
