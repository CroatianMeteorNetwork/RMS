from __future__ import print_function, division, absolute_import

import os
import logging
import numpy as np

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
        is_mask = kwargs.get('is_mask', False)
        if file_path and os.path.exists(file_path):
            current_mtime = os.path.getmtime(file_path)
            if file_path not in self.file_cache or current_mtime != self.file_cache[file_path][0]:
                content = loader_func(*args)
                if is_mask:
                    is_all_white = np.all(content.img == 255) if content is not None else None
                    self.file_cache[file_path] = (current_mtime, content, is_all_white)
                else:
                    self.file_cache[file_path] = (current_mtime, content, None)
                return content, True  # True indicates the file was reloaded
            return self.file_cache[file_path][1], False  # False indicates the cached version was used
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

        # Load mask
        mask_path = None
        for p in [os.path.join(dir_path, config.mask_file),
                  os.path.join(config.config_file_path, config.mask_file)]:
            if os.path.exists(p):
                mask_path = p
                break

        if mask_path is None:
            log_message = 'No mask file has been found.'
            log.info(log_message)
            return None, None, None

        mask, mask_reloaded = self.loadFile(mask_path, MaskImage.loadMask, mask_path, is_mask=True)

        if mask_reloaded or (mask_path in self.file_cache and self.file_cache[mask_path][2] is None):

            # Check if all white only if the mask was reloaded
            if mask is not None and np.all(mask.img == 255):
                log_message = 'Loaded mask is all white, setting it to None: {0}'.format(mask_path)
                log.info(log_message)
                mask = None
                # Update cache to reflect all-white status
                self.file_cache[mask_path] = (self.file_cache[mask_path][0], None, True)

            else:
                log_message = 'Loaded mask: {0}'.format(mask_path)
                log.info(log_message)

        # Check cached all-white status
        elif mask_path in self.file_cache and self.file_cache[mask_path][2]:
            log_message = 'Cached mask is all white, setting it to None: {0}'.format(mask_path)
            log.info(log_message)
            mask = None

        else:
            log_message = 'Using cached mask: {0}'.format(mask_path)
            log.info(log_message)

        # Load dark frame
        dark = None
        if config.use_dark:
            dark_path = None
            for p in [os.path.join(dir_path, config.dark_file),
                      os.path.abspath(config.dark_file)]:
                if os.path.exists(p):
                    dark_path = p
                    break

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

        # Load flat field image
        flat_struct = None
        if config.use_flat:
            flat_path = None
            for p in [os.path.join(dir_path, config.flat_file),
                      os.path.abspath(config.flat_file)]:
                if os.path.exists(p):
                    flat_path = p
                    break

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
        if mask is not None:
            binned_mask = mask.copy()
        else:
            binned_mask = None

        if binned_mask is not None:
            binned_mask.img = Image.binImage(binned_mask.img, binning_factor, binning_method)

        # Bin the dark
        if dark is not None:
            binned_dark = Image.binImage(dark, binning_factor, binning_method)
        else:
            binned_dark = None

        # Bin the flat
        if flat_struct is not None:
            binned_flat = flat_struct.copy()
        else:
            binned_flat = None

        if binned_flat is not None:
            binned_flat.binFlat(binning_factor, binning_method)

        return binned_mask, binned_dark, binned_flat

    def clearCache(self):
        """Clear the entire file cache."""
        self.file_cache.clear()
        self.binned_cache.clear()

    def remove_from_cache(self, file_path):
        """Remove a specific file from the cache."""
        if file_path in self.file_cache:
            del self.file_cache[file_path]


# Create a single instance to be used across modules
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
