""" FF-equivalent image pairs stored as PNG files.

Inputs which are not captured as FF files (e.g. video files processed through the frame interface) can store
the max pixel and the average pixel images of each frame chunk as a pair of PNG files:

    FF_<station>_<YYYYMMDD>_<HHMMSS>_<mmm>_<frame>_maxpixel.png
    FF_<station>_<YYYYMMDD>_<HHMMSS>_<mmm>_<frame>_avepixel.png

The max pixel file is the canonical FF name of the pair. It is the name which is listed by validFFName, stored
in CALSTARS and FTPdetectinfo files, and given to FFfile.read, which loads both images into an FF structure.
The images are stored with their native bit depth, and the chunk metadata is stored in PNG text chunks.
"""

from __future__ import print_function, division, absolute_import

import datetime
import os

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo

from RMS.Formats.FFStruct import FFStruct


PAIR_MAX_SUFFIX = '_maxpixel.png'
PAIR_AVE_SUFFIX = '_avepixel.png'

# Format of the beginning time stored in the metadata
BEGIN_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def isPairMaxName(file_name):
    """ Check if the given file name is the max pixel image (canonical FF name) of an image pair. """
    return file_name.startswith('FF') and file_name.endswith(PAIR_MAX_SUFFIX)


def isPairAveName(file_name):
    """ Check if the given file name is the average pixel image of an image pair. """
    return file_name.startswith('FF') and file_name.endswith(PAIR_AVE_SUFFIX)


def isPairName(file_name):
    """ Check if the given file name is either image of an image pair. """
    return isPairMaxName(file_name) or isPairAveName(file_name)


def pairNames(name):
    """ Return the names of both files of an image pair.

    Arguments:
        name: [str] Name of either file of the pair, or the common base name without the suffix.

    Return:
        (max_name, ave_name): [tuple of str]
    """

    for suffix in (PAIR_MAX_SUFFIX, PAIR_AVE_SUFFIX):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    return name + PAIR_MAX_SUFFIX, name + PAIR_AVE_SUFFIX


def _writePNG(file_path, img, pnginfo):
    """ Write an 8 or 16-bit grayscale image to a PNG file. A temporary file is renamed at the end, so a
        reader never sees a partially written image.
    """

    tmp_path = file_path + '.tmp'
    PILImage.fromarray(img).save(tmp_path, format='PNG', pnginfo=pnginfo)
    os.replace(tmp_path, file_path)


def writePair(directory, name, maxpixel, avepixel, meta=None):
    """ Write the max pixel and the average pixel images as a pair of PNG files.

    Arguments:
        directory: [str] Directory where the files will be written.
        name: [str] Name of the pair, either file name or the common base name.
        maxpixel: [ndarray] Max pixel image, uint8 or uint16.
        avepixel: [ndarray] Average pixel image, uint8 or uint16.

    Keyword arguments:
        meta: [dict] Metadata stored in the PNG text chunks of both files, e.g. nframes, fps, first_frame,
            binning, station, begin_utc (datetime), source_file. None by default.

    Return:
        (max_path, ave_path): [tuple of str] Paths of the written files.
    """

    pnginfo = PngInfo()
    if meta is not None:
        for key, value in meta.items():

            if isinstance(value, datetime.datetime):
                value = value.strftime(BEGIN_TIME_FORMAT)

            pnginfo.add_text(str(key), str(value))

    max_path, ave_path = [os.path.join(directory, file_name) for file_name in pairNames(name)]

    # Write the average pixel image first, so the canonical max pixel file never exists without it
    _writePNG(ave_path, avepixel, pnginfo)
    _writePNG(max_path, maxpixel, pnginfo)

    return max_path, ave_path


def readMeta(file_path):
    """ Read the metadata stored in the PNG text chunks.

    Arguments:
        file_path: [str] Path to the PNG file.

    Return:
        [dict] Metadata as strings.
    """

    # The text chunks are written before the image data, so they are read without decoding the image
    with PILImage.open(file_path) as img:
        return dict(img.info)


def readPair(directory, name, full_filename=False, verbose=True):
    """ Read an image pair into an FF structure.

    Arguments:
        directory: [str] Directory containing the pair.
        name: [str] Canonical FF name of the pair (the max pixel file name).

    Keyword arguments:
        full_filename: [bool] True if the name is a full path. False by default.
        verbose: [bool] Print a warning if the average pixel image is missing. True by default.

    Return:
        [FFStruct] FF structure, or None if the max pixel image could not be read. The std pixel and max frame
            images are not stored, so they are filled with zeros.
    """

    if full_filename:
        directory, name = os.path.split(name)

    max_name, ave_name = pairNames(name)
    max_path = os.path.join(directory, max_name)
    ave_path = os.path.join(directory, ave_name)

    maxpixel = cv2.imread(max_path, cv2.IMREAD_ANYDEPTH)
    if maxpixel is None:
        return None

    if os.path.isfile(ave_path):
        avepixel = cv2.imread(ave_path, cv2.IMREAD_ANYDEPTH)

    else:
        if verbose:
            print("Average pixel image {:s} is missing, using the max pixel image!".format(ave_name))
        avepixel = np.copy(maxpixel)

    meta = readMeta(max_path)

    # Upscale binned images to the full image size, so they agree with the star and meteor coordinates
    binning = int(meta.get('binning', 1))
    if binning > 1:
        maxpixel = np.repeat(np.repeat(maxpixel, binning, axis=0), binning, axis=1)
        avepixel = np.repeat(np.repeat(avepixel, binning, axis=0), binning, axis=1)


    ff = FFStruct()
    ff.nrows, ff.ncols = maxpixel.shape
    ff.nbits = 8*maxpixel.itemsize
    ff.nframes = int(meta.get('nframes', -1))
    ff.fps = float(meta.get('fps', -1))
    ff.first = int(meta.get('first_frame', 0))

    if 'begin_utc' in meta:
        ff.starttime = meta['begin_utc']

    ff.maxpixel = maxpixel
    ff.avepixel = avepixel
    ff.stdpixel = np.zeros_like(maxpixel)
    ff.maxframe = np.zeros(maxpixel.shape, dtype=np.uint8)

    return ff
