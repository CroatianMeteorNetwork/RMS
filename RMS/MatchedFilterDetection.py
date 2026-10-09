""" Matched-filter detection of faint moving objects in frame-based video (e.g. .vid files).

The normal detection thresholds the maxpixel of a window of frames, so an object has to be well above the
noise in single frames. Here the frames are instead summed along the motion of the object (shift-and-add
along a grid of velocities) before thresholding, which raises the signal-to-noise ratio of an object by about
the square root of the number of frames summed:

    1. Background: the per-pixel median and noise of every block of frames, smoothed over the neighbouring
       blocks and interpolated in time. The brightness of the stars changes with the transparency of the sky,
       so the best fitting scale of the star template is subtracted from every frame. Bright stars, the mask
       and the image border are masked.
    2. Search: the normalized frames are binned 2x2, smoothed with the PSF, and summed along a grid of
       velocities over runs of frames (short runs for the full range of speeds, long runs for slow objects).
       The best velocity of every pixel gives a detection image, whose local maxima above the threshold are
       the hits. Pixels above the threshold in a large part of the whole input are persistent and removed.
    3. Linking: the hits of consecutive runs are linked into tracks, using the velocity of each hit to predict
       the next one, and the tracks of the same object are merged.
    4. Measurement: a moving PSF is fitted jointly to short runs of the unbinned frames, at the position
       predicted from the track (see MatchedFilterKernels.fitMovingPSF). The run length is chosen per track,
       as short as the required position accuracy allows (every frame for bright objects), and the track is
       followed with a local motion model, so curved and accelerating tracks are measured too.
    5. Verification: the PSF-weighted signal along a smooth track through the measurements, minus the signal
       at the same positions when the object is elsewhere, over the pixels which are not on stars, has to be
       significant.

The results are written as an FTPdetectinfo file with the suffix "mf", apart from the results of the normal
detection, and recalibrated with the platepar. It runs in the monitor (mf_enable) after the normal detection,
or on files and directories from the command line (python -m RMS.MatchedFilterDetection), which skips the
files already processed so a run can be started again. The velocity search runs on the GPU when numba can
use CUDA, otherwise on all CPU cores (mf_threads).
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import json
import glob
import math
import shutil
import argparse
import datetime
import traceback
import collections
from time import time

import numpy as np
import cv2
from scipy import ndimage

import RMS.ConfigReader as cr
from RMS.Logger import getLogger
from RMS.Routines import Image
from RMS.Routines import MaskImage
from RMS.Routines.DynamicFTPCompressionCy import sampleMedianMAD
from RMS.Routines.MatchedFilterKernels import (VelocityStacker, fitMovingPSF, forcedTrackSignal, streakAperture,
                                               CUDA_AVAILABLE)
from RMS.Detection import getPolarLine, removeDuplicateDetections, joinContinuousDetections


log = getLogger("logger")


# Name of the directory of the results of the matched filter, inside the results directory of the normal
#   detection, so they are kept apart from the normal results (e.g. in the night reports of the monitor)
MATCHED_FILTER_DIR = 'matched_filter'

# Suffix of the names of the result files
MATCHED_FILTER_SUFFIX = 'mf'

# Name of the file which marks an input file as processed, with a summary of the processing
DONE_NAME = 'matched_filter_done.json'

# Extensions of the input files found in directories
INPUT_EXTENSIONS = ('.vid', '.mkv', '.mp4', '.avi', '.mov')

# Tracks slower than this (px per frame) are measured again on a background estimated without them
SLOW_SPEED = 0.1

# Very bright objects spill their charge along their row and column before the pixels reach the saturation
#   level: in a frame where the sum within SPILL_RADIUS (px) of the object is more than SPILL_RATIO times the sum
#   within the normal aperture, the measurement counts as saturated and the wider sum is its intensity. Only
#   checked when the normal sum is above SPILL_MIN_SNR times its noise
SPILL_RADIUS = 12.0
SPILL_RATIO = 1.3
SPILL_MIN_SNR = 100.0

# Search of very slow objects (see slowSearch): the medians of the blocks are compared with the medians of the
#   blocks SLOW_REF_BLOCKS (and one more) before and after, shifted by the motion of the stars; candidates above
#   SLOW_THRESHOLD (sigma) in at least SLOW_MIN_BLOCKS consecutive blocks are linked, up to SLOW_MAX_SPEED (px per
#   frame, faster objects are found by the search), and moving at least SLOW_MIN_RELATIVE_SPEED (px per frame)
#   relative to the stars
SLOW_REF_BLOCKS = 3
SLOW_THRESHOLD = 8.0
SLOW_MIN_BLOCKS = 3
SLOW_MAX_SPEED = 0.06
SLOW_MIN_RELATIVE_SPEED = 0.004

# Smallest fraction of the samples of a pixel left after masking the slow objects (see maskObjects)
MIN_MASKED_FRACTION = 0.3

# Smallest number of stars for the aperture correction of the intensities (see apertureCorrection)
MIN_APERTURE_STARS = 20

# Pixels of the star template: stars above this level in the median of the frames, in the noise of one frame,
#   and the smallest number of such pixels for the correction of the transparency
STAR_TEMPLATE_LEVEL = 0.3
MIN_STAR_TEMPLATE_PX = 100

# Trails of bright moving sources (see removeTrails): half width of the removed rows and columns beyond the
#   source (the trails are up to a few pixels off the source), and the margin kept around the source (px)
TRAIL_WIDTH = 4
TRAIL_KEEP = 6


# A hit: position and time of the middle of a run of frames, velocity (px/frame, unbinned), significance,
#   and the length of the run
Hit = collections.namedtuple('Hit', ['frame', 'x', 'y', 'vx', 'vy', 'z', 'run'])


class MatchedFilterOptions(object):
    def __init__(self, config, det_bin=1, fps=None, size=None):
        """ Options of the matched-filter detection, from the config.

        Arguments:
            config: [Config] Configuration object.

        Keyword arguments:
            det_bin: [int] Binning of the frames (detection binning), the pixel limits are in binned pixels. 1 by
                default.
            fps: [float] Frame rate of the input (e.g. measured from the frame times). config.fps by default.
            size: [tuple] (height, width) of the unbinned frames of the input. The size in the config by
                default.
        """

        self.block_frames = config.mf_block_frames
        self.threshold = config.mf_threshold
        self.run_frames = sorted(config.mf_run_frames)
        self.slow_run_frames = config.mf_slow_run_frames
        self.velocity_search = config.mf_velocity_search
        self.min_hits = config.mf_min_hits
        self.psf_sigma = config.mf_psf_sigma
        self.max_pos_error = config.mf_max_pos_error
        self.max_measure_frames = config.mf_max_measure_frames
        self.min_sample_snr = config.mf_min_sample_snr
        self.min_centroids = config.mf_min_centroids
        self.star_threshold = config.mf_star_threshold
        self.persistence = config.mf_persistence
        self.clip = config.mf_clip
        self.trail_level = config.mf_trail_level
        self.min_displacement = config.mf_min_displacement
        self.min_frames = config.mf_min_frames
        self.sigma_scale = config.mf_sigma_scale
        self.track_significance = config.mf_track_significance
        self.edge_margin = max(config.detection_border, config.mf_edge_margin)
        self.gpu = config.mf_gpu
        self.smooth_frames = config.mf_smooth_frames

        # Saturation level of the raw frames (ADU), 98% of the range of the bit depth if not given
        self.saturation_level = config.mf_saturation_level if config.mf_saturation_level > 0 \
            else int(round(0.98*(2**config.bit_depth - 1)))
        self.threads = config.mf_threads

        # Speed limits in px per frame of the (binned) frames, from the angular velocity limits
        fps = fps if fps else config.fps
        height, width = size if size else (config.height, config.width)
        scale = det_bin*(config.fov_h/float(height) + config.fov_w/float(width))/2.0
        self.speed_min = config.mf_ang_vel_min/scale/fps
        self.speed_max = config.mf_ang_vel_max/scale/fps

        # The runs of frames don't straddle the blocks, so a block holds a whole number of the longest runs and
        #   of the measurements
        unit = int(np.lcm.reduce([int(v) for v in self.run_frames + [self.slow_run_frames, 1]]))
        unit = int(np.lcm(unit, 2**int(math.floor(math.log2(max(self.max_measure_frames, 1))))))
        if self.block_frames % unit:
            fixed = max(unit, int(round(self.block_frames/float(unit)))*unit)
            log.warning('Matched filter: mf_block_frames {:d} is not a multiple of {:d}, {:d} used'.format(
                self.block_frames, unit, fixed))
            self.block_frames = fixed

        self.use_gpu = (self.gpu in ('on', 'auto')) and CUDA_AVAILABLE
        if (self.gpu == 'on') and not CUDA_AVAILABLE:
            log.warning('Matched filter: the GPU was requested, but CUDA is not available, using the CPU')



def velocityGrid(speed_max, run_frames, bin_factor):
    """ Velocities to search, in px per frame of the binned image. The spacing makes the largest position
        error of an object in a run half a binned pixel.

    Arguments:
        speed_max: [float] Largest speed (px/frame, unbinned).
        run_frames: [int] Number of frames in a run.
        bin_factor: [int] Binning of the searched image.

    Return:
        [ndarray] Velocities (vx, vy) of shape (n_vel, 2), binned px per frame.
    """

    step = 1.0/max(run_frames - 1, 1)
    v_max = speed_max/bin_factor + step
    grid = np.arange(-np.ceil(v_max/step), np.ceil(v_max/step) + 1)*step
    vx, vy = np.meshgrid(grid, grid)
    keep = vx**2 + vy**2 <= v_max**2
    vel = np.column_stack([vx[keep], vy[keep]])

    # Sorted by speed: for a slow object many velocities give nearly the same sum, and the first (slowest) of
    #   the equal ones is taken
    return vel[np.argsort(np.hypot(vel[:, 0], vel[:, 1]), kind='stable')]


def estimatePSFSigma(image, noise, n_stars=40, min_snr=20.0):
    """ Estimate the PSF sigma from bright isolated stars: a circular Gaussian with a constant is fitted to every
        star, and the median of their sigmas is taken.

    Arguments:
        image: [ndarray] Image of the stars with the sky subtracted (e.g. the mean of a block of frames minus
            the sky level).
        noise: [ndarray] Noise of the image (same units), per pixel or a scalar.

    Keyword arguments:
        n_stars: [int] Number of stars to use. 40 by default.
        min_snr: [float] Smallest peak signal-to-noise ratio of a star. 20 by default.

    Return:
        [float] PSF sigma (px), or None if fewer than 5 stars were measured.
    """

    from scipy.optimize import least_squares

    snr = image/noise
    smooth = cv2.GaussianBlur(snr.astype(np.float32), (0, 0), 1.0)
    r = 6
    local_max = (smooth == cv2.dilate(smooth, np.ones((2*r + 1, 2*r + 1), np.uint8))) & (smooth > min_snr)
    local_max[:2*r] = local_max[-2*r:] = False
    local_max[:, :2*r] = local_max[:, -2*r:] = False
    ys, xs = np.nonzero(local_max)

    # The brightest stars may be saturated, so the stars are taken in order of brightness after the first few
    order = np.argsort(smooth[ys, xs])[::-1]
    order = order[min(5, len(order)//4):]

    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]

    def residual(p, patch):
        amp, cx, cy, sigma, const = p
        return (amp*np.exp(-0.5*((xx - cx)**2 + (yy - cy)**2)/sigma**2) + const - patch).ravel()

    sigmas = []
    for i in order:
        y, x = ys[i], xs[i]

        # Isolated: no other star brighter than a tenth of this one within 2r
        near = smooth[y - 2*r:y + 2*r + 1, x - 2*r:x + 2*r + 1].copy()
        near[r:3*r + 1, r:3*r + 1] = -np.inf
        if near.max() > 0.1*smooth[y, x]:
            continue

        patch = image[y - r:y + r + 1, x - r:x + r + 1].astype(np.float64)
        p0 = [patch.max(), 0.0, 0.0, 1.0, 0.0]
        try:
            fit = least_squares(residual, p0, args=(patch,), bounds=([0, -2, -2, 0.3, -np.inf],
                                                                     [np.inf, 2, 2, 5.0, np.inf]))
        except Exception:
            continue
        if fit.success:
            sigmas.append(abs(fit.x[3]))
        if len(sigmas) >= n_stars:
            break

    if len(sigmas) < 5:
        return None

    return float(np.median(sigmas))


def binFrames(frames, bin_factor):
    """ Bin frames by summing bin_factor x bin_factor pixels, scaled to keep unit noise. """

    n, h, w = frames.shape
    h2, w2 = h//bin_factor, w//bin_factor
    binned = frames[:, :h2*bin_factor, :w2*bin_factor].reshape(n, h2, bin_factor, w2, bin_factor).sum(axis=(2, 4))

    return binned/bin_factor


class BlockBackground(object):
    def __init__(self, median, noise, n_samples):
        """ Background of a block of frames: per-pixel median and noise (ADU), and the number of samples. The
            star template (the stars of the median above the local sky, at the pixels in star_idx) is set by
            MatchedFilterDetector.backgroundPass, with the local sky level and noise, and the brightness of the
            stars relative to the noise of the sky.
        """

        self.median = median
        self.noise = noise
        self.n_samples = n_samples

        # Mean of all frames of the block (for the photometry of the stars, see apertureCorrection)
        self.mean = None

        # Median of the samples of this block alone (median is smoothed over the neighbouring blocks)
        self.raw_median = None
        self.stars = None
        self.star_idx = None
        self.sky = None
        self.sky_noise = None
        self.star_level = None



class MatchedFilterDetector(object):
    def __init__(self, img_handle, config, mask=None, dark=None, flat_struct=None):
        """ Matched-filter detection of moving objects in the frames of an image handle.

        Arguments:
            img_handle: [FrameInterface] Frame-based input (detection=True).
            config: [Config] Configuration object.

        Keyword arguments:
            mask: [MaskStruct] Mask, binned like the frames. None by default.
            dark: [ndarray] Dark frame, binned like the frames. None by default.
            flat_struct: [FlatStruct] Flat field, binned like the frames. None by default.
        """

        self.img_handle = img_handle
        self.config = config
        self.mask = mask
        self.dark = dark
        self.flat_struct = flat_struct

        self.height, self.width = img_handle.nrows, img_handle.ncols
        self.total_frames = img_handle.total_frames

        # Detection binning of the frame handle, the results are scaled to the unbinned image
        self.det_bin = config.detection_binning_factor if img_handle.input_type != 'ff' else 1
        fps = img_handle.fps if getattr(img_handle, 'fps', None) else config.fps
        self.opts = MatchedFilterOptions(config, det_bin=self.det_bin, fps=fps,
                                         size=(self.height*self.det_bin, self.width*self.det_bin))

        # The search is done on frames binned 2x2
        self.search_bin = 2

        # Saturation level of the (binned) frames: a summed bin of saturated pixels holds det_bin^2 times the
        #   level; an averaged bin is flagged only when all its pixels are saturated
        self.saturation_level = self.opts.saturation_level
        if (self.det_bin > 1) and (config.detection_binning_method != 'avg'):
            self.saturation_level *= self.det_bin**2

        self.backgrounds = {}
        self.static_masks = {}

        # Factor of the intensities from the stars (see apertureCorrection), None if not applied
        self.aperture_correction = None
        self.psf_sigma = self.opts.psf_sigma if self.opts.psf_sigma > 0 else None

        self.timing = collections.defaultdict(float)

        # Significances of the measured tracks: (first frame, last frame, significance, off time, reordered)
        self.significances = []


    ### Frames ###

    def readFrames(self, first, n, step=1, saturation=False):
        """ Read n frames from the first one, every step-th, with the dark and the flat applied, as float32. With
            saturation, also the mask of the saturated pixels of the raw frames (uint8). """

        frames = np.empty((n, self.height, self.width), dtype=np.float32)
        saturated = np.zeros((n, self.height, self.width), dtype=np.uint8) if saturation else None
        for i in range(n):
            self.img_handle.setFrame(first + i*step)
            frame = self.img_handle.loadFrame()
            if saturation:
                saturated[i] = frame >= self.saturation_level
            if self.dark is not None:
                frame = Image.applyDark(frame, self.dark)
            if self.flat_struct is not None:
                frame = Image.applyFlat(frame, self.flat_struct)
            frames[i] = frame

        if saturation:
            return frames, saturated

        return frames


    def blockBackground(self, samples):
        """ Per-pixel median and noise of the sampled frames (ADU). """

        median, mad = sampleMedianMAD(samples)
        noise = np.maximum(1.4826*mad, 1.0).astype(np.float32)

        return BlockBackground(median.astype(np.float32), noise, len(samples))


    def backgroundPass(self, step=4):
        """ The background of every block, from every step-th frame of the block, smoothed over the neighbouring
            blocks. The median of a block alone has a noise of about 0.15 of the noise of a frame, which is static
            within the block: summed over many frames, it would make faint static patterns which change from
            block to block. Averaging over three blocks, and interpolating between the blocks in time (see
            normalize), makes it smaller and continuous.
        """

        raw = []
        for first, last in zip(self.block_starts, self.block_ends):
            t1 = time()
            n = (last - first + step - 1)//step
            samples = self.readFrames(first, n, step=step)
            self.timing['read'] += time() - t1

            t1 = time()
            raw.append(self.blockBackground(samples))
            self.timing['background'] += time() - t1

        for k in range(len(raw)):
            near = raw[max(k - 1, 0):k + 2]
            median = np.mean([bg.median for bg in near], axis=0).astype(np.float32)
            noise = raw[k].noise
            bg = BlockBackground(median, noise, sum(bg.n_samples for bg in near))
            bg.raw_median = raw[k].median
            self.starTemplate(bg)
            self.backgrounds[k] = bg


    def starTemplate(self, bg):
        """ Set the local sky level and its noise, the brightness of the stars relative to the noise of the sky,
            and the template of the stars of a block background, from its median.
        """

        # Local sky level and its noise (the noise at a star is mostly the noise of the star itself), and the
        #   brightness of the stars relative to the noise of the sky in one frame
        bg.sky = self.skyLevel(bg.median)
        bg.sky_noise = np.maximum(self.skyLevel(bg.noise), 1.0)
        stars = bg.median - bg.sky
        level = cv2.GaussianBlur(stars/bg.sky_noise, (0, 0), 1.0)
        bg.star_level = level

        # Template of the stars, for the changes of their brightness with the transparency of the sky
        star_px = cv2.dilate((level > STAR_TEMPLATE_LEVEL).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        star_px &= stars > 0

        # The bright stars are masked anyway and would dominate the fit of the scale; they may also be saturated
        #   and then don't follow the transparency
        bright = cv2.dilate((level > self.opts.star_threshold).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        star_px &= ~bright
        bg.star_idx = np.flatnonzero(star_px)
        bg.stars = np.where(star_px, stars, 0).astype(np.float32)


    @staticmethod
    def skyLevel(median):
        """ The local sky level of a median image: a median filter, on a 4x smaller image for speed. """

        h, w = median.shape
        small = cv2.resize(median, (w//4, h//4), interpolation=cv2.INTER_AREA)

        return cv2.resize(ndimage.median_filter(small, size=5), (w, h), interpolation=cv2.INTER_LINEAR)


    def normalize(self, frames, k):
        """ Normalize the frames of the block k to unit noise, in place. The background of every frame is interpolated
            linearly in time between the backgrounds of the blocks (at their middle frames). The brightness of
            the stars changes with the transparency of the sky, by up to a few times within a minute in thin
            clouds: the change of the star template which fits the frame best is subtracted from every frame.
            Otherwise the stars which are too faint to be masked leave residuals, and the residuals of stars
            along a line make tracks of slow objects.

        Arguments:
            frames: [ndarray] All frames of the block (float32), normalized in place.
            k: [int] Index of the block.

        Return:
            [ndarray] The normalized frames.
        """

        first, last = self.block_starts[k], self.block_ends[k]
        mid = (first + last - 1)/2.0
        bg = self.backgrounds[k]

        z = frames
        for i in range(len(z)):
            f = first + i
            j = k - 1 if f < mid else k + 1
            z[i] -= bg.median
            if j in self.backgrounds:
                mid_j = (self.block_starts[j] + self.block_ends[j] - 1)/2.0
                z[i] -= np.float32((f - mid)/(mid_j - mid))*(self.backgrounds[j].median - bg.median)

        # Change of the brightness of the stars in every frame, relative to the template
        if (bg.star_idx is not None) and (len(bg.star_idx) >= MIN_STAR_TEMPLATE_PX):
            template = bg.stars.ravel()[bg.star_idx]
            scale = z.reshape(len(z), -1)[:, bg.star_idx].dot(template)/np.dot(template, template)
            for i in range(len(z)):
                z[i] -= scale[i]*bg.stars

        z /= bg.noise

        return z


    def staticMask(self, background):
        """ Mask of the bright static sources in the median of the frames (stars whose fluctuations are well
            above the noise), of the user mask, and of the image border. True means masked. Fainter stars
            are taken out by the correction of the transparency (see normalize), and sources which vary by
            themselves by their persistence over the whole input (see removePersistent).

        Arguments:
            background: [BlockBackground] Background of the block.

        Return:
            [ndarray] Boolean mask.
        """

        # Brightness of the static sources relative to the noise of the sky in one frame
        level = background.star_level
        static = cv2.dilate((level > self.opts.star_threshold).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

        if self.mask is not None:
            static |= self.mask.img == 0

        m = self.opts.edge_margin
        if m > 0:
            static[:m] = static[-m:] = True
            static[:, :m] = static[:, -m:] = True

        return static


    def removeTrails(self, frame, static):
        """ Remove the trails of very bright moving sources from a normalized frame, in place. On some sensors
            a bright source leaves a faint trail along its whole column and along its row, at a fraction of a
            percent of its peak. They move with the source, so the search finds tracks along them: the rows
            and columns of every source above trail_level are set to the background in the frame, except at
            the source itself.

        Arguments:
            frame: [ndarray] Normalized frame, modified in place.
            static: [ndarray] Mask of static sources (True = masked).
        """

        if self.opts.trail_level <= 0:
            return

        bright = frame > self.opts.trail_level
        if not bright.any():
            return
        bright &= ~static
        if not bright.any():
            return

        # The sources, brightest first. A component in the rows or columns of a brighter source is a part of its
        #   trail (the trail of a very bright source can itself be above the level), not a source
        labels, n = ndimage.label(bright)
        peaks = ndimage.maximum(frame, labels, index=np.arange(1, n + 1))
        boxes = ndimage.find_objects(labels)
        sources = []
        for i in np.argsort(peaks)[::-1]:
            sl = boxes[i]
            in_trail = False
            for s0 in sources:
                rows = (sl[0].start < s0[0].stop + TRAIL_WIDTH) and (sl[0].stop > s0[0].start - TRAIL_WIDTH)
                cols = (sl[1].start < s0[1].stop + TRAIL_WIDTH) and (sl[1].stop > s0[1].start - TRAIL_WIDTH)
                if rows or cols:
                    in_trail = True
                    break
            if not in_trail:
                sources.append(sl)

        # The source itself, with the wings of its PSF, is kept
        boxes = sources
        kept = []
        for sl in boxes:
            ys = slice(max(sl[0].start - TRAIL_KEEP, 0), sl[0].stop + TRAIL_KEEP)
            xs = slice(max(sl[1].start - TRAIL_KEEP, 0), sl[1].stop + TRAIL_KEEP)
            kept.append((ys, xs, frame[ys, xs].copy()))

        for sl in boxes:
            frame[max(sl[0].start - TRAIL_WIDTH, 0):sl[0].stop + TRAIL_WIDTH, :] = 0
            frame[:, max(sl[1].start - TRAIL_WIDTH, 0):sl[1].stop + TRAIL_WIDTH] = 0

        for ys, xs, patch in kept:
            frame[ys, xs] = patch


    ### Search ###

    def searchBlock(self, z, static, block_first, stackers):
        """ Velocity search of a block of normalized frames.

        Arguments:
            z: [ndarray] Frames normalized to unit noise, (n, height, width).
            static: [ndarray] Mask of static sources (True = masked).
            block_first: [int] Index of the first frame of the block.
            stackers: [list] (run_frames, VelocityStacker) to run.

        Return:
            [list] Hits.
        """

        # Single frame outliers (scintillating stars, cosmic rays) are clipped, a faint object is never that
        #   bright in one frame, and a bright one is detected anyway
        b = self.search_bin
        sigma_b = self.psf_sigma/b
        zb = np.empty((len(z), z.shape[1]//b, z.shape[2]//b), dtype=np.float32)
        for i in range(len(z)):
            self.removeTrails(z[i], static)

            # The masked pixels don't take part in the stacks of their neighbours either
            z[i][static] = 0
            binned = binFrames(np.clip(z[i:i + 1], -self.opts.clip, self.opts.clip), b)[0]
            zb[i] = cv2.GaussianBlur(binned.astype(np.float32), (0, 0), sigma_b)

        static_b = binFrames(static[None].astype(np.float32), b)[0] > 0
        static_b = cv2.dilate(static_b.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

        hits = []
        for run_frames, stacker in stackers:

            if run_frames not in self.persistence_counts:
                self.persistence_counts[run_frames] = np.zeros(static_b.shape, dtype=np.int32)
                self.persistence_runs[run_frames] = 0

            for r0 in range(0, len(zb) - run_frames + 1, run_frames):

                t1 = time()
                try:
                    stack_max, vel_idx = stacker.stackMax(zb[r0:r0 + run_frames])
                except Exception:
                    if not stacker.use_gpu:
                        raise
                    # E.g. out of GPU memory with several workers: continue on the CPU
                    log.error('Matched filter: the GPU failed, continuing on the CPU:\n' + traceback.format_exc())
                    stacker.use_gpu = False
                    self.opts.use_gpu = False
                    stack_max, vel_idx = stacker.stackMax(zb[r0:r0 + run_frames])
                self.timing['stack'] += time() - t1

                # Normalize by the distribution of the maximum over the velocities in the unmasked image
                vals = stack_max[~static_b]
                med = np.median(vals)
                scale = 1.4826*np.median(np.abs(vals - med))
                zmax = (stack_max - med)/max(scale, 1e-6)
                zmax[static_b] = 0

                above = zmax > self.opts.threshold
                self.persistence_counts[run_frames] += above
                self.persistence_runs[run_frames] += 1

                peaks = (zmax == cv2.dilate(zmax, np.ones((5, 5), np.uint8))) & above
                for yb, xb in zip(*np.nonzero(peaks)):
                    vxb, vyb = stacker.velocities[vel_idx[yb, xb]]
                    hits.append(Hit(block_first + r0 + (run_frames - 1)/2.0, (xb + 0.5)*b - 0.5,
                                    (yb + 0.5)*b - 0.5, vxb*b, vyb*b, float(zmax[yb, xb]), run_frames))

        return hits


    def search(self):
        """ Pass 1: background, static masks and the velocity search over the whole file.

        Return:
            [list] Hits.
        """

        n_block = self.opts.block_frames
        b = self.search_bin

        stackers = []
        speed_max_b = self.opts.speed_max if self.opts.velocity_search else 0.0
        for run_frames in self.opts.run_frames:
            stackers.append((run_frames, VelocityStacker(velocityGrid(speed_max_b, run_frames, b), run_frames,
                                                         use_gpu=self.opts.use_gpu)))
        if self.opts.slow_run_frames > max(self.opts.run_frames):
            slow_speed = min(4.0/self.opts.slow_run_frames*b, self.opts.speed_max)
            stackers.append((self.opts.slow_run_frames, VelocityStacker(
                velocityGrid(slow_speed, self.opts.slow_run_frames, b), self.opts.slow_run_frames,
                use_gpu=self.opts.use_gpu)))
        for run_frames, st in stackers:
            log.info('Matched filter: {:d} velocities for runs of {:d} frames{:s}'.format(len(st.velocities),
                run_frames, ' (GPU)' if st.use_gpu else ''))

        # Blocks of frames, the last one may be shorter
        starts = list(range(0, self.total_frames, n_block))
        if (len(starts) > 1) and (self.total_frames - starts[-1] < n_block//2):
            starts = starts[:-1]
        self.block_starts = starts
        self.block_ends = starts[1:] + [self.total_frames]

        # The background follows changes of the sky (e.g. thin clouds) on the time scale of the blocks. An
        #   object slower than about a PSF width per 100 frames is too slow to be detected anyway
        self.backgroundPass()

        # Number of runs in which every pixel of the search image is above the threshold
        self.persistence_counts = {}
        self.persistence_runs = {}

        hits = []
        for k, (first, last) in enumerate(zip(self.block_starts, self.block_ends)):

            t1 = time()
            frames = self.readFrames(first, last - first)
            self.timing['read'] += time() - t1

            bg = self.backgrounds[k]
            if self.psf_sigma is None:
                # The stars of the mean of the block, against the noise of the sky in the mean
                sigma = estimatePSFSigma(frames.mean(axis=0) - bg.sky, bg.sky_noise/math.sqrt(len(frames)))
                if sigma is None:
                    sigma = 1.0
                    log.warning('Matched filter: the PSF sigma could not be measured on the stars, 1.0 px '
                                'assumed (set mf_psf_sigma)')
                self.psf_sigma = sigma
                log.info('Matched filter: PSF sigma {:.2f} px'.format(self.psf_sigma))

            t1 = time()
            bg.mean = frames.mean(axis=0)
            static = self.staticMask(bg)
            self.static_masks[k] = static
            z = self.normalize(frames, k)
            del frames
            self.timing['background'] += time() - t1

            t1 = time()
            hits += self.searchBlock(z, static, first, stackers)
            self.timing['search'] += time() - t1

        return self.removePersistent(hits)


    def removePersistent(self, hits):
        """ Remove the hits at persistent pixels: above the threshold in a large fraction of all runs of the
            input (e.g. flickering pixels, variable stars). It is judged over the whole input, as a slow object
            stays at a position for a large part of a block: at the slowest speeds of the search it takes a few
            hundred frames to cross the PSF.

        Arguments:
            hits: [list] Hits of the search.

        Return:
            [list] Hits at pixels which are not persistent.
        """

        b = self.search_bin
        persistent = {}
        for run_frames, counts in self.persistence_counts.items():
            limit = max(2, self.opts.persistence*self.persistence_runs[run_frames])
            persistent[run_frames] = cv2.dilate((counts > limit).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

        kept = []
        for h in hits:
            mask = persistent[h.run]
            xb = int(round((h.x + 0.5)/b - 0.5))
            yb = int(round((h.y + 0.5)/b - 0.5))
            if not mask[yb, xb]:
                kept.append(h)

        return kept


    ### Search of very slow objects ###

    def starDrift(self, step=SLOW_REF_BLOCKS):
        """ Motion of the stars across the image (px per frame), e.g. the diurnal motion for a fixed camera,
            from the shift between the medians of blocks step apart. Zero if there are too few blocks.
        """

        n = len(self.block_starts)
        if n <= step:
            return 0.0, 0.0

        window = cv2.createHanningWindow((self.width, self.height), cv2.CV_32F)
        shifts = []
        for k in range(0, n - step, max(1, (n - step)//8)):
            a, b = self.backgrounds[k], self.backgrounds[k + step]
            img_a = np.clip(a.raw_median - a.sky, 0, None).astype(np.float32)
            img_b = np.clip(b.raw_median - b.sky, 0, None).astype(np.float32)
            (dx, dy), response = cv2.phaseCorrelate(img_a, img_b, window)
            dt = self.block_starts[k + step] - self.block_starts[k]
            if response > 0.05:
                shifts.append((dx/dt, dy/dt))

        if not shifts:
            return 0.0, 0.0

        return tuple(np.median(np.array(shifts), axis=0))


    def drift(self):
        """ The motion of the stars (px per frame), measured once. """

        if getattr(self, '_drift', None) is None:
            self._drift = self.starDrift()

        return self._drift


    def referenceBackground(self, k):
        """ Background of the block k without the objects which are part of its median: the median of the medians
            of the blocks SLOW_REF_BLOCKS (and one more) before and after it, shifted by the motion of the stars,
            and the same of their noise. A slow object is elsewhere in those blocks, while the stars are aligned.

        Return:
            (median, noise): [tuple of ndarrays] or (None, None) if there are fewer than two such blocks.
        """

        n = len(self.block_starts)
        refs = [j for j in (k - SLOW_REF_BLOCKS - 1, k - SLOW_REF_BLOCKS, k + SLOW_REF_BLOCKS,
                            k + SLOW_REF_BLOCKS + 1) if (0 <= j < n) and (self.backgrounds[j].raw_median is not None)]
        if len(refs) < 2:
            return None, None

        drift = np.array(self.drift())
        mid = (self.block_starts[k] + self.block_ends[k] - 1)/2.0
        medians, noises = [], []
        for j in refs:
            mid_j = (self.block_starts[j] + self.block_ends[j] - 1)/2.0
            dx, dy = drift*(mid - mid_j)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            for src, out in ((self.backgrounds[j].raw_median, medians), (self.backgrounds[j].noise, noises)):
                out.append(cv2.warpAffine(src, M, (self.width, self.height), flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE))

        return np.median(np.array(medians), axis=0), np.median(np.array(noises), axis=0)


    def slowSearch(self):
        """ Search of objects too slow and too bright for the search of runs of frames. Such an object stays on
            the same pixels for more than half of a block, so it is part of the median of the block (the
            background) and masked as a star. The stars move together (e.g. with the diurnal motion), so the
            median of a block is compared with the medians of blocks a few blocks before and after it, shifted
            by the motion of the stars: the stars cancel, and an object which moves relative to them stands out.
            Its positions in consecutive blocks are linked into a track.

        Return:
            [list] Tracks as arrays of hits, rows [frame, x, y, z, vx, vy, run].
        """

        n = len(self.block_starts)
        if (n < 2*SLOW_REF_BLOCKS) or (self.backgrounds[0].raw_median is None):
            return []

        sigma = self.psf_sigma
        m = self.opts.edge_margin + int(math.ceil(3*sigma))
        drift = np.array(self.drift())

        # Candidates in every block: (frame, x, y, significance)
        cands = []
        for k in range(n):
            bg = self.backgrounds[k]
            mid = (self.block_starts[k] + self.block_ends[k] - 1)/2.0
            ref, _ = self.referenceBackground(k)
            if ref is None:
                continue

            # The difference against its noise (robust, over the whole image), smoothed with the PSF
            diff = cv2.GaussianBlur((bg.raw_median - ref).astype(np.float32), (0, 0), sigma)
            noise = 1.4826*np.median(np.abs(diff - np.median(diff))) + 1e-6
            z = diff/noise

            # A residual of a star is small compared with the star itself
            star = cv2.GaussianBlur((ref - bg.sky).astype(np.float32), (0, 0), sigma)/noise

            peaks = (z == cv2.dilate(z, np.ones((5, 5), np.uint8))) & (z > SLOW_THRESHOLD) & (z > 0.5*star)
            if self.mask is not None:
                peaks &= self.mask.img > 0
            peaks[:m] = peaks[-m:] = False
            peaks[:, :m] = peaks[:, -m:] = False
            for y, x in zip(*np.nonzero(peaks)):
                cands.append((mid, float(x), float(y), float(z[y, x]), k))

        if not cands:
            return []

        # Link the candidates of consecutive blocks: an object moves less than the slow limit per frame, and
        #   relative to the stars (a star residual moves with the stars)
        cands = np.array(cands)
        block_len = float(self.opts.block_frames)
        reach = SLOW_MAX_SPEED*block_len + 2*sigma
        used = np.zeros(len(cands), dtype=bool)
        tracks = []
        for i in np.argsort(-cands[:, 3]):
            if used[i]:
                continue
            chain = [i]
            used[i] = True
            for direction in (1, -1):
                while True:
                    cur = chain[-1] if direction == 1 else chain[0]
                    k_next = cands[cur, 4] + direction
                    if len(chain) >= 2:
                        pts = cands[sorted(chain, key=lambda j: cands[j, 0])]
                        vx = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
                        vy = np.polyfit(pts[:, 0], pts[:, 2], 1)[0]
                        tol = 2*sigma + 1
                    else:
                        vx = vy = 0.0
                        tol = reach
                    best, best_d = None, None
                    for j in np.nonzero((cands[:, 4] == k_next) & ~used)[0]:
                        dt = cands[j, 0] - cands[cur, 0]
                        d = math.hypot(cands[j, 1] - cands[cur, 1] - vx*dt, cands[j, 2] - cands[cur, 2] - vy*dt)
                        if (d <= tol) and ((best_d is None) or (d < best_d)):
                            best, best_d = j, d
                    if best is None:
                        break
                    used[best] = True
                    if direction == 1:
                        chain.append(best)
                    else:
                        chain.insert(0, best)

            if len(chain) < SLOW_MIN_BLOCKS:
                continue

            pts = cands[sorted(chain, key=lambda j: cands[j, 0])]
            vx = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
            vy = np.polyfit(pts[:, 0], pts[:, 2], 1)[0]

            # Moving with the stars: a residual of a star
            if math.hypot(vx - drift[0], vy - drift[1]) < SLOW_MIN_RELATIVE_SPEED:
                continue

            run = self.opts.block_frames//2
            tracks.append(np.column_stack([pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], np.full(len(pts), vx),
                                           np.full(len(pts), vy), np.full(len(pts), run)]))

        return tracks


    ### Linking ###

    def linkHits(self, hits):
        """ Link the hits of consecutive runs into tracks, predicting the position of the next hit from the
            motion of the track so far.

        Arguments:
            hits: [list] Hits of all runs.

        Return:
            [list] Tracks, each an array of hits as rows [frame, x, y, z, vx, vy, run].
        """

        b = self.search_bin
        tracks = []

        for run_frames in sorted(set(h.run for h in hits)):

            group = [h for h in hits if h.run == run_frames]
            if not group:
                continue

            arr = np.array([[h.frame, h.x, h.y, h.z, h.vx, h.vy, h.run] for h in group])
            by_frame = collections.defaultdict(list)
            for i, f in enumerate(arr[:, 0]):
                by_frame[round(f, 1)].append(i)

            # Velocity grid step (px/frame) and the tolerances of the linking
            step = b/max(run_frames - 1.0, 1.0)
            used = np.zeros(len(arr), dtype=bool)

            for i in np.argsort(-arr[:, 3]):

                if used[i]:
                    continue

                chain = [i]
                used[i] = True

                for direction in (1, -1):
                    while True:

                        # Motion of the chain: a line fit of its hits, or the velocity of the hit
                        cur = chain[-1] if direction == 1 else chain[0]
                        if len(chain) >= 3:
                            pts = arr[sorted(chain, key=lambda j: arr[j, 0])]
                            vx = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
                            vy = np.polyfit(pts[:, 0], pts[:, 2], 1)[0]
                        else:
                            vx, vy = arr[cur, 4], arr[cur, 5]

                        best, best_z = None, 0
                        for gap in (1, 2, 3):
                            t = round(arr[cur, 0] + direction*gap*run_frames, 1)
                            for j in by_frame.get(t, []):
                                if used[j]:
                                    continue
                                dt = arr[j, 0] - arr[cur, 0]
                                dist = math.hypot(arr[j, 1] - (arr[cur, 1] + vx*dt),
                                                  arr[j, 2] - (arr[cur, 2] + vy*dt))
                                dv = math.hypot(arr[j, 4] - vx, arr[j, 5] - vy)
                                # The velocity of a hit is known to about a PSF width over the run
                                if (dist <= 1.5*b + 1.5*step*abs(dt)) and (dv <= 3*step) and (arr[j, 3] > best_z):
                                    best, best_z = j, arr[j, 3]
                            if best is not None:
                                break

                        if best is None:
                            break

                        used[best] = True
                        if direction == 1:
                            chain.append(best)
                        else:
                            chain.insert(0, best)

                if len(chain) >= self.opts.min_hits:
                    tracks.append(arr[chain])

        return self.mergeTracks(tracks)


    def mergeTracks(self, tracks):
        """ Merge the tracks of the same object found in runs of different lengths, or in pieces. Two tracks
            are the same object if they overlap in time and their interpolated positions are close, or one
            continues the other.

        Arguments:
            tracks: [list] Tracks as arrays of rows [frame, x, y, z, ...].

        Return:
            [list] Merged tracks, sorted by time.
        """

        tol = 2.0*self.search_bin

        # A merged track can now overlap or continue a track kept before, so the merging is repeated
        while True:
            merged = self._mergeOnce(tracks, tol)
            if len(merged) == len(tracks):
                break
            tracks = merged

        return sorted(merged, key=lambda t: t[0, 0])


    def _mergeOnce(self, tracks, tol):
        """ One pass of mergeTracks. """

        tracks = sorted(tracks, key=lambda t: -len(t))

        merged = []
        for tr in tracks:

            joined = False
            for k, m in enumerate(merged):

                # Overlap in time: compare the positions at the frames of the shorter track
                lo, hi = max(tr[0, 0], m[0, 0]), min(tr[-1, 0], m[-1, 0])
                if hi >= lo:
                    fr = np.concatenate([tr[:, 0], m[:, 0]])
                    fr = fr[(fr >= lo) & (fr <= hi)]
                    if len(fr):
                        dx = np.interp(fr, m[:, 0], m[:, 1]) - np.interp(fr, tr[:, 0], tr[:, 1])
                        dy = np.interp(fr, m[:, 0], m[:, 2]) - np.interp(fr, tr[:, 0], tr[:, 2])
                        if np.median(np.hypot(dx, dy)) <= tol:
                            merged[k] = self._union(m, tr)
                            joined = True
                            break

                # One continues the other: extrapolate the end of the earlier one
                first, second = (m, tr) if m[-1, 0] < tr[0, 0] else (tr, m)
                gap = second[0, 0] - first[-1, 0]
                if 0 < gap <= 4*self.opts.slow_run_frames:
                    end = first[-min(len(first), 5):]
                    if np.ptp(end[:, 0]) > 0:
                        vx = np.polyfit(end[:, 0], end[:, 1], 1)[0]
                        vy = np.polyfit(end[:, 0], end[:, 2], 1)[0]
                    else:
                        vx, vy = end[-1, 4], end[-1, 5]
                    pred = end[-1, 1:3] + np.array([vx, vy])*gap
                    if np.hypot(*(second[0, 1:3] - pred)) <= tol + 0.05*gap*math.hypot(vx, vy):
                        merged[k] = self._union(m, tr)
                        joined = True
                        break

            if not joined:
                merged.append(tr)

        return merged


    @staticmethod
    def _union(a, b):
        """ Hits of two tracks of the same object, sorted by time. """

        u = np.vstack([a, b])
        return u[np.argsort(u[:, 0], kind='stable')]


    def acceptTrack(self, track):
        """ A track is kept if its speed is in the range and it moves by more than a few PSF widths (the
            residuals of variable stars are stationary). """

        if np.ptp(track[:, 0]) <= 0:
            return False
        vx = np.polyfit(track[:, 0], track[:, 1], 1)[0]
        vy = np.polyfit(track[:, 0], track[:, 2], 1)[0]
        speed = math.hypot(vx, vy)

        return (self.opts.speed_min <= speed <= self.opts.speed_max*1.2) \
            and (speed*np.ptp(track[:, 0]) >= self.minDisplacement())


    def minDisplacement(self):
        """ Smallest motion of a detection (px). """

        return self.opts.min_displacement*2.355*self.psf_sigma


    ### Measurement ###

    def run(self):
        """ Detect and measure the objects of the whole input.

        Return:
            [list] Detections as [rho, theta, centroids], centroids rows [frame, x, y, intensity, background,
                snr, saturated] in the unbinned image, as from Detection.detectMeteors.
        """

        t0 = time()
        if self.total_frames < max(self.opts.run_frames):
            log.warning('Matched filter: only {:d} frames, nothing to search'.format(self.total_frames))
            self.timing['total'] = time() - t0
            return []

        hits = self.search()
        linked = self.linkHits(hits)

        # Bright objects too slow for the search (they are part of the background of a block)
        t1 = time()
        slow_tracks = self.slowSearch()
        if slow_tracks:
            linked = self.mergeTracks(linked + slow_tracks)
        self.timing['slow_search'] = time() - t1

        tracks = [tr for tr in linked if self.acceptTrack(tr)]
        self._debug = (hits, linked)
        log.info('Matched filter: {:d} hits, {:d} candidate tracks ({:d} from the search of slow objects)'.format(
            len(hits), len(tracks), len(slow_tracks)))

        # The background of the slow objects is estimated without them before they are measured, or they would
        #   be measured on a background which contains them (and on pixels masked as stars)
        if slow_tracks:
            t1 = time()
            self.maskObjects([tr[:, :3] for tr in slow_tracks])
            self.timing['slow_background'] += time() - t1

        t1 = time()
        measured = self.measure(tracks)
        self.timing['measure'] += time() - t1

        detections = []
        for cent in measured:

            # The positions of the measurements combined over the neighbouring measurements
            if self.opts.smooth_frames > 0:
                xy, keep = smoothPositions(cent, self.opts.smooth_frames)
                cent = cent[keep]
                cent[:, 1:3] = xy

            # Only tracks which last and move long enough are kept
            if (cent[-1, 0] - cent[0, 0] < self.opts.min_frames) \
                    or (np.hypot(*(cent[-1, 1:3] - cent[0, 1:3])) < self.minDisplacement()):
                continue

            # One row per frame: the intensity of every frame, at the position on the track
            cent = perFrameRows(cent, self.opts.max_measure_frames)

            # Unbinned coordinates, and the intensity of the unbinned image (averaged bins hold the mean of the
            #   pixels, summed bins already their sum)
            if self.det_bin > 1:
                cent[:, 1:3] = cent[:, 1:3]*self.det_bin + (self.det_bin - 1)/2.0
                if self.config.detection_binning_method == 'avg':
                    cent[:, 3] *= self.det_bin**2

            # The polar line in the binned image, as for the normal detections
            (xa, ya), (xb, yb) = (cent[[0, -1], 1:3] - (self.det_bin - 1)/2.0)/self.det_bin
            rho, theta = getPolarLine(xa, self.height - ya, xb, self.height - yb, self.height, self.width)
            detections.append([rho, theta, cent])

        # Join the pieces of tracks (a faint object can be lost for a while, e.g. behind stars) and remove the
        #   duplicates. A slow object can be lost for many seconds while moving only a few PSF widths, so pieces
        #   are joined across any gap in which the object moved less than 15 PSF widths, if the second piece
        #   begins within 2 PSF widths of the extrapolated end of the first one
        # The distances of the joining are in the binned image (as is the image size), the ones of the duplicates
        #   in the unbinned image of the centroids
        fwhm = 2.355*self.psf_sigma
        max_gap = 4*max(self.opts.run_frames + [self.opts.slow_run_frames])
        detections = joinContinuousDetections(detections, max_gap, 15*fwhm, 2*fwhm, self.height, self.width,
                                              bin_factor=self.det_bin)
        detections = removeDuplicateDetections(detections, fwhm*self.det_bin, 3*fwhm*self.det_bin)

        self.timing['total'] = time() - t0
        log.info('Matched filter: {:d} detections in {:.1f} s ({:s})'.format(len(detections),
            self.timing['total'], ', '.join('{:s} {:.1f} s'.format(k, v) for k, v in self.timing.items()
                                              if k != 'total')))

        return detections


    def apertureCorrection(self, star_list):
        """ Factor which puts the intensities of the detections on the scale of the intensities of the stars
            (CALSTARS), from which the magnitudes are calibrated. The stars are measured with the aperture of
            the detections (within 3 PSF sigmas) on the mean of the block of frames, minus the sky in an
            annulus around them, and
            the factor is the median ratio of their CALSTARS intensities to these sums. The two differ by up to
            ~15%, depending on how far the wings of the PSF reach beyond the aperture.

        Arguments:
            star_list: [list] Stars of the chunks, [ff_name, [(y, x, intensity, amplitude, fwhm, background, snr,
                saturated pixels), ...]] in the unbinned image, as from extractStarsFrameInterface.

        Return:
            (factor, n_stars): [tuple] The factor (None if there are fewer than MIN_APERTURE_STARS stars) and the
                number of stars it was measured on.
        """

        from RMS.Formats.FFfile import filenameToDatetime

        b = self.det_bin
        scale = b*b if (b > 1) and (self.config.detection_binning_method == 'avg') else 1.0
        radius = 3.0*self.psf_sigma

        # The sky around a star: the median of an annulus beyond the aperture
        r_in, r_out = radius + 3.0, radius + 7.0
        r = int(math.ceil(r_out))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        t_begin = self.img_handle.beginning_datetime
        starts = np.array(self.block_starts)
        m = self.opts.edge_margin + r + 1

        ratios = []
        for entry in star_list:
            ff_name, stars = entry[0], entry[1]
            try:
                dt = (filenameToDatetime(ff_name) - t_begin.replace(tzinfo=None)).total_seconds()
            except Exception:
                continue
            k = int(np.clip(np.searchsorted(starts, dt*self.img_handle.fps, side='right') - 1, 0, len(starts) - 1))
            bg = self.backgrounds.get(k)
            if bg is None:
                continue

            # The intensity of an object in a frame is on average the mean of its flux, and the stars of the
            #   calibration are measured on the average of their frames, so the mean of the block is used
            image = bg.mean if bg.mean is not None else bg.median

            for star in stars:
                y, x, intensity, saturated = star[0], star[1], star[2], star[7]
                if (intensity <= 0) or (saturated > 0):
                    continue
                xb, yb = (x - (b - 1)/2.0)/b, (y - (b - 1)/2.0)/b
                if not ((m <= xb < self.width - m) and (m <= yb < self.height - m)):
                    continue
                xi, yi = int(round(xb)), int(round(yb))
                patch = image[yi - r:yi + r + 1, xi - r:xi + r + 1]
                d2 = (xx + xi - xb)**2 + (yy + yi - yb)**2
                sky = np.median(patch[(d2 > r_in**2) & (d2 <= r_out**2)])
                total = float(np.sum(patch[d2 <= radius**2] - sky))
                if total > 0:
                    ratios.append(intensity/(total*scale))

        if len(ratios) < MIN_APERTURE_STARS:
            return None, len(ratios)

        return float(np.median(ratios)), len(ratios)


    def summary(self):
        """ Summary of the last run, for the logs and the done file of the input. """

        return {
            'aperture_correction': self.aperture_correction,
            'total_frames': int(self.total_frames),
            'psf_sigma': self.psf_sigma,
            'gpu': bool(self.opts.use_gpu),
            'timing_s': {k: round(v, 2) for k, v in self.timing.items()},
            'tracks': [{'first_frame': float(f0), 'last_frame': float(f1), 'significance': round(float(sig), 2),
                        'detected': bool(sig >= self.opts.track_significance)}
                       for f0, f1, sig in (s[:3] for s in self.significances)],
        }


    def measure(self, tracks):
        """ Pass 2: measure every track on the unbinned frames by fitting a moving PSF to short runs of frames,
            following the track with a local motion model.

        Arguments:
            tracks: [list] Tracks of hits.

        Return:
            [list] Centroid arrays of the accepted tracks.
        """

        if not tracks:
            return []

        states = [TrackMeasurement(tr, self.opts, self.psf_sigma) for tr in tracks]
        self.measurePass(states)
        cents = [st.centroids() for st in states]

        # A slow object stays on the same pixels for a good part of the frames from which the background is
        #   estimated, so the background contains part of it: the background is estimated again without the
        #   pixels of the slow objects, and they are measured again
        slow = [i for i, cent in enumerate(cents) if (cent is not None) and (trackSpeed(cent) < SLOW_SPEED)]
        if slow:
            t1 = time()
            self.maskObjects([cents[i] for i in slow])
            again = [TrackMeasurement(tracks[i], self.opts, self.psf_sigma) for i in slow]
            self.measurePass(again)
            for i, st in zip(slow, again):
                cent = st.centroids()
                if cent is not None:
                    cents[i] = cent
            self.timing['slow_background'] += time() - t1

        return self.verify([cent for cent in cents if cent is not None])


    def maskObjects(self, cents, step=4):
        """ Estimate the background (median and noise) again in the region of the given tracks, without the
            pixels within a few PSF sigmas of the objects at the times of the sampled frames. The background of every block is the
            median of the samples of the block and its neighbours (as in backgroundPass), and only the region
            around the tracks is replaced.

        Arguments:
            cents: [list] Centroid arrays of the tracks (rows [frame, x, y, ...]).

        Keyword arguments:
            step: [int] Every step-th frame is sampled. 4 by default.
        """

        radius = 4*self.psf_sigma + 1
        n_blocks = len(self.block_starts)
        samples = {}
        touched = set()
        references = {}

        def blockSamples(j):
            if j not in samples:
                first, last = self.block_starts[j], self.block_ends[j]
                fr = np.arange(first, last, step)
                samples[j] = (fr, self.readFrames(first, len(fr), step=step))
            return samples[j]

        for k in range(n_blocks):

            near = range(max(k - 1, 0), min(k + 2, n_blocks))
            lo, hi = self.block_starts[near[0]], self.block_ends[near[-1]]

            # Positions of the tracks during the frames of the neighbouring blocks
            tracks = [c for c in cents if (c[-1, 0] >= lo) and (c[0, 0] < hi)]
            if not tracks:
                continue

            # A region around every track, and in it the samples masked where any of the tracks is at the time
            #   of the sample
            bg = self.backgrounds[k]
            for c in tracks:
                sel = (c[:, 0] >= lo - 64) & (c[:, 0] < hi + 64)
                if sel.sum() == 0:
                    continue
                pos = c[sel, 1:3]
                x0 = int(max(np.floor(pos[:, 0].min() - radius - 2), 0))
                x1 = int(min(np.ceil(pos[:, 0].max() + radius + 3), self.width))
                y0 = int(max(np.floor(pos[:, 1].min() - radius - 2), 0))
                y1 = int(min(np.ceil(pos[:, 1].max() + radius + 3), self.height))
                if (x1 <= x0) or (y1 <= y0):
                    continue

                yy, xx = np.mgrid[y0:y1, x0:x1]
                crops = []
                for j in near:
                    fr, frames = blockSamples(j)
                    for f, img in zip(fr, frames):
                        crop = img[y0:y1, x0:x1].astype(np.float64)
                        for t in tracks:
                            if t[0, 0] - 8 <= f <= t[-1, 0] + 8:
                                px, py = np.interp(f, t[:, 0], t[:, 1]), np.interp(f, t[:, 0], t[:, 2])
                                crop[(xx - px)**2 + (yy - py)**2 <= radius**2] = np.nan
                        crops.append(crop)

                crops = np.array(crops)
                masked = nanMedian(crops)

                # The noise too: the pixels which the object crosses during the block vary between the sky and the
                #   object, and their noise would weigh down the object itself in the fits
                mad = 1.4826*nanMedian(np.abs(crops - masked))

                # A very slow object covers some pixels in most of the samples of three blocks: there, the
                #   background of blocks further away, aligned on the stars, is used
                few = np.isfinite(crops).sum(axis=0) < MIN_MASKED_FRACTION*len(crops)
                if few.any():
                    if k not in references:
                        references[k] = self.referenceBackground(k)
                    ref_median, ref_noise = references[k]
                    if ref_median is not None:
                        masked = np.where(few, ref_median[y0:y1, x0:x1], masked)
                        mad = np.where(few, ref_noise[y0:y1, x0:x1], mad)

                region = bg.median[y0:y1, x0:x1]
                bg.median[y0:y1, x0:x1] = np.where(np.isfinite(masked), masked, region).astype(np.float32)
                region = bg.noise[y0:y1, x0:x1]
                bg.noise[y0:y1, x0:x1] = np.where(np.isfinite(mad), np.maximum(mad, 1.0), region).astype(np.float32)
                touched.add(k)

            # Drop the samples which are not needed any more
            for j in list(samples):
                if j < k - 1:
                    del samples[j]

        # The star template and the static mask were made from the medians with the objects in them: a slow
        #   object would be partly in the template (subtracted from the frames), and its path masked as a star
        #   in the verification
        for k in touched:
            self.starTemplate(self.backgrounds[k])
            self.static_masks[k] = self.staticMask(self.backgrounds[k])


    def measurePass(self, states):
        """ Measure the tracks in all blocks of frames they overlap. """

        for k, (first, last) in enumerate(zip(self.block_starts, self.block_ends)):

            active = [st for st in states if st.overlaps(first, last)]
            if not active:
                continue

            t1 = time()
            frames, saturated = self.readFrames(first, last - first, saturation=True)
            self.timing['read'] += time() - t1
            bg = self.backgrounds[k]
            z = self.normalize(frames, k)

            for st in active:
                st.measureBlock(z, first, last, bg, self.static_masks[k], saturated)


    def verify(self, measured):
        """ Keep the detections whose signal along a smooth track through their measurements is significant
            and belongs to a moving object. The measurements of a track made of noise follow the noise peaks
            around the predicted positions, but a smooth track does not. A track along the residuals of static
            sources (e.g. stars which vary with the transparency of the sky) has signal at its positions at all
            times, while a moving object is at a position only at one time. Two tests compare the signal along
            the track with the signal at the same positions when the object is elsewhere, and the smaller
            significance is taken:
                - off time: the positions in frames before and after the track, when the object is at least 3
                  PSF widths away;
                - reordered: the positions visited in a different order in the frames of the track (reversed,
                  and shifted cyclically), which takes out the residuals of static sources that change on time
                  scales longer than the track, as the off time can be long before or after the track for slow
                  objects.

        Arguments:
            measured: [list] Centroid arrays.

        Return:
            [list] The significant ones.
        """

        if not measured:
            return []

        fwhm = 2.355*self.psf_sigma

        # For every detection: positions in every frame of its track (on time), and the same positions in
        #   frames before and after, when the object is at least 3 PSF widths away (off time)
        checks = []
        for cent in measured:
            frames = np.arange(int(math.ceil(cent[0, 0])), int(math.floor(cent[-1, 0])) + 1)
            xs, ys = smoothTrack(cent, frames)
            speed = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])/max(frames[-1] - frames[0], 1)
            shift = int(math.ceil(max(3*fwhm/max(speed, 1e-3), 16)))
            off = [(frames + d, xs, ys) for d in (-shift, shift)]
            off = [(f[(f >= 0) & (f < self.total_frames)], x[(f >= 0) & (f < self.total_frames)],
                    y[(f >= 0) & (f < self.total_frames)]) for f, x, y in off]
            off_f = np.concatenate([o[0] for o in off])
            off_x = np.concatenate([o[1] for o in off])
            off_y = np.concatenate([o[2] for o in off])

            n = len(frames)
            orders = [np.arange(n)[::-1], np.roll(np.arange(n), n//3), np.roll(np.arange(n), 2*n//3),
                      np.roll(np.arange(n)[::-1], n//2)]
            reord = (np.tile(frames, len(orders)), np.concatenate([xs[o] for o in orders]),
                     np.concatenate([ys[o] for o in orders]))

            checks.append({'on': (frames, xs, ys), 'off': (off_f, off_x, off_y), 'reord': reord,
                           'on_sum': [0.0, 0.0], 'off_sum': [0.0, 0.0], 'reord_sum': [0.0, 0.0]})

        # The positions of every test in each block of frames
        starts = np.array(self.block_starts)
        for c in checks:
            for key in ('on', 'off', 'reord'):
                c[key + '_idx'] = blockIndices(c[key][0], starts)

        radius = int(math.ceil(2.5*self.psf_sigma))
        for k, (first, last) in enumerate(zip(self.block_starts, self.block_ends)):

            active = [c for c in checks if (k in c['on_idx']) or (k in c['off_idx'])]
            if not active:
                continue

            t1 = time()
            frames = self.readFrames(first, last - first)
            self.timing['read'] += time() - t1
            z = self.normalize(frames, k)

            # The pixels of the stars are left out: the stars which are too faint to be masked still vary
            #   (scintillation), and a slow track over a few of them would collect their residuals
            excluded = (self.static_masks[k] | (self.backgrounds[k].stars > 0)).astype(np.uint8)

            for c in active:
                for key in ('on', 'off', 'reord'):
                    fr, x, y = c[key]
                    sel = c[key + '_idx'].get(k)
                    if sel is not None:
                        sgz, sgg = forcedTrackSignal(z[fr[sel] - first], x[sel], y[sel], self.psf_sigma, radius,
                                                     excluded)
                        c[key + '_sum'][0] += sgz
                        c[key + '_sum'][1] += sgg

        kept = []
        for cent, c in zip(measured, checks):

            on_gz, on_gg = c['on_sum']
            if (on_gg <= 0) or (c['off_sum'][1] <= 0) or (c['reord_sum'][1] <= 0):
                continue

            # Signal of the moving object: the signal along the track minus the signal of the test scaled to
            #   the same weights
            tests = []
            for key in ('off_sum', 'reord_sum'):
                gz, gg = c[key]
                tests.append((on_gz - gz*on_gg/gg)/math.sqrt(on_gg*(1 + on_gg/gg)))
            significance = min(tests)

            self.significances.append((cent[0, 0], cent[-1, 0], significance) + tuple(tests))
            log.debug('Matched filter: track {:.0f}-{:.0f} significance {:.1f} (off time {:.1f}, reordered '
                      '{:.1f})'.format(cent[0, 0], cent[-1, 0], significance, *tests))
            if significance >= self.opts.track_significance:
                kept.append(cent)

        return kept



def nanMedian(stack):
    """ Median along the first axis, ignoring NaNs (NaN where all values are NaN). The same as np.nanmedian,
        which is many times slower on large arrays.

    Arguments:
        stack: [ndarray] Values, (n, ...).

    Return:
        [ndarray] Median, stack.shape[1:].
    """

    srt = np.sort(stack, axis=0)
    n = np.isfinite(stack).sum(axis=0)
    lo = np.take_along_axis(srt, np.maximum((n - 1)//2, 0)[None], axis=0)[0]
    hi = np.take_along_axis(srt, np.maximum(n//2, 0)[None], axis=0)[0]
    med = (lo + hi)/2

    return np.where(n > 0, med, np.nan)


def blockIndices(frames, starts):
    """ Indices of the frames in every block of frames.

    Arguments:
        frames: [ndarray] Frame numbers.
        starts: [ndarray] First frames of the blocks, sorted.

    Return:
        [dict] Block index -> indices of its frames, in their order in frames.
    """

    blocks = np.searchsorted(starts, frames, side='right') - 1
    order = np.argsort(blocks, kind='stable')
    sorted_blocks = blocks[order]
    out = {}
    for k in np.unique(sorted_blocks):
        if k < 0:
            continue
        lo, hi = np.searchsorted(sorted_blocks, [k, k + 1])
        out[int(k)] = order[lo:hi]

    return out


def perFrameRows(cent, max_frames):
    """ Rows of every frame of a track from its measurements: the measurements of faint objects combine several
        frames, but the intensity of each frame is kept, so the light curve has the full time resolution (e.g.
        short flashes are not averaged out). The positions of the frames are interpolated along the track
        through the (smoothed) positions of the measurements, and extrapolated with the motion at its ends.

    Arguments:
        cent: [ndarray] Measurements, rows [frame, x, y, intensity, background, snr, saturated, n_frames,
            intensities of the frames (max_frames columns), saturated pixel counts (max_frames columns)], sorted
            by frame. The frame of a measurement is the middle of its frames.
        max_frames: [int] Largest number of frames of a measurement.

    Return:
        [ndarray] Rows [frame, x, y, intensity, background, snr, saturated], one per frame. The signal-to-noise
            ratio of a frame is the one of its measurement divided by the square root of its number of frames.
    """

    t, x, y = cent[:, 0], cent[:, 1], cent[:, 2]
    if len(cent) >= 2:
        vx_lo, vy_lo = (x[1] - x[0])/(t[1] - t[0]), (y[1] - y[0])/(t[1] - t[0])
        vx_hi, vy_hi = (x[-1] - x[-2])/(t[-1] - t[-2]), (y[-1] - y[-2])/(t[-1] - t[-2])
    else:
        vx_lo = vy_lo = vx_hi = vy_hi = 0.0

    rows = []
    for row in cent:
        n = int(round(row[7]))
        for k in range(n):
            f = row[0] - (n - 1)/2.0 + k
            if f < t[0]:
                xf, yf = x[0] + vx_lo*(f - t[0]), y[0] + vy_lo*(f - t[0])
            elif f > t[-1]:
                xf, yf = x[-1] + vx_hi*(f - t[-1]), y[-1] + vy_hi*(f - t[-1])
            else:
                xf, yf = np.interp(f, t, x), np.interp(f, t, y)
            rows.append([f, xf, yf, row[8 + k], row[4], row[5]/math.sqrt(n), row[8 + max_frames + k]])

    return np.array(rows)


def smoothTrack(cent, frames, segment=512):
    """ Smooth positions of a track at the given frames: a quadratic fit of the measurements in every segment
        of the track (lines for short segments). The model is stiff, so it cannot follow the noise.

    Arguments:
        cent: [ndarray] Centroids, rows [frame, x, y, ...], sorted by frame.
        frames: [ndarray] Frames at which the positions are needed.

    Keyword arguments:
        segment: [int] Longest segment of the track with one quadratic (frames). 512 by default.

    Return:
        (xs, ys): [tuple of ndarrays] Positions.
    """

    t = cent[:, 0]
    n_seg = max(1, int(math.ceil((t[-1] - t[0] + 1)/segment)))
    edges = np.linspace(t[0], t[-1] + 1e-6, n_seg + 1)

    xs = np.empty(len(frames))
    ys = np.empty(len(frames))
    for i in range(n_seg):
        sel = (t >= edges[i]) & (t <= edges[i + 1])

        # Frames before the first and after the last measurement are extrapolated from the end segments
        fsel = ((frames >= edges[i]) | (i == 0)) & ((frames <= edges[i + 1]) | (i == n_seg - 1))
        if sel.sum() < 2:
            sel = np.argsort(np.abs(t - (edges[i] + edges[i + 1])/2))[:2]
        tt = t[sel]
        deg = 2 if (len(tt) >= 6) and (np.ptp(tt) > 128) else (1 if np.ptp(tt) > 0 else 0)
        t0 = tt.mean()
        px = np.polyfit(tt - t0, cent[sel, 1], deg)
        py = np.polyfit(tt - t0, cent[sel, 2], deg)
        xs[fsel] = np.polyval(px, frames[fsel] - t0)
        ys[fsel] = np.polyval(py, frames[fsel] - t0)

    return xs, ys



def trackSpeed(cent):
    """ Mean speed of a track (px per frame). """

    return math.hypot(cent[-1, 1] - cent[0, 1], cent[-1, 2] - cent[0, 2])/max(cent[-1, 0] - cent[0, 0], 1.0)



def smoothPositions(cent, half_window, reject=5.0):
    """ Positions of a track combined over the neighbouring measurements: a fit of x(t) and y(t) to the
        measurements within +-half_window frames of every measurement, weighted by their signal-to-noise ratio,
        evaluated at the measurement (quadratic, as short fast tracks across a distorted field are curved over
        the whole window). Measurements further from the smoothed track than reject times its scatter (both in
        units of the position errors) are outliers: they are removed and the rest is smoothed again.

        On objects added to real frames, this reduces the position errors 2 to 5 times (64 frames) without a
        bias on tracks curved like the tracks across a distorted field, as the motion over a few seconds is
        smooth. The positions of nearby measurements are then correlated.

    Arguments:
        cent: [ndarray] Centroids, rows [frame, x, y, intensity, background, snr, ...], sorted by frame.
        half_window: [float] Half width of the window (frames).

    Keyword arguments:
        reject: [float] Outlier limit, in units of the robust scatter about the smoothed track. 5 by default.

    Return:
        (xy, keep): [tuple] Smoothed positions of the kept measurements, shape (n_kept, 2), and the boolean mask
            of the kept measurements.
    """

    t = cent[:, 0]
    # Weights of the fit: the inverse of the position errors, which scale as 1/snr
    w = np.maximum(cent[:, 5], 0.5)
    keep = np.ones(len(t), dtype=bool)

    def _smooth(use):
        tu, xu, yu, wu = t[use], cent[use, 1], cent[use, 2], w[use]
        out = np.column_stack([cent[:, 1], cent[:, 2]]).astype(np.float64)
        # The window keeps its width near the ends of the track, shifted to stay inside it (a window cut at the
        #   end would fit fewer measurements and extrapolate the quadratic)
        start = np.clip(t - half_window, tu[0], max(tu[0], tu[-1] - 2*half_window))
        lo = np.searchsorted(tu, start, side='left')
        hi = np.searchsorted(tu, start + 2*half_window, side='right')
        for i in range(len(t)):
            n = hi[i] - lo[i]
            if n < 4:
                continue
            sl = slice(lo[i], hi[i])
            tt = tu[sl] - t[i]
            deg = 1 if n < 6 else 2
            out[i, 0] = np.polyfit(tt, xu[sl], deg, w=wu[sl])[-1]
            out[i, 1] = np.polyfit(tt, yu[sl], deg, w=wu[sl])[-1]
        return out

    xy = _smooth(keep)
    for _ in range(2):
        # Distances in units of the position errors of the measurements (which scale as 1/snr), so the faint
        #   measurements of a track whose brightness changes are not taken for outliers
        d = np.hypot(cent[:, 1] - xy[:, 0], cent[:, 2] - xy[:, 1])*w
        scale = 1.4826*np.median(d[keep]) + 1e-3
        new_keep = d <= reject*scale
        if (new_keep == keep).all() or (new_keep.sum() < 4):
            break
        keep = new_keep
        xy = _smooth(keep)

    return xy[keep], keep



class TrackMeasurement(object):
    def __init__(self, hits, opts, psf_sigma):
        """ Measurement of one track: the runs of frames to fit, the motion model, and the centroids.

        Arguments:
            hits: [ndarray] Hits of the track, rows [frame, x, y, z, vx, vy, run].
            opts: [MatchedFilterOptions] Options.
            psf_sigma: [float] PSF sigma (px).
        """

        self.hits = hits
        self.opts = opts
        self.psf_sigma = psf_sigma

        run = int(np.max(hits[:, 6]))
        ext = 2*max(run, 16)
        self.first = int(math.floor(hits[0, 0] - ext))
        self.last = int(math.ceil(hits[-1, 0] + ext))

        # Frames per measurement, chosen on the first fits
        self.n_frames = None
        self.probe = []

        # Lines fitted through the hits, per 16 frames (see predict)
        self.line_cache = {}

        # Accepted measurements: [frame, x, y, sigma_x, sigma_y, amp, sigma_amp, intensity, background, n_frames,
        #   saturated pixel count]
        self.meas = []
        self.status = []


    def overlaps(self, first, last):
        return (self.first < last) and (self.last >= first)


    def predict(self, frame):
        """ Position and velocity at the frame. Where the hits of the search cover the track, from a robust
            local fit of the hits (they are dense and independent of the measurements, so the prediction cannot
            drift away with the noise). Beyond the hits, extrapolated from the nearby accepted measurements.
        """

        h = self.hits
        window = max(64, 4*int(np.max(h[:, 6])))

        # The line through the hits changes slowly along the track, it is fitted once per 16 frames
        key = int(frame//16)
        if key not in self.line_cache:
            centre = 16*key + 8
            near = h[np.abs(h[:, 0] - centre) <= window]
            self.line_cache[key] = self._robustLineFit(near[:, 0], near[:, 1], near[:, 2]) \
                if (len(near) >= 3) and (np.ptp(near[:, 0]) > 0) else None
        fit = self.line_cache[key]
        if fit is not None:
            px, py = fit
            return np.polyval(px, frame), np.polyval(py, frame), px[0], py[0]

        if len(self.meas) >= 4:
            m = np.array(self.meas)
            near = m[np.argsort(np.abs(m[:, 0] - frame))[:12]]
            if np.ptp(near[:, 0]) > 0:
                return self._robustLine(near[:, 0], near[:, 1], near[:, 2], frame)

        near = h[np.argsort(np.abs(h[:, 0] - frame))[:6]]
        if (len(near) >= 2) and (np.ptp(near[:, 0]) > 0):
            return self._robustLine(near[:, 0], near[:, 1], near[:, 2], frame)

        return near[0, 1] + near[0, 4]*(frame - near[0, 0]), near[0, 2] + near[0, 5]*(frame - near[0, 0]), \
            near[0, 4], near[0, 5]


    @classmethod
    def _robustLine(cls, t, x, y, frame, reject=3.0):
        """ Position and velocity at the frame from a line fit of positions vs time (see _robustLineFit). """

        px, py = cls._robustLineFit(t, x, y, reject=reject)

        return np.polyval(px, frame), np.polyval(py, frame), px[0], py[0]


    @staticmethod
    def _robustLineFit(t, x, y, reject=3.0):
        """ Line fit of positions vs time, refitted without the points more than reject px off.

        Return:
            (px, py): [tuple] Polynomial coefficients of x(t) and y(t).
        """

        keep = np.ones(len(t), dtype=bool)
        for _ in range(2):
            if (keep.sum() < 2) or (np.ptp(t[keep]) <= 0):
                break
            px = np.polyfit(t[keep], x[keep], 1)
            py = np.polyfit(t[keep], y[keep], 1)
            resid = np.hypot(x - np.polyval(px, t), y - np.polyval(py, t))
            new_keep = resid <= reject
            if new_keep.sum() < 2 or (new_keep == keep).all():
                break
            keep = new_keep

        return px, py


    def fitRun(self, z, first, f0, n, saturated=None):
        """ Fit the moving PSF to frames f0 .. f0 + n - 1 (absolute) of the block starting at first. The fit uses
            a patch around the predicted position, so a fit which moved away from the prediction is repeated
            around the fitted position: the patch cut at the prediction would pull it back (the prediction of a
            fast object is less accurate along its motion).

        The saturated pixels (mask of the block, optional) are left out of the fit.

        Return:
            (mid, x, y, vx, vy, res): middle frame, predicted position and velocity, and the fit (see
                fitMovingPSF).
        """

        mid = f0 + (n - 1)/2.0
        x, y, vx, vy = self.predict(mid)
        dt = np.arange(n) - (n - 1)/2.0
        radius = int(math.ceil(3*self.psf_sigma + 0.5*math.hypot(vx, vy) + 1))
        frames = z[f0 - first:f0 - first + n]
        skip = saturated[f0 - first:f0 - first + n] if saturated is not None \
            else np.zeros(frames.shape, dtype=np.uint8)
        res = fitMovingPSF(frames, skip, x, y, dt, vx, vy, self.psf_sigma, radius)

        for _ in range(2):
            if not np.isfinite(res[0]) or (math.hypot(res[0] - x, res[1] - y) <= 0.5):
                break
            again = fitMovingPSF(frames, skip, res[0], res[1], dt, vx, vy, self.psf_sigma, radius)
            if not np.isfinite(again[0]):
                break
            moved = math.hypot(again[0] - res[0], again[1] - res[1])
            res = again
            if moved <= 0.1:
                break

        return mid, x, y, vx, vy, res


    def measureBlock(self, z, first, last, bg, static=None, saturated=None):
        """ Measure the runs of frames of the track in a block of normalized frames. Measurements on masked
            static sources (bright stars) are not accepted, the fit can be pulled to the star. The saturated
            pixels (mask of the raw frames of the block) are left out of the fit and counted.

        The intensity of a measurement is the background-subtracted sum of the pixels within 3 PSF sigmas of the
        segment the object moved along during a frame, averaged over the frames of the measurement (the same
        scale as the intensities of the normal detection, which sums the pixels of the object in a frame).
        """

        # Choose the number of frames per measurement on a few fits of 8 frames: the smallest power of two
        #   whose expected position error is below the limit. The fits are made where the search found the
        #   object, so a fit of the noise before the object appears doesn't make it look fainter
        if self.n_frames is None:
            run = int(np.max(self.hits[:, 6]))
            lo = max(first, int(math.floor(self.hits[0, 0] - run/2.0)))
            hi = min(last, int(math.ceil(self.hits[-1, 0] + run/2.0)) + 1)
            for f0 in range(lo + (-lo) % 8, hi - 7, 8):
                mid, x, y, vx, vy, res = self.fitRun(z, first, f0, 8, saturated)
                ok = np.isfinite(res[0]) and (res[2] > self.opts.min_sample_snr*res[6])
                ok = ok and (math.hypot(res[0] - x, res[1] - y) <= 1.5 + 4*math.hypot(vx, vy))
                if ok:
                    self.probe.append(math.hypot(res[4], res[5]))
                if len(self.probe) >= 3:
                    break

            # Wait for the next block if no fit succeeded yet (and the hits continue), otherwise decide with the
            #   fits there are, so the beginning of the track in this block is measured
            if (not self.probe) and (last < self.last) and (last <= self.hits[-1, 0]):
                return

            # The position error scales as 1/sqrt(n) with the number of frames
            sigma8 = self.opts.sigma_scale*np.median(self.probe) if self.probe else 1e3
            n = 1
            while (2*n <= self.opts.max_measure_frames) and (sigma8*math.sqrt(8.0/n) > self.opts.max_pos_error):
                n *= 2
            self.n_frames = n

            # Measure from the beginning of the track (the earlier blocks are not available any more, the
            #   probing started in this block)
            self.first = max(self.first, first)

        n = self.n_frames
        start = max(first, self.first)
        start += (-start) % n
        for f0 in range(start, min(last, self.last + 1) - n + 1, n):

            mid, x, y, vx, vy, res = self.fitRun(z, first, f0, n, saturated)
            ok = np.isfinite(res[0]) and (res[2] > 0) and (res[2] >= self.opts.min_sample_snr*res[6])
            gate = 1.5 + 0.5*math.hypot(vx, vy)*n
            ok = ok and (math.hypot(res[0] - x, res[1] - y) <= gate)
            ok = ok and (self.opts.sigma_scale*math.hypot(res[4], res[5]) <= 2*self.opts.max_pos_error)

            xi = int(min(max(round(res[0]), 0), bg.noise.shape[1] - 1)) if ok else 0
            yi = int(min(max(round(res[1]), 0), bg.noise.shape[0] - 1)) if ok else 0
            ok = ok and ((static is None) or (not static[yi, xi]))

            self.status.append((mid, ok))
            if not ok:
                continue

            # Photometry: sum of the pixels along the motion in every frame of the measurement
            dt = np.arange(n) - (n - 1)/2.0
            sat = saturated[f0 - first:f0 - first + n] if saturated is not None \
                else np.zeros((n,) + z.shape[1:], dtype=np.uint8)
            frames = z[f0 - first:f0 - first + n]
            radius = 3.0*self.psf_sigma
            sums, n_sat = streakAperture(frames, sat, bg.noise, res[0] + vx*dt, res[1] + vy*dt, vx, vy, radius)

            # Charge spilled by a very bright object: the wider sum is its intensity, and it counts as saturated
            core_noise = float(bg.noise[yi, xi])*math.sqrt(math.pi)*radius
            if np.max(sums) > SPILL_MIN_SNR*core_noise:
                wide, wide_sat = streakAperture(frames, sat, bg.noise, res[0] + vx*dt, res[1] + vy*dt, vx, vy,
                                                SPILL_RADIUS)
                spill = (sums > SPILL_MIN_SNR*core_noise) & (wide > SPILL_RATIO*sums)
                sums = np.where(spill, wide, sums)
                n_sat = np.where(spill, np.maximum(wide_sat, 1), n_sat)

            # The photometry of every frame is kept (padded to the largest number of frames per measurement)
            pad = self.opts.max_measure_frames - n
            sums_pad = list(sums) + [0.0]*pad
            sat_pad = list(n_sat) + [0]*pad

            # The uncertainties of the fit are scaled to the actual errors on real noise
            sc = self.opts.sigma_scale
            self.meas.append([mid, res[0], res[1], sc*res[4], sc*res[5], res[2], sc*res[6], float(np.mean(sums)),
                              float(bg.median[yi, xi]), n, int(np.max(n_sat))] + sums_pad + sat_pad)


    def centroids(self):
        """ The accepted measurements of the track as centroid rows, or None if the track is rejected.

        Return:
            [ndarray] Rows [frame, x, y, intensity, background, snr, saturated, n_frames, the intensities of the
                frames of the measurement (max_measure_frames columns), their saturated pixel counts
                (max_measure_frames columns)], see perFrameRows.
        """

        if len(self.meas) < self.opts.min_centroids:
            return None

        m = np.array(self.meas)
        m = m[np.argsort(m[:, 0])]

        # The track ends where 3 measurements in a row failed outside the frames of the hits
        status = sorted(self.status)
        lo, hi = self.hits[0, 0], self.hits[-1, 0]
        keep = np.ones(len(m), dtype=bool)
        fails = 0
        for f, ok in status:
            if f >= lo:
                break
            fails = 0 if ok else fails + 1
            if fails >= 3:
                keep &= m[:, 0] > f
        fails = 0
        for f, ok in status[::-1]:
            if f <= hi:
                break
            fails = 0 if ok else fails + 1
            if fails >= 3:
                keep &= m[:, 0] < f
        m = m[keep]

        # A few measurements at an end of the track after a long gap are not trusted: the local motion model
        #   which judges the outliers has no support there
        max_gap = max(64, 8*self.n_frames if self.n_frames else 64)
        while len(m) >= self.opts.min_centroids:
            gaps = np.nonzero(np.diff(m[:, 0]) > max_gap)[0]
            if len(gaps) and (gaps[0] + 1 < self.opts.min_centroids):
                m = m[gaps[0] + 1:]
            elif len(gaps) and (len(m) - gaps[-1] - 1 < self.opts.min_centroids):
                m = m[:gaps[-1] + 1]
            else:
                break

        # Remove the outliers from a local quadratic motion model
        for _ in range(2):
            if len(m) < self.opts.min_centroids:
                return None
            resid = np.zeros(len(m))
            for i in range(len(m)):
                # The measurements nearest in time (m is sorted by time)
                lo = min(max(i - 7, 0), max(len(m) - 15, 0))
                near = np.arange(lo, min(lo + 15, len(m)))
                near = near[near != i]
                deg = 2 if len(near) >= 8 else 1
                t0 = m[i, 0]
                px = np.polyfit(m[near, 0] - t0, m[near, 1], deg)
                py = np.polyfit(m[near, 0] - t0, m[near, 2], deg)
                sig = max(math.hypot(m[i, 3], m[i, 4]), 0.2)
                resid[i] = math.hypot(m[i, 1] - px[-1], m[i, 2] - py[-1])/sig
            good = resid < 4
            if good.all():
                break
            m = m[good]

        if len(m) < self.opts.min_centroids:
            return None

        snr = np.minimum(m[:, 5]/np.maximum(m[:, 6], 1e-6), 99.99)
        cent = np.column_stack([m[:, 0], m[:, 1], m[:, 2], m[:, 7], m[:, 8], snr, m[:, 10], m[:, 9], m[:, 11:]])

        return cent



def detectMatchedFilter(img_handle, config, mask=None, dark=None, flat_struct=None, star_list=None,
                        return_detector=False):
    """ Run the matched-filter detection on an image handle.

    Arguments:
        img_handle: [FrameInterface] Frame-based input opened with detection=True.
        config: [Config] Configuration object.

    Keyword arguments:
        mask: [MaskStruct] Mask, not binned. None by default.
        dark: [ndarray] Dark frame, not binned. None by default.
        flat_struct: [FlatStruct] Flat field, not binned. None by default.
        star_list: [list] Stars extracted from the input (see extractStarsFrameInterface). With enough stars,
            the intensities are put on the scale of the star intensities (see apertureCorrection). None by
            default.
        return_detector: [bool] Also return the detector (for its summary). False by default.

    Return:
        [list] Detections as [rho, theta, centroids], or (detections, detector) with return_detector. The
            detector is None for FF files.
    """

    from RMS.DetectionTools import binImageCalibration

    # FF files keep only the maximum and the average of their frames, the frames themselves are needed
    if img_handle.input_type == 'ff':
        log.warning('Matched filter: FF files have no frames to search, skipped')
        return ([], None) if return_detector else []

    # The calibration is binned in place, and the caller's objects may already be binned for the normal
    #   detection, so copies are binned
    if config.detection_binning_factor > 1:
        import copy
        mask, dark, flat_struct = binImageCalibration(config, copy.deepcopy(mask), copy.deepcopy(dark),
                                                      copy.deepcopy(flat_struct))

    # Limit the threads of the compiled code, e.g. when several files are processed in parallel
    if config.mf_threads > 0:
        import numba
        numba.set_num_threads(min(config.mf_threads, numba.config.NUMBA_NUM_THREADS))

    detector = MatchedFilterDetector(img_handle, config, mask=mask, dark=dark, flat_struct=flat_struct)
    detections = detector.run()

    # The intensities on the scale of the stars of the photometric calibration
    if star_list and detections:
        factor, n_stars = detector.apertureCorrection(star_list)
        if (factor is not None) and (0.5 <= factor <= 2.0):
            for det in detections:
                det[2][:, 3] *= factor
            detector.aperture_correction = round(factor, 4)
            log.info('Matched filter: intensities scaled by {:.3f} to the scale of the stars ({:d} stars)'.format(
                factor, n_stars))
        else:
            log.warning('Matched filter: no aperture correction of the intensities ({:d} stars, factor {})'.format(
                n_stars, factor))

    return (detections, detector) if return_detector else detections



def saveMatchedFilterResults(detections, star_list, img_handle, config, output_dir, platepar_path=None,
    chunk_frames=128, chunk_images=None, load_all=False, ecsv_out=False):
    """ Save the detections of the matched filter as an FTPdetectinfo with the suffix "mf", and the stars as a
        CALSTARS with the same suffix. With a platepar and stars, the detections are recalibrated.

    Arguments:
        detections: [list] Detections as [rho, theta, centroids].
        star_list: [list] Stars of the chunks, as from the star extraction.
        img_handle: [FrameInterface] Image handle of the file.
        config: [Config] Configuration object.
        output_dir: [str] Output directory.

    Keyword arguments:
        platepar_path: [str] Platepar, copied to the output directory for the recalibration. None by default.
        chunk_frames: [int] Frames per star extraction chunk. 128 by default.
        chunk_images: [list] (first_frame, nframes, ff_name) of the saved chunk images, the detections are
            assigned to them. None by default.
        load_all: [bool] Recalibrate all chunks, see ApplyRecalibrate.applyRecalibrate. False by default.
        ecsv_out: [bool] Save the detections as ECSV files. False by default.

    Return:
        [str] Path of the FTPdetectinfo.
    """

    from RMS.DetectStarsAndMeteors import saveResultsFrameInterface

    os.makedirs(output_dir, exist_ok=True)

    _, _, ftp_name = saveResultsFrameInterface(star_list, detections, img_handle, config,
                                               chunk_frames=chunk_frames, output_suffix=MATCHED_FILTER_SUFFIX,
                                               output_dir=output_dir, chunk_images=chunk_images)
    ftp_path = os.path.join(output_dir, ftp_name)

    if platepar_path and star_list and detections:
        from RMS.Astrometry.ApplyRecalibrate import applyRecalibrate
        shutil.copy2(platepar_path, os.path.join(output_dir, config.platepar_name))
        applyRecalibrate(ftp_path, config, generate_plot=False, load_all=load_all, generate_ufoorbit=False,
                         ecsv_out=ecsv_out)

    return ftp_path



def mergeNightMatchedFilter(night_dir, results_paths, config):
    """ Merge the FTPdetectinfo files of the matched filter of the files of a night into one file, in the
        matched-filter directory of the night directory.

    Arguments:
        night_dir: [str] Night directory.
        results_paths: [list] Results directories of the files of the night.
        config: [Config] Configuration object.

    Return:
        [str] Path of the merged file, None if there were no results of the matched filter.
    """

    out_dir = os.path.join(night_dir, MATCHED_FILTER_DIR)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    ftp_name = 'FTPdetectinfo_{:s}_{:s}.txt'.format(os.path.basename(night_dir), MATCHED_FILTER_SUFFIX)

    return mergeFTPdetectinfo([os.path.join(path, MATCHED_FILTER_DIR) for path in results_paths], out_dir,
                              ftp_name, config)



def mergeFTPdetectinfo(dirs, out_dir, ftp_name, config):
    """ Merge the FTPdetectinfo files of the matched filter in the given directories into one file.

    Arguments:
        dirs: [list] Directories with the FTPdetectinfo files (missing ones are skipped).
        out_dir: [str] Directory of the merged file.
        ftp_name: [str] Name of the merged file.
        config: [Config] Configuration object.

    Return:
        [str] Path of the merged file, None if no FTPdetectinfo file was found.
    """

    from RMS.Formats import FTPdetectinfo

    meteor_list = []
    fps = None
    found = False
    for mf_dir in dirs:

        # Only finished results (a failed or interrupted file has no done file)
        if not os.path.isfile(os.path.join(mf_dir, DONE_NAME)):
            continue

        # The recalibration keeps a backup of the uncalibrated file, which is not merged
        for file_name in sorted(os.listdir(mf_dir)):
            if not FTPdetectinfo.validDefaultFTPdetectinfo(file_name):
                continue

            found = True
            for entry in FTPdetectinfo.readFTPdetectinfo(mf_dir, file_name):
                ff_name, _, meteor_No, _, meteor_fps, _, _, _, _, rho, phi, meteor_meas = entry
                meteor_list.append([ff_name, meteor_No, rho, phi, [line[1:] for line in meteor_meas],
                                    meteor_fps])
                fps = meteor_fps

    if not found:
        return None

    os.makedirs(out_dir, exist_ok=True)
    FTPdetectinfo.writeFTPdetectinfo(sorted(meteor_list, key=lambda entry: (entry[0], entry[1])), out_dir,
        ftp_name, out_dir, config.stationID, fps if fps else config.fps, celestial_coords_given=True)

    return os.path.join(out_dir, ftp_name)



def processFile(file_path, config, output_dir, platepar_path=None, dark_path=None, flat_path=None,
    mask_path=None, extract_stars=True):
    """ Run the matched-filter detection on one file and save the FTPdetectinfo (with the suffix "mf"), and
        the CALSTARS if the stars are extracted. With a platepar, the detections are recalibrated.

    Arguments:
        file_path: [str] Input file.
        config: [Config] Configuration object.
        output_dir: [str] Output directory.

    Keyword arguments:
        platepar_path: [str] Platepar, copied to the output directory for the recalibration. None by default.
        dark_path, flat_path, mask_path: [str] Calibration images. None by default.
        extract_stars: [bool] Extract the stars (needed for the recalibration). True by default.

    Return:
        [dict] Summary of the processing (also saved as the done file in the output directory), with the path
            of the FTPdetectinfo.
    """

    from RMS.DetectionTools import loadImageCalibration
    from RMS.ExtractStarsFrameInterface import extractStarsFrameInterface
    from RMS.Formats.FrameInterface import detectInputType

    t0 = time()

    config.use_dark = dark_path is not None
    config.use_flat = flat_path is not None
    if dark_path:
        config.dark_file = os.path.abspath(dark_path)
    if flat_path:
        config.flat_file = os.path.abspath(flat_path)
    if mask_path:
        config.mask_file = os.path.abspath(mask_path)

    img_handle = detectInputType(file_path, config, detection=True, preload_video=True)
    if img_handle.input_type == 'ff':
        raise ValueError('FF files have no frames to search: {:s}'.format(file_path))
    mask, dark, flat_struct = loadImageCalibration(os.path.dirname(os.path.abspath(file_path)), config,
        dtype=img_handle.ff.dtype, byteswap=img_handle.byteswap)

    star_list = []
    if extract_stars:
        star_list = extractStarsFrameInterface(img_handle, config, flat_struct=flat_struct, dark=dark, mask=mask,
                                               save_calstars=False)

    detections, detector = detectMatchedFilter(img_handle, config, mask=mask, dark=dark, flat_struct=flat_struct,
                                               star_list=star_list, return_detector=True)

    ftp_path = saveMatchedFilterResults(detections, star_list, img_handle, config, output_dir,
                                        platepar_path=platepar_path,
                                        chunk_frames=getattr(img_handle, 'chunk_frames', 128))

    return saveSummary(output_dir, detector, input_file=os.path.abspath(file_path),
                       ftpdetectinfo=os.path.basename(ftp_path), detections=len(detections),
                       stars_extracted=bool(star_list), processing_time_s=round(time() - t0, 1))



def saveSummary(output_dir, detector, **extra):
    """ Save the summary of the processing of a file as its done file in the output directory: the tracks
        with their significance, the timing, and the given items.

    Arguments:
        output_dir: [str] Output directory of the file.
        detector: [MatchedFilterDetector] Detector after its run, or None.

    Keyword arguments:
        Items added to the summary.

    Return:
        [dict] The summary.
    """

    summary = detector.summary() if detector is not None else {}
    summary.update(extra)
    summary['processed_at'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, DONE_NAME), 'w') as f:
        json.dump(summary, f, indent=1)

    return summary



def processFiles(files, config, output_dir, force=False, **kwargs):
    """ Process input files one after the other, each into its own subdirectory of the output directory, and
        merge their detections into one FTPdetectinfo in the output directory. A file whose directory has a
        done file is skipped (unless forced), so an interrupted run can be started again; a file which fails is
        logged and the others are processed.

    Arguments:
        files: [list] Input files.
        config: [Config] Configuration object.
        output_dir: [str] Output directory.

    Keyword arguments:
        force: [bool] Process the files which were already processed. False by default.
        **kwargs: Passed to processFile (platepar_path, dark_path, flat_path, mask_path, extract_stars).

    Return:
        [list] Files which failed.
    """

    log.info('Matched filter: {:d} input files'.format(len(files)))

    out_dirs = []
    failed = []
    for path, name in zip(files, outputNames(files)):

        out = os.path.join(output_dir, name)
        out_dirs.append(out)
        if os.path.isfile(os.path.join(out, DONE_NAME)) and not force:
            log.info('Matched filter: already processed, skipped: {:s}'.format(path))
            continue

        # Partial results of an interrupted run are not kept
        shutil.rmtree(out, ignore_errors=True)
        try:
            summary = processFile(path, config, out, **kwargs)
            log.info('Matched filter: {:d} detections in {:s} ({:.0f} s)'.format(summary['detections'], path,
                                                                               summary['processing_time_s']))
        except Exception:
            log.error('Matched filter: processing failed for {:s}:\n{:s}'.format(path, traceback.format_exc()))
            failed.append(path)

    merged = mergeFTPdetectinfo(out_dirs, output_dir, 'FTPdetectinfo_{:s}_{:s}.txt'.format(
        os.path.basename(os.path.normpath(output_dir)), MATCHED_FILTER_SUFFIX), config)
    if merged:
        log.info('Matched filter: merged detections saved to {:s}'.format(merged))

    if failed:
        log.error('Matched filter: {:d} files failed: {:s}'.format(len(failed), ', '.join(failed)))

    return failed



def outputNames(files):
    """ Names of the output directories of the input files: the file names without the extension, and for files
        with the same name in different directories, the name followed by a short hash of the full path, so every
        file has its own directory and keeps it from run to run.

    Arguments:
        files: [list] Input files.

    Return:
        [list] Directory names.
    """

    import hashlib

    stems = [os.path.splitext(os.path.basename(path))[0] for path in files]
    counts = collections.Counter(stems)

    return [stem if counts[stem] == 1 else '{:s}_{:s}'.format(stem,
            hashlib.sha1(os.path.abspath(path).encode('utf-8')).hexdigest()[:8]) for stem, path in zip(stems, files)]



def findInputFiles(paths):
    """ Input files from a list of files and directories (searched recursively for video files).

    Arguments:
        paths: [list] Files and directories.

    Return:
        [list] Files, sorted, without duplicates.
    """

    files = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                files += [os.path.join(root, name) for name in names if name.lower().endswith(INPUT_EXTENSIONS)]
        else:
            files.append(path)

    return sorted(set(os.path.abspath(f) for f in files))



if __name__ == "__main__":

    from RMS.Logger import LoggingManager

    arg_parser = argparse.ArgumentParser(description="Matched-filter detection of faint moving objects in "
        "frame-based video (e.g. .vid files). Every input file gets its own output directory, with an "
        "FTPdetectinfo with the suffix 'mf' and a done file; files with a done file are skipped, so an interrupted "
        "run can be started again. The detections of all files are merged into one FTPdetectinfo in the output "
        "directory.")

    arg_parser.add_argument('input', nargs='+', help='Input files or directories (searched for video files).')
    arg_parser.add_argument('-c', '--config', required=True, help='Config file.')
    arg_parser.add_argument('-o', '--output', required=True, help='Output directory.')
    arg_parser.add_argument('-p', '--platepar', help='Platepar, for the recalibration of the detections '
                            '(astrometry and photometry).')
    arg_parser.add_argument('--dark', help='Dark frame.')
    arg_parser.add_argument('--flat', help='Flat field.')
    arg_parser.add_argument('--mask', help='Mask.')
    arg_parser.add_argument('--no-velocity-search', action='store_true', help='Only search slow objects.')
    arg_parser.add_argument('--gpu', choices=['auto', 'on', 'off'], help='Use the GPU for the search.')
    arg_parser.add_argument('--threads', type=int, help='Number of CPU threads (0 = all cores).')
    arg_parser.add_argument('--no-stars', action='store_true', help="Don't extract the stars (no recalibration).")
    arg_parser.add_argument('--force', action='store_true', help='Process the files which were already processed.')

    args = arg_parser.parse_args()

    config = cr.parse(args.config)
    if args.no_velocity_search:
        config.mf_velocity_search = False
    if args.gpu:
        config.mf_gpu = args.gpu
    if args.threads is not None:
        config.mf_threads = args.threads

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    config.data_dir = output_dir
    config.log_dir = 'logs'
    LoggingManager().initLogging(config, 'matched_filter_')

    files = findInputFiles(args.input)
    if not files:
        log.error('Matched filter: no input files found in {:s}'.format(', '.join(args.input)))
        sys.exit(1)

    failed = processFiles(files, config, output_dir, platepar_path=args.platepar,
                          dark_path=args.dark, flat_path=args.flat, mask_path=args.mask,
                          extract_stars=not args.no_stars, force=args.force)
    sys.exit(1 if failed else 0)
