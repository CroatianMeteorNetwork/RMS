""" Matched-filter detection of faint moving objects in frame-based video (e.g. .vid files).

The normal detection thresholds the maxpixel of a window of frames, so an object has to be well above the
noise in single frames. Here the frames are instead summed along the motion of the object (shift-and-add
along a grid of velocities) before thresholding. N frames of unit noise summed along the motion of an object of
amplitude A per frame give N*A of signal and sqrt(N) of noise, so the signal-to-noise ratio grows as sqrt(N):
an object at 1.5 times the noise per frame is at 6 sigma in a run of 16 frames.

    1. Background: the per-pixel median and noise of every block of frames, smoothed over the neighbouring
       blocks and interpolated in time. The frames are normalized to unit noise, z = (frame - median)/noise.
       The brightness of the stars changes with the transparency of the sky, so the best fitting scale of the
       star template is subtracted from every frame. Bright stars, the mask, the image border, and the rows
       and columns of trails of very bright sources are masked.
    2. Search: the normalized frames are binned 2x2, smoothed with the PSF, and summed along a grid of
       velocities over runs of frames (short runs for all motions, long runs for small motions). The best
       velocity of every pixel gives a detection image, whose local maxima above the threshold are the hits.
       Pixels above the threshold in a large part of the whole input are persistent and removed. Objects
       which stay on the same pixels for most of a block (part of its median) are found by comparing the
       medians of the blocks, aligned on the stars.
    3. Linking: the hits of consecutive runs are linked into tracks, predicting the next hit from the motion of
       the track, and the tracks of the same object are merged (only where the merged track moves as one
       object). The sums along other velocities through a bright object give tracks along its path at other
       speeds (ghosts), which are removed.
    4. Measurement: a moving PSF is fitted jointly to short runs of the unbinned frames, at the position
       predicted from the track (see MatchedFilterKernels.fitMovingPSF). The run length is chosen per track,
       as short as the required position accuracy allows (every frame for bright objects), and the track is
       followed with a local motion model, so curved and accelerating tracks are measured too. The positions
       are then combined along the track with a weighted local quadratic fit.
    5. Verification: the PSF-weighted signal along a smooth track through the measurements, minus the signal
       at the same positions when the object is elsewhere, over the pixels which are not on stars, has to be
       significant, and the light of the object has to be concentrated like that of a point source.
    6. Photometry: the background-subtracted sum of the pixels along the motion in every frame of the track,
       on the scale of the star intensities of the photometric calibration.
    7. Duplicates: the pieces of a track are joined, and the detections of an object measured twice are removed,
       also a fast object measured at two places of its streak (moving together, with the same light curve).

The results are written as an FTPdetectinfo file with the suffix "mf", apart from the results of the normal
detection, and recalibrated with the platepar. It runs in the monitor (mf_enable) after the normal detection
or instead of it (mf_replace_detection), or on files and directories from the command line
(python -m RMS.MatchedFilterDetection), which skips the files already processed so a run can be started again.
The velocity search runs on the GPU when numba can use CUDA, otherwise on all CPU cores (mf_threads).
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import copy
import json
import math
import shutil
import hashlib
import argparse
import datetime
import traceback
import collections
from concurrent.futures import ThreadPoolExecutor
from time import time

import numpy as np
import cv2
import numba
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyRecalibrate import applyRecalibrate
from RMS.DetectionTools import binImageCalibration, loadImageCalibration
from RMS.DetectStarsAndMeteors import saveResultsFrameInterface
from RMS.ExtractStarsFrameInterface import extractStarsFrameInterface
from RMS.Formats import FTPdetectinfo
from RMS.Formats.FFfile import filenameToDatetime, validFFName
from RMS.Formats.FrameBuffer import SegmentBuffer, segmentMinimum, segmentOverlap, segmentRanges, segmentUnit
from RMS.Formats.FrameInterface import detectInputType
from RMS.Formats.Platepar import Platepar
from RMS.Logger import getLogger, LoggingManager
from RMS.Routines import Image
from RMS.Routines import MaskImage
from RMS.Routines import RollingShutterCorrection
from RMS.Routines.DynamicFTPCompressionCy import sampleMedianMAD
from RMS.Routines.SkyQuality import CalibrationError, SkyQualityMap, SkyReference, chunkFirstFrames
from RMS.Routines.MatchedFilterKernels import (DeviceFrames, VelocityStacker, fitMovingPSF, forcedTrackSignal,
                                               streakAperture, CUDA_AVAILABLE)
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
INPUT_EXTENSIONS = ('.vid', '.fblk', '.mkv', '.mp4', '.avi', '.mov')

# Directories of FITS frames (one frame per file) are inputs too, as for the fitsdirs input of the monitor (the
#   .fit files of FRIPON cameras are read with headers which other cameras don't have)
FITS_EXTENSIONS = ('.fits',)

# Tracks slower than this (px per frame) are measured again on a background estimated without them
SLOW_SPEED = 0.1

# A step between the times of neighbouring frames longer than this many median steps is a gap in the recording,
#   where frames are missing (e.g. while the clock of the camera is synchronised)
TIME_GAP_FACTOR = 1.5

# The ends of a track where at most FAINT_END_BRIGHT_ROWS of the rows (in windows of FAINT_END_WINDOW rows) reach
#   FAINT_END_FRACTION of the FAINT_END_PERCENTILE percentile of the SNR of the track (the level of its bright part,
#   also when that is only a tenth of the track) are cut (see trimFaintEnds)
FAINT_END_PERCENTILE = 90
FAINT_END_FRACTION = 0.1
FAINT_END_WINDOW = 9
FAINT_END_BRIGHT_ROWS = 0.2

# Only an end at the level of the noise is cut: a row at or above this SNR is the object, however much brighter the
#   rest of the track is (e.g. an object which brightens by magnitudes, or one whose bright part is saturated)
FAINT_END_MAX_SNR = 5.0

# Very bright objects spill their charge along their row and column before the pixels reach the saturation
#   level: in a frame where the sum within SPILL_RADIUS (px) of the object is more than SPILL_RATIO times the sum
#   within the normal aperture, the measurement counts as saturated and the wider sum is its intensity. Only
#   checked when the normal sum is above SPILL_MIN_SNR times its noise
SPILL_RADIUS = 12.0
SPILL_RATIO = 1.3
SPILL_MIN_SNR = 100.0

# Columns and rows of the median brighter than their neighbours by more than BRIGHT_LINE_LEVEL (in the noise of the
#   sky in one frame, and 10 times the scatter of the columns or rows) are masked (see staticMask)
BRIGHT_LINE_LEVEL = 0.2

# The bright columns and rows (see lineMask) are not masked in a frame within TRAIL_KEEP px of the pixels which are
#   LINE_KEEP_LEVEL (in the noise of a frame) above the line: an object much brighter than such a line is kept on it.
#   An object moving along a column (or row) for most of a block lights the column with its own trail, which makes
#   the column bright in the median of the block, and its path would be masked
LINE_KEEP_LEVEL = 10.0

# A pixel of a bright column (or row) is a source above its line if it is LINE_KEEP_LEVEL above the running median
#   of LINE_LOCAL px along the line (in the frame): a line which flickers as a whole (or brighter near the star of
#   its trail) is not a source, a compact source on it is
LINE_LOCAL = 31

# A track which stays within SHADOW_RADIUS (px) of a track at least SHADOW_STRENGTH times stronger (the sum of
#   the significances of the hits over the square root of their number), for at least SHADOW_OVERLAP of its hits,
#   moving the same way, is not measured (see removeShadows)
SHADOW_RADIUS = 20.0
SHADOW_STRENGTH = 3.0
SHADOW_OVERLAP = 0.8

# A detection which follows a longer one at a constant offset along the motion of up to DUPLICATE_ALONG PSF
#   FWHMs (on the same line) is the same object measured twice (e.g. on the wing or the trail of a very bright
#   object) if the image is above CONNECTED_LEVEL (in the noise of a frame) all the way between them (see
#   connectedLight); two objects which move together have no light between them
DUPLICATE_ALONG = 10.0
CONNECTED_LEVEL = 5.0

# A detection whose centroids are within a PSF FWHM of those of a longer one in at least this fraction of its frames
#   is the same object (see removeDuplicateDetections)
DUPLICATE_COINCIDENT = 1/3

# Smallest F statistic of the curvature of a short track (see curvatureSignificant): about a 5 sigma improvement
#   of the fit by the quadratic. The tracks of faint objects have no measurable curvature over a short time, and
#   a quadratic would fit their noise
CURVATURE_MIN_F = 25.0

# Shape of the detections (see extentConcentration): the frames are stacked along the track (every EXTENT_STEP-th
#   frame) within EXTENT_RADIUS px, and the concentration of the light at the centre tells a point source from an
#   extended structure (e.g. a cloud). Only detections below EXTENT_MAX_SIGNIFICANCE are tested. On recorded data,
#   objects are at 0.87-1 (a few faint ones and very bright ones down to 0.6), structures of thin drifting clouds
#   at 0.2-0.7
EXTENT_RADIUS = 9
EXTENT_RING = (4.0, 7.0)
EXTENT_STEP = 2
MIN_CONCENTRATION = 0.8
EXTENT_MAX_SIGNIFICANCE = 200.0

# A detection along a thin linear structure (e.g. a trail, or light left along the path of a bright source which
#   crossed the field) is concentrated at the centre of the ring but not along the structure: the light of the
#   ring is also taken along axes through the centre in LINE_ANGLES directions (the pixels of the ring within
#   LINE_HALF_WIDTH px of the axis), see extentLine. Below mf_line_concentration along the brightest axis, with
#   that axis at least LINE_MIN_SIGNIFICANCE (sigma) above the background, the detection is rejected. On recorded
#   data, tracks on such light are at 0.08-0.38, objects at 0.5-1 (lower only for tracks which run beside a bright
#   object or pick up its trails)
LINE_ANGLES = 16
LINE_HALF_WIDTH = 1.5
LINE_MIN_SIGNIFICANCE = 5.0

# Below MIN_CONCENTRATION_ALONE a detection is extended in any case; between it and MIN_CONCENTRATION, only if at
#   least COMOVING_MIN other candidate tracks at the same time move with it (within COMOVING_TOLERANCE of its
#   speed), as the structures of a drifting cloud do
MIN_CONCENTRATION_ALONE = 0.6
COMOVING_MIN = 1
COMOVING_TOLERANCE = 0.15

# A track within SAME_PATH_FWHM PSF FWHMs (at least 3 px) of the path of another one is a piece of the same object
#   (see onPath), not another structure moving with it
SAME_PATH_FWHM = 2.0

# Smallest sigma of the smoothing of the frames in the search (px), see searchSigma
SEARCH_MIN_SIGMA = 1.0

# Hits stronger than LINK_STRONG times the threshold are linked by their positions only: such a hit can come
#   from a single bright frame of its run (e.g. a flash), whose velocity is not known (see linkHits)
LINK_STRONG = 3.0

# Two tracks are merged only if their union moves as one object where they meet (see _validMerge): every hit within
#   MERGE_CHECK_RUNS runs of the junction is within the tolerance of the merging of a smooth path through its
#   neighbours within MERGE_CHECK_RUNS runs, a parabola from MERGE_CHECK_PARABOLA neighbours on, else a line. The
#   tolerance grows by MERGE_CHECK_SPREAD of the distance the object moves from the nearest neighbour (the error of
#   the velocity of the path over the extrapolation)
MERGE_CHECK_RUNS = 4
MERGE_CHECK_PARABOLA = 5
MERGE_CHECK_SPREAD = 0.05

# The sums of the search along other velocities through a bright object give hits along its path, which link into
#   tracks moving along the path at other speeds (ghosts, see removeGhosts). A track is a ghost of a track at least
#   GHOST_STRENGTH times stronger if GHOST_ON_PATH of its hits are within GHOST_RADIUS bins (of the coarser search
#   bin of the two) of the path of the stronger track, within its time, and its velocity differs from the velocity of
#   the stronger track by more than GHOST_SPEED_DIFF of its speed (a track at the speed of the stronger one is the
#   same object or one following it, e.g. in a train of objects, see removeShadows)
GHOST_STRENGTH = 2.0
GHOST_ON_PATH = 0.8
GHOST_RADIUS = 2.0
GHOST_SPEED_DIFF = 0.2

# A fast object (moving at least a PSF width per frame) can be measured twice, e.g. at the middle of its streak and
#   at its leading end (see removeStreakDuplicates). A detection is a second measurement of a longer one if they
#   share STREAK_DUPLICATE_SHARED of its frames, in which it keeps its offset from the longer one (scatter below a
#   PSF width), the offset is within the streak (the motion during a frame) plus STREAK_DUPLICATE_FWHM PSF widths,
#   and both have the same light curve: a correlation of the logarithms of their intensities of at least
#   STREAK_DUPLICATE_CORRELATION, or both flat (the scatter of the logarithm below STREAK_DUPLICATE_FLAT) with the
#   light connected between them (a flat light curve doesn't tell two objects moving together apart)
STREAK_DUPLICATE_SHARED = 0.5
STREAK_DUPLICATE_FWHM = 3.0
STREAK_DUPLICATE_CORRELATION = 0.7
STREAK_DUPLICATE_FLAT = 0.05

# Chains of faint objects (see linkFaintHits): a chain starts from two hits FAINT_SEED_FRAMES apart (frames), each
#   within FAINT_SEED_RADIUS (px) plus FAINT_SEED_SPREAD (px per frame, the error of the velocity of a hit) times the
#   time between them of the position predicted from the other one with its velocity. It takes the hits within
#   FAINT_LINE_RADIUS of the line through its hits, up to FAINT_SEED_FRAMES[1] beyond its ends, and needs hits in
#   FAINT_MIN_RUNS runs over FAINT_MIN_SPAN frames. The velocities of the hits of a chain agree within
#   FAINT_VELOCITY_STEPS steps of the velocity grids of their searches
FAINT_SEED_FRAMES = (16, 160)
FAINT_SEED_RADIUS = 4.0
FAINT_SEED_SPREAD = 0.1
FAINT_LINE_RADIUS = 3.0
FAINT_VELOCITY_STEPS = 3.0
FAINT_MIN_RUNS = 4
FAINT_MIN_SPAN = 30

# A chain is measured if the PSF-weighted signal along its line is FAINT_MIN_SIGNIFICANCE (sigma) above the same
#   positions at other times (see faintSignificance), below mf_track_significance as the line through the hits is
#   less accurate than the measured track. A measurement of a chain combines enough frames for FAINT_SAMPLE_SNR
#   times the noise, up to FAINT_MAX_MEASURE_FRAMES
FAINT_MIN_SIGNIFICANCE = 8.0
FAINT_SAMPLE_SNR = 4.0
FAINT_MAX_MEASURE_FRAMES = 64

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
Hit = collections.namedtuple('Hit', ['frame', 'x', 'y', 'vx', 'vy', 'z', 'run', 'bin'], defaults=(2,))


def matchedFilterBlockFrames(config):
    """ The length of the blocks of frames of the matched filter. The runs of frames don't straddle the blocks, so
        a block holds a whole number of the longest runs and of the measurements: the block length is
        mf_block_frames, made a multiple of the least common multiple of all run lengths and of the largest number
        of frames of a measurement (a power of two) if it isn't.

    Arguments:
        config: [Config] Configuration object.

    Return:
        [int] Number of frames of a block.
    """

    block_frames = int(config.mf_block_frames)
    unit = int(np.lcm.reduce([int(v) for v in list(config.mf_run_frames) + [config.mf_slow_run_frames, 1]]))
    unit = int(np.lcm(unit, 2**int(math.floor(math.log2(max(config.mf_max_measure_frames, 1))))))
    if block_frames % unit:
        block_frames = max(unit, int(round(block_frames/float(unit)))*unit)

    return block_frames


def slowReferenceBlocks(k, n):
    """ The blocks with which the slow search compares the block k of n blocks: SLOW_REF_BLOCKS (and one more)
        before and after it, those which exist. The nearer blocks are left out, as their medians share frames with
        the block k (see backgroundPass) and the object has not moved away yet.

    Arguments:
        k: [int] Index of the block.
        n: [int] Number of blocks.

    Return:
        [list] Indices of the reference blocks.
    """

    return [j for j in (k - SLOW_REF_BLOCKS - 1, k - SLOW_REF_BLOCKS, k + SLOW_REF_BLOCKS, k + SLOW_REF_BLOCKS + 1)
            if 0 <= j < n]


def slowSearchMinBlocks():
    """ The fewest blocks in which the slow search can find an object: an object has to be seen in
        SLOW_MIN_BLOCKS consecutive blocks, each compared with at least two reference blocks (see
        slowReferenceBlocks).

    Return:
        [int] Number of blocks.
    """

    n = SLOW_MIN_BLOCKS
    while True:

        # The longest run of consecutive blocks with two reference blocks
        run = longest = 0
        for k in range(n):
            run = (run + 1) if (len(slowReferenceBlocks(k, n)) >= 2) else 0
            longest = max(longest, run)

        if longest >= SLOW_MIN_BLOCKS:
            return n

        n += 1


def lineKeep(frame, lines, level=LINE_KEEP_LEVEL, window=LINE_LOCAL):
    """ The pixels of the bright columns and rows (see MatchedFilterDetector.lineMask) near the sources which stand
        out from the line itself in a frame: LINE_KEEP_LEVEL above the running median of LINE_LOCAL px along the
        line, grown by TRAIL_KEEP px. A line which flickers as a whole, or which is brighter near the star of its
        trail, has no such pixels.

    Arguments:
        frame: [ndarray] Normalized frame (unit noise).
        lines: [ndarray] Mask of the bright columns and rows (whole columns and rows are True).

    Keyword arguments:
        level: [float] Level above the line (in the noise of a frame). LINE_KEEP_LEVEL by default.
        window: [int] Length of the running median along the line (px), longer than twice the longest source along
            it (e.g. the streak of a fast object in a frame), so the source doesn't set the level of the line.
            LINE_LOCAL by default.

    Return:
        [ndarray] Boolean mask of the pixels to keep.
    """

    keep = np.zeros(frame.shape, dtype=bool)

    def lineLevel(sub):
        """ The running median along the lines. Beyond the ends of a line it is padded with the median of the whole
            line, so a source which reaches the edge of the image doesn't set the level.

        Arguments:
            sub: [ndarray] The pixels of the lines, along the first axis.

        Return:
            [ndarray] The level of the lines at every pixel, of the shape of sub.
        """

        # The lines padded with their medians at both ends, and the running median along them
        pad = window//2
        base = np.median(sub, axis=0, keepdims=True)
        padded = np.concatenate([np.repeat(base, pad, axis=0), sub, np.repeat(base, pad, axis=0)], axis=0)

        return ndimage.median_filter(padded, size=(window, 1), mode='nearest')[pad:pad + len(sub)]

    # The columns and the rows of the lines, and in them the pixels above the running median along the line
    cols = np.flatnonzero(lines.all(axis=0))
    rows = np.flatnonzero(lines.all(axis=1))
    if cols.size:
        sub = frame[:, cols]
        keep[:, cols] = (sub - lineLevel(sub)) > level
    if rows.size:
        sub = frame[rows, :]
        keep[rows, :] |= ((sub - lineLevel(sub.T).T) > level)

    # Grown to cover the source and the wings of its PSF
    if keep.any():
        box = np.ones((2*TRAIL_KEEP + 1, 2*TRAIL_KEEP + 1), np.uint8)
        keep = cv2.dilate(keep.astype(np.uint8), box) > 0

    return keep


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
        self.line_concentration = config.mf_line_concentration
        self.track_deviation_fwhm = config.mf_track_deviation_fwhm
        self.track_deviation_length = config.mf_track_deviation_length
        self.edge_margin = max(config.detection_border, config.mf_edge_margin)
        self.gpu = config.mf_gpu
        self.smooth_frames = config.mf_smooth_frames
        self.max_tracks = config.mf_max_tracks
        self.link_max_gap = config.mf_link_max_gap

        # Faint objects: the threshold of the hits linked into chains over the whole input (0 for none), and the
        #   largest number of chains measured (see MatchedFilterDetector.faintTracks)
        self.faint_threshold = config.mf_faint_threshold
        self.faint_max_tracks = config.mf_faint_max_tracks

        # Tiers of the velocity search: the largest speed of a tier in binned px per frame, and the coarsest bin
        #   (see MatchedFilterDetector.searchTiers)
        self.tier_speed = config.mf_tier_speed
        self.streak_search = config.mf_streak_search
        self.max_bin = config.mf_max_bin

        # Saturation level of the raw frames (ADU), 98% of the range of the bit depth if not given
        self.saturation_level = config.mf_saturation_level if config.mf_saturation_level > 0 \
            else int(round(0.98*(2**config.bit_depth - 1)))
        self.threads = config.mf_threads

        # Speed limits in px per frame of the (binned) frames, from the angular velocity limits: the plate scale
        #   (deg/px) is the mean of the two axes of the field of view over the image size, times the binning, so
        #   the speed in px/frame is (deg/s)/(deg/px)/(frames/s)
        fps = fps if fps else config.fps
        height, width = size if size else (config.height, config.width)
        scale = det_bin*(config.fov_h/float(height) + config.fov_w/float(width))/2.0
        self.speed_min = config.mf_ang_vel_min/scale/fps
        self.speed_max = config.mf_ang_vel_max/scale/fps

        # The block length, corrected to a whole number of runs and measurements (see matchedFilterBlockFrames)
        fixed = matchedFilterBlockFrames(config)
        if fixed != self.block_frames:
            log.warning('Matched filter: mf_block_frames {:d} is not a multiple of the runs and the measurements, '
                        '{:d} used'.format(self.block_frames, fixed))
            self.block_frames = fixed

        # The GPU is used when requested (or automatically) and numba can reach it
        self.use_gpu = (self.gpu in ('on', 'auto')) and CUDA_AVAILABLE
        if (self.gpu == 'on') and not CUDA_AVAILABLE:
            log.warning('Matched filter: the GPU was requested, but CUDA is not available, using the CPU')



def velocityGrid(speed_max, run_frames, bin_factor, speed_min=0.0):
    """ Velocities to search, in px per frame of the binned image. The spacing makes the largest position
        error of an object in a run half a binned pixel.

    An object whose velocity differs by dv from a velocity of the grid drifts by dv*(N - 1)/2 at the first and the
    last frame of a run of N frames, relative to the stack of that velocity. With the grid step 1/(N - 1) px per
    frame, the nearest velocity of the grid is at most half a step away in each axis, so the drift is at most
    1/4 px at the ends of the run, and the object stays within the PSF in the whole stack.

    Arguments:
        speed_max: [float] Largest speed (px/frame, unbinned).
        run_frames: [int] Number of frames in a run.
        bin_factor: [int] Binning of the searched image.

    Keyword arguments:
        speed_min: [float] Smallest speed (px/frame, unbinned), for the faster tiers of the search (see
            MatchedFilterDetector.searchTiers). The grid starts one step below it, so the tiers overlap. 0 by
            default.

    Return:
        [ndarray] Velocities (vx, vy) of shape (n_vel, 2), binned px per frame.
    """

    # The step of the grid, and the largest speed in binned px per frame (one step more, so an object at the
    #   largest speed is still between grid points)
    step = 1.0/max(run_frames - 1, 1)
    v_max = speed_max/bin_factor + step

    # A square grid of velocities, of which the ones within the circle of the largest speed are kept
    grid = np.arange(-np.ceil(v_max/step), np.ceil(v_max/step) + 1)*step
    vx, vy = np.meshgrid(grid, grid)
    keep = vx**2 + vy**2 <= v_max**2

    # For a faster tier, only the ring of velocities from one step below its smallest speed (the largest speed of
    #   the previous tier), so no velocity between the tiers is left out
    if speed_min > 0:
        keep &= vx**2 + vy**2 >= max(speed_min/bin_factor - step, 0)**2
    vel = np.column_stack([vx[keep], vy[keep]])

    # Sorted by speed: for a slow object many velocities give nearly the same sum, and the first (slowest) of
    #   the equal ones is taken
    return vel[np.argsort(np.hypot(vel[:, 0], vel[:, 1]), kind='stable')]


def estimatePSFSigma(image, noise, n_stars=40, min_snr=20.0):
    """ Estimate the PSF sigma from bright isolated stars: a circular Gaussian is fitted to every star, after
        subtracting the sky of an annulus around it, and the median of their sigmas is taken. (A constant fitted
        with the Gaussian would take up the wings of the PSF, and the sigma would be too small.)

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

    # The stars are the local maxima of the smoothed signal-to-noise image (the largest value within r px)
    #   above min_snr, away from the image border
    snr = image/noise
    smooth = cv2.GaussianBlur(snr.astype(np.float32), (0, 0), 1.0)
    r = 6
    r_sky = 9
    local_max = (smooth == cv2.dilate(smooth, np.ones((2*r + 1, 2*r + 1), np.uint8))) & (smooth > min_snr)
    local_max[:2*r] = local_max[-2*r:] = False
    local_max[:, :2*r] = local_max[:, -2*r:] = False
    ys, xs = np.nonzero(local_max)

    # The brightest stars may be saturated, so the stars are taken in order of brightness after the first few
    order = np.argsort(smooth[ys, xs])[::-1]
    order = order[min(5, len(order)//4):]

    # Pixel coordinates of the patch of a star (relative to its brightest pixel), and the annulus beyond it whose
    #   median is the local sky
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    ys_sky, xs_sky = np.mgrid[-r_sky:r_sky + 1, -r_sky:r_sky + 1]
    annulus = (xs_sky**2 + ys_sky**2 > (r + 1)**2) & (xs_sky**2 + ys_sky**2 <= r_sky**2)

    # The model of a star is a circular Gaussian amp*exp(-((x - cx)^2 + (y - cy)^2)/(2 sigma^2)) evaluated at the
    #   pixel centres, and the residuals of the least squares fit are the model minus the sky-subtracted patch
    def residual(p, patch):
        amp, cx, cy, sigma = p
        return (amp*np.exp(-0.5*((xx - cx)**2 + (yy - cy)**2)/sigma**2) - patch).ravel()

    sigmas = []
    for i in order:
        y, x = ys[i], xs[i]

        # Isolated: no other star brighter than a tenth of this one within 2r
        near = smooth[y - 2*r:y + 2*r + 1, x - 2*r:x + 2*r + 1].copy()
        near[r:3*r + 1, r:3*r + 1] = -np.inf
        if near.max() > 0.1*smooth[y, x]:
            continue

        # Subtract the local sky (the median of the annulus) from the patch
        if not ((r_sky <= y < image.shape[0] - r_sky) and (r_sky <= x < image.shape[1] - r_sky)):
            continue
        sky = np.median(image[y - r_sky:y + r_sky + 1, x - r_sky:x + r_sky + 1][annulus])
        patch = image[y - r:y + r + 1, x - r:x + r + 1].astype(np.float64) - sky

        # Fit the Gaussian, starting from the peak of the patch at its centre with sigma = 1 px; the centre is
        #   bounded to 2 px from the brightest pixel and the sigma to 0.3-5 px
        p0 = [patch.max(), 0.0, 0.0, 1.0]
        try:
            fit = least_squares(residual, p0, args=(patch,), bounds=([0, -2, -2, 0.3], [np.inf, 2, 2, 5.0]))
        except Exception:
            continue
        if fit.success:
            sigmas.append(abs(fit.x[3]))
        if len(sigmas) >= n_stars:
            break

    # The median of the sigmas of the stars, which is robust to the stars spoiled by a neighbour or a hot pixel
    if len(sigmas) < 5:
        return None

    return float(np.median(sigmas))


def binFrames(frames, bin_factor):
    """ Bin frames by summing bin_factor x bin_factor pixels, scaled to keep unit noise. The sum of b^2 pixels of
        unit noise has the noise b, so it is divided by b, and a point source spread over the binned pixel keeps
        all its signal while the noise stays 1.
    """

    # Crop to a whole number of bins. The mean of every bin (OpenCV's area resampling by a whole factor, much
    #   faster than summing the pixels with numpy) times b^2 is the sum, and the sum divided by b is b times the mean
    #   (the same as the sum of the pixels up to the rounding of the last bits, as the pixels are added in another
    #   order)
    n, h, w = frames.shape
    h2, w2 = h//bin_factor, w//bin_factor

    # Without binning, the frames as float32
    if bin_factor == 1:
        return frames[:, :h2, :w2].astype(np.float32)

    # Every frame is resampled on its own (OpenCV resamples single images), cropped to the whole bins
    binned = np.empty((n, h2, w2), dtype=np.float32)
    for i in range(n):
        binned[i] = cv2.resize(np.ascontiguousarray(frames[i, :h2*bin_factor, :w2*bin_factor], dtype=np.float32),
                               (w2, h2), interpolation=cv2.INTER_AREA)

    # The mean of a bin times b is its sum divided by b
    binned *= bin_factor

    return binned


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
    def __init__(self, img_handle, config, mask=None, dark=None, flat_struct=None, frame_range=None):
        """ Matched-filter detection of moving objects in the frames of an image handle.

        Arguments:
            img_handle: [FrameInterface] Frame-based input (detection=True).
            config: [Config] Configuration object.

        Keyword arguments:
            mask: [MaskStruct] Mask, binned like the frames. None by default.
            dark: [ndarray] Dark frame, binned like the frames. None by default.
            flat_struct: [FlatStruct] Flat field, binned like the frames. None by default.
            frame_range: [tuple] (first frame, end frame) of the frames to search, e.g. a segment of the input (see
                RMS.Formats.FrameBuffer), numbered as in the whole input. None by default (all frames).
        """

        self.img_handle = img_handle
        self.config = config
        self.mask = mask
        self.dark = dark
        self.flat_struct = flat_struct

        # Size of the (binned) frames and their number
        self.height, self.width = img_handle.nrows, img_handle.ncols
        self.total_frames = img_handle.total_frames

        # The frames which are searched
        self.frame_first, self.frame_end = frame_range if (frame_range is not None) else (0, self.total_frames)

        # Detection binning of the frame handle, the results are scaled to the unbinned image. The options use
        #   the frame rate measured from the frame times of the input when it has one
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

        # The background (BlockBackground) and the mask of the static sources of every block, by block index
        self.backgrounds = {}
        self.static_masks = {}

        # The bright columns and rows of every block (see lineMask), masked in the search except near bright sources
        self.line_masks = {}

        # The normalized frames of the blocks kept from the search for the measurement and the verification, which
        #   then don't read and normalize them again (see normalizedBlock): block index -> (normalized frames,
        #   saturation masks), within mf_cache_memory GB. A block is dropped when the background it was normalized
        #   with changes (see maskObjects)
        self.cache_bytes_max = getattr(config, 'mf_cache_memory', 0.0)*1e9
        self.block_cache = {}

        # Factor of the intensities from the stars (see apertureCorrection), None if not applied
        self.aperture_correction = None

        # The sky quality from the stars (SkyQualityMap): the clouded regions are not searched. None searches
        #   everywhere
        self.sky_map = None

        # Frames of the detections and detections removed on clouded sky (see run), and the first frames of the
        #   removed detections
        self.cloud_rows_removed = 0
        self.cloud_removed_row_frames = []
        self.cloud_detections_removed = 0
        self.cloud_removed_frames = []

        # Number of the weakest candidate tracks which were not measured (see mf_max_tracks), and of the measured ones
        self.tracks_dropped = 0
        self.tracks_measured = 0

        # Faint objects (see faintTracks): searched for if the faint threshold is below the threshold. The hits
        #   between the two thresholds as rows [frame, x, y, z, vx, vy, run, bin] (collected in parts during the
        #   search), and the number of the chains measured
        self.faint = 0 < self.opts.faint_threshold < self.opts.threshold
        self.faint_parts = []
        self.faint_hits = np.zeros((0, 8), dtype=np.float32)
        self.faint_measured = 0

        # Photometry of every frame of the verified tracks: id(centroids) -> (frames, intensities, saturated
        #   pixel counts, noise of the intensities)
        self.photometry = {}

        # The PSF sigma from the config, or None to measure it on the stars (see search)
        self.psf_sigma = self.opts.psf_sigma if self.opts.psf_sigma > 0 else None

        # Processing time of every step (s), for the summary
        self.timing = collections.defaultdict(float)

        # Significances of the measured tracks: (first frame, last frame, significance, off time, reordered)
        self.significances = []

        # Number of the detections rejected as wandering from a smooth path (see removeDeviatingDetections)
        self.deviation_rejected = 0


        # The persistence counts of the segments of the input before this one, {(run, bin): (counts, runs)} (see
        #   removePersistent), and the counts to carry to the next segment: those before the frame where it begins
        #   (persistence_until), so the runs of the overlap are counted once
        self.persistence_prior = {}
        self.persistence_until = None
        self.persistence_carry = None

        # Number of frames which the persistence counts are of (all frames of the input without segments), and the
        #   ones of the counts carried from the segments before this one and to the next one
        self.persistence_frames = None
        self.persistence_prior_frames = 0
        self.persistence_carry_frames = 0

        # All the persistence counts at the end of the search, carried instead when the next segment is not searched
        #   (its overlap with this one is then counted only here)
        self.persistence_full = None
        self.persistence_full_frames = 0


    ### Frames ###

    def readFrames(self, first, n, step=1, saturation=False):
        """ Read n frames from the first one, every step-th, with the dark and the flat applied, as float32. With
            saturation, also the mask of the saturated pixels of the raw frames (uint8). """

        frames = np.empty((n, self.height, self.width), dtype=np.float32)
        saturated = np.zeros((n, self.height, self.width), dtype=np.uint8) if saturation else None
        for i in range(n):
            self.img_handle.setFrame(first + i*step)
            frame = self.img_handle.loadFrame()

            # Saturation is judged on the raw values, before the dark and the flat change them
            if saturation:
                saturated[i] = frame >= self.saturation_level

            # Calibrate the frame as the normal detection does
            if self.dark is not None:
                frame = Image.applyDark(frame, self.dark)
            if self.flat_struct is not None:
                frame = Image.applyFlat(frame, self.flat_struct)
            frames[i] = frame

        if saturation:
            return frames, saturated

        return frames


    def cacheFits(self, n_frames):
        """ Whether a block of normalized frames, with their saturation masks, fits into the memory left for the
            kept blocks (mf_cache_memory).

        Arguments:
            n_frames: [int] Number of frames of the block.

        Return:
            [bool] True if the block can be kept.
        """

        # The memory of the blocks kept so far, and of the new block: 4 bytes of a normalized pixel and 1 byte of
        #   its saturation mask
        used = sum(z.nbytes + sat.nbytes for z, sat in self.block_cache.values())

        return used + n_frames*self.height*self.width*5 <= self.cache_bytes_max


    def normalizedBlock(self, k):
        """ The frames of a block normalized to unit noise, and the masks of their saturated pixels: kept from
            the search, or read and normalized again.

        Arguments:
            k: [int] Index of the block.

        Return:
            (z, saturated): [tuple of ndarrays] float32 normalized frames and uint8 saturation masks.
        """

        # The block kept from the search
        if k in self.block_cache:
            return self.block_cache[k]

        # Otherwise the frames and their saturated pixels are read again, and normalized as in the search
        first, last = self.block_starts[k], self.block_ends[k]
        t1 = time()
        frames, saturated = self.readFrames(first, last - first, saturation=True)
        self.timing['read'] += time() - t1

        return self.normalize(frames, k), saturated


    def blockBackground(self, samples):
        """ Per-pixel median and noise of the sampled frames (ADU). The median and the median absolute deviation
            (MAD) are robust to the objects and the cosmic rays which are in a few of the samples; for Gaussian
            noise, sigma = 1.4826*MAD. The noise is at least 1 ADU, so a pixel without noise (e.g. at the border of
            the mask) doesn't divide by zero.
        """

        # In parallel over the threads of the matched filter (mf_threads, 0 for all CPU cores)
        median, mad = sampleMedianMAD(samples, threads=self.config.mf_threads)
        noise = np.maximum(1.4826*mad, 1.0).astype(np.float32)

        return BlockBackground(median.astype(np.float32), noise, len(samples))


    def backgroundPass(self, step=4):
        """ The background of every block, from every step-th frame of the block, smoothed over the neighbouring
            blocks. The median of a block alone has a noise of about 0.15 of the noise of a frame, which is static
            within the block: summed over many frames, it would make faint static patterns which change from
            block to block. Averaging over three blocks, and interpolating between the blocks in time (see
            normalize), makes it smaller and continuous.
        """

        # The median and the noise of every block from every step-th frame (a quarter of the frames is enough for
        #   a robust median, and is four times faster to read)
        raw = []
        for first, last in zip(self.block_starts, self.block_ends):
            t1 = time()
            n = (last - first + step - 1)//step
            samples = self.readFrames(first, n, step=step)
            self.timing['read'] += time() - t1

            t1 = time()
            raw.append(self.blockBackground(samples))
            self.timing['background'] += time() - t1

        # The motion of the stars, from the shifts between the medians of the blocks (see starDrift), SLOW_REF_BLOCKS
        #   blocks apart, or fewer in a short input
        for k, bg in enumerate(raw):
            bg.raw_median = bg.median
            self.backgrounds[k] = bg
        self._drift = self.starDrift(step=max(1, min(SLOW_REF_BLOCKS, len(raw) - 1)))
        drift = np.array(self._drift, dtype=np.float64)
        mids = [(first + last - 1)/2.0 for first, last in zip(self.block_starts, self.block_ends)]

        # The median of every block is the mean of the medians of the block and of its neighbours (only one
        #   neighbour at the ends), which reduces its noise by sqrt(3). The medians of the neighbours are shifted by
        #   the motion of the stars to the middle of the block first, so the stars of the mean are where they are
        #   in the block (also at the ends, where the only neighbour is on one side). The noise is the one of the
        #   block alone, as it changes less from block to block, and its estimate doesn't add to the background of
        #   the frames. The median of the block alone is kept for the search of objects which stay on the same
        #   pixels (see slowSearch)
        for k in range(len(raw)):
            near = []
            for j in range(max(k - 1, 0), min(k + 2, len(raw))):

                # The median of the block itself as it is, the ones of the neighbours shifted to its middle
                dx, dy = drift*(mids[k] - mids[j])
                if (j == k) or (abs(dx) + abs(dy) < 1e-3):
                    near.append(raw[j].median)
                else:
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    near.append(cv2.warpAffine(raw[j].median, M, (self.width, self.height), flags=cv2.INTER_CUBIC,
                                               borderMode=cv2.BORDER_REPLICATE))
            # The mean of the aligned medians, with the noise of the block alone
            median = np.mean(near, axis=0).astype(np.float32)
            noise = raw[k].noise
            n_samples = sum(raw[j].n_samples for j in range(max(k - 1, 0), min(k + 2, len(raw))))
            bg = BlockBackground(median, noise, n_samples)
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

        # The level of the stars is smoothed with a Gaussian of 1 px, so it measures a star rather than its
        #   brightest pixel
        level = cv2.GaussianBlur(stars/bg.sky_noise, (0, 0), 1.0)
        bg.star_level = level

        # Template of the stars, for the changes of their brightness with the transparency of the sky: the
        #   pixels of the stars above STAR_TEMPLATE_LEVEL, grown by a pixel to include their wings, where the
        #   median is above the sky
        star_px = cv2.dilate((level > STAR_TEMPLATE_LEVEL).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        star_px &= stars > 0

        # The bright stars are masked anyway and would dominate the fit of the scale; they may also be saturated
        #   and then don't follow the transparency
        bright = cv2.dilate((level > self.opts.star_threshold).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        star_px &= ~bright

        # The template is the image of the stars (median minus sky) at these pixels, and the flat indices of the
        #   pixels for the fit of its scale in every frame (see normalize)
        bg.star_idx = np.flatnonzero(star_px)
        bg.stars = np.where(star_px, stars, 0).astype(np.float32)


    @staticmethod
    def skyLevel(median):
        """ The local sky level of a median image: a median filter, on a 4x smaller image for speed. The median
            over 5x5 pixels of the 4x smaller image (20x20 px) ignores the stars, which cover only a few pixels,
            and follows the large-scale changes of the sky (e.g. moonlight, vignetting).
        """

        # Shrink by averaging, filter, and interpolate back to the full size
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

        # The star template at its pixels, if there are enough of them for a stable fit of its scale (see
        #   normalizeFrames)
        use_template = (bg.star_idx is not None) and (len(bg.star_idx) >= MIN_STAR_TEMPLATE_PX)
        template = bg.stars.ravel()[bg.star_idx] if use_template else None

        # The frames are normalized in parallel threads, in slices of frames (numpy releases the GIL in the
        #   operations on whole frames); every frame is normalized by the same operations as in one thread
        n_threads = max(min(numba.get_num_threads(), len(frames)//8), 1)
        bounds = np.linspace(0, len(frames), n_threads + 1).astype(int)
        self.inThreads(lambda i: self.normalizeFrames(frames, k, first, mid, bg, template, bounds[i],
                                                         bounds[i + 1]), list(range(n_threads)))

        return frames


    def normalizeFrames(self, z, k, first, mid, bg, template, i0, i1):
        """ Normalize the frames i0 .. i1 - 1 of the block k in place (see normalize).

        Arguments:
            z: [ndarray] All frames of the block (float32).
            k: [int] Index of the block.
            first: [int] First frame of the block.
            mid: [float] Middle frame of the block.
            bg: [BlockBackground] Background of the block.
            template: [ndarray] The star template at its pixels (bg.star_idx), None without enough of them.
            i0, i1: [int] The range of the frames (in the block).
        """

        # Subtract the background of every frame: the median of this block, plus the linear interpolation towards
        #   the median of the neighbouring block on the side of the frame. With the middle frames m_k and m_j of
        #   the two blocks, the background of the frame f is median_k + (f - m_k)/(m_j - m_k)*(median_j - median_k),
        #   so the background changes continuously from block to block. Beyond the middle of the first and the last
        #   block, where there is no block on the side of the frame, the line through the neighbour on the other
        #   side is extrapolated, so the background follows the motion of the stars to the ends of the input
        for i in range(i0, i1):
            f = first + i
            j = k - 1 if f < mid else k + 1
            if j not in self.backgrounds:
                j = k + 1 if f < mid else k - 1
            z[i] -= bg.median
            if j in self.backgrounds:
                mid_j = (self.block_starts[j] + self.block_ends[j] - 1)/2.0
                z[i] -= np.float32((f - mid)/(mid_j - mid))*(self.backgrounds[j].median - bg.median)

        # Change of the brightness of the stars in every frame, relative to the template: after subtracting the
        #   background, a frame whose stars are brighter or fainter by the factor (1 + s) has the residual s*T at
        #   the pixels of the template T. The least squares scale is s = sum(z*T)/sum(T^2) over the pixels of the
        #   template, and s*T is subtracted from the frame (only with enough template pixels for a stable fit)
        if template is not None:
            scale = z[i0:i1].reshape(i1 - i0, -1)[:, bg.star_idx].dot(template)/np.dot(template, template)
            for i in range(i0, i1):
                z[i] -= scale[i - i0]*bg.stars

        # Divide by the noise of every pixel, so every pixel of the normalized frames has unit noise
        z[i0:i1] /= bg.noise


    def blockMask(self, k, background):
        """ The mask of a block of frames: its static sources (see staticMask) and its clouded regions (in any
            chunk of frames overlapping the block, see RMS.Routines.SkyQuality), which are not searched or
            measured.

        Arguments:
            k: [int] Index of the block.
            background: [BlockBackground] Background of the block.

        Return:
            [ndarray] Boolean mask of the binned image, True for the masked pixels.
        """

        mask = self.staticMask(background)
        if self.sky_map is not None:
            mask |= self.sky_map.cloudMask(self.block_starts[k], self.block_ends[k] - 1, self.height, self.width,
                                           bin_factor=self.det_bin)

        return mask


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

        # Brightness of the static sources relative to the noise of the sky in one frame: the sources above
        #   star_threshold are masked, grown by a pixel to cover their wings. On the bright columns and rows (see
        #   lineMask) the brightness is taken above the running median along the line, so a line bright enough to
        #   be above the threshold is not masked along its length as a source (it is masked as a line in the
        #   search, except near the sources which stand out from it), while the compact sources on it are
        level = background.star_level
        lines = self.lineMask(background)
        if lines.any():

            # The brightness of the sources before the smoothing (see starTemplate), with the level of the lines
            #   subtracted there, then smoothed: the smoothing of a very bright line would otherwise spread it into
            #   the neighbouring columns (rows), above the threshold
            rel = ((background.median - background.sky)/background.sky_noise).astype(np.float32)
            cols = np.flatnonzero(lines.all(axis=0))
            rows = np.flatnonzero(lines.all(axis=1))
            if cols.size:
                rel[:, cols] -= ndimage.median_filter(rel[:, cols], size=(LINE_LOCAL, 1), mode='nearest')
            if rows.size:
                rel[rows, :] -= ndimage.median_filter(rel[rows, :], size=(1, LINE_LOCAL), mode='nearest')
            level = cv2.GaussianBlur(rel, (0, 0), 1.0)
        static = cv2.dilate((level > self.opts.star_threshold).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

        # The user mask (0 = masked) and the image border
        if self.mask is not None:
            static |= self.mask.img == 0

        m = self.opts.edge_margin
        if m > 0:
            static[:m] = static[-m:] = True
            static[:, :m] = static[:, -m:] = True

        return static


    def lineMask(self, background):
        """ Mask of the columns and rows brighter than their neighbours (the trails of very bright stars along their
            row and column, also of stars outside the image, and bad columns): they flicker as a whole with the
            scintillation, which the stacks along them add up. They are masked in the search, except near sources
            much brighter than their flicker (see searchBlock). True means masked.

        Arguments:
            background: [BlockBackground] Background of the block.

        Return:
            [ndarray] Boolean mask.
        """

        # The median over a column (or row) of the median image above the sky, in the noise of the sky, is its
        #   profile; its excess over the median of the 15 neighbouring columns is compared with the robust scatter
        #   of the excess of all columns, and the columns above max(BRIGHT_LINE_LEVEL, 10*scatter) are masked,
        #   with one column on each side
        lines = np.zeros(background.median.shape, dtype=bool)
        sky_rel = (background.median - background.sky)/background.sky_noise
        m = max(self.opts.edge_margin, 1)
        for axis in (0, 1):
            profile = np.median(sky_rel[m:-m, :] if axis == 0 else sky_rel[:, m:-m], axis=axis)
            excess = profile - ndimage.median_filter(profile, size=15, mode='nearest')
            scatter = 1.4826*np.median(np.abs(excess))
            bright = excess > max(BRIGHT_LINE_LEVEL, 10*scatter)
            bright = ndimage.binary_dilation(bright, iterations=1)
            if axis == 0:
                lines[:, bright] = True
            else:
                lines[bright, :] = True

        return lines


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

        # The pixels above the level which are not on static sources (a quick test first, as most frames have none)
        bright = frame > self.opts.trail_level
        if not bright.any():
            return
        bright &= ~static
        if not bright.any():
            return

        # The sources, brightest first. A component in the rows or columns of a brighter source is a part of its
        #   trail (the trail of a very bright source can itself be above the level), not a source. The peak of
        #   every source is the largest value of its pixels (computed from the bright pixels only: ndimage.maximum
        #   sorts the whole frame, which is slow for large frames)
        #   The components and their boxes are found by OpenCV, which labels them as ndimage.label does (4-connected,
        #   numbered in the order of their first pixel) at a third of the time
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bright.view(np.uint8), connectivity=4,
                                                               ltype=cv2.CV_32S)

        # The label 0 is the background, the sources are 1 .. n
        n -= 1

        # The peak of every source
        peaks = np.full(n, -np.inf)
        np.maximum.at(peaks, labels[bright] - 1, frame[bright])

        # The box of every source as the slices of its rows and columns, as ndimage.find_objects gives them
        boxes = [(slice(y, y + h), slice(x, x + w)) for x, y, w, h in stats[1:, :4].tolist()]
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

        # The source itself, with the wings of its PSF, is kept: a copy of the box around it, grown by TRAIL_KEEP px
        boxes = sources
        kept = []
        for sl in boxes:
            ys = slice(max(sl[0].start - TRAIL_KEEP, 0), sl[0].stop + TRAIL_KEEP)
            xs = slice(max(sl[1].start - TRAIL_KEEP, 0), sl[1].stop + TRAIL_KEEP)
            kept.append((ys, xs, frame[ys, xs].copy()))

        # Set the rows and the columns of every source (its box grown by TRAIL_WIDTH px) to the background (0 in a
        #   normalized frame) over the whole frame
        for sl in boxes:
            frame[max(sl[0].start - TRAIL_WIDTH, 0):sl[0].stop + TRAIL_WIDTH, :] = 0
            frame[:, max(sl[1].start - TRAIL_WIDTH, 0):sl[1].stop + TRAIL_WIDTH] = 0

        # Put the sources back
        for ys, xs, patch in kept:
            frame[ys, xs] = patch


    ### Search ###

    def lineWindow(self):
        """ The length of the running median along a bright line (see lineKeep): at least LINE_LOCAL, and longer
            than twice the streak of the fastest searched object in a frame (with its PSF), so such a streak along
            the line is kept.

        Return:
            [int] Length of the running median (px, odd).
        """

        # The PSF before it is measured is taken as 1 px
        psf = self.psf_sigma if self.psf_sigma is not None else 1.0

        return max(LINE_LOCAL, 2*int(math.ceil(self.opts.speed_max + 6*psf)) + 1)


    def prepareFrame(self, frame, static, lines=None):
        """ Prepare a normalized frame for the search, in place: the bright columns and rows masked except near the
            sources which stand out from them (see lineKeep), the trails of the bright sources removed (see
            removeTrails), and the static sources masked.

        Arguments:
            frame: [ndarray] Normalized frame, modified in place.
            static: [ndarray] Mask of the static sources (True = masked).

        Keyword arguments:
            lines: [ndarray] Mask of the bright columns and rows (see lineMask), None for none.
        """

        # The bright columns and rows first, except near the sources which stand out from them (e.g. an object
        #   moving along a column, which lights the column with its own trail): a line above the level of the
        #   trails would otherwise be taken for a source whose trails cover the whole frame
        if (lines is not None) and lines.any():
            frame[lines & ~lineKeep(frame, lines, window=self.lineWindow())] = 0

        # The trails of the bright sources (also of those kept on the lines)
        self.removeTrails(frame, static)

        # The masked pixels don't take part in the stacks of their neighbours either
        frame[static] = 0


    def searchBlock(self, z, static, block_first, stackers, lines=None, keep_frames=False):
        """ Velocity search of a block of normalized frames.

        Arguments:
            z: [ndarray] Frames normalized to unit noise, (n, height, width).
            static: [ndarray] Mask of static sources (True = masked).
            block_first: [int] Index of the first frame of the block.
            stackers: [list] (run_frames, bin, VelocityStacker) to run.

        Keyword arguments:
            lines: [ndarray] Mask of the bright columns and rows (see lineMask), None for none. Their pixels are
                masked in every frame except near the sources which stand out from them (see lineKeep).
            keep_frames: [bool] Leave the frames unchanged (every frame is prepared in a copy). False by default.

        Return:
            [list] Hits.
        """

        ### Frames of the search ###

        # Every frame is prepared for the search: the trails of bright sources removed, the masked pixels set to
        #   the background, single-pixel outliers clipped to +-clip (scintillating stars, cosmic rays; a faint object
        #   is never that bright in one frame, and the frames of a flash still count fully), binned 2x2, and
        #   smoothed with the PSF. Binning and smoothing make the frames a matched filter of the PSF: the sum of the
        #   pixels weighted by the PSF is the best estimate of the amplitude of a point source, and the stack of
        #   such frames along the right velocity is the best estimate of the amplitude of the moving object
        #   The frames are binned for every bin of the tiers of the search (see searchTiers), and smoothed with the
        #   PSF in binned pixels
        bins = sorted(set(stacker_bin for _, stacker_bin, _ in stackers))
        zbs = {b: np.empty((len(z), z.shape[1]//b, z.shape[2]//b), dtype=np.float32) for b in bins}

        def prepareFrames(i0, i1):
            """ Prepare the frames i0 .. i1 - 1 of the block, and bin and smooth them for every bin.

            Arguments:
                i0, i1: [int] The range of the frames (in the block).
            """

            for i in range(i0, i1):

                # The frame is prepared in place, or in a copy if the block is kept unchanged (see search)
                frame = z[i].copy() if keep_frames else z[i]
                self.prepareFrame(frame, static, lines)

                # The outliers are clipped once, and the clipped frame binned and smoothed for every bin. The
                #   smoothing is the PSF in binned pixels, which is smaller than a pixel for the coarse bins (the
                #   binning itself is then most of the matched filter of the PSF)
                clipped = np.clip(frame[None], -self.opts.clip, self.opts.clip)
                for b in bins:
                    binned = binFrames(clipped, b)[0]
                    zbs[b][i] = cv2.GaussianBlur(binned.astype(np.float32), (0, 0), self.searchSigma()/b)

        # Every frame is prepared on its own, so the frames are prepared in parallel threads, in slices of frames
        #   (OpenCV and numpy release the GIL), with the same results as in one thread. The threads of the matched
        #   filter (mf_threads) take a slice each, of at least 8 frames
        n_threads = max(min(numba.get_num_threads(), len(z)//8), 1)
        bounds = np.linspace(0, len(z), n_threads + 1).astype(int)
        self.inThreads(lambda j: prepareFrames(bounds[j], bounds[j + 1]), list(range(n_threads)))

        # The static mask binned the same way, grown by a binned pixel (a bin is masked if any of its pixels is)
        statics_b = {}
        for b in bins:
            static_b = binFrames(static[None].astype(np.float32), b)[0] > 0
            statics_b[b] = cv2.dilate(static_b.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

        ### ###

        # On the GPU, the frames of every bin are copied there once, padded for the largest shift of the stackers
        #   which use them, and the stacks of all runs of a stacker are computed by one launch (see
        #   VelocityStacker.stackMaxRuns)
        device_frames = {}
        if self.opts.use_gpu:
            try:
                for b in bins:

                    # The padding of the frames of a bin is the largest shift of its stackers
                    pad = max(stacker.pad for _, stacker_bin, stacker in stackers if stacker_bin == b)
                    device_frames[b] = DeviceFrames(zbs[b], pad)

            # E.g. out of GPU memory with several workers: continue on the CPU
            except Exception:
                log.error('Matched filter: the GPU failed, continuing on the CPU:\n' + traceback.format_exc())
                self.useCPU(stackers)
                device_frames = {}

        hits = []
        faint = []
        for run_frames, b, stacker in stackers:

            # The binned frames and the static mask of the bin of this stacker
            zb = zbs[b]
            static_b = statics_b[b]

            # Counts of the runs in which every pixel is above the threshold, for every run length and bin (see
            #   removePersistent)
            key = (run_frames, b)
            if key not in self.persistence_counts:
                self.persistence_counts[key] = np.zeros(static_b.shape, dtype=np.int32)
                self.persistence_runs[key] = 0

            # Consecutive runs of run_frames frames of the block: the largest stack over the velocities of every
            #   pixel, and its velocity, for every run
            run_starts = list(range(0, len(zb) - run_frames + 1, run_frames))

            def cpuRuns(stacker=stacker, zb=zb, run_frames=run_frames, run_starts=run_starts):
                """ The stacks of the runs on the CPU, one run after the other, so only the stack of the run
                    being searched is in memory.

                Return:
                    [generator] (first frame, stack_max, vel_idx) of every run.
                """

                # Every run is stacked when the search reaches it (the stacking time is counted as on the GPU)
                for r0 in run_starts:
                    t1 = time()
                    stack_max, vel_idx = stacker.stackMax(zb[r0:r0 + run_frames])
                    self.timing['stack'] += time() - t1
                    yield r0, stack_max, vel_idx

            # On the GPU, the stacks of all runs are computed by one launch, from the frames already on the GPU
            if stacker.use_gpu:
                t1 = time()
                try:
                    stack_maxes, vel_idxs = stacker.stackMaxRuns(device_frames.get(b, zb), run_starts)
                    runs = zip(run_starts, stack_maxes, vel_idxs)

                # E.g. out of GPU memory with several workers: continue on the CPU
                except Exception:
                    log.error('Matched filter: the GPU failed, continuing on the CPU:\n' + traceback.format_exc())
                    self.useCPU(stackers)
                    device_frames = {}
                    runs = cpuRuns()
                self.timing['stack'] += time() - t1

            # On the CPU, run by run
            else:
                runs = cpuRuns()

            # The hits of every run, from its stack
            for r0, stack_max, vel_idx in runs:

                # The maximum over many velocities of noise is not the noise of a single stack (it is the largest of
                #   many draws, so its distribution is shifted up and narrower). The detection image is therefore
                #   normalized by the distribution of the maximum itself over the unmasked image: its median and
                #   robust sigma (1.4826*MAD). Then the threshold is in the units of the actual noise of the maximum,
                #   for any number of velocities
                vals = stack_max[~static_b]
                med = np.median(vals)
                scale = 1.4826*np.median(np.abs(vals - med))
                zmax = (stack_max - med)/max(scale, 1e-6)
                zmax[static_b] = 0

                # Pixels above the threshold, counted for the persistence
                above = zmax > self.opts.threshold
                self.persistence_counts[key] += above
                self.persistence_runs[key] += 1

                # The hits are the local maxima (the largest value in their neighbourhood) above the threshold.
                #   A hit is at the middle time of the run, at the centre of its binned pixel in unbinned
                #   coordinates (the binned pixel i covers the unbinned pixels b*i to b*i + b - 1, whose centre is
                #   (i + 0.5)*b - 0.5), with the velocity of the best stack converted to unbinned px per frame
                #   The neighbourhood of the local maxima is +-4 unbinned px (5x5 pixels of 2x2 bins), at least the 3x3
                #   neighbours on coarser bins, so the peaks of nearby objects are not suppressed by a coarser tier
                half = max(1, int(round(4.0/b)))
                local_max = zmax == cv2.dilate(zmax, np.ones((2*half + 1, 2*half + 1), np.uint8))
                peaks = local_max & above
                for yb, xb in zip(*np.nonzero(peaks)):
                    vxb, vyb = stacker.velocities[vel_idx[yb, xb]]
                    hits.append(Hit(block_first + r0 + (run_frames - 1)/2.0, (xb + 0.5)*b - 0.5,
                                    (yb + 0.5)*b - 0.5, vxb*b, vyb*b, float(zmax[yb, xb]), run_frames, b))

                # The local maxima between the faint threshold and the threshold, for the chains of faint objects
                #   (see faintTracks), kept as compact rows: there are many of them
                if self.faint:
                    yb, xb = np.nonzero(local_max & (zmax > self.opts.faint_threshold) & ~above)
                    vel = stacker.velocities[vel_idx[yb, xb]]*b
                    faint.append(np.column_stack([np.full(len(yb), block_first + r0 + (run_frames - 1)/2.0),
                                                  (xb + 0.5)*b - 0.5, (yb + 0.5)*b - 0.5, zmax[yb, xb], vel,
                                                  np.full(len(yb), run_frames), np.full(len(yb), b)]
                                                 ).astype(np.float32))

        # The faint hits of the block, combined with those of all blocks at the end of the search
        self.faint_parts += faint

        return hits


    def useCPU(self, stackers):
        """ Continue the search on the CPU (after the GPU failed, e.g. out of memory with several workers).

        Arguments:
            stackers: [list] (run_frames, bin, VelocityStacker) of the search, all switched to the CPU.
        """

        # The following blocks are searched on the CPU too
        self.opts.use_gpu = False
        for _, _, stacker in stackers:
            stacker.use_gpu = False


    def searchTiers(self):
        """ The tiers of the velocity search: the speeds are searched on frames binned more the faster they are.
            The number of velocities of a tier grows with the square of its largest speed in binned px per frame,
            and the work with the number of binned pixels too, so the fast speeds would cost most of the search
            on finely binned frames. A fast object moves several pixels during a frame and is long in every frame
            anyway, so coarser bins lose little of its signal, and fast objects are searched less deep (a
            coarser bin adds the noise of more pixels to a point source: about 0.4 mag for 4x4 instead of 2x2
            bins with a PSF sigma of 0.75 px).

        A tier with the bin B covers the speeds up to tier_speed*B px per frame (unbinned), i.e. tier_speed binned
        px per frame, and the next tier doubles the bin, up to max_bin, which covers the speeds up to the
        largest one. With tier_speed 0, all speeds are searched on the bins of the search.

        Return:
            [list] Tiers (bin, smallest speed, largest speed), the speeds in unbinned px per frame.
        """

        # Without the velocity search only the static stacks are searched; without tiers all speeds are searched on
        #   the bins of the search
        b = self.search_bin
        if not self.opts.velocity_search:
            return [(b, 0.0, 0.0)]
        if self.opts.tier_speed <= 0:
            return [(b, 0.0, self.opts.speed_max)]

        # The tiers from the finest bin up: every tier covers the speeds from the largest speed of the previous one
        #   up to tier_speed binned px per frame of its bin. The last tier (the largest speed is within it, or the
        #   next bin would be coarser than max_bin) covers the speeds up to the largest one
        tiers = []
        speed_lo = 0.0
        tier_bin = b
        while True:
            speed_hi = self.opts.tier_speed*tier_bin
            if (speed_hi >= self.opts.speed_max) or (2*tier_bin > self.opts.max_bin):
                tiers.append((tier_bin, speed_lo, self.opts.speed_max))
                return tiers
            tiers.append((tier_bin, speed_lo, speed_hi))

            # The next tier starts where this one ends, on bins twice as large
            speed_lo = speed_hi
            tier_bin *= 2


    def search(self):
        """ Pass 1: background, static masks and the velocity search over the whole file.

        Return:
            [list] Hits.
        """

        n_block = self.opts.block_frames
        b = self.search_bin

        # The smoothing of the search (px), for the weights of the sums of the streak samples, whose noise depends
        #   on it (see velocitySamples). The PSF sigma is measured later; until then 1 px is used, as below
        search_sigma = max(self.psf_sigma if self.psf_sigma is not None else 1.0, SEARCH_MIN_SIGMA)

        # The stackers of the runs: the short runs (8 and 16 frames) search all velocities up to the largest
        #   speed, in tiers of binning (see searchTiers); the long run (32 frames) only the small velocities, at
        #   most 4 binned px over the run, where its longer integration gains the most (a slow object stays within
        #   the PSF for many frames)
        stackers = []
        for run_frames in self.opts.run_frames:
            for tier_bin, speed_lo, speed_hi in self.searchTiers():
                stackers.append((run_frames, tier_bin, VelocityStacker(
                    velocityGrid(speed_hi, run_frames, tier_bin, speed_min=speed_lo), run_frames,
                    use_gpu=self.opts.use_gpu, streak=self.opts.streak_search, smooth_sigma=search_sigma/tier_bin)))
        if self.opts.slow_run_frames > max(self.opts.run_frames):
            slow_speed = min(4.0/self.opts.slow_run_frames*b, self.opts.speed_max)
            stackers.append((self.opts.slow_run_frames, b, VelocityStacker(
                velocityGrid(slow_speed, self.opts.slow_run_frames, b), self.opts.slow_run_frames,
                use_gpu=self.opts.use_gpu)))
        for run_frames, tier_bin, st in stackers:
            log.info('Matched filter: {:d} velocities for runs of {:d} frames, {:d}x{:d} binned{:s}'.format(
                len(st.velocities), run_frames, tier_bin, tier_bin, ' (GPU)' if st.use_gpu else ''))

        # Blocks of the frames which are searched, the last one may be shorter
        starts = list(range(self.frame_first, self.frame_end, n_block))
        if (len(starts) > 1) and (self.frame_end - starts[-1] < n_block//2):
            starts = starts[:-1]
        self.block_starts = starts
        self.block_ends = starts[1:] + [self.frame_end]

        # The background follows changes of the sky (e.g. thin clouds) on the time scale of the blocks. An
        #   object slower than about a PSF width per 100 frames is too slow to be detected anyway
        self.backgroundPass()

        # Number of runs in which every pixel of the search image is above the threshold, continued from the
        #   segments before this one when the input is read in segments
        self.persistence_counts = {key: counts.copy() for key, (counts, _) in self.persistence_prior.items()}
        self.persistence_runs = {key: runs for key, (_, runs) in self.persistence_prior.items()}
        self.persistence_frames = self.persistence_prior_frames + (self.frame_end - self.frame_first)

        # The PSF sigma is measured on the stars of the first block in which enough isolated stars are visible (the
        #   first blocks can be clouded), unless it is given. Until then 1 px is used: the smoothing of the search
        #   is at least 1 px (see searchSigma), so the search is the same as with the measured sigma if the PSF
        #   is narrower
        psf_measured = self.psf_sigma is not None
        if not psf_measured:
            self.psf_sigma = 1.0

        # The search, block by block: read the frames, normalize them, and search them
        hits = []
        for k, (first, last) in enumerate(zip(self.block_starts, self.block_ends)):

            # The persistence counts to carry to the next segment, before its frames
            self.persistenceCheckpoint(first)

            # With the normalized blocks kept for the measurement, the saturated pixels are taken now too
            t1 = time()
            keep = self.cacheFits(last - first)
            if keep:
                frames, saturated = self.readFrames(first, last - first, saturation=True)
            else:
                frames = self.readFrames(first, last - first)
            self.timing['read'] += time() - t1

            # Measure the PSF sigma on the stars of the mean of the block, against the noise of the sky in the mean
            bg = self.backgrounds[k]
            if not psf_measured:
                sigma = estimatePSFSigma(frames.mean(axis=0) - bg.sky, bg.sky_noise/math.sqrt(len(frames)))
                if sigma is not None:
                    self.psf_sigma = sigma
                    psf_measured = True
                    log.info('Matched filter: PSF sigma {:.2f} px (block {:d})'.format(self.psf_sigma, k))

                    # The weights of the streak samples depend on the smoothing of the search, which now follows
                    #   the measured PSF
                    for _, tier_bin, st in stackers:
                        st.setSmoothing(self.searchSigma()/tier_bin)

            # The mean of the block (for the photometry of the stars), the static mask, and the frames normalized to
            #   unit noise (in place, the raw frames are not needed any more)
            t1 = time()
            bg.mean = frames.mean(axis=0)
            static = self.blockMask(k, bg)
            self.static_masks[k] = static
            self.line_masks[k] = self.lineMask(bg)
            z = self.normalize(frames, k)
            del frames
            self.timing['background'] += time() - t1
            if keep:
                self.block_cache[k] = (z, saturated)

            # A kept block is searched on copies of its frames (the search changes the frames it prepares)
            t1 = time()
            hits += self.searchBlock(z, static, first, stackers, lines=self.line_masks[k], keep_frames=keep)
            self.timing['search'] += time() - t1

        if not psf_measured:
            log.warning('Matched filter: the PSF sigma could not be measured on the stars, 1.0 px assumed (set '
                        'mf_psf_sigma)')

        # Without a next segment (or before its frames were reached), all counts are carried
        self.persistenceCheckpoint(None)

        return self.removePersistent(hits)


    def persistenceCheckpoint(self, first):
        """ Keep the persistence counts to carry to the next segment of the input (see persistence_until), once:
            when the search reaches the block which begins at or after the first frame of the next segment, or at
            the end of the search (first is None).

        Arguments:
            first: [int] First frame of the block which is searched next, None at the end of the search.
        """

        if first is None:
            self.persistence_full = {key: (counts.copy(), self.persistence_runs[key])
                                     for key, counts in self.persistence_counts.items()}
            self.persistence_full_frames = self.persistence_prior_frames + (self.frame_end - self.frame_first)

        if self.persistence_carry is not None:
            return

        if (first is None) or ((self.persistence_until is not None) and (first >= self.persistence_until)):
            self.persistence_carry = {key: (counts.copy(), self.persistence_runs[key])
                                      for key, counts in self.persistence_counts.items()}
            last = self.frame_end if (first is None) else first
            self.persistence_carry_frames = self.persistence_prior_frames + (last - self.frame_first)


    def removePersistent(self, hits):
        """ Remove the hits at persistent pixels: above the threshold in a large fraction of all runs of the
            input (e.g. flickering pixels, variable stars). It is judged over the whole input (over the segments
            so far when it is read in segments), as an object which moves little stays at a position for a large
            part of a block: at the smallest motions of the search it takes a few hundred frames to cross the PSF.
            The faint hits collected by the search (faint_parts) are kept in faint_hits without those at persistent
            pixels.

        Arguments:
            hits: [list] Hits of the search.

        Return:
            [list] Hits at pixels which are not persistent.
        """

        # The persistent pixels of every run length and bin: above the threshold in more than the fraction
        #   persistence of all runs (and in at least 2 runs), grown by a binned pixel. When the counts are of a part of
        #   the input (its segments so far), the fraction is of the runs of the whole input: the counts only grow,
        #   so these pixels are persistent over the whole input too (the others are judged at its end, see
        #   removePersistentDetections)
        scale = max(1.0, self.total_frames/float(self.persistence_frames or self.total_frames))
        persistent = persistentMasks({key: (counts, self.persistence_runs[key]*scale)
                                      for key, counts in self.persistence_counts.items()}, self.opts.persistence)

        # Keep the hits whose binned pixel (the inverse of the conversion to unbinned coordinates in searchBlock)
        #   is not persistent in the runs of their length and bin
        kept = []
        for h in hits:
            b = h.bin
            mask = persistent[(h.run, b)]
            xb = int(round((h.x + 0.5)/b - 0.5))
            yb = int(round((h.y + 0.5)/b - 0.5))
            if not mask[yb, xb]:
                kept.append(h)

        # The faint hits of all blocks (see searchBlock), the same way
        if self.faint_parts:
            faint = np.vstack(self.faint_parts)
            self.faint_parts = []
            keep = np.ones(len(faint), dtype=bool)
            for (run_frames, b), mask in persistent.items():

                # The hits of the runs of this length and bin at the persistent pixels of that search are dropped
                sel = np.nonzero((faint[:, 6] == run_frames) & (faint[:, 7] == b))[0]
                xb = np.round((faint[sel, 1] + 0.5)/b - 0.5).astype(int)
                yb = np.round((faint[sel, 2] + 0.5)/b - 0.5).astype(int)
                keep[sel[mask[yb, xb]]] = False
            self.faint_hits = faint[keep]

        return kept


    ### Search of very slow objects ###

    def starDrift(self, step=SLOW_REF_BLOCKS):
        """ Motion of the stars across the image (px per frame), e.g. the diurnal motion for a fixed camera,
            from the shift between the medians of blocks step apart. Zero if there are too few blocks.

        Over a few blocks (tens of seconds) the stars of a narrow field move together, so the motion is one
        translation. Phase correlation finds the shift between two images from the phase of their cross-power
        spectrum, and is dominated by the many stars; a moving object or two don't change it.
        """

        n = len(self.block_starts)
        if n <= step:
            return 0.0, 0.0

        # The Hanning window tapers the images to zero at their edges, so the edges don't correlate
        window = cv2.createHanningWindow((self.width, self.height), cv2.CV_32F)

        # The shift between the stars (median minus sky, negative values clipped) of the blocks k and k + step, for
        #   up to 8 pairs of blocks over the input, divided by the time between their middles. A weak peak of the
        #   correlation (response) means no reliable shift, e.g. too few stars
        shifts = []
        for k in range(0, n - step, max(1, (n - step)//8)):
            a, b = self.backgrounds[k], self.backgrounds[k + step]
            img_a = np.clip(a.raw_median - self.skyLevel(a.raw_median), 0, None).astype(np.float32)
            img_b = np.clip(b.raw_median - self.skyLevel(b.raw_median), 0, None).astype(np.float32)
            (dx, dy), response = cv2.phaseCorrelate(img_a, img_b, window)

            # The medians are of the middles of the blocks (the last block can be longer or shorter)
            dt = ((self.block_starts[k + step] + self.block_ends[k + step]) - (self.block_starts[k]
                                                                             + self.block_ends[k]))/2.0
            if response > 0.05:
                shifts.append((dx/dt, dy/dt))

        if not shifts:
            return 0.0, 0.0

        # The median of the pairs, robust to a pair spoiled by e.g. a bright object
        return tuple(np.median(np.array(shifts), axis=0))


    def drift(self):
        """ The motion of the stars (px per frame), measured once. """

        if getattr(self, '_drift', None) is None:
            self._drift = self.starDrift(step=max(1, min(SLOW_REF_BLOCKS, len(self.block_starts) - 1)))

        return self._drift


    def referenceBackground(self, k):
        """ Background of the block k without the objects which are part of its median: the median of the medians
            of the blocks SLOW_REF_BLOCKS (and one more) before and after it, shifted by the motion of the stars,
            and the same of their noise. A slow object is elsewhere in those blocks, while the stars are aligned.

        Return:
            (median, noise): [tuple of ndarrays] or (None, None) if there are fewer than two such blocks.
        """

        # The reference blocks: SLOW_REF_BLOCKS and one more block before and after (those which exist). The nearer
        #   blocks are left out, as their medians share frames with this block (see backgroundPass) and the object
        #   has not moved away yet
        refs = [j for j in slowReferenceBlocks(k, len(self.block_starts)) if self.backgrounds[j].raw_median is not None]
        if len(refs) < 2:
            return None, None

        # Shift the median and the noise of every reference block by the motion of the stars between its middle
        #   and the middle of this block (an affine transform with only a translation, cubic interpolation), so
        #   its stars are where they are in this block
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

        # The median over the reference blocks: an object which moved relative to the stars is in at most one of
        #   them at any pixel, so the median doesn't contain it
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

        # The comparison needs reference blocks for consecutive blocks
        n = len(self.block_starts)
        if (n < slowSearchMinBlocks()) or (self.backgrounds[0].raw_median is None):
            return []

        # Candidates closer to the border than the edge margin plus the PSF are not used
        sigma = self.psf_sigma
        m = self.opts.edge_margin + int(math.ceil(3*sigma))
        drift = np.array(self.drift())


        ### Candidates in every block ###

        # (middle frame of the block, x, y, significance, block index)
        cands = []
        for k in range(n):
            bg = self.backgrounds[k]
            mid = (self.block_starts[k] + self.block_ends[k] - 1)/2.0
            ref, _ = self.referenceBackground(k)
            if ref is None:
                continue

            # The difference of the median of the block and the reference (the same sky and stars without the
            #   object), smoothed with the PSF, in units of its robust noise over the whole image. An object which
            #   is part of the median of this block is a positive peak of the difference
            diff = cv2.GaussianBlur((bg.raw_median - ref).astype(np.float32), (0, 0), sigma)
            noise = 1.4826*np.median(np.abs(diff - np.median(diff))) + 1e-6
            z = diff/noise

            # The aligned stars don't cancel perfectly (interpolation, scintillation), so the residual of a bright
            #   star can be above the threshold. Such a residual is small compared with the star itself: a peak
            #   has to be at least half of the (smoothed) star in the reference at its position
            star = cv2.GaussianBlur((ref - bg.sky).astype(np.float32), (0, 0), sigma)/noise

            # The candidates are the local maxima above SLOW_THRESHOLD, outside the mask and the border
            peaks = (z == cv2.dilate(z, np.ones((5, 5), np.uint8))) & (z > SLOW_THRESHOLD) & (z > 0.5*star)
            if self.mask is not None:
                peaks &= self.mask.img > 0
            peaks[:m] = peaks[-m:] = False
            peaks[:, :m] = peaks[:, -m:] = False
            for y, x in zip(*np.nonzero(peaks)):
                cands.append((mid, float(x), float(y), float(z[y, x]), k))

        if not cands:
            return []

        ### ###


        ### Linking of the candidates of consecutive blocks ###

        # A candidate is linked to the nearest candidate in the next (or previous) block. The first link can reach
        #   SLOW_MAX_SPEED px per frame over the block (plus the PSF), later links are predicted from the line
        #   fitted through the chain and have to be within 2 PSF sigmas plus a pixel of the prediction. The
        #   chains start from the strongest candidates
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

            # Extend the chain forward in time, then backward
            for direction in (1, -1):
                while True:

                    # The end of the chain in this direction, and the block to look in
                    cur = chain[-1] if direction == 1 else chain[0]
                    k_next = cands[cur, 4] + direction

                    # The motion of the chain (a line through its positions vs time), or none yet
                    if len(chain) >= 2:
                        pts = cands[sorted(chain, key=lambda j: cands[j, 0])]
                        vx = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
                        vy = np.polyfit(pts[:, 0], pts[:, 2], 1)[0]
                        tol = 2*sigma + 1
                    else:
                        vx = vy = 0.0
                        tol = reach

                    # The nearest unused candidate of the next block to the predicted position
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

            # An object has to be seen in at least SLOW_MIN_BLOCKS consecutive blocks
            if len(chain) < SLOW_MIN_BLOCKS:
                continue

            # The motion of the whole chain
            pts = cands[sorted(chain, key=lambda j: cands[j, 0])]
            vx = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
            vy = np.polyfit(pts[:, 0], pts[:, 2], 1)[0]

            # Moving with the stars: a residual of a star
            if math.hypot(vx - drift[0], vy - drift[1]) < SLOW_MIN_RELATIVE_SPEED:
                continue

            # The chain as a track of hits (rows [frame, x, y, z, vx, vy, run, bin]), with half a block as the
            #   length of its run, so the measurement extends half a block beyond its first and last hit (see
            #   TrackMeasurement)
            run = self.opts.block_frames//2
            tracks.append(np.column_stack([pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], np.full(len(pts), vx),
                                           np.full(len(pts), vy), np.full(len(pts), run),
                                           np.full(len(pts), self.search_bin)]))

        ### ###

        return tracks


    ### Linking ###

    def linkHits(self, hits):
        """ Link the hits of consecutive runs into tracks, predicting the position of the next hit from the
            motion of the track so far.

        Arguments:
            hits: [list] Hits of all runs.

        Return:
            [list] Tracks, each an array of hits as rows [frame, x, y, z, vx, vy, run, bin].
        """

        tracks = []

        # The hits of every run length and bin are linked separately (their runs are consecutive in time); the
        #   tracks of the different run lengths and bins are merged at the end
        for run_frames, b in sorted(set((h.run, h.bin) for h in hits)):

            group = [h for h in hits if (h.run == run_frames) and (h.bin == b)]
            if not group:
                continue

            # The hits as rows [frame, x, y, z, vx, vy, run, bin], and the hits of every run by its middle frame,
            #   for a quick lookup of the hits of the next run
            arr = np.array([[h.frame, h.x, h.y, h.z, h.vx, h.vy, h.run, h.bin] for h in group])
            by_frame = collections.defaultdict(list)

            # The middle frames of the hits as Python floats, which are rounded much quicker than numpy numbers
            frames = arr[:, 0].tolist()
            for i, f in enumerate(frames):
                by_frame[round(f, 1)].append(i)

            # A spatial index of the hits of every run, so only the hits near a predicted position are tested
            #   (large frames have many hits per run)
            by_frame = {t: np.array(idx) for t, idx in by_frame.items()}
            trees = {t: cKDTree(arr[idx, 1:3]) for t, idx in by_frame.items()}

            def nearHits(t, x, y, radius):
                """ Indices of the hits of the run at the middle frame t within radius of (x, y), in increasing
                    order (the order in which they would be tested one by one). """

                if t not in trees:
                    return []
                return sorted(by_frame[t][trees[t].query_ball_point((x, y), radius + 1e-6)])

            # Velocity grid step (px/frame, unbinned), the uncertainty of the velocity of a hit, which sets the
            #   tolerances of the linking
            step = b/max(run_frames - 1.0, 1.0)

            # The first look-ups of a chain of a single hit (with the motion of the hit, in the runs 1 to 3 runs
            #   before and after it) depend only on the hit, so they are made for all hits at once, one query per
            #   run, with the same positions and radii as one by one (most hits start a chain and find nothing)
            first_near = {}
            for direction in (1, -1):
                for gap in (1, 2, 3):

                    # The hits by the run gap runs before or after them
                    dt_run = direction*gap*run_frames
                    targets = collections.defaultdict(list)
                    for i, f in enumerate(frames):
                        targets[round(f + dt_run, 1)].append(i)

                    for t, idx in targets.items():

                        # A run without hits has nothing near any position
                        if t not in trees:
                            first_near.update(((i, direction, gap), []) for i in idx)
                            continue

                        # The positions predicted with the motion of every hit, looked up at once in the spatial
                        #   index of the run, within the radius of the look-up of a single hit
                        idx = np.array(idx)
                        points = np.column_stack([arr[idx, 1] + arr[idx, 4]*dt_run, arr[idx, 2] + arr[idx, 5]*dt_run])
                        found = trees[t].query_ball_point(points, 1.5*b + 1.5*step*abs(dt_run) + 1e-6)

                        # The hits found, in increasing order, as nearHits gives them
                        for i, near in zip(idx.tolist(), found):
                            first_near[(i, direction, gap)] = sorted(by_frame[t][near])

            # Hits this strong are judged by their positions only (see below)
            strong = LINK_STRONG*self.opts.threshold
            used = np.zeros(len(arr), dtype=bool)

            # Chains start from the strongest hits, and grow forward and backward in time
            for i in np.argsort(-arr[:, 3]):

                if used[i]:
                    continue

                chain = [i]
                used[i] = True

                for direction in (1, -1):
                    while True:

                        # Motion of the chain: a line fit of its hits, or the velocity of the hit. A strong hit
                        #   can come from a single bright frame of the run (a flash), whose velocity is not known,
                        #   but its position is: two strong hits give the motion
                        cur = chain[-1] if direction == 1 else chain[0]
                        strong_chain = all(arr[j, 3] >= strong for j in chain)
                        if (len(chain) >= 3) or ((len(chain) == 2) and strong_chain):
                            pts = arr[sorted(chain, key=lambda j: arr[j, 0])]
                            vx = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
                            vy = np.polyfit(pts[:, 0], pts[:, 2], 1)[0]
                        else:
                            vx, vy = arr[cur, 4], arr[cur, 5]

                        # Look for the next hit in the next run, or after a gap of up to 3 runs (the object can be
                        #   below the threshold in a run), or between strong hits up to link_max_gap frames (e.g.
                        #   between the flashes of a flashing object, which are the only frames above the
                        #   threshold). The first run with a matching hit is taken, and in it the strongest hit
                        best, best_z = None, 0
                        max_gap = max(3, int(self.opts.link_max_gap//run_frames))
                        for gap in range(1, max_gap + 1):

                            # Gaps longer than 3 runs are only allowed from a strong hit (see below)
                            if (gap > 3) and (arr[cur, 3] < strong):
                                break

                            t = round(frames[cur] + direction*gap*run_frames, 1)

                            # The hits which can pass the tests below: within the reach of the motion of the chain
                            #   around the predicted position, and for a single strong hit, also within the reach of
                            #   the largest speed around its position. For a chain of a single hit, the first look-ups
                            #   were made above for all hits at once
                            dt_run = direction*gap*run_frames
                            if (len(chain) == 1) and (gap <= 3):
                                near = first_near[(cur, direction, gap)]
                            else:
                                near = nearHits(t, arr[cur, 1] + vx*dt_run, arr[cur, 2] + vy*dt_run,
                                                1.5*b + 1.5*step*abs(dt_run))
                            if (len(chain) == 1) and strong_chain:
                                near = sorted(set(near) | set(nearHits(t, arr[cur, 1], arr[cur, 2],
                                    self.opts.speed_max*abs(dt_run) + 1.5*b)))

                            for j in near:
                                if used[j]:
                                    continue

                                # Longer gaps only between strong hits (flashes), so a track doesn't continue
                                #   into the noise beyond its ends
                                if (gap > 3) and ((arr[j, 3] < strong) or (arr[cur, 3] < strong)):
                                    continue
                                # Distance of the hit from the position predicted with the motion of the chain,
                                #   and the difference of its velocity from the motion of the chain
                                dt = arr[j, 0] - arr[cur, 0]
                                dist = math.hypot(arr[j, 1] - (arr[cur, 1] + vx*dt),
                                                  arr[j, 2] - (arr[cur, 2] + vy*dt))
                                dv = math.hypot(arr[j, 4] - vx, arr[j, 5] - vy)

                                # The velocity of a hit is known to about a PSF width over the run, except for
                                #   strong hits (see above). The predicted position is uncertain by the position of
                                #   a hit (1.5 binned px) plus the velocity error times the time (1.5 grid steps).
                                #   A single strong hit can be followed by another strong hit anywhere within the
                                #   reach of the largest speed
                                if (len(chain) == 1) and strong_chain and (arr[j, 3] >= strong):
                                    ok = math.hypot(arr[j, 1] - arr[cur, 1], arr[j, 2] - arr[cur, 2]) \
                                        <= self.opts.speed_max*abs(dt) + 1.5*b
                                else:
                                    ok = (dist <= 1.5*b + 1.5*step*abs(dt)) and ((dv <= 3*step) or
                                                                                 (arr[j, 3] >= strong))
                                if ok and (arr[j, 3] > best_z):
                                    best, best_z = j, arr[j, 3]
                            if best is not None:
                                break

                        # No more hits in this direction
                        if best is None:
                            break

                        used[best] = True
                        if direction == 1:
                            chain.append(best)
                        else:
                            chain.insert(0, best)

                # A track needs at least min_hits hits
                if len(chain) >= self.opts.min_hits:
                    tracks.append(arr[chain])

        # Merge the tracks of the same object (from runs of different lengths, or pieces of one track)
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

        # Two tracks are the same object if their positions are within 2 binned pixels (the position uncertainty
        #   of the hits), of the coarser bin of the two (see _mergeOnce)
        tol = 2.0

        # A merged track can now overlap or continue a track kept before, so the merging is repeated
        while True:
            merged = self._mergeOnce(tracks, tol)
            if len(merged) == len(tracks):
                break
            tracks = merged

        return sorted(merged, key=lambda t: t[0, 0])


    def _mergeOnce(self, tracks, tol_bins):
        """ One pass of mergeTracks. Every track is compared with the tracks kept so far (longest first) and
            merged into the first one it matches, or kept as a new track.

        The pairs which can't match are skipped without the comparison: the tracks too far apart in time, and the
        tracks overlapping in time whose bounding boxes are farther apart than the tolerance (their interpolated
        positions are inside their boxes). The result is the same as with the comparison of all pairs.
        """

        tracks = sorted(tracks, key=lambda t: -len(t))
        max_gap = 4*self.opts.slow_run_frames

        # The first and the last frame, the bounding box, the bin and the end velocity (computed when needed) of
        #   the tracks kept so far
        merged, info = [], []
        for tr in tracks:

            tr_info = self._trackInfo(tr)
            t0, t1 = tr_info[0], tr_info[1]

            # The tracks kept so far which overlap in time or are within the largest gap, in the order of merged
            if merged:
                m_t0 = np.array([inf[0] for inf in info])
                m_t1 = np.array([inf[1] for inf in info])
                candidates = np.nonzero((m_t0 - t1 <= max_gap) & (t0 - m_t1 <= max_gap))[0]
            else:
                candidates = []

            joined = False
            for k in candidates:

                m, m_info = merged[k], info[k]

                # The tolerance in binned pixels of the coarser of the two tracks (the bin is the last column of the
                #   hits, the bin of the search for tracks without it)
                tol = tol_bins*max(tr_info[6], m_info[6])

                # Overlap in time: compare the positions at the frames of both tracks within the overlap,
                #   interpolated linearly between the hits; the same object if the median distance is within tol.
                #   The distance is at least the distance between the bounding boxes of the tracks
                lo, hi = max(t0, m_info[0]), min(t1, m_info[1])
                if hi >= lo:
                    box_dx = max(0.0, m_info[2] - tr_info[3], tr_info[2] - m_info[3])
                    box_dy = max(0.0, m_info[4] - tr_info[5], tr_info[4] - m_info[5])
                    if math.hypot(box_dx, box_dy) <= tol:
                        fr = np.concatenate([tr[:, 0], m[:, 0]])
                        fr = fr[(fr >= lo) & (fr <= hi)]
                        if len(fr):
                            dx = np.interp(fr, m[:, 0], m[:, 1]) - np.interp(fr, tr[:, 0], tr[:, 1])
                            dy = np.interp(fr, m[:, 0], m[:, 2]) - np.interp(fr, tr[:, 0], tr[:, 2])
                            if (np.median(np.hypot(dx, dy)) <= tol) and self._validMerge(m, tr, tol):
                                merged[k] = self._union(m, tr)
                                info[k] = self._trackInfo(merged[k])
                                joined = True
                                break

                # One continues the other: extrapolate the end of the earlier one over the gap. The gap can be
                #   at most 4 runs of the slow search, so tracks lost for a while (e.g. behind a star) are joined
                if m_info[1] < t0:
                    first, second, first_info = m, tr, m_info
                else:
                    first, second, first_info = tr, m, tr_info
                gap = second[0, 0] - first[-1, 0]
                if 0 < gap <= max_gap:

                    # The velocity at the end of the earlier track (see _endVelocity), computed once per track
                    if first_info[7] is None:
                        first_info[7] = self._endVelocity(first)
                    vx, vy = first_info[7]

                    # The second track has to begin at the extrapolated position. The error of the extrapolation
                    #   grows with the distance travelled over the gap (5% of it, a velocity error of 5%)
                    pred = first[-1, 1:3] + np.array([vx, vy])*gap
                    if (np.hypot(*(second[0, 1:3] - pred)) <= tol + 0.05*gap*math.hypot(vx, vy)) \
                            and self._validMerge(m, tr, tol):
                        merged[k] = self._union(m, tr)
                        info[k] = self._trackInfo(merged[k])
                        joined = True
                        break

            if not joined:
                merged.append(tr)
                info.append(tr_info)

        return merged


    def _validMerge(self, a, b, tol):
        """ Whether the union of two tracks moves as one object where they meet. The tests of _mergeOnce compare
            the two tracks only with each other: a track merged with two hits of the noise which happen to continue
            it, or with a piece of another object which crosses it, meets them, and the merged track is then
            compared with the next tracks, so tracks of several objects and of the noise can grow into one track
            across the image (and a real object whose hits go into it is lost). Here every hit of the union near
            the junction (the overlap in time of the tracks, or the gap between them, and MERGE_CHECK_RUNS runs
            around it) has to be within the tolerance of a smooth path through its neighbours, fitted without it:
            a line, or a parabola with enough neighbours (curved tracks, e.g. by the distortion of the lens).

        Arguments:
            a: [ndarray] Hits of a track as rows [frame, x, y, z, vx, vy, run, ...], sorted by time.
            b: [ndarray] Hits of the other track, likewise.
            tol: [float] Tolerance of the merging (px).

        Return:
            [bool] True if the union is one object.
        """

        u = self._union(a, b)
        f = u[:, 0]

        # The junction: the overlap in time of the tracks, or the gap between them
        lo, hi = max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0])
        if hi < lo:
            lo, hi = hi, lo

        # The neighbours of a hit are within MERGE_CHECK_RUNS runs of the longest run of the hits (the run of a hit
        #   is its 7th column, the slow search has no runs)
        run = float(np.max(u[:, 6])) if u.shape[1] > 6 else float(self.opts.slow_run_frames)
        window = MERGE_CHECK_RUNS*run

        for i in np.nonzero((f >= lo - window) & (f <= hi + window))[0]:

            # The neighbours, without the hit and without the hits in its frame (of another tier of the search,
            #   whose positions are not independent of it)
            nb = np.nonzero((np.abs(f - f[i]) <= window) & (f != f[i]))[0]

            # A path needs neighbours at two times at least (the neighbours can share times too), a parabola at
            #   three
            n_times = len(np.unique(f[nb]))
            if n_times < 2:
                continue

            # A smooth path through the neighbours, at the time of the hit
            deg = 2 if (len(nb) >= MERGE_CHECK_PARABOLA) and (n_times >= 3) else 1
            t = f[nb] - f[i]
            x = np.polyval(np.polyfit(t, u[nb, 1], deg), 0.0)
            y = np.polyval(np.polyfit(t, u[nb, 2], deg), 0.0)

            # The tolerance of the merging, plus the error of the velocity of the path times the distance the
            #   object moves from the nearest neighbour (the path is extrapolated to a hit beyond its neighbours)
            vx, vy = np.polyfit(t, u[nb, 1], 1)[0], np.polyfit(t, u[nb, 2], 1)[0]
            reach = MERGE_CHECK_SPREAD*np.min(np.abs(t))*math.hypot(vx, vy)
            if math.hypot(u[i, 1] - x, u[i, 2] - y) > tol + reach:
                return False

        return True


    def _trackInfo(self, track):
        """ What _mergeOnce needs of a track: [first frame, last frame, x min, x max, y min, y max, bin, end
            velocity]. The end velocity is None until it is computed.
        """

        return [float(track[0, 0]), float(track[-1, 0]), float(np.min(track[:, 1])), float(np.max(track[:, 1])),
                float(np.min(track[:, 2])), float(np.max(track[:, 2])), self.trackBin(track), None]


    @staticmethod
    def _endVelocity(track):
        """ The velocity at the end of a track: a straight line fitted to its last 5 hits, or the velocity of the
            search of its last hit if they all are in the same frame.

        Arguments:
            track: [ndarray] Hits as rows [frame, x, y, z, vx, vy, ...], sorted by time.

        Return:
            (vx, vy) in pixels per frame.
        """

        end = track[-min(len(track), 5):]
        if np.ptp(end[:, 0]) > 0:
            return np.polyfit(end[:, 0], end[:, 1], 1)[0], np.polyfit(end[:, 0], end[:, 2], 1)[0]

        return end[-1, 4], end[-1, 5]


    def trackBin(self, track):
        """ The bin of the search of the hits of a track: the coarsest one, as a merged track can have hits of
            several tiers, and its positions are only as precise as its coarsest hits.

        Arguments:
            track: [ndarray] Hits as rows [frame, x, y, z, vx, vy, run, bin] (or without the bin column).

        Return:
            [float] The bin, the bin of the search for a track without the bin column.
        """

        return float(np.max(track[:, 7])) if track.shape[1] > 7 else float(self.search_bin)


    @staticmethod
    def _union(a, b):
        """ Hits of two tracks of the same object, sorted by time. """

        u = np.vstack([a, b])
        return u[np.argsort(u[:, 0], kind='stable')]


    def removeGhosts(self, tracks):
        """ Remove the ghosts of bright objects. The search sums the frames along every velocity, and the sums
            along other velocities than the one of a bright object still pass through the object: they give hits
            on the path of the object, at its positions of other times, which link into tracks moving along the
            path at other speeds (often many of them, of a few hits each). Such a track stays on the path of a
            much stronger track during its time, but moves at another velocity (see GHOST_STRENGTH and the
            following constants). Unlike a shadow (see removeShadows) it doesn't stay close to the stronger track at
            the same times.

        Arguments:
            tracks: [list] Tracks as arrays of rows [frame, x, y, z, vx, vy, run, bin], sorted by time.

        Return:
            [list] The tracks which are not ghosts, in the same order.
        """

        if len(tracks) < 2:
            return tracks

        # The strength of a track (the combined significance of its hits, see removeShadows), its velocity (a
        #   straight line fitted to its positions in time, or the median velocity of its hits if they are all in
        #   one frame), its time span and its bin
        strength = np.array([np.sum(tr[:, 3])/math.sqrt(len(tr)) for tr in tracks])
        vel = np.array([(np.polyfit(tr[:, 0], tr[:, 1], 1)[0], np.polyfit(tr[:, 0], tr[:, 2], 1)[0])
                        if np.ptp(tr[:, 0]) > 0 else (np.median(tr[:, 4]), np.median(tr[:, 5])) for tr in tracks])
        bins = np.array([self.trackBin(tr) for tr in tracks])
        run = np.array([float(np.max(tr[:, 6])) if tr.shape[1] > 6 else float(self.opts.slow_run_frames)
                        for tr in tracks])

        # Only the tracks of at least 3 hits have a path to compare with, strongest first
        ghost = np.zeros(len(tracks), dtype=bool)
        for i in np.argsort(-strength):
            if ghost[i] or (len(tracks[i]) < 3):
                continue
            ref = tracks[i]
            speed = math.hypot(*vel[i])
            if speed <= 0:
                continue

            # The segments of the path of the stronger track
            a, d = ref[:-1, 1:3], np.diff(ref[:, 1:3], axis=0)
            length2 = np.maximum(np.sum(d**2, axis=1), 1e-12)
            t0, t1 = ref[0, 0] - run[i], ref[-1, 0] + run[i]

            for j in np.nonzero((strength*GHOST_STRENGTH <= strength[i]) & ~ghost)[0]:
                tr = tracks[j]

                # Most of its hits within the time of the stronger track
                inside = (tr[:, 0] >= t0) & (tr[:, 0] <= t1)
                if inside.mean() < GHOST_ON_PATH:
                    continue

                # At another velocity than the stronger track
                if math.hypot(*(vel[j] - vel[i])) <= GHOST_SPEED_DIFF*speed:
                    continue

                # Most of its hits on the path: the distance of every hit from the nearest segment of the path
                p = tr[:, None, 1:3]
                u = np.clip(np.sum((p - a[None])*d[None], axis=2)/length2[None], 0.0, 1.0)
                dist = np.min(np.hypot(*(p - (a[None] + u[..., None]*d[None])).transpose(2, 0, 1)), axis=1)
                if np.mean(dist <= GHOST_RADIUS*max(bins[i], bins[j])) >= GHOST_ON_PATH:
                    ghost[j] = True

        if ghost.any():
            log.info('Matched filter: {:d} ghosts of bright objects removed'.format(int(ghost.sum())))

        return [tr for tr, g in zip(tracks, ghost) if not g]


    def removeShadows(self, tracks):
        """ Remove the tracks which follow a much stronger track at a small distance: the hits around a very
            bright object (its wings, trails, and the noise it adds) link into many tracks of their own, which
            would each be measured (on the bright object) before the duplicates are removed.

        Arguments:
            tracks: [list] Tracks as arrays of hits, rows [frame, x, y, z, vx, vy, run].

        Return:
            [list] The tracks without the shadows.
        """

        if len(tracks) < 2:
            return tracks

        # The strength of a track: the sum of the significances of its hits divided by the square root of their
        #   number, i.e. the significance of the hits combined (for independent hits of unit noise)
        strength = np.array([np.sum(tr[:, 3])/math.sqrt(len(tr)) for tr in tracks])

        # The velocity of every track (px/frame): the slopes of straight lines fitted to its positions in time,
        #   or the median velocity of the search if all hits are in the same frame
        vel = []
        for tr in tracks:
            if np.ptp(tr[:, 0]) > 0:
                vel.append((np.polyfit(tr[:, 0], tr[:, 1], 1)[0], np.polyfit(tr[:, 0], tr[:, 2], 1)[0]))
            else:
                vel.append((np.median(tr[:, 4]), np.median(tr[:, 5])))
        vel = np.array(vel)

        # Go from the strongest track to the weakest, and compare every track with the stronger tracks kept
        #   before it
        kept = []
        for i in np.argsort(-strength):
            tr = tracks[i]
            shadow = False
            for j in kept:
                ref = tracks[j]

                # Only a much stronger track can have shadows
                if strength[j] < SHADOW_STRENGTH*strength[i]:
                    continue

                # Most of the hits of the track have to be within the time of the stronger track
                inside = (tr[:, 0] >= ref[0, 0]) & (tr[:, 0] <= ref[-1, 0])
                if inside.mean() < SHADOW_OVERLAP:
                    continue

                # The distance of the hits from the stronger track, interpolated to their frames
                f = tr[inside, 0]
                dist = np.hypot(np.interp(f, ref[:, 0], ref[:, 1]) - tr[inside, 1],
                                np.interp(f, ref[:, 0], ref[:, 2]) - tr[inside, 2])

                # A shadow stays close to the stronger track and moves with it: the difference of the velocities
                #   is within 20% of the speed (plus twice the slowest speed of the search, for slow objects)
                speed = math.hypot(*vel[j])
                if (np.median(dist) <= SHADOW_RADIUS) and \
                        (math.hypot(*(vel[i] - vel[j])) <= 0.2*speed + 2*self.opts.speed_min):
                    shadow = True
                    break
            if not shadow:
                kept.append(i)

        if len(kept) < len(tracks):
            log.info('Matched filter: {:d} tracks following stronger ones removed'.format(len(tracks) - len(kept)))

        return [tracks[i] for i in sorted(kept)]


    def acceptTrack(self, track):
        """ A track is kept if its speed is in the range and it moves by more than a few PSF widths (the
            residuals of variable stars are stationary). """

        # All hits in the same frame: the motion can't be measured
        if np.ptp(track[:, 0]) <= 0:
            return False

        # The speed (px/frame) from straight lines fitted to the positions of the hits in time
        vx = np.polyfit(track[:, 0], track[:, 1], 1)[0]
        vy = np.polyfit(track[:, 0], track[:, 2], 1)[0]
        speed = math.hypot(vx, vy)

        # The speed has to be in the range of the search (with a 20% margin above it, the hits at the fastest
        #   velocities of the grid have errors of a grid step), and the motion over the track at least the
        #   minimum displacement
        return (self.opts.speed_min <= speed <= self.opts.speed_max*1.2) \
            and (speed*np.ptp(track[:, 0]) >= self.minDisplacement())


    def minDisplacement(self):
        """ Smallest motion of a detection (px): mf_min_displacement FWHMs of the search smoothing. """

        return self.opts.min_displacement*2.355*self.searchSigma()


    def searchSigma(self):
        """ Sigma of the smoothing of the search (px): the PSF sigma, but at least SEARCH_MIN_SIGMA. A narrower
            smoothing of the binned frames leaves many more noise peaks (and candidate tracks) for little gain,
            as the binning smooths too.
        """

        return max(self.psf_sigma, SEARCH_MIN_SIGMA)


    ### Faint objects ###

    def faintTracks(self, hits, tracks):
        """ Tracks of faint objects: chains of the hits down to the faint threshold over the whole input (see
            linkFaintHits), away from the tracks which are measured anyway, whose signal along their line is
            significant (see faintSignificance). The frames of their measurements follow from that signal.

        Arguments:
            hits: [list] Hits of the search (above the threshold).
            tracks: [list] Tracks which are measured, arrays of hits.

        Return:
            [list] (hits of the chain, frames per measurement) of the chains to measure, the most significant first.
        """

        # Only with a faint threshold below the threshold
        if not self.faint:
            return []

        # All hits down to the faint threshold (a faint object can have a few hits above the threshold too). The
        #   faint hits are not needed any more
        strong = np.array([[h.frame, h.x, h.y, h.z, h.vx, h.vy, h.run, h.bin] for h in hits]).reshape(-1, 8)
        pool = np.vstack([strong, self.faint_hits.astype(np.float64)])
        self.faint_hits = np.zeros((0, 8), dtype=np.float32)

        # Without the hits near the tracks which are measured anyway (their objects, and the sums of other velocities
        #   through them, see removeShadows), also in the frames which their measurement reaches beyond their hits
        #   (see TrackMeasurement), where their positions are extrapolated with the velocities of their end hits
        keep = np.ones(len(pool), dtype=bool)
        order = np.argsort(pool[:, 0], kind='stable')
        frames = pool[order, 0]
        margin = 2*max(max(self.opts.run_frames), 16)
        for tr in tracks:

            # The hits in the frames of the track and its margins, and the positions of the track at their frames:
            #   interpolated between its hits, extrapolated beyond them
            tr = tr[np.argsort(tr[:, 0], kind='stable')]
            i0, i1 = np.searchsorted(frames, [tr[0, 0] - margin, tr[-1, 0] + margin])
            idx = order[i0:i1]
            t = pool[idx, 0]
            x = np.interp(t, tr[:, 0], tr[:, 1])
            y = np.interp(t, tr[:, 0], tr[:, 2])
            before, after = t < tr[0, 0], t > tr[-1, 0]
            x[before] = tr[0, 1] + tr[0, 4]*(t[before] - tr[0, 0])
            y[before] = tr[0, 2] + tr[0, 5]*(t[before] - tr[0, 0])
            x[after] = tr[-1, 1] + tr[-1, 4]*(t[after] - tr[-1, 0])
            y[after] = tr[-1, 2] + tr[-1, 5]*(t[after] - tr[-1, 0])

            # The hits near the track are left out
            keep[idx[np.hypot(pool[idx, 1] - x, pool[idx, 2] - y) <= SHADOW_RADIUS]] = False

        # The chains in the speed range which move far enough (as the tracks, see acceptTrack)
        t1 = time()
        chains = [ch for ch in linkFaintHits(pool[keep]) if self.acceptTrack(ch)]
        self.timing['faint_link'] += time() - t1

        # The chains whose signal along their line is significant, the most significant ones
        t1 = time()
        significance = self.faintSignificance(chains)
        self.timing['faint_significance'] += time() - t1
        ranked = [i for i in np.argsort(-significance, kind='stable') if significance[i] >= FAINT_MIN_SIGNIFICANCE]
        ranked = ranked[:self.opts.faint_max_tracks]
        log.info('Matched filter: {:d} faint hits, {:d} chains, {:d} significant ({:d} measured)'.format(
            int(keep.sum()), len(chains), int(np.sum(significance >= FAINT_MIN_SIGNIFICANCE)), len(ranked)))

        faint = []
        for i in ranked:
            ch = chains[i]

            # The frames per measurement: the smallest power of two which gives FAINT_SAMPLE_SNR from the signal of
            #   a frame (the significance spread over the frames of the chain), which divides the blocks (so the
            #   measurements don't straddle them)
            frame_snr = significance[i]/math.sqrt(np.ptp(ch[:, 0]) + 1)
            n = 1
            while (2*n <= FAINT_MAX_MEASURE_FRAMES) and (self.opts.block_frames % (2*n) == 0) \
                    and (frame_snr*math.sqrt(n) < FAINT_SAMPLE_SNR):
                n *= 2
            faint.append((ch, n))

        return faint


    def faintSignificance(self, chains):
        """ The significance of the signal along the line of every chain: the PSF-weighted signal at the positions
            of the line in the frames from its first to its last hit, minus the signal at the same positions when
            the object is elsewhere, the smaller of two tests as in verify: before and after the chain, and in the
            frames of the chain in the reverse order.

        Arguments:
            chains: [list] Chains of hits, sorted by frame.

        Return:
            [ndarray] Significance of every chain (sigma).
        """

        if not chains:
            return np.zeros(0)

        # The three tests of every chain (on its line, off it in time, in reverse order) and the motion of its object
        fwhm = 2.355*self.psf_sigma
        starts = np.array(self.block_starts)
        tests, motions = [], []
        for ch in chains:

            # The line through the hits, in every frame from the first to the last hit
            frames = np.arange(int(math.ceil(ch[0, 0])), int(math.floor(ch[-1, 0])) + 1)
            fit_x, fit_y = TrackMeasurement._robustLineFit(ch[:, 0], ch[:, 1], ch[:, 2])
            xs, ys = np.polyval(fit_x, frames), np.polyval(fit_y, frames)

            # The same positions when the object is at least 3 PSF widths away (at least 16 frames before and
            #   after, in the frames of the input), and in the frames of the chain in the reverse order
            shift = int(math.ceil(max(3*fwhm/max(math.hypot(fit_x[0], fit_y[0]), 1e-3), 16)))
            off_f = np.concatenate([frames - shift, frames + shift])
            inside = (off_f >= 0) & (off_f < self.total_frames)
            sets = {'on': (frames, xs, ys), 'off': (off_f[inside], np.tile(xs, 2)[inside], np.tile(ys, 2)[inside]),
                    'reverse': (frames, xs[::-1], ys[::-1])}

            # The positions of every test in each block of frames, and their sums (sum(g*z), sum(g^2), see
            #   forcedTrackSignal)
            tests.append({key: (s, blockIndices(s[0], starts, end=self.block_ends[-1]), [0.0, 0.0])
                          for key, s in sets.items()})

            # With the streak search, the motion of the object during a frame (the PSF smeared along it, see
            #   verify)
            motions.append((fit_x[0], fit_y[0]) if self.opts.streak_search else (0.0, 0.0))

        # The sums of all tests, block by block, without the pixels of the stars (see verify)
        radius = int(math.ceil(2.5*self.psf_sigma))
        for k, (first, last) in enumerate(zip(self.block_starts, self.block_ends)):

            # The chains with a test in this block, and the normalized frames of the block without the stars
            active = [i for i, c in enumerate(tests) if any(k in idx for _, idx, _ in c.values())]
            if not active:
                continue
            z, _ = self.normalizedBlock(k)
            excluded = (self.static_masks[k] | (self.backgrounds[k].stars > 0)).astype(np.uint8)

            def sumBlock(i):
                """ Add the sums of the tests of a chain in this block. """

                # The patch includes half the streak of the object
                vx, vy = motions[i]
                radius_streak = radius + int(math.ceil(0.5*math.hypot(vx, vy)))
                for (fr, x, y), idx, sums in tests[i].values():
                    sel = idx.get(k)
                    if sel is not None:
                        sgz, sgg = forcedTrackSignal(z, fr[sel] - first, x[sel], y[sel], self.psf_sigma,
                                                     radius_streak, excluded, vx, vy)
                        sums[0] += sgz
                        sums[1] += sgg

            self.inThreads(sumBlock, active)

        # The signal along the line minus the signal of each test scaled to the same weights, in units of the noise
        #   of the difference (see verify), the smaller of the two
        significance = np.zeros(len(tests))
        for i, c in enumerate(tests):
            on_gz, on_gg = c['on'][2]
            values = []
            for key in ('off', 'reverse'):

                # A test without positions in the frames (e.g. a chain over the whole input) gives no significance
                gz, gg = c[key][2]
                if (on_gg <= 0) or (gg <= 0):
                    values = [0.0]
                    break
                values.append((on_gz - gz*on_gg/gg)/math.sqrt(on_gg*(1 + on_gg/gg)))
            significance[i] = min(values)

        return significance


    ### Measurement ###

    def polarLine(self, cent):
        """ The polar line (rho, theta) of a detection in the binned image, as for the normal detections. The
            coordinates of a pixel center of the unbinned image in the binned image are (x - (b - 1)/2)/b, and the
            y axis of the polar line points up.

        Arguments:
            cent: [ndarray] Centroids of the detection (unbinned), rows [frame, x, y, ...].

        Return:
            (rho, theta): [tuple] The polar line.
        """

        (xa, ya), (xb, yb) = (cent[[0, -1], 1:3] - (self.det_bin - 1)/2.0)/self.det_bin

        return getPolarLine(xa, self.height - ya, xb, self.height - yb, self.height, self.width)


    def connectedLight(self, frame, x1, y1, x2, y2):
        """ The fraction of the segment between two points which is well above the background (CONNECTED_LEVEL in
            the noise) in a frame. See removeDuplicateDetections.

        Arguments:
            frame: [float] Frame of the input.
            x1: [float] X of the first point (unbinned px).
            y1: [float] Y of the first point (unbinned px).
            x2: [float] X of the second point (unbinned px).
            y2: [float] Y of the second point (unbinned px).

        Return:
            [float] The fraction of the segment above the level, None if the frame has no background (e.g. outside
                the frames of this detector) or the segment is outside the image.
        """

        # The block of the frame
        f = int(round(frame))
        k = next((k for k, (a, b) in enumerate(zip(self.block_starts, self.block_ends)) if a <= f < b), None)
        if (k is None) or (k not in self.backgrounds) or (self.backgrounds[k].median is None):
            return None

        # The frame above the background of its block, in its noise
        bg = self.backgrounds[k]
        z = (self.readFrames(f, 1)[0] - bg.median)/bg.noise

        # The segment in the binned image, sampled every half pixel
        b = self.det_bin
        p1 = np.array([x1 - (b - 1)/2.0, y1 - (b - 1)/2.0])/b
        p2 = np.array([x2 - (b - 1)/2.0, y2 - (b - 1)/2.0])/b
        n = max(2, int(np.ceil(2*np.hypot(*(p2 - p1)))) + 1)
        t = np.linspace(0, 1, n)[:, None]
        pts = np.round(p1 + t*(p2 - p1)).astype(int)

        # Only the part of the segment inside the image is judged
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < self.width) & (pts[:, 1] >= 0) & (pts[:, 1] < self.height)
        if not inside.any():
            return None
        pts = pts[inside]

        # The fraction of the samples above the level
        return float(np.mean(z[pts[:, 1], pts[:, 0]] > CONNECTED_LEVEL))


    def joinDetections(self, detections, connected=None, cut=None):
        """ Join the pieces of tracks (a faint object can be lost for a while, e.g. behind stars) and remove the
            duplicates. An object can be lost for many seconds while moving only a few PSF widths, so pieces
            are joined across any gap in which the object moved less than 15 PSF widths, if the second piece
            begins within 2 PSF widths of the extrapolated end of the first one.

        Arguments:
            detections: [list] Detections as [rho, theta, centroids].

        Keyword arguments:
            connected: [function] The light between two points in a frame (see connectedLight), e.g. over the
                segments of an input (see connectedOver). None by default (connectedLight of this detector).
            cut: [set] Identities (id) of the centroid arrays of the detections whose faint ends were already cut
                (see trimFaintEnds), e.g. as they were measured; the arrays have to be held by the caller. None by
                default (the faint ends of all detections are cut here).

        Return:
            [list] The joined detections.
        """

        # The distances of the joining are in the binned image (as is the image size), the ones of the duplicates
        #   in the unbinned image of the centroids
        fwhm = 2.355*self.psf_sigma
        max_gap = 4*max(self.opts.run_frames + [self.opts.slow_run_frames])
        detections = joinContinuousDetections(detections, max_gap, 15*fwhm, 2*fwhm, self.height, self.width,
                                              bin_factor=self.det_bin)

        # A joined track can end in a piece much fainter than the rest of it (a piece built from the sums of other
        #   velocities through a bright object joins the object's track): its faint ends are cut (a track left too
        #   short or moving too little to be a detection is dropped), before the duplicates are removed, as the rest
        #   of such a track is often a second detection of the object. The tracks already cut (e.g. when they were
        #   measured, and not joined since) are not cut again, as the level of the cut follows the track
        trimmed = []
        for det in detections:
            if (cut is not None) and (id(det[2]) in cut):
                trimmed.append(det)
                continue

            # Cut, and kept if what is left is still a detection
            cent = trimFaintEnds(det[2])
            if len(cent) == len(det[2]):
                trimmed.append(det)
            elif isValidDetection(cent, self):
                trimmed.append([*self.polarLine(cent), cent])

        # The duplicates: detections of the same object close to a longer one, also further along the motion when
        #   the light between them is connected (the same bright object measured twice), or on it in a part of
        #   their frames (a piece of the sums of other velocities through the object which runs into it)
        detections = removeDuplicateDetections(trimmed, fwhm*self.det_bin, 3*fwhm*self.det_bin,
                                               connected=connected if connected is not None else self.connectedLight,
                                               max_connected_along=DUPLICATE_ALONG*fwhm*self.det_bin,
                                               min_coincident=DUPLICATE_COINCIDENT)

        # A fast object measured twice along its streak, further apart than the duplicates above
        return removeStreakDuplicates(detections, fwhm*self.det_bin,
                                      connected=connected if connected is not None else self.connectedLight)


    def run(self):
        """ Detect and measure the objects of the whole input.

        Return:
            [list] Detections as [rho, theta, centroids], centroids rows [frame, x, y, intensity, background,
                snr, saturated] in the unbinned image, as from Detection.detectMeteors.
        """

        t0 = time()

        # The search needs at least one run of the longest length
        if self.frame_end - self.frame_first < max(self.opts.run_frames):
            log.warning('Matched filter: only {:d} frames, nothing to search'.format(self.frame_end
                                                                                    - self.frame_first))
            self.timing['total'] = time() - t0
            return []

        # Pass 1: the velocity search on the binned frames, and the linking of its hits into tracks
        hits = self.search()
        linked = self.linkHits(hits)

        # Bright objects too slow for the search (they are part of the background of a block)
        t1 = time()
        slow_tracks = self.slowSearch()
        if slow_tracks:
            linked = self.mergeTracks(linked + slow_tracks)
        self.timing['slow_search'] = time() - t1

        # The hits of a track whose velocity disagrees with the motion of the track are not of the object (e.g.
        #   the sums of other velocities through a bright object, linked to its track by their position)
        linked = [cleanTrackHits(tr, min_hits=self.opts.min_hits) for tr in linked]

        # Keep the tracks in the speed range which move far enough, without the ghosts of bright objects (along
        #   their path at other speeds) and the tracks following stronger tracks
        tracks = self.removeShadows(self.removeGhosts([tr for tr in linked if self.acceptTrack(tr)]))

        # The time of the measurement grows with the number of tracks, which can be large in bad conditions
        #   (e.g. thin clouds, a very bright star): the strongest tracks are kept, so a file can't take much
        #   longer than usual
        if len(tracks) > self.opts.max_tracks:

            # The strength of a track is the combined significance of its hits (see removeShadows)
            strength = np.array([np.sum(tr[:, 3])/math.sqrt(len(tr)) for tr in tracks])
            keep = np.sort(np.argsort(-strength)[:self.opts.max_tracks])
            self.tracks_dropped = len(tracks) - len(keep)
            log.warning('Matched filter: {:d} candidate tracks, only the {:d} strongest are measured'.format(
                len(tracks), self.opts.max_tracks))
            tracks = [tracks[i] for i in keep]
        self.tracks_measured = len(tracks)
        self._debug = (hits, linked)
        log.info('Matched filter: {:d} hits, {:d} candidate tracks ({:d} from the search of slow objects)'.format(
            len(hits), len(tracks), len(slow_tracks)))

        # The background of the slow objects is estimated without them before they are measured, or they would
        #   be measured on a background which contains them (and on pixels masked as stars)
        if slow_tracks:
            t1 = time()
            self.maskObjects([tr[:, :3] for tr in slow_tracks])
            self.timing['slow_background'] += time() - t1

        # Faint objects: chains of the hits down to the faint threshold away from these tracks, measured with them
        faint = self.faintTracks(hits, tracks)

        # Pass 2: measure the tracks on the unbinned frames (the positions, verification and photometry)
        t1 = time()
        measured = self.measure(tracks, faint)
        self.timing['measure'] += time() - t1

        # Convert the measurements to the detections of the normal detection
        detections = []
        for cent in measured:

            # The photometry of every frame of the track, measured by measure()
            phot = self.photometry.get(id(cent))

            # The positions of the measurements combined over the neighbouring measurements
            if self.opts.smooth_frames > 0:
                xy, keep = smoothPositions(cent, self.opts.smooth_frames)
                cent = cent[keep]
                cent[:, 1:3] = xy

            # Only the measurements on clear sky are kept, as on clouded sky neither their positions nor their
            #   magnitudes can be trusted, and the track needs enough of them (in unbinned coordinates)
            if self.sky_map is not None:
                xy = cent[:, 1:3]*self.det_bin + (self.det_bin - 1)/2.0
                clear = self.sky_map.isClear(xy[:, 0], xy[:, 1], cent[:, 0])
                if clear.sum() < self.opts.min_centroids:
                    self.cloud_detections_removed += 1
                    self.cloud_removed_frames.append(int(round(cent[0, 0])))
                    continue
                cent = cent[clear]

            # Only tracks which last and move long enough are kept
            if (cent[-1, 0] - cent[0, 0] < self.opts.min_frames) \
                    or (np.hypot(*(cent[-1, 1:3] - cent[0, 1:3])) < self.minDisplacement()):
                continue

            # One row per frame: the intensity of every frame, at the position on the track
            cent = perFrameRows(cent, self.opts.max_measure_frames, phot)

            # The frames on clouded sky (between and beyond the measurements) are left out too
            if self.sky_map is not None:
                xy = cent[:, 1:3]*self.det_bin + (self.det_bin - 1)/2.0
                clear = self.sky_map.isClear(xy[:, 0], xy[:, 1], cent[:, 0])
                self.cloud_rows_removed += int((~clear).sum())
                self.cloud_removed_row_frames += cent[~clear, 0].tolist()
                first = cent[0, 0]
                cent = cent[clear]
                if len(cent) < 2:
                    self.cloud_detections_removed += 1
                    self.cloud_removed_frames.append(int(round(first)))
                    continue

            # The ends of a track where it is much fainter than along the rest of it are not of the object (e.g. a
            #   piece of track built from the sums of other velocities through a bright object, which continues
            #   the object over empty sky), and what is left has to be a detection: enough rows, lasting and
            #   moving long enough
            cent = trimFaintEnds(cent)
            if (len(cent) < max(2, self.opts.min_centroids)) or (cent[-1, 0] - cent[0, 0] < self.opts.min_frames) \
                    or (np.hypot(*(cent[-1, 1:3] - cent[0, 1:3])) < self.minDisplacement()):
                continue

            # Unbinned coordinates, and the intensity of the unbinned image (averaged bins hold the mean of the
            #   pixels, summed bins already their sum)
            if self.det_bin > 1:
                cent[:, 1:3] = cent[:, 1:3]*self.det_bin + (self.det_bin - 1)/2.0
                if self.config.detection_binning_method == 'avg':
                    cent[:, 3] *= self.det_bin**2

            detections.append([*self.polarLine(cent), cent])

        # Join the pieces of tracks and remove the duplicates (their faint ends were cut above)
        detections = self.joinDetections(detections, cut={id(det[2]) for det in detections})

        # The whole tracks which wander sideways from one smooth path are no objects (e.g. pieces of diffuse
        #   light)
        detections, rejected = removeDeviatingDetections(detections, 2.355*self.psf_sigma*self.det_bin
                                                         if self.psf_sigma is not None else None,
                                                         self.opts.track_deviation_fwhm,
                                                         self.opts.track_deviation_length)
        self.deviation_rejected += len(rejected)

        # The kept normalized frames are not needed any more (a segmented input keeps the detector of every
        #   segment until the end of the input)
        self.block_cache.clear()

        self.timing['total'] = time() - t0
        log.info('Matched filter: {:d} detections in {:.1f} s ({:s})'.format(len(detections),
            self.timing['total'], ', '.join('{:s} {:.1f} s'.format(k, v) for k, v in self.timing.items()
                                              if k != 'total')))

        return detections


    def apertureCorrection(self, star_list, first_frames=None):
        """ Factor which puts the intensities of the detections on the scale of the intensities of the stars
            (CALSTARS), from which the magnitudes are calibrated. The stars are measured with the aperture of
            the detections (within 3 PSF sigmas) on the mean of the block of frames, minus the sky in an
            annulus around them, and
            the factor is the median ratio of their CALSTARS intensities to these sums. The two differ by up to
            ~15%, depending on how far the wings of the PSF reach beyond the aperture.

        Arguments:
            star_list: [list] Stars of the chunks, [ff_name, [(y, x, intensity, amplitude, fwhm, background, snr,
                saturated pixels), ...]] in the unbinned image, as from extractStarsFrameInterface.

        Keyword arguments:
            first_frames: [list] The first frame of every chunk of the stars, -1 where it is not known (see
                chunkFirstFrames). After a gap in the recording the frames are not where the time and the frame
                rate put them. None by default (from the time of the chunk and the frame rate).

        Return:
            (factor, n_stars): [tuple] The factor (None if there are fewer than MIN_APERTURE_STARS stars) and the
                number of stars it was measured on.
        """

        # The sums of averaged bins are the mean of their pixels, so they are scaled to the sums of the
        #   unbinned pixels
        b = self.det_bin
        scale = b*b if (b > 1) and (self.config.detection_binning_method == 'avg') else 1.0

        # The aperture of the photometry of the detections
        radius = 3.0*self.psf_sigma

        # The sky around a star: the median of an annulus beyond the aperture
        r_in, r_out = radius + 3.0, radius + 7.0
        r = int(math.ceil(r_out))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        t_begin = self.img_handle.beginning_datetime
        starts = np.array(self.block_starts)
        m = self.opts.edge_margin + r + 1

        ratios = []
        for i, entry in enumerate(star_list):
            ff_name, stars = entry[0], entry[1]

            # The first frame of the chunk of the stars (from its time from the beginning of the input if not
            #   given), and the block of frames which contains it (only the frames which were searched)
            if first_frames is not None:
                frame = first_frames[i]
                if frame < 0:
                    continue
            else:
                try:
                    dt = (filenameToDatetime(ff_name) - t_begin.replace(tzinfo=None)).total_seconds()
                except Exception:
                    continue
                frame = dt*self.img_handle.fps
            if not (self.frame_first <= frame < self.frame_end):
                continue
            k = int(np.clip(np.searchsorted(starts, frame, side='right') - 1, 0, len(starts) - 1))
            bg = self.backgrounds.get(k)
            if bg is None:
                continue

            # The intensity of an object in a frame is on average the mean of its flux, and the stars of the
            #   calibration are measured on the average of their frames, so the mean of the block is used
            image = bg.mean if bg.mean is not None else bg.median

            for star in stars:

                # Saturated stars have wrong intensities in both
                y, x, intensity, saturated = star[0], star[1], star[2], star[7]
                if (intensity <= 0) or (saturated > 0):
                    continue

                # The position of the star in the binned image, away from the border
                xb, yb = (x - (b - 1)/2.0)/b, (y - (b - 1)/2.0)/b
                if not ((m <= xb < self.width - m) and (m <= yb < self.height - m)):
                    continue

                # The patch around the star, and the squared distance of every pixel from the star center
                xi, yi = int(round(xb)), int(round(yb))
                patch = image[yi - r:yi + r + 1, xi - r:xi + r + 1]
                d2 = (xx + xi - xb)**2 + (yy + yi - yb)**2

                # The sum within the aperture minus the sky of the annulus
                sky = np.median(patch[(d2 > r_in**2) & (d2 <= r_out**2)])
                total = float(np.sum(patch[d2 <= radius**2] - sky))
                if total > 0:
                    ratios.append(intensity/(total*scale))

        # The median ratio is robust to the stars with a neighbour in the aperture or annulus
        if len(ratios) < MIN_APERTURE_STARS:
            return None, len(ratios)

        return float(np.median(ratios)), len(ratios)


    def summary(self):
        """ Summary of the last run, for the logs and the done file of the input. """

        return {
            'aperture_correction': self.aperture_correction,
            'clear_sky_fraction': (round(float(np.mean(self.sky_map.clear)), 3) if self.sky_map is not None
                                   else None),
            'cloud_frames_removed': int(self.cloud_rows_removed),
            'cloud_detections_removed': int(self.cloud_detections_removed),
            'cloud_removed_utc': self.frameTimes(self.cloud_removed_frames),
            'tracks_dropped': int(self.tracks_dropped),
            'total_frames': int(self.total_frames),
            'input_end_utc': self.inputEnd(),
            'psf_sigma': self.psf_sigma,
            'gpu': bool(self.opts.use_gpu),
            'timing_s': {k: round(v, 2) for k, v in self.timing.items()},
            'tracks': [{'first_frame': float(s[0]), 'last_frame': float(s[1]), 'significance': round(float(s[2]), 2),
                        'detected': bool(s[2] >= self.opts.track_significance), 'linear_structure': bool(s[-1])}
                       for s in self.significances],
            'linear_structure_rejected': int(sum(bool(s[-1]) for s in self.significances)),
            'deviation_rejected': int(self.deviation_rejected),
            'faint_measured': int(self.faint_measured),
        }


    def inputEnd(self):
        """ The end of the input: the time of its last frame and the length of a frame (UTC, ISO format), None if
            it can't be read. """

        last = self.frameTimes([self.total_frames - 1])[0]
        if (last is None) or not self.img_handle.fps:
            return None

        end = datetime.datetime.strptime(last, '%Y-%m-%dT%H:%M:%S.%f') \
            + datetime.timedelta(seconds=1.0/self.img_handle.fps)

        return end.strftime('%Y-%m-%dT%H:%M:%S.%f')


    def frameTimes(self, frames):
        """ Look up the times of the given frames in the input, e.g. so the detections removed by clouds can be
            placed in the night.

        Arguments:
            frames: [list] Frame numbers.

        Return:
            times: [list] Times of the frames (UTC, ISO format), None for a frame whose time can't be read.
        """

        times = []
        for frame in frames:

            # Take the time of the frame from the input (as UTC without the time zone)
            try:
                t = self.img_handle.currentFrameTime(frame_no=int(frame), dt_obj=True)
                if t.tzinfo is not None:
                    t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
                times.append(t.strftime('%Y-%m-%dT%H:%M:%S.%f'))
            except Exception:
                times.append(None)

        return times


    def measure(self, tracks, faint=()):
        """ Pass 2: measure every track on the unbinned frames by fitting a moving PSF to short runs of frames,
            following the track with a local motion model.

        Arguments:
            tracks: [list] Tracks of hits.

        Keyword arguments:
            faint: [list] (hits of the chain, frames per measurement) of the chains of faint objects (see
                faintTracks), measured the same way with the given frames per measurement. Empty by default.

        Return:
            [list] Centroid arrays of the accepted tracks.
        """

        if not (tracks or faint):
            return []

        # The measurements of a faint chain combine up to FAINT_MAX_MEASURE_FRAMES frames, its position is
        #   predicted from its hits over the reach of the chain (they are sparse), and it is measured as far as that
        #   reach beyond its hits (the object is often visible beyond the runs where the noise added to it)
        faint_opts = copy.copy(self.opts)
        faint_opts.max_measure_frames = max(self.opts.max_measure_frames, FAINT_MAX_MEASURE_FRAMES)
        items = [(tr, None) for tr in tracks] + list(faint)

        def state(i):
            """ The measurement state of the i-th track. """

            # A track of the search, or a faint chain with its frames per measurement and the reach of its hits
            tr, n = items[i]
            if n is None:
                return TrackMeasurement(tr, self.opts, self.psf_sigma)
            return TrackMeasurement(tr, faint_opts, self.psf_sigma, n_frames=n, window=FAINT_SEED_FRAMES[1],
                                    extend=FAINT_SEED_FRAMES[1])

        # Every track is followed through the blocks of frames by its own measurement state, which predicts
        #   the position of the object from the hits of the search and then from its own measurements
        states = [state(i) for i in range(len(items))]
        self.measurePass(states)
        cents = [st.centroids() for st in states]

        # A slow object stays on the same pixels for a good part of the frames from which the background is
        #   estimated, so the background contains part of it: the background is estimated again without the
        #   pixels of the slow objects, and they are measured again
        slow = [i for i, cent in enumerate(cents) if (cent is not None) and (trackSpeed(cent) < SLOW_SPEED)]
        if slow:
            t1 = time()
            self.maskObjects([cents[i] for i in slow])
            again = [state(i) for i in slow]
            self.measurePass(again)
            for i, st in zip(slow, again):
                cent = st.centroids()
                if cent is not None:
                    cents[i] = cent
            self.timing['slow_background'] += time() - t1

        # The faint chains measured, counted in the summary
        self.faint_measured += len(faint)

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

        # Pixels within 4 PSF sigmas (plus 1 px for the position error) of an object contain its light
        radius = 4*self.psf_sigma + 1
        n_blocks = len(self.block_starts)

        # The motion of the stars (px per frame), which the samples are aligned with
        drift = np.array(self.drift(), dtype=np.float64)

        # The sampled frames of the blocks (read once, kept while they are needed), the blocks whose background
        #   was changed, and the reference backgrounds of the blocks (see referenceBackground)
        samples = {}
        touched = set()
        references = {}

        def blockSamples(j):
            """ The numbers and the frames of every step-th frame of the block j, read on the first use. """

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
            #   of the sample. The samples are aligned with the motion of the stars to the middle of the block (see
            #   below), which moves the object in them by up to the motion over the neighbouring blocks: the region
            #   is enlarged by that much
            bg = self.backgrounds[k]
            mid = (self.block_starts[k] + self.block_ends[k] - 1)/2.0
            pad_x, pad_y = np.abs(drift)*max(mid - lo, hi - mid)
            for c in tracks:

                # The bounding box of the positions of the track during the neighbouring blocks (with a margin of
                #   64 frames, the measurements can be up to 2 runs of frames apart), enlarged by the radius
                sel = (c[:, 0] >= lo - 64) & (c[:, 0] < hi + 64)
                if sel.sum() == 0:
                    continue
                pos = c[sel, 1:3]
                x0 = int(max(np.floor(pos[:, 0].min() - radius - 2 - pad_x), 0))
                x1 = int(min(np.ceil(pos[:, 0].max() + radius + 3 + pad_x), self.width))
                y0 = int(max(np.floor(pos[:, 1].min() - radius - 2 - pad_y), 0))
                y1 = int(min(np.ceil(pos[:, 1].max() + radius + 3 + pad_y), self.height))
                if (x1 <= x0) or (y1 <= y0):
                    continue

                # The region in every sample of the neighbouring blocks, with the pixels near any of the tracks at
                #   the time of the sample set to NaN (the positions are interpolated between the measurements, and
                #   the tracks are masked up to 8 frames beyond their ends). The region of a sample is taken where
                #   the stars were at its time (shifted by the motion of the stars from the middle of the block, with
                #   subpixel interpolation), so the stars of all samples are where they are in the block, as in the
                #   rest of its background (see backgroundPass); the positions of the objects are shifted the same way
                yy, xx = np.mgrid[y0:y1, x0:x1]
                centre = ((x0 + x1 - 1)/2.0, (y0 + y1 - 1)/2.0)
                crops = []
                smoothing = []
                for j in near:
                    fr, frames = blockSamples(j)
                    for f, img in zip(fr, frames):
                        sx, sy = drift*(f - mid)

                        # The bilinear interpolation of the shift averages neighbouring pixels, which reduces the
                        #   standard deviation of their independent noise by this factor (1 for whole pixels)
                        ax, ay = sx - np.floor(sx), sy - np.floor(sy)
                        smoothing.append(math.sqrt(((1 - ax)**2 + ax**2)*((1 - ay)**2 + ay**2)))

                        # The region of the sample where the stars were at its time, and the tracks masked in it
                        src = img if img.dtype == np.float32 else img.astype(np.float32)
                        crop = cv2.getRectSubPix(src, (x1 - x0, y1 - y0),
                                                 (centre[0] + sx, centre[1] + sy)).astype(np.float64)
                        for t in tracks:
                            if t[0, 0] - 8 <= f <= t[-1, 0] + 8:
                                px, py = np.interp(f, t[:, 0], t[:, 1]) - sx, np.interp(f, t[:, 0], t[:, 2]) - sy
                                crop[(xx - px)**2 + (yy - py)**2 <= radius**2] = np.nan
                        crops.append(crop)

                # The median of every pixel over the samples in which it is not covered by an object
                crops = np.array(crops)
                masked = nanMedian(crops)

                # The noise too: the pixels which the object crosses during the block vary between the sky and the
                #   object, and their noise would weigh down the object itself in the fits. It is the MAD of the
                #   unmasked samples, scaled by 1.4826 to the standard deviation of Gaussian noise. The deviations of
                #   every sample are divided by the smoothing of its interpolation, so the noise is the one of the
                #   frames, which are normalized with it
                smoothing = np.array(smoothing)[:, None, None]
                mad = 1.4826*nanMedian(np.abs(crops - masked)/smoothing)

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

                # Replace the background of the region where enough samples were left (the background is never
                #   below 1 ADU of noise, as in backgroundPass)
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
            self.static_masks[k] = self.blockMask(k, self.backgrounds[k])

        # The kept normalized blocks which used a changed background (their own or a neighbour's, see normalize)
        #   are read and normalized again when they are needed
        for k in touched:
            for j in (k - 1, k, k + 1):
                self.block_cache.pop(j, None)


    def inThreads(self, function, items):
        """ Call a function on every item (e.g. the measurement of a track in a block of frames, or a slice of the
            frames of a block), in parallel threads. The compiled kernels of the fits and the sums release the GIL,
            as numpy does in the operations on whole frames, so the threads run on several cores. The items have
            to be independent (each changing only its own state or its own frames); the results are the same as
            in a loop.

        Arguments:
            function: [callable] Function of an item.
            items: [list] Items.
        """

        # As many threads as the compiled code may use (mf_threads), at most one per item
        n_threads = min(numba.get_num_threads(), len(items))
        if n_threads <= 1:
            for item in items:
                function(item)
            return

        # The exceptions of the threads are raised here
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            for _ in executor.map(function, items):
                pass


    def measurePass(self, states):
        """ Measure the tracks in all blocks of frames they overlap. """

        # The frames are read block by block, and every block is read only once for all tracks in it
        for k, (first, last) in enumerate(zip(self.block_starts, self.block_ends)):

            active = [st for st in states if st.overlaps(first, last)]
            if not active:
                continue

            # The unbinned frames with their saturated pixels, normalized to unit noise
            bg = self.backgrounds[k]
            z, saturated = self.normalizedBlock(k)

            # Every track continues its measurements in this block. The tracks are independent (each changes only
            #   its own state, the frames are only read), so they are measured in parallel threads
            static = self.static_masks[k]
            self.inThreads(lambda st: st.measureBlock(z, first, last, bg, static, saturated), active)


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

            # The smooth track through the measurements, in every frame from the first to the last measurement
            frames = np.arange(int(math.ceil(cent[0, 0])), int(math.floor(cent[-1, 0])) + 1)
            xs, ys = smoothTrack(cent, frames)

            # The time in which the object moves by 3 PSF widths (at least 16 frames): the same positions are
            #   taken that many frames before and after the track, and the frames outside the input dropped
            span = max(frames[-1] - frames[0], 1)
            speed = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])/span
            shift = int(math.ceil(max(3*fwhm/max(speed, 1e-3), 16)))
            off = [(frames + d, xs, ys) for d in (-shift, shift)]
            off = [(f[(f >= 0) & (f < self.total_frames)], x[(f >= 0) & (f < self.total_frames)],
                    y[(f >= 0) & (f < self.total_frames)]) for f, x, y in off]
            off_f = np.concatenate([o[0] for o in off])
            off_x = np.concatenate([o[1] for o in off])
            off_y = np.concatenate([o[2] for o in off])

            # Reordered: the positions of the track in the frames of the track, in 4 other orders (reversed, shifted
            #   cyclically by 1/3 and 2/3 of the track, and reversed and shifted by 1/2). Every frame then has the
            #   position of a different time, at which the object was elsewhere
            n = len(frames)
            orders = [np.arange(n)[::-1], np.roll(np.arange(n), n//3), np.roll(np.arange(n), 2*n//3),
                      np.roll(np.arange(n)[::-1], n//2)]
            reord = (np.tile(frames, len(orders)), np.concatenate([xs[o] for o in orders]),
                     np.concatenate([ys[o] for o in orders]))

            # The motion during a frame, for the shape of the streak of a fast object in the PSF-weighted sums (only
            #   with the streak search; the same streak at the positions of all three tests)
            motion = ((xs[-1] - xs[0])/span, (ys[-1] - ys[0])/span) if self.opts.streak_search else (0.0, 0.0)

            # The sums of the three tests (sum(g*z), sum(g^2), see forcedTrackSignal), the photometry of every
            #   frame (the intensity, saturated pixels and noise), and the stack of the frames along the track
            checks.append({'on': (frames, xs, ys), 'off': (off_f, off_x, off_y), 'reord': reord, 'motion': motion,
                           'on_sum': [0.0, 0.0], 'off_sum': [0.0, 0.0], 'reord_sum': [0.0, 0.0],
                           'phot': np.zeros(n), 'phot_sat': np.zeros(n, dtype=np.int64),
                           'phot_noise': np.zeros(n),
                           'stack': np.zeros((2*EXTENT_RADIUS + 1, 2*EXTENT_RADIUS + 1)), 'stack_n': 0})

        # The positions of every test in each block of frames
        starts = np.array(self.block_starts)
        for c in checks:
            for key in ('on', 'off', 'reord'):
                c[key + '_idx'] = blockIndices(c[key][0], starts, end=self.block_ends[-1])

        # The half size of the patch of the PSF-weighted sums: the weights are negligible beyond 2.5 sigmas
        radius = int(math.ceil(2.5*self.psf_sigma))

        # The frames are read block by block, and the sums of all tests in the block accumulated
        for k, (first, last) in enumerate(zip(self.block_starts, self.block_ends)):

            active = [c for c in checks if (k in c['on_idx']) or (k in c['off_idx'])]
            if not active:
                continue

            # The normalized frames of the block (kept from the search, or read again) and the noise of its pixels
            z, saturated = self.normalizedBlock(k)
            noise = self.backgrounds[k].noise

            # The pixels of the stars are left out: the stars which are too faint to be masked still vary
            #   (scintillation), and a slow track over a few of them would collect their residuals
            excluded = (self.static_masks[k] | (self.backgrounds[k].stars > 0)).astype(np.uint8)

            # The clouded regions of the block, left out of the shape of the objects (cloud structure would make
            #   an object look extended)
            clouded = None
            if self.sky_map is not None:
                clouded = self.sky_map.cloudMask(first, last - 1, self.height, self.width, bin_factor=self.det_bin)

            # The tests of every track in this block. The tracks are independent (each adds to its own sums, the
            #   frames are only read), so they are run in parallel threads
            def verifyBlock(c):

                # The PSF-weighted sums of the three tests at their positions in the frames of this block, with the
                #   PSF smeared along the streak of a fast object (the patch then includes half the streak)
                vx, vy = c['motion']
                radius_streak = radius + int(math.ceil(0.5*math.hypot(vx, vy)))
                for key in ('on', 'off', 'reord'):
                    fr, x, y = c[key]
                    sel = c[key + '_idx'].get(k)
                    if sel is not None:
                        sgz, sgg = forcedTrackSignal(z, fr[sel] - first, x[sel], y[sel], self.psf_sigma,
                                                     radius_streak, excluded, vx, vy)
                        c[key + '_sum'][0] += sgz
                        c[key + '_sum'][1] += sgg

                # The intensity of every frame along the track, whether its position was measured or not (e.g.
                #   between the flashes of a flashing object), so the light curve has all frames
                sel = c['on_idx'].get(k)
                if sel is not None:
                    fr, x, y = c['on'][0][sel], c['on'][1][sel], c['on'][2][sel]

                    # The motion per frame in this block (the median of the differences of the positions), for
                    #   the segment the object moves along during a frame
                    vx = float(np.median(np.gradient(c['on'][1])[sel])) if len(c['on'][1]) > 1 else 0.0
                    vy = float(np.median(np.gradient(c['on'][2])[sel])) if len(c['on'][2]) > 1 else 0.0

                    # The sum within 3 PSF sigmas of the segment, and the number of saturated pixels
                    radius_phot = 3.0*self.psf_sigma
                    sums, n_sat = framePhotometry(z, saturated, fr - first, noise, x, y, vx, vy,
                                                  radius_phot)
                    c['phot'][sel] = sums
                    c['phot_sat'][sel] = n_sat

                    # The noise of the sum: the noise of a pixel times the square root of the number of pixels in
                    #   the aperture, sqrt(pi*r^2) = sqrt(pi)*r
                    xi = np.clip(np.round(x).astype(int), 0, noise.shape[1] - 1)
                    yi = np.clip(np.round(y).astype(int), 0, noise.shape[0] - 1)
                    c['phot_noise'][sel] = noise[yi, xi]*math.sqrt(math.pi)*radius_phot

                    # The frames stacked along the track, for the shape of the object (see extentConcentration).
                    #   Every EXTENT_STEP-th frame is enough; getRectSubPix shifts the patch by the subpixel
                    #   position with bilinear interpolation, so the object is at the centre of the stack
                    for f, xf, yf in zip(fr[::EXTENT_STEP], x[::EXTENT_STEP], y[::EXTENT_STEP]):
                        r = EXTENT_RADIUS
                        if (r <= xf < z.shape[2] - r - 1) and (r <= yf < z.shape[1] - r - 1):
                            if (clouded is not None) and clouded[int(round(yf)), int(round(xf))]:
                                continue
                            c['stack'] += cv2.getRectSubPix(z[f - first], (2*r + 1, 2*r + 1), (float(xf), float(yf)))
                            c['stack_n'] += 1

            self.inThreads(verifyBlock, active)

        # Significance, shape and motion of every track
        results = []
        for cent, c in zip(measured, checks):

            on_gz, on_gg = c['on_sum']
            if (on_gg <= 0) or (c['off_sum'][1] <= 0) or (c['reord_sum'][1] <= 0):
                continue

            # Signal of the moving object: the signal along the track minus the signal of the test scaled to
            #   the same weights. The amplitude of the PSF along the track is A_on = on_gz/on_gg with the variance
            #   1/on_gg, and at the positions of the test A = gz/gg with the variance 1/gg (see
            #   forcedTrackSignal). The significance of their difference is
            #       (A_on - A)/sqrt(1/on_gg + 1/gg) = (on_gz - gz*on_gg/gg)/sqrt(on_gg*(1 + on_gg/gg)),
            #   i.e. the signal-to-noise ratio of the track with the signal of static sources at its positions
            #   subtracted, and the noise of that subtraction added
            tests = []
            for key in ('off_sum', 'reord_sum'):
                gz, gg = c[key]
                tests.append((on_gz - gz*on_gg/gg)/math.sqrt(on_gg*(1 + on_gg/gg)))


            # The mean velocity of the track (px/frame), and the concentration of its stack (see
            #   extentConcentration)
            fr, xs, ys = c['on']
            span = max(fr[-1] - fr[0], 1)
            velocity = np.array([(xs[-1] - xs[0])/span, (ys[-1] - ys[0])/span])
            concentration = extentConcentration(c['stack']/max(c['stack_n'], 1), self.psf_sigma, motion=velocity)
            line = extentLine(c['stack']/max(c['stack_n'], 1), self.psf_sigma, c['stack_n'], motion=velocity)
            results.append((cent, c, min(tests), tests, concentration, velocity, line))

        kept = []
        for i, (cent, c, significance, tests, concentration, velocity, line) in enumerate(results):

            # Whether the track would have been a detection but lies on a thin linear structure
            linear = False

            # A point source: the light of the frames stacked along the track is concentrated at the centre.
            #   Structures of thin clouds drifting across the field are extended, and several of them move
            #   together with the clouds. A very bright object is spread by its own trails and spilled charge, so
            #   it is not tested
            if significance < EXTENT_MAX_SIGNIFICANCE:

                # Count the other tracks which overlap this one in time and have the same velocity (within
                #   COMOVING_TOLERANCE of the speed, at least the slowest speed of the search). A track on the path
                #   of this one is a piece of the same object (e.g. of a bright one, measured in pieces), not
                #   another structure moving with it
                together = 0
                for j, other in enumerate(results):
                    if (j == i) or (other[0][-1, 0] < cent[0, 0]) or (other[0][0, 0] > cent[-1, 0]):
                        continue
                    if np.hypot(*(other[5] - velocity)) <= max(COMOVING_TOLERANCE*np.hypot(*velocity),
                                                                 self.opts.speed_min):
                        if not onPath(other[1]['on'], c['on'], max(SAME_PATH_FWHM*fwhm, 3.0)):
                            together += 1
                # Rejected if the light is much less concentrated than for a point source, or somewhat less but
                #   other tracks move with it
                extended = (concentration < MIN_CONCENTRATION_ALONE) or \
                    ((concentration < MIN_CONCENTRATION) and (together >= COMOVING_MIN))
                if extended:
                    log.info('Matched filter: track {:.0f}-{:.0f} (significance {:.1f}) is extended (concentration '
                             '{:.2f}, {:d} tracks moving with it), rejected'.format(cent[0, 0], cent[-1, 0],
                                                                                   significance, concentration,
                                                                                   together))
                    significance = min(significance, 0.0)

                # A thin linear structure: the light is much less concentrated at the centre than along one
                #   axis through it, and that axis is clearly above the background
                elif (self.opts.line_concentration > 0) and (line[0] < self.opts.line_concentration) and \
                        (line[1] >= LINE_MIN_SIGNIFICANCE):
                    log.info('Matched filter: track {:.0f}-{:.0f} (significance {:.1f}) is on a linear structure '
                             '(concentration {:.2f} along it, {:.1f} sigma), rejected'.format(cent[0, 0],
                                                                                            cent[-1, 0],
                                                                                            significance,
                                                                                            *line))
                    linear = significance >= self.opts.track_significance
                    significance = min(significance, 0.0)

            # The significance of every candidate is kept for the done file, and the photometry of the kept ones
            #   for the conversion to detections (see run)
            self.significances.append((cent[0, 0], cent[-1, 0], significance) + tuple(tests) + (concentration, linear))
            log.debug('Matched filter: track {:.0f}-{:.0f} significance {:.1f} (off time {:.1f}, reordered '
                      '{:.1f})'.format(cent[0, 0], cent[-1, 0], significance, *tests))
            if significance >= self.opts.track_significance:
                kept.append(cent)
                self.photometry[id(cent)] = (c['on'][0], c['phot'], c['phot_sat'], c['phot_noise'])

        return kept



def nanMedian(stack):
    """ Median along the first axis, ignoring NaNs (NaN where all values are NaN). The same as np.nanmedian,
        which is many times slower on large arrays.

    Arguments:
        stack: [ndarray] Values, (n, ...).

    Return:
        [ndarray] Median, stack.shape[1:].
    """

    # np.sort puts the NaNs at the end, so the n finite values of every pixel are the first n of the sorted
    #   stack, and their median is the mean of the values at the indices (n - 1)//2 and n//2 (the same value
    #   for an odd n)
    srt = np.sort(stack, axis=0)
    n = np.isfinite(stack).sum(axis=0)
    lo = np.take_along_axis(srt, np.maximum((n - 1)//2, 0)[None], axis=0)[0]
    hi = np.take_along_axis(srt, np.maximum(n//2, 0)[None], axis=0)[0]
    med = (lo + hi)/2

    return np.where(n > 0, med, np.nan)


def blockIndices(frames, starts, end=None):
    """ Indices of the frames in every block of frames.

    Arguments:
        frames: [ndarray] Frame numbers.
        starts: [ndarray] First frames of the blocks, sorted.

    Keyword arguments:
        end: [int] End of the last block (the frame after it). The frames from it on are in no block. None by
            default (the last block has all the frames after its beginning).

    Return:
        [dict] Block index -> indices of its frames, in their order in frames.
    """

    # The block of every frame is the last block which begins at or before it (-1 before the first block and
    #   after the end of the last one)
    blocks = np.searchsorted(starts, frames, side='right') - 1
    if end is not None:
        blocks[np.asarray(frames) >= end] = -1

    # The frames sorted by block (a stable sort keeps their order within a block), and the range of every block
    #   in the sorted order
    order = np.argsort(blocks, kind='stable')
    sorted_blocks = blocks[order]
    out = {}
    for k in np.unique(sorted_blocks):
        if k < 0:
            continue
        lo, hi = np.searchsorted(sorted_blocks, [k, k + 1])
        out[int(k)] = order[lo:hi]

    return out


def onPath(track, path, radius):
    """ Whether two tracks lie on the same path: the median distance of the positions of the shorter one from
        the nearest positions of the longer one is within radius (symmetric). A piece of the same object is on
        its path, also when it is ahead of or behind the object along it.

    Arguments:
        track: [tuple] (frames, xs, ys) of the track.
        path: [tuple] (frames, xs, ys) of the other track, one position per frame.
        radius: [float] Largest median distance (px).

    Return:
        [bool] True if the track is on the path.
    """

    # The shorter track is compared with the path of the longer one (the longer one extends beyond a piece)
    if len(track[1]) > len(path[1]):
        track, path = path, track
    _, x, y = track
    _, px, py = path
    if (len(x) == 0) or (len(px) == 0):
        return False

    # The positions of the track (subsampled, enough for the median)
    step = max(1, len(x)//50)
    qx, qy = x[::step, None], y[::step, None]

    # A path of one position: the distance from it
    if len(px) < 2:
        d = np.hypot(qx[:, 0] - px[0], qy[:, 0] - py[0])
        return float(np.median(d)) <= radius

    # The distance of every position from the nearest segment between consecutive positions of the path (the
    #   positions of the path are a frame apart, which can be many px)
    ax, ay = px[None, :-1], py[None, :-1]
    sx, sy = np.diff(px)[None, :], np.diff(py)[None, :]
    length2 = np.maximum(sx*sx + sy*sy, 1e-12)
    t = np.clip(((qx - ax)*sx + (qy - ay)*sy)/length2, 0.0, 1.0)
    d = np.min(np.hypot(qx - (ax + t*sx), qy - (ay + t*sy)), axis=1)
    return float(np.median(d)) <= radius


def streakMask(r, psf_sigma, motion):
    """ The pixels of a stack along a track which the object lights during a frame: within 2 PSF sigmas of the
        segment it moves along in a frame, centred on it. Only an object which moves far enough during a frame to
        reach the ring of the shape tests (see extentConcentration) has a streak.

    Arguments:
        r: [int] Half size of the stack (px).
        psf_sigma: [float] PSF sigma (px).
        motion: [tuple] Motion of the object (px per frame), or None.

    Return:
        [ndarray] Boolean mask of the stack, or None if the object has no streak reaching the ring.
    """

    # Without the motion, or if the streak of a frame doesn't reach the ring, there is nothing to leave out
    if motion is None:
        return None
    speed = math.hypot(*motion)
    if 0.5*speed + 2*psf_sigma < EXTENT_RING[0]:
        return None

    # The distance of every pixel from the segment of the frame along the motion
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    ux, uy = motion[0]/speed, motion[1]/speed
    t = np.clip(xx*ux + yy*uy, -0.5*speed, 0.5*speed)
    return np.hypot(xx - t*ux, yy - t*uy) <= 2*psf_sigma


def extentConcentration(stack, psf_sigma, motion=None):
    """ How much of the light of an object is at its centre: the mean of the stack along its track within
        1.5 PSF sigmas of the centre, minus the mean in a ring EXTENT_RING px from it, over the mean at the
        centre. About 1 for a point source, near 0 for an extended structure (e.g. a cloud).

    Arguments:
        stack: [ndarray] Mean of the normalized frames along the track, centred on the object.
        psf_sigma: [float] PSF sigma (px).

    Keyword arguments:
        motion: [tuple] Motion of the object (px per frame). An object which moves far enough during a frame is a
            streak in every frame, which reaches into the ring along its motion: the pixels of the streak are
            then left out of the ring (see streakMask). None by default (the whole ring is used).

    Return:
        [float] Concentration (1 if the centre is not above the ring, as the shape is then not measurable).
    """

    # The distance of every pixel of the stack from its centre
    r = (stack.shape[0] - 1)//2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.hypot(xx, yy)

    # The ring, without the streak of a fast object, which is the object's own light (unless the streak covers
    #   all of it, e.g. for a very wide PSF)
    ring = (d >= EXTENT_RING[0]) & (d <= EXTENT_RING[1])
    streak = streakMask(r, psf_sigma, motion)
    if (streak is not None) and np.any(ring & ~streak):
        ring &= ~streak

    # The mean of the core (within 1.5 PSF sigmas, at least 1 px) and of the ring. The PSF of a point source is
    #   negligible in the ring, so its ring is at the level of the background (0 in normalized frames) and
    #   the concentration is ~1. An extended structure is as bright in the ring as in the core, giving ~0
    core = np.mean(stack[d <= max(1.5*psf_sigma, 1.0)])
    ring = np.mean(stack[ring])
    if core <= 0:
        return 1.0

    return float((core - ring)/core)


def extentLine(stack, psf_sigma, n_frames, motion=None):
    """ How much of the light of an object is at its centre compared with the brightest axis through it: the
        mean of the stack along its track within 1.5 PSF sigmas of the centre, minus the mean of the ring
        EXTENT_RING px from it along the brightest of LINE_ANGLES axes (within LINE_HALF_WIDTH px of the axis),
        over the mean at the centre. About 1 for a point source, low for an object on a thin linear structure,
        which lights the ring only along its own direction (and barely changes the mean of the whole ring, see
        extentConcentration).

    Arguments:
        stack: [ndarray] Mean of the normalized frames along the track, centred on the object.
        psf_sigma: [float] PSF sigma (px).
        n_frames: [int] Number of frames in the stack.

    Keyword arguments:
        motion: [tuple] Motion of the object (px per frame). An object which moves far enough during a frame is a
            streak in every frame, which reaches into the ring along its motion: the axes along the streak are
            then left out. None by default (all axes are used).

    Return:
        (concentration, significance): [tuple of floats] The concentration (1 if the centre is not above the
            axis, as the shape is then not measurable) and the significance of the brightest axis above the
            background (sigma).
    """

    # The distance of every pixel of the stack from its centre, and the pixels of the ring
    r = (stack.shape[0] - 1)//2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.hypot(xx, yy)
    ring = (d >= EXTENT_RING[0]) & (d <= EXTENT_RING[1])

    # The streak of the object in a frame, if it reaches the ring (see streakMask)
    streak = streakMask(r, psf_sigma, motion)

    # The mean of the ring along every axis: the pixels of the ring close to the line through the centre in
    #   that direction (on both sides of the centre). The axes which are mostly on the streak are left out
    axes = []
    for angle in np.arange(LINE_ANGLES)*np.pi/LINE_ANGLES:
        axis = ring & (np.abs(yy*math.cos(angle) - xx*math.sin(angle)) <= LINE_HALF_WIDTH)
        if (streak is not None) and (np.sum(axis & streak) > 0.25*np.sum(axis)):
            continue
        axes.append((float(np.mean(stack[axis])), int(np.sum(axis))))
    if not axes:
        return 1.0, 0.0
    axis_mean, axis_n = max(axes, key=lambda v: v[0])

    # The significance of the brightest axis. The normalized frames have unit noise, so the noise of a pixel of
    #   the stack is 1/sqrt(n_frames), and of the mean of an axis that over the square root of its pixels
    significance = axis_mean*math.sqrt(max(n_frames, 1)*axis_n)

    # The mean of the core (within 1.5 PSF sigmas, at least 1 px), compared with the brightest axis
    core = np.mean(stack[d <= max(1.5*psf_sigma, 1.0)])
    if core <= 0:
        return 1.0, float(significance)

    return float((core - axis_mean)/core), float(significance)


def framePhotometry(frames, saturated, frame_idx, noise, xs, ys, vx, vy, radius):
    """ Photometry of a moving object in a few frames: the sum within radius of the segment it moved along in
        every frame, and its saturated pixels. A very bright object spills its charge along its row and column
        before its pixels saturate: in a frame where the sum within SPILL_RADIUS is more than SPILL_RATIO times
        the normal sum, the wider sum is the intensity and the frame counts as saturated.

    Arguments:
        frames: [ndarray] Normalized frames (n, height, width), e.g. a block.
        saturated: [ndarray] uint8 mask of the saturated pixels of the frames.
        frame_idx: [ndarray] Indices of the frames of the object in frames (the frames are not copied).
        noise: [ndarray] Noise image (ADU) of the normalization.
        xs, ys: [ndarray] Positions of the object in the middle of the frames.
        vx, vy: [float] Velocity (px per frame).
        radius: [float] Radius of the aperture (px).

    Return:
        (sums, n_saturated): [tuple of ndarrays] Intensity (ADU) and saturated pixel count per frame.
    """

    # The sums within the aperture along the segment of every frame (see streakAperture)
    sums, n_sat = streakAperture(frames, saturated, frame_idx, noise, xs, ys, vx, vy, radius)

    # The noise of the sum, at the middle of the positions: the noise of a pixel times the square root of the
    #   number of pixels in the aperture
    xi = int(min(max(round(float(np.median(xs))), 0), noise.shape[1] - 1))
    yi = int(min(max(round(float(np.median(ys))), 0), noise.shape[0] - 1))
    core_noise = float(noise[yi, xi])*math.sqrt(math.pi)*radius

    # The spilled charge is only looked for in the frames of very bright objects (the wide aperture adds a lot of
    #   noise to faint ones)
    if np.max(sums) > SPILL_MIN_SNR*core_noise:
        wide, wide_sat = streakAperture(frames, saturated, frame_idx, noise, xs, ys, vx, vy, SPILL_RADIUS)
        spill = (sums > SPILL_MIN_SNR*core_noise) & (wide > SPILL_RATIO*sums)
        sums = np.where(spill, wide, sums)
        n_sat = np.where(spill, np.maximum(wide_sat, 1), n_sat)

    return sums, n_sat


def perFrameRows(cent, max_frames, phot=None):
    """ Rows of every frame of a track from its measurements: the measurements of faint objects combine several
        frames, but the intensity of each frame is kept, so the light curve has the full time resolution (e.g.
        short flashes are not averaged out). The positions of the frames are interpolated along the track
        through the (smoothed) positions of the measurements, and extrapolated with the motion at its ends.

    Arguments:
        cent: [ndarray] Measurements, rows [frame, x, y, intensity, background, snr, saturated, n_frames,
            intensities of the frames (max_frames columns), saturated pixel counts (max_frames columns)], sorted
            by frame. The frame of a measurement is the middle of its frames.
        max_frames: [int] Largest number of frames of a measurement.

    Keyword arguments:
        phot: [tuple] (frames, intensities, saturated pixel counts, noise of the intensities) of every frame of
            the track (see verify). With it, there is a row for every one of these frames (also the frames whose
            position was not measured), with this photometry. None by default.

    Return:
        [ndarray] Rows [frame, x, y, intensity, background, snr, saturated], one per frame. Without phot, the
            signal-to-noise ratio of a frame is the one of its measurement divided by the square root of its
            number of frames.
    """

    # The velocities at the two ends of the track, from the two first and the two last measurements, for the
    #   extrapolation of the frames beyond the first and the last measurement
    t, x, y = cent[:, 0], cent[:, 1], cent[:, 2]
    if len(cent) >= 2:
        vx_lo, vy_lo = (x[1] - x[0])/(t[1] - t[0]), (y[1] - y[0])/(t[1] - t[0])
        vx_hi, vy_hi = (x[-1] - x[-2])/(t[-1] - t[-2]), (y[-1] - y[-2])/(t[-1] - t[-2])
    else:
        vx_lo = vy_lo = vx_hi = vy_hi = 0.0

    def position(f):
        """ Position at the frame f: interpolated between the measurements, extrapolated beyond them. """

        if f < t[0]:
            return x[0] + vx_lo*(f - t[0]), y[0] + vy_lo*(f - t[0])
        if f > t[-1]:
            return x[-1] + vx_hi*(f - t[-1]), y[-1] + vy_hi*(f - t[-1])
        return np.interp(f, t, x), np.interp(f, t, y)

    # With the photometry of every frame: a row for every frame, with the background interpolated between the
    #   measurements and the signal-to-noise ratio of the intensity of the frame
    if phot is not None:
        frames, sums, n_sat, noise = phot
        rows = []
        for f, s, ns, nz in zip(frames, sums, n_sat, noise):
            xf, yf = position(f)
            rows.append([f, xf, yf, s, np.interp(f, t, cent[:, 4]), s/max(nz, 1e-6), ns])
        return np.array(rows)

    # Without it: a row for every frame of every measurement, with the intensity of the frame stored with the
    #   measurement. The n frames of a measurement are centred on its frame. The signal-to-noise ratio of a
    #   measurement of n frames is sqrt(n) times the one of a frame (for a constant intensity)
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


def curvatureSignificant(t, x, y, f_min=CURVATURE_MIN_F):
    """ Whether a track is clearly curved in time (its position is not linear in time): an F-test of a quadratic
        against a line, fitted to x(t) and y(t) together. The noise of the measurements is taken from the
        residuals of the quadratic, so the test needs no position errors, and only a curvature well above the
        scatter of the measurements passes.

    Arguments:
        t: [ndarray] Times of the measurements (frames).
        x, y: [ndarray] Positions of the measurements (px).

    Keyword arguments:
        f_min: [float] Smallest F statistic of a significant curvature. CURVATURE_MIN_F by default.

    Return:
        [bool] True if the quadratic is significantly better than the line.
    """

    # The quadratic has 2 more parameters than the line (one per axis), and leaves 2n - 6 degrees of freedom
    n = len(t)
    if n < 6:
        return False

    # The residual sums of squares of the line and of the quadratic, of both axes. The time is centred for a
    #   well-conditioned fit
    tc = t - t.mean()
    rss = []
    for deg in (1, 2):
        rss.append(sum(float(np.sum((v - np.polyval(np.polyfit(tc, v, deg), tc))**2)) for v in (x, y)))
    rss_line, rss_quad = rss

    # The improvement of the fit per added parameter, in units of the scatter left by the quadratic. A tiny
    #   floor keeps exact (noise-free) positions from dividing by zero
    scatter = max(rss_quad/(2*n - 6), 1e-12)
    return (rss_line - rss_quad)/2.0/scatter > f_min


def removeStreakDuplicates(detections, fwhm, connected=None):
    """ Remove the second measurements of fast objects. The light of an object moving several pixels per frame is
        a streak, and the object can be measured twice in it, e.g. at its middle and at its leading end, which gives
        two detections moving together, the length of the streak apart (often also across it, as the leading end
        of a bright streak is wide). They are further apart than the duplicates of removeDuplicateDetections, but
        they are one object: they move rigidly together and have the same light curve, which two objects don't.
        Objects of steady brightness have flat light curves, which can't tell them apart, so then the light
        also has to be connected between the two detections (the streak). The longer detection is kept (see
        STREAK_DUPLICATE_SHARED and the following constants).

    Arguments:
        detections: [list] Detections as [rho, theta, centroids], centroids rows [frame, x, y, intensity, ...].
        fwhm: [float] The PSF width (FWHM) in the pixels of the centroids.

    Keyword arguments:
        connected: [function] connected(frame, x1, y1, x2, y2) gives the fraction of the segment between two
            points which is lit (None if unknown), see MatchedFilterDetector.connectedLight. None by default, in
            which case detections with flat light curves are kept.

    Return:
        [list] Detections without the second measurements, in the original order.
    """

    # Longest first, so the second measurement is removed in favour of the longer detection
    order = sorted(range(len(detections)), key=lambda i: -len(detections[i][2]))

    kept = []
    for i in order:

        cent = np.asarray(detections[i][2])
        duplicate = False

        for j in kept:

            ref = np.asarray(detections[j][2])

            # The shared frames
            _, a, b = np.intersect1d(np.round(cent[:, 0], 1), np.round(ref[:, 0], 1), return_indices=True)
            if len(a) < max(3, STREAK_DUPLICATE_SHARED*len(cent)):
                continue

            # Only a fast object has a streak: the speed of the longer detection (px per frame)
            if np.ptp(ref[:, 0]) <= 0:
                continue
            speed = math.hypot(np.polyfit(ref[:, 0], ref[:, 1], 1)[0], np.polyfit(ref[:, 0], ref[:, 2], 1)[0])
            if speed < fwhm:
                continue

            # A rigid offset within the streak
            d = cent[a, 1:3] - ref[b, 1:3]
            offset = np.median(d, axis=0)
            if math.hypot(*offset) > speed + STREAK_DUPLICATE_FWHM*fwhm:
                continue
            if math.hypot(*(1.4826*np.median(np.abs(d - offset), axis=0))) > fwhm:
                continue

            # The same light curve; flat light curves only with the light connected between the detections in
            #   most of the sampled shared frames
            log_cent = np.log10(np.maximum(cent[a, 3], 1e-6))
            log_ref = np.log10(np.maximum(ref[b, 3], 1e-6))
            if max(np.std(log_cent), np.std(log_ref)) < STREAK_DUPLICATE_FLAT:
                if connected is not None:
                    lit = []
                    for k in np.unique(np.linspace(0, len(a) - 1, 5).astype(int)):
                        frac = connected(cent[a[k], 0], cent[a[k], 1], cent[a[k], 2], ref[b[k], 1], ref[b[k], 2])
                        if frac is not None:
                            lit.append(frac >= 0.9)
                    duplicate = (len(lit) > 0) and (np.mean(lit) >= 0.8)
            elif min(np.std(log_cent), np.std(log_ref)) > 0:
                duplicate = np.corrcoef(log_cent, log_ref)[0, 1] >= STREAK_DUPLICATE_CORRELATION

            if duplicate:
                break

        if not duplicate:
            kept.append(i)

    if len(kept) < len(detections):
        log.info('Matched filter: {:d} second measurements of fast objects removed'.format(len(detections)
                                                                                           - len(kept)))

    return [detections[i] for i in sorted(kept)]


def cleanTrackHits(hits, min_kept=0.5, min_hits=3):
    """ The hits of a track without the ones whose velocity disagrees with the motion of the track. The velocity
        of a hit is the velocity of the search whose sum was the largest there. A very bright object makes the
        sums of many other velocities large too, at positions near its track (they pass over it in some frames),
        and such hits can be linked to the track by their position, where they pull the motion model of the
        measurement away from the object.

    A hit is kept if its velocity is within 30% of the speed of the track, or within 3 steps of the velocity grid
    of its search (bin/(run - 1) px per frame) for slow tracks. The velocity of the track is the median of the
    velocities of its hits weighted by their significance: the hits of other velocities can be as many as the hits
    of the object, but they collect only part of its light. A track whose hits mostly disagree (by their weight)
    is kept as it is (e.g. a track whose speed changes a lot), so a track is never cut down to a few hits.

    Arguments:
        hits: [ndarray] Hits of the track, rows [frame, x, y, z, vx, vy, run, bin].

    Keyword arguments:
        min_kept: [float] Smallest fraction of the weight of the hits which has to be kept. 0.5 by default.
        min_hits: [int] Smallest number of hits of a track (at least 3): a track is not cut below it. 3 by
            default.

    Return:
        [ndarray] The kept hits.
    """

    # Too few hits to judge them, or hits without the velocity and the bin of their search
    if (len(hits) < 6) or (hits.shape[1] < 8):
        return hits

    # The motion of the track: the median velocity of its hits, weighted by their significance
    weights = np.maximum(hits[:, 3], 0.0)
    if weights.sum() <= 0:
        return hits
    v = np.array([weightedMedian(hits[:, 4], weights), weightedMedian(hits[:, 5], weights)])
    speed = math.hypot(*v)

    # The tolerance of every hit: 30% of the speed, at least 3 steps of the velocity grid of its own search
    step = hits[:, 7]/np.maximum(hits[:, 6] - 1, 1)
    tolerance = np.maximum(0.3*speed, 3.0*step)
    keep = np.hypot(hits[:, 4] - v[0], hits[:, 5] - v[1]) <= tolerance

    # A track which would keep too few hits, or less than min_kept of their weight, is kept as it is
    if (keep.sum() < max(3, min_hits)) or (weights[keep].sum() < min_kept*weights.sum()):
        return hits

    return hits[keep]


def linkFaintHits(hits, seed_frames=FAINT_SEED_FRAMES, seed_radius=FAINT_SEED_RADIUS,
                  seed_spread=FAINT_SEED_SPREAD, line_radius=FAINT_LINE_RADIUS, velocity_steps=FAINT_VELOCITY_STEPS,
                  min_runs=FAINT_MIN_RUNS, min_span=FAINT_MIN_SPAN):
    """ Link the hits of faint objects into chains along straight tracks. A faint object is above the threshold of
        its hits only in some runs (where the noise adds to it), with gaps of many runs between them, so its hits
        are linked along a line over long times instead of from run to run (see MatchedFilterDetector.linkHits).
        A chain starts from a pair of hits whose positions and velocities agree, and takes the hits along the line
        through its hits until they don't change. Most pairs are of the noise and find too few hits along their
        line, so the hits along the line of every pair are first counted for all pairs at once, and only the pairs
        with enough of them are grown into chains, the strongest first.

    The velocity of a hit is known to about a step of the velocity grid of its search (bin/(run - 1) px per frame),
    so the velocities are compared within velocity_steps such steps.

    Arguments:
        hits: [ndarray] Hits as rows [frame, x, y, z, vx, vy, run, bin].

    Keyword arguments:
        seed_frames: [tuple] Smallest and largest time between the two hits of a pair (frames), also the reach of a
            chain beyond its ends. FAINT_SEED_FRAMES by default.
        seed_radius: [float] Largest distance of a hit of a pair from the position predicted from the other one
            with its velocity (px), at the shortest time between them. FAINT_SEED_RADIUS by default.
        seed_spread: [float] Growth of that distance with the time between the hits (px per frame), from the error
            of the velocities of the hits. FAINT_SEED_SPREAD by default.
        line_radius: [float] Largest distance of a hit of a chain from its line (px). FAINT_LINE_RADIUS by default.
        velocity_steps: [float] Largest difference of the velocities of the hits of a pair, and of the velocity of
            a hit of a chain from the motion along its line, in steps of the velocity grid of the hit.
            FAINT_VELOCITY_STEPS by default.
        min_runs: [int] Smallest number of runs with a hit of a chain. FAINT_MIN_RUNS by default.
        min_span: [float] Smallest time from the first to the last hit of a chain (frames). FAINT_MIN_SPAN by
            default.

    Return:
        [list] Chains, each an array of hits sorted by frame, at most one hit per run.
    """

    # Too few hits for any chain
    if len(hits) < min_runs:
        return []

    # The hits by the middle frame of their run (the runs of every length have their own middle frames), with a
    #   spatial index of every run, and the largest velocity difference of every hit
    times, inverse = np.unique(hits[:, 0], return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    members = np.split(order, np.cumsum(np.bincount(inverse, minlength=len(times)))[:-1])
    trees = [cKDTree(hits[m, 1:3]) for m in members]
    dv_max = velocity_steps*hits[:, 7]/np.maximum(hits[:, 6] - 1, 1)

    ### Pairs of hits ###

    # For every run, the hits of the runs seed_frames before it, at the positions predicted with their velocity,
    #   are looked up all at once within the largest distance, and the nearest few hits taken (more than a few are
    #   rare away from bright objects)
    pairs = []
    for kb in range(len(times)):
        dt_runs = times[kb] - times
        earlier = np.nonzero((dt_runs >= seed_frames[0]) & (dt_runs <= seed_frames[1]))[0]
        if not len(earlier):
            continue
        a = np.concatenate([members[k] for k in earlier])
        dt = times[kb] - hits[a, 0]
        px = hits[a, 1] + hits[a, 4]*dt
        py = hits[a, 2] + hits[a, 5]*dt
        dist, near = trees[kb].query(np.column_stack([px, py]), k=8,
                                     distance_upper_bound=seed_radius + seed_spread*seed_frames[1])
        rows, cols = np.nonzero(np.isfinite(dist))
        a, b, dt = a[rows], members[kb][near[rows, cols]], dt[rows]

        # Each hit within the distance of the time between them from the position predicted from the other one
        #   (so the line through them agrees with the velocities of both), and their velocities alike
        reach = seed_radius + seed_spread*dt
        back_x, back_y = hits[b, 1] - hits[b, 4]*dt, hits[b, 2] - hits[b, 5]*dt
        ok = (np.hypot(hits[b, 1] - px[rows], hits[b, 2] - py[rows]) <= reach) \
            & (np.hypot(hits[a, 1] - back_x, hits[a, 2] - back_y) <= reach) \
            & (np.hypot(hits[b, 4] - hits[a, 4], hits[b, 5] - hits[a, 5]) <= np.maximum(dv_max[a], dv_max[b]))
        pairs.append(np.column_stack([a[ok], b[ok]]))

    # The pairs of all runs, no chains without them
    pairs = np.vstack(pairs) if pairs else np.zeros((0, 2), dtype=int)
    if not len(pairs):
        return []

    ### ###

    ### Hits along the line of every pair ###

    # The line through the two hits of every pair, and the runs it reaches (seed_frames[1] beyond the hits)
    a, b = pairs[:, 0], pairs[:, 1]
    span = hits[b, 0] - hits[a, 0]
    vx = (hits[b, 1] - hits[a, 1])/span
    vy = (hits[b, 2] - hits[a, 2])/span
    lo = hits[a, 0] - seed_frames[1]
    hi = hits[b, 0] + seed_frames[1]

    # The runs with a hit near the line of every pair (with a velocity like the motion along it), their number,
    #   the first and the last such run, and the sum of the significances of the hits (the nearest in every run)
    n_runs = np.zeros(len(pairs), dtype=int)
    first = np.full(len(pairs), np.inf)
    last = np.full(len(pairs), -np.inf)
    strength = np.zeros(len(pairs))
    for k, t in enumerate(times):
        sel = np.nonzero((lo <= t) & (hi >= t))[0]
        if not len(sel):
            continue
        px = hits[a[sel], 1] + vx[sel]*(t - hits[a[sel], 0])
        py = hits[a[sel], 2] + vy[sel]*(t - hits[a[sel], 0])
        dist, near = trees[k].query(np.column_stack([px, py]), k=4, distance_upper_bound=line_radius)
        rows, cols = np.nonzero(np.isfinite(dist))
        j = members[k][near[rows, cols]]
        ok = np.hypot(hits[j, 4] - vx[sel[rows]], hits[j, 5] - vy[sel[rows]]) <= dv_max[j]

        # Every pair counts the run once (the look-up gives the nearest hits first)
        rows, j = rows[ok], j[ok]
        rows, keep = np.unique(rows, return_index=True)
        sel, j = sel[rows], j[keep]
        n_runs[sel] += 1
        first[sel] = np.minimum(first[sel], t)
        last[sel] = np.maximum(last[sel], t)
        strength[sel] += hits[j, 3]

    # Only the pairs with enough runs along their line are grown into chains, the strongest first
    candidates = np.nonzero((n_runs >= min_runs) & (last - first >= min_span))[0]
    candidates = candidates[np.argsort(-strength[candidates], kind='stable')]

    ### ###

    ### Chains ###

    # One index of all hits, with the run as a coordinate far larger than the radius (the hits of other runs are
    #   never near), for the look-up of the hits along a line in all runs at once
    run_step = 10.0*line_radius
    tree = cKDTree(np.column_stack([run_step*inverse, hits[:, 1], hits[:, 2]]))
    used = np.zeros(len(hits), dtype=bool)
    chains = []
    for c in candidates:

        # The hits of a chain are in one chain only
        if used[a[c]] or used[b[c]]:
            continue

        # The chain grows from its pair: the hits near its line in the runs it reaches (the nearest one of every
        #   run), until they don't change (at most 4 times)
        chain = np.array([a[c], b[c]])
        for _ in range(4):

            # The line fitted through the hits of the chain, and the runs it reaches
            t = hits[chain, 0]
            fit_x = np.polyfit(t, hits[chain, 1], 1)
            fit_y = np.polyfit(t, hits[chain, 2], 1)
            ks = np.nonzero((times >= t.min() - seed_frames[1]) & (times <= t.max() + seed_frames[1]))[0]
            px, py = np.polyval(fit_x, times[ks]), np.polyval(fit_y, times[ks])
            dist, near = tree.query(np.column_stack([run_step*ks, px, py]), k=4, distance_upper_bound=line_radius)

            # The hits within both tolerances which are in no other chain, the nearest of every run
            rows, cols = np.nonzero(np.isfinite(dist))
            j = near[rows, cols]
            ok = (~used[j]) & (np.hypot(hits[j, 4] - fit_x[0], hits[j, 5] - fit_y[0]) <= dv_max[j])
            j = j[ok]
            j = j[np.unique(inverse[j], return_index=True)[1]]

            # A chain which lost its hits keeps the ones it had
            if len(j) < 2:
                break
            if np.array_equal(np.sort(j), np.sort(chain)):
                break
            chain = j

        # A chain needs hits in enough runs over a long enough time
        t = hits[chain, 0]
        if (len(chain) < min_runs) or (np.ptp(t) < min_span):
            continue
        used[chain] = True
        chains.append(hits[chain[np.argsort(t, kind='stable')]])

    ### ###

    return chains


def weightedMedian(values, weights):
    """ The median of the values weighted by the weights: the value at which half of the weight is reached. """

    # The values sorted, and the first one at which their cumulative weight reaches half of the total
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5*cumulative[-1])])


def trimFaintEnds(cent, fraction=FAINT_END_FRACTION, window=FAINT_END_WINDOW):
    """ Cut the ends of a track where it is much fainter than along the rest of it: the leading and the trailing
        rows in whose window (of window rows) at most FAINT_END_BRIGHT_ROWS of the rows reach fraction of the
        FAINT_END_PERCENTILE percentile of the SNR of the track (the level of its bright part, also when a faint
        piece is most of the track; saturated rows count as the brightest), and at most FAINT_END_MAX_SNR (an end at
        the level of the noise). Saturated rows are always bright (their SNR is not a measure of their
        brightness). A track
        which is faint all along (e.g. near the limit) is kept as it is, as only the ends fainter than the track
        itself are cut, and so is a track which is bright only in rare flashes.

    Arguments:
        cent: [ndarray] Rows of a track, one per frame, [frame, x, y, intensity, background, snr, saturated, ...],
            sorted by frame.

    Keyword arguments:
        fraction: [float] Fraction of the FAINT_END_PERCENTILE percentile of the SNR of the track.
            FAINT_END_FRACTION by default.
        window: [int] Number of rows of the window around a row. FAINT_END_WINDOW by default.

    Return:
        [ndarray] The rows which are kept.
    """

    if len(cent) < 2*window:
        return cent

    # A row is in a bright part of the track if more than FAINT_END_BRIGHT_ROWS of the rows of the window around
    #   it reach the level: a track faint between flashes (e.g. a tumbling object) is bright where it flashes, while
    #   a single noise row is not. The windows keep their length at the ends of the track (they are moved inside
    #   it), so a single row there doesn't count more than elsewhere
    snr = np.asarray(cent[:, 5], dtype=np.float64)
    saturated = (np.asarray(cent[:, 6], dtype=np.float64) > 0) if cent.shape[1] > 6 else np.zeros(len(snr), bool)
    half = window//2

    # The level from a row of the track (not interpolated between rows), so a bright part which is just a tenth of
    #   the track sets it, but never above the level of the noise. The saturated rows count as brighter than any
    #   other (their SNR is not a measure of their brightness): a track saturated in a tenth of its rows has a
    #   bright part whatever its other rows are (they may be only a continuation over empty sky)
    ref = np.sort(np.where(saturated, np.inf, snr))
    level = min(fraction*ref[int(math.ceil(FAINT_END_PERCENTILE/100.0*(len(ref) - 1)))], FAINT_END_MAX_SNR)
    above = (snr >= level) | saturated
    starts = np.clip(np.arange(len(snr)) - half, 0, len(snr) - window)
    bright = np.array([np.mean(above[k:k + window]) > FAINT_END_BRIGHT_ROWS for k in starts])

    # Without a bright part (e.g. rare flashes), nothing is an end of the track fainter than the rest of it
    if not bright.any():
        return cent

    # The rows from the first to the last bright one
    idx = np.nonzero(bright)[0]

    return cent[idx[0]:idx[-1] + 1]


def smoothTrack(cent, frames, segment=512):
    """ Smooth positions of a track at the given frames: a quadratic fit of the measurements in every segment
        of the track (lines for short segments, unless their curvature is clearly measured). The model is stiff,
        so it cannot follow the noise.

    Arguments:
        cent: [ndarray] Centroids, rows [frame, x, y, ...], sorted by frame.
        frames: [ndarray] Frames at which the positions are needed.

    Keyword arguments:
        segment: [int] Longest segment of the track with one quadratic (frames). 512 by default.

    Return:
        (xs, ys): [tuple of ndarrays] Positions.
    """

    # The track is split into segments of equal length of at most segment frames
    t = cent[:, 0]
    n_seg = max(1, int(math.ceil((t[-1] - t[0] + 1)/segment)))
    edges = np.linspace(t[0], t[-1] + 1e-6, n_seg + 1)

    xs = np.empty(len(frames))
    ys = np.empty(len(frames))
    for i in range(n_seg):
        sel = (t >= edges[i]) & (t <= edges[i + 1])

        # Frames before the first and after the last measurement are extrapolated from the end segments
        fsel = ((frames >= edges[i]) | (i == 0)) & ((frames <= edges[i + 1]) | (i == n_seg - 1))
        # A segment with fewer than 2 measurements uses the 2 measurements closest to its middle
        if sel.sum() < 2:
            sel = np.argsort(np.abs(t - (edges[i] + edges[i + 1])/2))[:2]
        tt = t[sel]

        # A quadratic needs enough measurements over a long enough time (otherwise its curvature fits the
        #   noise): a line for short segments, a constant if all measurements are in the same frame. A short
        #   segment is quadratic too if its curvature is clearly measured (a bright fast object whose apparent
        #   speed changes, which a line misses by several px in the middle and at the ends)
        if (len(tt) >= 6) and (np.ptp(tt) > 128):
            deg = 2
        elif np.ptp(tt) > 0:
            deg = 2 if curvatureSignificant(tt, cent[sel, 1], cent[sel, 2]) else 1
        else:
            deg = 0

        # The time is centred on the segment, for a well-conditioned fit
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

    # Weights of the fit: the inverse of the position errors, which scale as 1/snr (np.polyfit multiplies the
    #   residuals by the weights, so they are 1/sigma and not 1/sigma^2)
    w = np.maximum(cent[:, 5], 0.5)
    keep = np.ones(len(t), dtype=bool)

    def _smooth(use):
        """ Smoothed positions of all measurements, from the fits to the measurements selected by use. """

        tu, xu, yu, wu = t[use], cent[use, 1], cent[use, 2], w[use]

        # Measurements without enough neighbours keep their own positions
        out = np.column_stack([cent[:, 1], cent[:, 2]]).astype(np.float64)

        # The window keeps its width near the ends of the track, shifted to stay inside it (a window cut at the
        #   end would fit fewer measurements and extrapolate the quadratic)
        start = np.clip(t - half_window, tu[0], max(tu[0], tu[-1] - 2*half_window))
        lo = np.searchsorted(tu, start, side='left')
        hi = np.searchsorted(tu, start + 2*half_window, side='right')
        for i in range(len(t)):

            # At least 4 measurements in the window
            n = hi[i] - lo[i]
            if n < 4:
                continue

            # The time is relative to the measurement, so the constant term of the polynomial (the last
            #   coefficient of np.polyfit) is the smoothed position at the measurement. A line with fewer than 6
            #   measurements, as a quadratic would fit their noise
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

        # The robust scatter of the distances (the MAD about 0, scaled to a standard deviation), and the
        #   outliers beyond reject times it; the smoothing is repeated until the outliers don't change
        scale = 1.4826*np.median(d[keep]) + 1e-3
        new_keep = d <= reject*scale
        if (new_keep == keep).all() or (new_keep.sum() < 4):
            break
        keep = new_keep
        xy = _smooth(keep)

    return xy[keep], keep



class TrackMeasurement(object):
    def __init__(self, hits, opts, psf_sigma, n_frames=None, window=None, extend=None):
        """ Measurement of one track: the runs of frames to fit, the motion model, and the centroids.

        Arguments:
            hits: [ndarray] Hits of the track, rows [frame, x, y, z, vx, vy, run].
            opts: [MatchedFilterOptions] Options.
            psf_sigma: [float] PSF sigma (px).

        Keyword arguments:
            n_frames: [int] Frames per measurement, a power of two. None by default (chosen on the first fits, see
                measureBlock).
            window: [int] The hits within this many frames are used for the prediction of the position (see
                predict). None by default (64 frames, at least 4 runs of the longest search).
            extend: [int] Frames measured before the first and after the last hit. None by default (two runs of
                the longest search, at least 32 frames).
        """

        self.hits = hits
        self.opts = opts
        self.psf_sigma = psf_sigma
        self.window = window

        # The frames to measure: the frames of the hits, extended by two runs of the longest search (at least 32
        #   frames, unless given) at both ends, as the object can be visible beyond the runs in which it was found
        run = int(np.max(hits[:, 6]))
        ext = extend if extend is not None else 2*max(run, 16)
        self.first = int(math.floor(hits[0, 0] - ext))
        self.last = int(math.ceil(hits[-1, 0] + ext))

        # Frames per measurement, given or chosen on the first fits
        self.n_frames = n_frames
        self.probe = []

        # Lines fitted through the hits, per 16 frames (see predict)
        self.line_cache = {}

        # Accepted measurements: [frame, x, y, sigma_x, sigma_y, amp, sigma_amp, intensity, background, n_frames,
        #   saturated pixel count]
        self.meas = []

        # The middle frame of every fit and whether it was accepted (see centroids)
        self.status = []


    def overlaps(self, first, last):
        """ Whether the frames of the track overlap the block of frames first .. last - 1. """

        return (self.first < last) and (self.last >= first)


    def predict(self, frame):
        """ Position and velocity at the frame. Where the hits of the search cover the track, from a robust
            local fit of the hits (they are dense and independent of the measurements, so the prediction cannot
            drift away with the noise). Beyond the hits, extrapolated from the nearby accepted measurements.
        """

        # The hits within the window (at least 64 frames, 4 runs of the longest search, unless given) of the frame
        #   are used
        h = self.hits
        window = self.window if self.window else max(64, 4*int(np.max(h[:, 6])))

        # The line through the hits changes slowly along the track, it is fitted once per 16 frames
        key = int(frame//16)
        if key not in self.line_cache:
            centre = 16*key + 8
            near = h[np.abs(h[:, 0] - centre) <= window]
            self.line_cache[key] = self._robustLineFit(near[:, 0], near[:, 1], near[:, 2]) \
                if (len(near) >= 3) and (np.ptp(near[:, 0]) > 0) else None
        # The position from the line, and the velocity as its slope (the first coefficient of np.polyfit)
        fit = self.line_cache[key]
        if fit is not None:
            px, py = fit
            return np.polyval(px, frame), np.polyval(py, frame), px[0], py[0]

        # Beyond the hits: a line through the 12 measurements closest in time
        if len(self.meas) >= 4:
            m = np.array(self.meas)
            near = m[np.argsort(np.abs(m[:, 0] - frame))[:12]]
            if np.ptp(near[:, 0]) > 0:
                return self._robustLine(near[:, 0], near[:, 1], near[:, 2], frame)

        # Too few measurements: a line through the 6 hits closest in time
        near = h[np.argsort(np.abs(h[:, 0] - frame))[:6]]
        if (len(near) >= 2) and (np.ptp(near[:, 0]) > 0):
            return self._robustLine(near[:, 0], near[:, 1], near[:, 2], frame)

        # A single hit (or hits in one frame): extrapolated with the velocity of the search
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

        # At most two refits; stop when the kept points don't change or too few would be left
        keep = np.ones(len(t), dtype=bool)
        for _ in range(2):
            if (keep.sum() < 2) or (np.ptp(t[keep]) <= 0):
                break
            px = np.polyfit(t[keep], x[keep], 1)
            py = np.polyfit(t[keep], y[keep], 1)

            # The distance of every point from the line at its time
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

        # The position and velocity predicted at the middle of the run, and the times of the frames relative to
        #   it (the model of the fit is the PSF at the position x + vx*dt, y + vy*dt in every frame)
        mid = f0 + (n - 1)/2.0
        x, y, vx, vy = self.predict(mid)
        dt = np.arange(n) - (n - 1)/2.0

        # The half size of the patch: 3 PSF sigmas plus half the motion in a frame (the object is smeared along
        #   it) plus 1 px for the error of the prediction
        radius = int(math.ceil(3*self.psf_sigma + 0.5*math.hypot(vx, vy) + 1))

        # The frames of the run, and the pixels left out of the fit
        frames = z[f0 - first:f0 - first + n]
        skip = saturated[f0 - first:f0 - first + n] if saturated is not None \
            else np.zeros(frames.shape, dtype=np.uint8)
        res = fitMovingPSF(frames, skip, x, y, dt, vx, vy, self.psf_sigma, radius)

        # Refit around the fitted position (at most twice) while it is more than 0.5 px from the centre of the
        #   patch, until it moves by less than 0.1 px
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

            # The frames of the hits in this block (plus half a run at both ends), in runs of 8 frames aligned to
            #   multiples of 8
            run = int(np.max(self.hits[:, 6]))
            lo = max(first, int(math.floor(self.hits[0, 0] - run/2.0)))
            hi = min(last, int(math.ceil(self.hits[-1, 0] + run/2.0)) + 1)
            for f0 in range(lo + (-lo) % 8, hi - 7, 8):
                mid, x, y, vx, vy, res = self.fitRun(z, first, f0, 8, saturated)

                # A fit counts if its amplitude is significant and its position near the prediction (within
                #   1.5 px plus half the motion over the 8 frames)
                ok = np.isfinite(res[0]) and (res[2] > self.opts.min_sample_snr*res[6])
                ok = ok and (math.hypot(res[0] - x, res[1] - y) <= 1.5 + 4*math.hypot(vx, vy))

                # Its position error is the total of the formal errors in x and y of the fit
                if ok:
                    self.probe.append(math.hypot(res[4], res[5]))
                if len(self.probe) >= 3:
                    break

            # Wait for the next block if no fit succeeded yet (and the hits continue), otherwise decide with the
            #   fits there are, so the beginning of the track in this block is measured
            if (not self.probe) and (last < self.last) and (last <= self.hits[-1, 0]):
                return

            # The position error scales as 1/sqrt(n) with the number of frames, so the error of a fit of n frames
            #   is sigma8*sqrt(8/n), where sigma8 is the median error of the fits of 8 frames (scaled by
            #   sigma_scale to the actual errors). n is doubled from 1 until this is below max_pos_error, up to
            #   max_measure_frames. Without a successful fit, sigma8 is large and n is the largest
            sigma8 = self.opts.sigma_scale*np.median(self.probe) if self.probe else 1e3
            n = 1
            while (2*n <= self.opts.max_measure_frames) and (sigma8*math.sqrt(8.0/n) > self.opts.max_pos_error):
                n *= 2
            self.n_frames = n

            # Measure from the beginning of the track (the earlier blocks are not available any more, the
            #   probing started in this block)
            self.first = max(self.first, first)

        # The runs of n frames in this block, aligned to multiples of n (so the runs of consecutive blocks
        #   continue each other)
        n = self.n_frames
        start = max(first, self.first)
        start += (-start) % n
        for f0 in range(start, min(last, self.last + 1) - n + 1, n):

            mid, x, y, vx, vy, res = self.fitRun(z, first, f0, n, saturated)

            # A measurement is accepted if:
            #   - the amplitude is at least min_sample_snr times its error,
            #   - the position is within the gate around the prediction (1.5 px plus half the motion over the
            #     run),
            #   - the position error is at most twice the limit,
            #   - it is not on a masked static source
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

            # Photometry: sum of the pixels along the motion in every frame of the measurement, at the fitted
            #   position moved by the velocity to the time of every frame
            dt = np.arange(n) - (n - 1)/2.0
            sat = saturated[f0 - first:f0 - first + n] if saturated is not None \
                else np.zeros((n,) + z.shape[1:], dtype=np.uint8)
            sums, n_sat = framePhotometry(z[f0 - first:f0 - first + n], sat, np.arange(n), bg.noise,
                                          res[0] + vx*dt, res[1] + vy*dt, vx, vy, 3.0*self.psf_sigma)

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

        # The measurements sorted by time
        m = np.array(self.meas)
        m = m[np.argsort(m[:, 0])]

        # The track ends where 3 measurements in a row failed outside the frames of the hits (the extension
        #   beyond the hits continues into frames where the object may not be visible any more). Before the hits,
        #   going forward in time, the measurements before the last 3 failures in a row are dropped; after the
        #   hits the same, going backward
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
        #   which judges the outliers has no support there. Fewer than min_centroids measurements beyond a gap of
        #   more than 8 runs (at least 64 frames) are dropped
        max_gap = max(64, 8*self.n_frames if self.n_frames else 64)
        while len(m) >= self.opts.min_centroids:
            gaps = np.nonzero(np.diff(m[:, 0]) > max_gap)[0]
            if len(gaps) and (gaps[0] + 1 < self.opts.min_centroids):
                m = m[gaps[0] + 1:]
            elif len(gaps) and (len(m) - gaps[-1] - 1 < self.opts.min_centroids):
                m = m[:gaps[-1] + 1]
            else:
                break

        # Remove the outliers from a local quadratic motion model (two passes)
        for _ in range(2):
            if len(m) < self.opts.min_centroids:
                return None
            resid = np.zeros(len(m))
            for i in range(len(m)):

                # The 14 measurements nearest in time (m is sorted by time): a window of 15 centred on the
                #   measurement, shifted to stay inside the track at its ends, without the measurement itself
                lo = min(max(i - 7, 0), max(len(m) - 15, 0))
                near = np.arange(lo, min(lo + 15, len(m)))
                near = near[near != i]

                # The model predicts the position at the time of the measurement (the time is relative to it, so
                #   the prediction is the constant term, the last coefficient), a line with few neighbours
                deg = 2 if len(near) >= 8 else 1
                t0 = m[i, 0]
                px = np.polyfit(m[near, 0] - t0, m[near, 1], deg)
                py = np.polyfit(m[near, 0] - t0, m[near, 2], deg)

                # The distance from the prediction in units of the position error of the measurement (at least
                #   0.2 px, the formal errors of bright measurements are smaller than the error of the model)
                sig = max(math.hypot(m[i, 3], m[i, 4]), 0.2)
                resid[i] = math.hypot(m[i, 1] - px[-1], m[i, 2] - py[-1])/sig

            # Outliers are more than 4 errors from the model
            good = resid < 4
            if good.all():
                break
            m = m[good]

        if len(m) < self.opts.min_centroids:
            return None

        # The signal-to-noise ratio of a measurement is its amplitude over the error of the amplitude (limited to
        #   the width of the column of the FTPdetectinfo)
        snr = np.minimum(m[:, 5]/np.maximum(m[:, 6], 1e-6), 99.99)

        # Reorder the columns of the measurements to the rows of the centroids
        cent = np.column_stack([m[:, 0], m[:, 1], m[:, 2], m[:, 7], m[:, 8], snr, m[:, 10], m[:, 9], m[:, 11:]])

        return cent



def skyQuality(img_handle, config, star_list, platepar, reference_dir=None, input_name='', frame_range=None,
               update=True):
    """ The sky quality of an input from its stars (see RMS.Routines.SkyQuality), with mf_cloud_filter. With a
        reference directory, the clear-sky reference of the camera there is used and updated with this input, and
        its conditions are added to the log there.

    Arguments:
        img_handle: [FrameInterface] Frame-based input.
        config: [Config] Configuration object.
        star_list: [list] Stars of the chunks of frames (see extractStarsFrameInterface).
        platepar: [Platepar] Platepar of the camera, or None.

    Keyword arguments:
        reference_dir: [str] Directory of the clear-sky reference of the camera (see SkyReference). None by
            default.
        input_name: [str] Name of the input (its path), for the log of the conditions. Empty by default.
        frame_range: [tuple] (first frame, end frame) of the part of the input which the stars are from (e.g. a
            segment, see RMS.Formats.FrameBuffer). None by default (the whole input).
        update: [bool] Update the reference and the log with this input. False for a part of an input, as they
            are updated once for the whole input. True by default.

    Return:
        [SkyQualityMap] The sky quality, or None without the stars or the platepar, or if it failed.
    """

    # The sky quality needs the stars and the platepar
    if not (config.mf_cloud_filter and star_list and (platepar is not None)):
        return None

    t0 = time()
    try:

        # The clear-sky reference of the camera, and the sky quality of this input with the chunks at their frames
        #   (by the frame times of the input)
        reference = SkyReference(reference_dir, config.stationID) if reference_dir else None
        first_frame, end_frame = frame_range if (frame_range is not None) else (0, img_handle.total_frames)

        # The frames borrowed from the previous input (continuous detection, see RMS.Formats.FrameChain) are not
        #   part of this input: the sky quality is that of its own chunks (the nearest one for the borrowed frames),
        #   so the chunks of the previous input are neither clouded for lack of stars nor added to the reference
        #   and the log again
        first_frame = max(first_frame, getattr(img_handle, 'n_context', 0))
        sky_map = SkyQualityMap(star_list, platepar, config, img_handle.beginning_datetime.replace(tzinfo=None),
                                img_handle.fps, getattr(img_handle, 'chunk_frames', None),
                                max_offset=config.mf_cloud_max_offset, total_frames=end_frame,
                                min_matched=config.mf_cloud_min_matched,
                                first_frames=(chunkFirstFrames(star_list, img_handle,
                                                               frame_range=(first_frame, end_frame))
                                              if hasattr(img_handle, 'chunk_frames') else None),
                                reference=reference, input_name=input_name, first_frame=first_frame)

        # The stars of this input on clear sky are added to the reference, and its conditions to the log
        if update:
            sky_map.updateReference()

    # Without a calibration, a star catalog or references of the stars, the sky quality is not known, and the input
    #   is processed without it; other errors are logged, and don't fail the input either
    except CalibrationError as e:
        log.warning('Matched filter: no sky quality ({:s}), the clouded sky is not excluded'.format(str(e)))
        return None

    except Exception:
        log.warning('Matched filter: the sky quality failed, the clouded sky is not excluded:\n'
                    + traceback.format_exc())
        return None

    # The medians of the chunks, NaN if none is known
    def median(values):
        values = np.asarray(values)[np.isfinite(values)]
        return np.median(values) if len(values) else np.nan

    # A summary of the sky quality of the input for the log
    clear = sky_map.summary()
    log.info('Matched filter: sky quality in {:.1f} s, clear fraction of the chunks median {:.2f}, min {:.2f}, '
             'matched stars per chunk median {:.0f}, transparency median {:+.2f} mag, reliable stars seen median '
             '{:.2f}'.format(time() - t0, median(clear), np.min(clear), median(sky_map.n_matched),
                             median(sky_map.transparency), median(sky_map.reliable_seen)))

    return sky_map



def correctRollingShutter(detections, img_handle, config):
    """ Correct the frames of the centroids for a rolling shutter (deinterlace_order -1): a point at the row y is
        recorded later than the top row of its frame by y/rows of the readout time of the sensor (see
        RMS.Routines.RollingShutterCorrection). Applied once to the detections of the whole input, the frames
        within the input stay unchanged until then.

    Arguments:
        detections: [list] Detections as [rho, theta, centroids], centroids rows [frame, x, y, ...] in the
            unbinned image.
        img_handle: [FrameInterface] Image handle of the input.
        config: [Config] Configuration object.

    Return:
        [list] The detections, their frames corrected in place (unchanged without a rolling shutter).
    """

    # Only a rolling shutter is corrected
    if config.deinterlace_order != -1:
        return detections

    # The rows of the unbinned image; the readout time and the crop of the capture come from the config (only the
    #   inputs from the RMS capture are cropped)
    det_bin = config.detection_binning_factor if img_handle.input_type != 'ff' else 1
    n_rows = img_handle.nrows*det_bin
    cropped = RollingShutterCorrection.croppedInput(img_handle.input_type)

    # Move the frame of every centroid by the time its row was read after the top row
    for det in detections:
        cent = det[2]
        cent[:, 0] = RollingShutterCorrection.correctFrameTime(cent[:, 0], cent[:, 2], n_rows, config,
                                                               fps=getattr(img_handle, 'fps', None), cropped=cropped)

    return detections


def detectMatchedFilter(img_handle, config, mask=None, dark=None, flat_struct=None, star_list=None,
                        return_detector=False, platepar=None, reference_dir=None, input_name='', frame_range=None,
                        persistence_prior=None, sky_range=None, persistence_until=None, max_tracks=None,
                        persistence_prior_frames=0, track_deviation=True):
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
        platepar: [Platepar] Platepar of the camera. With the stars and mf_cloud_filter, the clouded sky is not
            searched and the measurements on it are removed (see RMS.Routines.SkyQuality). None by default.
        reference_dir: [str] Directory of the clear-sky reference of the camera, which is used and updated (see
            skyQuality). None by default (only this input is used).
        input_name: [str] Name of the input (its path), for the log of the conditions. Empty by default.
        frame_range: [tuple] (first frame, end frame) of a segment of the input to search (see
            detectMatchedFilterSegments), with the stars of its chunks. The sky reference and the log are then not
            updated. None by default (the whole input).
        persistence_prior: [dict] Persistence counts of the segments before this one (see
            MatchedFilterDetector.removePersistent). None by default.
        sky_range: [tuple] (first frame, end frame) of the stars of the sky quality, e.g. all the frames of the
            input read so far, as a short segment alone may have too few chunks for the references of the stars.
            None by default (frame_range).
        persistence_until: [int] First frame of the next segment, before which the persistence counts are kept to
            carry to it (see MatchedFilterDetector.persistenceCheckpoint). None by default.
        persistence_prior_frames: [int] Number of frames of the counts of persistence_prior. 0 by default.
        max_tracks: [int] Largest number of candidate tracks which are measured. None by default
            (mf_max_tracks).
        track_deviation: [bool] Whether the detections which wander from a smooth path are removed (see
            removeDeviatingDetections). False for the segments of an input, whose whole tracks are judged after
            they are joined. True by default.

    Return:
        [list] Detections as [rho, theta, centroids], or (detections, detector) with return_detector. The
            detector is None for FF files.
    """

    # FF files keep only the maximum and the average of their frames, the frames themselves are needed
    if img_handle.input_type == 'ff':
        log.warning('Matched filter: FF files have no frames to search, skipped')
        return ([], None) if return_detector else []

    # The calibration is binned in place, and the caller's objects may already be binned for the normal
    #   detection, so copies are binned
    if config.detection_binning_factor > 1:
        mask, dark, flat_struct = binImageCalibration(config, copy.deepcopy(mask), copy.deepcopy(dark),
                                                      copy.deepcopy(flat_struct))

    # Limit the threads of the compiled code, e.g. when several files are processed in parallel
    if config.mf_threads > 0:
        numba.set_num_threads(min(config.mf_threads, numba.config.NUMBA_NUM_THREADS))

    # The search follows the objects from frame to frame at a constant interval, so where frames are missing in
    #   the recording (a gap in the times of the frames) the parts before and after the gap are searched
    #   separately, as separate inputs would be
    first, end = frame_range if (frame_range is not None) else (0, img_handle.total_frames)
    gaps = timeGaps(img_handle, frame_range=(first, end))
    if gaps:
        detections, detector = detectAcrossGaps(img_handle, config, gaps, (first, end), mask=mask, dark=dark,
                                                flat_struct=flat_struct, star_list=star_list, platepar=platepar,
                                                reference_dir=reference_dir, input_name=input_name,
                                                whole_input=(frame_range is None),
                                                persistence_prior=persistence_prior, sky_range=sky_range,
                                                persistence_until=persistence_until, max_tracks=max_tracks,
                                                persistence_prior_frames=persistence_prior_frames,
                                                track_deviation=track_deviation)

    # Otherwise the whole range is searched at once
    else:
        detections, detector = detectRange(img_handle, config, mask=mask, dark=dark, flat_struct=flat_struct,
                                           star_list=star_list, platepar=platepar, reference_dir=reference_dir,
                                           input_name=input_name, frame_range=frame_range,
                                           persistence_prior=persistence_prior, sky_range=sky_range,
                                           persistence_until=persistence_until, max_tracks=max_tracks,
                                           persistence_prior_frames=persistence_prior_frames,
                                           track_deviation=track_deviation, update_sky=(frame_range is None))

    # The times of the detections of the whole input are corrected for a rolling shutter (the detections of the
    #   segments of an input are corrected together, see detectMatchedFilterSegments)
    if frame_range is None:
        detections = correctRollingShutter(detections, img_handle, config)

    return (detections, detector) if return_detector else detections



def detectRange(img_handle, config, mask=None, dark=None, flat_struct=None, star_list=None, platepar=None,
                reference_dir=None, input_name='', frame_range=None, persistence_prior=None, sky_range=None,
                persistence_until=None, max_tracks=None, persistence_prior_frames=0, track_deviation=True,
                update_sky=True, aperture=True):
    """ Detect and measure the objects in a range of frames without gaps in the recording, with the calibration
        already binned (see detectMatchedFilter, which describes the arguments).

    Keyword arguments:
        update_sky: [bool] Update the sky reference and the log of the conditions with the stars. True by default.
        aperture: [bool] Put the intensities on the scale of the stars (see apertureCorrection). True by default.

    Return:
        (detections, detector): [tuple] Detections as [rho, theta, centroids], and the detector after its run.
    """

    # Detect and measure the objects
    detector = MatchedFilterDetector(img_handle, config, mask=mask, dark=dark, flat_struct=flat_struct,
                                     frame_range=frame_range)
    detector.persistence_prior = persistence_prior or {}
    detector.persistence_until = persistence_until
    detector.persistence_prior_frames = persistence_prior_frames
    if max_tracks is not None:
        detector.opts.max_tracks = max_tracks
    if not track_deviation:
        detector.opts.track_deviation_fwhm = 0.0
    detector.sky_map = skyQuality(img_handle, config, star_list, platepar, reference_dir=reference_dir,
                                  input_name=input_name, frame_range=(sky_range or frame_range),
                                  update=update_sky)
    detections = detector.run()

    # The measurements on clouded sky were removed in the run
    if detector.sky_map is not None:
        log.info('Matched filter: {:d} detections and {:d} frames of detections on clouded sky removed'.format(
            detector.cloud_detections_removed, detector.cloud_rows_removed))

    # The intensities on the scale of the stars of the photometric calibration. A factor far from 1 means the
    #   stars were not measured properly (e.g. clouds), and the intensities are then left as they are
    if aperture and star_list and detections:
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

    return detections, detector



def timeGaps(img_handle, frame_range=None, factor=TIME_GAP_FACTOR):
    """ The gaps in the recording of an input: where the time between two neighbouring frames is longer than
        factor times the usual interval, frames are missing (e.g. while the clock of the camera is synchronised).
        Only an input with the time of every frame shows them; the times are those the input already holds after
        its frames were read (the star extraction reads them all), so nothing is read again.

    Arguments:
        img_handle: [FrameInterface] Frame-based input.

    Keyword arguments:
        frame_range: [tuple] (first frame, end frame) of the frames. None by default (the whole input).
        factor: [float] Interval, in median intervals of the frames, above which there is a gap.
            TIME_GAP_FACTOR by default.

    Return:
        [list] The first frame after every gap, in order, empty without gaps.
    """

    # The inputs without the time of every frame have their frames at a constant interval
    if (not hasattr(img_handle, 'hasAccurateTime')) or (not img_handle.hasAccurateTime()):
        return []

    # Too few frames to tell the usual interval from a gap
    first, end = frame_range if (frame_range is not None) else (0, img_handle.total_frames)
    if end - first < 3:
        return []

    try:

        # The times of the frames (s), from the table of the frame times of the input if it has one
        table = getattr(img_handle, 'frame_unix_times', None)
        if (table is not None) and (len(table) >= end):
            times = np.asarray(table[first:end], dtype=np.float64)

        # Otherwise from the time of every frame, relative to the first one
        else:
            t0 = None
            times = []
            for i in range(first, end):
                t = img_handle.currentFrameTime(frame_no=i, dt_obj=True)
                if t.tzinfo is not None:
                    t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
                if t0 is None:
                    t0 = t
                times.append((t - t0).total_seconds())
            times = np.array(times)

    except Exception:
        log.warning('Matched filter: the times of the frames could not be read, no gaps are looked for:\n'
                    + traceback.format_exc())
        return []

    # The gaps against the usual interval, the median of the positive steps (which the gaps hardly change): an input
    #   whose time stamps repeat (e.g. one stamp for every pair of frames) has steps of zero, which are no gaps and
    #   would make the median zero
    steps = np.diff(times)
    positive = steps[steps > 0]
    if len(positive) == 0:
        return []
    step = np.median(positive)

    return [first + int(i) + 1 for i in np.nonzero(steps > factor*step)[0]]



class GapPieces(object):
    def __init__(self, detectors, ranges, gaps, gap_seconds):
        """ The matched filter of an input (or of a segment of it) whose recording has gaps: the parts between
            the gaps are searched by their own detectors, and this stands for the detector of the whole range
            (its summary, the light between points, the persistence counts carried on). Everything else is that
            of the main detector: the first one with a measured PSF.

        Arguments:
            detectors: [list] The detector of every part, after its run.
            ranges: [list] (first frame, end frame) of every part.
            gaps: [list] The first frame after every gap.
            gap_seconds: [list] The length of every gap (s), the time from the frame before it to the frame after
                it.
        """

        # The parts and the gaps, and the main detector
        self.detectors = detectors
        self.ranges = ranges
        self.gaps = gaps
        self.gap_seconds = gap_seconds
        self.primary = next((det for det in detectors if det.psf_sigma is not None), detectors[0])

        # The counts and the time of the steps of the whole range, from all parts
        self.timing = collections.defaultdict(float)
        for det in detectors:
            for key, value in det.timing.items():
                self.timing[key] += value
        self.cloud_rows_removed = sum(det.cloud_rows_removed for det in detectors)
        self.cloud_removed_row_frames = [f for det in detectors for f in det.cloud_removed_row_frames]
        self.cloud_detections_removed = sum(det.cloud_detections_removed for det in detectors)
        self.cloud_removed_frames = [f for det in detectors for f in det.cloud_removed_frames]
        self.tracks_dropped = sum(det.tracks_dropped for det in detectors)
        self.tracks_measured = sum(det.tracks_measured for det in detectors)
        self.faint_measured = sum(det.faint_measured for det in detectors)
        self.significances = [sig for det in detectors for sig in det.significances]
        self.deviation_rejected = sum(det.deviation_rejected for det in detectors)
        self.photometry = {key: value for det in detectors for key, value in det.photometry.items()}

        # The blocks of all parts (a part too short to be searched has none)
        self.block_starts = [f for det in detectors for f in getattr(det, 'block_starts', [])]
        self.block_ends = [f for det in detectors for f in getattr(det, 'block_ends', [])]

        # Set after the parts are searched (see detectAcrossGaps)
        self.aperture_correction = None
        self.sky_map = self.primary.sky_map
        self.persistence_carry, self.persistence_carry_frames = None, 0
        self.persistence_full, self.persistence_full_frames = None, 0


    def __getattr__(self, name):
        """ Everything else is that of the main detector. """

        return getattr(self.__dict__['primary'], name)


    def connectedLight(self, frame, x1, y1, x2, y2):
        """ The light between two points in a frame (see MatchedFilterDetector.connectedLight), from the
            detector of the part which the frame belongs to, None outside of the parts which were searched. """

        # The part whose searched blocks hold the frame
        for det in self.detectors:
            starts = getattr(det, 'block_starts', [])
            if starts and (starts[0] <= frame < det.block_ends[-1]):
                return det.connectedLight(frame, x1, y1, x2, y2)

        return None


    def summary(self):
        """ Summary of the whole range (see MatchedFilterDetector.summary), from the summaries of its parts, with
            the gaps. """

        # The summary of the main detector, with the counts, the tracks and the time of all parts, and the gaps
        summary = self.primary.summary()
        tracks = [track for det in self.detectors for track in det.summary()['tracks']]
        summary.update({
            'aperture_correction': self.aperture_correction,
            'clear_sky_fraction': (round(float(np.mean(self.sky_map.clear)), 3) if self.sky_map is not None
                                   else None),
            'cloud_frames_removed': int(self.cloud_rows_removed),
            'cloud_detections_removed': int(self.cloud_detections_removed),
            'cloud_removed_utc': self.primary.frameTimes(sorted(self.cloud_removed_frames)),
            'tracks_dropped': int(self.tracks_dropped),
            'gpu': any(bool(det.opts.use_gpu) for det in self.detectors),
            'timing_s': {k: round(v, 2) for k, v in self.timing.items()},
            'tracks': tracks,
            'linear_structure_rejected': sum(int(track.get('linear_structure', False)) for track in tracks),
            'deviation_rejected': int(self.deviation_rejected),
            'faint_measured': int(self.faint_measured),
            'time_gaps': [{'frame': int(f), 'gap_s': round(float(g), 3)}
                          for f, g in zip(self.gaps, self.gap_seconds)],
        })

        return summary



def detectAcrossGaps(img_handle, config, gaps, frame_range, mask=None, dark=None, flat_struct=None,
                     star_list=None, platepar=None, reference_dir=None, input_name='', whole_input=True,
                     persistence_prior=None, sky_range=None, persistence_until=None, max_tracks=None,
                     persistence_prior_frames=0, track_deviation=True):
    """ Detect and measure the objects in a range of frames with gaps in the recording: every part between the
        gaps is searched as an input of its own (a track is not followed across a gap, as the object moved on while
        no frames were recorded; an object seen on both sides is two detections, as in two files), with the
        persistent pixels counted over all parts and the sky quality from the stars of the whole range. The
        calibration is already binned (see detectMatchedFilter, which describes the other arguments).

    Arguments:
        img_handle: [FrameInterface] Frame-based input.
        config: [Config] Configuration object.
        gaps: [list] The first frame after every gap (see timeGaps).
        frame_range: [tuple] (first frame, end frame) of the range which is searched.

    Keyword arguments:
        whole_input: [bool] The range is the whole input: the sky reference and the log of the conditions are
            updated with it, and the persistent pixels are judged over it. False for a segment of an input (see
            detectMatchedFilterSegments), which does both for the whole input. True by default.

    Return:
        (detections, detector): [tuple] Detections as [rho, theta, centroids], and the GapPieces of the range.
    """

    # The parts between the gaps
    first, end = frame_range
    bounds = [first] + list(gaps) + [end]
    ranges = list(zip(bounds[:-1], bounds[1:]))

    # The length of every gap, from the frame before it to the frame after it
    gap_seconds = []
    for g in gaps:
        t1 = img_handle.currentFrameTime(frame_no=g - 1, dt_obj=True)
        t2 = img_handle.currentFrameTime(frame_no=g, dt_obj=True)
        gap_seconds.append((t2 - t1).total_seconds())
    log.info('Matched filter: {:d} gaps in the recording ({:s} s), the {:d} parts between them are searched '
             'separately'.format(len(gaps), ', '.join('{:.2f}'.format(g) for g in gap_seconds), len(ranges)))

    # Every part is searched with the persistence counts of the parts before it, and the counts which reach the
    #   first frame of the next segment of the input are carried to it (see MatchedFilterDetector.persistence_until)
    prior, prior_frames = (persistence_prior or {}), persistence_prior_frames
    carry = None
    detectors, part_detections = [], []
    for a, b in ranges:

        # The counts up to the next segment, when it begins at or before this part
        if (persistence_until is not None) and (carry is None) and (persistence_until <= a):
            carry = (prior, prior_frames)
        until = persistence_until if (persistence_until is not None) and (a < persistence_until < b) else None

        detections, detector = detectRange(img_handle, config, mask=mask, dark=dark, flat_struct=flat_struct,
                                           star_list=star_list, platepar=platepar, reference_dir=reference_dir,
                                           input_name=input_name, frame_range=(a, b), persistence_prior=prior,
                                           sky_range=(sky_range or frame_range), persistence_until=until,
                                           max_tracks=max_tracks, persistence_prior_frames=prior_frames,
                                           track_deviation=track_deviation, update_sky=False, aperture=False)
        log.info('Matched filter: part {:d}-{:d}: {:d} detections'.format(a, b, len(detections)))

        # The counts carried to the next segment from the part where it begins, and the counts so far to the
        #   next part (a part too short to be searched has none)
        if until is not None:
            carry = ((detector.persistence_carry, detector.persistence_carry_frames)
                     if detector.persistence_carry is not None else (prior, prior_frames))
        if detector.persistence_full is not None:
            prior, prior_frames = detector.persistence_full, detector.persistence_full_frames
        detectors.append(detector)
        part_detections.append(detections)

    # The parts stand for one detector of the range, with the counts carried to the next segment (all of them
    #   if it begins after the range) and the counts of the whole range
    run = GapPieces(detectors, ranges, gaps, gap_seconds)
    run.persistence_carry, run.persistence_carry_frames = carry if carry is not None else (prior, prior_frames)
    run.persistence_full, run.persistence_full_frames = prior, prior_frames

    # The detections of all parts, which don't overlap in time
    detections = [det for dets in part_detections for det in dets]

    # The intensities on the scale of the stars, by the factor of every part from the stars of its chunks (at the
    #   frames of their times, see chunkFirstFrames). A part without such stars takes the median factor of the
    #   others
    if star_list and detections:
        first_frames = chunkFirstFrames(star_list, img_handle) if hasattr(img_handle, 'chunk_frames') else None
        factors = []
        for detector, dets in zip(detectors, part_detections):
            factor = None
            if dets and getattr(detector, 'block_starts', None):
                factor, _ = detector.apertureCorrection(star_list, first_frames=first_frames)
            factors.append(factor if (factor is not None) and (0.5 <= factor <= 2.0) else None)
        known = [f for f in factors if f is not None]
        if known:
            for factor, dets in zip(factors, part_detections):
                for det in dets:
                    det[2][:, 3] *= factor if (factor is not None) else np.median(known)
            run.aperture_correction = round(float(np.median(known)), 4)
            log.info('Matched filter: intensities scaled by {:s} to the scale of the stars'.format(
                ', '.join('{:.3f}'.format(f) if f is not None else 'the median' for f in factors)))
        else:
            log.warning('Matched filter: no aperture correction of the intensities (no part has enough stars)')

    # For the whole input, the detections mostly on pixels persistent over all parts are removed (each part only
    #   removed the hits on pixels persistent in the parts so far), and the sky reference and the log of the
    #   conditions are updated once
    if whole_input:
        n_before = len(detections)
        detections = removePersistentDetections(detections, prior, config.mf_persistence, run.det_bin)
        if len(detections) < n_before:
            log.info('Matched filter: {:d} detections on persistent pixels of the whole input removed'.format(
                n_before - len(detections)))
        sky_map = skyQuality(img_handle, config, star_list, platepar, reference_dir=reference_dir,
                             input_name=input_name)
        if sky_map is not None:
            run.sky_map = sky_map

    return detections, run



def isValidDetection(cent, detector):
    """ Whether the rows of a detection make a detection, with the checks of MatchedFilterDetector.run: enough of
        them, lasting and moving long enough (the motion in the binned frames of the detection).

    Arguments:
        cent: [ndarray] Rows of the detection (unbinned), [frame, x, y, ...], sorted by frame.
        detector: [MatchedFilterDetector] A detector of the input, for the limits.

    Return:
        [bool] True if it is a detection.
    """

    return (len(cent) >= max(2, detector.opts.min_centroids)) \
        and (cent[-1, 0] - cent[0, 0] >= detector.opts.min_frames) \
        and (np.hypot(*(cent[-1, 1:3] - cent[0, 1:3]))/detector.det_bin >= detector.minDisplacement())


def trackDeviation(cent):
    """ How far a track wanders sideways from one smooth path over its whole length: the RMS distance of its rows
        across the direction of motion from a quadratic in time fitted to x and to y. Real objects follow
        straight or smoothly curved paths (also over long passes, and with the distortion of the lens), while a
        track which goes back and forth (e.g. along diffuse light) does not. The distance along the motion is not
        used: the angular speed of a real object changes over a long pass more than the quadratic follows.

    Arguments:
        cent: [ndarray] Rows of the detection, [frame, x, y, ...], sorted by frame.

    Return:
        [float] The RMS distance across the motion (px), 0 for fewer than 5 rows.
    """

    if (len(cent) < 5) or (np.ptp(cent[:, 0]) <= 0):
        return 0.0

    # The time scaled to -0.5 .. 0.5, so the fit is well conditioned for long tracks
    t = (cent[:, 0] - cent[:, 0].mean())/np.ptp(cent[:, 0])
    px, py = np.polyfit(t, cent[:, 1], 2), np.polyfit(t, cent[:, 2], 2)
    dx = cent[:, 1] - np.polyval(px, t)
    dy = cent[:, 2] - np.polyval(py, t)

    # The direction of motion of the fitted path at every row, and the distance across it
    vx, vy = np.polyval(np.polyder(px), t), np.polyval(np.polyder(py), t)
    speed = np.hypot(vx, vy)
    speed[speed == 0] = 1.0
    across = (dy*vx - dx*vy)/speed

    return float(np.sqrt(np.mean(across**2)))


def removeDeviatingDetections(detections, fwhm, fwhm_factor, length_factor):
    """ Remove the detections which wander sideways from one smooth path by more than the limit (see
        trackDeviation): fwhm_factor PSF FWHMs (the scatter of the measurements, which scales with the size of the
        PSF) plus length_factor times the length of the track (the bending of a long track beyond the smooth path,
        e.g. by the distortion of a wide-field lens, which is the same fraction of the length in pixels as in
        degrees). Both factors are dimensionless, so the same values hold for cameras of other fields of view.

    Arguments:
        detections: [list] Detections as [rho, theta, centroids] (unbinned).
        fwhm: [float or function] FWHM of the PSF (px, unbinned), or a function which gives it for the rows of a
            detection (e.g. the PSF of the segment of the input which the detection belongs to).
        fwhm_factor: [float] Deviation allowed for the scatter of the measurements (FWHMs), 0 to keep all.
        length_factor: [float] Deviation allowed per pixel of the length of the track.

    Return:
        (detections, rejected): [tuple] The detections kept, and (first frame, deviation, limit) of the removed
            ones.
    """

    # The test is off, or the PSF is unknown
    if (fwhm_factor <= 0) or (fwhm is None):
        return list(detections), []

    # Every detection judged on its own
    kept, rejected = [], []
    for det in detections:
        cent = det[2]

        # The FWHM of the detection (a detection without a known PSF is kept)
        track_fwhm = fwhm(cent) if callable(fwhm) else fwhm
        if track_fwhm is None:
            kept.append(det)
            continue

        # Its deviation from the smooth path against the limit for its PSF and its length
        deviation = trackDeviation(cent)
        limit = fwhm_factor*track_fwhm + length_factor*float(np.hypot(*(cent[-1, 1:3] - cent[0, 1:3])))
        if deviation > limit:
            rejected.append((float(cent[0, 0]), deviation, limit))
            log.info('Matched filter: detection {:.0f}-{:.0f} wanders sideways from a smooth path by {:.2f} px '
                     'RMS (more than {:.2f} px), rejected'.format(cent[0, 0], cent[-1, 0], deviation, limit))
        else:
            kept.append(det)

    return kept, rejected


def removeInvalidDetections(detections, detector):
    """ The detections which pass the checks of a detection (see isValidDetection). """

    return [det for det in detections if isValidDetection(det[2], detector)]


def removeCloudedRows(detections, sky_map, detector):
    """ Remove the rows of the detections on clouded sky (see SkyQualityMap.isClear), and the detections which
        are then no detections any more (too few rows, too short in time or in motion, see
        MatchedFilterDetector.run), as the cloud filter of MatchedFilterDetector.run does.

    Arguments:
        detections: [list] Detections as [rho, theta, centroids] (unbinned).
        sky_map: [SkyQualityMap] Sky quality of the input.
        detector: [MatchedFilterDetector] A detector of the input, for the polar lines of the detections.

    Return:
        (detections, removed_rows, removed_first): [tuple] The detections which are kept, the frames of the rows
            which were removed, and the first frames of the detections which were removed.
    """

    kept, removed_rows, removed_first = [], [], []
    for det in detections:
        cent = det[2]
        clear = np.asarray(sky_map.isClear(cent[:, 1], cent[:, 2], cent[:, 0]), dtype=bool)

        # Nothing clouded: kept as it is
        if clear.all():
            kept.append(det)
            continue

        # The rows on clear sky, if they still make a detection
        removed_rows += cent[~clear, 0].tolist()
        rows = cent[clear]
        if not isValidDetection(rows, detector):
            removed_first.append(int(round(cent[0, 0])))
            continue
        kept.append([*detector.polarLine(rows), rows])

    return kept, removed_rows, removed_first


def persistentMasks(persistence, fraction):
    """ The persistent pixels of every run length and bin of the search: above the threshold in more than the
        given fraction of the runs (and in at least 2 runs), grown by a binned pixel.

    Arguments:
        persistence: [dict] {(run, bin): (counts, runs)}, the number of runs in which every pixel was above the
            threshold, and the number of runs.
        fraction: [float] Fraction of the runs (mf_persistence).

    Return:
        [dict] {(run, bin): mask}.
    """

    return {key: cv2.dilate((counts > max(2, fraction*runs)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            for key, (counts, runs) in persistence.items()}


def removePersistentDetections(detections, persistence, fraction, det_bin, max_share=0.5):
    """ Remove the detections which are mostly (more than max_share of their centroids) on pixels which are
        persistent over the whole input (see MatchedFilterDetector.removePersistent), in the runs of any length of
        the finest bin of the search. Used after the segments of an input, which only remove the hits on pixels
        which are persistent for sure.

    Arguments:
        detections: [list] Detections as [rho, theta, centroids] (unbinned).
        persistence: [dict] {(run, bin): (counts, runs)} of the whole input.
        fraction: [float] Fraction of the runs (mf_persistence).
        det_bin: [int] Detection binning of the frames.

    Keyword arguments:
        max_share: [float] Largest share of the centroids on persistent pixels. 0.5 by default.

    Return:
        [list] The detections which are kept.
    """

    if (not persistence) or (not detections):
        return detections

    # The persistent pixels in the runs of the finest bin
    masks = persistentMasks(persistence, fraction)
    finest = min(b for _, b in masks)
    masks = [(b, mask) for (_, b), mask in masks.items() if b == finest]

    kept = []
    for det in detections:
        cent = det[2]

        # The centroids in the binned frames of the detection, then in the bins of the search (as the hits, see
        #   removePersistent)
        x = (cent[:, 1] - (det_bin - 1)/2.0)/det_bin
        y = (cent[:, 2] - (det_bin - 1)/2.0)/det_bin
        on = np.zeros(len(cent), dtype=bool)
        for b, mask in masks:
            xb = np.clip(np.round((x + 0.5)/b - 0.5).astype(int), 0, mask.shape[1] - 1)
            yb = np.clip(np.round((y + 0.5)/b - 0.5).astype(int), 0, mask.shape[0] - 1)
            on |= mask[yb, xb]

        if np.mean(on) <= max_share:
            kept.append(det)

    return kept


# A detection of a segment is near one of the neighbouring segment within this many times the tolerance of the same
#   object (see stitchSegments)
NEAR_TOLERANCE_FACTOR = 4.0


def connectedOver(detectors):
    """ The light between two points in a frame (see MatchedFilterDetector.connectedLight) from the detector of the
        segment of the input whose blocks contain the frame, for the detections joined over the segments. A frame of
        a segment no longer held in memory is read again from the input (counted in the frames read outside the
        segments): at most 5 frames for a pair of detections, and only for the pairs which become duplicates after
        the segments are joined (see removeDuplicateDetections), as the pairs within a segment are judged while it
        is in memory.

    Arguments:
        detectors: [list] The detector of every segment, None where it was not searched.

    Return:
        [function] connected(frame, x1, y1, x2, y2), None for a frame of no searched segment.
    """

    def connected(frame, x1, y1, x2, y2):
        """ The light between two points in a frame from the detector of its segment (see connectedLight).

        Arguments:
            frame: [float] Frame of the input.
            x1: [float] X of the first point (unbinned px).
            y1: [float] Y of the first point (unbinned px).
            x2: [float] X of the second point (unbinned px).
            y2: [float] Y of the second point (unbinned px).

        Return:
            [float] The fraction of the segment above the level, None for a frame of no searched segment.
        """

        # The first searched segment whose blocks contain the frame
        for det in detectors:
            if (det is not None) and det.block_starts and (det.block_starts[0] <= frame < det.block_ends[-1]):
                return det.connectedLight(frame, x1, y1, x2, y2)

        return None

    return connected


def stitchSegments(segment_detections, cuts, overlaps, detector, tolerance, searched=None):
    """ Join the detections of neighbouring segments of an input. Every overlap of two segments is split at its
        middle: the frames before it belong to the first segment, the ones after it to the second one. An object which crosses the middle is found in both segments: its detections are joined
        there (the part before it from the first segment, the part after it from the second one). The other
        detections in the half of the other segment (or crossing the middle) are left to it (or cut at the middle)
        if it found something near them, e.g. the same object measured differently, so an object is not kept twice;
        those found in one segment only are kept whole (the segments are searched with their own statistics). The
        pieces are joined later, see MatchedFilterDetector.joinDetections.

    Arguments:
        segment_detections: [list] Detections of every segment, as [rho, theta, centroids].
        cuts: [list] Frame at the middle of the overlap of every pair of neighbouring segments.
        overlaps: [list] (first frame, end frame) of the overlap of every pair of neighbouring segments.
        detector: [MatchedFilterDetector] A detector of the input, for the polar lines of the joined detections.
        tolerance: [float] Largest median distance (unbinned px) between the two detections of an object in the
            overlap.

    Keyword arguments:
        searched: [list] Whether every segment was searched. The half of an overlap which belongs to a segment that
            was not searched stays with the other one. None by default (all were searched).

    Return:
        [list] Detections of the whole input.
    """

    def _positionsAt(cent, frames):
        """ The positions of a detection at the given frames, interpolated between its centroids. """

        order = np.argsort(cent[:, 0])
        f, x, y = cent[order, 0], cent[order, 1], cent[order, 2]
        return np.interp(frames, f, x), np.interp(frames, f, y)

    def _near(cent, others):
        """ Whether one of the other detections follows an object near this one (within a few PSF widths, a
            looser tolerance than for the same object) in the frames they share. """

        for other in others:
            common = cent[(cent[:, 0] >= other[:, 0].min()) & (cent[:, 0] <= other[:, 0].max())]
            if len(common) < 3:
                continue
            x, y = _positionsAt(other, common[:, 0])
            if np.median(np.hypot(common[:, 1] - x, common[:, 2] - y)) <= NEAR_TOLERANCE_FACTOR*tolerance:
                return True

        return False

    detections = list(segment_detections[0]) if segment_detections else []
    for k in range(1, len(segment_detections)):
        cut, (ov_first, ov_end) = cuts[k - 1], overlaps[k - 1]
        segment = segment_detections[k]

        # The detections of the two segments as they were found, to check if one has something near
        previous_cents = [det[2] for det in detections]
        segment_cents = [det[2] for det in segment]
        cut_previous = ((searched is None) or searched[k])
        cut_segment = ((searched is None) or searched[k - 1])

        # The detections of the previous segment entirely after the middle of the overlap, and the ones of this
        #   segment entirely before it, belong to the other segment if it found something near them (otherwise
        #   they were found in this segment only, and are kept)
        if cut_previous:
            detections = [det for det in detections
                          if (det[2][:, 0].min() < cut) or not _near(det[2], segment_cents)]
        if cut_segment:
            segment = [det for det in segment
                       if (det[2][:, 0].max() >= cut) or not _near(det[2], previous_cents)]

        # The detections of the previous segment which reach into the overlap can be joined
        candidates = [i for i, det in enumerate(detections) if det[2][:, 0].max() >= ov_first]
        used = set()
        new = []
        for det in segment:
            cent = det[2]

            # The centroids of this detection in the overlap, compared with the positions of every candidate at
            #   their frames (within the frames of the candidate)
            best, best_dist = None, tolerance
            in_overlap = cent[(cent[:, 0] >= ov_first) & (cent[:, 0] < ov_end)]
            for i in candidates:
                if i in used:
                    continue
                other = detections[i][2]
                common = in_overlap[(in_overlap[:, 0] >= other[:, 0].min()) & (in_overlap[:, 0] <= other[:, 0].max())]
                if len(common) < 3:
                    continue
                x, y = _positionsAt(other, common[:, 0])
                dist = float(np.median(np.hypot(common[:, 1] - x, common[:, 2] - y)))
                if dist <= best_dist:
                    best, best_dist = i, dist

            # Not the same object as one of the previous segment: its part after the middle of the overlap if the
            #   previous segment found something near it (e.g. the same object measured differently), all of it
            #   otherwise
            if best is None:
                if cut_segment and (cent[:, 0].min() < cut) and _near(cent, previous_cents):
                    cent = cent[cent[:, 0] >= cut]
                if len(cent) >= 2:
                    new.append([*detector.polarLine(cent), cent])
                continue

            # The same object: the centroids before the cut from the previous segment, the others from this one.
            #   Where one of them doesn't reach the cut, they are joined where the other one begins or ends, so no
            #   measured frames are lost
            other = detections[best][2]
            split = min(max(cut, cent[:, 0].min()), other[:, 0].max() + 1)
            joined = np.vstack([other[other[:, 0] < split], cent[cent[:, 0] >= split]])
            if len(joined) < 2:
                joined = other if len(other) >= len(cent) else cent
            detections[best] = [*detector.polarLine(joined), joined]
            used.add(best)

        # The detections of the previous segment which cross the middle of the overlap and were not joined: their
        #   part before it, if this segment found something near them
        if cut_previous:
            for i in candidates:
                cent = detections[i][2]
                if (i not in used) and (cent[:, 0].max() >= cut) and _near(cent, segment_cents):
                    cent = cent[cent[:, 0] < cut]
                    detections[i] = [*detector.polarLine(cent), cent] if len(cent) >= 2 else None
            detections = [det for det in detections if det is not None]
        detections += new

    detections.sort(key=lambda det: det[2][0, 0])

    return detections



class SegmentedRun(object):
    def __init__(self, img_handle, ranges, detectors, cuts, sky_map, buffer, timing, removed_rows=(),
                 removed_first=(), deviation_rejected=0):
        """ The matched filter of an input in segments (see detectMatchedFilterSegments), with the summary of the
            whole input in the form of MatchedFilterDetector.summary.

        Arguments:
            img_handle: [FrameInterface] The input.
            ranges: [list] (first frame, end frame) of every segment.
            detectors: [list] The detector of every segment, None where it was not searched.
            cuts: [list] Frames where the detections of neighbouring segments were joined.
            sky_map: [SkyQualityMap] Sky quality of the whole input, or None.
            buffer: [SegmentBuffer] The frames in memory.
            timing: [dict] Time of the steps outside the detectors (s).

        Keyword arguments:
            removed_rows: [list] Frames of the measurements removed on sky clouded in the sky quality of the whole
                input, after the segments (see detectMatchedFilterSegments). Empty by default.
            removed_first: [list] First frames of the detections removed then. Empty by default.
            deviation_rejected: [int] Number of the detections rejected as wandering from a smooth path (see
                removeDeviatingDetections). 0 by default.
        """

        self.img_handle = img_handle
        self.ranges = ranges
        self.detectors = detectors
        self.cuts = cuts
        self.sky_map = sky_map
        self.buffer = buffer
        self.timing = timing

        # The measurements and the detections removed on sky clouded in the sky quality of the whole input
        self.removed_rows = list(removed_rows)
        self.removed_first = list(removed_first)
        self.deviation_rejected = int(deviation_rejected)


    def summary(self):
        """ Summary of the input, from the summaries of its segments. The detections and the measurements
            removed by clouds and the measured tracks are taken from the segment which their time belongs to (each
            overlap is split at its middle), so those in an overlap are counted once. """

        run = [(k, det) for k, det in enumerate(self.detectors) if det is not None]

        # Without a searched segment the input was not searched, as an input without enough stars
        if not run:
            return {}

        # The frames which belong to every segment: up to the middle of its overlaps with the neighbouring
        #   segments which were searched, the whole overlap otherwise
        searched = [det is not None for det in self.detectors]
        bounds = []
        for k, (first, end) in enumerate(self.ranges):
            lo = self.cuts[k - 1] if (k > 0) and searched[k - 1] else first
            hi = self.cuts[k] if (k < len(self.cuts)) and searched[k + 1] else end
            bounds.append((lo, hi))

        # The detections removed by clouds in the frames which belong to every segment
        removed = list(self.removed_first)
        for k, det in run:
            removed += [f for f in det.cloud_removed_frames if bounds[k][0] <= f < bounds[k][1]]

        # The time of every step summed over the segments, with the steps outside them
        timing = collections.defaultdict(float)
        for _, det in run:
            for key, value in det.timing.items():
                timing[key] += value
        for key, value in self.timing.items():
            timing[key] += value

        first = run[0][1] if run else None
        corrections = [det.aperture_correction for _, det in run if det.aperture_correction is not None]

        # The candidate tracks of the frames which belong to every segment
        tracks = [track for k, det in run for track in det.summary()['tracks']
                  if bounds[k][0] <= (track['first_frame'] + track['last_frame'])/2.0 < bounds[k][1]]

        return {
            'aperture_correction': float(np.median(corrections)) if corrections else None,
            'clear_sky_fraction': (round(float(np.mean(self.sky_map.clear)), 3) if self.sky_map is not None
                                   else None),
            'cloud_frames_removed': len(self.removed_rows) + sum(int(bounds[k][0] <= f < bounds[k][1])
                                                                 for k, det in run
                                                                 for f in det.cloud_removed_row_frames),
            'cloud_detections_removed': len(removed),
            'cloud_removed_utc': first.frameTimes(sorted(removed)) if first is not None else [],
            'tracks_dropped': int(sum(det.tracks_dropped for _, det in run)),
            'total_frames': int(self.img_handle.total_frames),
            'input_end_utc': first.inputEnd() if first is not None else None,
            'psf_sigma': next((det.psf_sigma for _, det in run if det.psf_sigma is not None), None),
            'gpu': any(bool(det.opts.use_gpu) for _, det in run),
            'timing_s': {k: round(v, 2) for k, v in timing.items()},
            'tracks': tracks,
            'linear_structure_rejected': sum(int(track.get('linear_structure', False)) for track in tracks),
            'deviation_rejected': self.deviation_rejected,
            'faint_measured': int(sum(det.faint_measured for _, det in run)),
            'segments': [{'first_frame': int(f), 'end_frame': int(e), 'searched': det is not None}
                         for (f, e), det in zip(self.ranges, self.detectors)],
            'frames_read': int(self.buffer.frames_read),
            'frames_read_outside': int(self.buffer.frames_outside),
        }



def splitAtGaps(detections, gaps):
    """ Cut the detections at the gaps in the recording, so that no detection runs across one (see
        detectAcrossGaps). The pieces which are no detections any more are removed later (see
        removeInvalidDetections).

    Arguments:
        detections: [list] Detections as [rho, theta, centroids], centroids rows [frame, x, y, ...].
        gaps: [list] The first frame after every gap.

    Return:
        [list] The detections, those across a gap cut into their pieces.
    """

    # Without gaps there is nothing to cut
    if not gaps:
        return detections

    out = []
    for rho, theta, cent in detections:

        # The part between the gaps which every row belongs to; a detection within one part is kept as it is
        part = np.searchsorted(gaps, cent[:, 0], side='right')
        if part[0] == part[-1]:
            out.append([rho, theta, cent])
            continue

        # Otherwise one piece per part
        for k in np.unique(part):
            out.append([rho, theta, cent[part == k]])

    return out


def joinWithinGaps(detector, detections, gaps, connected=None, cut=None):
    """ Join the pieces of tracks and remove the duplicates (see MatchedFilterDetector.joinDetections) within every
        part of the recording between its gaps, as a track is not followed across a gap.

    Arguments:
        detector: [MatchedFilterDetector] The detector whose limits are used.
        detections: [list] Detections as [rho, theta, centroids], none across a gap (see splitAtGaps).
        gaps: [list] The first frame after every gap.

    Keyword arguments:
        connected: [function] The light between two points in a frame (see joinDetections). None by default.
        cut: [set] The detections whose faint ends were already cut (see joinDetections). None by default.

    Return:
        [list] The joined detections.
    """

    # Without gaps, all detections are joined together
    if not gaps:
        return detector.joinDetections(detections, connected=connected, cut=cut)

    # The detections of every part, by the part of their first row
    parts = collections.defaultdict(list)
    for det in detections:
        parts[int(np.searchsorted(gaps, det[2][0, 0], side='right'))].append(det)

    return [det for k in sorted(parts) for det in detector.joinDetections(parts[k], connected=connected, cut=cut)]



def detectMatchedFilterSegments(img_handle, config, segment_frames, mask=None, dark=None, flat_struct=None,
                                platepar=None, reference_dir=None, input_name='', chunk_callback=None):
    """ Extract the stars and run the matched filter on an input which is read only once, in segments held in
        memory (see RMS.Formats.FrameBuffer). Every segment is searched as an input of its own; neighbouring
        segments share mf_segment_overlap frames, and the detections of an object found in both are joined (see
        stitchSegments). The sky reference and the log of the conditions are updated once, for the whole input.

    Arguments:
        img_handle: [FrameInterface] Frame-based input opened with detection=True.
        config: [Config] Configuration object.
        segment_frames: [int] Number of frames of a segment (see FrameBuffer.segmentFrames).

    Keyword arguments:
        mask, dark, flat_struct: Calibration, not binned (see detectMatchedFilter). None by default.
        platepar: [Platepar] Platepar of the camera (see detectMatchedFilter). None by default.
        reference_dir: [str] Directory of the clear-sky reference of the camera. None by default.
        input_name: [str] Name of the input (its path), for the log of the conditions. Empty by default.
        chunk_callback: [callable] Called for every chunk of the star extraction (e.g. to save its images), see
            extractStarsImgHandle. None by default.

    Return:
        (star_list, detections, run, buffer): [tuple] The stars of all chunks, the detections, the run of the
            segments (SegmentedRun, with the summary) and the buffer with the frames of the last segment.
    """

    t0 = time()
    chunk_frames = img_handle.chunk_frames
    unit = segmentUnit(config, img_handle)

    # The segments, which share the overlap (in whole blocks)
    overlap = -(-int(config.mf_segment_overlap)//unit)*unit
    if img_handle.total_frames <= segment_frames:

        # The whole input is one segment, which overlaps nothing
        overlap = 0
    # A longer overlap than half a segment is shortened, so only neighbouring segments share frames
    elif overlap > segmentOverlap(segment_frames, overlap, unit):
        log.warning('Matched filter: mf_segment_overlap {:d} is more than half of segments of {:d} frames, {:d} '
                    'used'.format(int(config.mf_segment_overlap), segment_frames,
                                  segmentOverlap(segment_frames, overlap, unit)))
        overlap = segmentOverlap(segment_frames, overlap, unit)
    ranges = segmentRanges(img_handle.total_frames, segment_frames, overlap, unit,
                           minimum=min(segmentMinimum(config, img_handle), (segment_frames//2)//unit*unit))
    buffer = SegmentBuffer(img_handle, max(end - first for first, end in ranges))
    log.info('Matched filter: {:d} segments of up to {:d} frames ({:.1f} min), overlap {:d} frames'.format(
        len(ranges), segment_frames, segment_frames/img_handle.fps/60.0, overlap))

    # The middle of every overlap, where the frames of the neighbouring segments meet (see stitchSegments)
    cuts = [(ranges[k + 1][0] + ranges[k][1])//2 for k in range(len(ranges) - 1)]
    overlaps = [(ranges[k + 1][0], ranges[k][1]) for k in range(len(ranges) - 1)]

    timing = collections.defaultdict(float)
    star_list, star_firsts, detectors, segment_detections = [], [], [], []
    done_chunks = 0
    persistence, persistence_frames = {}, 0
    for k, (first, end) in enumerate(ranges):

        # Read the new frames of the segment
        t1 = time()
        buffer.load(first, end)
        timing['read_segments'] += time() - t1

        # The stars of the whole chunks which were not extracted yet (the overlap was in the previous segment;
        #   the last segment goes to the last chunk of the input)
        t1 = time()
        end_chunk = img_handle.total_fr_chunks if (k == len(ranges) - 1) else end//chunk_frames
        if end_chunk > done_chunks:
            new_stars = extractStarsFrameInterface(buffer, config, chunk_frames=chunk_frames,
                flat_struct=flat_struct, dark=dark, mask=mask, save_calstars=False, chunk_callback=chunk_callback,
                chunk_range=(done_chunks, end_chunk))
            star_firsts += chunkFirstFrames(new_stars, buffer,
                                            frame_range=(done_chunks*chunk_frames, end_chunk*chunk_frames))
            star_list += new_stars
            done_chunks = end_chunk
        timing['stars'] += time() - t1

        # The stars of the chunks which the frames of this segment belong to (the frames after the last whole chunk
        #   of the input belong to it)
        last_chunk = (img_handle.total_fr_chunks - 1)*chunk_frames
        segment_stars = [entry for entry, f in zip(star_list, star_firsts)
                         if (f >= 0) and (f < end)
                         and ((img_handle.total_frames if f == last_chunk else f + chunk_frames) > first)]

        # Like the whole input, a segment without enough stars (clouds, twilight) is not searched
        max_stars = max([len(entry[1]) for entry in segment_stars] + [0])
        if max_stars < config.ff_min_stars:
            log.info('Matched filter: segment {:d} ({:d}-{:d}) not searched, {:d} < {:d} stars'.format(
                k + 1, first, end, max_stars, config.ff_min_stars))

            # The overlap with the segment before it is not counted here, so all counts of that one are carried
            if detectors and (detectors[-1] is not None) and (detectors[-1].persistence_full is not None):
                persistence = detectors[-1].persistence_full
                persistence_frames = detectors[-1].persistence_full_frames
            detectors.append(None)
            segment_detections.append([])
            continue

        # The segment is searched, with the sky quality from the stars of all chunks read so far (the intensities
        #   are scaled to the stars of the segment itself, see apertureCorrection). The smooth path of the
        #   detections is judged on the whole tracks, after the segments are joined
        detections, detector = detectMatchedFilter(buffer, config, mask=mask, dark=dark, flat_struct=flat_struct,
                                                   star_list=list(star_list), return_detector=True,
                                                   platepar=platepar, reference_dir=reference_dir,
                                                   input_name=input_name, frame_range=(first, end),
                                                   persistence_prior=persistence,
                                                   persistence_prior_frames=persistence_frames, sky_range=(0, end),
                                                   persistence_until=(ranges[k + 1][0] if k + 1 < len(ranges)
                                                                      else None),
                                                   track_deviation=False)
        detectors.append(detector)

        # The persistent pixels are judged over all segments so far (the runs of every overlap once), as over the
        #   whole input. A segment which was too short to be searched has no counts, those so far are kept
        if (detector is not None) and (detector.persistence_carry is not None):
            persistence, persistence_frames = detector.persistence_carry, detector.persistence_carry_frames
        segment_detections.append(detections)
        log.info('Matched filter: segment {:d}/{:d} ({:d}-{:d}): {:d} detections'.format(k + 1, len(ranges), first,
                                                                                       end, len(detections)))

    # Join the detections at the middle of every overlap
    searched = [det for det in detectors if (det is not None) and (det.psf_sigma is not None)]
    detections = []
    if searched:
        tolerance = 3*2.355*searched[0].psf_sigma*searched[0].det_bin
        detections = stitchSegments(segment_detections, cuts, overlaps, searched[0], tolerance,
                                    searched=[det is not None for det in detectors])

        # A track is not followed across a gap in the recording (see detectAcrossGaps): the detections stitched over
        #   a gap are cut there, and the pieces are joined only within the parts between the gaps
        gaps = sorted(set(g for det in detectors if det is not None for g in getattr(det, 'gaps', [])))
        detections = splitAtGaps(detections, gaps)
        detections = joinWithinGaps(searched[0], detections, gaps, connected=connectedOver(detectors),
                                    cut={id(det[2]) for dets in segment_detections for det in dets})

        # The pieces cut at the middle of an overlap and not joined again may be no detections any more
        detections = removeInvalidDetections(detections, searched[0])

        # The detections mostly on pixels which are persistent over the whole input are removed, as their hits
        #   would be without segments (the segments only removed the hits on pixels persistent for sure)
        n_before = len(detections)
        detections = removePersistentDetections(detections, persistence, config.mf_persistence, searched[0].det_bin)
        if len(detections) < n_before:
            log.info('Matched filter: {:d} detections on persistent pixels of the whole input removed'.format(
                n_before - len(detections)))

    # The sky reference and the log of the conditions are updated once, with the stars of the whole input
    t1 = time()
    sky_map = skyQuality(img_handle, config, star_list, platepar, reference_dir=reference_dir,
                         input_name=input_name) if searched else None
    timing['sky_input'] += time() - t1

    # The measurements on sky which is clouded in the sky quality of the whole input are removed (the sky quality
    #   of a segment, from the stars so far, can take as clear what the whole input shows as clouded)
    removed_rows, removed_first = [], []
    if (sky_map is not None) and detections:
        detections, removed_rows, removed_first = removeCloudedRows(detections, sky_map, searched[0])
        if removed_rows:
            log.info('Matched filter: {:d} frames of detections and {:d} detections on sky clouded in the whole '
                     'input removed'.format(len(removed_rows), len(removed_first)))

    # The whole tracks (joined over the segments) which wander sideways from one smooth path are no objects. A track
    #   is judged with the PSF of the segment which its middle frame belongs to (the segments are split at the
    #   middle of their overlaps), or of the nearest segment with a measured PSF
    bounds = [0] + cuts + [img_handle.total_frames]
    measured = [k for k, det in enumerate(detectors) if (det is not None) and (det.psf_sigma is not None)]

    def segmentFWHM(cent):
        """ The FWHM of the PSF of the segment which a detection belongs to.

        Arguments:
            cent: [ndarray] Rows of the detection, [frame, x, y, ...].

        Return:
            [float] FWHM (px, unbinned), None if no segment has a measured PSF.
        """

        if not measured:
            return None

        # The segment of the middle frame of the detection, or the nearest one with a measured PSF
        mid = (cent[0, 0] + cent[-1, 0])/2.0
        k = int(np.clip(np.searchsorted(bounds, mid, side='right') - 1, 0, len(detectors) - 1))
        det = detectors[min(measured, key=lambda j: abs(j - k))]

        return 2.355*det.psf_sigma*det.det_bin

    detections, deviating = removeDeviatingDetections(detections, segmentFWHM, config.mf_track_deviation_fwhm,
                                                      config.mf_track_deviation_length)

    timing['segments_total'] = time() - t0
    log.info('Matched filter: {:d} detections in {:d} segments, {:d} frames read once ({:d} outside the '
             'segments), {:.1f} min'.format(len(detections), len(ranges), buffer.frames_read, buffer.frames_outside,
                                            timing['segments_total']/60.0))

    run = SegmentedRun(img_handle, ranges, detectors, cuts, sky_map, buffer, timing, removed_rows=removed_rows,
                       removed_first=removed_first, deviation_rejected=len(deviating))

    # The times of the detections of the whole input are corrected for a rolling shutter
    detections = correctRollingShutter(detections, img_handle, config)

    return star_list, detections, run, buffer



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

    os.makedirs(output_dir, exist_ok=True)

    # The FTPdetectinfo and CALSTARS, written as for the normal detection but with the suffix of the matched
    #   filter, so the two never overwrite each other
    _, _, ftp_name = saveResultsFrameInterface(star_list, detections, img_handle, config,
                                               chunk_frames=chunk_frames, output_suffix=MATCHED_FILTER_SUFFIX,
                                               output_dir=output_dir, chunk_images=chunk_images)
    ftp_path = os.path.join(output_dir, ftp_name)

    # The recalibration fits the platepar to the stars of every chunk and computes the coordinates and the
    #   magnitudes of the detections. It needs the platepar in the output directory
    if platepar_path and star_list and detections:
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

    # The merged file of an earlier report of the night is replaced
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

    meteor_list = []
    fps = None
    found = False
    calibrated = True
    for mf_dir in dirs:

        # Only finished results (a failed or interrupted file has no done file)
        if not os.path.isfile(os.path.join(mf_dir, DONE_NAME)):
            continue

        # The recalibration keeps a backup of the uncalibrated file, which is not merged
        for file_name in sorted(os.listdir(mf_dir)):
            if not FTPdetectinfo.validDefaultFTPdetectinfo(file_name):
                continue

            # Every detection is converted from the format of the reader to the format of the writer (without the
            #   calibration status column of the measurements)
            found = True
            for entry in FTPdetectinfo.readFTPdetectinfo(mf_dir, file_name):
                ff_name, _, meteor_No, _, meteor_fps, _, _, _, _, rho, phi, meteor_meas = entry
                meteor_list.append([ff_name, meteor_No, rho, phi, [line[1:] for line in meteor_meas],
                                    meteor_fps])
                fps = meteor_fps
                calibrated = calibrated and all(line[0] == 1 for line in meteor_meas)

    if not found:
        return None

    # The detections are sorted by file name (i.e. time) and detection number. The merged file is calibrated if
    #   all the merged detections are (the magnitudes of an uncalibrated one are placeholders)
    os.makedirs(out_dir, exist_ok=True)
    FTPdetectinfo.writeFTPdetectinfo(sorted(meteor_list, key=lambda entry: (entry[0], entry[1])), out_dir,
        ftp_name, out_dir, config.stationID, fps if fps else config.fps,
        calibration=('Merged from calibrated files' if (calibrated and meteor_list) else None),
        celestial_coords_given=True)

    return os.path.join(out_dir, ftp_name)



def processFile(file_path, config, output_dir, platepar_path=None, dark_path=None, flat_path=None,
    mask_path=None, extract_stars=True, reference_dir=None):
    """ Run the matched-filter detection on one file and save the FTPdetectinfo (with the suffix "mf"), and
        the CALSTARS if the stars are extracted. With a platepar, the detections are recalibrated.

    Arguments:
        file_path: [str] Input file, or a directory of FITS frames.
        config: [Config] Configuration object.
        output_dir: [str] Output directory.

    Keyword arguments:
        platepar_path: [str] Platepar, copied to the output directory for the recalibration, and used for the
            clouded sky (mf_cloud_filter). None by default.
        dark_path, flat_path, mask_path: [str] Calibration images. None by default.
        extract_stars: [bool] Extract the stars (needed for the recalibration). True by default.
        reference_dir: [str] Directory of the clear-sky reference of the camera, used and updated with this file
            (see skyQuality). None by default (only this file is used).

    Return:
        [dict] Summary of the processing (also saved as the done file in the output directory), with the path
            of the FTPdetectinfo.
    """

    t0 = time()

    # The calibration images given on the command line replace the ones of the config
    config.use_dark = dark_path is not None
    config.use_flat = flat_path is not None
    if dark_path:
        config.dark_file = os.path.abspath(dark_path)
    if flat_path:
        config.flat_file = os.path.abspath(flat_path)
    if mask_path:
        config.mask_file = os.path.abspath(mask_path)

    # Open the input and load the calibration images (the dark, flat and mask)
    img_handle = detectInputType(file_path, config, detection=True, preload_video=True)
    if img_handle.input_type == 'ff':
        raise ValueError('FF files have no frames to search: {:s}'.format(file_path))
    # The calibration images are in the directory of the input (for a directory of FITS frames, in the directory
    #   itself), as for the monitor
    mask, dark, flat_struct = loadImageCalibration(img_handle.dir_path, config, dtype=img_handle.ff.dtype,
        byteswap=img_handle.byteswap)

    # The stars of the chunks of frames, for the recalibration and the aperture correction
    star_list = []
    if extract_stars:
        star_list = extractStarsFrameInterface(img_handle, config, flat_struct=flat_struct, dark=dark, mask=mask,
                                               save_calstars=False)

    # The platepar, for the clouded sky
    platepar = None
    if platepar_path:
        platepar = Platepar()
        if platepar.read(platepar_path) is False:
            log.warning('Matched filter: the platepar can\'t be read: {:s}'.format(platepar_path))
            platepar = None

    # Detect, save the detections and recalibrate them
    detections, detector = detectMatchedFilter(img_handle, config, mask=mask, dark=dark, flat_struct=flat_struct,
                                               star_list=star_list, return_detector=True, platepar=platepar,
                                               reference_dir=reference_dir, input_name=os.path.abspath(file_path))

    ftp_path = saveMatchedFilterResults(detections, star_list, img_handle, config, output_dir,
                                        platepar_path=platepar_path,
                                        chunk_frames=getattr(img_handle, 'chunk_frames', 128))

    # The done file is written last, so a file with a done file was processed completely
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

    # The summary of the detector (its tracks, the timing, etc.) and the given items
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

        # Skip the files which were already processed
        out = os.path.join(output_dir, name)
        out_dirs.append(out)
        if os.path.isfile(os.path.join(out, DONE_NAME)) and not force:
            log.info('Matched filter: already processed, skipped: {:s}'.format(path))
            continue

        # Partial results of an interrupted run are not kept
        shutil.rmtree(out, ignore_errors=True)
        try:
            summary = processFile(path, config, out, reference_dir=output_dir, **kwargs)
            log.info('Matched filter: {:d} detections in {:s} ({:.0f} s)'.format(summary['detections'], path,
                                                                               summary['processing_time_s']))
        except Exception:
            log.error('Matched filter: processing failed for {:s}:\n{:s}'.format(path, traceback.format_exc()))
            failed.append(path)

    # Merge the detections of all files (also of the ones processed in earlier runs) into one FTPdetectinfo
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

    # The file names without the extension, and how many files have every name
    stems = [os.path.splitext(os.path.basename(path))[0] for path in files]
    counts = collections.Counter(stems)

    return [stem if counts[stem] == 1 else '{:s}_{:s}'.format(stem,
            hashlib.sha1(os.path.abspath(path).encode('utf-8')).hexdigest()[:8]) for stem, path in zip(stems, files)]



def findInputFiles(paths):
    """ Input files from a list of files and directories (searched recursively for video files and for
        directories of FITS frames, which are one input each).

    Arguments:
        paths: [list] Files and directories.

    Return:
        [list] Files (and directories of FITS frames), sorted, without duplicates. The FF files of the normal
            processing are FITS files too, but not frames: their directories are not inputs.
    """

    files = []
    for path in paths:

        # The video files in a directory and all its subdirectories, and the directories with FITS frames
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                files += [os.path.join(root, name) for name in names if name.lower().endswith(INPUT_EXTENSIONS)]
                if any(name.lower().endswith(FITS_EXTENSIONS) and not validFFName(name) for name in names):
                    files.append(root)
        else:
            files.append(path)

    return sorted(set(os.path.abspath(f) for f in files))



if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(description="Matched-filter detection of faint moving objects in "
        "frame-based video (e.g. .vid files). Every input file gets its own output directory, with an "
        "FTPdetectinfo with the suffix 'mf' and a done file; files with a done file are skipped, so an interrupted "
        "run can be started again. The detections of all files are merged into one FTPdetectinfo in the output "
        "directory.")

    arg_parser.add_argument('input', nargs='+', help='Input files or directories (searched for video files and '
                            'for directories of FITS frames, which are one input each).')
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

    #########################

    # Load the config and override its options with the ones from the command line
    config = cr.parse(args.config)
    if args.no_velocity_search:
        config.mf_velocity_search = False
    if args.gpu:
        config.mf_gpu = args.gpu
    if args.threads is not None:
        config.mf_threads = args.threads

    # The log is saved in the logs directory of the output directory
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    config.data_dir = output_dir
    config.log_dir = 'logs'
    LoggingManager().initLogging(config, 'matched_filter_')

    # Find the input files and process them
    files = findInputFiles(args.input)
    if not files:
        log.error('Matched filter: no input files found in {:s}'.format(', '.join(args.input)))
        sys.exit(1)

    failed = processFiles(files, config, output_dir, platepar_path=args.platepar,
                          dark_path=args.dark, flat_path=args.flat, mask_path=args.mask,
                          extract_stars=not args.no_stars, force=args.force)
    sys.exit(1 if failed else 0)
