""" Settings tuned on a reference camera, scaled to the frame rate and the pixel scale of the input.

Many settings of the detection and the calibration are counted in frames (e.g. the runs of the matched filter, the
shortest duration of a detection) or given in pixels (e.g. the margin from the image edge, the size of the star
search window). They are tuned for one camera, and a camera recording at another frame rate or binned differently
needs all of them changed. With the reference frame rate (mf_reference_fps) and the reference pixel scale
(mf_reference_scale, arcsec/px) of the camera the settings were tuned for, the settings are scaled to the input:
the settings in frames keep their durations, and the settings in pixels their angular sizes. Without a reference
(the default) the settings are used as they are.

Only the standard library is used (and the config), so the module can be imported in the main process of the
monitor, which must not import the matched filter (see RMS.MonitorProcessFrameInterface).
"""

from __future__ import print_function, division, absolute_import

import copy
import logging
import math


log = logging.getLogger("logger")


# Frames per chunk of the star extraction of the monitor (--chunk_frames) at the reference frame rate, by default
DEFAULT_CHUNK_FRAMES = 128

# Settings counted in frames, which keep their durations (0 turns some of them off and stays 0). The runs of the
#   matched filter are scaled together (see scaleRuns)
FRAME_OPTIONS = ('mf_block_frames', 'mf_min_frames', 'mf_link_max_gap', 'mf_segment_overlap')

# Sizes in whole pixels, which keep their angular sizes: the margins from the image edge of the matched filter and
#   of the detection, and the geometry of the star extraction
PIXEL_SIZE_OPTIONS = ('mf_edge_margin', 'detection_border', 'border', 'neighborhood_size', 'segment_radius')

# Distances in pixels which keep their angular sizes: the largest position error of a measurement of the matched
#   filter. (The star residuals with which a recalibrated platepar is accepted are scaled with the pixel scale of the
#   platepar wherever it is fitted, see recalibrationThresholds)
PIXEL_OPTIONS = ('mf_max_pos_error',)


def referenceEnabled(config):
    """ Whether the config gives a reference camera (mf_reference_fps or mf_reference_scale). Without one, the
        settings are used as they are, and the platepars and the references of the camera are used for frames of
        any size as before.

    Arguments:
        config: [Config] Configuration object.

    Return:
        [bool]
    """

    return (config.mf_reference_fps > 0) or (config.mf_reference_scale > 0)


def roundHalfUp(value):
    """ The nearest integer, halves rounded up (round() rounds them to the even integer). """

    return int(math.floor(value + 0.5))


def scaleFrames(n, factor):
    """ A number of frames scaled by the ratio of the frame rates, at least 1 frame. 0 (or less, off) stays.

    Arguments:
        n: [int] Number of frames at the reference frame rate.
        factor: [float] Frame rate of the input over the reference frame rate.

    Return:
        [int] Number of frames at the frame rate of the input.
    """

    if (n <= 0) or (factor == 1.0):
        return n

    return max(roundHalfUp(n*factor), 1)


def scaleRuns(runs, factor):
    """ The lengths of the runs of the matched filter scaled by the ratio of the frame rates, together: the runs are
        whole multiples of their common unit (the greatest common divisor), and the unit is scaled, so the runs keep
        their ratios. A block of frames holds a whole number of every run (see
        RMS.MatchedFilterDetection.matchedFilterBlockFrames), and runs rounded each on its own (e.g. 19, 38 and 77
        frames for 8, 16 and 32 at 2.4 times the frame rate) would need a block of tens of thousands of frames.

    Arguments:
        runs: [list] Lengths of the runs at the reference frame rate (frames). 0 (a run which is off) stays 0.
        factor: [float] Frame rate of the input over the reference frame rate.

    Return:
        [list] Lengths of the runs at the frame rate of the input (frames).
    """

    lengths = [int(n) for n in runs if n > 0]
    if (not lengths) or (factor == 1.0):
        return list(runs)

    # The common unit of the runs, scaled to at least 1 frame, and every run as the same multiple of it
    unit = 0
    for n in lengths:
        unit = math.gcd(unit, n)
    scaled_unit = max(roundHalfUp(unit*factor), 1)

    return [(int(n)//unit)*scaled_unit if n > 0 else n for n in runs]


def scalePowerOfTwo(n, factor):
    """ A number of frames which is a power of two (e.g. the frames of a measurement), scaled by the ratio of the
        frame rates and rounded up to a power of two, so its duration is at least the duration at the reference.

    Arguments:
        n: [int] Number of frames at the reference frame rate.
        factor: [float] Frame rate of the input over the reference frame rate.

    Return:
        [int] Number of frames at the frame rate of the input.
    """

    if (n <= 0) or (factor == 1.0):
        return n

    # The scaled number of frames, rounded first, so a factor which is a ratio of powers of two up to the rounding
    #   of a measured frame rate gives the exact power of two
    return 2**int(math.ceil(math.log2(max(roundHalfUp(n*factor), 1))))


def scaleSize(size, factor):
    """ A size in whole pixels scaled by the ratio of the pixel scales, at least 1 px.

    Arguments:
        size: [int] Size at the reference pixel scale (px).
        factor: [float] Reference pixel scale over the pixel scale of the input (arcsec/px over arcsec/px).

    Return:
        [int] Size at the pixel scale of the input (px).
    """

    if factor == 1.0:
        return size

    return max(roundHalfUp(size*factor), 1)


def inputScale(config, width, height, platepar=None):
    """ The pixel scale of the input: from the platepar if it is given and of the size of the frames, otherwise
        from the field of view in the config over the size of the frames (as the speed limits of the matched
        filter, see RMS.MatchedFilterDetection.MatchedFilterOptions).

    Arguments:
        config: [Config] Configuration object.
        width: [int] Width of the frames (px).
        height: [int] Height of the frames (px).

    Keyword arguments:
        platepar: [Platepar] Platepar of the input. None by default.

    Return:
        [float] Pixel scale (arcsec/px).
    """

    if (platepar is not None) and (platepar.F_scale > 0) and (platepar.X_res == width) \
            and (platepar.Y_res == height):
        return 3600.0/platepar.F_scale

    return 3600.0*(config.fov_h/float(height) + config.fov_w/float(width))/2.0


def referenceFactors(config, fps, scale):
    """ The factors which scale the settings tuned on the reference camera to the input.

    Arguments:
        config: [Config] Configuration object.
        fps: [float] Frame rate of the input. None or 0 if not known (the frame counts are not scaled).
        scale: [float] Pixel scale of the input (arcsec/px). None or 0 if not known (the sizes are not scaled).

    Return:
        (frame_factor, pixel_factor): [tuple] The frame rate of the input over the reference frame rate, which
            multiplies the frame counts, and the reference pixel scale over the pixel scale of the input, which
            multiplies the sizes in pixels. 1 without a reference.
    """

    frame_factor = 1.0
    if (config.mf_reference_fps > 0) and fps and (fps > 0):
        frame_factor = fps/float(config.mf_reference_fps)

    pixel_factor = 1.0
    if (config.mf_reference_scale > 0) and scale and (scale > 0):
        pixel_factor = float(config.mf_reference_scale)/scale

    return frame_factor, pixel_factor


def scaleToReference(config, fps, scale, size=None, logger=None):
    """ The settings tuned on the reference camera scaled to the frame rate and the pixel scale of the input (see
        the description of the module). The config is copied, and the copy remembers the factors (mf_frame_factor
        and mf_pixel_factor, which the matched filter uses for its constants), so a scaled config is not scaled
        again. The frame rate and the size of the frames of the input replace those of the config (which a camera
        recording differently would otherwise have to change too, e.g. the automatic calibration takes the size of
        the frames from the config).

    Arguments:
        config: [Config] Configuration object.
        fps: [float] Frame rate of the input (e.g. measured from the frame times). None if not known.
        scale: [float] Pixel scale of the input (arcsec/px), see inputScale. None if not known.

    Keyword arguments:
        size: [tuple] (width, height) of the unbinned frames of the input (px). None by default (not known).
        logger: [Logger] Logger for the scaled values. The module logger by default.

    Return:
        [Config] The config with the scaled settings (the given config without a reference).
    """

    # A config which was scaled already is used as it is, as is a config without a reference
    if getattr(config, 'mf_reference_applied', False) or (not referenceEnabled(config)):
        return config

    frame_factor, pixel_factor = referenceFactors(config, fps, scale)

    # The settings are changed on a copy, which remembers the factors. The lists of the config are replaced, not
    #   changed, so a shallow copy leaves the given config as it is
    config = copy.copy(config)
    config.mf_frame_factor = frame_factor
    config.mf_pixel_factor = pixel_factor
    config.mf_reference_applied = True

    # The frame rate and the size of the input
    if fps:
        config.fps = fps
    if size is not None:
        config.width, config.height = int(size[0]), int(size[1])

    # The frame counts keep their durations. The runs (also the run of the slow objects) are scaled together, so
    #   they keep their ratios and a block holds a whole number of all of them (see scaleRuns); the frames of a
    #   measurement double until the position error is small enough (see the matched filter), so their largest
    #   number stays a power of two
    runs = scaleRuns(list(config.mf_run_frames) + [config.mf_slow_run_frames], frame_factor)
    config.mf_run_frames, config.mf_slow_run_frames = runs[:-1], runs[-1]
    for name in FRAME_OPTIONS:
        setattr(config, name, scaleFrames(getattr(config, name), frame_factor))
    config.mf_max_measure_frames = scalePowerOfTwo(config.mf_max_measure_frames, frame_factor)

    # The sizes keep their angular sizes
    for name in PIXEL_SIZE_OPTIONS:
        setattr(config, name, scaleSize(getattr(config, name), pixel_factor))
    for name in PIXEL_OPTIONS:
        setattr(config, name, getattr(config, name)*pixel_factor)

    # A tier of the velocity search covers the speeds up to mf_tier_speed binned px per frame: on bins of the same
    #   angular size, the same angular speed is fewer binned px per frame at a higher frame rate
    config.mf_tier_speed = config.mf_tier_speed/frame_factor

    (logger or log).info("Settings scaled from the reference camera ({:.4g} fps, {:.4g} arcsec/px) to the input "
        "({} fps, {} arcsec/px): frames x{:.4f}, pixels x{:.4f}: runs {}, slow run {}, block {}, min frames {}, "
        "link gap {}, measurement frames {}, segment overlap {}, edge margin {}, detection border {}, max position "
        "error {:.3f} px, tier speed {:.3f}, star extraction border {}, neighborhood {}, segment radius {}".format(
            config.mf_reference_fps, config.mf_reference_scale,
            '{:.4g}'.format(fps) if fps else 'unknown', '{:.4g}'.format(scale) if scale else 'unknown',
            frame_factor, pixel_factor, config.mf_run_frames, config.mf_slow_run_frames, config.mf_block_frames,
            config.mf_min_frames, config.mf_link_max_gap, config.mf_max_measure_frames, config.mf_segment_overlap,
            config.mf_edge_margin, config.detection_border, config.mf_max_pos_error, config.mf_tier_speed,
            config.border, config.neighborhood_size, config.segment_radius))

    return config


def recalibrationThresholds(config, platepar):
    """ The largest mean star residuals (px) with which a fitted platepar is accepted (dist_check_threshold) and
        with which the quick recalibration is used (dist_check_quick_threshold), in the pixels of the platepar:
        with a reference pixel scale they keep their angular sizes at the pixel scale of the platepar. The
        platepar is known wherever a fit is checked (also outside the processing of a file, e.g. in the night
        report and the automatic calibration), so the thresholds are scaled here and not in the config.

    Arguments:
        config: [Config] Configuration object.
        platepar: [Platepar] The platepar which is fitted. None if not known (the thresholds are not scaled).

    Return:
        (dist_threshold, quick_threshold): [tuple] Thresholds (px).
    """

    factor = 1.0
    if (config.mf_reference_scale > 0) and (platepar is not None) and (platepar.F_scale > 0):
        factor = config.mf_reference_scale/(3600.0/platepar.F_scale)

    return config.dist_check_threshold*factor, config.dist_check_quick_threshold*factor


def scaleChunkFrames(chunk_frames, config, fps):
    """ The frames per chunk of the star extraction of the monitor: at its default (DEFAULT_CHUNK_FRAMES), with a
        reference frame rate, the chunks keep about the duration they have at the reference frame rate, so the
        stars of a chunk are as deep as at the reference. The chunks divide the blocks of the matched filter (as
        the default chunks divide the default blocks), so an input read once in segments, which hold whole blocks
        and whole chunks (see RMS.Formats.FrameBuffer.segmentUnit), is not cut into segments of tens of thousands
        of frames: the chunk is the whole fraction of the scaled block nearest to the scaled chunk. A chunk length
        which is given is used as it is.

    Arguments:
        chunk_frames: [int] Frames per chunk given to the monitor.
        config: [Config] Configuration object.
        fps: [float] Frame rate of the input. None if not known.

    Return:
        [int] Frames per chunk.
    """

    if (chunk_frames != DEFAULT_CHUNK_FRAMES) or (config.mf_reference_fps <= 0) or (not fps):
        return chunk_frames

    frame_factor, _ = referenceFactors(config, fps, None)
    target = chunk_frames*frame_factor

    # The block of the matched filter with the scaled frame counts (only the frames are needed, the pixel scale is
    #   not known yet). The matched filter is imported here, in the worker which processes the file, and not at the
    #   top of the module, which the main process of the monitor imports (see RMS.MonitorProcessFrameInterface)
    from RMS.MatchedFilterDetection import matchedFilterBlockFrames
    block = matchedFilterBlockFrames(scaleToReference(config, fps, None, logger=_QUIET))

    # The divisor of the block which gives the chunk nearest to the scaled length
    divisors = [n for n in range(1, block + 1) if block % n == 0]
    n = min(divisors, key=lambda d: abs(block/float(d) - target))

    return block//n


class _QuietLogger(object):
    """ A logger which logs nothing, for the scaling of the settings whose values are logged elsewhere. """

    def info(self, *args, **kwargs):
        pass


_QUIET = _QuietLogger()
