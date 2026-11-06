""" Shape of the light of a moving object, measured on the raw frames of the recording.

A point source moving across the sky leaves the same image in every frame: the point spread function (PSF) of the
camera, shifted along the track. When the frames of the track are shifted so that the object sits at the same place
and are averaged, a point source gives a compact image which is as wide as the PSF across the direction of motion.
An extended structure which moves with a similar speed, e.g. a cloud lit by the Moon and carried by the wind, gives
an image which is much wider than the PSF across the motion, as its light is spread over a larger area.

The measurement of a track:
    1) The background is the median of frames spread over the time of the track and some time before and after it,
        as the object stays on any pixel only briefly.
    2) The frames of the track, minus the background, are shifted (to a fraction of a pixel) so that the object
        is at the centre of a small window, and averaged.
    3) The averaged window is rotated so that the motion is along the x axis, and the profiles of the light across
        and along the motion (averaged over a 5 px strip through the centre) are fitted with a Gaussian.
    4) The spread is the Gaussian sigma across the motion divided by the PSF sigma: about 1 for a point source,
        clearly larger for an extended structure.

Only the raw frames of the recording are used, never compressed video (mp4, mkv, ...) or FF files, as the
compression smooths and blurs faint extended structures. The frames are read through the frame interface
(RMS.Formats.FrameInterface), so every raw input of RMS can be measured (.vid files, frame block files, image frames
such as UWO PNG frame dumps), and the frames are timed exactly as the processing timed them (frame_time_offset, the
point of the exposure of frame blocks, the repair of the timestamps). RawFrames joins the inputs around a track; any
object with the same methods (framesBetween, frame, height, width) can be measured instead.

Command line: python -m RMS.Routines.TrackShape FTPdetectinfo_... raw_path [-c config] [--psf_sigma 0.7]
writes a CSV file with the shape of every detection in the FTPdetectinfo file.
"""

from __future__ import print_function, division, absolute_import

import argparse
import csv
import datetime
import glob
import json
import math
import os
import re
import time

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit


# Half size of the window around the object in which the frames are averaged (px)
WINDOW_HALF = 20

# Margin around the extent of the track in which the background is computed (px)
BOX_MARGIN = 40

# Most frames of the track which are averaged (a longer track is subsampled evenly)
MAX_TRACK_FRAMES = 150

# Number of frames of which the median is the background
BACKGROUND_FRAMES = 40

# Shortest time before and after the track from which the frames of the background are taken (s), and the shortest
#   duration of a track used for that time (s)
MIN_BACKGROUND_PAD = 3.0
MIN_DURATION = 0.5

# Fewest frames averaged for a measurement (fewer, e.g. a track along the edge of the frame, show the shape of the
#   object in single frames, not along the track)
MIN_STACK_FRAMES = 3

# Half width of the strip through the centre over which the profiles are averaged (px)
STRIP_HALF = 2

# Starting sigma and the limits of the sigma (px) and of the centre (px) of the Gaussian fit of a profile
FIT_SIGMA_START = 1.5
FIT_SIGMA_LIMITS = (0.3, 15.0)
FIT_CENTRE_LIMIT = 3.0

# Columns of the CSV file of the command line
CSV_COLUMNS = ['ff_name', 'meteor_no', 'n_points', 'time_start_utc', 'time_end_utc', 'raw_file', 'psf_sigma',
               'sigma_cross', 'sigma_along', 'spread', 'snr', 'n_frames', 'n_stack', 'ok', 'flag', 'time_s']


### Raw frames, read through the frame interface ###

# File names which carry the time of the beginning of a raw recording: frame blocks (..._YYYYMMDD_hhmmss_ffffff.fblk)
#   and UWO .vid files (dump_<Unix time in hex>_<camera>.vid)
BLOCK_NAME_TIME = re.compile(r'(\d{8})_(\d{6})_(\d{6})\.fblk$', re.IGNORECASE)
VID_NAME_TIME = re.compile(r'_([0-9A-Fa-f]{8})_[^_]*\.vid$')

# Extensions of the raw recordings searched for in a directory
RAW_FILE_EXTENSIONS = ('.fblk', '.vid')

# Frames read when an input is opened (the frame interface loads a first chunk of frames)
OPEN_CHUNK_FRAMES = 2


def frameTimeOfInput(img_handle, frame_no):
    """ Time of a frame of an input of the frame interface, as the processing timed it (with frame_time_offset and
        the repair of the timestamps the input applies).

    Arguments:
        img_handle: [InputType] Input of the frame interface.
        frame_no: [int] Frame number of the input.

    Return:
        [float] Unix time (s).
    """

    from RMS.Astrometry.Conversions import datetime2UnixTime

    # A frame table of the input is used directly, else the time of the frame as the input gives it
    table = getattr(img_handle, 'frame_unix_times', None)
    if table is not None:
        return float(table[frame_no])

    return datetime2UnixTime(img_handle.currentFrameTime(frame_no=int(frame_no), dt_obj=True))



def closeInput(img_handle):
    """ Close the files of an input of the frame interface (the .vid file, the memory map of a frame block).

    Arguments:
        img_handle: [InputType] Input of the frame interface.
    """

    # The open file of a .vid input
    vid_file = getattr(img_handle, 'vid_file', None)
    if vid_file is not None:
        vid_file.close()

    # The memory map of a frame block
    reader = getattr(img_handle, 'reader', None)
    if reader is not None:
        reader.close()



class RawFrames(object):
    def __init__(self, img_handles, names=None):
        """ The frames of one or more inputs of the frame interface (RMS.Formats.FrameInterface), numbered
            continuously in time order, with the time of every frame as the processing timed it. The frames are
            read one at a time as they are needed, and their times only once.

        Arguments:
            img_handles: [list] Inputs of the frame interface, opened on the raw frames (no binning).

        Keyword arguments:
            names: [list] Names of the files or directories of the inputs, for the output. None by default.
        """

        # The inputs in the order of the times of their first frames
        if names is None:
            names = [getattr(h, 'file_name', '') or getattr(h, 'dir_path', '') for h in img_handles]
        starts = [frameTimeOfInput(h, 0) for h in img_handles]
        order = np.argsort(starts, kind='stable')
        self.img_handles = [img_handles[k] for k in order]
        self.names = [os.path.basename(str(names[k]).rstrip(os.sep)) for k in order]

        # The frames of every input follow those of the inputs before it
        counts = [int(h.total_frames) for h in self.img_handles]
        self.first_frames = np.concatenate([[0], np.cumsum(counts)]).astype(int)
        self.n_frames = int(self.first_frames[-1])

        # The size of the frames (all inputs of one camera have the same)
        self.height = int(self.img_handles[0].nrows) if self.img_handles else 0
        self.width = int(self.img_handles[0].ncols) if self.img_handles else 0

        # The times of the frames which were read
        self._times = {}


    def _input(self, i):
        """ The input of frame i and the number of the frame in it.

        Arguments:
            i: [int] Frame number.

        Return:
            (img_handle, frame_no): [tuple] The input and its frame.
        """

        # Only the frames of the inputs exist
        if (i < 0) or (i >= self.n_frames):
            raise IndexError('Frame {} is not in the raw frames'.format(i))

        # The last input which begins at or before the frame
        k = int(np.searchsorted(self.first_frames, i, side='right')) - 1

        return self.img_handles[k], int(i - self.first_frames[k])


    def frameTime(self, i):
        """ Time of frame i (Unix s).

        Arguments:
            i: [int] Frame number.

        Return:
            [float] Time of the frame.
        """

        # The time is read from the input only once
        if i not in self._times:
            img_handle, frame_no = self._input(int(i))
            self._times[i] = frameTimeOfInput(img_handle, frame_no)

        return self._times[i]


    def framesBetween(self, t0, t1):
        """ The frames whose times are between t0 and t1 (inclusive). The frames of every input are in time order,
            so the first one at or after t0 is found by a binary search over the times of its frames.

        Arguments:
            t0: [float] Beginning (Unix s).
            t1: [float] End (Unix s).

        Return:
            (indices, times): [tuple of ndarrays] Frame numbers and times, in time order.
        """

        indices, times = [], []
        for k in range(len(self.img_handles)):

            first, end = int(self.first_frames[k]), int(self.first_frames[k + 1])

            # The first frame of the input at or after t0
            lo, hi = first, end
            while lo < hi:
                mid = (lo + hi)//2
                if self.frameTime(mid) < t0:
                    lo = mid + 1
                else:
                    hi = mid

            # The frames which follow it, up to t1
            i = lo
            while (i < end) and (self.frameTime(i) <= t1):
                indices.append(i)
                times.append(self.frameTime(i))
                i += 1

        # Inputs which overlap in time are merged in time order
        indices, times = np.array(indices, dtype=int), np.array(times, dtype=np.float64)
        order = np.argsort(times, kind='stable')

        return indices[order], times[order]


    def frame(self, i):
        """ Frame i as float32, as the input reads it.

        Arguments:
            i: [int] Frame number.

        Return:
            [ndarray] The frame.
        """

        # The frame is read by its input, as the processing reads it
        img_handle, frame_no = self._input(int(i))
        img_handle.setFrame(frame_no)

        return np.asarray(img_handle.loadFrame(), dtype=np.float32)


    def close(self):
        """ Close the files of the inputs. """

        for img_handle in self.img_handles:
            closeInput(img_handle)



def rawFileStart(path):
    """ The time of the beginning of a raw recording from its name: a frame block
        (e.g. XX0001_20261027_101258_485479.fblk) or a UWO .vid file (dump_6AEAEB15_02G.vid, the Unix time in hex).
        None if the name has no time.

    Arguments:
        path: [str] Path to the file.

    Return:
        [float] Unix time (s), or None.
    """

    name = os.path.basename(path)

    # A frame block: the date and time of its first frame
    match = BLOCK_NAME_TIME.search(name)
    if match is not None:
        dt = datetime.datetime.strptime(''.join(match.groups()), '%Y%m%d%H%M%S%f')
        return dt.replace(tzinfo=datetime.timezone.utc).timestamp()

    # A .vid file: the Unix time of its beginning in hex
    match = VID_NAME_TIME.search(name)
    if match is not None:
        return float(int(match.group(1), 16))

    return None



def rawFilesAround(raw_dir, t0, t1, margin=0.0):
    """ The raw recording files of a directory (.fblk and .vid, searched recursively) which can hold frames
        between t0 and t1: the files which begin before t1 and whose next file begins after t0, from the times in
        their names. Files without a time in the name are all taken (their own times decide which frames are used).

    Arguments:
        raw_dir: [str] Directory of the raw files.
        t0: [float] Beginning (Unix s).
        t1: [float] End (Unix s).

    Keyword arguments:
        margin: [float] Widening of the interval on both sides (s), e.g. for a correction of the frame times.
            0 by default.

    Return:
        [list] Paths of the files, in time order.
    """

    paths = sorted(p for ext in RAW_FILE_EXTENSIONS
                   for p in glob.glob(os.path.join(raw_dir, '**', '*' + ext), recursive=True))

    # The files named by their time, in time order, and the others
    named = sorted((rawFileStart(p), p) for p in paths if rawFileStart(p) is not None)
    unnamed = [p for p in paths if rawFileStart(p) is None]

    # A file can hold frames of the interval if it begins before its end and the next file begins after its
    #   beginning
    t0, t1 = t0 - margin, t1 + margin
    selected = []
    for k, (start, path) in enumerate(named):
        next_start = named[k + 1][0] if k + 1 < len(named) else np.inf
        if (start <= t1) and (next_start >= t0):
            selected.append(path)

    return selected + unnamed



def openInput(path, config):
    """ Open a raw recording with the frame interface, on its raw frames (no binning). Only raw frames are
        accepted: compressed video smooths and blurs faint extended structures, and FF files hold images of
        chunks of frames, not the frames.

    Arguments:
        path: [str] A raw file (.vid, .fblk, ...) or a directory of images (e.g. UWO PNG frame dumps).
        config: [Config] Configuration of the camera.

    Return:
        [InputType] The input, None if the frame interface doesn't recognise it.
    """

    from RMS.Formats.FFfile import validFFName
    from RMS.Formats.FRbin import validFRName
    from RMS.Formats.FrameInterface import checkIfVideoFile, detectInputType

    # Compressed video and FF files are refused before anything is read
    name = os.path.basename(path.rstrip(os.sep))
    if os.path.isfile(path) and checkIfVideoFile(name):
        raise ValueError('{:s} is compressed video: the shape is measured on raw frames only (.vid, frame blocks, '
                         'image frames), as the compression blurs faint extended structures'.format(name))
    if os.path.isfile(path) and (validFFName(name) or validFRName(name)):
        raise ValueError('{:s} is an FF file, which holds images of a chunk of frames, not the frames'.format(name))

    # The input as the processing opens it, without the binning of the detection. An input whose beginning
    #   can't be found stops the frame interface (sys.exit), which is turned into an error here
    try:
        img_handle = detectInputType(path, config, detection=False, chunk_frames=OPEN_CHUNK_FRAMES)
    except SystemExit:
        raise ValueError('The frame interface could not open {:s}'.format(path))

    # A directory of FF files is not a recording of frames either
    if (img_handle is not None) and (getattr(img_handle, 'input_type', None) == 'ff'):
        closeInput(img_handle)
        raise ValueError('{:s} holds FF files, not the frames of a recording'.format(path))

    return img_handle



def openRawSource(raw_path, t0, t1, config=None, frame_time_offset=None, frame_block_time=None):
    """ The raw frames between t0 and t1, read through the frame interface: a raw file (.vid, .fblk), a directory
        of raw files (those around the interval are opened, see rawFilesAround), or a directory of image frames
        (e.g. UWO PNG frame dumps). The frames are timed as the processing timed them, from the config.

    Arguments:
        raw_path: [str] Path to the raw recording.
        t0: [float] Beginning (Unix s).
        t1: [float] End (Unix s).

    Keyword arguments:
        config: [Config] Configuration of the camera (frame_time_offset, frame_block_time, fps, ...). The RMS
            defaults if None (default).
        frame_time_offset: [float] frame_time_offset to use instead of the one of the config. None by default.
        frame_block_time: [str] frame_block_time to use instead of the one of the config. None by default.

    Return:
        [RawFrames] The frames, None if no raw frames were found.
    """

    import copy

    # The config of the camera (the RMS defaults without one), with the given timing of the frames
    if config is None:
        import RMS.ConfigReader as cr
        config = cr.Config()
    if (frame_time_offset is not None) or (frame_block_time is not None):
        config = copy.copy(config)
        if frame_time_offset is not None:
            config.frame_time_offset = float(frame_time_offset)
        if frame_block_time is not None:
            config.frame_block_time = frame_block_time

    # The raw files of a directory around the interval (the margin covers a correction of the times of the
    #   frames), else the directory as one input (e.g. image frames); a file as one input
    paths = [raw_path]
    if os.path.isdir(raw_path):
        around = rawFilesAround(raw_path, t0, t1, margin=abs(getattr(config, 'frame_time_offset', 0.0)) + 1.0)
        if around:
            paths = around
        elif any(os.path.isfile(p) for ext in RAW_FILE_EXTENSIONS
                 for p in glob.glob(os.path.join(raw_path, '**', '*' + ext), recursive=True)):
            return None

    # Every input opened by the frame interface (those it doesn't recognise are skipped)
    img_handles, names = [], []
    try:
        for path in paths:
            img_handle = openInput(path, config)
            if (img_handle is not None) and (img_handle.total_frames > 0):
                img_handles.append(img_handle)
                names.append(path)
    except Exception:
        for img_handle in img_handles:
            closeInput(img_handle)
        raise

    # The frames of all inputs as one sequence
    if not img_handles:
        return None

    return RawFrames(img_handles, names)



### Measurement ###

def _gauss(x, a, s, c, x0):
    """ Gaussian with an amplitude a, sigma s, constant level c and centre x0. """

    return a*np.exp(-(x - x0)**2/(2*s**2)) + c



def gaussianWidth(profile):
    """ Fit a Gaussian with a constant level to a profile centred on the object.

    Arguments:
        profile: [ndarray] Profile (odd length, the object at the middle).

    Return:
        (sigma, amplitude): [tuple of floats] Gaussian sigma (px) and amplitude, NaN if the fit failed.
    """

    x = np.arange(len(profile)) - len(profile)//2

    # The fit starts from the peak above the median level, with the centre allowed to move a few pixels
    try:
        p, _ = curve_fit(_gauss, x, profile,
                         p0=[max(profile.max() - np.median(profile), 1e-3), FIT_SIGMA_START, np.median(profile), 0],
                         bounds=([0, FIT_SIGMA_LIMITS[0], -np.inf, -FIT_CENTRE_LIMIT],
                                 [np.inf, FIT_SIGMA_LIMITS[1], np.inf, FIT_CENTRE_LIMIT]), maxfev=4000)
        return float(p[1]), float(p[0])
    except (RuntimeError, ValueError):
        return float('nan'), float('nan')



def timesOfFrames(source, frames):
    """ Times of (possibly fractional) frame numbers of a source, for a track given in frame numbers.

    Arguments:
        source: [object] Raw frame source.
        frames: [array] Frame numbers.

    Return:
        [ndarray] Times (Unix s).
    """

    frames = np.asarray(frames, dtype=np.float64)

    # The times of the whole frames around the track, interpolated to the fractional frames
    first = max(int(math.floor(frames.min())), 0)
    last = min(int(math.ceil(frames.max())), source.n_frames - 1)
    whole = np.arange(first, last + 1)

    return np.interp(frames, whole, [source.frameTime(int(i)) for i in whole])



def measureTrackShape(source, t, x, y, psf_sigma, window_half=WINDOW_HALF, max_track_frames=MAX_TRACK_FRAMES,
                      background_frames=BACKGROUND_FRAMES):
    """ Measure the shape of the light of a moving object on the raw frames: the width of the light across and
        along the motion in the frames averaged along the track (see the module docstring).

    Arguments:
        source: [object] Raw frames (RawFrames, or any object with framesBetween(t0, t1), frame(i), and the
            height and width of the frames).
        t: [array] Times of the measured positions of the object (Unix s, in the time base of the source).
        x: [array] X positions of the object (px).
        y: [array] Y positions of the object (px).
        psf_sigma: [float] Gaussian sigma of the point spread function (px).

    Keyword arguments:
        window_half: [int] Half size of the window in which the frames are averaged (px).
        max_track_frames: [int] Most frames of the track which are averaged.
        background_frames: [int] Number of frames of which the median is the background.

    Return:
        [dict] sigma_cross, sigma_along (Gaussian sigma across and along the motion, px), spread (sigma_cross
            divided by psf_sigma), snr (amplitude of the profile across the motion relative to the noise of the
            averaged window), n_frames (frames of the track), n_stack (frames averaged), ok (the widths were
            measured on at least MIN_STACK_FRAMES frames), flag ('ok', or why not: 'few points', 'no frames',
            'edge', 'few frames', 'fit failed').
    """

    result = dict(sigma_cross=float('nan'), sigma_along=float('nan'), spread=float('nan'), snr=float('nan'),
                  n_frames=0, n_stack=0, ok=False, flag='ok')

    # The positions in time order
    t, x, y = np.asarray(t, dtype=np.float64), np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    order = np.argsort(t, kind='stable')
    t, x, y = t[order], x[order], y[order]
    if len(t) < 2:
        result['flag'] = 'few points'
        return result

    # The frames of the track, and of the background: those over the track and a time before and after it, which
    #   is at least as long as the track
    duration = max(t[-1] - t[0], MIN_DURATION)
    pad = max(duration, MIN_BACKGROUND_PAD)
    all_idx, all_times = source.framesBetween(t[0] - pad, t[-1] + pad)
    on_track = (all_times >= t[0]) & (all_times <= t[-1])
    track_idx, track_times = all_idx[on_track], all_times[on_track]
    result['n_frames'] = int(len(track_idx))
    if len(track_idx) < 2:
        result['flag'] = 'no frames'
        return result

    # The box around the track in which the background is computed
    height, width = source.height, source.width
    bx0, bx1 = int(max(x.min() - BOX_MARGIN, 0)), int(min(x.max() + BOX_MARGIN, width))
    by0, by1 = int(max(y.min() - BOX_MARGIN, 0)), int(min(y.max() + BOX_MARGIN, height))

    # The frames are read once, as some are used for both the background and the track; only the box is kept, so
    #   the memory doesn't grow with the size of the frames
    cache = {}

    def readBox(i):
        """ The box of frame i of the source, read from the file only the first time. """
        if i not in cache:
            cache[i] = source.frame(int(i))[by0:by1, bx0:bx1].copy()
        return cache[i]

    # The background: the median of frames spread evenly over the track and the time around it (the object stays
    #   on any pixel only briefly, so it doesn't enter the median)
    bg_idx = all_idx[::max(1, len(all_idx)//background_frames)]
    background = np.median(np.array([readBox(i) for i in bg_idx]), axis=0)

    # The frames of the track, minus the background, shifted so the object is at the centre of the window and
    #   averaged. The window is cut 2 px larger on every side, so the shift doesn't bring in the edge values
    step = max(1, len(track_idx)//max_track_frames)
    n_win = 2*window_half + 1
    stack = np.zeros((n_win, n_win), np.float64)
    n_stack = 0
    for i, tf in zip(track_idx[::step], track_times[::step]):

        img = readBox(i) - background

        # The position of the object at the time of the frame, in the box
        cx, cy = np.interp(tf, t, x) - bx0, np.interp(tf, t, y) - by0
        ix, iy = int(round(cx)), int(round(cy))

        # Frames in which the window doesn't fit in the box are skipped
        if (ix - window_half - 2 < 0) or (iy - window_half - 2 < 0) or (ix + window_half + 3 > img.shape[1]) \
                or (iy + window_half + 3 > img.shape[0]):
            continue

        # The window shifted by the fraction of a pixel (bilinear), without the extra border
        win = img[iy - window_half - 2:iy + window_half + 3, ix - window_half - 2:ix + window_half + 3]
        win = ndimage.shift(win, (iy - cy, ix - cx), order=1, mode='nearest')[2:-2, 2:-2]
        stack += win
        n_stack += 1

    result['n_stack'] = n_stack
    if n_stack == 0:
        result['flag'] = 'edge'
        return result
    stack /= n_stack

    # The averaged window rotated so that the motion is along the x axis, and the profiles across the motion
    #   (along y) and along it (along x), averaged over a strip through the centre
    angle = math.atan2(y[-1] - y[0], x[-1] - x[0])
    rot = ndimage.rotate(stack, math.degrees(angle), reshape=False, order=1)
    c = window_half
    cross_profile = rot[:, c - STRIP_HALF:c + STRIP_HALF + 1].mean(axis=1)
    along_profile = rot[c - STRIP_HALF:c + STRIP_HALF + 1, :].mean(axis=0)

    # The Gaussian widths across and along the motion, and the signal to noise of the profile across the motion
    #   (the noise from the median absolute deviation of the averaged window)
    sigma_cross, amp_cross = gaussianWidth(cross_profile)
    sigma_along, _ = gaussianWidth(along_profile)
    noise = 1.4826*np.median(np.abs(stack - np.median(stack)))

    result.update(sigma_cross=sigma_cross, sigma_along=sigma_along, spread=sigma_cross/psf_sigma,
                  snr=amp_cross/max(noise, 1e-6))
    result['ok'] = bool(np.isfinite(sigma_cross)) and (n_stack >= MIN_STACK_FRAMES)
    if not np.isfinite(sigma_cross):
        result['flag'] = 'fit failed'
    elif n_stack < MIN_STACK_FRAMES:
        result['flag'] = 'few frames'

    return result



### Detections of an FTPdetectinfo file ###

def psfSigmaFromDone(ftp_path, config=None):
    """ The PSF sigma measured by the matched filter, from its done file next to the FTPdetectinfo file
        (in the pixels of the frames, i.e. times the detection binning of the config).

    Arguments:
        ftp_path: [str] Path to the FTPdetectinfo file.

    Keyword arguments:
        config: [Config] Configuration object, for the detection binning. None by default.

    Return:
        [float] PSF sigma (px), None if it isn't there.
    """

    done_path = os.path.join(os.path.dirname(os.path.abspath(ftp_path)), 'matched_filter_done.json')
    try:
        with open(done_path) as f:
            sigma = json.load(f).get('psf_sigma')
    except (IOError, OSError, ValueError):
        return None

    if not sigma:
        return None

    binning = getattr(config, 'detection_binning_factor', 1) if config is not None else 1

    return float(sigma)*binning



def psfSigmaForRawFrames(ftp_path, config=None, psf_sigma=None):
    """ The PSF sigma in the pixels of the raw frames: the one given, else the one the matched filter measured for
        the file (its done file next to the FTPdetectinfo file), else mf_psf_sigma of the config. The last two are
        in the pixels of the frames the matched filter searched, so they are multiplied by the detection binning.

    Arguments:
        ftp_path: [str] Path to the FTPdetectinfo file.

    Keyword arguments:
        config: [Config] Configuration object. None by default.
        psf_sigma: [float] PSF sigma in the pixels of the raw frames, used as given. None by default.

    Return:
        [float] PSF sigma (px), None if not known.
    """

    if psf_sigma is not None:
        return float(psf_sigma)

    # Measured by the matched filter for this file
    measured = psfSigmaFromDone(ftp_path, config)
    if measured is not None:
        return measured

    # The configured one, in the pixels of the binned frames
    if (config is not None) and (getattr(config, 'mf_psf_sigma', 0) > 0):
        return float(config.mf_psf_sigma)*getattr(config, 'detection_binning_factor', 1)

    return None



def rawInputFromDone(ftp_path, raw_dir):
    """ The raw input which the matched filter processed (a file or a directory of image frames, from its done
        file), looked up by name in the raw directory (it may have been moved since). None if not known or not
        found.

    Arguments:
        ftp_path: [str] Path to the FTPdetectinfo file.
        raw_dir: [str] Directory of the raw recordings.

    Return:
        [str] Path to the raw input, or None.
    """

    done_path = os.path.join(os.path.dirname(os.path.abspath(ftp_path)), 'matched_filter_done.json')
    try:
        with open(done_path) as f:
            input_file = json.load(f).get('input_file')
    except (IOError, OSError, ValueError):
        return None
    if not input_file:
        return None

    # The file or directory where it was processed, else in the raw directory
    if os.path.exists(input_file):
        return input_file
    found = glob.glob(os.path.join(raw_dir, '**', os.path.basename(input_file.rstrip(os.sep))), recursive=True)

    return found[0] if found else None



def detectionTimes(ff_name, fps, frames):
    """ Times of the measurements of a detection of an FTPdetectinfo file: the time in the name plus the frame
        divided by the frame rate (the frames of the inputs other than FF files are written so).

    Arguments:
        ff_name: [str] Name of the FF file of the detection.
        fps: [float] Frame rate of the detection.
        frames: [array] Frame numbers.

    Return:
        [ndarray] Times (Unix s).
    """

    from RMS.Formats.FFfile import filenameToDatetime

    ref = filenameToDatetime(ff_name).replace(tzinfo=datetime.timezone.utc).timestamp()

    return ref + np.asarray(frames, dtype=np.float64)/fps



def rawFrameNumbers(frames, y, config, n_rows, input_type, fps=None):
    """ The frames of the measurements of a detection on the raw frames. With a rolling shutter (deinterlace_order
        -1 and a measured rolling_shutter_readout), the processing moves the frame of a point by the time its row
        was read after the top row (see MatchedFilterDetection.correctRollingShutter), while a raw frame is timed by
        its top row; the same row delay is removed here, so the positions line up with the raw frames. Unchanged
        without a rolling shutter.

    Arguments:
        frames: [array] Frame numbers of the measurements, as written by the processing.
        y: [array] Rows of the measurements (px of the raw frames).
        config: [Config] Configuration of the camera, or None (no correction).
        n_rows: [int] Number of rows of the raw frames.
        input_type: [str] Type of the input of the frame interface (the crop of the capture applies to some).

    Keyword arguments:
        fps: [float] Frame rate of the input. None by default (the frame rate of the config).

    Return:
        [ndarray] Frame numbers on the raw frames.
    """

    from RMS.Routines import RollingShutterCorrection

    frames = np.asarray(frames, dtype=np.float64)

    # Only a rolling shutter moved the frames
    if (config is None) or (getattr(config, 'deinterlace_order', None) != -1):
        return frames

    # The row delay of every point, as the processing added it (the correction of frame 0 is the delay alone)
    delay = RollingShutterCorrection.correctFrameTime(np.zeros_like(frames), np.asarray(y, dtype=np.float64),
                                                      n_rows, config, fps=fps,
                                                      cropped=RollingShutterCorrection.croppedInput(input_type))

    return frames - delay



def measureFTPdetectinfo(ftp_path, raw_path, psf_sigma, config=None):
    """ Measure the shape of every detection of an FTPdetectinfo file on the raw recording.

    Arguments:
        ftp_path: [str] Path to the FTPdetectinfo file.
        raw_path: [str] Raw recording, any raw input of the frame interface: a raw file (.vid, .fblk), a directory
            of raw files, or a directory of image frames (see openRawSource).
        psf_sigma: [float] PSF sigma (px).

    Keyword arguments:
        config: [Config] Configuration of the camera, which times the raw frames as the processing did. The RMS
            defaults if None (default).

    Return:
        [list] A dict per detection: the measurement (see measureTrackShape) and the detection (CSV_COLUMNS).
    """

    from RMS.Formats.FTPdetectinfo import readFTPdetectinfo

    rows = []
    for met in readFTPdetectinfo(*os.path.split(os.path.abspath(ftp_path))):

        ff_name, meteor_no, fps = met[0], met[2], met[4]
        meas = np.array(met[-1], dtype=np.float64)
        t = detectionTimes(ff_name, fps, meas[:, 1])
        x, y = meas[:, 2], meas[:, 3]

        row = dict(ff_name=ff_name, meteor_no=meteor_no, n_points=len(t),
                   time_start_utc=datetime.datetime.fromtimestamp(t.min(), datetime.timezone.utc).isoformat(),
                   time_end_utc=datetime.datetime.fromtimestamp(t.max(), datetime.timezone.utc).isoformat(),
                   raw_file='', psf_sigma=psf_sigma)

        # The raw frames around the detection (the padding covers the row delay of a rolling shutter)
        t_start = time.time()
        pad = max(t.max() - t.min(), MIN_DURATION, MIN_BACKGROUND_PAD)
        source = openRawSource(raw_path, t.min() - pad, t.max() + pad, config=config)
        if source is None:
            row.update(sigma_cross=float('nan'), sigma_along=float('nan'), spread=float('nan'),
                       snr=float('nan'), n_frames=0, n_stack=0, ok=False, flag='no raw data')
        else:
            row['raw_file'] = ';'.join(source.names)

            # The times of the measurements on the raw frames, without the row delay of a rolling shutter
            first_input = source.img_handles[0]
            t_raw = detectionTimes(ff_name, fps, rawFrameNumbers(meas[:, 1], y, config, source.height,
                                                                 first_input.input_type,
                                                                 fps=getattr(first_input, 'fps', None)))
            try:
                row.update(measureTrackShape(source, t_raw, x, y, psf_sigma))
            finally:
                source.close()
        row['time_s'] = round(time.time() - t_start, 3)

        rows.append(row)

    return rows



if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(description="Measure the shape of the light of every detection of an "
        "FTPdetectinfo file on the raw frames of the recording (any raw input of RMS: .vid files, frame block files, image "
        "frames such as UWO PNG frame dumps; never compressed video): the width across and along the "
        "motion in the frames averaged along the track, and the spread (width across the motion in PSF sigmas, "
        "about 1 for a point source, larger for extended structures such as clouds). Writes a CSV file.")

    arg_parser.add_argument('ftpdetectinfo', help='FTPdetectinfo file.')
    arg_parser.add_argument('raw', help='Raw recording: a raw file (.vid, .fblk), a directory of raw files, or a '
                            'directory of image frames. For a directory, the input which the matched filter processed '
                            '(from its done file) is used if found and it is not a frame block, else the raw files '
                            'around the time of every detection.')
    arg_parser.add_argument('-c', '--config', help='Config file of the camera, which times the frames as the '
                            'processing did (frame_time_offset, frame_block_time, fps) and gives the detection binning '
                            'and the PSF sigma (mf_psf_sigma).')
    arg_parser.add_argument('--psf_sigma', type=float, help='PSF sigma (px of the raw frames). By default from the '
                            'done file of the matched filter next to the FTPdetectinfo file, else mf_psf_sigma of the '
                            'config, both times the detection binning.')
    arg_parser.add_argument('-o', '--output', help='Output CSV file. By default next to the FTPdetectinfo file, '
                            'with the suffix _track_shape.csv.')

    args = arg_parser.parse_args()

    #########################

    config = None
    if args.config:
        import RMS.ConfigReader as cr
        config = cr.parse(args.config)

    # The PSF sigma in raw pixels: given, else measured by the matched filter, else from the config
    psf_sigma = psfSigmaForRawFrames(args.ftpdetectinfo, config, args.psf_sigma)
    if psf_sigma is None:
        raise SystemExit('The PSF sigma is not known: give --psf_sigma or a config with mf_psf_sigma')

    # For a directory, the input which was processed if it is known; frame blocks are taken from the directory
    #   around every detection instead (the background is taken also from the frames before and after the block)
    raw_path = args.raw
    if os.path.isdir(raw_path):
        processed = rawInputFromDone(args.ftpdetectinfo, raw_path)
        if (processed is not None) and (not processed.lower().endswith('.fblk')):
            raw_path = processed

    # Measure every detection
    rows = measureFTPdetectinfo(args.ftpdetectinfo, raw_path, psf_sigma, config=config)

    # Save the CSV file
    out_path = args.output or os.path.splitext(os.path.abspath(args.ftpdetectinfo))[0] + '_track_shape.csv'
    with open(out_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print('{:d} detections measured, saved to {:s}'.format(len(rows), out_path))
