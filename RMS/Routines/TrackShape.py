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

Only the raw recording is used (.vid files and frame block files), never compressed video, as the compression
smooths and blurs faint extended structures. The frames are read through a raw frame source (VidSource,
FrameBlockSource), which gives the frames and their times; any object with the same methods can be used instead.

Command line: python -m RMS.Routines.TrackShape FTPdetectinfo_... raw_dir [-c config] [--psf_sigma 0.7]
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

# Half width of the strip through the centre over which the profiles are averaged (px)
STRIP_HALF = 2

# Starting sigma and the limits of the sigma (px) and of the centre (px) of the Gaussian fit of a profile
FIT_SIGMA_START = 1.5
FIT_SIGMA_LIMITS = (0.3, 15.0)
FIT_CENTRE_LIMIT = 3.0

# Columns of the CSV file of the command line
CSV_COLUMNS = ['ff_name', 'meteor_no', 'n_points', 'time_start_utc', 'time_end_utc', 'raw_file', 'psf_sigma',
               'sigma_cross', 'sigma_along', 'spread', 'snr', 'n_frames', 'n_stack', 'ok', 'flag', 'time_s']


### Raw frame sources ###

class VidSource(object):
    def __init__(self, path, frame_time_offset=0.0):
        """ Frames of a .vid file and their times (the time in the header of every frame), read one at a time
            as they are needed. The frames of a .vid file all have the same size, so a frame is found directly
            from its number, and the frames around a time are found by a binary search over the times in the
            headers (the frames are in time order).

        Arguments:
            path: [str] Path to the .vid file.

        Keyword arguments:
            frame_time_offset: [float] Correction added to the times of the frames (s), as frame_time_offset of
                the config. 0 by default.
        """

        # Imported here, so the module is usable without the .vid reader for the other sources
        from RMS.Formats.Vid import VidStruct, readFrame

        self._VidStruct = VidStruct
        self._readFrame = readFrame

        self.path = path
        self.frame_time_offset = float(frame_time_offset)
        self.fid = open(path, 'rb')

        # The size of the frames and of their records from the first header
        info = VidStruct()
        readFrame(info, self.fid, metadata_only=True)
        self.frame_bytes = info.seqlen
        self.height, self.width = info.ht, info.wid
        self.n_frames = os.path.getsize(path)//self.frame_bytes

        # The times of the frames which were read
        self._times = {}


    def frameTime(self, i):
        """ Time of frame i (Unix s, with the correction of the times).

        Arguments:
            i: [int] Frame number.

        Return:
            [float] Time of the frame.
        """

        # The time is read from the header of the frame only once
        if i not in self._times:
            st = self._VidStruct()
            self.fid.seek(i*self.frame_bytes)
            self._readFrame(st, self.fid, metadata_only=True)
            self._times[i] = st.ts + st.tu/1e6 + self.frame_time_offset

        return self._times[i]


    def framesBetween(self, t0, t1):
        """ The frames whose times are between t0 and t1 (inclusive).

        Arguments:
            t0: [float] Beginning (Unix s).
            t1: [float] End (Unix s).

        Return:
            (indices, times): [tuple of ndarrays] Frame numbers and times, in time order.
        """

        if self.n_frames == 0:
            return np.zeros(0, dtype=int), np.zeros(0)

        # The first frame at or after t0, by a binary search over the times in the headers
        lo, hi = 0, self.n_frames
        while lo < hi:
            mid = (lo + hi)//2
            if self.frameTime(mid) < t0:
                lo = mid + 1
            else:
                hi = mid

        # The frames which follow it, up to t1
        indices, times = [], []
        i = lo
        while (i < self.n_frames) and (self.frameTime(i) <= t1):
            indices.append(i)
            times.append(self.frameTime(i))
            i += 1

        return np.array(indices, dtype=int), np.array(times, dtype=np.float64)


    def frame(self, i):
        """ Frame i as float32 (the header in the first row of the frame is set to 0, see Vid.readFrame). """

        st = self._VidStruct()
        self.fid.seek(i*self.frame_bytes)
        img = self._readFrame(st, self.fid)
        if img is None:
            raise IndexError('Frame {} is not in {}'.format(i, self.path))

        # The time of the frame is kept, as its header was read too
        self._times[i] = st.ts + st.tu/1e6 + self.frame_time_offset

        return img.astype(np.float32)


    def close(self):
        """ Close the file. """

        if self.fid is not None:
            self.fid.close()
            self.fid = None



class FrameBlockSource(object):
    def __init__(self, paths, frame_time_offset=0.0, frame_block_time='middle'):
        """ Frames of one or more frame block files (see RMS.Formats.FrameBlock), numbered continuously in time
            order, with the time of every frame.

        Arguments:
            paths: [list] Paths to the block files.

        Keyword arguments:
            frame_time_offset: [float] Correction added to the times of the frames (s), as frame_time_offset of
                the config. 0 by default.
            frame_block_time: [str] The point of the exposure at which the frames are timed ('start', 'middle' or
                'end'), as frame_block_time of the config, so the times are those of the detections. 'middle' by
                default.
        """

        from RMS.Formats.FrameBlock import FrameBlockReader

        fraction = {'start': 0.0, 'middle': 0.5, 'end': 1.0}[frame_block_time]

        # The blocks with their frame times (as the frame interface of the blocks computes them), in time order
        blocks = []
        for path in paths:
            reader = FrameBlockReader(path)
            if reader.n_frames == 0:
                reader.close()
                continue
            exposure_s = np.asarray(reader.table['exposure_us'], dtype=np.float64)/1e6
            times = np.asarray(reader.times, dtype=np.float64) + float(frame_time_offset) + fraction*exposure_s
            blocks.append((times[0], reader, times))
        blocks.sort(key=lambda b: b[0])

        # Every frame is a (block, frame of the block), with its time
        self.readers = [b[1] for b in blocks]
        self.times = np.concatenate([b[2] for b in blocks]) if blocks else np.zeros(0)
        self._block = np.concatenate([np.full(len(b[2]), k, dtype=int) for k, b in enumerate(blocks)]) \
            if blocks else np.zeros(0, dtype=int)
        self._local = np.concatenate([np.arange(len(b[2])) for b in blocks]) if blocks else np.zeros(0, dtype=int)
        self.n_frames = len(self.times)
        self.height = self.readers[0].height if self.readers else 0
        self.width = self.readers[0].width if self.readers else 0


    def frameTime(self, i):
        """ Time of frame i (Unix s, with the correction of the times). """

        return float(self.times[i])


    def framesBetween(self, t0, t1):
        """ The frames whose times are between t0 and t1 (inclusive), see VidSource.framesBetween. """

        sel = np.where((self.times >= t0) & (self.times <= t1))[0]
        order = np.argsort(self.times[sel], kind='stable')

        return sel[order], self.times[sel][order]


    def frame(self, i):
        """ Frame i as float32 (the values as stored, without a shift to 16 bits). """

        return self.readers[self._block[i]].frame(int(self._local[i])).astype(np.float32)


    def close(self):
        """ Release the memory maps of the blocks. """

        for reader in self.readers:
            reader.close()



def frameBlockStart(path):
    """ The time of the beginning of a frame block from its name (e.g. XX0001_20261027_101258_485479.fblk), None if
        the name has no time.

    Arguments:
        path: [str] Path to the block file.

    Return:
        [float] Unix time (s), or None.
    """

    match = re.search(r'(\d{8})_(\d{6})_(\d{6})\.fblk$', os.path.basename(path))
    if match is None:
        return None

    dt = datetime.datetime.strptime(''.join(match.groups()), '%Y%m%d%H%M%S%f')

    return dt.replace(tzinfo=datetime.timezone.utc).timestamp()



def frameBlocksAround(raw_dir, t0, t1, margin=0.0):
    """ The frame block files of a directory (searched recursively) which can hold frames between t0 and t1: the
        blocks which begin before t1 and whose next block begins after t0, from the times in their names. Blocks
        without a time in the name are all taken (their own times decide which frames are used).

    Arguments:
        raw_dir: [str] Directory of the block files.
        t0: [float] Beginning (Unix s).
        t1: [float] End (Unix s).

    Keyword arguments:
        margin: [float] Widening of the interval on both sides (s), e.g. for a correction of the frame times.
            0 by default.

    Return:
        [list] Paths of the block files, in time order.
    """

    paths = sorted(glob.glob(os.path.join(raw_dir, '**', '*.fblk'), recursive=True))

    # The blocks named by their time, in time order, and the others
    named = sorted((frameBlockStart(p), p) for p in paths if frameBlockStart(p) is not None)
    unnamed = [p for p in paths if frameBlockStart(p) is None]

    # A block can hold frames of the interval if it begins before its end and the next block begins after its
    #   beginning
    t0, t1 = t0 - margin, t1 + margin
    selected = []
    for k, (start, path) in enumerate(named):
        next_start = named[k + 1][0] if k + 1 < len(named) else np.inf
        if (start <= t1) and (next_start >= t0):
            selected.append(path)

    return selected + unnamed



def openRawSource(raw_path, t0, t1, frame_time_offset=0.0, frame_block_time='middle'):
    """ A raw frame source for the frames between t0 and t1: a .vid file, a frame block file, or a directory of
        frame block files.

    Arguments:
        raw_path: [str] Path to a .vid file, a .fblk file, or a directory of .fblk files.
        t0: [float] Beginning (Unix s).
        t1: [float] End (Unix s).

    Keyword arguments:
        frame_time_offset: [float] Correction added to the times of the frames (s). 0 by default.
        frame_block_time: [str] Point of the exposure of the times of frame blocks, see FrameBlockSource.
            'middle' by default.

    Return:
        [VidSource or FrameBlockSource] The source, None if no raw frames were found.
    """

    if os.path.isdir(raw_path):
        paths = frameBlocksAround(raw_path, t0, t1, margin=abs(frame_time_offset) + 1.0)
        if not paths:
            return None
        return FrameBlockSource(paths, frame_time_offset=frame_time_offset, frame_block_time=frame_block_time)

    if raw_path.lower().endswith('.fblk'):
        return FrameBlockSource([raw_path], frame_time_offset=frame_time_offset, frame_block_time=frame_block_time)

    if raw_path.lower().endswith('.vid'):
        return VidSource(raw_path, frame_time_offset=frame_time_offset)

    return None



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
        source: [object] Raw frame source (VidSource, FrameBlockSource, or any object with framesBetween(t0, t1),
            frame(i), and the height and width of the frames).
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
            measured), flag ('ok', or why not: 'few points', 'no frames', 'edge', 'fit failed').
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

    # The frames are read once, as some are used for both the background and the track
    cache = {}

    def readFrame(i):
        if i not in cache:
            cache[i] = source.frame(int(i))
        return cache[i]

    # The box around the track in which the background is computed
    height, width = source.height, source.width
    bx0, bx1 = int(max(x.min() - BOX_MARGIN, 0)), int(min(x.max() + BOX_MARGIN, width))
    by0, by1 = int(max(y.min() - BOX_MARGIN, 0)), int(min(y.max() + BOX_MARGIN, height))

    # The background: the median of frames spread evenly over the track and the time around it (the object stays
    #   on any pixel only briefly, so it doesn't enter the median)
    bg_idx = all_idx[::max(1, len(all_idx)//background_frames)]
    background = np.median(np.array([readFrame(i)[by0:by1, bx0:bx1] for i in bg_idx]), axis=0)

    # The frames of the track, minus the background, shifted so the object is at the centre of the window and
    #   averaged. The window is cut 2 px larger on every side, so the shift doesn't bring in the edge values
    step = max(1, len(track_idx)//max_track_frames)
    n_win = 2*window_half + 1
    stack = np.zeros((n_win, n_win), np.float64)
    n_stack = 0
    for i, tf in zip(track_idx[::step], track_times[::step]):

        img = readFrame(i)[by0:by1, bx0:bx1] - background

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
    result['ok'] = bool(np.isfinite(sigma_cross))
    if not result['ok']:
        result['flag'] = 'fit failed'

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



def rawInputFromDone(ftp_path, raw_dir):
    """ The raw file which the matched filter processed (from its done file), looked up by name in the raw
        directory (the file may have been moved since). None if not known or not found.

    Arguments:
        ftp_path: [str] Path to the FTPdetectinfo file.
        raw_dir: [str] Directory of the raw files.

    Return:
        [str] Path to the raw file, or None.
    """

    done_path = os.path.join(os.path.dirname(os.path.abspath(ftp_path)), 'matched_filter_done.json')
    try:
        with open(done_path) as f:
            input_file = json.load(f).get('input_file')
    except (IOError, OSError, ValueError):
        return None
    if not input_file:
        return None

    # The file where it was processed, else in the raw directory
    if os.path.isfile(input_file):
        return input_file
    found = glob.glob(os.path.join(raw_dir, '**', os.path.basename(input_file)), recursive=True)

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



def measureFTPdetectinfo(ftp_path, raw_path, psf_sigma, frame_time_offset=0.0, frame_block_time='middle'):
    """ Measure the shape of every detection of an FTPdetectinfo file on the raw recording.

    Arguments:
        ftp_path: [str] Path to the FTPdetectinfo file.
        raw_path: [str] Raw recording: a .vid or .fblk file, or a directory of frame block files.
        psf_sigma: [float] PSF sigma (px).

    Keyword arguments:
        frame_time_offset: [float] Correction added to the times of the raw frames (s), as the processing applied
            it. 0 by default.
        frame_block_time: [str] Point of the exposure of the times of frame blocks. 'middle' by default.

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

        # The raw frames around the detection
        t_start = time.time()
        pad = max(t.max() - t.min(), MIN_DURATION, MIN_BACKGROUND_PAD)
        source = openRawSource(raw_path, t.min() - pad, t.max() + pad, frame_time_offset=frame_time_offset,
                               frame_block_time=frame_block_time)
        if source is None:
            row.update(sigma_cross=float('nan'), sigma_along=float('nan'), spread=float('nan'),
                       snr=float('nan'), n_frames=0, n_stack=0, ok=False, flag='no raw data')
        else:
            row['raw_file'] = ';'.join(os.path.basename(getattr(r, 'path', '')) for r in
                                       getattr(source, 'readers', [source]))
            try:
                row.update(measureTrackShape(source, t, x, y, psf_sigma))
            finally:
                source.close()
        row['time_s'] = round(time.time() - t_start, 3)

        rows.append(row)

    return rows



if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(description="Measure the shape of the light of every detection of an "
        "FTPdetectinfo file on the raw recording (.vid or frame block files): the width across and along the "
        "motion in the frames averaged along the track, and the spread (width across the motion in PSF sigmas, "
        "about 1 for a point source, larger for extended structures such as clouds). Writes a CSV file.")

    arg_parser.add_argument('ftpdetectinfo', help='FTPdetectinfo file.')
    arg_parser.add_argument('raw', help='Raw recording: a .vid or .fblk file, or a directory of frame block files. '
                            'For a directory, the file which the matched filter processed (from its done file) is '
                            'used if found, else the blocks around the time of every detection.')
    arg_parser.add_argument('-c', '--config', help='Config file, for the frame time correction, the timing of the '
                            'frame blocks, the detection binning and the PSF sigma (mf_psf_sigma).')
    arg_parser.add_argument('--psf_sigma', type=float, help='PSF sigma (px). By default from the done file of the '
                            'matched filter next to the FTPdetectinfo file, else mf_psf_sigma of the config.')
    arg_parser.add_argument('-o', '--output', help='Output CSV file. By default next to the FTPdetectinfo file, '
                            'with the suffix _track_shape.csv.')

    args = arg_parser.parse_args()

    #########################

    config = None
    if args.config:
        import RMS.ConfigReader as cr
        config = cr.parse(args.config)

    # The PSF sigma: given, else measured by the matched filter, else from the config
    psf_sigma = args.psf_sigma
    if psf_sigma is None:
        psf_sigma = psfSigmaFromDone(args.ftpdetectinfo, config)
    if (psf_sigma is None) and (config is not None) and (getattr(config, 'mf_psf_sigma', 0) > 0):
        psf_sigma = config.mf_psf_sigma
    if psf_sigma is None:
        raise SystemExit('The PSF sigma is not known: give --psf_sigma or a config with mf_psf_sigma')

    # For a directory, the .vid file which was processed if it is known; the frame blocks around every detection
    #   are taken from the directory (the background is taken also from the frames before and after the block)
    raw_path = args.raw
    if os.path.isdir(raw_path):
        processed = rawInputFromDone(args.ftpdetectinfo, raw_path)
        if (processed is not None) and processed.lower().endswith('.vid'):
            raw_path = processed

    # Measure every detection
    rows = measureFTPdetectinfo(args.ftpdetectinfo, raw_path, psf_sigma,
                                frame_time_offset=getattr(config, 'frame_time_offset', 0.0),
                                frame_block_time=getattr(config, 'frame_block_time', 'middle'))

    # Save the CSV file
    out_path = args.output or os.path.splitext(os.path.abspath(args.ftpdetectinfo))[0] + '_track_shape.csv'
    with open(out_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print('{:d} detections measured, saved to {:s}'.format(len(rows), out_path))
