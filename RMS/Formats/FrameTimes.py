""" The times of the individual frames of videos and image sequences, from a file with the time of every frame.

Some inputs are not evenly spaced in time. The timelapses of the saved frames (Utils/GenerateTimelapse.py) are
missing the frames which were not saved and their spacing jitters, and a video of the capture is missing the frames
which the camera or the network dropped. Timing their frames from the beginning and a frame rate puts every frame
after a missing one at the wrong time, so they are timed from a file with the time of every frame:

- the frametimes JSON file written next to a timelapse (<name>_frametimes.json): {video frame index: time};
- any JSON file with {frame index: time} or a list [time, ...], with the times as the stamp of the saved frames
  (YYYYMMDD_HHMMSS_mmm with an optional _d or _n), ISO 8601 (YYYY-MM-DDTHH:MM:SS[.ffffff], optionally with Z or
  a UTC offset) or Unix seconds;
- the FT files of the capture (FT_<station>_YYYYMMDD_HHMMSS.bin, see RMS/Formats/FTfile.py), with the capture time
  of every frame the capture processed: a single file, a folder of them (searched recursively), or the daily archive
  *_FT.tar(.bz2/.gz) read without extracting it. The frames of an FF file are the processed frames, one to one with
  the FT entries from the one at the time in its name on (alignCaptureTimes). A video of the capture
  (STATION_YYYYMMDD_HHMMSS_ffffff_video.mkv) is saved from the stream before the processing, which can drop frames
  the video keeps, so its frames are timed from their own timestamps in the video (the time in its name plus the
  presentation time of the frame), each snapped to the FT entry within half a frame (snapCaptureTimes).
  findCaptureTimeSources finds the FT files of an input in the station folders (SkyFit2 looks for them, other tools
  use them when they are given).

The times are the capture times of the frames (the timestamps of the FF files, the saved frames and the FT files),
without frame_time_offset, which the inputs apply as to their other times (not FF files, whose times are their
names).
"""

from __future__ import print_function, division, absolute_import

import datetime
import json
import os
import re
import tarfile

import cv2
import numpy as np

from RMS.Astrometry.Conversions import datetime2UnixTime, timeStringToDatetime
from RMS.Formats import FTfile
from RMS.Logger import getLogger
from RMS.Misc import UTCFromTimestamp
from Utils.GenerateTimelapse import frameTimesPaths


log = getLogger("rmslogger")


# The beginning and the end of a timelapse in its name (GenerateTimelapse.FNAME_TEMPLATE)
TIMELAPSE_NAME_PATTERN = re.compile(r"^(?P<station>.+?)_(?P<start>\d{8}-\d{6})_to_(?P<end>\d{8}-\d{6})_")

# The names of the FT files (with the time of their first frame) and of their daily archives (with their day)
FT_FILE_PATTERN = re.compile(r"^FT_.+\.bin$")
FT_FILE_TIME_PATTERN = re.compile(r"^FT_.+_(?P<time>\d{8}_\d{6})\.bin$")
FT_ARCHIVE_SUFFIXES = ('.tar', '.tar.bz2', '.tar.gz', '.tgz')
FT_ARCHIVE_DAY_PATTERN = re.compile(r"_(?P<day>\d{8})-\d{3}_FT\.")

# The capture times read from FT archives, by the path, the size and the time of change of the archive (reading
#   a daily archive takes seconds)
_CAPTURE_TIMES_CACHE = {}

# The name of a video of the capture (STATION_YYYYMMDD_HHMMSS_ffffff_video.mkv, see BufferedCapture.moveSegment)
CAPTURE_VIDEO_PATTERN = re.compile(r"^(?P<station>.+?)_(?P<time>\d{8}_\d{6}_\d{6})_video\.\w+$")

def parseTimelapseName(file_name):
    """ The station and the times of the first and the last frame (whole seconds) in the name of a timelapse (see
        GenerateTimelapse.FNAME_TEMPLATE), e.g. CAWEC4_20261122-224725_to_20261123-113909_frames_timelapse.mp4.

    Arguments:
        file_name: [str] Name or path of the file.

    Return:
        (station, start, end): [tuple] The station and the two times (datetime), None if it is not the name of a
            timelapse.
    """

    match = TIMELAPSE_NAME_PATTERN.match(os.path.basename(file_name))
    if match is None:
        return None

    try:
        start = datetime.datetime.strptime(match.group("start"), "%Y%m%d-%H%M%S")
        end = datetime.datetime.strptime(match.group("end"), "%Y%m%d-%H%M%S")

    except ValueError:
        return None

    return match.group("station"), start, end


def parseCaptureVideoName(file_name):
    """ The time of the first frame in the name of a video of the capture, e.g.
        CAWEA3_20260406_061051_773554_video.mkv.

    Arguments:
        file_name: [str] Name or path of the file.

    Return:
        [datetime] The time in the name, None if it is not the name of a video of the capture.
    """

    match = CAPTURE_VIDEO_PATTERN.match(os.path.basename(file_name))
    if match is None:
        return None

    try:
        return datetime.datetime.strptime(match.group("time"), "%Y%m%d_%H%M%S_%f")

    except ValueError:
        return None


def findFrameTimesFile(video_path):
    """ The frametimes JSON file next to a video (see GenerateTimelapse.frameTimesPaths), None if there is none. """

    for path in frameTimesPaths(video_path):
        if os.path.isfile(path):
            return path

    return None


def isCaptureTimesSource(path):
    """ Whether a source holds FT files: an FT file, a folder or an archive of FT files, a list of them, or the
        capture times already read (an array).
    """

    if isinstance(path, np.ndarray):
        return True

    if isinstance(path, (list, tuple)):
        return (len(path) > 0) and all(isCaptureTimesSource(entry) for entry in path)

    if not isinstance(path, str):
        return False

    name = os.path.basename(path)

    return os.path.isdir(path) or bool(FT_FILE_PATTERN.match(name)) or name.endswith(FT_ARCHIVE_SUFFIXES)


def ftFileTime(file_name):
    """ The time of the first frame in the name of an FT file, None if the name has none. """

    match = FT_FILE_TIME_PATTERN.match(os.path.basename(file_name))
    if match is None:
        return None

    return datetime.datetime.strptime(match.group("time"), "%Y%m%d_%H%M%S")


def findCaptureTimeSources(input_path, t_start, t_end, data_dirs=('VideoFiles', 'CapturedFiles', 'ArchivedFiles'),
                           times_dir='TimeFiles'):
    """ Find the FT files of the capture which hold the frames of an input between two times: the FT files and
        the daily FT archives (*_FT.tar.bz2 etc., only those of the days of the input) in the folder of the
        input, and the FT files of the hours of the input in the FT folders of the station
        (<data_dir>/<times_dir>/YYYY/YYYYMMDD-DoY/YYYYMMDD-DoY_HH/) when the input is in a station folder
        (<data_dir>/VideoFiles/..., <data_dir>/CapturedFiles/<night>/, <data_dir>/ArchivedFiles/<night>/).

    Arguments:
        input_path: [str] Path to the input (a file or a folder).
        t_start: [datetime] The time of the first frame of the input.
        t_end: [datetime] The time of the last frame of the input.

    Keyword arguments:
        data_dirs: [tuple] The folders of the station data folder which hold inputs (config.video_dir,
            config.captured_dir, config.archived_dir).
        times_dir: [str] The folder of the FT files in the station data folder (config.times_dir).

    Return:
        paths: [list of str] The FT files and archives, empty if there are none.
    """

    input_path = os.path.abspath(input_path)
    dir_path = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)

    # An FT file starts up to a block of frames before its frames, so the files which begin within a minute of
    #   the input are taken
    margin = datetime.timedelta(seconds=60)
    days = set()
    t = (t_start - margin).replace(hour=0, minute=0, second=0, microsecond=0)
    while t <= t_end + margin:
        days.add(t.strftime("%Y%m%d"))
        t += datetime.timedelta(days=1)

    def ftFileInRange(name):
        t_file = ftFileTime(name)
        return (t_file is None) or (t_start - margin <= t_file <= t_end + margin)

    paths = []

    # The FT files and the FT archives of the days of the input in its folder
    for name in sorted(os.listdir(dir_path)):

        if FT_FILE_PATTERN.match(name) and ftFileInRange(name):
            paths.append(os.path.join(dir_path, name))

        elif name.endswith(FT_ARCHIVE_SUFFIXES) and ('_FT' in name):
            match = FT_ARCHIVE_DAY_PATTERN.search(name)
            if (match is None) or (match.group("day") in days):
                paths.append(os.path.join(dir_path, name))

    # The station data folder, if the input is in one of its folders
    parts = dir_path.split(os.sep)
    data_dir = None
    for k in range(len(parts) - 1, 0, -1):
        if parts[k] in data_dirs:
            data_dir = os.sep.join(parts[:k])
            break

    # The FT files of the hours of the input in the FT folders of the station
    if data_dir is not None:
        t = (t_start - margin).replace(minute=0, second=0, microsecond=0)
        while t <= t_end + margin:
            hour_dir = os.path.join(data_dir, times_dir, t.strftime("%Y/%Y%m%d-%j/%Y%m%d-%j_%H"))
            if os.path.isdir(hour_dir):
                paths += [os.path.join(hour_dir, name) for name in sorted(os.listdir(hour_dir))
                          if FT_FILE_PATTERN.match(name) and ftFileInRange(name)]
            t += datetime.timedelta(hours=1)

    return paths


def readJSONFrameTimes(json_path):
    """ Read the times of the frames from a JSON file: {frame index: time} (e.g. the frametimes file of a
        timelapse) or a list of the times of the frames in order. The times can be in any form
        RMS.Astrometry.Conversions.timeStringToDatetime reads.

    Arguments:
        json_path: [str] Path to the JSON file.

    Return:
        (frame_times, invalid): [tuple]
            - frame_times: [dict] Frame index (int): time of the frame (naive UTC datetime).
            - invalid: [list] The entries which could not be read (key, value), left out of frame_times.

    Raises:
        ValueError: The file is not JSON or holds neither a dictionary nor a list. IOError/OSError: it cannot be
            read.
    """

    with open(json_path) as f:
        data = json.load(f)

    # The entries as (frame index, time)
    if isinstance(data, dict):
        entries = list(data.items())

    elif isinstance(data, list):
        entries = list(enumerate(data))

    else:
        raise ValueError("expected a dictionary of frame index: time or a list of times, got {}"\
            .format(type(data).__name__))

    frame_times = {}
    invalid = []

    for key, value in entries:

        # Read the frame index and the time of one frame, keep the entries which cannot be read aside
        try:
            frame_no = int(key)
            if frame_no < 0:
                raise ValueError
            frame_times[frame_no] = timeStringToDatetime(value)

        except (ValueError, TypeError, OverflowError, OSError):
            invalid.append((key, value))

    return frame_times, invalid


def readArchiveCaptureTimes(path):
    """ Read the capture times of the frames from an archive of FT files without extracting it, from the cache if
        the archive was read before (unchanged).

    Arguments:
        path: [str] Path to the archive (.tar, .tar.bz2, .tar.gz, .tgz).

    Return:
        [ndarray] The capture times (Unix seconds), unsorted.
    """

    stat = os.stat(path)
    key = (os.path.abspath(path), stat.st_size, stat.st_mtime)
    if key in _CAPTURE_TIMES_CACHE:
        return _CAPTURE_TIMES_CACHE[key]

    times = []
    with tarfile.open(path, 'r:*') as tar:
        for member in tar:
            if member.isfile() and FT_FILE_PATTERN.match(os.path.basename(member.name)):
                times.append(FTfile.readBuffer(tar.extractfile(member).read())[1])

    times = np.concatenate(times) if times else np.array([])
    _CAPTURE_TIMES_CACHE[key] = times

    return times


def readCaptureTimes(paths, t_start=None, t_end=None):
    """ Read the capture times of the frames from FT files: FT files, folders with FT files (searched
        recursively) and archives of FT files (read without extracting them, see readArchiveCaptureTimes).

    Arguments:
        paths: [str or list of str] The FT files, folders and archives.

    Keyword arguments:
        t_start: [datetime] Only the times from this time on. None by default.
        t_end: [datetime] Only the times up to this time. None by default.

    Return:
        [ndarray] The capture times of the frames (Unix seconds), sorted, without duplicates.

    Raises:
        ValueError: No FT files were found, or they hold fewer than two times. IOError/OSError: a file cannot be
            read.
    """

    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    times = []

    for path in paths:

        # A folder: every FT file in it and in its subfolders
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for name in sorted(files):
                    if FT_FILE_PATTERN.match(name):
                        with open(os.path.join(root, name), 'rb') as f:
                            times.append(FTfile.readBuffer(f.read())[1])

        # An archive: every FT file in it
        elif os.path.basename(path).endswith(FT_ARCHIVE_SUFFIXES):
            times.append(readArchiveCaptureTimes(path))

        # A single FT file
        else:
            with open(path, 'rb') as f:
                times.append(FTfile.readBuffer(f.read())[1])

    if not times:
        raise ValueError("no FT files found in {}".format(", ".join(paths)))

    # All times, in order and once (a frame is in one FT file only, but files can be given twice)
    times = np.unique(np.concatenate(times))
    times = times[times > 0]

    # Only the times of the given span
    if t_start is not None:
        times = times[times >= datetime2UnixTime(t_start)]
    if t_end is not None:
        times = times[times <= datetime2UnixTime(t_end)]

    if len(times) < 2:
        raise ValueError("the FT files in {} hold fewer than two frames of the input".format(", ".join(paths)))

    return times


def framePeriod(capture_times):
    """ The frame period of the capture (s), the median spacing of the capture times. """

    return float(np.median(np.diff(capture_times)))


def alignCaptureTimes(capture_times, start_time, n_frames):
    """ The capture times of the frames of an FF file: the first frame is the frame captured at the time in its name
        (the capture time within half a frame of it), and the following frames are the following captured frames.
        The FF files are made of the frames the capture processed, which are the frames of the FT files, one to one
        (a video of the capture is not, see snapCaptureTimes).

    Arguments:
        capture_times: [ndarray] The capture times (Unix seconds, sorted, see readCaptureTimes).
        start_time: [datetime] The time of the first frame (from the name of the FF file).
        n_frames: [int] The number of frames of the FF file.

    Return:
        frame_times: [dict] Frame index: capture time (datetime), for the frames within the capture times.

    Raises:
        ValueError: No capture time is within half a frame of the time of the first frame.
    """

    period = framePeriod(capture_times)
    t_start = datetime2UnixTime(start_time)

    # The capture time closest to the beginning of the video
    i = int(np.searchsorted(capture_times, t_start))
    candidates = [j for j in (i - 1, i) if 0 <= j < len(capture_times)]
    first = min(candidates, key=lambda j: abs(capture_times[j] - t_start))

    if abs(capture_times[first] - t_start) > period/2:
        raise ValueError("no capture time within half a frame ({:.3f} s) of the beginning {}"\
            .format(period/2, start_time))

    # The frames of the video are the captured frames from there on
    return {k: UTCFromTimestamp.utcfromtimestamp(float(t)) for k, t in enumerate(capture_times[first:first + n_frames])}


def readVideoTimestamps(video_path):
    """ Read the presentation times of the frames of a video file (from the time of its first frame), which keep
        the gaps of frames the camera or the network dropped.

    Arguments:
        video_path: [str] Path to the video file.

    Return:
        [ndarray] The times of the frames from the first one (s).

    Raises:
        ValueError: The video has no frames, or its timestamps do not increase (no usable timestamps).
    """

    # Read through the video (without converting the frames) and take the timestamp of every frame
    cap = cv2.VideoCapture(video_path)
    times = []
    try:
        while cap.grab():
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)

    finally:
        cap.release()

    times = np.array(times)

    if len(times) < 2:
        raise ValueError("fewer than two frames with timestamps in {:s}".format(video_path))

    if np.any(np.diff(times) <= 0):
        raise ValueError("the timestamps of the frames of {:s} do not increase".format(video_path))

    return times - times[0]


def snapCaptureTimes(frame_times, capture_times):
    """ Snap the times of the frames of a video of the capture (the time in its name plus the timestamps of the
        frames) to the capture times of the FT files: a frame takes the capture time within half a frame of it,
        and keeps its own time without one (a frame which the capture did not process).

    Arguments:
        frame_times: [ndarray] The times of the frames (Unix seconds).
        capture_times: [ndarray] The capture times (Unix seconds, sorted).

    Return:
        (times, matched): [tuple] The snapped times of the frames (Unix seconds), and which frames were snapped
            (bool array).
    """

    tol = framePeriod(capture_times)/2

    # The closest capture time to every frame
    i = np.clip(np.searchsorted(capture_times, frame_times), 1, len(capture_times) - 1)
    before, after = capture_times[i - 1], capture_times[i]
    nearest = np.where(np.abs(frame_times - before) <= np.abs(after - frame_times), before, after)

    matched = np.abs(nearest - frame_times) <= tol

    return np.where(matched, nearest, frame_times), matched


def loadFrameTimes(source):
    """ Load the times of the frames of an input from a JSON file (see readJSONFrameTimes). FT files time FF files
        and videos of the capture only (see alignCaptureTimes and snapCaptureTimes).

    Arguments:
        source: [str] Path to the JSON file.

    Return:
        frame_times: [dict] Frame index: time (naive UTC datetime).

    Raises:
        ValueError: The times cannot be read, or the source is FT files. IOError/OSError: the file cannot be read.
    """

    if isCaptureTimesSource(source):
        raise ValueError("FT files time the frames of FF files and of videos of the capture only")

    frame_times, invalid = readJSONFrameTimes(source)

    if invalid:
        log.warning("{:d} entries of the frame times file {:s} cannot be read and are left out (e.g. {:s})"\
            .format(len(invalid), source, str(invalid[0])))

    return frame_times


class FrameTimeTable(object):
    def __init__(self, frame_times):
        """ The times of the frames of an input, with the frames missing from them timed between the nearest
            frames with times (or at the median spacing before the first and after the last one).

        Arguments:
            frame_times: [dict] Frame index (int): time (datetime).

        Raises:
            ValueError: Fewer than two frames, or times which do not increase with the frames.
        """

        if len(frame_times) < 2:
            raise ValueError("fewer than two frames have times")

        # The frame indices in order, with their times in seconds from the time of the first one
        self.frame_times = dict(frame_times)
        self.frames = np.array(sorted(frame_times), dtype=np.float64)
        self.ref = frame_times[int(self.frames[0])]
        self.secs = np.array([(frame_times[int(fr)] - self.ref).total_seconds() for fr in self.frames])

        # The times have to increase with the frames
        if np.any(np.diff(self.secs) <= 0):
            raise ValueError("the times do not increase with the frames")

        # The median spacing of the frames, from the consecutive frames (any frames if no two are consecutive)
        steps = np.diff(self.secs)/np.diff(self.frames)
        consecutive = np.diff(self.frames) == 1
        if np.any(consecutive):
            steps = steps[consecutive]

        self.spacing = float(np.median(steps))

    def time(self, frame_no):
        """ The time of a frame, also of a frame without a time or a fractional frame (interpolated).

        Arguments:
            frame_no: [float] Frame index.

        Return:
            [datetime] The time of the frame.
        """

        frames, secs = self.frames, self.secs

        # Before the first frame with a time, at the median spacing
        if frame_no < frames[0]:
            t = secs[0] - (frames[0] - frame_no)*self.spacing

        # After the last frame with a time, at the median spacing
        elif frame_no > frames[-1]:
            t = secs[-1] + (frame_no - frames[-1])*self.spacing

        # The time of the frame, or interpolated between the nearest frames with times
        else:
            t = float(np.interp(frame_no, frames, secs))

        return self.ref + datetime.timedelta(seconds=t)

    def missing(self, n_frames):
        """ The frames of an input with n_frames frames which have no time. """

        return [i for i in range(n_frames) if i not in self.frame_times]

    def gaps(self, factor=1.5):
        """ The gaps between consecutive frames longer than factor times the median spacing.

        Return:
            [list] (frame index after the gap, length of the gap in s) for every gap.
        """

        gaps = []
        for i in range(1, len(self.frames)):
            if self.frames[i] - self.frames[i - 1] != 1:
                continue
            step = self.secs[i] - self.secs[i - 1]
            if step > factor*self.spacing:
                gaps.append((int(self.frames[i]), step))

        return gaps
