"""
Tests of the timing of the frames of videos and image sequences from a file with the time of every frame
(RMS/Formats/FrameTimes.py), and of its use in the inputs (RMS/Formats/FrameInterface.py) and in SkyFit2.

A timelapse of the saved frames (Utils/GenerateTimelapse.py) is missing the frames which were not saved and its
spacing jitters, so its frames are timed from the frametimes JSON file next to it. A video of the capture and the FF
files are timed from the FT files of the capture, from the frame captured at the time in their name on (SkyFit2
finds the FT files, also in their daily archives). Without a file of frame times, the frames are timed as before.
"""

import datetime
import json
import os
import tarfile
import types

import cv2
import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Formats import FTfile
from RMS.Formats.FTStruct import FTStruct
from RMS.Formats.FrameInterface import InputTypeImages, InputTypeVideo, detectInputType
from RMS.Formats import FrameTimes
from RMS.Formats.FrameTimes import FrameTimeTable, alignCaptureTimes, datetimeToUnix, findCaptureTimeSources, \
    findFrameTimesFile, frameTimesPaths, parseCaptureVideoName, parseTime, parseTimelapseName, readCaptureTimes, \
    readJSONFrameTimes, unixToDatetime
from RMS.Formats import FFfile
from RMS.Formats.FFStruct import FFStruct
from RMS.Formats.FrameInterface import InputTypeFRFF
from RMS.Routines import RollingShutterCorrection
from Utils.GenerateTimelapse import generateTimelapseFromFrames, isFfmpegWorking


OFFSET = 0.084

# Playback rate of the synthetic timelapses (the rate GenerateTimelapse encodes them at)
PLAYBACK_FPS = 30.0

# Frame rate of the synthetic capture
CAPTURE_FPS = 25.0

# Capture time of the first frame of the synthetic timelapses
T0 = datetime.datetime(2026, 11, 22, 22, 47, 25, 2000)

# Seconds of the frames from the first: 5 s apart with jitter, 9 frames not saved after the second frame (as in a
#   real night), and a spacing of 5.1 and 4.9 s
FRAME_SECS = [0.0, 5.003, 55.012, 60.015, 65.019, 70.119, 75.018, 80.022, 85.025, 90.029, 95.033, 100.037]

TIMELAPSE_NAME = "XX0001_20261122-224725_to_20261122-224905"


def makeConfig(offset=0.0):
    """ A default config read through OpenCV, with the given correction of the timestamps. """

    config = cr.Config()
    config.media_backend = 'cv2'
    config.frame_time_offset = offset

    return config


def writeVideo(path, n_frames=len(FRAME_SECS), fps=PLAYBACK_FPS):
    """ A small video of blank frames (MP4, or AVI for other names). """

    fourcc = cv2.VideoWriter_fourcc(*('mp4v' if path.endswith('.mp4') else 'MJPG'))
    writer = cv2.VideoWriter(path, fourcc, fps, (64, 48), True)
    for i in range(n_frames):
        writer.write(np.full((48, 64, 3), 10*i, dtype=np.uint8))
    writer.release()


def stamp(dt, mode='_n'):
    """ The time part of the name of a saved frame (as in the frametimes JSON file), milliseconds truncated. """

    return dt.strftime("%Y%m%d_%H%M%S_") + "{:03d}".format(dt.microsecond//1000) + mode


def captureTimes(secs=FRAME_SECS):
    """ The capture times of the frames of the synthetic timelapse. """

    return [T0 + datetime.timedelta(seconds=s) for s in secs]


def writeJSON(path, data):
    """ Write a JSON file. """

    with open(str(path), 'w') as f:
        json.dump(data, f)


def writeTimelapse(dir_path, name=TIMELAPSE_NAME + "_frames_timelapse.mp4",
                   json_name=TIMELAPSE_NAME + "_frametimes.json", times=None):
    """ A timelapse video with its frametimes JSON file (not written if json_name is None), named as
        GenerateTimelapse names them.
    """

    video_path = os.path.join(str(dir_path), name)
    writeVideo(video_path)

    if json_name is not None:
        if times is None:
            times = {str(i): stamp(t) for i, t in enumerate(captureTimes())}
        writeJSON(os.path.join(str(dir_path), json_name), times)

    return video_path


def frameTimes(handle, frames=range(len(FRAME_SECS))):
    """ The times of some frames of an input. """

    return [handle.currentFrameTime(frame_no=i, dt_obj=True) for i in frames]


def syntheticCapture(t_start, duration, gaps=(), fps=CAPTURE_FPS, seed=1):
    """ Capture times (Unix seconds) of a capture at the given frame rate with a jitter of 2 ms, without the frames
        in the gaps.

    Arguments:
        t_start: [datetime] Time of the first frame.
        duration: [float] Length of the capture (s).

    Keyword arguments:
        gaps: [list] (start, end) of the gaps, in seconds from the beginning.
    """

    rng = np.random.RandomState(seed)
    secs = np.arange(0, duration, 1/fps)
    secs = secs + rng.uniform(-0.002, 0.002, len(secs))
    for g0, g1 in gaps:
        secs = secs[(secs < g0) | (secs >= g1)]

    return datetimeToUnix(t_start) + secs


def writeFT(dir_path, times, block=256, station="XX0001"):
    """ Write the capture times into FT files of blocks of frames, as the capture does. Return the paths. """

    paths = []
    for i in range(0, len(times), block):
        ft = FTStruct()
        ft.timestamps = [(i + k, float(t)) for k, t in enumerate(times[i:i + block])]
        name = unixToDatetime(times[i]).strftime("FT_{:s}_%Y%m%d_%H%M%S.bin".format(station))
        FTfile.write(ft, str(dir_path), name)
        paths.append(os.path.join(str(dir_path), name))

    return paths


### Reading the times ###

@pytest.mark.parametrize("value, expected", [
    ("20261122_224725_002_n", datetime.datetime(2026, 11, 22, 22, 47, 25, 2000)),
    ("20261122_224725_002_d", datetime.datetime(2026, 11, 22, 22, 47, 25, 2000)),
    ("20261122_224725_002", datetime.datetime(2026, 11, 22, 22, 47, 25, 2000)),
    ("2026-11-22T22:47:25.002000", datetime.datetime(2026, 11, 22, 22, 47, 25, 2000)),
    ("2026-11-22T22:47:25Z", datetime.datetime(2026, 11, 22, 22, 47, 25)),
    ("2026-11-22 23:47:25.5+01:00", datetime.datetime(2026, 11, 22, 22, 47, 25, 500000)),
    (1795387645.25, datetime.datetime(2026, 11, 22, 22, 47, 25, 250000)),
    ("1795387645", datetime.datetime(2026, 11, 22, 22, 47, 25)),
])
def test_parse_time(value, expected):

    # The stamp of the saved frames, ISO 8601 (UTC or with an offset) and Unix seconds
    assert parseTime(value) == expected


@pytest.mark.parametrize("value", ["broken", "2026-13-01T00:00:00", True, None, [1]])
def test_parse_time_rejects(value):

    with pytest.raises(ValueError):
        parseTime(value)


def test_read_json_dictionary_and_list(tmp_path):

    # A dictionary with the stamps of the saved frames, an entry without a time and one with a bad index
    path = tmp_path/"a_frametimes.json"
    writeJSON(path, {"0": "20261122_224725_002_n", "1": "20261122_224730_005_d", "2": "20261122_224735_007",
                     "3": "broken", "x": "20261122_224740_000_n"})
    frame_times, invalid = readJSONFrameTimes(str(path))

    assert frame_times == {0: datetime.datetime(2026, 11, 22, 22, 47, 25, 2000),
                           1: datetime.datetime(2026, 11, 22, 22, 47, 30, 5000),
                           2: datetime.datetime(2026, 11, 22, 22, 47, 35, 7000)}
    assert sorted(key for key, _ in invalid) == ["3", "x"]

    # A list of ISO and Unix times in the order of the frames
    path = tmp_path/"b.json"
    writeJSON(path, ["2026-11-22T22:47:25.002Z", 1795387650.005])
    frame_times, invalid = readJSONFrameTimes(str(path))

    assert frame_times == {0: datetime.datetime(2026, 11, 22, 22, 47, 25, 2000),
                           1: datetime.datetime(2026, 11, 22, 22, 47, 30, 5000)}
    assert invalid == []


def test_names():

    # The two names of the frametimes file written by GenerateTimelapse
    assert frameTimesPaths("/d/XX0001_a_to_b_frames_timelapse.mp4")[0] == "/d/XX0001_a_to_b_frametimes.json"
    assert frameTimesPaths("/d/night.mp4") == ["/d/night_frametimes.json"]

    # The name of a timelapse, and of a video of the capture
    assert parseTimelapseName("/d/CAWEC4_20261122-224725_to_20261123-113909_frames_timelapse.mp4") == \
        ("CAWEC4", datetime.datetime(2026, 11, 22, 22, 47, 25), datetime.datetime(2026, 11, 23, 11, 39, 9))
    assert parseTimelapseName("CAWEC4_20261122_224725_002000_video.mkv") is None
    assert parseCaptureVideoName("CAWEC4_20261122_224725_002000_video.mkv") == \
        datetime.datetime(2026, 11, 22, 22, 47, 25, 2000)
    assert parseCaptureVideoName("night.mp4") is None


def test_frame_time_table():

    # Frames 3, 4 and 6 have no time
    times = {i: t for i, t in enumerate(captureTimes()) if i not in (3, 4, 6)}
    table = FrameTimeTable(times)
    expected = captureTimes()

    # The frames with times keep them, the others are interpolated, before and after at the median spacing
    assert table.time(2) == expected[2]
    assert table.time(3) == expected[2] + (expected[5] - expected[2])/3
    assert table.time(6) == expected[5] + (expected[7] - expected[5])/2
    assert (table.time(-1) - expected[0]).total_seconds() == pytest.approx(-table.spacing)
    assert table.missing(12) == [3, 4, 6]

    # The gap of the frames which were not saved
    assert [fr for fr, _ in table.gaps()] == [2]

    # Fewer than two frames, or times which go back
    with pytest.raises(ValueError):
        FrameTimeTable({0: T0})
    with pytest.raises(ValueError):
        FrameTimeTable({0: T0, 1: T0 - datetime.timedelta(seconds=1)})


### FT files ###

def test_read_capture_times_from_file_folder_and_archive(tmp_path):

    # FT files of a capture in the folder tree of the station
    times = syntheticCapture(T0, 30.0)
    hour_dir = tmp_path/"TimeFiles"/"2026"/"20261122-326"/"20261122-326_22"
    hour_dir.mkdir(parents=True)
    paths = writeFT(hour_dir, times)

    # A single file, the folder (searched recursively) and an archive of it
    assert np.array_equal(readCaptureTimes(paths[0]), times[:256])
    assert np.array_equal(readCaptureTimes(str(tmp_path/"TimeFiles")), times)

    archive = str(tmp_path/"XX0001_20261122-326_FT.tar.bz2")
    with tarfile.open(archive, 'w:bz2') as tar:
        tar.add(str(tmp_path/"TimeFiles"/"2026"/"20261122-326"), arcname="20261122-326")
    assert np.array_equal(readCaptureTimes(archive), times)

    # A file given twice counts once
    assert np.array_equal(readCaptureTimes([paths[0], str(tmp_path/"TimeFiles")]), times)


def test_ft_file_read_unchanged(tmp_path):
    """ Guard: FTfile.read gives the frame counters and the times as before. """

    ft = FTStruct()
    ft.timestamps = [(1, 0.033), (2, 0.066), (3, 1795387645.123456)]
    FTfile.write(ft, str(tmp_path), "FT_test.bin")

    assert FTfile.read(str(tmp_path), "FT_test.bin").timestamps == ft.timestamps


def test_align_capture_times_with_dropped_frames():

    # A capture which dropped the frames between 2 and 3 s
    times = syntheticCapture(T0, 10.0, gaps=[(2.0, 3.0)])

    # A video beginning at a captured frame (its name has the time to the microsecond, close to the frame)
    start = unixToDatetime(times[10]) + datetime.timedelta(microseconds=300)
    frame_times = alignCaptureTimes(times, start, 100)

    # The frames are the captured frames from there on, also after the dropped ones
    assert frame_times[0] == unixToDatetime(times[10])
    assert frame_times[60] == unixToDatetime(times[70])
    assert np.max(np.diff([datetimeToUnix(frame_times[k]) for k in range(100)])) > 0.9

    # No captured frame at the beginning
    with pytest.raises(ValueError):
        alignCaptureTimes(times, unixToDatetime(times[0]) - datetime.timedelta(seconds=5), 100)


### Video, image and FF inputs ###

def test_dropped_and_jittered_frames_get_their_capture_times(tmp_path):

    # A timelapse with dropped frames and jitter, named as GenerateTimelapse names it
    handle = InputTypeVideo(writeTimelapse(tmp_path), makeConfig())

    # Every frame has its capture time, not the time from the beginning and the playback rate
    assert frameTimes(handle) == captureTimes()
    assert handle.beginning_datetime == T0

    # The time of the current frame (as shown in SkyFit2) is the capture time
    handle.setFrame(5)
    assert handle.currentFrameTime(dt_obj=True) == captureTimes()[5]

    # The frame rate is the median spacing of the frames, not the playback rate
    assert 1.0/handle.fps == pytest.approx(np.median(np.diff(FRAME_SECS)))


def test_chunk_time_from_capture_times(tmp_path):

    # Chunks of 4 frames: the mean time of the frames of the chunk, or the time of its first frame
    handle = InputTypeVideo(writeTimelapse(tmp_path), makeConfig(), chunk_frames=4)
    expected = captureTimes()

    def meanTime(frames):
        return expected[frames[0]] + datetime.timedelta(
            seconds=np.mean([(expected[i] - expected[frames[0]]).total_seconds() for i in frames]))

    assert abs((handle.currentTime(dt_obj=True) - meanTime([0, 1, 2, 3])).total_seconds()) < 1e-6
    assert handle.currentTime(dt_obj=True, beginning=True) == expected[0]

    handle.nextChunk()
    handle.loadChunk()
    assert abs((handle.currentTime(dt_obj=True) - meanTime([4, 5, 6, 7])).total_seconds()) < 1e-6


def test_renamed_video_with_given_frame_times(tmp_path):

    # A renamed timelapse with its JSON file next to it under its name, and with a JSON file of ISO times given
    path = writeTimelapse(tmp_path, name="night.mp4", json_name="night_frametimes.json")
    assert findFrameTimesFile(path) == str(tmp_path/"night_frametimes.json")
    assert frameTimes(InputTypeVideo(path, makeConfig())) == captureTimes()

    iso_path = str(tmp_path/"times.json")
    writeJSON(iso_path, [t.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for t in captureTimes()])
    other = writeTimelapse(tmp_path, name="renamed.mp4", json_name=None)
    assert frameTimes(InputTypeVideo(other, makeConfig(), frame_times=iso_path)) == captureTimes()

    # The same through the detection of the input type (as SkyFit2 and the detection open their inputs)
    assert frameTimes(detectInputType(other, makeConfig(), frame_times=iso_path)) == captureTimes()


def test_timelapse_without_frame_times_is_timed_from_its_name(tmp_path):

    # A timelapse without its frametimes file: it begins at the time in its name, at the interval of the saved
    #   frames (it could not be loaded before)
    handle = InputTypeVideo(writeTimelapse(tmp_path, json_name=None), makeConfig())

    assert handle.frame_time_table is None
    assert handle.timelapse_without_times
    assert handle.beginning_datetime == T0.replace(microsecond=0)
    assert handle.currentFrameTime(frame_no=3, dt_obj=True) == T0.replace(microsecond=0) \
        + datetime.timedelta(seconds=15)

    # The warning says what can be wrong and how to fix it
    message = handle.timelapseWithoutTimesMessage()
    assert (TIMELAPSE_NAME + "_frametimes.json") in message
    assert "--frametimes" in message


def test_frames_missing_from_the_json_are_interpolated(tmp_path):

    # Frames 3 and 4 have no time (an image whose name could not be read), frame 6 has a broken one, and the
    #   last frame is missing
    times = {str(i): stamp(t) for i, t in enumerate(captureTimes())}
    del times["3"], times["4"], times[str(len(FRAME_SECS) - 1)]
    times["6"] = "broken"

    handle = InputTypeVideo(writeTimelapse(tmp_path, times=times), makeConfig())
    t = frameTimes(handle)
    expected = captureTimes()

    for i in [0, 1, 2, 5, 7, 8, 9, 10]:
        assert t[i] == expected[i]
    assert t[3] == expected[2] + (expected[5] - expected[2])/3
    assert t[6] == expected[5] + (expected[7] - expected[5])/2
    assert (t[-1] - expected[-2]).total_seconds() == pytest.approx(1.0/handle.fps, abs=1e-6)


@pytest.mark.parametrize("content", ["{not json", "[1, 2, 3", '{"0": "20261122_224725_002_n"}',
                                     '{"0": "20261122_224730_002_n", "1": "20261122_224725_002_n"}'])
def test_unusable_json_times_every_interval(tmp_path, content):

    # A video timed by its name with a frametimes file which is not JSON, has a single frame or times which go back
    name = T0.strftime("%Y%m%d_%H%M%S.%f")
    path = writeTimelapse(tmp_path, name=name + ".mp4", json_name=None)
    with open(os.path.join(str(tmp_path), name + "_frametimes.json"), 'w') as f:
        f.write(content)

    handle = InputTypeVideo(path, makeConfig())

    # The video is a timelapse without usable frame times: its frames are timed every 5 s (the interval of the
    #   saved frames) from the beginning time in its name, with the warning
    assert getattr(handle, "frame_time_table", None) is None
    assert handle.timelapse_without_times
    assert handle.beginning_datetime == T0
    assert frameTimes(handle, [0, 7]) == [T0, T0 + datetime.timedelta(seconds=35)]


def test_frame_time_offset_applied(tmp_path):

    path = writeTimelapse(tmp_path)
    plain = InputTypeVideo(path, makeConfig(0.0))
    corrected = InputTypeVideo(path, makeConfig(OFFSET))

    # The beginning, every frame and the chunk times are later by the correction
    assert (corrected.beginning_datetime - plain.beginning_datetime).total_seconds() == pytest.approx(OFFSET)
    for a, b in zip(frameTimes(plain), frameTimes(corrected)):
        assert (b - a).total_seconds() == pytest.approx(OFFSET, abs=1e-6)
    assert (corrected.currentTime(dt_obj=True) - plain.currentTime(dt_obj=True)).total_seconds() \
        == pytest.approx(OFFSET, abs=1e-6)


def test_given_beginning_time_does_not_override_the_json(tmp_path):

    # A beginning time given by hand (the only way to load a timelapse before) is not used with the JSON file
    handle = InputTypeVideo(writeTimelapse(tmp_path), makeConfig(),
                            beginning_time=T0.replace(microsecond=0) - datetime.timedelta(seconds=30))

    assert frameTimes(handle) == captureTimes()


def test_video_without_frame_times_unchanged(tmp_path):
    """ Guard: a video without a file of frame times is timed from the beginning time in its name and the
        playback rate, as before. """

    name = T0.strftime("%Y%m%d_%H%M%S.%f")
    handle = InputTypeVideo(writeTimelapse(tmp_path, name=name + ".mp4", json_name=None), makeConfig(OFFSET),
                            chunk_frames=4)

    t_beg = T0 + datetime.timedelta(seconds=OFFSET)
    assert getattr(handle, "frame_time_table", None) is None
    assert handle.fps == pytest.approx(PLAYBACK_FPS)
    assert frameTimes(handle, [0, 7]) == [t_beg, t_beg + datetime.timedelta(seconds=7/handle.fps)]
    assert handle.currentTime(dt_obj=True) == t_beg + datetime.timedelta(seconds=2/handle.fps)


def test_image_folder_with_frame_times(tmp_path):

    # A folder of images with irregular times, without a time in its name
    img_dir = tmp_path/"images"
    img_dir.mkdir()
    for i in range(len(FRAME_SECS)):
        cv2.imwrite(str(img_dir/"img_{:03d}.png".format(i)), np.full((48, 64), 20, dtype=np.uint8))

    times_path = str(tmp_path/"times.json")
    writeJSON(times_path, {str(i): stamp(t) for i, t in enumerate(captureTimes())})

    config = makeConfig(OFFSET)
    config.width, config.height = 64, 48
    handle = InputTypeImages(str(img_dir), config, frame_times=times_path, chunk_frames=4)

    offset = datetime.timedelta(seconds=OFFSET)
    assert frameTimes(handle) == [t + offset for t in captureTimes()]
    assert handle.beginning_datetime == T0 + offset


def test_generated_timelapse_round_trip(tmp_path):

    # A timelapse made by GenerateTimelapse from saved frames with a frame not saved and jitter
    if not isFfmpegWorking():
        pytest.skip("ffmpeg is not available")

    frames_dir = tmp_path/"frames"
    frames_dir.mkdir()
    image_paths = []
    for t in captureTimes():
        image_path = str(frames_dir/("XX0001_" + stamp(t) + ".jpg"))
        cv2.imwrite(image_path, np.full((48, 64), 50, dtype=np.uint8))
        image_paths.append(image_path)

    video_path = str(tmp_path/(TIMELAPSE_NAME + "_frames_timelapse.mp4"))
    video_path, json_path = generateTimelapseFromFrames(image_paths, str(frames_dir), video_path)
    assert json_path == str(tmp_path/(TIMELAPSE_NAME + "_frametimes.json"))

    # The video read back has the capture times in the names of the images
    handle = InputTypeVideo(video_path, makeConfig())
    assert handle.total_frames == len(FRAME_SECS)
    assert frameTimes(handle) == captureTimes()


def test_capture_video_timed_from_given_ft_files(tmp_path):

    # A video of the capture with its FT files given; the capture dropped the frames between 0.5 and 0.7 s
    times = syntheticCapture(T0, 3.0, gaps=[(0.5, 0.7)])
    ft_dir = tmp_path/"ft"
    ft_dir.mkdir()
    writeFT(ft_dir, times)

    name = "XX0001_" + unixToDatetime(times[0]).strftime("%Y%m%d_%H%M%S_%f") + "_video.avi"
    path = str(tmp_path/name)
    writeVideo(path, n_frames=40, fps=CAPTURE_FPS)

    # The frames are the captured frames, the frames after the dropped ones are later than by the frame rate
    handle = InputTypeVideo(path, makeConfig(OFFSET), frame_times=str(ft_dir))
    offset = datetime.timedelta(seconds=OFFSET)
    assert frameTimes(handle, range(40)) == [unixToDatetime(t) + offset for t in times[:40]]
    assert (handle.currentFrameTime(frame_no=20, dt_obj=True) - handle.currentFrameTime(frame_no=0, dt_obj=True))\
        .total_seconds() > 20/CAPTURE_FPS + 0.15

    # Without the FT files given, the video is timed from its name and its frame rate, as before
    plain = InputTypeVideo(path, makeConfig())
    assert plain.frame_time_table is None
    assert plain.currentFrameTime(frame_no=20, dt_obj=True) == parseCaptureVideoName(name) \
        + datetime.timedelta(seconds=20/plain.fps)


def test_ft_files_do_not_time_a_timelapse(tmp_path):

    # FT files given for a timelapse are not used (its frames are seconds apart): its frametimes file is
    times = syntheticCapture(T0, 30.0)
    ft_dir = tmp_path/"ft"
    ft_dir.mkdir()
    writeFT(ft_dir, times)

    handle = InputTypeVideo(writeTimelapse(tmp_path), makeConfig(), frame_times=str(ft_dir))
    assert frameTimes(handle) == captureTimes()


def ffName(t, station="XX0001"):
    """ The name of an FF file which begins at the given time. """

    return t.strftime("FF_{:s}_%Y%m%d_%H%M%S_".format(station)) + "{:03d}".format(t.microsecond//1000) \
        + "_0000000.fits"


def writeFF(dir_path, file_name):
    """ Write a small FF file with flat images. """

    ff = FFStruct()
    ff.nrows, ff.ncols = 48, 64
    ff.nbits = 8
    ff.nframes = 256
    ff.fps = CAPTURE_FPS
    ff.starttime = "2026-11-22T22:47:25.002"
    ff.maxpixel = np.full((48, 64), 30, dtype=np.uint8)
    ff.avepixel = np.full((48, 64), 20, dtype=np.uint8)
    ff.stdpixel = np.zeros((48, 64), dtype=np.uint8)
    ff.maxframe = np.zeros((48, 64), dtype=np.uint8)

    FFfile.write(ff, str(dir_path), file_name, fmt='fits')


def ffCapture(n_blocks=3, gap=(3.0, 3.4)):
    """ The capture times of blocks of 256 frames, with frames dropped in the gap of the first block, and the
        names of the FF files of the blocks (the time of their first frame, truncated to the millisecond).
    """

    times = syntheticCapture(T0, n_blocks*256/CAPTURE_FPS + 5.0, gaps=[gap])
    names = [ffName(unixToDatetime(times[256*k])) for k in range(n_blocks)]

    return times, names


def test_ff_files_timed_from_ft_files(tmp_path):

    # A folder of FF files with the FT files of the capture; frames were dropped in the first block
    times, names = ffCapture()
    for name in names:
        writeFF(tmp_path, name)
    ft_dir = tmp_path/"ft"
    ft_dir.mkdir()
    writeFT(ft_dir, times)

    config = makeConfig(0.0)
    handle = InputTypeFRFF(str(tmp_path), config, frame_times=str(ft_dir))

    # Every frame of every FF file is its captured frame (no frame_time_offset, as for the names of FF files)
    for k, name in enumerate(names):
        handle.setCurrentFF(name)
        assert frameTimes(handle, range(256)) == [unixToDatetime(t) for t in times[256*k:256*(k + 1)]]

        # The middle of the file is the time at half of its frames
        assert handle.currentTime(dt_obj=True) == handle.frame_time_table.time(128)

    # After the dropped frames, the times are later than from the name and the frame rate
    handle.setCurrentFF(names[0])
    from_name = FFfile.filenameToDatetime(names[0]) + datetime.timedelta(seconds=200/CAPTURE_FPS)
    assert (handle.currentFrameTime(frame_no=200, dt_obj=True) - from_name).total_seconds() > 0.3


def test_ff_file_without_ft_times_unchanged(tmp_path):
    """ Guard: an FF file without capture times at its time is timed from its name and the frame rate. """

    times, names = ffCapture()
    writeFF(tmp_path, names[0])

    # FT files of another time
    other = times + 3600.0
    handle = InputTypeFRFF(os.path.join(str(tmp_path), names[0]), makeConfig(), single_ff=True,
                           frame_times=other)

    assert handle.frame_time_table is None
    assert handle.currentFrameTime(frame_no=100, dt_obj=True) == FFfile.filenameToDatetime(names[0]) \
        + datetime.timedelta(seconds=100/handle.fps)


def test_find_capture_time_sources(tmp_path):

    # The station folders: FF files of a night, the FT files in the FT folders and the daily archives
    t_start = datetime.datetime(2026, 11, 22, 22, 47, 25)
    data_dir = tmp_path/"RMS_data"
    night_dir = data_dir/"ArchivedFiles"/"XX0001_20261122_224724_669632"
    night_dir.mkdir(parents=True)
    for day in ["20261121-325", "20261122-326", "20261123-327", "20261124-328"]:
        (night_dir/("XX0001_" + day + "_FT.tar.bz2")).write_bytes(b"")
    (night_dir/"FT_XX0001_20261122_224710.bin").write_bytes(b"")
    (night_dir/"FT_XX0001_20261122_120000.bin").write_bytes(b"")

    hour_dir = data_dir/"TimeFiles"/"2026"/"20261122-326"/"20261122-326_22"
    hour_dir.mkdir(parents=True)
    (hour_dir/"FT_XX0001_20261122_224715.bin").write_bytes(b"")
    (hour_dir/"FT_XX0001_20261122_221000.bin").write_bytes(b"")

    paths = findCaptureTimeSources(str(night_dir), t_start, datetime.datetime(2026, 11, 23, 11, 39, 9))
    names = sorted(os.path.basename(path) for path in paths)

    # Only the archives of the days of the input, and the FT files of its span
    assert names == ["FT_XX0001_20261122_224710.bin", "FT_XX0001_20261122_224715.bin",
                     "XX0001_20261122-326_FT.tar.bz2", "XX0001_20261123-327_FT.tar.bz2"]


def test_archive_read_once(tmp_path, monkeypatch):

    # A daily archive of FT files is read once, then taken from the cache
    times = syntheticCapture(T0, 30.0)
    ft_dir = tmp_path/"20261122-326"
    ft_dir.mkdir()
    writeFT(ft_dir, times)
    archive = str(tmp_path/"XX0001_20261122-326_FT.tar.bz2")
    with tarfile.open(archive, 'w:bz2') as tar:
        tar.add(str(ft_dir), arcname="20261122-326")

    opened = []
    real_open = tarfile.open
    monkeypatch.setattr(FrameTimes.tarfile, "open", lambda *args, **kwargs: opened.append(args[0])
                        or real_open(*args, **kwargs))

    first = readCaptureTimes(archive)
    second = readCaptureTimes(archive, t_start=unixToDatetime(times[100] - 0.001),
                              t_end=unixToDatetime(times[200] + 0.001))

    assert np.array_equal(first, times)
    assert np.array_equal(second, times[100:201])
    assert opened == [archive]


### SkyFit2 ###

PICK_FRAMES = [1, 2, 5]
PICK_Y = 47.0


def makePlateTool(skyfit2, handle, config, deinterlace_order):
    """ A stand-in of the SkyFit2 PlateTool with what saveECSV and saveFTPdetectinfo read, with picks on three
        frames at the bottom of the image (the largest rolling shutter delay).
    """

    from RMS.Formats.Platepar import Platepar

    # A platepar of the size of the video
    pp = Platepar()
    pp.lat, pp.lon, pp.elev = 45.0, 15.0, 100.0
    pp.X_res, pp.Y_res = 64, 48
    pp.F_scale = 1.0
    pp.az_centre, pp.alt_centre = 180.0, 60.0
    pp.JD = 2461000.5
    pp.station_code = "XX0001"
    pp.updateRefRADec(skip_rot_update=True)

    config.deinterlace_order = deinterlace_order
    config.stationID = "XX0001"

    pick_list = {}
    for frame in PICK_FRAMES:
        pick_list[frame] = {'mode': 1, 'x_centroid': 32.0, 'y_centroid': PICK_Y, 'intensity_sum': 100,
                            'background_intensity': 10, 'saturated': False, 'snr': 10.0}

    pt = types.SimpleNamespace(img_handle=handle, config=config, platepar=pp, pick_list=pick_list,
                               meas_ground_points=False, mag_band_string="V", exposure_ratio=1.0,
                               dir_path=str(handle.dir_path), img=types.SimpleNamespace(data=np.zeros((64, 48))),
                               computeIntensitySum=lambda: None)

    # The methods of PlateTool which the savers call
    for name in ['computeExposureRatioCorrection', 'getRollingShutterCorrectedFrameNo', 'perFrameTimes',
                 'pickTime']:
        if hasattr(skyfit2.PlateTool, name):
            setattr(pt, name, types.MethodType(getattr(skyfit2.PlateTool, name), pt))

    return pt


def expectedPickTimes(config, deinterlace_order):
    """ The capture times of the frames of the picks, plus the rolling shutter delay of their row within one frame
        period of the camera. """

    delay = 0.0
    if deinterlace_order == -1:
        config.deinterlace_order = -1
        delay = RollingShutterCorrection.correctFrameTime(0.0, PICK_Y, 48, config, fps=config.fps,
                                                          cropped=False)/config.fps

    return [captureTimes()[fr] + datetime.timedelta(seconds=delay) for fr in PICK_FRAMES]


@pytest.mark.parametrize("deinterlace_order", [-2, -1])
def test_skyfit2_ecsv_has_capture_times(tmp_path, deinterlace_order):

    # The picks of a timelapse saved by SkyFit2 into an ECSV file, with a global and a rolling shutter
    skyfit2 = pytest.importorskip("Utils.SkyFit2")
    handle = InputTypeVideo(writeTimelapse(tmp_path), makeConfig())
    config = makeConfig()
    pt = makePlateTool(skyfit2, handle, config, deinterlace_order)

    assert skyfit2.PlateTool.saveECSV(pt)

    ecsv_files = [f for f in os.listdir(str(tmp_path)) if f.endswith(".ecsv")]
    with open(os.path.join(str(tmp_path), ecsv_files[0])) as f:
        rows = [line for line in f if not line.startswith("#") and not line.startswith("datetime")]

    times = [datetime.datetime.strptime(row.split(",")[0], "%Y-%m-%dT%H:%M:%S.%f") for row in rows]
    for t, expected in zip(times, expectedPickTimes(config, deinterlace_order)):
        assert abs((t - expected).total_seconds()) < 2e-6


@pytest.mark.parametrize("deinterlace_order", [-2, -1])
def test_skyfit2_ftpdetectinfo_has_capture_times(tmp_path, deinterlace_order):

    # The picks of a timelapse saved by SkyFit2 into an FTPdetectinfo file: the times read from the file (the time
    #   in the FF name plus the frame over the frame rate in the file) are the times of the picks
    skyfit2 = pytest.importorskip("Utils.SkyFit2")
    from RMS.Formats.FFfile import filenameToDatetime

    handle = InputTypeVideo(writeTimelapse(tmp_path), makeConfig())
    config = makeConfig()
    pt = makePlateTool(skyfit2, handle, config, deinterlace_order)

    skyfit2.PlateTool.saveFTPdetectinfo(pt)

    ftp_files = [f for f in os.listdir(str(tmp_path)) if f.startswith("FTPdetectinfo_")]
    with open(os.path.join(str(tmp_path), ftp_files[0])) as f:
        lines = f.read().splitlines()

    # The FF name, the frame rate in the header of the meteor, and the frames of the picks
    ff_index = [i for i, line in enumerate(lines) if line.startswith("FF_")][0]
    fps = float(lines[ff_index + 2].split()[3])
    frames = [float(line.split()[0]) for line in lines[ff_index + 3:ff_index + 3 + len(PICK_FRAMES)]]
    t_ref = filenameToDatetime(lines[ff_index])

    times = [t_ref + datetime.timedelta(seconds=fr/fps) for fr in frames]
    for t, expected in zip(times, expectedPickTimes(config, deinterlace_order)):
        assert abs((t - expected).total_seconds()) < 0.003


def test_skyfit2_state_keeps_the_frame_times_file(tmp_path):

    # The path to the file with the times of the frames is saved in the state, so a reloaded state times the frames
    #   in the same way
    skyfit2 = pytest.importorskip("Utils.SkyFit2")
    from RMS.Pickling import loadPickle

    handle = InputTypeVideo(writeTimelapse(tmp_path), makeConfig())
    pt = types.SimpleNamespace(img_handle=handle, dir_path=str(tmp_path), pick_list={},
                               frame_times_path=str(tmp_path/"times.json"))

    skyfit2.PlateTool.saveState(pt)

    state = loadPickle(str(tmp_path), 'skyFitMR_latest.state')
    assert state['frame_times_path'] == str(tmp_path/"times.json")
    assert 'img_handle' not in state


def test_skyfit2_rolling_shutter_delay_with_a_given_fps(tmp_path):

    # A timelapse loaded with the FPS given as before (-f 0.2 replaces the frame rate of the camera in the config):
    #   the rolling shutter delay of a pick is still within one frame period of the camera, not 5 s
    skyfit2 = pytest.importorskip("Utils.SkyFit2")
    handle = InputTypeVideo(writeTimelapse(tmp_path), makeConfig())
    config = makeConfig()
    expected = expectedPickTimes(config, -1)

    config.fps = 0.2
    pt = makePlateTool(skyfit2, handle, config, -1)
    pt.camera_fps = CAPTURE_FPS

    for frame, t in zip(PICK_FRAMES, expected):
        assert abs((skyfit2.PlateTool.pickTime(pt, frame, pt.pick_list[frame]) - t).total_seconds()) < 1e-6


def skyFitStandIn(skyfit2, input_path, config):
    """ A stand-in of the SkyFit2 PlateTool with what detectInputType reads. """

    pt = types.SimpleNamespace(input_path=input_path, dir_path=input_path if os.path.isdir(input_path)
                               else os.path.dirname(input_path), config=config, flipud=False, use_fr_files=False,
                               frame_times_path=None)
    pt.findCaptureTimes = types.MethodType(skyfit2.PlateTool.findCaptureTimes, pt)

    return pt


def test_skyfit2_finds_the_ft_archive_of_ff_files(tmp_path):

    # A night folder of the station with FF files and the daily archive of the FT files (no FT file on its own)
    skyfit2 = pytest.importorskip("Utils.SkyFit2")
    times, names = ffCapture()
    night_dir = tmp_path/"RMS_data"/"ArchivedFiles"/"XX0001_20261122_224724_669632"
    night_dir.mkdir(parents=True)
    for name in names:
        writeFF(night_dir, name)

    day = T0.strftime("%Y%m%d-%j")
    ft_dir = tmp_path/day
    ft_dir.mkdir()
    writeFT(ft_dir, times)
    with tarfile.open(str(night_dir/("XX0001_" + day + "_FT.tar.bz2")), 'w:bz2') as tar:
        tar.add(str(ft_dir), arcname=day)

    pt = skyFitStandIn(skyfit2, str(night_dir), makeConfig())
    skyfit2.PlateTool.detectInputType(pt)

    # The frames of the FF files are the captured frames
    handle = pt.img_handle
    handle.setCurrentFF(names[1])
    assert frameTimes(handle, [0, 100, 255]) == [unixToDatetime(times[256 + k]) for k in [0, 100, 255]]


def test_skyfit2_finds_the_ft_files_of_a_capture_video(tmp_path):

    # A video of the capture in the station folders, with its FT files in the FT folder of the same hour
    skyfit2 = pytest.importorskip("Utils.SkyFit2")
    times = syntheticCapture(T0, 3.0, gaps=[(0.5, 0.7)])
    hour = T0.strftime("%Y/%Y%m%d-%j/%Y%m%d-%j_%H")
    video_dir = tmp_path/"VideoFiles"/hour
    ft_dir = tmp_path/"TimeFiles"/hour
    video_dir.mkdir(parents=True)
    ft_dir.mkdir(parents=True)
    writeFT(ft_dir, times)

    path = str(video_dir/("XX0001_" + unixToDatetime(times[0]).strftime("%Y%m%d_%H%M%S_%f") + "_video.avi"))
    writeVideo(path, n_frames=40, fps=CAPTURE_FPS)

    pt = skyFitStandIn(skyfit2, path, makeConfig())
    skyfit2.PlateTool.detectInputType(pt)

    assert frameTimes(pt.img_handle, range(40)) == [unixToDatetime(t) for t in times[:40]]


def test_skyfit2_warns_about_a_timelapse_without_frame_times(tmp_path, monkeypatch):

    # A timelapse without its frametimes file: SkyFit2 shows the warning in a message box (mocked here)
    skyfit2 = pytest.importorskip("Utils.SkyFit2")
    boxes = []
    monkeypatch.setattr(skyfit2, "messagebox", lambda title, message: boxes.append((title, message)))

    pt = skyFitStandIn(skyfit2, writeTimelapse(tmp_path, json_name=None), makeConfig())
    skyfit2.PlateTool.detectInputType(pt)

    assert len(boxes) == 1
    assert "--frametimes" in boxes[0][1]

    # With the frametimes file, no warning
    boxes[:] = []
    pt = skyFitStandIn(skyfit2, writeTimelapse(tmp_path), makeConfig())
    skyfit2.PlateTool.detectInputType(pt)
    assert boxes == []
