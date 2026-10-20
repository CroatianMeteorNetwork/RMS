"""
Tests of the rolling-shutter correction of the times of the points (RMS.Routines.RollingShutterCorrection): the
rows of a rolling shutter are recorded one after the other within the readout time of the sensor, by default the
whole frame period, or the readout time given in the config (rolling_shutter_readout).
"""

import os

import numpy as np

import RMS.ConfigReader as cr
from RMS.Routines import RollingShutterCorrection as rsc


def test_no_correction_without_a_readout_time():
    """ Without a measured readout time there is no correction: the readout time in frames is 0, and the time of a
        point is the frame time on every row. """

    config = cr.Config()
    assert config.rolling_shutter_readout == 0
    assert rsc.readoutFrames(config, 25.0) == 0
    assert rsc.correctRollingShutterTemporal(10, 360, 720, readout_frames=rsc.readoutFrames(config, 25.0)) == 10

    # The rows read over the whole frame (readout time 1/fps): the middle row half a frame later
    assert rsc.correctRollingShutterTemporal(10, 360, 720) == 10.5


def test_readout_time_in_frames():
    """ A sensor read in 32 ms recording at 10 fps reads its rows in 0.32 of a frame: a point at the middle row
        is 0.16 frames later than the frame time, and the frame rate of the input is used when given. """

    config = cr.Config()
    config.rolling_shutter_readout = 0.032
    config.fps = 25.0

    assert np.isclose(rsc.readoutFrames(config, 10.0), 0.32)
    assert np.isclose(rsc.readoutFrames(config), 0.8)
    assert np.isclose(rsc.correctRollingShutterTemporal(10, 540, 1080, readout_frames=0.32), 10.16)

    # The list version, frame by frame
    corrected = rsc.correctRollingShutterTemporalList([1, 2], [0, 1080], 1080, readout_frames=0.32)
    assert np.allclose(corrected, [1, 2.32])


def test_readout_time_parsed(tmp_path):
    """ The readout time is read from the [Capture] section of the config. """

    path = os.path.join(str(tmp_path), '.config')
    with open(path, 'w') as f:
        f.write("[System]\nstationID: XX0001\n\n[Capture]\ndeinterlace_order: -1\nrolling_shutter_readout: 0.032\n")

    config = cr.parse(path)

    assert config.deinterlace_order == -1
    assert config.rolling_shutter_readout == 0.032


def test_cropped_capture():
    """ An image cropped in the capture still comes from all the rows of the sensor: a 1080-row sensor read in
        32 ms at 10 fps, its video cropped by 40 rows at the top and 80 at the bottom, gives 960-row frames whose
        first row is 40 rows of the sensor (40/1080 of the readout) after the frame time, and whose last row is
        80 rows before the end of the readout. A region of interest removes rows the same way. A raw video saved
        before the crop (taller than the processed frames) and an image without removed rows are the whole
        sensor. """

    config = cr.Config()
    config.rolling_shutter_readout = 0.032
    config.fps = 10.0
    config.height = config.height_device = 1080

    # No crop: the image is the whole sensor
    assert rsc.sensorRows(config, 1080) == (0, 1080)
    assert np.isclose(rsc.correctFrameTime(10, 540, 1080, config), 10.16)

    # Cropped video: the rows of the image are offset on the sensor, the readout covers all the sensor rows
    config.video_crop = "top=40 bottom=80 left=15 right=15"
    config.height = config.height_device = 960
    assert rsc.sensorRows(config, 960) == (40, 1080)
    assert np.isclose(rsc.correctFrameTime(10, 0, 960, config), 10 + 0.32*40/1080)
    assert np.isclose(rsc.correctFrameTime(10, 960, 960, config), 10 + 0.32*1000/1080)

    # Arrays of points, and the frame rate of the input
    corrected = rsc.correctFrameTime(np.array([1.0, 2.0]), np.array([0.0, 520.0]), 960, config, fps=20.0)
    assert np.allclose(corrected, [1 + 0.64*40/1080, 2 + 0.64*560/1080])

    # A raw video saved before the crop has all the sensor rows, also when its height is that of the processed
    #   frames (a capture which scales the video up before the crop)
    assert rsc.sensorRows(config, 1080) == (0, 1080)
    assert rsc.sensorRows(config, 960, cropped=False) == (0, 960)
    assert np.isclose(rsc.correctFrameTime(10, 480, 960, config, cropped=False), 10.16)

    # Only the inputs of the RMS capture are cropped: not the raw videos, image folders or the files of other
    #   capture systems
    assert [rsc.croppedInput(t) for t in ('ff', 'fblk', 'images', 'video', 'vid', 'dfn')] == \
        [True, True, False, False, False, False]

    # The input overwrites the height in the config with its own (binned) size: the crop still applies
    config.height = 480
    assert rsc.sensorRows(config, 960) == (40, 1080)

    # A region of interest (rows 100 to 900 of the cropped frames) on top of the crop
    config.roi_up, config.roi_down = 100, 900
    assert rsc.sensorRows(config, 800) == (140, 1080)

    # The capture crops only with the GStreamer backend and a well-formed crop
    config.roi_up, config.roi_down = -1, -1
    config.media_backend = 'v4l2'
    assert rsc.sensorRows(config, 960) == (0, 960)
    config.media_backend = 'gst'
    config.video_crop = "top=40 bottom=80 left=15 right=x"
    assert rsc.sensorRows(config, 960) == (0, 960)

    # Only the top and bottom of the crop matter
    config = cr.Config()
    config.height = config.height_device = 1080
    config.video_crop = "left=100 right=100"
    assert rsc.sensorRows(config, 1080) == (0, 1080)
