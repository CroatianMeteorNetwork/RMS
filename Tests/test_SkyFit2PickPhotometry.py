"""
Tests of the photometry of the manual reduction picks in SkyFit2: the intensity sum of the pixels coloured on each
frame, and the saving of the picks to the ECSV file. They are run on the code of PlateTool, without the GUI.
"""

import datetime
import os
import types

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

import RMS.ConfigReader as cr
from Utils.SkyFit2 import PlateTool
from Tests.test_PlateparDecentering import makePlatepar


# A 5x5 block of photometry pixels, 50 above a flat background of 20
BACKGROUND = 20.0
SIGNAL = 50.0
PIXELS = [(x, y) for x in range(900, 905) for y in range(500, 505)]


class FrameTimes(object):
    """ The input of the frames: FF file times at 25 FPS. """

    def __init__(self, avepixel):

        self.input_type = 'ff'
        self.use_fr_files = False
        self.fps = 25.0
        self.beginning_datetime = datetime.datetime(2026, 10, 20, 3, 15, 42, 120000)
        self.current_ff_file = "FF_XX0001_20261020_031542_120_0000000.fits"
        self.ff = types.SimpleNamespace(avepixel=avepixel.T)

    def currentFrameTime(self, frame_no=None, dt_obj=False):

        dt = self.beginning_datetime + datetime.timedelta(seconds=frame_no/self.fps)

        if dt_obj:
            return dt

        return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)


class PickingSession(object):
    """ The state of SkyFit2 used by the photometry and the saving of the picks. """

    computeIntensitySum = PlateTool.computeIntensitySum
    computeExposureRatioCorrection = PlateTool.computeExposureRatioCorrection
    getCurrentPick = PlateTool.getCurrentPick
    saveECSV = PlateTool.saveECSV

    def __init__(self, dir_path, avepixel=None, nobg=False, peribg=False):

        self.platepar = makePlatepar()
        self.platepar.station_code = "XX0001"

        if avepixel is None:
            avepixel = np.full((self.platepar.X_res, self.platepar.Y_res), BACKGROUND)

        data = avepixel.copy()
        for x, y in PIXELS:
            data[x, y] = BACKGROUND + SIGNAL

        self.mode = 'manualreduction'
        self.frame = 10
        self.img = types.SimpleNamespace(data=data, img_showing='frame', getFrame=lambda: self.frame)
        self.img_handle = FrameTimes(avepixel)
        self.config = cr.Config()
        self.config.stationID = "XX0001"
        self.dir_path = dir_path
        self.mag_band_string = "V"
        self.meas_ground_points = False
        self.exposure_ratio = 1.0
        self.dark = None
        self.flat_struct = None
        self.no_background_subtraction = nobg
        self.peripheral_background_subtraction = peribg
        self.saturation_threshold = int(round(0.98*(2**self.config.bit_depth - 1)))
        self.pick_list = {}

    def pick(self, frame, x, y, pixels):
        """ A pick on the given frame, with its photometry not computed yet. """

        self.pick_list[frame] = {'x_centroid': x, 'y_centroid': y, 'mode': 1, 'intensity_sum': 1,
                                 'photometry_pixels': list(pixels), 'background_intensity': 0, 'snr': 1.0,
                                 'saturated': False}

        return self.pick_list[frame]


def readTable(path):
    """ The column names and the rows of a file written by SkyFit2 (ECSV header in comments). """

    with open(path) as f:
        lines = [line.strip() for line in f if not line.startswith('#')]

    names = lines[0].split(',')

    return names, [dict(zip(names, [v.strip() for v in line.split(',')])) for line in lines[1:]]


@pytest.mark.parametrize("mode, img_showing", [('skyfit', 'avepixel'), ('manualreduction', 'maxpixel')])
def test_photometry_is_only_measured_on_the_frame(tmp_path, mode, img_showing):
    """ The SkyFit mode (avepixel) and the maxpixel view keep the picks and the current frame: computing the
        photometry while they are shown (e.g. on saving, or leaving the frame) leaves the intensity sum measured on
        the frame as it is. """

    session = PickingSession(str(tmp_path))
    pick = session.pick(session.frame, 902.0, 502.0, PIXELS)
    session.computeIntensitySum()

    session.mode = mode
    session.img.img_showing = img_showing
    session.img.data = session.img_handle.ff.avepixel.T
    session.computeIntensitySum()

    assert pick['intensity_sum'] == len(PIXELS)*SIGNAL


def test_dfn_image_is_measured_in_manual_reduction(tmp_path):
    """ A DFN image has no frames: it is shown as the avepixel in manual reduction, and is measured there (with
        the DFN exposure correction of 431). """

    session = PickingSession(str(tmp_path))
    session.img_handle.input_type = 'dfn'
    session.img.img_showing = 'avepixel'
    pick = session.pick(session.frame, 902.0, 502.0, PIXELS)

    session.computeIntensitySum()

    assert pick['intensity_sum'] == 431*len(PIXELS)*SIGNAL


def test_ecsv_is_saved_with_a_pick_without_signal(tmp_path):
    """ Pixels coloured where the frame is the avepixel (an FF frame away from the meteor) have no signal and an
        SNR of 0: the pick is saved like one with an unknown SNR (SNR 0, no random magnitude error) instead of the
        saving failing on the error 2.5 log10(1 + 1/SNR). """

    session = PickingSession(str(tmp_path))
    session.img.data = session.img_handle.ff.avepixel.T.copy()
    pick = session.pick(session.frame, 902.0, 502.0, PIXELS)

    session.computeIntensitySum()
    assert pick['snr'] == 0

    assert session.saveECSV()

    ecsv_files = [name for name in os.listdir(str(tmp_path)) if name.endswith('.ecsv')]
    _, rows = readTable(os.path.join(str(tmp_path), ecsv_files[0]))

    assert float(rows[0]['snr']) == 0
    assert float(rows[0]['err_plus_mag']) == session.platepar.mag_lev_stddev
