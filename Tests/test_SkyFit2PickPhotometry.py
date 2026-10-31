"""
Tests of the photometry of the manual reduction picks in SkyFit2: the intensity sum of the pixels coloured on each
frame, the saving of the picks to the ECSV file, and the photometry pixels file (--photpixels, or Save Photometry
Pixels in the GUI) saved next to it. They are run on the code of PlateTool, without the GUI.
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
    photometryPixelLines = PlateTool.photometryPixelLines
    savePhotometryPixels = PlateTool.savePhotometryPixels

    def __init__(self, dir_path, avepixel=None, save_photometry_pixels=True, nobg=False, peribg=False):

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
        self.save_photometry_pixels = save_photometry_pixels
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
    assert 'photometry_pixel_values' in pick


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


@pytest.mark.parametrize("background_kwargs, background", [
    (dict(nobg=True), 0.0),
    (dict(peribg=True), BACKGROUND),
    (dict(), BACKGROUND),
    ])
def test_pixel_values_add_up_to_the_intensity_sum(tmp_path, background_kwargs, background):
    """ With every background subtraction (none, peripheral, avepixel) the stored pixels are the coloured ones,
        with the frame value as raw intensity, the background that was subtracted, and background-subtracted
        intensities whose sum is the intensity sum of the pick. """

    session = PickingSession(str(tmp_path), **background_kwargs)
    pick = session.pick(session.frame, 902.0, 502.0, PIXELS)

    session.computeIntensitySum()
    values = pick['photometry_pixel_values']

    assert sorted(zip(values['x'], values['y'])) == sorted(PIXELS)
    assert np.allclose(values['intensity_raw'], BACKGROUND + SIGNAL)
    assert np.allclose(values['background'], background)
    assert np.allclose(values['intensity_nobg'], BACKGROUND + SIGNAL - background)
    assert int(abs(np.nansum(values['intensity_nobg']))) == pick['intensity_sum']


def test_star_masked_pixels_are_left_out_of_the_intensity_sum(tmp_path):
    """ A coloured pixel on a star of the avepixel does not count in the intensity sum: its background and its
        background-subtracted intensity are NaN, and the other 24 pixels add up to the sum. """

    avepixel = np.full((1920, 1080), BACKGROUND)
    avepixel[902, 502] = 200.0

    session = PickingSession(str(tmp_path), avepixel=avepixel)
    pick = session.pick(session.frame, 902.0, 502.0, PIXELS)

    session.computeIntensitySum()
    values = pick['photometry_pixel_values']
    on_star = (values['x'] == 902) & (values['y'] == 502)

    assert np.isnan(values['intensity_nobg'][on_star]).all()
    assert np.isnan(values['background'][on_star]).all()
    assert not np.isnan(values['background'][~on_star]).any()
    assert pick['intensity_sum'] == 24*SIGNAL
    assert np.nansum(values['intensity_nobg']) == pick['intensity_sum']


def test_values_are_only_kept_from_the_last_computation_with_the_option_on(tmp_path):
    """ The values are not stored with the option off, and a new computation drops the old ones: with the option
        off, or with the coloured pixels removed, nothing of a previous photometry is left to be saved. """

    session = PickingSession(str(tmp_path), save_photometry_pixels=False)
    pick = session.pick(session.frame, 902.0, 502.0, PIXELS)

    session.computeIntensitySum()
    assert 'photometry_pixel_values' not in pick

    session.save_photometry_pixels = True
    session.computeIntensitySum()
    assert 'photometry_pixel_values' in pick

    session.save_photometry_pixels = False
    session.computeIntensitySum()
    assert 'photometry_pixel_values' not in pick

    session.save_photometry_pixels = True
    session.computeIntensitySum()
    pick['photometry_pixels'] = []
    session.computeIntensitySum()
    assert 'photometry_pixel_values' not in pick


def test_ecsv_has_the_photometry_of_the_current_frame(tmp_path):
    """ Pixels coloured on the current frame and saved right away are in the ECSV file: the photometry of the
        current frame is computed on saving, not only when leaving the frame. """

    session = PickingSession(str(tmp_path), save_photometry_pixels=False)
    session.pick(session.frame, 902.0, 502.0, PIXELS)

    assert session.saveECSV()

    ecsv_files = [name for name in os.listdir(str(tmp_path)) if name.endswith('.ecsv')]
    _, rows = readTable(os.path.join(str(tmp_path), ecsv_files[0]))

    assert int(rows[0]['integrated_pixel_value']) == len(PIXELS)*SIGNAL

    # Without the option no photometry pixels file
    assert os.listdir(str(tmp_path)) == ecsv_files


def test_saved_pixels_match_the_ecsv_picks(tmp_path, capsys):
    """ The pixels file is next to the ECSV file, with the same name and a _photometry_pixels.csv ending. Its rows
        join the picks by the datetime; a pixel at the position of a pick has the sky coordinates of the pick (to
        the last written digit), and the pixels of a pick add up to its integrated pixel value. The pick whose
        photometry was computed with the option off is not in the file and is reported. """

    session = PickingSession(str(tmp_path))

    # A pick measured with the option off, then the pick of the current frame
    session.save_photometry_pixels = False
    session.frame = 9
    session.pick(9, 901.0, 501.0, PIXELS)
    session.computeIntensitySum()

    session.save_photometry_pixels = True
    session.frame = 10
    session.pick(10, 902.0, 502.0, PIXELS)

    assert session.saveECSV()

    ecsv_name = [name for name in os.listdir(str(tmp_path)) if name.endswith('.ecsv')][0]
    pixels_name = ecsv_name[:-len('.ecsv')] + "_photometry_pixels.csv"
    assert sorted(os.listdir(str(tmp_path))) == sorted([ecsv_name, pixels_name])

    _, picks = readTable(os.path.join(str(tmp_path), ecsv_name))
    names, pixels = readTable(os.path.join(str(tmp_path), pixels_name))

    assert names == ['datetime', 'frame', 'fragment', 'x_image', 'y_image', 'ra', 'dec', 'azimuth', 'altitude',
                     'intensity_raw', 'background', 'intensity_nobg']
    assert len(pixels) == len(PIXELS)
    assert {row['frame'] for row in pixels} == {'10'}
    assert {row['fragment'] for row in pixels} == {'0'}
    assert {row['datetime'] for row in pixels} == {picks[1]['datetime']}

    at_pick = [row for row in pixels if (row['x_image'], row['y_image']) == ('902', '502')][0]
    for coord in ['ra', 'dec', 'azimuth', 'altitude']:
        assert abs(float(at_pick[coord]) - float(picks[1][coord])) <= 1.5e-6

    assert int(abs(sum(float(row['intensity_nobg']) for row in pixels))) \
        == int(picks[1]['integrated_pixel_value'])

    assert "frames 9 were not saved" in capsys.readouterr().out


def test_pixels_file_is_read_by_astropy(tmp_path):
    """ The header of the pixels file is a valid ECSV header: astropy reads the columns with their types and units,
        and the NaN of the pixels left out of the intensity sum. """

    ascii = pytest.importorskip("astropy.io.ascii")

    avepixel = np.full((1920, 1080), BACKGROUND)
    avepixel[902, 502] = 200.0

    session = PickingSession(str(tmp_path), avepixel=avepixel)
    session.pick(session.frame, 902.0, 502.0, PIXELS)
    session.saveECSV()

    pixels_name = [name for name in os.listdir(str(tmp_path)) if name.endswith('_photometry_pixels.csv')][0]
    table = ascii.read(os.path.join(str(tmp_path), pixels_name), format='ecsv')

    assert len(table) == len(PIXELS)
    assert table['x_image'].dtype.kind == 'i'
    assert table['fragment'].dtype.kind == 'i'
    assert str(table['ra'].unit) == 'deg'
    assert np.isnan(table['intensity_nobg']).sum() == 1
    assert table.meta['ecsv_file'].endswith('.ecsv')
