""" Tests for the SkyFit2 manual reduction annotations (fragment ID, flare, trajectory use) and their
    ECSV columns.

    The SkyFit2 functions are run on a minimal stand-in for the GUI, so no data files are needed.
"""

import datetime
import os

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from RMS.Formats.Platepar import Platepar
from RMS.Routines.CustomPyqtgraphClasses import AnnotationsWidget, FRAGMENT_COLORS
from Utils.SkyFit2 import GREAT_CIRCLE_COLOR, PlateTool


LEGACY_COLUMNS = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image',
                  'integrated_pixel_value', 'background_pixel_value', 'saturated_pixels', 'mag_data',
                  'err_minus_mag', 'err_plus_mag', 'snr']


class ImageHandle(object):
    """ Stand-in for an FF image handle. """

    input_type = 'ff'
    current_ff_file = 'FF_XX0001_20240101_000000_000_0000000.fits'
    fps = 25.0
    total_frames = 256
    beginning_datetime = datetime.datetime(2024, 1, 1, 0, 0, 0)

    def currentFrameTime(self, frame_no=None, dt_obj=False):

        dt = self.beginning_datetime + datetime.timedelta(seconds=frame_no/self.fps)

        if dt_obj:
            return dt

        return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)


class Reduction(object):
    """ Stand-in for PlateTool with only what the pick and ECSV functions use. """

    saveECSV = PlateTool.saveECSV
    loadECSV = PlateTool.loadECSV
    addFragmentPoint = PlateTool.addFragmentPoint
    removeFragmentPoint = PlateTool.removeFragmentPoint

    def __init__(self, dir_path):

        self.dir_path = str(dir_path)
        self.img_handle = ImageHandle()
        self.config = type('Config', (), {'stationID': 'XX0001', 'deinterlace_order': -2})()
        self.mag_band_string = 'V'
        self.meas_ground_points = False

        self.platepar = Platepar()
        self.platepar.station_code = 'XX0001'
        self.platepar.lat, self.platepar.lon, self.platepar.elev = 45.0, 16.0, 100.0
        self.platepar.RA_d, self.platepar.dec_d = 120.0, 40.0

        self.pick_list = {}
        self.fragment_picks = {}

    def computeExposureRatioCorrection(self):
        return 0.0

    def computeIntensitySum(self):
        """ The picks have no coloured photometry pixels, so their intensity stays 1. """
        pass

    def addPick(self, frame, x, y, **annotations):
        """ Add a main fragment pick in the same format as PlateTool.addCentroid. """

        self.pick_list[frame] = {'x_centroid': x, 'y_centroid': y, 'mode': 1, 'intensity_sum': 1,
                                 'photometry_pixels': None, 'background_intensity': 0, 'snr': 1,
                                 'saturated': False}
        self.pick_list[frame].update(annotations)


def saveAndRead(reduction):
    """ Save the ECSV and return its meta lines, column names and rows (as lists of strings). """

    assert reduction.saveECSV()

    file_name = [name for name in os.listdir(reduction.dir_path) if name.endswith('.ecsv')][0]
    file_path = os.path.join(reduction.dir_path, file_name)

    with open(file_path) as f:
        lines = f.read().splitlines()

    meta = [line for line in lines if line.startswith('#')]
    data = [[part.strip() for part in line.split(',')] for line in lines if not line.startswith('#')]

    return file_path, meta, data[0], data[1:]


def column(columns, rows, name):
    """ Return the values of the given column in every row. """

    return [row[columns.index(name)] for row in rows]


def test_unannotated_reduction(tmp_path):
    """ Without annotations, the ECSV has exactly the previous columns. """

    reduction = Reduction(tmp_path)
    for frame in (10, 11, 12):
        reduction.addPick(frame, 100.0 + frame, 200.0)

    _, meta, columns, rows = saveAndRead(reduction)

    assert columns == LEGACY_COLUMNS
    assert "# - {no_frags: 1}" in meta
    assert len(rows) == 3


def test_annotated_reduction(tmp_path):
    """ Flares and main picks excluded from the trajectory add the annotation columns. """

    reduction = Reduction(tmp_path)
    reduction.addPick(10, 110.0, 200.0)
    reduction.addPick(11, 111.0, 200.0, flare=True)
    reduction.addPick(12, 112.0, 200.0, flare=True, trajectory_use=False)

    _, meta, columns, rows = saveAndRead(reduction)

    assert columns == LEGACY_COLUMNS + ['frame_number', 'flare', 'trajectory_use']
    assert "# - {no_frags: 1}" in meta
    assert column(columns, rows, 'frame_number') == ['10', '11', '12']
    assert column(columns, rows, 'flare') == ['False', 'True', 'True']
    assert column(columns, rows, 'trajectory_use') == ['True', 'True', 'False']


def test_fragment_columns(tmp_path):
    """ Additional fragments are written in their own columns, one row per frame (GDEF standard), with
        empty values on the frames where a fragment or the main fragment was not picked.
    """

    reduction = Reduction(tmp_path)
    reduction.addPick(10, 110.0, 200.0)
    reduction.addPick(11, 111.0, 200.0)
    reduction.addPick(12, 112.0, 200.0)
    reduction.addFragmentPoint(11, 1, 111.0, 210.0)
    reduction.addFragmentPoint(12, 1, 112.0, 210.0)
    reduction.addFragmentPoint(12, 2, 112.0, 220.0)

    # Fragment 1 is still visible after the main fragment
    reduction.addFragmentPoint(13, 1, 113.0, 210.0)

    _, meta, columns, rows = saveAndRead(reduction)

    fragment_columns = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image']
    assert columns == LEGACY_COLUMNS + ['frame_number', 'flare', 'trajectory_use'] \
        + [name + '1' for name in fragment_columns] + [name + '2' for name in fragment_columns]
    assert "# - {no_frags: 3}" in meta

    assert column(columns, rows, 'frame_number') == ['10', '11', '12', '13']
    assert column(columns, rows, 'x_image') == ['110.000', '111.000', '112.000', '']
    assert column(columns, rows, 'x_image1') == ['', '111.000', '112.000', '113.000']
    assert column(columns, rows, 'y_image2') == ['', '', '220.000', '']

    # The main fragment columns are empty on the frame where only fragment 1 was picked
    assert set(rows[3][:len(LEGACY_COLUMNS)] + [rows[3][columns.index('flare')]]) == {''}

    # With a global shutter, a fragment point has the time of its frame
    assert column(columns, rows, 'datetime1')[1] == column(columns, rows, 'datetime')[1]


def test_fragment_ids_0_to_8(tmp_path):
    """ The main fragment (0) and the 8 others can be stored on a single frame, and each one has its own 
        color. As in the GFE standard, fragment k is written to the columns with the suffix k. """

    reduction = Reduction(tmp_path)
    reduction.addPick(20, 120.0, 200.0)
    for fragment_id in range(1, 9):
        reduction.addFragmentPoint(20, fragment_id, 120.0 + fragment_id, 200.0)

    _, meta, columns, rows = saveAndRead(reduction)

    assert "# - {no_frags: 9}" in meta
    assert [float(column(columns, rows, 'x_image' + str(k))[0]) for k in range(1, 9)] \
        == [120.0 + fragment_id for fragment_id in range(1, 9)]

    assert len(FRAGMENT_COLORS) == 9
    assert FRAGMENT_COLORS[0] == (255, 0, 0)
    assert len(set(FRAGMENT_COLORS)) == 9


def test_no_duplicate_fragment_point(tmp_path):
    """ Adding the same fragment on the same frame again moves the point. """

    reduction = Reduction(tmp_path)
    reduction.addPick(10, 110.0, 200.0)

    reduction.addFragmentPoint(10, 1, 110.0, 210.0)
    reduction.addFragmentPoint(10, 1, 115.0, 215.0)

    assert list(reduction.fragment_picks[10].keys()) == [1]
    assert reduction.fragment_picks[10][1]['x_centroid'] == 115.0

    # The main pick on that frame is not changed
    assert reduction.pick_list[10]['x_centroid'] == 110.0

    reduction.removeFragmentPoint(10, 1)
    assert reduction.fragment_picks == {}


def test_round_trip(tmp_path):
    """ Saving and loading the ECSV keeps the frames, positions and annotations. """

    reduction = Reduction(tmp_path)
    reduction.addPick(10, 110.0, 200.0)
    reduction.addPick(11, 111.0, 200.0, flare=True, trajectory_use=False)
    reduction.addPick(12, 112.0, 200.0)
    reduction.addFragmentPoint(11, 1, 111.0, 210.0)
    reduction.addFragmentPoint(11, 3, 111.0, 230.0)
    reduction.addFragmentPoint(13, 1, 113.0, 210.0)
    file_path, _, _, _ = saveAndRead(reduction)

    loaded = Reduction(tmp_path)
    pick_list = loaded.loadECSV(file_path)

    assert sorted(pick_list.keys()) == [10, 11, 12]
    assert pick_list[11]['x_centroid'] == pytest.approx(111.0)
    assert [(pick_list[frame]['flare'], pick_list[frame]['trajectory_use']) for frame in (10, 11, 12)] \
        == [(False, True), (True, False), (False, True)]

    assert {frame: sorted(fragments) for frame, fragments in loaded.fragment_picks.items()} \
        == {11: [1, 3], 13: [1]}
    assert loaded.fragment_picks[11][3]['y_centroid'] == pytest.approx(230.0)
    assert loaded.fragment_picks[13][1]['x_centroid'] == pytest.approx(113.0)


def test_load_legacy_ecsv(tmp_path):
    """ An ECSV without the annotation columns loads as before, with the default annotations. """

    reduction = Reduction(tmp_path)
    for frame in (10, 11, 12):
        reduction.addPick(frame, 100.0 + frame, 200.0)
    file_path, _, columns, _ = saveAndRead(reduction)
    assert columns == LEGACY_COLUMNS

    loaded = Reduction(tmp_path)
    loaded.fragment_picks = {11: {1: {'x_centroid': 1.0, 'y_centroid': 1.0}}}
    pick_list = loaded.loadECSV(file_path)

    assert sorted(pick_list.keys()) == [10, 11, 12]
    assert all(not pick['flare'] and pick['trajectory_use'] for pick in pick_list.values())

    # Points of other fragments are only replaced by files which have annotations
    assert list(loaded.fragment_picks.keys()) == [11]


def shiftEcsvTimes(file_path, milliseconds):
    """ Move the times of the rows of an ECSV file, as the rolling shutter correction does. """

    with open(file_path) as f:
        lines = f.read().splitlines()

    header_seen = False
    for i, line in enumerate(lines):

        if line.startswith('#'):
            continue

        if not header_seen:
            header_seen = True
            continue

        time_str, rest = line.split(',', 1)

        # Frames with only additional fragments have no main time
        if not time_str:
            continue

        time = datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f')
        time += datetime.timedelta(milliseconds=milliseconds)
        lines[i] = time.strftime('%Y-%m-%dT%H:%M:%S.%f') + ',' + rest

    with open(file_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def test_load_rolling_shutter_times(tmp_path, monkeypatch):
    """ A file with annotations gets its frames from the frame numbers: times with the rolling shutter correction
        don't match the times of the frames. """

    reduction = Reduction(tmp_path)
    reduction.addPick(10, 110.0, 200.0)
    reduction.addPick(11, 111.0, 200.0, flare=True)
    reduction.addPick(12, 112.0, 200.0)
    reduction.addFragmentPoint(13, 1, 113.0, 210.0)
    file_path, _, _, _ = saveAndRead(reduction)
    shiftEcsvTimes(file_path, 13)

    # The message of a file which can't be read needs a running application
    monkeypatch.setattr('Utils.SkyFit2.qmessagebox', lambda **kwargs: None)

    loaded = Reduction(tmp_path)
    pick_list = loaded.loadECSV(file_path)

    assert sorted(pick_list.keys()) == [10, 11, 12]
    assert pick_list[11]['x_centroid'] == pytest.approx(111.0)
    assert pick_list[11]['flare']
    assert sorted(loaded.fragment_picks.keys()) == [13]


def test_load_frame_number_out_of_range(tmp_path, monkeypatch):
    """ A file with annotations whose frames are not in the input is refused, as one whose times don't match. """

    reduction = Reduction(tmp_path)
    reduction.addPick(200, 110.0, 200.0, flare=True)
    file_path, _, _, _ = saveAndRead(reduction)

    loaded = Reduction(tmp_path)
    loaded.img_handle.total_frames = 100

    messages = []
    monkeypatch.setattr('Utils.SkyFit2.qmessagebox', lambda **kwargs: messages.append(kwargs))

    assert loaded.loadECSV(file_path) is None
    assert len(messages) == 1


def test_valid_ecsv(tmp_path):
    """ The column declarations in the header match the data columns, including the empty values. """

    table_module = pytest.importorskip("astropy.table")

    reduction = Reduction(tmp_path)
    reduction.addPick(10, 110.0, 200.0, flare=True)
    reduction.addFragmentPoint(10, 1, 110.0, 210.0)
    reduction.addFragmentPoint(11, 1, 111.0, 210.0)
    file_path, _, _, _ = saveAndRead(reduction)

    table = table_module.Table.read(file_path, format='ascii.ecsv')

    assert list(table['frame_number']) == [10, 11]
    assert table['flare'][0] and table['flare'].mask[1]
    assert table['x_image'].mask.tolist() == [False, True]
    assert list(table['x_image1']) == pytest.approx([110.0, 111.0])


class GreatCircles(Reduction):
    """ Stand-in for PlateTool with the great circle lines. """

    updateGreatCircle = PlateTool.updateGreatCircle
    drawGreatCircle = PlateTool.drawGreatCircle

    def __init__(self, dir_path):

        Reduction.__init__(self, dir_path)

        self.img_handle.currentFrameTime = lambda dt_obj=False: self.img_handle.beginning_datetime
        self.platepar.F_scale = 20.0
        self.platepar.updateRefRADec(skip_rot_update=True)

        self.great_circle_line = pg.PlotCurveItem()
        self.fragment_great_circle_lines = {k: pg.PlotCurveItem() for k in range(1, len(FRAGMENT_COLORS))}


def lineColor(line):
    return line.opts['pen'].color().getRgb()[:3]


def distanceToLine(line, x, y):
    """ Distance in pixels from an image point to the closest segment of a line. """

    x_line, y_line = line.getData()
    xa, ya, xb, yb = x_line[:-1], y_line[:-1], x_line[1:], y_line[1:]

    t = np.clip(((x - xa)*(xb - xa) + (y - ya)*(yb - ya))/((xb - xa)**2 + (yb - ya)**2), 0, 1)

    return np.min(np.hypot(xa + t*(xb - xa) - x, ya + t*(yb - ya) - y))


def test_great_circle_colour_follows_the_fragments(qapp, tmp_path):
    """ The main line is purple until there are other fragments, and a fragment needs two points for its own. """

    session = GreatCircles(tmp_path)
    session.addPick(10, 200.0, 100.0)
    session.addPick(14, 400.0, 200.0)
    session.addPick(18, 600.0, 300.0)

    session.updateGreatCircle()
    main_line = session.great_circle_line.getData()[0].copy()
    assert lineColor(session.great_circle_line) == GREAT_CIRCLE_COLOR

    # With the first point of a fragment only the colour of the main line changes
    session.addFragmentPoint(10, 3, 200.0, 400.0)
    session.updateGreatCircle()
    assert lineColor(session.great_circle_line) == FRAGMENT_COLORS[0]
    assert np.array_equal(session.great_circle_line.getData()[0], main_line)
    assert session.fragment_great_circle_lines[3].getData()[0] is None

    # With the second point the fragment gets its line, and the others have none
    session.addFragmentPoint(14, 3, 500.0, 420.0)
    session.updateGreatCircle()
    assert len(session.fragment_great_circle_lines[3].getData()[0]) > 0
    assert session.fragment_great_circle_lines[1].getData()[0] is None

    # Removing the points goes back step by step
    session.removeFragmentPoint(14, 3)
    session.updateGreatCircle()
    assert session.fragment_great_circle_lines[3].getData()[0] is None
    assert lineColor(session.great_circle_line) == FRAGMENT_COLORS[0]

    session.removeFragmentPoint(10, 3)
    session.updateGreatCircle()
    assert lineColor(session.great_circle_line) == GREAT_CIRCLE_COLOR


def test_great_circle_of_a_fragment_goes_through_its_points(qapp, tmp_path):
    """ Each line is fitted only to the points of its fragment, also when they share frames. """

    session = GreatCircles(tmp_path)
    session.addPick(10, 200.0, 100.0)
    session.addPick(14, 400.0, 200.0)
    session.addPick(18, 600.0, 300.0)
    session.addFragmentPoint(10, 1, 200.0, 400.0)
    session.addFragmentPoint(14, 1, 500.0, 420.0)
    session.addFragmentPoint(14, 2, 700.0, 100.0)
    session.addFragmentPoint(18, 2, 800.0, 600.0)
    session.updateGreatCircle()

    # The lines are in pixel centre coordinates
    for x, y in ((200.5, 100.5), (400.5, 200.5), (600.5, 300.5)):
        assert distanceToLine(session.great_circle_line, x, y) < 1

    line1, line2 = session.fragment_great_circle_lines[1], session.fragment_great_circle_lines[2]
    for x, y in ((200.5, 400.5), (500.5, 420.5)):
        assert distanceToLine(line1, x, y) < 1
    for x, y in ((700.5, 100.5), (800.5, 600.5)):
        assert distanceToLine(line2, x, y) < 1

    # The points of a fragment are far from the line of another one
    assert distanceToLine(line1, 700.5, 100.5) > 50


def test_great_circle_of_main_leaves_out_unused_picks_only(qapp, tmp_path):
    """ The main picks not used in the trajectory don't count, and those of the fragments are all used. """

    session = GreatCircles(tmp_path)
    session.addPick(10, 200.0, 100.0)
    session.addPick(14, 400.0, 200.0, trajectory_use=False)
    session.addFragmentPoint(10, 1, 200.0, 400.0)
    session.addFragmentPoint(14, 1, 500.0, 420.0)
    session.updateGreatCircle()

    assert session.great_circle_line.getData()[0] is None
    assert len(session.fragment_great_circle_lines[1].getData()[0]) > 0


def test_no_great_circles_with_ground_points(qapp, tmp_path):
    """ As for the main fragment, ground points have no great circle. """

    session = GreatCircles(tmp_path)
    session.meas_ground_points = True
    session.addFragmentPoint(10, 1, 200.0, 400.0)
    session.addFragmentPoint(14, 1, 500.0, 420.0)
    session.updateGreatCircle()

    assert session.fragment_great_circle_lines[1].getData()[0] is None


class Gui(object):
    """ Stand-in for PlateTool with what the Annotations tab uses. """

    def __init__(self):

        self.frame = 10
        self.img = type('Img', (), {'getFrame': lambda _: self.frame})()
        self.pick_list = {10: {'x_centroid': 110.0, 'y_centroid': 200.0, 'mode': 1}}
        self.fragment_picks = {}
        self.redraws = 0
        self.refits = 0
        self.has_data = True

    def hasData(self):
        return self.has_data

    def updatePicks(self):
        self.redraws += 1

    def updateGreatCircle(self):
        self.refits += 1

    def updateLeftLabels(self):
        pass


@pytest.fixture(scope='module')
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp):
    return AnnotationsWidget(Gui())


def test_widget_without_data(qapp):
    """ SkyFit2 started with no input has no current frame, so the tab has no pick to annotate. """

    gui = Gui()
    gui.has_data = False

    def noFrame(_):
        raise AttributeError("'NoneType' object has no attribute 'current_frame'")

    gui.img = type('Img', (), {'getFrame': noFrame})()

    widget = AnnotationsWidget(gui)

    assert widget.currentPick() is None
    assert not widget.flare.isEnabled()
    assert not widget.trajectory_use.isEnabled()


def test_widget_main_fragment(widget):
    """ The main fragment pick can be flagged as a flare and excluded from the trajectory. """

    assert widget.fragment.count() == 9
    assert widget.fragment.itemText(0) == '0 — Main'
    assert widget.fragment.itemText(8) == '8 — Fragment 8'

    assert widget.trajectory_use.isEnabled() and widget.trajectory_use.isChecked()
    assert not widget.flare.isChecked()

    widget.flare.click()
    widget.trajectory_use.click()

    pick = widget.gui.pick_list[10]
    assert pick['flare'] is True
    assert pick['trajectory_use'] is False

    # The pick marker is redrawn and the great circle refitted right away on every change
    assert widget.gui.redraws == 2
    assert widget.gui.refits == 2


def test_widget_secondary_fragment(widget):
    """ Other fragments cannot be annotated, are never used in the main trajectory and do not change
        the main pick.
    """

    widget.fragment.setCurrentIndex(1)

    assert not widget.trajectory_use.isEnabled()
    assert not widget.trajectory_use.isChecked()

    assert not widget.flare.isEnabled()

    # Also when a point of fragment 1 exists on this frame
    widget.gui.fragment_picks = {10: {1: {'x_centroid': 1.0, 'y_centroid': 1.0, 'flare': False}}}
    widget.updateAnnotations()
    assert not widget.flare.isEnabled() and not widget.flare.isChecked()
    assert not widget.trajectory_use.isEnabled()

    widget.onAnnotationChanged()
    assert widget.gui.fragment_picks[10][1] == {'x_centroid': 1.0, 'y_centroid': 1.0, 'flare': False}
    assert 'flare' not in widget.gui.pick_list[10]

    # On the next frame there is no point of fragment 1 yet, and the fragment stays selected
    widget.gui.frame = 11
    widget.updateAnnotations()
    assert widget.fragment.currentIndex() == 1
    assert not widget.flare.isEnabled() and not widget.flare.isChecked()

    # Back to the main fragment, the pick is used in the trajectory again
    widget.gui.frame = 10
    widget.fragment.setCurrentIndex(0)
    assert widget.trajectory_use.isEnabled() and widget.trajectory_use.isChecked()
