"""
Tests of the choice of the stars detected on the current image in SkyFit2 (PlateTool.getDetectedStars).

The stars re-detected in the Star Detection tab take precedence over CALSTARS while the override is enabled, and the
detections removed or the stars marked by the user (Mark Stars) correct them. The star picking, the catalog
balancing, the quick alignment and the astrometry.net fits all read the detections, and they must all read the same
list, so the choice is made in one method only.
"""

import ast
import os
from types import SimpleNamespace

import numpy as np
import pytest


SKYFIT2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "Utils", "SkyFit2.py")

FF_NAME = "FF_XX0001_20261021_201500_123_0000512.fits"
FR_NAME = "FR_XX0001_20261021_201500_123_0000512.bin"

# Two rows in the CALSTARS column order Y X IntensSum Ampltd FWHM BgLvl SNR NSatPx
CALSTARS_STARS = [[100.0, 200.0, 5000.0, 300.0, 2.5, 40.0, 25.0, 0.0],
                  [150.0, 250.0, 3000.0, 200.0, 2.4, 41.0, 18.0, 0.0]]
OVERRIDE_STARS = [[101.0, 201.0, 5100.0, 310.0, 2.6, 39.0, 26.0, 0.0]]

# A star marked by the user
MARKED_STAR = [300.0, 400.0, 800.0, 0.0, 2.2, 0.0, 9.0, 0.0]


def makePlateTool(name, override_enabled, override_data, calstars, marked=None, removed=None):
    """ A stand-in with only the attributes which getDetectedStars reads. """

    return SimpleNamespace(
        img_handle=SimpleNamespace(name=lambda: name),
        star_detection_override_enabled=override_enabled,
        star_detection_override_data=override_data,
        calstars=calstars,
        star_detection_marked={} if marked is None else marked,
        star_detection_removed={} if removed is None else removed
    )


@pytest.fixture(scope="module")
def getDetectedStars():
    skyfit2 = pytest.importorskip("Utils.SkyFit2")
    return skyfit2.PlateTool.getDetectedStars


def test_override_takes_precedence_when_enabled(getDetectedStars):
    pt = makePlateTool(FF_NAME, True, {FF_NAME: OVERRIDE_STARS}, {FF_NAME: CALSTARS_STARS})
    np.testing.assert_array_equal(getDetectedStars(pt), np.array(OVERRIDE_STARS))


def test_disabled_override_falls_back_to_calstars(getDetectedStars):
    pt = makePlateTool(FF_NAME, False, {FF_NAME: OVERRIDE_STARS}, {FF_NAME: CALSTARS_STARS})
    np.testing.assert_array_equal(getDetectedStars(pt), np.array(CALSTARS_STARS))


def test_image_not_redetected_falls_back_to_calstars(getDetectedStars):
    pt = makePlateTool(FF_NAME, True, {}, {FF_NAME: CALSTARS_STARS})
    np.testing.assert_array_equal(getDetectedStars(pt), np.array(CALSTARS_STARS))


def test_image_without_detections_returns_none(getDetectedStars):
    pt = makePlateTool(FF_NAME, True, {}, {})
    assert getDetectedStars(pt) is None


def test_fr_file_reads_the_detections_of_its_ff_file(getDetectedStars):
    pt = makePlateTool(FR_NAME, False, {}, {FF_NAME: CALSTARS_STARS})
    np.testing.assert_array_equal(getDetectedStars(pt), np.array(CALSTARS_STARS))


def test_marked_stars_are_added_after_the_detections(getDetectedStars):
    pt = makePlateTool(FF_NAME, False, {}, {FF_NAME: CALSTARS_STARS}, marked={FF_NAME: [MARKED_STAR]})
    np.testing.assert_array_equal(getDetectedStars(pt), np.array(CALSTARS_STARS + [MARKED_STAR]))


def test_marked_stars_on_an_image_without_detections(getDetectedStars):
    pt = makePlateTool(FF_NAME, False, {}, {}, marked={FF_NAME: [MARKED_STAR]})
    np.testing.assert_array_equal(getDetectedStars(pt), np.array([MARKED_STAR]))


def test_removed_detection_stays_removed_after_redetection(getDetectedStars):
    """ A detection is removed by its position: the same star re-detected 1 px away is left out too, and the
        other stars are kept, with or without the override. """

    # Removed at x=200, y=100, then re-detected at x=201, y=101
    pt = makePlateTool(FF_NAME, True, {FF_NAME: OVERRIDE_STARS + [CALSTARS_STARS[1]]}, {FF_NAME: CALSTARS_STARS},
                       removed={FF_NAME: [(200.0, 100.0)]})
    np.testing.assert_array_equal(getDetectedStars(pt), np.array([CALSTARS_STARS[1]]))

    pt.star_detection_override_enabled = False
    np.testing.assert_array_equal(getDetectedStars(pt), np.array([CALSTARS_STARS[1]]))


def test_marked_stars_take_the_columns_of_the_detections(getDetectedStars):
    """ Old CALSTARS files have fewer columns, the marked stars are cut to them. """

    old_calstars = [row[:4] for row in CALSTARS_STARS]
    pt = makePlateTool(FF_NAME, False, {}, {FF_NAME: old_calstars}, marked={FF_NAME: [MARKED_STAR]})
    np.testing.assert_array_equal(getDetectedStars(pt), np.array(old_calstars + [MARKED_STAR[:4]]))


def test_fr_file_reads_the_corrections_of_its_ff_file(getDetectedStars):
    pt = makePlateTool(FR_NAME, False, {}, {FF_NAME: CALSTARS_STARS}, marked={FF_NAME: [MARKED_STAR]},
                       removed={FF_NAME: [(250.0, 150.0)]})
    np.testing.assert_array_equal(getDetectedStars(pt), np.array([CALSTARS_STARS[0], MARKED_STAR]))


@pytest.fixture
def markingSession():
    """ A stand-in with the PlateTool methods which mark and remove stars, and the state they read. A click is
        centroided on the clicked position, with the other values of MARKED_STAR. """

    PlateTool = pytest.importorskip("Utils.SkyFit2").PlateTool

    class MarkingSession(object):

        getDetectedStars = PlateTool.getDetectedStars
        markStar = PlateTool.markStar
        unmarkStar = PlateTool.unmarkStar
        clearStarMarks = PlateTool.clearStarMarks

        def __init__(self):
            self.__dict__.update(vars(makePlateTool(FF_NAME, False, {}, {FF_NAME: CALSTARS_STARS})))
            self.star_aperture_radius = 5
            self.mouse_x = self.mouse_y = 0.0

        def click(self, x, y):
            # The mouse coordinates are offset by half a pixel from the image coordinates
            self.mouse_x, self.mouse_y = x + 0.5, y + 0.5

        def centroid(self):
            _, _, intensity, _, fwhm, _, snr, n_saturated = MARKED_STAR
            return self.mouse_x - 0.5, self.mouse_y - 0.5, fwhm, intensity, snr, bool(n_saturated)

        def updateCalstars(self):
            pass

    return MarkingSession()


def test_click_marks_a_missed_star_but_not_a_detected_one(markingSession):
    session = markingSession

    session.click(400.0, 300.0)
    session.markStar()
    assert session.star_detection_marked[FF_NAME] == [MARKED_STAR]

    # Within the aperture of a detection nothing is added
    session.click(202.0, 101.0)
    session.markStar()
    assert len(session.star_detection_marked[FF_NAME]) == 1


def test_right_click_removes_a_marked_star_first_then_a_detection(markingSession):
    session = markingSession
    session.click(400.0, 300.0)
    session.markStar()

    session.unmarkStar()
    assert session.star_detection_marked[FF_NAME] == []
    assert FF_NAME not in session.star_detection_removed

    session.click(251.0, 149.0)
    session.unmarkStar()
    assert session.star_detection_removed[FF_NAME] == [(250.0, 150.0)]
    np.testing.assert_array_equal(session.getDetectedStars(), np.array([CALSTARS_STARS[0]]))

    session.clearStarMarks()
    np.testing.assert_array_equal(session.getDetectedStars(), np.array(CALSTARS_STARS))


def test_only_getDetectedStars_chooses_between_override_and_calstars():
    """ A method which reads both self.calstars[...] and self.star_detection_override_data[...] chooses between
    the two sources itself, and would not see a change to how getDetectedStars chooses. """

    with open(SKYFIT2_PATH, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    choosers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        read_attrs = {sub.value.attr for sub in ast.walk(node)
                      if isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Load)
                      and isinstance(sub.value, ast.Attribute) and isinstance(sub.value.value, ast.Name)
                      and (sub.value.value.id == "self")}

        if {"calstars", "star_detection_override_data"} <= read_attrs:
            choosers.append(node.name)

    assert choosers == ["getDetectedStars"]
