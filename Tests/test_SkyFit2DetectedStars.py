"""
Tests of the choice of the stars detected on the current image in SkyFit2 (PlateTool.getDetectedStars).

The stars re-detected in the Star Detection tab take precedence over CALSTARS while the override is enabled. The
star picking, the catalog balancing, the quick alignment and the astrometry.net fits all read the detections, and
they must all read the same list, so the choice is made in one method only.
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


def makePlateTool(name, override_enabled, override_data, calstars):
    """ A stand-in with only the attributes which getDetectedStars reads. """

    return SimpleNamespace(
        img_handle=SimpleNamespace(name=lambda: name),
        star_detection_override_enabled=override_enabled,
        star_detection_override_data=override_data,
        calstars=calstars
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
