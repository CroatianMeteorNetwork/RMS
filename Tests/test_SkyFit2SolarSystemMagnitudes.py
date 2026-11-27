""" Pure unit tests for SkyFit2 planet and Moon phase geometry and visual magnitudes. """

from __future__ import print_function, division, absolute_import

import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def skyfit2():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("pyqtgraph")
    return pytest.importorskip("Utils.SkyFit2")


def testPlanetPhaseGeometry(skyfit2):
    """ planetPhaseGeometry must return exact distances and phase angles for elementary configurations. """

    # Opposition: Sun at (-1, 0, 0), outer planet at (4, 0, 0) -> r=5, delta=4, alpha=0 deg
    r, delta, alpha = skyfit2.planetPhaseGeometry([4.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
    assert r == pytest.approx(5.0)
    assert delta == pytest.approx(4.0)
    assert alpha == pytest.approx(0.0)

    # Quadrature / 90 deg phase: Sun at (1, 0, 0), body at (0, 1, 0) -> r=sqrt(2), delta=1, alpha=45 deg
    r2, delta2, alpha2 = skyfit2.planetPhaseGeometry([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    assert r2 == pytest.approx(np.sqrt(2.0))
    assert delta2 == pytest.approx(1.0)
    assert alpha2 == pytest.approx(45.0)


@pytest.mark.parametrize("body_name, date_str, expected_mag, tol", [
    # Published ephemeris / Astronomical Almanac visual magnitudes
    ("mercury", "2021-01-24T00:00:00", -0.6, 0.25),   # greatest eastern elongation (old formula gave +2.31)
    ("venus",   "2023-06-04T00:00:00", -4.4, 0.25),   # greatest eastern elongation
    ("mars",    "2022-12-08T00:00:00", -1.9, 0.25),   # opposition
    ("jupiter", "2023-11-03T00:00:00", -2.9, 0.25),   # opposition
    ("saturn",  "2021-08-02T00:00:00",  0.2, 0.25),   # opposition with tilted rings (old formula gave +0.99)
    ("uranus",  "2023-11-13T00:00:00",  5.6, 0.25),   # opposition
    ("neptune", "2023-09-19T00:00:00",  7.8, 0.25),   # opposition
    ("moon",    "2023-09-29T10:00:00", -12.7, 0.35),  # full moon
    ("moon",    "2023-09-22T19:30:00", -10.1, 0.35),  # first quarter (old formula gave -11.95)
])
def testSolarSystemMagnitudesMatchPublishedValues(skyfit2, body_name, date_str, expected_mag, tol):
    """ computeSolarSystemMagnitude must agree with published visual magnitudes across planets and Moon. """

    astropy_time = pytest.importorskip("astropy.time")
    astropy_coords = pytest.importorskip("astropy.coordinates")

    t = astropy_time.Time(date_str, scale="utc")
    sun = astropy_coords.get_sun(t)
    body = astropy_coords.get_body(body_name, t)

    mag = skyfit2.computeSolarSystemMagnitude(body_name, body, sun, t)
    assert mag == pytest.approx(expected_mag, abs=tol)
