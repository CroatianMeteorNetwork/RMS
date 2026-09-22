""" Tests for time conversion helpers in RMS.Astrometry.Conversions. """

from __future__ import print_function, division, absolute_import

import datetime

import pytest

np = pytest.importorskip("numpy")

from RMS.Astrometry.Conversions import jd2YearsFromJ2000, datetime2JD


def testJd2YearsFromJ2000AtEpoch():
    """ J2000.0 itself is zero years from J2000. """

    assert jd2YearsFromJ2000(2451545.0) == 0.0


def testJd2YearsFromJ2000OneJulianYear():
    """ One Julian year is exactly 365.25 days. """

    assert jd2YearsFromJ2000(2451545.0 + 365.25) == pytest.approx(1.0)
    assert jd2YearsFromJ2000(2451545.0 - 2*365.25) == pytest.approx(-2.0)


def testJd2YearsFromJ2000KeepsDayFraction():
    """ Half a day must contribute to the result (a .days based difference would truncate it). """

    # Two epochs half a day apart
    years_full = jd2YearsFromJ2000(2451545.0 + 100.0)
    years_half = jd2YearsFromJ2000(2451545.0 + 100.5)

    # The difference must be exactly half a day expressed in Julian years
    assert years_half - years_full == pytest.approx(0.5/365.25)


def testJd2YearsFromJ2000MatchesDatetimeDifference():
    """ Agrees with the total_seconds based computation used elsewhere in RMS. """

    # An arbitrary time with a sub-second component
    dt = datetime.datetime(2026, 9, 21, 3, 17, 45, 250000)
    jd = datetime2JD(dt)

    # The same quantity computed straight from the datetime difference
    expected = (dt - datetime.datetime(2000, 1, 1, 12, 0, 0)).total_seconds()/(365.25*24*3600)

    assert jd2YearsFromJ2000(jd) == pytest.approx(expected, abs=1e-9)
