""" Tests of the Moon check in the flux cloud detection (Utils.Flux.detectMoon).

    detectMoon used to decide if the Moon is up by comparing the next rising and setting times. In ephem < 4.2
    that search loops forever for the Moon at high latitudes: on 2026-09-13 at Arrival Heights, Antarctica
    (AQ000A) it hung the night processing, and every restart hung again on the same night. The Moon is now
    checked from its current altitude, which is the same test without the search.
"""

from __future__ import print_function, division, absolute_import

import os
import tarfile
import types
from datetime import datetime, timedelta

import ephem
import pytest

from RMS.Formats.Platepar import Platepar
import Utils.Flux as flux


STATIONS_TAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ExampleStationData', 'stations.tar.bz2')

# Arrival Heights, Antarctica (AQ000A)
AQ000A = (-77.830215, 166.661952, 224)

# Image times of the night that hung (from the CALSTARS file of AQ000A_20260913_083938)
HUNG_NIGHT_FRAMES = [
    "FF_AQ000A_20260913_112302_123_0204288.fits",
    "FF_AQ000A_20260913_113558_456_0223232.fits",
    "FF_AQ000A_20260913_121014_789_0273408.fits",
    "FF_AQ000A_20260913_130845_380_0401152.fits",
    "FF_AQ000A_20260913_154122_359_0628992.fits",
    "FF_AQ000A_20260913_154842_988_0640000.fits",
]


@pytest.fixture
def aq000a(tmp_path):
    """ A platepar and config at Arrival Heights. The platepar is an example one moved to that location. """

    with tarfile.open(STATIONS_TAR) as tar:
        member = tar.getmember('Stations/AU000A/platepar_cmn2010.cal')
        member.name = 'platepar_cmn2010.cal'
        tar.extract(member, str(tmp_path))

    platepar = Platepar()
    platepar.read(os.path.join(str(tmp_path), 'platepar_cmn2010.cal'))
    platepar.lat, platepar.lon, platepar.elev = AQ000A

    config = types.SimpleNamespace(latitude=AQ000A[0], longitude=AQ000A[1], elevation=AQ000A[2], fps=25)

    return platepar, config


def test_detect_moon_does_not_search_rise_set(aq000a, monkeypatch):
    """ The night that hung: detectMoon must not call the Moon rising/setting search at all. """

    platepar, config = aq000a

    def forbidden(*args, **kwargs):
        raise AssertionError("detectMoon must not search for the rising/setting times")

    monkeypatch.setattr(ephem.Observer, 'next_rising', forbidden)
    monkeypatch.setattr(ephem.Observer, 'next_setting', forbidden)

    kept = flux.detectMoon(HUNG_NIGHT_FRAMES, platepar, config)

    assert set(kept) <= set(HUNG_NIGHT_FRAMES)


def moonUpRiseSet(o, m):
    """ The old test: the Moon is up if it sets before it rises next.

    Return:
        (up, exception): up [bool], and whether it was decided from an AlwaysUp/NeverUp error.
    """

    try:
        return not (o.next_rising(m) < o.next_setting(m)), False
    except ephem.AlwaysUpError:
        return True, True
    except ephem.NeverUpError:
        return False, True


def moonUp(o, m):
    """ The new test: altitude of the upper limb above the horizon. """
    return not ((float(m.alt) + float(m.radius)) < float(o.horizon))


@pytest.mark.parametrize('site', [AQ000A, (43.1928, -81.3157, 323), (0.0, 0.0, 0), (78.2232, 15.6267, 30)])
def test_moon_altitude_matches_rise_set(site):
    """ The altitude test gives the same answer as the rising/setting comparison, except within a few minutes
        of a rising or setting, and where the old test fell back on an AlwaysUp/NeverUp error (at high
        latitudes, where it was wrong - see the next test). Checked through a year every 3 hours. """

    o = ephem.Observer()
    o.lat, o.long, o.elevation = str(site[0]), str(site[1]), site[2]
    o.horizon = '0:0'

    t = datetime(2026, 1, 1)
    checked = 0
    while t < datetime(2027, 1, 1):

        o.date = t
        m = ephem.Moon()
        m.compute(o)

        # Altitude first: the rising/setting search recomputes the body at the event time
        up_altitude = moonUp(o, m)
        up_rise_set, from_exception = moonUpRiseSet(o, m)
        if (up_altitude != up_rise_set) and (not from_exception):

            # Must be right at a rising or setting: the state changes within a few minutes
            states = []
            for dt in (-5, 5):
                o.date = t + timedelta(minutes=dt)
                mm = ephem.Moon()
                mm.compute(o)
                states.append(moonUp(o, mm))

            assert states[0] != states[1], "Disagreement at {} not near a rising/setting".format(t)

        checked += 1
        t += timedelta(hours=3)

    assert checked > 2900


def test_moon_below_horizon_before_circumpolar_rise():
    """ Arrival Heights, 2026-01-10 06:00 UTC: the Moon is 2.5 deg below the horizon and rises at 09:09, then
        stays up for more than a day. The old test got AlwaysUpError from the setting search and called the
        Moon up. """

    o = ephem.Observer()
    o.lat, o.long, o.elevation = str(AQ000A[0]), str(AQ000A[1]), AQ000A[2]
    o.horizon = '0:0'
    o.date = datetime(2026, 1, 10, 6, 0)
    m = ephem.Moon()
    m.compute(o)

    assert not moonUp(o, m)
    assert moonUpRiseSet(o, m) == (True, True)


def test_rise_set_search_moves_the_moon():
    """ Why the old code was also wrong about the Moon position: the rising/setting search leaves the body
        computed at the event time, and detectMoon then used m.ra/m.dec to check if the Moon is in the FOV. """

    o = ephem.Observer()
    o.lat, o.long, o.elevation = '43.1928', '-81.3157', 323
    o.date = datetime(2026, 3, 1, 3, 0)
    m = ephem.Moon()
    m.compute(o)
    ra_before = float(m.ra)

    o.next_setting(m)

    assert abs(float(m.ra) - ra_before) > 1e-4
