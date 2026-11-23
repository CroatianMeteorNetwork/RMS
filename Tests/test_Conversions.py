"""
Tests of the conversions of times in RMS.Astrometry.Conversions: the reading of a time given as a string or a
number (timeStringToDatetime, used for the files of frame times), and the conversion of Unix times to naive UTC
datetimes and back (UTCFromTimestamp, datetime2UnixTime) to the microsecond.
"""

import datetime

import numpy as np
import pytest

from RMS.Astrometry.Conversions import datetime2UnixTime, timeStringToDatetime
from RMS.Misc import UTCFromTimestamp


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
def test_time_string_to_datetime(value, expected):

    # The time stamp of the saved frames, ISO 8601 (UTC or with an offset) and Unix seconds
    assert timeStringToDatetime(value) == expected


@pytest.mark.parametrize("value", ["broken", "2026-13-01T00:00:00", True, None, [1]])
def test_time_string_to_datetime_rejects(value):

    with pytest.raises(ValueError):
        timeStringToDatetime(value)


def test_unix_time_conversions_to_the_microsecond():

    # Unix times of capture frames (to the microsecond and finer) as datetimes, the same as the epoch plus the
    #   time, and back
    rng = np.random.RandomState(0)
    times = np.concatenate([[0.0, 1795387645.002211, 1795387650.005963, 1795387700.999999],
                            rng.uniform(1.5e9, 2.0e9, 1000)])

    for t in times:
        dt = UTCFromTimestamp.utcfromtimestamp(float(t))
        assert dt == datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=float(t))
        assert abs(datetime2UnixTime(dt) - t) <= 0.5e-6
