import datetime

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")
pytest.importorskip("cv2")

from RMS.Astrometry.Conversions import datetime2JD
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert
from Utils.FluxBatch import computeDateAxisTicks, datetimeToSolPlot, solToDatetime, uniqueBinYears


def _solDeg(dt):
    # Solar longitude (deg) at the given time, used as the reference for the plotted axis
    return np.degrees(jd2SolLonSteyaert(datetime2JD(dt)))


def test_short_outburst_uses_time_labels_and_dates_at_midnight():
    # A few hour outburst across midnight is labelled with clock times, and the date replaces 00:00
    dt_beg = datetime.datetime(2026, 10, 22, 22, 10)
    dt_end = datetime.datetime(2026, 10, 23, 2, 5)

    major_dts, major_labels, minor_dts, axis_label = computeDateAxisTicks(dt_beg, dt_end)

    assert major_labels == ["22:30", "23:00", "23:30", "Oct 23", "00:30", "01:00", "01:30", "02:00"]
    assert axis_label == "UTC 2026"
    assert minor_dts == []


def test_short_window_without_midnight_shows_date_in_axis_label():
    # Without a 00 UTC label nothing on the axis shows the date, so it has to be in the axis label
    dt_beg = datetime.datetime(2026, 10, 23, 11, 23)
    dt_end = datetime.datetime(2026, 10, 23, 12, 3)

    major_dts, major_labels, _, axis_label = computeDateAxisTicks(dt_beg, dt_end)

    assert major_labels == ["11:30", "11:40", "11:50", "12:00"]
    assert axis_label == "UTC 2026 Oct 23"


@pytest.mark.parametrize("days", [1.4, 3, 12, 40, 70, 150])
def test_label_count_and_midnight_ticks(days):
    # From a day long peak to a long shower, the labels stay readable and fall inside the plotted range
    dt_beg = datetime.datetime(2026, 9, 5, 7, 13)
    dt_end = dt_beg + datetime.timedelta(days=days)

    major_dts, major_labels, minor_dts, _ = computeDateAxisTicks(dt_beg, dt_end)

    assert 2 <= len(major_dts) <= 12
    assert len(major_dts) == len(major_labels)
    assert all(dt_beg <= dt <= dt_end for dt in major_dts + minor_dts)

    # Every 00 UTC in the range is marked for showers up to a few months long
    if days <= 120:
        midnights = {dt for dt in major_dts + minor_dts if (dt.hour == 0) and (dt.minute == 0)}
        assert len(midnights) == int(dt_end.toordinal() - dt_beg.toordinal())


def test_year_crossing_axis_label():
    # A shower crossing the New Year shows both years, and the 1st of January is labelled
    _, major_labels, _, axis_label = computeDateAxisTicks(datetime.datetime(2025, 12, 27, 3),
        datetime.datetime(2026, 1, 11, 4))

    assert axis_label == "UTC 2025/26"
    assert "Jan 1" in major_labels


@pytest.mark.parametrize("dt_true", [
    datetime.datetime(2026, 1, 3, 21, 17),
    datetime.datetime(2025, 12, 28, 6, 0),
    datetime.datetime(2026, 3, 20, 14, 46),
    datetime.datetime(2026, 10, 21, 3, 30),
    ])
def test_sol_datetime_roundtrip(dt_true):
    # The time read off the date axis must match the true time, also across the New Year and the equinox

    # Use a reference epoch near the tested time, as in a single shower plot
    dt_ref = datetime.datetime(2026, 1, 3, 12, 0)
    if dt_true.month > 2:
        dt_ref = dt_true - datetime.timedelta(days=10)
    sol_ref_deg = _solDeg(dt_ref)

    # Convert to the plotted solar longitude and back
    sol_plot = datetimeToSolPlot(dt_true, sol_ref_deg)
    assert abs(sol_plot - sol_ref_deg) <= 180

    dt_back = solToDatetime(sol_plot, dt_ref, sol_ref_deg)
    assert abs((dt_back - dt_true).total_seconds()) < 60


def test_unique_bin_years_ignores_duplicate_entries():
    # Several bin entries from the same season must count as one year, so the date axis isn't dropped
    dt_a = datetime.datetime(2026, 10, 1)
    dt_b = datetime.datetime(2026, 11, 1)
    single_year = [[(dt_a, dt_b), []], [(dt_a, dt_b + datetime.timedelta(minutes=5)), []]]
    multi_year = single_year + [[(dt_a.replace(year=2025), dt_b.replace(year=2025)), []]]

    assert uniqueBinYears(single_year) == [2026]
    assert uniqueBinYears(multi_year) == [2025, 2026]


def test_dropped_midnight_label_keeps_date_in_axis_label():
    # The only 00 UTC label is in the unlabelled strip at the left edge, so the date must move to the
    #   axis label
    dt_beg = datetime.datetime(2026, 10, 22, 23, 58)
    dt_end = datetime.datetime(2026, 10, 23, 4, 0)
    label_beg = datetime.datetime(2026, 10, 23, 0, 5)

    major_dts, major_labels, minor_dts, axis_label = computeDateAxisTicks(dt_beg, dt_end,
        label_beg=label_beg)

    assert "Oct 23" not in major_labels
    assert datetime.datetime(2026, 10, 23, 0, 0) in minor_dts
    assert all(dt >= label_beg for dt in major_dts)
    assert axis_label == "UTC 2026 Oct 23"
