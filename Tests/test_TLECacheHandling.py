""" Tests for atomic TLE cache downloads, validation, exact group lookup, and offline fallback. """

from __future__ import print_function, division, absolute_import

import datetime
import os
from unittest.mock import patch

import pytest

from RMS.Routines.SatellitePositions import (
    findClosestTLEFile,
    findNewestTLECacheFile,
    isValidTLEFile,
    listTLECacheFiles,
    loadTLEs,
)

SAMPLE_TLE = (
    "ISS (ZARYA)\n"
    "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991\n"
    "2 25544  51.6439 201.2643 0007417 356.5673 139.3661 15.50083952202315\n"
)


def testListAndFindClosestTLEMatchExactGroupOnly(tmp_path):
    """ Exact group matching must distinguish active.txt from gp_active.txt and starlink.txt and skip empty files. """

    f_active = tmp_path/"TLE_20260101_000000_active.txt"
    f_gp_active = tmp_path/"TLE_20260102_000000_gp_active.txt"
    f_starlink = tmp_path/"TLE_20260102_000000_starlink.txt"
    f_empty = tmp_path/"TLE_20260103_000000_active.txt"
    f_corrupt = tmp_path/"TLE_20260104_000000_active.txt"

    f_active.write_text(SAMPLE_TLE)
    f_gp_active.write_text(SAMPLE_TLE)
    f_starlink.write_text(SAMPLE_TLE)
    f_empty.write_text("")
    f_corrupt.write_text("<html>429 Too Many Requests</html>\n")

    listed = [os.path.basename(p) for _, p in listTLECacheFiles(str(tmp_path), "active.txt")]
    assert sorted(listed) == [
        "TLE_20260101_000000_active.txt",
        "TLE_20260103_000000_active.txt",
        "TLE_20260104_000000_active.txt",
    ]

    target = datetime.datetime(2026, 1, 2, 12, 0, 0, tzinfo=datetime.timezone.utc)
    closest = findClosestTLEFile(str(tmp_path), target, cache_file_name="active.txt")
    assert closest == str(f_active)
    assert findNewestTLECacheFile(str(tmp_path), "active.txt") == str(f_active)


def testRateLimitOrInterruptedDownloadNeverCachedAndCleansPartFile(tmp_path, monkeypatch):
    """ Non-TLE responses, failed downloads, and interrupts must not create cache files or leave .part files. """

    import RMS.Routines.SatellitePositions as sat_pos
    monkeypatch.setattr(sat_pos, "SKYFIELD_AVAILABLE", True)

    old_cache = tmp_path/"TLE_20250101_000000_active.txt"
    old_cache.write_text(SAMPLE_TLE)

    # 1. Rate-limit HTML response -> rejected, falls back to old daily cache, no new file or .part left
    def _writeHtml(url, path):
        with open(path, "w") as f:
            f.write("<html><body>Rate limit exceeded</body></html>\n")

    with patch("urllib.request.urlretrieve", side_effect=_writeHtml):
        with patch("RMS.Routines.SatellitePositions.loadRobustTLEs", return_value=["ok"]) as mock_load:
            res = loadTLEs(str(tmp_path), "active.txt", max_age_hours=24, use_daily_cache=True)
            assert res == ["ok"]
            mock_load.assert_called_once_with(str(old_cache))

    remaining = sorted(os.listdir(str(tmp_path)))
    assert remaining == ["TLE_20250101_000000_active.txt"]

    # 2. KeyboardInterrupt mid-download -> .part removed in finally
    def _interruptDownload(url, path):
        with open(path, "w") as f:
            f.write("partial data")
        raise KeyboardInterrupt()

    with patch("urllib.request.urlretrieve", side_effect=_interruptDownload):
        with pytest.raises(KeyboardInterrupt):
            loadTLEs(str(tmp_path), "active.txt", max_age_hours=24, use_daily_cache=True)

    assert sorted(os.listdir(str(tmp_path))) == ["TLE_20250101_000000_active.txt"]


def testIsValidTLEFileValidation(tmp_path):
    """ isValidTLEFile accepts real TLE line pairs and rejects plain text or empty files. """

    good = tmp_path/"good.txt"
    bad = tmp_path/"bad.txt"
    empty = tmp_path/"empty.txt"

    good.write_text(SAMPLE_TLE)
    bad.write_text("429 Too Many Requests\n")
    empty.write_text("")

    assert isValidTLEFile(str(good)) is True
    assert isValidTLEFile(str(bad)) is False
    assert isValidTLEFile(str(empty)) is False
