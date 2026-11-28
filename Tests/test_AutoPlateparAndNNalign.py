""" Tests for AutoPlatepar and NNalign time/epoch handling, frame selection, and input guards. """

from __future__ import print_function, division, absolute_import

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from RMS.Astrometry.Conversions import date2JD, jd2YearsFromJ2000
from RMS.Astrometry.AutoPlatepar import (
    directoryReferenceJD,
    frameReferenceTime,
    loadCatalogStars,
    selectBestFrame,
)
from RMS.Astrometry import NNalign
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Formats.Platepar import Platepar


def testJd2YearsFromJ2000PreservesDayFraction():
    """ jd2YearsFromJ2000 must preserve the sub-day fraction (unlike datetime .days truncation). """

    jd_j2000 = 2451545.0
    assert jd2YearsFromJ2000(jd_j2000) == pytest.approx(0.0)
    assert jd2YearsFromJ2000(jd_j2000 + 365.25) == pytest.approx(1.0)
    assert jd2YearsFromJ2000(jd_j2000 + 0.5) == pytest.approx(0.5/365.25)


def testFrameReferenceTimeUsesMiddleOfFF():
    """ frameReferenceTime must offset by half the FF duration instead of using frame 0. """

    ff_name = "FF_XX0001_20261127_040000_000_0000000.fits"
    t0 = filenameToDatetime(ff_name)
    ff_dt, jd = frameReferenceTime(ff_name, fps=25.0, ff_frames=256)

    # 256 frames at 25 fps -> midpoint is 128/25 = 5.12 s after the first frame
    assert (ff_dt - t0).total_seconds() == pytest.approx(128.0/25.0, abs=1e-3)
    jd0 = date2JD(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second, t0.microsecond/1000.0)
    assert (jd - jd0)*86400.0 == pytest.approx(128.0/25.0, abs=1e-2)


def testDirectoryReferenceJDFindsFirstFFFromCalstarsOrFiles(tmp_path):
    """ directoryReferenceJD must derive the reference JD from CALSTARS entries or FF files on disk. """

    from RMS.Formats import CALSTARS

    ff_disk = "FF_XX0001_20261127_030000_000_0000000.fits"
    ff_cal = "FF_XX0001_20261127_020000_000_0000000.fits"

    # Empty directory -> None
    assert directoryReferenceJD(str(tmp_path), fps=25.0) is None

    # Non-standard FF-like name is skipped instead of raising; valid FF file on disk is used
    (tmp_path/"FF_calibration.fits").write_bytes(b"0")
    (tmp_path/ff_disk).write_bytes(b"0")
    jd_disk = directoryReferenceJD(str(tmp_path), fps=25.0)
    assert jd_disk == pytest.approx(frameReferenceTime(ff_disk, 25.0, ff_frames=256)[1])

    # When a CALSTARS file exists, its FF entry, chunk_frames, and header FPS take precedence over disk/config
    star_row = [[100.0, 200.0, 1000.0, 500.0, 2.5, 20.0, 15.0, 0.0]]
    CALSTARS.writeCALSTARS([[ff_cal, star_row]], str(tmp_path), "CALSTARS_XX0001_20261127.txt",
                           cam_code="XX0001", nrows=720, ncols=1280, chunk_frames=128)
    jd_cal = directoryReferenceJD(str(tmp_path), fps=25.0)
    assert jd_cal == pytest.approx(frameReferenceTime(ff_cal, 25.0, ff_frames=128)[1], abs=1e-9)

    # Header FPS in CALSTARS overrides the caller's fallback fps
    CALSTARS.writeCALSTARS([[ff_cal, star_row]], str(tmp_path), "CALSTARS_XX0001_20261127.txt",
                           cam_code="XX0001", nrows=720, ncols=1280, chunk_frames=256, fps=100.0)
    jd_hdr_fps = directoryReferenceJD(str(tmp_path), fps=25.0)
    assert jd_hdr_fps == pytest.approx(frameReferenceTime(ff_cal, 100.0, ff_frames=256)[1], abs=1e-9)
    assert abs(jd_hdr_fps - frameReferenceTime(ff_cal, 25.0, ff_frames=256)[1])*86400.0 == pytest.approx(3.84, abs=1e-2)


def testSelectBestFrameDoesNotPenalizeSensitiveCamera():
    """ On a night where clear frames routinely have ~400 stars, selectBestFrame must not pick a sparse frame. """

    rng = np.random.RandomState(42)

    def _makeFrame(n_stars):
        y = rng.uniform(20, 700, n_stars)
        x = rng.uniform(20, 1260, n_stars)
        intens = np.full(n_stars, 5000.0)
        amp = np.full(n_stars, 500.0)
        fwhm = np.full(n_stars, 2.5)
        bg = np.full(n_stars, 20.0)
        snr = np.full(n_stars, 15.0)
        sat = np.zeros(n_stars)
        return np.column_stack([y, x, intens, amp, fwhm, bg, snr, sat])

    calstars = {
        "FF_clear_1.fits": _makeFrame(400),
        "FF_clear_2.fits": _makeFrame(410),
        "FF_clear_3.fits": _makeFrame(390),
        "FF_clear_4.fits": _makeFrame(405),
        "FF_cloudy_sparse.fits": _makeFrame(120),
    }

    best_ff, _, _ = selectBestFrame(calstars, img_width=1280, img_height=720, min_stars=10, max_stars=200)
    assert best_ff != "FF_cloudy_sparse.fits"


def testLoadCatalogStarsUsesFractionalYearsFromJ2000(tmp_path, monkeypatch):
    """ loadCatalogStars must forward fractional years_from_J2000 rather than truncating to whole days. """

    from RMS.Formats import StarCatalog

    seen = {}

    def _fakeRead(path, fname, lim_mag=None, mag_band_ratios=None, years_from_J2000=0.0):
        seen["years"] = years_from_J2000
        return (np.zeros((1, 3)), None, None)

    monkeypatch.setattr(StarCatalog, "readStarCatalog", _fakeRead)

    cfg = SimpleNamespace(
        star_catalog_path=str(tmp_path),
        rms_root_dir=str(tmp_path),
        star_catalog_file="cat.bin",
        star_catalog_band_ratios=[0, 1, 0, 0, 0, 0, 0],
    )
    jd = 2451545.0 + 365.25*26 + 0.5
    loadCatalogStars(cfg, 6.5, jd=jd)
    assert seen["years"] == pytest.approx(26.0 + 0.5/365.25)


@pytest.mark.parametrize("bad_coords", [
    [],
    np.empty((0, 8)),
    np.array([10.0, 20.0, 30.0]),
    np.array([[10.0], [20.0]]),
])
def testAlignPlateparHandlesEmptyOrInvalidCoordsAndMsOverflow(bad_coords):
    """ alignPlatepar must handle ms rounding to 1000000 us and invalid/empty coords without crashing. """

    pp = Platepar()
    cfg = SimpleNamespace(catalog_mag_limit=6.5)
    orig_cfg = copy.deepcopy(cfg)

    # Millisecond value 999.9999 previously triggered ValueError: microsecond must be in 0..999999
    calstars_time = (2026, 11, 27, 4, 0, 0, 999.9999)
    res_pp, res_lm = NNalign.alignPlatepar(cfg, pp, calstars_time, bad_coords)

    assert res_pp is pp
    assert res_lm == pytest.approx(6.5)
    assert cfg.catalog_mag_limit == orig_cfg.catalog_mag_limit
