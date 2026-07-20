""" Tests for the nightly star-scoring product format (RMS.Formats.StarScoring). """

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from RMS.Formats.StarScoring import saveStarScoring, loadStarScoring, SCHEMA_VERSION


def test_roundtrip(tmp_path):
    frames = dict(
        frame_names=["FF_T_1.fits", "FF_T_2.fits"],
        frame_time_unix=[1.5e9, 1.5e9 + 300],
        sun_alt=[-25.0, -14.0],
        moon_alt=[-10.0, 5.0],
        moon_phase=[20.0, 20.0],
        n_detected=[300, 420],
        in_flux_domain=[True, False],
    )
    stars = dict(
        star_frame=[0, 0, 1],
        star_x=[10.5, 600.25, 300.0],
        star_y=[20.0, 400.0, 100.0],
        star_mag=[4.5, 6.25, 5.0],
        star_p=[0.9, 0.25, 0.6],
        calstars_row=[3, -1, 17],
    )
    header = dict(stationID="TEST01", night="TEST01_20260101_000000_000000",
                  catalog_lim_mag=7.4)

    path = saveStarScoring(str(tmp_path), "TEST01_20260101_000000_000000",
                           header, frames, stars)
    h, f, s = loadStarScoring(path)

    assert h["schema_version"] == SCHEMA_VERSION
    assert h["stationID"] == "TEST01"
    assert h["catalog_lim_mag"] == pytest.approx(7.4)
    assert list(f["frame_names"]) == frames["frame_names"]
    assert f["in_flux_domain"].tolist() == [True, False]
    assert f["n_detected"].tolist() == [300, 420]
    assert s["calstars_row"].tolist() == [3, -1, 17]
    assert s["star_x"].tolist() == pytest.approx([10.5, 600.25, 300.0])
    # quantized columns keep working precision
    assert s["star_mag"].astype(float).tolist() == pytest.approx([4.5, 6.25, 5.0], abs=0.01)
    assert s["star_p"].astype(float).tolist() == pytest.approx([0.9, 0.25, 0.6], abs=0.005)


def test_unmatched_convention(tmp_path):
    # -1 calstars_row marks unmatched; matched can be derived without a separate column
    stars = dict(star_frame=[0], star_x=[1.0], star_y=[1.0], star_mag=[5.0],
                 star_p=[0.5], calstars_row=[-1])
    frames = dict(frame_names=["FF_T_1.fits"], frame_time_unix=[0.0], sun_alt=[-20.0],
                  moon_alt=[0.0], moon_phase=[0.0], n_detected=[0],
                  in_flux_domain=[True])
    path = saveStarScoring(str(tmp_path), "N", {}, frames, stars)
    _, _, s = loadStarScoring(path)
    assert (s["calstars_row"] >= 0).sum() == 0
