""" Tests for the beam-blended effective source catalog (RMS.EffectiveCatalog). """

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from RMS.EffectiveCatalog import buildEffectiveSources, beamRadiusArcsec


BEAM = 600.0   # arcsec, ~2.6 px at an RMS plate scale


def test_far_stars_stay_separate():
    src = buildEffectiveSources([10.0, 12.0], [0.0, 0.0], [5.0, 5.5], BEAM)
    assert len(src["mag"]) == 2


def test_close_pair_merges_with_summed_flux():
    # two stars 100 arcsec apart, well inside the beam
    src = buildEffectiveSources([10.0, 10.0 + 100/3600.0], [0.0, 0.0], [6.0, 6.0], BEAM)
    assert len(src["mag"]) == 1
    assert src["mag"][0] == pytest.approx(6.0 - 2.5*np.log10(2), abs=1e-6)
    assert src["n_members"][0] == 2
    assert src["mag_brightest"][0] == 6.0


def test_chain_does_not_daisy_merge():
    # A(bright) - B - C spaced so B is inside A's beam, C inside B's but outside A's:
    # local-maxima semantics demand TWO sources (A+B, and C alone), not one
    step = 0.9*BEAM/3600.0
    src = buildEffectiveSources([10.0, 10.0 + step, 10.0 + 2*step],
                                [0.0, 0.0, 0.0], [4.0, 6.0, 6.5], BEAM)
    assert len(src["mag"]) == 2
    assert src["n_members"].tolist() == [2, 1]


def test_brightest_first_assignment():
    # a faint star near one of two well-separated bright seeds joins the nearer seed
    src = buildEffectiveSources(
        [10.0, 10.0 + 900/3600.0, 10.0 + 180/3600.0],
        [0.0, 0.0, 0.0], [4.0, 4.5, 7.0], BEAM)
    assert len(src["mag"]) == 2
    k = int(np.argmin(src["mag_brightest"]))          # the mag 4.0 source
    assert src["n_members"][k] == 2


def test_flux_weighted_centroid():
    # equal-flux pair: source position lands midway
    ra2 = 10.0 + 200/3600.0
    src = buildEffectiveSources([10.0, ra2], [0.0, 0.0], [6.0, 6.0], BEAM)
    assert src["ra"][0] == pytest.approx((10.0 + ra2)/2, abs=1e-5)


def test_empty_input():
    src = buildEffectiveSources([], [], [], BEAM)
    assert len(src["mag"]) == 0


def test_beam_radius_from_fwhm():
    # FWHM 3 px at 229 arcsec/px -> sigma 1.27 px -> radius ~467 arcsec
    r = beamRadiusArcsec(3.0, 229.0)
    assert r == pytest.approx(1.6*(3.0/2.355)*229.0, rel=1e-6)


def test_dense_field_reduces_source_count():
    # random dense cluster merges into fewer sources; sparse field does not
    rng = np.random.default_rng(3)
    n = 300
    ra_dense = 10.0 + rng.uniform(0, 0.5, n)          # ~30 arcmin box
    dec_dense = rng.uniform(0, 0.5, n)
    mags = rng.uniform(5, 9, n)
    dense = buildEffectiveSources(ra_dense, dec_dense, mags, BEAM)
    sparse = buildEffectiveSources(10.0 + rng.uniform(0, 60, n), rng.uniform(0, 60, n),
                                   mags, BEAM)
    assert len(dense["mag"]) < 0.8*n
    assert len(sparse["mag"]) > 0.97*n
    # flux conservation: total flux is preserved by merging
    assert np.sum(10**(-0.4*dense["mag"])) == pytest.approx(np.sum(10**(-0.4*mags)), rel=1e-9)
