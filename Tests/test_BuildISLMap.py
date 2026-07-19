""" Tests for the ISL map prototype (Utils.BuildISLMap). """

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from Utils.BuildISLMap import buildISLMap, evalISL, equatorialToGalactic


def test_galactic_conversion_landmarks():
    l, b = equatorialToGalactic([266.405, 192.8595], [-28.936, 27.1284])
    assert abs(l[0]) < 0.1 or abs(l[0] - 360) < 0.1     # Sgr A* at l ~ 0
    assert abs(b[0]) < 0.1
    assert b[1] == pytest.approx(90.0, abs=0.01)        # NGP at b = 90


def test_map_brightness_scales_with_star_density():
    rng = np.random.default_rng(5)
    # dense strip at b ~ 0, sparse elsewhere - build in galactic coords via inverse
    # trick: just use equatorial positions ON the galactic plane vs pole
    n = 4000
    # plane sample: near Sgr A* (l~0, b~0)
    ra_p = 266.4 + rng.uniform(-3, 3, n)
    dec_p = -28.9 + rng.uniform(-3, 3, n)
    # pole sample: near NGP, 10x fewer stars
    ra_o = 192.86 + rng.uniform(-3, 3, n//10)
    dec_o = 27.13 + rng.uniform(-3, 3, n//10)
    cat = np.column_stack([np.concatenate([ra_p, ra_o]),
                           np.concatenate([dec_p, dec_o]),
                           np.full(n + n//10, 8.0)])
    m = buildISLMap(cat, grid_step=2.0)
    # evaluate at the actual sample centroids (l wraps wildly near the pole)
    lp, bp = equatorialToGalactic(ra_p, dec_p)
    lo, bo = equatorialToGalactic(ra_o, dec_o)
    sb_plane = float(evalISL(m, np.median(lp), np.median(bp)))
    sb_pole = float(evalISL(m, np.median(lo), np.median(bo)))
    assert sb_plane < sb_pole    # brighter (smaller mag) where denser
    assert sb_pole - sb_plane == pytest.approx(2.5, abs=0.8)   # ~10x flux ratio


def test_flux_conservation_per_cell():
    # one star of known mag in one cell: surface brightness must equal the star's
    # flux times the completion factor over the cell area
    cat = np.array([[266.405, -28.936, 5.0]])
    m = buildISLMap(cat, grid_step=1.0, completion_factor=1.0)
    sb = float(evalISL(m, 0.0, 0.0))
    cell_arcsec2 = 3600.0**2*np.cos(np.radians(0.5))
    expected = 5.0 + 2.5*np.log10(cell_arcsec2)
    assert sb == pytest.approx(expected, abs=0.02)
