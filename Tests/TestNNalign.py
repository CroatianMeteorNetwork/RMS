""" Tests for the input guards of the nearest-neighbour platepar alignment. """

from __future__ import print_function, division, absolute_import

import types

import pytest

np = pytest.importorskip("numpy")

from RMS.Astrometry.NNalign import alignPlatepar
from RMS.Formats.Platepar import Platepar


def _makeConfig():
    """ Minimal config with only the field alignPlatepar reads before the input guards. """

    return types.SimpleNamespace(catalog_mag_limit=5.5)


@pytest.mark.parametrize("coords", [
    [],                       # empty list
    np.zeros((0, 8)),         # empty 2D array
    np.array([1.0, 2.0]),     # a single 1D entry instead of a table
])
def testAlignPlateparRejectsDegenerateStarInput(coords):
    """ Empty or 1-D star input returns the original platepar instead of raising IndexError. """

    platepar = Platepar()
    config = _makeConfig()

    platepar_out, lim_mag = alignPlatepar(config, platepar, (2020, 1, 1, 0, 0, 0, 0), coords)

    assert platepar_out is platepar
    assert lim_mag == config.catalog_mag_limit


def testAlignPlateparDoesNotModifyCallerConfig(monkeypatch):
    """ The LM inferred from calibrated photometry is used internally but the caller's config is kept. """

    import RMS.Astrometry.NNalign as NNalign

    platepar = Platepar()
    platepar.mag_lev = 12.0
    platepar.mag_lev_stddev = 0.1

    config = types.SimpleNamespace(catalog_mag_limit=5.5, star_catalog_path='.', star_catalog_file='x',
                                   star_catalog_band_ratios=None)

    # Stop right after the catalog LM was chosen, recording the LM the catalog was requested with
    requested = {}

    class _Stop(Exception):
        pass

    def readStarCatalog(*args, **kwargs):
        requested['lim_mag'] = kwargs['lim_mag']
        raise _Stop()

    monkeypatch.setattr(NNalign.StarCatalog, 'readStarCatalog', readStarCatalog)

    # CALSTARS-like rows (y, x, intensity)
    coords = np.column_stack([np.linspace(10, 500, 50), np.linspace(10, 700, 50), np.full(50, 100.0)])

    with pytest.raises(_Stop):
        alignPlatepar(config, platepar, (2020, 1, 1, 0, 0, 0, 0), coords)

    assert requested['lim_mag'] != 5.5
    assert config.catalog_mag_limit == 5.5
