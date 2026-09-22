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
