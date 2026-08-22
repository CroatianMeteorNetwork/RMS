""" Tests for pair provenance: only automatically matched pairs may be filtered away
    (RMS.Astrometry.StarClasses, RMS.Astrometry.StarFilters). """

from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from RMS.Astrometry.StarClasses import CatalogStar, PairedStars
from RMS.Astrometry.StarFilters import filterBlendedStars

from Tests.TestStarFilters import buildCatalog, buildPairedStars, makePlatepar


def test_pairs_are_hand_picked_unless_marked():
    # The default matters: anything whose origin is unknown (an old saved state, a loaded
    # pairs file from before provenance was recorded) must be treated as hand-picked
    paired = PairedStars()
    paired.addPair(10.0, 20.0, 3.0, 1000.0, CatalogStar(1.0, 2.0, 5.0))
    paired.addPair(30.0, 40.0, 3.0, 1000.0, CatalogStar(3.0, 4.0, 5.0), auto=True)

    assert paired.autoFlags() == [False, True]


def test_filters_carry_provenance_through_a_rebuild():
    # The filters rebuild the pair list from scratch - the flag has to survive that, or a
    # hand-picked pair silently becomes fair game for the next filter
    pp = makePlatepar()
    positions = [(300, 200), (500, 250), (700, 300), (900, 350), (400, 500)]
    paired = buildPairedStars(pp, positions, fwhm=3.0)

    # Mark every other pair as automatic
    for i, entry in enumerate(paired.paired_stars):
        entry[7] = (i%2 == 0)
    expected = [i%2 == 0 for i in range(len(positions))]

    # A blend on the last star, so the filter removes something and rebuilds the list
    catalog = buildCatalog(pp, positions + [(404.5, 500)])
    filtered, removed = filterBlendedStars(paired, catalog, pp, pp.JD, 6.0)

    # The blend sits on the last pair, so exactly that one goes and the surviving
    # flags must still line up with the pairs they belong to
    assert removed == 1
    assert filtered.autoFlags() == expected[:-1]


def test_auto_flags_track_the_entries():
    paired = PairedStars()
    for i in range(5):
        paired.addPair(float(i), float(i), 3.0, 1000.0, CatalogStar(float(i), 1.0, 5.0),
                       auto=(i < 3))

    assert paired.autoFlags() == [True, True, True, False, False]

    paired.removeClosestPair(0.0, 0.0)
    assert paired.autoFlags() == [True, True, False, False]
