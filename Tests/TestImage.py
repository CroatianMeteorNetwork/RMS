""" Tests for coordinate filtering in RMS.Routines.Image. """

from __future__ import print_function, division, absolute_import

import pytest

np = pytest.importorskip("numpy")

from RMS.Routines.Image import CoordinateFilter


def testFilterCoordinatesAcceptsList():
    """ A plain list of (x, y) pairs must work, not only an ndarray. """

    coord_filter = CoordinateFilter((100, 200), None, 10)

    coords = [[5, 50], [50, 50], [150, 5], [150, 50], [195, 95]]
    filtered, flags = coord_filter.filterCoordinates(coords)

    assert list(flags) == [False, True, False, True, False]
    assert np.array_equal(filtered, np.array([[50, 50], [150, 50]]))


def testFilterCoordinatesArrayAndListAgree():
    """ List and ndarray input give identical results. """

    coord_filter = CoordinateFilter((100, 200), None, 10)

    coords = [[20.5, 30.2], [0.0, 0.0], [199.0, 99.0], [100.0, 50.0]]
    filtered_list, flags_list = coord_filter.filterCoordinates(coords)
    filtered_arr, flags_arr = coord_filter.filterCoordinates(np.array(coords))

    assert np.array_equal(flags_list, flags_arr)
    assert np.array_equal(filtered_list, filtered_arr)


def testFilterCoordinatesEmpty():
    """ Empty input returns empty output without raising. """

    coord_filter = CoordinateFilter((100, 200), None, 10)

    filtered, flags = coord_filter.filterCoordinates([])

    assert filtered.size == 0
    assert flags.size == 0


def testFilterCoordinatesMask():
    """ Coordinates inside the masked (zero) region are rejected. """

    mask_img = np.full((100, 200), 255, dtype=np.uint8)
    mask_img[40:60, 90:110] = 0

    coord_filter = CoordinateFilter((100, 200), mask_img, 10)

    filtered, flags = coord_filter.filterCoordinates([[100, 50], [20, 20]])

    assert list(flags) == [False, True]
    assert np.array_equal(filtered, np.array([[20, 20]]))
