""" Tests for CoordinateFilter list input and Platepar.write FOV exception narrowing. """

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest

from RMS.Formats.Platepar import Platepar
from RMS.Routines.Image import CoordinateFilter


def testCoordinateFilterAcceptsListAndArrayInput():
    """ CoordinateFilter.filterCoordinates must accept plain Python lists as well as ndarrays. """

    mask = np.full((100, 100), 255, dtype=np.uint8)
    mask[40:60, 40:60] = 0
    cf = CoordinateFilter(image_shape=(100, 100), mask=mask, border=5)

    coords_list = [
        [2.0, 50.0],    # outside border
        [50.0, 50.0],   # inside masked region
        [20.0, 20.0],   # valid
        [80.0, 80.0],   # valid
    ]

    filtered_list, flags_list = cf.filterCoordinates(coords_list)
    filtered_arr, flags_arr = cf.filterCoordinates(np.array(coords_list))

    assert np.array_equal(flags_list, [False, False, True, True])
    assert np.array_equal(flags_list, flags_arr)
    assert np.allclose(filtered_list, [[20.0, 20.0], [80.0, 80.0]])
    assert np.allclose(filtered_list, filtered_arr)


def testPlateparWriteNarrowsComputeFOVExceptions(tmp_path, monkeypatch):
    """ Platepar.write keeps old FOV on expected math/value errors but propagates unexpected errors. """

    import RMS.Astrometry.ApplyAstrometry as aa

    pp = Platepar()
    pp.fov_h, pp.fov_v = 55.0, 31.0

    # Expected ValueError: keeps existing FOV and succeeds
    monkeypatch.setattr(aa, "computeFOVSize", lambda p: (_ for _ in ()).throw(ValueError("uncalibrated")))
    out_ok = str(tmp_path/"ok.cal")
    pp.write(out_ok)
    assert pp.fov_h == pytest.approx(55.0)
    assert pp.fov_v == pytest.approx(31.0)

    # Unexpected RuntimeError: must not be silently swallowed
    monkeypatch.setattr(aa, "computeFOVSize", lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        pp.write(str(tmp_path/"fail.cal"))
