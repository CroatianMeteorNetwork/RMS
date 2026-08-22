""" Tests for SkyFit2's adaptive gate factor tuning (Utils.SkyFit2). """

from __future__ import absolute_import, division, print_function

import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

PlateTool = pytest.importorskip("Utils.SkyFit2").PlateTool


def test_flat_curve_goes_deep_but_not_past_the_precision_it_measured():
    # Measured US005E curve: precision is flat well past the 3.0 default, so the sweep is
    # free to go deeper - but only as far as the precision holds up
    results = [(20.0, 13, 0.97), (10.0, 27, 0.97), (3.0, 142, 0.97), (2.5, 178, 0.96),
               (2.0, 244, 0.93)]
    assert PlateTool._selectGateFactor(results) == 2.5


def test_flooding_camera_is_pulled_back_to_its_knee():
    # Measured CN0008 mid-night: fixed-pattern peaks flood the frame below ~12, so the
    # deepest gate within the slack is the knee, NOT the deepest swept factor
    results = [(20.0, 37, 0.96), (15.0, 45, 0.96), (12.0, 58, 0.93), (10.0, 67, 0.85),
               (8.0, 98, 0.68)]
    assert PlateTool._selectGateFactor(results) == 15.0


def test_selection_can_exceed_the_default_factor():
    # The whole point of the sweep: a camera whose usable gate is far above the 3.0 default
    # must be able to reach it
    results = [(20.0, 40, 0.96), (15.0, 49, 0.96), (12.0, 70, 0.90), (10.0, 86, 0.84)]
    assert PlateTool._selectGateFactor(results) > 3.0


def test_noise_dominated_sweep_is_rejected():
    # No factor is worth recommending when even the best one is mostly junk (bad platepar,
    # clouds, daylight)
    results = [(20.0, 10, 0.20), (10.0, 30, 0.30), (3.0, 500, 0.10)]
    assert PlateTool._selectGateFactor(results) is None


def test_empty_and_starless_sweeps_return_none():
    assert PlateTool._selectGateFactor([]) is None
    assert PlateTool._selectGateFactor([(20.0, 0, 0.0), (10.0, 0, 0.0)]) is None
