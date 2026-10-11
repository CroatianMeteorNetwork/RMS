""" Rolling shutter correction: exact (row x line period) when the camera reports its line period,
otherwise from the config, whose defaults are the stock IMX291 at 25 fps and 720p.

Run with: python -m pytest Tests/TestRollingShutter.py
"""

from __future__ import print_function, division, absolute_import

import types

import pytest

from RMS.ConfigReader import Config
from RMS.Formats.FFStruct import FFStruct
from RMS.Routines.RollingShutterCorrection import correctRollingShutterTemporal, lineTime, scanTime


def correctedTime(frame, y, n_rows, fps, scan):
    return correctRollingShutterTemporal(frame, y, n_rows, fps=fps, scan_rate=scan)/fps


def testDefaultsAreStockImx291():
    config = Config()
    assert config.deinterlace_order == -1
    assert (config.width, config.height, config.fps) == (1280, 720, 25.0)
    # 1080 sensor rows at HMAX 4400 / 148.5 MHz, downscaled to the 720 image rows
    assert config.scan_rate == pytest.approx(1080*4400/148.5e6, abs=1e-12)

    # Bottom row of a 720p image: one full scan after row 0
    t = correctedTime(10, 720, 720, config.fps, scanTime(config, 720)) - 10/config.fps
    assert t == pytest.approx(0.032, abs=1e-12)


def testReportedLinePeriodIsExact():
    config = Config()
    ff = FFStruct()
    ff.trowns = 26666                       # IMX662 (UST002)
    handle = types.SimpleNamespace(ff=ff)

    for y in (0.0, 1.0, 540.25, 1079.0):
        t = correctedTime(0, y, 1080, 25.0, scanTime(config, 1080, lineTime(handle)))
        assert t == pytest.approx(y*26666e-9, abs=1e-12)

    # The camera's report wins over a config that says global shutter or has another scan rate
    config.deinterlace_order = -2
    config.scan_rate = 0.03197
    assert scanTime(config, 1080, lineTime(handle)) == pytest.approx(1080*26666e-9, abs=1e-15)


def testVideoHandleLinePeriod():
    # A video handle carries the SEI line period itself; its FF stand-in has none
    handle = types.SimpleNamespace(ff=FFStruct(), t_row_ns=29629)
    assert lineTime(handle) == 29629
    assert lineTime(types.SimpleNamespace(ff=FFStruct())) is None
    assert lineTime(None) is None


def testConfigFallback():
    config = Config()
    assert scanTime(config, 720, None) == config.scan_rate

    # Global shutter or interlaced: no correction
    for order in (-2, 0, 1):
        config.deinterlace_order = order
        assert scanTime(config, 720, None) is None
