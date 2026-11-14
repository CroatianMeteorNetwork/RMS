""" Tests of stopping the FF compression while the capture keeps running, e.g. when the next night is started
    right away at a polar night split. """

from __future__ import print_function, division, absolute_import

import os
import time
import ctypes
import multiprocessing

import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Compression import Compressor, blockToCompress, stopAction
from RMS.Misc import frameBufferShape


REPO_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config')


def test_block_choice_without_stop():
    """ Normal operation: whichever block is ready, the first buffer first. """

    assert blockToCompress(100.0, 0.0) == 1
    assert blockToCompress(0.0, 100.0) == 2
    assert blockToCompress(100.0, 90.0) == 1
    assert blockToCompress(0.0, -1.0) is None


def test_block_choice_when_stopping():
    """ Stopping: only the blocks which started before the stop request, the oldest first. """

    assert blockToCompress(90.0, 0.0, exit_time=100.0) == 1
    assert blockToCompress(110.0, 90.0, exit_time=100.0) == 2
    assert blockToCompress(95.0, 90.0, exit_time=100.0) == 2
    assert blockToCompress(110.0, 0.0, exit_time=100.0) is None


def test_stop_action():
    """ A block from before the stop is compressed, a newer one is left for the next compressor, and the
        compression waits only while a block is being compressed. """

    # A waiting block captured before the stop
    assert stopAction(90.0, 0.0, 100.0) == 'compress'

    # Live capture goes on: one buffer is being filled, the other holds a block from after the stop
    assert stopAction(0.0, 110.0, 100.0) == 'exit'

    # Nothing waiting
    assert stopAction(0.0, 0.0, 100.0) == 'exit'

    # A block is still being compressed
    assert stopAction(-1.0, 0.0, 100.0) == 'wait'


@pytest.fixture
def smallConfig(tmp_path):
    """ A config with a small frame size, so a block compresses in a moment. """

    config = cr.parse(REPO_CONFIG)
    config.width = 64
    config.height = 48
    config.data_dir = str(tmp_path)
    config.live_jpg = False
    config.enable_fireball_detection = False

    return config


def test_stop_during_live_capture(smallConfig, tmp_path):
    """ The compressor is stopped while the capture is running: the block waiting from before the stop is
        compressed and the compressor exits at once, leaving the block captured after the stop for the next
        compressor. It used to wait for both buffers to be empty, which does not happen while the capture
        runs, and gave up after 30 s without compressing the waiting block. """

    buffer_len = int(np.prod(frameBufferShape(smallConfig)))
    array1 = multiprocessing.Array(ctypes.c_uint8, buffer_len)
    array2 = multiprocessing.Array(ctypes.c_uint8, buffer_len)
    start_time1 = multiprocessing.Value('d', 0.0, lock=False)
    start_time2 = multiprocessing.Value('d', 0.0, lock=False)

    out_dir = tmp_path/"night"
    out_dir.mkdir()

    compressor = Compressor(str(out_dir), array1, start_time1, array2, start_time2, smallConfig)

    # Both buffers hold a block when the stop comes: one captured before the stop, and one after it (the
    #   capture goes on). The stop is requested before the compressor runs, so it finds both waiting
    t_stop = time.time()
    start_time1.value = t_stop - 20.0
    start_time2.value = t_stop + 10.0
    compressor.exit_time.value = t_stop
    compressor.exit.set()

    compressor.start()
    compressor.stop()
    stop_duration = time.time() - t_stop

    ff_files = [f for f in os.listdir(str(out_dir)) if f.startswith('FF_')]

    # The waiting block was compressed, the newer one was left alone, and the stop did not time out
    assert len(ff_files) == 1
    assert start_time2.value == t_stop + 10.0
    assert stop_duration < 15
