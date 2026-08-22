""" Tests for the fit progress/abort hook (RMS.Formats.Platepar). """

from __future__ import absolute_import, division, print_function

import time

import numpy as np
import pytest

from RMS.Formats.Platepar import (PROGRESS_CALLBACK_INTERVAL, _lstsqFit,
    _withProgressCallback)


class Aborted(Exception):
    """ Stands in for SkyFit's OperationCancelled. """

    @staticmethod
    def raiseIt():
        raise Aborted()


def test_callback_is_throttled():
    # The optimizers evaluate their cost thousands of times a second - the hook must not run
    # on every evaluation
    calls = []
    wrapped = _withProgressCallback(lambda x: x, lambda: calls.append(1), interval=0.05)

    t_end = time.time() + 0.2
    while time.time() < t_end:
        wrapped(1)

    # ~4 windows of 0.05 s; allow slack for scheduling
    assert 1 <= len(calls) <= 8


def test_wrapped_function_still_returns_its_value():
    wrapped = _withProgressCallback(lambda x, k=0: x + k, lambda: None, interval=0.0)
    assert wrapped(2, k=3) == 5


def test_raising_from_the_callback_aborts_a_running_fit():
    # This is how Stop reaches a fit already inside scipy: the exception has to travel out
    # through the optimizer, not be swallowed by it. The residual is slowed down so the fit
    # outlives one throttle window - the only fits worth aborting are the slow ones anyway
    evaluations = {'n': 0}

    def residuals(params, target):
        evaluations['n'] += 1
        time.sleep(PROGRESS_CALLBACK_INTERVAL/2)
        return params - target

    with pytest.raises(Aborted):
        _lstsqFit(residuals, np.zeros(20), (np.arange(20, dtype=float),),
                  progress_callback=Aborted.raiseIt)

    # Aborted while fitting, not after running to convergence
    assert evaluations['n'] < 20


def test_fit_without_a_callback_is_unchanged():
    def residuals(params, target):
        return params - target

    target = np.array([1.0, -2.0, 3.0])
    plain = _lstsqFit(residuals, np.zeros(3), (target,))
    hooked = _lstsqFit(residuals, np.zeros(3), (target,), progress_callback=lambda: None)

    np.testing.assert_allclose(plain.x, hooked.x)
    np.testing.assert_allclose(plain.x, target, atol=1e-6)


def test_default_interval_is_sane():
    # Slow enough to be free, fast enough that a Stop click feels immediate
    assert 0.01 <= PROGRESS_CALLBACK_INTERVAL <= 0.5
