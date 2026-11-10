"""
Tests of the joint fit of the night in the monitor (MonitorNightReport.nightJointFit): a platepar which the
automatic calibration installed during a night is refitted on the whole night at its report, installed only if the
fit meets its quality control, once per night; nothing happens without an install during the night. The mask
of the night is used for the clear chunks and the quality control.

The fit itself is replaced by a stand-in whose outcome each test sets.
"""

from __future__ import print_function, division, absolute_import

import os

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.MonitorNightReport as mnr
import RMS.Astrometry.AutoCalibration as ac
import RMS.Astrometry.NightCalibration as nc
from RMS.Formats.Platepar import Platepar


JD0 = 2461000.0


def _platepar(value):
    """ A platepar told apart by its mag_lev. """

    pp = Platepar()
    pp.X_res, pp.Y_res = 1280, 720
    pp.mag_lev = value

    return pp


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    """ An output directory with an installed automatic platepar and a night of chunks, and a stand-in fit.

    Return:
        (config, output_dir, plan): plan sets the outcome of the fit ('qc_ok', 'value') and records the calls.
    """

    config = cr.Config()
    config.monitor_auto_calibration = True
    output_dir = str(tmp_path)
    ac.writePlatepar(_platepar(1.0), ac.autoPlateparPath(output_dir, config))

    chunks = [('FF_{:04d}'.format(m), JD0 + m/1440.0, np.zeros((50, 8))) for m in range(300)]
    plan = {'qc_ok': True, 'value': 2.0, 'calls': []}

    def _fit(clear, start, config, night_dir, night_hours, **kwargs):
        plan['calls'].append((len(clear), start.mag_lev, round(night_hours, 2)))
        plan['fit_mask'] = kwargs.get('mask')
        report = {'qc': {'ok': plan['qc_ok'], 'failed': [] if plan['qc_ok'] else ['converged'], 'checks': []}}
        return _platepar(plan['value']), report

    monkeypatch.setattr(mnr, 'scanNights', lambda output_dir, config: {'20261105': {'r1': {}, 'r2': {}}})
    monkeypatch.setattr(ac, 'readChunks', lambda path, config: (chunks[:150] if path.endswith('r1')
                                                                else chunks[150:], 256))
    def _clear(chunks, *args, **kwargs):
        plan['clear_mask'] = kwargs.get('mask')
        return list(chunks)

    monkeypatch.setattr(ac, 'clearChunks', _clear)
    monkeypatch.setattr(nc, 'jointFitNight', _fit)

    return config, output_dir, plan


def _install(output_dir, jd):
    """ Record an install of the automatic calibration at the given time. """

    state = ac.readState(output_dir)
    state['installs'].append({'jd': jd, 'path': 'x', 'reason': 'first', 'dry_run': False})
    ac.writeState(output_dir, state)


def _installed(config, output_dir):
    """ The value of the installed automatic platepar. """

    return ac._platepar(ac.autoPlateparPath(output_dir, config), False).mag_lev


def test_refit_installed_once_per_night(monitor):
    """ After an install during the night the platepar is refitted on all chunks of the night; a fit which meets
        its quality control is installed, the version it replaces is returned (for the backfill), and the night
        isn't refitted again. """

    config, output_dir, plan = monitor
    _install(output_dir, JD0 + 0.05)

    replaced = mnr.nightJointFit(output_dir, '20261105', config)
    assert replaced is not None
    assert _installed(config, output_dir) == 2.0
    assert plan['calls'] == [(300, 1.0, round(299/60.0, 2))]
    assert ac.readState(output_dir)['night_joint_fits']['20261105']['verdict'] == 'OK'

    # Once per night
    assert mnr.nightJointFit(output_dir, '20261105', config) is None
    assert len(plan['calls']) == 1


def test_refit_failing_quality_control_not_installed(monitor):
    """ A fit which fails its quality control is not installed, nothing is returned for the backfill. """

    config, output_dir, plan = monitor
    plan['qc_ok'] = False
    _install(output_dir, JD0 + 0.05)

    assert mnr.nightJointFit(output_dir, '20261105', config) is None
    assert _installed(config, output_dir) == 1.0
    assert ac.readState(output_dir)['night_joint_fits']['20261105']['verdict'].startswith('FAIL')


def test_no_refit_without_an_install_during_the_night(monitor):
    """ Without an install during the night (or only one of a dry run, or one of another night) nothing is
        fitted. """

    config, output_dir, plan = monitor
    _install(output_dir, JD0 - 1.0)
    state = ac.readState(output_dir)
    state['installs'].append({'jd': JD0 + 0.05, 'path': 'x', 'reason': 'first', 'dry_run': True})
    ac.writeState(output_dir, state)

    assert mnr.nightJointFit(output_dir, '20261105', config) is None
    assert not plan['calls']


def test_dry_run_installs_a_candidate(monitor):
    """ In a dry run the refit is saved as a candidate and the installed platepar stays. """

    config, output_dir, plan = monitor
    config.monitor_auto_calibration_dry_run = True
    _install(output_dir, JD0 + 0.05)

    assert mnr.nightJointFit(output_dir, '20261105', config) is None
    assert _installed(config, output_dir) == 1.0
    assert os.path.isfile(os.path.join(output_dir, 'auto_candidate_' + config.platepar_name))


def test_mask_of_the_night_is_used(monitor, tmp_path, monkeypatch):
    """ The mask the files of the night were processed with is used to pick the clear chunks and in the quality
        control of the fit; without a mask there is none. """

    config, output_dir, plan = monitor
    _install(output_dir, JD0 + 0.05)

    # Without a mask file
    mnr.nightJointFit(output_dir, '20261105', config)
    assert plan['clear_mask'] is None
    assert plan['fit_mask'] is None

    # With the mask of the files of the night
    mask_path = str(tmp_path/'mask.bmp')
    open(mask_path, 'w').close()
    mask_img = np.ones((720, 1280), dtype=np.uint8)
    monkeypatch.setattr(mnr, 'scanNights', lambda output_dir, config: {'20261106': {'r1': {'mask_path': mask_path},
                                                                                    'r2': {}}})
    monkeypatch.setattr(mnr, 'loadMask', lambda path: type('Mask', (), {'img': mask_img})())

    mnr.nightJointFit(output_dir, '20261106', config)
    assert plan['clear_mask'] is mask_img
    assert plan['fit_mask'] is mask_img
