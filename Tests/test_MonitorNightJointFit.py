"""
Tests of the joint fit of the night in the monitor (MonitorNightReport.nightJointFit): a platepar which the
automatic calibration installed during a night is refitted on the whole night at its report, installed only if the
fit meets its quality control, once per night; nothing happens without an install during the night. The mask
of the night is used for the clear chunks and the quality control. The fit waits for the end of the night, takes
the flat from the processing of the files, and leaves an old night alone when a newer platepar was installed. The
files calibrated with the replaced platepar get the refit even where their own judgement passed it.

The fit itself is replaced by a stand-in whose outcome each test sets.
"""

from __future__ import print_function, division, absolute_import

import datetime
import json
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
        plan['fit_use_flat'] = config.use_flat
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

    # The night is over unless a test says otherwise
    night_end = datetime.datetime(2026, 11, 6, 12, 0, 0)
    monkeypatch.setattr(mnr, 'nightBoundsFromName', lambda config, name: (night_end - datetime.timedelta(hours=12),
                                                                          plan.get('night_end', night_end)))

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


@pytest.mark.parametrize('use_flat', [False, True])
def test_vignetting_refitted_outside_the_joint_holdout(monitor, monkeypatch, use_flat):
    """ With monitor_auto_calibration_fit_vignetting the vignetting of the start is refitted before the joint fit,
        on chunks which the joint fit doesn't hold out, and the joint fit starts from the refitted platepar; never
        with a flat. """

    config, output_dir, plan = monitor
    config.monitor_auto_calibration_fit_vignetting = True
    config.use_flat = use_flat
    _install(output_dir, JD0 + 0.05)

    refits = []

    def _refit(platepar, chunks, catalog_stars, config, **kwargs):
        refits.append(set(chunk[0] for chunk in chunks))
        return _platepar(5.0), {'kept': True, 'reason': 'stand-in'}

    monkeypatch.setattr(ac, 'refitVignetting', _refit)
    monkeypatch.setattr(ac, '_loadCatalog', lambda config, jd: np.zeros((0, 3)))

    mnr.nightJointFit(output_dir, '20261105', config)

    if use_flat:
        assert not refits
        assert plan['calls'][0][1] == 1.0
    else:
        clear = sorted(plan_chunk for plan_chunk in ('FF_{:04d}'.format(m) for m in range(300)))
        held = nc.nightHoldout([(name, 0, None) for name in clear])
        assert len(refits) == 1
        assert refits[0] and not (refits[0] & held)
        assert refits[0] | held == set(clear)
        assert plan['calls'][0][1] == 5.0


def test_not_before_the_end_of_the_night(monitor):
    """ A report during the night (a partial report) doesn't fit the night and doesn't use up its fit. """

    config, output_dir, plan = monitor
    _install(output_dir, JD0 + 0.05)
    plan['night_end'] = datetime.datetime.utcnow() + datetime.timedelta(hours=3)

    assert mnr.nightJointFit(output_dir, '20261105', config) is None
    assert not plan['calls']
    assert '20261105' not in ac.readState(output_dir).get('night_joint_fits', {})

    # After the end of the night it is fitted
    plan['night_end'] = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
    assert mnr.nightJointFit(output_dir, '20261105', config) is not None
    assert len(plan['calls']) == 1


def test_old_night_does_not_replace_a_newer_platepar(monitor):
    """ An old night reported after a newer platepar was installed (e.g. a backlog) isn't fitted, and the newer
        platepar stays. """

    config, output_dir, plan = monitor
    _install(output_dir, JD0 + 0.05)
    _install(output_dir, JD0 + 2.0)

    assert mnr.nightJointFit(output_dir, '20261105', config) is None
    assert not plan['calls']
    assert _installed(config, output_dir) == 1.0
    assert ac.readState(output_dir)['night_joint_fits']['20261105']['verdict'].startswith('skipped')


@pytest.mark.parametrize('flat_applied', [False, True])
def test_flat_of_the_processing(monitor, monkeypatch, flat_applied):
    """ The flat comes from how the files of the night were processed, not from the configuration of the report:
        with a flat no vignetting is refitted, and the fit gets the flat setting of the files. """

    config, output_dir, plan = monitor
    config.use_flat = not flat_applied
    config.monitor_auto_calibration_fit_vignetting = True
    _install(output_dir, JD0 + 0.05)
    monkeypatch.setattr(mnr, 'scanNights', lambda output_dir, config: {
        '20261105': {'r1': {'flat_applied': flat_applied}, 'r2': {'flat_applied': flat_applied}}})

    refits = []
    monkeypatch.setattr(ac, 'refitVignetting', lambda platepar, chunks, *args, **kwargs:
                        (refits.append(1) or platepar, {'kept': False, 'reason': 'stand-in'})[0:2])
    monkeypatch.setattr(ac, '_loadCatalog', lambda config, jd: np.zeros((0, 3)))

    mnr.nightJointFit(output_dir, '20261105', config)
    assert plan['fit_use_flat'] is flat_applied
    assert bool(refits) is (not flat_applied)

    # The configuration of the caller is not changed
    assert config.use_flat is (not flat_applied)


def test_backfill_refines_files_whose_judgement_passed(tmp_path):
    """ The files calibrated with a replaced platepar which their own judgement passed keep it, unless the
        installed platepar refines it (the joint fit of the night). """

    from RMS.MonitorProcessFrameInterface import _needsCalibration

    results_path = str(tmp_path)
    with open(os.path.join(results_path, ac.FILE_RESULT_NAME), 'w') as f:
        json.dump({'current': {'ok': True}}, f)
    info = {'calibrated': True, 'platepar_mtime': 100.0}

    assert not _needsCalibration(info, results_path, 200.0, [100.0], ac.FILE_RESULT_NAME)
    assert _needsCalibration(info, results_path, 200.0, [100.0], ac.FILE_RESULT_NAME, refine_replaced=True)

    # A version which wasn't replaced stays either way
    assert not _needsCalibration(info, results_path, 200.0, [50.0], ac.FILE_RESULT_NAME, refine_replaced=True)


def test_replaced_version_kept_until_no_file_has_it(monitor, monkeypatch):
    """ The version an installed refit replaced is kept in the state, so a backfill which stopped or failed on a
        file is made again at the next report; once no file of the night or of the nights not reported yet has it,
        it is forgotten. """

    config, output_dir, plan = monitor
    _install(output_dir, JD0 + 0.05)

    replaced = mnr.nightJointFit(output_dir, '20261105', config)
    assert ac.readState(output_dir)['joint_fit_replaced'] == [replaced]

    # A file still has the replaced version (e.g. its recalibration failed): the backfill gets it again
    files = {'r1': {'calibrated': True, 'platepar_mtime': replaced}, 'r2': {'calibrated': True,
                                                                          'platepar_mtime': replaced + 1}}
    monkeypatch.setattr(mnr, 'scanNights', lambda output_dir, config: {'20261105': files})
    assert mnr.refinementBackfillPending(output_dir, config, '20261105') == [replaced]
    assert ac.readState(output_dir)['joint_fit_replaced'] == [replaced]

    # A reported night doesn't count
    monkeypatch.setattr(mnr, 'scanNights', lambda output_dir, config: {'20261105': {}, '20261104': files})
    monkeypatch.setattr(mnr, 'readReportStates', lambda output_dir: {
        'nights': {'20261104': {'reported_at': '2026-11-05T12:00:00'}}})
    assert mnr.refinementBackfillPending(output_dir, config, '20261105') == []

    # No file has it any more: forgotten
    assert ac.readState(output_dir)['joint_fit_replaced'] == []
