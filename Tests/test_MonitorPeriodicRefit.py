"""
Tests of the periodic refit of the lens distortion in the monitor (MonitorNightReport.periodicRefit): with
monitor_auto_calibration_periodic_refit, the report of a night refits the distortion of the platepar in use on the
recent nights every auto_calibration_refit_days days (NightCalibration.refitNights), installs the refit only if it
is clearly better, keeps the replaced version for the backfill, and records the time and the verdict in the state
of the automatic calibration. It is off by default, waits for the end of the night, never runs while a platepar is
on trial, saves only a candidate in a dry run, takes the flat from the processing of the files, and an error never
stops the report.

The joint fit and its decision are replaced by a stand-in whose outcome each test sets.
"""

from __future__ import print_function, division, absolute_import

import datetime
import os

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.MonitorNightReport as mnr
import RMS.Astrometry.AutoCalibration as ac
import RMS.Astrometry.NightCalibration as nc
from RMS.Formats.Platepar import Platepar


JD0 = 2461000.0

# Ten nights, the oldest first; the last one is being reported
NIGHTS = ['XX0001_202611{:02d}_220000_000000'.format(day) for day in range(1, 11)]
NIGHT = NIGHTS[-1]

# The time of the report, after the end of the night
NOW = datetime.datetime(2026, 11, 11, 15, 0, 0)

# The version (modification time) of the given platepar the files were calibrated with
GIVEN_MTIME = 1.7e9


def _platepar(value):
    """ A platepar told apart by its mag_lev. """

    pp = Platepar()
    pp.X_res, pp.Y_res = 1280, 720
    pp.mag_lev = value

    return pp


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    """ An output directory with ten nights of two processed files each, calibrated with the given platepar (its
        copy in every results directory), all but the last night reported, and a stand-in refit.

    Return:
        (config, output_dir, plan): plan sets the outcome of the refit ('better', 'raise', 'during') and the clear
            hours of the nights ('hours'), and records the calls.
    """

    config = cr.Config()
    config.stationID = 'XX0001'
    config.monitor_auto_calibration = True
    config.monitor_auto_calibration_periodic_refit = True
    output_dir = str(tmp_path)

    # Every night has two files, each with a copy of the platepar it was calibrated with
    files = {}
    for night in NIGHTS:
        for k in range(2):
            results_dir = os.path.join('results', night, 'f{:d}'.format(k))
            os.makedirs(os.path.join(output_dir, results_dir))
            path = os.path.join(output_dir, results_dir, config.platepar_name)
            ac.writePlatepar(_platepar(1.0), path)
            os.utime(path, (GIVEN_MTIME, GIVEN_MTIME))
            files.setdefault(night, {})[results_dir] = {'night': night, 'calibrated': True,
                                                        'platepar_mtime': GIVEN_MTIME, 'flat_applied': False,
                                                        'mask_path': None}

    # All nights but the last one were reported
    for night in NIGHTS[:-1]:
        mnr.updateReportState(output_dir, night_name=night, reported_at='2026-11-01T12:00:00')

    plan = {'files': files, 'hours': {}, 'better': True, 'calls': [], 'references': []}
    monkeypatch.setattr(mnr, 'scanNights', lambda output_dir, config: plan['files'])

    # The chunks of a file: half of the clear hours of its night, at the time of the night
    def _readChunks(results_path, config):
        """ Stand-in of AutoCalibration.readChunks. """
        night = os.path.basename(os.path.dirname(results_path))
        k = int(os.path.basename(results_path)[1:])
        n = int(plan['hours'].get(night, 2.0)/2*3600*config.fps/256)
        jd = JD0 + NIGHTS.index(night) + k*0.1
        return [('FF_{:s}_{:d}_{:05d}'.format(night, k, m), jd + m/1e5, np.zeros((50, 8))) for m in range(n)], 256

    def _clear(chunks, *args, **kwargs):
        """ Stand-in of AutoCalibration.clearChunks: every chunk is clear. """
        plan['references'].append(kwargs.get('reference_stars'))
        return list(chunks)

    # The refit: the nights it is given, its start and its flat; a better refit replaces the platepar
    def _refitDistortion(selected, platepar, config, mask_img, nproc):
        """ Stand-in of NightCalibration._refitDistortion. """
        plan['calls'].append(sorted(os.path.basename(path) for path in selected))
        plan['start'] = platepar.mag_lev
        plan['use_flat'] = config.use_flat
        plan['night_platepars'] = [os.path.isfile(os.path.join(path, config.platepar_name)) for path in selected]
        if plan.get('during'):
            plan['during']()
        if plan.get('raise'):
            raise RuntimeError('stand-in error')
        report = {'qc': {'ok': True, 'failed': [], 'checks': []}}
        if plan['better']:
            return _platepar(2.0), report, 'the worst zone 10.00" better on {:d} nights'.format(len(selected))
        return None, report, 'not clearly better (worst zone +1.00", needed 5.00")'

    import RMS.Astrometry.JointDistortionFit as jdf
    monkeypatch.setattr(ac, 'readChunks', _readChunks)
    monkeypatch.setattr(ac, 'clearChunks', _clear)
    monkeypatch.setattr(nc, '_refitDistortion', _refitDistortion)
    monkeypatch.setattr(nc, 'writeChunks', lambda chunks, dir_path, *args, **kwargs: os.makedirs(dir_path,
                                                                                                 exist_ok=True))
    monkeypatch.setattr(jdf, 'formatReport', lambda report: 'stand-in report')

    # The night is over unless a test says otherwise
    monkeypatch.setattr(mnr, 'nightBoundsFromName', lambda config, name: (
        NOW - datetime.timedelta(hours=20), plan.get('night_end', NOW - datetime.timedelta(hours=3))))

    return config, output_dir, plan


def _installed(config, output_dir):
    """ The value of the installed automatic platepar, None if there is none. """

    platepar = ac._platepar(ac.autoPlateparPath(output_dir, config), False)

    return None if platepar is None else platepar.mag_lev


def test_off_by_default(monitor, tmp_path):
    """ The option is off by default and in the default config file, and then the report refits nothing. """

    config, output_dir, plan = monitor

    assert cr.Config().monitor_auto_calibration_periodic_refit is False
    assert cr.parse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config')) \
        .monitor_auto_calibration_periodic_refit is False

    config.monitor_auto_calibration_periodic_refit = False
    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is None
    assert not plan['calls']
    assert 'periodic_refit' not in ac.readState(output_dir)

    # Nor without the automatic calibration, or with the interval of the refit 0
    config.monitor_auto_calibration_periodic_refit = True
    config.monitor_auto_calibration = False
    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is None
    config.monitor_auto_calibration = True
    config.auto_calibration_refit_days = 0
    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is None
    assert not plan['calls']


def test_option_read_from_the_config_file(tmp_path):
    """ The option is read from the section of the monitor. """

    repo_config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config')
    with open(repo_config) as f:
        text = f.read()
    config_path = str(tmp_path/'test.config')
    with open(config_path, 'w') as f:
        f.write(text.replace('monitor_auto_calibration_periodic_refit: false',
                             'monitor_auto_calibration_periodic_refit: true'))

    assert cr.parse(config_path).monitor_auto_calibration_periodic_refit is True


def test_runs_once_per_interval_on_the_newest_nights(monitor):
    """ The refit runs on the newest nights (this one and the reported ones before it, at most
        auto_calibration_refit_nights) from the platepar in use, records its time, and isn't run again before the
        interval is over. """

    config, output_dir, plan = monitor
    config.auto_calibration_refit_nights = 6

    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is not None
    assert plan['calls'] == [sorted(NIGHTS[-6:])]
    assert plan['start'] == 1.0
    assert all(plan['night_platepars'])

    refit_state = ac.readState(output_dir)['periodic_refit']
    assert refit_state['last_utc'] == NOW.strftime(mnr.PERIODIC_REFIT_TIME_FORMAT)
    assert refit_state['night'] == NIGHT
    assert refit_state['verdict'].startswith('installed the refit')
    assert os.path.isfile(os.path.join(mnr.nightDirPath(output_dir, NIGHT, config), mnr.PERIODIC_REFIT_REPORT_NAME))

    # The clear chunks are judged against the clearest sky seen by the camera, when it is known
    assert all(reference is None for reference in plan['references'])

    # Not again within the interval, again after it
    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW + datetime.timedelta(days=3)) is None
    assert len(plan['calls']) == 1
    mnr.periodicRefit(output_dir, NIGHT, config, now=NOW + datetime.timedelta(days=8))
    assert len(plan['calls']) == 2


def test_nights_and_clear_sky_of_the_refit(monitor):
    """ A night not reported yet (other than this one) isn't in the refit, and the clear chunks are judged
        against the clearest sky seen by the camera. """

    config, output_dir, plan = monitor
    config.auto_calibration_refit_nights = 4
    mnr.updateReportState(output_dir, night_name=NIGHTS[-2], reported_at=None)
    state = ac.readState(output_dir)
    state['star_counts'] = list(range(101))
    ac.writeState(output_dir, state)

    mnr.periodicRefit(output_dir, NIGHT, config, now=NOW)
    assert plan['calls'] == [sorted([NIGHT] + NIGHTS[-5:-2])]
    assert plan['references'] and all(reference == ac.CLEAR_REFERENCE_PERCENTILE
                                      for reference in plan['references'])


def test_interval_not_over(monitor):
    """ A refit recorded less than auto_calibration_refit_days ago: nothing is done. """

    config, output_dir, plan = monitor
    state = ac.readState(output_dir)
    state['periodic_refit'] = {'last_utc': (NOW - datetime.timedelta(days=2)).strftime(
        mnr.PERIODIC_REFIT_TIME_FORMAT), 'night': NIGHTS[-3], 'verdict': 'kept the platepar'}
    ac.writeState(output_dir, state)

    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is None
    assert not plan['calls']
    assert ac.readState(output_dir)['periodic_refit']['night'] == NIGHTS[-3]


def test_accepted_refit_installed_for_the_backfill(monitor):
    """ A clearly better refit is installed as the automatic platepar; the version of the platepar in use it
        replaces is returned and kept for the backfill, so the files calibrated with it get the refit. """

    config, output_dir, plan = monitor

    replaced = mnr.periodicRefit(output_dir, NIGHT, config, now=NOW)
    assert replaced == GIVEN_MTIME
    assert _installed(config, output_dir) == 2.0

    state = ac.readState(output_dir)
    assert state['joint_fit_replaced'] == [GIVEN_MTIME]
    assert state['installs'][-1]['dry_run'] is False
    assert state['installs'][-1]['reason'].startswith('refit on the recent nights')

    # The files of the night still have the replaced version: the backfill refines them
    assert mnr.refinementBackfillPending(output_dir, config, NIGHT) == [GIVEN_MTIME]


def test_refit_starts_from_the_newest_platepar_in_use(monitor):
    """ The refit starts from the platepar the monitor uses: the automatic platepar when it is newer than the one
        of the files, and it is the version which is replaced. """

    config, output_dir, plan = monitor
    auto_path = ac.autoPlateparPath(output_dir, config)
    ac.writePlatepar(_platepar(3.0), auto_path)
    os.utime(auto_path, (GIVEN_MTIME + 100, GIVEN_MTIME + 100))

    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) == GIVEN_MTIME + 100
    assert plan['start'] == 3.0
    assert _installed(config, output_dir) == 2.0


def test_rejected_refit_installs_nothing(monitor):
    """ A refit which isn't clearly better installs nothing and leaves nothing for the backfill; its time and
        verdict are recorded. """

    config, output_dir, plan = monitor
    plan['better'] = False

    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is None
    assert _installed(config, output_dir) is None

    state = ac.readState(output_dir)
    assert not state.get('joint_fit_replaced')
    assert not state['installs']
    assert state['periodic_refit']['verdict'].startswith('kept the platepar (not clearly better')
    assert state['periodic_refit']['last_utc'] == NOW.strftime(mnr.PERIODIC_REFIT_TIME_FORMAT)


def test_too_few_clear_nights(monitor):
    """ With fewer nights of enough clear sky than the refit needs, nothing is fitted and the verdict says so. """

    config, output_dir, plan = monitor
    plan['hours'] = {night: 0.5 for night in NIGHTS[:-2]}

    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is None
    assert not plan['calls']
    assert ac.readState(output_dir)['periodic_refit']['verdict'].startswith('kept the platepar (only 2 nights')


def test_dry_run_saves_a_candidate(monitor):
    """ In a dry run a better refit is only saved as a candidate: nothing is installed and nothing is left for the
        backfill. """

    config, output_dir, plan = monitor
    config.monitor_auto_calibration_dry_run = True

    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is None
    assert _installed(config, output_dir) is None
    assert os.path.isfile(os.path.join(output_dir, 'auto_candidate_' + config.platepar_name))

    state = ac.readState(output_dir)
    assert not state.get('joint_fit_replaced')
    assert state['installs'][-1]['dry_run'] is True
    assert state['periodic_refit']['verdict'].startswith('saved a candidate of the refit')


@pytest.mark.parametrize('decision', ['trial', 'pending'])
def test_skipped_while_a_platepar_is_on_trial(monitor, decision):
    """ While a platepar is on trial (or a candidate waits for its confirmation) nothing is fitted and no time is
        recorded, so the refit runs at a later report. """

    config, output_dir, plan = monitor
    state = ac.readState(output_dir)
    state[decision] = {'path': 'x', 'previous': 'y', 'fitted_jd': JD0}
    ac.writeState(output_dir, state)

    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is None
    assert not plan['calls']
    assert 'periodic_refit' not in ac.readState(output_dir)

    # Once the decision is made, the refit runs
    state = ac.readState(output_dir)
    state[decision] = None
    ac.writeState(output_dir, state)
    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is not None


def test_not_before_the_end_of_the_night(monitor):
    """ A partial report during the night neither fits nor uses up the refit. """

    config, output_dir, plan = monitor
    plan['night_end'] = NOW + datetime.timedelta(hours=3)

    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is None
    assert not plan['calls']
    assert 'periodic_refit' not in ac.readState(output_dir)


def test_old_night_reported_again(monitor):
    """ An old night reported again after newer nights were reported refits nothing. """

    config, output_dir, plan = monitor

    assert mnr.periodicRefit(output_dir, NIGHTS[-4], config, now=NOW) is None
    assert not plan['calls']


def test_platepar_replaced_during_the_refit(monitor):
    """ A platepar installed while the refit runs (e.g. by the processing of a backlog) is not replaced. """

    config, output_dir, plan = monitor

    def _install():
        """ Another platepar installed during the refit. """
        ac.writePlatepar(_platepar(4.0), ac.autoPlateparPath(output_dir, config))

    plan['during'] = _install

    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is None
    assert _installed(config, output_dir) == 4.0
    assert ac.readState(output_dir)['periodic_refit']['verdict'].startswith('not installed')


def test_error_is_logged_and_recorded(monitor, monkeypatch):
    """ An error in the refit is logged and recorded with the time (so it isn't repeated at every report), never
        raised. """

    config, output_dir, plan = monitor
    plan['raise'] = True
    errors = []
    monkeypatch.setattr(mnr.log, 'error', lambda message, *args: errors.append(message))

    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is None
    assert errors and 'stand-in error' in errors[0]
    assert _installed(config, output_dir) is None
    assert ac.readState(output_dir)['periodic_refit']['verdict'].startswith('failed')


@pytest.mark.parametrize('flat_applied', [False, True])
def test_flat_of_the_processing(monitor, flat_applied):
    """ The flat comes from how the files were processed, not from the configuration of the report, and the
        configuration of the caller is not changed. """

    config, output_dir, plan = monitor
    config.use_flat = not flat_applied
    for night in plan['files'].values():
        for info in night.values():
            info['flat_applied'] = flat_applied

    mnr.periodicRefit(output_dir, NIGHT, config, now=NOW)
    assert plan['use_flat'] is flat_applied
    assert config.use_flat is (not flat_applied)


def test_report_continues_after_an_error(tmp_path, monkeypatch):
    """ An error raised by the periodic refit doesn't stop the report of the night. """

    from Tests.test_MonitorNightReport import NIGHT as REPORT_NIGHT, _config, _makeResults

    config = _config()
    config.monitor_auto_calibration = True
    config.monitor_auto_calibration_periodic_refit = True
    output_dir = str(tmp_path)
    _makeResults(output_dir, config, 'file1', n_chunks=3)

    calls = []

    def _fail(output_dir, night_name, config):
        """ A refit which raises. """
        calls.append(night_name)
        raise RuntimeError('stand-in error')

    monkeypatch.setattr(mnr, 'periodicRefit', _fail)

    night_state = mnr.generateNightReport(output_dir, REPORT_NIGHT, config)
    assert calls == [REPORT_NIGHT]
    assert 'archive' in night_state['ok_steps']


@pytest.mark.parametrize('worker', ['install', 'trial'])
def test_platepar_installed_while_the_refit_waits_for_the_lock(monitor, monkeypatch, worker):
    """ A worker which installs a platepar (or puts one on trial) while it holds the lock of the state, after the
        fit and before the refit gets the lock, keeps its platepar: the platepar in use and the trial are checked
        under the lock. After a new trial the time isn't recorded, so the refit is made again later. """

    config, output_dir, plan = monitor
    real_lock = ac._StateLock

    class _WorkerFirst(object):
        """ The lock of the state, taken first by a worker which installs a platepar. """

        def __init__(self, *args, **kwargs):
            self.lock = real_lock(*args, **kwargs)

        def __enter__(self):

            # The worker installs its platepar while it holds the lock
            with real_lock(output_dir):
                ac.writePlatepar(_platepar(5.0), ac.autoPlateparPath(output_dir, config))
                if worker == 'trial':
                    state = ac.readState(output_dir)
                    state['trial'] = {'path': ac.autoPlateparPath(output_dir, config), 'previous': 'y',
                                      'fitted_jd': JD0}
                    ac.writeState(output_dir, state)

            return self.lock.__enter__()

        def __exit__(self, *args):
            return self.lock.__exit__(*args)

    monkeypatch.setattr(ac, '_StateLock', _WorkerFirst)

    assert mnr.periodicRefit(output_dir, NIGHT, config, now=NOW) is None
    assert _installed(config, output_dir) == 5.0

    state = ac.readState(output_dir)
    assert not state.get('joint_fit_replaced')
    assert state['periodic_refit']['verdict'].startswith('not installed')
    if worker == 'trial':
        assert state['periodic_refit']['last_utc'] is None


def _backlog(config, output_dir, plan, mtime):
    """ A backlog: the nights after NIGHTS[5] were processed, with a newer given platepar of the given version,
        but not reported yet. """

    for night in NIGHTS[6:]:
        mnr.updateReportState(output_dir, night_name=night, reported_at=None)
        for results_dir, info in plan['files'][night].items():
            path = os.path.join(output_dir, results_dir, config.platepar_name)
            ac.writePlatepar(_platepar(6.0), path)
            os.utime(path, (mtime, mtime))
            info['platepar_mtime'] = mtime


def test_backlog_refit_starts_from_the_newest_given_platepar(monitor):
    """ The report of a night of a backlog refits the platepar in use, also when it is only in the files of the
        newer nights not reported yet (a newer given platepar), and that is the version it replaces; the refit is
        fitted on the reported nights. """

    config, output_dir, plan = monitor
    config.auto_calibration_refit_nights = 4
    _backlog(config, output_dir, plan, GIVEN_MTIME + 500)

    assert mnr.periodicRefit(output_dir, NIGHTS[5], config, now=NOW) == GIVEN_MTIME + 500
    assert plan['start'] == 6.0
    assert plan['calls'] == [sorted(NIGHTS[2:6])]


def test_backlog_newer_given_platepar_during_the_refit(monitor):
    """ A newer given platepar first used by a file of a night not reported yet while the refit runs isn't
        replaced by the refit. """

    config, output_dir, plan = monitor
    _backlog(config, output_dir, plan, GIVEN_MTIME)

    def _newer():
        """ A file of the newest night processed with a newer given platepar. """
        results_dir = sorted(plan['files'][NIGHTS[-1]])[-1]
        path = os.path.join(output_dir, results_dir, config.platepar_name)
        os.utime(path, (GIVEN_MTIME + 900, GIVEN_MTIME + 900))

    plan['during'] = _newer

    assert mnr.periodicRefit(output_dir, NIGHTS[5], config, now=NOW) is None
    assert _installed(config, output_dir) is None
    assert ac.readState(output_dir)['periodic_refit']['verdict'].startswith('not installed, the platepar was '
                                                                             'replaced')
