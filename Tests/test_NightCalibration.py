"""
Tests of the automatic calibration of a classic camera during the night and at its end
(RMS.Astrometry.NightCalibration).

The fits and the scores are replaced by stand-ins whose outcome each test sets, so the tests check the decisions:
when a platepar is fitted, installed, checked and refitted, the retries and their limit, that the installed platepar
is only ever replaced by a better one, the reading of the stars of the detection, and the decisions at the end of
the night.
"""

from __future__ import print_function, division, absolute_import

import copy
import datetime
import json
import os

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.Astrometry.AutoCalibration as ac
import RMS.Astrometry.NightCalibration as nc
from RMS.Formats.Platepar import Platepar


JD0 = 2461000.0


class _Score(object):
    """ A stand-in score: whether it is good and a value which compares platepars. """

    def __init__(self, ok, value):
        self.ok, self.value = ok, value

    def toDict(self):
        return {'ok': self.ok, 'value': self.value}


def _platepar(value):
    """ A platepar whose quality is its mag_lev (the stand-in scores read it). """

    pp = Platepar()
    pp.X_res, pp.Y_res = 1280, 720
    pp.mag_lev = value

    return pp


def _chunks(minutes, start=0):
    """ Chunks every minute, from the given minute, as (name, jd, stars). """

    return [('FF_XX0001_{:06d}.fits'.format(start + m), JD0 + (start + m)/1440.0, np.zeros((40, 8)))
            for m in range(minutes)]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """ A config with the platepar installed in a temporary directory, and stand-ins for the fits and scores.

    Return:
        (config, night_dir, plan): plan sets the outcomes: 'first' (list of the values of the first fits, None
            for a failure, a negative value for a platepar which fails on held-out chunks), 'refit' (value of the
            refits, None for a failure), 'good_above' (a platepar is good on new stars if its value is above it),
            and 'calls' records the fits.
    """

    config = cr.Config()
    config.config_file_path = str(tmp_path/"config")
    os.makedirs(config.config_file_path)
    night_dir = str(tmp_path/"night")
    os.makedirs(night_dir)

    plan = {'first': [], 'refit': None, 'good_above': 0, 'calls': []}

    def _fitCandidate(chunk_dirs, pool, held_out, catalog_stars, config, **kwargs):
        plan['calls'].append(('first', len(pool), kwargs.get('fit_vignetting')))
        value = plan['first'].pop(0) if plan['first'] else None
        return (None, {'reason': 'no fit'}) if value is None else (_platepar(value), {})

    def _jointFit(dir_paths, platepar, config, **kwargs):
        plan['calls'].append(('refit', platepar.mag_lev, kwargs.get('max_chunks')))
        if plan['refit'] is None:
            return None, None
        report = {'qc': {'ok': plan.get('qc_ok', True), 'failed': [] if plan.get('qc_ok', True) else ['converged'],
                         'checks': []},
                  'nights': [], 'held_out': [], 'distortion_type': 'radial7-odd', 'chunks': 0, 'match_radius': 1,
                  'lim_mag': 6, 'snr_min': 0, 'iterations': [{'iteration': 0, 'recalibrated': 0, 'stats': None}]}
        return _platepar(plan['refit']), report

    def _score(platepar, chunks, *args, **kwargs):
        return _Score(platepar.mag_lev > plan['good_above'], platepar.mag_lev)

    monkeypatch.setattr(ac, 'fitCandidate', _fitCandidate)
    monkeypatch.setattr(ac, 'scorePlatepar', _score)
    monkeypatch.setattr(ac, 'judgeScore', lambda score: (score.ok, [] if score.ok else ['fails']))
    monkeypatch.setattr(ac, 'candidateIsBetter', lambda cand, cur: (cand.value > cur.value,
                                                                    'value {:.1f}'.format(cand.value)))
    monkeypatch.setattr(ac, 'clearChunks', lambda chunks, *args, **kwargs: list(chunks))
    monkeypatch.setattr(ac, '_loadCatalog', lambda config, jd: None)
    monkeypatch.setattr(nc, '_quietJointFit', _jointFit)
    monkeypatch.setattr(nc, '_mask', lambda config, night_dir: None)
    monkeypatch.setattr(nc, 'writeChunks', lambda *args, **kwargs: None)
    monkeypatch.setattr(nc, 'skyStable', lambda chunks, now_jd: True)
    _vignettingStandIn(monkeypatch, plan)

    return config, night_dir, plan


def _vignettingStandIn(monkeypatch, plan):
    """ A stand-in refit of the vignetting: it records the value of the platepar it refits ('vig_calls') and,
        if plan['vig_kept'], returns a copy whose value is higher by 0.25. """

    plan.setdefault('vig_kept', False)
    plan.setdefault('vig_calls', [])

    def _refitVignetting(platepar, chunks, catalog_stars, config, ff_frames=256, mask=None):
        plan['vig_calls'].append(platepar.mag_lev)
        if plan['vig_kept']:
            refitted = copy.deepcopy(platepar)
            refitted.mag_lev += 0.25
            return refitted, {'kept': True, 'reason': 'better'}
        return platepar, {'kept': False, 'reason': 'not better'}

    monkeypatch.setattr(ac, 'refitVignetting', _refitVignetting)


def _run(calibration, minutes):
    """ The calibration of the night stepped every minute up to the given minute. """

    actions = []
    for m in range(1, minutes + 1):
        action = calibration.update(_chunks(m))
        if action:
            actions.append((m, action))

    return actions


def _installed(config):
    """ The value of the installed platepar, None if there is none. """

    pp = ac._platepar(nc.installPath(config), False)
    return None if pp is None else pp.mag_lev


def test_first_platepar_installed_then_checked_and_refitted(setup):
    """ The first platepar is installed as soon as there are enough clear chunks; after the check time it is
        checked on new stars and refitted, the better refit is installed, and when it is good the night is done. """

    config, night_dir, plan = setup
    plan['first'] = [2.0]
    plan['refit'] = 3.0
    plan['good_above'] = 1.5

    calibration = nc.NightCalibration(config, night_dir)
    actions = _run(calibration, 120)

    # Installed at the first chunk count which allows a fit
    assert actions[0][0] == ac.MIN_CLEAR_CHUNKS
    assert 'installed the first platepar' in actions[0][1]

    # Checked after the check time on new chunks, the refit installed and the night done
    check_minute = ac.MIN_CLEAR_CHUNKS + int(config.auto_calibration_check_min)
    assert actions[1][0] == check_minute
    assert 'installed the refit' in actions[1][1] and 'done for the night' in actions[1][1]
    assert len(actions) == 2
    assert _installed(config) == 3.0
    assert calibration.state['done'] and (calibration.state['attempts'] == 2)

    # The refit started from the installed platepar, with a bounded number of chunks; the first fit fitted the
    #   vignetting (no flat)
    assert plan['calls'][1] == ('refit', 2.0, nc.NIGHT_REFIT_MAX_CHUNKS)
    assert plan['calls'][0][2] is True

    # The first platepar is kept as replaced
    assert any(name.startswith('replaced_') for name in os.listdir(config.config_file_path))


def test_failed_fits_are_retried_up_to_the_limit(setup):
    """ A first fit which fails (no fit, or failing on held-out chunks) is retried after the retry time, at most
        auto_calibration_max_fits times a night; nothing is installed. """

    config, night_dir, plan = setup
    plan['first'] = [None, -1.0, None, None, None, None]
    plan['good_above'] = 0.0

    calibration = nc.NightCalibration(config, night_dir)
    actions = _run(calibration, 200)

    attempts = [m for m, action in actions if 'first fit' in action]
    assert len(attempts) == config.auto_calibration_max_fits
    assert np.all(np.diff(attempts) == int(config.auto_calibration_retry_min))
    assert 'no more attempts' in actions[-1][1]
    assert _installed(config) is None


def test_installed_platepar_never_replaced_by_a_worse_one(setup):
    """ A refit which isn't better is not installed; while the installed platepar fails on new stars, the next
        attempt comes after the retry time with stars after the last check, until the limit. """

    config, night_dir, plan = setup
    plan['first'] = [1.0]
    plan['refit'] = 0.5
    plan['good_above'] = 0.9

    # The first platepar is good on its held-out chunks, then fails on the new stars
    calibration = nc.NightCalibration(config, night_dir)
    calibration.update(_chunks(ac.MIN_CLEAR_CHUNKS))
    assert _installed(config) == 1.0
    plan['good_above'] = 5.0
    actions = _run(calibration, 300)

    refits = [(m, action) for m, action in actions if action.startswith('attempt')]
    assert len(refits) == config.auto_calibration_max_fits - 1
    assert all('kept the installed platepar' in action for _, action in refits)
    assert _installed(config) == 1.0

    # The first check after the check time, the next ones after the retry time
    minutes = [m for m, _ in refits]
    assert minutes[0] == ac.MIN_CLEAR_CHUNKS + int(config.auto_calibration_check_min)
    assert np.all(np.diff(minutes) == int(config.auto_calibration_retry_min))


def test_check_waits_for_new_stars(setup):
    """ The check needs enough clear chunks taken after the installed platepar: without new clear chunks (e.g.
        clouds after the install) nothing is done and no attempt is counted. """

    config, night_dir, plan = setup
    plan['first'] = [1.0]
    plan['refit'] = 2.0

    calibration = nc.NightCalibration(config, night_dir)
    calibration.update(_chunks(ac.MIN_CLEAR_CHUNKS))
    assert calibration.state['attempts'] == 1

    # Long after the check time, but no chunk after the install
    late = [(name, JD0 + 0.5, stars) for name, _, stars in _chunks(ac.MIN_CLEAR_CHUNKS)]
    calibration.state['installed_jd'] = calibration.state['check_from_jd'] = JD0 + 0.5
    calibration.state['next_attempt_jd'] = JD0
    assert calibration.update(late) is None
    assert calibration.state['attempts'] == 1


def test_state_survives_a_restart(setup):
    """ The state is kept in the night directory: a calibration started again (e.g. after a restart of the
        capture) continues where the previous one was. """

    config, night_dir, plan = setup
    plan['first'] = [1.0]

    calibration = nc.NightCalibration(config, night_dir)
    calibration.update(_chunks(ac.MIN_CLEAR_CHUNKS))

    again = nc.NightCalibration(config, night_dir)
    assert again.state['attempts'] == 1
    assert again.state['installed'] == nc.installPath(config)
    assert again.update(_chunks(ac.MIN_CLEAR_CHUNKS + 1)) is None


def test_dry_run_installs_nothing(setup):
    """ In a dry run the platepars are saved as candidates next to the install path, which stays empty. """

    config, night_dir, plan = setup
    plan['first'] = [1.0]

    calibration = nc.NightCalibration(config, night_dir, dry_run=True)
    calibration.update(_chunks(ac.MIN_CLEAR_CHUNKS))

    assert not os.path.isfile(nc.installPath(config))
    assert os.path.isfile(os.path.join(config.config_file_path, 'candidate_' + config.platepar_name))


def test_flat_fielded_camera_fits_no_vignetting(setup):
    """ A camera with a flat gets no vignetting fit in its first platepar. """

    config, night_dir, plan = setup
    config.use_flat = True
    plan['first'] = [1.0]

    nc.NightCalibration(config, night_dir).update(_chunks(ac.MIN_CLEAR_CHUNKS))
    assert plan['calls'][0][2] is False


def test_read_detection_backups(tmp_path):
    """ The stars of the detection are read from its backup files once each, as CALSTARS rows; files which can't
        be read yet are read later. """

    from RMS.Pickling import savePickle
    from RMS.QueuedPool import BackupContainer

    night_dir = str(tmp_path)
    star_data = ['FF_A.fits', [10.0, 20.0], [5.0, 6.0], [100, 200], [1000, 2000], [2.0, 2.5], [30, 40], [8.0, 9.0],
                 [0, 1]]
    savePickle(BackupContainer(['dir', 'FF_A.fits'], ['FF_A.fits', star_data, [], {}]), night_dir,
               'rms_queue_bkup_1_abc.pickle')
    savePickle(BackupContainer(['dir', 'FF_B.fits'], ['FF_B.fits', ['FF_B.fits', [], [], [], [], [], [], [], []],
                                                      [], {}]), night_dir, 'rms_queue_bkup_2_abc.pickle')
    with open(os.path.join(night_dir, 'rms_queue_bkup_3_abc.pickle'), 'w') as f:
        f.write('partial')

    seen = set()
    chunks = nc.readDetectionBackups(night_dir, seen)
    assert [name for name, _ in chunks] == ['FF_A.fits']
    rows = chunks[0][1]
    assert rows.shape == (2, 8)
    assert list(rows[0]) == [5.0, 10.0, 1000, 100, 2.0, 30, 8.0, 0]

    # Read once; the partial file is tried again
    assert nc.readDetectionBackups(night_dir, seen) == []
    assert 'rms_queue_bkup_3_abc.pickle' not in seen


def _endOfNight(setup, monkeypatch, platepar, fitted_tonight, chunks=240):
    """ The end of the night with the given platepar, on chunks of the given number of minutes. """

    config, night_dir, plan = setup
    monkeypatch.setattr(ac, 'readChunks', lambda night_dir, config, file_name=None: (_chunks(chunks), 256))
    if fitted_tonight:
        nc.writeState(night_dir, dict(nc.readState(night_dir), installed=nc.installPath(config)))
    path = nc.installPath(config)
    if platepar is not None:
        ac.writePlatepar(platepar, path)

    return nc.endOfNightCalibration(config, night_dir, 'CALSTARS.txt', platepar, path, 'json', nproc=1)


def test_end_of_night_installs_a_good_joint_fit(setup, monkeypatch):
    """ The platepar installed tonight is refitted on the whole night, and the fit which is good its quality
        control is installed and used for the night. """

    config, night_dir, plan = setup
    plan['refit'] = 3.0

    platepar, path, fmt = _endOfNight(setup, monkeypatch, _platepar(1.0), True)
    assert platepar.mag_lev == 3.0
    assert _installed(config) == 3.0
    assert os.path.isfile(os.path.join(night_dir, 'auto_calibration_joint_report.txt'))

    # The quality control limits of a night: hours of clear sky over a share of the night
    assert plan['calls'][-1] == ('refit', 1.0, nc.END_MAX_CHUNKS)


def test_end_of_night_keeps_the_platepar_when_the_fit_fails(setup, monkeypatch):
    """ A joint fit which fails its quality control is not installed, the platepar of the night is kept. """

    config, night_dir, plan = setup
    plan['refit'] = 3.0
    plan['qc_ok'] = False

    platepar, path, fmt = _endOfNight(setup, monkeypatch, _platepar(1.0), True)
    assert platepar.mag_lev == 1.0
    assert _installed(config) == 1.0


def test_end_of_night_keeps_a_good_existing_platepar(setup, monkeypatch):
    """ An existing platepar (not fitted tonight) which is good on the night's stars is kept without a fit; one
        which fails is refitted. """

    config, night_dir, plan = setup
    plan['refit'] = 3.0
    plan['good_above'] = 0.5

    platepar, _, _ = _endOfNight(setup, monkeypatch, _platepar(1.0), False)
    assert platepar.mag_lev == 1.0
    assert not [call for call in plan['calls'] if call[0] == 'refit']

    plan['good_above'] = 2.0
    platepar, _, _ = _endOfNight(setup, monkeypatch, _platepar(1.0), False)
    assert platepar.mag_lev == 3.0


def test_end_of_night_without_a_platepar(setup, monkeypatch):
    """ Without a platepar (e.g. the night processed again, or no clear sky during the night) one is fitted from
        scratch on the night and refitted; if the joint fit fails, the one from scratch is installed if it
        is good. """

    config, night_dir, plan = setup
    plan['first'] = [1.0, 1.0]
    plan['refit'] = 3.0
    plan['qc_ok'] = False

    platepar, _, _ = _endOfNight(setup, monkeypatch, None, False)
    assert platepar.mag_lev == 1.0
    assert _installed(config) == 1.0


def test_end_of_night_without_clear_sky(setup, monkeypatch):
    """ A night with too little clear sky changes nothing. """

    config, night_dir, plan = setup
    plan['refit'] = 3.0

    platepar, _, _ = _endOfNight(setup, monkeypatch, _platepar(1.0), True, chunks=ac.MIN_CLEAR_CHUNKS - 1)
    assert platepar.mag_lev == 1.0
    assert not plan['calls']


def test_sky_stable():
    """ The sky is stable once the stars per chunk stop growing: not while they grow at dusk, nor without enough
        chunks to compare. """

    def _night(counts):
        return [('FF_{:d}'.format(m), JD0 + m/1440.0, np.zeros((n, 8))) for m, n in enumerate(counts)]

    dusk = _night([10 + 5*m for m in range(20)])
    assert not nc.skyStable(dusk, dusk[-1][1])

    dark = _night([100]*20)
    assert nc.skyStable(dark, dark[-1][1])

    assert not nc.skyStable(dark[:3], dark[2][1])


def test_night_calibration_process_starts_and_stops(tmp_path):
    """ The calibration of the night runs in its own process next to the capture and stops when the capture ends,
        also while it waits for new stars. StartCapture starts it only for a camera without a platepar when the
        automatic calibration is on. """

    import time

    import RMS.StartCapture as sc

    config = cr.Config()
    config.config_file_path = str(tmp_path/"config")
    os.makedirs(config.config_file_path)
    night_dir = str(tmp_path/"night")
    os.makedirs(night_dir)

    # Off by default: nothing starts
    sc.startNightCalibration(config, night_dir)
    assert night_dir not in sc.NIGHT_CALIBRATIONS

    # On, without a platepar: it runs until it is stopped
    config.auto_calibration = True
    sc.startNightCalibration(config, night_dir)
    calibration = sc.NIGHT_CALIBRATIONS[night_dir]
    time.sleep(1)
    assert calibration.process.is_alive()
    start = time.time()
    sc.stopNightCalibration(night_dir)
    assert not calibration.process.is_alive()
    assert time.time() - start < 30
    assert night_dir not in sc.NIGHT_CALIBRATIONS

    # With a platepar nothing starts
    with open(nc.installPath(config), 'w') as f:
        f.write('{}')
    sc.startNightCalibration(config, night_dir)
    assert night_dir not in sc.NIGHT_CALIBRATIONS

    # Unless the calibration of the night installed it and isn't done (the capture was restarted): it resumes
    nc.writeState(night_dir, dict(nc.readState(night_dir), installed=nc.installPath(config), done=False))
    sc.startNightCalibration(config, night_dir)
    assert night_dir in sc.NIGHT_CALIBRATIONS
    sc.stopNightCalibration(night_dir)

    # A finished calibration doesn't start again
    nc.writeState(night_dir, dict(nc.readState(night_dir), done=True))
    sc.startNightCalibration(config, night_dir)
    assert night_dir not in sc.NIGHT_CALIBRATIONS


def test_platepar_of_the_last_attempt_is_checked(setup):
    """ A first platepar installed by the last allowed attempt is still checked on new stars after the check time
        (without another fit), which ends the night. """

    config, night_dir, plan = setup
    config.auto_calibration_max_fits = 1
    plan['first'] = [2.0]
    plan['refit'] = 3.0
    plan['good_above'] = 1.0

    calibration = nc.NightCalibration(config, night_dir)
    actions = _run(calibration, 120)

    assert 'installed the first platepar' in actions[0][1]
    assert actions[1][0] == ac.MIN_CLEAR_CHUNKS + int(config.auto_calibration_check_min)
    assert 'checked the platepar of the last attempt' in actions[1][1] and 'done for the night' in actions[1][1]
    assert len(actions) == 2
    assert not [call for call in plan['calls'] if call[0] == 'refit']
    assert calibration.state['done'] and (calibration.state['attempts'] == 1)
    assert _installed(config) == 2.0

### Periodic refit over many nights ###

def _stats(worst):
    """ Residual statistics whose worst zone is the given size (arcsec). """

    zones = [[(0.1, 0.0, 100)]*3 for _ in range(3)]
    zones[0][0] = (worst, 0.0, 100)

    return {'zones': zones, 'corners': {name: (0.1, 0.0, 100) for name in ('TL', 'TR', 'BL', 'BR')}, 'rms': 50.0}


@pytest.fixture
def refit(tmp_path, monkeypatch):
    """ Archived nights with a stand-in fit: plan sets the clear hours of every night, the verdict of the quality
        control and the worst zone before and after.

    Return:
        (config, plan)
    """

    config = cr.Config()
    config.config_file_path = str(tmp_path/"config")
    os.makedirs(config.config_file_path)
    config.data_dir = str(tmp_path/"data")
    archived = os.path.join(config.data_dir, config.archived_dir)
    for n in range(20):
        os.makedirs(os.path.join(archived, '{:s}_202611{:02d}_000000_000000'.format(config.stationID, n + 1)))
    ac.writePlatepar(_platepar(1.0), nc.installPath(config))

    plan = {'hours': {}, 'qc_ok': True, 'before': 30.0, 'after': 5.0, 'calls': []}

    def _readChunks(night_dir, config, file_name=None):
        hours = plan['hours'].get(os.path.basename(night_dir), 2.0)
        n = int(hours*3600*config.fps/256)
        return [('FF_{:05d}'.format(m), JD0 + m/1e4, np.full((10, 8), 2.0)) for m in range(n)], 256

    def _jointFit(dir_paths, platepar, config, **kwargs):
        plan['calls'].append((sorted(os.path.basename(d) for d in dir_paths), kwargs.get('holdout'),
                              kwargs.get('holdout_baseline')))
        plan['qc_mask'] = kwargs.get('mask')
        report = {'qc': {'ok': plan['qc_ok'], 'failed': [] if plan['qc_ok'] else ['converged'], 'checks': []},
                  'held_out_stats': {'start': {'stats': _stats(plan['before'])},
                                     'fitted': {'stats': _stats(plan['after'])}},
                  'nights': [], 'held_out': [], 'distortion_type': 'radial7-odd', 'chunks': 0, 'match_radius': 1,
                  'lim_mag': 6, 'snr_min': 0, 'iterations': [{'iteration': 0, 'recalibrated': 0, 'stats': None}]}
        fitted = _platepar(2.0)
        fitted.F_scale = 3600.0/60.0
        return fitted, report

    monkeypatch.setattr(ac, 'readChunks', _readChunks)
    def _clearChunks(chunks, *args, **kwargs):
        plan['masks'].append(kwargs.get('mask'))
        return list(chunks)

    plan['masks'] = []
    monkeypatch.setattr(ac, 'clearChunks', _clearChunks)
    monkeypatch.setattr(nc, '_mask', lambda config, night_dir: type('Mask', (), {'img': 'mask image'})())
    monkeypatch.setattr(nc, '_quietJointFit', _jointFit)
    monkeypatch.setattr(nc, 'writeChunks', lambda chunks, dir_path, *args, **kwargs: os.makedirs(dir_path,
                                                                                                 exist_ok=True))
    import RMS.Astrometry.JointDistortionFit as jdf
    monkeypatch.setattr(jdf, 'transferPointing', lambda platepar, pointing, **kwargs: (platepar, 0.0))
    monkeypatch.setattr(jdf, 'formatReport', lambda report: '')
    monkeypatch.setattr(ac, '_loadCatalog', lambda config, jd: None)
    _vignettingStandIn(monkeypatch, plan)

    return config, plan


def test_refit_on_the_newest_clear_nights(refit):
    """ The refit uses the newest nights with enough clear sky, every few held out, and installs a clearly better
        fit; it isn't run again before the interval. """

    config, plan = refit
    plan['hours']['{:s}_20261120_000000_000000'.format(config.stationID)] = 0.5
    now = datetime.datetime(2026, 11, 21, tzinfo=datetime.timezone.utc)

    action = nc.periodicRefit(config, nproc=1, now=now)
    assert action.startswith('installed the refit')
    assert _installed(config) == 2.0

    nights, holdout, baseline = plan['calls'][0]
    assert len(nights) == config.auto_calibration_refit_nights
    assert '{:s}_20261120_000000_000000'.format(config.stationID) not in nights
    assert '{:s}_20261119_000000_000000'.format(config.stationID) in nights
    assert holdout == nc.REFIT_HOLDOUT

    # The held-out nights are compared with the platepar of the camera, which the refit would replace
    assert baseline == 'start'

    # The clear chunks are judged with the mask, also in the quality control of the fit
    assert plan['masks'] and all(mask == 'mask image' for mask in plan['masks'])
    assert plan['qc_mask'] == 'mask image'

    # Not again within the interval, again after it
    assert nc.periodicRefit(config, nproc=1, now=now + datetime.timedelta(days=3)) is None
    assert nc.periodicRefit(config, nproc=1, now=now + datetime.timedelta(days=8)) is not None


def test_refit_not_installed_unless_clearly_better(refit):
    """ A refit which fails the quality control, or whose worst zone isn't better by the needed number of standard
        errors, isn't installed; nor without enough nights of clear sky. """

    config, plan = refit
    now = datetime.datetime(2026, 11, 21, tzinfo=datetime.timezone.utc)

    plan['qc_ok'] = False
    assert 'quality control' in nc.periodicRefit(config, nproc=1, now=now)
    assert _installed(config) == 1.0

    # The worst zone better by less than 3 standard errors (3 x 50"/sqrt(100) = 15")
    plan['qc_ok'], plan['after'] = True, 20.0
    assert 'not clearly better' in nc.periodicRefit(config, nproc=1, now=now + datetime.timedelta(days=8))
    assert _installed(config) == 1.0

    # Too few clear nights
    plan['hours'] = {name: 0.5 for name in os.listdir(os.path.join(config.data_dir, config.archived_dir))}
    assert 'only 0 nights' in nc.periodicRefit(config, nproc=1, now=now + datetime.timedelta(days=16))


def test_refit_off(refit):
    """ With auto_calibration_refit_days 0 there is no refit. """

    config, plan = refit
    config.auto_calibration_refit_days = 0
    assert nc.periodicRefit(config, nproc=1) is None
    assert not plan['calls']


def test_worst_zones_compared_over_the_same_zones():
    """ The worst zone before and after a refit is taken over the same zones (enough stars in both): a zone which
        loses its stars after the fit doesn't make the fit look better. """

    before, after = _stats(60.0), _stats(61.0)
    after['zones'][0][0] = (61.0, 0.0, 5)
    worst_before, n_before, worst_after = nc._worstZones(before, after, 10)
    assert worst_before == pytest.approx(0.1) and worst_after == pytest.approx(0.1)

    # The zone with enough stars in both counts
    assert nc._worstZones(_stats(60.0), _stats(61.0), 10) == (60.0, 100, 61.0)

    # No zone with enough stars
    assert nc._worstZones(_stats(60.0), _stats(61.0), 1000) is None


def test_vignetting_not_refitted_unless_asked(setup, monkeypatch):
    """ Without auto_calibration_fit_vignetting (the default) the vignetting of an existing platepar is never
        refitted: neither in the refits during the night nor at its end. """

    config, night_dir, plan = setup
    plan['first'] = [2.0]
    plan['refit'] = 3.0
    plan['good_above'] = 1.5
    _run(nc.NightCalibration(config, night_dir), 120)
    _endOfNight(setup, monkeypatch, _platepar(3.0), True)

    assert not plan['vig_calls']


def test_vignetting_refitted_during_the_night(setup):
    """ With auto_calibration_fit_vignetting the refit during the night also gets its vignetting refitted, before
        it is compared with the installed platepar. """

    config, night_dir, plan = setup
    config.auto_calibration_fit_vignetting = True
    plan['first'] = [2.0]
    plan['refit'] = 3.0
    plan['good_above'] = 1.5
    plan['vig_kept'] = True

    _run(nc.NightCalibration(config, night_dir), 120)
    assert plan['vig_calls'] == [3.0]
    assert _installed(config) == 3.25


def test_vignetting_of_a_good_platepar_refitted_at_the_end_of_the_night(setup, monkeypatch):
    """ With auto_calibration_fit_vignetting a good existing platepar gets only its vignetting refitted at the end
        of the night, installed if it is kept; nothing changes if it isn't, nor for a camera with a flat. """

    config, night_dir, plan = setup
    config.auto_calibration_fit_vignetting = True
    plan['refit'] = 3.0
    plan['good_above'] = 0.5

    platepar, _, _ = _endOfNight(setup, monkeypatch, _platepar(1.0), False)
    assert platepar.mag_lev == 1.0
    assert plan['vig_calls'] == [1.0]

    plan['vig_kept'] = True
    platepar, _, _ = _endOfNight(setup, monkeypatch, _platepar(1.0), False)
    assert platepar.mag_lev == 1.25
    assert _installed(config) == 1.25
    assert not [call for call in plan['calls'] if call[0] == 'refit']

    config.use_flat = True
    plan['vig_calls'] = []
    platepar, _, _ = _endOfNight(setup, monkeypatch, _platepar(1.0), False)
    assert platepar.mag_lev == 1.0
    assert not plan['vig_calls']


def test_joint_fit_of_the_night_starts_from_the_refitted_vignetting(setup, monkeypatch):
    """ With auto_calibration_fit_vignetting the joint fit of the night starts from the platepar with its
        vignetting refitted, so its quality control judges it; if the fit fails its quality control, the platepar
        with the refitted vignetting is installed. """

    config, night_dir, plan = setup
    config.auto_calibration_fit_vignetting = True
    plan['refit'] = 3.0
    plan['qc_ok'] = False
    plan['vig_kept'] = True

    platepar, _, _ = _endOfNight(setup, monkeypatch, _platepar(1.0), True)
    assert plan['calls'][-1][:2] == ('refit', 1.25)
    assert platepar.mag_lev == 1.25
    assert _installed(config) == 1.25


def test_vignetting_refitted_in_the_periodic_refit(refit):
    """ With auto_calibration_fit_vignetting the periodic refit also refits the vignetting on the newest night:
        of the fitted platepar if it replaces the camera's, otherwise of the camera's; a kept vignetting alone is
        installed too. """

    config, plan = refit
    config.auto_calibration_fit_vignetting = True
    plan['vig_kept'] = True
    now = datetime.datetime(2026, 11, 21, tzinfo=datetime.timezone.utc)

    # The distortion not clearly better: the camera's platepar with the refitted vignetting
    plan['after'] = 20.0
    action = nc.periodicRefit(config, nproc=1, now=now)
    assert 'vignetting refitted' in action
    assert plan['vig_calls'] == [1.0]
    assert _installed(config) == 1.25

    # The distortion clearly better: the fitted platepar with the refitted vignetting
    plan['after'] = 5.0
    action = nc.periodicRefit(config, nproc=1, now=now + datetime.timedelta(days=8))
    assert plan['vig_calls'][-1] == 2.0
    assert _installed(config) == 2.25
