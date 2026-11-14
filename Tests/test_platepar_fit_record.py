"""
Tests of the fit record of the platepar (Platepar.fit_record): what fitted the distortion and the pointing,
when, with which RMS code and with which astrometric model (ASTROMETRY_MODEL), kept through saving and
loading, and the warning when detections are computed with a platepar of an older model used as it is.
"""

import copy
import datetime
import os
import pickle
import shutil
import subprocess

import numpy as np
import pytest

from RMS.Astrometry import ApplyRecalibrate
from RMS.Astrometry.JointDistortionFit import transferPointing
from RMS.Formats import Platepar as PlateparModule
from RMS.Formats.Platepar import ASTROMETRY_MODEL, Platepar
from RMS.Misc import gitCommit, getRmsRootDir, rmsCommit
from Tests.test_RecalibrateLSQRefine import makeConfig, makeImage, makePlatepar, perturbed, recalibrate


def _imageStars(rows):
    """ Image stars (x, y, intensity) of fitAstrometry from CALSTARS rows (y, x, intensity, ...). """

    return np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])


def _checkEntry(entry, fit_by):
    """ A record of a fit made now by this code. """

    assert entry['model'] == ASTROMETRY_MODEL
    assert entry['by'] == fit_by
    assert entry['commit'] == rmsCommit()

    # ISO 8601 in UTC, within a few minutes of now
    fit_time = datetime.datetime.strptime(entry['utc'], '%Y-%m-%dT%H:%M:%SZ')
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    assert abs((now - fit_time).total_seconds()) < 600


def test_new_platepar_has_no_recorded_fit():

    # Nothing fitted yet: both fits are unknown and the model is the oldest one
    pp = Platepar()
    assert pp.fit_record == {'distortion': None, 'pointing': None}
    assert pp.astrometryModel() == 1
    assert pp.version == 3


def test_full_fit_records_the_distortion_and_the_pointing():

    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)

    # A fit of the distortion and the pointing (radial models fit them together)
    fitted = perturbed(pp)
    fitted.fitAstrometry(pp.JD, _imageStars(rows), catalog_stars, fit_by='test full fit')

    # One entry for both fits
    _checkEntry(fitted.fit_record['distortion'], 'test full fit')
    assert fitted.fit_record['pointing'] == fitted.fit_record['distortion']
    assert fitted.astrometryModel() == ASTROMETRY_MODEL


def test_default_label_of_a_fit():

    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)

    # Without a label the fit is named after the function
    fitted = perturbed(pp)
    fitted.fitAstrometry(pp.JD, _imageStars(rows), catalog_stars)
    assert fitted.fit_record['distortion']['by'] == 'fitAstrometry'


def test_pointing_only_fit_records_only_the_pointing():

    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)

    # A pointing-only fit of a platepar fitted before the record existed
    fitted = perturbed(pp)
    fitted.fitAstrometry(pp.JD, _imageStars(rows), catalog_stars, fit_only_pointing=True,
                         fit_by='test pointing')

    # The pointing is recorded, the distortion is still of an unknown fit, so of the oldest model
    _checkEntry(fitted.fit_record['pointing'], 'test pointing')
    assert fitted.fit_record['distortion'] is None
    assert fitted.astrometryModel() == 1

    # The pointing fit alone records the same way
    fitted = perturbed(pp)
    fitted.fitPointing(pp.JD, _imageStars(rows), catalog_stars)
    _checkEntry(fitted.fit_record['pointing'], 'fitPointing')
    assert fitted.fit_record['distortion'] is None


def test_recalibration_records_the_pointing_and_keeps_the_distortion():

    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)
    config = makeConfig()

    # The distortion of the starting platepar was fitted by something else
    start = perturbed(pp)
    start.fit_record['distortion'] = {'model': ASTROMETRY_MODEL, 'commit': 'abc1234',
                                      'utc': '2026-11-01T00:00:00Z', 'by': 'SkyFit2'}

    # The recalibration (with the least squares refinement) refits only the pointing
    result = recalibrate(start, rows, catalog_stars, config)
    _checkEntry(result.fit_record['pointing'], 'recalibrateFF+LSQ')
    assert result.fit_record['distortion'] == start.fit_record['distortion']

    # Without the refinement the label is the search alone
    config.recalibration_lsq_refine = False
    result = recalibrate(start, rows, catalog_stars, config)
    _checkEntry(result.fit_record['pointing'], 'recalibrateFF')


def test_record_is_saved_and_loaded(tmp_path):

    pp = makePlatepar()
    pp.recordFit(pointing=True, distortion=True, fit_by='test save')
    pp.recordFit(pointing=True, fit_by='test pointing')

    # Written as JSON and read back
    path = str(tmp_path/'platepar_cmn2010.cal')
    pp.write(path)
    loaded = Platepar()
    loaded.read(path)

    assert loaded.fit_record == pp.fit_record
    assert loaded.fit_record['distortion']['by'] == 'test save'
    assert loaded.fit_record['pointing']['by'] == 'test pointing'
    assert loaded.version == 3
    assert loaded.astrometryModel() == ASTROMETRY_MODEL


def test_platepar_without_record_is_of_the_first_model(tmp_path):

    import json

    # A platepar file written before the record existed (version 2, no fit_record)
    pp = makePlatepar()
    pp.recordFit(pointing=True, distortion=True, fit_by='test')
    data = json.loads(pp.jsonStr())
    del data['fit_record']
    data['version'] = 2
    path = str(tmp_path/'platepar_cmn2010.cal')
    with open(path, 'w') as f:
        json.dump(data, f)

    loaded = Platepar()
    loaded.read(path)
    assert loaded.fit_record == {'distortion': None, 'pointing': None}
    assert loaded.astrometryModel() == 1

    # The same for a platepar pickled before the record existed (e.g. a SkyFit2 state file)
    old = copy.deepcopy(pp)
    del old.__dict__['fit_record']
    restored = pickle.loads(pickle.dumps(old))
    assert restored.fit_record == {'distortion': None, 'pointing': None}
    assert restored.astrometryModel() == 1


def test_malformed_records_are_not_known():

    # Entries which are not records, and a record without a valid model
    pp = Platepar()
    pp.fit_record = {'distortion': 'refit', 'pointing': None, 'other': {}}
    assert pp.astrometryModel() == 1
    pp.fit_record = {'distortion': {'model': 'two'}}
    assert pp.astrometryModel() == 1
    assert PlateparModule.normalFitRecord(None) == {'distortion': None, 'pointing': None}
    assert PlateparModule.normalFitRecord({'pointing': {'model': 2}, 'x': 1}) == {'distortion': None,
                                                                                  'pointing': {'model': 2}}


def test_transferred_pointing_keeps_the_record_of_its_fit():

    # The distortion of one platepar and the pointing of another, each with its own record
    distortion = makePlatepar()
    distortion.recordFit(pointing=True, distortion=True, fit_by='distortion fit')
    pointing = perturbed(makePlatepar())
    pointing.recordFit(pointing=True, fit_by='pointing fit')

    transferred, _ = transferPointing(distortion, pointing)
    assert transferred.fit_record['distortion'] == distortion.fit_record['distortion']
    assert transferred.fit_record['pointing'] == pointing.fit_record['pointing']


def test_git_commit_outside_a_repository(tmp_path):

    # A directory which is not in a git working tree, one which does not exist, and one inside the RMS working
    #   tree which is not its root
    assert gitCommit(str(tmp_path)) == 'unknown'
    assert gitCommit(str(tmp_path/'missing')) == 'unknown'
    assert gitCommit(os.path.join(getRmsRootDir(), 'RMS')) == 'unknown'

    # The commit of this process is read once
    assert rmsCommit() is rmsCommit()


@pytest.mark.skipif(shutil.which('git') is None, reason='git not installed')
def test_git_commit_and_changes_of_the_code(tmp_path):

    repo = str(tmp_path)

    def _git(*args):
        """ Run git in the test working tree, as a test user. """
        subprocess.check_call(['git', '-C', repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.com']
                              + list(args), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # A working tree with a code file and a configuration file
    with open(os.path.join(repo, 'code.py'), 'w') as f:
        f.write('x = 1\n')
    with open(os.path.join(repo, '.config'), 'w') as f:
        f.write('[System]\n')
    _git('init', '-q')
    _git('add', 'code.py', '.config')
    _git('commit', '-q', '-m', 'First')
    head = subprocess.check_output(['git', '-C', repo, 'rev-parse', '--short', 'HEAD']).decode().strip()

    # A clean tree, then an edited configuration (not code), then edited code
    assert gitCommit(repo) == head
    with open(os.path.join(repo, '.config'), 'w') as f:
        f.write('[System]\nstationID: XX0001\n')
    assert gitCommit(repo) == head
    with open(os.path.join(repo, 'code.py'), 'w') as f:
        f.write('x = 2\n')
    assert gitCommit(repo) == head + '+'


def test_warning_for_an_older_model_used_as_it_is(monkeypatch):

    # The warnings of the recalibration
    warnings = []
    monkeypatch.setattr(ApplyRecalibrate.log, 'warning', lambda message, *args: warnings.append(message))

    # A platepar fitted before the record existed is of the first model
    old = makePlatepar()
    assert ApplyRecalibrate.warnOlderAstrometryModel(old, 'FF_XX0001_20261114_010203_000_0000000.fits')
    assert len(warnings) == 1
    assert 'FF_XX0001_20261114_010203_000_0000000.fits' in warnings[0]
    assert 'model 1' in warnings[0]

    # Its pointing refitted with the current model does not change the model of the distortion
    old.recordFit(pointing=True, fit_by='recalibrateFF')
    assert ApplyRecalibrate.warnOlderAstrometryModel(old, 'FF_2')
    assert len(warnings) == 2

    # A platepar whose distortion was fitted with the current model needs no warning
    current = makePlatepar()
    current.recordFit(pointing=True, distortion=True, fit_by='SkyFit2')
    assert not ApplyRecalibrate.warnOlderAstrometryModel(current, 'FF_3')
    assert len(warnings) == 2
