""" Tests of the automatic calibration (RMS.Astrometry.AutoCalibration): the zones of the image, the clear sky, the
    judgement of a platepar, the agreement of the sections and the decisions of the controller. """

from __future__ import print_function, division, absolute_import

import copy
import json
import os

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.Astrometry.AutoCalibration as ac
from RMS.Formats.Platepar import Platepar


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_CONFIG = os.path.join(REPO_DIR, '.config')
TEMPLATE_PLATEPAR = os.path.join(REPO_DIR, 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')


def _stars(x, y, fwhm=2.0):
    """ CALSTARS rows of stars at the given positions. """

    x, y = np.atleast_1d(x), np.atleast_1d(y)
    rows = np.zeros((len(x), 8))
    rows[:, ac.STAR_Y] = y
    rows[:, ac.STAR_X] = x
    rows[:, ac.STAR_INTENSITY] = 1000
    rows[:, ac.STAR_FWHM] = fwhm

    return rows


def _score(zone_share, zone_stars=50, recalibrated=10, chunks=10, dist=0.2, mag_stddev=0.08, mag_residual=0.0):
    """ A score with the given values of the zones. """

    score = ac.CalibrationScore()
    score.zone_share = np.array(zone_share, dtype=np.float64)
    score.zone_stars = np.full(len(score.zones), zone_stars)
    score.zone_mag_residual = np.full(len(score.zones), mag_residual)
    score.recalibrated = recalibrated
    score.chunks = chunks
    score.dist = dist
    score.mag_lev = np.full(chunks, 10.0)
    score.mag_lev_stddev = np.full(chunks, mag_stddev)

    return score


### Zones ###

@pytest.mark.parametrize('width, height', [(512, 512), (1920, 1080)])
def test_zones_cover_the_image(width, height):
    """ Every position is in one ring, the positions of the outer frame are in one side, and the sides have a similar
        area for any aspect ratio. """

    x, y = np.meshgrid(np.arange(0, width, 4) + 0.5, np.arange(0, height, 4) + 0.5)
    membership = ac.zoneMembership(x.ravel(), y.ravel(), width, height)
    rings, sides = membership[:4], membership[4:]

    assert np.all(np.sum(rings, axis=0) == 1)
    assert np.all(np.sum(sides, axis=0) <= 1)

    # The sides of the outer frame have a similar area (within a factor of 2)
    areas = np.sum(sides, axis=1)
    assert areas.min() > 0.5*areas.max()


def test_zone_of_the_corner_and_the_centre():

    membership = ac.zoneMembership([256, 2, 510], [256, 2, 256], 512, 512)
    names = ac.zoneNames()

    # The centre is in the inner ring, the corner in the outer ring and on the left or the top side, the middle of
    #   the right edge on the right side
    assert membership[names.index('r0-25'), 0]
    assert membership[names.index('r75-100'), 1]
    assert membership[names.index('side_left'), 1] or membership[names.index('side_top'), 1]
    assert membership[names.index('side_right'), 2]


### Clear sky ###

def test_coverage_counts_only_unmasked_cells():

    # Stars only in the left half
    x, y = np.meshgrid(np.linspace(10, 250, 8), np.linspace(10, 500, 8))
    stars = _stars(x.ravel(), y.ravel())
    assert ac.chunkCoverage(stars, 512, 512) < 0.6

    # With the right half masked, the stars cover the whole usable image
    mask = np.ones((512, 512), np.uint8)
    mask[:, 256:] = 0
    assert ac.chunkCoverage(stars, 512, 512, mask=mask) == pytest.approx(1.0)


def test_clear_chunks():
    """ A chunk is clear if it has enough stars compared to the clearest chunks, which cover the image. """

    rng = np.random.default_rng(1)
    full = _stars(rng.uniform(0, 512, 200), rng.uniform(0, 512, 200))
    cloudy = full[:40]
    half = full[full[:, ac.STAR_X] < 200]

    chunks = [('a', 1.0, full), ('b', 2.0, full), ('c', 3.0, cloudy), ('d', 4.0, half)]
    assert [name for name, _, _ in ac.clearChunks(chunks, 512, 512)] == ['a', 'b']

    # Against the reference of the camera, a sky with fewer stars than usual is not clear
    assert ac.clearChunks(chunks, 512, 512, reference_stars=500) == []


### Judgement ###

def test_judge_passes_a_platepar_which_fits_everywhere():

    ok, reasons = ac.judgeScore(_score([0.95]*8))
    assert ok, reasons


@pytest.mark.parametrize('kwargs, reason', [
    ({'zone_share': [0.95, 0.95, 0.95, 0.2, 0.95, 0.95, 0.95, 0.95]}, 'zone r75-100'),
    ({'zone_share': [0.95]*8, 'recalibrated': 5}, 'recalibrated'),
    ({'zone_share': [0.95]*8, 'dist': 0.6}, 'distance'),
    ({'zone_share': [0.95]*8, 'mag_stddev': 0.3}, 'photometric scatter'),
    ({'zone_share': [0.95]*8, 'mag_residual': 0.4}, 'photometric residual'),
    ({'zone_share': [0.95]*8, 'mag_residual': -0.3}, 'photometric residual'),
])
def test_judge_fails(kwargs, reason):

    ok, reasons = ac.judgeScore(_score(**kwargs))
    assert not ok
    assert any(reason in r for r in reasons), reasons


def test_zones_without_enough_stars_are_not_judged():

    score = _score([0.95, 0.95, 0.95, 0.0, 0.95, 0.95, 0.95, 0.95])
    score.zone_stars[3] = ac.MIN_ZONE_STARS - 1

    assert ac.judgeScore(score)[0]


def test_candidate_comparison():

    good = _score([0.95]*8)
    edge = _score([0.95, 0.95, 0.9, 0.3, 0.8, 0.8, 0.8, 0.8])

    # Without a platepar, and against a failing platepar, a passing candidate is better
    assert ac.candidateIsBetter(good, None)[0]
    assert ac.candidateIsBetter(good, edge)[0]

    # A candidate which is about as good is not used, nor one which fails
    assert not ac.candidateIsBetter(copy.deepcopy(good), good)[0]
    assert not ac.candidateIsBetter(edge, None)[0]

    # A candidate which is worse in a zone is not used even if its worst zone is better
    current = _score([0.95, 0.95, 0.95, 0.8, 0.95, 0.95, 0.95, 0.95])
    candidate = _score([0.95, 0.85, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95])
    assert not ac.candidateIsBetter(candidate, current)[0]

    # A clearly better worst zone without any worse zone is better
    candidate = _score([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95])
    assert ac.candidateIsBetter(candidate, current)[0]


### Agreement of the sections ###

def _platepar():
    """ A platepar without distortion, so its forward and reverse mappings agree exactly (the forward and reverse
        distortion of the templates are not fitted to each other). """

    platepar = Platepar()
    assert platepar.read(TEMPLATE_PLATEPAR) is not False
    for name in ['x_poly_fwd', 'y_poly_fwd', 'x_poly_rev', 'y_poly_rev']:
        setattr(platepar, name, np.zeros_like(getattr(platepar, name)))

    # A station and a pointing high in the sky (the template points at the horizon from the equator)
    platepar.lat, platepar.lon = 45.0, -80.0
    platepar.az_centre, platepar.alt_centre = 180.0, 60.0
    platepar.updateRefRADec()

    return platepar


def test_section_difference_ignores_the_pointing():
    """ Platepars which differ only in the pointing agree (the recalibration refits it), a different distortion
        doesn't. """

    platepar = _platepar()
    jd = platepar.JD

    assert ac.sectionDifference(platepar, platepar, jd, 2.0)[1] < 0.01

    pointed = copy.deepcopy(platepar)
    pointed.pos_angle_ref += 0.5
    pointed.RA_d += 0.2
    assert ac.sectionDifference(platepar, pointed, jd, 2.0)[1] < 0.2

    distorted = copy.deepcopy(platepar)
    distorted.x_poly_rev = distorted.x_poly_rev.copy()
    distorted.y_poly_rev = distorted.y_poly_rev.copy()
    distorted.x_poly_rev[5] += 0.05
    distorted.y_poly_rev[5] += 0.05
    assert ac.sectionDifference(platepar, distorted, jd, 2.0)[1] > ac.MAX_SECTION_DIFF_MAX


def test_agreeing_subset():

    agree = [[True, True, False, True],
             [True, True, False, True],
             [False, False, True, False],
             [True, True, False, True]]

    assert ac._agreeingSubset(4, agree) == [0, 1, 3]


### Controller ###

@pytest.fixture
def camera(tmp_path, monkeypatch):
    """ A camera with an output directory, and a file whose clear chunks are given directly. The fit and the scoring
        of platepars are replaced: the platepars carry their score in their mag_lev (the worst zone share). """

    config = cr.parse(REPO_CONFIG)
    config.width, config.height = 512, 512
    config.monitor_auto_calibration = True
    output_dir = str(tmp_path/'out')
    os.makedirs(output_dir)

    rng = np.random.default_rng(2)
    state = {'jd': 2461000.5, 'candidate_share': 0.95, 'fits': 0}

    def _readChunks(results_dir, config):
        jd = state['jd']
        return [('FF_XX0001_{:d}'.format(i), jd + i/1440.0, _stars(rng.uniform(0, 512, 150),
                                                                   rng.uniform(0, 512, 150)))
                for i in range(10)], 256

    def _scorePlatepar(platepar, chunks, catalog_stars, config, **kwargs):
        share = platepar.mag_lev
        return _score([share, 0.95, 0.95, share, 0.95, 0.95, 0.95, 0.95])

    def _fitCandidate(chunk_dirs, section_chunks, held_out, catalog_stars, config, **kwargs):
        state['fits'] += 1
        platepar = _platepar()
        platepar.mag_lev = state['candidate_share']
        return platepar, {'sections': []}

    monkeypatch.setattr(ac, 'readChunks', _readChunks)
    monkeypatch.setattr(ac, 'scorePlatepar', _scorePlatepar)
    monkeypatch.setattr(ac, 'fitCandidate', _fitCandidate)
    monkeypatch.setattr(ac, '_loadCatalog', lambda config, jd: None)

    def _process(name, platepar_path=None):
        results_dir = str(tmp_path/name)
        os.makedirs(results_dir, exist_ok=True)
        result = ac.autoCalibrateFile(results_dir, config, output_dir, platepar_path)
        state['jd'] += 10/1440.0
        return result

    return config, output_dir, state, _process


def _writePlatepar(path, share):

    platepar = _platepar()
    platepar.mag_lev = share
    platepar.write(path)

    return path


def test_first_platepar_is_installed(camera):

    from RMS.MonitorNightReport import autoPlateparPath

    config, output_dir, state, process = camera

    result = process('file1')
    assert result['action'] == 'installed the first platepar'
    assert result['installed'] == autoPlateparPath(output_dir, config)
    assert os.path.isfile(result['installed'])

    # The judgement is saved with the results of the file, and the state of the camera
    with open(os.path.join(os.path.dirname(output_dir), 'file1', ac.FILE_RESULT_NAME)) as f:
        assert json.load(f)['action'] == 'installed the first platepar'
    assert len(ac.readState(output_dir)['installs']) == 1


def test_good_platepar_is_kept_and_checked_periodically(camera, tmp_path):

    config, output_dir, state, process = camera
    good = _writePlatepar(str(tmp_path/'good.cal'), 0.95)

    # A candidate which is not better is not used
    assert process('file1', good)['action'] == 'candidate not better'

    # The next files are only judged until the interval of the periodic check has passed
    assert process('file2', good)['action'] == 'judged'
    assert process('file3', good)['action'] == 'judged'
    assert state['fits'] == 1
    process('file4', good)
    assert state['fits'] == 2


def test_failing_platepar_is_replaced_after_confirmation(camera, tmp_path):

    from RMS.MonitorNightReport import autoPlateparPath

    config, output_dir, state, process = camera
    bad = _writePlatepar(str(tmp_path/'bad.cal'), 0.2)

    # The candidate waits for the confirmation on a later file
    result = process('file1', bad)
    assert result['current']['ok'] is False
    assert result['action'] == 'candidate waits for the confirmation'
    assert not os.path.isfile(autoPlateparPath(output_dir, config))

    # No new candidate is fitted while one waits, and it is confirmed once there are 15 minutes of chunks after the
    #   fitted ones which it was not fitted on (the second file after it, with files of 10 minutes)
    result = process('file2', bad)
    assert result['action'] == 'candidate waits for the confirmation'
    assert state['fits'] == 1

    result = process('file3', bad)
    assert result['action'] == 'installed the confirmed candidate'
    assert os.path.isfile(autoPlateparPath(output_dir, config))


def test_unconfirmed_candidate_is_dropped(camera, tmp_path, monkeypatch):

    config, output_dir, state, process = camera
    bad = _writePlatepar(str(tmp_path/'bad.cal'), 0.2)

    process('file1', bad)
    assert ac.readState(output_dir)['pending'] is not None

    # On the next file the candidate fails (e.g. it was fitted on a sky which changed), so it is dropped
    pending_path = ac.readState(output_dir)['pending']['path']
    _writePlatepar(pending_path, 0.1)
    process('file2', bad)
    result = process('file3', bad)
    assert result['pending']['better'] is False
    assert result['action'] != 'installed the confirmed candidate'


def test_dry_run_installs_nothing(camera):

    from RMS.MonitorNightReport import autoPlateparPath

    config, output_dir, state, process = camera

    results_dir = os.path.join(os.path.dirname(output_dir), 'file1')
    os.makedirs(results_dir)
    result = ac.autoCalibrateFile(results_dir, config, output_dir, None, dry_run=True)

    assert result['installed'] is None
    assert not os.path.isfile(autoPlateparPath(output_dir, config))
    assert os.path.isfile(os.path.join(output_dir, 'auto_candidate_' + config.platepar_name))


def test_not_enough_clear_sky(camera, monkeypatch):

    config, output_dir, state, process = camera
    monkeypatch.setattr(ac, 'clearChunks', lambda *args, **kwargs: [])

    assert process('file1')['action'] == 'not enough clear sky'
    assert state['fits'] == 0


def test_installing_keeps_the_replaced_platepar(tmp_path):

    from RMS.MonitorNightReport import autoPlateparPath

    config = cr.parse(REPO_CONFIG)
    output_dir = str(tmp_path)

    ac.installPlatepar(_platepar(), output_dir, config, 'first')
    ac.installPlatepar(_platepar(), output_dir, config, 'second')

    assert os.path.isfile(autoPlateparPath(output_dir, config))
    assert len([name for name in os.listdir(output_dir) if name.startswith('auto_replaced_')]) == 1


def test_damaged_state_starts_over(tmp_path):

    with open(ac.statePath(str(tmp_path)), 'w') as f:
        f.write('{damaged')

    assert ac.readState(str(tmp_path))['installs'] == []


def test_failed_first_attempt_is_retried_with_more_clear_sky(camera, monkeypatch):
    """ Without a platepar, a failed attempt is retried as soon as the clear chunks in the window grew enough, not
        only after RETRY_MINUTES (e.g. short files of a new camera). """

    config, output_dir, state, process = camera

    # The candidate fails at first
    state['candidate_share'] = 0.2
    clear = {'n': 4}
    monkeypatch.setattr(ac, 'clearChunks', lambda chunks, *args, **kwargs: chunks[:clear['n']])

    # Files of a minute: 4 clear chunks each, the window grows by 4 per file
    def _short(name):
        result = process(name)
        state['jd'] -= 9/1440.0
        return result

    assert _short('file1')['action'] == 'not enough clear sky'
    assert _short('file2')['action'].startswith('candidate not better')
    assert state['fits'] == 1

    # 12 clear chunks are 1.5 times the 8 of the first attempt, so it is retried (and fails again), then 16 are
    #   not 1.5 times 12, but 20 are
    assert _short('file3')['action'].startswith('candidate not better')
    assert _short('file4')['action'] == 'judged'
    state['candidate_share'] = 0.95
    assert _short('file5')['action'] == 'installed the first platepar'
    assert state['fits'] == 3


def test_candidate_with_worse_photometry_is_not_used():

    current = _score([0.95, 0.95, 0.95, 0.8, 0.95, 0.95, 0.95, 0.95], mag_residual=0.05)
    candidate = _score([0.95]*8, mag_residual=0.2)

    better, reason = ac.candidateIsBetter(candidate, current)
    assert not better
    assert 'photometry' in reason
