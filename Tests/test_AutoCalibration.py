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


# The repository config and a platepar template
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_CONFIG = os.path.join(REPO_DIR, '.config')
TEMPLATE_PLATEPAR = os.path.join(REPO_DIR, 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')


def _stars(x, y, fwhm=2.0):
    """ CALSTARS rows of stars at the given positions. """

    # Only the positions, the intensity and the width are set
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    rows = np.zeros((len(x), 8))
    rows[:, ac.STAR_Y] = y
    rows[:, ac.STAR_X] = x
    rows[:, ac.STAR_INTENSITY] = 1000
    rows[:, ac.STAR_FWHM] = fwhm

    return rows


def _score(zone_share, zone_stars=50, recalibrated=10, chunks=10, dist=0.2, mag_stddev=0.08, mag_residual=0.0):
    """ A score with the given values of the zones. """

    # The same values in every zone and chunk, apart from the given match shares
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

    # A dense grid of positions over the image
    x, y = np.meshgrid(np.arange(0, width, 4) + 0.5, np.arange(0, height, 4) + 0.5)
    membership = ac.zoneMembership(x.ravel(), y.ravel(), width, height)
    rings, sides = membership[:4], membership[4:]

    assert np.all(np.sum(rings, axis=0) == 1)
    assert np.all(np.sum(sides, axis=0) <= 1)

    # The sides of the outer frame have a similar area (within a factor of 2)
    areas = np.sum(sides, axis=1)
    assert areas.min() > 0.5*areas.max()


def test_zone_of_the_corner_and_the_centre():
    """ The zones of the centre, a corner and the middle of the right edge. """

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
    """ The coverage of the stars counts only the cells which are mostly not masked. """

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

    # A clear chunk, one with few stars, and one with stars in only a part of the image
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
    """ A platepar which fits the stars in all zones passes. """

    ok, reasons = ac.judgeScore(_score([0.95]*8))
    assert ok, reasons


@pytest.mark.parametrize('kwargs, reason', [
    # A bad outer ring, few recalibrated chunks, stars far from the catalog, and bad photometry
    ({'zone_share': [0.95, 0.95, 0.95, 0.2, 0.95, 0.95, 0.95, 0.95]}, 'zone r75-100'),
    ({'zone_share': [0.95]*8, 'recalibrated': 5}, 'recalibrated'),
    ({'zone_share': [0.95]*8, 'dist': 0.6}, 'distance'),
    ({'zone_share': [0.95]*8, 'mag_stddev': 0.3}, 'photometric scatter'),
    ({'zone_share': [0.95]*8, 'mag_residual': 0.4}, 'photometric residual'),
    ({'zone_share': [0.95]*8, 'mag_residual': -0.3}, 'photometric residual'),
])
def test_judge_fails(kwargs, reason):
    """ A platepar fails if a zone, the recalibration, the distances or the photometry are bad. """

    # The platepar fails for the given reason
    ok, reasons = ac.judgeScore(_score(**kwargs))
    assert not ok
    assert any(reason in r for r in reasons), reasons


def test_zones_without_enough_stars_are_not_judged():
    """ A zone with too few stars doesn't fail the platepar. """

    score = _score([0.95, 0.95, 0.95, 0.0, 0.95, 0.95, 0.95, 0.95])
    score.zone_stars[3] = ac.MIN_ZONE_STARS - 1

    assert ac.judgeScore(score)[0]


def test_candidate_comparison():
    """ A candidate is used only if it passes and is clearly better without being worse in any zone. """

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

    # The template without the distortion
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

    # A different pointing (rotation and position)
    pointed = copy.deepcopy(platepar)
    pointed.pos_angle_ref += 0.5
    pointed.RA_d += 0.2
    assert ac.sectionDifference(platepar, pointed, jd, 2.0)[1] < 0.2

    # A different distortion
    distorted = copy.deepcopy(platepar)
    distorted.x_poly_rev = distorted.x_poly_rev.copy()
    distorted.y_poly_rev = distorted.y_poly_rev.copy()
    distorted.x_poly_rev[5] += 0.05
    distorted.y_poly_rev[5] += 0.05
    assert ac.sectionDifference(platepar, distorted, jd, 2.0)[1] > ac.MAX_SECTION_DIFF_MAX


def test_agreeing_subset():
    """ The largest group of sections which all agree with each other. """

    # Sections 0, 1 and 3 agree with each other, 2 with none
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

    # A 512x512 camera with the automatic calibration
    config = cr.parse(REPO_CONFIG)
    config.width, config.height = 512, 512
    config.monitor_auto_calibration = True
    output_dir = str(tmp_path/'out')
    os.makedirs(output_dir)

    rng = np.random.default_rng(2)
    state = {'jd': 2461000.5, 'candidate_share': 0.95, 'fits': 0}

    def _readChunks(results_dir, config):
        """ The chunks of a file at the current time of the state. """

        # Ten chunks of 150 stars, a minute apart, from the time of the file
        jd = state['jd']
        return [('FF_XX0001_{:d}'.format(i), jd + i/1440.0, _stars(rng.uniform(0, 512, 150),
                                                                   rng.uniform(0, 512, 150)))
                for i in range(10)], 256

    def _scorePlatepar(platepar, chunks, catalog_stars, config, **kwargs):
        """ The score of the share stored in the mag_lev of the platepar, in the outer ring. """

        share = platepar.mag_lev
        return _score([share, 0.95, 0.95, share, 0.95, 0.95, 0.95, 0.95])

    def _fitCandidate(chunk_dirs, section_chunks, held_out, catalog_stars, config, **kwargs):
        """ A candidate with the share of state['candidate_share']. """

        # The fits are counted with their options, the candidate has the share of state['candidate_share']
        state['fits'] += 1
        state['fit_kwargs'] = kwargs
        platepar = _platepar()
        platepar.mag_lev = state['candidate_share']
        return platepar, {'sections': []}

    # Replace the reading of the stars, the fit, the scoring and the catalog
    monkeypatch.setattr(ac, 'readChunks', _readChunks)
    monkeypatch.setattr(ac, 'scorePlatepar', _scorePlatepar)
    monkeypatch.setattr(ac, 'fitCandidate', _fitCandidate)
    monkeypatch.setattr(ac, '_loadCatalog', lambda config, jd: None)

    def _process(name, platepar_path=None):
        """ Run the automatic calibration on the next file, return its judgement. """

        # The results of the file, and the time moves 10 minutes to the next file
        results_dir = str(tmp_path/name)
        os.makedirs(results_dir, exist_ok=True)
        result = ac.autoCalibrateFile(results_dir, config, output_dir, platepar_path)
        state['jd'] += 10/1440.0
        return result

    return config, output_dir, state, _process


def _writePlatepar(path, share):
    """ Write a platepar which scores the given share in the camera fixture, return its path. """

    # The share the platepar scores is stored in its mag_lev
    platepar = _platepar()
    platepar.mag_lev = share
    platepar.write(path)

    return path


def test_first_platepar_is_installed(camera):
    """ Without a platepar, a candidate which passes is installed right away. """

    from RMS.MonitorNightReport import autoPlateparPath

    config, output_dir, state, process = camera

    # The candidate fitted on the first file is installed
    result = process('file1')
    assert result['action'] == 'installed the first platepar'
    assert result['installed'] == autoPlateparPath(output_dir, config)
    assert os.path.isfile(result['installed'])

    # The judgement is saved with the results of the file, and the state of the camera
    with open(os.path.join(os.path.dirname(output_dir), 'file1', ac.FILE_RESULT_NAME)) as f:
        assert json.load(f)['action'] == 'installed the first platepar'
    assert len(ac.readState(output_dir)['installs']) == 1


def test_good_platepar_is_kept_and_checked_periodically(camera, tmp_path):
    """ A good platepar is kept, and a better one is looked for every interval. """

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
    """ A failing platepar which is not far worse than the candidate is replaced by it once it is confirmed on
        later chunks. """

    from RMS.MonitorNightReport import autoPlateparPath

    config, output_dir, state, process = camera
    bad = _writePlatepar(str(tmp_path/'bad.cal'), 0.75)

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

    # Confirmed on the second file after the fit, the saved candidate is removed
    pending_path = ac.readState(output_dir)['pending']['path']
    result = process('file3', bad)
    assert result['action'] == 'installed the confirmed candidate'
    assert os.path.isfile(autoPlateparPath(output_dir, config))
    assert not os.path.isfile(pending_path)


def test_unconfirmed_candidate_is_dropped(camera, tmp_path, monkeypatch):
    """ A candidate which isn't better on the later chunks is dropped. """

    config, output_dir, state, process = camera
    bad = _writePlatepar(str(tmp_path/'bad.cal'), 0.75)

    process('file1', bad)
    assert ac.readState(output_dir)['pending'] is not None

    # On the next file the candidate fails (e.g. it was fitted on a sky which changed), so it is dropped
    pending_path = ac.readState(output_dir)['pending']['path']
    _writePlatepar(pending_path, 0.1)
    process('file2', bad)
    result = process('file3', bad)
    assert result['pending']['better'] is False
    assert result['action'] != 'installed the confirmed candidate'

    # As the platepar still fails, a new candidate is fitted in its place
    assert state['fits'] == 2
    assert ac.readState(output_dir)['pending']['fitted_jd'] > state['jd'] - 15/1440.0


def test_dry_run_installs_nothing(camera):
    """ A dry run saves the candidate only, the camera keeps running without a platepar. """

    from RMS.MonitorNightReport import autoPlateparPath

    config, output_dir, state, process = camera

    # A file processed in a dry run, without a platepar
    results_dir = os.path.join(os.path.dirname(output_dir), 'file1')
    os.makedirs(results_dir)
    result = ac.autoCalibrateFile(results_dir, config, output_dir, None, dry_run=True)

    # Nothing is installed, the candidate is saved only
    assert result['installed'] is None
    assert result['action'] == 'dry run: installed the first platepar'
    assert not os.path.isfile(autoPlateparPath(output_dir, config))
    assert os.path.isfile(os.path.join(output_dir, 'auto_candidate_' + config.platepar_name))
    assert ac.readState(output_dir)['installs'][-1]['dry_run'] is True


def test_dry_run_judges_its_candidate_as_installed(camera):
    """ A dry run takes its saved candidate as the platepar in use, so it decides as it would with the candidate
        installed, and doesn't fit a new first platepar on every file. """

    from RMS.MonitorNightReport import autoPlateparPath

    config, output_dir, state, process = camera

    for i in range(3):
        results_dir = os.path.join(os.path.dirname(output_dir), 'file{:d}'.format(i))
        os.makedirs(results_dir)
        result = ac.autoCalibrateFile(results_dir, config, output_dir, None, dry_run=True)
        state['jd'] += 10/1440.0

    # The candidate of the first file is judged on the next ones, nothing is fitted again within the interval
    assert result['action'] == 'judged'
    assert result['current']['ok'] is True
    assert state['fits'] == 1
    assert not os.path.isfile(autoPlateparPath(output_dir, config))


def test_terrible_platepar_is_replaced_on_trial_and_kept(camera, tmp_path):
    """ A platepar which is far worse than the candidate is replaced right away, and the new platepar is kept once
        it passes on the chunks of the following CONFIRM_MINUTES. """

    from RMS.MonitorNightReport import autoPlateparPath

    config, output_dir, state, process = camera
    bad = _writePlatepar(str(tmp_path/'bad.cal'), 0.2)

    # Installed on trial right away, the replaced platepar is kept for a roll back, and its version reported so the
    #   files calibrated with it are calibrated again
    result = process('file1', bad)
    assert result['action'] == 'installed the platepar on trial'
    assert result['replaced_platepar_mtime'] == pytest.approx(os.path.getmtime(bad))
    assert result['installed'] == autoPlateparPath(output_dir, config)
    trial = ac.readState(output_dir)['trial']
    assert os.path.isfile(trial['previous'])

    # A file still processed with the replaced platepar (by another worker) judges the trial platepar, nothing is
    #   fitted while it is tested
    result = process('file2', bad)
    assert result['action'] == 'the platepar on trial is tested'
    assert result['current']['ok'] is True
    assert state['fits'] == 1

    # It passes on the chunks after the fit, so it is kept
    result = process('file3', autoPlateparPath(output_dir, config))
    assert result['action'] == 'kept the platepar on trial'
    assert ac.readState(output_dir)['trial'] is None
    assert not os.path.isfile(trial['previous'])
    with open(autoPlateparPath(output_dir, config)) as f:
        assert json.load(f)['mag_lev'] == pytest.approx(0.95)


def test_failing_trial_platepar_is_rolled_back(camera, tmp_path):
    """ A platepar on trial which fails on the following chunks, where the replaced one is clearly better, is
        replaced by the previous platepar again. """

    from RMS.MonitorNightReport import autoPlateparPath

    config, output_dir, state, process = camera
    bad = _writePlatepar(str(tmp_path/'bad.cal'), 0.2)
    process('file1', bad)
    trial = ac.readState(output_dir)['trial']

    # The sky changes: the trial platepar fails on the later chunks, while the previous one fits them
    _writePlatepar(autoPlateparPath(output_dir, config), 0.1)
    trial_mtime = os.path.getmtime(autoPlateparPath(output_dir, config))
    _writePlatepar(trial['previous'], 0.95)
    process('file2', bad)
    result = process('file3', bad)

    # The previous platepar is put back, and the version of the rejected platepar is reported, so the files
    #   calibrated with it are calibrated again
    assert result['action'] == 'rolled back the platepar on trial'
    assert result['replaced_platepar_mtime'] == pytest.approx(trial_mtime)
    assert result['installed'] == autoPlateparPath(output_dir, config)
    with open(autoPlateparPath(output_dir, config)) as f:
        assert json.load(f)['mag_lev'] == pytest.approx(0.95)
    assert ac.readState(output_dir)['trial'] is None


def test_failing_trial_platepar_without_a_better_previous_is_refitted(camera, tmp_path):
    """ A platepar on trial which fails, while the replaced one is no better, is replaced by a new fit. """

    from RMS.MonitorNightReport import autoPlateparPath

    config, output_dir, state, process = camera
    bad = _writePlatepar(str(tmp_path/'bad.cal'), 0.2)
    process('file1', bad)

    # The trial platepar fails on the later chunks, the previous one too
    _writePlatepar(autoPlateparPath(output_dir, config), 0.1)
    process('file2', bad)
    result = process('file3', bad)

    # A new candidate is fitted, far better than the failing trial platepar, so it goes on trial
    assert result['trial']['kept'] is False
    assert result['action'] == 'installed the platepar on trial'
    assert state['fits'] == 2


def test_not_enough_clear_sky(camera, monkeypatch):
    """ Without enough clear chunks nothing is judged or fitted. """

    config, output_dir, state, process = camera
    monkeypatch.setattr(ac, 'clearChunks', lambda *args, **kwargs: [])

    assert process('file1')['action'] == 'not enough clear sky'
    assert state['fits'] == 0


def test_installing_keeps_the_replaced_platepar(tmp_path):
    """ Installing a platepar keeps the one it replaces. """

    from RMS.MonitorNightReport import autoPlateparPath

    config = cr.parse(REPO_CONFIG)
    output_dir = str(tmp_path)

    ac.installPlatepar(_platepar(), output_dir, config, 'first')
    ac.installPlatepar(_platepar(), output_dir, config, 'second')

    assert os.path.isfile(autoPlateparPath(output_dir, config))
    assert len([name for name in os.listdir(output_dir) if name.startswith('auto_replaced_')]) == 1


def test_damaged_state_starts_over(tmp_path):
    """ A damaged state file is replaced by a new state. """

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
        """ Process a file of a minute. """

        # The next file begins a minute later
        result = process(name)
        state['jd'] -= 9/1440.0
        return result

    # The first attempt is made with 8 clear chunks, and fails
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
    """ A candidate whose photometry is worse in a zone is not used. """

    current = _score([0.95, 0.95, 0.95, 0.8, 0.95, 0.95, 0.95, 0.95], mag_residual=0.05)
    candidate = _score([0.95]*8, mag_residual=0.2)

    # The candidate is better in the worst zone but its photometry is worse
    better, reason = ac.candidateIsBetter(candidate, current)
    assert not better
    assert 'photometry' in reason


@pytest.mark.parametrize('current_worst, current_share, far_better', [
    (0.6, 0.6, True),      # The candidate matches 1.58 times the stars of the worst zone
    (0.75, 0.75, False),   # 1.27 times: the candidate waits for the confirmation
    (0.75, 0.5, True),     # 1.9 times the stars over the whole image (e.g. chunks not recalibrated)
    (0.0, 0.0, True),      # Nothing matches
])
def test_far_better_candidate(current_worst, current_share, far_better):
    """ A candidate is installed on trial only if it matches TRIAL_MATCH_RATIO times the share of the stars of a
        failing platepar, over the whole image or in the worst zone. """

    candidate = _score([0.95]*8)
    candidate.share = 0.95
    current = _score([current_worst] + [0.95]*7)
    current.share = current_share

    assert ac.candidateIsFarBetter(candidate, current) is far_better


def test_passing_platepar_is_never_replaced_on_trial():
    """ A platepar which passes is not replaced on trial, however much better the candidate is. """

    candidate = _score([0.95]*8)
    candidate.share = 0.95
    current = _score([0.5]*8)
    current.share = 0.5

    assert ac.judgeScore(current)[0]
    assert not ac.candidateIsFarBetter(candidate, current)


def test_older_file_finishing_late_makes_no_decision(camera, tmp_path):
    """ A file older than the last decision (a worker which finished late) is judged and its chunks are kept, but
        no platepar is fitted, installed or put on trial from its older window. """

    config, output_dir, state, process = camera
    good = _writePlatepar(str(tmp_path/'good.cal'), 0.95)

    # A newer file makes the periodic attempt
    state['jd'] += 30/1440.0
    process('newer', good)
    decided = ac.readState(output_dir)['last_attempt_jd']
    assert state['fits'] == 1

    # An older file finishes later: judged only, the state of the decisions doesn't go back
    state['jd'] -= 30/1440.0
    result = process('older', good)
    assert result['action'] == 'judged'
    assert state['fits'] == 1
    assert ac.readState(output_dir)['last_attempt_jd'] == decided


def test_fit_is_hinted_with_the_pointing_of_a_passing_platepar(camera, tmp_path):
    """ The periodic fit of a camera whose platepar passes searches only around its pointing (much faster); a fit
        without a platepar or for a failing one searches the whole sky, so a camera which moved is still found. """

    config, output_dir, state, process = camera

    # Without a platepar: no hint
    process('file0')
    assert state['fit_kwargs'].get('position_hint') is None

    # A passing platepar: a hint around its pointing, with a radius of a few fields of view
    good = _writePlatepar(str(tmp_path/'good.cal'), 0.95)
    state['jd'] += 60/1440.0
    process('file1', good)
    hint = state['fit_kwargs'].get('position_hint')
    assert hint is not None
    ra, dec, radius = hint
    assert (0 <= ra < 360) and (-90 <= dec <= 90) and (radius > 10)

    # A failing platepar: no hint
    bad = _writePlatepar(str(tmp_path/'bad.cal'), 0.75)
    state['jd'] += 60/1440.0
    process('file2', bad)
    assert state['fit_kwargs'].get('position_hint') is None


def test_installed_platepar_never_missing_during_an_install(tmp_path, monkeypatch):
    """ While a platepar replaces the installed one, the installed path always exists (other workers read it
        without the lock). """

    import RMS.MonitorNightReport as mnr
    from RMS.MonitorNightReport import autoPlateparPath

    config = cr.parse(REPO_CONFIG)
    output_dir = str(tmp_path)
    ac.installPlatepar(_platepar(), output_dir, config, 'first')

    # The write of the new platepar finds the old one still in place
    seen = []
    real_write = mnr._writePlatepar
    monkeypatch.setattr(mnr, '_writePlatepar', lambda pp, path: (seen.append(os.path.isfile(path)),
                                                                 real_write(pp, path)))
    ac.installPlatepar(_platepar(), output_dir, config, 'second')

    assert seen == [True]
    assert os.path.isfile(autoPlateparPath(output_dir, config))
    assert len([name for name in os.listdir(output_dir) if name.startswith('auto_replaced_')]) == 1


def test_reprocessing_an_older_night_still_fits(camera):
    """ A file much older than the last attempt (an older night processed again) is not a late worker: without a
        platepar a candidate is fitted for it. """

    config, output_dir, state, process = camera
    state['candidate_share'] = 0.2

    # The last attempt is from a file a day later
    process('newer')
    assert state['fits'] == 1
    state['jd'] -= 1.0
    process('older')
    assert state['fits'] == 2


def test_unreadable_trial_platepar_ends_the_trial(camera, tmp_path):
    """ A platepar on trial which can't be read ends the trial instead of failing every following file. """

    from RMS.MonitorNightReport import autoPlateparPath

    config, output_dir, state, process = camera
    bad = _writePlatepar(str(tmp_path/'bad.cal'), 0.2)
    process('file1', bad)
    assert ac.readState(output_dir)['trial'] is not None

    with open(autoPlateparPath(output_dir, config), 'w') as f:
        f.write('{damaged')
    process('file2', bad)
    assert ac.readState(output_dir)['trial'] is None


def test_mask_drops_mostly_masked_zones_and_the_mask_edge():
    """ With a mask, a zone mostly under the mask is not judged, and detections next to the mask (e.g. lit branches
        of a tree, half-hidden stars) are not counted, so a good platepar doesn't fail on them. """

    import cv2

    # A platepar and a chunk of stars which all match (the catalog is made from the stars themselves)
    pp = _platepar()
    pp.X_res, pp.Y_res = 512, 512
    config = cr.parse(REPO_CONFIG)
    rng = np.random.default_rng(1)
    x, y = rng.uniform(5, 507, 800), rng.uniform(5, 507, 800)
    stars = _stars(x, y, fwhm=2.0)

    # The bottom right corner is masked, and a tree on the left side with unmatched detections along its edge
    mask = np.full((512, 512), 255, np.uint8)
    mask[380:, 380:] = 0
    cv2.circle(mask, (40, 256), 60, 0, -1)
    keep = mask[y.astype(int), x.astype(int)] > 0
    stars = stars[keep]
    edge = cv2.dilate((mask == 0).astype(np.uint8), np.ones((5, 5), np.uint8)) & (mask > 0)
    ey, ex = np.nonzero(edge)
    k = rng.choice(len(ex), 150, replace=False)
    spurious = _stars(ex[k] + 0.3, ey[k] + 0.3, fwhm=2.0)

    # Matching: every real star is matched, the spurious ones are not
    def _match(platepar, jd, chunk_stars, catalog_stars, lim_mag, match_radius):
        n = len(chunk_stars)
        dist = np.where(np.arange(n) < len(stars), 0.1, np.inf)
        return dist, np.zeros(n), np.zeros(n), np.zeros(n)

    import unittest.mock
    with unittest.mock.patch.object(ac, 'matchChunk', _match):
        chunks = [('FF_XX0001_20251225_010000_000_0000000.fits', 2461000.5, np.vstack([stars, spurious]))]
        without = ac.scorePlatepar(pp, chunks, None, config, recalibrate=False)
        masked = ac.scorePlatepar(pp, chunks, None, config, recalibrate=False, mask=mask)

    left = without.zones.index('side_left')
    assert without.zone_share[left] < 0.8
    assert masked.zone_share[left] > 0.99

    # The masked corner: the outer ring there is mostly masked, but the ring has other corners; a zone which is
    #   mostly masked is not judged
    mask2 = np.full((512, 512), 255, np.uint8)
    mask2[256:, :] = 0
    mask2[:, :180] = 0
    with unittest.mock.patch.object(ac, 'matchChunk', _match):
        masked2 = ac.scorePlatepar(pp, [chunks[0][:2] + (stars,)], None, config, recalibrate=False, mask=mask2)
    assert not masked2.judged()[masked2.zones.index('side_bottom')]
    assert not masked2.judged()[masked2.zones.index('side_left')]


def test_sections_which_fail_on_held_out_chunks_are_dropped(monkeypatch):
    """ A section fitted to a wrong solution (it can pair many stars of its own chunk) fails on the held-out chunks
        and is dropped before the agreement: two sections which pass and agree give the candidate, one wrong fit no
        longer blocks it. """

    import RMS.Astrometry.AutoPlatepar as ap

    config = cr.parse(REPO_CONFIG)
    chunks = [('FF_XX0001_20251225_0100{:02d}_000_0000000.fits'.format(i), 2461000.5 + i/1440.0,
               _stars(np.linspace(10, 500, 60), np.linspace(10, 500, 60))) for i in range(8)]

    # Sections 0 and 3 are wrong fits (their mag_lev marks them), 1 and 2 are good
    calls = {'n': 0}

    def _fit(dir_path, config, catalog_stars, ff_name=None, **kwargs):
        pp = _platepar()
        pp.mag_lev = 0.05 if calls['n'] in (0, 3) else 0.95
        calls['n'] += 1
        return pp, list(range(80)), ff_name

    monkeypatch.setattr(ap, 'autoFitPlatepar', _fit)
    # A wrong fit matches few stars on the held-out chunks, which can't be recalibrated with it
    monkeypatch.setattr(ac, 'scorePlatepar', lambda pp, *a, **k: _score([pp.mag_lev]*8,
                                                                        recalibrated=(0 if pp.mag_lev < 0.5 else 10)))
    monkeypatch.setattr(ac, 'sectionDifference', lambda a, b, jd, fwhm: (0.0, 0.0) if a.mag_lev == b.mag_lev
                        else (5.0, 10.0))
    monkeypatch.setattr(ac, 'fitPhotometry', lambda *a, **k: {})

    candidate, info = ac.fitCandidate({c[0]: '.' for c in chunks}, chunks, chunks[:2], None, config)

    assert candidate is not None, info
    assert candidate.mag_lev == pytest.approx(0.95)
    assert len(info['agreeing_sections']) == 2
    assert sum(1 for s in info['sections'] if s.get('held_out_ok') is False) == 2


def test_sections_are_judged_with_their_own_photometry(monkeypatch):
    """ A new camera: the sections come from the plate solver with the default vignetting coefficient, which can be
        far from the camera's (e.g. a wide 1080p camera without a flat). Every section's photometry, with its
        vignetting, is fitted before it is judged on the held-out chunks, as the candidate is; otherwise every
        section fails on the photometry alone. """

    import RMS.Astrometry.AutoPlatepar as ap

    config = cr.parse(REPO_CONFIG)
    chunks = [('FF_XX0001_20251225_0100{:02d}_000_0000000.fits'.format(i), 2461000.5 + i/1440.0,
               _stars(np.linspace(10, 500, 60), np.linspace(10, 500, 60))) for i in range(8)]

    # The plate solver gives every section the default vignetting coefficient
    def _fit(dir_path, config, catalog_stars, ff_name=None, **kwargs):
        pp = _platepar()
        pp.vignetting_coeff = 0.00067
        return pp, list(range(80)), ff_name

    monkeypatch.setattr(ap, 'autoFitPlatepar', _fit)

    # The fitted photometry finds the camera's coefficient, with which the photometry passes; with the default one
    #   the photometric residual of the outer zone fails the judgement
    fitted = []

    def _photometry(platepar, chunks, catalog_stars, config, ff_frames=256, fit_vignetting=False):
        if fit_vignetting:
            platepar.vignetting_coeff = 0.00027
        fitted.append(fit_vignetting)
        return {'vignetting_coeff': platepar.vignetting_coeff, 'vignetting_fitted': fit_vignetting}

    def _scorePlatepar(pp, *args, **kwargs):
        score = _score([0.95]*8)
        if pp.vignetting_coeff != 0.00027:
            score.zone_mag_residual = np.full(len(score.zones), -0.57)
        return score

    monkeypatch.setattr(ac, 'fitPhotometry', _photometry)
    monkeypatch.setattr(ac, 'scorePlatepar', _scorePlatepar)
    monkeypatch.setattr(ac, 'sectionDifference', lambda a, b, jd, fwhm: (0.0, 0.0))

    candidate, info = ac.fitCandidate({c[0]: '.' for c in chunks}, chunks, chunks[:2], None, config,
                                      fit_vignetting=True)

    # Every section passes with its own photometry, and the candidate keeps the fitted coefficient
    assert candidate is not None, info
    assert all(s.get('held_out_ok') for s in info['sections'] if s['fitted'])
    assert candidate.vignetting_coeff == pytest.approx(0.00027)
    assert all(fitted)
