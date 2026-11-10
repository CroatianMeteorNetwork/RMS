"""
Tests of the joint fit of the lens distortion on the stars of many nights (RMS.Astrometry.JointDistortionFit).

The synthetic nights are made with a known distortion (a real 1280x720 calibration, radial7-odd with the asymmetry
correction) and a different pointing on every night: catalog stars are put on the images of the chunks of every
night by that distortion with the night's pointing, measured with a noise, and written to a CALSTARS file in the
directory of the night. The platepar of every night has the night's pointing but a wrong distortion (the stars are
off by up to about a pixel at the corners), as does the starting platepar of the fit. The fit has to recover the
known distortion, which the nights of the fit and a held-out night show.

The pointing transfer carries the pointing of a platepar over to another distortion (transferPointing).
"""

import copy
import json
import logging
import os
import shutil

import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Astrometry import JointDistortionFit
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.AutoPlatepar import loadCatalogStars
from RMS.Astrometry.Conversions import date2JD, jd2Date, JD2HourAngle
from RMS.Astrometry.JointDistortionFit import (QC_DEFAULTS, heldOutNights, jointDistortionFit, mappingDifference,
                                               matchStars, modelSanity, qualityControl, readNights, readPlatepar,
                                               residualStats, transferPointing)
from RMS.Formats import CALSTARS
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Formats.Platepar import Platepar


# Noise of the measured star positions (px), and the faintest synthetic star
NOISE_PX = 0.1
LIM_MAG = 6.0

# Number of nights, chunks per night (minutes apart), and the held-out night
N_NIGHTS = 4
N_CHUNKS = 3
CHUNK_MINUTES = 30
HELD_OUT = 2

# The wrong distortion of the platepars of the nights: stars moved outwards by this much (px) at the corners
WRONG_DISTORTION_PX = 1.0

# The photometry of the starting platepar, which the fit has to keep (a vignetting coefficient which the fit must
#   neither fit nor reset)
START_MAG_LEV = 10.3
START_VIGNETTING = 0.0012


def makePlatepar():
    """ A real 1280x720 calibration, radial7-odd with the asymmetry correction and an equal aspect. """

    pp = Platepar()
    pp.lat, pp.lon, pp.elev = 50.192509, 5.254718, 295.0
    pp.X_res, pp.Y_res = 1280, 720
    pp.F_scale = 15.600916448923877
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = 349.8434689573242, -4.063815352866793, 19.34236856738213
    pp.JD, pp.Ho = 2461018.279334375, 358.3338455827907
    pp.refraction = True
    pp.equal_aspect, pp.asymmetry_corr, pp.force_distortion_centre = True, True, False
    pp.setDistortionType("radial7-odd", reset_params=True)
    pp.x_poly_fwd = np.array([-0.0017811686386760266, 0.031678484905771256, -0.007740143231304088,
                              -1.0140484087110697, 0.06835307986186301, 0.010258192855205479,
                              0.005386010898662939])
    pp.x_poly_rev = np.array([-0.0017856520990328477, 0.03170770867400316, 0.007074982440201784,
                              0.4820702423628897, 0.06802381881224333, -0.001903408962282129,
                              0.00014273818898309323])
    pp.y_poly_fwd = np.array([0.5, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    pp.y_poly_rev = np.array([0.5, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    pp.updateRefAltAz()
    pp.mag_lev = 10.5
    pp.vignetting_coeff = 0.0

    return pp


def makeConfig():
    """ The default configuration for a 1280x720 camera, with the catalog limit of the synthetic stars. """

    config = cr.Config()
    config.width, config.height = 1280, 720
    config.catalog_mag_limit = LIM_MAG
    config.use_flat = True

    return config


def withPointing(pp, jd, d_az, d_alt, d_rot, d_scale):
    """ The platepar with the pointing of another night: the camera turned by the given angles (deg) and the scale
        changed by the given share, at the reference time jd. """

    pp = copy.deepcopy(pp)
    rotation = pp.rotationWrtHorizon()
    pp.JD, pp.Ho = jd, JD2HourAngle(jd)
    pp.az_centre += d_az
    pp.alt_centre += d_alt
    pp.rotation_from_horiz = rotation + d_rot
    pp.updateRefRADec(preserve_rotation=True)
    pp.F_scale *= 1 + d_scale

    return pp


def grid(pp, n=25, margin=20):
    """ Image positions on a grid over the image (inside a margin). """

    xs, ys = np.meshgrid(np.linspace(margin, pp.X_res - margin, n), np.linspace(margin, pp.Y_res - margin, n))

    return xs.ravel(), ys.ravel()


def gridDifference(pa, pb):
    """ The largest distance (px) between the image positions with the two platepars of the sky positions of a grid
        over the image of platepar pa (at its reference time). """

    x, y = grid(pa)
    _, ra, dec, _ = xyToRaDecPP(np.full(x.size, pa.JD), x, y, np.ones(x.size), pa, extinction_correction=False,
                                jd_time=True)
    xa, ya = raDecToXYPP(np.asarray(ra), np.asarray(dec), pa.JD, pa)
    xb, yb = raDecToXYPP(np.asarray(ra), np.asarray(dec), pa.JD, pb)

    return np.max(np.hypot(xb - xa, yb - ya))


def wrongDistortion(pp):
    """ A platepar of the same type whose stars are moved outwards by up to WRONG_DISTORTION_PX at the corners:
        fitted on a grid of the platepar's stars with their image positions moved. """

    x, y = grid(pp, n=40)
    _, ra, dec, _ = xyToRaDecPP(np.full(x.size, pp.JD), x, y, np.ones(x.size), pp, extinction_correction=False,
                                jd_time=True)

    # The stars moved outwards by a share of the radius which grows with its square
    cx, cy = pp.X_res/2.0, pp.Y_res/2.0
    share = WRONG_DISTORTION_PX/np.hypot(cx, cy)*((x - cx)**2 + (y - cy)**2)/(cx**2 + cy**2)
    x_moved, y_moved = x + (x - cx)*share, y + (y - cy)*share

    wrong = copy.deepcopy(pp)
    wrong.fitAstrometry(pp.JD, np.column_stack([x_moved, y_moved, np.ones(x.size)]),
                        np.column_stack([ra, dec, np.full(x.size, 5.0)]))

    return wrong


def chunkName(jd):
    """ The name of an FF file which starts at the given time. """

    year, month, day, hour, minute, second, millisecond = jd2Date(jd)

    return "FF_XX0001_{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}_{:03d}_0000000.fits".format(
        year, month, day, hour, minute, int(second), int(millisecond))


def chunkStars(pp, jd, catalog_stars, rng, noise=NOISE_PX):
    """ The stars of a chunk as CALSTARS rows: the catalog stars on the image with the platepar, with the noise.

    Arguments:
        pp: [Platepar] Platepar of the chunk.
        jd: [float] Julian date of the chunk.
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag).
        rng: [Generator] Random generator of the noise.

    Keyword arguments:
        noise: [float] Noise of the positions (px). NOISE_PX by default.

    Return:
        [ndarray] The stars as CALSTARS rows.
    """

    # The catalog stars near the centre of the field of view
    _, ra_c, dec_c, _ = xyToRaDecPP([jd2Date(jd)], [pp.X_res/2], [pp.Y_res/2], [1], pp, extinction_correction=False)
    ra, dec, mag = catalog_stars[:, 0], catalog_stars[:, 1], catalog_stars[:, 2]
    cos_sep = np.sin(np.radians(dec))*np.sin(np.radians(dec_c[0])) \
        + np.cos(np.radians(dec))*np.cos(np.radians(dec_c[0]))*np.cos(np.radians(ra - ra_c[0]))
    near = cos_sep > np.cos(np.radians(60))
    ra, dec, mag = ra[near], dec[near], mag[near]

    # Their image positions, those inside the image, with the noise
    x, y = raDecToXYPP(ra, dec, jd, pp)
    inside = (x > 15) & (x < pp.X_res - 15) & (y > 15) & (y < pp.Y_res - 15)
    x, y, mag = x[inside], y[inside], mag[inside]
    x = x + rng.normal(0, noise, x.size)
    y = y + rng.normal(0, noise, y.size)

    # The intensities follow the photometry of the platepar
    intensity = 10**((pp.mag_lev - mag)/2.5)
    n = x.size

    return np.column_stack([y, x, intensity, intensity/10, np.full(n, 2.5), np.full(n, 10.0), np.full(n, 50.0),
                            np.zeros(n)])


def makeNights(tmp_path_factory, config, truth, start, prefix, n_nights=N_NIGHTS, n_chunks=N_CHUNKS,
               noise=NOISE_PX):
    """ Synthetic nights: a directory per night with a CALSTARS file of stars put on the images by the known
        distortion with the night's pointing, and a platepar with the night's pointing on the starting distortion.

    Arguments:
        tmp_path_factory: [TempPathFactory] Factory of the directories.
        config: [Config] Configuration.
        truth: [Platepar] The known distortion.
        start: [Platepar] The starting distortion.
        prefix: [str] Prefix of the names of the directories.

    Keyword arguments:
        n_nights: [int] Number of nights. N_NIGHTS by default.
        n_chunks: [int] Number of chunks of every night. N_CHUNKS by default.
        noise: [float] Noise of the star positions (px). NOISE_PX by default.

    Return:
        (dirs, pointings): the directories and the true platepar of every night.
    """

    catalog_stars = loadCatalogStars(config, LIM_MAG, jd=truth.JD)
    rng = np.random.default_rng(0)

    dirs, pointings = [], []
    for n in range(n_nights):

        # The pointing of the night, a few nights apart: the camera turned and refocused a little
        jd_night = truth.JD + 3*n
        pointing = withPointing(truth, jd_night, 0.3*n, -0.2*n, 0.1*n, 0.0005*n)

        # The chunks of the night, their stars at the middle of the chunk (the time of the recalibration)
        star_list = []
        for c in range(n_chunks):
            name = chunkName(jd_night + c*CHUNK_MINUTES/1440.0)
            jd = date2JD(*getMiddleTimeFF(name, config.fps, ret_milliseconds=True, ff_frames=256))
            star_list.append([name, chunkStars(pointing, jd, catalog_stars, rng, noise=noise).tolist()])

        # The directory of the night, with the night's pointing on the starting distortion
        dir_path = str(tmp_path_factory.mktemp("{:s}_{:d}".format(prefix, n)))
        CALSTARS.writeCALSTARS(star_list, dir_path, "CALSTARS_XX0001.txt", "XX0001", truth.Y_res, truth.X_res)
        transferPointing(start, pointing)[0].write(os.path.join(dir_path, config.platepar_name))

        dirs.append(dir_path)
        pointings.append(pointing)

    return dirs, pointings


@pytest.fixture(scope="module")
def nights(tmp_path_factory):
    """ The synthetic nights of a known distortion, with a wrong distortion in the platepars of the nights and in
        the starting platepar (see makeNights).

    Return:
        (config, truth, start, dirs, pointings): the configuration, the known distortion, the starting platepar
            (the wrong distortion), the directories and the true platepar of every night.
    """

    config = makeConfig()
    truth = makePlatepar()
    start = wrongDistortion(truth)
    start.mag_lev, start.vignetting_coeff = START_MAG_LEV, START_VIGNETTING
    dirs, pointings = makeNights(tmp_path_factory, config, truth, start, "night")

    return config, truth, start, dirs, pointings


@pytest.fixture(scope="module")
def fit(nights):
    """ The joint fit of the synthetic nights, one night held out. """

    config, truth, start, dirs, pointings = nights
    fitted, report = jointDistortionFit(dirs, start, config, distortion_type="radial7-odd",
                                        holdout=(N_NIGHTS, HELD_OUT), n_iter=2, chunk_step=1, nproc=4)

    return fitted, report


def test_transfer_onto_the_same_platepar():
    """ The pointing of a platepar carried over to itself doesn't change it. """

    pp = makePlatepar()
    transferred, rms = transferPointing(pp, pp)

    assert rms < 1e-4
    assert gridDifference(pp, transferred) < 1e-3
    assert abs(transferred.RA_d - pp.RA_d) < 1e-6
    assert abs(transferred.dec_d - pp.dec_d) < 1e-6


def test_transfer_recovers_a_pointing():
    """ The pointing of another night carried over to the same distortion is that night's pointing. """

    pp = makePlatepar()
    pointing = withPointing(pp, pp.JD + 10, 1.0, -0.5, 0.3, 0.002)

    # Start from the platepar with its own pointing: only the distortion is kept
    transferred, rms = transferPointing(pp, pointing)

    assert rms < 1e-3
    assert gridDifference(pointing, transferred) < 1e-2
    assert abs(transferred.F_scale/pointing.F_scale - 1) < 1e-6
    assert abs(transferred.pos_angle_ref - pointing.pos_angle_ref) < 1e-4


def test_transfer_onto_another_distortion():
    """ Carried over to a distortion with a different centre, the pointing puts the stars where the pointing
        platepar does, which copying the pointing parameters doesn't. """

    pp = makePlatepar()

    # The same lens with the distortion centre elsewhere, its own pointing fitted to the platepar
    other = copy.deepcopy(pp)
    for coeffs in (other.x_poly_fwd, other.x_poly_rev):
        coeffs[0] += 0.01
        coeffs[1] -= 0.01
    other, rms_models = transferPointing(other, pp)

    # Another night's pointing, carried over and copied
    pointing = withPointing(pp, pp.JD + 10, 1.0, -0.5, 0.3, 0.002)
    transferred, rms = transferPointing(other, pointing)
    copied = copy.deepcopy(other)
    for name in ('RA_d', 'dec_d', 'JD', 'Ho', 'pos_angle_ref', 'F_scale'):
        setattr(copied, name, getattr(pointing, name))

    # The transfer is as close as the two distortions allow (as on the night they were matched on), the copy is
    #   several times further off
    assert rms < 1.1*rms_models
    assert gridDifference(pointing, transferred) < 1.1*gridDifference(pp, other)
    assert gridDifference(pointing, copied) > 3*gridDifference(pointing, transferred)


def test_held_out_nights():
    """ Every N-th night from the K-th, or the nights of the given directories. """

    nights = [{'dir': 'night_{:d}'.format(n)} for n in range(10)]

    assert heldOutNights(nights, None) == set()
    assert heldOutNights(nights, (4, 1)) == {1, 5, 9}
    assert heldOutNights(nights, ['night_3', 'night_7']) == {3, 7}


def test_fit_recovers_the_distortion(nights, fit):
    """ The fitted distortion with the pointing of every night (also the held-out one) puts the stars where the
        known distortion does, much closer than the noise of the stars, while the starting one is far off. """

    config, truth, start, dirs, pointings = nights
    fitted, report = fit

    assert fitted.distortion_type == "radial7-odd"
    for pointing in pointings:
        assert gridDifference(pointing, transferPointing(start, pointing)[0]) > 3*NOISE_PX
        assert gridDifference(pointing, transferPointing(fitted, pointing)[0]) < 0.4*NOISE_PX


def test_residuals_of_the_fit(nights, fit):
    """ After the fit the residuals of the stars of the fit are the noise, also in the corners. """

    config, truth, start, dirs, pointings = nights
    fitted, report = fit

    # The rms distance of the noise in both coordinates (arcsec)
    noise = np.sqrt(2)*NOISE_PX*3600/truth.F_scale

    first, last = report['iterations'][0], report['iterations'][-1]
    assert last['recalibrated'] == report['chunks']
    assert last['stats']['rms'] < 1.15*noise
    assert last['stats']['rms'] < first['stats']['rms']
    for dx, dy, n in last['stats']['corners'].values():
        assert np.hypot(dx, dy) < 0.5*noise

    # The photometry of the starting platepar is kept (the recalibration of the reference chunk fitted another
    #   zero point)
    assert fitted.vignetting_coeff == START_VIGNETTING
    assert fitted.mag_lev == START_MAG_LEV


def test_held_out_night_fits(nights, fit):
    """ The held-out night fits the fitted distortion to the noise, better than its own platepar. """

    config, truth, start, dirs, pointings = nights
    fitted, report = fit

    noise = np.sqrt(2)*NOISE_PX*3600/truth.F_scale

    assert report['held_out'] == [dirs[HELD_OUT]]
    held = report['held_out_stats']
    assert held['fitted']['recalibrated'] == N_CHUNKS
    assert held['fitted']['stats']['rms'] < 1.15*noise
    assert held['fitted']['stats']['rms'] < held['start']['stats']['rms']


class _Records(logging.Handler):
    """ Keeps the messages of a logger. """

    def __init__(self):
        logging.Handler.__init__(self, level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture(scope="module")
def fit_poly(nights):
    """ The joint fit of the synthetic nights with another distortion type (poly3+radial5), without workers, with
        the messages of the log kept.

    Return:
        (fitted, report, messages, levels): the fitted platepar, the report, the messages of the log, and the level
            of the log before and after the fit.
    """

    config, truth, start, dirs, pointings = nights
    rms_log = logging.getLogger("rmslogger")
    previous_level = rms_log.level
    rms_log.setLevel(logging.INFO)
    records = _Records()
    rms_log.addHandler(records)
    try:
        fitted, report = jointDistortionFit(dirs, start, config, distortion_type="poly3+radial5", n_iter=1,
                                            chunk_step=1, nproc=1)
        levels = (logging.INFO, rms_log.level)
    finally:
        rms_log.removeHandler(records)
        rms_log.setLevel(previous_level)

    return fitted, report, records.messages, levels


def test_type_change_without_workers(nights, fit_poly):
    """ Another distortion type is fitted from no distortion and the change is logged; without workers the
        log keeps its level and every iteration is logged. """

    config, truth, start, dirs, pointings = nights
    fitted, report, messages, levels = fit_poly

    noise = np.sqrt(2)*NOISE_PX*3600/truth.F_scale

    # The new type fits the stars nearly to the noise (it can't describe the known distortion exactly)
    assert fitted.distortion_type == "poly3+radial5"
    assert report['distortion_type'] == "poly3+radial5"
    assert report['iterations'][-1]['stats']['rms'] < 1.2*noise
    for pointing in pointings:
        assert gridDifference(pointing, transferPointing(fitted, pointing)[0]) < 2*NOISE_PX

    # The change of the type and the iterations are logged, and the level of the log is the same after the fit
    assert any('changes from radial7-odd' in message for message in messages)
    assert any(message.startswith('Iteration 1') for message in messages)
    assert levels[0] == levels[1]

    # No data of the recalibration stays behind in this process
    assert not JointDistortionFit._worker_data


def test_cli_keeps_the_type_and_carries_the_pointing(nights, fit_poly, tmp_path, monkeypatch):
    """ The command line: without a type the one of the starting platepar is kept (not the config's), and the
        result is written with the pointing of the given platepar. """

    config, truth, start, dirs, pointings = nights
    fitted_poly = fit_poly[0]

    # The starting platepar of another type than the config's
    config = copy.deepcopy(config)
    config.platepar_distortion_type = "radial7-odd"
    monkeypatch.setattr(cr, "loadConfigFromDirectory", lambda *args: config)
    start_path = str(tmp_path/"start.cal")
    fitted_poly.write(start_path)

    out_path = str(tmp_path/"joint.cal")
    pointing_path = os.path.join(dirs[1], config.platepar_name)
    code = JointDistortionFit.main(dirs + ['-p', start_path, '-o', out_path, '--iter', '1', '--nproc', '2',
                                           '--holdout', '4:3', '--pointing', pointing_path, '--noqc'])
    assert code == 0
    assert os.path.isfile(os.path.join(str(tmp_path), "joint_report.txt"))

    # The type is kept, and the platepar has the pointing of the night (the reference chunk is of another night)
    written = readPlatepar(out_path, True)
    assert written.distortion_type == "poly3+radial5"
    assert gridDifference(pointings[1], written) < 2*NOISE_PX
    assert written.vignetting_coeff == fitted_poly.vignetting_coeff


def test_cli_stops_on_unreadable_platepars(nights, tmp_path, monkeypatch):
    """ A starting platepar or a platepar of the pointing which can't be read stops the command line with an
        error, before anything is written; so does a number of iterations below 1. """

    config, truth, start, dirs, pointings = nights
    monkeypatch.setattr(cr, "loadConfigFromDirectory", lambda *args: config)
    start_path = str(tmp_path/"start.cal")
    start.write(start_path)
    out_path = str(tmp_path/"joint.cal")

    assert JointDistortionFit.main(dirs + ['-p', str(tmp_path/"missing.cal"), '-o', out_path]) == 1
    assert JointDistortionFit.main(dirs + ['-p', start_path, '-o', out_path, '--pointing',
                                           str(tmp_path/"missing.cal")]) == 1
    assert not os.path.exists(out_path)

    with pytest.raises(SystemExit) as e:
        JointDistortionFit.main(dirs + ['-p', start_path, '-o', out_path, '--iter', '0'])
    assert e.value.code != 0


def test_unreadable_platepars(nights, tmp_path):
    """ Platepars which can't be read or describe no image are none, a night with one starts from the starting
        platepar, and the pointing isn't carried over from a broken platepar or between different cameras. """

    config, truth, start, dirs, pointings = nights

    # A file which isn't a platepar, and a platepar without a scale
    garbage = tmp_path/"garbage.cal"
    garbage.write_text("not a platepar\n")
    assert readPlatepar(str(garbage), True) is None
    assert readPlatepar(str(tmp_path/"missing.cal"), True) is None
    no_scale = json.loads(start.jsonStr())
    no_scale['F_scale'] = -1.0
    (tmp_path/"no_scale.cal").write_text(json.dumps(no_scale))
    assert readPlatepar(str(tmp_path/"no_scale.cal"), True) is None

    # A night with a platepar which can't be read is used without it
    night_dir = tmp_path/"night"
    shutil.copytree(dirs[0], str(night_dir))
    shutil.copy(str(garbage), str(night_dir/config.platepar_name))
    nights_read = readNights([str(night_dir)], config)
    assert len(nights_read) == 1
    assert nights_read[0]['platepar'] is None

    # The pointing isn't carried over from a platepar without a scale, nor to a lens without the distortion
    broken = copy.deepcopy(start)
    broken.F_scale = -1.58
    with pytest.raises(ValueError):
        transferPointing(start, broken)
    no_distortion = copy.deepcopy(truth)
    no_distortion.resetDistortionParameters()
    with pytest.raises(ValueError):
        transferPointing(no_distortion, truth)
    assert transferPointing(no_distortion, truth, max_rms=None)[1] > 1.0


def test_failures(nights, tmp_path, caplog):
    """ No fit without iterations or without usable chunks; the nights whose stars are all below the minimum SNR
        are left out with a warning. """

    config, truth, start, dirs, pointings = nights

    with pytest.raises(ValueError):
        jointDistortionFit(dirs, start, config, n_iter=0)

    # Too few stars in every chunk
    few = copy.deepcopy(config)
    few.ff_min_stars = 100000
    with pytest.raises(ValueError):
        jointDistortionFit(dirs, start, few, n_iter=1, nproc=1)

    # The SNR of all stars (50) below the minimum: no night is used, and the warning says why
    with caplog.at_level(logging.WARNING, logger="rmslogger"):
        assert readNights(dirs, config, snr_min=60) == []
        with pytest.raises(ValueError):
            jointDistortionFit(dirs, start, config, n_iter=1, snr_min=60, nproc=1)
    assert any('SNR' in record.getMessage() and 'older CALSTARS' in record.getMessage()
               for record in caplog.records)


def test_held_out_numpy_and_missing_dirs(caplog):
    """ The held-out nights take numpy integers, and directories which are not among the nights are reported. """

    nights = [{'dir': 'night_{:d}'.format(n)} for n in range(10)]

    assert heldOutNights(nights, (np.int64(4), np.int64(1))) == {1, 5, 9}
    with caplog.at_level(logging.WARNING, logger="rmslogger"):
        assert heldOutNights(nights, ['night_3', 'elsewhere']) == {3}
    assert any('elsewhere' in record.getMessage() for record in caplog.records)


def test_match_stars_one_to_one_and_isolated():
    """ A star is matched only to an isolated catalog star within the match radius, which no other star matches. """

    pp = makePlatepar()
    jd = pp.JD

    # Catalog stars at chosen image positions: an isolated one, a close pair, one which two stars are near, and one
    #   too far from its star
    cat_xy = np.array([[400.0, 300.0], [700.0, 400.0], [702.0, 400.0], [900.0, 200.0], [500.0, 500.0]])
    _, ra, dec, _ = xyToRaDecPP(np.full(len(cat_xy), jd), cat_xy[:, 0], cat_xy[:, 1], np.ones(len(cat_xy)), pp,
                                extinction_correction=False, jd_time=True)
    catalog_stars = np.column_stack([ra, dec, np.full(len(cat_xy), 3.0)])
    x_cat, y_cat = raDecToXYPP(np.array(ra), np.array(dec), jd, pp)

    # The stars (CALSTARS rows: y, x, intensity, amplitude, FWHM, background, SNR, saturated)
    star_xy = [(x_cat[0] + 0.2, y_cat[0]), (x_cat[1], y_cat[1]), (x_cat[3] + 0.2, y_cat[3]),
               (x_cat[3] - 0.2, y_cat[3]), (x_cat[4] + 1.5, y_cat[4])]
    stars = np.array([[y, x, 1000, 100, 2.5, 10, 50, 0] for x, y in star_xy], dtype=np.float64)

    matched = matchStars(pp, jd, stars, catalog_stars, 1.0, 6.0)

    # Only the isolated catalog star is matched, to its star
    assert len(matched) == 1
    assert np.allclose(matched[0, :2], star_xy[0])
    assert np.allclose(matched[0, 4:6], catalog_stars[0, :2])
    assert np.allclose(matched[0, 7:9], [x_cat[0], y_cat[0]])


def test_residual_stats():
    """ The residual statistics: distances, the radial component (outwards positive), zones and corners. """

    # Stars (x, y) with their catalog stars (x, y) on a 100x100 image at 2 arcsec/px: one right of the centre
    #   moved outwards, one left of the centre moved the same way (inwards), one in the top left corner moved up
    stars = [(90, 50, 89, 50), (10, 50, 9, 50), (5, 5, 5, 6)]
    matched = np.zeros((len(stars), 9))
    for i, (x, y, x_cat, y_cat) in enumerate(stars):
        matched[i, [0, 1, 7, 8]] = x, y, x_cat, y_cat
    stats = residualStats(matched, np.full(len(stars), 2.0), 100, 100)

    assert stats['n'] == 3
    assert np.isclose(stats['median'], 2.0)
    assert np.isclose(stats['rms'], 2.0)

    # The two stars at 40 px from the centre cancel in the radial mean, the corner star is moved up, so outwards
    #   by its share along the diagonal
    ring = [r for r in stats['radial'] if r[0] <= 40 < r[1]][0]
    assert ring[3] == 2 and np.isclose(ring[2], 0.0)
    corner_ring = [r for r in stats['radial'] if r[0] <= np.hypot(45, 45) < r[1]][0]
    assert corner_ring[3] == 1 and np.isclose(corner_ring[2], 2.0/np.sqrt(2))

    # The zones (rows from the top) and the corners
    assert stats['zones'][1][2] == (2.0, 0.0, 1)
    assert stats['zones'][1][0] == (2.0, 0.0, 1)
    assert stats['zones'][0][0] == (0.0, -2.0, 1)
    assert stats['corners']['TL'] == (0.0, -2.0, 1)
    assert stats['corners']['BR'][2] == 0 and np.isnan(stats['corners']['BR'][0])

    assert residualStats(np.zeros((0, 9)), np.zeros(0), 100, 100) is None


def test_warning_when_the_fit_gets_worse(caplog):
    """ A final distortion whose residuals are clearly larger than an earlier one's is reported. """

    def _iterations(values):
        return [{'stats': {'rms': value}} for value in values]

    with caplog.at_level(logging.WARNING, logger="rmslogger"):
        assert JointDistortionFit._warnIfWorse(_iterations([5.0, 4.0, 4.5]))
        assert not JointDistortionFit._warnIfWorse(_iterations([5.0, 4.0, 4.02]))


# The decentering of the known distortion of the decentering tests (p1, p2, p3, p4 of the sky to image mapping),
#   which moves the stars by up to about 0.9 px after the pointing is fitted, and the nights, chunks and noise (px)
#   of its synthetic nights (a smaller noise, as the asymmetry correction takes up a part of the decentering)
DECENTERING = [0.0002, -0.00015, 0.0001, 0.0]
DECENTERED_NIGHTS = 4
DECENTERED_CHUNKS = 2
DECENTERED_NOISE = 0.05


@pytest.fixture(scope="module")
def decentered_nights(tmp_path_factory):
    """ The synthetic nights of a known distortion with the decentering correction, the platepars of the nights and
        the starting platepar without it (the same distortion otherwise).

    Return:
        (config, truth, start, dirs, pointings): as the nights fixture.
    """

    config = makeConfig()
    truth = makePlatepar()
    truth.remapCoeffsForFlagChange('decentering_corr', True)
    truth.x_poly_rev[-4:] = DECENTERING
    start = makePlatepar()
    dirs, pointings = makeNights(tmp_path_factory, config, truth, start, "decentered", n_nights=DECENTERED_NIGHTS,
                                 n_chunks=DECENTERED_CHUNKS, noise=DECENTERED_NOISE)

    return config, truth, start, dirs, pointings


@pytest.fixture(scope="module")
def decentering_on(decentered_nights):
    """ The joint fit of the decentered nights with the decentering correction switched on. """

    config, truth, start, dirs, pointings = decentered_nights

    return jointDistortionFit(dirs, start, config, n_iter=2, chunk_step=1, nproc=4, decentering=True)[0]


def decenteringCoeffsClose(coeffs):
    """ Whether fitted decentering coefficients are those of the known distortion: p1 and p3 within 20%, p2 within
        30%, p4 near zero (the radial terms and the asymmetry correction take up a part of the decentering, so the
        coefficients are not exactly those of the known distortion even when the stars are fitted to the noise).

    Arguments:
        coeffs: [list] Fitted p1, p2, p3, p4.

    Return:
        [bool] True if they are close.
    """

    p1, p2, p3, p4 = coeffs
    return (abs(p1/DECENTERING[0] - 1) < 0.2) and (abs(p3/DECENTERING[2] - 1) < 0.2) \
        and (abs(p2/DECENTERING[1] - 1) < 0.3) and (abs(p4) < 0.1*DECENTERING[0])


def test_decentering_switched_on_recovers_it(decentered_nights, decentering_on):
    """ Switched on, the decentering correction is fitted: its coefficients are those of the known distortion and
        the fitted distortion puts the stars where the known one does, much closer than the noise. """

    config, truth, start, dirs, pointings = decentered_nights
    fitted = decentering_on

    assert fitted.decentering_corr and (fitted.distortion_type == "radial7-odd")
    assert decenteringCoeffsClose(fitted.x_poly_rev[-4:])
    for pointing in pointings:
        assert gridDifference(pointing, transferPointing(start, pointing, max_rms=None)[0]) > 10*DECENTERED_NOISE
        assert gridDifference(pointing, transferPointing(fitted, pointing)[0]) < 0.5*DECENTERED_NOISE


def test_decentering_kept_off_by_default(decentered_nights, decentering_on):
    """ Not asked for, the decentering correction stays off as in the starting platepar. Without it the radial
        distortion and its asymmetry correction take up most of the decentering, but leave several times more
        than the fit with it. """

    config, truth, start, dirs, pointings = decentered_nights
    fitted = jointDistortionFit(dirs, start, config, n_iter=1, chunk_step=1, nproc=4)[0]

    assert not fitted.decentering_corr
    assert fitted.poly_length == start.poly_length
    for pointing in pointings:
        assert gridDifference(pointing, transferPointing(fitted, pointing)[0]) \
            > 2*gridDifference(pointing, transferPointing(decentering_on, pointing)[0])


def test_decentering_kept_on_by_default(decentered_nights, decentering_on):
    """ Not asked for, the decentering correction of a starting platepar which has it stays on and is refitted. """

    config, truth, start, dirs, pointings = decentered_nights
    fitted = jointDistortionFit(dirs, decentering_on, config, n_iter=1, chunk_step=1, nproc=4)[0]

    assert fitted.decentering_corr
    assert decenteringCoeffsClose(fitted.x_poly_rev[-4:])


def test_decentering_switched_off_from_the_command_line(decentered_nights, decentering_on, tmp_path, monkeypatch):
    """ Switched off on the command line, the decentering coefficients are dropped from the fitted platepar. """

    config, truth, start, dirs, pointings = decentered_nights
    monkeypatch.setattr(cr, "loadConfigFromDirectory", lambda *args: config)
    start_path, out_path = str(tmp_path/"start.cal"), str(tmp_path/"joint.cal")
    decentering_on.write(start_path)

    assert JointDistortionFit.main(dirs + ['-p', start_path, '-o', out_path, '--iter', '1', '--step', '2',
                                           '--nproc', '4', '--decentering', 'off', '--noqc']) == 0
    written = readPlatepar(out_path, True)
    assert not written.decentering_corr
    assert written.poly_length == start.poly_length
    assert len(written.x_poly_rev) == start.poly_length


def test_decentering_switched_on_with_a_type_change_from_the_command_line(nights, fit_poly, tmp_path,
                                                                         monkeypatch):
    """ On the command line, a type change with the decentering correction switched on fits the new radial type
        with the correction (from no distortion), from a starting platepar of a polynomial type. """

    config, truth, start, dirs, pointings = nights
    monkeypatch.setattr(cr, "loadConfigFromDirectory", lambda *args: config)
    start_path, out_path = str(tmp_path/"start.cal"), str(tmp_path/"joint.cal")
    fit_poly[0].write(start_path)

    assert JointDistortionFit.main(dirs + ['-p', start_path, '-o', out_path, '--iter', '2', '--step', '2',
                                           '--nproc', '4', '-t', 'radial7-odd', '--decentering', 'on', '--noqc']) == 0
    written = readPlatepar(out_path, True)
    assert written.distortion_type == "radial7-odd"
    assert written.decentering_corr
    assert len(written.x_poly_rev) == written.poly_length == start.poly_length + 4


def test_decentering_with_a_polynomial_type_fails(nights, fit_poly, tmp_path, monkeypatch):
    """ The decentering correction exists only for the radial types: switched on with a polynomial type the fit
        fails before it starts, the command line with an error. """

    config, truth, start, dirs, pointings = nights

    with pytest.raises(ValueError):
        jointDistortionFit(dirs, fit_poly[0], config, decentering=True)
    with pytest.raises(ValueError):
        jointDistortionFit(dirs, start, config, distortion_type="poly3+radial5", decentering=True)

    monkeypatch.setattr(cr, "loadConfigFromDirectory", lambda *args: config)
    start_path, out_path = str(tmp_path/"start.cal"), str(tmp_path/"joint.cal")
    start.write(start_path)
    assert JointDistortionFit.main(dirs + ['-p', start_path, '-o', out_path, '-t', 'poly3+radial5',
                                           '--decentering', 'on']) == 1
    assert not os.path.exists(out_path)



# The limits of the quality control for the few chunks of the synthetic nights
SMALL_QC = {'min_chunks': 3, 'min_held_chunks': 1, 'min_zone_stars': 0, 'judge_min_chunks': 1}


@pytest.fixture(scope="module")
def fit_held_chunks(nights):
    """ The joint fit of the synthetic nights with the middle chunk of every night held out (and no night held
        out), with the quality control for few chunks. """

    config, truth, start, dirs, pointings = nights

    # The synthetic stars have neither vignetting nor extinction, which the judgement of the photometry would see
    start = copy.deepcopy(start)
    start.vignetting_coeff = 0.0
    start.extinction_scale = 0.0

    return jointDistortionFit(dirs, start, config, distortion_type="radial7-odd", n_iter=3, chunk_step=1, nproc=4,
                              holdout_chunks=N_CHUNKS, qc_options=SMALL_QC)


def test_held_out_chunks_of_the_nights_of_the_fit(nights, fit_held_chunks):
    """ Every N-th chunk of the nights of the fit is held out: the middle chunk of every night, judged with the
        starting platepars and with the fitted distortion, which is better on them. """

    config, truth, start, dirs, pointings = nights
    fitted, report = fit_held_chunks

    noise = np.sqrt(2)*NOISE_PX*3600/truth.F_scale

    assert report['held_out'] == []
    assert report['chunks'] == N_NIGHTS*(N_CHUNKS - 1)
    assert report['held_chunks'] == N_NIGHTS
    held = report['held_out_stats']
    assert [night[0] for night in held['fitted']['nights']] == dirs
    assert held['fitted']['recalibrated'] == N_NIGHTS
    assert held['fitted']['stats']['rms'] < 1.2*noise
    assert held['fitted']['stats']['rms'] < held['start']['stats']['rms']

    # Every chunk is in the fit or held out, never both
    with pytest.raises(ValueError):
        jointDistortionFit(dirs, start, config, holdout_chunks=1)


def test_quality_control_accepts_a_good_fit(fit_held_chunks):
    """ A fit which recovers the distortion meets every check of the quality control, and the report lists them;
        the mapping converged. """

    fitted, report = fit_held_chunks

    qc = report['qc']
    assert qc['ok'], qc
    names = [check['name'] for check in qc['checks']]
    for name in ('chunks of the fit', 'held-out chunks', 'held-out stars not worse', 'no held-out zone worse',
                 'worst held-out zone not worse',
                 'judged on the held-out chunks', 'converged', 'few outlier chunks',
                 'change from the starting platepar', 'no fold-over', 'forward and reverse agree'):
        assert name in names
    assert report['iterations'][-2]['change_px'][0] < QC_DEFAULTS['max_change_px']
    assert 'Quality control: OK' in JointDistortionFit.formatReport(report)


def _failed(fitted, start, report, config, **options):
    """ The names of the failed checks of the quality control with the given limits (judgement off). """

    limits = dict(QC_DEFAULTS, judge=False, **SMALL_QC)
    limits.update(options)

    return qualityControl(fitted, start, report, config, limits)['failed']


def test_quality_control_failures(nights, fit_held_chunks):
    """ Every check fails when its condition isn't met: too few chunks or stars in a zone, worse held-out stars or
        zones, no convergence, many outlier chunks, a mapping far from the start, no held-out stars. """

    config, truth, start, dirs, pointings = nights
    fitted, report = fit_held_chunks

    assert _failed(fitted, start, report, config) == []

    # Too few chunks, held-out chunks, stars in a zone; too few hours or too short a night
    assert _failed(fitted, start, report, config, min_chunks=1000) == ['chunks of the fit']
    assert _failed(fitted, start, report, config, min_held_chunks=1000) == ['held-out chunks']
    assert 'stars in every zone and corner' in _failed(fitted, start, report, config, min_zone_stars=10**6)
    assert qualityControl(fitted, start, report, config, dict(QC_DEFAULTS, judge=False, min_hours=2.0,
                                                              min_span_hours=4.0, **SMALL_QC),
                          clear_hours=1.0, longest_span_hours=3.0)['failed'] == ['hours of clear sky',
                                                                                 'span of the night']

    # The held-out stars worse than with the starting platepars, overall and in the central zone
    worse = copy.deepcopy(report)
    after = worse['held_out_stats']['fitted']['stats']
    after['median'] *= 10
    after['zones'][1][1] = (500.0, 500.0, 100)
    failed = _failed(fitted, start, worse, config)
    assert 'held-out stars not worse' in failed and 'no held-out zone worse' in failed
    assert 'worst held-out zone not worse' in failed

    # Without held-out stars
    no_held = copy.deepcopy(report)
    no_held.pop('held_out_stats')
    assert set(_failed(fitted, start, no_held, config)) == {'held-out chunks', 'held-out stars not worse'}

    # No held-out zone with enough stars to compare
    failed = _failed(fitted, start, report, config, min_zone_stars=10**6)
    assert 'no held-out zone worse' in failed and 'worst held-out zone not worse' in failed

    # Not converged, or many outlier chunks
    unconverged = copy.deepcopy(report)
    unconverged['iterations'][-2]['change_px'] = (1.0, 2.0)
    assert _failed(fitted, start, unconverged, config) == ['converged']
    outliers = copy.deepcopy(report)
    outliers['iterations'][-1]['outlier_chunks'] = report['chunks']
    assert _failed(fitted, start, outliers, config) == ['few outlier chunks']

    # A mapping far from the starting platepar's
    assert _failed(fitted, start, report, config, max_model_change_diag=1e-6) == ['change from the starting platepar']


def test_model_sanity():
    """ A good distortion neither folds over nor disagrees between its directions; one with a huge high-order term
        folds over at the edges of the image. """

    pp = makePlatepar()
    folds, roundtrip = modelSanity(pp)
    assert folds == 0
    assert roundtrip < 0.2

    folded = copy.deepcopy(pp)
    folded.x_poly_rev = np.array(folded.x_poly_rev)
    folded.x_poly_rev[5] = 50.0
    folds, roundtrip = modelSanity(folded)
    assert (folds > 0) or (roundtrip > QC_DEFAULTS['max_roundtrip_px'])

    # The mapping of a platepar against itself doesn't differ
    assert mappingDifference(pp, pp)[1] < 1e-3


def test_outlier_chunks_are_left_out():
    """ A chunk whose residuals are far larger than the others' is an outlier, and so are all chunks of a night
        whose residuals are far larger than the other nights'. """

    def _stars(dist, n=30):
        stars = np.zeros((n, 9))
        stars[:, 0], stars[:, 7] = 100.0 + dist, 100.0
        stars[:, 3] = 50.0
        return stars

    pp = makePlatepar()
    results = {(n, c): (pp, _stars(0.2)) for n in range(4) for c in range(5)}
    results[(0, 2)] = (pp, _stars(2.0))
    assert JointDistortionFit._outliers(results, list(results), 0) == {(0, 2)}

    results = {(n, c): (pp, _stars(0.2 if n else 0.5)) for n in range(4) for c in range(5)}
    assert JointDistortionFit._outliers(results, list(results), 0) == set((0, c) for c in range(5))


def test_cli_exit_code_of_a_failed_quality_control(nights, fit_poly, tmp_path, monkeypatch):
    """ A fit which fails its quality control (the synthetic nights have too few chunks for the default limits)
        is written with its report, and the command line ends with exit code 2. """

    config, truth, start, dirs, pointings = nights
    monkeypatch.setattr(cr, "loadConfigFromDirectory", lambda *args: config)
    start_path, out_path = str(tmp_path/"start.cal"), str(tmp_path/"joint.cal")
    start.write(start_path)

    assert JointDistortionFit.main(dirs + ['-p', start_path, '-o', out_path, '--iter', '1', '--nproc', '2',
                                           '--holdoutchunks', '3', '--maxchunks', '6']) == 2
    assert os.path.isfile(out_path)
    with open(os.path.join(str(tmp_path), "joint_report.txt")) as f:
        assert 'Quality control: FAIL (chunks of the fit' in f.read()


def test_held_out_baseline_of_the_starting_distortion(nights):
    """ With the held-out baseline 'start', the held-out nights are compared with the starting distortion (with
        their pointing carried over), not with their own platepars: starting from the known distortion, its
        held-out residuals are the noise, while the platepars of the nights (the wrong distortion) are worse. """

    config, truth, start, dirs, pointings = nights
    noise = np.sqrt(2)*NOISE_PX*3600/truth.F_scale

    reports = {}
    for baseline in ('night', 'start'):
        reports[baseline] = jointDistortionFit(dirs, truth, config, holdout=(N_NIGHTS, HELD_OUT), n_iter=1,
                                               chunk_step=1, nproc=4, qc=False, holdout_baseline=baseline)[1]

    assert reports['start']['held_out_stats']['start']['stats']['rms'] < 1.15*noise
    assert reports['night']['held_out_stats']['start']['stats']['rms'] > 1.3*noise

    with pytest.raises(ValueError):
        jointDistortionFit(dirs, truth, config, holdout_baseline='other')


def test_quality_control_selects_clear_chunks_with_the_mask(nights, fit_held_chunks, monkeypatch):
    """ The judgement of the quality control selects the clear held-out chunks with the mask of the camera. """

    import RMS.Astrometry.AutoCalibration as ac

    config, truth, start, dirs, pointings = nights
    fitted, report = fit_held_chunks

    masks = []
    monkeypatch.setattr(ac, 'clearChunks', lambda chunks, width, height, mask=None, **kwargs:
                        masks.append(mask) or [])
    qc = qualityControl(fitted, start, report, config, dict(QC_DEFAULTS, **SMALL_QC),
                        judge_groups=[('night', fitted, start, [('FF', 0.0, np.zeros((3, 8)), 256)])],
                        mask='mask image')
    assert masks == ['mask image']
    assert 'judged on the held-out chunks' in qc['failed']


def test_chunks_timed_by_the_frame_rate_of_the_header(tmp_path):
    """ The chunks of a CALSTARS file with the frame rate in its header are timed with it, and their duration
        (for the hours of clear sky of the quality control) is that of the header's frame rate. """

    config = makeConfig()
    config.fps = 25.0
    name = 'FF_XX0001_20261105_101112_345_0000000.fits'
    CALSTARS.writeCALSTARS([[name, [(10.0, 20.0, 100, 10, 2.0, 5, 20.0, 0)]]], str(tmp_path), 'CALSTARS_test.txt',
                           'XX0001', 720, 1280, chunk_frames=128, fps=10.0)

    chunk = JointDistortionFit.readChunks(str(tmp_path), config)[0]
    assert chunk[1] == pytest.approx(date2JD(*getMiddleTimeFF(name, 10.0, ret_milliseconds=True, ff_frames=128)),
                                     abs=1e-9)
    assert chunk[3] == 128
    assert chunk[4] == pytest.approx(12.8)
