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
import os

import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.AutoPlatepar import loadCatalogStars
from RMS.Astrometry.Conversions import date2JD, jd2Date, JD2HourAngle
from RMS.Astrometry.JointDistortionFit import heldOutNights, jointDistortionFit, transferPointing
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


def chunkStars(pp, jd, catalog_stars, rng):
    """ The stars of a chunk as CALSTARS rows: the catalog stars on the image with the platepar, with the noise. """

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
    x = x + rng.normal(0, NOISE_PX, x.size)
    y = y + rng.normal(0, NOISE_PX, y.size)

    # The intensities follow the photometry of the platepar
    intensity = 10**((pp.mag_lev - mag)/2.5)
    n = x.size

    return np.column_stack([y, x, intensity, intensity/10, np.full(n, 2.5), np.full(n, 10.0), np.full(n, 50.0),
                            np.zeros(n)])


@pytest.fixture(scope="module")
def nights(tmp_path_factory):
    """ The synthetic nights: a directory per night with a CALSTARS file and a platepar with the night's pointing
        and a wrong distortion.

    Return:
        (config, truth, start, dirs, pointings): the configuration, the known distortion, the starting platepar
            (the wrong distortion), the directories and the true platepar of every night.
    """

    config = makeConfig()
    truth = makePlatepar()
    start = wrongDistortion(truth)
    catalog_stars = loadCatalogStars(config, LIM_MAG, jd=truth.JD)
    rng = np.random.default_rng(0)

    dirs, pointings = [], []
    for n in range(N_NIGHTS):

        # The pointing of the night, a few nights apart: the camera turned and refocused a little
        jd_night = truth.JD + 3*n
        pointing = withPointing(truth, jd_night, 0.3*n, -0.2*n, 0.1*n, 0.0005*n)

        # The chunks of the night, their stars at the middle of the chunk (the time of the recalibration)
        star_list = []
        for c in range(N_CHUNKS):
            name = chunkName(jd_night + c*CHUNK_MINUTES/1440.0)
            jd = date2JD(*getMiddleTimeFF(name, config.fps, ret_milliseconds=True, ff_frames=256))
            star_list.append([name, chunkStars(pointing, jd, catalog_stars, rng).tolist()])

        # The directory of the night, with the night's pointing on the wrong distortion
        dir_path = str(tmp_path_factory.mktemp("night_{:d}".format(n)))
        CALSTARS.writeCALSTARS(star_list, dir_path, "CALSTARS_XX0001.txt", "XX0001", truth.Y_res, truth.X_res)
        transferPointing(start, pointing)[0].write(os.path.join(dir_path, config.platepar_name))

        dirs.append(dir_path)
        pointings.append(pointing)

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

    # The photometry of the starting platepar is kept
    assert fitted.vignetting_coeff == start.vignetting_coeff
    assert fitted.mag_lev == start.mag_lev


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
