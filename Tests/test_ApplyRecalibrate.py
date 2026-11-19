"""
Tests of the star coverage gate of the recalibration on individual images (ApplyRecalibrate.recalibrateFF and
ApplyRecalibrate.coverageMatchFraction).

The gate rejects a recalibration which matched only a small fraction of the stars which could have been matched.
Only the stars down to the catalog limit can be matched, so a camera which detects fainter stars must not have
its fits rejected. The tests use a synthetic 1280x720 image with a 60 deg field: catalog stars at known image
positions, detections of them, and detections of fainter stars with no catalog counterpart.
"""

import copy

import numpy as np

import RMS.ConfigReader as cr
from RMS.Astrometry import ApplyRecalibrate
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.Conversions import JD2HourAngle
from RMS.Formats.Platepar import Platepar


# Julian date of the synthetic image (2023-02-28 ~18 UT)
JD_IMG = 2460004.25

# Limiting magnitude of the matching, the catalog stars are brighter
LIM_MAG = 4.5

# Smallest distance between two stars of the synthetic image (px), so every detection has one counterpart
MIN_SEPARATION = 20.0


def makePlatepar():
    """ A 1280x720 platepar with a 60 deg field pointed 60 deg up to the south, without distortion. """

    # The station and the image
    pp = Platepar()
    pp.lat, pp.lon, pp.elev = 45.0, 15.0, 100.0
    pp.X_res, pp.Y_res = 1280, 720
    pp.JD = JD_IMG

    # The reference hour angle goes with the reference time, as when a platepar is fitted or loaded
    pp.Ho = JD2HourAngle(pp.JD)

    # The scale and the pointing, with the reference RA and Dec of the pointing at the reference time
    pp.F_scale = pp.X_res/60.0
    pp.az_centre, pp.alt_centre = 180.0, 60.0
    pp.updateRefRADec(skip_rot_update=True)
    pp.updateRefAltAz()

    return pp


def makeConfig():
    """ The default configuration for a 1280x720 camera. """

    config = cr.Config()
    config.width, config.height = 1280, 720

    return config


def randomPositions(rng, n, pp, taken=None, margin=30):
    """ Random image positions at least MIN_SEPARATION apart, and as far from the already taken positions.

    Arguments:
        rng: [Generator] Random number generator.
        n: [int] Number of positions.
        pp: [Platepar instance] Platepar which gives the image size.

    Keyword arguments:
        taken: [ndarray] (N, 2) positions which are already taken. None by default.
        margin: [float] Distance of the positions from the image edges (px). 30 by default.

    Return:
        [ndarray] (n, 2) positions (x, y).
    """

    # Draw positions until there are enough of them far enough from all others
    taken = [] if taken is None else [tuple(xy) for xy in taken]
    new = []
    while len(new) < n:
        x, y = rng.uniform(margin, pp.X_res - margin), rng.uniform(margin, pp.Y_res - margin)
        if all(np.hypot(x - xi, y - yi) > MIN_SEPARATION for xi, yi in taken + new):
            new.append((x, y))

    return np.array(new)


def catalogAt(pp, xy, mags):
    """ Catalog stars at the given image positions, sorted by descending declination as the catalog subsetting
        expects.

    Arguments:
        pp: [Platepar instance] Platepar which maps the image positions to the sky.
        xy: [ndarray] (N, 2) image positions (x, y).
        mags: [ndarray] Magnitudes of the stars.

    Return:
        [ndarray] (N, 3) catalog stars (ra, dec, mag).
    """

    # The sky positions of the image positions at the time of the image
    n = len(xy)
    _, ra, dec, _ = xyToRaDecPP(np.full(n, pp.JD), xy[:, 0], xy[:, 1], np.ones(n), pp,
                                extinction_correction=False, jd_time=True)
    catalog = np.column_stack([ra, dec, mags]).astype(np.float64)

    return catalog[np.argsort(-catalog[:, 1])]


def detections(xy, mags):
    """ Detected stars at the given image positions, with the intensities of their magnitudes.

    Arguments:
        xy: [ndarray] (N, 2) image positions (x, y).
        mags: [ndarray] Magnitudes of the stars.

    Return:
        [ndarray] (N, 8) CALSTARS rows (y, x, intensity, amplitude, FWHM, background, SNR, saturated).
    """

    # Brighter stars have larger intensities
    n = len(xy)
    intensity = 10**(-0.4*(np.asarray(mags) - 12.0))

    return np.column_stack([xy[:, 1], xy[:, 0], intensity, intensity/10, np.full(n, 2.5), np.full(n, 20.0),
                            np.full(n, 30.0), np.zeros(n)]).astype(np.float64)


def deepCameraImage(n_catalog, n_faint, seed):
    """ An image of a camera which detects stars fainter than the catalog limit: every catalog star in the
        image is detected, together with fainter stars which have no catalog counterpart.

    Arguments:
        n_catalog: [int] Number of catalog stars in the image.
        n_faint: [int] Number of detected stars fainter than the catalog limit.
        seed: [int] Seed of the random numbers.

    Return:
        (pp, catalog_stars, star_rows): [tuple] The true platepar, the catalog stars and the detections.
    """

    pp = makePlatepar()
    rng = np.random.default_rng(seed)

    # The catalog stars and their detections
    cat_xy = randomPositions(rng, n_catalog, pp)
    cat_mags = rng.uniform(1.0, LIM_MAG - 0.1, n_catalog)

    # The fainter stars, away from the catalog stars
    faint_xy = randomPositions(rng, n_faint, pp, taken=cat_xy)

    rows = np.r_[detections(cat_xy, cat_mags), detections(faint_xy, np.full(n_faint, LIM_MAG + 1.0))]

    return pp, catalogAt(pp, cat_xy, cat_mags), rows


def startPlatepar(pp):
    """ The platepar with the pointing off by about half a pixel, so the recalibration has to run a fit. """

    # Move the pointing by half a pixel in RA
    pp = copy.deepcopy(pp)
    pp.RA_d += 0.5/pp.F_scale/np.cos(np.radians(pp.dec_d))
    pp.updateRefAltAz()

    return pp


def test_deep_camera_fit_is_accepted():
    """ All 100 catalog stars in the image are detected and matched, together with 150 fainter detections. Only
        100/250 = 0.40 of the detections can ever be matched, but the fit matched all possible stars. """

    pp, catalog_stars, rows = deepCameraImage(100, 150, seed=1)
    config = makeConfig()

    # All detections are used for the fit
    result, _ = ApplyRecalibrate.recalibrateFF(config, startPlatepar(pp), JD_IMG, {JD_IMG: rows}, catalog_stars,
                                               lim_mag=LIM_MAG, ignore_max_stars=True)

    # The fit is accepted with all catalog stars matched
    assert result is not None
    assert result.auto_recalibrated
    assert len(result.star_list) == 100


def test_deep_camera_fit_with_sampled_stars_is_accepted():
    """ With more detections than recalibration_max_stars, a random sample of them is fitted. The 100 catalog
        stars are among 400 detections, so about 50 of them are in the sample of 200: the possible matches are
        those 50, not all 100 catalog stars in the image. """

    pp, catalog_stars, rows = deepCameraImage(100, 300, seed=2)
    config = makeConfig()

    # More detections than the recalibration uses
    assert len(rows) > config.recalibration_max_stars

    # A sample of the detections is used for the fit
    result, _ = ApplyRecalibrate.recalibrateFF(config, startPlatepar(pp), JD_IMG, {JD_IMG: rows}, catalog_stars,
                                               lim_mag=LIM_MAG)

    # The fit is accepted with the catalog stars in the sample matched
    assert result is not None
    assert result.auto_recalibrated
    assert len(result.star_list) >= config.min_matched_stars


def test_sparse_spurious_fit_is_still_rejected():
    """ A fit which matched only 29 of 308 detections, with 400 catalog stars in the image, is rejected even when
        the goodness of fit and the distance checks are skipped (guard: rejected before and after the change). """

    pp = makePlatepar()
    rng = np.random.default_rng(3)
    config = makeConfig()

    # 400 catalog stars in the image, of which only 29 are detected, and 279 unrelated detections
    cat_xy = randomPositions(rng, 400, pp)
    cat_mags = rng.uniform(1.0, LIM_MAG - 0.1, 400)
    other_xy = randomPositions(rng, 279, pp, taken=cat_xy, margin=10)
    rows = np.r_[detections(cat_xy[:29], cat_mags[:29]), detections(other_xy, np.full(279, 3.0))]
    catalog_stars = catalogAt(pp, cat_xy, cat_mags)

    # The goodness of fit and the distance checks are skipped, so only the coverage gate can reject the fit
    kwargs = dict(lim_mag=LIM_MAG, ignore_max_stars=True, force_platepar_save=True, ignore_distance_threshold=True)
    result, _ = ApplyRecalibrate.recalibrateFF(config, startPlatepar(pp), JD_IMG, {JD_IMG: rows}, catalog_stars,
                                               **kwargs)

    assert result is None

    # Without the gate the same fit is accepted with the 29 stars
    result, _ = ApplyRecalibrate.recalibrateFF(config, startPlatepar(pp), JD_IMG, {JD_IMG: rows}, catalog_stars,
                                               min_match_fraction=0.0, **kwargs)

    assert result is not None
    assert len(result.star_list) == 29


def test_coverage_fraction_of_the_possible_matches():
    """ The possible matches are the smaller of the number of detections and the number of catalog stars which
        project inside the image. Catalog stars outside the image don't count. """

    pp = makePlatepar()
    rng = np.random.default_rng(4)

    # 100 catalog stars inside the image and 30 outside it, to the left
    inside_xy = randomPositions(rng, 100, pp)
    outside_xy = np.column_stack([rng.uniform(-200, -50, 30), rng.uniform(50, pp.Y_res - 50, 30)])
    catalog_stars = catalogAt(pp, np.r_[inside_xy, outside_xy], np.full(130, 3.0))

    # A deep camera: more detections than catalog stars in the image
    fraction, n_possible = ApplyRecalibrate.coverageMatchFraction(90, 250, catalog_stars, JD_IMG, pp)
    assert n_possible == 100
    assert np.isclose(fraction, 0.9)

    # A camera which detects fewer stars than the catalog has: the fraction of the detections, as before
    fraction, n_possible = ApplyRecalibrate.coverageMatchFraction(45, 60, catalog_stars, JD_IMG, pp)
    assert n_possible == 60
    assert np.isclose(fraction, 0.75)


def test_coverage_fraction_with_sampled_detections():
    """ When only a sample of the detections is matched, only the same share of the catalog stars can have its
        detection in the sample. """

    # 100 catalog stars inside the image
    pp = makePlatepar()
    rng = np.random.default_rng(5)
    catalog_stars = catalogAt(pp, randomPositions(rng, 100, pp), np.full(100, 3.0))

    # 200 of 400 detections used: half of the 100 catalog stars can be matched
    fraction, n_possible = ApplyRecalibrate.coverageMatchFraction(45, 200, catalog_stars, JD_IMG, pp,
                                                                  n_detected_all=400)
    assert np.isclose(n_possible, 50)
    assert np.isclose(fraction, 0.9)

    # All detections used
    fraction, n_possible = ApplyRecalibrate.coverageMatchFraction(45, 200, catalog_stars, JD_IMG, pp,
                                                                  n_detected_all=200)
    assert np.isclose(n_possible, 100)
    assert np.isclose(fraction, 0.45)


def test_coverage_fraction_without_catalog_stars():
    """ No catalog stars give a zero fraction instead of a division by zero. """

    pp = makePlatepar()

    # An empty catalog
    fraction, n_possible = ApplyRecalibrate.coverageMatchFraction(0, 100, np.zeros((0, 3)), JD_IMG, pp)

    assert n_possible == 0
    assert fraction == 0.0
