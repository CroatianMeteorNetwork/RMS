"""
Tests of the star coverage gate of the recalibration on individual images (ApplyRecalibrate.recalibrateFF and
ApplyRecalibrate.coverageMatchFraction), and of its photometry: the photometric fit of every image, and the
averaging of the photometric offsets in the neighbourhoods of the images with meteors
(ApplyRecalibrate.averageNeighbourPhotometry).

The gate rejects a recalibration which matched only a small fraction of the stars which could have been matched.
Only the stars down to the catalog limit can be matched, so a camera which detects fainter stars must not have
its fits rejected. The tests use a synthetic 1280x720 image with a 60 deg field: catalog stars at known image
positions, detections of them, and detections of fainter stars with no catalog counterpart.

The neighbourhood averaging must leave out the degenerate photometric fits (the initial guess 10.0 with a zero
standard deviation) but give them the average, include the first image of the night, and give the same offsets
whatever the order of the meteors.
"""

import copy
import datetime
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_zero_intensity_star_keeps_the_recalibrated_photometry():
    """ One matched star with an intensity of 0 (an integer IntensSum of 0 in CALSTARS) must not turn the
        photometric fit of the recalibration into its initial guess (mag_lev = 10.0 with a zero standard
        deviation). """

    # An image with 100 catalog stars detected and 20 fainter detections
    pp, catalog_stars, rows = deepCameraImage(100, 20, seed=6)
    config = makeConfig()

    # The photometry of the image as it is
    clean, _ = ApplyRecalibrate.recalibrateFF(config, startPlatepar(pp), JD_IMG, {JD_IMG: rows}, catalog_stars,
                                              lim_mag=LIM_MAG, ignore_max_stars=True)
    assert clean is not None

    # The same image with one star of zero intensity
    rows = rows.copy()
    rows[0, 2] = 0
    result, _ = ApplyRecalibrate.recalibrateFF(config, startPlatepar(pp), JD_IMG, {JD_IMG: rows}, catalog_stars,
                                               lim_mag=LIM_MAG, ignore_max_stars=True)

    # The offset of the other stars, with a real standard deviation
    assert result is not None
    assert abs(result.mag_lev - clean.mag_lev) < 0.05
    assert result.mag_lev_stddev > 0


def photometryFits(values):
    """ Recalibrated platepars which only carry a photometric offset and its standard deviation.

    Arguments:
        values: [dict] FF name -> (mag_lev, mag_lev_stddev).

    Return:
        [dict] FF name -> object with the mag_lev and mag_lev_stddev attributes.
    """

    return {ff_name: SimpleNamespace(mag_lev=mag_lev, mag_lev_stddev=mag_lev_stddev)
            for ff_name, (mag_lev, mag_lev_stddev) in values.items()}


def overlappingNeighbourhoodNight():
    """ A night of 8 images where the neighbourhoods of meteors two or one images apart overlap, with one
        degenerate photometric fit (FF_3) between them.

    Return:
        (calstars_ffs, recalibrated): [tuple] FF names in the CALSTARS order, and the recalibrated platepars.
    """

    # FF_0 is not recalibrated, the fit of FF_3 is degenerate
    calstars_ffs = ['FF_{:d}'.format(i) for i in range(8)]
    recalibrated = photometryFits({
        'FF_1': (11.0, 0.1),
        'FF_2': (11.4, 0.1),
        'FF_3': (10.0, 0.0),
        'FF_4': (11.2, 0.1),
        'FF_5': (10.6, 0.1),
        'FF_6': (11.8, 0.1),
        'FF_7': (11.1, 0.1),
    })

    return calstars_ffs, recalibrated


def test_neighbourhood_average_of_usable_fits():
    """ Guard (the same before and after the change): a meteor image with two usable neighbours gets the mean
        offset of the three and the combined standard deviation, and so do the neighbours. """

    # A meteor on FF_2, with usable fits of FF_1, FF_2 and FF_3
    calstars_ffs = ['FF_0', 'FF_1', 'FF_2', 'FF_3']
    recalibrated = photometryFits({'FF_1': (11.0, 0.1), 'FF_2': (11.3, 0.2), 'FF_3': (11.2, 0.1)})

    ApplyRecalibrate.averageNeighbourPhotometry(recalibrated, ['FF_2'], calstars_ffs)

    # The combined standard deviation of the three fits, assuming equal sample sizes
    mean = np.mean([11.0, 11.3, 11.2])
    stddev = np.sqrt(np.mean([0.1**2 + (11.0 - mean)**2, 0.2**2 + (11.3 - mean)**2, 0.1**2 + (11.2 - mean)**2]))

    # All three images get the average
    for ff_name in ['FF_1', 'FF_2', 'FF_3']:
        assert recalibrated[ff_name].mag_lev == pytest.approx(mean)
        assert recalibrated[ff_name].mag_lev_stddev == pytest.approx(stddev)


def test_degenerate_fit_receives_the_neighbourhood_average():
    """ An image with a degenerate photometric fit (mag_lev 10.0 from the initial guess, zero standard deviation)
        must not contribute to the neighbourhood average, but must receive it. """

    # A meteor on FF_2, whose fit is degenerate, between two usable fits
    calstars_ffs = ['FF_0', 'FF_1', 'FF_2', 'FF_3']
    recalibrated = photometryFits({'FF_1': (11.0, 0.1), 'FF_2': (10.0, 0.0), 'FF_3': (11.2, 0.1)})

    ApplyRecalibrate.averageNeighbourPhotometry(recalibrated, ['FF_2'], calstars_ffs)

    # The average only uses FF_1 and FF_3, and all three images receive it
    for ff_name in ['FF_1', 'FF_2', 'FF_3']:
        assert recalibrated[ff_name].mag_lev == pytest.approx(11.1)
        assert recalibrated[ff_name].mag_lev_stddev > 0


def test_first_image_is_in_its_own_neighbourhood():
    """ A meteor on the first image of CALSTARS: the image itself belongs to its neighbourhood, contributes to the
        average and receives it. """

    # A meteor on FF_0, with usable fits of FF_0 and FF_1
    calstars_ffs = ['FF_0', 'FF_1', 'FF_2', 'FF_3']
    recalibrated = photometryFits({'FF_0': (11.0, 0.1), 'FF_1': (11.2, 0.1)})

    ApplyRecalibrate.averageNeighbourPhotometry(recalibrated, ['FF_0'], calstars_ffs)

    # Both images get the average of both
    for ff_name in ['FF_0', 'FF_1']:
        assert recalibrated[ff_name].mag_lev == pytest.approx(11.1)


def test_neighbourhood_average_does_not_depend_on_the_order():
    """ Overlapping neighbourhoods must give every image the same offset in any order of the meteors. With the
        meteors two images apart, writing the averages back while iterating gave FF_3 an offset of 11.21 in one
        order and of 11.185 in the other. """

    # The same night, averaged with the meteors in both orders
    calstars_ffs, recalibrated = overlappingNeighbourhoodNight()

    forward = copy.deepcopy(recalibrated)
    backward = copy.deepcopy(recalibrated)

    ApplyRecalibrate.averageNeighbourPhotometry(forward, ['FF_3', 'FF_5'], calstars_ffs, neighbourhood_size=5)
    ApplyRecalibrate.averageNeighbourPhotometry(backward, ['FF_5', 'FF_3'], calstars_ffs, neighbourhood_size=5)

    # Every image gets the same offset in both orders
    for ff_name in recalibrated:
        assert forward[ff_name].mag_lev == pytest.approx(backward[ff_name].mag_lev)
        assert forward[ff_name].mag_lev_stddev == pytest.approx(backward[ff_name].mag_lev_stddev)


def test_neighbourhood_average_of_the_original_fits_only():
    """ An offset which is itself an average, or a degenerate fit which received an average, must never count again
        as an independent measurement. The neighbourhood of FF_3 (FF_1..FF_5) averages FF_1, FF_2, FF_4 and FF_5.
        The neighbourhood of FF_4 (FF_2..FF_6) must average the original FF_2, FF_4, FF_5 and FF_6, and still leave
        out FF_3. """

    # Meteors on FF_3 and FF_4, with neighbourhoods of 5 images
    calstars_ffs, recalibrated = overlappingNeighbourhoodNight()

    ApplyRecalibrate.averageNeighbourPhotometry(recalibrated, ['FF_3', 'FF_4'], calstars_ffs, neighbourhood_size=5)

    # Each meteor image carries the average of its own neighbourhood, of the original fits
    assert recalibrated['FF_3'].mag_lev == pytest.approx(np.mean([11.0, 11.4, 11.2, 10.6]))
    assert recalibrated['FF_4'].mag_lev == pytest.approx(np.mean([11.4, 11.2, 10.6, 11.8]))


def ffName(i):
    """ Name of the i-th FF file of a synthetic night, the files 10.24 s apart.

    Arguments:
        i: [int] Index of the FF file.

    Return:
        [str] FF file name.
    """

    # The time of the first frame of the file
    dt = datetime.datetime(2023, 2, 28, 18, 0, 0) + datetime.timedelta(seconds=10.24*i)

    return 'FF_XX0001_{:s}_{:03d}_{:07d}.fits'.format(dt.strftime('%Y%m%d_%H%M%S'), dt.microsecond//1000, 256*i)


def test_meteor_on_the_first_image_with_a_degenerate_fit(tmp_path, monkeypatch):
    """ Through the recalibration of a night: a meteor on the first image of CALSTARS, whose photometric fit is
        degenerate (the initial guess 10.0 with a zero standard deviation), is measured with the offset of the
        neighbouring image. The star matching is replaced by the photometric fits of the images. """

    # A night of four images
    config = makeConfig()
    ff_names = [ffName(i) for i in range(4)]

    # The photometric fits of the recalibration, the one of the first image is degenerate
    fits = {ff_names[0]: (10.0, 0.0), ff_names[1]: (11.2, 0.1), ff_names[2]: (11.3, 0.1), ff_names[3]: (11.1, 0.1)}

    # Every image is recalibrated, with its photometric fit
    def recalibrate(platepar, ff_processing_list, *args, **kwargs):

        recalibrated = {}
        for ff_name in ff_processing_list:
            recalibrated[ff_name] = copy.deepcopy(platepar)
            recalibrated[ff_name].auto_recalibrated = True
            recalibrated[ff_name].mag_lev, recalibrated[ff_name].mag_lev_stddev = fits[ff_name]

        return recalibrated

    # The photometric offset of the platepar which measures every meteor
    applied = {}

    # The measurement only records that offset
    def apply(ff_name, fps, meteor_meas, platepar, add_calstatus=False):
        applied[ff_name] = platepar.mag_lev
        return meteor_meas

    # One meteor on the first image
    meteor_list = [[ff_names[0], 1, 0.0, 0.0, []]]

    # The catalog, the FTPdetectinfo reading and writing, the star matching and the measurement are replaced
    monkeypatch.setattr(ApplyRecalibrate.StarCatalog, 'readStarCatalog',
                        lambda *args, **kwargs: (np.zeros((0, 3)), None, config.star_catalog_band_ratios))
    monkeypatch.setattr(ApplyRecalibrate.FTPdetectinfo, 'readFTPdetectinfo',
                        lambda *args, **kwargs: ('XX0001', config.fps, meteor_list))
    monkeypatch.setattr(ApplyRecalibrate.FTPdetectinfo, 'writeFTPdetectinfo', lambda *args, **kwargs: None)
    monkeypatch.setattr(ApplyRecalibrate, 'recalibratePlateparsForFF', recalibrate)
    monkeypatch.setattr(ApplyRecalibrate, 'applyPlateparToCentroids', apply)

    # The FTPdetectinfo file and CALSTARS with the four images
    ftpdetectinfo_path = tmp_path/'FTPdetectinfo_XX0001_20230228_180000_000000.txt'
    ftpdetectinfo_path.write_text('')
    calstars_data = ([[ff_name, np.zeros((0, 8))] for ff_name in ff_names], 256)

    # The recalibration of the night
    ApplyRecalibrate.recalibrateIndividualFFsAndApplyAstrometry(str(tmp_path), str(ftpdetectinfo_path),
                                                                calstars_data, config, makePlatepar(),
                                                                generate_plot=False)

    # The meteor is measured with the offset of the second image, the only usable fit of its neighbourhood
    assert applied[ff_names[0]] == pytest.approx(11.2)
