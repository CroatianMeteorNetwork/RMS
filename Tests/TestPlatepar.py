""" Tests for the pointing parameter helpers in RMS.Formats.Platepar. """

from __future__ import print_function, division, absolute_import

import pytest

np = pytest.importorskip("numpy")

from RMS.Formats.Platepar import normalizeRaDec


@pytest.mark.parametrize("ra, dec", [(0.0, 0.0), (123.4, 45.6), (359.9, -89.9), (10.0, 89.999)])
def testNormalizeRaDecInsideRange(ra, dec):
    """ Values already on the sphere are returned unchanged. """

    # Nothing to normalize, the input is already inside the valid range
    ra_out, dec_out, pos_angle_offset = normalizeRaDec(ra, dec)

    assert ra_out == pytest.approx(ra)
    assert dec_out == pytest.approx(dec)
    assert pos_angle_offset == 0.0


@pytest.mark.parametrize("dec", [90.0, -90.0])
def testNormalizeRaDecExactlyAtPoles(dec):
    """ The poles themselves are fixed points and the RA is not flipped. """

    # A declination of exactly +/- 90 deg is on the boundary and must not be reflected
    ra_out, dec_out, pos_angle_offset = normalizeRaDec(50.0, dec)

    assert ra_out == pytest.approx(50.0)
    assert dec_out == pytest.approx(dec)
    assert pos_angle_offset == 0.0


@pytest.mark.parametrize("ra, dec, ra_expected, dec_expected", [
    # Just past the north pole: reflect, RA advances by 180
    (50.0, 95.0, 230.0, 85.0),
    # Just past the south pole
    (50.0, -95.0, 230.0, -85.0),
    # Infinitesimally past the north pole must stay next to the north pole, not jump to -90
    (50.0, 90.000001, 230.0, 89.999999),
    # RA wraps around after the 180 deg advance
    (300.0, 95.0, 120.0, 85.0),
    # Dec of 180 is the antipode of dec 0 on the same meridian
    (0.0, 180.0, 180.0, 0.0),
    # Dec of 270 is the same as -90 (no RA flip needed)
    (0.0, 270.0, 0.0, -90.0),
    # Negative RA wraps into [0, 360)
    (-10.0, 20.0, 350.0, 20.0),
])
def testNormalizeRaDecReflectsAcrossPole(ra, dec, ra_expected, dec_expected):
    """ Declinations past a pole are reflected, with the RA advanced by 180 degrees. """

    # Every parametrized case crosses a pole, so the RA has to move by 180 deg
    ra_out, dec_out, pos_angle_offset = normalizeRaDec(ra, dec)

    assert ra_out == pytest.approx(ra_expected)
    assert dec_out == pytest.approx(dec_expected)

    # A reflection across the pole turns the local tangent frame by 180 deg
    reflected = abs(((ra - ra_expected + 180)%360) - 180) > 1e-9
    assert pos_angle_offset == (180.0 if reflected else 0.0)


def testNormalizeRaDecReflectionPreservesDirection():
    """ The reflected point must be the same unit vector as the raw (ra, dec) pair. """

    # A declination 7.5 deg past the north pole
    ra, dec = 37.0, 97.5
    ra_out, dec_out, pos_angle_offset = normalizeRaDec(ra, dec)

    def toVector(ra_d, dec_d):
        """ Convert an (RA, Dec) pair in degrees to a unit vector.

        Arguments:
            ra_d: [float] Right ascension (deg).
            dec_d: [float] Declination (deg).

        Return:
            vect: [ndarray] Unit vector pointing at the given direction.
        """

        ra_r, dec_r = np.radians(ra_d), np.radians(dec_d)
        return np.array([np.cos(dec_r)*np.cos(ra_r), np.cos(dec_r)*np.sin(ra_r), np.sin(dec_r)])

    # The normalized pair must point in exactly the same direction and be inside the valid ranges
    assert np.allclose(toVector(ra, dec), toVector(ra_out, dec_out))
    assert 0.0 <= ra_out < 360.0
    assert -90.0 <= dec_out <= 90.0


def testProjectionIsContinuousAcrossThePole():
    """ Stepping the declination parameter past the pole must not jump the projected field.

        The optimizers step the declination freely, so the parameter regularly crosses 90 deg for a
        camera pointing near the celestial pole. normalizeRaDec reflects the pointing back onto the
        sphere; without also rotating the position angle by the returned offset, the field would flip
        by 180 deg at the crossing and the cost function would be discontinuous.
    """

    from RMS.Formats.Platepar import Platepar
    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP

    platepar = Platepar()
    platepar.X_res, platepar.Y_res = 1920, 1080
    platepar.F_scale = 27.0
    platepar.JD = 2460000.5
    platepar.refraction = False

    ra_param = 30.0
    pos_angle_param = 40.0

    # Catalog stars a couple of degrees from the pole, so they stay in the field the whole way
    star_ra = np.array([30.0, 120.0, 210.0])
    star_dec = np.array([88.0, 88.5, 88.2])


    def projectStars(dec_param):
        """ Project the stars with the pointing given by the raw declination parameter. """

        ra_d, dec_d, pos_angle_offset = normalizeRaDec(ra_param, dec_param)

        platepar.RA_d = ra_d
        platepar.dec_d = dec_d
        platepar.pos_angle_ref = (pos_angle_param + pos_angle_offset)%360

        x_arr, y_arr = raDecToXYPP(star_ra, star_dec, platepar.JD, platepar)

        return np.c_[x_arr, y_arr]


    # Walk the declination parameter across the pole in even steps
    dec_params = [89.90, 89.96, 89.99, 90.01, 90.04, 90.10]
    positions = [projectStars(dec_param) for dec_param in dec_params]

    # Every step of the parameter moves the stars by a comparable amount, the crossing included
    steps = [np.max(np.abs(positions[i + 1] - positions[i])) for i in range(len(positions) - 1)]

    assert max(steps) < 5.0


RADIAL_TYPES = ["radial3-all", "radial4-all", "radial5-all", "radial3-odd", "radial5-odd", "radial7-odd",
                "radial9-odd"]

RADIAL_FLAGS = [
    # (flag name, restricted value, free value)
    ('force_distortion_centre', True, False),
    ('equal_aspect', True, False),
    ('asymmetry_corr', False, True),
]


def _radialPlatepar(dist_type):
    """ A platepar with a radial distortion with all flags in the restricted state and non-zero ks. """

    from RMS.Formats.Platepar import Platepar

    pp = Platepar()
    pp.X_res, pp.Y_res = 1920, 1080
    pp.F_scale = 1920/90.0
    pp.JD = 2460000.5
    pp.RA_d, pp.dec_d = 100.0, 30.0
    pp.pos_angle_ref = 20.0
    pp.lat, pp.lon = 45.0, 15.0
    pp.refraction = False

    pp.force_distortion_centre = True
    pp.equal_aspect = True
    pp.asymmetry_corr = False
    pp.setDistortionType(dist_type, reset_params=True)

    # Set small, non-zero radial coefficients (the array only contains the ks in this flag state)
    n = pp.poly_length
    ks = 0.02*(np.arange(n) + 1)*(-1)**np.arange(n)
    pp.x_poly_fwd = ks.copy()
    pp.x_poly_rev = -ks.copy()
    pp.y_poly_fwd = ks.copy()
    pp.y_poly_rev = -ks.copy()
    pp.x_poly = pp.x_poly_fwd
    pp.y_poly = pp.y_poly_fwd

    return pp


def _projection(pp):
    """ Project an image grid to the sky and a sky grid to the image, both used as a fingerprint. """

    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP

    xs, ys = np.meshgrid(np.linspace(10, pp.X_res - 10, 9), np.linspace(10, pp.Y_res - 10, 7))
    xs, ys = xs.ravel(), ys.ravel()

    _, ra, dec, _ = xyToRaDecPP(len(xs)*[pp.JD], xs, ys, np.ones(len(xs)), pp, extinction_correction=False,
        jd_time=True)
    x_back, y_back = raDecToXYPP(ra, dec, pp.JD, pp)

    return np.c_[ra, dec], np.c_[x_back, y_back]


@pytest.mark.parametrize("dist_type", RADIAL_TYPES)
def testBuildRadialCoeffsMatchesPolyLength(dist_type):
    """ Rebuilding the coefficients must give an array of the length the distortion type uses. """

    pp = _radialPlatepar(dist_type)

    for flag_state in [(True, True, False), (False, False, True)]:
        pp.force_distortion_centre, pp.equal_aspect, pp.asymmetry_corr = flag_state
        pp.setDistortionType(dist_type, reset_params=False)

        coeffs = pp.buildRadialCoeffs(pp.extractRadialCoeffs(np.arange(1, 20, dtype=float)))

        assert len(coeffs) == pp.poly_length


@pytest.mark.parametrize("dist_type", RADIAL_TYPES)
@pytest.mark.parametrize("flag_name, restricted_value, free_value", RADIAL_FLAGS)
def testRadialFlagToggleKeepsProjection(dist_type, flag_name, restricted_value, free_value):
    """ Freeing a distortion flag (and restricting it back) must not change the projection.

        The newly freed parameters start at the values the restricted state implies (the forced centre is
        half a pixel from the image centre, zero aspect and asymmetry), and every radial coefficient must be
        carried over.
    """

    pp = _radialPlatepar(dist_type)
    sky_ref, img_ref = _projection(pp)

    # Free the flag
    assert pp.remapCoeffsForFlagChange(flag_name, free_value)
    assert len(pp.x_poly_fwd) == pp.poly_length
    sky, img = _projection(pp)
    assert np.max(np.abs(img - img_ref)) < 1e-6
    assert np.max(np.abs(sky - sky_ref)) < 1e-6/3600

    # Restrict it again
    assert pp.remapCoeffsForFlagChange(flag_name, restricted_value)
    sky, img = _projection(pp)
    assert np.max(np.abs(img - img_ref)) < 1e-6
    assert np.max(np.abs(sky - sky_ref)) < 1e-6/3600


def _nnStartPlatepar():
    """ A platepar set up for the NN fit as AutoPlatepar does, with a stale star list. """

    from RMS.Formats.Platepar import Platepar

    pp = Platepar()
    pp.X_res, pp.Y_res = 1280, 720
    pp.F_scale = 1280/60.0
    pp.JD = 2460000.5
    pp.RA_d, pp.dec_d = 100.0, 30.0
    pp.pos_angle_ref = 20.0
    pp.lat, pp.lon = 45.0, 15.0
    pp.refraction = False
    pp.equal_aspect = True
    pp.asymmetry_corr = False
    pp.force_distortion_centre = False
    pp.setDistortionType("radial5-odd", reset_params=True)
    pp.x_poly_fwd[-1] = 0.01
    pp.x_poly_rev[-1] = -0.01
    pp.star_list = [[pp.JD, 1.0, 2.0, 3.0, 100.0, 30.0, 4.0]]

    return pp


def _nnCatalog(pp, n_stars, rng):
    """ Catalog stars (RA, dec, mag) at random positions inside the image of the given platepar. """

    from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP

    xs = rng.uniform(20, pp.X_res - 20, n_stars)
    ys = rng.uniform(20, pp.Y_res - 20, n_stars)
    _, ra, dec, _ = xyToRaDecPP(n_stars*[pp.JD], xs, ys, np.ones(n_stars), pp, extinction_correction=False,
        jd_time=True)

    return np.c_[ra, dec, rng.uniform(1, 5, n_stars)]


def _plateparState(pp):
    """ The fitted parameters of a platepar, for comparing before and after a fit. """

    return (pp.distortion_type, pp.poly_length, pp.RA_d, pp.dec_d, pp.pos_angle_ref, pp.F_scale,
            list(pp.x_poly_fwd), list(pp.x_poly_rev), list(pp.y_poly_fwd), list(pp.y_poly_rev))


def testNNFitTooFewStarsDoesNotPairByIndex():
    """ Too few stars for the NN fit must return None without building index-paired bogus star pairs. """

    pp = _nnStartPlatepar()
    rng = np.random.default_rng(0)
    catalog = _nnCatalog(pp, 50, rng)
    img_stars = np.c_[rng.uniform(0, pp.X_res, 3), rng.uniform(0, pp.Y_res, 3), np.ones(3)]

    state = _plateparState(pp)
    result = pp.fitAstrometry(pp.JD, img_stars, catalog, first_platepar_fit=True, use_nn_cost=True)

    assert result is None
    assert _plateparState(pp) == state
    assert pp.star_list == []


def testNNFitRejectionRestoresPlatepar():
    """ A rejected NN fit (RMSD too large) must leave the platepar parameters as they were at entry. """

    pp = _nnStartPlatepar()
    rng = np.random.default_rng(1)
    catalog = _nnCatalog(pp, 150, rng)

    # Detections unrelated to the catalog, so no pointing fits them
    n_det = 60
    img_stars = np.c_[rng.uniform(20, pp.X_res - 20, n_det), rng.uniform(20, pp.Y_res - 20, n_det),
                      np.ones(n_det)]

    # Point the platepar somewhere the catalog does not cover much, so the RMSD is large. The scale is held
    #   fixed so the fit cannot shrink the field onto a single catalog star, and the scene is seeded, so the
    #   fit is always rejected
    pp.RA_d = (pp.RA_d + 25.0)%360
    state = _plateparState(pp)

    result = pp.fitAstrometry(pp.JD, img_stars, catalog, first_platepar_fit=True, use_nn_cost=True,
        fixed_scale=True)

    assert result is None
    assert _plateparState(pp) == state
    assert pp.star_list == []


@pytest.mark.parametrize("n_det", [30, 60])
def testNNFitWithPolyDistortionIsRejectedBeforeFitting(monkeypatch, n_det):
    """ The NN mode with a polynomial distortion must be rejected before any fit on index-paired stars.

        The catalog and the detections have different lengths, which used to make the polynomial fit
        raise before the rejection was reached.
    """

    from RMS.Formats import Platepar as PlateparModule

    pp = _nnStartPlatepar()
    pp.setDistortionType("poly3+radial", reset_params=True)
    rng = np.random.default_rng(3)
    catalog = _nnCatalog(pp, 50, rng)
    img_stars = np.c_[rng.uniform(20, pp.X_res - 20, n_det), rng.uniform(20, pp.Y_res - 20, n_det),
                      np.ones(n_det)]

    # No least squares fit may run
    def noFit(*args, **kwargs):
        raise AssertionError("A distortion fit was run in the NN mode with a polynomial distortion")

    monkeypatch.setattr(PlateparModule.scipy.optimize, 'least_squares', noFit)

    state = _plateparState(pp)
    result = pp.fitAstrometry(pp.JD, img_stars, catalog, first_platepar_fit=True, use_nn_cost=True)

    assert result is None
    assert _plateparState(pp) == state
    assert pp.star_list == []


def _nnPointingScene(false_fraction, dec_d=None, seed=2):
    """ Synthetic camera, catalog and detections from the astrometry benchmark. """

    from Tests.BenchmarkAstrometryFit import (buildSyntheticPlatepar, buildSyntheticCatalog,
        buildSyntheticDetections)

    pp = buildSyntheticPlatepar()

    # Optionally point the camera close to the celestial pole
    if dec_d is not None:
        pp.F_scale = 40.0
        pp.refraction = False
        pp.RA_d, pp.dec_d, pp.pos_angle_ref = 10.0, dec_d, 30.0
        pp.updateRefAltAz()

    catalog = buildSyntheticCatalog(pp, np.random.RandomState(1), n_target=1500)
    img_stars, _ = buildSyntheticDetections(pp, catalog, np.random.RandomState(seed),
        false_fraction=false_fraction)

    return pp, catalog, img_stars


def _medianProjectionError(pp_true, pp_fit, catalog):
    """ Median pixel distance between the catalog projected with the true and the fitted platepar. """

    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP

    x0, y0 = raDecToXYPP(catalog[:, 0], catalog[:, 1], pp_true.JD, pp_true)
    x1, y1 = raDecToXYPP(catalog[:, 0], catalog[:, 1], pp_true.JD, pp_fit)
    in_img = (x0 >= 0) & (x0 < pp_true.X_res) & (y0 >= 0) & (y0 < pp_true.Y_res)

    return np.median(np.hypot(x1 - x0, y1 - y0)[in_img])


def testFitPointingNNNormalizesPolarPointing():
    """ A polar camera fitted from the other side of the pole must get a valid, normalized pointing.

        Starting at RA + 180 deg with the position angle rotated by 180 deg is the same field seen across
        the pole, so the optimizer converges beyond dec = 90 deg. That must be stored as a proper pointing.
    """

    import copy

    pp, catalog, img_stars = _nnPointingScene(0.1, dec_d=89.7)

    pp_fit = copy.deepcopy(pp)
    pp_fit.RA_d = (pp_fit.RA_d + 180.0)%360
    pp_fit.pos_angle_ref += 180.0

    success, _, _, _ = pp_fit.fitPointingNN(pp.JD, img_stars, catalog)

    assert success
    assert -90.0 <= pp_fit.dec_d <= 90.0
    assert 0.0 <= pp_fit.RA_d < 360.0
    assert _medianProjectionError(pp, pp_fit, catalog) < 0.1


@pytest.mark.parametrize("false_fraction", [0.1, 0.3])
def testFitPointingNNRobustToFalseDetections(false_fraction):
    """ False detections must not bias the NN pointing fit (the plain RMSD gave 0.2-0.5 px errors). """

    import copy

    pp, catalog, img_stars = _nnPointingScene(false_fraction)

    pp_fit = copy.deepcopy(pp)
    pp_fit.RA_d += 0.4
    pp_fit.dec_d += 0.3
    pp_fit.pos_angle_ref += 0.2

    success, _, inlier_fraction, _ = pp_fit.fitPointingNN(pp.JD, img_stars, catalog)

    assert success
    assert inlier_fraction > 0.7
    assert _medianProjectionError(pp, pp_fit, catalog) < 0.1


def testNearestCatalogStarsEmptyCatalog():
    """ An empty catalog gives NaN separations instead of an IndexError. """

    from RMS.Formats.Platepar import _nearestCatalogStars

    indices, seps = _nearestCatalogStars(np.array([10.0, 20.0]), np.array([5.0, 6.0]), np.array([]),
        np.array([]))

    assert len(indices) == 2
    assert np.all(np.isnan(seps))


def _nnRansacScene():
    """ Synthetic scene and start platepar for the NN RANSAC fit, as the benchmark sets it up. """

    import copy

    from Tests.BenchmarkAstrometryFit import (buildSyntheticPlatepar, buildSyntheticCatalog,
        buildSyntheticDetections)

    pp = buildSyntheticPlatepar()
    catalog = buildSyntheticCatalog(pp, np.random.RandomState(1), n_target=800)
    img_stars, _ = buildSyntheticDetections(pp, catalog, np.random.RandomState(2))

    pp_start = copy.deepcopy(pp)
    pp_start.RA_d = (pp_start.RA_d + 0.05)%360
    pp_start.dec_d = pp_start.dec_d + 0.05
    pp_start.equal_aspect = True
    pp_start.asymmetry_corr = False
    pp_start.force_distortion_centre = False
    pp_start.setDistortionType("radial5-odd", reset_params=True)

    return pp, pp_start, catalog, img_stars


def testNNFitHandlesNonFiniteDetection():
    """ A detection with a non-finite position must not crash the KD-tree cost of the NN fit. """

    pp, pp_start, catalog, img_stars = _nnRansacScene()

    img_stars = np.array(img_stars, dtype=np.float64)
    img_stars[5, 0] = np.nan

    result = pp_start.fitAstrometry(pp.JD, img_stars, catalog, first_platepar_fit=True, use_nn_cost=True)

    assert result is not None
    img_matched, cat_matched = result
    assert np.all(np.isfinite(img_matched[:, :2]))
    assert len(img_matched) > 0.5*len(img_stars)


def testNNFitHandlesZeroIntensities():
    """ Zero intensities (median <= 0) must fall back to uniform RANSAC sampling instead of raising. """

    pp, pp_start, catalog, img_stars = _nnRansacScene()

    img_stars = np.array(img_stars, dtype=np.float64)
    img_stars[:, 2] = 0.0

    result = pp_start.fitAstrometry(pp.JD, img_stars, catalog, first_platepar_fit=True, use_nn_cost=True)

    assert result is not None
    assert len(result[0]) > 0.5*len(img_stars)


@pytest.mark.parametrize("flag_state", [(True, True, False), (False, False, True)])
def testRadial3OddMatchesRadial5OddWithZeroK2(flag_state):
    """ radial3-odd (k1 only, the array ends there) must project exactly like radial5-odd with k2 = 0. """

    projections = []
    for dist_type in ["radial3-odd", "radial5-odd"]:

        pp = _radialPlatepar(dist_type)
        pp.force_distortion_centre, pp.equal_aspect, pp.asymmetry_corr = flag_state
        pp.setDistortionType(dist_type, reset_params=True)

        # Same k1 in both, the extra radial5-odd k2 is zero
        coeffs = {'x0': 0.001, 'y0': -0.002, 'xy': 0.003, 'a1': 0.002, 'a2': 0.1, 'k1': 0.05}
        pp.x_poly_fwd = pp.buildRadialCoeffs(coeffs)
        pp.x_poly_rev = pp.buildRadialCoeffs(dict(coeffs, k1=-0.05))
        pp.y_poly_fwd = pp.x_poly_fwd.copy()
        pp.y_poly_rev = pp.x_poly_rev.copy()
        pp.x_poly = pp.x_poly_fwd
        pp.y_poly = pp.y_poly_fwd

        assert len(pp.x_poly_fwd) == pp.poly_length
        projections.append(_projection(pp))

    assert np.array_equal(projections[0][0], projections[1][0])
    assert np.array_equal(projections[0][1], projections[1][1])
