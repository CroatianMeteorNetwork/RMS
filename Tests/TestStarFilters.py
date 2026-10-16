""" Tests for the catalog pre-filters used by the blended-star rejection.

Covers the FOV cone pre-filter (units, exactness and the epoch of its centre) and the
neighbour search in filterBlendedStars.
"""

from __future__ import print_function, division, absolute_import

import os

import pytest

np = pytest.importorskip("numpy")

from RMS.Astrometry.ApplyAstrometry import getFOVSelectionRadius, raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.StarClasses import CatalogStar, PairedStars
from RMS.Astrometry.StarFilters import catalogStarsInFOV, filterBlendedStars
from RMS.Formats.Platepar import Platepar


# Real 720p platepar shipped with RMS. Using a fitted platepar rather than a hand-made one
#   matters here: it carries actual distortion coefficients, and its reference RA/Dec is
#   consistent with the projection, which a synthetic platepar is not.
TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'share', 'platepar_templates', 'template_generic_720p_4mm.cal')


def makePlatepar():
    """ Platepar from the shipped 4 mm 720p template.

    The template's reference JD is left alone - its RA/Dec, Alt/Az and pointing are only
    mutually consistent at the epoch it was fitted for.
    """

    if not os.path.isfile(TEMPLATE):
        pytest.skip("platepar template not available: {:s}".format(TEMPLATE))

    pp = Platepar()
    pp.read(TEMPLATE)
    pp.refraction = False

    return pp


def pixelToRaDec(pp, x, y, jd=None):
    """ RA/Dec of the given image coordinates, as arrays. """

    if jd is None:
        jd = pp.JD

    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))

    _, ra, dec, _ = xyToRaDecPP(len(x)*[jd], x, y, np.ones(len(x)), pp,
                                extinction_correction=False, jd_time=True)

    return ra, dec


def fovCentre(pp, jd=None):
    """ RA/Dec of the FOV centre at the given time. """

    ra, dec = pixelToRaDec(pp, pp.X_res/2.0, pp.Y_res/2.0, jd=jd)

    return ra[0], dec[0]


def angularSeparationDeg(ra1, dec1, ra2, dec2):
    """ Great-circle separation between two points (deg). """

    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    cos_sep = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2)

    return np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))


class TestCatalogStarsInFOV(object):

    def test_units_regression(self):
        """ The cone must not be a hemisphere pass-through.

        Guards the original bug: multiplying by F_scale (px/deg) instead of dividing put
        the radius in the thousands of degrees, so the min(..., 90) cap always fired and
        every catalog star in front of the camera survived.
        """

        pp = makePlatepar()
        ra_c, dec_c = fovCentre(pp)

        # A star 85 deg off-axis is in front of the camera but nowhere near the FOV, so it
        #   is only accepted if the cone has degenerated to a hemisphere
        far_ra = np.array([ra_c])
        far_dec = np.array([dec_c - 85.0])

        assert angularSeparationDeg(ra_c, dec_c, far_ra[0], far_dec[0]) > 80
        assert not catalogStarsInFOV(far_ra, far_dec, pp, pp.JD)[0]

    def test_image_corners_are_inside(self):
        """ Every pixel of the image, corners included, must survive the pre-filter. """

        pp = makePlatepar()

        xs = [0, pp.X_res, 0, pp.X_res, pp.X_res/2.0]
        ys = [0, pp.Y_res, pp.Y_res, 0, pp.Y_res/2.0]
        ra, dec = pixelToRaDec(pp, xs, ys)

        assert np.all(catalogStarsInFOV(ra, dec, pp, pp.JD))

        # Still true with essentially no margin - getFOVSelectionRadius circumscribes the
        #   image, so the corners land on the radius itself. A hair over 1.0 keeps this off
        #   an exact float boundary, since the two sides compute the separation differently.
        assert np.all(catalogStarsInFOV(ra, dec, pp, pp.JD, margin=1.001))

    def test_radius_is_at_least_the_selection_radius(self):
        """ The accepted cone must cover getFOVSelectionRadius, i.e. the whole image.

        Direct regression guard on the units bug, phrased as the property that matters:
        anything the platepar can image has to be considered.
        """

        pp = makePlatepar()
        ra_c, dec_c = fovCentre(pp)
        fov_radius = getFOVSelectionRadius(pp)

        # Walk outwards along a meridian and find where the filter starts rejecting
        offsets = np.arange(0, 90, 0.5)
        ra = np.full(offsets.shape, ra_c)
        dec = dec_c - offsets

        accepted = catalogStarsInFOV(ra, dec, pp, pp.JD)
        seps = angularSeparationDeg(ra_c, dec_c, ra, dec)

        assert np.max(seps[accepted]) >= fov_radius

    def test_cone_centre_follows_jd(self):
        """ The cone is centred on the pointing at jd, not on platepar.RA_d/dec_d.

        An alt-az camera sweeps ~15 deg/hour in RA, so a platepar reused across a night
        must not keep filtering against the pointing it was fitted at.
        """

        pp = makePlatepar()

        jd_later = pp.JD + 0.5  # 12 h later, i.e. roughly the opposite side of the sky
        ra_later, dec_later = fovCentre(pp, jd=jd_later)
        ra_arr, dec_arr = np.array([ra_later]), np.array([dec_later])

        # Where the camera is actually pointing at jd_later
        assert catalogStarsInFOV(ra_arr, dec_arr, pp, jd_later)[0]

        # ...is not where it pointed at platepar.JD
        assert not catalogStarsInFOV(ra_arr, dec_arr, pp, pp.JD)[0]

    def test_empty_catalog(self):
        """ An empty catalog gives an empty mask rather than an error. """

        pp = makePlatepar()
        mask = catalogStarsInFOV(np.array([]), np.array([]), pp, pp.JD)

        assert len(mask) == 0


def buildCatalog(pp, pixel_positions, mag=5.0, jd=None):
    """ Catalog array [ra, dec, mag] for stars at the given image coordinates. """

    xs = [p[0] for p in pixel_positions]
    ys = [p[1] for p in pixel_positions]
    ra, dec = pixelToRaDec(pp, xs, ys, jd=jd)

    return np.column_stack([ra, dec, np.full(len(ra), mag)])


def buildPairedStars(pp, pixel_positions, fwhm=3.0, mag=5.0, jd=None):
    """ PairedStars whose catalog coordinates project back to the given image coordinates. """

    catalog = buildCatalog(pp, pixel_positions, mag=mag, jd=jd)

    paired = PairedStars()
    for (x, y), (ra, dec, star_mag) in zip(pixel_positions, catalog):
        paired.addPair(x, y, fwhm, 1000.0, CatalogStar(ra, dec, star_mag))

    return paired


class TestFilterBlendedStars(object):

    # 5 matched stars spread across the frame - the filter no-ops below 5 pairs
    BASE = [(300, 200), (500, 250), (700, 300), (900, 350), (400, 500)]

    def test_close_neighbour_is_flagged(self):
        """ A catalog star at 1.5x FWHM is inside the 2x FWHM blend radius. """

        pp = makePlatepar()
        paired = buildPairedStars(pp, self.BASE, fwhm=3.0)

        # 4.5 px away from the first matched star, blend radius is 2*3 = 6 px
        catalog = buildCatalog(pp, self.BASE + [(304.5, 200)])

        filtered, removed = filterBlendedStars(paired, catalog, pp, pp.JD, 6.0)

        assert removed == 1
        assert len(filtered) == len(paired) - 1

    def test_distant_neighbour_is_kept(self):
        """ A catalog star at 3x FWHM is outside the blend radius. """

        pp = makePlatepar()
        paired = buildPairedStars(pp, self.BASE, fwhm=3.0)

        # 9 px away, blend radius is 6 px
        catalog = buildCatalog(pp, self.BASE + [(309, 200)])

        filtered, removed = filterBlendedStars(paired, catalog, pp, pp.JD, 6.0)

        assert removed == 0
        assert len(filtered) == len(paired)

    def test_self_match_is_not_a_blend(self):
        """ A star's own catalog entry sits at d ~ 0 and must not flag it. """

        pp = makePlatepar()
        paired = buildPairedStars(pp, self.BASE, fwhm=3.0)
        catalog = buildCatalog(pp, self.BASE)

        _, removed = filterBlendedStars(paired, catalog, pp, pp.JD, 6.0)

        assert removed == 0

    def test_duplicate_catalog_entry_does_not_hide_a_blend(self):
        """ Duplicate entries at the star's own position must not mask a real neighbour.

        The nearest-neighbour search asks for several neighbours precisely so that two
        coincident catalog rows cannot crowd out the genuine one.
        """

        pp = makePlatepar()
        paired = buildPairedStars(pp, self.BASE, fwhm=3.0)

        # The first matched star's position twice, plus a genuine neighbour at 4.5 px
        catalog = buildCatalog(pp, self.BASE + [(300, 200), (304.5, 200)])

        _, removed = filterBlendedStars(paired, catalog, pp, pp.JD, 6.0)

        assert removed == 1

    def test_matches_bruteforce_reference(self):
        """ The KD-tree search must agree with an explicit O(N*M) distance scan.

        Replaces the chunked-vs-unchunked check the chunking approach needed.
        """

        pp = makePlatepar()
        rng = np.random.RandomState(42)

        matched_positions = [(float(x), float(y)) for x, y in
                             rng.uniform([100, 100], [1100, 600], size=(40, 2))]
        catalog_positions = [(float(x), float(y)) for x, y in
                             rng.uniform([100, 100], [1100, 600], size=(400, 2))]

        fwhm = 4.0
        paired = buildPairedStars(pp, matched_positions, fwhm=fwhm)
        catalog = buildCatalog(pp, matched_positions + catalog_positions)

        filtered, removed = filterBlendedStars(paired, catalog, pp, pp.JD, 6.0)

        # Reference: project everything the same way the filter does, then scan
        cat_x, cat_y = raDecToXYPP(catalog[:, 0], catalog[:, 1], pp.JD, pp)
        matched_x = np.array([p[0] for p in matched_positions])
        matched_y = np.array([p[1] for p in matched_positions])

        blend_radius = 2.0*fwhm
        expected = 0
        for mx, my in zip(matched_x, matched_y):
            dist = np.hypot(cat_x - mx, cat_y - my)
            if np.any((dist < blend_radius) & (dist > 0.1)):
                expected += 1

        assert expected > 0, "test data should contain at least one blend"
        assert removed == expected
        assert len(filtered) == len(matched_positions) - expected

    def test_deep_catalog_outside_the_fov_is_cheap(self):
        """ A catalog covering the whole sky must not be carried into the distance search.

        The pre-filters exist so a deep catalog cannot allocate an (n_matched x n_catalog)
        matrix; this checks the result is unaffected by stars that cannot possibly blend.
        """

        pp = makePlatepar()
        paired = buildPairedStars(pp, self.BASE, fwhm=3.0)

        near_catalog = buildCatalog(pp, self.BASE + [(304.5, 200)])

        # Whole-sky catalog of equally bright stars, almost all of it behind the camera or
        #   far off-axis
        rng = np.random.RandomState(7)
        ra_all = rng.uniform(0, 360, 20000)
        dec_all = np.degrees(np.arcsin(rng.uniform(-1, 1, 20000)))
        sky = np.column_stack([ra_all, dec_all, np.full(len(ra_all), 5.0)])

        catalog = np.vstack([near_catalog, sky])

        _, removed = filterBlendedStars(paired, catalog, pp, pp.JD, 6.0)

        assert removed == 1

    def test_below_minimum_pairs_is_a_noop(self):
        """ Fewer than 5 pairs returns the input untouched. """

        pp = makePlatepar()
        paired = buildPairedStars(pp, self.BASE[:4], fwhm=3.0)
        catalog = buildCatalog(pp, self.BASE[:4])

        filtered, removed = filterBlendedStars(paired, catalog, pp, pp.JD, 6.0)

        assert removed == 0
        assert filtered is paired
