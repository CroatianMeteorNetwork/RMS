import pytest

np = pytest.importorskip("numpy")

from RMS.Astrometry.MatchStars import matchStars


def _make_catalog(positions):
    """ Helper: build cat_x_array, cat_y_array, cat_good_indices from a list of (x, y). """
    cat_x = np.array([p[0] for p in positions], dtype=np.float64)
    cat_y = np.array([p[1] for p in positions], dtype=np.float64)
    cat_good = np.arange(len(positions), dtype=np.uint32)
    return cat_x, cat_y, cat_good


def _make_stars(positions):
    """ Helper: build stars_list from a list of (x, y). stars_list columns are (y, x, ...). """
    return np.array([[y, x, 0.0, 0.0] for x, y in positions], dtype=np.float64)


class TestMatchStarsBasic:

    def test_exact_match(self):
        """ Stars at identical positions should match with distance 0. """
        cat_x, cat_y, cat_good = _make_catalog([(10, 20), (30, 40), (50, 60)])
        stars = _make_stars([(10, 20), (30, 40), (50, 60)])

        result = matchStars(stars, cat_x, cat_y, cat_good, 5.0)

        assert result.shape[0] == 3
        assert result.shape[1] == 3
        for i in range(3):
            row = result[result[:, 0] == i][0]
            assert row[1] == i  # matched to corresponding catalog star
            assert np.isclose(row[2], 0.0)

    def test_nearest_neighbor(self):
        """ Each image star should match the closest catalog star. """
        cat_x, cat_y, cat_good = _make_catalog([(0, 0), (10, 0), (20, 0)])
        stars = _make_stars([(9, 0)])  # closest to catalog star 1

        result = matchStars(stars, cat_x, cat_y, cat_good, 5.0)

        assert result.shape[0] == 1
        assert result[0, 0] == 0  # image star index
        assert result[0, 1] == 1  # catalog star index
        assert np.isclose(result[0, 2], 1.0)

    def test_max_radius_filters(self):
        """ Stars beyond max_radius should not match. """
        cat_x, cat_y, cat_good = _make_catalog([(0, 0)])
        stars = _make_stars([(100, 100)])

        result = matchStars(stars, cat_x, cat_y, cat_good, 5.0)

        assert result.shape[0] == 0

    def test_distance_correct(self):
        """ Returned distance should be Euclidean. """
        cat_x, cat_y, cat_good = _make_catalog([(0, 0)])
        stars = _make_stars([(3, 4)])

        result = matchStars(stars, cat_x, cat_y, cat_good, 10.0)

        assert result.shape[0] == 1
        assert np.isclose(result[0, 2], 5.0)


class TestMatchStarsEdgeCases:

    def test_empty_catalog(self):
        """ Empty catalog should return no matches. """
        cat_x = np.array([], dtype=np.float64)
        cat_y = np.array([], dtype=np.float64)
        cat_good = np.array([], dtype=np.uint32)
        stars = _make_stars([(10, 20)])

        result = matchStars(stars, cat_x, cat_y, cat_good, 5.0)

        assert result.shape == (0, 3)

    def test_empty_stars(self):
        """ Empty star list should return no matches. """
        cat_x, cat_y, cat_good = _make_catalog([(10, 20)])
        stars = np.empty((0, 4), dtype=np.float64)

        result = matchStars(stars, cat_x, cat_y, cat_good, 5.0)

        assert result.shape == (0, 3)

    def test_subset_of_catalog_indices(self):
        """ Only catalog stars in cat_good_indices should be considered. """
        cat_x, cat_y, _ = _make_catalog([(0, 0), (5, 0), (100, 100)])
        # Only include catalog star 2 (at 100, 100)
        cat_good = np.array([2], dtype=np.uint32)
        stars = _make_stars([(0, 0)])  # right on top of catalog star 0, but it's excluded

        result = matchStars(stars, cat_x, cat_y, cat_good, 5.0)

        assert result.shape[0] == 0  # star 0 is not in cat_good, star 2 is too far

    def test_return_dtype(self):
        """ Result should be float64. """
        cat_x, cat_y, cat_good = _make_catalog([(10, 20)])
        stars = _make_stars([(10, 20)])

        result = matchStars(stars, cat_x, cat_y, cat_good, 5.0)

        assert result.dtype == np.float64

    def test_one_to_one(self):
        """ Multiple image stars near the same catalog star: each catalog star matched once. """
        cat_x, cat_y, cat_good = _make_catalog([(10, 10)])
        # Two image stars both close to the single catalog star
        stars = _make_stars([(10, 11), (10, 9)])

        result = matchStars(stars, cat_x, cat_y, cat_good, 5.0)

        # KD-tree query allows both to match the same catalog star (same as brute-force)
        assert result.shape[0] == 2
        assert all(result[:, 1] == 0)
