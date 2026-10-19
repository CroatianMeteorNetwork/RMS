""" Catalog (J2000) directions have to be precessed to the epoch of date before they are converted to alt/az.
    Converting them as if they were of date puts them 18-25 arcmin off between 2021 and 2030. """

from __future__ import print_function, division, absolute_import

import numpy as np

from RMS.Astrometry.Conversions import geocentricToApparentRadiantAndVelocity, trueRaDec2ApparentAltAz, \
    date2JD, geo2Cartesian
from RMS.Astrometry.CyFunctions import subsetCatalog


def test_zenith_attraction_moves_the_radiant_vertically():
    """ The zenith attraction only pulls the radiant towards the zenith (the geocentric vertical, which the
        function uses): the apparent radiant keeps the azimuth of the geocentric one. Evaluated about the zenith
        of a J2000 radiant taken as a radiant of date, it also moved sideways, by up to 260 arcsec for slow
        meteors. """

    rng = np.random.default_rng(11)

    for _ in range(200):

        ra_g, dec_g = rng.uniform(0, 360), rng.uniform(-30, 80)
        vg = rng.uniform(11, 20)*1000
        lat, lon = rng.uniform(-45, 60), rng.uniform(-180, 180)
        jd = date2JD(2026, 8, 12, 0, 0, 0) + rng.uniform(0, 1)

        x, y, z = geo2Cartesian(lat, lon, 100000, jd)
        lat_geocentric = np.degrees(np.arctan2(z, np.hypot(x, y)))

        azim_g, elev_g = trueRaDec2ApparentAltAz(ra_g, dec_g, jd, lat_geocentric, lon, refraction=False)
        if not (5 < elev_g < 60):
            continue

        ra_a, dec_a, _ = geocentricToApparentRadiantAndVelocity(ra_g, dec_g, vg, lat, lon, 100000, jd,
            include_rotation=False)
        azim_a, elev_a = trueRaDec2ApparentAltAz(ra_a, dec_a, jd, lat_geocentric, lon, refraction=False)

        assert elev_a > elev_g
        assert abs((azim_a - azim_g + 180)%360 - 180)*np.cos(np.radians(elev_a))*3600 < 0.5


def test_catalog_horizon_cut_at_the_epoch_of_date():
    """ subsetCatalog keeps exactly the stars above -5 deg, with the elevation of the J2000 catalog
        positions taken at the epoch of date. """

    rng = np.random.default_rng(5)
    n = 20000
    catalog = np.column_stack([rng.uniform(0, 360, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
        rng.uniform(-1, 9, n)])
    catalog = catalog[np.argsort(-catalog[:, 1])]

    jd = date2JD(2026, 1, 15, 3, 0, 0)
    lat, lon = 45.0, 16.0

    indices, _ = subsetCatalog(catalog, 0.0, 90.0, jd, lat, lon, 180.0, 10.0)
    kept = np.isin(np.arange(n), indices)

    _, elev = trueRaDec2ApparentAltAz(catalog[:, 0].copy(), catalog[:, 1].copy(), jd, lat, lon, refraction=False)

    # Away from a 1 arcsec band around the cut, the subset is exactly the stars above -5 deg
    clear = np.abs(elev + 5) > 1/3600
    assert np.array_equal(kept[clear], (elev >= -5)[clear])
