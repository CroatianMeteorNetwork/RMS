""" The platepar reference pointing RA_d/dec_d is read by the projection kernels as true equatorial coordinates
in the EPOCH OF DATE of the platepar JD (pointingCorrection shifts it by sidereal time and precesses to J2000
itself). The alt/az <-> RA/Dec helpers must use the same convention, otherwise a pointing set from alt/az
lands ~20 arcmin (the precession since 2000) from where it was requested.
Run with: python -m unittest Tests.test_reference_pointing_epoch
"""

from __future__ import print_function, division, absolute_import

import os
import unittest

import numpy as np

from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP
from RMS.Astrometry.Conversions import jd2Date
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz, j2000FromTrueOfDate
from RMS.Misc import getRmsRootDir


TEMPLATE = os.path.join(getRmsRootDir(), 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')

# Refraction apparent -> true and true -> apparent are two different approximations (Bennett and
#   Saemundsson), which disagree by about 1.1 arcsec at 45 deg: a round trip through both directions closes to
#   this level, without refraction to rounding
REFRACTION_ROUND_TRIP = {True: 1.5, False: 1.0e-3}


def separationArcsec(az1, alt1, az2, alt2):
    """ Helper: angular separation (arcsec) of two alt/az directions (radians), from atan2 so that it resolves
        below a milliarcsecond. """

    v1 = np.array([np.cos(alt1)*np.sin(az1), np.cos(alt1)*np.cos(az1), np.sin(alt1)])
    v2 = np.array([np.cos(alt2)*np.sin(az2), np.cos(alt2)*np.cos(az2), np.sin(alt2)])

    return np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2)), v1.dot(v2)))*3600


class TestReferencePointingEpoch(unittest.TestCase):

    def testPointingSetFromAltAzLandsWhereRequested(self):

        for refraction in (True, False):

            pp = Platepar()
            pp.read(TEMPLATE)
            pp.lat, pp.lon, pp.elev = 45.0, 15.0, 300.0
            pp.refraction = refraction

            # Distortion-free, so that the pixel round trip is exact and only the frames are tested
            pp.resetDistortionParameters()
            pp.x_poly_fwd[0] = pp.x_poly_fwd[1] = pp.x_poly_rev[0] = pp.x_poly_rev[1] = 0.0
            pp.alt_centre, pp.az_centre, pp.pos_angle_ref = 45.0, 200.0, 20.0
            pp.updateRefRADec(skip_rot_update=True)

            # Pixel of the reference direction (raDecToXYPP takes J2000), then back through the calibrated
            # path to apparent alt/az
            ra_j, dec_j = j2000FromTrueOfDate(pp.JD, np.radians(pp.RA_d), np.radians(pp.dec_d))
            xc, yc = raDecToXYPP(np.array([np.degrees(ra_j)]), np.array([np.degrees(dec_j)]), pp.JD, pp)
            jd_arr, ra, dec, _ = xyToRaDecPP([jd2Date(pp.JD)], [xc[0]], [yc[0]], [1], pp,
                extinction_correction=False)
            az, alt = cyTrueRaDec2ApparentAltAz(np.radians(ra[0]), np.radians(dec[0]), jd_arr[0],
                np.radians(pp.lat), np.radians(pp.lon), refraction)

            # Two refraction round trips: the pointing set from alt/az, and the pixel back to alt/az
            self.assertLess(separationArcsec(az, alt, np.radians(200.0), np.radians(45.0)),
                2*REFRACTION_ROUND_TRIP[refraction] + 0.01, "refraction={}".format(refraction))

            # The inverse helper recovers the alt/az the pointing was set from (the azimuth is not refracted)
            pp.updateRefAltAz()
            self.assertLess(abs(pp.az_centre - 200.0)*3600, 1.0e-3)
            self.assertLess(abs(pp.alt_centre - 45.0)*3600, REFRACTION_ROUND_TRIP[refraction])

            # computeRefAltAz has no side effects and agrees with updateRefAltAz
            az_c, alt_c = pp.computeRefAltAz()
            self.assertAlmostEqual(az_c, pp.az_centre, places=9)
            self.assertAlmostEqual(alt_c, pp.alt_centre, places=9)


if __name__ == "__main__":
    unittest.main()
