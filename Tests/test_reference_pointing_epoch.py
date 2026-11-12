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

def separationArcmin(az1, alt1, az2, alt2):
    cos_sep = np.sin(alt1)*np.sin(alt2) + np.cos(alt1)*np.cos(alt2)*np.cos(az1 - az2)
    return np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))*60


class TestReferencePointingEpoch(unittest.TestCase):

    def testPointingSetFromAltAzLandsWhereRequested(self):

        for refraction in (True, False):

            pp = Platepar()
            pp.read(TEMPLATE)
            pp.lat, pp.lon, pp.elev = 45.0, 15.0, 300.0
            pp.refraction = refraction
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

            self.assertLess(separationArcmin(az, alt, np.radians(200.0), np.radians(45.0)), 0.1,
                "refraction={}".format(refraction))

            # The inverse helper recovers the alt/az the pointing was set from
            pp.updateRefAltAz()
            self.assertAlmostEqual(pp.az_centre, 200.0, places=3)
            self.assertAlmostEqual(pp.alt_centre, 45.0, places=3)

            # computeRefAltAz has no side effects and agrees with updateRefAltAz
            az_c, alt_c = pp.computeRefAltAz()
            self.assertAlmostEqual(az_c, pp.az_centre, places=9)
            self.assertAlmostEqual(alt_c, pp.alt_centre, places=9)


if __name__ == "__main__":
    unittest.main()
