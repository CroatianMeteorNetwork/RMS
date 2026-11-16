""" The true equator and equinox of date outside the astrometric kernels: the J2000 -> date rotation matrix, the
    RA/Dec of AST plates and UWO event files, and the alt/az of the FOV area. Run with:
    python -m unittest Tests.test_frame_of_date """

from __future__ import print_function, division, absolute_import

import os
import shutil
import tempfile
import types
import unittest

import numpy as np

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.Conversions import (date2JD, jd2Date, JD2LST, JD2HourAngle, geo2Cartesian, altAz2RADec,
    trueOfDateRaDec2J2000, j2000RaDec2TrueOfDate, j2000ToTrueOfDateMatrix, trueRaDec2ApparentAltAz)
from RMS.Formats.Platepar import Platepar
from RMS.Misc import getRmsRootDir


DATES = [date2JD(2021, 3, 1, 3, 0, 0), date2JD(2026, 1, 15, 3, 0, 0), date2JD(2029, 6, 20, 3, 0, 0)]

TEMPLATE = os.path.join(getRmsRootDir(), 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')


def unitVector(ra, dec):
    """ Helper: unit vectors of RA/Dec in degrees. """

    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)], axis=-1)


def separationArcsec(ra1, dec1, ra2, dec2):
    """ Helper: angle between directions given in degrees (arcsec), from the cross and dot products. """

    v1, v2 = unitVector(ra1, dec1), unitVector(ra2, dec2)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2), axis=-1), np.sum(v1*v2, axis=-1)))*3600


class TestFrameOfDate(unittest.TestCase):

    def testMatrixIsTheTransformOfTheConverters(self):
        """ The matrix rotates J2000 vectors to the same true-of-date directions as j2000RaDec2TrueOfDate, its
            transpose goes back, and it is a rotation. """

        rng = np.random.default_rng(1)
        ra = rng.uniform(0, 360, 200)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 200)))

        for jd in DATES:

            m = j2000ToTrueOfDateMatrix(jd)
            self.assertLess(np.max(np.abs(m.dot(m.T) - np.eye(3))), 1e-14)
            self.assertAlmostEqual(np.linalg.det(m), 1.0, places=14)

            # Forward against the converter
            v_date = unitVector(ra, dec).dot(m.T)
            ra_t, dec_t = j2000RaDec2TrueOfDate(ra, dec, jd)
            ra_m = np.degrees(np.arctan2(v_date[:, 1], v_date[:, 0]))
            dec_m = np.degrees(np.arcsin(v_date[:, 2]))
            self.assertLess(np.max(separationArcsec(ra_t, dec_t, ra_m, dec_m)), 1e-6)

            # And back
            v_j2000 = v_date.dot(m)
            self.assertLess(np.max(np.abs(v_j2000 - unitVector(ra, dec))), 1e-14)


    def testMatrixFrameIsTheFrameOfTheObserver(self):
        """ The frame of date of the matrix is the one of geo2Cartesian (the apparent sidereal time): a line of
            sight rotated into it has the alt/az of trueRaDec2ApparentAltAz, measured from the observer's position
            in that frame. """

        lat, lon, ht = 43.2, -81.3, 330.0
        for jd in DATES:

            # The observer's position, at the right ascension of the local sidereal time
            station = np.array(geo2Cartesian(lat, lon, ht, jd))
            ra_station = np.degrees(np.arctan2(station[1], station[0]))%360
            self.assertLess(abs((ra_station - JD2LST(jd, lon)[0] + 180)%360 - 180)*3600, 1e-6)

            # A J2000 line of sight 30 deg east of the meridian and 20 deg north: its hour angle and declination
            #   in the frame of the matrix give the alt/az of the converter
            ra_j, dec_j = trueOfDateRaDec2J2000(JD2LST(jd, lon)[0] + 30.0, 20.0, jd)
            v_date = j2000ToTrueOfDateMatrix(jd).dot(unitVector(ra_j, dec_j))
            ha = np.radians(JD2LST(jd, lon)[0]) - np.arctan2(v_date[1], v_date[0])
            dec = np.arcsin(v_date[2])
            phi = np.radians(lat)
            alt = np.degrees(np.arcsin(np.sin(phi)*np.sin(dec) + np.cos(phi)*np.cos(dec)*np.cos(ha)))
            az = np.degrees(np.arctan2(-np.cos(dec)*np.sin(ha),
                np.sin(dec)*np.cos(phi) - np.cos(dec)*np.sin(phi)*np.cos(ha)))%360
            az_c, alt_c = trueRaDec2ApparentAltAz(ra_j, dec_j, jd, lat, lon, refraction=False)
            self.assertLess(abs(alt - alt_c)*3600, 1e-4)
            self.assertLess(abs((az - az_c + 180)%360 - 180)*np.cos(np.radians(alt))*3600, 1e-4)


    def testAstPlateGivesJ2000(self):
        """ The RA/Dec of an AST plate are J2000: the alt/az of the plate taken to the true equator and equinox
            of the date of each point, then to J2000 (in 2026 the frame of date is ~22 arcmin from J2000), and
            back to the same alt/az, as Detection does for the ASGARD event file. """

        from RMS.Formats import AST

        # Two points a day apart, with the alt/az the plate gives for their pixels
        az, alt = np.array([123.4, 250.0]), np.array([47.5, 21.0])
        time_data = [[2026, 1, 15, 3, 0, 0, 0], [2026, 1, 16, 3, 0, 0, 0]]
        ast = types.SimpleNamespace(lat=np.radians(43.2), lon=np.radians(-81.3))

        original = AST.plateASTMap
        AST.plateASTMap = lambda ast_, x, y: (np.radians(90.0 - alt), np.radians(90.0 - az))
        try:
            jd, ra, dec, _ = AST.xyToRaDecAST(time_data, [10.0, 20.0], [30.0, 40.0], [1000, 1000], ast, 10.0)
        finally:
            AST.plateASTMap = original

        for i in range(2):

            # Independently: the RA/Dec of date of the alt/az, then to J2000
            ra_d, dec_d = altAz2RADec(float(az[i]), float(alt[i]), jd[i], 43.2, -81.3)
            ra_e, dec_e = trueOfDateRaDec2J2000(ra_d, dec_d, jd[i])
            self.assertLess(separationArcsec(ra[i], dec[i], ra_e, dec_e), 0.01)

            # Back to the alt/az of the plate
            az_b, alt_b = trueRaDec2ApparentAltAz(ra[i], dec[i], jd[i], 43.2, -81.3, refraction=False)
            self.assertLess(abs(alt_b - alt[i])*3600, 0.01)
            self.assertLess(abs((az_b - az[i] + 180)%360 - 180)*np.cos(np.radians(alt[i]))*3600, 0.01)


    def testEventFileGivesJ2000(self):
        """ The RA/Dec read from a UWO event file (which FTPdetectinfo stores as J2000) are J2000. """

        from RMS.Formats.ConvertEvUWOtoFTPdetectinfo import readEvFile

        lat, lon = 43.2, -81.3
        theta, phi = np.array([40.0, 41.0, 42.0]), np.array([10.0, 12.0, 14.0])
        lines = ["#unix = 1768446000.000000", "#site = 01", "#latlon = {:.6f} {:.6f} 300.0".format(lat, lon),
            "#stream = A"]
        for i in range(3):
            lines.append("{:d} {:.3f} 1000 0 {:.2f} {:.2f} {:.4f} {:.4f} 0 {:.2f}".format(i, 0.01*i, 100.0 + i,
                200.0 + i, theta[i], phi[i], 2.0 + i))

        dir_path = tempfile.mkdtemp()
        try:
            with open(os.path.join(dir_path, "ev_test.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            out = readEvFile(dir_path, "ev_test.txt")
        finally:
            shutil.rmtree(dir_path)

        jd_ref, ra, dec = out[0], out[9], out[10]
        for i in range(3):
            ra_d, dec_d = altAz2RADec(90.0 - phi[i], 90.0 - theta[i], jd_ref, lat, lon)
            ra_e, dec_e = trueOfDateRaDec2J2000(ra_d, dec_d, jd_ref)
            self.assertLess(separationArcsec(ra[i], dec[i], ra_e, dec_e), 0.01)


    def testFovAreaUsesTheAltAzOfTheImagePoint(self):
        """ The FOV area traces the line of sight of an image point at its true alt/az, the same at any time for a
            camera fixed to the Earth: on a plate with the aberration modelled and the refraction on, as measured
            at the time of the plate. """

        from RMS.Routines import FOVArea

        pp = Platepar()
        pp.read(TEMPLATE)
        pp.aberration = True
        pp.refraction = True
        pp.lat, pp.lon, pp.elev = 43.2, -81.3, 330.0
        pp.JD = date2JD(2026, 1, 15, 3, 0, 0)
        pp.Ho = JD2HourAngle(pp.JD)
        pp.alt_centre, pp.az_centre, pp.pos_angle_ref = 50.0, 120.0, 10.0
        pp.updateRefRADec(skip_rot_update=True)
        pp.resetDistortionParameters()

        calls = []
        original = FOVArea.AEH2LatLonAlt

        def _spy(azim, elev, *args):
            calls.append((azim, elev))
            return original(azim, elev, *args)

        FOVArea.AEH2LatLonAlt = _spy
        try:
            for x, y in ((pp.X_res/2, pp.Y_res/2), (100.0, 100.0)):
                FOVArea.xyHt2Geo(pp, x, y, 100000)
        finally:
            FOVArea.AEH2LatLonAlt = original

        self.assertEqual(len(calls), 2)
        for (x, y), (azim, elev) in zip(((pp.X_res/2, pp.Y_res/2), (100.0, 100.0)), calls):
            self.assertGreater(elev, 10.0)
            _, ra, dec, _ = xyToRaDecPP([jd2Date(pp.JD)], [x], [y], [1], pp, extinction_correction=False,
                measurement=True)
            az_e, alt_e = trueRaDec2ApparentAltAz(ra[0], dec[0], pp.JD, pp.lat, pp.lon, refraction=False)
            self.assertLess(abs(elev - alt_e)*3600, 0.05)
            self.assertLess(abs((azim - az_e + 180)%360 - 180)*np.cos(np.radians(alt_e))*3600, 0.05)


if __name__ == "__main__":
    unittest.main()
