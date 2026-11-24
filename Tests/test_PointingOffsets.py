"""
Tests of the offsets of the pointing used by the fits of the pointing (Platepar.PointingOffsets).

The fits turn the camera in its own axes about a reference pointing: the optical axis is moved along the image x
and y directions and the camera is rolled about it, with the offsets in pixels. The tests check that zero offsets
give the reference back, that the camera the projection builds from the new pointing is the reference camera turned
by exactly these angles (so a star on the optical axis moves by the offsets on the image, the same at the equator,
at the poles and at the zenith, with and without the refraction), that the offsets of a pointing are the inverse of
the pointing of the offsets, and that turns compose as rotations.
"""

import copy

import numpy as np
import pytest

from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.Astrometry.Conversions import JD2HourAngle
from RMS.Astrometry.CyFunctions import eqRefractionApparentToTrue, pointingCorrection, refractionScale
from RMS.Formats.Platepar import (Platepar, PointingOffsets, frameToPointing, offsetRotation, pointingFrame,
    rotationOffsets)
from RMS.Math import angularSeparationDeg


# Julian date of the reference time of the platepars (2023-02-28 ~18 UT)
JD_REF = 2460004.25

# Pointings of the camera: (name, latitude, right ascension, declination) (deg), with the right ascension
#   relative to the local sidereal time of the reference time (0 is on the meridian). The zenith and the low
#   pointings are given by the elevation instead (see makePlatepar)
POINTINGS = [
    ("equator", 45.0, -20.0, 0.0),
    ("dec 60", 45.0, 30.0, 60.0),
    ("dec 89.999", 45.0, 70.0, 89.999),
    ("north pole", 45.0, 0.0, 90.0),
    ("south pole", -45.0, 0.0, -90.0),
    ("dec -89.999", -45.0, 200.0, -89.999),
    ("zenith", 45.0, None, 90.0),
    ("elevation 5", 45.0, None, 5.0),
]


def makePlatepar(lat, ra_lst, dec, pos_angle=37.0, refraction=True):
    """ A 1280x720 platepar with a 60 deg field and no distortion, pointing to the given direction.

    Arguments:
        lat: [float] Latitude of the station (deg).
        ra_lst: [float] Right ascension of the pointing from the local sidereal time of the reference time (deg),
            or None to point at the elevation dec to the south (90 is the zenith).
        dec: [float] Declination of the pointing (deg), or the elevation if ra_lst is None.

    Keyword arguments:
        pos_angle: [float] Position angle (deg). 37 by default.
        refraction: [bool] Correct the refraction. True by default.

    Return:
        [Platepar] The platepar.
    """

    pp = Platepar()
    pp.lat, pp.lon, pp.elev = lat, 15.0, 100.0
    pp.X_res, pp.Y_res = 1280, 720
    pp.F_scale = pp.X_res/60.0
    pp.JD = JD_REF
    pp.Ho = JD2HourAngle(pp.JD)
    pp.refraction = refraction
    pp.setDistortionType("poly3+radial", reset_params=True)

    # The pointing from the right ascension and declination, or from the elevation to the south
    if ra_lst is not None:
        pp.RA_d, pp.dec_d = (pp.Ho + pp.lon + ra_lst)%360, dec
    else:
        pp.az_centre, pp.alt_centre = 180.0, dec
        pp.updateRefRADec(skip_rot_update=True)
    pp.pos_angle_ref = pos_angle
    pp.updateRefAltAz()

    return pp


def kernelFrame(pp, jd=JD_REF):
    """ The axes of the camera as the projection builds them at the given time (J2000).

    Arguments:
        pp: [Platepar] Platepar.

    Keyword arguments:
        jd: [float] Julian date. JD_REF by default.

    Return:
        [ndarray] The axes (pointingFrame of the corrected pointing).
    """

    ra, dec, pos_angle = pointingCorrection(jd, np.radians(pp.lat), np.radians(pp.lon), np.radians(pp.Ho), pp.JD,
        np.radians(pp.RA_d), np.radians(pp.dec_d), np.radians(pp.pos_angle_ref), refraction=pp.refraction,
        refraction_scale=refractionScale(pp.elev))

    return pointingFrame(np.degrees(ra), np.degrees(dec), np.degrees(pos_angle))


def frameTolerance(pp):
    """ How well the projected camera of a pointing is defined by the numbers of the pointing (rad). With the
        refraction, the position angle is measured from the meridian at the refracted pointing, which turns by
        the move of the pointing divided by its distance from the pole: near a pole the rounding of the pointing
        (1e-16 rad, more after the refraction) turns the camera by more than that (1e-11 rad at 0.001 deg from the
        pole). Without the refraction, and at a pole itself, the position angle follows the right ascension.

    Arguments:
        pp: [Platepar] Platepar.

    Return:
        [float] The tolerance (rad).
    """

    # Distance of the refracted pointing from the pole (rad)
    axis = kernelFrame(pp)[:, 2]
    pole_distance = np.hypot(axis[0], axis[1])

    if (not pp.refraction) or (pole_distance == 0):
        return 1e-12

    return 1e-12 + 1e-14/pole_distance


def turned(pp, offsets):
    """ A copy of the platepar turned by the given offsets.

    Arguments:
        pp: [Platepar] Platepar (not changed).
        offsets: [list] Offsets (px).

    Return:
        [Platepar] The turned copy.
    """

    pp_turned = copy.deepcopy(pp)
    PointingOffsets(pp).apply(pp_turned, offsets)

    return pp_turned


def axisStar(pp):
    """ The catalog direction of a star on the optical axis of the camera at its reference time, without the
        aberration.

    Arguments:
        pp: [Platepar] Platepar with the aberration off.

    Return:
        (ra, dec): [tuple of ndarrays] Right ascension and declination of the star (deg, J2000, one element).
    """

    # The optical axis as the projection sees it (refracted), taken back through the refraction of the stars
    axis = kernelFrame(pp)[:, 2]
    ra, dec = np.arctan2(axis[1], axis[0]), np.arctan2(axis[2], np.hypot(axis[0], axis[1]))
    if pp.refraction:
        ra, dec = eqRefractionApparentToTrue(ra, dec, pp.JD, np.radians(pp.lat), np.radians(pp.lon),
            refractionScale(pp.elev))

    return np.array([np.degrees(ra)]), np.array([np.degrees(dec)])


@pytest.fixture(params=[pytest.param(pointing, id=pointing[0]) for pointing in POINTINGS])
def pointing(request):
    """ The name, latitude, right ascension from the sidereal time and declination of a test pointing. """

    return request.param


### Zero offsets ###

@pytest.mark.parametrize("refraction", [True, False])
def test_zero_offsets_give_the_reference(pointing, refraction):

    # The pointing without offsets is the reference: the same direction to 1e-12 deg, the same projected camera
    #   (to 1e-12 rad, or as well as the numbers define it, see frameTolerance), and the same numbers where the
    #   right ascension and the position angle are defined well (away from a pole)
    _, lat, ra_lst, dec = pointing
    pp = makePlatepar(lat, ra_lst, dec, refraction=refraction)

    ra_d, dec_d, pos_angle_ref = PointingOffsets(pp).toPointing([0.0, 0.0, 0.0])

    assert angularSeparationDeg(ra_d, dec_d, pp.RA_d, pp.dec_d) < 1e-12
    assert np.max(np.abs(kernelFrame(turned(pp, [0.0, 0.0, 0.0])) - kernelFrame(pp))) < frameTolerance(pp)

    if abs(pp.dec_d) < 89.0:
        assert abs((ra_d - pp.RA_d + 180)%360 - 180) < 1e-12
        assert abs(dec_d - pp.dec_d) < 1e-12
        assert abs((pos_angle_ref - pp.pos_angle_ref + 180)%360 - 180) < 1e-12


def test_pointing_frame_round_trip():

    # frameToPointing is the inverse of pointingFrame, the camera (not the numbers) at the poles
    rng = np.random.default_rng(3)
    for ra, dec, pos_angle in zip(rng.uniform(0, 360, 300), rng.uniform(-90, 90, 300), rng.uniform(0, 360, 300)):
        frame = pointingFrame(ra, dec, pos_angle)
        out = frameToPointing(frame[:, 0], frame[:, 2])
        assert np.max(np.abs(pointingFrame(*out) - frame)) < 1e-14
        assert abs(out[1] - dec) < 1e-12

    # At a pole the right ascension of the axes comes out as 0 and the position angle carries the field
    for dec in [90.0, -90.0]:
        frame = pointingFrame(123.0, dec, 45.0)
        assert np.max(np.abs(pointingFrame(*frameToPointing(frame[:, 0], frame[:, 2])) - frame)) < 1e-14


### The projected camera is turned by the offsets ###

@pytest.mark.parametrize("refraction", [True, False])
@pytest.mark.parametrize("offsets", [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [3.0, -4.0, 2.0],
    [-500.0, 300.0, -100.0]])
def test_projected_camera_is_turned_by_the_offsets(pointing, refraction, offsets):

    # The axes the projection builds for the turned pointing are the reference axes turned by the rotation of
    #   the offsets in the axes of the camera, at the reference time and a day later
    _, lat, ra_lst, dec = pointing
    pp = makePlatepar(lat, ra_lst, dec, refraction=refraction)
    pointing_offsets = PointingOffsets(pp)
    rotation = offsetRotation(offsets[0]*pointing_offsets.shift_unit, offsets[1]*pointing_offsets.shift_unit,
                              offsets[2]*pointing_offsets.roll_unit)

    for jd in [JD_REF, JD_REF + 1.3]:
        frame_turned = kernelFrame(turned(pp, offsets), jd=jd)
        assert np.max(np.abs(kernelFrame(pp, jd=jd).T.dot(frame_turned) - rotation)) \
            < frameTolerance(pp) + frameTolerance(turned(pp, offsets))


@pytest.mark.parametrize("refraction", [True, False])
def test_star_on_the_axis_moves_by_the_offsets(pointing, refraction):

    # A pixel of offset moves a star on the optical axis by a pixel on the image, along x for dx and along y for
    #   dy, at every pointing (the poles and the zenith included), and the roll turns the image about the axis
    #   so that a star at a quarter of the image diagonal from it moves by the roll
    _, lat, ra_lst, dec = pointing
    pp = makePlatepar(lat, ra_lst, dec, refraction=refraction)
    pp.aberration = False

    # The star on the optical axis
    star_ra, star_dec = axisStar(pp)

    def _image(offsets):
        """ Image of the axis star with the camera turned by the offsets. """

        x, y = raDecToXYPP(star_ra, star_dec, pp.JD, turned(pp, offsets))
        return np.array([x[0], y[0]])

    # The centre of the projection, from images of the star off the centre: the projection measures the
    #   distance from the centre with an arc cosine, which within 1e-6 rad of the centre is rounded by up to
    #   1e-5 px
    centre = (_image([1.0, 0.0, 0.0]) + _image([-1.0, 0.0, 0.0]))/2

    # The shifts move the star the other way by the offsets
    for dx, dy in [(1.0, 0.0), (0.0, 1.0), (-3.0, 4.0), (30.0, -20.0)]:
        assert np.max(np.abs(_image([dx, dy, 0.0]) - (centre - [dx, dy]))) < 1e-6

    # The roll keeps it in place, to the rounding at the centre
    assert np.max(np.abs(_image([0.0, 0.0, 2.0]) - centre)) < 1e-4

    # The star at a quarter of the diagonal along +x, found as the axis star of the camera turned by that much
    #   (its image is then that far from the centre with the reference camera)
    radius = np.hypot(pp.X_res, pp.Y_res)/4
    edge_ra, edge_dec = axisStar(turned(pp, [radius, 0.0, 0.0]))
    x_edge, y_edge = raDecToXYPP(edge_ra, edge_dec, pp.JD, pp)
    assert np.max(np.abs([x_edge[0] - (centre[0] + radius), y_edge[0] - centre[1]])) < 1e-6

    # Rolling the camera by droll turns the image the other way about the axis, by droll/radius
    droll = 2.0
    x, y = raDecToXYPP(edge_ra, edge_dec, pp.JD, turned(pp, [0.0, 0.0, droll]))
    angle = -droll/radius
    assert abs(x[0] - (centre[0] + radius*np.cos(angle))) < 1e-6
    assert abs(y[0] - (centre[1] + radius*np.sin(angle))) < 1e-6


def test_one_offset_is_the_same_angle_everywhere():

    # The angle on the sky between the reference and the turned optical axis is hypot(dx, dy)/F_scale (deg),
    #   whatever the pointing (a step of the right ascension is cos(dec) of it on the sky, and nothing at a pole)
    for _, lat, ra_lst, dec in POINTINGS:
        for refraction in [True, False]:
            pp = makePlatepar(lat, ra_lst, dec, refraction=refraction)
            axis = kernelFrame(pp)[:, 2]
            axis_turned = kernelFrame(turned(pp, [3.0, 4.0, 0.0]))[:, 2]
            angle = np.degrees(np.arccos(np.clip(np.dot(axis, axis_turned), -1, 1)))
            assert angle == pytest.approx(5.0/pp.F_scale, rel=1e-6)


### Inverse, composition, round trips ###

@pytest.mark.parametrize("refraction", [True, False])
def test_offsets_of_a_pointing_are_the_inverse(pointing, refraction):

    # fromPointing gives back the offsets of toPointing, and toPointing the pointing of fromPointing
    _, lat, ra_lst, dec = pointing
    pp = makePlatepar(lat, ra_lst, dec, refraction=refraction)
    pointing_offsets = PointingOffsets(pp)

    for offsets in [[0.0, 0.0, 0.0], [1e-6, -1e-6, 1e-6], [2.5, -7.0, 1.5], [800.0, -400.0, 300.0]]:
        out = pointing_offsets.fromPointing(*pointing_offsets.toPointing(offsets))
        assert np.max(np.abs(out - offsets)) < 1e-8


def test_turns_compose_as_rotations():

    # Turning by q1 and then by q2 about the turned pointing is turning by the product of their rotations, and
    #   turning back by the inverse rotation gives the reference again
    pp = makePlatepar(45.0, 70.0, 89.999)
    pointing_offsets = PointingOffsets(pp)
    unit = np.array([pointing_offsets.shift_unit]*2 + [pointing_offsets.roll_unit])

    q1, q2 = np.array([40.0, -25.0, 10.0]), np.array([-15.0, 30.0, -5.0])
    pp_1 = turned(pp, q1)
    pp_12 = turned(pp_1, q2)

    # The product of the rotations, as offsets about the reference
    rotation = offsetRotation(*(q1*unit)).dot(offsetRotation(*(q2*unit)))
    q12 = np.array(rotationOffsets(rotation))/unit
    assert np.max(np.abs(kernelFrame(turned(pp, q12)) - kernelFrame(pp_12))) < 3*frameTolerance(pp_12)

    # The inverse rotation from the turned pointing back to the reference
    q_back = np.array(rotationOffsets(offsetRotation(*(q1*unit)).T))/unit
    assert np.max(np.abs(kernelFrame(turned(pp_1, q_back)) - kernelFrame(pp))) < 3*frameTolerance(pp)


def test_random_round_trips():

    # 10000 random references (anywhere on the sky, a third of them within a degree of a pole, with and without
    #   the refraction) and offsets up to 2000 px: the offsets of the turned pointing are the offsets given, and
    #   the projected camera is the reference turned by them
    rng = np.random.default_rng(7)
    n_cases = 10000
    for i in range(n_cases):

        lat = rng.uniform(-70, 70)
        if i%3 == 0:
            dec = np.sign(lat)*rng.uniform(89.0, 90.0)
        else:
            dec = rng.uniform(lat - 80 if lat > 0 else -90, lat + 80 if lat < 0 else 90)
        pp = makePlatepar(lat, rng.uniform(-60, 60), dec, pos_angle=rng.uniform(0, 360),
                          refraction=bool(i%2))
        pointing_offsets = PointingOffsets(pp)

        # Turns of the optical axis up to 133 deg and rolls up to 2.7 rad (the roll comes back in (-pi, pi])
        offsets = np.append(rng.uniform(-2000, 2000, 2), rng.uniform(-1000, 1000))*rng.choice([1e-6, 1e-2, 1.0])

        ra_d, dec_d, pos_angle_ref = pointing_offsets.toPointing(offsets)
        assert (0.0 <= ra_d < 360.0) and (-90.0 <= dec_d <= 90.0) and (0.0 <= pos_angle_ref < 360.0)
        assert np.max(np.abs(pointing_offsets.fromPointing(ra_d, dec_d, pos_angle_ref) - offsets)) < 1e-7

        # The projected camera, checked on a part of the cases (it is slower)
        if i%20 == 0:
            unit = np.array([pointing_offsets.shift_unit]*2 + [pointing_offsets.roll_unit])
            rotation = kernelFrame(pp).T.dot(kernelFrame(turned(pp, offsets)))
            assert np.max(np.abs(rotation - offsetRotation(*(offsets*unit)))) \
                < 10*(frameTolerance(pp) + frameTolerance(turned(pp, offsets)))


### Across a pole and across RA 0 ###

@pytest.mark.parametrize("refraction", [True, False])
@pytest.mark.parametrize("lat, dec", [(45.0, 89.99), (-45.0, -89.99)])
def test_stepping_over_the_pole_moves_the_stars_evenly(refraction, lat, dec):

    # Stepping the optical axis over the pole in even steps of the offset moves the stars by even steps: the
    #   offsets have no special point there, with the refraction too (where the position angle of the
    #   projection, measured from the meridian at the refracted pointing, turns fast)
    pp = makePlatepar(lat, 0.0, dec, refraction=refraction)
    pole_px = 0.01*pp.F_scale
    star_ra, star_dec = np.array([30.0, 120.0, 210.0, 300.0]), np.sign(dec)*np.array([88.0, 88.5, 88.2, 89.5])

    # The offsets which put the optical axis on the pole, and steps along y through it (one of them 1e-4 deg from
    #   the pole)
    pole_dx, pole_dy, _ = PointingOffsets(pp).fromPointing(0.0, np.sign(dec)*90.0, 0.0)

    positions = []
    for dy in pole_dy + np.linspace(-3*pole_px, 3*pole_px, 13):
        x, y = raDecToXYPP(star_ra, star_dec, pp.JD, turned(pp, [pole_dx + 0.01*pole_px, dy, 0.0]))
        positions.append(np.column_stack([x, y]))

    steps = np.array([np.max(np.hypot(*(positions[i + 1] - positions[i]).T)) for i in range(len(positions) - 1)])
    assert np.max(steps) < 1.01*np.min(steps)
    assert np.max(steps) == pytest.approx(pole_px/2, rel=0.02)


def test_right_ascension_wraps():

    # A pointing moved across right ascension 0 stays in [0, 360) and moves on the sky by the offset (with the
    #   position angle 0 the +x direction points west)
    pp = makePlatepar(45.0, 0.0, 10.0, refraction=False)
    pp.RA_d, pp.pos_angle_ref = 0.001, 0.0
    pp.updateRefAltAz()
    pointing_offsets = PointingOffsets(pp)

    for dx in [-0.1*pp.F_scale, 0.1*pp.F_scale]:
        ra_d, dec_d, _ = pointing_offsets.toPointing([dx, 0.0, 0.0])
        assert 0.0 <= ra_d < 360.0
        assert angularSeparationDeg(ra_d, dec_d, pp.RA_d, pp.dec_d) == pytest.approx(0.1, rel=1e-3)


### Scale ###

def test_scale_offset():

    # The scale offset changes the scale so that the stars at a quarter of the image diagonal from the centre
    #   move by it, and fromScale is its inverse
    pp = makePlatepar(45.0, 0.0, 30.0)
    pointing_offsets = PointingOffsets(pp)
    radius = np.hypot(pp.X_res, pp.Y_res)/4

    assert pointing_offsets.toScale(0.0) == pp.F_scale
    assert pointing_offsets.toScale(radius) == pytest.approx(2*pp.F_scale)
    assert pointing_offsets.fromScale(pointing_offsets.toScale(3.7)) == pytest.approx(3.7)

    pp_scaled = turned(pp, [0.0, 0.0, 0.0, 1.0])
    assert pp_scaled.F_scale == pytest.approx(pp.F_scale*(1 + 1/radius))
