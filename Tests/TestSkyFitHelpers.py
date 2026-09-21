""" Tests for the pure (non-Qt) helpers behind the SkyFit2 WASD panning and pointing indicator, in
    RMS.Astrometry.ApplyAstrometry. """

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest

from RMS.Astrometry.ApplyAstrometry import screenNudgeToAzAltDelta, fovCentreZenithDirection
from RMS.Astrometry.Conversions import JD2HourAngle
from RMS.Formats.Platepar import Platepar
from RMS.Math import angularSeparationDeg


def makePlatepar(az_centre, alt_centre, pos_angle_ref=0.0, f_scale=14.5):
    """ Platepar pointed at the given apparent az/alt with a consistent reference RA/Dec.

    Arguments:
        az_centre: [float] Apparent azimuth of the pointing (deg).
        alt_centre: [float] Apparent altitude of the pointing (deg).

    Keyword arguments:
        pos_angle_ref: [float] Position angle (celestial roll) of the plate (deg).
        f_scale: [float] Plate scale (px/deg).

    Return:
        pp: [Platepar]
    """

    pp = Platepar()
    pp.lat, pp.lon = 45.0, 15.0
    pp.JD = 2460000.5
    pp.Ho = JD2HourAngle(pp.JD)
    pp.F_scale = f_scale
    pp.pos_angle_ref = pos_angle_ref
    pp.az_centre, pp.alt_centre = az_centre, alt_centre
    pp.updateRefRADec(skip_rot_update=True)

    return pp


@pytest.mark.parametrize("az_centre, alt_centre, pos_angle_ref", [
    (120.0, 45.0, 0.0),
    (120.0, 45.0, 137.0),
    (0.5, 30.0, 30.0),
    (270.0, 80.0, 200.0),
])
@pytest.mark.parametrize("screen_dx, screen_dy", [(1, 0), (-1, 0), (0, 1), (0, -1)])
def testNudgeMovesExactlyOneStep(az_centre, alt_centre, pos_angle_ref, screen_dx, screen_dy):
    """ Every WASD press advances the pointing by exactly the requested angle, whatever the roll. """

    pp = makePlatepar(az_centre, alt_centre, pos_angle_ref)
    step = 1.5

    d_az, d_alt = screenNudgeToAzAltDelta(pp, screen_dx, screen_dy, step)

    moved = angularSeparationDeg(pp.az_centre, pp.alt_centre, pp.az_centre + d_az, pp.alt_centre + d_alt)

    assert moved == pytest.approx(step, abs=1e-4)


def testOppositeKeysAreAntisymmetric():
    """ Left/right and up/down produce opposite pointing changes (to first order in the step). """

    pp = makePlatepar(200.0, 40.0, 75.0)
    step = 0.2

    right = screenNudgeToAzAltDelta(pp, 1, 0, step)
    left = screenNudgeToAzAltDelta(pp, -1, 0, step)
    up = screenNudgeToAzAltDelta(pp, 0, 1, step)
    down = screenNudgeToAzAltDelta(pp, 0, -1, step)

    assert right[0] == pytest.approx(-left[0], abs=1e-3)
    assert right[1] == pytest.approx(-left[1], abs=1e-3)
    assert up[0] == pytest.approx(-down[0], abs=1e-3)
    assert up[1] == pytest.approx(-down[1], abs=1e-3)


def testAzimuthDeltaIsWrappedAcrossNorth():
    """ Nudging a pointing just below 360 deg azimuth eastward gives a small positive delta, not -359. """

    pp = makePlatepar(359.8, 30.0, 0.0)

    # Pick the screen direction that increases the azimuth from the zenith/east geometry
    _, east_screen, _, _, valid = fovCentreZenithDirection(pp)
    assert valid
    ea = np.radians(east_screen)

    d_az, _ = screenNudgeToAzAltDelta(pp, np.cos(ea), np.sin(ea), 1.0)

    assert 0.0 < d_az < 2.0
    assert (pp.az_centre + d_az + 360)%360 < 2.0


def testNudgeTowardZenithRaisesAltitude():
    """ The zenith arrow of the pointing indicator and the pan agree: nudging along the arrow direction
        raises the pointing by (almost) the full step, nudging along the East direction increases the
        azimuth. """

    for pos_angle_ref in (0.0, 45.0, 190.0):
        pp = makePlatepar(150.0, 35.0, pos_angle_ref)

        # The readout is evaluated at the distortion centre pixel, which sits a fraction of a degree
        #   from the reference pointing in the RMS projection model, hence the loose tolerance
        angle_screen, east_screen, azimuth, elevation, valid = fovCentreZenithDirection(pp,
            centre=pp.getDistortionCentre())
        assert valid
        assert azimuth == pytest.approx(150.0, abs=0.5)
        assert elevation == pytest.approx(35.0, abs=0.5)

        # Along the arrow: altitude goes up by about the full step
        za = np.radians(angle_screen)
        d_az, d_alt = screenNudgeToAzAltDelta(pp, np.cos(za), np.sin(za), 1.0)
        assert d_alt == pytest.approx(1.0, abs=0.05)
        assert abs(d_az) < 0.2

        # Along the East direction of the horizon bar: azimuth goes up, altitude barely changes
        ea = np.radians(east_screen)
        d_az, d_alt = screenNudgeToAzAltDelta(pp, np.cos(ea), np.sin(ea), 1.0)
        assert d_az > 0.5
        assert abs(d_alt) < 0.2


def testCrossingTheZenithFlipsAzimuth():
    """ A step larger than the distance to the zenith carries the pointing over the top: the altitude
        comes back down and the azimuth flips by about 180 deg. """

    pp = makePlatepar(90.0, 89.5, 0.0)

    # So close to the zenith the finite-difference zenith direction of the indicator is unreliable (the
    #   indicator draws a dashed ring there), so find the screen direction that raises the altitude the
    #   most with a tiny step instead
    angles = np.radians(np.arange(0, 360, 5.0))
    d_alts = [screenNudgeToAzAltDelta(pp, np.cos(a), np.sin(a), 0.01)[1] for a in angles]
    za = angles[int(np.argmax(d_alts))]

    # Push over the zenith with a step twice the distance to it
    d_az, d_alt = screenNudgeToAzAltDelta(pp, np.cos(za), np.sin(za), 1.0)

    new_alt = pp.alt_centre + d_alt
    new_az = (pp.az_centre + d_az)%360

    assert new_alt < 90.0
    assert new_alt == pytest.approx(89.5, abs=0.05)
    assert abs(((new_az - 270.0) + 180)%360 - 180) < 15.0


def testDegenerateNudgeIsNoop():
    """ A zero screen direction is a no-op. """

    pp = makePlatepar(120.0, 45.0, 0.0)
    assert screenNudgeToAzAltDelta(pp, 0, 0, 1.0) == (0.0, 0.0)
