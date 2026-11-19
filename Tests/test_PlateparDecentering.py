"""
Tests of the decentering correction of the radial distortion (Platepar.decentering_corr): the Brown-Conrady terms
of a lens element off the optical axis, p1, p2 (amount and direction of the shift, with the square of the radius) and
p3, p4 (the same with its fourth power), stored after the radial coefficients when the correction is on.

The platepar of the tests is a real 1920x1080 calibration (radial7-odd with the asymmetry correction).
"""

import os

import numpy as np

from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP


RADIAL_TYPES = ["radial3-odd", "radial5-odd", "radial7-odd", "radial9-odd", "radial3-all", "radial4-all",
                "radial5-all"]


def makePlatepar():
    """ A real 1920x1080 calibration, radial7-odd with the asymmetry correction and an equal aspect. """

    pp = Platepar()
    pp.lat, pp.lon, pp.elev = 43.19301, -81.315555, 327.0
    pp.X_res, pp.Y_res = 1920, 1080
    pp.F_scale = 109.77153469234013
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = 358.8547132398961, 83.09558620179469, 193.1290062962394
    pp.JD, pp.Ho = 2461317.6327126734, 60.60690352972597
    pp.refraction = False
    pp.equal_aspect, pp.asymmetry_corr, pp.force_distortion_centre = True, True, False
    pp.setDistortionType("radial7-odd", reset_params=True)
    pp.x_poly_fwd = np.array([0.10294621581941152, -0.07894876838186547, -0.49422571182978947, 0.07401737914355552,
                              -0.00276175996668039, 0.0005543572801092884, -0.00012132737766823207])
    pp.x_poly_rev = np.array([0.10295021728524656, -0.07889225008815977, -0.5027730739388016, 0.07302020386100035,
                              -0.0027689973215856718, 0.0005216343597457173, -0.00010247089852210629])
    pp.y_poly_fwd = np.zeros(7)
    pp.y_poly_rev = np.zeros(7)

    return pp


def grid(pp, nx=16, ny=9):
    """ Image positions on a grid over the whole image (inside a margin). """

    xs, ys = np.meshgrid(np.linspace(40, pp.X_res - 40, nx), np.linspace(40, pp.Y_res - 40, ny))
    return xs.ravel(), ys.ravel()


def mapping(pp):
    """ Both directions of the mapping on a grid: sky positions of the image grid, and image positions of them. """

    x, y = grid(pp)
    _, ra, dec, _ = xyToRaDecPP(np.full(x.size, pp.JD), x, y, np.ones(x.size), pp, extinction_correction=False,
                                jd_time=True)
    xi, yi = raDecToXYPP(np.asarray(ra), np.asarray(dec), pp.JD, pp)
    return np.concatenate([ra, dec, xi, yi])


def test_flag_adds_and_removes_the_coefficients():
    """ The correction adds 4 coefficients after the radial ones of every radial type, keeps all others, and
        turning it off gives the original arrays back. """

    for dist_type in RADIAL_TYPES:

        # Distinct coefficients of the type, to see where each one goes
        pp = makePlatepar()
        pp.setDistortionType(dist_type, reset_params=True)
        pp.x_poly_fwd = np.arange(1, pp.poly_length + 1)/100.0
        pp.x_poly_rev = -np.arange(1, pp.poly_length + 1)/100.0
        before_fwd, before_rev, length = pp.x_poly_fwd.copy(), pp.x_poly_rev.copy(), pp.poly_length

        # On: 4 zero coefficients after the others
        pp.remapCoeffsForFlagChange('decentering_corr', True)
        assert pp.poly_length == length + 4
        assert np.array_equal(pp.x_poly_fwd[:length], before_fwd)
        assert np.array_equal(pp.x_poly_fwd[length:], np.zeros(4))
        coeffs = pp.extractRadialCoeffs()
        assert (coeffs['p1'], coeffs['p2'], coeffs['p3'], coeffs['p4']) == (0, 0, 0, 0)

        # Off again: the original arrays
        pp.remapCoeffsForFlagChange('decentering_corr', False)
        assert pp.poly_length == length
        assert np.array_equal(pp.x_poly_fwd, before_fwd) and np.array_equal(pp.x_poly_rev, before_rev)


def test_zero_decentering_changes_nothing():
    """ With the correction on and all its coefficients zero, the mapping is the same as without it. """

    pp = makePlatepar()
    off = mapping(pp)
    pp.remapCoeffsForFlagChange('decentering_corr', True)
    assert np.array_equal(mapping(pp), off)


def test_decentering_displacement():
    """ The displacement of the image by p1 is the Brown-Conrady pattern: at a point on the x axis of the radial
        distortion it is 3*p1*r^2 along x (radial), at a point on the y axis p1*r^2 along x (tangential), in units of
        half the image width; p3 adds the same pattern with r^4. """

    pp = makePlatepar()
    pp.remapCoeffsForFlagChange('decentering_corr', True)

    # Without the radial distortion, the asymmetry and the offset of the centre, the image positions are the
    #   standard coordinates plus the decentering
    pp.x_poly_rev = np.zeros(pp.poly_length)
    _, ra, dec, _ = xyToRaDecPP(np.full(2, pp.JD), np.array([960 + 480, 960.0]), np.array([540.0, 540 - 480]),
                                np.ones(2), pp, extinction_correction=False, jd_time=True)
    x0, y0 = raDecToXYPP(np.asarray(ra), np.asarray(dec), pp.JD, pp)

    p1, p3 = 0.001, 0.5
    pp.x_poly_rev[-4:] = [p1, 0.0, p3, 0.0]
    x1, y1 = raDecToXYPP(np.asarray(ra), np.asarray(dec), pp.JD, pp)

    # The standard coordinates of the two points relative to the centre, in units of half the image width
    u = (x0 - 960)/960.0
    v = (y0 - 540)/960.0
    r2 = u**2 + v**2

    # The image is displaced by minus the distortion offset
    expected_dx = -((p1 + p3*r2)*(r2 + 2*u**2))*960
    expected_dy = -(2*(p1 + p3*r2)*u*v)*960
    assert np.allclose(x1 - x0, expected_dx, atol=1e-6)
    assert np.allclose(y1 - y0, expected_dy, atol=1e-6)


def test_fit_recovers_the_decentering(capsys):
    """ Stars displaced by a decentered lens element: the fit with the correction recovers it and matches the stars
        to a few hundredths of a pixel, the fit without it leaves residuals several times larger. """

    truth = makePlatepar()
    truth.remapCoeffsForFlagChange('decentering_corr', True)
    truth.x_poly_rev[-4:] = [0.0015, -0.001, 0.0005, 0.0]

    # The catalogue: the sky positions of a grid of stars, and their image positions through the decentered lens
    x, y = grid(truth, 24, 14)
    _, ra, dec, _ = xyToRaDecPP(np.full(x.size, truth.JD), x, y, np.ones(x.size), makePlatepar(),
                                extinction_correction=False, jd_time=True)
    ra, dec = np.asarray(ra), np.asarray(dec)
    img_x, img_y = raDecToXYPP(ra, dec, truth.JD, truth)
    img_stars = np.column_stack([img_x, img_y, np.full(x.size, 1000.0)])
    catalog_stars = np.column_stack([ra, dec, np.full(x.size, 5.0)])

    rms = {}
    for decentering in (False, True):
        pp = makePlatepar()
        if decentering:
            pp.remapCoeffsForFlagChange('decentering_corr', True)
        pp.fitAstrometry(truth.JD, img_stars, catalog_stars)
        fx, fy = raDecToXYPP(ra, dec, truth.JD, pp)
        rms[decentering] = np.sqrt(np.mean((fx - img_x)**2 + (fy - img_y)**2))
    capsys.readouterr()

    assert rms[True] < 0.05
    assert rms[False] > 3*rms[True]


def test_saved_and_loaded(tmp_path):
    """ The correction and its coefficients are saved and read back; a platepar saved without it reads with the
        correction off and unchanged arrays. """

    # A platepar without the correction, and the same with it
    pp = makePlatepar()
    plain = os.path.join(str(tmp_path), 'plain.cal')
    pp.write(plain)

    pp.remapCoeffsForFlagChange('decentering_corr', True)
    pp.x_poly_fwd[-4:] = [0.001, -0.002, 0.0005, -0.0003]
    path = os.path.join(str(tmp_path), 'decentering.cal')
    pp.write(path)

    # Read back with its coefficients
    loaded = Platepar()
    loaded.read(path)
    assert loaded.decentering_corr
    assert np.allclose(loaded.x_poly_fwd, pp.x_poly_fwd)

    # A platepar written without the key (as before the correction existed)
    import json
    d = json.load(open(plain))
    d.pop('decentering_corr', None)
    json.dump(d, open(plain, 'w'))
    old = Platepar()
    old.read(plain)
    assert not old.decentering_corr
    assert np.allclose(old.x_poly_fwd, makePlatepar().x_poly_fwd)


def test_type_conversion_keeps_the_decentering():
    """ Converting between radial types (as the staged fit does) keeps the decentering coefficients. """

    pp = makePlatepar()
    pp.remapCoeffsForFlagChange('decentering_corr', True)
    pp.x_poly_fwd[-4:] = [0.001, -0.002, 0.0005, -0.0003]

    # Convert to radial3-odd as the staged fit does
    x_new, _ = pp.convertDistortionCoeffs("radial3-odd")
    pp.setDistortionType("radial3-odd", reset_params=False)
    pp.x_poly_fwd = x_new
    coeffs = pp.extractRadialCoeffs()
    assert np.allclose([coeffs['p1'], coeffs['p2'], coeffs['p3'], coeffs['p4']], [0.001, -0.002, 0.0005, -0.0003])
    assert len(pp.x_poly_fwd) == pp.poly_length


def test_flag_changes_keep_the_radial_coefficients_of_every_type():
    """ Turning the asymmetry correction off and on keeps all radial coefficients of every radial type (the types
        with all powers have more of them than the odd ones of the same order). """

    for dist_type in RADIAL_TYPES:
        pp = makePlatepar()
        pp.setDistortionType(dist_type, reset_params=True)
        pp.x_poly_fwd = np.arange(1, pp.poly_length + 1)/100.0
        before = pp.x_poly_fwd.copy()

        # The asymmetry off and on again: the centre and the radial coefficients stay
        pp.remapCoeffsForFlagChange('asymmetry_corr', False)
        pp.remapCoeffsForFlagChange('asymmetry_corr', True)
        coeffs = pp.extractRadialCoeffs()
        assert np.array_equal(pp.x_poly_fwd[:2], before[:2])
        assert np.array_equal(pp.x_poly_fwd[4:], before[4:]), dist_type
        assert coeffs['k%d' % pp.radialTermCount()] == before[-1]


def test_radial_order_change_keeps_the_decentering():
    """ Changing the radial order with the decentering correction on (as the distortion type selector of SkyFit2
        does) keeps the decentering coefficients and the radial coefficients of the lower order. """

    pp = makePlatepar()
    pp.remapCoeffsForFlagChange('decentering_corr', True)
    pp.x_poly_fwd[-4:] = [0.001, -0.002, 0.0005, -0.0003]
    before = pp.extractRadialCoeffs()

    # Down to radial3-odd: k1 and the decentering coefficients stay, k2 and k3 go
    pp.setDistortionType("radial3-odd", reset_params=False)
    coeffs = pp.extractRadialCoeffs()
    assert len(pp.x_poly_fwd) == pp.poly_length
    assert coeffs['k1'] == before['k1']
    assert [coeffs[k] for k in ('p1', 'p2', 'p3', 'p4')] == [before[k] for k in ('p1', 'p2', 'p3', 'p4')]

    # Back up to radial7-odd: the decentering coefficients stay, the higher radial ones start from zero
    pp.setDistortionType("radial7-odd", reset_params=False)
    coeffs = pp.extractRadialCoeffs()
    assert len(pp.x_poly_fwd) == pp.poly_length
    assert (coeffs['k1'], coeffs['k2'], coeffs['k3']) == (before['k1'], 0.0, 0.0)
    assert [coeffs[k] for k in ('p1', 'p2', 'p3', 'p4')] == [before[k] for k in ('p1', 'p2', 'p3', 'p4')]


def test_old_pickled_platepar():
    """ A platepar pickled before the decentering correction existed (e.g. in a SkyFit2 state file) loads with
        the correction off and keeps working: its coefficients, the flag toggles and a change of the radial order. """

    import pickle

    # A platepar as pickled before the option existed: without the attribute
    pp = makePlatepar()
    del pp.__dict__['decentering_corr']
    old = pickle.loads(pickle.dumps(pp))
    assert old.decentering_corr is False
    assert np.allclose(old.x_poly_fwd, makePlatepar().x_poly_fwd)

    # Usable as any platepar
    assert old.extractRadialCoeffs()['p1'] == 0.0
    old.remapCoeffsForFlagChange('decentering_corr', True)
    assert old.poly_length == makePlatepar().poly_length + 4
    old.setDistortionType("radial5-odd", reset_params=False)
    assert len(old.x_poly_fwd) == old.poly_length
