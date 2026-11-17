""" Astrometry of a faint point source along a predicted path (RMS.Routines.PathAstrometry): the path, the fit of the
    source, the mapping between stacks, and the whole measurement on synthetic frames.

The synthetic frames are made with a real 1280x720 calibration and the real star catalog: the
catalog stars are put on the frames by the platepar at the time of every frame, as Gaussians of a brightness from
their magnitude, with a moving source too faint to be seen in one frame, on a background with noise. The source is
put a known distance off its predicted path, which the measurement has to recover. Run with:
    python -m pytest Tests/test_PathAstrometry.py
"""

from __future__ import print_function, division, absolute_import

import copy
import csv
import datetime
import os

import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.AutoPlatepar import loadCatalogStars
from RMS.Formats.Platepar import Platepar
from RMS.Misc import getRmsRootDir
import RMS.Routines.PathAstrometry as PA


# Size (px) and scale (px/deg) of the image: a field of about 11 x 8 deg
WIDTH, HEIGHT = 640, 480
F_SCALE = 60.0

# Faintest catalog star on the frames
STAR_LIM_MAG = 7.5

# Frames per second, frames per stack, stacks per group and per window of the synthetic recording
FPS = 25.0
STACK = 4
GROUP = 4
WINDOW = 8

# Background, its noise, the peak of a star of magnitude 7 and the brightest peak (inside the 8-bit range of the
#   star extraction), the Gaussian sigma of the stars and of the source (px)
BACKGROUND = 30.0
NOISE = 3.0
PEAK_MAG7 = 40.0
PEAK_MAX = 200.0
PSF_SIGMA = 1.0

# Peak of the source in one frame (a few times the noise of one frame, the source moves across the frames of a
#   stack and is measured in a group of stacks)
SOURCE_PEAK = 15.0

# Offset of the source from its predicted path (px), which the measurement has to recover
OFFSET = (0.4, -0.3)

# Time of the beginning of the synthetic recording: the reference time of the calibration (makePlatepar)
T_START = (2461018.279334375 - 2440587.5)*86400.0



def makeConfig():
    """ The default configuration for the synthetic camera, its star extraction for 8-bit images. """

    config = cr.Config()
    config.width, config.height = WIDTH, HEIGHT
    config.fps = FPS
    config.catalog_mag_limit = STAR_LIM_MAG
    config.min_matched_stars = 15
    config.max_stars = 1000

    return config



def makePlatepar():
    """ A narrow field of a real calibration (the lens of a 1280x720 camera of the tests of the joint distortion
        fit, radial7-odd with the asymmetry correction, at a scale of a narrow-field camera), at the time of the
        synthetic recording. """

    pp = Platepar()
    pp.lat, pp.lon, pp.elev = 50.192509, 5.254718, 295.0
    pp.X_res, pp.Y_res = WIDTH, HEIGHT
    pp.F_scale = F_SCALE
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = 349.8434689573242, -4.063815352866793, 19.34236856738213
    pp.JD, pp.Ho = 2461018.279334375, 358.3338455827907
    pp.refraction = True
    pp.equal_aspect, pp.asymmetry_corr, pp.force_distortion_centre = True, True, False
    pp.setDistortionType("radial7-odd", reset_params=True)
    pp.x_poly_fwd = np.array([-0.0017811686386760266, 0.031678484905771256, -0.007740143231304088,
                              -1.0140484087110697, 0.06835307986186301, 0.010258192855205479,
                              0.005386010898662939])
    pp.x_poly_rev = np.array([-0.0017856520990328477, 0.03170770867400316, 0.007074982440201784,
                              0.4820702423628897, 0.06802381881224333, -0.001903408962282129,
                              0.00014273818898309323])
    pp.y_poly_fwd = np.array([0.5, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    pp.y_poly_rev = np.array([0.5, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    pp.exact_reverse = True
    pp.updateRefAltAz()
    pp.mag_lev = 10.5
    pp.vignetting_coeff = 0.0

    return pp



class FakeFrames(object):
    def __init__(self, pp, catalog, path_fn, n_frames, offset=OFFSET, seed=0):
        """ Synthetic raw frames with the methods of RMS.Routines.TrackShape.RawFrames which the measurement uses
            (framesBetween, frame). The stars follow the platepar, the source follows path_fn with an offset.

        Arguments:
            pp: [Platepar] Platepar of the frames.
            catalog: [ndarray] Catalog stars (ra, dec, mag).
            path_fn: [function] Direction of the source (ra, dec) at a Unix time.
            n_frames: [int] Number of frames.

        Keyword arguments:
            offset: [tuple] Offset of the source from its path (px).
            seed: [int] Seed of the noise.
        """

        self.pp, self.catalog, self.path_fn, self.offset = pp, catalog, path_fn, offset
        self.times = T_START + np.arange(n_frames)/FPS
        self.rng = np.random.default_rng(seed)
        self.height, self.width = int(pp.Y_res), int(pp.X_res)


    def framesBetween(self, t0, t1):
        """ The frames whose times are between t0 and t1 (inclusive). """

        idx = np.where((self.times >= t0) & (self.times <= t1))[0]

        return idx, self.times[idx]


    def frame(self, i):
        """ Frame i: the stars, the source and the noise. """

        t = self.times[i]
        jd = PA.unixToJD(t)
        img = np.full((self.height, self.width), BACKGROUND)

        # The stars of the catalog where the platepar puts them
        x, y = raDecToXYPP(self.catalog[:, 0], self.catalog[:, 1], jd, self.pp)
        peaks = np.minimum(PEAK_MAG7*10**(-0.4*(self.catalog[:, 2] - 7.0)), PEAK_MAX)
        inside = (x > 5) & (x < self.width - 5) & (y > 5) & (y < self.height - 5)
        sources = list(zip(x[inside], y[inside], peaks[inside]))

        # The source, off its path by the offset
        ra, dec = self.path_fn(t)
        xs, ys = raDecToXYPP(np.array([ra]), np.array([dec]), jd, self.pp, measurement=True)
        sources.append((xs[0] + self.offset[0], ys[0] + self.offset[1], SOURCE_PEAK))

        # Every source as a Gaussian in a small box around it
        for xc, yc, peak in sources:
            xi, yi = int(round(xc)), int(round(yc))
            y0, y1, x0, x1 = max(yi - 5, 0), min(yi + 6, self.height), max(xi - 5, 0), min(xi + 6, self.width)
            if (y0 >= y1) or (x0 >= x1):
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            img[y0:y1, x0:x1] += peak*np.exp(-((xx - xc)**2 + (yy - yc)**2)/(2*PSF_SIGMA**2))

        return (img + self.rng.normal(0, NOISE, img.shape)).astype(np.float32)



def movingPath(pp, rate_px=0.6):
    """ A path which moves against the stars across the middle of the image: the sky positions, as seen at the
        start, of a straight line of image positions, moving rate_px pixels per frame.

    Return:
        [function] Direction (ra, dec) at a Unix time.
    """

    jd = PA.unixToJD(T_START)

    def fn(t):
        x = 250.0 + rate_px*(t - T_START)*FPS
        y = 200.0 + 0.5*rate_px*(t - T_START)*FPS
        _, ra, dec, _ = xyToRaDecPP([jd], [x], [y], [1], pp, extinction_correction=False, measurement=True,
                                    jd_time=True)
        return float(ra[0]), float(dec[0])

    return fn



def pathOf(fn, t0, t1, step=0.2):
    """ A Path through the directions of a function between two times. """

    times = np.arange(t0, t1 + step, step)
    radec = np.array([fn(t) for t in times])

    return PA.Path(times, radec[:, 0], radec[:, 1])



@pytest.fixture(scope="module")
def setup():
    """ The configuration, the platepar and the catalog of the synthetic frames. """

    config = makeConfig()
    pp = makePlatepar()
    catalog = loadCatalogStars(config, STAR_LIM_MAG, jd=pp.JD)

    return config, pp, catalog



def test_path_interpolation():
    """ The path goes through its points, between them along the arc, and is NaN outside of them. """

    path = PA.Path(np.array([0.0, 10.0]), np.array([10.0, 12.0]), np.array([20.0, 20.0]))
    ra, dec = path.at(np.array([0.0, 5.0, 10.0, -1.0, 11.0]))

    assert abs(ra[0] - 10.0) < 1e-9 and abs(ra[2] - 12.0) < 1e-9
    # Half way in time is the middle of the great circle arc between the points
    mid = PA.unitVectors([10.0, 12.0], [20.0, 20.0]).sum(axis=0)
    ra_mid, dec_mid = PA.raDecOfVectors(mid)
    assert abs(ra[1] - ra_mid[0]) < 1e-9 and abs(dec[1] - dec_mid[0]) < 1e-9
    assert np.all(np.isnan(ra[3:])) and np.all(np.isnan(dec[3:]))

    # Across RA = 0 the path doesn't go the long way around
    path = PA.Path(np.array([0.0, 10.0]), np.array([359.0, 1.0]), np.array([0.0, 0.0]))
    ra, _ = path.at(np.array([5.0]))
    assert min(abs(ra[0] - 0.0), abs(ra[0] - 360.0)) < 1e-6



def test_read_path(tmp_path):
    """ A path file with ISO times and one with Unix times give the same path. """

    rows = [(T_START, 10.0, 20.0), (T_START + 10, 10.1, 20.05)]
    iso = tmp_path/'iso.csv'
    unix = tmp_path/'unix.csv'
    with open(str(iso), 'w') as f:
        f.write('time,ra,dec\n')
        for t, ra, dec in rows:
            f.write('{:s},{},{}\n'.format(datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(), ra, dec))
    with open(str(unix), 'w') as f:
        f.write('time,ra,dec\n')
        for t, ra, dec in rows:
            f.write('{:.3f},{},{}\n'.format(t, ra, dec))

    a, b = PA.readPath(str(iso)), PA.readPath(str(unix))
    np.testing.assert_allclose(a.times, b.times)
    np.testing.assert_allclose(a.vectors, b.vectors)



def test_fit_source():
    """ A Gaussian source is centred to a small fraction of a pixel; nothing far from the prediction, or within the
        noise, is taken. """

    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:60, 0:60]
    img = 10.0 + 20.0*np.exp(-((xx - 30.3)**2 + (yy - 29.6)**2)/(2*1.0**2)) + rng.normal(0, 1.0, xx.shape)

    fit = PA.fitSource(img, 30.0, 30.0)
    assert fit is not None
    assert abs(fit['x'] - 30.3) < 0.05 and abs(fit['y'] - 29.6) < 0.05
    assert fit['sigma_x'] < 0.05 and fit['snr'] > 10

    # Predicted too far from the source, or no source above the noise
    assert PA.fitSource(img, 36.0, 30.0) is None
    assert PA.fitSource(10.0 + rng.normal(0, 1.0, xx.shape), 30.0, 30.0) is None

    # A box which reaches out of the image or into the unknown
    assert PA.fitSource(img, 2.0, 30.0) is None
    holed = img.copy()
    holed[29, 31] = np.nan
    assert PA.fitSource(holed, 30.0, 30.0) is None



def test_stack_mapping(setup):
    """ The mapping between two stacks puts a star of the first where it is in the second; between a stack and
        itself it is the identity (to the agreement of the two directions of the plate). """

    config, pp, catalog = setup
    t0, t1 = T_START, T_START + 60.0

    rows, cols = PA.stackMapping(pp, t0, pp, t0, WIDTH, HEIGHT)
    for y, x in ((50, 50), (240, 320), (430, 600)):
        assert np.hypot(rows[y, x] - y, cols[y, x] - x) < 0.05

    # Stars over the image, a minute later
    rows, cols = PA.stackMapping(pp, t0, pp, t1, WIDTH, HEIGHT)
    for y, x in ((30, 40), (240, 320), (450, 600)):
        _, ra, dec, _ = xyToRaDecPP([PA.unixToJD(t0)], [float(x)], [float(y)], [1], pp, extinction_correction=False,
                                    jd_time=True)
        x1, y1 = raDecToXYPP(np.array(ra), np.array(dec), PA.unixToJD(t1), pp)
        assert np.hypot(rows[y, x] - y1[0], cols[y, x] - x1[0]) < 0.02



def test_measure_path(setup):
    """ The source is found along its path in the synthetic frames, at its offset from the path, and converted to
        the sky with the recalibrated platepar of the middle stack of its group. """

    config, pp, catalog = setup
    fn = movingPath(pp)
    n_frames = 2*WINDOW*STACK
    frames = FakeFrames(pp, catalog, fn, n_frames)
    path = pathOf(fn, frames.times[0], frames.times[-1])

    # The calibration starts from a platepar pointing slightly off, which the recalibration corrects
    start = copy.deepcopy(pp)
    start.RA_d += 0.05
    start.dec_d -= 0.03

    measurements, info = PA.measurePath(frames, path, start, config, catalog_stars=catalog, stack_frames=STACK,
                                        window_stacks=WINDOW, group_stacks=GROUP)

    assert info['stacks'] == n_frames//STACK
    assert info['recalibrated'] == info['stacks']
    assert info['windows'] == 2 and info['windows_static'] == 0
    assert len(measurements) == 4

    for meas in measurements:

        # The offset of the source from its path is recovered in the image
        assert abs((meas['x'] - meas['x_pred']) - OFFSET[0]) < 0.15
        assert abs((meas['y'] - meas['y_pred']) - OFFSET[1]) < 0.15

        # On the sky: the true position of the source (path + offset through the true platepar) at the time
        jd = PA.unixToJD(meas['time_unix'])
        ra_p, dec_p = fn(meas['time_unix'])
        xt, yt = raDecToXYPP(np.array([ra_p]), np.array([dec_p]), jd, pp, measurement=True)
        _, ra_t, dec_t, _ = xyToRaDecPP([jd], [xt[0] + OFFSET[0]], [yt[0] + OFFSET[1]], [1], pp,
                                        extinction_correction=False, measurement=True, jd_time=True)
        sep = np.degrees(np.arccos(np.clip(np.dot(PA.unitVectors(meas['ra'], meas['dec'])[0],
                                                  PA.unitVectors(ra_t, dec_t)[0]), -1, 1)))*3600
        scale = 3600.0/pp.F_scale
        assert sep < 0.2*scale

        # A stated uncertainty of a small part of a pixel, the source well above the noise
        assert 0 < meas['sigma_x'] < 0.2 and meas['snr'] > 5
        assert meas['n_stacks'] == GROUP and meas['n_frames'] == GROUP*STACK



def test_static_source_not_measured(setup):
    """ A source which doesn't move against the stars would be kept by the median of the stars: its windows are
        counted and not measured. """

    config, pp, catalog = setup
    jd0 = PA.unixToJD(T_START)

    # A source fixed on the sky, like a star
    _, ra, dec, _ = xyToRaDecPP([jd0], [250.0], [200.0], [1], pp, extinction_correction=False, measurement=True,
                                jd_time=True)
    fn = lambda t: (float(ra[0]), float(dec[0]))
    frames = FakeFrames(pp, catalog, fn, WINDOW*STACK)
    path = pathOf(fn, frames.times[0], frames.times[-1])

    measurements, info = PA.measurePath(frames, path, pp, config, catalog_stars=catalog, stack_frames=STACK,
                                        window_stacks=WINDOW, group_stacks=GROUP)

    assert measurements == []
    assert info['windows'] == 1 and info['windows_static'] == 1



def test_command_line_output(setup, tmp_path, monkeypatch):
    """ The command line writes the measurements with the columns of CSV_COLUMNS. """

    config, pp, catalog = setup
    fn = movingPath(pp)
    frames = FakeFrames(pp, catalog, fn, WINDOW*STACK)

    # The path file, and the raw source and the catalog given by the synthetic frames
    path_file = tmp_path/'path.csv'
    with open(str(path_file), 'w') as f:
        f.write('time,ra,dec\n')
        for t in np.arange(frames.times[0], frames.times[-1] + 0.2, 0.2):
            ra, dec = fn(t)
            f.write('{:.4f},{:.8f},{:.8f}\n'.format(t, ra, dec))
    config_file = tmp_path/'test.config'
    with open(os.path.join(getRmsRootDir(), '.config')) as f:
        text = f.read()
    config_file.write_text(text)
    pp_file = tmp_path/'platepar.cal'
    pp.write(str(pp_file))

    class Closing(FakeFrames):
        def close(self):
            """ The synthetic frames have no files to close. """

    monkeypatch.setattr('RMS.Routines.TrackShape.openRawSource',
                        lambda *args, **kwargs: Closing(pp, catalog, fn, WINDOW*STACK))
    monkeypatch.setattr(cr, 'parse', lambda path: config)
    monkeypatch.setattr('RMS.Astrometry.AutoPlatepar.loadCatalogStars', lambda *args, **kwargs: catalog)

    out = tmp_path/'out.csv'
    assert PA.main([str(tmp_path), str(path_file), '-c', str(config_file), '-p', str(pp_file), '-o', str(out),
                    '--stack', str(STACK), '--group', str(GROUP), '--window', str(WINDOW)]) == 0

    with open(str(out)) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == PA.CSV_COLUMNS
    assert len(rows) == 2
