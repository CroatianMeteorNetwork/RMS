"""
Tests of the settings tuned on a reference camera scaled to the input (RMS.Routines.ReferenceScale): the frame counts
keep their durations at another frame rate and the pixel sizes their angular sizes at another pixel scale, the
platepar of frames of another size is rescaled (Platepar.rescaled), and the clear-sky reference and the state of the
automatic calibration are not mixed between frame sizes. Without a reference everything is as before.

The case of the tests is a 1080p camera (16.4 arcsec per sensor pixel) which recorded 1920x1080 frames (binned 2x2)
at 10 fps, and then 960x540 frames (binned 4x4) at 25 fps.
"""

import datetime
import json
import os

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.MatchedFilterDetection as mfd
from RMS.Astrometry import AutoCalibration
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Formats.Platepar import Platepar
from RMS.Routines import ReferenceScale as rs
from RMS.Routines.SkyQuality import SkyReference


# The reference camera: 10 fps, 32.8 arcsec/px (1920x1080), and the input: 25 fps, 65.6 arcsec/px (960x540)
REF_FPS, REF_SCALE = 10.0, 32.8
INPUT_FPS, INPUT_SCALE = 25.0, 65.6


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
    pp.vignetting_coeff = 0.00043697301249064584

    return pp


def referenceConfig():
    """ A config with the settings of the reference camera (the defaults, with a few values of the camera) and the
        reference given. """

    config = cr.Config()
    config.mf_min_frames = 10
    config.mf_reference_fps = REF_FPS
    config.mf_reference_scale = REF_SCALE
    config.border, config.neighborhood_size, config.segment_radius, config.detection_border = 15, 20, 10, 5

    return config


def test_no_reference_changes_nothing():
    """ Without a reference the config is returned as it is, and the constants of the matched filter are its own. """

    config = cr.Config()
    assert (config.mf_reference_fps, config.mf_reference_scale) == (0.0, 0.0)
    assert rs.scaleToReference(config, INPUT_FPS, INPUT_SCALE, size=(960, 540)) is config

    # The options of the matched filter use the constants of the module unchanged
    opts = mfd.MatchedFilterOptions(config, fps=INPUT_FPS, size=(540, 960))
    assert opts.faint_min_span == mfd.FAINT_MIN_SPAN
    assert opts.faint_seed_spread == mfd.FAINT_SEED_SPREAD
    assert opts.faint_max_measure_frames == mfd.FAINT_MAX_MEASURE_FRAMES
    assert opts.extent_step == mfd.EXTENT_STEP
    assert (opts.slow_speed, opts.slow_max_speed, opts.slow_min_relative_speed) == \
        (mfd.SLOW_SPEED, mfd.SLOW_MAX_SPEED, mfd.SLOW_MIN_RELATIVE_SPEED)
    assert (opts.shadow_radius, opts.spill_radius) == (mfd.SHADOW_RADIUS, mfd.SPILL_RADIUS)
    assert opts.min_star_template_px == mfd.MIN_STAR_TEMPLATE_PX

    # The chunks of the star extraction and the recalibration thresholds stay too
    assert rs.scaleChunkFrames(rs.DEFAULT_CHUNK_FRAMES, config, INPUT_FPS) == rs.DEFAULT_CHUNK_FRAMES
    assert rs.recalibrationThresholds(config, makePlatepar()) == (config.dist_check_threshold,
                                                                   config.dist_check_quick_threshold)


def test_reference_scales_the_settings():
    """ The settings of the reference camera scaled to 25 fps and 65.6 arcsec/px (with the frame rate measured
        from the frame times, a little off 25): the values tuned by hand for this case. """

    config = referenceConfig()
    scaled = rs.scaleToReference(config, 25.0029, 65.5, size=(960, 540))

    # The frame counts keep their durations
    assert scaled.mf_run_frames == [20, 40]
    assert scaled.mf_slow_run_frames == 80
    assert scaled.mf_block_frames == 640
    assert scaled.mf_min_frames == 25
    assert scaled.mf_link_max_gap == 240
    assert scaled.mf_segment_overlap == 1280

    # The frames of a measurement double, so their largest number is the power of two at least as long as at the
    #   reference (8 frames at 10 fps -> 20 -> 32)
    assert scaled.mf_max_measure_frames == 32

    # The pixel sizes keep their angular sizes
    assert scaled.mf_edge_margin == 4
    assert scaled.mf_max_pos_error == pytest.approx(0.25, abs=0.002)
    assert (scaled.border, scaled.neighborhood_size, scaled.segment_radius, scaled.detection_border) == (8, 10, 5, 3)

    # A tier of the search covers the same angular speeds on bins of the same angular size
    assert scaled.mf_tier_speed == pytest.approx(0.8, abs=0.001)

    # The frame rate and the size of the input replace those of the config
    assert scaled.fps == 25.0029
    assert (scaled.width, scaled.height) == (960, 540)

    # The given config is not changed, and a scaled config is not scaled again
    assert config.mf_run_frames == [8, 16] and config.mf_block_frames == 256 and config.border == 15
    assert rs.scaleToReference(scaled, 25.0029, 65.5) is scaled

    # The block length is a whole number of the runs and the measurements
    assert mfd.matchedFilterBlockFrames(scaled) == 640


@pytest.mark.parametrize('fps', [7.0, 13.0, 15.0, 20.0, 23.0, 24.0, 25.0, 29.97, 30.0, 50.0])
def test_runs_stay_compatible(fps):
    """ The runs are scaled together, so the block (a whole number of all runs and of the measurements, see
        matchedFilterBlockFrames) stays close to the scaled block length at any frame rate, as at the reference. """

    config = referenceConfig()
    scaled = rs.scaleToReference(config, fps, REF_SCALE, size=(1920, 1080))
    requested = rs.scaleFrames(config.mf_block_frames, fps/REF_FPS)
    block = mfd.matchedFilterBlockFrames(scaled)
    assert abs(block - requested) <= 0.15*requested

    # The runs keep the ratios of the reference runs
    runs = scaled.mf_run_frames + [scaled.mf_slow_run_frames]
    assert [r/runs[0] for r in runs] == [1, 2, 4]


def test_reference_scales_the_constants_of_the_matched_filter():
    """ The constants of the matched filter in frames and pixels follow the factors of the scaled config (except
        those which follow the PSF or the runs, see MatchedFilterOptions). """

    scaled = rs.scaleToReference(referenceConfig(), INPUT_FPS, INPUT_SCALE, size=(960, 540))
    opts = mfd.MatchedFilterOptions(scaled, fps=INPUT_FPS, size=(540, 960))

    # Frames: x2.5; speeds (px per frame): x0.5/2.5; distances: x0.5; areas: x0.25
    assert opts.faint_min_span == 75
    assert opts.faint_seed_spread == pytest.approx(mfd.FAINT_SEED_SPREAD/2.5)
    assert opts.faint_max_measure_frames == 256
    assert opts.extent_step == 5
    assert opts.slow_max_speed == pytest.approx(mfd.SLOW_MAX_SPEED*0.2)
    assert opts.slow_speed == pytest.approx(mfd.SLOW_SPEED*0.2)
    assert opts.slow_min_relative_speed == pytest.approx(mfd.SLOW_MIN_RELATIVE_SPEED*0.2)
    assert opts.shadow_radius == pytest.approx(mfd.SHADOW_RADIUS/2)
    assert opts.spill_radius == pytest.approx(mfd.SPILL_RADIUS/2)
    assert opts.min_star_template_px == 25


def test_rounding():
    """ Off settings (0) stay off, a factor of 1 keeps every value, the sizes are at least 1 px. """

    assert rs.scaleFrames(0, 2.5) == 0
    assert rs.scaleFrames(20, 1.0) == 20
    assert rs.scaleFrames(1, 0.1) == 1
    assert rs.scalePowerOfTwo(20, 1.0) == 20
    assert rs.scalePowerOfTwo(8, 2.0) == 16
    assert rs.scalePowerOfTwo(8, 0.5) == 4
    assert rs.scaleSize(1, 0.25) == 1
    assert rs.scaleSize(5, 0.5) == 3


def test_chunk_frames():
    """ The default chunks of the star extraction keep their duration, given chunks stay. """

    config = referenceConfig()
    assert rs.scaleChunkFrames(rs.DEFAULT_CHUNK_FRAMES, config, INPUT_FPS) == 320
    assert rs.scaleChunkFrames(375, config, INPUT_FPS) == 375
    assert rs.scaleChunkFrames(rs.DEFAULT_CHUNK_FRAMES, config, None) == rs.DEFAULT_CHUNK_FRAMES


@pytest.mark.parametrize('fps', [7.0, 13.0, 15.0, 20.0, 23.0, 24.0, 25.0, 29.97, 30.0, 50.0])
def test_chunks_divide_the_blocks(fps):
    """ The default chunks of the star extraction divide the blocks of the matched filter, as at the reference (2
        chunks per block), so the segments of an input read once (see RMS.Formats.FrameBuffer.segmentUnit, the least
        common multiple of both) stay short, and they keep their duration within the rounding to a divisor. """

    config = referenceConfig()
    chunk = rs.scaleChunkFrames(rs.DEFAULT_CHUNK_FRAMES, config, fps)
    block = mfd.matchedFilterBlockFrames(rs.scaleToReference(config, fps, REF_SCALE, size=(1920, 1080)))
    assert block % chunk == 0
    assert abs(chunk - rs.DEFAULT_CHUNK_FRAMES*fps/REF_FPS) <= 0.25*rs.DEFAULT_CHUNK_FRAMES*fps/REF_FPS


def test_standalone_matched_filter_extracts_stars_with_the_scaled_config(tmp_path, monkeypatch):
    """ The standalone processing of a file by the matched filter scales the config before the stars are extracted,
        so the stars, the detection and the results use the same settings (as in the monitor). """

    class Handle(object):
        input_type = 'fblk'
        dir_path = str(tmp_path)
        byteswap = False
        fps = INPUT_FPS
        ncols, nrows = 960, 540
        chunk_frames = 320

        class ff(object):
            dtype = np.uint16

    used = {}

    def extract(img_handle, config, **kwargs):
        used['stars'] = (config.border, config.neighborhood_size, config.segment_radius)
        return []

    def detect(img_handle, config, **kwargs):
        used['detection'] = (config.border, config.mf_run_frames)
        return [], None

    def save(detections, star_list, img_handle, config, output_dir, **kwargs):
        used['results'] = config.mf_run_frames
        return str(tmp_path/'FTPdetectinfo_mf.txt')

    monkeypatch.setattr(mfd, 'detectInputType', lambda *a, **k: Handle())
    monkeypatch.setattr(mfd, 'loadImageCalibration', lambda *a, **k: (None, None, None))
    monkeypatch.setattr(mfd, 'extractStarsFrameInterface', extract)
    monkeypatch.setattr(mfd, 'detectMatchedFilter', detect)
    monkeypatch.setattr(mfd, 'saveMatchedFilterResults', save)
    monkeypatch.setattr(mfd, 'saveSummary', lambda *a, **k: {})

    # The field of view of the frames at 65.6 arcsec/px
    config = referenceConfig()
    config.fov_w, config.fov_h = 960*INPUT_SCALE/3600, 540*INPUT_SCALE/3600
    mfd.processFile(str(tmp_path/'input.fblk'), config, str(tmp_path))

    assert used['stars'] == (8, 10, 5)
    assert used['detection'] == (8, [20, 40])
    assert used['results'] == [20, 40]


def test_frame_size_of_ff_inputs():
    """ The handle of an FF file takes its size from the config, the size of the frames is that of the FF file. """

    import RMS.MonitorProcessFrameInterface as mon

    class FF(object):
        ncols, nrows = 960, 540

    class Handle(object):
        input_type = 'ff'
        ncols, nrows = 1920, 1080
        ff = FF()

    config = referenceConfig()
    assert mon.inputFrameSize(Handle(), config) == (960, 540)

    # Other inputs: the size of their (binned) frames, unbinned
    Handle.input_type = 'fblk'
    config.detection_binning_factor = 2
    assert mon.inputFrameSize(Handle(), config) == (3840, 2160)


def test_recalibration_rescales_a_new_platepar_to_the_frames_of_the_file(tmp_path, monkeypatch):
    """ A platepar installed at another frame size (e.g. after the camera was binned differently) is rescaled to the
        frames of every file it recalibrates, also the earlier files of the night (the backfill), whose frame size
        is in their done flag. Without a reference it is copied as it is. """

    import RMS.MonitorProcessFrameInterface as mon
    from RMS.MonitorNightReport import writeDoneFlag

    monkeypatch.setattr(mon, 'applyRecalibrate', lambda *a, **k: None)

    # A platepar of 960x540 frames, and the results of a file of 1920x1080 frames
    new_path = str(tmp_path/'new.cal')
    makePlatepar().rescaled(960, 540).write(new_path)
    results_dir = tmp_path/'results'
    results_dir.mkdir()
    (results_dir/'FTPdetectinfo_XX0001_20261026_000000_000000.txt').write_text('')
    writeDoneFlag(str(results_dir), {'night': 'n', 'frame_size': [1920, 1080]})

    config = referenceConfig()
    mon.recalibrateResults(str(results_dir), config, platepar_path=new_path)
    copied = Platepar()
    copied.read(str(results_dir/config.platepar_name))
    assert (copied.X_res, copied.Y_res) == (1920, 1080)
    assert os.path.getmtime(str(results_dir/config.platepar_name)) == pytest.approx(os.path.getmtime(new_path))

    # Without a reference the platepar is copied as it is
    plain = cr.Config()
    mon.recalibrateResults(str(results_dir), plain, platepar_path=new_path)
    copied.read(str(results_dir/plain.platepar_name))
    assert (copied.X_res, copied.Y_res) == (960, 540)


def test_recalibration_thresholds():
    """ The star residual thresholds keep their angular sizes at the pixel scale of the platepar. """

    config = referenceConfig()
    pp = makePlatepar()
    dist, quick = rs.recalibrationThresholds(config, pp)
    assert dist == pytest.approx(config.dist_check_threshold*REF_SCALE/(3600/pp.F_scale))

    # Twice the pixel scale, half the thresholds
    dist2, quick2 = rs.recalibrationThresholds(config, pp.rescaled(960, 540))
    assert (dist2, quick2) == pytest.approx((dist/2, quick/2))


def test_platepar_rescale_maps_the_sky_to_the_scaled_pixels():
    """ A sky position maps to half the pixel coordinates on the platepar of frames binned twice as much (within
        0.05 px, the difference of the forward and the reverse distortion), the field of view is the same. """

    pp = makePlatepar()
    half = pp.rescaled(960, 540)
    assert (half.X_res, half.Y_res) == (960, 540)
    assert half.F_scale == pytest.approx(pp.F_scale/2)
    assert half.vignetting_coeff == pytest.approx(2*pp.vignetting_coeff)
    assert np.array_equal(half.x_poly_fwd, pp.x_poly_fwd) and np.array_equal(half.x_poly_rev, pp.x_poly_rev)

    xs, ys = np.meshgrid(np.linspace(40, 1880, 12), np.linspace(40, 1040, 7))
    xs, ys = xs.ravel(), ys.ravel()
    _, ra, dec, _ = xyToRaDecPP(np.full(xs.size, pp.JD), xs, ys, np.ones(xs.size), pp, extinction_correction=False,
                                jd_time=True)
    x2, y2 = raDecToXYPP(np.asarray(ra), np.asarray(dec), pp.JD, half)
    assert np.max(np.abs(2*np.asarray(x2) - xs)) < 0.1
    assert np.max(np.abs(2*np.asarray(y2) - ys)) < 0.1
    assert np.max(np.hypot(np.asarray(x2) - xs/2, np.asarray(y2) - ys/2)) < 0.05

    # The original is not changed
    assert (pp.X_res, pp.Y_res) == (1920, 1080)


def test_platepar_rescale_refuses():
    """ Frames of another aspect, or a distortion model in pixels, are not rescaled. """

    pp = makePlatepar()
    assert pp.rescaled(960, 600) is None

    poly = makePlatepar()
    poly.setDistortionType("poly3+radial", reset_params=True)
    assert poly.rescaled(960, 540) is None


def test_sky_reference_per_frame_size(tmp_path):
    """ A clear-sky reference of frames of another size is kept renamed and a new one is started; a reference
        without a size is used. """

    keys = np.array([1, 2, 3])
    values = np.array([0.1, 0.2, 0.3])
    rates = np.ones(3)
    t = datetime.datetime(2026, 10, 26, 4, 0, 0)

    # A reference of 1920x1080 frames
    ref = SkyReference(str(tmp_path), 'XX0001', frame_size=(1920, 1080))
    ref.update(t, keys, values, rates, 17.9, [])
    assert len(SkyReference(str(tmp_path), 'XX0001', frame_size=(1920, 1080)).keys) == 3

    # Frames of another size don't use it, and the next update keeps it renamed
    new = SkyReference(str(tmp_path), 'XX0001', frame_size=(960, 540))
    assert len(new.keys) == 0 and len(new.zeroPoints()) == 0
    new.update(t, keys[:1], values[:1], rates[:1], 15.5, [])
    assert os.path.isfile(os.path.join(str(tmp_path), 'sky_reference_XX0001_1920x1080.npz'))
    again = SkyReference(str(tmp_path), 'XX0001', frame_size=(960, 540))
    assert len(again.keys) == 1 and list(again.zeroPoints()) == [15.5]

    # Without a size (as before) the reference is used whatever its size
    assert len(SkyReference(str(tmp_path), 'XX0001').keys) == 1


def test_auto_calibration_state_per_frame_size(tmp_path):
    """ The state of the automatic calibration of frames of another size is kept renamed and the camera starts
        over; a state without a size is used. """

    out = str(tmp_path)

    # A state without a size gets the size of the frames
    AutoCalibration.writeState(out, {'files': [{'results_dir': 'a'}], 'star_counts': [100]})
    state = AutoCalibration._stateForSize(out, AutoCalibration.readState(out), (1920, 1080))
    assert state['frame_size'] == [1920, 1080] and state['star_counts'] == [100]
    AutoCalibration.writeState(out, state)

    # Frames of another size start over
    state = AutoCalibration._stateForSize(out, AutoCalibration.readState(out), (960, 540))
    assert state['frame_size'] == [960, 540] and state['star_counts'] == [] and state['files'] == []
    with open(os.path.join(out, 'auto_calibration_state_1920x1080.json')) as f:
        assert json.load(f)['star_counts'] == [100]
