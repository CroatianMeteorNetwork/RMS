"""
Tests for the matched-filter detection of faint moving objects (RMS.MatchedFilterDetection and
RMS.Routines.MatchedFilterKernels), on synthetic frames: a constant sky with noise, static stars, and objects
moving along straight or curved tracks well below the single-frame detection limit.
"""

import os
import math
import datetime

import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Formats import FTPdetectinfo
from RMS.DetectStarsAndMeteors import saveResultsFrameInterface
import RMS.MatchedFilterDetection as mfd
from RMS.MatchedFilterDetection import velocityGrid, binFrames, MatchedFilterDetector, smoothTrack, smoothPositions
from RMS.Routines.MatchedFilterKernels import CUDA_AVAILABLE, VelocityStacker, fitMovingPSF


BEG_TIME = datetime.datetime(2026, 10, 8, 3, 0, 0)

SIZE = 128
SKY = 100.0
NOISE = 5.0
PSF_SIGMA = 1.0


def renderObject(img, x, y, peak, sigma=PSF_SIGMA, vx=0.0, vy=0.0, n_sub=5):
    """ Add a Gaussian PSF with the given peak (ADU) at (x, y), smeared over the frame by the motion. """

    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    for s in range(n_sub):
        d = (s + 0.5)/n_sub - 0.5
        img += peak/n_sub*np.exp(-0.5*((xx - x - vx*d)**2 + (yy - y - vy*d)**2)/sigma**2)


class _SynthHandle(object):
    """ Frame-based image handle of synthetic frames: sky, noise, static stars, and moving objects whose
        positions are functions of the frame number.
    """

    input_type = 'vid'
    byteswap = False

    def __init__(self, objects, total_frames=512, fps=25.0, seed=1, n_stars=12, n_faint_stars=0,
                 star_gain=None, clip=None):

        self.objects = objects
        self.total_frames = total_frames
        self.nrows = self.ncols = SIZE
        self.fps = fps
        self.beginning_datetime = BEG_TIME
        self.dir_path = None
        self.seed = seed
        self.current_frame = 0

        # Brightness of the stars in every frame relative to the average (e.g. the transparency of the sky)
        self.star_gain = star_gain

        # Saturation level of the frames (pixel values are clipped there), None for no saturation
        self.clip = clip

        rng = np.random.default_rng(seed + 1000)
        self.stars = np.zeros((SIZE, SIZE))

        # Positions and fluxes of the bright stars (x, y, flux)
        self.star_list = []
        for _ in range(n_stars):
            x, y, peak = rng.uniform(10, SIZE - 10), rng.uniform(10, SIZE - 10), rng.uniform(5, 50)*NOISE
            renderObject(self.stars, x, y, peak)
            self.star_list.append((x, y, peak*2*np.pi*PSF_SIGMA**2))
        for _ in range(n_faint_stars):
            renderObject(self.stars, rng.uniform(5, SIZE - 5), rng.uniform(5, SIZE - 5),
                         rng.uniform(0.5, 2)*NOISE)

    def setFrame(self, fr_num):
        self.current_frame = fr_num

    def loadFrame(self, avepixel=False):

        f = self.current_frame
        rng = np.random.default_rng(self.seed*100003 + f)
        gain = self.star_gain(f) if self.star_gain is not None else 1.0
        img = SKY + gain*self.stars + rng.normal(0, NOISE, (SIZE, SIZE))
        for obj in self.objects:
            if obj['f0'] <= f < obj['f1']:
                x, y = obj['pos'](f)
                x2, y2 = obj['pos'](f + 0.5)
                x1, y1 = obj['pos'](f - 0.5)

                # The brightness can change with time (e.g. flashes)
                snr = obj['snr'](f) if callable(obj['snr']) else obj['snr']
                renderObject(img, x, y, snr*NOISE, vx=x2 - x1, vy=y2 - y1)

                # Charge spilled along the row and the column of a very bright object: the given fraction of
                #   its flux spread over +-10 px
                if obj.get('spill'):
                    xi, yi = int(round(x)), int(round(y))
                    flux = snr*NOISE*2*np.pi*PSF_SIGMA**2*obj['spill']
                    for d in list(range(-10, -3)) + list(range(4, 11)):
                        if 0 <= xi + d < SIZE:
                            img[yi, xi + d] += flux/28
                        if 0 <= yi + d < SIZE:
                            img[yi + d, xi] += flux/28

                # Trails of a bright source along its whole column, and along its row on one side
                if obj.get('trail'):
                    xi, yi = int(round(x)), int(round(y))
                    img[:, xi] += obj['trail']*NOISE
                    img[yi, :max(xi - 3, 0)] += 2*obj['trail']*NOISE

        if self.clip is not None:
            img = np.minimum(img, self.clip)

        return img.astype(np.float32)

    def currentFrameTime(self, frame_no=None, dt_obj=False):

        if frame_no is None:
            frame_no = self.current_frame

        return BEG_TIME + datetime.timedelta(seconds=frame_no/self.fps)


def linearObject(snr, x0, y0, vx, vy, f0, f1):
    """ An object moving at a constant velocity, at (x0, y0) in the frame f0. """

    return {'snr': snr, 'f0': f0, 'f1': f1, 'pos': lambda f: (x0 + vx*(f - f0), y0 + vy*(f - f0))}


@pytest.fixture
def config():

    config = cr.Config()
    config.stationID = 'XX0001'
    config.width = config.height = SIZE
    config.fps = 25.0

    # 0.04 deg/px: the speeds of 0.01 - 2 deg/s are 0.01 - 2 px/frame
    config.fov_w = config.fov_h = 0.04*SIZE
    config.detection_binning_factor = 1
    config.mf_psf_sigma = PSF_SIGMA
    config.mf_gpu = 'off'

    return config


def truthError(cent, obj):
    """ Distances of the centroids from the true positions (px). """

    truth = np.array([obj['pos'](f) for f in cent[:, 0]])
    return np.hypot(cent[:, 1] - truth[:, 0], cent[:, 2] - truth[:, 1])


### Kernels ###

def test_velocity_grid_sorted_and_bounded():

    vel = velocityGrid(2.0, 8, 2)
    speed = np.hypot(vel[:, 0], vel[:, 1])

    assert np.all(np.diff(speed) >= 0)
    assert speed[0] == 0
    assert speed.max() <= 2.0/2 + 1.0/7 + 1e-9

    # The spacing is a binned pixel over the run: the largest position error at the run end is half a pixel
    assert np.isclose(np.min(np.abs(np.diff(np.unique(vel[:, 0])))), 1.0/7)


def test_bin_frames_keeps_unit_noise():

    rng = np.random.default_rng(0)
    binned = binFrames(rng.normal(0, 1, (4, 128, 128)).astype(np.float32), 2)

    assert binned.shape == (4, 64, 64)
    assert abs(np.std(binned) - 1) < 0.03


def test_velocity_stack_finds_moving_object():

    rng = np.random.default_rng(1)
    n = 8
    frames = rng.normal(0, 1, (n, 64, 64)).astype(np.float32)
    vel_true = np.array([1.0, -0.5])
    for k in range(n):
        dt = k - (n - 1)/2.0
        x, y = 30 + vel_true[0]*dt, 32 + vel_true[1]*dt
        frames[k, int(round(y)), int(round(x))] += 3.0

    vel = velocityGrid(1.5, n, 1)
    out_max, out_idx = VelocityStacker(vel, n, use_gpu=False).stackMax(frames)

    y, x = np.unravel_index(np.argmax(out_max), out_max.shape)
    assert (x, y) == (30, 32)
    assert np.hypot(*(vel[out_idx[y, x]] - vel_true)) <= 1.0/(n - 1) + 1e-9

    # The mean along the right velocity is the object signal plus the noise of the mean
    assert out_max[y, x] > 2.0


def test_velocity_stack_matches_direct_sum():
    """ The stack of every velocity is the mean of the shifted frames, with pixels outside the image as zero. """

    rng = np.random.default_rng(4)
    n = 6
    frames = rng.normal(0, 1, (n, 20, 30)).astype(np.float32)
    vel = velocityGrid(2.0, n, 1)
    stacker = VelocityStacker(vel, n, use_gpu=False)
    out_max, out_idx = stacker.stackMax(frames)

    padded = np.pad(frames, ((0, 0), (10, 10), (10, 10)))
    sums = np.zeros((len(vel), 20, 30))
    for v in range(len(vel)):
        for k in range(n):
            dx, dy = stacker.shifts[v, k]
            sums[v] += padded[k, 10 + dy:30 + dy, 10 + dx:40 + dx]
    assert np.allclose(out_max, sums.max(axis=0)/n, atol=1e-5)
    assert np.all(sums[out_idx, np.arange(20)[:, None], np.arange(30)] >= sums.max(axis=0) - 1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_velocity_stack_cpu_gpu_agree():

    rng = np.random.default_rng(2)
    frames = rng.normal(0, 1, (8, 96, 80)).astype(np.float32)
    vel = velocityGrid(2.0, 8, 2)

    max_cpu, idx_cpu = VelocityStacker(vel, 8, use_gpu=False).stackMax(frames)
    max_gpu, idx_gpu = VelocityStacker(vel, 8, use_gpu=True).stackMax(frames)

    assert np.allclose(max_cpu, max_gpu, atol=1e-5)
    assert np.mean(idx_cpu == idx_gpu) > 0.999


@pytest.mark.parametrize("vx, vy", [(0.0, 0.0), (1.5, -0.7)])
def test_moving_psf_fit_unbiased_with_realistic_errors(vx, vy):
    """ The joint fit of the moving PSF to runs of 8 frames at a peak SNR of 2 per frame: no bias, and the
        scatter of the positions matches the uncertainties of the fit.
    """

    rng = np.random.default_rng(3)
    n = 8
    dt = np.arange(n) - (n - 1)/2.0
    radius = int(math.ceil(3*PSF_SIGMA + 0.5*math.hypot(vx, vy) + 1))

    errs, sig = [], []
    for _ in range(300):
        xt, yt = 20 + rng.uniform(-0.5, 0.5), 20 + rng.uniform(-0.5, 0.5)
        frames = rng.normal(0, 1, (n, 40, 40))
        for k in range(n):
            renderObject(frames[k], xt + vx*dt[k], yt + vy*dt[k], 2.0, vx=vx, vy=vy)

        # Predicted 0.7 px off the truth
        res = fitMovingPSF(frames.astype(np.float32), np.zeros(frames.shape, np.uint8), xt + 0.5, yt - 0.5, dt, vx, vy,
                           PSF_SIGMA, radius)
        if res[9] and np.isfinite(res[0]):
            errs.append((res[0] - xt, res[1] - yt))
            sig.append((res[4], res[5]))

    errs = np.array(errs)
    sig = np.array(sig)

    assert len(errs) > 280
    assert np.all(np.abs(np.mean(errs, axis=0)) < 0.05)
    ratio = np.std(errs, axis=0)/np.median(sig, axis=0)
    assert np.all((ratio > 0.8) & (ratio < 1.3))


def test_smooth_track_follows_curve():

    frames = np.arange(0, 800, 4.0)
    cent = np.column_stack([frames, 10 + 0.1*frames + 2e-5*frames**2, 50 - 0.05*frames])
    xs, ys = smoothTrack(cent, np.arange(0, 797))
    f = np.arange(0, 797)

    assert np.max(np.abs(xs - (10 + 0.1*f + 2e-5*f**2))) < 0.05
    assert np.max(np.abs(ys - (50 - 0.05*f))) < 0.05


def test_smooth_positions_reduce_errors_without_bias():
    """ Noisy positions along a curved track (a bend of 4 px over 400 frames, as across a distorted field):
        combining them over +-64 frames reduces the error several times and does not follow the noise.
    """

    rng = np.random.default_rng(8)
    f = np.arange(0, 400, 4.0)
    true_x = 20 + 0.3*f
    true_y = 30 + 0.1*f + 8*4.0/400**2*0.5*(f - 200)**2
    cent = np.column_stack([f, true_x + rng.normal(0, 0.4, len(f)), true_y + rng.normal(0, 0.4, len(f)),
                            np.zeros(len(f)), np.zeros(len(f)), np.full(len(f), 3.0)])

    # One measurement far off (e.g. a fit pulled to a noise peak at the end of the track)
    cent[0, 1] += 8.0

    smooth, keep = smoothPositions(cent, 64)
    raw_err = np.hypot(cent[keep, 1] - true_x[keep], cent[keep, 2] - true_y[keep])
    err = np.hypot(smooth[:, 0] - true_x[keep], smooth[:, 1] - true_y[keep])

    assert not keep[0] and keep[1:].sum() >= len(f) - 3
    assert np.sqrt(np.mean(err**2)) < np.sqrt(np.mean(raw_err**2))/2.5
    assert abs(np.mean(smooth[:, 1] - true_y[keep])) < 0.05

    # The ends of the track are not worse than the raw measurements
    assert np.all(err[:5] < 0.6) and np.all(err[-5:] < 0.6)


### Detection ###

def runDetector(handle, config):

    det = MatchedFilterDetector(handle, config)
    return det, det.run()


def test_detects_faint_object_below_single_frame_limit(config):
    """ An object with a peak SNR of 1.5 per frame (far below the normal detection) moving at 0.3 px/frame is
        found and measured to about half a pixel. Its positions are measured on several frames together, but
        there is a row for every frame (with the intensity of that frame).
    """

    obj = linearObject(1.5, 25, 30, 0.3, 0.12, 40, 470)
    det, detections = runDetector(_SynthHandle([obj]), config)

    assert len(detections) == 1
    cent = detections[0][2]

    err = truthError(cent, obj)
    assert np.sqrt(np.mean(err**2)) < 0.6

    # Most of the track is measured, every frame
    assert cent[-1, 0] - cent[0, 0] > 0.7*(obj['f1'] - obj['f0'])
    assert np.all(np.diff(cent[:, 0]) >= 1) and (np.median(np.diff(cent[:, 0])) == 1)


def test_bright_object_measured_every_frame(config):

    obj = linearObject(12.0, 20, 100, 0.8, -0.3, 100, 220)
    _, detections = runDetector(_SynthHandle([obj]), config)

    assert len(detections) == 1
    cent = detections[0][2]
    assert np.median(np.diff(cent[:, 0])) == 1
    assert np.sqrt(np.mean(truthError(cent, obj)**2)) < 0.25


def test_detects_fast_faint_object_with_velocity_search(config):

    obj = linearObject(2.5, 15, 20, 1.6, 0.9, 200, 255)
    _, detections = runDetector(_SynthHandle([obj]), config)

    assert len(detections) == 1
    assert np.sqrt(np.mean(truthError(detections[0][2], obj)**2)) < 0.5

    # Without the velocity search, the object is too fast to be found
    config.mf_velocity_search = False
    _, detections = runDetector(_SynthHandle([obj]), config)
    assert len(detections) == 0


def test_follows_curved_track(config):
    """ A track bent like an object seen through a distorting lens: the measurements follow the curve. """

    def pos(f):
        t = f - 50
        return 20 + 0.22*t, 20 + 0.02*t + 0.0004*t**2

    obj = {'snr': 2.5, 'f0': 50, 'f1': 480, 'pos': pos}
    _, detections = runDetector(_SynthHandle([obj]), config)

    assert len(detections) == 1
    cent = detections[0][2]
    assert np.sqrt(np.mean(truthError(cent, obj)**2)) < 0.5
    assert cent[-1, 0] - cent[0, 0] > 0.7*(obj['f1'] - obj['f0'])


def test_no_detections_in_noise_and_stars(config):

    for seed in (5, 6):
        _, detections = runDetector(_SynthHandle([], seed=seed), config)
        assert len(detections) == 0


@pytest.mark.parametrize('correct', [True, False])
def test_transparency_changes_are_removed(config, monkeypatch, correct):
    """ Faint stars (too faint to be masked) whose brightness changes with the transparency of the sky leave
        no residuals in the normalized frames, which would make faint static sources varying in time.
    """

    import RMS.MatchedFilterDetection as mfd

    if not correct:
        monkeypatch.setattr(mfd, 'MIN_STAR_TEMPLATE_PX', 10**9)

    handle = _SynthHandle([], n_stars=0, n_faint_stars=150,
                          star_gain=lambda f: 1.0 + 0.8*math.sin(2*math.pi*f/300.0))
    det = MatchedFilterDetector(handle, config)
    det.block_starts, det.block_ends = [0, 256], [256, 512]
    det.backgroundPass()

    # Residuals of the stars relative to their brightness, in the mean of 32 frames when the stars are
    #   brightest (1.8 times the average)
    z = det.normalize(det.readFrames(0, 256), 0)
    star_px = handle.stars > 0.5*NOISE
    resid = np.mean(np.mean(z[60:92], axis=0)[star_px]/(handle.stars[star_px]/NOISE))

    if correct:
        assert abs(resid) < 0.1
    else:
        assert resid > 0.5


def test_faint_object_found_with_changing_transparency(config):

    obj = linearObject(2.0, 25, 30, 0.2, 0.1, 40, 470)
    handle = _SynthHandle([obj], n_stars=4, n_faint_stars=150,
                          star_gain=lambda f: 1.0 + 0.8*math.sin(2*math.pi*f/300.0))
    _, detections = runDetector(handle, config)

    assert len(detections) == 1
    assert np.sqrt(np.mean(truthError(detections[0][2], obj)**2)) < 0.6


def test_short_track_rejected(config):
    """ Tracks shorter than mf_min_frames are not kept. """

    obj = linearObject(15.0, 40, 40, 1.0, 0.0, 300, 314)
    config.mf_min_frames = 20
    _, detections = runDetector(_SynthHandle([obj]), config)

    assert len(detections) == 0


### Output ###

def test_ftpdetectinfo_round_trip(config, tmp_path):

    obj = linearObject(4.0, 25, 30, 0.3, 0.15, 40, 300)
    handle = _SynthHandle([obj])
    _, detections = runDetector(handle, config)
    assert len(detections) == 1

    _, _, ftp_name = saveResultsFrameInterface([], detections, handle, config, output_suffix='mf',
                                               output_dir=str(tmp_path))
    assert ftp_name.endswith('_mf.txt')

    _, _, meteors = FTPdetectinfo.readFTPdetectinfo(str(tmp_path), ftp_name, ret_input_format=True)
    assert len(meteors) == 1

    # Centroid rows: frame (relative to the time in the FF name), x, y, ...
    picks = np.array(meteors[0][4], dtype=np.float64)
    cent = detections[0][2]
    assert len(picks) == len(cent)
    assert np.allclose(picks[:, 1], cent[:, 1], atol=0.01)
    assert np.allclose(picks[:, 2], cent[:, 2], atol=0.01)

    # The times are kept, the frames are counted from the time in the name
    assert np.allclose(np.diff(picks[:, 0]), np.diff(cent[:, 0]), atol=0.01)


def test_config_parse(tmp_path):

    path = os.path.join(str(tmp_path), '.config')
    with open(path, 'w') as f:
        f.write("[System]\nstationID: XX0001\n\n[MatchedFilter]\nmf_enable: true\nmf_run_frames: 4, 8\n"
                "mf_threshold: 6.5\nmf_gpu: off\nmf_min_frames: 30\n")

    config = cr.parse(path)

    assert config.mf_enable is True
    assert config.mf_run_frames == [4, 8]
    assert config.mf_threshold == 6.5
    assert config.mf_gpu == 'off'
    assert config.mf_min_frames == 30


### Operation ###

def test_ff_input_is_skipped(config):

    class _FF(object):
        input_type = 'ff'

    assert mfd.detectMatchedFilter(_FF(), config) == []
    assert mfd.detectMatchedFilter(_FF(), config, return_detector=True) == ([], None)


def test_summary_lists_tracks_with_significance(config):

    obj = linearObject(2.0, 25, 30, 0.3, 0.12, 40, 470)
    det, detections = runDetector(_SynthHandle([obj]), config)
    summary = det.summary()

    assert summary['total_frames'] == 512
    assert sum(t['detected'] for t in summary['tracks']) == len(detections) == 1
    assert all((t['significance'] >= config.mf_track_significance) == t['detected'] for t in summary['tracks'])
    assert 'measure' in summary['timing_s']


def test_find_input_files(tmp_path):

    (tmp_path/'night'/'sub').mkdir(parents=True)
    for name in ('a.vid', 'sub/b.vid', 'sub/c.MKV', 'notes.txt'):
        (tmp_path/'night'/name).write_text('x')

    files = mfd.findInputFiles([str(tmp_path/'night'), str(tmp_path/'night'/'a.vid')])

    assert [os.path.basename(f) for f in files] == ['a.vid', 'b.vid', 'c.MKV']


def test_process_files_resumes_and_continues_after_failure(config, tmp_path, monkeypatch):
    """ Files with a done file are skipped, a failing file doesn't stop the others, and the detections of all
        files are merged.
    """

    calls = []

    def _process(path, config, out, **kwargs):
        calls.append(os.path.basename(path))
        if 'bad' in path:
            raise RuntimeError('broken file')
        os.makedirs(out)
        picks = [[f, 10.0 + f, 20.0, 1.0, 2.0, 3.0, 4.0, 100, 9.0, 50, 5.0, 0] for f in (1.0, 2.0)]
        # The recalibration leaves a backup of the uncalibrated file next to the calibrated one
        for name in ('FTPdetectinfo_{:s}_mf.txt', 'FTPdetectinfo_{:s}_mf_backup_20261008_184146.444121.txt'):
            FTPdetectinfo.writeFTPdetectinfo([['FF_XX0001_20261008_030000_000_0000000.fits', 1, 10.0, 45.0,
                                               picks, 25.0]], out, name.format(os.path.basename(out)), out,
                                             'XX0001', 25.0, celestial_coords_given=True)
        return mfd.saveSummary(out, None, detections=1, processing_time_s=1.0)

    monkeypatch.setattr(mfd, 'processFile', _process)
    files = [str(tmp_path/name) for name in ('a.vid', 'bad.vid', 'c.vid')]
    out = str(tmp_path/'out')

    assert mfd.processFiles(files, config, out) == [files[1]]
    assert calls == ['a.vid', 'bad.vid', 'c.vid']

    merged = [f for f in os.listdir(out) if f.startswith('FTPdetectinfo')]
    assert merged == ['FTPdetectinfo_out_mf.txt']
    assert len(FTPdetectinfo.readFTPdetectinfo(out, merged[0])) == 2

    # Started again: only the failed file is processed
    calls.clear()
    assert mfd.processFiles(files, config, out) == [files[1]]
    assert calls == ['bad.vid']


def test_output_names_unique_for_same_file_names(tmp_path):

    files = [str(tmp_path/'cam1'/'a.vid'), str(tmp_path/'cam2'/'a.vid'), str(tmp_path/'cam1'/'b.vid')]
    names = mfd.outputNames(files)

    assert names[2] == 'b'
    assert len(set(names)) == 3 and all(n.startswith('a_') for n in names[:2])
    assert mfd.outputNames(files) == names


@pytest.mark.parametrize('method, factor', [('avg', 4), ('sum', 1)])
def test_binned_frames_speed_limits_and_intensity(config, method, factor):
    """ With 2x2 detection binning the speed limits are in binned pixels, and the intensity is scaled to the
        unbinned image only for averaged bins.
    """

    config.detection_binning_factor = 2
    config.detection_binning_method = method

    opts = mfd.MatchedFilterOptions(config, det_bin=2)
    unbinned = mfd.MatchedFilterOptions(config)
    assert np.isclose(opts.speed_max, unbinned.speed_max/2)
    assert np.isclose(opts.speed_min, unbinned.speed_min/2)

    obj = linearObject(12.0, 20, 100, 0.8, -0.3, 100, 220)
    det = MatchedFilterDetector(_SynthHandle([obj]), config)
    detections = det.run()
    config.detection_binning_factor = 1
    det1 = MatchedFilterDetector(_SynthHandle([obj]), config)
    detections1 = det1.run()

    assert len(detections) == len(detections1) == 1
    ratio = np.median(detections[0][2][:, 3])/np.median(detections1[0][2][:, 3])
    assert np.isclose(ratio, factor, rtol=0.1)



### Photometry and saturation ###

@pytest.mark.parametrize('snr, vx', [(3.0, 0.3), (12.0, 0.8), (12.0, 1.8)])
def test_intensity_is_the_flux_per_frame(config, snr, vx):
    """ The intensity of a measurement is the background-subtracted sum of the pixels of the object in a frame,
        as for the normal detection: the flux of the PSF (peak*2*pi*sigma^2), within a few percent.
    """

    obj = linearObject(snr, 20, 60, vx, 0.1, 60, 60 + int(80/vx))
    config.mf_smooth_frames = 0
    _, detections = runDetector(_SynthHandle([obj], n_stars=4), config)

    assert len(detections) == 1
    flux = snr*NOISE*2*np.pi*PSF_SIGMA**2
    intensity = np.median(detections[0][2][:, 3])
    assert abs(intensity/flux - 1) < (0.06 if snr > 5 else 0.15)


def test_saturated_object_flagged_and_measured(config):
    """ An object whose core is clipped by saturation: the measurements are flagged, and leaving the clipped
        pixels out of the fit keeps the positions accurate.
    """

    obj = linearObject(60.0, 20, 60, 0.6, 0.2, 60, 200)
    clip = SKY + 25*NOISE
    config.mf_saturation_level = clip - 1
    config.mf_smooth_frames = 0
    _, detections = runDetector(_SynthHandle([obj], n_stars=4, clip=clip), config)

    assert len(detections) == 1
    cent = detections[0][2]
    assert np.median(cent[:, 6]) >= 1
    assert np.sqrt(np.mean(truthError(cent, obj)**2)) < 0.15

    # Without a saturation level, nothing is flagged
    config.mf_saturation_level = 0
    _, detections = runDetector(_SynthHandle([obj], n_stars=4, clip=clip), config)
    assert np.all(detections[0][2][:, 6] == 0)



@pytest.mark.parametrize('mask_slow', [True, False])
def test_slow_object_photometry_uses_background_without_it(config, monkeypatch, mask_slow):
    """ A slow object stays on the same pixels for a large part of the frames of the background, which then
        contains part of it; the background is estimated again without it, so its intensity is the flux.
    """

    if not mask_slow:
        monkeypatch.setattr(mfd, 'SLOW_SPEED', 0.0)

    # 1536 frames: the object stays on its pixels for a tenth of them (in a 10-minute file, under 2%)
    obj = linearObject(6.0, 30, 60, 0.03, 0.015, 0, 1536)
    config.mf_smooth_frames = 0
    _, detections = runDetector(_SynthHandle([obj], n_stars=4, total_frames=1536), config)

    assert len(detections) == 1
    flux = 6.0*NOISE*2*np.pi*PSF_SIGMA**2
    ratio = np.median(detections[0][2][:, 3])/flux
    if mask_slow:
        assert abs(ratio - 1) < 0.1
    else:
        assert ratio < 0.85


@pytest.mark.parametrize('trail_level', [25.0, 0.0])
def test_trails_of_bright_object_removed(config, trail_level):
    """ A very bright object which leaves faint trails along its column and its row: the trails move with it
        and make tracks along them, unless they are removed from the search.
    """

    obj = linearObject(400.0, 110, 50, -0.6, 0.1, 100, 260)
    obj['trail'] = 1.0
    config.mf_trail_level = trail_level
    _, detections = runDetector(_SynthHandle([obj], n_stars=4), config)

    if trail_level > 0:
        assert len(detections) == 1
        assert np.median(truthError(detections[0][2], obj)) < 0.3
    else:
        assert len(detections) > 1


### Light curves, bright objects, photometric scale ###

def test_flashes_kept_in_light_curve(config):
    """ An object below the single-frame limit which flashes for one frame every 24 frames: it is found, and
        the light curve has the flashes at their full brightness
        (the positions of faint objects are measured on several frames, but the intensity of every frame is
        kept).
    """

    flash_snr, base_snr = 40.0, 1.5
    flashes = set(range(110, 460, 24))
    snr = lambda f: flash_snr if int(f) in flashes else base_snr
    obj = linearObject(1.0, 25, 30, 0.25, 0.1, 100, 460)
    obj['snr'] = snr
    _, detections = runDetector(_SynthHandle([obj]), config)

    assert len(detections) == 1
    cent = detections[0][2]
    flux = lambda s: s*NOISE*2*np.pi*PSF_SIGMA**2
    at_flash = np.isin(np.round(cent[:, 0]).astype(int), list(flashes))
    assert at_flash.sum() >= 0.8*len(flashes)
    assert abs(np.median(cent[at_flash, 3])/flux(flash_snr) - 1) < 0.15
    assert abs(np.median(cent[~at_flash, 3])/flux(base_snr) - 1) < 0.3


def test_spilled_charge_flagged_and_counted(config):
    """ A very bright object whose charge spills along its row and column (without saturated pixels): the
        measurements are flagged as saturated and the intensity includes the spilled charge.
    """

    obj = linearObject(400.0, 30, 64, 0.5, 0.05, 100, 200)
    obj['spill'] = 1.0
    config.mf_smooth_frames = 0

    # No pixel reaches the saturation level
    config.mf_saturation_level = 10**6
    _, detections = runDetector(_SynthHandle([obj], n_stars=4), config)

    assert len(detections) == 1
    cent = detections[0][2]
    total = 400.0*NOISE*2*np.pi*PSF_SIGMA**2*2.0
    assert np.median(cent[:, 6]) >= 1
    assert abs(np.median(cent[:, 3])/total - 1) < 0.1

    # A bright object without spill is not flagged
    obj['spill'] = 0.0
    _, detections = runDetector(_SynthHandle([obj], n_stars=4), config)
    assert np.all(detections[0][2][:, 6] == 0)


def test_intensities_on_the_scale_of_the_stars(config):
    """ The intensities are scaled to the intensities of the stars of the calibration: here the stars are
        listed 10% brighter than their flux, so the intensity of the object is 10% above its flux.
    """

    obj = linearObject(8.0, 20, 40, 0.4, 0.1, 100, 300)
    handle = _SynthHandle([obj], n_stars=30)
    stars = [(y, x, 1.1*flux, 0, 2.0, SKY, 50.0, 0) for x, y, flux in handle.star_list]
    star_list = [['FF_XX0001_20261008_030000_000_0000000.fits', stars],
                 ['FF_XX0001_20261008_030010_000_0000256.fits', stars]]
    config.mf_smooth_frames = 0

    detections, detector = mfd.detectMatchedFilter(handle, config, star_list=star_list, return_detector=True)
    plain = mfd.detectMatchedFilter(_SynthHandle([obj], n_stars=30), config)

    assert len(detections) == len(plain) == 1
    ratio = np.median(detections[0][2][:, 3])/np.median(plain[0][2][:, 3])
    assert abs(detector.aperture_correction/1.1 - 1) < 0.05
    assert abs(ratio/detector.aperture_correction - 1) < 0.01
    assert detector.summary()['aperture_correction'] == detector.aperture_correction


### Robustness ###

def test_strong_trail_is_not_restored(config):
    """ The trails of a very bright object which are themselves above trail_level are removed as well (only
        the object itself is kept).
    """

    obj = linearObject(4000.0, 100, 50, -0.6, 0.1, 100, 260)
    obj['trail'] = 30.0
    _, detections = runDetector(_SynthHandle([obj], n_stars=4), config)

    assert len(detections) == 1


def test_no_edge_margin(config):
    """ With no edge margin, the image is not masked. """

    config.mf_edge_margin = 0
    config.detection_border = 0
    obj = linearObject(2.5, 25, 30, 0.3, 0.12, 40, 470)
    _, detections = runDetector(_SynthHandle([obj]), config)

    assert len(detections) == 1


def test_too_few_frames(config):

    _, detections = runDetector(_SynthHandle([], total_frames=4), config)
    assert detections == []


def test_calibration_of_the_caller_not_binned(config):
    """ With detection binning, the caller's mask is not binned again (the normal detection may have binned it
        already in place).
    """

    from RMS.Routines.MaskImage import MaskStructure

    config.detection_binning_factor = 2
    mask = MaskStructure(np.full((SIZE, SIZE), 255, dtype=np.uint8))

    class _Binned(_SynthHandle):
        def loadFrame(self, avepixel=False):
            img = super(_Binned, self).loadFrame()
            return img.reshape(SIZE//2, 2, SIZE//2, 2).mean(axis=(1, 3)).astype(np.float32)

    handle = _Binned([linearObject(4.0, 20, 30, 0.4, 0.1, 50, 300)])
    handle.nrows = handle.ncols = SIZE//2
    mfd.detectMatchedFilter(handle, config, mask=mask)
    assert mask.img.shape == (SIZE, SIZE)


def test_slow_object_kept_after_background_without_it(config):
    """ A slow object is measured again on a background estimated without it; the star template and the masks
        made with it in the background must be updated too, or its own path is left out of the verification.
    """

    for snr in (6.0, 10.0):
        obj = linearObject(snr, 30, 60, 0.012, 0.006, 0, 2048)
        _, detections = runDetector(_SynthHandle([obj], n_stars=4, total_frames=2048), config)
        assert len(detections) == 1, snr


def test_smoothing_keeps_faint_measurements():
    """ Measurements of a track whose brightness changes: the faint ones scatter more, in proportion to their
        errors, and are not outliers.
    """

    rng = np.random.default_rng(3)
    t = np.arange(400, dtype=float)
    snr = np.where(t < 340, 20.0, 3.0)
    err = 0.5/snr
    cent = np.column_stack([t, 10 + 0.1*t + rng.normal(0, err), 20 + 0.05*t + rng.normal(0, err),
                            np.ones_like(t), np.ones_like(t), snr])
    _, keep = smoothPositions(cent, 64)

    assert keep[t >= 340].mean() > 0.95


def test_merge_tracks_through_a_bridge(config):
    """ Pieces of one track: A and C are too far apart to be merged directly, B fills the gap; and a sparse
        track of 32-frame runs which overlaps a track of 8-frame runs in its gap.
    """

    det = MatchedFilterDetector(_SynthHandle([]), config)
    row = lambda f, run: [f, 20 + 0.1*f, 30 + 0.05*f, 6.0, 0.1, 0.05, run]

    a = np.array([row(f, 8) for f in np.arange(3.5, 400, 8)])
    b = np.array([row(f, 8) for f in np.arange(403.5, 460, 8)])
    c = np.array([row(f, 8) for f in np.arange(600 + 3.5, 1000, 8)])
    b2 = np.array([row(f, 8) for f in np.arange(459.5, 600, 8)])
    assert len(det.mergeTracks([a, c, b, b2])) == 1

    sparse = np.array([row(f, 32) for f in (15.5, 47.5, 175.5, 207.5)])
    dense = np.array([row(f, 8) for f in np.arange(59.5, 164, 8)])
    assert len(det.mergeTracks([sparse, dense])) == 1



def test_flashes_without_light_between_are_linked(config):
    """ An object seen only in its flashes (2 frames every 16 frames): the hits of the runs with a flash have a
        poorly known velocity, so they are linked by their positions. The flashes count fully in the search only
        with a higher clipping level of the frames.
    """

    flashes = {f for f in range(100, 320) if (f - 100) % 16 < 2}
    obj = linearObject(1.0, 15, 20, 0.4, 0.3, 100, 320)
    obj['snr'] = lambda f: 60.0 if int(f) in flashes else 0.0
    config.mf_clip = 10.0
    config.mf_saturation_level = 10**6
    _, detections = runDetector(_SynthHandle([obj], n_stars=4), config)

    assert len(detections) == 1
    cent = detections[0][2]
    assert np.median(truthError(cent[np.isin(np.round(cent[:, 0]).astype(int), list(flashes))], obj)) < 0.3
