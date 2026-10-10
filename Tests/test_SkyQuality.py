"""
Tests of the sky quality from the stars of the chunks of frames (RMS.Routines.SkyQuality): clear sky is clear
everywhere, a region of dimmed stars or of missing stars is clouded in its chunks, a chunk with too few stars is
clouded everywhere, and the measurements of detections on clouded sky are removed.
"""

import datetime

import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Formats.Platepar import Platepar
from RMS.Routines.SkyQuality import CalibrationError, SkyQualityMap, filterClouded


# Size of the image, frames per second and frames per chunk of the synthetic inputs
WIDTH, HEIGHT = 512, 512
FPS = 25.0
CHUNK_FRAMES = 128
N_CHUNKS = 12
BEGIN = datetime.datetime(2025, 12, 25, 10, 0, 0)


def _chunkName(i):
    """ FF name of the chunk i (its first frame). """

    t = BEGIN + datetime.timedelta(seconds=i*CHUNK_FRAMES/FPS)
    return 'FF_XX0001_{:s}_{:03d}_0000000.fits'.format(t.strftime('%Y%m%d_%H%M%S'), t.microsecond//1000)


def _skyMap(n_stars=500, cloud=None, opaque=None, few_stars_chunk=None, seed=0, skipped=(), total_frames=None):
    """ Sky quality of synthetic chunks with uniformly distributed matched stars with residuals of 0.05 mag.

    Keyword arguments:
        n_stars: [int] Matched stars per chunk.
        cloud: [tuple] (x, y, radius, chunks, dimming): the stars in the circle are fainter by the dimming (mag) in
            the given chunks.
        opaque: [tuple] (x, y, radius, chunks): no stars in the circle in the given chunks.
        few_stars_chunk: [int] A chunk with only 20 stars.
        skipped: [tuple] Chunks left out of the star list (e.g. skipped by the star extraction).
        total_frames: [int] Number of frames of the input.
    """

    rng = np.random.default_rng(seed)

    config = cr.Config()
    config.ff_min_stars = 100

    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # The same stars in every chunk (a fixed field), of which 10% are randomly missed in a chunk
    field_x, field_y = rng.uniform(0, WIDTH, n_stars), rng.uniform(0, HEIGHT, n_stars)

    star_list, matched = [], []
    for i in range(N_CHUNKS):

        found = rng.random(n_stars) > 0.1
        if i == few_stars_chunk:
            found[np.nonzero(found)[0][20:]] = False
        x, y = field_x[found], field_y[found]
        n = len(x)
        res = rng.normal(0, 0.05, n)

        if (cloud is not None) and (i in cloud[3]):
            res[np.hypot(x - cloud[0], y - cloud[1]) < cloud[2]] += cloud[4]

        keep = np.ones(n, dtype=bool)
        if (opaque is not None) and (i in opaque[3]):
            keep = np.hypot(x - opaque[0], y - opaque[1]) >= opaque[2]

        # The stars as from the star extraction, (y, x, intensity, amplitude, fwhm, background, snr, saturated)
        stars = [(yy, xx, 1000.0, 100.0, 2.0, 500.0, 20.0, 0) for xx, yy in zip(x[keep], y[keep])]
        if i in skipped:
            continue
        star_list.append([_chunkName(i), stars])
        matched.append(np.c_[x[keep], y[keep], res[keep]])

    return SkyQualityMap(star_list, platepar, config, BEGIN, FPS, CHUNK_FRAMES, matched=matched,
                         total_frames=total_frames)


def test_clear_sky_is_clear():
    """ Without clouds, the sky is clear everywhere and in every chunk. """

    sky_map = _skyMap()

    assert sky_map.clear.all()
    assert not sky_map.cloudMask(0, N_CHUNKS*CHUNK_FRAMES - 1, HEIGHT//2, WIDTH//2, bin_factor=2).any()


def test_dimmed_stars_are_clouded():
    """ Stars dimmed by 0.5 mag in a region are a cloud: clouded in its chunks and the neighbouring ones (the cloud
        moves during a chunk), clear elsewhere and at other times.
    """

    sky_map = _skyMap(cloud=(150, 300, 80, (5, 6), 0.5))

    frame_in = 5*CHUNK_FRAMES + 10
    assert not sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([frame_in]))[0]

    # The neighbouring chunks are not clear, the farther ones are
    assert not sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([4*CHUNK_FRAMES + 10]))[0]
    assert sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([1*CHUNK_FRAMES]))[0]
    assert sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([10*CHUNK_FRAMES]))[0]

    # Far from the cloud, the sky is clear
    assert sky_map.isClear(np.array([450.0]), np.array([50.0]), np.array([frame_in]))[0]

    # The search mask of a block covering the cloud, 2x2 binned
    mask = sky_map.cloudMask(5*CHUNK_FRAMES, 6*CHUNK_FRAMES, HEIGHT//2, WIDTH//2, bin_factor=2)
    assert mask[300//2, 150//2] and not mask[50//2, 450//2]


def test_small_dimming_is_clear():
    """ A dimming below mf_cloud_max_offset (0.15 mag) is not a cloud. """

    sky_map = _skyMap(cloud=(150, 300, 80, (5, 6), 0.05))
    assert sky_map.clear.all()


def test_uniform_haze_is_clear():
    """ A haze dimming the whole image uniformly is calibrated by the photometric offset of the chunk, it is not a
        cloud.
    """

    sky_map = _skyMap(cloud=(WIDTH/2, HEIGHT/2, 2*WIDTH, (5, 6), 0.5))
    assert sky_map.clear.all()


def test_missing_stars_are_clouded():
    """ A region without stars (an opaque cloud) is clouded. """

    sky_map = _skyMap(opaque=(300, 200, 100, (7,)))
    assert not sky_map.isClear(np.array([300.0]), np.array([200.0]), np.array([7*CHUNK_FRAMES + 50]))[0]
    assert sky_map.isClear(np.array([60.0]), np.array([460.0]), np.array([7*CHUNK_FRAMES + 50]))[0]


def test_chunk_with_few_stars_is_clouded():
    """ A chunk with fewer stars than ff_min_stars is clouded everywhere. """

    sky_map = _skyMap(few_stars_chunk=3)
    assert not sky_map.clear[3].any()
    assert sky_map.clear[[0, 1, 5, 6]].all()


def test_filter_clouded_measurements():
    """ The measurements of a detection on clouded sky are removed, and a detection left with too few
        measurements is removed.
    """

    sky_map = _skyMap(cloud=(150, 300, 80, (5, 6), 0.5))

    # A detection crossing the cloud: 100 measurements from x = 0 to 400 at y = 300 during chunks 5 and 6, and one
    #   far from it
    frames = np.linspace(5*CHUNK_FRAMES, 7*CHUNK_FRAMES - 1, 100)
    crossing = np.c_[frames, np.linspace(0, 400, 100), np.full(100, 300.0), np.ones(100)]
    clear = np.c_[frames, np.linspace(380, 480, 100), np.full(100, 40.0), np.ones(100)]

    kept, n_rows, n_dets = filterClouded([[0.0, 0.0, crossing], [0.0, 0.0, clear]], sky_map, min_rows=6)
    assert n_dets == 0
    assert len(kept[1][2]) == 100
    assert 0 < n_rows < 100
    assert np.all(np.abs(kept[0][2][:, 1] - 150) > 60)

    # With a higher minimum number of measurements, the crossing detection is removed
    kept, n_rows, n_dets = filterClouded([[0.0, 0.0, crossing], [0.0, 0.0, clear]], sky_map, min_rows=99)
    assert n_dets == 1 and len(kept) == 1


@pytest.mark.parametrize('frame', [-50, 10**6])
def test_frames_outside_the_chunks(frame):
    """ Frames before the first and after the last chunk use the nearest chunk. """

    sky_map = _skyMap()
    assert sky_map.isClear(np.array([200.0]), np.array([200.0]), np.array([frame]))[0]


def test_missing_chunks_are_clouded():
    """ Chunks missing from the star list (at the beginning, in the middle and at the end of the input) are
        clouded, and the frames of the chunks present are clear.
    """

    sky_map = _skyMap(skipped=(0, 5, 11), total_frames=N_CHUNKS*CHUNK_FRAMES)
    assert len(sky_map.chunk_first) == N_CHUNKS

    x, y = np.array([200.0]), np.array([200.0])
    for chunk in (0, 5, 11):
        assert not sky_map.isClear(x, y, np.array([chunk*CHUNK_FRAMES + 60]))[0]
    assert sky_map.isClear(x, y, np.array([8*CHUNK_FRAMES + 60]))[0]

    # Every other chunk missing: with the chunk size given, the missing ones are clouded
    sky_map = _skyMap(skipped=(1, 3, 5, 7, 9, 11), total_frames=N_CHUNKS*CHUNK_FRAMES)
    assert len(sky_map.chunk_first) == N_CHUNKS
    assert not sky_map.isClear(x, y, np.array([3*CHUNK_FRAMES + 60]))[0]

    # Without the number of frames, the frames after the last chunk belong to it
    sky_map = _skyMap(skipped=(11,))
    assert len(sky_map.chunk_first) == N_CHUNKS - 1


def test_no_catalog_is_unknown(tmp_path):
    """ Without the star catalog the sky quality is not known (CalibrationError), rather than clouded. """

    config = cr.Config()
    config.star_catalog_path = str(tmp_path)
    config.star_catalog_file = 'missing_catalog.bin'

    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    stars = [(100.0, 100.0, 1000.0, 100.0, 2.0, 500.0, 20.0, 0)]*200
    with pytest.raises(CalibrationError):
        SkyQualityMap([[_chunkName(i), stars] for i in range(3)], platepar, config, BEGIN, FPS, CHUNK_FRAMES)
