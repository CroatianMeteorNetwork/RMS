"""
Tests of the sky quality from the stars of the chunks of frames (RMS.Routines.SkyQuality): clear sky is clear
everywhere, a region of dimmed stars or of missing stars is clouded in its chunks, a chunk with too few stars is
clouded everywhere, and the measurements of detections on clouded sky are removed.
"""

import os
import datetime

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.Routines.SkyQuality as sq
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.Astrometry.Conversions import datetime2JD
from RMS.Formats.Platepar import Platepar
from RMS.Routines.SkyQuality import (CalibrationError, SkyQualityMap, SkyReference, chunkFirstFrames,
                                     filterClouded)


# Size of the image, frames per second and frames per chunk of the synthetic inputs
WIDTH, HEIGHT = 512, 512
FPS = 25.0
CHUNK_FRAMES = 128
N_CHUNKS = 12
BEGIN = datetime.datetime(2025, 12, 25, 10, 0, 0)


def _chunkName(i, begin=BEGIN):
    """ FF name of the chunk i (its first frame) of an input beginning at begin. """

    t = begin + datetime.timedelta(seconds=i*CHUNK_FRAMES/FPS)
    return 'FF_XX0001_{:s}_{:03d}_0000000.fits'.format(t.strftime('%Y%m%d_%H%M%S'), t.microsecond//1000)


def _skyMap(n_stars=500, cloud=None, opaque=None, few_stars_chunk=None, seed=0, skipped=(), total_frames=None,
            begin=BEGIN, reference=None, n_chunks=N_CHUNKS):
    """ Sky quality of synthetic chunks with uniformly distributed matched stars with residuals of 0.05 mag.

    Keyword arguments:
        n_stars: [int] Matched stars per chunk.
        cloud: [tuple] (x, y, radius, chunks, dimming): the stars in the circle are fainter by the dimming (mag) in
            the given chunks.
        opaque: [tuple] (x, y, radius, chunks): no stars in the circle in the given chunks.
        few_stars_chunk: [int] A chunk with only 20 stars.
        skipped: [tuple] Chunks left out of the star list (e.g. skipped by the star extraction).
        total_frames: [int] Number of frames of the input.
        begin: [datetime] Beginning of the input.
        reference: [SkyReference] Clear-sky reference of the camera.
        n_chunks: [int] Number of chunks.
    """

    rng = np.random.default_rng(seed)

    # The chunks need ff_min_stars stars to be usable
    config = cr.Config()
    config.ff_min_stars = 100

    # Only the size of the image is used, as the matches are given
    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # The same stars in every chunk (a fixed field, the same in every input), of which 10% are randomly missed in a
    #   chunk
    field = np.random.default_rng(100)
    field_x, field_y = field.uniform(0, WIDTH, n_stars), field.uniform(0, HEIGHT, n_stars)

    star_list, matched = [], []
    for i in range(n_chunks):

        # The stars found in this chunk (only 20 in the chunk with few stars), with their index as the key
        found = rng.random(n_stars) > 0.1
        if i == few_stars_chunk:
            found[np.nonzero(found)[0][20:]] = False
        x, y, keys = field_x[found], field_y[found], np.nonzero(found)[0].astype(np.float64)
        n = len(x)

        # Their residuals: the scatter of a star from chunk to chunk on clear sky, plus the dimming of the cloud
        res = rng.normal(0, 0.05, n)
        if (cloud is not None) and (i in cloud[3]):
            res[np.hypot(x - cloud[0], y - cloud[1]) < cloud[2]] += cloud[4]

        # The stars behind an opaque cloud are not found
        keep = np.ones(n, dtype=bool)
        if (opaque is not None) and (i in opaque[3]):
            keep = np.hypot(x - opaque[0], y - opaque[1]) >= opaque[2]

        # The stars as from the star extraction, (y, x, intensity, amplitude, fwhm, background, snr, saturated)
        stars = [(yy, xx, 1000.0, 100.0, 2.0, 500.0, 20.0, 0) for xx, yy in zip(x[keep], y[keep])]

        # The skipped chunks are not in the star list
        if i in skipped:
            continue
        star_list.append([_chunkName(i, begin), stars])
        matched.append(np.c_[x[keep], y[keep], res[keep], keys[keep]])

    sky_map = SkyQualityMap(star_list, platepar, config, begin, FPS, CHUNK_FRAMES, matched=matched,
                            total_frames=total_frames, reference=reference)

    # For building other maps of the same stars
    sky_map.platepar_for_test, sky_map.matched_for_test = platepar, matched

    return sky_map


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

    # In the cloud, during its chunks
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

    # The first, a middle and the last chunk missing: they are added back as clouded chunks
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

    # The frames after the last full chunk, fewer than a chunk (which the star extraction doesn't use), belong to
    #   the last chunk, they are not a missing chunk
    sky_map = _skyMap(total_frames=N_CHUNKS*CHUNK_FRAMES + 20)
    assert len(sky_map.chunk_first) == N_CHUNKS
    assert sky_map.isClear(x, y, np.array([N_CHUNKS*CHUNK_FRAMES + 10]))[0]
    assert [row['stars'] for row in sky_map.conditions()].count(0) == 0

    # A whole chunk missing at the end is clouded, and the frames after it belong to it
    sky_map = _skyMap(total_frames=(N_CHUNKS + 1)*CHUNK_FRAMES + 20)
    assert len(sky_map.chunk_first) == N_CHUNKS + 1
    for frame in (N_CHUNKS*CHUNK_FRAMES + 10, (N_CHUNKS + 1)*CHUNK_FRAMES + 10):
        assert not sky_map.isClear(x, y, np.array([frame]))[0]

    # Without the number of frames, the frames after the last chunk belong to it
    sky_map = _skyMap(skipped=(11,))
    assert len(sky_map.chunk_first) == N_CHUNKS - 1


def test_no_catalog_is_unknown(tmp_path):
    """ Without the star catalog the sky quality is not known (CalibrationError), rather than clouded. """

    # A catalog which doesn't exist
    config = cr.Config()
    config.star_catalog_path = str(tmp_path)
    config.star_catalog_file = 'missing_catalog.bin'

    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # Any stars: the catalog is read before they are matched
    stars = [(100.0, 100.0, 1000.0, 100.0, 2.0, 500.0, 20.0, 0)]*200
    with pytest.raises(CalibrationError):
        SkyQualityMap([[_chunkName(i), stars] for i in range(3)], platepar, config, BEGIN, FPS, CHUNK_FRAMES)


class _GapInput(object):
    """ An input with timestamped frames and a gap of 10 s in the recording before the chunk 6. """

    def __init__(self):
        self.chunk_frames = CHUNK_FRAMES
        self.total_frames = N_CHUNKS*CHUNK_FRAMES
        self.fps = FPS
        self.beginning_datetime = BEGIN

    def currentFrameTime(self, frame_no=None, dt_obj=False):

        # The frames from the chunk 6 on are 10 s later than the frame rate would put them
        gap = 10.0 if frame_no >= 6*CHUNK_FRAMES else 0.0
        return BEGIN + datetime.timedelta(seconds=frame_no/FPS + gap)


def test_first_frames_after_a_recording_gap():
    """ The chunks after a gap in the recording are at their frames, not where their times would put them, and
        they are clear.
    """

    # The chunks named by their times (10 s later from the chunk 6 on), and a chunk at a time of no chunk of the
    #   input
    img_handle = _GapInput()
    names = [_chunkName(i) for i in range(6)] + [_chunkName(i + 10.0*FPS/CHUNK_FRAMES) for i in range(6, N_CHUNKS)]
    star_list = [[name, []] for name in names] + [['FF_XX0001_20251225_120000_000_0000000.fits', []]]

    # The chunks are at their sequential frames, also after the gap
    first_frames = chunkFirstFrames(star_list, img_handle)
    assert first_frames[:N_CHUNKS] == [i*CHUNK_FRAMES for i in range(N_CHUNKS)]

    # A chunk at a time which is not the time of a chunk of the input is not known
    assert first_frames[-1] == -1

    # The sky quality with these first frames: no missing chunks after the gap, and clear sky there
    clear_map = _skyMap()
    stars = [[name, [(0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0)]*500] for name in names]
    sky_map = SkyQualityMap(stars, clear_map.platepar_for_test, clear_map.config, BEGIN, FPS, CHUNK_FRAMES,
                            matched=clear_map.matched_for_test, total_frames=N_CHUNKS*CHUNK_FRAMES,
                            first_frames=first_frames[:N_CHUNKS])
    assert len(sky_map.chunk_first) == N_CHUNKS
    assert sky_map.isClear(np.array([200.0]), np.array([200.0]), np.array([8*CHUNK_FRAMES + 10]))[0]

    # The last chunk missing: it is added after the last chunk present, by its time (after the gap)
    sky_map = SkyQualityMap(stars[:-1], clear_map.platepar_for_test, clear_map.config, BEGIN, FPS, CHUNK_FRAMES,
                            matched=clear_map.matched_for_test[:-1], total_frames=N_CHUNKS*CHUNK_FRAMES,
                            first_frames=first_frames[:N_CHUNKS - 1])
    assert len(sky_map.chunk_first) == N_CHUNKS
    assert sky_map.chunk_times[-1] == sky_map.chunk_times[-2] + datetime.timedelta(seconds=CHUNK_FRAMES/FPS)

    # The chunks before the gap missing: they are added from the beginning of the input
    sky_map = SkyQualityMap(stars[6:N_CHUNKS], clear_map.platepar_for_test, clear_map.config, BEGIN, FPS, CHUNK_FRAMES,
                            matched=clear_map.matched_for_test[6:], total_frames=N_CHUNKS*CHUNK_FRAMES,
                            first_frames=first_frames[6:N_CHUNKS])
    assert len(sky_map.chunk_first) == N_CHUNKS
    assert sky_map.chunk_times[0] == BEGIN


def _patchCalibration(monkeypatch, catalog, offset):
    """ Replace the catalog, the recalibration and the projection: the catalog stars are projected to their
        (ra, dec) as (x, y), shifted by offset px (a platepar off by that much).
    """

    # The recalibration always succeeds, with a platepar of the size of the image
    def recalibrate(platepar, ff_names, calstars, catalog_stars, config, **kwargs):
        pp = Platepar()
        pp.X_res, pp.Y_res = WIDTH, HEIGHT
        pp.auto_recalibrated = True
        return {ff_names[0]: pp}

    # The given catalog, the projection to (ra + offset, dec), all catalog stars near the field, and no extinction
    monkeypatch.setattr(sq.StarCatalog, 'readStarCatalog', lambda *args, **kwargs: (catalog, None, None))
    monkeypatch.setattr(sq, 'recalibratePlateparsForFF', recalibrate)
    monkeypatch.setattr(sq, 'raDecToXYPP', lambda ra, dec, jd, pp: (ra + offset, dec))
    monkeypatch.setattr(sq, 'catalogNearField', lambda catalog, pp, jds: np.ones(len(catalog), dtype=bool))
    monkeypatch.setattr(sq, 'extinctionCorrectionTrueToApparent', lambda mags, ra, dec, jd, pp: mags)


@pytest.mark.parametrize('offset, trusted', [(0.0, True), (5.0, False)])
def test_calibration_matching_few_stars_is_not_trusted(monkeypatch, offset, trusted):
    """ A calibration which matches dozens of the stars of the clearest chunk is used; one which matches only a
        few (e.g. a platepar off by a few pixels, accepted by the recalibration) leaves the sky quality unknown
        instead of making every chunk clouded.
    """

    # 300 stars at their catalog positions
    rng = np.random.default_rng(1)
    x, y = rng.uniform(20, WIDTH - 20, 300), rng.uniform(20, HEIGHT - 20, 300)
    catalog = np.c_[x, y, np.full(300, 8.0)]

    # 10 stars which the offset platepar still matches (catalog entries 5 px to their left, projected onto them)
    extra = np.c_[x[:10] - offset, y[:10]]
    catalog = np.vstack([catalog, np.c_[extra, np.full(10, 8.0)]]) if offset else catalog

    _patchCalibration(monkeypatch, catalog, offset)

    config = cr.Config()
    config.ff_min_stars = 100
    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # The same stars in 6 chunks
    stars = [(yy, xx, 1000.0, 100.0, 2.0, 500.0, 20.0, 0) for xx, yy in zip(x, y)]
    star_list = [[_chunkName(i), stars] for i in range(6)]

    # Without the offset nearly all stars match; with it, only the 10 extra ones
    if trusted:
        sky_map = SkyQualityMap(star_list, platepar, config, BEGIN, FPS, CHUNK_FRAMES)
        assert np.all(sky_map.n_matched > 250)
    else:
        with pytest.raises(CalibrationError, match='only 1[0-9] of the 300'):
            SkyQualityMap(star_list, platepar, config, BEGIN, FPS, CHUNK_FRAMES)


def _residuals(keys, value):
    """ The same visit value for all stars. """

    return np.full(len(keys), value), np.full(len(keys), 1.0)


def test_reference_order_and_reprocessing(tmp_path):
    """ The visits of the inputs are kept by the time of the observation: in any order of the processing, an input
        processed again replaces its visit, and only the latest MAX_VISITS visits are kept.
    """

    # Three stars, and the inputs of every 10 minutes (more than the visits which are kept)
    keys = np.array([11, 22, 33])
    times = [BEGIN + datetime.timedelta(minutes=10*i) for i in range(SkyReference.MAX_VISITS + 2)]

    reference = SkyReference(str(tmp_path), 'XX0001')

    # Out of order: the second input first
    reference.update(times[1], keys, *_residuals(keys, 0.2), 0.0, [])
    reference.update(times[0], keys, *_residuals(keys, 0.1), 0.0, [])
    values, _ = SkyReference(str(tmp_path), 'XX0001').lookup(keys)
    assert sorted(values[0][np.isfinite(values[0])]) == pytest.approx([0.1, 0.2])

    # Processed again: replaced, not added
    reference.update(times[1], keys, *_residuals(keys, 0.3), 0.0, [])
    values, _ = reference.lookup(keys)
    assert sorted(values[0][np.isfinite(values[0])]) == pytest.approx([0.1, 0.3])

    # Processed again with fewer stars and an unknown zero point: the stars and the zero point which are not in
    #   the new processing are removed too
    reference.update(times[1], keys[:1], *_residuals(keys[:1], 0.4), np.nan, [])
    values, _ = reference.lookup(keys)
    assert sorted(values[0][np.isfinite(values[0])]) == pytest.approx([0.1, 0.4])
    assert values[1][np.isfinite(values[1])] == pytest.approx([0.1])
    assert len(reference.zeroPoints()) == 1
    reference.update(times[1], keys, *_residuals(keys, 0.3), 0.0, [])

    # The earlier processing of an input can be left out
    values, _ = reference.lookup(keys, exclude_time=(times[1] - datetime.datetime(1970, 1, 1)).total_seconds())
    assert values[0][np.isfinite(values[0])] == pytest.approx([0.1])

    # Only the latest visits are kept
    for i, t in enumerate(times):
        reference.update(t, keys, *_residuals(keys, float(i)), 0.0, [])
    values, _ = reference.lookup(keys)
    assert sorted(values[0][np.isfinite(values[0])]) == pytest.approx(list(range(2, SkyReference.MAX_VISITS + 2)))


def test_conditions_log(tmp_path):
    """ The conditions of every chunk are appended to the log of the date of the chunk, with the time of the
        processing.
    """

    reference = SkyReference(str(tmp_path), 'XX0001')
    sky_map = _skyMap(cloud=(150, 300, 80, (5, 6), 0.5), reference=reference)
    sky_map.updateReference()

    # The log of the date of the input: the description, the header and a row per chunk
    path = tmp_path/'sky_conditions_XX0001_20251225.csv'
    lines = [line for line in path.read_text().splitlines() if not line.startswith('#')]
    header = lines[0].split(',')
    rows = [dict(zip(header, line.split(','))) for line in lines[1:]]
    assert len(rows) == N_CHUNKS

    # The time of the first chunk, the cloud in the chunk 5, and the transparency of clear sky
    assert rows[0]['chunk_utc'] == '2025-12-25T10:00:00.000'
    assert float(rows[5]['clear_fraction']) < 1.0 and float(rows[0]['clear_fraction']) == 1.0
    assert abs(float(rows[0]['transparency_mag'])) < 0.05


def test_stationary_cloud_found_with_reference(tmp_path):
    """ A cloud over the same region during a whole input: with the stars of that input alone it is the
        reference (not found), with the clear-sky reference of an earlier input it is found.
    """

    cloud = (150, 300, 80, tuple(range(N_CHUNKS)), 0.5)
    later = BEGIN + datetime.timedelta(minutes=10)

    # Alone
    sky_map = _skyMap(cloud=cloud, begin=later)
    assert sky_map.clear.all()

    # With the reference of an earlier, clear input
    reference = SkyReference(str(tmp_path), 'XX0001')
    _skyMap(reference=reference).updateReference()
    sky_map = _skyMap(cloud=cloud, begin=later, reference=SkyReference(str(tmp_path), 'XX0001'))
    assert not sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([5*CHUNK_FRAMES]))[0]
    assert sky_map.isClear(np.array([450.0]), np.array([60.0]), np.array([5*CHUNK_FRAMES]))[0]

    # A dimming of 0.2 mag (just above the limit) with a single earlier input: the input itself doesn't enter the
    #   references of the stars which have an earlier one
    cloud = (150, 300, 80, tuple(range(N_CHUNKS)), 0.2)
    sky_map = _skyMap(cloud=cloud, begin=later, reference=SkyReference(str(tmp_path), 'XX0001'))
    assert not sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([5*CHUNK_FRAMES]))[0]


def test_missing_reliable_stars_are_clouded(monkeypatch, tmp_path):
    """ Where the stars which are reliably seen on clear sky are missing (an opaque cloud), the sky is clouded. """

    # 400 stars at their catalog positions, matched through the patched calibration
    rng = np.random.default_rng(3)
    x, y = rng.uniform(20, WIDTH - 20, 400), rng.uniform(20, HEIGHT - 20, 400)
    catalog = np.c_[x, y, np.full(400, 8.0)]
    _patchCalibration(monkeypatch, catalog, 0.0)

    config = cr.Config()
    config.ff_min_stars = 100
    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # The stars of every chunk, without the ones within 90 px of (300, 200) in the hidden chunk (or in all chunks)
    def starList(begin, hidden_chunk=None, hidden_always=False):
        out = []
        for i in range(N_CHUNKS):
            keep = np.ones(len(x), dtype=bool)
            if (i == hidden_chunk) or hidden_always:
                keep = np.hypot(x - 300, y - 200) > 90
            out.append([_chunkName(i, begin), [(yy, xx, 1000.0, 100.0, 2.0, 500.0, 20.0, 0)
                                               for xx, yy in zip(x[keep], y[keep])]])
        return out

    # A clear input makes the stars reliable, then an input with an opaque cloud in the chunk 7
    reference = SkyReference(str(tmp_path), 'XX0001')
    SkyQualityMap(starList(BEGIN), platepar, config, BEGIN, FPS, CHUNK_FRAMES, reference=reference).updateReference()

    later = BEGIN + datetime.timedelta(minutes=10)
    sky_map = SkyQualityMap(starList(later, hidden_chunk=7), platepar, config, later, FPS, CHUNK_FRAMES,
                            reference=SkyReference(str(tmp_path), 'XX0001'))
    assert sky_map.reliable_seen[0] == pytest.approx(1.0, abs=0.05)
    assert sky_map.reliable_seen[7] < 0.9
    assert not sky_map.isClear(np.array([300.0]), np.array([200.0]), np.array([7*CHUNK_FRAMES + 10]))[0]
    assert sky_map.isClear(np.array([60.0]), np.array([460.0]), np.array([7*CHUNK_FRAMES + 10]))[0]

    # An opaque cloud during the whole input: the stars behind it are not matched in it at all, but they are
    #   expected from the reference
    sky_map = SkyQualityMap(starList(later, hidden_always=True), platepar, config, later, FPS, CHUNK_FRAMES,
                            reference=SkyReference(str(tmp_path), 'XX0001'))
    assert not sky_map.isClear(np.array([300.0]), np.array([200.0]), np.array([7*CHUNK_FRAMES + 10]))[0]
    assert sky_map.isClear(np.array([60.0]), np.array([460.0]), np.array([7*CHUNK_FRAMES + 10]))[0]


def test_saturated_reliable_stars_are_seen(monkeypatch, tmp_path):
    """ Reliable stars which are extracted but not matched (saturated pixels, left out of the photometry) are
        seen: the region around them is not clouded.
    """

    # 400 stars at their catalog positions, matched through the patched calibration
    rng = np.random.default_rng(4)
    x, y = rng.uniform(20, WIDTH - 20, 400), rng.uniform(20, HEIGHT - 20, 400)
    catalog = np.c_[x, y, np.full(400, 8.0)]
    _patchCalibration(monkeypatch, catalog, 0.0)

    config = cr.Config()
    config.ff_min_stars = 100
    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # The stars of every chunk, the ones within 90 px of (300, 200) with saturated pixels in the saturated chunk
    def starList(begin, saturated_chunk=None):
        out = []
        for i in range(N_CHUNKS):
            saturated = (np.hypot(x - 300, y - 200) < 90) & (i == saturated_chunk)
            out.append([_chunkName(i, begin), [(yy, xx, 1000.0, 100.0, 2.0, 500.0, 20.0, int(sat))
                                               for xx, yy, sat in zip(x, y, saturated)]])
        return out

    # A clear input makes the stars reliable, then an input with the saturated stars in the chunk 7
    reference = SkyReference(str(tmp_path), 'XX0001')
    SkyQualityMap(starList(BEGIN), platepar, config, BEGIN, FPS, CHUNK_FRAMES, reference=reference).updateReference()

    later = BEGIN + datetime.timedelta(minutes=10)
    sky_map = SkyQualityMap(starList(later, saturated_chunk=7), platepar, config, later, FPS, CHUNK_FRAMES,
                            reference=SkyReference(str(tmp_path), 'XX0001'))
    assert sky_map.reliable_seen[7] == pytest.approx(1.0, abs=0.05)
    assert sky_map.isClear(np.array([300.0]), np.array([200.0]), np.array([7*CHUNK_FRAMES + 10]))[0]


def test_short_input(tmp_path):
    """ An input with fewer chunks than MIN_STAR_OBS (e.g. 30 s of frames): alone, its own stars give the references
        (a single chunk doesn't, its sky quality is not known); with the reference of an earlier input, a cloud in the
        short input is found.
    """

    # A single chunk: no star has a reference
    with pytest.raises(CalibrationError):
        _skyMap(n_chunks=1)

    # Three chunks: the stars seen in all of them have references, the clear sky is clear
    assert _skyMap(n_chunks=3).clear.all()

    # With the reference of an earlier input, a cloud in the short input is found
    reference = SkyReference(str(tmp_path), 'XX0001')
    _skyMap(reference=reference).updateReference()
    later = BEGIN + datetime.timedelta(minutes=10)
    sky_map = _skyMap(n_chunks=3, begin=later, cloud=(150, 300, 80, (0, 1, 2), 0.5),
                      reference=SkyReference(str(tmp_path), 'XX0001'))
    assert not sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([CHUNK_FRAMES]))[0]
    assert sky_map.isClear(np.array([450.0]), np.array([60.0]), np.array([CHUNK_FRAMES]))[0]


def test_short_inputs_build_the_reference(tmp_path):
    """ A camera with only short inputs (two chunks each) builds its reference from them, so that a cloud over the
        same region during a whole later short input is found (which its own stars alone can't show). """

    # Ten clear inputs of two chunks, 30 s apart, add their stars to the reference
    for k in range(10):
        begin = BEGIN + datetime.timedelta(seconds=30*k)
        _skyMap(n_chunks=2, begin=begin, seed=k, reference=SkyReference(str(tmp_path), 'XX0001')).updateReference()
    reference = SkyReference(str(tmp_path), 'XX0001')
    assert len(reference.keys) > 400
    assert np.median(np.isfinite(reference.values).sum(axis=1)) >= 5

    # A later short input under a cloud in both its chunks
    later = BEGIN + datetime.timedelta(minutes=10)
    sky_map = _skyMap(n_chunks=2, begin=later, cloud=(150, 300, 80, (0, 1), 0.5), seed=50,
                      reference=SkyReference(str(tmp_path), 'XX0001'))
    assert not sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([CHUNK_FRAMES]))[0]
    assert sky_map.isClear(np.array([450.0]), np.array([60.0]), np.array([CHUNK_FRAMES]))[0]


def test_stars_entering_during_the_input(monkeypatch):
    """ With the sky moving across the image during the input (twice the image width), the stars which are in the
        image only at its beginning or its end are matched too.
    """

    # Stars over three image widths in x
    rng = np.random.default_rng(4)
    n = 1500
    ra, dec = rng.uniform(-WIDTH, 2*WIDTH, n), rng.uniform(20, HEIGHT - 20, n)
    catalog = np.c_[ra, dec, np.full(n, 8.0)]
    _patchCalibration(monkeypatch, catalog, 0.0)

    # The sky moves by 2*WIDTH px over the input, so a star at ra is at x = ra - shift(t)
    duration = N_CHUNKS*CHUNK_FRAMES/FPS
    jd0 = datetime2JD(BEGIN)

    def project(r, d, jd, pp):
        return r - 2*WIDTH*(jd - jd0)*86400/duration + WIDTH/2, d

    monkeypatch.setattr(sq, 'raDecToXYPP', project)

    config = cr.Config()
    config.ff_min_stars = 50
    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # The stars in the image in every chunk, at their positions at the middle of the chunk
    star_list = []
    for i in range(N_CHUNKS):
        t = BEGIN + datetime.timedelta(seconds=i*CHUNK_FRAMES/FPS)
        x, y = project(ra, dec, datetime2JD(t) + CHUNK_FRAMES/(2*FPS)/86400, None)
        inside = (x > 15) & (x < WIDTH - 15)
        star_list.append([_chunkName(i), [(yy, xx, 1000.0, 100.0, 2.0, 500.0, 20.0, 0)
                                          for xx, yy in zip(x[inside], y[inside])]])

    # The first and the last chunk see stars which are far from the field in the middle of the input
    sky_map = SkyQualityMap(star_list, platepar, config, BEGIN, FPS, CHUNK_FRAMES)
    assert sky_map.n_matched[0] > 0.8*len(star_list[0][1])
    assert sky_map.n_matched[-1] > 0.8*len(star_list[-1][1])


def test_catalog_near_field():
    """ Every catalog star which is in the image at some time of an input is selected near the field, and most of
        the sky is not.
    """

    # A template platepar of the repository
    platepar = Platepar()
    platepar.read(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'share',
                               'platepar_templates', 'template_generic_720p_6mm.cal'))

    # Stars over the whole sky
    rng = np.random.default_rng(5)
    n = 20000
    catalog = np.c_[rng.uniform(0, 360, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n))), np.full(n, 5.0)]

    # An input of two hours (the sky moves by 30 deg), the epochs of the selection over it
    jd0 = datetime2JD(BEGIN)
    jds = list(jd0 + np.linspace(0, 2/24, 21))
    near = sq.catalogNearField(catalog, platepar, jds)

    # The stars in the image at any of many times of the input
    inside = np.zeros(n, dtype=bool)
    for jd in jd0 + np.linspace(0, 2/24, 61):
        x, y = raDecToXYPP(catalog[:, 0], catalog[:, 1], jd, platepar)
        inside |= (x >= 0) & (x < platepar.X_res) & (y >= 0) & (y < platepar.Y_res)

    # All stars in the image are near the field, and most of the sky is not
    assert inside.sum() > 100
    assert np.all(near[inside])
    assert near.mean() < 0.5


def _writeConditions(path, rows):
    """ Write a log of the conditions with the given rows (chunk_utc, input, clear_fraction, processed_utc). """

    with open(path, 'w') as f:
        f.write('# Sky conditions\n')
        f.write(','.join(sq.CONDITION_COLUMNS) + ',processed_utc\n')
        for chunk_utc, input_name, clear, processed in rows:
            f.write('{:s},{:s},500,300,{:.3f},0.010,0.950,{:s}\n'.format(chunk_utc, input_name, clear, processed))


def test_read_conditions(tmp_path):
    """ The conditions of the given inputs are read from the logs of the dates of the given time: the rows of the
        newest processing of every input (an input processed again has newer rows, and the rows of its earlier
        processing are not taken even for chunks the newer one doesn't have), sorted by time.
    """

    _writeConditions(tmp_path/'sky_conditions_XX0001_20251225.csv', [
        ('2025-12-25T23:59:54.000', '/data/a.vid', 1.0, '2025-12-26T01:00:00'),
        ('2025-12-25T23:59:58.000', '/data/a.vid', 1.0, '2025-12-26T01:00:00'),
        ('2025-12-25T23:59:58.000', '/data/a.vid', 0.5, '2025-12-26T02:00:00'),
        ('2025-12-25T23:59:54.000', '/data/b.vid', 0.0, '2025-12-26T01:00:00')])
    _writeConditions(tmp_path/'sky_conditions_XX0001_20251226.csv', [
        ('2025-12-26T00:00:02.000', '/data/a.vid', 1.0, '2025-12-26T02:00:00')])
    _writeConditions(tmp_path/'sky_conditions_XX0001_20251227.csv', [
        ('2025-12-27T00:00:02.000', '/data/a.vid', 1.0, '2025-12-28T02:00:00')])
    _writeConditions(tmp_path/'sky_conditions_XX0002_20251226.csv', [
        ('2025-12-26T00:00:06.000', '/data/a.vid', 1.0, '2025-12-26T01:00:00')])

    start, end = datetime.datetime(2025, 12, 25, 12, 0, 0), datetime.datetime(2025, 12, 26, 12, 0, 0)
    rows = sq.readConditions(str(tmp_path), 'XX0001', inputs={'/data/a.vid'}, start=start, end=end)
    assert [row['chunk_utc'] for row in rows] == ['2025-12-25T23:59:58.000', '2025-12-26T00:00:02.000']
    assert rows[0]['clear_fraction'] == 0.5
    assert rows[0]['time'] == datetime.datetime(2025, 12, 25, 23, 59, 58)

    # All inputs of the camera in the time
    assert len(sq.readConditions(str(tmp_path), 'XX0001', start=start, end=end)) == 3


def test_conditions_log_quotes_the_input(tmp_path):
    """ The input is logged by its path, which is quoted if it has a comma. """

    reference = SkyReference(str(tmp_path), 'XX0001')
    sky_map = _skyMap(reference=reference)
    sky_map.input_name = '/data/night, 1/a.vid'
    sky_map.updateReference()

    rows = sq.readConditions(str(tmp_path), 'XX0001')
    assert len(rows) == N_CHUNKS and all(row['input'] == '/data/night, 1/a.vid' for row in rows)


def test_plot_sky_conditions(tmp_path):
    """ The plot of the conditions is saved, and the time is split into clear, partly clouded, clouded and not
        searched.
    """

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    rows = []
    for i in range(100):
        clear = 1.0 if i < 60 else (0.4 if i < 80 else 0.0)
        rows.append({'time': begin + datetime.timedelta(seconds=4*i), 'clear_fraction': clear,
                     'transparency_mag': 0.02 if clear == 1.0 else 0.3, 'reliable_seen': clear,
                     'stars': 500*clear + 10, 'matched': 300*clear + 5})

    # An input which was not searched for 100 s after a gap
    ns_times = [begin + datetime.timedelta(seconds=500 + 5*k) for k in range(20)]
    not_searched = [(ns_times[0], ns_times[-1] + datetime.timedelta(seconds=5), ns_times, [20]*20)]

    plot_path = str(tmp_path/'conditions.png')
    summary = sq.plotSkyConditions(rows, plot_path, 'Test', not_searched=not_searched,
                                   detection_times=[begin + datetime.timedelta(seconds=30)], removed_detections=2,
                                   cloud_max_offset=0.15, min_matched=30)

    assert os.path.getsize(plot_path) > 10000
    assert summary['observed_h'] == pytest.approx(500/3600.0)
    assert summary['clear'] == pytest.approx(240/500.0)
    assert summary['partly'] == pytest.approx(80/500.0)
    assert summary['clouded'] == pytest.approx(80/500.0)
    assert summary['not_searched'] == pytest.approx(100/500.0)
    assert summary['median_dimming'] == pytest.approx(0.02)


def test_plot_sky_conditions_inputs_at_the_same_times(tmp_path):
    """ The chunks of two inputs at the same times (e.g. a recording which was copied) don't break the plot. """

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    rows = [{'time': begin + datetime.timedelta(seconds=4*i), 'input': name, 'clear_fraction': 1.0,
             'transparency_mag': 0.0, 'reliable_seen': 1.0, 'stars': 500, 'matched': 300}
            for i in range(50) for name in ('a.vid', 'b.vid')]
    rows.sort(key=lambda row: row['time'])

    summary = sq.plotSkyConditions(rows, str(tmp_path/'conditions.png'), 'Test')
    assert summary['clear'] == pytest.approx(1.0)
    assert summary['observed_h'] == pytest.approx(200/3600.0)


def test_plot_sky_conditions_overlapping_inputs(tmp_path):
    """ The time of overlapping inputs is counted once, and the worse state of the sky holds where they
        overlap. The running median of the dimming is over a minute of every input.
    """

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    rows = []
    for name, offset, clear in [('a.vid', 0, 1.0), ('b.vid', 200, 0.0)]:
        rows += [{'time': begin + datetime.timedelta(seconds=offset + 4*i), 'input': name, 'clear_fraction': clear,
                  'transparency_mag': 0.0, 'reliable_seen': 1.0, 'stars': 500, 'matched': 300} for i in range(100)]
    rows.sort(key=lambda row: row['time'])

    # a.vid from 0 to 400 s, b.vid (clouded) from 200 to 600 s
    summary = sq.plotSkyConditions(rows, str(tmp_path/'conditions.png'), 'Test')
    assert summary['observed_h'] == pytest.approx(600/3600.0)
    assert summary['clear'] == pytest.approx(200/600.0)
    assert summary['clouded'] == pytest.approx(400/600.0)


def test_running_median():
    """ The running median is over the time window, of the finite values only. """

    times = [0, 10, 20, 30, 100]
    values = [1.0, 2.0, np.nan, 4.0, 5.0]
    assert np.allclose(sq._runningMedian(times, values, 15.0), [1.5, 1.5, 3.0, 4.0, 5.0])
    assert np.isnan(sq._runningMedian([0], [np.nan], 30.0)[0])


def test_plot_sky_conditions_without_conditions(tmp_path):
    """ A night of inputs which were all not searched (e.g. clouded all night) has a plot of their stars. """

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    ns_times = [begin + datetime.timedelta(seconds=5*k) for k in range(20)]
    not_searched = [(ns_times[0], ns_times[-1] + datetime.timedelta(seconds=5), ns_times, [10]*20)]

    summary = sq.plotSkyConditions([], str(tmp_path/'conditions.png'), 'Test', not_searched=not_searched,
                                   cloud_max_offset=0.15, min_matched=30)
    assert os.path.isfile(str(tmp_path/'conditions.png'))
    assert summary['not_searched'] == pytest.approx(1.0)


def test_read_conditions_skips_rows_which_cant_be_read(tmp_path):
    """ A row which was not written completely or has values which can't be read is skipped, and a log with bytes
        which are not UTF-8 is read.
    """

    _writeConditions(tmp_path/'sky_conditions_XX0001_20251225.csv', [
        ('2025-12-25T23:59:50.000', 'a.vid', 1.0, '2025-12-26T01:00:00')])
    with open(str(tmp_path/'sky_conditions_XX0001_20251225.csv'), 'a') as f:
        f.write('2025-12-25T23:59:54.000,a.vid,500,300,1.000,0.0\n')
        f.write('2025-12-25T23:59:58.000,a.vid,500,300,x,0.010,0.950,2025-12-26T01:00:00\n')
        f.write('not a time,a.vid,500,300,1.000,0.010,0.950,2025-12-26T01:00:00\n')
        f.write(','.join(sq.CONDITION_COLUMNS) + ',processed_utc\n')
    with open(str(tmp_path/'sky_conditions_XX0001_20251225.csv'), 'ab') as f:
        f.write(b'2025-12-25T23:59:59.000,a\xff.vid,500,300,1.000,0.010,0.950,2025-12-26T01:00:00\n')

    rows = sq.readConditions(str(tmp_path), 'XX0001', inputs={'a.vid'})
    assert [row['chunk_utc'] for row in rows] == ['2025-12-25T23:59:50.000']


def test_plot_sky_conditions_inputs_of_different_cadence(tmp_path):
    """ The time of the chunks of every input is the spacing of its own chunks. """

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    rows = [{'time': begin + datetime.timedelta(seconds=4*i), 'input': 'a.vid', 'clear_fraction': 1.0,
             'transparency_mag': 0.0, 'reliable_seen': 1.0, 'stars': 500, 'matched': 300} for i in range(100)]
    rows += [{'time': begin + datetime.timedelta(seconds=1000 + 20*i), 'input': 'b.vid', 'clear_fraction': 0.0,
              'transparency_mag': 0.0, 'reliable_seen': 1.0, 'stars': 500, 'matched': 300} for i in range(10)]
    rows.sort(key=lambda row: row['time'])

    # a.vid 400 s, b.vid 200 s (not 10 gaps of 20 s with chunks of 4 s)
    summary = sq.plotSkyConditions(rows, str(tmp_path/'conditions.png'), 'Test')
    assert summary['observed_h'] == pytest.approx(600/3600.0)
    assert summary['clouded'] == pytest.approx(200/600.0)


def test_plot_sky_conditions_input_of_a_few_chunks_with_a_pause(tmp_path):
    """ The time of the chunks of an input of a few chunks with a pause between them is not the pause. """

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    rows = [{'time': begin + datetime.timedelta(seconds=t), 'input': 'a.vid', 'clear_fraction': 1.0,
             'transparency_mag': 0.0, 'reliable_seen': 1.0, 'stars': 500, 'matched': 300} for t in (0, 4, 600)]

    summary = sq.plotSkyConditions(rows, str(tmp_path/'conditions.png'), 'Test')
    assert summary['observed_h'] == pytest.approx(12/3600.0)


def test_plot_sky_conditions_input_of_two_chunks_with_a_pause(tmp_path):
    """ The time of the chunks of an input of two chunks with a pause between them is the given one, or at most
        the one of the other inputs.
    """

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    def _rows(name, offsets):
        return [{'time': begin + datetime.timedelta(seconds=t), 'input': name, 'clear_fraction': 1.0,
                 'transparency_mag': 0.0, 'reliable_seen': 1.0, 'stars': 500, 'matched': 300} for t in offsets]

    summary = sq.plotSkyConditions(_rows('a.vid', (0, 600)), str(tmp_path/'conditions.png'), 'Test',
                                   chunk_seconds={'a.vid': 4.0})
    assert summary['observed_h'] == pytest.approx(8/3600.0)

    rows = sorted(_rows('a.vid', (0, 600)) + _rows('b.vid', (1000, 1004, 1008)), key=lambda row: row['time'])
    summary = sq.plotSkyConditions(rows, str(tmp_path/'conditions.png'), 'Test')
    assert summary['observed_h'] == pytest.approx(20/3600.0)


@pytest.mark.parametrize('given_chunks, clipped', [(True, False), (False, False), (True, True)])
def test_night_timeline_random(given_chunks, clipped):
    """ The time line of random nights (inputs of different times of their chunks, pauses, overlapping inputs and
        inputs which were not searched) agrees with the state of every moment counted on a fine grid: a chunk
        lasts until the next chunk of its input (for the time of a chunk before a pause of more than three
        chunks), and the worse state holds where they overlap.
    """

    rng = np.random.default_rng(1)
    order = ['clouded', 'not_searched', 'partly', 'clear']
    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    q = 0.25

    for _ in range(60):
        rows, chunk_seconds, intervals = [], {}, []
        for k in range(int(rng.integers(1, 4))):
            name = 'in{:d}.vid'.format(k)
            chunk = q*int(rng.integers(8, 33))
            t = q*int(rng.integers(0, 2000))
            n = int(rng.integers(3, 40))
            offsets = []
            for i in range(n):
                offsets.append(t)
                t += chunk*(int(rng.integers(5, 50)) if (i >= 2) and (rng.random() < 0.1) else 1)
            chunk_seconds[name] = chunk
            for i, t in enumerate(offsets):
                clear = float(rng.choice([1.0, 0.5, 0.0]))
                rows.append({'time': begin + datetime.timedelta(seconds=t), 'input': name, 'clear_fraction': clear,
                             'transparency_mag': 0.0, 'reliable_seen': 1.0, 'stars': 500, 'matched': 300})
                nxt = offsets[i + 1] if i + 1 < len(offsets) else None
                end = nxt if (nxt is not None) and (nxt - t <= 3*chunk) else t + chunk
                intervals.append(('clear' if clear == 1.0 else ('partly' if clear > 0 else 'clouded'), t, end))
        not_searched = []
        for _ in range(int(rng.integers(0, 3))):
            a = q*int(rng.integers(0, 3000))
            b = a + q*int(rng.integers(4, 400))
            not_searched.append((begin + datetime.timedelta(seconds=a), begin + datetime.timedelta(seconds=b), [], []))
            intervals.append(('not_searched', a, b))
        rows.sort(key=lambda row: (row['time'], row['input']))

        # The state of every moment on a grid of q s, the worst of the intervals which cover it, within the night
        top = int(max(end for _, _, end in intervals)/q) + 1
        grid = np.full(top, len(order))
        for state, a, b in intervals:
            sl = slice(int(round(a/q)), int(round(b/q)))
            grid[sl] = np.minimum(grid[sl], order.index(state))
        bounds = None
        if clipped:
            lo, hi = sorted(int(v) for v in rng.integers(0, top, 2))
            grid[:lo] = len(order)
            grid[hi:] = len(order)
            bounds = (begin + datetime.timedelta(seconds=lo*q), begin + datetime.timedelta(seconds=hi*q))
        observed = np.sum(grid < len(order))*q

        summary = sq.nightTimeline(rows, not_searched=not_searched,
                                   chunk_seconds=chunk_seconds if given_chunks else None, bounds=bounds)['summary']
        assert summary['observed_h']*3600 == pytest.approx(observed, abs=1e-6)
        for i, state in enumerate(order):
            assert summary[state]*summary['observed_h']*3600 == pytest.approx(np.sum(grid == i)*q, abs=1e-6)


def test_night_timeline_within_the_night():
    """ The time of the chunks is taken within the night: a chunk which begins before it counts from its
        beginning, and the last chunk only until its end.
    """

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    rows = [{'time': begin + datetime.timedelta(seconds=4*i), 'input': 'a.vid', 'clear_fraction': 1.0,
             'transparency_mag': 0.0, 'reliable_seen': 1.0, 'stars': 500, 'matched': 300} for i in range(10)]
    bounds = (begin + datetime.timedelta(seconds=2), begin + datetime.timedelta(seconds=38))

    summary = sq.nightTimeline(rows, chunk_seconds={'a.vid': 4.0}, bounds=bounds)['summary']
    assert summary['observed_h']*3600 == pytest.approx(36.0)


def test_night_timeline_until_the_end_of_the_input():
    """ The last chunk of an input lasts until the end of the input, which includes the frames after it (fewer
        than a chunk, which the star extraction doesn't use), so consecutive inputs join without a gap. An end
        which is not just after the last chunk is not taken.
    """

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)

    def _input(name, start, n):
        return [{'time': start + datetime.timedelta(seconds=4*i), 'input': name, 'clear_fraction': 1.0,
                 'transparency_mag': 0.0, 'reliable_seen': 1.0, 'stars': 500, 'matched': 300} for i in range(n)]

    # Two inputs of 10 chunks of 4 s and 2.5 s more, the second one beginning at the end of the first
    second = begin + datetime.timedelta(seconds=42.5)
    rows = _input('a.vid', begin, 10) + _input('b.vid', second, 10)
    ends = {'a.vid': second, 'b.vid': second + datetime.timedelta(seconds=42.5)}

    timeline = sq.nightTimeline(rows, chunk_seconds={'a.vid': 4.0, 'b.vid': 4.0}, input_ends=ends)
    assert timeline['gaps'] == []
    assert timeline['summary']['observed_h']*3600 == pytest.approx(85.0)
    assert timeline['segments'][-1][2] == ends['b.vid']

    # Without the ends, the frames after the last chunks are not covered
    timeline = sq.nightTimeline(rows, chunk_seconds={'a.vid': 4.0, 'b.vid': 4.0})
    assert timeline['summary']['observed_h']*3600 == pytest.approx(80.0)

    # An end far after the last chunk (e.g. a wrong number of frames) is not taken
    timeline = sq.nightTimeline(_input('a.vid', begin, 10), chunk_seconds={'a.vid': 4.0},
                                input_ends={'a.vid': begin + datetime.timedelta(seconds=100)})
    assert timeline['summary']['observed_h']*3600 == pytest.approx(40.0)


@pytest.mark.parametrize('mags, removed', [((4.5, 7.2, None), 2), ((None, None, None), 1), ((4.5, 7.2, 9.0), 0),
                                           ((), 2)])
def test_plot_of_the_detections(tmp_path, mags, removed):
    """ The detections are plotted by their median magnitudes, also when some or all of them have none, with the
        detections removed by clouds.
    """

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    rows = [{'time': begin + datetime.timedelta(seconds=4*i), 'input': 'a.vid', 'clear_fraction': 1.0,
             'transparency_mag': 0.0, 'reliable_seen': 1.0, 'stars': 500, 'matched': 300} for i in range(100)]
    detection_times = [begin + datetime.timedelta(seconds=60*(i + 1)) for i in range(len(mags))]
    removed_times = [begin + datetime.timedelta(seconds=30 + 60*i) for i in range(removed)]

    plot_path = str(tmp_path/'conditions.png')
    sq.plotSkyConditions(rows, plot_path, 'Test', detection_times=detection_times, detection_mags=list(mags),
                         removed_detections=removed, removed_times=removed_times, chunk_seconds={'a.vid': 4.0})
    assert os.path.getsize(plot_path) > 0


def test_plot_until_the_end_of_the_last_input(tmp_path, monkeypatch):
    """ The time axis of the plot reaches the end of the last input (its frames after the last chunk). """

    import matplotlib.dates as mdates
    import matplotlib.figure

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    rows = [{'time': begin + datetime.timedelta(seconds=4*i), 'input': 'a.vid', 'clear_fraction': 1.0,
             'transparency_mag': 0.0, 'reliable_seen': 1.0, 'stars': 500, 'matched': 300} for i in range(10)]
    end = begin + datetime.timedelta(seconds=42.5)

    # Keep the figure which is saved
    saved = {}
    real_save = matplotlib.figure.Figure.savefig
    def _save(self, *args, **kwargs):
        saved['fig'] = self
        return real_save(self, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', _save)

    summary = sq.plotSkyConditions(rows, str(tmp_path/'conditions.png'), 'Test', chunk_seconds={'a.vid': 4.0},
                                   input_ends={'a.vid': end})
    assert summary['observed_h']*3600 == pytest.approx(42.5)
    x_end = saved['fig'].axes[3].get_xlim()[1]
    assert x_end >= mdates.date2num(end) - 1e-9


def test_night_timeline_of_an_input_shorter_than_a_chunk():
    """ An input shorter than a chunk (its one chunk has fewer frames) lasts until its end. """

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    rows = [{'time': begin, 'input': 'a.vid', 'clear_fraction': 1.0, 'transparency_mag': 0.0, 'reliable_seen': 1.0,
             'stars': 500, 'matched': 300}]
    timeline = sq.nightTimeline(rows, chunk_seconds={'a.vid': 4.0},
                                input_ends={'a.vid': begin + datetime.timedelta(seconds=2.0)})
    assert timeline['summary']['observed_h']*3600 == pytest.approx(2.0)


def test_plot_until_the_last_removed_detection(tmp_path, monkeypatch):
    """ The time axis of the plot reaches the detections removed by clouds after the last chunk. """

    import matplotlib.dates as mdates
    import matplotlib.figure

    begin = datetime.datetime(2025, 12, 25, 22, 0, 0)
    rows = [{'time': begin + datetime.timedelta(seconds=4*i), 'input': 'a.vid', 'clear_fraction': 0.0,
             'transparency_mag': 1.0, 'reliable_seen': 0.2, 'stars': 50, 'matched': 30} for i in range(10)]
    removed = begin + datetime.timedelta(seconds=42.0)

    saved = {}
    real_save = matplotlib.figure.Figure.savefig
    def _save(self, *args, **kwargs):
        saved['fig'] = self
        return real_save(self, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', _save)

    sq.plotSkyConditions(rows, str(tmp_path/'conditions.png'), 'Test', chunk_seconds={'a.vid': 4.0},
                         removed_detections=1, removed_times=[removed])
    assert saved['fig'].axes[3].get_xlim()[1] >= mdates.date2num(removed) - 1e-9
