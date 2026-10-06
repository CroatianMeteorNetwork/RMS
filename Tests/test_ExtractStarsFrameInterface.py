"""
Tests for star extraction on frame interface inputs (RMS.ExtractStars.extractStarsImgHandle) and saving of its
results (RMS.DetectStarsAndMeteors).

A fake image handle with chunks of known content is used, so every chunk can be identified by its pixel
values.
"""

import datetime

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.ExtractStars as es
from RMS.DetectStarsAndMeteors import unbinStarList


BEG_TIME = datetime.datetime(2025, 12, 25, 10, 0, 0)


class _FakeFF(object):
    def __init__(self, avepixel):
        self.avepixel = avepixel
        self.maxpixel = avepixel


class _FakeHandle(object):
    """ Image handle mimicking the video handles: chunks of chunk_frames frames. The left half of each chunk
        image is the given chunk level and the right half is one count brighter, so the median rounds down to
        the chunk level and masking one half changes the image.
    """

    def __init__(self, chunk_levels, chunk_frames=128, fps=100.0):

        self.chunk_levels = chunk_levels
        self.total_fr_chunks = len(chunk_levels)
        self.chunk_frames = chunk_frames
        self.fps = fps
        self.current_frame_chunk = 0
        self.current_frame = 0

        # Keep the loaded chunks, as a real handle caches them
        self.cache = {}
        self.loaded_chunks = []

    def setFrame(self, fr_num):
        self.current_frame = fr_num

    def nextChunk(self):
        self.current_frame_chunk = (self.current_frame_chunk + 1)%self.total_fr_chunks
        self.current_frame = self.current_frame_chunk*self.chunk_frames

    def loadChunk(self, first_frame=None, read_nframes=None):

        chunk = self.current_frame_chunk
        self.loaded_chunks.append(chunk)

        if chunk not in self.cache:
            self.cache[chunk] = _FakeFF(self.chunkImage(chunk))

        return self.cache[chunk]

    def chunkImage(self, chunk):
        img = np.full((64, 64), self.chunk_levels[chunk], dtype=np.uint16)
        img[:, 32:] += 1
        return img

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        return BEG_TIME + datetime.timedelta(seconds=frame_no/self.fps)

    def currentTime(self, dt_obj=False, beginning=False):
        return self.currentFrameTime(frame_no=self.current_frame_chunk*self.chunk_frames)


@pytest.fixture
def config():

    config = cr.Config()
    config.stationID = 'XX0001'
    config.bit_depth = 8
    config.max_global_intensity = 150

    return config


@pytest.fixture
def fake_extract(monkeypatch):
    """ Replace the star extraction with a stub which returns one star whose amplitude is the chunk level, and
        fails for chunk levels listed in fail_levels.
    """

    calls = {'levels': [], 'fail_levels': set()}

    def _extractStars(img, **kwargs):

        level = int(np.median(img))
        calls['levels'].append(level)

        if level in calls['fail_levels']:
            return False

        one = lambda v: np.array([v])
        return one(10.0), one(20.0), one(level), one(1000), one(2.5), one(5), one(8.0), one(0)

    monkeypatch.setattr(es, 'extractStars', _extractStars)

    return calls


def _starLevels(star_list):
    """ Return the chunk level (stored as the amplitude) of every entry in the star list. """
    return [int(entry[1][0][3]) for entry in star_list]


def test_all_chunks_processed_in_order(config, fake_extract):

    handle = _FakeHandle([10, 20, 30, 40])
    star_list = es.extractStarsImgHandle(handle, config=config)

    assert handle.loaded_chunks == [0, 1, 2, 3]
    assert _starLevels(star_list) == [10, 20, 30, 40]


def test_failed_chunk_advances_to_next(config, fake_extract):

    fake_extract['fail_levels'] = {20}

    handle = _FakeHandle([10, 20, 30, 40])
    star_list = es.extractStarsImgHandle(handle, config=config)

    # The failed chunk must not be loaded again in place of the following chunks
    assert handle.loaded_chunks == [0, 1, 2, 3]
    assert _starLevels(star_list) == [10, 30, 40]


def test_bright_chunk_is_skipped_not_fatal(config, fake_extract):

    # Chunk 1 is above max_global_intensity (150)
    handle = _FakeHandle([10, 200, 30, 40])
    star_list = es.extractStarsImgHandle(handle, config=config)

    assert handle.loaded_chunks == [0, 1, 2, 3]
    assert _starLevels(star_list) == [10, 30, 40]


def test_all_chunks_failing_returns_error_return(config, fake_extract):

    handle = _FakeHandle([200, 200])
    star_list = es.extractStarsImgHandle(handle, config=config)

    assert star_list == [[], [], [], [], [], [], [], [], []]


def test_handle_reset_to_first_chunk(config, fake_extract):

    handle = _FakeHandle([10, 20, 30])

    # Start from a chunk other than the first one
    handle.current_frame_chunk = 2

    es.extractStarsImgHandle(handle, config=config)

    assert handle.loaded_chunks == [0, 1, 2]
    assert handle.current_frame_chunk == 0


def test_masking_does_not_modify_cached_chunk(config, fake_extract):

    handle = _FakeHandle([10, 20])

    # Mask out the left half of the image
    mask = np.ones((64, 64), dtype=np.uint8)
    mask[:, :32] = 0

    class _Mask(object):
        img = mask

    es.extractStarsImgHandle(handle, mask=_Mask(), config=config)

    for chunk in range(len(handle.chunk_levels)):
        assert np.array_equal(handle.cache[chunk].avepixel, handle.chunkImage(chunk))


def test_ff_names_use_chunk_start_time(config, fake_extract):

    handle = _FakeHandle([10, 20, 30], chunk_frames=128, fps=100.0)
    star_list = es.extractStarsImgHandle(handle, config=config)

    names = [entry[0] for entry in star_list]

    # Chunks start at 0, 1.28 and 2.56 s
    assert names == [
        'FF_XX0001_20251225_100000_000_0000000.fits',
        'FF_XX0001_20251225_100001_280_0000000.fits',
        'FF_XX0001_20251225_100002_560_0000000.fits',
    ]


### Star unbinning ###

def _starList():
    return [['FF_a.fits', [(10.0, 20.0, 1000, 50, 2.0, 7, 9.0, 0)]], [[]]]


@pytest.mark.parametrize('method, intens_factor', [('avg', 4), ('sum', 1)])
def test_unbinStarList(config, method, intens_factor):

    config.detection_binning_factor = 2
    config.detection_binning_method = method

    star_list = unbinStarList(_starList(), config)

    y, x, intens, ampl, fwhm, bg, snr, n_sat = star_list[0][1][0]

    assert (y, x, fwhm) == (20.0, 40.0, 4.0)
    assert intens == 1000*intens_factor
    assert (ampl, bg, snr, n_sat) == (50, 7, 9.0, 0)

    # Malformed entries are kept unchanged
    assert star_list[1] == [[]]


def test_unbinStarList_no_binning(config):

    config.detection_binning_factor = 1

    assert unbinStarList(_starList(), config) == _starList()


### Matching meteors to CALSTARS chunks in recalibration ###

def _closestChunk(meteor_list, fps_list, chunk_names, chunk_frames, fps):

    from RMS.Astrometry.ApplyRecalibrate import ftpMatchTimes
    from RMS.Formats import FFfile

    calstars_datetime_dict = {name: FFfile.getMiddleTimeFF(name, fps, dt_obj=True, ff_frames=chunk_frames)
                              for name in chunk_names}

    ftp_times = ftpMatchTimes(meteor_list, fps_list, calstars_datetime_dict, fps)

    return {ff_name: min(calstars_datetime_dict,
                         key=lambda x: abs((t - calstars_datetime_dict[x]).total_seconds()))
            for ff_name, t in ftp_times.items()}


def _chunkNames(n, chunk_frames, fps):
    from RMS.Formats import FFfile
    return [FFfile.constructFFName('XX0001', BEG_TIME + datetime.timedelta(seconds=i*chunk_frames/fps))
            for i in range(n)]


@pytest.mark.parametrize('seconds_into_chunk', [0.1, 1.0, 2.5, 3.9])
def test_first_pick_name_matched_to_containing_chunk(seconds_into_chunk):

    from RMS.Formats import FFfile

    # 128-frame chunks at 32 fps are 4 s long; the meteor begins in chunk 1
    fps = 32.0
    chunk_names = _chunkNames(4, 128, fps)

    pick_time = BEG_TIME + datetime.timedelta(seconds=4 + seconds_into_chunk)
    ff_name = FFfile.constructFFName('XX0001', pick_time)
    meteor_list = [[ff_name, 1, 0, 0, [[0.0, 10, 20], [1.0, 11, 21]]]]

    assert _closestChunk(meteor_list, [fps], chunk_names, 128, fps)[ff_name] == chunk_names[1]


def test_name_in_calstars_matched_to_itself():

    # A meteor late in an FF (frame 250 of 256) is still matched to its own FF
    fps = 25.0
    chunk_names = _chunkNames(3, 256, fps)
    meteor_list = [[chunk_names[1], 1, 0, 0, [[250.0, 10, 20]]]]

    assert _closestChunk(meteor_list, [None], chunk_names, 256, fps)[chunk_names[1]] == chunk_names[1]


def test_first_frame_offset_is_used():

    from RMS.Formats import FFfile

    # The name is the start of chunk 0, but the meteor begins at frame 200 (in chunk 1)
    fps = 32.0
    chunk_names = _chunkNames(3, 128, fps)
    ff_name = FFfile.constructFFName('XX0001', BEG_TIME + datetime.timedelta(seconds=0.5))
    meteor_list = [[ff_name, 1, 0, 0, [[184.0, 10, 20]]]]

    assert _closestChunk(meteor_list, [fps], chunk_names, 128, fps)[ff_name] == chunk_names[1]
