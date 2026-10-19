""" Tests of the continuous detection over consecutive files (RMS.Formats.FrameChain, RMS.ContinuousResults). """

from __future__ import print_function, division, absolute_import

import os
import json
import time
import datetime

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.ContinuousResults as cont
from RMS.Formats.FFfile import constructFFName
from RMS.Formats.FrameChain import ChainedInput, contiguous


T0 = datetime.datetime(2026, 10, 18, 1, 15, 28)


class _Handle(object):
    """ An input of n frames every 0.1 s from t0, the value of every frame its time (s from T0). """

    def __init__(self, t0_s, n, fps=10.0):
        self.input_type = 'fblk'
        self.nrows, self.ncols = 4, 5
        self.fps = fps
        self.chunk_frames = 4
        self.total_frames = n
        self.t0_s = t0_s
        self.beginning_datetime = T0 + datetime.timedelta(seconds=t0_s)
        self.frame_unix_times = np.array([(self.beginning_datetime + datetime.timedelta(seconds=i/fps))
                                          .replace(tzinfo=datetime.timezone.utc).timestamp() for i in range(n)])
        self.current = 0

    def setFrame(self, i):
        self.current = i

    def loadFrame(self, avepixel=False):
        return np.full((4, 5), self.t0_s + self.current/self.fps, dtype=np.float32)

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        i = self.current if frame_no is None else frame_no
        return self.beginning_datetime + datetime.timedelta(seconds=i/self.fps)

    def hasAccurateTime(self):
        return True

    def name(self, beginning=False):
        return str(self.beginning_datetime)


def test_chained_input_puts_the_last_frames_of_the_previous_input_in_front():
    """ The chained input gives the last frames of the previous input, then the frames of the input, with their
        times; it begins at its first frame and is named after the input. """

    prev, cur = _Handle(0.0, 30), _Handle(3.0, 20)
    chain = ChainedInput(prev, 8, cur)
    assert chain.total_frames == 28 and chain.n_context == 8
    chain.setFrame(0)
    assert chain.loadFrame()[0, 0] == pytest.approx(2.2)
    chain.setFrame(8)
    assert chain.loadFrame()[0, 0] == pytest.approx(3.0)
    assert chain.currentFrameTime(frame_no=27, dt_obj=True) == cur.currentFrameTime(frame_no=19, dt_obj=True)
    assert chain.beginning_datetime == prev.currentFrameTime(frame_no=22, dt_obj=True)
    assert chain.name_datetime == cur.beginning_datetime
    assert np.allclose(np.diff(chain.frame_unix_times), 0.1)

    # The previous input follows without missing frames, not with a gap
    assert contiguous(prev, cur)
    assert not contiguous(prev, _Handle(3.5, 20))

    # Nor with frames of another size
    other = _Handle(3.0, 20)
    other.nrows = 8
    assert not contiguous(prev, other)

    # Nor at another frame rate
    assert not contiguous(prev, _Handle(3.0, 20, fps=20.0))


def test_previous_input_of_files_and_directories(tmp_path):
    """ The previous input is the previous file of the directory, else the last one of the previous directory, or
        the previous directory of frames. """

    for d, files in (('20261018_011458', ['A_011458.fblk']), ('20261018_011528', ['A_011528.fblk', 'A_011541.fblk']),
                     ('.hidden', ['A_011600.fblk'])):
        os.makedirs(str(tmp_path/d))
        for f in files:
            open(str(tmp_path/d/f), 'w').close()

    assert cont.previousInput(str(tmp_path/'20261018_011528'/'A_011541.fblk')) == \
        str(tmp_path/'20261018_011528'/'A_011528.fblk')
    assert cont.previousInput(str(tmp_path/'20261018_011528'/'A_011528.fblk')) == \
        str(tmp_path/'20261018_011458'/'A_011458.fblk')
    assert cont.previousInput(str(tmp_path/'20261018_011458'/'A_011458.fblk')) is None
    assert cont.previousInput(str(tmp_path/'20261018_011528')) == str(tmp_path/'20261018_011458')

    # Not outside the monitored directory (e.g. the input of another camera next to it)
    assert cont.previousInput(str(tmp_path/'20261018_011528'/'A_011528.fblk'),
                              root=str(tmp_path/'20261018_011528')) is None
    assert cont.previousInput(str(tmp_path/'20261018_011528'/'A_011528.fblk'), root=str(tmp_path)) == \
        str(tmp_path/'20261018_011458'/'A_011458.fblk')


@pytest.fixture
def config():
    c = cr.parse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config'))
    c.stationID = 'TEST01'
    return c


def _det(t_first, t_last, x0=100.0, vx=20.0, y=300.0, fps=10.0):
    """ A detection of an object moving at vx px/s from x0 at T0, rows every frame from t_first to t_last (s). """

    times_s = np.arange(int(round(t_first*fps)), int(round(t_last*fps)) + 1)/fps
    t_unix = np.array([(T0 + datetime.timedelta(seconds=t)).replace(tzinfo=datetime.timezone.utc).timestamp()
                       for t in times_s])
    rows = np.c_[np.zeros(len(times_s)), x0 + vx*times_s, np.full(len(times_s), y), np.full(len(times_s), 500.0),
                 np.full(len(times_s), 300.0), np.full(len(times_s), 10.0), np.zeros(len(times_s))]
    ff = constructFFName('TEST01', T0 + datetime.timedelta(seconds=times_s[0]))
    return dict(ff_name=ff, fps=fps, rho=0.0, phi=0.0, times=t_unix, rows=rows)


def _results(tmp_path, name, dets, config):
    d = str(tmp_path/name)
    os.makedirs(d)
    open(os.path.join(d, 'CALSTARS_TEST01_{:s}.txt'.format(name)), 'w').close()
    open(os.path.join(d, 'done.flag'), 'w').close()
    if dets:
        cont.writeDetections(d, dets, config)
    return d


def test_detections_across_the_boundary_are_joined_where_they_begin(tmp_path, config):
    """ An object crossing the boundary is one detection in the results of the earlier file, made of its rows
        before the boundary and of the later file's rows after it (and before the earlier detection begins); an
        object of the later file which begins in the borrowed frames without a counterpart is moved to the earlier
        file; the others stay. """

    boundary_s = 30.0
    boundary = (T0 + datetime.timedelta(seconds=boundary_s)).replace(tzinfo=datetime.timezone.utc).timestamp()

    # Earlier file: the crossing object from 20 s to its end at the boundary. Later file (12.8 s borrowed): the same
    #   object from 17.2 s to 40 s, an object which begins at 25 s (not found in the earlier file), and one of its own
    prev = _results(tmp_path, 'prev', [_det(20.0, 29.9)], config)
    cur = _results(tmp_path, 'cur', [_det(17.2, 40.0), _det(25.0, 35.0, x0=900.0, y=500.0),
                                     _det(31.0, 38.0, x0=50.0, y=800.0)], config)

    result = cont.joinResults(prev, cur, boundary, 12.8, config)
    assert result['n_joined'] == 1 and result['n_moved'] == 1

    earlier, later = cont.readDetections(prev), cont.readDetections(cur)
    assert len(earlier) == 2 and len(later) == 1
    joined = min(earlier, key=lambda d: d['times'][0])
    assert joined['times'][0] == pytest.approx(boundary - 12.8, abs=1e-3)
    assert joined['times'][-1] == pytest.approx(boundary + 10.0, abs=1e-3)
    assert np.allclose(np.diff(joined['times']), 0.1, atol=2e-3)
    assert later[0]['rows'][0, 2] == pytest.approx(800.0)

    # The record tells where the detections went, for the join of the next file
    with open(os.path.join(cur, cont.JOIN_RECORD)) as f:
        assert {m['home'] for m in json.load(f)['moved']} == {prev}


def test_pending_joins_in_order_after_the_files_are_processed(tmp_path, config):
    """ The joins are done once both files are processed, a file's join after the join of the file before it, and
        an object crossing two boundaries is one detection where it begins. """

    out = str(tmp_path/'out')
    os.makedirs(out)
    b1 = (T0 + datetime.timedelta(seconds=30)).replace(tzinfo=datetime.timezone.utc).timestamp()
    b2 = (T0 + datetime.timedelta(seconds=60)).replace(tzinfo=datetime.timezone.utc).timestamp()

    # An object through three files, from 17.2 s (in the borrowed frames of the second file) to 70 s
    a = _results(tmp_path, 'a', [_det(20.0, 29.9)], config)
    b = _results(tmp_path, 'b', [_det(17.2, 59.9)], config)
    c = _results(tmp_path, 'c', [_det(47.2, 70.0)], config)
    os.remove(os.path.join(c, 'done.flag'))
    cont.writePending(out, 'b', {'results_dir': b, 'previous_results_dir': a, 'boundary_unix': b1, 'context_s': 12.8})
    cont.writePending(out, 'c', {'results_dir': c, 'previous_results_dir': b, 'boundary_unix': b2, 'context_s': 12.8})

    # The later file is not processed yet: only the first join
    assert cont.processPendingJoins(out, config) == 1
    open(os.path.join(c, 'done.flag'), 'w').close()
    assert cont.processPendingJoins(out, config) == 1
    assert cont.processPendingJoins(out, config) == 0

    dets = cont.readDetections(a)
    assert len(dets) == 1 and not cont.readDetections(b) and not cont.readDetections(c)
    assert dets[0]['times'][-1] - dets[0]['times'][0] == pytest.approx(52.8, abs=1e-3)



def test_failed_join_is_tried_again(tmp_path, config):
    """ A join which fails in the recalibration stays pending, holds back the join after it, and is finished (only
        the recalibrations left) when it is tried again. """

    out = str(tmp_path/'out')
    os.makedirs(out)
    b1 = (T0 + datetime.timedelta(seconds=30)).replace(tzinfo=datetime.timezone.utc).timestamp()
    b2 = (T0 + datetime.timedelta(seconds=60)).replace(tzinfo=datetime.timezone.utc).timestamp()
    a = _results(tmp_path, 'a', [_det(20.0, 29.9)], config)
    b = _results(tmp_path, 'b', [_det(17.2, 59.9)], config)
    c = _results(tmp_path, 'c', [_det(47.2, 70.0)], config)
    cont.writePending(out, 'b', {'results_dir': b, 'previous_results_dir': a, 'boundary_unix': b1, 'context_s': 12.8})
    cont.writePending(out, 'c', {'results_dir': c, 'previous_results_dir': b, 'boundary_unix': b2, 'context_s': 12.8})

    # The first recalibration fails
    calls = []
    def recalibrate(d):
        """ Records the recalibrations; the first one fails. """
        calls.append(d)
        if len(calls) == 1:
            raise RuntimeError('recalibration failed')

    assert cont.processPendingJoins(out, config, recalibrate=recalibrate) == 0
    assert sorted(os.listdir(os.path.join(out, cont.PENDING_DIR))) == ['b.json', 'c.json']

    # Tried again: the recalibrations are finished, then the join after it is done
    assert cont.processPendingJoins(out, config, recalibrate=recalibrate) == 2
    assert not os.listdir(os.path.join(out, cont.PENDING_DIR))
    assert sorted(set(calls)) == sorted({a, b, c})
    dets = cont.readDetections(a)
    assert len(dets) == 1 and not cont.readDetections(b) and not cont.readDetections(c)
    assert dets[0]['times'][-1] - dets[0]['times'][0] == pytest.approx(52.8, abs=1e-3)



def test_sky_quality_of_a_chained_input_from_its_own_chunks(config, monkeypatch):
    """ The sky quality of a chained input is that of the chunks of the input (its stars), at their frames in the
        chained input also when the borrowed frames are not a whole number of chunks; the borrowed frames are not
        missing chunks of the input. """

    import RMS.MatchedFilterDetection as mf

    prev, cur = _Handle(0.0, 6), _Handle(0.6, 20)
    chain = ChainedInput(prev, 8, cur)
    assert chain.n_context == 6
    star_list = [[constructFFName(config.stationID, cur.currentFrameTime(frame_no=4*k, dt_obj=True)), []]
                 for k in range(5)]

    # The arguments of the sky quality map
    seen = {}
    class _Map(object):
        """ Records the arguments of the sky quality map. """
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
        def summary(self):
            return np.ones(1)
        n_matched, transparency, reliable_seen = np.ones(1), np.zeros(1), np.ones(1)

    monkeypatch.setattr(mf, 'SkyQualityMap', _Map)
    config.mf_cloud_filter = True
    assert mf.skyQuality(chain, config, star_list, object(), update=False) is not None
    assert seen['first_frame'] == 6
    assert seen['first_frames'] == [6, 10, 14, 18, 22]



def test_join_of_a_file_never_processed_expires(tmp_path, config):
    """ A file whose processing never ends (no done flag) holds back the joins after it only until the pending
        joins expire, then the later files are joined. """

    out = str(tmp_path/'out')
    os.makedirs(out)
    bound = lambda s: (T0 + datetime.timedelta(seconds=s)).replace(tzinfo=datetime.timezone.utc).timestamp()
    a = _results(tmp_path, 'a', [_det(20.0, 29.9)], config)
    b = _results(tmp_path, 'b', [_det(40.0, 45.0)], config)
    c = _results(tmp_path, 'c', [_det(70.0, 89.9)], config)
    d = _results(tmp_path, 'd', [_det(80.0, 100.0)], config)
    os.remove(os.path.join(b, 'done.flag'))
    for name, cur, prev, t in (('b', b, a, 30), ('c', c, b, 60), ('d', d, c, 90)):
        cont.writePending(out, name, {'results_dir': cur, 'previous_results_dir': prev, 'boundary_unix': bound(t),
                                      'context_s': 12.8})

    # Held back while the file may still be processed
    assert cont.processPendingJoins(out, config) == 0
    assert len(os.listdir(os.path.join(out, cont.PENDING_DIR))) == 3

    # Expired: the joins with the unprocessed file are dropped, the one after them is done
    assert cont.processPendingJoins(out, config, now=time.time() + cont.PENDING_MAX_WAIT + 1) == 1
    assert not os.listdir(os.path.join(out, cont.PENDING_DIR))
    assert len(cont.readDetections(c)) == 1 and len(cont.readDetections(d)) == 0



def test_joins_wait_for_results_newer_than_a_forced_start(tmp_path, config):
    """ With all files processed again (--force), a done flag from before the start is old: the join waits until
        both files have new results. """

    out = str(tmp_path/'out')
    os.makedirs(out)
    b1 = (T0 + datetime.timedelta(seconds=30)).replace(tzinfo=datetime.timezone.utc).timestamp()
    a = _results(tmp_path, 'a', [_det(20.0, 29.9)], config)
    b = _results(tmp_path, 'b', [_det(17.2, 40.0)], config)
    cont.writePending(out, 'b', {'results_dir': b, 'previous_results_dir': a, 'boundary_unix': b1, 'context_s': 12.8})

    # The earlier file's done flag is from before the start
    start = time.time() - 50
    os.utime(os.path.join(a, 'done.flag'), (start - 100, start - 100))
    assert cont.processPendingJoins(out, config, done_after=start) == 0

    # Processed again: joined
    open(os.path.join(a, 'done.flag'), 'w').close()
    assert cont.processPendingJoins(out, config, done_after=start) == 1



def test_moved_detection_named_after_the_chunk_image_and_results_marked_new(tmp_path, config):
    """ A detection moved to the previous file takes the name of the image pair of the chunk with its first frame
        (as the detections of the file do), and the changed results get new done flags (the night report takes
        the results by the times of their done flags). """

    boundary = (T0 + datetime.timedelta(seconds=30)).replace(tzinfo=datetime.timezone.utc).timestamp()
    prev = _results(tmp_path, 'prev', [], config)
    cur = _results(tmp_path, 'cur', [_det(25.0, 35.0, x0=900.0, y=500.0)], config)
    old = time.time() - 100
    for d in (prev, cur):
        os.utime(os.path.join(d, 'done.flag'), (old, old))

    # The images of the previous file: chunks beginning at 17.2 s and 22.4 s
    unix = lambda s: (T0 + datetime.timedelta(seconds=s)).replace(tzinfo=datetime.timezone.utc).timestamp()
    images = [[unix(17.2), 'FF_TEST01_20261018_011545_200_0000000.fits'],
              [unix(22.4), 'FF_TEST01_20261018_011550_400_0000052.fits']]
    cont.joinResults(prev, cur, boundary, 12.8, config, previous_chunk_images=images)

    moved = cont.readDetections(prev)
    assert len(moved) == 1 and moved[0]['ff_name'] == images[1][1]
    assert moved[0]['times'][0] == pytest.approx(unix(25.0), abs=1e-3)
    for d in (prev, cur):
        assert os.path.getmtime(os.path.join(d, 'done.flag')) > old + 50
