""" Continuous detection over consecutive input files of a recording (monitor_continuous).

A file which follows the previous file of the recording without missing frames is searched with the last frames of
the previous file in front of it (see RMS.Formats.FrameChain), so the search continues across the boundary of the
files as in one long file. A detection of an object which crosses the boundary is then found in both files: in the
earlier one up to its end, and in the later one from the borrowed frames on. Once both files are processed, the two
are joined into one detection (the frames before the boundary from the earlier file, the others from the later one),
which is kept in the results of the file where the object first appears and removed from the other one. A detection
of the later file which begins in the borrowed frames and has no counterpart in the earlier file is moved to the
results of the earlier file. The changed results are recalibrated, so their astrometry and photometry (and the ECSV
files) are those of the joined detections.

Every processed file which has a previous file leaves a pending join in the .continuous directory of the output
directory, which the monitor handles in the order of the boundaries (see processPendingJoins), once both files are
processed and the earlier file's own join is done. The join records where its detections went (continuous.json in
the results directory of the later file), so that a detection joined over several files is continued in the results
where it begins.
"""

from __future__ import print_function, division, absolute_import

import os
import glob
import json
import time
import datetime

import numpy as np

from RMS.Formats import FTPdetectinfo
from RMS.Formats.FFfile import constructFFName, filenameToDatetime
from RMS.Logger import getLogger

log = getLogger("logger")


# Directory of the pending joins in the output directory, and the record of a join in the results directory
PENDING_DIR = '.continuous'
JOIN_RECORD = 'continuous.json'

# A pending join one of whose files has no results this long (s) after the processing of the later file began is
#   dropped (e.g. a file was skipped, failed or its processing was stopped); the detections are kept as they are
PENDING_MAX_WAIT = 3600.0

# A join which fails is tried again when the monitor next handles the pending joins, up to this many times, after
#   which it is dropped (the detections stay as they are)
MAX_JOIN_ATTEMPTS = 3

# Two detections are the same object if they share at least this many frames of the borrowed frames, at a median
#   distance of at most this many pixels
MIN_COMMON_FRAMES = 3
MAX_COMMON_DISTANCE = 3.0



def previousInput(file_path, root=None):
    """ The input recorded just before an input of a recording, by the names (which begin with the time): the
        previous file with the same extension in the same directory or in the previous directory of the parent
        directory, or the previous directory for a directory of frames.

    Arguments:
        file_path: [str] Input file, or a directory of frames.

    Keyword arguments:
        root: [str] The monitored input directory, outside which no input is looked for (e.g. the input of another
            camera next to it). None by default (no limit).

    Return:
        [str] The previous input, None if there is none.
    """

    file_path = os.path.abspath(file_path)
    parent, name = os.path.split(file_path)
    root = os.path.abspath(root) if root is not None else None

    def inside(path):
        """ Whether a path is inside the monitored input directory (or there is no limit). """
        return (root is None) or (path == root) or path.startswith(root + os.sep)

    # An input outside the monitored directory has no previous input
    if not inside(parent):
        return None

    def visible(entries):
        """ The entries of a directory without the hidden ones. """
        return [e for e in entries if not e.startswith('.')]

    # A directory of frames: the previous sibling directory
    if os.path.isdir(file_path):
        siblings = sorted(e for e in visible(os.listdir(parent)) if os.path.isdir(os.path.join(parent, e))
                          and e < name)
        return os.path.join(parent, siblings[-1]) if siblings else None

    # A file: the previous file with the same extension in its directory, else the last one of the previous
    #   sibling directory (e.g. the directories of 30 s of a session)
    ext = os.path.splitext(name)[1].lower()
    same = sorted(e for e in visible(os.listdir(parent)) if e.lower().endswith(ext) and e < name)
    if same:
        return os.path.join(parent, same[-1])

    # The previous directory is looked for only inside the monitored directory
    grand, parent_name = os.path.split(parent)
    if (parent == root) or (not inside(grand)):
        return None
    dirs = sorted(e for e in visible(os.listdir(grand)) if os.path.isdir(os.path.join(grand, e)) and e < parent_name)
    if dirs:
        files = sorted(e for e in visible(os.listdir(os.path.join(grand, dirs[-1]))) if e.lower().endswith(ext))
        if files:
            return os.path.join(grand, dirs[-1], files[-1])

    return None



def writePending(output_dir, unique_id, pending):
    """ Leave a pending join for the monitor (see processPendingJoins).

    Arguments:
        output_dir: [str] Output directory of the camera.
        unique_id: [str] ID of the later file.
        pending: [dict] The join: results_dir, previous_results_dir, boundary_unix (the time of the first frame of
            the later file), context_s (the length of the borrowed frames, s).
    """

    # The join with the ID of the file and the time it was left (kept when it is written again)
    pending_dir = os.path.join(output_dir, PENDING_DIR)
    os.makedirs(pending_dir, exist_ok=True)
    pending = dict(pending, unique_id=unique_id)
    pending.setdefault('created_unix', time.time())

    # Written to a temporary file and renamed, so the monitor never reads a partly written join
    path = os.path.join(pending_dir, unique_id + '.json')
    with open(path + '.tmp', 'w') as f:
        json.dump(pending, f, indent=1)
    os.replace(path + '.tmp', path)



def _ftpPath(results_dir):
    """ The FTPdetectinfo of the detections of a results directory, None if there is none. """

    # The FTPdetectinfo files of the detections (not the backups, nor those of the separate passes)
    names = sorted(n for n in os.listdir(results_dir) if FTPdetectinfo.validDefaultFTPdetectinfo(n))

    return os.path.join(results_dir, names[0]) if names else None



def readDetections(results_dir):
    """ The detections of a results directory, with the time of every row.

    Arguments:
        results_dir: [str] Results directory.

    Return:
        [list] Detections as dicts: ff_name, fps, rho, phi, times (unix s), rows (frame, x, y, intensity,
            background, snr, saturated pixels), dir.
    """

    path = _ftpPath(results_dir)
    if path is None:
        return []

    dets = []
    for entry in FTPdetectinfo.readFTPdetectinfo(os.path.dirname(path), os.path.basename(path)):
        ff_name, fps, rho, phi, meas = entry[0], entry[4], entry[9], entry[10], np.array(entry[-1], dtype=float)

        # The time of every row from the time in the name and the frame, and the columns which are written back
        t_ref = filenameToDatetime(ff_name).replace(tzinfo=datetime.timezone.utc).timestamp()
        rows = meas[:, [1, 2, 3, 8, 10, 11, 12]]
        dets.append(dict(ff_name=ff_name, fps=fps, rho=rho, phi=phi, times=t_ref + meas[:, 1]/fps, rows=rows,
                         dir=results_dir))

    return dets



def writeDetections(results_dir, dets, config):
    """ Write the detections of a results directory (uncalibrated, as the processing writes them), named after
        the first row of every detection.

    Arguments:
        results_dir: [str] Results directory.
        dets: [list] Detections (see readDetections).
        config: [Config] Configuration.
    """

    path = _ftpPath(results_dir)
    if path is None:

        # The name of the FTPdetectinfo of a directory without detections, as from its CALSTARS
        calstars = sorted(glob.glob(os.path.join(results_dir, 'CALSTARS_*.txt')))
        prefix = os.path.basename(calstars[0])[len('CALSTARS_'):-len('.txt')] if calstars \
            else os.path.basename(os.path.normpath(results_dir))
        path = os.path.join(results_dir, 'FTPdetectinfo_' + prefix + '.txt')

    # The detections in the order of their beginnings, numbered within every FF name
    results, counts = [], {}
    for det in sorted(dets, key=lambda d: d['times'][0]):

        # The frames relative to the time in the name, at the frame rate of the detection
        fps = det['fps']
        t_ref = filenameToDatetime(det['ff_name']).replace(tzinfo=datetime.timezone.utc).timestamp()
        rows = np.array(det['rows'], dtype=np.float64)
        rows[:, 0] = (np.asarray(det['times']) - t_ref)*fps
        counts[det['ff_name']] = counts.get(det['ff_name'], 0) + 1
        results.append([det['ff_name'], counts[det['ff_name']], det['rho'], det['phi'], rows, fps])

    FTPdetectinfo.writeFTPdetectinfo(results, results_dir, os.path.basename(path), results_dir, config.stationID,
                                     config.fps)



def _sameObject(a, b, t_lo, t_hi, fps):
    """ Whether two detections are the same object: rows at the same times within [t_lo, t_hi), at a median
        distance of at most MAX_COMMON_DISTANCE px.

    Return:
        [bool]
    """

    # The rows of the first detection in the time range
    ia = np.nonzero((a['times'] >= t_lo) & (a['times'] < t_hi))[0]
    if len(ia) == 0:
        return False

    # The distance to the row of the second detection at the same time (within a third of a frame)
    tol = 0.3/fps
    dist = []
    for i in ia:
        j = int(np.argmin(np.abs(b['times'] - a['times'][i])))
        if abs(b['times'][j] - a['times'][i]) <= tol:
            dist.append(np.hypot(*(a['rows'][i, 1:3] - b['rows'][j, 1:3])))

    return (len(dist) >= MIN_COMMON_FRAMES) and (np.median(dist) <= MAX_COMMON_DISTANCE)



def _merge(earlier, later, boundary):
    """ One detection from two of the same object: the rows of the earlier one before the boundary and of the later
        one from the boundary on (also earlier rows of the later one, before the beginning of the earlier one). """

    # The rows of each detection which are kept, in the order of their times
    keep_e = earlier['times'] < boundary
    keep_l = (later['times'] >= boundary) | (later['times'] < earlier['times'][0])
    times = np.r_[earlier['times'][keep_e], later['times'][keep_l]]
    rows = np.vstack([earlier['rows'][keep_e], later['rows'][keep_l]])
    order = np.argsort(times)

    return dict(earlier, times=times[order], rows=rows[order])



def joinResults(previous_dir, results_dir, boundary, context_s, config, recalibrate=None,
                previous_chunk_images=None):
    """ Join the detections of a file with those of the previous file of the recording (see the module
        description).

    Arguments:
        previous_dir: [str] Results directory of the previous file.
        results_dir: [str] Results directory of the file.
        boundary: [float] Time of the first frame of the file (unix s).
        context_s: [float] Length of the borrowed frames before the boundary (s).
        config: [Config] Configuration.

    Keyword arguments:
        recalibrate: [function] recalibrate(results_dir) of the changed results. None by default (not recalibrated).
        previous_chunk_images: [list] [time of the first frame (unix s), FF name] of the image pairs of the chunks
            of the previous file, sorted by time: a detection moved to it is named after the image of its first
            frame, as the detections of the file. None by default (named after the time of its first frame).

    Return:
        [dict] The record of the join: the number of joined and moved detections, and where they went.
    """

    # A join tried before which failed after writing the detections only has recalibrations left
    own_record = os.path.join(results_dir, JOIN_RECORD)
    if os.path.isfile(own_record):
        with open(own_record) as f:
            result = json.load(f)
        if result.get('to_recalibrate') and (result.get('previous_results_dir') == previous_dir):
            return _recalibrateJoined(results_dir, result, recalibrate)

    # The detections of the earlier side: those of the previous file and those it gave to the files before it
    #   (a detection joined over several files is kept where it begins)
    homes = [previous_dir]
    record_path = os.path.join(previous_dir, JOIN_RECORD)
    if os.path.isfile(record_path):
        with open(record_path) as f:
            homes += [m['home'] for m in json.load(f).get('moved', []) if os.path.isdir(m['home'])]
    earlier = {}
    for home in dict.fromkeys(homes):
        earlier[home] = readDetections(home)
    later = readDetections(results_dir)

    # Every detection of the later file which reaches into the borrowed frames is joined or moved
    joined, moved, record = 0, 0, []
    changed = set()
    keep_later = []
    for det in later:

        # Only the detections which begin in the borrowed frames reach into the previous file
        if det['times'][0] >= boundary:
            keep_later.append(det)
            continue

        # The counterpart on the earlier side, in the borrowed frames
        partner = None
        for home, dets in earlier.items():
            for k, other in enumerate(dets):
                if _sameObject(det, other, boundary - context_s, boundary, det['fps']):
                    partner = (home, k)
                    break
            if partner is not None:
                break

        # Joined with the counterpart where it begins, or moved to the previous file where it first appears
        if partner is not None:
            home, k = partner
            earlier[home][k] = _merge(earlier[home][k], det, boundary)
            joined += 1
        else:
            home = previous_dir

            # Only in the borrowed frames: the previous file searched them itself
            if det['times'][-1] < boundary:
                changed.add(results_dir)
                continue
            earlier[home].append(dict(det, ff_name=_movedName(det['times'][0], config, previous_chunk_images),
                                      dir=home))
            moved += 1

        # Both results change, and the record tells where the detection went
        changed.update([home, results_dir])
        record.append({'home': home, 't_first': float(det['times'][0]), 't_last': float(det['times'][-1])})

    # The changed results are written, and the record of the join with the results still to be recalibrated
    for home, dets in earlier.items():
        if home in changed:
            writeDetections(home, dets, config)
    if results_dir in changed:
        writeDetections(results_dir, keep_later, config)


    result = {'joined': joined, 'moved': record, 'boundary_unix': boundary, 'previous_results_dir': previous_dir,
              'n_joined': joined, 'n_moved': moved, 'changed': sorted(changed),
              'to_recalibrate': sorted(changed) if recalibrate is not None else []}
    _writeRecord(results_dir, result)

    return _recalibrateJoined(results_dir, result, recalibrate)



def _movedName(t_first, config, chunk_images=None):
    """ The FF name of a detection moved to the previous file: the image pair of the chunk with its first frame,
        else the time of its first frame. """

    # The last image pair which begins at or before the first frame (within 0.1 ms, the precision of the names)
    if chunk_images:
        names = [name for t, name in chunk_images if t <= t_first + 1e-4]
        if names:
            return names[-1]

    # Without the image pairs, the name of the time of the first frame
    first_time = datetime.datetime.fromtimestamp(t_first, tz=datetime.timezone.utc).replace(tzinfo=None)

    return constructFFName(config.stationID, first_time)



def _writeRecord(results_dir, result):
    """ Write the record of a join (JOIN_RECORD) in the results directory of the later file. """

    # Written to a temporary file and renamed, so the record is never partly written
    path = os.path.join(results_dir, JOIN_RECORD)
    with open(path + '.tmp', 'w') as f:
        json.dump(result, f, indent=1)
    os.replace(path + '.tmp', path)



def _recalibrateJoined(results_dir, result, recalibrate):
    """ Recalibrate the results changed by a join, one by one, keeping in its record those still to be done (so a
        join which failed in the recalibration is finished when it is tried again).
    """

    # Every recalibration done is taken off the record at once
    while result.get('to_recalibrate'):
        if recalibrate is not None:
            recalibrate(result['to_recalibrate'][0])
        result['to_recalibrate'] = result['to_recalibrate'][1:]
        _writeRecord(results_dir, result)

    # The changed results are new results: the night report finds them by the time of their done flags
    for d in result.get('changed', []):
        flag = os.path.join(d, 'done.flag')
        if os.path.isfile(flag):
            os.utime(flag, None)

    return result



def processPendingJoins(output_dir, config, recalibrate=None, now=None, done_after=None):
    """ Do the pending joins whose files are both processed, in the order of their boundaries, the join of a file
        only after the join of the file before it. Called by the monitor in its loop.

    Arguments:
        output_dir: [str] Output directory of the camera.
        config: [Config] Configuration.

    Keyword arguments:
        recalibrate: [function] recalibrate(results_dir), see joinResults. None by default.
        now: [float] Current time (unix s), for the waiting limit. None by default (the clock).
        done_after: [float] Only the done flags written at or after this time (unix s) mark processed files, e.g.
            the start of a monitor which processes all files again (--force), so the old results of a file not
            yet processed again are not joined. None by default (any done flag).

    Return:
        [int] Number of joins done.
    """

    def isDone(results_dir):
        """ Whether a file is processed: its results have a done flag (written after done_after). """
        flag = os.path.join(results_dir, 'done.flag')
        if not os.path.isfile(flag):
            return False
        return (done_after is None) or (os.path.getmtime(flag) >= done_after)

    pending_dir = os.path.join(output_dir, PENDING_DIR)
    if not os.path.isdir(pending_dir):
        return 0
    now = time.time() if now is None else now

    # The pending joins in the order of their boundaries (one being written is read the next time), and the
    #   results whose own join is still pending
    pendings = []
    for path in glob.glob(os.path.join(pending_dir, '*.json')):
        try:
            with open(path) as f:
                pendings.append((path, json.load(f)))
        except (OSError, ValueError):
            continue
    pendings.sort(key=lambda p: p[1]['boundary_unix'])
    waiting_dirs = {p['results_dir'] for _, p in pendings}

    done = 0
    for path, p in pendings:

        # Whether both files are processed
        cur_done = isDone(p['results_dir'])
        prev_done = isDone(p['previous_results_dir'])

        # One of the files never got results (skipped, failed, or its processing was stopped): nothing to join,
        #   and the joins after it are not held back
        if ((not cur_done) or (not prev_done)) and (now - p['created_unix'] > PENDING_MAX_WAIT):
            missing = p['results_dir'] if not cur_done else p['previous_results_dir']
            log.warning('Continuous detection: {} has no results, {} is not joined to {}'.format(
                missing, p['results_dir'], p['previous_results_dir']))
            os.remove(path)
            waiting_dirs.discard(p['results_dir'])
            continue

        # Both files processed, and the earlier file joined to the one before it
        if (not cur_done) or (not prev_done) or (p['previous_results_dir'] in waiting_dirs):
            continue
        try:
            result = joinResults(p['previous_results_dir'], p['results_dir'], p['boundary_unix'], p['context_s'],
                                 config, recalibrate=recalibrate,
                                 previous_chunk_images=p.get('previous_chunk_images'))
            log.info('Continuous detection: {} joined to {}: {:d} detections joined, {:d} moved'.format(
                os.path.basename(p['results_dir']), os.path.basename(p['previous_results_dir']),
                result['n_joined'], result['n_moved']))

        # A failed join stays pending (and holds back the joins after it) to be tried again, up to a limit
        except Exception as e:
            attempts = p.get('attempts', 0) + 1
            if attempts < MAX_JOIN_ATTEMPTS:
                log.warning('Continuous detection: the join of {} failed (attempt {:d}), tried again later: {}'.format(
                    p['results_dir'], attempts, repr(e)))
                writePending(output_dir, p['unique_id'], dict(p, attempts=attempts))
                continue
            log.warning('Continuous detection: the join of {} failed {:d} times, dropped: {}'.format(
                p['results_dir'], attempts, repr(e)))

        # Done (or given up): the joins after it can go on
        os.remove(path)
        waiting_dirs.discard(p['results_dir'])
        done += 1

    return done
