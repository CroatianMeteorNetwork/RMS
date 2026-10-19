""" Local sky quality from the stars of the chunks of frames: where and when the sky is clear enough for the
measurements of the detections to be trusted.

The stars of every chunk are matched to the star catalog, and the difference between the instrumental magnitude of
every matched star (with a fixed zero point) and its catalog magnitude is its photometric residual. On clear sky the
residual of a star is nearly the same in every chunk, but it differs from star to star by about 0.1 mag (the errors
of the catalog magnitudes and the colours of the stars). The reference of every star (SkyReference) is the median
over the other inputs of the camera of its residual on clear sky in an input (a low quantile of its residuals
there), relative to the zero point of the input (the median of these residuals over its stars), so the transparency
of the inputs doesn't enter the reference. The input itself is used only for the stars without other inputs, so a
cloud during the whole input doesn't enter the references of the stars behind it. Clouds dim the stars behind them,
so a cloud shows up as a region of stars fainter than their references, and an opaque cloud as a region where stars
which are reliably seen on clear sky are missing.

The local photometric offset (the median excess over the references of the k nearest matched stars) and the local
fraction of the reliable stars which are seen are computed on a grid. A point of the image at a given frame is clear
if the chunk of the frame and the neighbouring chunks have enough stars, the local offset is below a limit (relative
to the clear part of the chunk, as a uniform thin haze is calibrated by the photometric offset of the chunk), and
most of the reliable stars around it are seen. Without enough reliable stars (e.g. the first inputs of a camera), the
distance to the k-th nearest matched star relative to the clearest chunks of the input is used instead.

The reference also keeps a log of the conditions (the clear fraction and the transparency of every chunk), by the
time of the observation, so the inputs can be processed in any order and again.
"""

from __future__ import print_function, division, absolute_import

import os
import csv
import collections
import copy
import glob
import math
import datetime
import warnings

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from RMS.Astrometry.ApplyAstrometry import (computeFOVSize, extinctionCorrectionTrueToApparent, photomLine,
                                            raDecToXYPP, xyToRaDecPP)
from RMS.Astrometry.ApplyRecalibrate import recalibratePlateparsForFF
from RMS.Astrometry.Conversions import datetime2JD
from RMS.Formats import StarCatalog
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Logger import getLogger
from RMS.Math import angularSeparationDeg

log = getLogger("logger")


# Smallest fraction of the stars of the chunk the platepar is fitted on which have to match the catalog for the
#   calibration to be trusted (on clear sky, most of the extracted stars match unambiguously within 1 px)
MIN_MATCHED_FRACTION = 0.15

# Smallest number of observations of a star in an input for its residual there to be kept (its clear-sky residual is
#   a low quantile of its residuals). A short input (e.g. 30 s of frames) has fewer chunks: there, a star has to be seen
#   in all its chunks, but in at least MIN_STAR_OBS_SHORT of them, so that short inputs also build the reference of
#   the camera (the median over the latest inputs of every star smooths their fewer observations)
MIN_STAR_OBS = 5
MIN_STAR_OBS_SHORT = 2

# Quantile of the residuals of a star in an input taken as its residual on clear sky there (%), and of the zero
#   points of the inputs taken as the zero point of clear sky (%)
STAR_QUANTILE = 25

# A star is reliable if it is seen (an extracted star within the match radius, also if it is not matched, e.g. with
#   saturated pixels) in at least this fraction of the chunks in which it is in the image on clear sky, and a region
#   is clouded if fewer than MIN_SEEN_FRACTION of the reliable stars around it which are expected to be seen (from
#   their fractions) are seen
RELIABLE_FRACTION = 0.9
MIN_SEEN_FRACTION = 0.5

# Smallest number of reliable stars in the image for the missing stars to be used (instead of the density of the
#   matched stars)
MIN_RELIABLE = 100



def minStarObs(n_chunks):
    """ The smallest number of observations of a star in an input of n_chunks used chunks for its residual there to
        be kept: MIN_STAR_OBS, or all the chunks of a shorter input, but at least MIN_STAR_OBS_SHORT.

    Arguments:
        n_chunks: [int] Number of the used chunks of the input.

    Return:
        [int]
    """

    return min(MIN_STAR_OBS, max(MIN_STAR_OBS_SHORT, int(n_chunks)))


def starKeys(ra, dec):
    """ Keys of catalog stars: their positions rounded to 0.001 deg, as integers (unique for the stars which are
        matched, as there is no other catalog star within 3 match radii of them).

    Arguments:
        ra, dec: [ndarray] Right ascensions and declinations (deg).

    Return:
        [ndarray] int64 keys.
    """

    # The right ascension (0-360 deg) and the declination shifted to 0-180 deg, in units of 0.001 deg; the
    #   declination takes the lower 6 digits of the key, which it never exceeds (180,000)
    ra_key = np.round(np.asarray(ra)*1000).astype(np.int64)
    dec_key = np.round((np.asarray(dec) + 90)*1000).astype(np.int64)

    return ra_key*1000000 + dec_key



class CalibrationError(Exception):
    """ The sky quality is not known: the platepar could not be fitted to the stars, there is no star catalog, or
        no star has a reference.
    """
    pass



def catalogNearField(catalog, platepar, jds):
    """ The catalog stars near the field of view at any of the given times: within 0.65 of the diagonal of the
        field (its half plus a margin of 30% of it) of the centre of the field at one of the times. An angular cut is
        much faster than projecting the whole catalog into the image.

    Arguments:
        catalog: [ndarray] Catalog stars, rows (ra, dec, mag) in degrees.
        platepar: [Platepar] Platepar of the camera.
        jds: [list] Julian dates.

    Return:
        [ndarray] Boolean, True for the stars near the field.
    """

    # The centre of the field at every time
    n = len(jds)
    _, ra_c, dec_c, _ = xyToRaDecPP(np.array(jds), np.full(n, platepar.X_res/2.0), np.full(n, platepar.Y_res/2.0),
                                    np.ones(n), platepar, extinction_correction=False, jd_time=True)

    # The largest distance from the centre: half the diagonal of the field (to its corners), plus 30% of it for
    #   the uncertainty of the size of the field and the motion of the sky between the times
    fov_h, fov_v = computeFOVSize(platepar)
    radius = 0.65*math.hypot(fov_h, fov_v)

    # A star is near the field if it is within the radius of the centre at any of the times
    near = np.zeros(len(catalog), dtype=bool)
    for ra, dec in zip(ra_c, dec_c):
        near |= angularSeparationDeg(catalog[:, 0], catalog[:, 1], ra, dec) < radius

    return near



def unixTime(dt):
    """ Seconds since 1970-01-01 of a naive UTC datetime. """

    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()



class SkyReference(object):

    # Number of inputs (visits) kept per star, and of zero points of the inputs, the latest ones by the time of the
    #   observation
    MAX_VISITS = 10
    MAX_ZERO_POINTS = 1000

    def __init__(self, dir_path, station_id):
        """ Clear-sky reference of a camera, kept across its inputs in its output directory:
            - sky_reference_<station>.npz: for every star (by its key, see starKeys) the clear-sky residual relative
              to the zero point of the input and the fraction of the chunks in which it was seen, of each of its
              latest inputs, with the time of the observation (the beginning of the input), and the zero points
              of the inputs (the median clear-sky residual of their stars) with their times,
            - sky_conditions_<station>_<YYYYMMDD>.csv: the conditions of every chunk (one file per UTC date of the
              observation), appended by every input. A chunk processed again has a newer row, the latest row of a
              chunk (by the processing time) is the valid one.

        The inputs can be processed in any order and by several processes: the files are updated under a lock, and
        the reference is replaced atomically.

        Arguments:
            dir_path: [str] Output directory of the camera.
            station_id: [str] Station code.
        """

        self.dir_path = dir_path
        self.station_id = station_id

        # The file of the reference of the stars, and the lock file of the updates (see update)
        self.stars_path = os.path.join(dir_path, 'sky_reference_{:s}.npz'.format(station_id))
        self.lock_path = os.path.join(dir_path, '.sky_reference_{:s}.lock'.format(station_id))

        self.load()


    def load(self):
        """ Read the reference of the stars (empty if there is none yet). """

        # The visits of the stars: a row per star (its key), and a column per visit with its time (unix s, NaN
        #   for an empty slot), clear-sky residual and seen fraction. The zero points of the inputs, with their
        #   times
        self.keys = np.zeros(0, dtype=np.int64)
        self.times = np.zeros((0, self.MAX_VISITS))
        self.values = np.zeros((0, self.MAX_VISITS), dtype=np.float32)
        self.rates = np.zeros((0, self.MAX_VISITS), dtype=np.float32)
        self.zp_times = np.zeros(0)
        self.zp_values = np.zeros(0)

        # A reference which can't be read (e.g. damaged by a full disk) is started again: it is rebuilt from the
        #   following inputs
        if os.path.isfile(self.stars_path):
            try:
                with np.load(self.stars_path) as data:
                    self.keys, self.times = data['keys'], data['times']
                    self.values, self.rates = data['values'], data['rates']
                    self.zp_times, self.zp_values = data['zp_times'], data['zp_values']
            except Exception as e:
                log.warning('Sky quality: the reference {:s} can\'t be read ({:s}), it is started again'.format(
                    self.stars_path, str(e)))

        # The row of every star by its key
        self.index = {int(k): i for i, k in enumerate(self.keys)}


    def zeroPoints(self, exclude_time=None):
        """ The zero points of the inputs.

        Keyword arguments:
            exclude_time: [float] The zero point at this time (unix s) is left out. None by default.

        Return:
            [ndarray] Zero points (mag).
        """

        # The times are those of the beginnings of the inputs, the same for every processing of an input (within a
        #   second, the resolution of the times kept)
        use = np.ones(len(self.zp_times), dtype=bool)
        if exclude_time is not None:
            use &= np.abs(self.zp_times - exclude_time) > 1.0

        return self.zp_values[use]


    def lookup(self, keys, exclude_time=None):
        """ The visits of stars.

        Arguments:
            keys: [ndarray] Keys of the stars.

        Keyword arguments:
            exclude_time: [float] The visits at this time (unix s) are left out (the earlier processing of the same
                input). None by default.

        Return:
            (values, rates): [ndarray] (stars, MAX_VISITS) arrays of the clear-sky residuals relative to the zero
                points of the inputs, and the seen fractions of the visits, NaN where there is no visit.
        """

        values = np.full((len(keys), self.MAX_VISITS), np.nan)
        rates = np.full((len(keys), self.MAX_VISITS), np.nan)
        for i, key in enumerate(keys):

            # A star which is not in the reference has no visits
            row = self.index.get(int(key))
            if row is None:
                continue

            # The filled slots of the star, without the visit at the excluded time
            use = np.isfinite(self.times[row])
            if exclude_time is not None:
                use &= np.abs(self.times[row] - exclude_time) > 1.0
            values[i, use] = self.values[row, use]
            rates[i, use] = self.rates[row, use]

        return values, rates


    def update(self, obs_time, keys, values, rates, zero_point, conditions):
        """ Add the visits of an input (replacing an earlier processing of it) and append its conditions.

        Arguments:
            obs_time: [datetime] Time of the beginning of the input (UTC, naive).
            keys, values, rates: [ndarray] Keys, clear-sky residuals relative to the zero point of the input, and
                seen fractions of its stars.
            zero_point: [float] Zero point of the input (NaN if not known).
            conditions: [list] Rows of the conditions of its chunks (see SkyQualityMap.conditions).
        """

        t = unixTime(obs_time)

        with _FileLock(self.lock_path):

            # The reference may have been updated by another process since it was read
            self.load()

            # The stars which are not in the reference yet get new rows, with empty slots
            keys_all, times, vals, rts = list(self.keys), [self.times], [self.values], [self.rates]
            new_rows = [k for k in keys if int(k) not in self.index]
            if new_rows:
                keys_all += [int(k) for k in new_rows]
                times.append(np.full((len(new_rows), self.MAX_VISITS), np.nan))
                vals.append(np.full((len(new_rows), self.MAX_VISITS), np.nan, dtype=np.float32))
                rts.append(np.full((len(new_rows), self.MAX_VISITS), np.nan, dtype=np.float32))
            # All rows as arrays, and the row of every star by its key
            keys_all = np.array(keys_all, dtype=np.int64)
            times, vals, rts = np.vstack(times), np.vstack(vals), np.vstack(rts)
            index = {int(k): i for i, k in enumerate(keys_all)}

            # An earlier processing of this input is removed completely (also the visits of the stars which are not
            #   in this processing)
            earlier = np.abs(times - t) <= 1.0
            times[earlier], vals[earlier], rts[earlier] = np.nan, np.nan, np.nan

            for key, value, rate in zip(keys, values, rates):
                row = index[int(key)]

                # The slot of this input: an empty slot, or the oldest visit if it is older than this input
                empty = np.nonzero(~np.isfinite(times[row]))[0]
                if len(empty):
                    slot = empty[0]
                else:
                    slot = int(np.argmin(times[row]))
                    if times[row, slot] > t:
                        continue

                # The visit of this input
                times[row, slot], vals[row, slot], rts[row, slot] = t, value, rate

            # The zero point of this input (replacing an earlier processing of it, also if it is not known now), the
            #   latest ones kept
            keep = np.abs(self.zp_times - t) > 1.0
            zp_times, zp_values = self.zp_times[keep], self.zp_values[keep]
            if np.isfinite(zero_point):
                zp_times, zp_values = np.r_[zp_times, t], np.r_[zp_values, zero_point]
                order = np.argsort(zp_times)[-self.MAX_ZERO_POINTS:]
                zp_times, zp_values = zp_times[order], zp_values[order]

            # Write the reference to a temporary file and replace the old one with it, so a process reading it never
            #   sees a partly written file
            tmp_path = self.stars_path + '.tmp.npz'
            np.savez_compressed(tmp_path, keys=keys_all, times=times, values=vals, rates=rts, zp_times=zp_times,
                                zp_values=zp_values)
            os.replace(tmp_path, self.stars_path)

            # The reference in memory is the one written
            self.keys, self.times, self.values, self.rates, self.index = keys_all, times, vals, rts, index
            self.zp_times, self.zp_values = zp_times, zp_values

            # Append the conditions to the file of the date of every chunk, with the time of the processing (UTC),
            #   which tells the rows of a chunk processed again apart
            processed = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
            for date, rows in _groupByDate(conditions):
                path = os.path.join(self.dir_path, 'sky_conditions_{:s}_{:s}.csv'.format(self.station_id, date))
                new_file = not os.path.isfile(path)
                with open(path, 'a') as f:

                    # A new file starts with a description and the header of the columns
                    if new_file:
                        f.write('# Sky conditions of the chunks of frames (RMS.Routines.SkyQuality). An input '
                                'processed again has newer rows (processed_utc), which are the valid ones\n')
                        f.write(','.join(CONDITION_COLUMNS) + ',processed_utc\n')
                    writer = csv.writer(f, lineterminator='\n')
                    for row in rows:
                        writer.writerow([row[c] for c in CONDITION_COLUMNS] + [processed])



# Columns of the log of the conditions: the time of the beginning of the chunk (UTC), the input file, the number of
#   extracted and matched stars, the fraction of the image which is clear, the transparency (the median excess of the
#   residuals of the stars over their clear-sky residuals, mag; positive is less transparent), and the fraction of
#   the reliable stars which are seen
CONDITION_COLUMNS = ('chunk_utc', 'input', 'stars', 'matched', 'clear_fraction', 'transparency_mag',
                     'reliable_seen')



def _groupByDate(conditions):
    """ The rows of the conditions grouped by the UTC date of their chunks (YYYYMMDD). """

    # The date is the beginning of the time of the chunk (YYYY-MM-DD), without the dashes
    groups = {}
    for row in conditions:
        groups.setdefault(row['chunk_utc'][:10].replace('-', ''), []).append(row)

    return sorted(groups.items())



class _FileLock(object):
    """ An exclusive lock on a file, for the updates of the reference by several processes (no lock where fcntl
        is not available).
    """

    def __init__(self, path):
        self.path = path
        self.f = None

    def __enter__(self):

        # The lock file is created if it doesn't exist (and never written to), and the call waits until the lock
        #   is free. fcntl is not available on Windows, where the updates are not locked
        self.f = open(self.path, 'a')
        try:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_EX)
        except ImportError:
            pass
        return self

    def __exit__(self, *args):

        # The lock is released, also if the update failed
        try:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_UN)
        except ImportError:
            pass
        self.f.close()



class SkyQualityMap(object):
    def __init__(self, star_list, platepar, config, begin_time, fps, chunk_frames, n_neighbours=10,
                 max_offset=0.15, max_radius_ratio=2.0, grid_cells=32, match_radius=1.0, matched=None,
                 total_frames=None, first_frames=None, min_matched=30, reference=None, input_name='',
                 first_frame=0):
        """ Local sky quality of the chunks of frames of one input.

        Arguments:
            star_list: [list] Stars of the chunks, [ff_name, [(y, x, intensity, amplitude, fwhm, background, snr,
                saturated), ...]] per chunk (see extractStarsFrameInterface).
            platepar: [Platepar] Platepar of the camera. It is recalibrated on the chunk with the most stars.
            config: [Config] Configuration object (the star catalog and ff_min_stars).
            begin_time: [datetime] Time of the first frame of the input (UTC, naive).
            fps: [float] Frames per second.
            chunk_frames: [int] Number of frames in a chunk, None for the spacing of the chunks.

        Keyword arguments:
            n_neighbours: [int] Number of nearest matched stars of a grid point. 10 by default.
            max_offset: [float] Largest local photometric offset of clear sky (mag). 0.15 by default.
            max_radius_ratio: [float] Largest distance to the k-th nearest matched star relative to the
                reference (a ratio of 2 is a quarter of the reference density). 2.0 by default.
            grid_cells: [int] Number of grid cells along the longer image side. 32 by default.
            match_radius: [float] Largest distance between a star and its catalog star (px). 1.0 by default.
            matched: [list] The matched stars of the chunks (see matchStars), instead of matching them. None by
                default.
            total_frames: [int] Number of frames of the input, for the whole chunks missing at its end. None by
                default (the frames after the last chunk belong to it).
            min_matched: [int] Smallest number of matched stars of the chunk the platepar is fitted on, for the
                calibration to be trusted. 30 by default.
            first_frames: [list] The first frame of every chunk of star_list (see chunkFirstFrames). None by
                default, the first frames are computed from the times of the chunks and the frame rate (which
                assumes no gaps in the recording).
            reference: [SkyReference] Clear-sky reference of the camera, from its other inputs. None by default
                (only this input is used).
            input_name: [str] Name of the input (its path), for the log of the conditions. Empty by default.
            first_frame: [int] First frame of the part of the input which the stars are from, e.g. of a segment
                (see RMS.Formats.FrameBuffer); with total_frames as its end. 0 by default (the whole input).
        """

        # The options
        self.config = config
        self.fps = fps
        self.k = n_neighbours
        self.max_offset = max_offset
        self.max_radius_ratio = max_radius_ratio
        self.match_radius = match_radius
        self.min_matched = min_matched
        self.reference = reference
        self.input_name = input_name
        self.begin_time = begin_time

        # The size of the image (unbinned)
        self.width, self.height = platepar.X_res, platepar.Y_res

        # The grid points are the centres of square cells
        self.cell = max(self.width, self.height)/grid_cells
        gx = np.arange(self.cell/2, self.width, self.cell)
        gy = np.arange(self.cell/2, self.height, self.cell)
        self.grid_shape = (len(gy), len(gx))
        grid_x, grid_y = np.meshgrid(gx, gy)
        self.grid = np.c_[grid_x.ravel(), grid_y.ravel()]

        # The first frame of every chunk, given or from its time (the entries of the star list without stars are
        #   left out)
        keep = [i for i, entry in enumerate(star_list) if len(entry) > 1]
        star_list = [star_list[i] for i in keep]
        if first_frames is not None:
            self.chunk_first = np.array([first_frames[i] for i in keep], dtype=np.int64)
        else:
            self.chunk_first = np.array([int(round((filenameToDatetime(entry[0]) - begin_time).total_seconds()*fps))
                                         for entry in star_list])

        # Chunks whose first frame is not known are left out (they are then missing, i.e. clouded)
        known = self.chunk_first >= 0
        star_list = [entry for entry, k in zip(star_list, known) if k]
        self.chunk_first = self.chunk_first[known]
        if matched is not None:
            matched = [matched[i] for i, k in zip(keep, known) if k]

        # The chunks in the order of their frames
        order = np.argsort(self.chunk_first)
        star_list = [star_list[i] for i in order]
        self.chunk_first = self.chunk_first[order]

        # The number of frames of a chunk: given (the chunks of the star extraction), or the spacing of the chunks
        if chunk_frames is None:
            chunk_frames = int(np.median(np.diff(self.chunk_first))) if len(self.chunk_first) > 1 else 128
        self.chunk_frames = max(int(chunk_frames), 1)

        # The matched stars of every chunk, as rows [x, y, residual, key], and the catalog stars in the image of
        #   every chunk, as rows [x, y, key, seen] (None if the matches are given)
        self.n_stars = np.array([len(entry[1]) for entry in star_list])
        self.chunk_times = [filenameToDatetime(entry[0]) for entry in star_list]
        self.in_image = None
        if matched is None:
            self.matched = self.matchStars(star_list, platepar)
        else:
            self.matched = [matched[i] for i in order]

        # Chunks without stars (e.g. skipped by the star extraction, or failed) are clouded: empty chunks are put
        #   in the gaps of the chunks, and before the first and after the last one
        self.addMissingChunks(total_frames, first_frame=first_frame)
        self.n_matched = np.array([len(m) for m in self.matched])

        # Which grid points of which chunks are clear
        self.clear = self.computeClear()


    def addMissingChunks(self, total_frames, first_frame=0):
        """ Add empty chunks (no stars, so clouded) where chunks are missing: in the gaps between the chunks,
            before the first chunk, and after the last one up to total_frames (also for a part of a chunk).

        Arguments:
            total_frames: [int] Number of frames of the input, or None.

        Keyword arguments:
            first_frame: [int] First frame of the input (or of its part). 0 by default.
        """

        cf = self.chunk_frames
        if len(self.chunk_first) == 0:
            return

        # The first frames of the chunks, with the missing ones (a chunk is missing where the next one begins
        #   more than half a chunk after the expected frame). Before the first chunk: the chunks back to the
        #   beginning of the input
        firsts = list(np.arange(self.chunk_first[0] - cf, first_frame - cf/2.0, -cf)[::-1].astype(int))
        for k, f0 in enumerate(self.chunk_first):

            # In the gap between this chunk and the previous one, every chunk which fits in it
            if k > 0:
                expected = self.chunk_first[k - 1] + cf
                while expected <= f0 - cf/2.0:
                    firsts.append(int(expected))
                    expected += cf
            firsts.append(int(f0))
        # After the last chunk, every whole chunk which fits in the remaining frames is missing. The frames at the
        #   end of the input, fewer than a chunk, belong to the chunk before them (the star extraction uses only
        #   whole chunks, so they never have stars of their own)
        if total_frames is not None:
            expected = self.chunk_first[-1] + cf
            while expected + cf <= total_frames:
                firsts.append(int(expected))
                expected += cf

        # Nothing is missing
        if len(firsts) == len(self.chunk_first):
            return

        # The stars of the existing chunks, none for the added ones. The time of an added chunk is taken at the
        #   frame rate from the nearest chunk present before it, so a gap in the recording before it is kept (from
        #   the beginning of the input before the first chunk)
        existing = dict(zip(self.chunk_first.tolist(), range(len(self.chunk_first))))
        self.matched = [self.matched[existing[f]] if f in existing else np.zeros((0, 4)) for f in firsts]
        self.n_stars = np.array([self.n_stars[existing[f]] if f in existing else 0 for f in firsts])

        def _addedTime(f):
            k = int(np.searchsorted(self.chunk_first, f, side='right')) - 1
            if k < 0:
                return self.begin_time + datetime.timedelta(seconds=f/self.fps)
            return self.chunk_times[k] + datetime.timedelta(seconds=(f - self.chunk_first[k])/self.fps)

        self.chunk_times = [self.chunk_times[existing[f]] if f in existing else _addedTime(f) for f in firsts]
        if self.in_image is not None:
            self.in_image = [self.in_image[existing[f]] if f in existing else np.zeros((0, 4)) for f in firsts]
        self.chunk_first = np.array(firsts)


    def matchStars(self, star_list, platepar):
        """ Match the stars of every chunk to the catalog and compute their photometric residuals.

        The platepar is recalibrated once, on the chunk with the most stars, and the catalog is projected into
        every chunk at its time. A star is matched if its nearest catalog star is within match_radius and there is
        no other catalog star within 3 match radii (blended stars have wrong intensities).

        Arguments:
            star_list: [list] Stars of the chunks, sorted by time.
            platepar: [Platepar] Platepar of the camera.

        The catalog stars in the image of every chunk are kept in self.in_image, as rows [x, y, key, seen]. A star
        is seen if an extracted star is within match_radius of it, also if it is not matched (e.g. a bright star with
        saturated pixels).

        Return:
            [list] Per chunk, an array of rows [x, y, residual, key] of the matched stars. The residual is the
                instrumental magnitude (with the zero point 0) minus the apparent catalog magnitude, positive for
                stars fainter than expected, and the key identifies the star (see starKeys).
        """

        matched = [np.zeros((0, 4)) for _ in star_list]
        self.in_image = [np.zeros((0, 4)) for _ in star_list]
        if not star_list:
            return matched

        # The star catalog, to the limiting magnitude of the recalibration
        years_from_J2000 = (filenameToDatetime(star_list[0][0])
                            - datetime.datetime(2000, 1, 1, 12, 0, 0)).total_seconds()/(365.25*24*3600)
        catalog = StarCatalog.readStarCatalog(self.config.star_catalog_path, self.config.star_catalog_file,
                                              years_from_J2000=years_from_J2000,
                                              lim_mag=self.config.catalog_mag_limit + 1,
                                              mag_band_ratios=self.config.star_catalog_band_ratios)
        if not catalog:
            raise CalibrationError('the star catalog could not be loaded')
        catalog = catalog[0]

        # Recalibrate the platepar on the chunk with the most stars (or the next ones, if it fails). The pointing
        #   of a fixed camera doesn't change within a file, so this one fit is used for all chunks
        calstars = {entry[0]: entry[1] for entry in star_list}
        # The chunk times of the recalibration are computed with the frame rate of the config, which is set to the
        #   one of the input
        config = copy.deepcopy(self.config)
        config.fps = self.fps
        pp = None
        for i_fit in np.argsort(-self.n_stars)[:3]:
            ff_name = star_list[i_fit][0]
            recalibrated = recalibratePlateparsForFF(platepar, [ff_name], calstars, catalog, config,
                                                     ff_frames=self.chunk_frames)
            if (ff_name in recalibrated) and recalibrated[ff_name].auto_recalibrated:
                pp = recalibrated[ff_name]
                break

        # Without a calibration, the stars can't be matched to the catalog
        if pp is None:
            raise CalibrationError('the platepar could not be recalibrated on the stars of the input')

        # Only the catalog stars near the field are projected for every chunk: the stars near the image at any
        #   of up to 21 times spread over the input
        step = max(len(star_list)//20, 1)
        jds = [datetime2JD(filenameToDatetime(entry[0])) for entry in star_list[::step] + [star_list[-1]]]
        catalog = catalog[catalogNearField(catalog, pp, jds)]
        if len(catalog) < 2:
            raise CalibrationError('no catalog stars in the field')
        catalog_keys = starKeys(catalog[:, 0], catalog[:, 1])

        for i, (ff_name, stars) in enumerate(star_list):

            stars = np.array(stars, dtype=np.float64)
            if len(stars) == 0:
                continue

            # The catalog at the time of the chunk (the middle of it)
            jd = datetime2JD(filenameToDatetime(ff_name)) + self.chunk_frames/(2*self.fps)/86400.0
            cat_x, cat_y = raDecToXYPP(catalog[:, 0], catalog[:, 1], jd, pp)

            # The catalog stars in the image (away from the edge, where the star extraction doesn't look), and
            #   whether they are seen
            inside = (cat_x > 10) & (cat_x < self.width - 10) & (cat_y > 10) & (cat_y < self.height - 10)
            seen = cKDTree(stars[:, [1, 0]]).query(np.c_[cat_x[inside], cat_y[inside]])[0] <= self.match_radius
            self.in_image[i] = np.c_[cat_x[inside], cat_y[inside], catalog_keys[inside], seen]

            # Unambiguous matches with a positive intensity, without saturated pixels (their intensities are too
            #   low)
            dist, idx = cKDTree(np.c_[cat_x, cat_y]).query(stars[:, [1, 0]], k=2)
            ok = (dist[:, 0] <= self.match_radius) & (dist[:, 1] > 3*self.match_radius) & (stars[:, 2] > 0)
            if stars.shape[1] > 7:
                ok &= stars[:, 7] == 0
            if not ok.any():
                continue
            cat = catalog[idx[ok, 0]]

            # The residuals: the instrumental magnitudes corrected for the vignetting, with the zero point 0 (so the
            #   reference doesn't depend on the photometric offset of the platepar), minus the catalog magnitudes
            #   corrected for the extinction
            app_mags = np.array(extinctionCorrectionTrueToApparent(cat[:, 2], cat[:, 0], cat[:, 1], jd, pp))
            radius = np.hypot(stars[ok, 0] - self.height/2, stars[ok, 1] - self.width/2)
            inst_mags = photomLine((stars[ok, 2], radius), 0.0, pp.vignetting_coeff)
            matched[i] = np.c_[stars[ok, 1], stars[ok, 0], inst_mags - app_mags, catalog_keys[idx[ok, 0]]]

        # A calibration which matches only a few of the stars of the clearest chunk (the one it was fitted on) is
        #   not trusted: with it, every chunk would look clouded. The clearest chunk has to match at least
        #   min_matched stars and MIN_MATCHED_FRACTION of its stars
        n_fit = len(matched[i_fit])
        if (n_fit < self.min_matched) or (n_fit < MIN_MATCHED_FRACTION*self.n_stars[i_fit]):
            raise CalibrationError('only {:d} of the {:d} stars of the clearest chunk match the catalog'.format(
                n_fit, int(self.n_stars[i_fit])))

        return matched


    def computeClear(self):
        """ Which grid points of which chunks are clear (see the module description).

        Return:
            [ndarray] Boolean array (chunks, grid points), True for clear sky.
        """

        # Nothing is clear until shown to be, and the transparency, the seen fraction of the reliable stars and the
        #   local offsets of every chunk are kept for the log of the conditions (NaN where not known)
        n_chunks = len(self.matched)
        clear = np.zeros((n_chunks, len(self.grid)), dtype=bool)
        self.transparency = np.full(n_chunks, np.nan)
        self.reliable_seen = np.full(n_chunks, np.nan)
        self.offset = np.full((n_chunks, len(self.grid)), np.nan)
        self.star_keys = np.zeros(0)

        # Chunks with too few stars (clouds, twilight) are not clear anywhere
        usable = (self.n_stars >= self.config.ff_min_stars) & (self.n_matched >= self.k)
        if not usable.any():
            return clear

        # The reference residual of every star, and the excess of the residuals over it (which includes the zero
        #   point of the chunk)
        self.star_keys, star_ref, star_seen, clear_zero_point = self.starReferences(usable)
        reliable = star_seen >= RELIABLE_FRACTION

        # Without the reference of any star (an input of a single usable chunk, and no reference of the camera
        #   yet) the sky quality is not known, rather than clouded everywhere
        if not np.isfinite(star_ref).any():
            raise CalibrationError('no star has a reference (too few chunks and no reference of the camera)')

        # The excess of every matched star over its reference (NaN for the stars without one), found by a binary
        #   search in the sorted keys
        excess = []
        for m in self.matched:
            ref = np.full(len(m), np.nan)
            if len(m) and len(self.star_keys):
                pos = np.clip(np.searchsorted(self.star_keys, m[:, 3]), 0, len(self.star_keys) - 1)
                found = self.star_keys[pos] == m[:, 3]
                ref[found] = star_ref[pos[found]]
            excess.append(m[:, 2] - ref)

        # The reliable stars and their seen fractions, to tell where stars are missing: the matched stars of this
        #   input, and the catalog stars in the image which are not matched in it but are reliable in the reference
        #   (behind a cloud which stays during the whole input)
        reliable_keys, reliable_seen = self.star_keys[reliable], star_seen[reliable]
        if (self.reference is not None) and (self.in_image is not None):
            # The catalog stars in the image of the usable chunks which are not matched in this input
            in_keys = np.unique(np.concatenate([self.in_image[i][:, 2] for i in np.nonzero(usable)[0]]))
            in_keys = in_keys[~np.isin(in_keys, self.star_keys)]
            if len(in_keys):

                # Their seen fractions in the other inputs; the reliable ones are added (sorted by key)
                _, ref_rates = self.reference.lookup(in_keys, exclude_time=unixTime(self.begin_time))
                with np.errstate(all='ignore'), warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    ref_seen = np.nanmedian(ref_rates, axis=1)
                more = np.isfinite(ref_seen) & (ref_seen >= RELIABLE_FRACTION)
                reliable_keys = np.r_[reliable_keys, in_keys[more]]
                reliable_seen = np.r_[reliable_seen, ref_seen[more]]
                order = np.argsort(reliable_keys)
                reliable_keys, reliable_seen = reliable_keys[order], reliable_seen[order]

        # The distance to the k-th nearest matched star of every grid point, with its reference (the densest
        #   chunks), for the chunks without enough reliable stars
        radius = np.full((n_chunks, len(self.grid)), np.nan)
        for i in np.nonzero(usable)[0]:
            radius[i] = cKDTree(self.matched[i][:, :2]).query(self.grid, k=self.k)[0][:, -1]
        radius_ref = np.nanpercentile(radius, 20, axis=0)

        # Every usable chunk (the others stay clouded everywhere)
        for i in np.nonzero(usable)[0]:

            # The local offset: the median excess of the k nearest matched stars with a clear-sky residual
            has_ref = np.isfinite(excess[i])
            if has_ref.sum() < self.k:
                continue
            idx = cKDTree(self.matched[i][has_ref, :2]).query(self.grid, k=self.k)[1]
            offset = np.median(excess[i][has_ref][idx], axis=1)
            self.offset[i] = offset

            # The transparency of the chunk (its zero point relative to the one of clear sky, for the log), and the
            #   offset relative to the clear part of the chunk: a uniform haze changes the zero point of the whole
            #   chunk, which the photometric calibration of the chunk takes care of
            self.transparency[i] = np.median(excess[i][has_ref]) - clear_zero_point
            flagged = (offset - np.percentile(offset, 20)) > self.max_offset

            # Opaque clouds: the reliable stars in the image which are not seen. Without enough of them, the
            #   matched stars much sparser than in the densest chunks
            expected = None
            if self.in_image is not None:
                expected = self.in_image[i][np.isin(self.in_image[i][:, 2], reliable_keys)]
            if (expected is not None) and (len(expected) >= MIN_RELIABLE):

                # The stars seen around every grid point, relative to the number expected from their seen fractions
                rate = reliable_seen[np.searchsorted(reliable_keys, expected[:, 2])]
                seen = expected[:, 3]
                idx = cKDTree(expected[:, :2]).query(self.grid, k=self.k)[1]
                flagged |= seen[idx].sum(axis=1) < MIN_SEEN_FRACTION*rate[idx].sum(axis=1)
                self.reliable_seen[i] = seen.sum()/rate.sum()
            else:
                flagged |= radius[i] > self.max_radius_ratio*radius_ref

            # The offsets are smoothed over the k nearest stars, which spread over several grid cells, so a cloud
            #   flags a group of cells. A single flagged cell is noise (e.g. at the image edge, with fewer stars),
            #   and it is not flagged
            flagged = flagged.reshape(self.grid_shape)
            neighbours = ndimage.convolve(flagged.astype(np.int32), np.ones((3, 3), dtype=np.int32),
                                          mode='constant')
            flagged &= neighbours >= 2

            clear[i] = ~flagged.ravel()

        # Kept for the diagnostics
        self.radius, self.radius_ref, self.excess = radius, radius_ref, excess

        return clear


    def starReferences(self, usable):
        """ The reference residual of every star of this input (relative to the zero point of an input), whether it
            is reliably seen on clear sky and how often, and the zero point of clear sky.

        The residual of a star in an input is the STAR_QUANTILE quantile of its residuals in the usable chunks
        (with at least MIN_STAR_OBS of them, or all of them for an input of fewer chunks, see minStarObs),
        relative to the zero point of the input (the median of these over the stars). Its reference is the median
        of its residuals in the other inputs in the reference, or, without them, its residual in this input (which
        a cloud during the whole input would bias towards the cloud). A star is reliable if it is seen in at least
        RELIABLE_FRACTION of the chunks in which it is in the image, in the median of the other inputs (or in this
        one, without them). The zero point of clear sky is the STAR_QUANTILE quantile of the zero points of the
        other inputs (or the one of this input, without them).

        Arguments:
            usable: [ndarray] Boolean, the usable chunks.

        Return:
            (keys, residuals, rates, zero_point): sorted keys of the matched stars, their reference residuals (NaN
                if not known), their seen fractions (NaN if not known), and the zero point of clear sky.
        """

        # The visits of this input: the residuals of its stars relative to its zero point (the median of the
        #   residuals of its stars), so they can be compared with the visits of other inputs, whatever the
        #   transparency of the sky was
        keys, residuals, seen = self.inputVisits(usable)
        zero_point = np.nanmedian(residuals) if np.isfinite(residuals).any() else np.nan
        residuals = residuals - zero_point
        zero_points = np.array([zero_point])

        # The medians over the visits of the other inputs (the earlier processing of this one left out), where
        #   there are any
        if (self.reference is not None) and len(keys):
            t = unixTime(self.begin_time)
            ref_values, ref_rates = self.reference.lookup(keys, exclude_time=t)
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                ref_residuals = np.nanmedian(ref_values, axis=1)
                ref_seen = np.nanmedian(ref_rates, axis=1)
            # The visits of the other inputs replace the one of this input (which a cloud during the whole input
            #   would bias), for the stars which have them
            residuals = np.where(np.isfinite(ref_residuals), ref_residuals, residuals)
            seen = np.where(np.isfinite(ref_seen), ref_seen, seen)

            # The zero points of the other inputs, if there are any
            ref_zero_points = self.reference.zeroPoints(exclude_time=t)
            if np.isfinite(ref_zero_points).any():
                zero_points = ref_zero_points

        # The zero point of clear sky: a low quantile of the zero points, as clouds only make the stars fainter
        #   (the zero point larger)
        zero_points = zero_points[np.isfinite(zero_points)]
        clear_zero_point = np.percentile(zero_points, STAR_QUANTILE) if len(zero_points) else np.nan

        return keys, residuals, seen, clear_zero_point


    def inputVisits(self, use, clear=None):
        """ The visit of this input of every matched star: the STAR_QUANTILE quantile of its residuals, and the
            fraction of the chunks in which it is in the image in which it is seen.

        Arguments:
            use: [ndarray] Boolean, the chunks which are used.

        Keyword arguments:
            clear: [ndarray] The clear grid points of the chunks: only the observations on clear sky are used. None
                by default (all).

        Return:
            (keys, values, rates): sorted keys, residuals (NaN with fewer observations than minStarObs) and seen
                fractions (NaN without the catalog stars in the image).
        """

        # The observations of the stars in the used chunks (on clear sky), as rows [residual, key]
        obs = []
        for i in np.nonzero(use)[0]:
            m = self.matched[i]
            if clear is not None:
                m = m[clear[i, self.gridIndex(m[:, 0], m[:, 1])]]
            obs.append(m[:, 2:4])
        obs = np.vstack(obs) if obs else np.zeros((0, 2))
        if len(obs) == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)

        # The residuals of every star: the observations sorted by star (the stable sort keeps their order in time),
        #   so the observations of the star j are order[bounds[j]:bounds[j + 1]]
        keys, inverse, counts = np.unique(obs[:, 1], return_inverse=True, return_counts=True)
        order = np.argsort(inverse, kind='stable')
        bounds = np.r_[0, np.cumsum(counts)]
        values = np.full(len(keys), np.nan)
        for j in np.nonzero(counts >= minStarObs(np.count_nonzero(use)))[0]:
            values[j] = np.percentile(obs[order[bounds[j]:bounds[j + 1]], 0], STAR_QUANTILE)

        # The fraction of the chunks in which every star is in the image (on clear sky) in which it is seen
        rates = np.full(len(keys), np.nan)
        if self.in_image is not None:
            n_in, n_seen = np.zeros(len(keys)), np.zeros(len(keys))
            for i in np.nonzero(use)[0]:
                # The catalog stars in the image of the chunk (on clear sky)
                inside = self.in_image[i]
                if clear is not None:
                    inside = inside[clear[i, self.gridIndex(inside[:, 0], inside[:, 1])]]

                # Counted for the matched stars of this input (the keys are sorted, so the row of a star is found by
                #   a binary search)
                pos = np.clip(np.searchsorted(keys, inside[:, 2]), 0, len(keys) - 1)
                hit = keys[pos] == inside[:, 2]
                np.add.at(n_in, pos[hit], 1)
                np.add.at(n_seen, pos[hit], inside[hit, 3])

            with np.errstate(all='ignore'):
                rates = np.where(n_in > 0, n_seen/n_in, np.nan)

        return keys, values, rates


    def conditions(self):
        """ Rows of the log of the conditions of the chunks (see CONDITION_COLUMNS). """

        # One row per chunk (also the missing ones, which are clouded), the values rounded to 0.001 (NaN where they
        #   are not known, e.g. without the reliable stars)
        clear = self.summary()
        rows = []
        for i, t in enumerate(self.chunk_times):
            rows.append({'chunk_utc': t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
                         'input': self.input_name,
                         'stars': int(self.n_stars[i]),
                         'matched': int(self.n_matched[i]),
                         'clear_fraction': '{:.3f}'.format(clear[i]),
                         'transparency_mag': '{:.3f}'.format(self.transparency[i]),
                         'reliable_seen': '{:.3f}'.format(self.reliable_seen[i])})

        return rows


    def updateReference(self):
        """ Add the stars of this input on clear sky to the reference, and its conditions to the log. """

        if self.reference is None:
            return

        # The visits of the stars of this input from their observations on clear sky only, so the clouds of this
        #   input don't enter the reference
        use = self.clear.any(axis=1)
        keys, values, rates = self.inputVisits(use, clear=self.clear)
        keep = np.isfinite(values)

        # The residuals relative to the zero point of this input (the median of its stars), see starReferences
        zero_point = np.median(values[keep]) if keep.any() else np.nan
        self.reference.update(self.begin_time, keys[keep], values[keep] - zero_point, rates[keep], zero_point,
                              self.conditions())


    def chunkIndex(self, frames):
        """ The chunk of every frame (the nearest one for frames before the first and after the last chunk). """

        # The last chunk which begins at or before the frame
        idx = np.searchsorted(self.chunk_first, np.asarray(frames), side='right') - 1
        return np.clip(idx, 0, len(self.chunk_first) - 1)


    def gridIndex(self, x, y):
        """ The grid cell of every image point. """

        # The cells are square, the points outside the image are in the nearest cell, and the index runs along the
        #   rows of the grid (as the grid points)
        ix = np.clip((np.asarray(x)/self.cell).astype(int), 0, self.grid_shape[1] - 1)
        iy = np.clip((np.asarray(y)/self.cell).astype(int), 0, self.grid_shape[0] - 1)
        return iy*self.grid_shape[1] + ix


    def isClear(self, x, y, frames):
        """ Whether the sky is clear at image points at given frames: clear in the chunk of the frame and in the
            neighbouring chunks (a cloud moves during a chunk).

        Arguments:
            x, y: [ndarray] Image coordinates (unbinned).
            frames: [ndarray] Frames.

        Return:
            [ndarray] Boolean, True for clear sky.
        """

        # Without chunks there is no clear sky
        if len(self.chunk_first) == 0:
            return np.zeros(np.shape(x), dtype=bool)

        # The cell of every point in its chunk and in the chunks before and after it (the first and the last chunk
        #   are their own neighbours)
        chunk = self.chunkIndex(frames)
        cell = self.gridIndex(x, y)
        ok = np.ones(np.shape(x), dtype=bool)
        for d in (-1, 0, 1):
            ok &= self.clear[np.clip(chunk + d, 0, len(self.chunk_first) - 1), cell]

        return ok


    def cloudMask(self, first_frame, last_frame, height, width, bin_factor=1):
        """ The image regions which are not clear in any chunk of a range of frames.

        Arguments:
            first_frame, last_frame: [int] The range of frames (inclusive).
            height, width: [int] Size of the mask (binned).

        Keyword arguments:
            bin_factor: [int] Binning of the mask. 1 by default.

        Return:
            [ndarray] Boolean mask (height, width), True where the sky is not clear.
        """

        # Without chunks there is no clear sky
        if len(self.chunk_first) == 0:
            return np.ones((height, width), dtype=bool)

        # The chunks overlapping the frames and their neighbours (as in isClear), sampled every half chunk so no
        #   chunk is skipped; a cell is cloudy if it is not clear in any of them
        chunks = np.unique(self.chunkIndex(np.arange(first_frame - self.chunk_frames,
                                                     last_frame + self.chunk_frames + 1,
                                                     max(self.chunk_frames//2, 1))))
        cloudy = ~np.all(self.clear[chunks], axis=0).reshape(self.grid_shape)

        # Every grid cell to the pixels of the mask
        ys = np.minimum((np.arange(height)*bin_factor/self.cell).astype(int), self.grid_shape[0] - 1)
        xs = np.minimum((np.arange(width)*bin_factor/self.cell).astype(int), self.grid_shape[1] - 1)

        return cloudy[np.ix_(ys, xs)]


    def summary(self):
        """ Fraction of the clear sky per chunk, for the log. """

        # The fraction of the clear grid cells of every chunk
        return self.clear.mean(axis=1) if len(self.clear) else np.zeros(0)



def filterClouded(detections, sky_map, min_rows):
    """ Remove the measurements of the detections which are not on clear sky, and the detections left with too
        few measurements.

    Arguments:
        detections: [list] Detections as [rho, theta, centroids], centroids rows [frame, x, y, ...].
        sky_map: [SkyQualityMap] Sky quality of the input.
        min_rows: [int] Smallest number of measurements of a detection.

    Return:
        (detections, n_rows_removed, n_detections_removed)
    """

    kept, n_rows, n_dets = [], 0, 0
    for det in detections:

        # The measurements on clear sky, at their positions and frames
        cent = det[2]
        ok = sky_map.isClear(cent[:, 1], cent[:, 2], cent[:, 0])
        n_rows += int((~ok).sum())

        # A detection with too few measurements left is removed, the others keep their measurements on clear sky
        if ok.sum() < min_rows:
            n_dets += 1
            continue
        kept.append([det[0], det[1], cent[ok]] + list(det[3:]))

    return kept, n_rows, n_dets



def chunkFirstFrames(star_list, img_handle, frame_range=None):
    """ The first frame of every chunk of the stars: the frame of the input at the time of the chunk (the time in
        its name), among the first frames of the chunks of the input. With timestamped frames, the frames after a
        gap in the recording are not where the time and the frame rate would put them.

    Arguments:
        star_list: [list] Stars of the chunks, [ff_name, stars] per chunk.
        img_handle: [FrameInterface] The input of the stars, with its chunk_frames.

    Keyword arguments:
        frame_range: [tuple] (first frame, end frame) of the part of the input which the stars are from (e.g. a
            segment, see RMS.Formats.FrameBuffer). None by default (the whole input).

    Return:
        [list] The first frame of every chunk, -1 for a chunk whose time is not the time of a chunk of the input.
    """

    # The times are relative to the beginning of the input
    chunk_frames = img_handle.chunk_frames
    begin = img_handle.beginning_datetime.replace(tzinfo=None)

    # The times of the first frames of all chunks of the input, or of its part (s from the beginning). The chunks
    #   begin at the multiples of chunk_frames, unless the input gives where they begin
    first, end = frame_range if (frame_range is not None) else (0, img_handle.total_frames)
    if hasattr(img_handle, 'chunk_first_frames'):
        firsts = np.array([f for f in img_handle.chunk_first_frames if first <= f < end], dtype=int)
    else:
        firsts = np.arange(-(-first//chunk_frames)*chunk_frames, end, chunk_frames)
    times = np.array([(img_handle.currentFrameTime(frame_no=int(f), dt_obj=True).replace(tzinfo=None)
                       - begin).total_seconds() for f in firsts])

    # Every chunk of the stars at the nearest chunk of the input, if within half a frame of it (the names have a
    #   resolution of 1 ms)
    result = []
    for entry in star_list:
        t = (filenameToDatetime(entry[0]) - begin).total_seconds()
        k = int(np.argmin(np.abs(times - t)))
        result.append(int(firsts[k]) if abs(times[k] - t) <= 0.5/img_handle.fps + 0.001 else -1)

    return result



### Conditions of a night ###

# Clear fraction of a chunk at and above which it is shown as clear, and below which as partly clouded (a chunk
#   with no clear grid cell is clouded)
CLEAR_FRACTION_CLEAR = 0.999


def readConditions(dir_path, station_id, inputs=None, start=None, end=None):
    """ Read the log of the conditions of the chunks (see SkyReference). Only the rows of the newest processing of
        every input are taken, so an input which was processed again isn't counted twice.

    Arguments:
        dir_path: [str] Directory of the log (the output directory of the camera).
        station_id: [str] Station code.

    Keyword arguments:
        inputs: [set] Names of the inputs whose chunks are read (as in the log), None for all. None by default.
        start: [datetime] Beginning of the time of the chunks which are read (UTC, naive), only the logs of the
            dates from it on are read. None by default.
        end: [datetime] End of the time of the chunks which are read (UTC, naive), only the logs of the dates up to
            it are read. None by default.

    Return:
        rows: [list] Rows sorted by time, dicts with the columns of the log (numbers as floats) and the time of
            the chunk as a datetime (UTC, naive) in 'time'.
    """

    # Find the logs, one per UTC date
    log_paths = sorted(glob.glob(os.path.join(dir_path, 'sky_conditions_{:s}_*.csv'.format(station_id))))

    # Only read the logs of the dates which are requested
    if (start is not None) or (end is not None):
        first = start.strftime('%Y%m%d') if start is not None else ''
        last = end.strftime('%Y%m%d') if end is not None else '99999999'
        log_paths = [path for path in log_paths if first <= path[-12:-4] <= last]


    columns = list(CONDITION_COLUMNS) + ['processed_utc']

    # Group the rows of every input by the time they were processed
    processings = {}
    for log_path in log_paths:

        # Read the lines of the log without the comments (bytes which are not text don't stop the reading)
        with open(log_path, newline='', encoding='utf-8', errors='replace') as f:
            lines = [line for line in f if line.strip() and not line.startswith('#')]

        for values in csv.reader(lines):

            # Skip the header and the lines which were not written completely
            if (len(values) != len(columns)) or (values[0] == columns[0]):
                continue

            # Skip the inputs which were not requested
            row = dict(zip(columns, values))
            if (inputs is not None) and (row['input'] not in inputs):
                continue

            # Convert the values, skip the row if they can't be read
            try:
                for name in ('stars', 'matched', 'clear_fraction', 'transparency_mag', 'reliable_seen'):
                    row[name] = float(row[name])
                row['time'] = datetime.datetime.strptime(row['chunk_utc'], '%Y-%m-%dT%H:%M:%S.%f')
            except ValueError:
                continue

            processings.setdefault(row['input'], {}).setdefault(row['processed_utc'], []).append(row)


    # Take the rows of the newest processing of every input, within the requested time
    rows = []
    for by_time in processings.values():
        for row in by_time[max(by_time)]:
            if ((start is None) or (row['time'] >= start)) and ((end is None) or (row['time'] <= end)):
                rows.append(row)

    return sorted(rows, key=lambda row: (row['time'], row['input']))


def _runningMedian(times, values, half_window):
    """ Running median of a time series, ignoring the values which are not finite.

    Arguments:
        times: [array] Times of the values (s), sorted.
        values: [array] Values.
        half_window: [float] Half of the length of the window (s).

    Return:
        out: [ndarray] Median within the window around every value, NaN where there are no finite values.
    """

    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    # Find the first and the last value within the window of every value
    lo = np.searchsorted(times, times - half_window, side='left')
    hi = np.searchsorted(times, times + half_window, side='right')

    # Compute the median of the finite values in every window
    out = np.full(len(values), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        window = values[a:b]
        window = window[np.isfinite(window)]
        if len(window):
            out[i] = np.median(window)

    return out


def _spans(times, labels, max_gap, chunk):
    """ Join consecutive chunks with the same label into spans of time. A chunk lasts until the next one, but a
        gap longer than max_gap is a pause in the data, so the chunk before it (and the last chunk) only lasts
        for the time of one chunk.

    Arguments:
        times: [list] Times of the chunks (datetime), sorted.
        labels: [list] Label of every chunk.
        max_gap: [timedelta] Longest time between chunks which is not a pause.
        chunk: [timedelta] Time of one chunk.

    Return:
        spans: [list] Spans as (label, start, end).
    """

    spans = []
    for i, (t, label) in enumerate(zip(times, labels)):

        # The chunk lasts until the next one, unless there is a pause after it
        end = times[i + 1] if (i + 1 < len(times)) and (times[i + 1] - t <= max_gap) else t + chunk

        # Extend the current span if the label is the same and there was no pause, otherwise start a new one
        if spans and (spans[-1][0] == label) and (spans[-1][2] >= t):
            spans[-1][2] = end
        else:
            spans.append([label, t, end])

    return [tuple(span) for span in spans]


def _stateSegments(intervals, order):
    """ Combine the intervals of several states into one time line. Intervals can overlap (e.g. two recordings
        of the same time), and where they do, the state which is first in the order is taken.

    Arguments:
        intervals: [list] Intervals as (state, start, end).
        order: [list] States, ordered from the one which is taken first.

    Return:
        segments: [list] Segments as (state, start, end) which don't overlap, in time order, with neighbouring
            segments of the same state joined.
    """

    # List the beginnings and the ends of the intervals in time order. At the same time an end comes first, so
    #   touching intervals are not counted as overlapping
    events = sorted([(start, 1, state) for state, start, end in intervals if end > start]
                    + [(end, -1, state) for state, start, end in intervals if end > start],
                    key=lambda event: (event[0], event[1]))

    segments = []
    active = collections.Counter()
    for i, (t, change, state) in enumerate(events):

        # Count the open intervals of every state
        active[state] += change

        # Add the time until the next event (if it is later), with the first state in the order which is open
        if (i + 1 < len(events)) and (events[i + 1][0] > t):
            current = next((st for st in order if active[st] > 0), None)

            # Skip the time when no interval is open
            if current is None:
                continue

            # Extend the last segment if it has the same state and ends here, otherwise start a new one
            if segments and (segments[-1][0] == current) and (segments[-1][2] == t):
                segments[-1][2] = events[i + 1][0]
            else:
                segments.append([current, t, events[i + 1][0]])

    return [tuple(segment) for segment in segments]


def _withGaps(times, values, max_gap):
    """ Insert a NaN into every gap of a time series which is longer than max_gap, so the line of the plot is
        broken there instead of joining the data across the gap.

    Arguments:
        times: [list] Times of the values (datetime), sorted.
        values: [array] Values.
        max_gap: [timedelta] Longest gap which is not broken.

    Return:
        (out_t, out_v): [tuple] Times and values with the NaNs inserted.
    """

    out_t, out_v = [], []
    for i, (t, v) in enumerate(zip(times, values)):

        # Insert a NaN just after the previous value if there is a gap before this one
        if i and (t - times[i - 1] > max_gap):
            out_t.append(times[i - 1] + max_gap)
            out_v.append(np.nan)

        out_t.append(t)
        out_v.append(v)

    return out_t, np.array(out_v, dtype=np.float64)


def _legendPlace(fig, ax, handles):
    """ Find where to put the legend on the right side of a panel (upper, center or lower), so it covers as few
        of the plotted points as possible.

    Arguments:
        fig: [Figure] The figure.
        ax: [Axes] The panel.
        handles: [list] Handles of the legend.

    Return:
        loc: [str] Location of the legend. The upper right if it can't be found, as the place of the legend is
            only cosmetic.
    """

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    try:
        renderer = FigureCanvasAgg(fig).get_renderer()

        # Collect the plotted points in display coordinates. Lines of only a few points (e.g. a horizontal line)
        #   are resampled along their length, so they count where they go and not only at their ends
        points = []
        for line in ax.get_lines():
            xy = np.asarray(line.get_xydata(), dtype=np.float64)
            if not len(xy):
                continue
            if len(xy) < 50:
                xy = np.column_stack([np.interp(np.linspace(0, len(xy) - 1, 200), np.arange(len(xy)), xy[:, k])
                                      for k in range(2)])
            xy = line.get_transform().transform(xy)
            points.append(xy[np.all(np.isfinite(xy), axis=1)])
        points = np.concatenate(points) if points else np.zeros((0, 2))

        # Try every place and count the points the legend would cover
        best = None
        for loc in ('upper right', 'center right', 'lower right'):
            legend = ax.legend(handles=handles, loc=loc, fontsize=9, handlelength=2.2)
            box = legend.get_window_extent(renderer)
            legend.remove()
            covered = int(np.sum((points[:, 0] >= box.x0) & (points[:, 0] <= box.x1)
                                 & (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1)))
            if (best is None) or (covered < best[0]):
                best = (covered, loc)

        return best[1]

    except Exception as e:
        log.debug("The place of the legend could not be found: {:s}".format(repr(e)))
        return 'upper right'


def nightTimeline(rows, not_searched=(), chunk_seconds=None, bounds=None, clear_level=CLEAR_FRACTION_CLEAR,
                  input_ends=None):
    """ Build the time line of the state of the sky through a night. Every chunk is clear, partly clouded or
        clouded (see CLEAR_FRACTION_CLEAR) until the next chunk of its input, the inputs which were not searched
        (too few stars) fill their time, and where inputs overlap the worse state is taken (clouded, not
        searched, partly clouded, clear). The time between them without any data is a gap.

    Arguments:
        rows: [list] Conditions of the chunks in time order, see readConditions.

    Keyword arguments:
        not_searched: [list] Spans of the inputs which were not searched (start, end, times, stars).
        chunk_seconds: [dict] Time of the chunks of the inputs (s) by their names, from the spacing of their chunks
            for the inputs which are not in it. None by default.
        bounds: [tuple] Beginning and end of the night (datetime, UTC), the time of the chunks is taken within them.
            None by default.
        clear_level: [float] Clear fraction of a chunk at and above which it is clear. CLEAR_FRACTION_CLEAR by
            default.
        input_ends: [dict] End of the inputs (datetime, UTC) by their names: the last chunk lasts until it, as the
            frames after the last chunk (fewer than a chunk, which the star extraction doesn't use) belong to it.
            None by default (the last chunk lasts for the time of a chunk).

    Return:
        timeline: [dict] input_rows (the indices of the rows of every input), steps (the time of the chunks of
            every input), chunk (the time of a chunk of all inputs), max_gap (the longest time between chunks
            which is not a gap), labels (the state of every chunk), segments (state, start, end), gaps (start,
            end), summary (the observed time in hours, and its fractions in every state), and ends (the end of
            the last chunk of every input).
    """

    times = [row['time'] for row in rows]
    clear = np.array([row['clear_fraction'] for row in rows], dtype=np.float64)

    # Group the rows by input (in time order, as the rows)
    input_rows = collections.defaultdict(list)
    for i, row in enumerate(rows):
        input_rows[row.get('input')].append(i)


    ### Time of the chunks of every input ###

    # The time of a chunk is given for every input (from its frame rate) when it is known. Otherwise it is the
    #   shortest time between its chunks, as the chunks are regular but an input of a few chunks can have a
    #   pause between them
    chunk_seconds = chunk_seconds or {}

    def _spacing(idx):
        dt = [(times[j] - times[i]).total_seconds() for i, j in zip(idx[:-1], idx[1:])]
        dt = [d for d in dt if d > 0]
        return (float(min(dt)), len(dt)) if dt else (None, 0)

    spacings = {name: _spacing(idx) for name, idx in input_rows.items()}
    steps = {name: (chunk_seconds[name] if (chunk_seconds.get(name) or 0) > 0 else spacings[name][0])
             for name in input_rows}

    # The typical time of a chunk, from the inputs where it is reliable
    known = [st for name, st in steps.items()
             if (st is not None) and ((chunk_seconds.get(name) or 0) > 0 or (spacings[name][1] > 1))]
    step = float(np.median(known)) if known else 4.0

    # An input of one chunk takes the typical time, and an input of two chunks at most the typical time (its one
    #   spacing could be a pause)
    for name, st in steps.items():
        if st is None:
            steps[name] = step
        elif not (chunk_seconds.get(name) or 0) > 0 and (spacings[name][1] == 1) and known:
            steps[name] = min(st, step)
    steps = {name: datetime.timedelta(seconds=st) for name, st in steps.items()}
    chunk = datetime.timedelta(seconds=step)

    # A gap of more than a few chunks is a pause in the data
    max_gap = 3*max(list(steps.values()) + [chunk])

    ### ###


    # Classify every chunk by how much of the image is clear
    labels = np.where(clear >= clear_level, 'clear', np.where(clear > 0, 'partly', 'clouded'))

    # Turn the chunks of every input into spans of time, and add the inputs which were not searched
    input_ends = input_ends or {}
    intervals, ends = [], {}
    for name, idx in input_rows.items():
        spans = _spans([times[i] for i in idx], [labels[i] for i in idx], 3*steps[name], steps[name])

        # The last chunk lasts until the end of the input, if that is after its beginning and less than a chunk
        #   after its own end (it is shorter in an input shorter than a chunk)
        end, last = input_ends.get(name), times[idx[-1]]
        ends[name] = last + steps[name]
        if spans and (end is not None) and (last < end <= last + 2*steps[name]):
            spans[-1] = spans[-1][:2] + (end,)
            ends[name] = end
        intervals += spans
    intervals += [('not_searched', start, end) for start, end, _, _ in not_searched]

    # Only take the time within the night (a chunk can begin before it or end after it)
    if bounds is not None:
        intervals = [(state, max(start, bounds[0]), min(end, bounds[1])) for state, start, end in intervals]

    # Combine the overlapping inputs into one time line, the worse state is taken
    segments = _stateSegments(intervals, ['clouded', 'not_searched', 'partly', 'clear'])

    # Sum up the observed time and the fraction of it in every state
    durations = collections.defaultdict(float)
    for state, start, end in segments:
        durations[state] += (end - start).total_seconds()
    observed_s = sum(durations.values())
    summary = {'observed_h': observed_s/3600.0}
    for state in ('clear', 'partly', 'clouded', 'not_searched'):
        summary[state] = durations[state]/observed_s if observed_s else 0.0

    # Find the time without data (e.g. no recording, or the files were not processed)
    gaps = [(a[2], b[1]) for a, b in zip(segments[:-1], segments[1:]) if b[1] - a[2] > max_gap]

    return {'input_rows': input_rows, 'steps': steps, 'chunk': chunk, 'max_gap': max_gap, 'labels': labels,
            'segments': segments, 'gaps': gaps, 'summary': summary, 'ends': ends}


def plotSkyConditions(rows, plot_path, title, not_searched=(), detection_times=(), removed_detections=0,
                      cloud_max_offset=None, min_matched=None, chunk_seconds=None, bounds=None, input_ends=None,
                      removed_times=(), detection_mags=None):
    """ Plot the sky conditions of a night: the dimming of the stars, the clear fraction of the image and the stars
        through the night, with the time when the sky was clouded and partly clouded shaded, and the times of
        the detections.

    Arguments:
        rows: [list] Conditions of the chunks, see readConditions.
        plot_path: [str] Path of the saved plot.
        title: [str] Title of the plot.

    Keyword arguments:
        not_searched: [list] Spans (start, end) of the inputs which were not searched, as they had too few stars,
            with the stars of their chunks: (start, end, times, stars).
        detection_times: [list] Times of the detections (datetime, UTC).
        removed_detections: [int] Number of the detections removed on clouded sky.
        cloud_max_offset: [float] Dimming above which the sky is clouded (mag), drawn if given.
        min_matched: [int] Smallest number of matched stars, drawn if given.
        chunk_seconds: [dict] Time of the chunks of the inputs (s) by their names, from the spacing of their chunks
            for the inputs which are not in it. None by default.
        bounds: [tuple] Beginning and end of the night (datetime, UTC), the plot and the summary are within them.
            None by default.
        input_ends: [dict] End of the inputs (datetime, UTC) by their names, see nightTimeline. None by default.
        removed_times: [list] Times of the detections removed on clouded sky (datetime, UTC), where they are
            known. Empty by default.
        detection_mags: [list] Median magnitude of every detection (None where it is not known). None by default
            (no magnitudes).

    Return:
        [dict] Summary: the observed time (h, overlapping inputs counted once), and the fractions of it which
            were clear, partly clouded, clouded and not searched, and the median dimming of the chunks which
            were not clouded.
    """

    # Imported here, the plotting is not needed for the processing. A figure without pyplot doesn't depend on
    #   the backend or the display
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
    from matplotlib.lines import Line2D
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    import matplotlib.transforms as mtransforms

    # Unpack the conditions of the chunks
    times = [row['time'] for row in rows]
    clear = np.array([row['clear_fraction'] for row in rows])
    dimming = np.array([row['transparency_mag'] for row in rows])
    seen = np.array([row['reliable_seen'] for row in rows])
    stars = np.array([row['stars'] for row in rows])
    matched = np.array([row['matched'] for row in rows])

    # Build the time line of the state of the sky, which is shaded on the panels and summarized at the top
    timeline = nightTimeline(rows, not_searched=not_searched, chunk_seconds=chunk_seconds, bounds=bounds,
                             input_ends=input_ends)
    input_rows, steps, chunk, max_gap, ends = (timeline[k] for k in ('input_rows', 'steps', 'chunk', 'max_gap',
                                                                     'ends'))
    labels, segments, gaps, summary = (timeline[k] for k in ('labels', 'segments', 'gaps', 'summary'))
    step = chunk.total_seconds()
    n = len(rows)

    # Add the median dimming of the chunks which were not clouded to the summary
    usable = (labels != 'clouded') & np.isfinite(dimming)
    summary['median_dimming'] = float(np.median(dimming[usable])) if usable.any() else float('nan')

    # Colours which can be told apart also with colour vision deficiencies (from the palette of Okabe and Ito):
    #   the clear sky is light green (as its summary card), the clouded sky is a darker orange than the partly
    #   clouded sky, the time without stars or data is grey, and the values are blue, green and black
    shade = {'clear': ('#009E73', 0.12), 'partly': ('#E69F00', 0.25), 'clouded': ('#D55E00', 0.45),
             'not_searched': ('#000000', 0.4), 'no_data': ('#000000', 0.07)}
    cards_bg = {'partly': '#f9e7bf', 'clouded': '#ecb48c', 'not_searched': '#999999'}
    blue, green, grey, dark = '#0072B2', '#009E73', '#4b5563', '#111827'


    # Four panels over the same time: the dimming, the clear part of the image, the stars and the detections
    fig = Figure(figsize=(14, 11.6), dpi=110)
    axes = fig.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2, 2, 1.8], 'hspace': 0.3})
    ax_dim, ax_clear, ax_stars, ax_det = axes

    # Shade the state of the sky on all panels, so every value can be read against it
    for ax in axes:
        for state, start, end in segments:
            if state in shade:
                ax.axvspan(start, end, color=shade[state][0], alpha=shade[state][1], lw=0, zorder=0)
        for start, end in gaps:
            ax.axvspan(start, end, color=shade['no_data'][0], alpha=shade['no_data'][1], lw=0, zorder=0)

        # Light grid, no frame on the top and on the right
        ax.grid(True, color='#e5e7eb', lw=0.8, zorder=1)
        ax.tick_params(labelsize=10)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    def _plotInputs(ax, values, *args, **kwargs):
        """ Plot the values of every input as its own line, broken at the pauses, so inputs which overlap are
            not joined into one zigzag line. """

        for name, idx in input_rows.items():
            ax.plot(*_withGaps([times[i] for i in idx], np.asarray(values)[idx], 3*steps[name]), *args, **kwargs)

    ### Dimming of the stars ###

    # Plot the dimming of the stars vs clear sky in every chunk (smaller and lighter points for a full night)
    dense = n > 2000
    ax_dim.plot(times, dimming, '.', color=blue, ms=2 if dense else 3.5, alpha=0.25 if dense else 0.5, zorder=2)

    # Smooth it with a running median over a minute, separately for every input
    smooth = np.full(n, np.nan)
    for idx in input_rows.values():
        seconds = [(times[i] - times[idx[0]]).total_seconds() for i in idx]
        smooth[idx] = _runningMedian(seconds, dimming[idx], 30.0)

    # Don't draw the median where the dimming was not measured (e.g. no stars behind a thick cloud)
    smooth[~np.isfinite(dimming)] = np.nan
    _plotInputs(ax_dim, smooth, '-', color=blue, lw=1.6, zorder=3)

    # Mark the dimming of clear sky and the limit above which the sky is clouded
    if cloud_max_offset is not None:
        ax_dim.axhline(cloud_max_offset, color=dark, lw=1, ls='--', zorder=2)
    ax_dim.axhline(0, color=grey, lw=0.8, zorder=2)

    # Fit the range to the bulk of the values and the cloud limit, so a few extreme values don't flatten the plot
    finite = dimming[np.isfinite(dimming)]
    if len(finite):
        lo, hi = np.percentile(finite, [0.5, 99.5])
        top = max(hi, (cloud_max_offset or 0)*2, 0.3)
        bottom, top = min(lo, -0.1), top + 0.05*(top - min(lo, -0.1))
        ax_dim.set_ylim(bottom, top)

        # Mark the chunks dimmed beyond the top of the plot (thick clouds) at its top
        above = np.isfinite(dimming) & (dimming > top)
        if above.any():
            ax_dim.plot([t for t, a in zip(times, above) if a], np.full(int(above.sum()), top), '^', color=blue,
                        ms=4, alpha=0.6, clip_on=False, zorder=3)
    ax_dim.set_ylabel('mag', fontsize=10)

    ### ###


    ### Clear part of the image ###

    # Fill the clear part of the image of every chunk, the last chunk of every input until its end
    for name, idx in input_rows.items():
        t_cl, v_cl = _withGaps([times[i] for i in idx] + [ends[name]], 100*clear[idx + idx[-1:]], 3*steps[name])
        ax_clear.fill_between(t_cl, 0, v_cl, step='post', color=green, alpha=0.12, lw=0, zorder=2)
        ax_clear.plot(t_cl, v_cl, '-', drawstyle='steps-post', color=green, lw=1.4, zorder=3)

    # Plot the share of the stars which are reliably seen on clear sky that are seen
    _plotInputs(ax_clear, 100*seen, '--', color='#111827', lw=1.0, zorder=3)
    ax_clear.set_ylim(0, 105)
    ax_clear.set_ylabel('%', fontsize=10)

    ### ###


    ### Stars ###

    # Plot the extracted and the matched stars, also of the inputs which were not searched
    _plotInputs(ax_stars, stars, '-', color=blue, lw=1.0, zorder=3)
    _plotInputs(ax_stars, matched, '-', color=dark, lw=1.0, zorder=3)
    for _, _, ns_times, ns_stars in not_searched:
        ax_stars.plot(ns_times, ns_stars, '-', color=blue, lw=1.0, zorder=3)

    # Mark the fewest matched stars with which the sky is judged
    if min_matched is not None:
        ax_stars.axhline(min_matched, color=dark, lw=1, ls='--', zorder=2)

    # Use a log scale with plain numbers, as the stars drop by orders of magnitude under clouds
    ax_stars.set_yscale('log')
    ax_stars.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax_stars.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: '{:g}'.format(v)))
    ax_stars.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Fit the range to the star counts and the limit of the matched stars
    star_counts = np.r_[stars, matched, [c for span in not_searched for c in span[3]]]
    star_counts = star_counts[np.isfinite(star_counts) & (star_counts > 0)]
    if len(star_counts):
        ax_stars.set_ylim(max(min(star_counts.min(), min_matched or star_counts.min())/1.5, 1),
                          max(star_counts.max(), min_matched or 0)*1.5)
    ax_stars.set_ylabel('stars', fontsize=10)

    ### ###


    # Every detection is a dot at its median magnitude (brighter up). The detections removed on clouded sky, whose
    #   magnitudes were not measured, are crosses of another colour in the middle of the panel, as are the
    #   detections without a magnitude (hollow dots), so none of them can be taken for the shading
    det_times = list(detection_times)
    mags = np.array([np.nan if m is None else m for m in (list(detection_mags) if detection_mags is not None
                                                          else [None]*len(det_times))], dtype=np.float64)
    known = np.isfinite(mags)
    removed_times = list(removed_times)
    det_legend = []
    if known.any():

        # The range of the magnitudes, the markers without a magnitude in its middle
        bright, faint = float(np.min(mags[known])), float(np.max(mags[known]))
        pad = max(0.1*(faint - bright), 0.3)
        y_mid = (bright + faint)/2
        ax_det.set_ylim(faint + pad, bright - pad)
        ax_det.set_ylabel('mag', fontsize=10)
        ax_det.plot([t for t, k in zip(det_times, known) if k], mags[known], 'o', ms=5.5, color=dark, mec='white',
                    mew=0.6, alpha=0.85, zorder=3)
        det_legend.append(Line2D([], [], marker='o', ms=5.5, color=dark, mec='white', ls='none',
                                 label='Median mag'))
    else:
        y_mid = 0.5
        ax_det.set_ylim(0, 1)
        ax_det.set_yticks([])

    # The detections without a magnitude (e.g. not calibrated)
    if (~known).any():
        ax_det.plot([t for t, k in zip(det_times, known) if not k], np.full(int((~known).sum()), y_mid), 'o',
                    ms=5.5, mfc='white', mec=dark, mew=1.0, zorder=3)
        det_legend.append(Line2D([], [], marker='o', ms=5.5, mfc='white', mec=dark, ls='none',
                                 label='No mag'))

    # The detections removed on clouded sky
    if removed_times:
        ax_det.plot(removed_times, np.full(len(removed_times), y_mid), 'X', ms=7, color='#CC79A7', mec=dark,
                    mew=0.5, zorder=3)
        det_legend.append(Line2D([], [], marker='X', ms=7, color='#CC79A7', mec=dark, mew=0.5, ls='none',
                                 label='Removed by clouds (no mag)'))
    ax_det.grid(True, axis='y', color='#e5e7eb', lw=0.8, zorder=1)
    ax_det.grid(False, axis='x')

    # Set the time axis (UTC) to the data within the night, with the seconds shown for a short time
    ax_det.set_xlabel('Time (UTC)', fontsize=10)
    all_times = times + [t for span in not_searched for t in span[:2]] + list(detection_times) + removed_times
    time_format = '%H:%M'
    if all_times:
        x_start, x_end = min(all_times), max([max(all_times) + chunk] + list(ends.values()))
        if bounds is not None:
            x_start, x_end = max(x_start, bounds[0]), min(x_end, bounds[1])
        ax_det.set_xlim(x_start, x_end)
        if max(all_times) - min(all_times) < datetime.timedelta(minutes=30):
            time_format = '%H:%M:%S'
    ax_det.xaxis.set_major_formatter(mdates.DateFormatter(time_format))

    # Write the name of the time without stars or without data in the dimming panel where there is room, so it
    #   can be read without telling the colours apart
    x0, x1 = ax_det.get_xlim()
    label_transform = mtransforms.blended_transform_factory(ax_dim.transData, ax_dim.transAxes)
    named = [(start, end, 'too few stars') for state, start, end in segments if state == 'not_searched'] \
        + [(start, end, 'no data') for start, end in gaps]
    for start, end, name in named:
        a, b = mdates.date2num(start), mdates.date2num(end)
        if (b - a) > 0.02*(x1 - x0):
            ax_dim.text((a + b)/2, 0.5, name, transform=label_transform, ha='center',
                        va='center', rotation=90, fontsize=9.5,
                        color='#111827', alpha=0.8, zorder=4)

    # Name every panel above it
    for ax, name in [(ax_dim, 'Dimming of the stars vs clear sky'), (ax_clear, 'Clear part of the image'),
                     (ax_stars, 'Stars per chunk'), (ax_det, 'Detections')]:
        ax.set_title(name, loc='left', fontsize=10.5, fontweight='bold', color='#111827', pad=4)

    # The legend of the values of every panel (placed when the layout is final, see below)
    legends = [
        (ax_dim, [Line2D([], [], color=blue, marker='.', ls='none', ms=6, alpha=0.6,
                         label='Every chunk (~{:.0f} s)'.format(step)),
                  Line2D([], [], color=blue, lw=1.6, label='1-minute median'),
                  Line2D([], [], color=grey, lw=0.8, label='Clear sky')]
                 + ([Line2D([], [], color=dark, lw=1, ls='--',
                            label='Cloud limit ({:.2f} mag)'.format(cloud_max_offset))]
                    if cloud_max_offset is not None else [])
                 + [Line2D([], [], color=blue, marker='^', ls='none', ms=5, alpha=0.6, label='Off the scale')]),
        (ax_clear, [Patch(facecolor=green, alpha=0.3, edgecolor=green, label='Clear part of the image'),
                    Line2D([], [], color='#111827', lw=1.0, ls='--', label='Reliable stars seen')]),
        (ax_stars, [Line2D([], [], color=blue, lw=1.0, label='Extracted'),
                    Line2D([], [], color=dark, lw=1.0, label='Matched to the catalog')]
                   + ([Line2D([], [], color=dark, lw=1, ls='--',
                              label='Fewest matched ({:d})'.format(int(min_matched)))]
                      if min_matched is not None else [])),
    ] + ([(ax_det, det_legend)] if det_legend else [])

    # The legend of the shading, above the panels as it applies to all of them
    shading = [Patch(facecolor=shade['clear'][0], alpha=shade['clear'][1], label='Clear'),
               Patch(facecolor=shade['partly'][0], alpha=shade['partly'][1],
                     label='Partly clouded: detections removed where clouded'),
               Patch(facecolor=shade['clouded'][0], alpha=shade['clouded'][1],
                     label='Clouded: detections removed'),
               Patch(facecolor=shade['not_searched'][0], alpha=shade['not_searched'][1],
                     label='Too few stars: not searched'),
               Patch(facecolor=shade['no_data'][0], alpha=shade['no_data'][1], label='No data')]
    fig.legend(handles=shading, loc='upper center', ncol=5, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.53, 0.862), title='Shading of all panels', title_fontsize=9.5)

    ### Summary cards ###

    # Summarize the night at the top as a row of cards, every card a large value with its name below it. The
    #   shares of the states are joined into one card coloured as the shading, and so are the detections and the
    #   ones removed by clouds
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.985, x=0.53)
    states = [('{:.0f}%'.format(100*summary['clear']), 'clear', '#007a5a', '#e0f2ec'),
              ('{:.0f}%'.format(100*summary['partly']), 'partly clouded', '#a36f00', cards_bg['partly']),
              ('{:.0f}%'.format(100*summary['clouded']), 'clouded', '#8a3a00', cards_bg['clouded'])]
    if not_searched:
        states.append(('{:.0f}%'.format(100*summary['not_searched']), 'too few stars', '#111827',
                       cards_bg['not_searched']))
    detections = [('{:d}'.format(len(detection_times)), 'detections', '#111827', '#f3f4f6')]
    if removed_detections:
        detections.append(('\u2212{:d}'.format(removed_detections), 'removed by clouds', '#a36f00',
                           cards_bg['partly']))
    groups = [[('{:.1f} h'.format(summary['observed_h']), 'observed', '#111827', '#f9fafb')], states, detections]

    # Draw the groups of cards across the width of the panels, the cards of a group in one rounded box
    left, right, bottom, height, space = 0.08, 0.98, 0.875, 0.068, 0.01
    width = (right - left - space*(len(groups) - 1))/sum(len(group) for group in groups)
    x = left
    for group in groups:
        box = FancyBboxPatch((x, bottom), width*len(group), height, boxstyle='round,pad=0,rounding_size=0.008',
                             transform=fig.transFigure, facecolor='none', edgecolor='#d1d5db', lw=0.8, zorder=2)
        fig.patches.append(box)
        for i, (value, name, color, background) in enumerate(group):
            part = Rectangle((x + i*width, bottom), width, height, transform=fig.transFigure,
                             facecolor=background, edgecolor='none', zorder=1)
            part.set_clip_path(box)
            fig.patches.append(part)
            if i:
                fig.lines.append(Line2D([x + i*width]*2, [bottom, bottom + height], transform=fig.transFigure,
                                        color='white', lw=1.5, zorder=1.5))
            fig.text(x + (i + 0.5)*width, bottom + 0.62*height, value, ha='center', va='center', fontsize=15,
                     fontweight='bold', color=color)
            fig.text(x + (i + 0.5)*width, bottom + 0.2*height, name, ha='center', va='center', fontsize=9,
                     color='#374151')
        x += width*len(group) + space

    ### ###


    fig.subplots_adjust(left=0.08, right=0.98, top=0.785, bottom=0.055)

    # Put the legend of every panel on its right, where it covers the fewest points
    for ax, handles in legends:
        loc = _legendPlace(fig, ax, handles)
        ax.legend(handles=handles, loc=loc, fontsize=9, handlelength=2.2, frameon=True, framealpha=0.92,
                  facecolor='white', edgecolor='#e5e7eb').set_zorder(10)

    fig.savefig(plot_path)

    return summary
