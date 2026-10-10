""" Local sky quality from the stars of the chunks of frames: where and when the sky is clear enough for the
measurements of the detections to be trusted.

The stars of every chunk are matched to the star catalog, and the difference between the instrumental magnitude of
every matched star (with a fixed zero point) and its catalog magnitude is its photometric residual. On clear sky the
residual of a star is nearly the same in every chunk, but it differs from star to star by about 0.1 mag (the errors
of the catalog magnitudes and the colours of the stars). The reference of every star (SkyReference) is the median
over the other inputs of the camera of its residual on clear sky in an input (a low quantile of its residuals
there), relative to the zero point of the input (the median of these residuals over its stars), so the transparency
of the inputs doesn't enter the reference. The input itself is used only for the stars without other inputs, so a
cloud during the whole input doesn't enter the references of the stars behind it. Clouds dim the stars behind them, so a cloud shows up as a region of stars
fainter than their references, and an opaque cloud as a region where stars which are reliably seen on clear sky are
missing.

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
import copy
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
#   a low quantile of its residuals)
MIN_STAR_OBS = 5

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



def starKeys(ra, dec):
    """ Keys of catalog stars: their positions rounded to 0.001 deg, as integers (unique for the stars which are
        matched, as there is no other catalog star within 3 match radii of them).

    Arguments:
        ra, dec: [ndarray] Right ascensions and declinations (deg).

    Return:
        [ndarray] int64 keys.
    """

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

    fov_h, fov_v = computeFOVSize(platepar)
    radius = 0.65*math.hypot(fov_h, fov_v)

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
        self.stars_path = os.path.join(dir_path, 'sky_reference_{:s}.npz'.format(station_id))
        self.lock_path = os.path.join(dir_path, '.sky_reference_{:s}.lock'.format(station_id))

        self.load()


    def load(self):
        """ Read the reference of the stars (empty if there is none yet). """

        self.keys = np.zeros(0, dtype=np.int64)
        self.times = np.zeros((0, self.MAX_VISITS))
        self.values = np.zeros((0, self.MAX_VISITS), dtype=np.float32)
        self.rates = np.zeros((0, self.MAX_VISITS), dtype=np.float32)
        self.zp_times = np.zeros(0)
        self.zp_values = np.zeros(0)

        if os.path.isfile(self.stars_path):
            try:
                with np.load(self.stars_path) as data:
                    self.keys, self.times = data['keys'], data['times']
                    self.values, self.rates = data['values'], data['rates']
                    self.zp_times, self.zp_values = data['zp_times'], data['zp_values']
            except Exception as e:
                log.warning('Sky quality: the reference {:s} can\'t be read ({:s}), it is started again'.format(
                    self.stars_path, str(e)))

        self.index = {int(k): i for i, k in enumerate(self.keys)}


    def zeroPoints(self, exclude_time=None):
        """ The zero points of the inputs.

        Keyword arguments:
            exclude_time: [float] The zero point at this time (unix s) is left out. None by default.

        Return:
            [ndarray] Zero points (mag).
        """

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
            row = self.index.get(int(key))
            if row is None:
                continue
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

            keys_all, times, vals, rts = list(self.keys), [self.times], [self.values], [self.rates]
            new_rows = [k for k in keys if int(k) not in self.index]
            if new_rows:
                keys_all += [int(k) for k in new_rows]
                times.append(np.full((len(new_rows), self.MAX_VISITS), np.nan))
                vals.append(np.full((len(new_rows), self.MAX_VISITS), np.nan, dtype=np.float32))
                rts.append(np.full((len(new_rows), self.MAX_VISITS), np.nan, dtype=np.float32))
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

                times[row, slot], vals[row, slot], rts[row, slot] = t, value, rate

            # The zero point of this input (replacing an earlier processing of it, also if it is not known now), the
            #   latest ones kept
            keep = np.abs(self.zp_times - t) > 1.0
            zp_times, zp_values = self.zp_times[keep], self.zp_values[keep]
            if np.isfinite(zero_point):
                zp_times, zp_values = np.r_[zp_times, t], np.r_[zp_values, zero_point]
                order = np.argsort(zp_times)[-self.MAX_ZERO_POINTS:]
                zp_times, zp_values = zp_times[order], zp_values[order]

            # Write the reference to a temporary file and replace the old one with it
            tmp_path = self.stars_path + '.tmp.npz'
            np.savez_compressed(tmp_path, keys=keys_all, times=times, values=vals, rates=rts, zp_times=zp_times,
                                zp_values=zp_values)
            os.replace(tmp_path, self.stars_path)

            self.keys, self.times, self.values, self.rates, self.index = keys_all, times, vals, rts, index
            self.zp_times, self.zp_values = zp_times, zp_values

            # Append the conditions to the file of the date of every chunk
            processed = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
            for date, rows in _groupByDate(conditions):
                path = os.path.join(self.dir_path, 'sky_conditions_{:s}_{:s}.csv'.format(self.station_id, date))
                new_file = not os.path.isfile(path)
                with open(path, 'a') as f:
                    if new_file:
                        f.write('# Sky conditions of the chunks of frames (RMS.Routines.SkyQuality). A chunk processed '
                                'again has a newer row (processed_utc), which is the valid one\n')
                        f.write(','.join(CONDITION_COLUMNS) + ',processed_utc\n')
                    for row in rows:
                        f.write(','.join(str(row[c]) for c in CONDITION_COLUMNS) + ',' + processed + '\n')



# Columns of the log of the conditions: the time of the beginning of the chunk (UTC), the input file, the number of
#   extracted and matched stars, the fraction of the image which is clear, the transparency (the median excess of the
#   residuals of the stars over their clear-sky residuals, mag; positive is less transparent), and the fraction of
#   the reliable stars which are seen
CONDITION_COLUMNS = ('chunk_utc', 'input', 'stars', 'matched', 'clear_fraction', 'transparency_mag',
                     'reliable_seen')



def _groupByDate(conditions):
    """ The rows of the conditions grouped by the UTC date of their chunks (YYYYMMDD). """

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
        self.f = open(self.path, 'a')
        try:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_EX)
        except ImportError:
            pass
        return self

    def __exit__(self, *args):
        try:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_UN)
        except ImportError:
            pass
        self.f.close()



class SkyQualityMap(object):
    def __init__(self, star_list, platepar, config, begin_time, fps, chunk_frames, n_neighbours=10,
                 max_offset=0.15, max_radius_ratio=2.0, grid_cells=32, match_radius=1.0, matched=None,
                 total_frames=None, first_frames=None, min_matched=30, reference=None, input_name=''):
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
            total_frames: [int] Number of frames of the input, for the chunks missing at its end. None by
                default (the frames after the last chunk belong to it).
            min_matched: [int] Smallest number of matched stars of the chunk the platepar is fitted on, for the
                calibration to be trusted. 30 by default.
            first_frames: [list] The first frame of every chunk of star_list (see chunkFirstFrames). None by
                default, the first frames are computed from the times of the chunks and the frame rate (which
                assumes no gaps in the recording).
            reference: [SkyReference] Clear-sky reference of the camera, from its other inputs. None by default
                (only this input is used).
            input_name: [str] Name of the input, for the log of the conditions. Empty by default.
        """

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

        self.width, self.height = platepar.X_res, platepar.Y_res

        # The grid points are the centres of square cells
        self.cell = max(self.width, self.height)/grid_cells
        gx = np.arange(self.cell/2, self.width, self.cell)
        gy = np.arange(self.cell/2, self.height, self.cell)
        self.grid_shape = (len(gy), len(gx))
        grid_x, grid_y = np.meshgrid(gx, gy)
        self.grid = np.c_[grid_x.ravel(), grid_y.ravel()]

        # The first frame of every chunk, given or from its time
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

        order = np.argsort(self.chunk_first)
        star_list = [star_list[i] for i in order]
        self.chunk_first = self.chunk_first[order]
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
        self.addMissingChunks(total_frames)
        self.n_matched = np.array([len(m) for m in self.matched])

        # Which grid points of which chunks are clear
        self.clear = self.computeClear()


    def addMissingChunks(self, total_frames):
        """ Add empty chunks (no stars, so clouded) where chunks are missing: in the gaps between the chunks,
            before the first chunk, and after the last one up to total_frames (also for a part of a chunk).

        Arguments:
            total_frames: [int] Number of frames of the input, or None.
        """

        cf = self.chunk_frames
        if len(self.chunk_first) == 0:
            return

        # The first frames of the chunks, with the missing ones (a chunk is missing where the next one begins
        #   more than half a chunk after the expected frame)
        firsts = list(np.arange(self.chunk_first[0] - cf, -cf/2.0, -cf)[::-1].astype(int))
        for k, f0 in enumerate(self.chunk_first):
            if k > 0:
                expected = self.chunk_first[k - 1] + cf
                while expected <= f0 - cf/2.0:
                    firsts.append(int(expected))
                    expected += cf
            firsts.append(int(f0))
        # After the last chunk, every remaining frame is in a missing chunk (the star extraction uses only full
        #   chunks, so the frames at the end of the input have no stars)
        if total_frames is not None:
            expected = self.chunk_first[-1] + cf
            while expected < total_frames:
                firsts.append(int(expected))
                expected += cf

        if len(firsts) == len(self.chunk_first):
            return

        # The stars of the existing chunks, none for the added ones
        existing = dict(zip(self.chunk_first.tolist(), range(len(self.chunk_first))))
        self.matched = [self.matched[existing[f]] if f in existing else np.zeros((0, 4)) for f in firsts]
        self.n_stars = np.array([self.n_stars[existing[f]] if f in existing else 0 for f in firsts])
        self.chunk_times = [self.chunk_times[existing[f]] if f in existing
                            else self.begin_time + datetime.timedelta(seconds=f/self.fps) for f in firsts]
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

        # Without the reference of any star (a short input, with fewer than MIN_STAR_OBS chunks, and no reference
        #   of the camera yet) the sky quality is not known, rather than clouded everywhere
        if not np.isfinite(star_ref).any():
            raise CalibrationError('no star has a reference (too few chunks and no reference of the camera)')
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
            in_keys = np.unique(np.concatenate([self.in_image[i][:, 2] for i in np.nonzero(usable)[0]]))
            in_keys = in_keys[~np.isin(in_keys, self.star_keys)]
            if len(in_keys):
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

        self.radius, self.radius_ref, self.excess = radius, radius_ref, excess

        return clear


    def starReferences(self, usable):
        """ The reference residual of every star of this input (relative to the zero point of an input), whether it
            is reliably seen on clear sky and how often, and the zero point of clear sky.

        The residual of a star in an input is the STAR_QUANTILE quantile of its residuals in the usable chunks
        (with at least MIN_STAR_OBS of them), relative to the zero point of the input (the median of these over
        the stars). Its reference is the median of its residuals in the other inputs in the reference, or, without
        them, its residual in this input (which a cloud during the whole input would bias towards the cloud). A
        star is reliable if it is seen in at least RELIABLE_FRACTION of the chunks in which it is in the image, in
        the median of the other inputs (or in this one, without them). The zero point of clear sky is the
        STAR_QUANTILE quantile of the zero points of the other inputs (or the one of this input, without them).

        Arguments:
            usable: [ndarray] Boolean, the usable chunks.

        Return:
            (keys, residuals, rates, zero_point): sorted keys of the matched stars, their reference residuals (NaN
                if not known), their seen fractions (NaN if not known), and the zero point of clear sky.
        """

        # The visits of this input
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
            residuals = np.where(np.isfinite(ref_residuals), ref_residuals, residuals)
            seen = np.where(np.isfinite(ref_seen), ref_seen, seen)

            ref_zero_points = self.reference.zeroPoints(exclude_time=t)
            if np.isfinite(ref_zero_points).any():
                zero_points = ref_zero_points

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
            (keys, values, rates): sorted keys, residuals (NaN with fewer than MIN_STAR_OBS observations) and seen
                fractions (NaN without the catalog stars in the image).
        """

        obs = []
        for i in np.nonzero(use)[0]:
            m = self.matched[i]
            if clear is not None:
                m = m[clear[i, self.gridIndex(m[:, 0], m[:, 1])]]
            obs.append(m[:, 2:4])
        obs = np.vstack(obs) if obs else np.zeros((0, 2))
        if len(obs) == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)

        # The residuals of every star
        keys, inverse, counts = np.unique(obs[:, 1], return_inverse=True, return_counts=True)
        order = np.argsort(inverse, kind='stable')
        bounds = np.r_[0, np.cumsum(counts)]
        values = np.full(len(keys), np.nan)
        for j in np.nonzero(counts >= MIN_STAR_OBS)[0]:
            values[j] = np.percentile(obs[order[bounds[j]:bounds[j + 1]], 0], STAR_QUANTILE)

        # The fraction of the chunks in which every star is in the image (on clear sky) in which it is seen
        rates = np.full(len(keys), np.nan)
        if self.in_image is not None:
            n_in, n_seen = np.zeros(len(keys)), np.zeros(len(keys))
            for i in np.nonzero(use)[0]:
                inside = self.in_image[i]
                if clear is not None:
                    inside = inside[clear[i, self.gridIndex(inside[:, 0], inside[:, 1])]]
                pos = np.clip(np.searchsorted(keys, inside[:, 2]), 0, len(keys) - 1)
                hit = keys[pos] == inside[:, 2]
                np.add.at(n_in, pos[hit], 1)
                np.add.at(n_seen, pos[hit], inside[hit, 3])
            with np.errstate(all='ignore'):
                rates = np.where(n_in > 0, n_seen/n_in, np.nan)

        return keys, values, rates


    def conditions(self):
        """ Rows of the log of the conditions of the chunks (see CONDITION_COLUMNS). """

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

        use = self.clear.any(axis=1)
        keys, values, rates = self.inputVisits(use, clear=self.clear)
        keep = np.isfinite(values)
        zero_point = np.median(values[keep]) if keep.any() else np.nan
        self.reference.update(self.begin_time, keys[keep], values[keep] - zero_point, rates[keep], zero_point,
                              self.conditions())


    def chunkIndex(self, frames):
        """ The chunk of every frame (the nearest one for frames before the first and after the last chunk). """

        idx = np.searchsorted(self.chunk_first, np.asarray(frames), side='right') - 1
        return np.clip(idx, 0, len(self.chunk_first) - 1)


    def gridIndex(self, x, y):
        """ The grid cell of every image point. """

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

        if len(self.chunk_first) == 0:
            return np.zeros(np.shape(x), dtype=bool)

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
        cent = det[2]
        ok = sky_map.isClear(cent[:, 1], cent[:, 2], cent[:, 0])
        n_rows += int((~ok).sum())
        if ok.sum() < min_rows:
            n_dets += 1
            continue
        kept.append([det[0], det[1], cent[ok]] + list(det[3:]))

    return kept, n_rows, n_dets



def chunkFirstFrames(star_list, img_handle):
    """ The first frame of every chunk of the stars: the frame of the input at the time of the chunk (the time in
        its name), among the first frames of the chunks of the input. With timestamped frames, the frames after a
        gap in the recording are not where the time and the frame rate would put them.

    Arguments:
        star_list: [list] Stars of the chunks, [ff_name, stars] per chunk.
        img_handle: [FrameInterface] The input of the stars, with its chunk_frames.

    Return:
        [list] The first frame of every chunk, -1 for a chunk whose time is not the time of a chunk of the input.
    """

    chunk_frames = img_handle.chunk_frames
    begin = img_handle.beginning_datetime.replace(tzinfo=None)

    # The times of the first frames of all chunks of the input (s from the beginning)
    firsts = np.arange(0, img_handle.total_frames, chunk_frames)
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
