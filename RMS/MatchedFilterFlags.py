""" Flags for the detections of the matched filter which are real light but not the reflected sunlight of an object
    moving above the atmosphere:

    - aircraft: the lights of an aircraft give several bright detections at the same time, side by side and moving
      together (one per light),
    - not sunlit: wherever along the line of sight the object would be for its angular velocity to come from a
      speed of at most mf_flag_sunlit_speed km/s across the line of sight, it would be in the shadow of the Earth,
      so it shines by its own light (aircraft lights, other transient light).

    The flagged detections are not written to the FTPdetectinfo of the results: they are saved apart, in a file in
    the same format but named FlaggedDetections_*.txt (so no tool which looks for FTPdetectinfo files picks it up),
    with a CSV of the reasons (see saveFlaggedDetections).
"""

from __future__ import print_function, division, absolute_import

import csv
import glob
import logging
import os
import shutil
import tempfile

import numpy as np

from RMS.Astrometry.ApplyAstrometry import applyAstrometryFTPdetectinfo, xyToRaDecPP
from RMS.Astrometry.Conversions import J2000_JD, datetime2JD, geo2Cartesian, raDec2Vector
from RMS.Astrometry.CyFunctions import equatorialCoordPrecession
from RMS.Detection import subframePickTime
from RMS.Formats import FTPdetectinfo

log = logging.getLogger("logger")

# Name of the file of the flagged detections (in the FTPdetectinfo format) and of the CSV of their flags
FLAGGED_PREFIX = 'FlaggedDetections'

# Equatorial radius of the Earth (km), the radius of its shadow (a cylinder; the penumbra is ignored)
EARTH_RADIUS_KM = 6378.137

# Ranges along the line of sight (km) at which the shadow is tested: fine near the observer, coarse far away
SHADOW_RANGES_KM = np.concatenate([np.arange(0, 5000, 2.0), np.arange(5000, 50000, 20.0),
                                   np.arange(50000, 500001, 500.0)])


class Track(object):
    """ The measurements of a detection on the sky: times (Julian dates), image positions, celestial coordinates
        (deg) and magnitudes of its points. """

    def __init__(self, jd, x, y, ra, dec, mag):

        order = np.argsort(jd)
        self.jd = np.asarray(jd, dtype=np.float64)[order]
        self.x = np.asarray(x, dtype=np.float64)[order]
        self.y = np.asarray(y, dtype=np.float64)[order]
        self.ra = np.asarray(ra, dtype=np.float64)[order]
        self.dec = np.asarray(dec, dtype=np.float64)[order]
        self.mag = np.asarray(mag, dtype=np.float64)[order]


    @property
    def duration(self):
        """ Duration (s). """

        return (self.jd[-1] - self.jd[0])*86400.0


    def peakMagnitude(self):
        """ Magnitude at the brightest part of the track: the brightest of the medians of 3 consecutive points (so
            a single bright outlier doesn't count). """

        mag = self.mag[np.isfinite(self.mag)]
        if len(mag) == 0:
            return np.inf
        if len(mag) < 3:
            return float(np.min(mag))

        return float(np.min(np.median(np.stack([mag[:-2], mag[1:-1], mag[2:]]), axis=0)))


    def angularVelocity(self):
        """ Mean angular velocity (deg/s) from the first to the last point. """

        if self.duration <= 0:
            return 0.0

        return angularSeparation(self.ra[0], self.dec[0], self.ra[-1], self.dec[-1])/self.duration


    def imageDirection(self):
        """ Direction of motion in the image (unit vector), from the first to the last point. """

        d = np.array([self.x[-1] - self.x[0], self.y[-1] - self.y[0]])
        n = np.hypot(*d)

        return d/n if n > 0 else d


    def imageSpeed(self):
        """ Mean speed in the image (px/s). """

        if self.duration <= 0:
            return 0.0

        return float(np.hypot(self.x[-1] - self.x[0], self.y[-1] - self.y[0]))/self.duration



def angularSeparation(ra1, dec1, ra2, dec2):
    """ Angle between two directions (deg). """

    v1, v2 = np.array(raDec2Vector(ra1, dec1)), np.array(raDec2Vector(ra2, dec2))

    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))))



def sunDirection(jd):
    """ Unit vector of the direction of the Sun (equatorial, epoch of date), from ephem. """

    import ephem

    # ephem counts days from 1899 Dec 31 12:00 (JD 2415020.0); ra/dec are the apparent coordinates of the date
    sun = ephem.Sun(ephem.Date(jd - 2415020.0))

    return np.array(raDec2Vector(np.degrees(float(sun.ra)), np.degrees(float(sun.dec))))



def firstSunlitRange(jd, ra, dec, lat, lon, elev):
    """ The shortest distance along a line of sight at which a point is outside the shadow of the Earth.

    Arguments:
        jd: [float] Julian date.
        ra, dec: [float] Direction of the line of sight (deg, J2000).
        lat, lon: [float] Position of the observer (deg).
        elev: [float] Elevation of the observer (m).

    Return:
        [float] Range (km), inf if the whole line of sight up to the last tested range is in the shadow.
    """

    # The observer, the line of sight and the Sun in the same frame: equatorial, epoch of date (the observer from
    #   the sidereal time, the J2000 line of sight precessed to the date)
    station = np.array(geo2Cartesian(lat, lon, elev, jd))/1000.0
    ra_date, dec_date = equatorialCoordPrecession(J2000_JD.days, jd, np.radians(ra), np.radians(dec))
    sight = np.array(raDec2Vector(np.degrees(ra_date), np.degrees(dec_date)))
    sun = sunDirection(jd)

    # A point is in the shadow if it is on the night side and within the radius of the Earth from the axis of the
    #   shadow (the line through the centre of the Earth away from the Sun)
    points = station[None, :] + SHADOW_RANGES_KM[:, None]*sight[None, :]
    along = points.dot(sun)
    across = np.linalg.norm(points - along[:, None]*sun[None, :], axis=1)
    sunlit = (along > 0) | (across > EARTH_RADIUS_KM)

    if not np.any(sunlit):
        return np.inf

    return float(SHADOW_RANGES_KM[np.argmax(sunlit)])



def sunlitRatio(track, lat, lon, elev, max_speed_km_s):
    """ How much farther away than possible a track would have to be to be sunlit: the distance along the line of
        sight (at the middle of the track) to the first sunlit point, over the largest distance at which its
        angular velocity can come from a speed across the line of sight of at most max_speed_km_s.

    Return:
        [float] The ratio; below 1 the track can be a sunlit object, above 1 it can't be.
    """

    omega = np.radians(track.angularVelocity())
    if omega <= 0:
        return 0.0

    mid = len(track.jd)//2
    r_sunlit = firstSunlitRange(track.jd[mid], track.ra[mid], track.dec[mid], lat, lon, elev)
    r_max = max_speed_km_s/omega

    return r_sunlit/r_max



def overlapGap(t1, t2):
    """ Time between two tracks (s): 0 if they overlap, otherwise the gap between them. """

    gap = max(t1.jd[0], t2.jd[0]) - min(t1.jd[-1], t2.jd[-1])

    return max(gap, 0.0)*86400.0



def parallelCompanion(t1, t2, max_angle, max_speed_frac, max_offset_px):
    """ Whether the track t2 moves alongside t1 at the same time: they overlap in time, move in the same direction
        (within max_angle deg) at the same speed (within the fraction max_speed_frac), and t2 is at most
        max_offset_px from the line of t1. """

    if overlapGap(t1, t2) > 0:
        return False

    d1, d2 = t1.imageDirection(), t2.imageDirection()
    if np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0))) > max_angle:
        return False

    s1, s2 = t1.imageSpeed(), t2.imageSpeed()
    if (s1 <= 0) or (abs(s2 - s1) > max_speed_frac*s1):
        return False

    # Distance of the points of t2 across the line of t1
    normal = np.array([-d1[1], d1[0]])
    offset = np.abs((t2.x - t1.x[0])*normal[0] + (t2.y - t1.y[0])*normal[1])

    return float(np.median(offset)) <= max_offset_px



def flagTracks(tracks, lat, lon, elev, config):
    """ The flags of every track of one input (see the module description).

    Arguments:
        tracks: [list] Track objects of the detections of the input.
        lat, lon, elev: [float] Position of the observer (deg, deg, m).
        config: [Config] Configuration (the mf_flag_* options).

    Return:
        [list] For every track, a list of its flags (empty if none) and a dictionary of the values they come from.
    """

    flags = [[] for _ in tracks]
    values = [{} for _ in tracks]

    # The ratio of the sunlit range, needed by both flags. Without a largest speed it can't be computed, and then
    #   neither flag is applied (the aircraft flag must not take a sunlit group of objects for aircraft)
    if config.mf_flag_sunlit_speed <= 0:
        return flags, values
    for i, track in enumerate(tracks):
        values[i]['sunlit_ratio'] = sunlitRatio(track, lat, lon, elev, config.mf_flag_sunlit_speed)

    # Not sunlit (mf_flag_sunlit_factor 0 turns this flag alone off)
    if config.mf_flag_sunlit_factor > 0:
        for i in range(len(tracks)):
            if values[i]['sunlit_ratio'] > config.mf_flag_sunlit_factor:
                flags[i].append('not_sunlit')

    # Aircraft: a bright track with another bright track within mf_flag_aircraft_gap s, or with a track moving
    #   alongside it. A bright group of objects moving together which can be sunlit (e.g. a group released
    #   together) is not flagged: its ratio of the sunlit range is below 1
    if config.mf_flag_aircraft_mag != 0:
        scale = (config.fov_h/float(config.height) + config.fov_w/float(config.width))/2.0
        max_offset_px = config.mf_flag_parallel_offset/scale
        peaks = [track.peakMagnitude() for track in tracks]
        for i, track in enumerate(tracks):
            values[i]['peak_mag'] = peaks[i]
            if (peaks[i] > config.mf_flag_aircraft_mag) or (values[i]['sunlit_ratio'] < 1.0):
                continue

            for j, other in enumerate(tracks):
                if j == i:
                    continue
                bright_pair = (peaks[j] <= config.mf_flag_aircraft_mag) \
                    and (overlapGap(track, other) <= config.mf_flag_aircraft_gap)
                if bright_pair or parallelCompanion(track, other, config.mf_flag_parallel_angle,
                                                    config.mf_flag_parallel_speed, max_offset_px):
                    flags[i].append('aircraft')
                    values[i]['companion'] = j
                    break

    return flags, values



def detectionTrack(det, img_handle, fps, platepar):
    """ The Track of a detection of the matched filter, with the coordinates and magnitudes from the platepar.

    Arguments:
        det: [list] Detection as [rho, theta, centroids], centroids rows [frame, x, y, intensity, ...].
        img_handle: [FrameInterface] Input of the detection: the times of its points are the recorded times of
            their frames (gaps and dropped frames included), as for the results (see Detection.subframePickTime).
        fps: [float] Frame rate, for the fractional part of the frame numbers.
        platepar: [Platepar] Platepar of the camera.

    Return:
        [Track]
    """

    cent = np.asarray(det[2], dtype=np.float64)
    jd = np.array([datetime2JD(subframePickTime(img_handle, frame, fps)) for frame in cent[:, 0]])
    _, ra, dec, mag = xyToRaDecPP(jd, cent[:, 1], cent[:, 2], np.maximum(cent[:, 3], 1.0), platepar,
                                  jd_time=True, extinction_correction=False)

    return Track(jd, cent[:, 1], cent[:, 2], ra, dec, mag)



def splitFlagged(detections, img_handle, config, platepar):
    """ Split the detections of an input into the kept and the flagged ones (see flagTracks).

    Arguments:
        detections: [list] Detections as [rho, theta, centroids].
        img_handle: [FrameInterface] Input of the detections (the recorded times of its frames).
        config: [Config] Configuration.
        platepar: [Platepar] Platepar of the camera; without one, nothing is flagged.

    Return:
        (kept, flagged): [tuple] The kept detections, and the flagged ones as (detection, flags, values).
    """

    # Nothing to flag: the detections are returned as they are
    if (platepar is None) or (not detections) \
            or (config.mf_flag_sunlit_speed <= 0) \
            or ((config.mf_flag_aircraft_mag == 0) and (config.mf_flag_sunlit_factor <= 0)):
        return detections, []

    fps = img_handle.fps if getattr(img_handle, 'fps', None) else config.fps
    tracks = [detectionTrack(det, img_handle, fps, platepar) for det in detections]
    flags, values = flagTracks(tracks, platepar.lat, platepar.lon, platepar.elev, config)

    kept, flagged = [], []
    for det, det_flags, det_values in zip(detections, flags, values):
        if det_flags:
            flagged.append((det, det_flags, det_values))
            log.info('Matched filter: detection {:.0f}-{:.0f} flagged as {:s}, saved apart'.format(
                det[2][0][0], det[2][-1][0], ', '.join(det_flags)))
        else:
            kept.append(det)

    if not flagged:
        return detections, []

    return kept, flagged



def saveFlaggedDetections(flagged, img_handle, config, output_dir, platepar, chunk_frames=128, chunk_images=None):
    """ Save the flagged detections apart from the results: FlaggedDetections_<station>_<time>.txt in the format of
        an FTPdetectinfo (with the coordinates and magnitudes from the platepar, not recalibrated), and
        FlaggedDetections_<station>_<time>.csv with the flags of every detection.

    Arguments:
        flagged: [list] Flagged detections as (detection, flags, values), see splitFlagged.
        img_handle: [FrameInterface] Input of the detections.
        config: [Config] Configuration.
        output_dir: [str] Directory of the results.
        platepar: [Platepar] Platepar of the camera.

    Keyword arguments:
        chunk_frames, chunk_images: As for DetectStarsAndMeteors.saveResultsFrameInterface.

    Return:
        [str] Path of the file of the detections, None if there are none.
    """

    from RMS.DetectStarsAndMeteors import saveResultsFrameInterface

    # The flagged detections of an earlier processing of the same input are removed (named as the results of the
    #   input, see saveResultsFrameInterface)
    timestamp = getattr(img_handle, 'name_datetime', img_handle.beginning_datetime).strftime("%Y%m%d_%H%M%S_%f")
    stale = '{:s}_{:s}_{:s}*'.format(FLAGGED_PREFIX, config.stationID, timestamp)
    for path in glob.glob(os.path.join(output_dir, stale)):
        os.remove(path)

    if not flagged:
        return None

    # The detections are written as an FTPdetectinfo in a temporary directory (so the names of the image chunks are
    #   assigned as for the results), the platepar is applied there, and the file is moved to the results under
    #   its own name
    tmp_dir = tempfile.mkdtemp(prefix='mf_flagged_')
    try:
        _, _, ftp_name = saveResultsFrameInterface([], [entry[0] for entry in flagged], img_handle, config,
                                                   chunk_frames=chunk_frames, output_dir=tmp_dir,
                                                   write_empty=False, chunk_images=chunk_images)
        applyAstrometryFTPdetectinfo(tmp_dir, ftp_name, None, platepar=platepar)

        # The folders in the header are those of the results, not the temporary one
        name = ftp_name.replace('FTPdetectinfo', FLAGGED_PREFIX, 1)
        out_path = os.path.join(output_dir, name)
        with open(os.path.join(tmp_dir, ftp_name)) as f:
            text = f.read()
        with open(out_path, 'w') as f:
            f.write(text.replace(tmp_dir, output_dir))

        # The flags, by the image and the number of every detection in the file. The writer keeps the order of
        #   the detections, so the entries read back are paired with the flagged detections in order
        entries = FTPdetectinfo.readFTPdetectinfo(output_dir, name)
        with open(os.path.splitext(out_path)[0] + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['ff_name', 'meteor_no', 'flags', 'peak_mag', 'sunlit_ratio'])
            for entry, (_, det_flags, det_values) in zip(entries, flagged):
                writer.writerow([entry[0], int(entry[2]), ';'.join(det_flags),
                                 '{:.2f}'.format(det_values.get('peak_mag', np.nan)),
                                 '{:.2f}'.format(det_values.get('sunlit_ratio', np.nan))])

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_path



def mergeNightFlagged(night_dir, results_paths, config):
    """ Merge the flagged detections of the files of a night (FlaggedDetections_* in their results directories and
        the matched-filter subdirectories) into FlaggedDetections_<night>.txt and .csv in the night directory.

    Arguments:
        night_dir: [str] Night directory.
        results_paths: [list] Results directories of the files of the night.
        config: [Config] Configuration.

    Return:
        [str] Path of the merged file, None if no file had flagged detections.
    """

    from RMS.MatchedFilterDetection import MATCHED_FILTER_DIR

    name = '{:s}_{:s}'.format(FLAGGED_PREFIX, os.path.basename(night_dir))
    for ext in ('.txt', '.csv'):
        if os.path.isfile(os.path.join(night_dir, name + ext)):
            os.remove(os.path.join(night_dir, name + ext))

    # The detections are converted from the format of the reader to that of the writer (as in
    #   MatchedFilterDetection.mergeFTPdetectinfo), and the rows of the CSV files are joined
    meteor_list, csv_rows, header, fps = [], [], None, None
    for path in results_paths:
        for d in (path, os.path.join(path, MATCHED_FILTER_DIR)):
            for txt in sorted(glob.glob(os.path.join(d, FLAGGED_PREFIX + '_*.txt'))):
                for entry in FTPdetectinfo.readFTPdetectinfo(d, os.path.basename(txt)):
                    ff_name, _, meteor_No, _, meteor_fps, _, _, _, _, rho, phi, meteor_meas = entry
                    meteor_list.append([ff_name, meteor_No, rho, phi, [line[1:] for line in meteor_meas],
                                        meteor_fps])
                    fps = meteor_fps
                csv_path = os.path.splitext(txt)[0] + '.csv'
                if os.path.isfile(csv_path):
                    with open(csv_path) as f:
                        rows = list(csv.reader(f))
                    if rows:
                        header = rows[0]
                        csv_rows += rows[1:]

    if not meteor_list:
        return None

    FTPdetectinfo.writeFTPdetectinfo(sorted(meteor_list, key=lambda entry: (entry[0], entry[1])), night_dir,
                                     name + '.txt', night_dir, config.stationID, fps if fps else config.fps,
                                     calibration='Platepar applied, not recalibrated', celestial_coords_given=True)
    if header is not None:
        with open(os.path.join(night_dir, name + '.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(sorted(csv_rows))

    log.info('Merged {:d} flagged detections into {:s}'.format(len(meteor_list), name + '.txt'))

    return os.path.join(night_dir, name + '.txt')
