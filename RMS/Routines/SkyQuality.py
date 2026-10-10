""" Local sky quality from the stars of the chunks of frames: where and when the sky is clear enough for the
measurements of the detections to be trusted.

The stars of every chunk are matched to the star catalog, and the difference between the instrumental magnitude of
every matched star (with a fixed zero point) and its catalog magnitude is its photometric residual. Clouds dim the
stars behind them, so a cloud shows up as a region of stars with larger residuals than the rest of the image, and
an opaque cloud as a region without matched stars. The local photometric offset and the local density of the
matched stars are computed on a grid, from the k nearest matched stars of every grid point, and compared with the
reference of the same grid point: the clearest chunks of the input (this removes the residuals of the flat and the
vignetting, and the star density of the field and the image corners).

A point of the image at a given frame is clear if the chunk of the frame and the neighbouring chunks have enough
matched stars, the local offset is below a limit (relative to the clear part of the chunk, as a uniform thin haze
is calibrated by the photometric offset of the chunk), and the k nearest matched stars are not much farther than in
the reference.
"""

from __future__ import print_function, division, absolute_import

import copy
import datetime

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from RMS.Astrometry.ApplyAstrometry import extinctionCorrectionTrueToApparent, photomLine, raDecToXYPP
from RMS.Astrometry.ApplyRecalibrate import recalibratePlateparsForFF
from RMS.Astrometry.Conversions import datetime2JD
from RMS.Formats import StarCatalog
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Logger import getLogger

log = getLogger("logger")


class CalibrationError(Exception):
    """ The platepar could not be fitted to the stars, so the sky quality is not known. """
    pass



class SkyQualityMap(object):
    def __init__(self, star_list, platepar, config, begin_time, fps, chunk_frames, n_neighbours=10,
                 max_offset=0.15, max_radius_ratio=2.0, grid_cells=32, match_radius=1.0, matched=None):
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
        """

        self.config = config
        self.fps = fps
        self.k = n_neighbours
        self.max_offset = max_offset
        self.max_radius_ratio = max_radius_ratio
        self.match_radius = match_radius

        self.width, self.height = platepar.X_res, platepar.Y_res

        # The grid points are the centres of square cells
        self.cell = max(self.width, self.height)/grid_cells
        gx = np.arange(self.cell/2, self.width, self.cell)
        gy = np.arange(self.cell/2, self.height, self.cell)
        self.grid_shape = (len(gy), len(gx))
        grid_x, grid_y = np.meshgrid(gx, gy)
        self.grid = np.c_[grid_x.ravel(), grid_y.ravel()]

        # The first frame of every chunk, from its time
        star_list = [entry for entry in star_list if len(entry) > 1]
        self.chunk_first = np.array([int(round((filenameToDatetime(entry[0]) - begin_time).total_seconds()*fps))
                                     for entry in star_list])
        order = np.argsort(self.chunk_first)
        star_list = [star_list[i] for i in order]
        self.chunk_first = self.chunk_first[order]
        if chunk_frames is None:
            chunk_frames = int(np.median(np.diff(self.chunk_first))) if len(self.chunk_first) > 1 else 128
        self.chunk_frames = max(int(chunk_frames), 1)

        # The matched stars of every chunk, as rows [x, y, residual]
        self.n_stars = np.array([len(entry[1]) for entry in star_list])
        self.matched = self.matchStars(star_list, platepar) if matched is None else [matched[i] for i in order]
        self.n_matched = np.array([len(m) for m in self.matched])

        # Which grid points of which chunks are clear
        self.clear = self.computeClear()


    def matchStars(self, star_list, platepar):
        """ Match the stars of every chunk to the catalog and compute their photometric residuals.

        The platepar is recalibrated once, on the chunk with the most stars, and the catalog is projected into
        every chunk at its time. A star is matched if its nearest catalog star is within match_radius and there is
        no other catalog star within 3 match radii (blended stars have wrong intensities).

        Arguments:
            star_list: [list] Stars of the chunks, sorted by time.
            platepar: [Platepar] Platepar of the camera.

        Return:
            [list] Per chunk, an array of rows [x, y, residual] of the matched stars. The residual is the
                instrumental magnitude (with the zero point of the given platepar) minus the apparent catalog
                magnitude, positive for stars fainter than expected.
        """

        matched = [np.zeros((0, 3)) for _ in star_list]
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
            log.warning('Sky quality: the star catalog could not be loaded')
            return matched
        catalog = catalog[0]

        # Recalibrate the platepar on the chunk with the most stars (or the next ones, if it fails). The pointing
        #   of a fixed camera doesn't change within a file, so this one fit is used for all chunks
        calstars = {entry[0]: entry[1] for entry in star_list}
        config = copy.deepcopy(self.config)
        pp = None
        for i in np.argsort(-self.n_stars)[:3]:
            ff_name = star_list[i][0]
            recalibrated = recalibratePlateparsForFF(platepar, [ff_name], calstars, catalog, config,
                                                     ff_frames=self.chunk_frames)
            if (ff_name in recalibrated) and recalibrated[ff_name].auto_recalibrated:
                pp = recalibrated[ff_name]
                break

        # Without a calibration, the stars can't be matched to the catalog
        if pp is None:
            raise CalibrationError('the platepar could not be recalibrated on the stars of the input')

        # Only the catalog stars near the field are projected for every chunk (the margin covers the motion of
        #   the sky within the input)
        jd_mid = datetime2JD(filenameToDatetime(star_list[len(star_list)//2][0]))
        x, y = raDecToXYPP(catalog[:, 0], catalog[:, 1], jd_mid, pp)
        margin = 0.3*max(self.width, self.height)
        near = (x > -margin) & (x < self.width + margin) & (y > -margin) & (y < self.height + margin)
        catalog = catalog[near]

        for i, (ff_name, stars) in enumerate(star_list):

            stars = np.array(stars, dtype=np.float64)
            if (len(stars) == 0) or (len(catalog) < 2):
                continue

            # The catalog at the time of the chunk (the middle of it)
            jd = datetime2JD(filenameToDatetime(ff_name)) + self.chunk_frames/(2*self.fps)/86400.0
            cat_x, cat_y = raDecToXYPP(catalog[:, 0], catalog[:, 1], jd, pp)

            # Unambiguous matches with a positive intensity, without saturated pixels (their intensities are too
            #   low)
            dist, idx = cKDTree(np.c_[cat_x, cat_y]).query(stars[:, [1, 0]], k=2)
            ok = (dist[:, 0] <= self.match_radius) & (dist[:, 1] > 3*self.match_radius) & (stars[:, 2] > 0)
            if stars.shape[1] > 7:
                ok &= stars[:, 7] == 0
            if not ok.any():
                continue
            cat = catalog[idx[ok, 0]]

            # The residuals: the instrumental magnitudes corrected for the vignetting, with the zero point of the
            #   given platepar, minus the catalog magnitudes corrected for the extinction
            app_mags = np.array(extinctionCorrectionTrueToApparent(cat[:, 2], cat[:, 0], cat[:, 1], jd, pp))
            radius = np.hypot(stars[ok, 0] - self.height/2, stars[ok, 1] - self.width/2)
            inst_mags = photomLine((stars[ok, 2], radius), platepar.mag_lev, pp.vignetting_coeff)
            matched[i] = np.c_[stars[ok, 1], stars[ok, 0], inst_mags - app_mags]

        return matched


    def computeClear(self):
        """ Which grid points of which chunks are clear (see the module description).

        Return:
            [ndarray] Boolean array (chunks, grid points), True for clear sky.
        """

        n_chunks = len(self.matched)
        clear = np.zeros((n_chunks, len(self.grid)), dtype=bool)

        # Chunks with too few stars (clouds, twilight) are not clear anywhere
        usable = (self.n_stars >= self.config.ff_min_stars) & (self.n_matched >= self.k)
        if not usable.any():
            return clear

        # The local offset (the median residual of the k nearest matched stars) and the distance to the k-th
        #   nearest matched star of every grid point
        offset = np.full((n_chunks, len(self.grid)), np.nan)
        radius = np.full((n_chunks, len(self.grid)), np.nan)
        for i in np.nonzero(usable)[0]:
            dist, idx = cKDTree(self.matched[i][:, :2]).query(self.grid, k=self.k)
            offset[i] = np.median(self.matched[i][idx, 2], axis=1)
            radius[i] = dist[:, -1]

        # The reference of every grid point: the clearest of the chunks (the smallest offsets and radii)
        offset_ref = np.nanpercentile(offset, 20, axis=0)
        radius_ref = np.nanpercentile(radius, 20, axis=0)

        for i in np.nonzero(usable)[0]:

            # The offset relative to the clear part of the chunk: a uniform haze changes the zero point of the
            #   whole chunk, which the photometric calibration of the chunk takes care of
            excess = offset[i] - offset_ref
            excess -= np.percentile(excess, 20)

            flagged = (excess > self.max_offset) | (radius[i] > self.max_radius_ratio*radius_ref)

            # The offsets and the radii are smoothed over the k nearest stars, which spread over several grid
            #   cells, so a cloud flags a group of cells. A single flagged cell is noise (e.g. at the image edge,
            #   with fewer stars), and it is not flagged
            flagged = flagged.reshape(self.grid_shape)
            neighbours = ndimage.convolve(flagged.astype(np.int32), np.ones((3, 3), dtype=np.int32),
                                          mode='constant')
            flagged &= neighbours >= 2

            clear[i] = ~flagged.ravel()

        self.offset, self.radius, self.offset_ref, self.radius_ref = offset, radius, offset_ref, radius_ref

        return clear


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
