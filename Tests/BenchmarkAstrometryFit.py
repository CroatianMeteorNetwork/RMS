""" Benchmark of the astrometric fitting hot spots on a synthetic star field.

    Builds a synthetic 1920x1080 platepar with a mild radial7-odd distortion, a synthetic catalog of a
    few thousand stars in the FOV and synthetic detections (the catalog projected with 0.3 px noise,
    30% of stars dropped, 10% false detections). Then times:

        (a) Platepar.fitPointingNN on a platepar with the pointing perturbed by ~0.5 deg,
        (b) the full RANSAC Platepar.fitAstrometry path (use_nn_cost=True),
        (c) StarFilters.filterBlendedStars,
        (d) the ExtractStars PSF duplicate removal.

    This is a script, not a pytest test. Run it as:

        python -m Tests.BenchmarkAstrometryFit [--repeats N] [--seed S]
"""

from __future__ import print_function, division, absolute_import

import argparse
import contextlib
import copy
import io
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.Astrometry.Conversions import JD2HourAngle
from RMS.Astrometry.StarClasses import CatalogStar, PairedStars
from RMS.Astrometry.StarFilters import filterBlendedStars
from RMS.Formats.Platepar import Platepar
import RMS.ExtractStars as ExtractStars


# Image and optics of the synthetic camera
X_RES = 1920
Y_RES = 1080
F_SCALE = 27.0  # px/deg
JD_OBS = 2459580.75  # 2022-01-01 06:00 UTC


### Synthetic field builders ###

def buildSyntheticPlatepar():
    """ Build a synthetic radial7-odd platepar with small distortion coefficients.

    Return:
        pp: [Platepar] Platepar of the synthetic 1920x1080 camera, pointed south at 45 deg altitude.
    """

    # Start from a radial7-odd platepar and set the image size and the plate scale
    pp = Platepar(distortion_type="radial7-odd")

    pp.X_res = X_RES
    pp.Y_res = Y_RES
    pp.F_scale = F_SCALE
    pp.fov_h = X_RES/F_SCALE
    pp.fov_v = Y_RES/F_SCALE

    # Station location and time of the synthetic observation
    pp.lat = 43.0
    pp.lon = -81.0
    pp.elev = 300.0
    pp.JD = JD_OBS
    pp.Ho = JD2HourAngle(JD_OBS)
    pp.refraction = True

    # Point the camera south at 45 deg altitude and derive the reference RA/Dec from that
    pp.az_centre = 180.0
    pp.alt_centre = 45.0
    pp.updateRefRADec(skip_rot_update=True)
    pp.pos_angle_ref = 92.0
    pp.updateRefAltAz()

    # Small radial distortion (the last three coefficients are k1, k2, k3 for radial7-odd)
    pp.setDistortionType("radial7-odd", reset_params=True)
    pp.x_poly_fwd[-3:] = [0.01, 0.002, 0.0005]
    pp.x_poly_rev[-3:] = [-0.01, -0.002, -0.0005]

    return pp


def buildSyntheticCatalog(pp, rng, n_target=3000, cone_radius_deg=45.0):
    """ Draw random catalog stars in the FOV of the platepar.

    Arguments:
        pp: [Platepar] Synthetic platepar.
        rng: [np.random.RandomState] Random generator.

    Keyword arguments:
        n_target: [int] Approximate number of stars to keep in the FOV. 3000 by default.
        cone_radius_deg: [float] Radius of the cone around the pointing to sample from (deg). 45 by
            default.

    Return:
        catalog_stars: [ndarray] (M, 3) array of (ra, dec, mag), sorted by magnitude.
    """

    # Sample points uniformly in a cone around the pointing direction
    n_draw = 20*n_target
    cos_r = np.cos(np.radians(cone_radius_deg))
    z = rng.uniform(cos_r, 1.0, n_draw)
    phi = rng.uniform(0.0, 2*np.pi, n_draw)
    s = np.sqrt(1.0 - z**2)
    local = np.column_stack([s*np.cos(phi), s*np.sin(phi), z])

    # Rotate the local cone (about +z) to the pointing direction
    ra0 = np.radians(pp.RA_d)
    dec0 = np.radians(pp.dec_d)
    k = np.array([np.cos(dec0)*np.cos(ra0), np.cos(dec0)*np.sin(ra0), np.sin(dec0)])
    north = np.array([-np.sin(dec0)*np.cos(ra0), -np.sin(dec0)*np.sin(ra0), np.cos(dec0)])
    east = np.cross(k, north)
    vec = local[:, 0:1]*east + local[:, 1:2]*north + local[:, 2:3]*k

    ra = np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 360
    dec = np.degrees(np.arcsin(np.clip(vec[:, 2], -1.0, 1.0)))
    mag = rng.uniform(2.0, 9.0, n_draw)

    # Keep the stars that project inside the image
    x, y = raDecToXYPP(ra, dec, JD_OBS, pp)
    in_img = (x >= 0) & (x < X_RES) & (y >= 0) & (y < Y_RES)
    catalog = np.column_stack([ra, dec, mag])[in_img][:n_target]

    return catalog[np.argsort(catalog[:, 2])]


def buildSyntheticDetections(pp, catalog_stars, rng, noise_px=0.3, drop_fraction=0.3, false_fraction=0.1):
    """ Project the catalog with the platepar, add noise, drop stars and add false detections.

    Arguments:
        pp: [Platepar] Synthetic platepar.
        catalog_stars: [ndarray] (M, 3) catalog.
        rng: [np.random.RandomState] Random generator.

    Keyword arguments:
        noise_px: [float] Gaussian centroid noise (px). 0.3 by default.
        drop_fraction: [float] Fraction of catalog stars that are not detected. 0.3 by default.
        false_fraction: [float] Fraction of false detections added, relative to the kept stars. 0.1 by
            default.

    Return:
        img_stars: [ndarray] (N, 3) array of (x, y, intensity).
        truth_cat_idx: [ndarray] Catalog index for every detection (-1 for false detections).
    """

    # Project the catalog onto the image and blur the centroids with Gaussian noise
    ra, dec, mag = catalog_stars.T
    x, y = raDecToXYPP(ra, dec, JD_OBS, pp)
    x = x + rng.normal(0.0, noise_px, len(x))
    y = y + rng.normal(0.0, noise_px, len(y))
    intensity = 1000.0*10**(-0.4*(mag - 8.0))

    # Drop a fraction of the stars and everything that fell outside the image
    keep = rng.uniform(size=len(x)) > drop_fraction
    in_img = (x >= 0) & (x < X_RES) & (y >= 0) & (y < Y_RES)
    keep &= in_img

    # Sprinkle false detections over the image, with intensities in the faint half of the real range
    n_false = int(false_fraction*np.sum(keep))
    x_false = rng.uniform(0, X_RES, n_false)
    y_false = rng.uniform(0, Y_RES, n_false)
    int_false = 10**rng.uniform(np.log10(intensity.min()), np.log10(np.median(intensity)), n_false)

    # Stack the real and the false detections, keeping the catalog index of every real one
    img_stars = np.vstack([
        np.column_stack([x[keep], y[keep], intensity[keep]]),
        np.column_stack([x_false, y_false, int_false]),
    ])
    truth = np.concatenate([np.where(keep)[0], -np.ones(n_false, dtype=int)])

    # Shuffle so false detections are not all at the end
    order = rng.permutation(len(img_stars))

    return img_stars[order], truth[order]


def buildPairedStars(pp, catalog_stars, img_stars, truth_cat_idx, rng):
    """ Build a PairedStars object from the true detections.

    Arguments:
        pp: [Platepar] Synthetic platepar. Kept for a uniform builder signature, it is not read.
        catalog_stars: [ndarray] (M, 3) array of (ra, dec, mag).
        img_stars: [ndarray] (N, 3) array of (x, y, intensity).
        truth_cat_idx: [ndarray] Catalog index for every detection (-1 for false detections).
        rng: [np.random.RandomState] Random generator.

    Return:
        paired_stars: [PairedStars] Image/catalog pairs for every true detection, with a random FWHM.
    """

    # Pair every real detection with the catalog star it was projected from, skipping false detections
    paired_stars = PairedStars()
    for (x, y, intens), cat_i in zip(img_stars, truth_cat_idx):
        if cat_i < 0:
            continue
        ra, dec, mag = catalog_stars[cat_i]
        fwhm = rng.uniform(1.8, 3.5)
        paired_stars.addPair(x, y, fwhm, intens, CatalogStar(ra, dec, mag), snr=10.0, saturated=False)

    return paired_stars


def _duplicateKeepMaskReference(x_arr, y_arr, intens_arr, radius):
    """ Inline copy of the PSF duplicate removal loop from ExtractStars.fitPSF, used when the library
        does not expose the helper yet.

    Arguments:
        x_arr: [ndarray] X image coordinates of the detections.
        y_arr: [ndarray] Y image coordinates of the detections.
        intens_arr: [ndarray] Intensity of every detection.
        radius: [float] Detections closer than this (px) are treated as duplicates.

    Return:
        keep: [ndarray] Boolean mask of the detections that survive the duplicate removal.
    """

    # Find all pairs of detections that are closer than the radius
    keep = np.ones(len(x_arr), dtype=bool)
    tree = cKDTree(np.column_stack([x_arr, y_arr]))
    pairs = tree.query_pairs(radius, output_type='ndarray')

    # For each duplicate pair, discard the fainter detection unless one was already dropped
    for i, j in pairs:
        if not keep[i] or not keep[j]:
            continue
        if intens_arr[j] > intens_arr[i]:
            keep[i] = False
        else:
            keep[j] = False

    return keep


### Benchmarks ###

def timeIt(func, repeats):
    """ Run func repeats times and return the best and the mean wall time.

    Arguments:
        func: [function] Zero-argument callable to time.
        repeats: [int] Number of times to run it.

    Return:
        best: [float] Shortest run time (s).
        mean: [float] Mean run time (s).
        result: [object] Return value of the last run.
    """

    # Time every repeat separately, the shortest run is the least noisy estimate
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - t0)

    return min(times), float(np.mean(times)), result


def benchmarkFitPointingNN(pp, catalog_stars, img_stars, repeats):
    """ (a) fitPointingNN on a pointing perturbed by ~0.5 deg.

    Arguments:
        pp: [Platepar] Synthetic ground-truth platepar.
        catalog_stars: [ndarray] (M, 3) array of (ra, dec, mag).
        img_stars: [ndarray] (N, 3) array of (x, y, intensity).
        repeats: [int] Number of times to run the fit.

    Return:
        None
    """

    # Perturb the pointing and refit it, suppressing the per-fit console output
    def run():
        pp_pert = copy.deepcopy(pp)
        pp_pert.RA_d = (pp_pert.RA_d + 0.4/np.cos(np.radians(pp.dec_d))) % 360
        pp_pert.dec_d = pp_pert.dec_d + 0.3
        pp_pert.pos_angle_ref = pp_pert.pos_angle_ref + 0.2
        with contextlib.redirect_stdout(io.StringIO()):
            res = pp_pert.fitPointingNN(JD_OBS, img_stars, catalog_stars, fixed_scale=True)
        return res, pp_pert

    # Report the timing together with how far the recovered pointing is from the truth
    best, mean, (res, pp_fit) = timeIt(run, repeats)
    dra = ((pp_fit.RA_d - pp.RA_d + 180) % 360 - 180)*np.cos(np.radians(pp.dec_d))
    ddec = pp_fit.dec_d - pp.dec_d
    print("(a) fitPointingNN:      best {:7.3f} s  mean {:7.3f} s   success={} rmsd={:.3f} px "
          "inlier_frac={:.2f} dRA*cos={:.4f} deg dDec={:.4f} deg".format(
              best, mean, res[0], res[1], res[2], dra, ddec))


def benchmarkFitAstrometryRansac(pp, catalog_stars, img_stars, repeats):
    """ (b) full RANSAC fitAstrometry path with the NN cost, as run by AutoPlatepar.

    Arguments:
        pp: [Platepar] Synthetic ground-truth platepar.
        catalog_stars: [ndarray] (M, 3) array of (ra, dec, mag).
        img_stars: [ndarray] (N, 3) array of (x, y, intensity).
        repeats: [int] Number of times to run the fit.

    Return:
        None
    """

    # Start from a slightly offset pointing and a reset radial5-odd distortion, as AutoPlatepar does
    def run():
        pp_start = copy.deepcopy(pp)
        pp_start.RA_d = (pp_start.RA_d + 0.05) % 360
        pp_start.dec_d = pp_start.dec_d + 0.05
        pp_start.refraction = True
        pp_start.equal_aspect = True
        pp_start.asymmetry_corr = False
        pp_start.force_distortion_centre = False
        pp_start.setDistortionType("radial5-odd", reset_params=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = pp_start.fitAstrometry(JD_OBS, img_stars, catalog_stars, first_platepar_fit=True,
                                            use_nn_cost=True)
        return result, pp_start

    best, mean, (result, pp_fit) = timeIt(run, repeats)

    # A failed fit returns nothing to measure the residuals on
    if result is None:
        print("(b) fitAstrometry RANSAC: best {:7.3f} s  mean {:7.3f} s   -> returned None".format(
            best, mean))
        return

    # Compute the image residual of the matched pairs under the fitted platepar
    img_matched, cat_matched = result
    x, y = raDecToXYPP(cat_matched[:, 0], cat_matched[:, 1], JD_OBS, pp_fit)
    rmsd = np.sqrt(np.mean((x - img_matched[:, 0])**2 + (y - img_matched[:, 1])**2))
    print("(b) fitAstrometry RANSAC: best {:7.3f} s  mean {:7.3f} s   {:d} matched, rmsd={:.3f} px, "
          "x_poly_fwd[-3:]={}".format(best, mean, len(img_matched), rmsd,
                                      np.array2string(pp_fit.x_poly_fwd[-3:], precision=5)))


def benchmarkFilterBlendedStars(pp, catalog_stars, paired_stars, repeats):
    """ (c) filterBlendedStars on the true pairs.

    Arguments:
        pp: [Platepar] Synthetic ground-truth platepar.
        catalog_stars: [ndarray] (M, 3) array of (ra, dec, mag).
        paired_stars: [PairedStars] Image/catalog pairs of the true detections.
        repeats: [int] Number of times to run the filter.

    Return:
        None
    """

    def run():
        return filterBlendedStars(paired_stars, catalog_stars, pp, JD_OBS, lim_mag=9.0)

    best, mean, (filtered, n_removed) = timeIt(run, repeats)
    print("(c) filterBlendedStars: best {:7.3f} s  mean {:7.3f} s   {:d} in, {:d} removed".format(
        best, mean, len(paired_stars), n_removed))


def benchmarkDuplicateRemoval(img_stars, rng, repeats):
    """ (d) ExtractStars PSF duplicate removal on detections with injected duplicates.

    Arguments:
        img_stars: [ndarray] (N, 3) array of (x, y, intensity).
        rng: [np.random.RandomState] Random generator used to inject the duplicates.
        repeats: [int] Number of times to run the duplicate removal.

    Return:
        None
    """

    # Inject duplicates: 15% of the stars get a second detection converging within the segment radius
    n_dup = int(0.15*len(img_stars))
    dup_idx = rng.choice(len(img_stars), n_dup, replace=False)
    dup = img_stars[dup_idx].copy()
    dup[:, 0] += rng.normal(0.0, 0.5, n_dup)
    dup[:, 1] += rng.normal(0.0, 0.5, n_dup)
    dup[:, 2] *= rng.uniform(0.5, 1.5, n_dup)
    all_stars = np.vstack([img_stars, dup])
    all_stars = all_stars[rng.permutation(len(all_stars))]
    x_arr, y_arr, intens_arr = all_stars.T
    segment_radius = 4

    # Prefer the library helper, fall back to the inline copy on older branches
    helper = getattr(ExtractStars, "duplicateDetectionKeepMask", None)
    label = "library helper" if helper is not None else "inline reference copy"
    if helper is None:
        helper = _duplicateKeepMaskReference

    def run():
        return helper(x_arr, y_arr, intens_arr, segment_radius)

    best, mean, keep = timeIt(run, repeats)
    print("(d) duplicate removal:  best {:7.3f} s  mean {:7.3f} s   {:d} in, {:d} removed ({})".format(
        best, mean, len(x_arr), int(np.sum(~keep)), label))


### Entry point ###

def main():
    """ Parse the command line arguments, build the synthetic field and run the selected benchmarks.

    Return:
        None
    """

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Benchmark the astrometric fitting hot spots.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per benchmark (best is reported).")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for the synthetic field.")
    parser.add_argument("--n-catalog", type=int, default=3000, help="Number of catalog stars in the FOV.")
    parser.add_argument("--skip", default="", help="Comma-separated benchmarks to skip, e.g. 'b,d'.")
    args = parser.parse_args()

    # Seed the generator and read the list of benchmarks to skip
    rng = np.random.RandomState(args.seed)
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    # Build the synthetic star field once and share it between the benchmarks
    pp = buildSyntheticPlatepar()
    catalog_stars = buildSyntheticCatalog(pp, rng, n_target=args.n_catalog)
    img_stars, truth = buildSyntheticDetections(pp, catalog_stars, rng)
    paired_stars = buildPairedStars(pp, catalog_stars, img_stars, truth, rng)

    # Describe the field that the benchmarks run on
    print("Synthetic field: {:d} catalog stars in FOV, {:d} detections ({:d} false), {:d} true pairs".format(
        len(catalog_stars), len(img_stars), int(np.sum(truth < 0)), len(paired_stars)))
    print("Platepar: RA={:.3f} Dec={:.3f} rot={:.2f} F_scale={:.2f} {}".format(
        pp.RA_d, pp.dec_d, pp.pos_angle_ref, pp.F_scale, pp.distortion_type))
    print()

    # Run the benchmarks that were not skipped
    if "a" not in skip:
        benchmarkFitPointingNN(pp, catalog_stars, img_stars, args.repeats)
    if "b" not in skip:
        benchmarkFitAstrometryRansac(pp, catalog_stars, img_stars, args.repeats)
    if "c" not in skip:
        benchmarkFilterBlendedStars(pp, catalog_stars, paired_stars, args.repeats)
    if "d" not in skip:
        benchmarkDuplicateRemoval(img_stars, np.random.RandomState(args.seed + 1), args.repeats)


if __name__ == "__main__":
    main()
