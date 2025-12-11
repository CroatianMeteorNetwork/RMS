
from __future__ import print_function, division, absolute_import

import os
import inspect
import math

import numpy as np
from PIL import Image
from astropy.wcs import WCS

from RMS.ExtractStars import extractStarsAuto
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.AstrometryNetNova import novaAstrometryNetSolve
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz
from RMS.Logger import getLogger

try:
    import astrometry
    ASTROMETRY_NET_AVAILABLE = True

except ImportError:
    ASTROMETRY_NET_AVAILABLE = False


def matchStarsIterative(x_data, y_data, input_intensities, catalog_stars, wcs_obj,
                        img_width, img_height, lat, lon, scale_px_per_deg, jd, verbose=False):
    """
    Iteratively match input stars to catalog stars, starting with bright stars.

    Uses WCS for first pass, then fits a platepar and uses it for subsequent passes
    with progressively more stars and tighter radius. This handles lens distortion
    by fitting distortion parameters early with bright, unambiguous matches.

    Args:
        x_data, y_data: Input star pixel coordinates (arrays)
        input_intensities: Input star intensities for sorting (array, can be None)
        catalog_stars: List of catalog stars from astrometry.net (match.stars)
        wcs_obj: Initial WCS from astrometry.net solution
        img_width, img_height: Image dimensions
        lat, lon: Station latitude/longitude (degrees)
        scale_px_per_deg: Image scale in pixels per degree (F_scale)
        jd: Julian date
        verbose: Print progress info

    Returns:
        matched_pairs: List of dicts with input_x, input_y, catalog_ra, catalog_dec, dist_px
        pp: Fitted Platepar object (or None if fitting failed)
    """
    # Sort catalog stars by magnitude (brightest first, lower mag = brighter)
    catalog_sorted = sorted(catalog_stars,
                           key=lambda s: s.metadata.get('mag', 99) if hasattr(s, 'metadata') else 99)

    # Sort input stars by intensity (brightest first, higher intensity = brighter)
    if input_intensities is not None and len(input_intensities) == len(x_data):
        bright_order = np.argsort(-np.array(input_intensities))  # descending
    else:
        bright_order = np.arange(len(x_data))

    # Get image center RA/Dec from WCS
    ra_center, dec_center = wcs_obj.all_pix2world(img_width/2, img_height/2, 1)

    # Create a minimal platepar for fitting
    pp = Platepar()
    pp.lat = lat
    pp.lon = lon
    pp.X_res = int(img_width)
    pp.Y_res = int(img_height)
    pp.F_scale = scale_px_per_deg
    pp.RA_d = float(ra_center)
    pp.dec_d = float(dec_center)
    pp.pos_angle_ref = 0.0
    pp.JD = jd
    pp.refraction = False

    # Compute Ho (hour angle offset)
    J2000_DAYS = 2451545.0
    T = (jd - J2000_DAYS) / 36525.0
    pp.Ho = (280.46061837 + 360.98564736629*(jd - J2000_DAYS)
             + 0.000387933*T**2 - T**3/38710000.0) % 360

    # Set distortion type and zero distortion for initial fit
    pp.setDistortionType("radial3-odd", reset_params=True)

    # Compute az/alt center for platepar
    azim, alt = cyTrueRaDec2ApparentAltAz(
        math.radians(pp.RA_d), math.radians(pp.dec_d),
        jd, math.radians(lat), math.radians(lon), refraction=False
    )
    pp.az_centre = math.degrees(azim)
    pp.alt_centre = math.degrees(alt)

    # Iteration parameters: n_catalog_stars, n_input_stars, radius_px, do_fit
    iterations = [
        {'n_cat': 10,  'n_input': 10,  'radius_px': 50, 'fit': True,  'desc': 'bright'},
        {'n_cat': 30,  'n_input': 50,  'radius_px': 20, 'fit': True,  'desc': 'medium'},
        {'n_cat': 999, 'n_input': 999, 'radius_px': 10, 'fit': False, 'desc': 'all'},
    ]

    matched_pairs = []
    use_platepar = False  # Start with WCS

    print("Iterative star matching:")
    print("  Catalog stars available: {:d}".format(len(catalog_sorted)))
    print("  Input stars available: {:d}".format(len(x_data)))
    if input_intensities is not None:
        print("  Input intensities: provided (sorting by brightness)")
    else:
        print("  Input intensities: not provided (using original order)")

    for iteration in iterations:
        # Select brightest N catalog stars
        n_cat = min(iteration['n_cat'], len(catalog_sorted))
        cat_subset = catalog_sorted[:n_cat]

        # Select brightest N input stars
        n_input = min(iteration['n_input'], len(x_data))
        input_indices = bright_order[:n_input]

        print("  Pass '{}': {} cat stars, {} input stars, radius={} px".format(
            iteration['desc'], n_cat, n_input, iteration['radius_px']))

        # Project catalog stars to pixel coordinates
        catalog_pixels = []
        for star in cat_subset:
            if not use_platepar:
                # First iteration: use WCS
                cx, cy = wcs_obj.all_world2pix(star.ra_deg, star.dec_deg, 1)
            else:
                # Subsequent: use fitted platepar
                cx_arr, cy_arr = raDecToXYPP(
                    np.array([star.ra_deg]), np.array([star.dec_deg]), jd, pp
                )
                cx, cy = cx_arr[0], cy_arr[0]

            cat_mag = star.metadata.get('mag', 99) if hasattr(star, 'metadata') else 99
            catalog_pixels.append((float(cx), float(cy), star.ra_deg, star.dec_deg, cat_mag))

        # Match input stars to catalog (nearest within radius)
        new_pairs = []
        used_catalog = set()  # Prevent multiple inputs matching same catalog star

        for idx in input_indices:
            ix, iy = x_data[idx], y_data[idx]
            best_dist = float('inf')
            best_cat_idx = None
            best_cat = None

            for cat_idx, (cx, cy, ra, dec, mag) in enumerate(catalog_pixels):
                if cat_idx in used_catalog:
                    continue
                dist = np.sqrt((ix - cx)**2 + (iy - cy)**2)
                if dist < best_dist and dist < iteration['radius_px']:
                    best_dist = dist
                    best_cat_idx = cat_idx
                    best_cat = (cx, cy, ra, dec, mag)

            if best_cat is not None:
                used_catalog.add(best_cat_idx)
                new_pairs.append({
                    'input_x': float(ix),
                    'input_y': float(iy),
                    'catalog_x': best_cat[0],
                    'catalog_y': best_cat[1],
                    'catalog_ra': best_cat[2],
                    'catalog_dec': best_cat[3],
                    'dist_px': best_dist
                })

        matched_pairs = new_pairs
        print("    -> {:d} matches found".format(len(matched_pairs)))

        # Fit platepar with current matches (except last iteration)
        if iteration['fit'] and len(matched_pairs) >= 4:
            img_stars = np.array([[m['input_x'], m['input_y'], 1.0] for m in matched_pairs])
            cat_stars = np.array([[m['catalog_ra'], m['catalog_dec'], 1.0] for m in matched_pairs])

            try:
                pp.fitAstrometry(jd, img_stars, cat_stars, first_platepar_fit=True)
                use_platepar = True  # Switch to using platepar for next iteration
                print("    -> Fit OK, RA={:.2f} Dec={:.2f}".format(pp.RA_d, pp.dec_d))
            except Exception as e:
                print("    -> Fit FAILED: {}".format(e))

    return matched_pairs, pp if use_platepar else None


def astrometryNetSolveLocal(ff_file_path=None, img=None, mask=None, x_data=None, y_data=None,
                            fov_w_range=None, max_stars=100, verbose=False, x_center=None, y_center=None,
                            lat=None, lon=None, jd=None, input_intensities=None):
    """ Find an astrometric solution of X, Y image coordinates of stars detected on an image using the
        local installation of astrometry.net.

    Keyword arguments:
        ff_file_path: [str] Path to the FF file to load.
        img: [ndarray] Numpy array containing image data.
        mask: [ndarray] Mask image. None by default.
        x_data: [list] A list of star x image coordinates.
        y_data: [list] A list of star y image coordinates
        fov_w_range: [2 element tuple] A tuple of scale_lower and scale_upper, i.e. the estimate of the
            width of the FOV in degrees.
        max_stars: [int] Maximum number of stars to use for the astrometry.net solution. Default is 100.
        verbose: [bool] Print verbose output. Default is False.
        x_center: [float] X coordinate of the image center. Default is None.
        y_center: [float] Y coordinate of the image center. Default is None.
        lat: [float] Station latitude in degrees. Required for iterative matching.
        lon: [float] Station longitude in degrees. Required for iterative matching.
        jd: [float] Julian date. Required for iterative matching.
        input_intensities: [ndarray] Star intensities for brightness-based matching. Optional.

    Returns:
        [tuple] A tuple containing the following elements:
            - ra_mid: [float] Right ascension of the image center in degrees.
            - dec_mid: [float] Declination of the image center in degrees.
            - rot_eq_standard: [float] Equatorial orientation in degrees.
            - scale: [float] Scale in arcsec/pixel.
            - fov_w: [float] Width of the FOV in degrees.
            - fov_h: [float] Height of the FOV in degrees.
            - star_data: [list] A list of star data, where star_data = [x_data, y_data].
    """

    # Read the FF file, if given
    if ff_file_path is not None:
        
        # Read the FF file
        ff = readFF(*os.path.split(ff_file_path))
        img = ff.avepixel
        bit_depth = ff.nbits

    # If the image is given as a numpy array, use it
    elif img is not None:
        img = img

        bit_depth = img.dtype.itemsize*8

    else:
        img = None

    
    # Get an astrometry.net solution on an image
    if img is not None:

        # If an image has been given and no star x and y coordinates have been given, extract the stars
        if x_data is None or y_data is None:
            
            # Automatically extract stars from the image
            x_data, y_data, _, _, _, _, _, _  = extractStarsAuto(img, mask=mask, max_star_candidates=1500, 
                segment_radius=8, min_stars_detect=50, max_stars_detect=150, bit_depth=bit_depth, 
                verbose=verbose
            )
            


    # If there are too many stars, select the brightest ones (or random if no intensities)
    if len(x_data) > max_stars:

        if verbose:
            print("Too many stars found: ", len(x_data))

        if input_intensities is not None and len(input_intensities) == len(x_data):
            # Select the brightest stars (highest intensity)
            if verbose:
                print("Selecting {:d} brightest stars...".format(max_stars))
            bright_indices = np.argsort(-np.array(input_intensities))[:max_stars]
            x_data = x_data[bright_indices]
            y_data = y_data[bright_indices]
            input_intensities = np.array(input_intensities)[bright_indices]
        else:
            # Fall back to random selection if no intensities available
            if verbose:
                print("Randomly selecting {:d} stars...".format(max_stars))
            rand_indices = np.random.choice(len(x_data), max_stars, replace=False)
            x_data = x_data[rand_indices]
            y_data = y_data[rand_indices]


    # Print the found star coordinates
    if verbose:
        print("Stars for astrometry.net: ", len(x_data))
        print("        X,       Y")   
        for x, y in zip(x_data, y_data):
            print("{:8.2f} {:8.2f}".format(x, y))



        print()
        print("Solving the image using the local installation of astrometry.net...")

    # Get a path to this file
    this_file_path = os.path.abspath(__file__)

    # Construct a path to RMS/share/astrometry_cache, which is ../../share/astrometry_cache from this file
    astrometry_cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(this_file_path))), "share", "astrometry_cache"
        )


    # Solve the image using the local installation of astrometry.net
    solver = astrometry.Solver(
        astrometry.series_4100.index_files(
            cache_directory=astrometry_cache_path,
            scales={15, 18}, # Skymark diameter scales, chosen for meteor cameras. See: 
                             # https://pypi.org/project/astrometry/#choosing-series
        )
    )

    size_hint = None

    if fov_w_range is not None:

        # Get the image width
        if img is not None:
            img_width = img.shape[1]
        else:
            img_width = np.max(x_data)

        # Compute arcsec per pixel for the FOV range
        lower_arcsec_per_pixel = fov_w_range[0]*3600/img_width
        upper_arcsec_per_pixel = fov_w_range[1]*3600/img_width

        if verbose:
            print("FOV range:")
            print("  {:.2f} - {:.2f} deg".format(fov_w_range[0], fov_w_range[1]))
            print("  {:.2f} - {:.2f} arcsec/pixel".format(lower_arcsec_per_pixel, upper_arcsec_per_pixel))


        size_hint=astrometry.SizeHint(
            lower_arcsec_per_pixel=lower_arcsec_per_pixel,
            upper_arcsec_per_pixel=upper_arcsec_per_pixel
        )

    # Print progress info
    if verbose:
        getLogger(level="INFO")

    # Init solution parameters
    solution_parameters = astrometry.SolutionParameters(
            # Return the first solution if the log odds ratio is greater than 100 (very good solution)
            logodds_callback=lambda logodds_list: (
            astrometry.Action.STOP
            if logodds_list[0] > 100.0
            else astrometry.Action.CONTINUE
            )
    )

    # If the solver.solve has the argument "stars", use a 2D array of stars instead of stars_xs and stars_ys
    solve_args = inspect.getfullargspec(solver.solve).args
    if "stars" in solve_args:
        star_data = np.array([x_data, y_data]).T

        solution = solver.solve(
            stars=star_data,
            size_hint=size_hint,
            position_hint=None,
            solution_parameters=solution_parameters
            )

    else:

        solution = solver.solve(
            stars_xs=x_data,
            stars_ys=y_data,
            size_hint=size_hint,
            position_hint=None,
            solution_parameters=solution_parameters
            )

    if solution.has_match():
        
        # print()
        # print("Found solution for image center:")

        if verbose:
            # Print the WCS fields
            print()
            print("WCS fields:")
            print('-----------------------------')
            for key, value in solution.best_match().wcs_fields.items():
                print("{:8s}: {}".format(key, ", ".join(map(str, value))))
            
            print()

        # Load the solution into an astropy WCS object
        wcs_obj = WCS(solution.best_match().wcs_fields)

        # Get the image center in pixel coordinates
        if (x_center is None) or (y_center is None):
            if img is not None:
                x_center = img.shape[1]/2
                y_center = img.shape[0]/2
            else:
                x_center = np.median(x_data)
                y_center = np.median(y_data)

        # print("Image center, x = {:.2f}, y = {:.2f}".format(x_center, y_center))

        # Use wcs.all_pix2world to get the RA and Dec at the new center
        ra_mid, dec_mid = wcs_obj.all_pix2world(x_center, y_center, 1)

        # Image coordinate slightly right of the centre
        x_right = x_center + 10
        y_right = y_center
        ra_right, dec_right = wcs_obj.all_pix2world(x_right, y_right, 1)

        # Compute the equatorial orientation
        rot_eq_standard = np.degrees(np.arctan2(np.radians(dec_mid) - np.radians(dec_right), \
            np.radians(ra_mid) - np.radians(ra_right)))%360


        # Compute the scale in px/deg
        scale = 3600/solution.best_match().scale_arcsec_per_pixel

        # Compute the FOV size in degrees
        if img is not None:

            img_wid, img_ht = np.max(img.shape), np.min(img.shape)

            fov_w = img_wid*solution.best_match().scale_arcsec_per_pixel/3600
            fov_h = img_ht *solution.best_match().scale_arcsec_per_pixel/3600

        else:
            # Take the range of image coordinates as a FOV indicator
            x_max = np.max(x_data)
            y_max = np.max(y_data)

            fov_w = x_max*solution.best_match().scale_arcsec_per_pixel/3600
            fov_h = y_max*solution.best_match().scale_arcsec_per_pixel/3600


        star_data = [x_data, y_data]

        match = solution.best_match()

        # Note: match.stars contains ALL catalog stars in the FOV region from the index,
        # NOT just the ones that matched to input stars.
        # We don't use these - RMS has its own better star catalog for matching.

        # Extract quad stars (the specific catalog stars used for initial geometric matching)
        quad_stars = []
        if hasattr(match, 'quad_stars') and match.quad_stars:
            for star in match.quad_stars:
                x_pix, y_pix = wcs_obj.all_world2pix(star.ra_deg, star.dec_deg, 1)
                quad_stars.append({
                    'ra_deg': star.ra_deg,
                    'dec_deg': star.dec_deg,
                    'x_pix': float(x_pix),
                    'y_pix': float(y_pix),
                    'metadata': star.metadata if hasattr(star, 'metadata') else {}
                })

        # Additional solution info - no matched pairs from astrometry.net
        # Star matching will be done in SkyFit2 using RMS's own catalog
        solution_info = {
            'logodds': match.logodds,
            'quad_stars': quad_stars,
            'index_path': str(match.index_path) if hasattr(match, 'index_path') else None,
            'wcs_obj': wcs_obj,
            'input_star_count': len(x_data)  # How many stars we sent to the solver
        }

        if verbose:
            print()
            print("Quad stars: {:d}".format(len(quad_stars)))
            print("Log odds: {:.2f}".format(match.logodds))

        return ra_mid, dec_mid, rot_eq_standard, scale, fov_w, fov_h, star_data, solution_info
    

    else:
        print("No solution found.")
        return None


def astrometryNetSolve(ff_file_path=None, img=None, mask=None, x_data=None, y_data=None, fov_w_range=None,
                       max_stars=100, verbose=False, x_center=None, y_center=None,
                       lat=None, lon=None, jd=None, input_intensities=None):
    """ Find an astrometric solution of X, Y image coordinates of stars detected on an image using the
        local installation of astrometry.net.

    Keyword arguments:
        ff_file_path: [str] Path to the FF file to load.
        img: [ndarray] Numpy array containing image data.
        mask: [ndarray] Mask image. None by default.
        x_data: [list] A list of star x image coordinates.
        y_data: [list] A list of star y image coordinates
        fov_w_range: [2 element tuple] A tuple of scale_lower and scale_upper, i.e. the estimate of the
            width of the FOV in degrees.
        max_stars: [int] Maximum number of stars to use for the astrometry.net solution. Default is 100.
        verbose: [bool] Print verbose output. Default is False.
        x_center: [float] X coordinate of the image center. Default is None.
        y_center: [float] Y coordinate of the image center. Default is None.
        lat: [float] Station latitude in degrees. Required for iterative matching.
        lon: [float] Station longitude in degrees. Required for iterative matching.
        jd: [float] Julian date. Required for iterative matching.
        input_intensities: [ndarray] Star intensities for brightness-based matching. Optional.
    """

    # If the local installation of astrometry.net is not available, use the nova.astrometry.net API
    if not ASTROMETRY_NET_AVAILABLE:
        return novaAstrometryNetSolve(
            ff_file_path=ff_file_path, img=img, x_data=x_data, y_data=y_data,
            fov_w_range=fov_w_range, x_center=x_center, y_center=y_center
            )


    else:

        # Try to solve the image using the local installation of astrometry.net
        try:
            return astrometryNetSolveLocal(
                ff_file_path=ff_file_path, img=img, mask=mask, x_data=x_data, y_data=y_data,
                fov_w_range=fov_w_range, max_stars=max_stars, verbose=verbose,
                x_center=x_center, y_center=y_center,
                lat=lat, lon=lon, jd=jd, input_intensities=input_intensities
                )

        # If it fails, use the nova.astrometry.net API
        except Exception as e:

            print("Local astrometry.net solver failed with error:")
            print(e)
            print("Trying the nova.astrometry.net API...")

            return novaAstrometryNetSolve(
                ff_file_path=ff_file_path, img=img, x_data=x_data, y_data=y_data,
                fov_w_range=fov_w_range, x_center=x_center, y_center=y_center
                )


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Run astrometry.net on an FF file.")

    arg_parser.add_argument('input_path', metavar='INPUT_PATH', type=str, 
                            help='Path to the FF file to load.')
    
    arg_parser.add_argument('--fov', metavar='FOV', type=float, default=None,
                            help='Estimate of the width of the FOV in degrees. E.g. --fov 20')
    
    # Parse the arguments
    cml_args = arg_parser.parse_args()


    # Set the FOV range
    if cml_args.fov is not None:
        fov_w_range = (0.5*cml_args.fov, 2*cml_args.fov)
    else:
        fov_w_range = None

    # Run the astrometry.net solver
    status = astrometryNetSolve(ff_file_path=cml_args.input_path, fov_w_range=fov_w_range, verbose=True)

    if status is not None:

        ra_mid, dec_mid, rot_eq_standard, scale, fov_w, fov_h, star_data, solution_info = status

        print("Astrometry.net solution:")
        print()
        print("RA  = {:.2f} deg".format(ra_mid))
        print("Dec = {:+.2f} deg".format(dec_mid))
        print("Scale = {:.2f} arcmin/pixel".format(scale))
        print("Rot. eq. standard = {:.2f} deg".format(rot_eq_standard))
        print("FOV = {:.2f} x {:.2f} deg".format(fov_w, fov_h))

        # Print matched star info if available
        if solution_info is not None:
            matched_stars = solution_info.get('matched_stars', [])
            quad_stars = solution_info.get('quad_stars', [])
            logodds = solution_info.get('logodds')

            print()
            print("Solution info:")
            if logodds is not None:
                print("  Log odds: {:.2f}".format(logodds))
            print("  Matched stars: {:d}".format(len(matched_stars)))
            print("  Quad stars: {:d}".format(len(quad_stars)))

            if matched_stars:
                print()
                print("Matched stars (first 10):")
                print("  {:>10s} {:>10s} {:>8s} {:>8s}".format("RA", "Dec", "X", "Y"))
                for star in matched_stars[:10]:
                    print("  {:10.5f} {:+10.5f} {:8.2f} {:8.2f}".format(
                        star['ra_deg'], star['dec_deg'], star['x_pix'], star['y_pix']))

    else:
        print("No solution found.")