
from __future__ import print_function, division, absolute_import

import inspect
import math
import os

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

# Allow disabling local astrometry.net via environment variable for testing
import os
if os.environ.get('RMS_DISABLE_LOCAL_ASTROMETRY', '').lower() in ('1', 'true', 'yes'):
    ASTROMETRY_NET_AVAILABLE = False
    print("NOTE: Local astrometry.net disabled via RMS_DISABLE_LOCAL_ASTROMETRY")


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
                            fov_w_range=None, fov_w_hint=None, max_stars=100, verbose=False, x_center=None, y_center=None,
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



    # For very wide FOV images (>90 deg), filter to central region FIRST before selecting brightest
    # This avoids distortion issues at the edges of fisheye lenses
    # Use fov_w_hint (best guess from config) if available, otherwise use range midpoint
    estimated_fov = fov_w_hint if fov_w_hint is not None else (
        (fov_w_range[0] + fov_w_range[1]) / 2.0 if fov_w_range is not None else None
    )

    print("DEBUG: fov_w_hint={}, fov_w_range={}, estimated_fov={}".format(fov_w_hint, fov_w_range, estimated_fov))

    if estimated_fov is not None and estimated_fov > 90:

        # Determine image center
        if img is not None:
            img_center_x = img.shape[1] / 2.0
            img_center_y = img.shape[0] / 2.0
        elif x_center is not None and y_center is not None:
            img_center_x = x_center
            img_center_y = y_center
        else:
            # Estimate center from star positions
            img_center_x = (np.max(x_data) + np.min(x_data)) / 2.0
            img_center_y = (np.max(y_data) + np.min(y_data)) / 2.0

        # Calculate the radius that corresponds to ~60 deg FOV from center
        if img is not None:
            img_radius = min(img.shape[0], img.shape[1]) / 2.0
        elif x_center is not None and y_center is not None:
            img_radius = min(x_center, y_center)
        else:
            img_radius = max(np.max(x_data) - np.min(x_data), np.max(y_data) - np.min(y_data)) / 2.0

        # Use central 60 degrees for very wide FOV cameras
        central_fov_fraction = min(60.0 / estimated_fov, 0.7)
        central_radius = img_radius * central_fov_fraction

        # Filter stars to only those within the central region
        distances = np.sqrt((x_data - img_center_x)**2 + (y_data - img_center_y)**2)
        central_mask = distances <= central_radius
        central_star_count = np.sum(central_mask)

        print("Wide FOV check: estimated_fov={:.1f} deg, central_fov_fraction={:.2f}".format(
            estimated_fov, central_fov_fraction))
        print("  Image center: ({:.1f}, {:.1f}), radius: {:.1f} px".format(
            img_center_x, img_center_y, img_radius))
        print("  Central radius: {:.1f} px, stars in central region: {:d}/{:d}".format(
            central_radius, central_star_count, len(x_data)))

        if central_star_count >= 20:  # Only filter if we have enough central stars
            original_count = len(x_data)
            x_data = x_data[central_mask]
            y_data = y_data[central_mask]
            if input_intensities is not None:
                input_intensities = np.array(input_intensities)[central_mask]

            # Update FOV range to reflect the filtered central region
            filtered_fov = estimated_fov * central_fov_fraction * 2
            fov_w_range = [filtered_fov * 0.75, filtered_fov * 1.5]
            print("  -> Filtering to central {:.1f} deg, Stars: {:d} -> {:d}".format(
                filtered_fov, original_count, len(x_data)))
            print("  -> Updated FOV range for scale selection: {:.1f} - {:.1f} deg".format(
                fov_w_range[0], fov_w_range[1]))
        else:
            print("  -> NOT filtering (need >= 20 central stars)")

    # Select brightest stars if too many
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

    # Default scales for ~45 deg FOV height (covers 4.5° to 45° quads)
    scales = {14, 15, 16, 17, 18, 19}

    size_hint = None

    if fov_w_range is not None:

        # Get image dimensions to compute aspect ratio
        if img is not None:
            img_width = img.shape[1]
            img_height = img.shape[0]
        else:
            img_width = np.max(x_data)
            img_height = np.max(y_data) if y_data is not None else img_width * 0.75

        aspect_ratio = img_height / img_width

        # Use the average FOV estimate for the width
        avg_fov_w = (fov_w_range[0] + fov_w_range[1]) / 2.0

        # Compute FOV height (shorter dimension for landscape images)
        fov_h = avg_fov_w * aspect_ratio

        # Use the "10% to 100% of image size" rule on the short side
        # Quad sizes should range from 10% to 100% of the shorter FOV dimension
        min_quad_size_deg = fov_h * 0.10
        max_quad_size_deg = fov_h

        if verbose:
            print("FOV range: {:.2f} - {:.2f} deg (width)".format(fov_w_range[0], fov_w_range[1]))
            print("FOV height estimate: {:.2f} deg".format(fov_h))
            print("Quad size range: {:.2f} - {:.2f} deg".format(min_quad_size_deg, max_quad_size_deg))

        # Index 4100 series quad diameter ranges (in degrees):
        # Scale 14: 240-340 arcmin    (4.0-5.7°)
        # Scale 15: 340-480 arcmin    (5.7-8.0°)
        # Scale 16: 480-680 arcmin    (8.0-11.3°)
        # Scale 17: 680-1000 arcmin   (11.3-16.7°)
        # Scale 18: 1000-1400 arcmin  (16.7-23.3°)
        # Scale 19: 1400-2000 arcmin  (23.3-33.3°)

        # Quad diameter boundaries for 4100 series scales
        # Formula: quad diameter = 240/sqrt(2)^7 * sqrt(2)^N arcmin
        # This gives scale 14 = 240-340 arcmin, matching astrometry.net docs
        base_arcmin = 240.0 / math.pow(math.sqrt(2), 7)  # ~21.2 arcmin
        scale_quad_ranges_deg = []
        for scale_num in range(7, 20):
            lower_arcmin = base_arcmin * math.pow(math.sqrt(2), scale_num - 7)
            upper_arcmin = base_arcmin * math.pow(math.sqrt(2), scale_num - 6)
            lower_deg = lower_arcmin / 60.0
            upper_deg = upper_arcmin / 60.0
            scale_quad_ranges_deg.append((scale_num, lower_deg, upper_deg))

        # Find scales whose quad ranges overlap with our desired range
        # Add 10% margin on the lower bound to avoid boundary precision issues
        min_quad_with_margin = min_quad_size_deg * 0.9
        matching_scales = []

        # Target quad size is ~30% of FOV height (empirically good for matching)
        target_quad_deg = fov_h * 0.30

        for scale_num, lower_deg, upper_deg in scale_quad_ranges_deg:
            # Check if this scale's quad range overlaps with our desired quad range
            if min_quad_with_margin <= upper_deg and max_quad_size_deg >= lower_deg:
                # Calculate how close this scale's midpoint is to our target
                mid_deg = (lower_deg + upper_deg) / 2.0
                distance = abs(mid_deg - target_quad_deg)
                matching_scales.append((scale_num, distance))

        # Sort by distance to target (closest first), then convert to set for astrometry lib
        matching_scales.sort(key=lambda x: x[1])
        scales = {s[0] for s in matching_scales}

        # If no scales found, fall back to defaults for ~45 deg FOV height
        if not scales:
            scales = {14, 15, 16, 17, 18, 19}

        if verbose:
            # Show scales in preferred order (closest to target quad size first)
            preferred_order = [s[0] for s in matching_scales]
            print("Using index scales (preferred order): {}".format(preferred_order))
            print("Target quad size: {:.1f} deg (30% of FOV height)".format(target_quad_deg))
            for s, dist in matching_scales:
                lower_arcmin = base_arcmin * math.pow(math.sqrt(2), s - 7)
                upper_arcmin = base_arcmin * math.pow(math.sqrt(2), s - 6)
                mid_deg = (lower_arcmin + upper_arcmin) / 2.0 / 60.0
                print("  Scale {}: {:.0f}-{:.0f} arcmin ({:.1f}-{:.1f} deg), mid={:.1f} deg".format(
                    s, lower_arcmin, upper_arcmin, lower_arcmin/60, upper_arcmin/60, mid_deg))

        # Compute pixel scale for size_hint (helps solver converge faster)
        lower_arcsec_per_pixel = fov_w_range[0] * 3600 / img_width
        upper_arcsec_per_pixel = fov_w_range[1] * 3600 / img_width

        size_hint = astrometry.SizeHint(
            lower_arcsec_per_pixel=lower_arcsec_per_pixel,
            upper_arcsec_per_pixel=upper_arcsec_per_pixel
        )

    # Solve the image using the local installation of astrometry.net
    solver = astrometry.Solver(
        astrometry.series_4100.index_files(
            cache_directory=astrometry_cache_path,
            scales=scales,
        )
    )

    # Print progress info
    if verbose:
        getLogger(level="INFO")

    # Init solution parameters
    solution_parameters = astrometry.SolutionParameters(
            # Return the first solution if the log odds ratio is greater than 60 (good solution)
            # Lower threshold means faster solving - logodds > 50 is typically reliable
            logodds_callback=lambda logodds_list: (
            astrometry.Action.STOP
            if logodds_list[0] > 60.0
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

        # Use wcs.all_pix2world to get the RA and Dec at the new center
        ra_mid, dec_mid = wcs_obj.all_pix2world(x_center, y_center, 1)

        if verbose:
            # Print the WCS fields
            print()
            print("WCS fields from astrometry.net:")
            print('-----------------------------')
            for key, value in solution.best_match().wcs_fields.items():
                print("{:8s}: {}".format(key, ", ".join(map(str, value))))
            print()
            print("Image center: x={:.1f}, y={:.1f}".format(x_center, y_center))
            print("RA/Dec at image center: {:.4f}, {:.4f}".format(float(ra_mid), float(dec_mid)))
            print()

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
                       fov_w_hint=None, max_stars=100, verbose=False, x_center=None, y_center=None,
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

    # Import API URLs
    from RMS.Astrometry.AstrometryNetNova import PRIMARY_API_URL, FALLBACK_API_URL

    # Helper to try coordinate-only first, then fall back to image if available
    def _tryRemoteSolve(api_url, ff_path, image, x_coords, y_coords, fov_range, x_cen, y_cen):
        """Try coordinate-only solve first, fall back to image if that fails."""

        # If we have coordinates, try coordinate-only first (faster, less bandwidth)
        if x_coords is not None and y_coords is not None:
            print("  Trying coordinate-only solve...")
            try:
                result = novaAstrometryNetSolve(
                    ff_file_path=None, img=None, x_data=x_coords, y_data=y_coords,
                    fov_w_range=fov_range, x_center=x_cen, y_center=y_cen,
                    api_url=api_url
                )
                if result is not None:
                    return result
                print("  Coordinate-only solve failed.")
            except Exception as e:
                print(f"  Coordinate-only solve error: {e}")

            # Fall back to image if available
            if image is not None or ff_path is not None:
                print("  Falling back to image upload...")
                return novaAstrometryNetSolve(
                    ff_file_path=ff_path, img=image, x_data=None, y_data=None,
                    fov_w_range=fov_range, x_center=x_cen, y_center=y_cen,
                    api_url=api_url
                )
            return None

        # No coordinates, just try with image
        return novaAstrometryNetSolve(
            ff_file_path=ff_path, img=image, x_data=x_coords, y_data=y_coords,
            fov_w_range=fov_range, x_center=x_cen, y_center=y_cen,
            api_url=api_url
        )

    # If the local installation of astrometry.net is not available, use remote API
    if not ASTROMETRY_NET_AVAILABLE:
        # Try primary server (contrailcast) first
        print("Local astrometry.net not available. Trying remote API...")
        print(f"Trying primary server: {PRIMARY_API_URL}")

        try:
            result = _tryRemoteSolve(
                PRIMARY_API_URL, ff_file_path, img, x_data, y_data,
                fov_w_range, x_center, y_center
            )
            if result is not None:
                return result
        except Exception as e:
            print(f"Primary server failed: {e}")

        # Fall back to nova.astrometry.net
        print(f"Trying fallback server: {FALLBACK_API_URL}")
        return _tryRemoteSolve(
            FALLBACK_API_URL, ff_file_path, img, x_data, y_data,
            fov_w_range, x_center, y_center
        )

    else:

        # Try to solve the image using the local installation of astrometry.net
        try:
            return astrometryNetSolveLocal(
                ff_file_path=ff_file_path, img=img, mask=mask, x_data=x_data, y_data=y_data,
                fov_w_range=fov_w_range, fov_w_hint=fov_w_hint, max_stars=max_stars, verbose=verbose,
                x_center=x_center, y_center=y_center,
                lat=lat, lon=lon, jd=jd, input_intensities=input_intensities
                )

        # If local fails, try remote APIs
        except Exception as e:

            print("Local astrometry.net solver failed with error:")
            print(e)

            # Try primary server (contrailcast) first
            print(f"Trying primary server: {PRIMARY_API_URL}")
            try:
                result = _tryRemoteSolve(
                    PRIMARY_API_URL, ff_file_path, img, x_data, y_data,
                    fov_w_range, x_center, y_center
                )
                if result is not None:
                    return result
            except Exception as e2:
                print(f"Primary server failed: {e2}")

            # Fall back to nova.astrometry.net
            print(f"Trying fallback server: {FALLBACK_API_URL}")
            return _tryRemoteSolve(
                FALLBACK_API_URL, ff_file_path, img, x_data, y_data,
                fov_w_range, x_center, y_center
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