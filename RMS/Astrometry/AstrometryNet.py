
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
from RMS.Astrometry.AstrometryNetNova import (novaAstrometryNetSolve, rotationEqStandard, PRIMARY_API_URL,
    FALLBACK_API_URL)
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz
from RMS.Logger import getLogger

try:
    import astrometry
    ASTROMETRY_NET_AVAILABLE = True

except ImportError:
    ASTROMETRY_NET_AVAILABLE = False

# Allow disabling local astrometry.net via environment variable for testing
if os.environ.get('RMS_DISABLE_LOCAL_ASTROMETRY', '').lower() in ('1', 'true', 'yes'):
    ASTROMETRY_NET_AVAILABLE = False
    print("NOTE: Local astrometry.net disabled via RMS_DISABLE_LOCAL_ASTROMETRY")


def astrometryNetSolveLocal(ff_file_path=None, img=None, mask=None, x_data=None, y_data=None,
                            fov_w_range=None, fov_w_hint=None, max_stars=100, verbose=False, x_center=None,
                            y_center=None, lat=None, lon=None, jd=None, input_intensities=None):
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
        fov_w_hint: [float] Best guess of the FOV width in degrees (e.g. from the config). None by default,
            in which case the midpoint of fov_w_range is used to decide whether the image is a very wide
            field that should be solved on its central part only.
        max_stars: [int] Maximum number of stars to use for the astrometry.net solution. Default is 100.
        verbose: [bool] Print verbose output. Default is False.
        x_center: [float] X coordinate of the image center. Default is None.
        y_center: [float] Y coordinate of the image center. Default is None.
        lat: [float] Station latitude in degrees. Accepted for API compatibility, currently unused.
        lon: [float] Station longitude in degrees. Accepted for API compatibility, currently unused.
        jd: [float] Julian date. Accepted for API compatibility, currently unused.
        input_intensities: [ndarray] Star intensities, used to keep the brightest stars when there are
            more than max_stars. None by default, in which case the stars are picked at random.

    Return:
        (ra_mid, dec_mid, rot_eq_standard, scale, fov_w, fov_h, star_data, solution_info): [tuple] None if
            no solution was found. Otherwise:
            ra_mid: [float] Right ascension of the image center in degrees.
            dec_mid: [float] Declination of the image center in degrees.
            rot_eq_standard: [float] Equatorial orientation in degrees.
            scale: [float] Scale in px/deg.
            fov_w: [float] Width of the FOV in degrees.
            fov_h: [float] Height of the FOV in degrees.
            star_data: [list] A list of star data, where star_data = [x_data, y_data].
            solution_info: [dict] Additional solution details - the log odds, the quad stars used for
                the geometric match, the index file path, the astropy WCS object and the number of
                input stars.
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



    # For very wide FOV images (>90 deg), filter to the central region FIRST before selecting the brightest
    #   stars. This avoids distortion issues at the edges of fisheye lenses. Use fov_w_hint (best guess
    #   from the config) if available, otherwise use the range midpoint
    estimated_fov = fov_w_hint if fov_w_hint is not None else (
        (fov_w_range[0] + fov_w_range[1]) / 2.0 if fov_w_range is not None else None
    )

    if verbose:
        print("FOV hint: fov_w_hint={}, fov_w_range={}, estimated_fov={}".format(
            fov_w_hint, fov_w_range, estimated_fov))

    # Get the full image dimensions (before any central star filtering). The given FOV range is for the
    #   full image width, so the pixel scale hint must divide it by the full image width in pixels
    if img is not None:
        img_width = img.shape[1]
        img_height = img.shape[0]
    elif (x_center is not None) and (y_center is not None):
        img_width = 2*x_center
        img_height = 2*y_center
    else:
        img_width = np.max(x_data)
        img_height = np.max(y_data) if y_data is not None else img_width*0.75

    # Keep the full image FOV range for the pixel scale hint, the central filtering below only changes the
    #   range used to select the index quad scales
    fov_w_range_full = fov_w_range

    if estimated_fov is not None and estimated_fov > 90:

        # Determine the image center from the image, the given center or as a last resort the stars
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

        # Use the central 60 degrees for very wide FOV cameras
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

        # Only filter if there are enough central stars to solve on
        if central_star_count >= 20:
            original_count = len(x_data)
            x_data = x_data[central_mask]
            y_data = y_data[central_mask]
            if input_intensities is not None:
                input_intensities = np.array(input_intensities)[central_mask]

            # Update the FOV range to reflect the filtered central region
            filtered_fov = estimated_fov * central_fov_fraction * 2
            fov_w_range = [filtered_fov * 0.75, filtered_fov * 1.5]
            print("  -> Filtering to central {:.1f} deg, Stars: {:d} -> {:d}".format(
                filtered_fov, original_count, len(x_data)))
            print("  -> Updated FOV range for scale selection: {:.1f} - {:.1f} deg".format(
                fov_w_range[0], fov_w_range[1]))
        else:
            print("  -> NOT filtering (need >= 20 central stars)")

    # Select the brightest stars if there are too many
    if len(x_data) > max_stars:

        if verbose:
            print("Too many stars found: ", len(x_data))

        # Select the brightest stars (highest intensity)
        if input_intensities is not None and len(input_intensities) == len(x_data):
            if verbose:
                print("Selecting {:d} brightest stars...".format(max_stars))
            bright_indices = np.argsort(-np.array(input_intensities))[:max_stars]
            x_data = x_data[bright_indices]
            y_data = y_data[bright_indices]
            input_intensities = np.array(input_intensities)[bright_indices]
        # Fall back to a random selection if no intensities are available
        else:
            if verbose:
                print("Randomly selecting {:d} stars...".format(max_stars))

            # Use a local seeded generator so repeated solves of the same input pick the same stars
            rng = np.random.RandomState(0)
            rand_indices = rng.choice(len(x_data), max_stars, replace=False)
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

    # Default scales for a ~45 deg FOV height (covers 4.5 to 45 deg quads)
    scales = {14, 15, 16, 17, 18, 19}

    size_hint = None

    if fov_w_range is not None:

        # Compute the aspect ratio from the full image dimensions
        aspect_ratio = img_height / img_width

        # Use the average FOV estimate for the width
        avg_fov_w = (fov_w_range[0] + fov_w_range[1]) / 2.0

        # Compute the FOV height (shorter dimension for landscape images)
        fov_h = avg_fov_w * aspect_ratio

        # Use the "10% to 100% of image size" rule on the short side - quad sizes should range from 10% to
        #   100% of the shorter FOV dimension
        min_quad_size_deg = fov_h * 0.10
        max_quad_size_deg = fov_h

        if verbose:
            print("FOV range: {:.2f} - {:.2f} deg (width)".format(fov_w_range[0], fov_w_range[1]))
            print("FOV height estimate: {:.2f} deg".format(fov_h))
            print("Quad size range: {:.2f} - {:.2f} deg".format(min_quad_size_deg, max_quad_size_deg))

        # Index 4100 series quad diameter ranges:
        #   Scale 14: 240-340 arcmin    (4.0-5.7 deg)
        #   Scale 15: 340-480 arcmin    (5.7-8.0 deg)
        #   Scale 16: 480-680 arcmin    (8.0-11.3 deg)
        #   Scale 17: 680-1000 arcmin   (11.3-16.7 deg)
        #   Scale 18: 1000-1400 arcmin  (16.7-23.3 deg)
        #   Scale 19: 1400-2000 arcmin  (23.3-33.3 deg)

        # Quad diameter boundaries for the 4100 series scales
        #   Formula: quad diameter = 240/sqrt(2)^7 * sqrt(2)^N arcmin
        #   This gives scale 14 = 240-340 arcmin, matching the astrometry.net docs
        base_arcmin = 240.0 / math.pow(math.sqrt(2), 7)  # ~21.2 arcmin
        scale_quad_ranges_deg = []
        for scale_num in range(7, 20):
            lower_arcmin = base_arcmin * math.pow(math.sqrt(2), scale_num - 7)
            upper_arcmin = base_arcmin * math.pow(math.sqrt(2), scale_num - 6)
            lower_deg = lower_arcmin / 60.0
            upper_deg = upper_arcmin / 60.0
            scale_quad_ranges_deg.append((scale_num, lower_deg, upper_deg))

        # Find the scales whose quad ranges overlap with the desired range. Add a 10% margin on the lower
        #   bound to avoid boundary precision issues
        min_quad_with_margin = min_quad_size_deg * 0.9
        matching_scales = []

        # The target quad size is ~30% of the FOV height (empirically good for matching)
        target_quad_deg = fov_h * 0.30

        for scale_num, lower_deg, upper_deg in scale_quad_ranges_deg:

            # Check if this scale's quad range overlaps with the desired quad range
            if min_quad_with_margin <= upper_deg and max_quad_size_deg >= lower_deg:

                # Calculate how close this scale's midpoint is to the target
                mid_deg = (lower_deg + upper_deg) / 2.0
                distance = abs(mid_deg - target_quad_deg)
                matching_scales.append((scale_num, distance))

        # Sort by the distance to the target (closest first), then convert to a set for the astrometry lib
        matching_scales.sort(key=lambda x: x[1])
        scales = {s[0] for s in matching_scales}

        # If no scales were found, fall back to the defaults for a ~45 deg FOV height
        if not scales:
            scales = {14, 15, 16, 17, 18, 19}

        # Show the scales in the preferred order (closest to the target quad size first)
        if verbose:
            preferred_order = [s[0] for s in matching_scales]
            print("Using index scales (preferred order): {}".format(preferred_order))
            print("Target quad size: {:.1f} deg (30% of FOV height)".format(target_quad_deg))
            for s, dist in matching_scales:
                lower_arcmin = base_arcmin * math.pow(math.sqrt(2), s - 7)
                upper_arcmin = base_arcmin * math.pow(math.sqrt(2), s - 6)
                mid_deg = (lower_arcmin + upper_arcmin) / 2.0 / 60.0
                print("  Scale {}: {:.0f}-{:.0f} arcmin ({:.1f}-{:.1f} deg), mid={:.1f} deg".format(
                    s, lower_arcmin, upper_arcmin, lower_arcmin/60, upper_arcmin/60, mid_deg))

        # Compute the pixel scale for the size hint (helps the solver converge faster). Use the full image
        #   FOV range and width, as filtering the stars to the centre does not change the pixel scale
        lower_arcsec_per_pixel = fov_w_range_full[0] * 3600 / img_width
        upper_arcsec_per_pixel = fov_w_range_full[1] * 3600 / img_width

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
            # Return the first solution if the log odds ratio is greater than 60 (good solution). A lower
            #   threshold means faster solving - logodds > 50 is typically reliable
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

        # Print the WCS fields
        if verbose:
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
        rot_eq_standard = rotationEqStandard(ra_mid, dec_mid, ra_right, dec_right)


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

        # Note: match.stars contains ALL catalog stars in the FOV region from the index, NOT just the ones
        #   that matched to input stars. They are not used - RMS has its own better star catalog for
        #   matching

        # Extract the quad stars (the specific catalog stars used for the initial geometric matching)
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

        # Additional solution info - no matched pairs from astrometry.net, the star matching will be done
        #   in SkyFit2 using RMS's own catalog
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
        local installation of astrometry.net, falling back to the remote astrometry.net compatible
        servers if the local solver is unavailable or fails.

    Keyword arguments:
        ff_file_path: [str] Path to the FF file to load.
        img: [ndarray] Numpy array containing image data.
        mask: [ndarray] Mask image. None by default.
        x_data: [list] A list of star x image coordinates.
        y_data: [list] A list of star y image coordinates
        fov_w_range: [2 element tuple] A tuple of scale_lower and scale_upper, i.e. the estimate of the
            width of the FOV in degrees.
        fov_w_hint: [float] Best guess of the FOV width in degrees (e.g. from the config). None by default,
            in which case the midpoint of fov_w_range is used to decide whether the image is a very wide
            field that should be solved on its central part only.
        max_stars: [int] Maximum number of stars to use for the astrometry.net solution. Default is 100.
        verbose: [bool] Print verbose output. Default is False.
        x_center: [float] X coordinate of the image center. Default is None.
        y_center: [float] Y coordinate of the image center. Default is None.
        lat: [float] Station latitude in degrees. Accepted for API compatibility, currently unused.
        lon: [float] Station longitude in degrees. Accepted for API compatibility, currently unused.
        jd: [float] Julian date. Accepted for API compatibility, currently unused.
        input_intensities: [ndarray] Star intensities, used to keep the brightest stars when there are
            more than max_stars. None by default, in which case the stars are picked at random.

    Return:
        [tuple] The same 8-tuple as astrometryNetSolveLocal, or None if no solver found a solution. The
            remote solvers fill solution_info with whatever the server provides.
    """

    def _tryRemoteSolve(api_url, ff_path, image, x_coords, y_coords, fov_range, x_cen, y_cen):
        """ Try a coordinate-only solve on the given server first, and fall back to uploading the image
            if that fails.

        Arguments:
            api_url: [str] Base URL of the astrometry.net compatible API.
            ff_path: [str] Path to the FF file, or None.
            image: [ndarray] Image data, or None.
            x_coords: [list] Star x image coordinates, or None.
            y_coords: [list] Star y image coordinates, or None.
            fov_range: [2 element tuple] Estimate of the FOV width range in degrees.
            x_cen: [float] X coordinate of the image center.
            y_cen: [float] Y coordinate of the image center.

        Return:
            [tuple] The solution tuple from novaAstrometryNetSolve, or None if nothing solved.
        """

        # If there are coordinates, try a coordinate-only solve first (faster, less bandwidth)
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

            # Fall back to the image if available
            if image is not None or ff_path is not None:
                print("  Falling back to image upload...")
                return novaAstrometryNetSolve(
                    ff_file_path=ff_path, img=image, x_data=None, y_data=None,
                    fov_w_range=fov_range, x_center=x_cen, y_center=y_cen,
                    api_url=api_url
                )
            return None

        # No coordinates, just try with the image
        return novaAstrometryNetSolve(
            ff_file_path=ff_path, img=image, x_data=x_coords, y_data=y_coords,
            fov_w_range=fov_range, x_center=x_cen, y_center=y_cen,
            api_url=api_url
        )

    # If the local installation of astrometry.net is not available, use the remote API
    if not ASTROMETRY_NET_AVAILABLE:

        # Try the primary server (contrailcast) first
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

        # If the local solver fails, try the remote APIs
        except Exception as e:

            print("Local astrometry.net solver failed with error:")
            print(e)

            # Try the primary server (contrailcast) first
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

        # Print the matched star info if available
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
