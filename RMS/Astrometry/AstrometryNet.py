
from __future__ import print_function, division, absolute_import

import os
import inspect

import numpy as np
from PIL import Image
from astropy.wcs import WCS

from RMS.ExtractStars import extractStarsAuto
from RMS.Formats.FFfile import read as readFF
from RMS.Astrometry.AstrometryNetNova import novaAstrometryNetSolve
from RMS.Logger import getLogger

try:
    import astrometry
    ASTROMETRY_NET_AVAILABLE = True

except ImportError:
    ASTROMETRY_NET_AVAILABLE = False



def astrometryNetSolveLocal(ff_file_path=None, img=None, mask=None, x_data=None, y_data=None, 
                            fov_w_range=None, max_stars=100, verbose=False, x_center=None, y_center=None):
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
            


    # If there are too many stars (more than 100), randomly select them to reduce the number
    if len(x_data) > max_stars:
        
        if verbose:
            print("Too many stars found: ", len(x_data))
            print("Randomly selecting {:d} stars...".format(max_stars))

        # Randomly select max_stars stars
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

        return ra_mid, dec_mid, rot_eq_standard, scale, fov_w, fov_h, star_data
    

    else:
        print("No solution found.")
        return None


def astrometryNetSolve(ff_file_path=None, img=None, mask=None, x_data=None, y_data=None, fov_w_range=None,
                       max_stars=100, verbose=False, x_center=None, y_center=None):
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
                x_center=x_center, y_center=y_center
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

        ra_mid, dec_mid, rot_eq_standard, scale, fov_w, fov_h = status

        print("Astrometry.net solution:")
        print()
        print("RA  = {:.2f} deg".format(ra_mid))
        print("Dec = {:+.2f} deg".format(dec_mid))
        print("Scale = {:.2f} arcmin/pixel".format(scale))
        print("Rot. eq. standard = {:.2f} deg".format(rot_eq_standard))
        print("FOV = {:.2f} x {:.2f} deg".format(fov_w, fov_h))

    else:
        print("No solution found.")