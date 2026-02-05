""" Functions for computing the FOV area from a platepar file. """

from __future__ import print_function, division, absolute_import, unicode_literals

import argparse
import numpy as np

# Assuming these modules are in the python path
# You might need to adjust the import paths based on your project structure
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.Conversions import jd2Date, J2000_JD
from RMS.Formats.Platepar import Platepar
from RMS.Math import sphericalPolygonArea
from RMS.Routines.MaskImage import loadMask, MaskStructure




def fovSkyArea(platepar, mask=None, side_points=20):
    """ Given a platepar file, compute the solid angle of the FOV.

    Arguments:
        platepar: [Platepar object]

    Keyword arguments:
        mask: [Mask object] Mask object, None by default. If given, the area
            of the unmasked FOV will be computed.
        side_points: [int] How many points to use to evaluate the FOV on each 
            side of the image.

    Return:
        [float] FOV area in square degrees.

    """

    # If the mask has wrong dimensions, disregard it
    if mask is not None:
        if (mask.img.shape[0] != platepar.Y_res) or (mask.img.shape[1] != platepar.X_res):
            print("The mask has the wrong shape, so it will be ignored!")
            print("     Mask     = {:d}x{:d}".format(mask.img.shape[1], mask.img.shape[0]))
            print("     Platepar = {:d}x{:d}".format(platepar.X_res, platepar.Y_res))
            mask = None

    # If the mask is not given, make a dummy mask with all white pixels
    if mask is None:
        mask = MaskStructure(255 + np.zeros((platepar.Y_res, platepar.X_res), dtype=np.uint8))


    # Get image dimensions
    width = platepar.X_res
    height = platepar.Y_res

    # Compute the number of points for the sides, normalized to the longest side
    if width >= height:
        longer_side_points = side_points
        shorter_side_points = int(np.ceil(side_points*height/width))
    else:
        shorter_side_points = side_points
        longer_side_points = int(np.ceil(side_points*width/height))

    # Generate border points in pixel coordinates
    border_pixels = []

    # Top edge (left to right)
    border_pixels.extend(zip(np.linspace(0, width - 1, longer_side_points, endpoint=False), 
                             np.full(longer_side_points, 0))
                             )
    # Right edge (top to bottom)
    border_pixels.extend(zip(np.full(shorter_side_points, width - 1), 
                             np.linspace(0, height - 1, shorter_side_points, endpoint=False))
                             )
    # Bottom edge (right to left)
    border_pixels.extend(zip(np.linspace(width - 1, 0, longer_side_points, endpoint=False), 
                             np.full(longer_side_points, height - 1))
                             )
    # Left edge (bottom to top)
    border_pixels.extend(zip(np.zeros(shorter_side_points), 
                             np.linspace(height - 1, 0, shorter_side_points, endpoint=False))
                             )

    x_coords = []
    y_coords = []

    # For every point on the border, find the first unmasked pixel by searching inwards
    for x0, y0 in border_pixels:
        x0, y0 = int(x0), int(y0)

        # Determine search direction (inwards from the edge)
        if x0 == 0: dx = 1
        elif x0 == width - 1: dx = -1
        else: dx = 0

        if y0 == 0: dy = 1
        elif y0 == height - 1: dy = -1
        else: dy = 0
        
        # Refine search direction for corners
        if x0 == 0 and y0 == 0: dx, dy = 1, 1
        elif x0 == width -1 and y0 == 0: dx, dy = -1, 1
        elif x0 == width -1 and y0 == height -1: dx, dy = -1, -1
        elif x0 == 0 and y0 == height -1: dx, dy = 1, -1


        # Search for the first unmasked pixel
        x, y = x0, y0
        unmasked_point_found = False
        # Limit search to avoid excessive loops
        for _ in range(max(width, height)):
            if not (0 <= x < width and 0 <= y < height):
                break
            
            if mask.img[y, x] > 0:
                x_coords.append(x)
                y_coords.append(y)
                unmasked_point_found = True
                break
            
            # Move to the next pixel inwards
            x += dx
            y += dy

        # If no unmasked pixel was found along the search path, use the border pixel
        if not unmasked_point_found:
            x_coords.append(x0)
            y_coords.append(y0)


    # Compute RA/Dec in J2000 for all border points at once
    jd_j2000 = jd2Date(J2000_JD.days)
    magnitudes = [1]*len(x_coords)

    _, ra_list, dec_list, _ = xyToRaDecPP([jd_j2000]*len(x_coords), x_coords, y_coords, magnitudes, platepar,
        extinction_correction=False)

    # Unwrap RA coordinates to handle polygons crossing the 0h/24h line
    ra_unwrapped = np.unwrap(ra_list, period=360)
    
    # Combine RA and Dec into a list of points
    fov_points = np.array(list(zip(ra_unwrapped, dec_list)))

    # Compute the area of the spherical polygon defined by the FOV points
    area = sphericalPolygonArea(fov_points)

    return area


if __name__ == "__main__":

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Compute the FOV area in square degrees from a platepar file.""")

    arg_parser.add_argument('platepar', metavar='PLATEPAR_PATH', type=str,
        help="Path to the platepar file.")
    
    arg_parser.add_argument('mask', metavar='MASK_PATH', type=str, nargs='?', default=None,
        help="Path to the optional mask file.")

    arg_parser.add_argument('-s', '--side_points', type=int, default=40,
        help="Number of points to sample on each side of the FOV. Default is 40.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # Load the platepar file
    pp = Platepar()
    pp.read(cml_args.platepar)

    # Load the mask file if provided
    mask = None
    if cml_args.mask is not None:
        mask = loadMask(cml_args.mask)
        print("Mask file: {:s}".format(cml_args.mask))


    # Compute the FOV area
    fov_area_sq_deg = fovSkyArea(pp, mask=mask, side_points=cml_args.side_points)

    print("Platepar file: {:s}".format(cml_args.platepar))
    print("Resolution: {:d} x {:d}".format(pp.X_res, pp.Y_res))
    print("FOV area: {:.2f} square degrees".format(fov_area_sq_deg))

