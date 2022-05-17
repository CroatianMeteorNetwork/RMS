""" Functions for computing the FOV area at a given height. """

from __future__ import print_function, division, absolute_import, unicode_literals


import numpy as np

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDec2AltAz
from RMS.Astrometry.Conversions import jd2Date, J2000_JD, AEH2LatLonAlt
from RMS.Formats.Platepar import Platepar
from RMS.Routines.MaskImage import loadMask, MaskStructure



def xyHt2Geo(platepar, x, y, area_ht, indicate_limit=False, elev_limit=5):
    """ Given pixel coordiantes on the image and a height above sea level, compute geo coordiantes of the
        point. The elevation is limited to 5 deg above horizon.

    Arguments:
        platepar: [Platepar object]
        x: [float] Image X coordinate.
        y: [float] Image Y coordiante.
        area_ht: [float] Height above sea level (meters).

    Keyword arguments:
        indicate_limit: [bool] Indicate that the elevation was below the limit of 5 deg by setting the
            height to -1. False by default.
        elev_limit: [float] Limit of elevation above horizon (deg). 5 degrees by default.

    
    Return:
        (r, lat, lon, ht): [tuple of floats] range in meters, latitude and longitude in degrees, \
            WGS84 height in meters

    """

    
    # Compute RA/Dec in J2000 of the image point, at J2000 epoch time so we don't have to precess
    _, ra, dec, _ = xyToRaDecPP([jd2Date(J2000_JD.days)], [x], [y], [1], platepar, \
        extinction_correction=False)

    # Compute alt/az of the point
    azim, elev = raDec2AltAz(ra[0], dec[0], J2000_JD.days, platepar.lat, platepar.lon)

    # Limit the elevation to elev_limit degrees above the horizon
    limit_hit = False
    if elev < elev_limit:
        elev = elev_limit
        limit_hit = True

    # Compute the geo location of the point along the line of sight
    p_r, p_lat, p_lon, p_ht = AEH2LatLonAlt(azim, elev, area_ht, platepar.lat, platepar.lon, \
        platepar.elev)


    # If the elevation limit was hit, and the indicate flag is True, set the elevation to -1
    if indicate_limit and limit_hit:
        p_ht = -1

    return p_r, p_lat, p_lon, p_ht



def fovArea(platepar, mask=None, area_ht=100000, side_points=10, elev_limit=5):
    """ Given a platepar file and a mask file, compute geo points of the FOV edges at the given height.

    Arguments:
        platepar: [Platepar object]

    Keyword arguments:
        mask: [Mask object] Mask object, None by default.
        area_ht: [float] Height in meters of the computed area.
        side_points: [int] How many points to use to evaluate the FOV on seach side of the image. Normalized
            to the longest side.
        elev_limit: [float] Limit of elevation above horizon (deg). 5 degrees by default.

    Return:
        [list] A list points for every side of the image, and every side is a list of (lat, lon, elev) 
            describing the sides of the FOV area. Values are in degrees and meters.

    """

    # If the mask has wrong dimensions, disregard it
    if mask is not None:
        if (mask.img.shape[0] != platepar.Y_res) or (mask.img.shape[1] != platepar.X_res):
            print("The mask has the wrong shape, so it will be ignored!")
            print("    Mask     = {:d}x{:d}".format(mask.img.shape[1], mask.img.shape[0]))
            print("    Platepar = {:d}x{:d}".format(platepar.X_res, platepar.Y_res))
            mask = None

    # If the mask is not given, make a dummy mask with all white pixels
    if mask is None:
        mask = MaskStructure(255 + np.zeros((platepar.Y_res, platepar.X_res), dtype=np.uint8))


    # Compute the number of points for the sizes
    longer_side_points = side_points
    shorter_side_points = int(np.ceil(side_points*platepar.Y_res/platepar.X_res))

    # Define operations for each side (number of points, axis of sampling, sampling start, direction of sampling, reverse sampling direction)
    side_operations = [
        [shorter_side_points, 'y', 0                 ,  1, False], # left
        [longer_side_points,  'x', platepar.Y_res - 1, -1, False], # bottom
        [shorter_side_points, 'y', platepar.X_res - 1, -1, True],  # right
        [longer_side_points,  'x', 0                 ,  1, True]]  # up
    


    # Sample the points on image borders
    side_points_list = []
    for n_sample, axis, c0, sampling_direction, reverse_sampling in side_operations:

        # Reverse some ordering to make the sampling counter-clockwise, starting in the top-left corner
        sampling_offsets = range(n_sample + 1)
        if reverse_sampling:
            sampling_offsets = reversed(sampling_offsets)

        # Sample points on every side
        side_points = []
        for i_sample in sampling_offsets:

            # Compute x, y coordinate of the sampled pixel
            if axis == 'x':
                axis_side = platepar.X_res
                other_axis_side = platepar.Y_res
                x0 = int((i_sample/n_sample)*(axis_side - 1))
                y0 = c0
            else:
                axis_side = platepar.Y_res
                other_axis_side = platepar.X_res
                x0 = c0
                y0 = int((i_sample/n_sample)*(axis_side - 1))


            # Find a pixel position along the axis that is not masked using increments of 10 pixels
            unmasked_point_found = False

            # Make a list of points to sample
            for mask_offset in np.arange(0, other_axis_side, 10):

                # Compute the current pixel position
                if axis == 'x':
                    x = x0
                    y = y0 + sampling_direction*mask_offset
                else:
                    x = x0 + sampling_direction*mask_offset
                    y = y0


                # If the position is not masked, stop searching for unmasked point
                if mask.img[y, x] > 0:
                    unmasked_point_found = True
                    break


            # Find azimuth and altitude at the given pixel, if a found unmask pixel was found along this
            #   line
            if unmasked_point_found:

                # Compute the geo location of the point along the line of sight
                _, p_lat, p_lon, p_elev = xyHt2Geo(platepar, x, y, area_ht, elev_limit=elev_limit)


                side_points.append([x, y, p_lat, p_lon, p_elev])
                

        # Add points from every side to the list (store a copy)
        side_points_list.append(list(side_points))


    # Postprocess the point list by removing points which intersect points on the previous side
    side_points_list_filtered = []
    for i, (n_sample, axis, c0, sampling_direction, reverse_sampling) in enumerate(side_operations):

        # Get the current and previous points list
        side_points = side_points_list[i]
        side_points_prev = side_points_list[i - 1]

        # Remove all points from the list that intersect points on the previous side
        side_points_filtered = []
        for x, y, p_lat, p_lon, p_elev in side_points:

            # Check all points from the previous side
            skip_point = False
            for entry_prev in side_points_prev:
                
                x_prev, y_prev = entry_prev[:2]

                # # Skip duplicates
                # if (x == x_prev) and (y == y_prev):
                #     skip_point = True
                #     break

                if axis == 'x':

                    if reverse_sampling:
                        if (y_prev < y) and (x_prev < x):
                            skip_point = True
                            break
                    else:
                        if (y_prev > y) and (x_prev > x):
                            skip_point = True
                            break

                else:
                    if reverse_sampling:
                        if (y_prev < y) and (x_prev > x):
                            skip_point = True
                            break
                    else:
                        if (y_prev > y) and (x_prev < x):
                            skip_point = True
                            break


            # If the point should not be skipped, add it to the final list
            if not skip_point:
                side_points_filtered.append([p_lat, p_lon, p_elev])

            #     print("ADDING   = {:4d}, {:4d}, {:10.6f}, {:11.6f}, {:.2f}".format(int(x), int(y), p_lat, p_lon, p_elev))

            # else:
            #     print("SKIPPING = {:4d}, {:4d}, {:10.6f}, {:11.6f}, {:.2f}".format(int(x), int(y), p_lat, p_lon, p_elev))


        side_points_list_filtered.append(side_points_filtered)


    return side_points_list_filtered




if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Compute the FOV area given the platepar and mask files. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('platepar', metavar='PLATEPAR', type=str, \
                    help="Path to the platepar file.")

    arg_parser.add_argument('mask', metavar='MASK', type=str, nargs='?', \
                    help="Path to the mask file.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################
    

    # Load the platepar file
    pp = Platepar()
    pp.read(cml_args.platepar)


    # Load the mask file
    if cml_args.mask is not None:
        mask = loadMask(cml_args.mask)
    else:
        mask = None


    # Compute the FOV geo points
    area_list = fovArea(pp, mask)

    for side_points in area_list:
        print(side_points)