""" Make a KML file outlining the field of view of the camera. """


from __future__ import print_function, division, absolute_import, unicode_literals

import os
import sys

import numpy as np

import RMS.ConfigReader as cr
from RMS.Formats.Platepar import Platepar
from RMS.Routines.MaskImage import loadMask
from RMS.Routines.FOVArea import fovArea


def fovKML(dir_path, platepar, mask=None, area_ht=100000, side_points=10, plot_station=True):
    """ Given a platepar file and a mask file, make a KML file outlining the camera FOV.

    Arguments:
        dir_path: [str] Path where the KML file will be saved.
        platepar: [Platepar object]

    Keyword arguments:
        mask: [Mask object] Mask object, None by default.
        area_ht: [float] Height in meters of the computed area.
        side_points: [int] How many points to use to evaluate the FOV on seach side of the image. Normalized
            to the longest side.
        plot_station: [bool] Plot the location of the station. True by default.

    Return:
        kml_path: [str] Path to the saved KML file.

    """

    # Find lat/lon/elev describing the view area
    polygon_sides = fovArea(platepar, mask, area_ht, side_points)

    # Make longitued in the same wrap region
    lon_list = []
    for side in polygon_sides:
        side = np.array(side)
        lat, lon, elev = side.T
        lon_list += lon.tolist()
    
    # Unwrap longitudes
    lon_list = np.degrees(np.unwrap(2*np.radians(lon_list))/2)

    # Assign longitudes back to the proper sides
    prev_len = 0
    polygon_sides_lon_wrapped = []
    for side in polygon_sides:
        side = np.array(side)
        lat, _, elev = side.T

        # Extract the wrapped longitude
        lon = lon_list[prev_len:prev_len + len(lat)]

        side = np.c_[lat, lon, elev]
        polygon_sides_lon_wrapped.append(side.tolist())

        prev_len += len(lat)

    polygon_sides = polygon_sides_lon_wrapped


    # List of polygons to plot
    polygon_list = []


    # Join points from all sides to create the collection area at the given height
    area_vertices = []
    for side in polygon_sides:
        for side_p in side:
            area_vertices.append(side_p)

    polygon_list.append(area_vertices)


        
    # If the station is plotted, connect every side to the station
    if plot_station:
        
        for side in polygon_sides:

            side_vertices = []

            # Add coordinates of the station (first point)
            side_vertices.append([platepar.lat, platepar.lon, platepar.elev])

            for side_p in side:
                side_vertices.append(side_p)        

            # Add coordinates of the station (last point)
            side_vertices.append([platepar.lat, platepar.lon, platepar.elev])

            polygon_list.append(list(side_vertices))



    ### MAKE A KML ###

    kml = "<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://earth.google.com/kml/2.1'><Folder><name>{:s}</name><open>1</open><Placemark id='{:s}'>".format(platepar.station_code, platepar.station_code) \
        + """
                 <Style id='camera'>
                  <LineStyle>
                <width>1.5</width>
                  </LineStyle>
                  <PolyStyle>
                <color>40000800</color>
                  </PolyStyle>
                </Style>
                <styleUrl>#camera</styleUrl>\n""" \
        + "<name>{:s}</name>\n".format(platepar.station_code) \
        + "                <description>Area height: {:d} km\n".format(int(area_ht/1000))

    # Only add station info if the station is plotted
    if plot_station:
        kml += "Longitude: {:10.6f} deg\n".format(platepar.lat) \
             + "Latitude:  {:11.6f} deg\n".format(platepar.lon) \
             + "Altitude: {:.2f} m\n".format(platepar.elev) \

    kml += """
    </description>
    
    <MultiGeometry>"""


    ### Plot all polygons ###
    for polygon_points in polygon_list:
        kml += \
"""    <Polygon>
        <extrude>0</extrude>
        <altitudeMode>absolute</altitudeMode>
        <outerBoundaryIs>
            <LinearRing>
                <coordinates>\n"""

        # Add the polygon points to the KML
        for p_lat, p_lon, p_elev in polygon_points:
            kml += "                    {:.6f},{:.6f},{:.0f}\n".format(p_lon, p_lat, p_elev)

        kml += \
"""                </coordinates>
            </LinearRing>
        </outerBoundaryIs>
    </Polygon>"""
    ### ###


    kml += \
"""    </MultiGeometry>
    </Placemark>
    </Folder>
    </kml> """
    ###


    # Save the KML file to the directory with the platepar
    kml_path = os.path.join(dir_path, "{:s}-{:d}km.kml".format(platepar.station_code, int(area_ht/1000)))
    with open(kml_path, 'w') as f:
        f.write(kml)

    print("KML saved to:", kml_path)


    return kml_path



if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Compute the FOV area given the platepar and mask files. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('input', metavar='DIR_OR_PP', type=str, \
                    help="Path to the directory where the platepar is, or path to the platepar file.")

    arg_parser.add_argument('mask', metavar='MASK', type=str, nargs='?',\
                    help="Path to the mask file.")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one. To load the .config file in the given data directory, write '.' (dot).")

    arg_parser.add_argument('-e', '--elev', metavar='ELEVATION', type=float, \
        help="Height of area polygon (km). 100 km by default.", default=100)

    arg_parser.add_argument('-p', '--pts', metavar='SIDE_POINT', type=int, \
        help="Number of points to evaluate on the longest side. 10 by default.", default=10)

    arg_parser.add_argument('-s', '--station', action="store_true", \
        help="""Plot the location of the station.""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # If the path to the platepar is given
    if os.path.isfile(cml_args.input):
        platepar_path = cml_args.input
        dir_path = os.path.dirname(platepar_path)

    # If the path to the dirctory is given
    else:
        dir_path = cml_args.input


    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)
    

    # If the path to the directory is given, automatically locate the platepar and mask
    if not os.path.isfile(cml_args.input):

        platepar_path = None
        mask_path = None

        for file_name in os.listdir(dir_path):
            
            # Locate platepar
            if file_name == config.platepar_name:
                platepar_path = os.path.join(dir_path, file_name)

            # Locate mask
            if file_name == config.mask_file:
                mask_path = os.path.join(dir_path, file_name)

    
        if platepar_path is None:
            print("No platepar find was found in {:s}!".format(dir_path))
            sys.exit()

        else:
            print("Found platepar!")



    # Load the platepar file
    pp = Platepar()
    pp.read(platepar_path)


    # Assign mask
    mask_path = None
    if cml_args.mask is not None:
        mask_path = cml_args.mask

    # Load the mask file
    if mask_path is not None:
        mask = loadMask(mask_path)
        print("Loading mask:", mask_path)
    else:
        mask = None


    # Generate a KML file from the platepar
    fovKML(dir_path, pp, mask, area_ht=1000*cml_args.elev, side_points=cml_args.pts, \
        plot_station=cml_args.station)