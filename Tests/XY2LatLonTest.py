import logging
logging.getLogger('matplotlib.font_manager').disabled = True

import json 

import numpy as np
import matplotlib.pyplot as plt

from RMS.Astrometry.ApplyAstrometry import xyHt2Geo, fovEdgePolygon, geoHt2XYInsideFOV
from RMS.Astrometry.Conversions import J2000_JD
from RMS.Formats.Platepar import Platepar




if __name__ == "__main__":

    pp_json = """
{
    "F_scale": 15.701176160218697,
    "Ho": 313.86851592408493,
    "JD": 2460740.915217384,
    "RA_d": 109.18334207962252,
    "UT_corr": 0,
    "X_res": 1280,
    "Y_res": 720,
    "alt_centre": 37.96222492347106,
    "asymmetry_corr": true,
    "auto_check_fit_refined": false,
    "auto_recalibrated": false,
    "az_centre": 175.86983442281368,
    "dec_d": -74.08689102624128,
    "distortion_type": "radial7-odd",
    "distortion_type_list": [
        "poly3+radial",
        "poly3+radial3",
        "poly3+radial5",
        "radial3-all",
        "radial4-all",
        "radial5-all",
        "radial3-odd",
        "radial5-odd",
        "radial7-odd",
        "radial9-odd"
    ],
    "distortion_type_poly_length": [
        12,
        13,
        14,
        7,
        8,
        9,
        6,
        7,
        8,
        9
    ],
    "elev": 271.0,
    "equal_aspect": true,
    "extinction_scale": 0.6,
    "force_distortion_centre": false,
    "fov_h": 88.65210403644974,
    "fov_v": 47.014859261006805,
    "gamma": 1.0,
    "lat": -22.47,
    "lon": 143.18,
    "mag_0": -2.5,
    "mag_lev": 10.009485462304724,
    "mag_lev_stddev": 0.2002279247131813,
    "measurement_apparent_to_true_refraction": false,
    "poly_length": 7,
    "pos_angle_ref": 343.0805344830111,
    "refraction": true,
    "rotation_from_horiz": -1.4937393230898501,
    "star_list": [
    ],
    "station_code": "AU004P",
    "version": 2,
    "vignetting_coeff": 0.0006141654982057213,
    "vignetting_fixed": false,
    "x_poly": [
        0.032439499444704714,
        0.030238153496385207,
        0.01114345505377356,
        -0.2604269175454886,
        0.0752264534197823,
        -0.0006981065281311777,
        0.012335178584800453
    ],
    "x_poly_fwd": [
        0.032439499444704714,
        0.030238153496385207,
        0.01114345505377356,
        -0.2604269175454886,
        0.0752264534197823,
        -0.0006981065281311777,
        0.012335178584800453
    ],
    "x_poly_rev": [
        0.03244007706072434,
        0.03027620597179506,
        0.010881721890645384,
        -0.25797257944584695,
        0.07273219719598978,
        -0.00720498334132414,
        0.0023641562141605544
    ],
    "y_poly": [
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ],
    "y_poly_fwd": [
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ],
    "y_poly_rev": [
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ]
}"""

    # Load a platepar file
    pp = Platepar()
    pp.loadFromDict(json.loads(pp_json))

    # Reference height
    ht = 10000 # m

    # Get the edges of the FOV
    x_vert, y_vert, ra_vert, dec_vert = fovEdgePolygon(pp, J2000_JD.days)

    # Plot the FOV in Xy (left) and RaDec (right) coordinates
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x_vert, y_vert, 'k-')
    plt.scatter(x_vert, y_vert, c=[i for i in range(len(x_vert))])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()
    plt.gca().invert_yaxis()

    # Make the RA dec plot a polar plot
    plt.subplot(122, polar=True)
    plt.plot(np.radians(ra_vert), dec_vert, 'k-')
    plt.scatter(np.radians(ra_vert), dec_vert, c=[i for i in range(len(ra_vert))])
    plt.xlabel("RA")
    
    plt.show()


    # Generate samples along the diagonal of the image
    x_test = np.linspace(5, pp.X_res - 5, 10)
    y_test = np.linspace(5, pp.Y_res - 5, 10)

    print()

    # Compare the input and output X and Y points
    for i in range(len(x_test)):
        
        x_i = x_test[i]
        y_i = y_test[i]

        # Map the input X, Y to lat, lon
        lat, lon = xyHt2Geo(pp, x_i, y_i, ht)

        # Map back to X and Y
        x_o, y_o, _ = geoHt2XYInsideFOV(pp, lat, lon, ht)

        print("Test point {}:".format(i))
        print("  IN  X: {:.2f}, Y: {:.2f}".format(x_i, y_i))
        print("  Latitude: {:.5f}, Longitude: {:.5f}".format(lat[0], lon[0]))

        if len(x_o) == 0:
            print("  OUT X: N/A, Y: N/A")
        else:
            print("  OUT X: {:.2f}, Y: {:.2f}".format(x_o[0], y_o[0]))


    


    # Create a grid of lat between -22.0 and -24.0 and lon between 142.0 and 143.5
    lat_samples = np.linspace(-22.5, -23.4, 20)
    lon_samples = np.linspace(142.5, 144.0, 20)
    test_points = np.array(np.meshgrid(lon_samples, lat_samples)).T.reshape(-1, 2)
    test_points = np.array(test_points)
    lat_points, lon_points = test_points[:, 1], test_points[:, 0]

    # Check which test points are inside the polygon
    x_coords, y_coords, inside_mask = geoHt2XYInsideFOV(pp, lat_points, lon_points, ht)

    # Print all pairs of lat/lon and X/Y which are inside the FOV
    print("\nPoints inside the FOV:")
    inside_count = 0
    for i in range(len(lat_points)):
        if inside_mask[i]:
            print("Lat: {:.5f}, Lon: {:.5f} -> X: {:.2f}, Y: {:.2f}".format(lat_points[i], lon_points[i], x_coords[inside_count], y_coords[inside_count]))
            inside_count += 1

    # Plot the grid of lat/lon points on the map and on the image
    plt.figure()

    # Left - lat/lon map with marked points inside
    plt.subplot(121)
    plt.scatter(lon_points, lat_points, c='r', label="Input", marker='o')

    # Mark the lat/lon inside the FOV separately
    plt.scatter(lon_points[inside_mask], lat_points[inside_mask], c='g', label="Inside", marker='x')

    # Right - X/Y with lat/lon inside the image
    plt.subplot(122)
    plt.scatter(x_coords, y_coords, c='b', label="Output", marker='x')

    plt.show()

    # # Plot the input X, Y and the output X, Y
    # plt.figure()
    # plt.scatter(x_test, y_test, c='r', label="Input", marker='o')
    # plt.scatter(x_coords, y_coords, c='b', label="Output", marker='x')

    # plt.show()




