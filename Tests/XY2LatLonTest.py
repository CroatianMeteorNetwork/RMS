import logging
logging.getLogger('matplotlib.font_manager').disabled = True

import numpy as np
import matplotlib.pyplot as plt

from RMS.Astrometry.ApplyAstrometry import xyHt2Geo, fovEdgePolygon, geoHt2XYInsideFOV
from RMS.Astrometry.Conversions import J2000_JD
from RMS.Formats.Platepar import Platepar




if __name__ == "__main__":

    # Load a platepar file
    pp = Platepar()
    pp.read("C:\\temp\\AU004P_20250323_085721_715219_detected\\platepar_cmn2010.cal")

    # Reference height
    ht = 10_000 # m

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

        print(f"Test point {i}:")
        print(f"  IN  X: {x_i:.2f}, Y: {y_i:.2f}")
        print(f"  Latitude: {lat[0]:.5f}, Longitude: {lon[0]:.5f}")

        if len(x_o) == 0:
            print("  OUT X: N/A, Y: N/A")
        else:
            print(f"  OUT X: {x_o[0]:.2f}, Y: {y_o[0]:.2f}")


    


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
            print(f"Lat: {lat_points[i]:.5f}, Lon: {lon_points[i]:.5f} -> X: {x_coords[inside_count]:.2f}, Y: {y_coords[inside_count]:.2f}")
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




