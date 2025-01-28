import numpy as np
import matplotlib.pyplot as plt


def raDecToXYZ(ra, dec):
    """
    Convert spherical coordinates (RA, Dec) to Cartesian coordinates.

    Arguments:
        ra: [float] Right Ascension in degrees.
        dec: [float] Declination in degrees.

    Returns:
        [ndarray] Cartesian coordinates as a 1D array [x, y, z].
    """

    ra_rad, dec_rad = np.radians(ra), np.radians(dec)
    x = np.cos(ra_rad)*np.cos(dec_rad)
    y = np.sin(ra_rad)*np.cos(dec_rad)
    z = np.sin(dec_rad)
    return np.array([x, y, z])


def projectOntoPlane(v, normal):
    """
    Project a vector onto a plane defined by its normal vector.

    Arguments:
        v: [ndarray] Input vector as a 1D array [x, y, z].
        normal: [ndarray] Normal vector of the plane as a 1D array [x, y, z].

    Returns:
        [ndarray] Projected vector as a 1D array [x, y, z].
    """
    return v - np.dot(v, normal)*normal


def rotateToZ(v, normal):
    """
    Rotate a vector so the plane's normal aligns with the z-axis.

    Arguments:
        v: [ndarray] Input vector as a 1D array [x, y, z].
        normal: [ndarray] Normal vector of the plane as a 1D array [x, y, z].

    Returns:
        [ndarray] Rotated vector as a 1D array [x, y, z].
    """
    
    x = np.array([1, 0, 0])
    y = np.cross(normal, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, normal)
    rotation_matrix = np.array([x, y, normal])

    return np.dot(rotation_matrix, v)


def rayTracing(x, y, poly):
    """
    Optimized ray-tracing algorithm to determine if a 2D point is inside a polygon.

    Arguments:
        x: [float] X-coordinate of the point.
        y: [float] Y-coordinate of the point.
        poly: [ndarray] Array of polygon vertices as [[x1, y1], [x2, y2], ...].

    Returns:
        [bool] True if the point is inside the polygon, False otherwise.
    """
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]

    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        
        # Check if point is within the y-bounds of the edge
        if y > min(p1y, p2y) and y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                # Calculate intersection point x-coordinate
                if p1y != p2y:  # Avoid division by zero
                    xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xints = p1x  # Horizontal line
                
                # Toggle inside if the point is to the left of the intersection
                if p1x == p2x or x <= xints:
                    inside = not inside
        
        p1x, p1y = p2x, p2y

    return inside


def fitPlane(points):
    """
    Fits a plane to a set of 3D points using SVD.

    Parameters:
        points (np.ndarray): Nx3 array of 3D points.

    Returns:
        normal (np.ndarray): Normal vector of the plane [a, b, c].
        d (float): The d coefficient in the plane equation ax + by + cz + d = 0.
    """
    # Step 1: Compute the centroid
    centroid = np.mean(points, axis=0)
    
    # Step 2: Center the points
    centered_points = points - centroid
    
    # Step 3: Perform SVD
    _, _, vh = np.linalg.svd(centered_points)
    
    # Step 4: Extract the normal vector (last row of V^T)
    normal = vh[-1, :]
    
    # Step 5: Compute d
    d = -np.dot(normal, centroid)
    
    return normal, d


def sphericalPolygonCheck(polygon, test_points, show_plot=False):
    """ Check if a set of points in spherical coordinates are inside a polygon.
    
    Arguments:
        polygon: [list] List of polygon vertices as [[ra1, dec1], [ra2, dec2], ...].
            Coordinates are in degrees.
        test_points: [ndarray] Array of test points as [[ra1, dec1], [ra2, dec2], ...].
            Coordinates are in degrees.

    Returns:
        [list] List of booleans indicating whether each test point is inside the polygon.
    
    """

    # Convert polygon vertices to numpy array
    polygon = np.array(polygon)

    # Close the polygon if it is not already closed
    if not np.all(polygon[0] == polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    # Convert polygon vertices to Cartesian
    cartesian_polygon = np.array([raDecToXYZ(ra, dec) for ra, dec in polygon])

    # Fit a plane to the polygon vertices
    normal, _ = fitPlane(cartesian_polygon)

    # Determine the direction of the polygon (not necessarily aligned with the normal)
    # Store the direction vector which is the same as the normal, but it points towards the polygon
    v0 = cartesian_polygon[0]
    if np.dot(v0, normal) > 0:
        polygon_direction = normal
    else:
        polygon_direction = -normal

    # Project polygon onto plane and rotate to align with the z-axis
    projected_polygon = np.array([rotateToZ(projectOntoPlane(v, normal), normal)[:2] for v in cartesian_polygon])

    # Convert test points to Cartesian, project, and rotate
    cartesian_points = np.array([raDecToXYZ(ra, dec) for ra, dec in test_points])
    projected_points = np.array([rotateToZ(projectOntoPlane(v, normal), normal) for v in cartesian_points])

    # Ignore all points which are behind the normal plane with respect to the polygon
    ignore_indices = np.array([np.dot(v, polygon_direction) < 0 for v in cartesian_points])

    # Determine which points are inside the polygon
    inside = np.array([rayTracing(x, y, projected_polygon) for x, y, _ in projected_points])

    # Set points behind the normal plane to False
    inside[ignore_indices] = False

    if show_plot:

        # Plot a 3D plot of the polygon and test points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the polygon
        ax.plot(cartesian_polygon[:, 0], cartesian_polygon[:, 1], cartesian_polygon[:, 2], 'k-')

        # Plot the normal plane
        xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        zz = (-normal[0]*xx - normal[1]*yy)/normal[2]
        ax.plot_surface(xx, yy, zz, alpha=0.5)

        # Plot the normal vector
        ax.quiver(0, 0, 0, normal[0], normal[1], normal[2], color='b', label='Normal')

        # Plot the direction of the polygon
        ax.quiver(0, 0, 0, polygon_direction[0], polygon_direction[1], polygon_direction[2], color='r', label='Direction')
        
        # Plot the test points that are inside the polygon green
        ax.scatter(cartesian_points[inside, 0], cartesian_points[inside, 1], cartesian_points[inside, 2], c='g')

        # Plot the test points that are outside the polygon red
        ax.scatter(cartesian_points[~inside, 0], cartesian_points[~inside, 1], cartesian_points[~inside, 2], c='r')

        # Set limits of the 3D plot to -1 to 1
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        plt.legend()

        plt.show()


        # Plot the polygon and test points
        fig, ax = plt.subplots()
        
        # Plot the polygon
        ax.plot(projected_polygon[:, 0], projected_polygon[:, 1], 'k-')

        # Plot the test points that are inside the polygon green
        ax.plot(projected_points[inside, 0], projected_points[inside, 1], 'go')

        # Plot the test points that are outside the polygon red
        ax.plot(projected_points[~inside, 0], projected_points[~inside, 1], 'ro')

        plt.show()



    return inside


def testSphericalPolygonCheck():
    """ Test the sphericalPolygonCheck function. """

    # # US0020 manual
    # polygon = [
    #     [46.00, 79.00], [136.00, 84.00], [216.55, 75.82],
    #     [245.85, 61.53], [319.86, 56.92], [326.00, 70.64], [0.00, 77.71]
    # ]

    # US0020
    polygon = [
        (0.55, 36.68),
        (347.46, 43.15),
        (331.99, 45.77),
        (315.97, 44.23),
        (301.11, 38.12),
        (283.91, 59.51),
        (239.95, 68.72),
        (195.93, 59.46),
        (178.34, 38.02),
        (163.94, 43.93),
        (148.00, 45.49),
        (132.59, 42.86),
        (119.58, 36.30),
        (99.46, 56.19),
        (60.00, 64.18),
        (20.60, 56.25),
        (0.55, 36.68),
    ]

    # # NZ004U
    # polygon = [
    #     (343.73, -28.75),
    #     (344.85, -16.60),
    #     (343.68, -5.23),
    #     (340.54, 5.77),
    #     (334.87, 16.67),
    #     (0.13, 22.09),
    #     (22.40, 26.85),
    #     (45.65, 30.93),
    #     (73.72, 33.82),
    #     (71.39, 21.97),
    #     (71.52, 10.64),
    #     (73.56, -0.51),
    #     (77.77, -11.76),
    #     (53.55, -15.24),
    #     (32.26, -19.28),
    #     (10.41, -23.77),
    #     (343.73, -28.75),
    # ]

    # Close the polygon if it is not already closed
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    polygon = np.array(polygon)

    ra_samples = np.linspace(0, 360, 20)
    dec_samples = np.linspace(-90, 90, 20)
    test_points = np.array(np.meshgrid(ra_samples, dec_samples)).T.reshape(-1, 2)
    test_points = np.array(test_points)

    # Check which test points are inside the polygon
    inside = np.array(sphericalPolygonCheck(polygon, test_points, show_plot=True))

    # Plot a polar projection of the polygon and test points
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    ax.plot(np.radians(polygon[:, 0]), 90 - polygon[:, 1], 'k-')
    
    # Plot the test points that are inside the polygon green
    ax.plot(np.radians(test_points[inside, 0]), 90 - test_points[inside, 1], 'go')

    # Plot the test points that are outside the polygon red
    ax.plot(np.radians(test_points[~inside, 0]), 90 - test_points[~inside, 1], 'ro')

    plt.show()



if __name__ == "__main__":

    # Run the test function
    testSphericalPolygonCheck()
